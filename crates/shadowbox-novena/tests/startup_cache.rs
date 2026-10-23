//! Project-owned shaders through the host registration and restart path.
use novena::{
    api::{Object, ShaderTranslation},
    startup_cache::{StartupCacheConfig, TranslatedShader, TranslationContext, TranslationStatus},
    Host, Instance, Registers, ShaderTranslator,
};
use shadowbox_novena::ShadowboxTranslator;
use std::{
    ffi::c_void,
    fs,
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
struct Memory(Vec<u8>);
unsafe extern "C" fn read(user: *mut c_void, address: u64, out: *mut u8, size: u64) -> i32 {
    let memory = unsafe { &*user.cast::<Memory>() };
    let Some(start) = address
        .checked_sub(0x1000)
        .and_then(|n| usize::try_from(n).ok())
    else {
        return 1;
    };
    let Some(bytes) = memory.0.get(start..start.saturating_add(size as usize)) else {
        return 1;
    };
    unsafe { std::slice::from_raw_parts_mut(out, size as usize) }.copy_from_slice(bytes);
    0
}
fn register(instance: &Instance) -> Arc<TranslatedShader> {
    let mut registers = Registers {
        x: [9, 1, 0x10000, 0, 0, 0, 0, 0],
        ..Default::default()
    };
    assert_eq!(
        instance.call(
            novena::functions::lookup("nvnProgramSetShaders").unwrap(),
            &mut registers
        ),
        novena::Status::Ok
    );
    let Some(Object::Program {
        shader_translations,
        ..
    }) = instance.objects.get(9)
    else {
        panic!("program missing")
    };
    let ShaderTranslation::Cached(request) = &shader_translations[0] else {
        panic!("cached request missing")
    };
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        match request.poll() {
            TranslationStatus::Ready(shader) => return shader,
            TranslationStatus::Failed(error) => panic!("{error}"),
            _ => {
                assert!(Instant::now() < deadline);
                std::thread::yield_now();
            }
        }
    }
}
fn instance(
    memory: &mut Memory,
    directory: PathBuf,
    translator: Arc<dyn ShaderTranslator>,
) -> Instance {
    let host = Host {
        user: (memory as *mut Memory).cast(),
        read_memory: Some(read),
        ..Default::default()
    };
    let instance = unsafe {
        Instance::with_host_startup_cache(host, StartupCacheConfig::new(directory), translator)
    }
    .unwrap();
    instance.objects.put(
        7,
        Object::MemoryPool {
            device: 0,
            flags: 0,
            storage: 0x1000,
            size: memory.0.len() as u64,
            gpu_address: None,
            observed_gpu_address: None,
        },
    );
    instance.objects.put(
        9,
        Object::Program {
            device: 0,
            shader_records: Vec::new(),
            shader_translations: Vec::new(),
        },
    );
    instance
}

struct Counted {
    calls: AtomicUsize,
    adapter: ShadowboxTranslator,
}
impl ShaderTranslator for Counted {
    fn translate(&self, stage: u32, code: &[u8]) -> Result<Vec<u32>, String> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.adapter.translate(stage, code)
    }
    fn translation_cache_key(
        &self,
        stage: u32,
        code: &[u8],
        context: &TranslationContext,
    ) -> Result<Option<String>, String> {
        self.adapter.translation_cache_key(stage, code, context)
    }
    fn translate_for_context(
        &self,
        stage: u32,
        code: &[u8],
        context: &TranslationContext,
    ) -> Result<TranslatedShader, String> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.adapter.translate_for_context(stage, code, context)
    }
    fn encode_cache_record(
        &self,
        shader: &TranslatedShader,
        context: &TranslationContext,
    ) -> Result<Option<Vec<u8>>, String> {
        self.adapter.encode_cache_record(shader, context)
    }
    fn decode_cache_record(
        &self,
        key: &str,
        bytes: &[u8],
    ) -> Result<Option<TranslatedShader>, String> {
        self.adapter.decode_cache_record(key, bytes)
    }
}

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "adapter-cache-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

fn memory(program: &[u8]) -> Memory {
    // Existing host program layout, Novena provenance 0021 and 0031.
    let mut memory = Memory(vec![0; 0x10000]);
    memory.0[..4].copy_from_slice(&0x1234_5678_u32.to_le_bytes());
    memory.0[0x30..0x80].copy_from_slice(&program[..80]);
    memory.0[0x80..0x80 + program.len() - 80].copy_from_slice(&program[80..]);
    // The guest compiler ends a program with a bundle holding an unconditional
    // branch to itself; novena reads code up to it (its provenance 0046).
    let end = 0x80 + program.len() - 80;
    for (slot, word) in END_BUNDLE.into_iter().enumerate() {
        memory.0[end + 8 * slot..end + 8 * slot + 8].copy_from_slice(&word.to_le_bytes());
    }
    memory.0[0xf000..0xf008].copy_from_slice(&0x1000_u64.to_le_bytes());
    memory
}

/// A scheduling word, two NOPs and the BRA-to-self end marker
/// (novena tests/shadowbox/drawing.rs pack_shader).
const END_BUNDLE: [u64; 4] = [
    0,
    0x50b0_0000_0007_0f00,
    0x50b0_0000_0007_0f00,
    0xe240_0fff_ff87_000f,
];

fn token_program() -> Vec<u8> {
    // Project fixture: translation_cache.rs token_region_feature_requirement_survives_disk_cache.
    // Token fields: Mesa sm50.rs:3038-3051; VOTE: :3233-3247.
    let pt = 7u64 << 16;
    let vote = 0x50d8u64 << 48 | 7 << 39 | 7 << 45 | pt;
    let mut program = vec![0; 80];
    program.extend(
        [
            0,
            0xe290u64 << 48 | 32 << 20 | pt | 15,
            0xf0f8u64 << 48 | 15,
            vote,
            0,
            0xf0f8u64 << 48 | pt | 15,
            vote,
            0xe300u64 << 48 | pt | 15,
        ]
        .into_iter()
        .flat_map(u64::to_le_bytes),
    );
    program
}

fn corpus_program(name: &str) -> Vec<u8> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus");
    let mut program = fs::read(root.join(format!("{name}.hdr"))).unwrap();
    program.truncate(80);
    program.extend(fs::read(root.join(format!("{name}.bin"))).unwrap());
    program
}

#[test]
fn startup_records_restore_words_and_requirements_through_host_registration() {
    let directory = Directory::new();
    let translator = Arc::new(Counted {
        calls: AtomicUsize::new(0),
        adapter: ShadowboxTranslator::new(),
    });
    let programs = [
        corpus_program("add_inputs"),
        corpus_program("branch"),
        token_program(),
        corpus_program("gamelike_alpha_test"),
    ];
    let mut originals = Vec::new();
    for program in &programs {
        let mut memory = memory(program);
        let cold = instance(&mut memory, directory.0.clone(), translator.clone());
        let shader = register(&cold);
        assert_eq!(cold.startup_cache_stats().unwrap().misses, 1);
        originals.push(shader);
        drop(cold); // Join publication before reopening.
    }
    assert!(originals[2].requires_maximal_reconvergence);
    assert!(originals[2].requires_subgroup_size_32);
    assert!(!originals[2].requires_demote_to_helper_invocation);
    assert!(originals[3].requires_demote_to_helper_invocation);
    assert_eq!(translator.calls.load(Ordering::Relaxed), 4);
    for (program, original) in programs.iter().zip(&originals) {
        let mut memory = memory(program);
        let warm = instance(&mut memory, directory.0.clone(), translator.clone());
        assert_eq!(warm.startup_cache_stats().unwrap().loaded, 4);
        assert_eq!(register(&warm).as_ref(), original.as_ref());
        assert_eq!(warm.startup_cache_stats().unwrap().hits, 1);
        assert_eq!(warm.startup_cache_stats().unwrap().translated, 0);
        assert!(warm.take_startup_cache_diagnostics().is_empty());
    }
    assert_eq!(translator.calls.load(Ordering::Relaxed), 4);
}

#[test]
fn demotion_requirement_round_trips_in_adapter_record() {
    let translator = ShadowboxTranslator::new();
    let mut shader = translator
        .translate_for_context(0, &token_program(), &TranslationContext::default())
        .unwrap();
    shader.key = translator
        .translation_cache_key(0, &token_program(), &TranslationContext::default())
        .unwrap()
        .unwrap();
    shader.requires_demote_to_helper_invocation = true;
    let bytes = translator
        .encode_cache_record(&shader, &TranslationContext::default())
        .unwrap()
        .unwrap();
    assert_eq!(bytes[104] & 4, 4);
    assert_eq!(
        translator
            .decode_cache_record(&shader.key, &bytes)
            .unwrap()
            .unwrap(),
        shader
    );
}

fn copy_sources(source: &Path, destination: &Path) {
    fs::create_dir_all(destination).unwrap();
    for entry in fs::read_dir(source).unwrap() {
        let entry = entry.unwrap();
        let output = destination.join(entry.file_name());
        if entry.file_type().unwrap().is_dir() {
            copy_sources(&entry.path(), &output);
        } else {
            fs::copy(entry.path(), output).unwrap();
        }
    }
}

#[test]
fn different_translator_build_is_a_clean_runtime_miss() {
    let directory = Directory::new();
    let program = token_program();
    let adapter = ShadowboxTranslator::new();
    // Novena keys the program record it reads from guest memory (header, code
    // to the end marker, cb1), not these bytes; a scratch cold start names it.
    let key = {
        let scratch = Directory::new();
        let mut memory = memory(&program);
        let cold = instance(
            &mut memory,
            scratch.0.clone(),
            Arc::new(ShadowboxTranslator::new()),
        );
        register(&cold);
        drop(cold); // Join publication.
        let records: Vec<_> = fs::read_dir(&scratch.0)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .filter(|path| path.extension().is_some_and(|x| x == "sbc"))
            .collect();
        let [record] = records.as_slice() else {
            panic!("one startup record expected, found {records:?}")
        };
        record.file_stem().unwrap().to_str().unwrap().to_owned()
    };

    // Build an isolated translator source copy under the configured build output.
    // A source change changes its automatic identity. The live project stays intact.
    let output = std::env::var_os("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("adapter-foreign-builds"));
    let build = output.join(format!("foreign-record-{}", std::process::id()));
    fs::create_dir_all(build.join("crates/shadowbox")).unwrap();
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    copy_sources(
        &root.join("crates/shadowbox/src"),
        &build.join("crates/shadowbox/src"),
    );
    for name in ["Cargo.toml", "build.rs"] {
        fs::copy(
            root.join("crates/shadowbox").join(name),
            build.join("crates/shadowbox").join(name),
        )
        .unwrap();
    }
    fs::copy(root.join("Cargo.lock"), build.join("Cargo.lock")).unwrap();
    fs::write(
        build.join("Cargo.toml"),
        r#"[workspace]
members = ["crates/shadowbox"]
resolver = "2"
[workspace.package]
version = "0.1.0"
edition = "2021"
license = "PolyForm-Noncommercial-1.0.0 OR AGPL-3.0-or-later"
"#,
    )
    .unwrap();
    let foreign_source = build.join("crates/shadowbox/src/lib.rs");
    let mut source = fs::read_to_string(&foreign_source).unwrap();
    source.push_str("\n// Project experiment: a separate translator build identity.\n");
    fs::write(foreign_source, source).unwrap();
    fs::create_dir_all(build.join("writer/src")).unwrap();
    fs::write(
        build.join("writer/Cargo.toml"),
        r#"[package]
name = "foreign-record-writer"
version = "0.0.0"
edition = "2021"
[workspace]
[dependencies]
shadowbox = { path = "../crates/shadowbox" }
"#,
    )
    .unwrap();
    fs::write(
        build.join("writer/src/main.rs"),
        r#"fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let program = std::fs::read(&args[0]).unwrap();
    let output = Result::from(shadowbox::translate_header_prefixed(shadowbox::QuietRun, &program)).unwrap();
    let record = shadowbox::cache::encode_record(&args[1], Default::default(), &output).unwrap();
    std::fs::write(&args[2], record).unwrap();
}
"#,
    )
    .unwrap();
    let input = build.join("program.bin");
    fs::write(&input, &program).unwrap();
    let record = directory.0.join(format!("{key}.sbc"));
    let result = std::process::Command::new(env!("CARGO"))
        .args(["run", "--offline", "--quiet", "--manifest-path"])
        .arg(build.join("writer/Cargo.toml"))
        .arg("--")
        .arg(input)
        .arg(&key)
        .arg(&record)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "foreign writer failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    fs::remove_dir_all(build).unwrap();

    let translator = Arc::new(Counted {
        calls: AtomicUsize::new(0),
        adapter,
    });
    let mut memory = memory(&program);
    let cold = instance(&mut memory, directory.0.clone(), translator.clone());
    let stats = cold.startup_cache_stats().unwrap();
    assert_eq!((stats.loaded, stats.invalid), (0, 0));
    assert!(cold.take_startup_cache_diagnostics().is_empty());
    let original = register(&cold);
    assert_eq!(cold.startup_cache_stats().unwrap().misses, 1);
    drop(cold);
    assert_eq!(translator.calls.load(Ordering::Relaxed), 1);
    let warm = instance(&mut memory, directory.0.clone(), translator.clone());
    assert_eq!(warm.startup_cache_stats().unwrap().loaded, 1);
    assert_eq!(register(&warm).as_ref(), original.as_ref());
    assert_eq!(translator.calls.load(Ordering::Relaxed), 1);
    assert!(warm.take_startup_cache_diagnostics().is_empty());
}

#[cfg(feature = "gpu-check")]
#[test]
#[ignore = "requires an offscreen device"]
fn restarted_demote_module_reaches_production_device_creation() {
    let directory = Directory::new();
    let translator = Arc::new(Counted {
        calls: AtomicUsize::new(0),
        adapter: ShadowboxTranslator::new(),
    });
    let program = corpus_program("gamelike_alpha_test");
    let mut host_memory = memory(&program);
    let cold = instance(&mut host_memory, directory.0.clone(), translator.clone());
    let original = register(&cold);
    assert!(original.requires_demote_to_helper_invocation);
    drop(cold);
    let warm = instance(&mut host_memory, directory.0.clone(), translator.clone());
    let restored = register(&warm);
    assert_eq!(restored, original);
    assert_eq!(warm.startup_cache_stats().unwrap().loaded, 1);
    assert_eq!(warm.startup_cache_stats().unwrap().translated, 0);
    assert_eq!(translator.calls.load(Ordering::Relaxed), 1);
    let mut context = novena::gpu::Context::new().expect("offscreen device required");
    assert!(context.shader_features.demote_to_helper_invocation);
    context.shader_features.demote_to_helper_invocation = false;
    assert_eq!(
        context
            .shader_features
            .validate_translated_shader(&restored)
            .unwrap_err(),
        "shaderDemoteToHelperInvocation is unavailable"
    );
    assert_eq!(
        context
            .create_global_shader_module(&restored.words)
            .unwrap_err(),
        "shaderDemoteToHelperInvocation is unavailable"
    );
    context.shader_features.demote_to_helper_invocation = true;
    context
        .shader_features
        .validate_translated_shader(&restored)
        .unwrap();
    let module = context
        .create_global_shader_module(&restored.words)
        .unwrap();
    unsafe {
        context.device.destroy_shader_module(module, None);
    }
}
