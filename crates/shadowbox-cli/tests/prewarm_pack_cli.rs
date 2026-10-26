//! Synthetic inputs only.
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs::{self, File},
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicUsize, Ordering},
    time::SystemTime,
};

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir()
            .join("shadowbox-prewarm-pack-tests")
            .join(format!(
                "{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn cli(args: &[&Path]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_shadowbox"))
        .args(args)
        .output()
        .unwrap()
}
fn success(args: &[&Path]) -> Output {
    let output = cli(args);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}
fn pack(out: &Path, romfs: &Path, caches: &[&Path]) -> Output {
    let mut args = vec![Path::new("prewarm-pack"), out, Path::new("--romfs"), romfs];
    for cache in caches {
        args.extend([Path::new("--emulator-cache"), cache]);
    }
    success(&args)
}
fn manifest(out: &Path) -> Value {
    serde_json::from_slice(&fs::read(out.join("manifest.json")).unwrap()).unwrap()
}
fn shader(stage: u32, immediate: u32, bad: bool) -> Vec<u8> {
    // Mesa src/nouveau/compiler/nak/sph.rs:112-120,204-215: type/version/stage;
    // :305-312,326-350: geometry count/output/maximum vertices.
    // Mesa src/nouveau/compiler/nak/sm50.rs:220-244,3418-3428: PT/barriers/bundles;
    // :1938-1951,321-328: MOV32I opcode/immediate/lanes/destination;
    // :3106-3110: EXIT/CC.T; :3183-3187: real OpNop/CC.T.
    let mut bytes = vec![0; 80];
    if stage != 0 {
        bytes[..4].copy_from_slice(
            &((stage << 10) | (3 << 5) | if stage == 5 { 2 } else { 1 }).to_le_bytes(),
        );
    }
    if stage == 4 {
        bytes[11] = 1;
        bytes[15] = 1;
        bytes[16] = 1;
    }
    // TCS count deliberately zero to test a real translation blocker.
    let control = 0x7e0u64 | (0x7e0 << 21) | (0x7e0 << 42);
    bytes.extend(
        [
            control,
            if bad {
                0x1234_0000_0007_0000
            } else {
                0x0100_0000_0007_f000 | (u64::from(immediate) << 20)
            },
            0xe300_0000_0007_000f,
            if bad {
                0x5678_0000_0007_0000
            } else {
                0x50b0_0000_0007_0f00
            },
        ]
        .into_iter()
        .flat_map(u64::to_le_bytes),
    );
    bytes
}
fn extracted(shader: &[u8]) -> Vec<u8> {
    // Project envelope: docs/reference-sources.md, permitted Novena
    // docs/reference-sources.md (magic/SPH/code offsets).
    let compute = shader[..80].iter().all(|b| *b == 0);
    let mut bytes = vec![0; if compute { 0x100 } else { 0x80 }];
    bytes[..4].copy_from_slice(&(if compute { 0x1234_5679u32 } else { 0x1234_5678 }).to_le_bytes());
    if !compute {
        bytes[0x30..0x80].copy_from_slice(&shader[..80]);
    }
    bytes.extend_from_slice(&shader[80..]);
    bytes
}
// (stage, immediate, topology, native state byte length, unknown instructions).
fn cache(dir: &Path, records: &[(u32, u32, u32, usize, bool)]) {
    // Ryujinx Graphics.Gpu/Shader/DiskCache/DiskCacheGuestStorage.cs:14-21,
    // 28-85,395-413: guest TOC version/header/offset/size/CB1/hash.
    // DiskCacheHostStorage.cs:16-28,37-68,592-619: shared TOC/version/offset/
    // mask/indices. BinarySerializer.cs:227-245: compression tag 0 (none).
    // ShaderSpecializationState.cs:974-1021: queried flags, compute, state,
    // remaining no-CB/no-pipeline/no-texture tail; BinarySerializer.cs:111-116:
    // native state magic/size; GpuChannelGraphicsState.cs:18-28: topology +4.
    fs::create_dir_all(dir).unwrap();
    let mut guest_toc = vec![0; 32];
    guest_toc[..4].copy_from_slice(b"TOCG");
    guest_toc[4..8].copy_from_slice(&0x10001u32.to_le_bytes());
    guest_toc[12..16].copy_from_slice(&(records.len() as u32).to_le_bytes());
    let mut shared_toc = vec![0; 32];
    shared_toc[..4].copy_from_slice(b"TOCS");
    shared_toc[4..8].copy_from_slice(&0x10002u32.to_le_bytes());
    let (mut guest_data, mut shared_data) = (Vec::new(), Vec::new());
    for (index, &(stage, imm, topology, size, bad)) in records.iter().enumerate() {
        let record = shader(stage, imm, bad);
        let code = if stage == 0 {
            &record[80..]
        } else {
            &record[..]
        };
        for field in [guest_data.len() as u32, code.len() as u32, 0, 0] {
            guest_toc.extend(field.to_le_bytes());
        }
        guest_data.push(0);
        guest_data.extend(code);
        shared_toc.extend((shared_data.len() as u64).to_le_bytes());
        shared_data.push(0);
        shared_data.extend((if stage == 0 { 0u32 } else { 1 << stage }).to_le_bytes());
        shared_data.extend((index as i32).to_le_bytes());
        shared_data.extend(0u32.to_le_bytes());
        shared_data.push(u8::from(stage == 0));
        shared_data.extend(if stage == 0 { b"COMS" } else { b"GFXS" });
        shared_data.extend((size as u32).to_le_bytes());
        let mut state = vec![0; size];
        if size >= 8 {
            state[4..8].copy_from_slice(&topology.to_le_bytes());
        }
        shared_data.extend(state);
        shared_data.extend([0, 0]);
        shared_data.extend(0u32.to_le_bytes());
    }
    for (name, data) in [
        ("guest.toc", guest_toc),
        ("guest.data", guest_data),
        ("shared.toc", shared_toc),
        ("shared.data", shared_data),
    ] {
        fs::write(dir.join(name), data).unwrap();
    }
}
fn snapshot(dir: &Path) -> BTreeMap<PathBuf, (Vec<u8>, SystemTime)> {
    let mut result = BTreeMap::new();
    fn visit(root: &Path, dir: &Path, result: &mut BTreeMap<PathBuf, (Vec<u8>, SystemTime)>) {
        for entry in fs::read_dir(dir).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                visit(root, &path, result);
            } else {
                result.insert(
                    path.strip_prefix(root).unwrap().to_owned(),
                    (
                        fs::read(&path).unwrap(),
                        path.metadata().unwrap().modified().unwrap(),
                    ),
                );
            }
        }
    }
    visit(dir, dir, &mut result);
    result
}

#[test]
fn combines_sources_state_variants_and_first_blockers_deterministically() {
    let temp = Temp::new();
    let romfs = temp.0.join("romfs");
    let a = temp.0.join("a");
    let b = temp.0.join("b");
    let out = temp.0.join("out");
    fs::create_dir(&romfs).unwrap();
    for stage in [2, 3, 4, 5] {
        fs::write(
            romfs.join(format!("stage{stage}")),
            extracted(&shader(stage, 1, false)),
        )
        .unwrap();
    }
    cache(
        &a,
        &[
            (5, 1, 0, 12, false),
            (4, 1, 0, 12, false),
            (0, 4, 0, 12, false),
            (5, 9, 0, 12, true),
            (5, 2, 0, 12, false),
        ],
    );
    cache(
        &b,
        &[
            (4, 1, 4, 12, false),
            (3, 1, 0, 0, false),
            (5, 1, 0, 12, false),
        ],
    );
    pack(&out, &romfs, &[&a, &b]);
    let value = manifest(&out);
    assert_eq!(
        value["counts"],
        serde_json::json!({"programs_found_by_scan": 4, "programs_found_by_cache": 6,
        "programs_found_by_both": 3, "translated_complete": 3, "state_missing": 2, "failed": 2, "total_programs": 7})
    );
    assert_eq!(value["runtime_cache_entries"], 4);
    assert_eq!(value["failures"].as_array().unwrap().len(), 2);
    // Inspect diagnostics only. Automatic identities may contain any hex substring.
    let blockers = value["failures"]
        .as_array()
        .unwrap()
        .iter()
        .map(|failure| failure["first_blocker"].as_str().unwrap())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(blockers.contains("1234"));
    assert!(!blockers.contains("5678"));
    let geometry = value["programs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["stage"] == "geometry")
        .unwrap();
    assert_eq!(geometry["variants"].as_array().unwrap().len(), 2);
    let before = snapshot(&out.join(".work"));
    let cached = snapshot(&out.join("cache"));
    pack(&out, &romfs, &[&b, &a, &a]);
    assert_eq!(manifest(&out), value);
    assert_eq!(snapshot(&out.join(".work")), before);
    assert_eq!(snapshot(&out.join("cache")), cached);
    let fresh = temp.0.join("fresh");
    pack(&fresh, &romfs, &[&b, &a]);
    assert_eq!(manifest(&fresh), value);
    assert_eq!(
        shadowbox::disk_cache_stats(&fresh.join("cache"))
            .unwrap()
            .invalid,
        0
    );
    for (name, (bytes, _)) in cached {
        assert_eq!(fs::read(fresh.join("cache").join(name)).unwrap(), bytes);
    }
    // Runtime translation really hits the produced package, preserving bytes.
    let record = shader(5, 1, false);
    let input =
        shadowbox::header_prefixed_input(&record, shadowbox::TargetGeneration::Sm5).unwrap();
    let mut runtime = shadowbox::Cache::new(shadowbox::CacheConfig {
        memory: shadowbox::cache::Keep::nothing(),
        disk: shadowbox::cache::Disk::at(out.join("cache")),
    });
    let warm = Result::from(shadowbox::translate_cached_with_context(
        shadowbox::QuietRun,
        &input,
        shadowbox::CacheContext::default(),
        &mut runtime,
    ))
    .unwrap();
    assert_eq!(runtime.stats.disk_hits, 1);
    assert_eq!(runtime.stats.misses, 0);
    assert_eq!(
        warm.spirv,
        Result::from(shadowbox::translate(shadowbox::QuietRun, &input))
            .unwrap()
            .spirv
    );
}

#[test]
fn resumes_by_content_and_removes_entries_from_changed_inputs() {
    let temp = Temp::new();
    let romfs = temp.0.join("romfs");
    let guest = temp.0.join("guest");
    let out = temp.0.join("out");
    fs::create_dir(&romfs).unwrap();
    let program = romfs.join("program");
    fs::write(&program, extracted(&shader(5, 1, false))).unwrap();
    cache(&guest, &[(5, 2, 0, 12, false)]);
    pack(&out, &romfs, &[&guest]);
    let original = manifest(&out);
    let checkpoints = snapshot(&out.join(".work"));
    // An identical rewrite changes mtime, while input SHA-256 remains unchanged.
    fs::write(&program, fs::read(&program).unwrap()).unwrap();
    pack(&out, &romfs, &[&guest]);
    assert_eq!(snapshot(&out.join(".work")), checkpoints);
    // Change bytes with the same length and restore mtime: hashes must invalidate.
    let mtime = program.metadata().unwrap().modified().unwrap();
    fs::write(&program, extracted(&shader(5, 3, false))).unwrap();
    File::options()
        .write(true)
        .open(&program)
        .unwrap()
        .set_modified(mtime)
        .unwrap();
    pack(&out, &romfs, &[&guest]);
    let current = manifest(&out);
    assert_ne!(current["romfs_sha256"], original["romfs_sha256"]);
    let old = original["programs"]
        .as_array()
        .unwrap()
        .iter()
        .find(|p| p["sources"] == serde_json::json!(["scan"]))
        .unwrap();
    let old_key = old["variants"][0]["translation_key"].as_str().unwrap();
    assert!(!out.join("cache").join(format!("{old_key}.sbc")).exists());
    for (name, saved) in &checkpoints {
        if name
            .components()
            .next()
            .unwrap()
            .as_os_str()
            .to_string_lossy()
            .starts_with("import-")
        {
            assert_eq!(snapshot(&out.join(".work"))[name], *saved);
        }
    }
    fs::remove_file(program).unwrap();
    pack(&out, &romfs, &[]);
    assert_eq!(manifest(&out)["counts"]["total_programs"], 0);
    assert_eq!(
        shadowbox::disk_cache_stats(&out.join("cache"))
            .unwrap()
            .entries,
        0
    );
}

#[test]
fn repairs_corrupt_outputs_and_unfinished_stages() {
    let temp = Temp::new();
    let romfs = temp.0.join("romfs");
    let guest = temp.0.join("guest");
    let out = temp.0.join("out");
    fs::create_dir(&romfs).unwrap();
    cache(&guest, &[(5, 1, 0, 12, false)]);
    pack(&out, &romfs, &[&guest]);
    let value = manifest(&out);
    let before = snapshot(&out.join(".work"));
    let runtime = fs::read_dir(out.join("cache"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    fs::write(&runtime, b"corrupt").unwrap();
    pack(&out, &romfs, &[&guest]);
    assert_eq!(manifest(&out), value);
    assert_eq!(snapshot(&out.join(".work")), before);
    assert_eq!(
        shadowbox::disk_cache_stats(&out.join("cache"))
            .unwrap()
            .invalid,
        0
    );
    let imported = fs::read_dir(out.join(".work"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| {
            p.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("import-")
        })
        .unwrap();
    let artifact = fs::read_dir(imported.join("cache"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    fs::write(&artifact, b"corrupt").unwrap();
    pack(&out, &romfs, &[&guest]);
    assert_eq!(manifest(&out), value);
    assert_eq!(
        shadowbox::disk_cache_stats(&imported.join("cache"))
            .unwrap()
            .invalid,
        0
    );
    // Simulate process death between writing outputs and committing its checkpoint.
    fs::remove_file(imported.join("checkpoint.json")).unwrap();
    fs::write(imported.join("unfinished.partial"), b"partial").unwrap();
    fs::write(out.join("manifest.pending"), b"partial").unwrap();
    fs::write(out.join(".prewarm-pack.lock"), b"dead process\n").unwrap();
    pack(&out, &romfs, &[&guest]);
    assert_eq!(manifest(&out), value);
    assert!(!imported.join("unfinished.partial").exists());
    assert!(!out.join("manifest.pending").exists());
    // A result bit flip must also invalidate a checkpoint even if output files match.
    let checkpoint = imported.join("checkpoint.json");
    let mut corrupt: Value = serde_json::from_slice(&fs::read(&checkpoint).unwrap()).unwrap();
    corrupt["result"]["programs"][0]["status"] = Value::String("failed".into());
    fs::write(checkpoint, corrupt.to_string()).unwrap();
    pack(&out, &romfs, &[&guest]);
    assert_eq!(manifest(&out), value);
}

#[test]
fn completed_import_survives_another_inputs_error() {
    let temp = Temp::new();
    let romfs = temp.0.join("romfs");
    let a = temp.0.join("a");
    let b = temp.0.join("b");
    let out = temp.0.join("out");
    fs::create_dir(&romfs).unwrap();
    cache(&a, &[(5, 1, 0, 12, false)]);
    cache(&b, &[(5, 2, 0, 12, false)]);
    fs::write(b.join("guest.toc"), b"bad").unwrap();
    let args = [
        Path::new("prewarm-pack"),
        &out,
        Path::new("--romfs"),
        &romfs,
        Path::new("--emulator-cache"),
        &a,
        Path::new("--emulator-cache"),
        &b,
    ];
    assert!(!cli(&args).status.success());
    assert!(!out.join("manifest.json").exists());
    let saved = snapshot(&out.join(".work"));
    assert!(saved
        .keys()
        .any(|p| p.file_name() == Some(std::ffi::OsStr::new("checkpoint.json"))));
    cache(&b, &[(5, 2, 0, 12, false)]);
    pack(&out, &romfs, &[&a, &b]);
    let after = snapshot(&out.join(".work"));
    for (name, entry) in saved {
        if name.file_name() == Some(std::ffi::OsStr::new("checkpoint.json")) {
            assert_eq!(after[&name], entry);
        }
    }
    let published = fs::read(out.join("manifest.json")).unwrap();
    let runtime = snapshot(&out.join("cache"));
    fs::write(b.join("guest.toc"), b"bad again").unwrap();
    assert!(!cli(&args).status.success());
    assert_eq!(fs::read(out.join("manifest.json")).unwrap(), published);
    assert_eq!(snapshot(&out.join("cache")), runtime);
}

#[test]
fn validates_arguments_output_boundaries_keys_and_lock() {
    let temp = Temp::new();
    let romfs = temp.0.join("romfs");
    let out = temp.0.join("out");
    fs::create_dir(&romfs).unwrap();
    for args in [
        vec![Path::new("prewarm-pack"), &out],
        vec![Path::new("prewarm-pack"), &out, Path::new("--romfs")],
        vec![
            Path::new("prewarm-pack"),
            &out,
            Path::new("--romfs"),
            &romfs,
            Path::new("--wat"),
        ],
    ] {
        assert!(!cli(&args).status.success());
    }
    let nested = romfs.join("output");
    assert!(!cli(&[
        Path::new("prewarm-pack"),
        &nested,
        Path::new("--romfs"),
        &romfs
    ])
    .status
    .success());
    fs::create_dir(&out).unwrap();
    fs::write(out.join("sentinel"), b"keep").unwrap();
    assert!(!cli(&[
        Path::new("prewarm-pack"),
        &out,
        Path::new("--romfs"),
        &romfs
    ])
    .status
    .success());
    assert_eq!(fs::read(out.join("sentinel")).unwrap(), b"keep");
    fs::remove_file(out.join("sentinel")).unwrap();
    pack(&out, &romfs, &[]);
    let lock = File::options()
        .write(true)
        .open(out.join(".prewarm-pack.lock"))
        .unwrap();
    lock.lock().unwrap();
    let blocked = cli(&[
        Path::new("prewarm-pack"),
        &out,
        Path::new("--romfs"),
        &romfs,
    ]);
    assert!(!blocked.status.success());
    assert!(String::from_utf8_lossy(&blocked.stderr).contains("holds the package lock"));
    drop(lock);
    pack(&out, &romfs, &[]);
    let key = temp.0.join("private-key");
    fs::write(&key, "11".repeat(32)).unwrap();
    let keyed = [
        Path::new("prewarm-pack"),
        &out,
        Path::new("--romfs"),
        &romfs,
        Path::new("--pak-aes-key-file"),
        &key,
    ];
    success(&keyed);
    let first = manifest(&out);
    let before = snapshot(&out.join(".work"));
    success(&keyed);
    assert_eq!(snapshot(&out.join(".work")), before);
    fs::write(&key, "22".repeat(32)).unwrap();
    success(&keyed);
    assert_ne!(manifest(&out)["sources"], first["sources"]);
    assert!(!fs::read_to_string(out.join("manifest.json"))
        .unwrap()
        .contains(&"22".repeat(32)));
    fs::write(&key, b"secret invalid content").unwrap();
    let error = cli(&keyed);
    assert!(!error.status.success());
    let text = String::from_utf8_lossy(&error.stderr);
    assert!(!text.contains("private-key"));
    assert!(!text.contains("secret invalid content"));
}

#[cfg(unix)]
#[test]
fn rejects_symlinks_without_writing_through_them() {
    use std::os::unix::fs::symlink;
    let temp = Temp::new();
    let romfs = temp.0.join("romfs");
    let out = temp.0.join("out");
    let elsewhere = temp.0.join("elsewhere");
    fs::create_dir(&romfs).unwrap();
    fs::create_dir(&elsewhere).unwrap();
    fs::write(elsewhere.join("sentinel"), b"keep").unwrap();
    symlink(&elsewhere, romfs.join("link")).unwrap();
    assert!(!cli(&[
        Path::new("prewarm-pack"),
        &out,
        Path::new("--romfs"),
        &romfs
    ])
    .status
    .success());
    fs::remove_file(romfs.join("link")).unwrap();
    pack(&out, &romfs, &[]);
    fs::remove_dir(out.join("cache")).unwrap();
    symlink(&elsewhere, out.join("cache")).unwrap();
    assert!(!cli(&[
        Path::new("prewarm-pack"),
        &out,
        Path::new("--romfs"),
        &romfs
    ])
    .status
    .success());
    assert_eq!(fs::read(elsewhere.join("sentinel")).unwrap(), b"keep");
}

#[test]
fn manifest_records_identity_and_foreign_checkpoints_are_rebuilt() {
    let temp = Temp::new();
    let romfs = temp.0.join("romfs");
    let guest = temp.0.join("guest");
    let out = temp.0.join("out");
    fs::create_dir(&romfs).unwrap();
    fs::write(romfs.join("program"), extracted(&shader(5, 1, false))).unwrap();
    cache(&guest, &[(5, 2, 0, 12, false)]);
    pack(&out, &romfs, &[&guest]);
    let identity = shadowbox::translator_identity(
        shadowbox::TargetGeneration::Sm5,
        shadowbox::TranslationOptions::default(),
    );
    let expected = manifest(&out);
    assert_eq!(expected["translator_identity"], identity);
    assert_eq!(
        expected["translator_version"],
        shadowbox::TRANSLATOR_VERSION
    );
    assert_eq!(
        expected["cache_format_version"],
        shadowbox::CACHE_FORMAT_VERSION
    );
    let checkpoints = snapshot(&out.join(".work"))
        .into_keys()
        .filter(|path| path.file_name() == Some(std::ffi::OsStr::new("checkpoint.json")))
        .collect::<Vec<_>>();
    assert_eq!(checkpoints.len(), 3); // scan, import, merge
    for path in &checkpoints {
        let path = out.join(".work").join(path);
        let mut old: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        assert_eq!(old["translator_identity"], identity);
        // Keep input/output/result hashes correct. Only the translator differs.
        old["translator_identity"] = Value::String("0".repeat(64));
        fs::write(path, old.to_string()).unwrap();
    }
    pack(&out, &romfs, &[&guest]);
    assert_eq!(manifest(&out), expected);
    for path in checkpoints {
        let checkpoint: Value =
            serde_json::from_slice(&fs::read(out.join(".work").join(path)).unwrap()).unwrap();
        assert_eq!(checkpoint["translator_identity"], identity);
    }
}
