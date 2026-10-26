#![allow(unexpected_cfgs)]
//! Frozen output identity and single-process translation/allocation benchmark.
#[path = "support/translation_inputs.rs"]
mod inputs;
#[path = "support/speed3_inputs.rs"]
mod speed3_inputs;
#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use sha2::{Digest, Sha256};
#[cfg(not(speed3_baseline))]
use shadowbox::Translator;
use shadowbox::{interface::ShaderInput, translate};
#[path = "support/families.rs"]
mod families;
#[path = "support/family_inputs.rs"]
mod family_inputs;
use std::{
    alloc::{GlobalAlloc, Layout, System},
    fmt::Write,
    hint::black_box,
    sync::atomic::{AtomicBool, AtomicUsize, Ordering},
    time::{Duration, Instant},
};

struct Allocator;
static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static BYTES: AtomicUsize = AtomicUsize::new(0);

// Counting is enabled only for the isolated, untimed allocation sweep.
unsafe impl GlobalAlloc for Allocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        }
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            BYTES.fetch_add(size, Ordering::Relaxed);
        }
        unsafe { System.realloc(ptr, layout, size) }
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: Allocator = Allocator;

fn run(case: &inputs::Case) -> shadowbox::interface::ShaderOutput {
    Result::from(translate(
        shadowbox::QuietRun,
        &ShaderInput {
            code: &case.code,
            program_record: &case.header,
        },
    ))
    .unwrap_or_else(|e| panic!("{}: {e}", case.name))
}

fn root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn hash_output(
    output: Result<shadowbox::interface::ShaderOutput, shadowbox::interface::Stopped>,
) -> String {
    let mut hash = Sha256::new();
    match output {
        Ok(output) => {
            hash.update([0, u8::from(output.requires_subgroup_size_32)]);
            hash.update((output.spirv.len() as u64).to_le_bytes());
            for word in output.spirv {
                hash.update(word.to_le_bytes());
            }
            hash.update(format!("{:?}", output.surface_descriptors));
        }
        Err(error) => {
            hash.update([1]);
            hash.update(error.to_string().as_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}

#[test]
fn synthetic_before_after_hashes() {
    let mut rows = String::new();
    for case in speed3_inputs::synthetic() {
        let output = run(&case);
        let mut hash = Sha256::new();
        hash.update([u8::from(output.requires_subgroup_size_32)]);
        hash.update((output.spirv.len() as u64).to_le_bytes());
        for word in output.spirv {
            hash.update(word.to_le_bytes());
        }
        writeln!(rows, "{}\t{:x}", case.name, hash.finalize()).unwrap();
    }
    if std::env::var_os("SHADOWBOX_RECORD_TRANSLATE_HASHES").is_some() {
        std::fs::write(
            root().join("docs/benchmarks/xlate-speed3-hashes.tsv"),
            &rows,
        )
        .unwrap();
    }
    assert_eq!(
        rows,
        std::fs::read_to_string(root().join("docs/benchmarks/xlate-speed3-hashes.tsv")).unwrap()
    );
}

#[test]
fn all_family_before_after_hashes() {
    let mut rows = String::new();
    let mut cases = inputs::corpus();
    cases.extend(inputs::synthetic());
    cases.extend(speed3_inputs::synthetic());
    for case in cases {
        let output = Result::from(translate(
            shadowbox::QuietRun,
            &ShaderInput {
                code: &case.code,
                program_record: &case.header,
            },
        ));
        writeln!(rows, "{}\t{}", case.name, hash_output(output)).unwrap();
    }
    for family in families::FAMILIES {
        let code = family_inputs::program(*family);
        let (header, state) = family_inputs::family_context(*family);
        let output = Result::from(shadowbox::translate_with_graphics_state(
            shadowbox::QuietRun,
            &ShaderInput {
                code: &code,
                program_record: &header,
            },
            &state,
        ));
        writeln!(rows, "family:{}\t{}", family.name, hash_output(output)).unwrap();
    }
    let file = std::env::var_os("SPEED3_HASH_FILE")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| root().join("docs/benchmarks/family-before-speed.tsv"));
    if std::env::var_os("SPEED3_RECORD_HASHES").is_some() {
        std::fs::write(&file, &rows).unwrap();
    }
    assert_eq!(rows, std::fs::read_to_string(file).unwrap());
}

#[test]
#[ignore = "manual benchmark, run alone with --test-threads=1"]
fn translation_measurements() {
    measure(false);
}

#[cfg(not(speed3_baseline))]
#[test]
#[ignore = "manual batch benchmark, run alone with --test-threads=1"]
fn parallel_translation_measurements() {
    measure(true);
}

fn measure(batch: bool) {
    let mut groups = vec![("corpus".to_owned(), inputs::corpus())];
    let synthetic = speed3_inputs::synthetic();
    for length in [200, 500, 1000, 2000] {
        let prefix = format!("mixed_{length}_");
        let cases = synthetic
            .iter()
            .filter(|c| c.name.starts_with(&prefix))
            .map(|c| inputs::Case {
                name: c.name.clone(),
                code: c.code.clone(),
                header: c.header.clone(),
            })
            .collect();
        groups.push((format!("mixed_{length}"), cases));
    }
    groups.push(("legacy".to_owned(), inputs::synthetic()));
    let filter = std::env::var("SPEED3_FILTER").unwrap_or_default();
    let samples: usize = std::env::var("SPEED3_SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);
    let ms: u64 = std::env::var("SPEED3_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);
    for (name, mut cases) in groups {
        if batch {
            let n = cases.len();
            for index in n..256 {
                let case = &cases[index % n];
                cases.push(inputs::Case {
                    name: case.name.clone(),
                    code: case.code.clone(),
                    header: case.header.clone(),
                });
            }
        }
        if !filter.is_empty() && filter != name {
            continue;
        }
        #[cfg(not(speed3_baseline))]
        let inputs = cases
            .iter()
            .map(|case| ShaderInput {
                code: &case.code,
                program_record: &case.header,
            })
            .collect::<Vec<_>>();
        #[cfg(not(speed3_baseline))]
        let mut translator = Translator::default();
        let mut sweep = || {
            #[cfg(speed3_baseline)]
            assert!(!batch);
            #[cfg(not(speed3_baseline))]
            if batch {
                let outputs = translator.translate_batch(shadowbox::QuietRun, black_box(&inputs));
                for (case, output) in cases.iter().zip(outputs) {
                    black_box(
                        Result::from(output)
                            .unwrap_or_else(|error| panic!("{}: {error}", case.name)),
                    );
                }
                return;
            }
            for case in &cases {
                black_box(run(black_box(case)));
            }
        };
        sweep();
        ALLOCATIONS.store(0, Ordering::Relaxed);
        BYTES.store(0, Ordering::Relaxed);
        COUNTING.store(true, Ordering::Relaxed);
        sweep();
        COUNTING.store(false, Ordering::Relaxed);
        let allocations = ALLOCATIONS.load(Ordering::Relaxed) as f64 / cases.len() as f64;
        let bytes = BYTES.load(Ordering::Relaxed) as f64 / cases.len() as f64;
        let mut rates = Vec::new();
        for _ in 0..samples {
            let start = Instant::now();
            let mut programs = 0;
            loop {
                sweep();
                programs += cases.len();
                if start.elapsed() >= Duration::from_millis(ms) {
                    break;
                }
            }
            rates.push(programs as f64 / start.elapsed().as_secs_f64());
        }
        rates.sort_by(f64::total_cmp);
        println!(
            "{name}\t{:.2}\t{:.2}\t{:.2}\t{allocations:.2}\t{bytes:.0}",
            rates[rates.len() / 2],
            rates[0],
            rates[rates.len() - 1]
        );
    }
}

#[test]
#[ignore = "export the generated corpus for CLI profiling"]
fn export_synthetic() {
    let dir = std::env::var_os("SPEED3_OUT")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("CARGO_TARGET_DIR")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(std::env::temp_dir)
                .join("xlate-speed3")
        })
        .join("inputs");
    std::fs::create_dir_all(&dir).unwrap();
    for case in speed3_inputs::synthetic() {
        std::fs::write(dir.join(format!("{}.bin", case.name)), case.code).unwrap();
        std::fs::write(dir.join(format!("{}.hdr", case.name)), case.header).unwrap();
    }
}

#[test]
fn synthetic_modules_validate() {
    for case in speed3_inputs::synthetic() {
        let output = run(&case);
        let result = spirv_validation::validate(&output.spirv);
        assert!(
            result.status.success(),
            "{}: {}",
            case.name,
            String::from_utf8_lossy(&result.stderr)
        );
    }
}

// Repeated batches alternate stage, header size, workload size and failures.
// Compare with cold emission as well as the frozen hashes, so prior scratch
// contents cannot silently affect output or diagnostics.
#[cfg(not(speed3_baseline))]
#[test]
fn batch_matches_cold_translation_after_failures() {
    let mut cases = inputs::corpus();
    cases.extend(speed3_inputs::synthetic());
    cases.push(inputs::Case {
        name: "malformed".into(),
        code: vec![0],
        header: Vec::new(),
    });
    cases.push(inputs::Case {
        name: "short_header".into(),
        code: Vec::new(),
        header: vec![0; 4],
    });
    cases.push(inputs::Case {
        name: "unsupported".into(),
        code: vec![0xff; 32],
        header: Vec::new(),
    });
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let mut translator = Translator::default();
    assert!(translator
        .translate_batch(shadowbox::QuietRun, &[])
        .is_empty());
    for round in 0..3 {
        if round == 1 {
            cases.reverse();
        }
        let inputs = cases
            .iter()
            .map(|case| ShaderInput {
                code: &case.code,
                program_record: &case.header,
            })
            .collect::<Vec<_>>();
        let state = shadowbox::interface::GraphicsState::default();
        let results = pool.install(|| {
            if round == 2 {
                translator.translate_batch_with_options_and_graphics_state(
                    shadowbox::QuietRun,
                    &inputs,
                    shadowbox::TranslationOptions {
                        global_delta: shadowbox::GlobalDelta::Zero,
                        float64: shadowbox::HostFloat64::Unsupported,
                    },
                    &state,
                )
            } else {
                translator.translate_batch(shadowbox::QuietRun, &inputs)
            }
        });
        assert_eq!(results.len(), inputs.len());
        for ((case, input), actual) in cases.iter().zip(&inputs).zip(results) {
            let mut cold = Translator::default();
            let expected = Result::from(cold.translate_with_options_and_graphics_state(
                shadowbox::QuietRun,
                input,
                shadowbox::TranslationOptions {
                    float64: shadowbox::HostFloat64::Unsupported,
                    global_delta: if round == 2 {
                        shadowbox::GlobalDelta::Zero
                    } else {
                        shadowbox::GlobalDelta::Pushed
                    },
                },
                &state,
            ));
            if case.name == "unsupported" {
                assert!(matches!(
                    &expected,
                    Err(shadowbox::interface::Stopped::Unknown(_))
                ));
            }
            match (Result::from(actual), expected) {
                (Ok(a), Ok(b)) => {
                    assert_eq!(a.spirv, b.spirv, "{}", case.name);
                    assert_eq!(a.requires_subgroup_size_32, b.requires_subgroup_size_32);
                    assert_eq!(a.surface_descriptors, b.surface_descriptors);
                    // Reusing the translator must not mutate an earlier output.
                    Result::from(translator.translate(shadowbox::QuietRun, &inputs[0])).ok();
                    assert_eq!(a.spirv, b.spirv, "{}", case.name);
                }
                (Err(a), Err(b)) => assert_eq!(a.to_string(), b.to_string(), "{}", case.name),
                (a, b) => panic!("{}: warm={}, cold={}", case.name, a.is_ok(), b.is_ok()),
            }
        }
    }
}

#[cfg(not(speed3_baseline))]
#[test]
fn parallel_families_match_serial_with_driver_state() {
    let cases = families::FAMILIES
        .iter()
        .map(|family| {
            let code = family_inputs::program(*family);
            let (header, _) = family_inputs::family_context(*family);
            (family.name, code, header)
        })
        .collect::<Vec<_>>();
    let (_, state) = family_inputs::family_context(families::FAMILIES[0]);
    let inputs = cases
        .iter()
        .map(|(_, code, header)| ShaderInput {
            code,
            program_record: header,
        })
        .collect::<Vec<_>>();
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let mut translator = Translator::default();
        for _ in 0..2 {
            let results = pool.install(|| {
                translator.translate_batch_with_options_and_graphics_state(
                    shadowbox::QuietRun,
                    &inputs,
                    shadowbox::TranslationOptions::default(),
                    &state,
                )
            });
            for (((name, _, _), input), output) in cases.iter().zip(&inputs).zip(results) {
                let expected = Result::from(shadowbox::translate_with_graphics_state(
                    shadowbox::QuietRun,
                    input,
                    &state,
                ));
                assert_eq!(
                    hash_output(Result::from(output)),
                    hash_output(expected),
                    "{name}"
                );
            }
        }
    }
}
