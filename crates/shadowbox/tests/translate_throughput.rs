//! End-to-end translate throughput and reviewed reachable-code output hashes.
//! The extent rule is documented in docs/reference-sources.md
#[path = "support/fuzz_inputs.rs"]
mod fuzz_inputs;
#[path = "support/translation_inputs.rs"]
mod inputs;
#[path = "support/spirv_validation.rs"]
mod spirv_validation;
use inputs::Case;
use sha2::{Digest, Sha256};
use shadowbox::{interface::ShaderInput, translate};
use std::{
    fmt::Write,
    hint::black_box,
    time::{Duration, Instant},
};

fn run(case: &Case) -> Result<shadowbox::interface::ShaderOutput, shadowbox::interface::Stopped> {
    Result::from(translate(
        shadowbox::QuietRun,
        &ShaderInput {
            code: &case.code,
            program_record: &case.header,
        },
    ))
}

fn hash_result(
    hash: &mut Sha256,
    result: Result<shadowbox::interface::ShaderOutput, shadowbox::interface::Stopped>,
) -> bool {
    match result {
        Ok(output) => {
            hash.update([1, u8::from(output.requires_subgroup_size_32)]);
            hash.update((output.spirv.len() as u64).to_le_bytes());
            for word in output.spirv {
                hash.update(word.to_le_bytes());
            }
            true
        }
        Err(error) => {
            let error = error.to_string();
            hash.update([0]);
            hash.update((error.len() as u64).to_le_bytes());
            hash.update(error.as_bytes());
            false
        }
    }
}

fn check_hashes(file: &str, actual: &str) {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let path = root.join("docs/benchmarks").join(file);
    if std::env::var_os("SHADOWBOX_RECORD_TRANSLATE_HASHES").is_some() {
        std::fs::write(path, actual).unwrap();
    } else {
        assert_eq!(
            actual,
            std::fs::read_to_string(path).unwrap(),
            "translation differs from the reviewed reachable-code snapshot"
        );
    }
}

#[test]
fn corpus_and_synthetic_output_hashes() {
    let mut rows = String::new();
    for case in inputs::corpus().into_iter().chain(inputs::synthetic()) {
        let mut hash = Sha256::new();
        let result = run(&case);
        if let Ok(output) = &result {
            let check = spirv_validation::validate(&output.spirv);
            assert!(
                check.status.success(),
                "{}: {}",
                case.name,
                String::from_utf8_lossy(&check.stderr)
            );
        }
        let accepted = hash_result(&mut hash, result);
        assert!(
            accepted || !case.name.starts_with("synthetic_"),
            "{} must translate",
            case.name
        );
        writeln!(
            rows,
            "{}\t{}\t{:x}",
            case.name,
            if accepted { "ok" } else { "err" },
            hash.finalize()
        )
        .unwrap();
    }
    check_hashes("translate-reachable-outputs.tsv", &rows);
}

#[test]
#[ignore = "full eight-sample fuzz output identity gate"]
fn fuzz_output_hash() {
    let mut rng = fuzz_inputs::Rng(0x5eed_5eed_5eed_5eed);
    let headers = fuzz_inputs::fuzz_headers();
    let state = fuzz_inputs::graphics_state();
    let mut hash = Sha256::new();
    let mut accepted = 0;
    for prefix in 0..=0xffff_u64 {
        for _ in 0..8 {
            let raw = prefix << 48 | (rng.next() & 0xffff_ffff_ffff);
            let code = fuzz_inputs::fuzz_program(raw);
            for header in &headers {
                hash.update(raw.to_le_bytes());
                accepted += usize::from(hash_result(
                    &mut hash,
                    Result::from(shadowbox::translate_with_graphics_state(
                        shadowbox::QuietRun,
                        &ShaderInput {
                            code: &code,
                            program_record: header,
                        },
                        &state,
                    )),
                ));
            }
        }
    }
    let row = format!(
        "samples=8 seed=5eed5eed5eed5eed accepted={accepted} sha256={:x}\n",
        hash.finalize()
    );
    eprint!("{row}");
    check_hashes("translate-reachable-fuzz.sha256", &row);
}

#[test]
#[ignore = "manual translate throughput benchmark"]
fn translate_throughput() {
    let corpus = inputs::corpus();
    let (accepted, rejected): (Vec<_>, Vec<_>) =
        corpus.into_iter().partition(|case| run(case).is_ok());
    let mut groups = vec![
        ("corpus_accepted".into(), accepted),
        ("corpus_rejected".into(), rejected),
    ];
    groups.extend(
        inputs::synthetic()
            .into_iter()
            .map(|case| (case.name.clone(), vec![case])),
    );
    let filter = std::env::var("SHADOWBOX_BENCH_FILTER").unwrap_or_default();
    let samples = std::env::var("SHADOWBOX_BENCH_SAMPLES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(7);
    let ms = std::env::var("SHADOWBOX_BENCH_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);
    println!("workload\tshaders\tsamples\tmedian_shaders_per_second\tmin\tmax");
    for (name, cases) in groups {
        if !name.contains(&filter) {
            continue;
        }
        if cases.is_empty() {
            println!("{name}\t0\t0\t0.00\t0.00\t0.00");
            continue;
        }
        for case in &cases {
            black_box(run(case)).ok();
        }
        let mut rates = Vec::new();
        for _ in 0..samples {
            let start = Instant::now();
            let mut rounds = 0;
            loop {
                for case in &cases {
                    black_box(run(black_box(case))).ok();
                }
                rounds += 1;
                if start.elapsed() >= Duration::from_millis(ms) {
                    break;
                }
            }
            rates.push((rounds * cases.len()) as f64 / start.elapsed().as_secs_f64());
        }
        rates.sort_by(f64::total_cmp);
        println!(
            "{name}\t{}\t{samples}\t{:.2}\t{:.2}\t{:.2}",
            cases.len(),
            rates[samples / 2],
            rates[0],
            rates[samples - 1]
        );
    }
}

#[test]
fn synthetic_programs_validate() {
    let dir = std::env::temp_dir().join(format!(
        "shadowbox-translate-validation-{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    for case in inputs::synthetic() {
        let output = run(&case).unwrap();
        let path = dir.join(format!("{}.spv", case.name));
        std::fs::write(
            &path,
            output
                .spirv
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let result = std::process::Command::new("spirv-val")
            .args(["--target-env", "vulkan1.2"])
            .arg(path)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}: {}",
            case.name,
            String::from_utf8_lossy(&result.stderr)
        );
    }
    std::fs::remove_dir_all(dir).unwrap();
}
