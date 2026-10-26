//! Frozen complete translation results for the speed work.
#[path = "support/translation_inputs.rs"]
mod inputs;
#[path = "support/speed3_inputs.rs"]
mod speed3_inputs;
#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use sha2::{Digest, Sha256};
use shadowbox::{interface::ShaderInput, translate};
use std::fmt::Write;

#[test]
fn semantic_checkpoint_outputs_remain_byte_identical_after_speed_merge() {
    let mut rows = String::new();
    let mut synthetic = String::new();
    for case in inputs::corpus()
        .into_iter()
        .chain(inputs::synthetic())
        .chain(speed3_inputs::synthetic())
    {
        let result = Result::from(translate(
            shadowbox::QuietRun,
            &ShaderInput {
                code: &case.code,
                program_record: &case.header,
            },
        ));
        let mut hash = Sha256::new();
        match &result {
            Ok(output) => {
                let check = spirv_validation::validate(&output.spirv);
                assert!(
                    check.status.success(),
                    "{}: {}",
                    case.name,
                    String::from_utf8_lossy(&check.stderr)
                );
                hash.update([u8::from(output.requires_subgroup_size_32)]);
                hash.update((output.spirv.len() as u64).to_le_bytes());
                for word in &output.spirv {
                    hash.update(word.to_le_bytes());
                }
                let digest = hash.clone().finalize();
                if case.name.starts_with("mixed_") {
                    writeln!(synthetic, "{}\t{:x}", case.name, digest).unwrap();
                }
                // Surface bindings are part of the public result too.
                hash.update(format!("{:?}", output.surface_descriptors));
            }
            Err(error) => {
                hash.update(error.to_string());
            }
        }
        writeln!(
            rows,
            "{}\t{}\t{:x}",
            case.name,
            if result.is_ok() { "ok" } else { "err" },
            hash.finalize()
        )
        .unwrap();
    }
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    for (file, actual) in [
        ("translate-before-speed.tsv", rows),
        ("xlate-speed3-hashes.tsv", synthetic),
    ] {
        let path = root.join("docs/benchmarks").join(file);
        if std::env::var_os("SHADOWBOX_RECORD_TRANSLATE_HASHES").is_some() {
            std::fs::write(&path, &actual).unwrap();
        }
        assert_eq!(
            actual,
            std::fs::read_to_string(path).unwrap(),
            "speed integration changed the frozen semantic checkpoint"
        );
    }
}
