//! Main-accepted synthetic forms, replayed through production translation.
#[path = "support/batch12_fixtures.rs"]
mod fixtures;
#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::spirv::Op;
use shadowbox::{interface::ShaderInput, translate};
use std::collections::BTreeMap;

#[test]
fn main_sample_fetch_and_surface_forms_translate_and_emit_image_operations() {
    let baseline: BTreeMap<_, _> = include_str!("data/families-main-d3f01ae.tsv")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| line.split_once('\t').unwrap())
        .collect();
    let cases = fixtures::cases();
    assert_eq!(cases.len(), baseline.len(), "fixture matrix changed");
    let header = fixtures::header();
    let mut counts = [0; 4];
    for (name, instructions) in cases {
        assert_eq!(
            baseline[name.as_str()],
            "PASS",
            "main must accept the fixture"
        );
        let code = fixtures::program(&instructions);
        let words = Result::from(translate(
            shadowbox::QuietRun,
            &ShaderInput {
                code: &code,
                program_record: &header,
            },
        ))
        .unwrap_or_else(|error| panic!("{name}: {error}"))
        .spirv;
        let check = spirv_validation::validate(&words);
        assert!(
            check.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&check.stderr)
        );
        let mut opcodes = Vec::new();
        let mut at = 5;
        while at < words.len() {
            opcodes.push(Op::of(words[at]));
            at += (words[at] >> 16) as usize;
        }
        // SPIR-V operation numbers: Mesa src/compiler/spirv/spirv.h:1792,1815-1827.
        // Ryujinx Spirv/Instructions.cs:1411-1445 emits ImageFetch;
        // InstEmitSurface.cs:682-690 emits ImageStore. Verify real operations
        // survive optimization rather than accepting an empty translation.
        let family = if name.starts_with("AST.") {
            assert!(
                opcodes.contains(&Op::Store),
                "{name}: missing output stores"
            );
            0
        } else if name.starts_with("TEX.Sample.") {
            assert!(
                opcodes.iter().any(|op| matches!(
                    op,
                    Op::ImageSampleImplicitLod | Op::ImageSampleExplicitLod
                )),
                "{name}: missing sample"
            );
            1
        } else if name.starts_with("TEX.Fetch.") {
            assert!(opcodes.contains(&Op::ImageFetch), "{name}: missing fetch");
            2
        } else {
            assert!(name.starts_with("SUST."));
            assert!(
                opcodes.contains(&Op::ImageWrite),
                "{name}: missing surface store"
            );
            3
        };
        counts[family] += 1;
    }
    assert_eq!(counts, [8, 84, 40, 24]);
}
