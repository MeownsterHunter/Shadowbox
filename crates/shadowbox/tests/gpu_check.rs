#![cfg(feature = "gpu-check")]

#[path = "support/fragment.rs"]
mod fragment;
use shadowbox::interface::ShaderInput;
use std::{fs, path::Path};

#[test]
fn constant_color_matches_reference_on_radv() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let temp = std::env::temp_dir().join(format!("shadowbox-reference-{}", std::process::id()));
    fs::create_dir_all(&temp).unwrap();
    let reference = fragment::compile(
        &fs::read_to_string(root.join("corpus/constant_color.frag")).unwrap(),
        "frag",
        &temp.join("reference"),
    );
    let code = fs::read(root.join("corpus/constant_color.bin")).unwrap();
    let header = fs::read(root.join("corpus/constant_color.hdr")).unwrap();
    let translated = Result::from(shadowbox::translate(
        shadowbox::QuietRun,
        &ShaderInput {
            code: &code,
            program_record: &header,
        },
    ))
    .unwrap()
    .spirv;
    let as_words = |bytes: &[u8]| {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect::<Vec<_>>()
    };
    let decoded = shadowbox::decoder::decode_stream(shadowbox::QuietRun, &as_words(&code));
    let parsed_header = shadowbox::headers::parse(&as_words(&header)).unwrap();
    let unoptimized =
        shadowbox::emitter::emit_unoptimized_with_header(&decoded, Some(&parsed_header))
            .unwrap()
            .words;

    fragment::assert_pixels(
        &[
            ("GLSL reference", reference),
            ("optimized", translated),
            ("raw", unoptimized),
        ],
        [1.0, 0.25, 0.5, 1.0],
    );
    fs::remove_dir_all(temp).unwrap();
}

#[path = "support/pipeline_bench.rs"]
mod pipeline_bench;
