#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{Conversion, F2fSource, Instruction, InstructionKind, Register, Source},
};

struct Form {
    name: &'static str,
    raw: u64,
    src: Source,
    abs: bool,
    negate: bool,
    saturate: bool,
}

fn forms() -> [Form; 6] {
    let r2 = Source::Register(Register(2));
    [
        Form {
            name: "R2.H0",
            raw: 0x5ca8_0000_0027_0601,
            src: r2,
            abs: false,
            negate: false,
            saturate: false,
        },
        Form {
            name: "negated",
            raw: 0x5ca8_2000_0027_0601,
            src: r2,
            abs: false,
            negate: true,
            saturate: false,
        },
        Form {
            name: "absolute",
            raw: 0x5caa_0000_0027_0601,
            src: r2,
            abs: true,
            negate: false,
            saturate: false,
        },
        Form {
            name: "absolute negated",
            raw: 0x5caa_2000_0027_0601,
            src: r2,
            abs: true,
            negate: true,
            saturate: false,
        },
        Form {
            name: "saturated",
            raw: 0x5cac_0000_0027_0601,
            src: r2,
            abs: false,
            negate: false,
            saturate: true,
        },
        Form {
            name: "cbuf negated",
            raw: 0x4ca8_2000_0037_0601,
            src: Source::ConstantBuffer {
                binding: 0,
                offset: 12,
            },
            abs: false,
            negate: true,
            saturate: false,
        },
    ]
}

#[test]
fn six_f2f_half_source_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for form in forms() {
        let name = form.name;
        let raw = form.raw;
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        assert_eq!(
            instruction.kind,
            InstructionKind::Convert(ops::Convert {
                kind: Conversion::F2F {
                    source: F2fSource::Float16Low,
                    destination: shadowbox::ir::F2fDestination::Float32,
                    integer_round: None,
                    abs: form.abs,
                    negate: form.negate,
                    saturate: form.saturate,
                    ftz: false,
                },
                dst: Register(1),
                src: form.src,
            }),
            "{name}"
        );
        let mut module = emit_unoptimized_with_header(
            &[
                instruction,
                Instruction {
                    address: 16,
                    predicate: None,
                    kind: InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
                },
            ],
            Some(&header),
        )
        .unwrap();
        let raw_check = spirv_validation::validate(&module.words);
        assert!(
            raw_check.status.success(),
            "{name} raw: {}",
            String::from_utf8_lossy(&raw_check.stderr)
        );
        module.optimize();
        let optimized_check = spirv_validation::validate(&module.words);
        assert!(
            optimized_check.status.success(),
            "{name} optimized: {}",
            String::from_utf8_lossy(&optimized_check.stderr)
        );
    }
}
