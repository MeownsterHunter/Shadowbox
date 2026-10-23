#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{Conversion, F2fSource, Instruction, InstructionKind, Register, RoundingMode, Source},
};

struct Form {
    name: &'static str,
    raw: u64,
    immediate: u32,
    integer_round: Option<RoundingMode>,
    abs: bool,
    negate: bool,
    saturate: bool,
    ftz: bool,
}

fn forms() -> [Form; 6] {
    [
        Form {
            name: "pass 1.5",
            raw: 0x38a8_003f_c007_0a01,
            immediate: 1.5f32.to_bits(),
            integer_round: None,
            abs: false,
            negate: false,
            saturate: false,
            ftz: false,
        },
        Form {
            name: "floor -2.5",
            raw: 0x39a8_04c0_2007_0a01,
            immediate: (-2.5f32).to_bits(),
            integer_round: Some(RoundingMode::NegativeInfinity),
            abs: false,
            negate: false,
            saturate: false,
            ftz: false,
        },
        Form {
            name: "ceil negated 2.5",
            raw: 0x38a8_2540_2007_0a01,
            immediate: 2.5f32.to_bits(),
            integer_round: Some(RoundingMode::PositiveInfinity),
            abs: false,
            negate: true,
            saturate: false,
            ftz: false,
        },
        Form {
            name: "round absolute -2.5",
            raw: 0x39aa_0440_2007_0a01,
            immediate: (-2.5f32).to_bits(),
            integer_round: Some(RoundingMode::NearestEven),
            abs: true,
            negate: false,
            saturate: false,
            ftz: false,
        },
        Form {
            name: "trunc saturated 1.5",
            raw: 0x38ac_05bf_c007_0a01,
            immediate: 1.5f32.to_bits(),
            integer_round: Some(RoundingMode::Zero),
            abs: false,
            negate: false,
            saturate: true,
            ftz: false,
        },
        Form {
            name: "FTZ subnormal",
            raw: 0x38a8_1000_4007_0a01,
            immediate: 0x0040_0000,
            integer_round: None,
            abs: false,
            negate: false,
            saturate: false,
            ftz: true,
        },
    ]
}

#[test]
fn six_f2f_immediate_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for form in forms() {
        let name = form.name;
        let raw = form.raw;
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        assert_eq!(
            instruction.kind,
            InstructionKind::Convert(ops::Convert {
                kind: Conversion::F2F {
                    source: F2fSource::Float32,
                    destination: shadowbox::ir::F2fDestination::Float32,
                    integer_round: form.integer_round,
                    abs: form.abs,
                    negate: form.negate,
                    saturate: form.saturate,
                    ftz: form.ftz,
                },
                dst: Register(1),
                src: Source::Immediate(form.immediate),
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
