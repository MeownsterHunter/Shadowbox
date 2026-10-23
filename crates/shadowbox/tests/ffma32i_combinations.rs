#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{FFmaModifiers, FloatFlags, Flush, Instruction, InstructionKind, Register, Source},
};

struct Form {
    name: &'static str,
    raw: u64,
    dst: u8,
    lhs: u8,
    immediate: u32,
    modifiers: FFmaModifiers,
}

const PLAIN: FFmaModifiers = FFmaModifiers {
    product_negate: false,
    addend_negate: false,
    saturate: false,
    flush: Flush::Keep,
};

fn forms() -> [Form; 6] {
    [
        Form {
            name: "negated A and C 1.5",
            raw: 0x0f03_fc00_0007_0201,
            dst: 1,
            lhs: 2,
            immediate: 0x3fc0_0000,
            modifiers: FFmaModifiers {
                product_negate: true,
                addend_negate: true,
                ..PLAIN
            },
        },
        Form {
            name: "saturate negated A 2.0",
            raw: 0x0d84_0000_0007_0201,
            dst: 1,
            lhs: 2,
            immediate: 0x4000_0000,
            modifiers: FFmaModifiers {
                product_negate: true,
                saturate: true,
                ..PLAIN
            },
        },
        Form {
            name: "saturate negated C -1.5",
            raw: 0x0e8b_fc00_0007_0201,
            dst: 1,
            lhs: 2,
            immediate: 0xbfc0_0000,
            modifiers: FFmaModifiers {
                addend_negate: true,
                saturate: true,
                ..PLAIN
            },
        },
        Form {
            name: "saturate negated A and C 0.25",
            raw: 0x0f83_e800_0007_0201,
            dst: 1,
            lhs: 2,
            immediate: 0x3e80_0000,
            modifiers: FFmaModifiers {
                product_negate: true,
                addend_negate: true,
                saturate: true,
                flush: Flush::Keep,
            },
        },
        Form {
            name: "R5 R3 4.0",
            raw: 0x0c04_0800_0007_0305,
            dst: 5,
            lhs: 3,
            immediate: 0x4080_0000,
            modifiers: PLAIN,
        },
        Form {
            name: "R7 R6 -0.0",
            raw: 0x0c08_0000_0007_0607,
            dst: 7,
            lhs: 6,
            immediate: 0x8000_0000,
            modifiers: PLAIN,
        },
    ]
}

#[test]
fn six_ffma32i_combinations_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for form in forms() {
        let name = form.name;
        let raw = form.raw;
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        assert_eq!(
            instruction.kind,
            InstructionKind::FFma(ops::FFma {
                dst: Register(form.dst),
                lhs: Register(form.lhs),
                rhs: Source::Immediate(form.immediate),
                addend: Source::Register(Register(form.dst)),
                flags: FloatFlags::Kept,
                modifiers: form.modifiers,
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
