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
            name: "plain 1.5",
            raw: 0x0c03_fc00_0007_0201,
            immediate: 0x3fc0_0000,
            modifiers: PLAIN,
        },
        Form {
            name: "negated A -0.75",
            raw: 0x0d0b_f400_0007_0201,
            immediate: 0xbf40_0000,
            modifiers: FFmaModifiers {
                product_negate: true,
                ..PLAIN
            },
        },
        Form {
            name: "negated C 2.0",
            raw: 0x0e04_0000_0007_0201,
            immediate: 0x4000_0000,
            modifiers: FFmaModifiers {
                addend_negate: true,
                ..PLAIN
            },
        },
        Form {
            name: "saturate 0.5",
            raw: 0x0c83_f000_0007_0201,
            immediate: 0x3f00_0000,
            modifiers: FFmaModifiers {
                saturate: true,
                ..PLAIN
            },
        },
        Form {
            name: "FTZ 3.0",
            raw: 0x0c24_0400_0007_0201,
            immediate: 0x4040_0000,
            modifiers: FFmaModifiers {
                flush: Flush::Ftz,
                ..PLAIN
            },
        },
        Form {
            name: "FMZ 0.0",
            raw: 0x0c40_0000_0007_0201,
            immediate: 0,
            modifiers: FFmaModifiers {
                flush: Flush::Dnz,
                ..PLAIN
            },
        },
    ]
}

#[test]
fn six_ffma32i_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for form in forms() {
        let name = form.name;
        let raw = form.raw;
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        assert_eq!(
            instruction.kind,
            InstructionKind::FFma(ops::FFma {
                dst: Register(1),
                lhs: Register(2),
                rhs: Source::Immediate(form.immediate),
                addend: Source::Register(Register(1)),
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
