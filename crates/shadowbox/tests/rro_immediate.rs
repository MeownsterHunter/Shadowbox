#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{Instruction, InstructionKind, Predicate, Register, Source},
};

struct Form {
    name: &'static str,
    raw: u64,
    value: u32,
    abs: bool,
    negate: bool,
    ex2: bool,
}

fn forms() -> [Form; 6] {
    [
        Form {
            name: "SINCOS 1.0",
            raw: 0x3890_003f_8007_0003,
            value: 0x3f80_0000,
            abs: false,
            negate: false,
            ex2: false,
        },
        Form {
            name: "EX2 -2.5",
            raw: 0x3990_00c0_2007_0003,
            value: 0xc020_0000,
            abs: false,
            negate: false,
            ex2: true,
        },
        Form {
            name: "SINCOS |-3.0|",
            raw: 0x3992_0040_4007_0003,
            value: 0xc040_0000,
            abs: true,
            negate: false,
            ex2: false,
        },
        Form {
            name: "EX2 -(0.5)",
            raw: 0x3890_20bf_0007_0003,
            value: 0x3f00_0000,
            abs: false,
            negate: true,
            ex2: true,
        },
        Form {
            name: "SINCOS -|+inf|",
            raw: 0x3892_207f_8007_0003,
            value: 0x7f80_0000,
            abs: true,
            negate: true,
            ex2: false,
        },
        Form {
            name: "EX2 -|-qNaN|",
            raw: 0x3992_20ff_c007_0003,
            value: 0xffc0_0000,
            abs: true,
            negate: true,
            ex2: true,
        },
    ]
}

#[test]
fn six_rro_immediate_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for form in forms() {
        let name = form.name;
        let raw = form.raw;
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        assert_eq!(
            instruction.kind,
            InstructionKind::Rro(ops::Rro {
                dst: Register(3),
                src: Source::Immediate(form.value),
                abs: form.abs,
                negate: form.negate,
                ex2: form.ex2,
            }),
            "{name}"
        );
        // Unguarded, and guarded by an inverted predicate.
        for predicate in [
            None,
            Some(Predicate {
                register: 1,
                inverted: true,
            }),
        ] {
            let mut module = emit_unoptimized_with_header(
                &[
                    Instruction {
                        predicate,
                        ..instruction
                    },
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
}
