#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{FloatFlags, Instruction, InstructionKind, Predicate, Register, Source},
};

struct Form {
    name: &'static str,
    raw: u64,
    rhs: Source,
    min_when: Predicate,
    abs: [bool; 2],
    negate: [bool; 2],
}

const B3: Source = Source::Register(Register(3));

const fn p(register: u8, inverted: bool) -> Predicate {
    Predicate { register, inverted }
}

fn forms() -> [Form; 6] {
    [
        Form {
            name: "register P0",
            raw: 0x5c60_0000_0037_0201,
            rhs: B3,
            min_when: p(0, false),
            abs: [false; 2],
            negate: [false; 2],
        },
        Form {
            name: "register !P1",
            raw: 0x5c60_0480_0037_0201,
            rhs: B3,
            min_when: p(1, true),
            abs: [false; 2],
            negate: [false; 2],
        },
        Form {
            name: "register |A| -B P3",
            raw: 0x5c60_6180_0037_0201,
            rhs: B3,
            min_when: p(3, false),
            abs: [true, false],
            negate: [false, true],
        },
        Form {
            name: "register -A |B| !P6",
            raw: 0x5c63_0700_0037_0201,
            rhs: B3,
            min_when: p(6, true),
            abs: [false, true],
            negate: [true, false],
        },
        Form {
            name: "cbuf !P0",
            raw: 0x4c60_0400_0017_0201,
            rhs: Source::ConstantBuffer {
                binding: 0,
                offset: 4,
            },
            min_when: p(0, true),
            abs: [false; 2],
            negate: [false; 2],
        },
        Form {
            name: "f20 immediate 1.5 P4",
            raw: 0x3860_023f_c007_0201,
            rhs: Source::Immediate(0x3fc0_0000),
            min_when: p(4, false),
            abs: [false; 2],
            negate: [false; 2],
        },
    ]
}

#[test]
fn six_fmnmx_select_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for form in forms() {
        let name = form.name;
        let raw = form.raw;
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        assert_eq!(
            instruction.kind,
            InstructionKind::FMnMxSelect(ops::FMnMxSelect {
                dst: Register(1),
                lhs: Register(2),
                rhs: form.rhs,
                min_when: form.min_when,
                flags: FloatFlags::Kept,
                abs: form.abs,
                negate: form.negate,
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
