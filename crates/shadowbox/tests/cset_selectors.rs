#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{CSetPCondition, Instruction, InstructionKind, Predicate, Register, SetOp, SetResult},
};

struct Form {
    name: &'static str,
    raw: u64,
    condition: CSetPCondition,
    set_op: SetOp,
    accum: Predicate,
    result: SetResult,
}

const PT: Predicate = Predicate {
    register: 7,
    inverted: false,
};

fn forms() -> [Form; 6] {
    [
        Form {
            name: "LE and P2",
            raw: 0x5098_0100_0007_0301,
            condition: CSetPCondition::decode(3),
            set_op: SetOp::And,
            accum: Predicate {
                register: 2,
                inverted: false,
            },
            result: SetResult::Mask,
        },
        Form {
            name: "NE or PT float-one",
            raw: 0x5098_3380_0007_0501,
            condition: CSetPCondition::decode(5),
            set_op: SetOp::Or,
            accum: PT,
            result: SetResult::FloatOne,
        },
        Form {
            name: "GE xor !P1",
            raw: 0x5098_4480_0007_0601,
            condition: CSetPCondition::decode(6),
            set_op: SetOp::Xor,
            accum: Predicate {
                register: 1,
                inverted: true,
            },
            result: SetResult::Mask,
        },
        Form {
            name: "LTU and PT float-one",
            raw: 0x5098_1380_0007_0901,
            condition: CSetPCondition::decode(9),
            set_op: SetOp::And,
            accum: PT,
            result: SetResult::FloatOne,
        },
        Form {
            name: "LO or !P0",
            raw: 0x5098_2400_0007_1101,
            condition: CSetPCondition::decode(17),
            set_op: SetOp::Or,
            accum: Predicate {
                register: 0,
                inverted: true,
            },
            result: SetResult::Mask,
        },
        Form {
            name: "RLE and P5",
            raw: 0x5098_0280_0007_1e01,
            condition: CSetPCondition::decode(30),
            set_op: SetOp::And,
            accum: Predicate {
                register: 5,
                inverted: false,
            },
            result: SetResult::Mask,
        },
    ]
}

#[test]
fn six_cset_selector_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for form in forms() {
        let name = form.name;
        let raw = form.raw;
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        assert_eq!(
            instruction.kind,
            InstructionKind::CSet(ops::CSet {
                dst: Register(1),
                condition: form.condition,
                set_op: form.set_op,
                accum: form.accum,
                result: form.result,
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
