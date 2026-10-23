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
            name: "LT and PT",
            raw: 0x5098_0380_0007_0101,
            condition: CSetPCondition::decode(1),
            set_op: SetOp::And,
            accum: PT,
            result: SetResult::Mask,
        },
        Form {
            name: "GT and PT float-one",
            raw: 0x5098_1380_0007_0401,
            condition: CSetPCondition::decode(4),
            set_op: SetOp::And,
            accum: PT,
            result: SetResult::FloatOne,
        },
        Form {
            name: "EQ or !P0",
            raw: 0x5098_2400_0007_0201,
            condition: CSetPCondition::decode(2),
            set_op: SetOp::Or,
            accum: Predicate {
                register: 0,
                inverted: true,
            },
            result: SetResult::Mask,
        },
        Form {
            name: "HI xor P1",
            raw: 0x5098_4080_0007_1401,
            condition: CSetPCondition::decode(20),
            set_op: SetOp::Xor,
            accum: Predicate {
                register: 1,
                inverted: false,
            },
            result: SetResult::Mask,
        },
        Form {
            name: "NEU or !P3 float-one",
            raw: 0x5098_3580_0007_0d01,
            condition: CSetPCondition::decode(13),
            set_op: SetOp::Or,
            accum: Predicate {
                register: 3,
                inverted: true,
            },
            result: SetResult::FloatOne,
        },
        Form {
            name: "RGT and PT",
            raw: 0x5098_0380_0007_1f01,
            condition: CSetPCondition::decode(31),
            set_op: SetOp::And,
            accum: PT,
            result: SetResult::Mask,
        },
    ]
}

#[test]
fn six_cset_forms_decode_and_validate_raw_and_optimized() {
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
