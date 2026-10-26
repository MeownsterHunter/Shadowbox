#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{
        DoubleSetResult, DoubleSource, FloatCompare, FloatFlags, Instruction, InstructionKind,
        Predicate, Register, RegisterPair, Relation, SetOp, SetResult,
    },
};

struct Form {
    name: &'static str,
    raw: u64,
    rhs: DoubleSource,
    compare: FloatCompare,
    set_op: SetOp,
    accum: Predicate,
    abs: [bool; 2],
    negate: [bool; 2],
    result: SetResult,
}

const P7: Predicate = Predicate {
    register: 7,
    inverted: false,
};
const B4: DoubleSource = DoubleSource::RegisterPair(RegisterPair(Register(4)));

fn forms() -> [Form; 6] {
    [
        Form {
            name: "register GT float-one",
            raw: 0x5914_0380_0047_0201,
            rhs: B4,
            compare: FloatCompare::Ordered(Relation::Gt),
            set_op: SetOp::And,
            accum: P7,
            abs: [false; 2],
            negate: [false; 2],
            result: SetResult::FloatOne,
        },
        Form {
            name: "register GE or !P0",
            raw: 0x5906_2400_0047_0201,
            rhs: B4,
            compare: FloatCompare::Ordered(Relation::Ge),
            set_op: SetOp::Or,
            accum: Predicate {
                register: 0,
                inverted: true,
            },
            abs: [false; 2],
            negate: [false; 2],
            result: SetResult::Mask,
        },
        Form {
            name: "register LTU xor P1",
            raw: 0x5909_4080_0047_0201,
            rhs: B4,
            compare: FloatCompare::Unordered(Relation::Lt),
            set_op: SetOp::Xor,
            accum: Predicate {
                register: 1,
                inverted: false,
            },
            abs: [false; 2],
            negate: [false; 2],
            result: SetResult::Mask,
        },
        Form {
            name: "register LT negated A absolute B",
            raw: 0x5901_1b80_0047_0201,
            rhs: B4,
            compare: FloatCompare::Ordered(Relation::Lt),
            set_op: SetOp::And,
            accum: P7,
            abs: [false, true],
            negate: [true, false],
            result: SetResult::Mask,
        },
        Form {
            name: "register EQU absolute A negated B",
            raw: 0x596a_0380_0047_0201,
            rhs: B4,
            compare: FloatCompare::Unordered(Relation::Eq),
            set_op: SetOp::And,
            accum: P7,
            abs: [true, false],
            negate: [false, true],
            result: SetResult::Mask,
        },
        Form {
            name: "cbuf GEU negated A float-one",
            raw: 0x491e_0b80_0047_0201,
            rhs: DoubleSource::ConstantBuffer {
                binding: 0,
                offset: 16,
            },
            compare: FloatCompare::Unordered(Relation::Ge),
            set_op: SetOp::And,
            accum: P7,
            abs: [false; 2],
            negate: [true, false],
            result: SetResult::FloatOne,
        },
    ]
}

#[test]
fn six_dset_field_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for form in forms() {
        let name = form.name;
        let raw = form.raw;
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        assert_eq!(
            instruction.kind,
            InstructionKind::DoubleSet(ops::DoubleSet {
                dst: Register(1),
                lhs: RegisterPair(Register(2)),
                rhs: form.rhs,
                compare: form.compare,
                set_op: form.set_op,
                accum: form.accum,
                abs: form.abs,
                negate: form.negate,
                result: match form.result {
                    SetResult::Mask => DoubleSetResult::Mask(FloatFlags::Kept),
                    SetResult::FloatOne => DoubleSetResult::FloatOne(FloatFlags::Kept),
                },
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
