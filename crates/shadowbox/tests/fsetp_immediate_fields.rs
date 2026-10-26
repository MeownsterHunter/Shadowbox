#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{FloatCompare, Instruction, InstructionKind, Register, Relation, SetOp, Source},
};

struct Form {
    name: &'static str,
    raw: u64,
    rhs: Source,
    compare: FloatCompare,
    set_op: SetOp,
    accum: u8,
    accum_negate: bool,
    abs: [bool; 2],
    negate: [bool; 2],
}

const ONE: Source = Source::Immediate(0x3f80_0000);
const MINUS_TWO: Source = Source::Immediate(0xc000_0000);

fn forms() -> [Form; 6] {
    [
        Form {
            name: "GT negated 1.0",
            raw: 0x36b4_03bf_8007_0247,
            rhs: ONE,
            compare: FloatCompare::Ordered(Relation::Gt),
            set_op: SetOp::And,
            accum: 7,
            accum_negate: false,
            abs: [false; 2],
            negate: [false, true],
        },
        Form {
            name: "LT absolute -2.0",
            raw: 0x37b1_13c0_0007_0207,
            rhs: MINUS_TWO,
            compare: FloatCompare::Ordered(Relation::Lt),
            set_op: SetOp::And,
            accum: 7,
            accum_negate: false,
            abs: [false, true],
            negate: [false; 2],
        },
        Form {
            name: "GE negated absolute -0.5",
            raw: 0x37b6_13bf_0007_0247,
            rhs: Source::Immediate(0xbf00_0000),
            compare: FloatCompare::Ordered(Relation::Ge),
            set_op: SetOp::And,
            accum: 7,
            accum_negate: false,
            abs: [false, true],
            negate: [false, true],
        },
        Form {
            name: "EQ negated 1.0 or !P1",
            raw: 0x36b2_24bf_8007_0247,
            rhs: ONE,
            compare: FloatCompare::Ordered(Relation::Eq),
            set_op: SetOp::Or,
            accum: 1,
            accum_negate: true,
            abs: [false; 2],
            negate: [false, true],
        },
        Form {
            name: "NEU -|A| negated 1.0",
            raw: 0x36bd_0bbf_8007_02c7,
            rhs: ONE,
            compare: FloatCompare::Unordered(Relation::Ne),
            set_op: SetOp::And,
            accum: 7,
            accum_negate: false,
            abs: [true, false],
            negate: [true, true],
        },
        Form {
            name: "LE |A| absolute -2.0 xor P2",
            raw: 0x37b3_5140_0007_0287,
            rhs: MINUS_TWO,
            compare: FloatCompare::Ordered(Relation::Le),
            set_op: SetOp::Xor,
            accum: 2,
            accum_negate: false,
            abs: [true, true],
            negate: [false; 2],
        },
    ]
}

#[test]
fn six_fsetp_immediate_field_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for form in forms() {
        let name = form.name;
        let raw = form.raw;
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        assert_eq!(
            instruction.kind,
            InstructionKind::FSetP(ops::FSetP {
                dst: [0, 7],
                lhs: Register(2),
                rhs: form.rhs,
                compare: form.compare,
                set_op: form.set_op,
                accum: form.accum,
                accum_negate: form.accum_negate,
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
