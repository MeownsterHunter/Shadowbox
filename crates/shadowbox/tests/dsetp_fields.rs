#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{
        DoubleSource, FloatCompare, Instruction, InstructionKind, Predicate, Register,
        RegisterPair, Relation, SetOp,
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
}

const P7: Predicate = Predicate {
    register: 7,
    inverted: false,
};
const B4: DoubleSource = DoubleSource::RegisterPair(RegisterPair(Register(4)));

fn forms() -> [Form; 6] {
    [
        Form {
            name: "register GT or !P0",
            raw: 0x5b84_2400_0047_0201,
            rhs: B4,
            compare: FloatCompare::Ordered(Relation::Gt),
            set_op: SetOp::Or,
            accum: Predicate {
                register: 0,
                inverted: true,
            },
            abs: [false; 2],
            negate: [false; 2],
        },
        Form {
            name: "register LTU xor P1",
            raw: 0x5b89_4080_0047_0201,
            rhs: B4,
            compare: FloatCompare::Unordered(Relation::Lt),
            set_op: SetOp::Xor,
            accum: Predicate {
                register: 1,
                inverted: false,
            },
            abs: [false; 2],
            negate: [false; 2],
        },
        Form {
            name: "register LT negated A absolute B",
            raw: 0x5b81_1b80_0047_0201,
            rhs: B4,
            compare: FloatCompare::Ordered(Relation::Lt),
            set_op: SetOp::And,
            accum: P7,
            abs: [false, true],
            negate: [true, false],
        },
        Form {
            name: "register NEU negated absolute A and B",
            raw: 0x5b8d_1b80_0047_02c1,
            rhs: B4,
            compare: FloatCompare::Unordered(Relation::Ne),
            set_op: SetOp::And,
            accum: P7,
            abs: [true; 2],
            negate: [true; 2],
        },
        Form {
            name: "cbuf GEU negated A or !P2",
            raw: 0x4b8e_2d00_0047_0201,
            rhs: DoubleSource::ConstantBuffer {
                binding: 0,
                offset: 16,
            },
            compare: FloatCompare::Unordered(Relation::Ge),
            set_op: SetOp::Or,
            accum: Predicate {
                register: 2,
                inverted: true,
            },
            abs: [false; 2],
            negate: [true, false],
        },
        Form {
            name: "immediate GTU negated absolute A xor !P1",
            raw: 0x368c_4cbf_f007_0281,
            rhs: DoubleSource::Immediate(0x3ff0_0000),
            compare: FloatCompare::Unordered(Relation::Gt),
            set_op: SetOp::Xor,
            accum: Predicate {
                register: 1,
                inverted: true,
            },
            abs: [true, false],
            negate: [true, false],
        },
    ]
}

#[test]
fn six_dsetp_field_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for form in forms() {
        let name = form.name;
        let raw = form.raw;
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        assert_eq!(
            instruction.kind,
            InstructionKind::DoubleSetP(ops::DoubleSetP {
                dst: [0, 1],
                lhs: RegisterPair(Register(2)),
                rhs: form.rhs,
                compare: form.compare,
                set_op: form.set_op,
                accum: form.accum,
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
