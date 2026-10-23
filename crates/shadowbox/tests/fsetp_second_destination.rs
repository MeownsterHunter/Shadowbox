#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{
        FloatCompare, Instruction, InstructionKind, Predicate, Register, Relation, SetOp, Source,
    },
};

struct Form {
    name: &'static str,
    raw: u64,
    dst: [u8; 2],
    rhs: Source,
    compare: FloatCompare,
    set_op: SetOp,
    accum: u8,
    accum_negate: bool,
    abs: [bool; 2],
    negate: [bool; 2],
}

const B4: Source = Source::Register(Register(4));

fn forms() -> [Form; 6] {
    [
        Form {
            name: "register GT P0 P1",
            raw: 0x5bb4_0380_0047_0201,
            dst: [0, 1],
            rhs: B4,
            compare: FloatCompare::Ordered(Relation::Gt),
            set_op: SetOp::And,
            accum: 7,
            accum_negate: false,
            abs: [false; 2],
            negate: [false; 2],
        },
        Form {
            name: "register LTU or !P0",
            raw: 0x5bb9_2400_0047_020a,
            dst: [1, 2],
            rhs: B4,
            compare: FloatCompare::Unordered(Relation::Lt),
            set_op: SetOp::Or,
            accum: 0,
            accum_negate: true,
            abs: [false; 2],
            negate: [false; 2],
        },
        Form {
            name: "register LE -|A| |B| xor P2",
            raw: 0x5bb3_5900_0047_0298,
            dst: [3, 0],
            rhs: B4,
            compare: FloatCompare::Ordered(Relation::Le),
            set_op: SetOp::Xor,
            accum: 2,
            accum_negate: false,
            abs: [true, true],
            negate: [true, false],
        },
        Form {
            name: "cbuf NE negated B",
            raw: 0x4bb5_0380_0047_0255,
            dst: [2, 5],
            rhs: Source::ConstantBuffer {
                binding: 0,
                offset: 16,
            },
            compare: FloatCompare::Ordered(Relation::Ne),
            set_op: SetOp::And,
            accum: 7,
            accum_negate: false,
            abs: [false; 2],
            negate: [false, true],
        },
        Form {
            name: "immediate GE 1.0 and !P1",
            raw: 0x36b6_04bf_8007_0203,
            dst: [0, 3],
            rhs: Source::Immediate(0x3f80_0000),
            compare: FloatCompare::Ordered(Relation::Ge),
            set_op: SetOp::And,
            accum: 1,
            accum_negate: true,
            abs: [false; 2],
            negate: [false; 2],
        },
        Form {
            name: "immediate NEU -2.0 |A| or P3",
            raw: 0x37bd_21c0_0007_02a6,
            dst: [4, 6],
            rhs: Source::Immediate(0xc000_0000),
            compare: FloatCompare::Unordered(Relation::Ne),
            set_op: SetOp::Or,
            accum: 3,
            accum_negate: false,
            abs: [true, false],
            negate: [false; 2],
        },
    ]
}

#[test]
fn six_fsetp_second_destination_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for form in forms() {
        let name = form.name;
        let raw = form.raw;
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        assert_eq!(
            instruction.kind,
            InstructionKind::FSetP(ops::FSetP {
                dst: form.dst,
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

        // Unguarded, and guarded by a predicate the instruction overwrites.
        for predicate in [
            None,
            Some(Predicate {
                register: form.dst[0],
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
