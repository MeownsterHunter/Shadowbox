#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{
        Instruction, InstructionKind, IntCompare, IntegerCompareMode, Predicate, Register,
        Relation, SetOp, Signedness, Source,
    },
};

struct Form {
    name: &'static str,
    raw: u64,
    dst: [u8; 2],
    rhs: Source,
    compare: Relation,
    signedness: Signedness,
    mode: IntegerCompareMode,
    set_op: SetOp,
    accum: u8,
    accum_negate: bool,
}

const B4: Source = Source::Register(Register(4));

fn forms() -> [Form; 6] {
    [
        Form {
            name: "register U32 LT P0 P1",
            raw: 0x5b62_0380_0047_0201,
            dst: [0, 1],
            rhs: B4,
            compare: Relation::Lt,
            signedness: Signedness::Unsigned,
            mode: IntegerCompareMode::Ordinary,
            set_op: SetOp::And,
            accum: 7,
            accum_negate: false,
        },
        Form {
            name: "register S32 GE or !P0",
            raw: 0x5b6d_2400_0047_020a,
            dst: [1, 2],
            rhs: B4,
            compare: Relation::Ge,
            signedness: Signedness::Signed,
            mode: IntegerCompareMode::Ordinary,
            set_op: SetOp::Or,
            accum: 0,
            accum_negate: true,
        },
        Form {
            name: "register S32 NE.X xor P2",
            raw: 0x5b6b_4900_0047_0218,
            dst: [3, 0],
            rhs: B4,
            compare: Relation::Ne,
            signedness: Signedness::Signed,
            mode: IntegerCompareMode::Extended,
            set_op: SetOp::Xor,
            accum: 2,
            accum_negate: false,
        },
        Form {
            name: "cbuf U32 GT P2 P5",
            raw: 0x4b68_0380_0047_0215,
            dst: [2, 5],
            rhs: Source::ConstantBuffer {
                binding: 0,
                offset: 16,
            },
            compare: Relation::Gt,
            signedness: Signedness::Unsigned,
            mode: IntegerCompareMode::Ordinary,
            set_op: SetOp::And,
            accum: 7,
            accum_negate: false,
        },
        Form {
            name: "immediate S32 LE -5 and !P1",
            raw: 0x3767_04ff_ffb7_0203,
            dst: [0, 3],
            rhs: Source::Immediate(-5_i32 as u32),
            compare: Relation::Le,
            signedness: Signedness::Signed,
            mode: IntegerCompareMode::Ordinary,
            set_op: SetOp::And,
            accum: 1,
            accum_negate: true,
        },
        Form {
            name: "immediate U32 EQ 7 or P3, PT P6",
            raw: 0x3664_2180_0077_023e,
            dst: [7, 6],
            rhs: Source::Immediate(7),
            compare: Relation::Eq,
            signedness: Signedness::Unsigned,
            mode: IntegerCompareMode::Ordinary,
            set_op: SetOp::Or,
            accum: 3,
            accum_negate: false,
        },
    ]
}

#[test]
fn six_isetp_second_destination_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for form in forms() {
        let name = form.name;
        let raw = form.raw;
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        assert_eq!(
            instruction.kind,
            InstructionKind::ISetP(ops::ISetP {
                dst: form.dst,
                lhs: Register(2),
                rhs: form.rhs,
                compare: IntCompare::Relation(form.compare),
                signedness: form.signedness,
                mode: form.mode,
                set_op: form.set_op,
                accum: form.accum,
                accum_negate: form.accum_negate,
                abs: [false; 2],
                negate: [false; 2],
            }),
            "{name}"
        );

        // Unguarded, and guarded by a predicate the instruction overwrites.
        let written = if form.dst[0] == 7 {
            form.dst[1]
        } else {
            form.dst[0]
        };
        for predicate in [
            None,
            Some(Predicate {
                register: written,
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
