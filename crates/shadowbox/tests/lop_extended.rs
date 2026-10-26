#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{Instruction, InstructionKind, LogicOp, PredicateOp, PredicateOutput, Register, Source},
};

struct Form {
    name: &'static str,
    raw: u64,
    rhs: Source,
    op: LogicOp,
    invert: [bool; 2],
    predicate_output: Option<PredicateOutput>,
    extended: bool,
    cc: bool,
}

fn forms() -> [Form; 6] {
    [
        Form {
            name: "LOP.AND.X.CC register",
            raw: 0x5c47_8800_0037_0201,
            rhs: Source::Register(Register(3)),
            op: LogicOp::And,
            invert: [false; 2],
            predicate_output: None,
            extended: true,
            cc: true,
        },
        Form {
            name: "LOP.OR.X.CC c[1][8]",
            raw: 0x4c47_8a04_0027_0201,
            rhs: Source::ConstantBuffer {
                binding: 1,
                offset: 8,
            },
            op: LogicOp::Or,
            invert: [false; 2],
            predicate_output: None,
            extended: true,
            cc: true,
        },
        Form {
            name: "LOP.XOR.X.CC ~A immediate",
            raw: 0x3847_8c92_3457_0201,
            rhs: Source::Immediate(0x12345),
            op: LogicOp::Xor,
            invert: [true, false],
            predicate_output: None,
            extended: true,
            cc: true,
        },
        Form {
            name: "LOP.PASS_B.X.CC.Z P0 ~B",
            raw: 0x5c40_af00_0037_0201,
            rhs: Source::Register(Register(3)),
            op: LogicOp::PassB,
            invert: [false, true],
            predicate_output: Some(PredicateOutput {
                op: PredicateOp::Z,
                dst: 0,
            }),
            extended: true,
            cc: true,
        },
        Form {
            name: "LOP32I.PASS_B",
            raw: 0x0461_2345_6787_0201,
            rhs: Source::Immediate(0x1234_5678),
            op: LogicOp::PassB,
            invert: [false; 2],
            predicate_output: None,
            extended: false,
            cc: false,
        },
        Form {
            name: "LOP32I.AND.X.CC ~B",
            raw: 0x0710_000f_fff7_0201,
            rhs: Source::Immediate(0xffff),
            op: LogicOp::And,
            invert: [false, true],
            predicate_output: None,
            extended: true,
            cc: true,
        },
    ]
}

#[test]
fn six_lop_extended_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for form in forms() {
        let name = form.name;
        let raw = form.raw;
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        assert_eq!(
            instruction.kind,
            InstructionKind::Lop(ops::Lop {
                dst: Register(1),
                lhs: Register(2),
                rhs: form.rhs,
                op: form.op,
                invert: form.invert,
                predicate_output: form.predicate_output,
                extended: form.extended,
                cc: form.cc,
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
