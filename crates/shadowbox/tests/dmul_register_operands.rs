#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{DoubleProductSign, DoubleSource, Instruction, InstructionKind, Register, RegisterPair},
};

fn forms() -> [(u64, Register, Register, Register, DoubleProductSign); 5] {
    [
        (
            0x5c80_0000_0007_0000,
            Register(0),
            Register(0),
            Register(0),
            DoubleProductSign::Positive,
        ),
        (
            0x5c80_0000_0007_0001,
            Register(1),
            Register(0),
            Register(0),
            DoubleProductSign::Positive,
        ),
        (
            0x5c80_0000_0007_0200,
            Register(0),
            Register(2),
            Register(0),
            DoubleProductSign::Positive,
        ),
        (
            0x5c80_0000_0047_0000,
            Register(0),
            Register(0),
            Register(4),
            DoubleProductSign::Positive,
        ),
        (
            0x5c81_0000_0007_0000,
            Register(0),
            Register(0),
            Register(0),
            DoubleProductSign::Negated,
        ),
    ]
}

#[test]
fn five_dmul_register_operand_forms_decode_and_validate() {
    let header = headers::parse(&[0; 32]).unwrap();
    for (raw, expected_dst, expected_lhs, expected_rhs, expected_sign) in forms() {
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        let InstructionKind::DoubleMultiply(ops::DoubleMultiply {
            dst,
            lhs,
            rhs,
            sign,
            ..
        }) = instruction.kind
        else {
            panic!("{raw:016x} did not decode as DMUL: {:?}", instruction.kind);
        };
        assert_eq!(dst, RegisterPair(expected_dst), "{raw:016x}");
        assert_eq!(lhs, RegisterPair(expected_lhs), "{raw:016x}");
        assert_eq!(
            rhs,
            DoubleSource::RegisterPair(RegisterPair(expected_rhs)),
            "{raw:016x}"
        );
        assert_eq!(sign, expected_sign, "{raw:016x}");

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
            "{raw:016x} raw: {}",
            String::from_utf8_lossy(&raw_check.stderr)
        );
        module.optimize();
        let optimized_check = spirv_validation::validate(&module.words);
        assert!(
            optimized_check.status.success(),
            "{raw:016x} optimized: {}",
            String::from_utf8_lossy(&optimized_check.stderr)
        );
    }
}
