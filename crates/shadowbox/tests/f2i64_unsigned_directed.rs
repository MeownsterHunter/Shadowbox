#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{
        DoubleSource, DoubleSourceModifier, Instruction, InstructionKind, IntegerWidth, Register,
        RegisterPair, RoundingMode, Signedness,
    },
};

fn forms() -> [(&'static str, u64, IntegerWidth, RoundingMode); 6] {
    [
        (
            "U32 floor",
            0x5cb0_0080_0027_0e01,
            IntegerWidth::Bits32,
            RoundingMode::NegativeInfinity,
        ),
        (
            "U32 ceil",
            0x5cb0_0100_0027_0e01,
            IntegerWidth::Bits32,
            RoundingMode::PositiveInfinity,
        ),
        (
            "U32 trunc",
            0x5cb0_0180_0027_0e01,
            IntegerWidth::Bits32,
            RoundingMode::Zero,
        ),
        (
            "U16 floor",
            0x5cb0_0080_0027_0d01,
            IntegerWidth::Bits16,
            RoundingMode::NegativeInfinity,
        ),
        (
            "U16 ceil",
            0x5cb0_0100_0027_0d01,
            IntegerWidth::Bits16,
            RoundingMode::PositiveInfinity,
        ),
        (
            "U16 trunc",
            0x5cb0_0180_0027_0d01,
            IntegerWidth::Bits16,
            RoundingMode::Zero,
        ),
    ]
}

#[test]
fn six_unsigned_directed_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for (name, raw, expected_width, expected_round) in forms() {
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        let InstructionKind::DoubleToInt(ops::DoubleToInt {
            dst,
            src,
            modifier,
            signedness,
            width,
            round,
        }) = instruction.kind
        else {
            panic!("{name} did not decode as F2I.F64: {:?}", instruction.kind);
        };
        assert_eq!(dst, Register(1), "{name}");
        assert_eq!(
            src,
            DoubleSource::RegisterPair(RegisterPair(Register(2))),
            "{name}"
        );
        assert_eq!(modifier, DoubleSourceModifier::Plain, "{name}");
        assert_eq!(signedness, Signedness::Unsigned, "{name}");
        assert_eq!(width, expected_width, "{name}");
        assert_eq!(round, expected_round, "{name}");

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
