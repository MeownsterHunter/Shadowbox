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

struct Form {
    name: &'static str,
    raw: u64,
    modifier: DoubleSourceModifier,
    signedness: Signedness,
    round: RoundingMode,
}

fn forms() -> [Form; 6] {
    [
        Form {
            name: "S32 nearest-even, -R2",
            raw: 0x5cb0_2000_0027_1e01,
            modifier: DoubleSourceModifier::Negated,
            signedness: Signedness::Signed,
            round: RoundingMode::NearestEven,
        },
        Form {
            name: "S32 nearest-even, |R2|",
            raw: 0x5cb2_0000_0027_1e01,
            modifier: DoubleSourceModifier::Absolute,
            signedness: Signedness::Signed,
            round: RoundingMode::NearestEven,
        },
        Form {
            name: "S32 nearest-even, -|R2|",
            raw: 0x5cb2_2000_0027_1e01,
            modifier: DoubleSourceModifier::NegatedAbsolute,
            signedness: Signedness::Signed,
            round: RoundingMode::NearestEven,
        },
        Form {
            name: "S32 truncate, -R2",
            raw: 0x5cb0_2180_0027_1e01,
            modifier: DoubleSourceModifier::Negated,
            signedness: Signedness::Signed,
            round: RoundingMode::Zero,
        },
        Form {
            name: "U32 floor, |R2|",
            raw: 0x5cb2_0080_0027_0e01,
            modifier: DoubleSourceModifier::Absolute,
            signedness: Signedness::Unsigned,
            round: RoundingMode::NegativeInfinity,
        },
        Form {
            name: "U32 truncate, -R2",
            raw: 0x5cb0_2180_0027_0e01,
            modifier: DoubleSourceModifier::Negated,
            signedness: Signedness::Unsigned,
            round: RoundingMode::Zero,
        },
    ]
}

#[test]
fn six_f2i64_source_modifier_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for form in forms() {
        let name = form.name;
        let raw = form.raw;
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        assert_eq!(
            instruction.kind,
            InstructionKind::DoubleToInt(ops::DoubleToInt {
                dst: Register(1),
                src: DoubleSource::RegisterPair(RegisterPair(Register(2))),
                modifier: form.modifier,
                signedness: form.signedness,
                width: IntegerWidth::Bits32,
                round: form.round,
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
