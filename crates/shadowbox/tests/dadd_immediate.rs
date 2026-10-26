#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{
        DoubleSource, DoubleSourceModifier, Instruction, InstructionKind, Register, RegisterPair,
    },
};

fn forms() -> [(&'static str, u64, u32, DoubleSourceModifier); 5] {
    use DoubleSourceModifier::{Negated, NegatedAbsolute, Plain};
    [
        ("positive one", 0x3870_003f_f007_0206, 0x3ff0_0000, Plain),
        ("negative one", 0x3970_003f_f007_0206, 0xbff0_0000, Plain),
        (
            "positive half lhs absolute",
            0x3870_403f_e007_0206,
            0x3fe0_0000,
            DoubleSourceModifier::Absolute,
        ),
        (
            "positive two lhs negated",
            0x3871_0040_0007_0206,
            0x4000_0000,
            Negated,
        ),
        (
            "negative two lhs negated absolute",
            0x3971_4040_0007_0206,
            0xc000_0000,
            NegatedAbsolute,
        ),
    ]
}

#[test]
fn five_dadd_immediate_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for (name, raw, immediate, lhs_modifier) in forms() {
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        let InstructionKind::DoubleAdd(ops::DoubleAdd {
            dst,
            lhs,
            rhs,
            modifiers,
            ..
        }) = instruction.kind
        else {
            panic!("{name} did not decode as DADD.I: {:?}", instruction.kind);
        };
        assert_eq!(dst, RegisterPair(Register(6)), "{name}");
        assert_eq!(lhs, RegisterPair(Register(2)), "{name}");
        assert_eq!(rhs, DoubleSource::Immediate(immediate), "{name}");
        assert_eq!(
            modifiers,
            [lhs_modifier, DoubleSourceModifier::Plain],
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
