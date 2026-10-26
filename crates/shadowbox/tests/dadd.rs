#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{
        DoubleSource, DoubleSourceModifier, FloatFlags, Instruction, InstructionKind, Register,
        RegisterPair, RoundingMode,
    },
};

fn forms() -> [(&'static str, u64, [DoubleSourceModifier; 2]); 5] {
    use DoubleSourceModifier::{Absolute, Negated, Plain};
    [
        ("plain", 0x5c70_0000_0047_0206, [Plain, Plain]),
        ("lhs absolute", 0x5c70_4000_0047_0206, [Absolute, Plain]),
        ("lhs negated", 0x5c71_0000_0047_0206, [Negated, Plain]),
        ("rhs absolute", 0x5c72_0000_0047_0206, [Plain, Absolute]),
        ("rhs negated", 0x5c70_2000_0047_0206, [Plain, Negated]),
    ]
}

#[test]
fn five_dadd_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for (name, raw, expected_modifiers) in forms() {
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        let InstructionKind::DoubleAdd(ops::DoubleAdd {
            dst,
            lhs,
            rhs,
            modifiers,
            round,
            flags,
        }) = instruction.kind
        else {
            panic!("{name} did not decode as DADD: {:?}", instruction.kind);
        };
        assert_eq!(dst, RegisterPair(Register(6)), "{name}");
        assert_eq!(lhs, RegisterPair(Register(2)), "{name}");
        assert_eq!(
            rhs,
            DoubleSource::RegisterPair(RegisterPair(Register(4))),
            "{name}"
        );
        assert_eq!(modifiers, expected_modifiers, "{name}");
        assert_eq!(round, RoundingMode::NearestEven, "{name}");
        assert_eq!(flags, FloatFlags::Kept, "{name}");

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
