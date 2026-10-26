#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{
        DoubleMinMaxOperation, DoubleSource, DoubleSourceModifier, FloatFlags, Instruction,
        InstructionKind, Register, RegisterPair,
    },
    spirv::Op,
};

fn forms() -> [(
    &'static str,
    u64,
    DoubleMinMaxOperation,
    [DoubleSourceModifier; 2],
); 6] {
    use DoubleMinMaxOperation::{Maximum, Minimum};
    use DoubleSourceModifier::{Absolute, Negated, Plain};
    [
        ("minimum", 0x5c50_0380_0047_0208, Minimum, [Plain, Plain]),
        ("maximum", 0x5c50_0780_0047_0208, Maximum, [Plain, Plain]),
        (
            "lhs absolute",
            0x5c50_4380_0047_0208,
            Minimum,
            [Absolute, Plain],
        ),
        (
            "lhs negated",
            0x5c51_0380_0047_0208,
            Minimum,
            [Negated, Plain],
        ),
        (
            "rhs absolute",
            0x5c52_0380_0047_0208,
            Minimum,
            [Plain, Absolute],
        ),
        (
            "rhs negated",
            0x5c50_2380_0047_0208,
            Minimum,
            [Plain, Negated],
        ),
    ]
}

#[test]
fn six_dmnmx_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for (name, raw, expected_operation, expected_modifiers) in forms() {
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        let InstructionKind::DoubleMinMax(ops::DoubleMinMax {
            dst,
            lhs,
            rhs,
            operation,
            modifiers,
            flags,
        }) = instruction.kind
        else {
            panic!("{name} did not decode as DMNMX: {:?}", instruction.kind);
        };
        assert_eq!(dst, RegisterPair(Register(8)), "{name}");
        assert_eq!(lhs, RegisterPair(Register(2)), "{name}");
        assert_eq!(
            rhs,
            DoubleSource::RegisterPair(RegisterPair(Register(4))),
            "{name}"
        );
        assert_eq!(operation, expected_operation, "{name}");
        assert_eq!(modifiers, expected_modifiers, "{name}");
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
        let expected_glsl = match expected_operation {
            DoubleMinMaxOperation::Minimum => 37,
            DoubleMinMaxOperation::Maximum => 40,
            DoubleMinMaxOperation::Selected(_) => {
                unreachable!("{name}: the forms are fixed operations")
            }
        };
        let mut at = 5;
        let mut glsl_operations = Vec::new();
        while at < module.words.len() {
            let words = (module.words[at] >> 16) as usize;
            assert!(words != 0, "{name}: zero-width SPIR-V instruction");
            if Op::of(module.words[at]) == Op::ExtInst {
                glsl_operations.push(module.words[at + 4]);
            }
            at += words;
        }
        assert!(glsl_operations.contains(&expected_glsl), "{name}");
        assert!(
            !glsl_operations.iter().any(|op| matches!(op, 79 | 80)),
            "{name}"
        );
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
