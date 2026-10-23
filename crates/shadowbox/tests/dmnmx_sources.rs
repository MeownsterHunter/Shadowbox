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
    DoubleSource,
    DoubleMinMaxOperation,
    [DoubleSourceModifier; 2],
); 6] {
    use DoubleMinMaxOperation::{Maximum, Minimum};
    use DoubleSourceModifier::{Absolute, Negated, Plain};
    let cbuf = |binding, offset| DoubleSource::ConstantBuffer { binding, offset };
    [
        (
            "minimum, c[0][0x10]",
            0x4c50_0380_0047_0208,
            cbuf(0, 0x10),
            Minimum,
            [Plain, Plain],
        ),
        (
            "maximum, c[0][0x10]",
            0x4c50_0780_0047_0208,
            cbuf(0, 0x10),
            Maximum,
            [Plain, Plain],
        ),
        (
            "maximum, -R2, |c[1][0x20]|",
            0x4c53_0784_0087_0208,
            cbuf(1, 0x20),
            Maximum,
            [Negated, Absolute],
        ),
        (
            "minimum, +1.0",
            0x3850_03bf_f007_0208,
            DoubleSource::Immediate(0x3ff0_0000),
            Minimum,
            [Plain, Plain],
        ),
        (
            "maximum, -2.0",
            0x3950_07c0_0007_0208,
            DoubleSource::Immediate(0xc000_0000),
            Maximum,
            [Plain, Plain],
        ),
        (
            "minimum, |R2|, -(0.5)",
            0x3850_63bf_e007_0208,
            DoubleSource::Immediate(0x3fe0_0000),
            Minimum,
            [Absolute, Negated],
        ),
    ]
}

#[test]
fn six_dmnmx_source_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for (name, raw, expected_rhs, expected_operation, expected_modifiers) in forms() {
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        assert_eq!(
            instruction.kind,
            InstructionKind::DoubleMinMax(ops::DoubleMinMax {
                dst: RegisterPair(Register(8)),
                lhs: RegisterPair(Register(2)),
                rhs: expected_rhs,
                operation: expected_operation,
                modifiers: expected_modifiers,
                flags: FloatFlags::Kept,
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
