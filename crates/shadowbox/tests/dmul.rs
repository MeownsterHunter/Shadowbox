#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{
        DoubleProductSign, DoubleSource, FloatFlags, Instruction, InstructionKind, Register,
        RegisterPair,
    },
};

fn forms() -> [(&'static str, u64, DoubleSource, DoubleProductSign); 5] {
    [
        (
            "register plain",
            0x5c80_0000_0047_0206,
            DoubleSource::RegisterPair(RegisterPair(Register(4))),
            DoubleProductSign::Positive,
        ),
        (
            "register negated",
            0x5c81_0000_0047_0206,
            DoubleSource::RegisterPair(RegisterPair(Register(4))),
            DoubleProductSign::Negated,
        ),
        (
            "cbuf plain",
            0x4c80_0000_0047_0206,
            DoubleSource::ConstantBuffer {
                binding: 0,
                offset: 16,
            },
            DoubleProductSign::Positive,
        ),
        (
            "cbuf negated",
            0x4c81_0000_0047_0206,
            DoubleSource::ConstantBuffer {
                binding: 0,
                offset: 16,
            },
            DoubleProductSign::Negated,
        ),
        (
            "immediate positive one",
            0x3880_003f_f007_0206,
            DoubleSource::Immediate(0x3ff0_0000),
            DoubleProductSign::Positive,
        ),
    ]
}

#[test]
fn five_dmul_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for (name, raw, expected_rhs, expected_sign) in forms() {
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        let InstructionKind::DoubleMultiply(ops::DoubleMultiply {
            dst,
            lhs,
            rhs,
            sign,
            flags,
        }) = instruction.kind
        else {
            panic!("{name} did not decode as DMUL: {:?}", instruction.kind);
        };
        assert_eq!(dst, RegisterPair(Register(6)), "{name}");
        assert_eq!(lhs, RegisterPair(Register(2)), "{name}");
        assert_eq!(rhs, expected_rhs, "{name}");
        assert_eq!(sign, expected_sign, "{name}");
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
