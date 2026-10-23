#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{
        DoubleAddendSign, DoubleProductSign, DoubleSource, FloatFlags, Instruction,
        InstructionKind, Register, RegisterPair,
    },
};

fn forms() -> [(
    u64,
    Register,
    Register,
    Register,
    DoubleProductSign,
    DoubleAddendSign,
); 6] {
    [
        (
            0x5b70_0000_0007_0000,
            Register(0),
            Register(0),
            Register(0),
            DoubleProductSign::Positive,
            DoubleAddendSign::Added,
        ),
        (
            0x5b71_0000_0007_0000,
            Register(0),
            Register(0),
            Register(0),
            DoubleProductSign::Negated,
            DoubleAddendSign::Added,
        ),
        (
            0x5b72_0000_0007_0000,
            Register(0),
            Register(0),
            Register(0),
            DoubleProductSign::Positive,
            DoubleAddendSign::Subtracted,
        ),
        (
            0x5b70_0000_0007_0200,
            Register(2),
            Register(0),
            Register(0),
            DoubleProductSign::Positive,
            DoubleAddendSign::Added,
        ),
        (
            0x5b70_0000_0047_0000,
            Register(0),
            Register(4),
            Register(0),
            DoubleProductSign::Positive,
            DoubleAddendSign::Added,
        ),
        (
            0x5b70_0200_0007_0000,
            Register(0),
            Register(0),
            Register(4),
            DoubleProductSign::Positive,
            DoubleAddendSign::Added,
        ),
    ]
}

#[test]
fn six_dfma_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for (raw, lhs, multiplier, addend, product_sign, addend_sign) in forms() {
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        assert_eq!(
            instruction.kind,
            InstructionKind::DoubleFma(ops::DoubleFma {
                dst: RegisterPair(Register(0)),
                lhs: RegisterPair(lhs),
                multiplier: DoubleSource::RegisterPair(RegisterPair(multiplier)),
                addend: DoubleSource::RegisterPair(RegisterPair(addend)),
                product_sign,
                addend_sign,
                flags: FloatFlags::Kept,
            }),
            "{raw:016x}"
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
