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

const CBUF: DoubleSource = DoubleSource::ConstantBuffer {
    binding: 0,
    offset: 0x10,
};
const R4: DoubleSource = DoubleSource::RegisterPair(RegisterPair(Register(4)));

fn forms() -> [(
    u64,
    DoubleSource,
    DoubleSource,
    DoubleProductSign,
    DoubleAddendSign,
); 6] {
    [
        (
            0x4b70_0200_0047_0206,
            CBUF,
            R4,
            DoubleProductSign::Positive,
            DoubleAddendSign::Added,
        ),
        (
            0x4b72_0200_0047_0206,
            CBUF,
            R4,
            DoubleProductSign::Positive,
            DoubleAddendSign::Subtracted,
        ),
        (
            0x5370_0200_0047_0206,
            R4,
            CBUF,
            DoubleProductSign::Positive,
            DoubleAddendSign::Added,
        ),
        (
            0x5371_0200_0047_0206,
            R4,
            CBUF,
            DoubleProductSign::Negated,
            DoubleAddendSign::Added,
        ),
        (
            0x3670_023f_f007_0206,
            DoubleSource::Immediate(1.0_f64.to_bits().wrapping_shr(32) as u32),
            R4,
            DoubleProductSign::Positive,
            DoubleAddendSign::Added,
        ),
        (
            0x3772_0240_0007_0206,
            DoubleSource::Immediate((-2.0_f64).to_bits().wrapping_shr(32) as u32),
            R4,
            DoubleProductSign::Positive,
            DoubleAddendSign::Subtracted,
        ),
    ]
}

#[test]
fn six_dfma_source_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for (raw, multiplier, addend, product_sign, addend_sign) in forms() {
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        assert_eq!(
            instruction.kind,
            InstructionKind::DoubleFma(ops::DoubleFma {
                dst: RegisterPair(Register(6)),
                lhs: RegisterPair(Register(2)),
                multiplier,
                addend,
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
