#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{
        FMulModifiers, FMulRounding, FMulScale, Flush, Instruction, InstructionKind, Register,
        Saturation, Source, SourceNegation,
    },
};

struct Form {
    name: &'static str,
    raw: u64,
    rhs: Source,
    modifiers: FMulModifiers,
}

const OPERAND: FMulModifiers = FMulModifiers::Operand {
    rhs_negation: SourceNegation::Plain,
    saturation: Saturation::Unclamped,
    flush: Flush::Dnz,
    rounding: FMulRounding::NearestEven,
    scale: FMulScale::Unscaled,
};

fn forms() -> [Form; 6] {
    [
        Form {
            name: "register FMZ",
            raw: 0x5c68_2000_0037_0201,
            rhs: Source::Register(Register(3)),
            modifiers: OPERAND,
        },
        Form {
            name: "cbuf FMZ negated B",
            raw: 0x4c69_2000_0017_0201,
            rhs: Source::ConstantBuffer {
                binding: 0,
                offset: 4,
            },
            modifiers: FMulModifiers::Operand {
                rhs_negation: SourceNegation::Negated,
                saturation: Saturation::Unclamped,
                flush: Flush::Dnz,
                rounding: FMulRounding::NearestEven,
                scale: FMulScale::Unscaled,
            },
        },
        Form {
            name: "f20 FMZ saturate 1.5",
            raw: 0x386c_203f_c007_0201,
            rhs: Source::Immediate(0x3fc0_0000),
            modifiers: FMulModifiers::Operand {
                rhs_negation: SourceNegation::Plain,
                saturation: Saturation::Saturated,
                flush: Flush::Dnz,
                rounding: FMulRounding::NearestEven,
                scale: FMulScale::Unscaled,
            },
        },
        Form {
            name: "register FMZ A/2",
            raw: 0x5c68_2200_0037_0201,
            rhs: Source::Register(Register(3)),
            modifiers: FMulModifiers::Operand {
                rhs_negation: SourceNegation::Plain,
                saturation: Saturation::Unclamped,
                flush: Flush::Dnz,
                rounding: FMulRounding::NearestEven,
                scale: FMulScale::Divide2,
            },
        },
        Form {
            name: "FMUL32I FMZ 1.5",
            raw: 0x1e43_fc00_0007_0201,
            rhs: Source::Immediate(0x3fc0_0000),
            modifiers: FMulModifiers::Immediate32 {
                saturation: Saturation::Unclamped,
                flush: Flush::Dnz,
            },
        },
        Form {
            name: "FMUL32I FMZ saturate 1.5",
            raw: 0x1ec3_fc00_0007_0201,
            rhs: Source::Immediate(0x3fc0_0000),
            modifiers: FMulModifiers::Immediate32 {
                saturation: Saturation::Saturated,
                flush: Flush::Dnz,
            },
        },
    ]
}

#[test]
fn six_fmul_fmz_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for form in forms() {
        let name = form.name;
        let raw = form.raw;
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        assert_eq!(
            instruction.kind,
            InstructionKind::FMul(ops::FMul {
                dst: Register(1),
                lhs: Register(2),
                rhs: form.rhs,
                modifiers: form.modifiers,
                flags: shadowbox::ir::FloatFlags::Kept,
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
