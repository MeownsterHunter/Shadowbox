#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{
        DoubleSource, DoubleSourceModifier, Instruction, InstructionKind, IntegerWidth, Register,
        RoundingMode, Signedness,
    },
};

struct Form {
    name: &'static str,
    raw: u64,
    binding: u8,
    offset: u16,
    modifier: DoubleSourceModifier,
    signedness: Signedness,
    width: IntegerWidth,
    round: RoundingMode,
}

fn forms() -> [Form; 6] {
    [
        Form {
            name: "S32 nearest-even, c[0][0x48]",
            raw: 0x4cb0_0000_0127_1e01,
            binding: 0,
            offset: 0x48,
            modifier: DoubleSourceModifier::Plain,
            signedness: Signedness::Signed,
            width: IntegerWidth::Bits32,
            round: RoundingMode::NearestEven,
        },
        Form {
            name: "U32 truncate, c[25][0x1b8]",
            raw: 0x4cb0_01e4_06e7_0e01,
            binding: 25,
            offset: 0x1b8,
            modifier: DoubleSourceModifier::Plain,
            signedness: Signedness::Unsigned,
            width: IntegerWidth::Bits32,
            round: RoundingMode::Zero,
        },
        Form {
            name: "S16 floor, c[14][0x268]",
            raw: 0x4cb0_00b8_09a7_1d01,
            binding: 14,
            offset: 0x268,
            modifier: DoubleSourceModifier::Plain,
            signedness: Signedness::Signed,
            width: IntegerWidth::Bits16,
            round: RoundingMode::NegativeInfinity,
        },
        Form {
            name: "U16 ceil, c[25][0x2b0]",
            raw: 0x4cb0_0164_0ac7_0d01,
            binding: 25,
            offset: 0x2b0,
            modifier: DoubleSourceModifier::Plain,
            signedness: Signedness::Unsigned,
            width: IntegerWidth::Bits16,
            round: RoundingMode::PositiveInfinity,
        },
        Form {
            name: "S32 nearest-even, -|c[13][0x100]|",
            raw: 0x4cb2_2034_0407_1e01,
            binding: 13,
            offset: 0x100,
            modifier: DoubleSourceModifier::NegatedAbsolute,
            signedness: Signedness::Signed,
            width: IntegerWidth::Bits32,
            round: RoundingMode::NearestEven,
        },
        Form {
            name: "U32 floor, |c[9][0x178]|",
            raw: 0x4cb2_00a4_05e7_0e01,
            binding: 9,
            offset: 0x178,
            modifier: DoubleSourceModifier::Absolute,
            signedness: Signedness::Unsigned,
            width: IntegerWidth::Bits32,
            round: RoundingMode::NegativeInfinity,
        },
    ]
}

#[test]
fn six_f2i64_cbuf_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for form in forms() {
        let name = form.name;
        let raw = form.raw;
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        assert_eq!(
            instruction.kind,
            InstructionKind::DoubleToInt(ops::DoubleToInt {
                dst: Register(1),
                src: DoubleSource::ConstantBuffer {
                    binding: form.binding,
                    offset: form.offset,
                },
                modifier: form.modifier,
                signedness: form.signedness,
                width: form.width,
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
