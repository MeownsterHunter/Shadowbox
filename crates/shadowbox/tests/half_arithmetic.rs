//! Encoding boundary checks for packed arithmetic. Field citations are in
//! decoder.rs.
use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    ir::{HalfFlush, HalfOp, InstructionKind, Source},
};

fn decode(raw: u64) -> InstructionKind {
    decode_word([raw as u32, (raw >> 32) as u32], 8).kind
}

#[test]
fn every_mode_is_represented_and_reserved_modes_stay_unknown() {
    // envytools envydis/gm107.c:1987,1157-1167,1083-1086,1782-1785.
    for (opcode, shift, bits) in [
        (0x5d10_u64, 39, 1),
        (0x7a00, 39, 1),
        (0x7a80, 39, 1),
        (0x2c00, 55, 1),
        (0x5d08, 39, 2),
        (0x7800, 39, 2),
        (0x7880, 39, 2),
        (0x2a00, 55, 2),
        (0x5d00, 37, 2),
        (0x7000, 57, 2),
        (0x7080, 57, 2),
        (0x6080, 57, 2),
        (0x2800, 55, 2),
    ] {
        for mode in 0..1_u64 << bits {
            let raw = opcode << 48 | mode << shift | 7 << 16;
            if mode == 3 {
                assert!(
                    matches!(decode(raw), InstructionKind::Unknown(ops::Unknown { .. })),
                    "{raw:016x}"
                );
            } else {
                let expected = [HalfFlush::Preserve, HalfFlush::Ftz, HalfFlush::Fmz][mode as usize];
                assert!(
                    matches!(decode(raw), InstructionKind::HalfArithmetic(ops::HalfArithmetic { flush, .. }) if flush == expected),
                    "{raw:016x}"
                );
            }
        }
    }
}

#[test]
fn rc_exchanges_sources_but_keeps_product_and_addend_signs() {
    // envytools gm107.c:1983; Ryujinx InstEmitFloatArithmetic.cs:324-332.
    let raw =
        0x6080_u64 << 48 | 1 << 56 | 1 << 51 | 3 << 53 | 5 << 39 | 2 << 34 | 8 << 20 | 7 << 16;
    assert!(matches!(decode(raw),
        InstructionKind::HalfArithmetic(ops::HalfArithmetic {
            op: HalfOp::Fma,
            src: [_, Source::Register(r), Source::ConstantBuffer { binding: 2, offset: 32 }],
            swizzle: [0, 3, 1],
            negate: [false, true, true],
            ..
        }) if r.0 == 5));
}

#[test]
fn wide_immediate_fma_swizzle_is_not_part_of_the_literal() {
    // envytools gm107.c:2157,1774-1785. Ryujinx's generated 32i decoder
    // (InstDecoders.cs:2459-2470) overlaps the literal at 47 and puts FMZ
    // at 57. Use the independent encoding authority instead.
    let raw = 0x2800_u64 << 48 | 2 << 53 | 0xdead_beef << 20 | 7 << 16 | 9;
    assert!(matches!(decode(raw),
        InstructionKind::HalfArithmetic(ops::HalfArithmetic {
            op: HalfOp::Fma,
            src: [_, Source::Immediate(0xdead_beef), Source::Register(r)],
            swizzle: [2, 0, 0],
            ..
        }) if r.0 == 9));
}

#[test]
fn only_the_two_allowed_families_decode_as_nop() {
    let source = include_str!("../src/decoder/families.rs");
    assert_eq!(
        source
            .matches("x.guarded(InstructionKind::Nop(ops::Nop))")
            .count(),
        2
    );
    // NOP and scheduling facts: Mesa nak/sm50.rs:3178-3184;
    // envytools gm107.c:1877-1878. Real half/VMAD/bound texture instructions
    // must retain their operation or stay Unknown, regardless of modifiers.
    for prefix in [
        0x5d00_u64, 0x5d08, 0x5d10, 0x5d18, 0x5d20, 0x7000, 0x7080, 0x6080, 0x7800, 0x7880, 0x7a00,
        0x7a80, 0x7c00, 0x7c80, 0x7e00, 0x7e80, 0x2800, 0x2a00, 0x2c00, 0x5f00, 0xc038, 0xdf48,
    ] {
        for lower in [0, 0xffff_ffff_ffff] {
            assert!(!matches!(
                decode(prefix << 48 | lower),
                InstructionKind::Nop(ops::Nop)
            ));
        }
    }
}

#[test]
fn every_half_comparison_form_carries_ftz() {
    // envytools envydis/gm107.c:1973-1976,1985-1986. Only one FTZ bit;
    // these encodings have neither FMZ nor wide-immediate forms.
    for (opcode, shift) in [
        (0x5d18_u64, 50),
        (0x7c00, 54),
        (0x7c80, 54),
        (0x5d20, 6),
        (0x7e00, 6),
        (0x7e80, 6),
    ] {
        for enabled in [false, true] {
            let raw = opcode << 48 | u64::from(enabled) << shift | 7 << 16;
            assert!(
                matches!(decode(raw), InstructionKind::HalfSet(ops::HalfSet { ftz, .. }) if ftz == enabled)
            );
        }
    }
}
