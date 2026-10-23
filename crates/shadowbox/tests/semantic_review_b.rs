//! Structural regressions for semantic defects that spirv-val cannot detect.
//! Reference evidence and scope are in docs/semantic-review-B.md.
use shadowbox::ir::ops;
use shadowbox::spirv::Op;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    ir::{Instruction, InstructionKind, Register, Source},
};

fn decode(opcode: u64, fields: u64) -> Instruction {
    let raw = opcode << 48 | 7 << 16 | fields;
    decode_word([raw as u32, (raw >> 32) as u32], 16)
}

fn ops(instruction: Instruction) -> Vec<(Op, Vec<u32>)> {
    let words = emit_unoptimized_with_header(
        &[
            Instruction {
                address: 8,
                predicate: None,
                kind: InstructionKind::Mov(ops::Mov {
                    quad_lanes: 0xf,
                    dst: Register(1),
                    src: Source::Immediate(0xffff_ff00),
                }),
            },
            instruction,
            Instruction {
                address: 24,
                predicate: None,
                kind: InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
            },
        ],
        None,
    )
    .unwrap()
    .words;
    let mut at = 5;
    let mut result = Vec::new();
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        result.push((Op::of(words[at]), words[at + 1..at + count].to_vec()));
        at += count;
    }
    result
}

#[test]
fn i2f_immediate_is_signed_twenty_bits() {
    // Mesa sm50.rs:335-346,1856-1858; Ryujinx InstEmitHelper.cs:238-240.
    for (opcode, immediate, expected) in [(0x38b8, 0x12345, 0x12345), (0x39b8, 0x7ffff, u32::MAX)] {
        assert!(matches!(
            decode(opcode, 2 << 8 | 2 << 10 | 1 << 13 | immediate << 20).kind,
            InstructionKind::Convert(ops::Convert { src: Source::Immediate(value), .. }) if value == expected
        ));
    }
}

#[test]
fn i2f_modifies_the_full_word_before_signed_byte_extraction() {
    // Ryujinx InstEmitConversion.cs:257-265: abs, negate, then extract.
    let instructions = ops(decode(
        0x5cba,
        2 << 8 | 1 << 13 | 1 << 41 | 1 << 45 | 1 << 20,
    ));
    let abs = instructions
        .iter()
        .find(|(op, a)| *op == Op::ExtInst && a[3] == 5)
        .unwrap();
    let neg = instructions
        .iter()
        .find(|(op, _)| *op == Op::SNegate)
        .unwrap();
    let extract = instructions
        .iter()
        .find(|(op, _)| *op == Op::BitFieldSExtract)
        .unwrap();
    let convert = instructions
        .iter()
        .find(|(op, _)| *op == Op::ConvertSToF)
        .unwrap();
    assert_eq!(neg.1[2], abs.1[1]);
    assert_eq!(extract.1[2], neg.1[1]);
    assert_eq!(convert.1[2], extract.1[1]);
}

#[test]
fn i2f_cc_stays_unknown() {
    // envytools gm107.c:1995,2069,2115; Ryujinx InstEmitConversion.cs:283.
    for opcode in [0x5cb8, 0x4cb8, 0x38b8] {
        assert!(matches!(
            decode(opcode, 0xa << 8 | 1 << 47).kind,
            InstructionKind::Unknown(ops::Unknown { .. })
        ));
    }
}

#[test]
fn f2i_unimplemented_cc_stays_unknown() {
    // Mesa sm50.rs:1837; envytools gm107.c:1996,2070.
    for opcode in [0x5cb0, 0x4cb0] {
        assert!(matches!(
            decode(opcode, 0xa << 8 | 1 << 47).kind,
            InstructionKind::Unknown(ops::Unknown { .. })
        ));
    }
}

#[test]
fn f2f_unimplemented_cc_stays_unknown() {
    // Mesa sm50.rs:1793; envytools gm107.c:1997,2071.
    for opcode in [0x5ca8, 0x4ca8] {
        assert!(matches!(
            decode(opcode, 0xa << 8 | 1 << 47).kind,
            InstructionKind::Unknown(ops::Unknown { .. })
        ));
    }
}

#[test]
fn f2i_sanitizes_nan_before_the_conversion() {
    // SPIR-V conversion is undefined on NaN. conversions2 emitter.rs:2317-2380
    // already sanitizes it before FClamp and ConvertFToS/U.
    for (opcode, cast) in [(0x5cb0, Op::ConvertFToU), (0x5cb0, Op::ConvertFToS)] {
        let instructions = ops(decode(
            opcode,
            0xa << 8 | u64::from(cast == Op::ConvertFToS) << 12 | 1 << 20,
        ));
        let convert = instructions.iter().find(|(op, _)| *op == cast).unwrap();
        let clamp = instructions
            .iter()
            .find(|(op, a)| *op == Op::ExtInst && a[3] == 43 && a[1] == convert.1[2])
            .unwrap();
        let sanitized = instructions
            .iter()
            .find(|(op, a)| *op == Op::Select && a[1] == clamp.1[4]);
        assert!(
            sanitized.is_some(),
            "clamp input must be a NaN-sanitizing select"
        );
        let select = &sanitized.unwrap().1;
        assert!(instructions
            .iter()
            .any(|(op, a)| *op == Op::IsNan && a[1] == select[2]));
    }
}

#[test]
fn f2i_ftz_flushes_input_bits_before_rounding() {
    // Mesa sm50.rs:1837; conversions2 emitter.rs:2220-2244. Flushing must
    // happen before floor/ceil, which can turn a nonzero subnormal into 1.
    let instructions = ops(decode(0x5cb0, 0xa << 8 | 1 << 44 | 2 << 39 | 1 << 20));
    let rounded = instructions
        .iter()
        .find(|(op, a)| *op == Op::ExtInst && a[3] == 9)
        .unwrap();
    let float = instructions
        .iter()
        .find(|(op, a)| *op == Op::Bitcast && a[1] == rounded.1[4])
        .unwrap();
    let flushed = instructions
        .iter()
        .find(|(op, a)| *op == Op::Select && a[1] == float.1[2])
        .unwrap();
    assert!(instructions
        .iter()
        .any(|(op, a)| *op == Op::ULessThan && a[1] == flushed.1[2]));
    assert!(instructions
        .iter()
        .any(|(op, a)| *op == Op::BitwiseAnd && a[1] == flushed.1[3]));
}

#[test]
fn f2f_ftz_flushes_input_bits_before_integral_rounding() {
    // Mesa sm50.rs:1793; Ryujinx InstEmitConversion.cs:134-143.
    let instructions = ops(decode(
        0x5ca8,
        0xa << 8 | 1 << 44 | 1 << 42 | 2 << 39 | 1 << 20,
    ));
    let rounded = instructions
        .iter()
        .find(|(op, a)| *op == Op::ExtInst && a[3] == 9)
        .unwrap();
    let float = instructions
        .iter()
        .find(|(op, a)| *op == Op::Bitcast && a[1] == rounded.1[4])
        .unwrap();
    let flushed = instructions
        .iter()
        .find(|(op, a)| *op == Op::Select && a[1] == float.1[2])
        .unwrap();
    assert!(instructions
        .iter()
        .any(|(op, a)| *op == Op::ULessThan && a[1] == flushed.1[2]));
}
