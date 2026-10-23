//! KIL must preserve shader execution, including the stack and later Unknowns.
//! Mesa nak/from_nir.rs:3035-3050 lowers demote to KIL, not termination.
use shadowbox::ir::ops;
use shadowbox::{
    decoder::{decode_stream, decode_word},
    emitter::emit_unoptimized_with_header,
    ir::InstructionKind,
};

// Mesa sm50.rs:1938-1951,3099-3111,3172-3174,3418-3468.
fn words(tail: u64) -> Vec<u32> {
    [0u64, 0xe330_0000_0007_000f, tail, 0xe300_0000_0007_000f]
        .into_iter()
        .flat_map(|w| [w as u32, (w >> 32) as u32])
        .collect()
}

#[test]
fn unconditional_kil_keeps_fallthrough_and_rejects_executed_unknown() {
    let xs = decode_stream(shadowbox::QuietRun, &words(0xffff_ffff_ffff_ffff));
    assert!(xs
        .iter()
        .any(|x| matches!(x.kind, InstructionKind::Unknown(ops::Unknown { .. }))));
    assert!(emit_unoptimized_with_header(&xs, None).is_err());
}

#[test]
fn unconditional_kil_preserves_the_following_instruction() {
    let xs = decode_stream(shadowbox::QuietRun, &words(0x0100_0000_02a7_f001));
    assert!(xs.iter().any(|x| matches!(
        x.kind,
        InstructionKind::Mov(ops::Mov {
            dst: shadowbox::ir::Register(1),
            src: shadowbox::ir::Source::Immediate(42),
            ..
        })
    )));
    let module = emit_unoptimized_with_header(&xs, None).unwrap();
    assert!(module.words.windows(1).any(|w| w[0] == (1 << 16 | 5380)));
}

#[test]
fn kil_rejects_every_unnamed_bit_and_nontrivial_cc() {
    // Mesa sm50.rs:225-235,3172-3174: CC0..4=T, guard16..19.
    let base = 0xe330_0000_0007_000fu64;
    let decode = |raw: u64| decode_word([raw as u32, (raw >> 32) as u32], 0).kind;
    for bit in (5..16).chain(20..48) {
        assert!(matches!(
            decode(base | 1 << bit),
            InstructionKind::Unknown(ops::Unknown { .. })
        ));
    }
    for cc in 0..32 {
        let instruction = decode(base & !31 | cc);
        assert_eq!(instruction == InstructionKind::Kill(ops::Kill), cc == 15);
    }
}
