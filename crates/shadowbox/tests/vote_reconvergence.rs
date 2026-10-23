//! VOTE at a token target must execute after the region's SPIR-V loop merge.
use shadowbox::ir::ops;
use shadowbox::spirv::Op;
use shadowbox::{
    decoder::decode_stream, emitter::emit_unoptimized_with_header, headers, ir::InstructionKind,
};
const PT: u64 = 7 << 16;
fn words(slots: &[u64]) -> Vec<u32> {
    // Mesa nak/sm50.rs:3418-3468: scheduling slot and three instructions.
    slots
        .chunks(3)
        .flat_map(|s| [0, s[0], s[1], s[2]])
        .flat_map(|x| [x as u32, (x >> 32) as u32])
        .collect()
}
fn vote() -> u64 {
    // Mesa nak/sm50.rs:3233-3247: ALL, P2 destination, P1 source, R50 ballot.
    0x50d8 << 48 | 2 << 45 | 1 << 39 | PT | 50
}
#[test]
fn ssy_vote_at_merge_has_nested_loop_reconvergence() {
    // SSY byte08 targets byte30: relative0x20 from nextPC.
    // Mesa nak/sm50.rs:3006-3016,3032-3052,3106-3110.
    let code = words(&[
        0xe290 << 48 | 32 << 20 | PT | 15,
        0xf0f8 << 48 | 15,
        vote(),
        0xf0f8 << 48 | PT | 15,
        vote(),
        0xe300 << 48 | PT | 15,
    ]);
    let decoded = decode_stream(shadowbox::QuietRun, &code);
    assert!(decoded
        .iter()
        .all(|x| !matches!(x.kind, InstructionKind::Unknown(ops::Unknown { .. }))));
    assert_eq!(
        decoded
            .iter()
            .filter(|x| matches!(x.kind, InstructionKind::Vote(ops::Vote { .. })))
            .count(),
        2
    );
    let header = headers::parse(&[0; 32]).unwrap();
    let module = emit_unoptimized_with_header(&decoded, Some(&header)).unwrap();
    assert!(module.requires_maximal_reconvergence);
    let mut at = 5;
    let mut loops = 0;
    while at < module.words.len() {
        loops += usize::from(Op::of(module.words[at]) == Op::LoopMerge);
        at += (module.words[at] >> 16) as usize;
    }
    assert_eq!(loops, 2);
}
#[test]
fn single_block_predicated_vote_is_still_supported() {
    // Predication guards the collective itself. Ryujinx Translator.cs:359-393,
    // InstEmitWarp.cs:48-101. No inter-block reconvergence is involved.
    let code = words(&[
        vote() & !PT,
        0xe300 << 48 | PT | 15,
        0x50b0 << 48 | PT | 15 << 8,
    ]);
    let header = headers::parse(&[0; 32]).unwrap();
    assert!(emit_unoptimized_with_header(
        &decode_stream(shadowbox::QuietRun, &code),
        Some(&header)
    )
    .is_ok());
}

// An ordinary BRA must retain its operation and cannot consume the SSY.
// Mesa sm50.rs:3019-3028 encodes BRA; :3038-3051 encodes SSY/SYNC.
// Ryujinx InstEmitFlowControl.cs:239-275 distinguishes pushes and pops.
#[test]
fn ordinary_branch_keeps_its_operation_and_ordered_tail() {
    let code = words(&[
        0xe290 << 48 | 32 << 20 | PT | 15,
        0xe240 << 48 | 24 << 20 | 15,
        vote(),
        0xf0f8 << 48 | PT | 15,
        vote(),
        0xe300 << 48 | PT | 15,
    ]);
    let decoded = decode_stream(shadowbox::QuietRun, &code);
    assert!(matches!(
        decoded[1].kind,
        InstructionKind::Branch(ops::Branch { .. })
    ));
    assert!(matches!(
        decoded[3].kind,
        InstructionKind::Reconverge(ops::Reconverge { .. })
    ));
    let header = headers::parse(&[0; 32]).unwrap();
    let module = emit_unoptimized_with_header(&decoded, Some(&header)).unwrap();
    assert!(module.requires_maximal_reconvergence);
}

#[test]
fn plain_branch_leaving_a_used_token_region_is_ordered() {
    // A CC-tested pop can fail and fall through. That path reaches the
    // plain BRA, which preserves the token rather than performing SYNC.
    // envytools gm107.c:1920-1921,426-458; Ryujinx FlowControl.cs:295-312.
    let code = words(&[
        0xe2a0 << 48 | 32 << 20 | PT | 15,
        0xe340 << 48 | PT | 2,
        0xe240 << 48 | 16 << 20 | PT | 15,
        0x50b0 << 48 | PT | 15 << 8,
        vote(),
        0xe300 << 48 | PT | 15,
    ]);
    let header = headers::parse(&[0; 32]).unwrap();
    let module =
        emit_unoptimized_with_header(&decode_stream(shadowbox::QuietRun, &code), Some(&header))
            .unwrap();
    assert!(module.requires_maximal_reconvergence);
}

#[test]
fn early_backedge_with_two_latches_reports_address_and_full_reason() {
    // BRA fields: Mesa sm50.rs:3019-3028. Two branches return to byte 08.
    // The first could restart an iteration before the VOTE at byte 18.
    let code = words(&[
        0x50b0 << 48 | PT | 15 << 8,
        0xe240 << 48 | 0xff_fff0 << 20 | 15,
        vote(),
        0xe240 << 48 | 0xff_ffd8 << 20 | PT | 15,
        vote(),
        0xe300 << 48 | PT | 15,
    ]);
    let header = headers::parse(&[0; 32]).unwrap();
    let error =
        emit_unoptimized_with_header(&decode_stream(shadowbox::QuietRun, &code), Some(&header))
            .err()
            .unwrap();
    assert_eq!(error.to_string(),
        "unsupported control flow: warp reconvergence at 0x10: loop at 0x8 has multiple latches; iteration ordering is unprovable");
}
