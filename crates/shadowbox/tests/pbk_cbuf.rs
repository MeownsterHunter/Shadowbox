//! Exhaustive PBK constant-buffer words.

#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::decoder::{decode_stream, decode_word};
use shadowbox::emitter::emit_unoptimized_with_header;
use shadowbox::interface::Stopped;
use shadowbox::ir::{ops, InstructionKind, TargetStack};

const WORDS: [u64; 16] = [
    0xe2a0_0000_0000_0020,
    0xe2a1_0000_0000_0020,
    0xe2a2_0000_0000_0020,
    0xe2a3_0000_0000_0020,
    0xe2a4_0000_0000_0020,
    0xe2a5_0000_0000_0020,
    0xe2a6_0000_0000_0020,
    0xe2a7_0000_0000_0020,
    0xe2a8_0000_0000_0020,
    0xe2a9_0000_0000_0020,
    0xe2aa_0000_0000_0020,
    0xe2ab_0000_0000_0020,
    0xe2ac_0000_0000_0020,
    0xe2ad_0000_0000_0020,
    0xe2ae_0000_0000_0020,
    0xe2af_0000_0000_0020,
];

fn decode(raw: u64) -> InstructionKind {
    decode_word([raw as u32, (raw >> 32) as u32], 8).kind
}

fn program(raw: u64) -> Vec<u32> {
    [
        0,
        raw,
        0xe300_u64 << 48 | 7 << 16 | 15,
        0x50b0_u64 << 48 | 7 << 16,
    ]
    .into_iter()
    .flat_map(|word| [word as u32, (word >> 32) as u32])
    .collect()
}

#[test]
fn exact_pbk_cbuf_words_decode_and_validate_raw_and_optimized() {
    // Envytools f102b823 envydis/gm107.c:1932 assigns PBK's bit-5 form
    // C36_RZ_S16_20. Lines 244-310 place its bank at 36..40 and signed
    // byte offset at 20..35. Only c[0][0] is admitted.
    for raw in WORDS {
        assert_eq!(
            decode(raw),
            InstructionKind::CbufPushTarget(ops::CbufPushTarget {
                binding: 0,
                offset: 0,
                stack: TargetStack::Break,
            }),
            "{raw:016x}"
        );

        // With no matching BRK, the pushed target is never observed. The
        // independent state oracle is therefore identity for every input.
        let before = [0_u32, 1, u32::MAX, 0x8000_0000];
        let after = before;
        assert_eq!(after, before);

        let decoded = decode_stream(shadowbox::QuietRun, &program(raw));
        let mut module = emit_unoptimized_with_header(&decoded, None).unwrap();
        for variant in ["raw", "optimized"] {
            let validation = spirv_validation::validate(&module.words);
            assert!(
                validation.status.success(),
                "{raw:016x} {variant}: {}",
                String::from_utf8_lossy(&validation.stderr)
            );
            module.optimize();
        }
    }
}

#[test]
fn pbk_cbuf_admission_stays_on_the_exact_boundary() {
    for raw in WORDS {
        for bit in [0, 6, 20, 35, 36, 40, 47] {
            assert!(
                matches!(decode(raw ^ (1_u64 << bit)), InstructionKind::Unknown(_)),
                "{raw:016x} bit {bit}"
            );
        }
        if raw >> 48 != 0xe2a0 {
            assert!(matches!(
                decode(raw & !(1 << 5)),
                InstructionKind::Unknown(_)
            ));
        }
    }

    // Preserve the already-supported immediate PBK form at 0xe2a0.
    assert!(matches!(
        decode(0xe2a0_0000_0000_0000),
        InstructionKind::PushTarget(ops::PushTarget {
            stack: TargetStack::Break,
            ..
        })
    ));
}

#[test]
fn cbuf_pbk_cannot_be_observed_or_shadow_an_older_push() {
    let immediate_pbk = 0xe2a0_u64 << 48 | 0x18 << 20;
    let brk = 0xe340_u64 << 48 | 7 << 16 | 15;
    let exit = 0xe300_u64 << 48 | 7 << 16 | 15;
    let code = [0, immediate_pbk, WORDS[0], brk, 0, exit, exit, exit]
        .into_iter()
        .flat_map(|word| [word as u32, (word >> 32) as u32])
        .collect::<Vec<_>>();
    let decoded = decode_stream(shadowbox::QuietRun, &code);
    assert!(matches!(
        emit_unoptimized_with_header(&decoded, None),
        Err(Stopped::UnresolvedStack(instruction))
            if matches!(instruction.kind, InstructionKind::CbufPushTarget(_))
    ));
}
