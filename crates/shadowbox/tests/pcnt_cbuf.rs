#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::{
    decoder::{decode_stream, decode_word},
    emitter::{emit_unoptimized_with_header, emit_with_header},
    headers,
    interface::Stopped,
    ir::{ops, InstructionKind, TargetStack},
};

const WORDS: [u64; 16] = [
    0xe2b0_0000_0000_0020,
    0xe2b1_0000_0000_0020,
    0xe2b2_0000_0000_0020,
    0xe2b3_0000_0000_0020,
    0xe2b4_0000_0000_0020,
    0xe2b5_0000_0000_0020,
    0xe2b6_0000_0000_0020,
    0xe2b7_0000_0000_0020,
    0xe2b8_0000_0000_0020,
    0xe2b9_0000_0000_0020,
    0xe2ba_0000_0000_0020,
    0xe2bb_0000_0000_0020,
    0xe2bc_0000_0000_0020,
    0xe2bd_0000_0000_0020,
    0xe2be_0000_0000_0020,
    0xe2bf_0000_0000_0020,
];

fn decode(raw: u64) -> InstructionKind {
    decode_word([raw as u32, (raw >> 32) as u32], 8).kind
}

fn program(slots: &[u64]) -> Vec<u32> {
    slots
        .chunks(3)
        .flat_map(|bundle| std::iter::once(0).chain(bundle.iter().copied()))
        .flat_map(|raw| [raw as u32, (raw >> 32) as u32])
        .collect()
}

#[test]
fn cbuf_pcnt_words_decode_typed_continue_pushes() {
    for raw in WORDS {
        assert_eq!(
            decode(raw),
            InstructionKind::CbufPushTarget(ops::CbufPushTarget {
                binding: 0,
                offset: 0,
                stack: TargetStack::Continue,
            }),
            "{raw:016x}"
        );
    }
}

#[test]
fn cbuf_pcnt_admission_is_exact() {
    for raw in WORDS {
        for bit in 0..48 {
            let neighbor = raw ^ (1_u64 << bit);
            assert!(
                !matches!(decode(neighbor), InstructionKind::CbufPushTarget(_)),
                "unlisted word decoded as cbuf PCNT: {neighbor:016x}"
            );
            if bit == 5 {
                assert!(matches!(decode(neighbor), InstructionKind::PushTarget(_)));
                continue;
            }
            assert!(
                matches!(decode(neighbor), InstructionKind::Unknown(_)),
                "unlisted cbuf PCNT decoded: {neighbor:016x}"
            );
        }
    }
}

#[test]
fn unused_cbuf_pcnt_validates_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    let exit = 0xe300_0000_0007_000f;
    for raw in WORDS {
        let xs = decode_stream(shadowbox::QuietRun, &program(&[raw, exit]));
        for module in [
            emit_unoptimized_with_header(&xs, Some(&header)).unwrap(),
            emit_with_header(&xs, Some(&header)).unwrap(),
        ] {
            let validation = spirv_validation::validate(&module.words);
            assert!(
                validation.status.success(),
                "{raw:016x}: {}",
                String::from_utf8_lossy(&validation.stderr)
            );
        }
    }
}

#[test]
fn cbuf_pcnt_with_reachable_cont_remains_unproven() {
    let cont = 0xe350_0000_0007_000f;
    let exit = 0xe300_0000_0007_000f;
    for raw in WORDS {
        let xs = decode_stream(shadowbox::QuietRun, &program(&[raw, cont, exit]));
        assert!(xs
            .iter()
            .any(|x| matches!(x.kind, InstructionKind::PopJump(_))));
        assert!(matches!(
            emit_unoptimized_with_header(&xs, None),
            Err(Stopped::UnprovenPop(_))
        ));
    }
}
