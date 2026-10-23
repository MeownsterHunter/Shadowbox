#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::{
    decoder::{decode_stream, decode_word},
    emitter::{emit_unoptimized_with_header, emit_with_header},
    headers,
    ir::{ops, BranchTarget, InstructionKind, TargetStack},
};

const PACKET_WORDS: [u64; 15] = [
    0xe2b1_0000_0000_0000,
    0xe2b2_0000_0000_0000,
    0xe2b3_0000_0000_0000,
    0xe2b4_0000_0000_0000,
    0xe2b5_0000_0000_0000,
    0xe2b6_0000_0000_0000,
    0xe2b7_0000_0000_0000,
    0xe2b8_0000_0000_0000,
    0xe2b9_0000_0000_0000,
    0xe2ba_0000_0000_0000,
    0xe2bb_0000_0000_0000,
    0xe2bc_0000_0000_0000,
    0xe2bd_0000_0000_0000,
    0xe2be_0000_0000_0000,
    0xe2bf_0000_0000_0000,
];

fn decode(raw: u64) -> InstructionKind {
    decode_word([raw as u32, (raw >> 32) as u32], 8).kind
}

fn program(raw: u64) -> Vec<u32> {
    let exit = 0xe300_0000_0007_000f_u64;
    [0, raw, exit, 0]
        .into_iter()
        .flat_map(|word| [word as u32, (word >> 32) as u32])
        .collect()
}

#[test]
fn packets_p007061_through_p007064_decode_typed_continue_pushes() {
    for raw in PACKET_WORDS {
        assert_eq!(
            decode(raw),
            InstructionKind::PushTarget(ops::PushTarget {
                target: BranchTarget::reaching(8, 16),
                stack: TargetStack::Continue,
            }),
            "{raw:016x}"
        );
    }
}

#[test]
fn direct_pcnt_alias_admission_is_exact() {
    for raw in PACKET_WORDS {
        for bit in 0..48 {
            let neighbor = raw ^ (1_u64 << bit);
            if bit == 5 {
                assert!(matches!(
                    decode(neighbor),
                    InstructionKind::CbufPushTarget(_)
                ));
            } else {
                assert!(matches!(decode(neighbor), InstructionKind::Unknown(_)));
            }
        }
    }
}

#[test]
fn unused_direct_pcnt_aliases_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for raw in PACKET_WORDS {
        let xs = decode_stream(shadowbox::QuietRun, &program(raw));
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
