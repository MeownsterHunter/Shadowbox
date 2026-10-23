//! Exact TexCoord0 IPA.PASS words observed in Novena's strict 13-shader replay.

#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_graphics_state,
    headers,
    interface::GraphicsState,
    ir::{ops, InstructionKind, LegacyInterpolation, LegacyVarying, Register},
};

const WORDS: [(u64, u8, u8); 4] = [
    (0xe003_ffb0_0ff7_ff04, 4, 0),
    (0xe003_ffb0_8ff7_ff02, 2, 2),
    (0xe003_ffb0_0ff7_ff00, 0, 0),
    (0xe003_ffb0_4ff7_ff01, 1, 1),
];
const ADMITTED_WORDS: [u64; 6] = [
    0xe003_ffb0_0ff7_ff04,
    0xe003_ffb0_8ff7_ff02,
    0xe003_ffb0_0ff7_ff00,
    0xe003_ffb0_4ff7_ff01,
    0xe003_ffb0_cff7_ff04,
    0xe003_ffb0_4ff7_ff05,
];

fn kind(raw: u64) -> InstructionKind {
    decode_word([raw as u32, (raw >> 32) as u32], 0).kind
}

fn is_admitted(raw: u64) -> bool {
    ADMITTED_WORDS.contains(&raw)
}

#[test]
fn exact_words_decode_to_closed_typed_state_and_neighbors_do_not() {
    for (raw, dst, component) in WORDS {
        assert_eq!(
            kind(raw),
            InstructionKind::LegacyVaryingLoad(ops::LegacyVaryingLoad {
                dst: Register(dst),
                varying: LegacyVarying::TextureCoordinate {
                    index: 0,
                    component,
                },
                interpolation: LegacyInterpolation::Pass,
                inv_w: Register(255),
            })
        );
        for bit in 0..64 {
            let neighbor = raw ^ (1_u64 << bit);
            assert_eq!(
                matches!(kind(neighbor), InstructionKind::LegacyVaryingLoad(_)),
                is_admitted(neighbor),
                "neighbor {neighbor:016x} from {raw:016x} bit {bit}"
            );
        }
    }
}

fn independent_nth_free(used: u32, ordinal: usize) -> Option<u16> {
    (0..32)
        .filter(|location| used & (1 << location) == 0)
        .nth(ordinal)
        .map(|location| location as u16)
}

#[test]
fn independent_cpu_oracle_covers_all_replay_words() {
    let masks = [0, 1, 0x55, 0x8000_001f, u32::MAX];
    let values = [
        (0.0_f32, 0.5_f32, 0x0000_0000),
        (-1.25, 0.25, 0xbea0_0000),
        (f32::MIN_POSITIVE, 2.0, 0x0100_0000),
        (-0.0, 1.0, 0x8000_0000),
    ];
    let mut evaluations = 0;
    for (raw, _, component) in WORDS {
        let varying = LegacyVarying::TextureCoordinate {
            index: 0,
            component,
        };
        for used in masks {
            assert_eq!(varying.location(used), independent_nth_free(used, 5));
            evaluations += 1;
        }
        for (input, reciprocal_w, expected_bits) in values {
            assert_eq!((input * reciprocal_w).to_bits(), expected_bits);
            evaluations += 1;
        }
        assert!(matches!(kind(raw), InstructionKind::LegacyVaryingLoad(_)));
    }
    assert_eq!(evaluations, 36);
}

#[test]
fn all_replay_words_validate_as_raw_and_optimized_spirv() {
    let mut header_words = [0; 32];
    header_words[0] = 5 << 10;
    header_words[18] = 0xf;
    let header = headers::parse(&header_words).unwrap();
    let mut raw_count = 0;
    let mut optimized_count = 0;
    for (raw, _, _) in WORDS {
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 0);
        let mut module = emit_unoptimized_with_graphics_state(
            &[instruction],
            Some(&header),
            &GraphicsState::default(),
        )
        .unwrap_or_else(|error| panic!("{raw:016x}: {error}"));
        let validation = spirv_validation::validate(&module.words);
        assert!(
            validation.status.success(),
            "{raw:016x}/raw: {}",
            String::from_utf8_lossy(&validation.stderr)
        );
        raw_count += 1;
        module.optimize();
        let validation = spirv_validation::validate(&module.words);
        assert!(
            validation.status.success(),
            "{raw:016x}/optimized: {}",
            String::from_utf8_lossy(&validation.stderr)
        );
        optimized_count += 1;
    }
    assert_eq!((raw_count, optimized_count), (4, 4));
}
