//! Exact FrontColorDiffuse IPA.SC lane recovered from the Novena replay.

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
    (0xe0c3_ffa8_0047_ff00, 0, 0),
    (0xe0c3_ffa8_4047_ff01, 1, 1),
    (0xe0c3_ffa8_8047_ff02, 2, 2),
    (0xe0c3_ffa8_c047_ff03, 3, 3),
];

fn kind(raw: u64) -> InstructionKind {
    decode_word([raw as u32, (raw >> 32) as u32], 0).kind
}

#[test]
fn exact_word_decodes_to_closed_typed_state_and_neighbors_do_not() {
    for (word, dst, component) in WORDS {
        assert_eq!(
            kind(word),
            InstructionKind::LegacyVaryingLoad(ops::LegacyVaryingLoad {
                dst: Register(dst),
                varying: LegacyVarying::FrontColorDiffuse { component },
                interpolation: LegacyInterpolation::State,
                inv_w: Register(4),
            })
        );
        for bit in 0..64 {
            let neighbor = word ^ (1_u64 << bit);
            assert!(
                !matches!(kind(neighbor), InstructionKind::LegacyVaryingLoad(_)),
                "neighbor {neighbor:016x} bit {bit} entered the exact path"
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
fn independent_cpu_oracle_covers_smooth_flat_and_layout() {
    let masks = [0, 1, 0x55, 0x8000_001f, u32::MAX];
    let values = [
        (0.0_f32, 0.5_f32, 2.0_f32, 0x0000_0000, 0x0000_0000),
        (-1.25, 0.25, -0.75, 0x3e70_0000, 0xbfa0_0000),
        (f32::MIN_POSITIVE, 2.0, 0.5, 0x0080_0000, 0x0080_0000),
        (-0.0, 1.0, -1.0, 0x0000_0000, 0x8000_0000),
    ];
    let mut evaluations = 0;
    for component in 0..4 {
        let varying = LegacyVarying::FrontColorDiffuse { component };
        for used in masks {
            assert_eq!(varying.location(used), independent_nth_free(used, 1));
            assert_eq!(varying.component(), component);
            evaluations += 1;
        }
    }
    for (input, reciprocal_w, multiplier, expected_smooth, expected_flat) in values {
        let smooth = input * reciprocal_w * multiplier;
        let flat = input;
        assert_eq!(smooth.to_bits(), expected_smooth);
        assert_eq!(flat.to_bits(), expected_flat);
        evaluations += 2;
    }
    assert_eq!(evaluations, 28);
}

#[test]
fn word_validates_as_raw_and_optimized_spirv_for_both_shading_states() {
    let mut header_words = [0; 32];
    header_words[0] = 5 << 10;
    header_words[18] = 0xf;
    let header = headers::parse(&header_words).unwrap();
    let mut raw_count = 0;
    let mut optimized_count = 0;
    for (word, _, _) in WORDS {
        let instruction = decode_word([word as u32, (word >> 32) as u32], 0);
        for flat in [false, true] {
            let mut module = emit_unoptimized_with_graphics_state(
                &[instruction],
                Some(&header),
                &GraphicsState {
                    legacy_color_flat_shading: Some(flat),
                    ..Default::default()
                },
            )
            .unwrap();
            let validation = spirv_validation::validate(&module.words);
            assert!(
                validation.status.success(),
                "raw/{word:016x}/{flat}: {}",
                String::from_utf8_lossy(&validation.stderr)
            );
            raw_count += 1;
            module.optimize();
            let validation = spirv_validation::validate(&module.words);
            assert!(
                validation.status.success(),
                "optimized/{word:016x}/{flat}: {}",
                String::from_utf8_lossy(&validation.stderr)
            );
            optimized_count += 1;
        }
    }
    assert_eq!((raw_count, optimized_count), (8, 8));
}
