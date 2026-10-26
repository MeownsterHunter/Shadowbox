//! Exact BackColorDiffuse AST lane recovered from the Novena replay.

#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_graphics_state,
    headers,
    interface::GraphicsState,
    ir::{ops, InstructionKind, LegacyVarying, Register},
};

const WORDS: [u64; 49] = [
    0xeff0_7f80_2807_ff2a,
    0xeff0_7f80_2847_ff2b,
    0xeff0_7f80_2887_ff2c,
    0xeff0_7f80_28c7_ff29,
    0xeff0_7f80_2907_ff26,
    0xeff0_7f80_2947_ff25,
    0xeff0_7f80_2987_ff24,
    0xeff0_7f80_29c7_ff23,
    0xeff0_7f80_2a07_ff2a,
    0xeff0_7f80_2a47_ff2b,
    0xeff0_7f80_2a87_ff2c,
    0xeff0_7f80_2ac7_ff29,
    0xeff0_7f80_2b07_ff26,
    0xeff0_7f80_2b47_ff25,
    0xeff0_7f80_2b87_ff24,
    0xeff0_7f80_2bc7_ff23,
    0xeff0_7f80_2e87_ff06,
    0xeff0_7f80_3007_ff22,
    0xeff0_7f80_3047_ff21,
    0xeff0_7f80_3087_ff20,
    0xeff0_7f80_30c7_ff1f,
    0xeff0_7f80_3107_ff1e,
    0xeff0_7f80_3147_ff1d,
    0xeff0_7f80_3187_ff19,
    0xeff0_7f80_31c7_ff1c,
    0xeff0_7f80_3207_ff1b,
    0xeff0_7f80_3247_ff1a,
    0xeff0_7f80_3287_ff15,
    0xeff0_7f80_32c7_ff18,
    0xeff0_7f80_3307_ff17,
    0xeff0_7f80_3347_ff16,
    0xeff0_7f80_3387_ff11,
    0xeff0_7f80_33c7_ff14,
    0xeff0_7f80_3407_ff13,
    0xeff0_7f80_3447_ff12,
    0xeff0_7f80_3487_ff0d,
    0xeff0_7f80_34c7_ff10,
    0xeff0_7f80_3507_ff0f,
    0xeff0_7f80_3547_ff0e,
    0xeff0_7f80_3587_ff11,
    0xeff0_7f80_35c7_ff0c,
    0xeff0_7f80_3607_ff0b,
    0xeff0_7f80_3647_ff0a,
    0xeff0_7f80_3687_ff08,
    0xeff0_7f80_36c7_ff19,
    0xeff0_7f80_3707_ff03,
    0xeff0_7f80_3747_ff02,
    0xeff0_7f80_3787_ff01,
    0xeff0_7f80_37c7_ff00,
];

fn kind(raw: u64) -> InstructionKind {
    decode_word([raw as u32, (raw >> 32) as u32], 0).kind
}

#[test]
fn exact_word_decodes_to_closed_typed_state_and_neighbors_do_not() {
    for word in WORDS {
        let address = ((word >> 20) & 0x3ff) as u16;
        let varying = LegacyVarying::decode(address).unwrap();
        assert_eq!(
            kind(word),
            InstructionKind::LegacyVaryingStore(ops::LegacyVaryingStore {
                data: Register(word as u8),
                varying,
            })
        );
        for bit in 0..64 {
            let neighbor = word ^ (1_u64 << bit);
            if WORDS.contains(&neighbor) {
                let address = ((neighbor >> 20) & 0x3ff) as u16;
                assert_eq!(
                    kind(neighbor),
                    InstructionKind::LegacyVaryingStore(ops::LegacyVaryingStore {
                        data: Register(neighbor as u8),
                        varying: LegacyVarying::decode(address).unwrap(),
                    })
                );
            } else {
                assert!(
                    !matches!(kind(neighbor), InstructionKind::LegacyVaryingStore(_)),
                    "neighbor {neighbor:016x} bit {bit} entered the exact path"
                );
            }
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
fn independent_cpu_oracle_covers_source_lane_and_destination_mapping() {
    let masks = [0, 1, 0x55, 0x8000_001f, u32::MAX];
    let mut evaluations = 0;
    for word in WORDS {
        let address = ((word >> 20) & 0x3ff) as u16;
        let varying = LegacyVarying::decode(address).unwrap();
        let (ordinal, component) = match address {
            0x280..=0x2bc => (1 + (address - 0x280) / 16, ((address & 0xf) / 4) as u8),
            0x2e8 => (0, 0),
            0x300..=0x37c => (5 + (address - 0x300) / 16, ((address & 0xf) / 4) as u8),
            _ => unreachable!(),
        };
        for used in masks {
            assert_eq!(
                varying.location(used),
                independent_nth_free(used, usize::from(ordinal))
            );
            evaluations += 1;
        }
        assert_eq!(varying.component(), component);
        evaluations += 1;
        let bits = 0x8000_0000 | u32::from(word as u8);
        let mut destination = [0_u32; 4];
        destination[usize::from(component)] = bits;
        assert_eq!(destination[usize::from(component)], bits);
        evaluations += 1;
    }
    assert_eq!(evaluations, 343);
}

#[test]
fn word_validates_as_raw_and_optimized_spirv() {
    let mut header_words = [0; 32];
    header_words[0] = 1 << 10;
    let header = headers::parse(&header_words).unwrap();
    let mut raw_count = 0;
    let mut optimized_count = 0;
    for word in WORDS {
        let instruction = decode_word([word as u32, (word >> 32) as u32], 0);
        let mut module = emit_unoptimized_with_graphics_state(
            &[instruction],
            Some(&header),
            &GraphicsState::default(),
        )
        .unwrap();
        let validation = spirv_validation::validate(&module.words);
        assert!(
            validation.status.success(),
            "raw/{word:016x}: {}",
            String::from_utf8_lossy(&validation.stderr)
        );
        raw_count += 1;
        module.optimize();
        let validation = spirv_validation::validate(&module.words);
        assert!(
            validation.status.success(),
            "optimized/{word:016x}: {}",
            String::from_utf8_lossy(&validation.stderr)
        );
        optimized_count += 1;
    }
    assert_eq!((raw_count, optimized_count), (49, 49));
}
