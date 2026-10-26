//! Exact PointSize AST exposed by Novena strict-cache cases 09 through 13.

#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_graphics_state,
    headers,
    interface::GraphicsState,
    ir::{ops, InstructionKind, Register},
};

const WORD: u64 = 0xeff0_7f80_06c7_ff05;

fn kind(raw: u64) -> InstructionKind {
    decode_word([raw as u32, (raw >> 32) as u32], 0).kind
}

#[test]
fn exact_word_decodes_to_point_size_and_every_neighbor_stays_out() {
    assert_eq!(
        kind(WORD),
        InstructionKind::PointSizeStore(ops::PointSizeStore { data: Register(5) })
    );
    for bit in 0..64 {
        let neighbor = WORD ^ (1_u64 << bit);
        assert!(
            !matches!(kind(neighbor), InstructionKind::PointSizeStore(_)),
            "neighbor {neighbor:016x} bit {bit} entered the exact path"
        );
    }
}

#[test]
fn independent_cpu_oracle_covers_fields_and_bit_exact_output() {
    let source = (WORD & 0xff) as u8;
    let address = ((WORD >> 20) & 0x3ff) as u16;
    let components = ((WORD >> 47) & 3) as u8 + 1;
    let offset = ((WORD >> 8) & 0xff) as u8;
    let patch = (WORD >> 31) & 1 != 0;
    assert_eq!(
        (source, address, components, offset, patch),
        (5, 0x6c, 1, 0xff, false)
    );

    let mut evaluations = 0;
    for bits in [0_u32, 0x8000_0000, 0x3f80_0000, 0x7f80_0000, 0x7fc0_1234] {
        let output = f32::from_bits(bits);
        assert_eq!(output.to_bits(), bits);
        evaluations += 1;
    }
    for bit in 0..64 {
        assert!(!matches!(
            kind(WORD ^ (1_u64 << bit)),
            InstructionKind::PointSizeStore(_)
        ));
        evaluations += 1;
    }
    assert_eq!(evaluations, 69);
}

#[test]
fn exact_word_validates_as_raw_and_optimized_spirv() {
    let mut header_words = [0; 32];
    header_words[0] = 1 << 10;
    let header = headers::parse(&header_words).unwrap();
    let instruction = decode_word([WORD as u32, (WORD >> 32) as u32], 0);
    let mut module = emit_unoptimized_with_graphics_state(
        &[instruction],
        Some(&header),
        &GraphicsState::default(),
    )
    .unwrap();
    let raw = spirv_validation::validate(&module.words);
    assert!(
        raw.status.success(),
        "raw: {}",
        String::from_utf8_lossy(&raw.stderr)
    );
    module.optimize();
    let optimized = spirv_validation::validate(&module.words);
    assert!(
        optimized.status.success(),
        "optimized: {}",
        String::from_utf8_lossy(&optimized.stderr)
    );
}
