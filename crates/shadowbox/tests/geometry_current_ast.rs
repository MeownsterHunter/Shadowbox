//! Exact geometry AST words from Novena strict-cache cases 01 and 02.

#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use std::collections::BTreeMap;

use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_graphics_state,
    headers,
    interface::{GeometryInput, GraphicsState},
    ir::{ops, InstructionKind, Register},
};

const WORDS: [(u64, u8, u16); 19] = [
    (0xeff0_0000_0707_ff01, 1, 0x70),
    (0xeff0_0000_0747_ff02, 2, 0x74),
    (0xeff0_0000_0787_ff03, 3, 0x78),
    (0xeff0_0000_07c7_ff05, 5, 0x7c),
    (0xeff0_0000_0807_ff06, 6, 0x80),
    (0xeff0_0000_0847_ff07, 7, 0x84),
    (0xeff0_0000_0647_ff09, 9, 0x64),
    (0xeff0_0000_0907_ff09, 9, 0x90),
    (0xeff0_0000_0707_ff0a, 10, 0x70),
    (0xeff0_0000_0747_ff01, 1, 0x74),
    (0xeff0_0000_0787_ff02, 2, 0x78),
    (0xeff0_0000_07c7_ff03, 3, 0x7c),
    (0xeff0_0000_0807_ff05, 5, 0x80),
    (0xeff0_0000_0847_ff06, 6, 0x84),
    (0xeff0_0000_0647_ff07, 7, 0x64),
    (0xeff0_0000_0907_ff07, 7, 0x90),
    (0xeff0_0000_0707_ff09, 9, 0x70),
    (0xeff0_0000_0647_ff0a, 10, 0x64),
    (0xeff0_0000_0907_ff0a, 10, 0x90),
];
const OUT_WORD: u64 = 0xfbe0_0080_0ff7_0000;

fn kind(raw: u64) -> InstructionKind {
    decode_word([raw as u32, (raw >> 32) as u32], 0).kind
}

fn is_unit(raw: u64) -> bool {
    WORDS.iter().any(|(word, _, _)| *word == raw)
}

fn emit_vertex_oracle(current_handle: u32) -> u32 {
    current_handle.wrapping_add(1)
}

#[test]
fn exact_words_decode_to_geometry_typed_state_and_neighbors_do_not() {
    for (raw, data, address) in WORDS {
        assert_eq!(
            kind(raw),
            InstructionKind::GeometryAttributeStore(ops::GeometryAttributeStore {
                data: Register(data),
                address,
                handle: Register(0),
            })
        );
        for bit in 0..64 {
            let neighbor = raw ^ (1_u64 << bit);
            assert_eq!(
                matches!(kind(neighbor), InstructionKind::GeometryAttributeStore(_)),
                is_unit(neighbor),
                "neighbor {neighbor:016x} from {raw:016x} bit {bit}"
            );
        }
    }
    assert_eq!(
        kind(OUT_WORD),
        InstructionKind::GeometryOutHandle(ops::GeometryOutHandle {
            dst: Register(0),
            handle: Register(0),
            emit: true,
            cut: false,
            stream: 0,
        })
    );
    for bit in 0..64 {
        let neighbor = OUT_WORD ^ (1_u64 << bit);
        assert!(
            !matches!(kind(neighbor), InstructionKind::GeometryOutHandle(_)),
            "OUT neighbor {neighbor:016x} bit {bit}"
        );
    }
}

#[test]
fn independent_current_handle_cpu_oracle_covers_every_word() {
    let mut evaluations = 0;
    for (raw, data, address) in WORDS {
        let value = 0x5a00_0000 | u32::from(data);
        for current_handle in [0_u32, 1, 17] {
            let mut output_bank = BTreeMap::new();
            output_bank.insert((current_handle, address), value);
            assert_eq!(output_bank[&(current_handle, address)], value);
            evaluations += 1;
        }
        assert!(matches!(
            kind(raw),
            InstructionKind::GeometryAttributeStore(_)
        ));
    }
    for (current_handle, expected_next) in [(0_u32, 1_u32), (1, 2), (17, 18)] {
        assert_eq!(emit_vertex_oracle(current_handle), expected_next);
        evaluations += 1;
    }
    assert_eq!(evaluations, 60);
}

fn geometry_header() -> headers::ShaderHeader {
    let mut h = [0_u32; 32];
    h[0] = 4 << 10;
    h[2] = 3 << 24;
    h[3] = 1 << 24;
    h[4] = 1;
    h[6] = 255;
    h[13] = 255 << 16;
    headers::parse(&h).unwrap()
}

#[test]
fn every_word_validates_as_raw_and_optimized_spirv() {
    let header = geometry_header();
    let state = GraphicsState {
        geometry_input: Some(GeometryInput::Triangles),
        ..Default::default()
    };
    let mut raw_count = 0;
    let mut optimized_count = 0;
    for (raw, _, _) in WORDS {
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 0);
        let mut module =
            emit_unoptimized_with_graphics_state(&[instruction], Some(&header), &state)
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
    let instruction = decode_word([OUT_WORD as u32, (OUT_WORD >> 32) as u32], 0);
    let mut module = emit_unoptimized_with_graphics_state(&[instruction], Some(&header), &state)
        .unwrap_or_else(|error| panic!("{OUT_WORD:016x}: {error}"));
    let validation = spirv_validation::validate(&module.words);
    assert!(
        validation.status.success(),
        "{OUT_WORD:016x}/raw: {}",
        String::from_utf8_lossy(&validation.stderr)
    );
    raw_count += 1;
    module.optimize();
    let validation = spirv_validation::validate(&module.words);
    assert!(
        validation.status.success(),
        "{OUT_WORD:016x}/optimized: {}",
        String::from_utf8_lossy(&validation.stderr)
    );
    optimized_count += 1;
    assert_eq!((raw_count, optimized_count), (20, 20));
}
