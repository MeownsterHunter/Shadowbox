//! Project-authored reference-matching fixtures. Ryujinx's integer-zero
//! default is policy, not hardware proof: docs/reference-sources.md
#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::{decode_stream, decode_word},
    emitter::{emit_unoptimized_with_graphics_state, emit_with_graphics_state},
    headers,
    interface::{
        GeometryInput, GraphicsState, ShaderInput, TessellationDomain, TessellationSpacing,
        TessellationState,
    },
    ir::{InstructionKind as K, Predicate, Register},
    translate_with_graphics_state,
};

fn read(index: u8, dst: u8, guard: u8) -> u64 {
    // S2R opcode/destination/index: Mesa nak/sm50.rs:321-327,3220-3224.
    // Guard16..18/invert19: envytools envydis/gm107.c:371,408-411.
    (0xf0c8_u64 << 48) | u64::from(dst) | (u64::from(guard) << 16) | (u64::from(index) << 20)
}

fn record(stage: u32) -> Vec<u8> {
    // Stage10..13, geometry input count88..95, output topology120..123
    // and max vertices128..139: Mesa nak/sph.rs:305-312,326-350;
    // src/nouveau/headers/nvidia/classes/cla097sph.h:37-42,66-73.
    let mut words = [0_u32; 20];
    words[0] = stage << 10;
    words[2] = 3 << 24;
    words[3] = 7 << 24;
    words[4] = 3;
    words.into_iter().flat_map(u32::to_le_bytes).collect()
}

fn driver_state() -> GraphicsState<'static> {
    GraphicsState {
        geometry_input: Some(GeometryInput::Triangles),
        tessellation: Some(TessellationState {
            domain: TessellationDomain::Triangles,
            spacing: TessellationSpacing::Equal,
            clockwise: false,
        }),
        ..GraphicsState::default()
    }
}

fn bundle(raw: u64, slot: usize) -> Vec<u8> {
    // Mesa nak/sm50.rs:3418-3428: control word + three instruction slots.
    // Padding is real OpNop (3178-3184); EXIT is 3101-3110.
    let nop = (0x50b0_u64 << 48) | (7 << 16);
    let exit = (0xe300_u64 << 48) | (7 << 16) | 15;
    let mut words = [0, nop, nop, nop, 0, exit, nop, nop];
    words[slot] = raw;
    words.into_iter().flat_map(u64::to_le_bytes).collect()
}

#[test]
fn unpacked_wscale_reads_decode_and_emit_in_every_stage() {
    // Indices/names: envytools envydis/gm107.c:148-149. RZ is register255
    // (Mesa nak/sm50.rs:204-205); each read must remain S2R even for RZ.
    for index in [0x1e, 0x1f] {
        for dst in [0, 17, 254, 255] {
            for guard in [7, 0, 8] {
                let raw = read(index, dst, guard);
                let words = [raw as u32, (raw >> 32) as u32];
                let instruction = decode_word(words, 0);
                assert_eq!(
                    instruction.kind,
                    K::S2R(ops::S2R {
                        dst: Register(dst),
                        idx: index
                    })
                );
                assert_eq!(
                    instruction.predicate,
                    match guard {
                        7 => None,
                        0 => Some(Predicate {
                            register: 0,
                            inverted: false
                        }),
                        8 => Some(Predicate {
                            register: 0,
                            inverted: true
                        }),
                        _ => unreachable!(),
                    }
                );
                for stage in 0..=5 {
                    let bytes = record(stage);
                    let mut header_words: Vec<_> = bytes
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| u32::from_le_bytes(*b))
                        .collect();
                    // The direct parser takes the project's padded header
                    // buffer; production translation pads the SM5 record.
                    header_words.resize(32, 0);
                    let header = headers::parse(&header_words).unwrap();
                    for result in [
                        emit_with_graphics_state(
                            std::slice::from_ref(&instruction),
                            Some(&header),
                            &driver_state(),
                        ),
                        emit_unoptimized_with_graphics_state(
                            std::slice::from_ref(&instruction),
                            Some(&header),
                            &driver_state(),
                        ),
                    ] {
                        let module = result.unwrap();
                        let validation = spirv_validation::validate(&module.words);
                        assert!(
                            validation.status.success(),
                            "stage{stage}, SR{index:02x}, R{dst}, guard{guard}: {}",
                            String::from_utf8_lossy(&validation.stderr)
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn bundled_wscale_reads_translate_in_every_stage_and_slot() {
    for stage in 0..=5 {
        let record = record(stage);
        for slot in 1..=3 {
            // Positive control: a known driver specialization read proves
            // the stage, driver state and bundle fixture can translate.
            let known = bundle(read(0x12, 0, 7), slot);
            Result::from(translate_with_graphics_state(
                shadowbox::QuietRun,
                &ShaderInput {
                    code: &known,
                    program_record: &record,
                },
                &driver_state(),
            ))
            .unwrap();
            for index in [0x1e, 0x1f] {
                for dst in [0, 255] {
                    let raw = read(index, dst, 7);
                    let code = bundle(raw, slot);
                    let words: Vec<_> = code
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|b| u32::from_le_bytes(*b))
                        .collect();
                    let decoded = decode_stream(shadowbox::QuietRun, &words);
                    let instruction = decoded
                        .iter()
                        .find(|x| x.address == slot as u32 * 8)
                        .unwrap();
                    assert_eq!(
                        instruction.kind,
                        K::S2R(ops::S2R {
                            dst: Register(dst),
                            idx: index
                        })
                    );
                    let output = Result::from(translate_with_graphics_state(
                        shadowbox::QuietRun,
                        &ShaderInput {
                            code: &code,
                            program_record: &record,
                        },
                        &driver_state(),
                    ))
                    .unwrap();
                    let validation = spirv_validation::validate(&output.spirv);
                    assert!(
                        validation.status.success(),
                        "stage{stage}, slot{slot}, SR{index:02x}, R{dst}: {}",
                        String::from_utf8_lossy(&validation.stderr)
                    );
                }
            }
        }
    }
}

#[test]
fn wscale_unnamed_fields_remain_unknown() {
    // Mesa nak/sm50.rs:3220-3224 only encodes dst0..7/index20..27.
    // Guard16..19 is envytools envydis/gm107.c:371,408-411.
    // All other operand bits and unimplemented opcode variants must reject.
    for index in [0x1e, 0x1f] {
        let raw = read(index, 17, 7);
        for bit in (8..16).chain(28..51) {
            let changed = raw | 1 << bit;
            let words = [changed as u32, (changed >> 32) as u32];
            assert_eq!(
                decode_word(words, 0).kind,
                K::Unknown(ops::Unknown { raw: words }),
                "SR{index:02x}, bit{bit}"
            );
        }
    }
}
