//! Synthetic VOTE.VTG fixtures. This is reference matching under operator
//! policy, not hardware proof. See docs/reference-sources.md
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
    ir::{Instruction, InstructionKind as K, Predicate, VoteVtgMode, U28},
    translate_with_graphics_state,
};

const PT: u64 = 7 << 16;
const NOP: u64 = (0x50b0_u64 << 48) | PT; // Mesa nak/sm50.rs:3178-3184.

fn vote(mode: u64, immediate: u32, guard: u64) -> u64 {
    // Row/mask: envytools envydis/gm107.c:2053. R/A/RA48..49:
    // :1633-1637. Imm28 at bit20: Ryujinx Decoders/InstDecoders.cs:5231.
    // Guard16..18/invert19: envytools :371,408-411.
    ((0x50e0 | mode) << 48) | (u64::from(immediate) << 20) | (guard << 16)
}

fn decode(raw: u64) -> Instruction {
    decode_word([raw as u32, (raw >> 32) as u32], 0)
}

#[test]
fn every_mode_immediate_and_guard_is_visible_and_has_no_destinations() {
    for (mode, expected) in [
        (0, VoteVtgMode::R),
        (1, VoteVtgMode::A),
        (2, VoteVtgMode::Ra),
    ] {
        for immediate in [0, 1, 0x800000, 0xffffff, 0x1000000, 0x8000000, 0xfffffff] {
            for guard in 0..16 {
                let instruction = decode(vote(mode, immediate, guard));
                assert_eq!(
                    instruction.kind,
                    K::VoteVtg(ops::VoteVtg {
                        mode: expected,
                        immediate: U28::new(immediate).unwrap(),
                    })
                );
                assert_eq!(
                    instruction.predicate,
                    if guard == 7 {
                        None
                    } else {
                        Some(Predicate {
                            register: (guard & 7) as u8,
                            inverted: guard & 8 != 0,
                        })
                    }
                );
                assert!(instruction
                    .to_string()
                    .contains(&format!("VOTE.VTG.{expected:?} 0x{immediate:07x}")));
                // No Rd or predicate output: Ryujinx Decoders/InstTable.cs:320.
                assert_eq!(
                    instruction.kind.written_registers(),
                    shadowbox::ir::Written::NOTHING
                );
                assert!(!instruction.kind.may_write_predicate());
                assert!(instruction.kind.sources().is_empty());
            }
        }
    }
    // Every defined operand bit changes complete IR and disassembly. The
    // exception permits unchanged SPIR-V, but no field disappears from audit.
    let plain = decode(vote(0, 0, 7));
    for bit in 16..50 {
        let changed = decode(vote(0, 0, 7) ^ (1 << bit));
        assert_ne!(changed, plain, "bit{bit}");
        assert_ne!(changed.to_string(), plain.to_string(), "bit{bit}");
    }
}

#[test]
fn undefined_modes_and_unassigned_bits_remain_unknown() {
    for mode in 0..8 {
        let raw = vote(mode, 0xfffffff, 7);
        if mode >= 3 {
            assert_eq!(
                decode(raw).kind,
                K::Unknown(ops::Unknown {
                    raw: [raw as u32, (raw >> 32) as u32]
                })
            );
        } else {
            // No source assigns bits0..15 or 50: envytools
            // envydis/gm107.c:2053,1633-1637,371,408-411; Ryujinx
            // Decoders/InstDecoders.cs:5224-5231.
            for bit in (0..16).chain(std::iter::once(50)) {
                let changed = raw | (1 << bit);
                assert_eq!(
                    decode(changed).kind,
                    K::Unknown(ops::Unknown {
                        raw: [changed as u32, (changed >> 32) as u32]
                    }),
                    "mode{mode}, bit{bit}"
                );
            }
        }
    }
    // Opcode bits51..63 are fixed by the row mask (:2053). Some neighbors
    // are real instructions; none may enter the policy exception.
    for bit in 51..64 {
        assert!(!matches!(
            decode(vote(2, 0, 7) ^ (1 << bit)).kind,
            K::VoteVtg(ops::VoteVtg { .. })
        ));
    }
    assert_eq!(
        include_str!("../src/decoder/families.rs")
            .matches("x.guarded(InstructionKind::Nop(ops::Nop))")
            .count(),
        2
    );
}

fn record(stage: u32) -> Vec<u8> {
    // Stage10..13, geometry input count88..95, output topology120..123,
    // max vertices128..139, generic map432..559: Mesa nak/sph.rs:
    // 134,169,305-312,326-350; headers/nvidia/classes/cla097sph.h:37-42,66-73.
    let mut words = [0_u32; 20];
    words[0] = stage << 10;
    words[2] = 3 << 24;
    words[3] = 7 << 24;
    words[4] = 3;
    words[13] = 1 << 16;
    words.into_iter().flat_map(u32::to_le_bytes).collect()
}

fn state() -> GraphicsState<'static> {
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

fn fixture(raw: u64, slot: usize, stage: u32) -> Vec<u32> {
    // MOV32I: Mesa nak/sm50.rs:1938-1941,1951. Quad12..15, dst0..7
    // (:321-327), immediate20..51. Common guard above.
    let mov = |dst: u64, value: u64| (0x0100 << 48) | (value << 20) | PT | (15 << 12) | dst;
    // P0 = R0 == RZ, secondary output PT. envytools gm107.c:2028,
    // :203-207,368-369,373,1445-1453,1430-1434: dst3, secondary0, lhs8,
    // rhs20, accumulator39, EQ49..51, boolean AND45..46.
    let setp = (0x5b64 << 48) | (7 << 39) | (255 << 20) | PT | 7;
    // IADD immediate: Mesa nak/sm50.rs:1265-1286. dst0..7, lhs8..15,
    // literal20..38, guard16..19. P0 and inverted P0 follow VOTE.VTG.
    let add = |guard: u64, value: u64| (0x3810 << 48) | (value << 20) | (guard << 16) | 2;
    let mut instructions = vec![mov(0, u64::from(stage & 1)), mov(2, 0x12345678), setp];
    instructions.extend(std::iter::repeat_n(NOP, slot - 1));
    instructions.extend([raw, add(0, 0x111), add(8, 0x222)]);
    // AST generic location0: Mesa nak/sm50.rs:2886-2896. data R2,
    // offset/vertex RZ8/39, address0x80 at20, output32, count1 at47.
    instructions
        .push((0xeff0 << 48) | (255 << 39) | (1 << 32) | (0x80 << 20) | PT | (255 << 8) | 2);
    if stage == 4 {
        // OUT emit, stream RZ, unused handle/dst: Mesa sm50.rs:
        // 3260-3261,3274-3284. Emit bit39.
        instructions.push((0xfbe0 << 48) | (1 << 39) | (255 << 20) | PT | (255 << 8) | 255);
    }
    // EXIT: Mesa nak/sm50.rs:3101-3110. Control-word + three slots:
    // :3418-3428. Padding uses actual OpNop (:3178-3184).
    instructions.push((0xe300 << 48) | PT | 15);
    let mut words = Vec::new();
    for chunk in instructions.chunks(3) {
        words.extend([0, 0]);
        for index in 0..3 {
            let instruction = chunk.get(index).copied().unwrap_or(NOP);
            words.extend([instruction as u32, (instruction >> 32) as u32]);
        }
    }
    words
}

#[test]
fn vertex_tessellation_and_geometry_preserve_surrounding_code() {
    for stage in 1..=4 {
        let record = record(stage);
        let mut header_words: Vec<_> = record
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        header_words.resize(32, 0);
        let header = headers::parse(&header_words).unwrap();
        for slot in 1..=3 {
            let control = fixture(NOP, slot, stage);
            let control_ir = decode_stream(shadowbox::QuietRun, &control);
            let baseline = [
                emit_with_graphics_state(&control_ir, Some(&header), &state())
                    .unwrap()
                    .words,
                emit_unoptimized_with_graphics_state(&control_ir, Some(&header), &state())
                    .unwrap()
                    .words,
            ];
            for mode in 0..3 {
                for immediate in [0, 0xffffff] {
                    for guard in [7, 0, 8, 15] {
                        let words = fixture(vote(mode, immediate, guard), slot, stage);
                        let decoded = decode_stream(shadowbox::QuietRun, &words);
                        let vtg = decoded
                            .iter()
                            .find(|x| matches!(x.kind, K::VoteVtg(ops::VoteVtg { .. })))
                            .unwrap();
                        assert_eq!(vtg.address, 32 + slot as u32 * 8);
                        assert!(decoded
                            .iter()
                            .any(|x| matches!(x.kind, K::ISetP(ops::ISetP { .. }))));
                        assert!(decoded.iter().any(|x| matches!(
                            x.kind,
                            K::AttributeStore(ops::AttributeStore { .. })
                        )));
                        for (actual, expected) in [
                            emit_with_graphics_state(&decoded, Some(&header), &state())
                                .unwrap()
                                .words,
                            emit_unoptimized_with_graphics_state(&decoded, Some(&header), &state())
                                .unwrap()
                                .words,
                        ]
                        .into_iter()
                        .zip(&baseline)
                        {
                            assert_eq!(&actual, expected, "stage{stage}, slot{slot}, mode{mode}, immediate{immediate:x}, guard{guard}");
                            let validation = spirv_validation::validate(&actual);
                            assert!(
                                validation.status.success(),
                                "{}",
                                String::from_utf8_lossy(&validation.stderr)
                            );
                        }
                        let code: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                        let output = Result::from(translate_with_graphics_state(
                            shadowbox::QuietRun,
                            &ShaderInput {
                                code: &code,
                                program_record: &record,
                            },
                            &state(),
                        ))
                        .unwrap();
                        assert_eq!(output.spirv, baseline[0]);
                    }
                }
            }
        }
    }
}

#[test]
fn public_ir_cannot_hold_an_immediate_past_28_bits() {
    assert_eq!(U28::new(0x0fff_ffff).map(U28::get), Some(0x0fff_ffff));
    assert_eq!(U28::new(0x1000_0000), None);
    assert_eq!(U28::low_bits(0x1fff_ffff).get(), 0x0fff_ffff);
}

#[test]
fn rejected_encodings_keep_graphics_translation_failing() {
    // Row/mask: envytools envydis/gm107.c:2053. Mode3 OOPS: :1637.
    // Bits0,50 have no operand in that row or the reference decoder
    // (Ryujinx Decoders/InstDecoders.cs:5224-5231); bit51 leaves its opcode mask.
    for raw in [
        vote(3, 0, 7),
        vote(2, 0, 7) | 1,
        vote(2, 0, 7) | (1 << 50),
        vote(2, 0, 7) | (1 << 51),
    ] {
        assert!(matches!(decode(raw).kind, K::Unknown(ops::Unknown { .. })));
        for stage in 1..=4 {
            let record = record(stage);
            for slot in 1..=3 {
                let words = fixture(raw, slot, stage);
                let code: Vec<_> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                assert!(
                    Result::from(translate_with_graphics_state(
                        shadowbox::QuietRun,
                        &ShaderInput {
                            code: &code,
                            program_record: &record
                        },
                        &state(),
                    ))
                    .is_err(),
                    "word{raw:016x}, stage{stage}, slot{slot}"
                );
            }
        }
    }
}

#[test]
fn csm_exit_selectors_fall_through_exactly_like_never_taken_exit() {
    // EXIT CC0..4, keep-reference-count5: envytools gm107.c:1925,451-456.
    // Ryujinx InstEmitFlowControl.cs:153-181 evaluates EXIT through
    // GetCondition(.., IrConsts.False); InstEmitConditionCode.cs:52-82 has
    // no CSM24..29 case, so the reference never takes it. CC.F (0) is the
    // project's existing never-taken form and must emit the same module.
    let exit = |guard: u64, test: u64| (0xe300 << 48) | (guard << 16) | test;
    for stage in 1..=4 {
        let record = record(stage);
        let mut header_words: Vec<_> = record
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect();
        header_words.resize(32, 0);
        let header = headers::parse(&header_words).unwrap();
        for slot in 1..=3 {
            for guard in [7, 0, 8] {
                let never =
                    decode_stream(shadowbox::QuietRun, &fixture(exit(guard, 0), slot, stage));
                let expected = [
                    emit_with_graphics_state(&never, Some(&header), &state())
                        .unwrap()
                        .words,
                    emit_unoptimized_with_graphics_state(&never, Some(&header), &state())
                        .unwrap()
                        .words,
                ];
                for test in 24..=29 {
                    let words = fixture(exit(guard, test), slot, stage);
                    let decoded = decode_stream(shadowbox::QuietRun, &words);
                    assert!(decoded.iter().any(|x| x.kind
                        == K::Exit(ops::Exit {
                            cc_test: test as u8
                        })));
                    // The code after the CSM EXIT stays live.
                    assert!(decoded
                        .iter()
                        .any(|x| matches!(x.kind, K::AttributeStore(ops::AttributeStore { .. }))));
                    let actual = [
                        emit_with_graphics_state(&decoded, Some(&header), &state())
                            .unwrap()
                            .words,
                        emit_unoptimized_with_graphics_state(&decoded, Some(&header), &state())
                            .unwrap()
                            .words,
                    ];
                    assert_eq!(
                        actual, expected,
                        "stage{stage}, slot{slot}, guard{guard}, cc{test}"
                    );
                    let validation = spirv_validation::validate(&actual[0]);
                    assert!(
                        validation.status.success(),
                        "{}",
                        String::from_utf8_lossy(&validation.stderr)
                    );
                    // Keep-reference-count mode stays Unknown (gm107.c:1925).
                    assert!(matches!(
                        decode(exit(guard, test) | (1 << 5)).kind,
                        K::Unknown(ops::Unknown { .. })
                    ));
                }
            }
        }
    }
}
