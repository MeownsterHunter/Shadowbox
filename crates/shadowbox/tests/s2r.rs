//! Project-owned S2R execution inputs. Encoding: Mesa nak/sm50.rs:3220-3224;
//! semantics: Ryujinx Instructions/InstEmitMove.cs:78-215. No GPU is needed.
#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::spirv::Op;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_graphics_state,
    headers,
    interface::{
        GeometryInput, GraphicsState, TessellationDomain, TessellationSpacing, TessellationState,
    },
    ir::{Instruction, InstructionKind as K, Predicate, Register, Source},
};
use std::collections::HashMap;

fn s2r(dst: u8, index: u8) -> Instruction {
    // Destination0..7, index20..27: Mesa sm50.rs:3220-3224.
    // Common guard16..19: envytools gm107.c:371,408-419.
    let raw = (0xf0c8u64 << 48) | u64::from(dst) | (7 << 16) | (u64::from(index) << 20);
    decode_word([raw as u32, (raw >> 32) as u32], 0)
}
fn header(stage: u32) -> headers::ShaderHeader {
    // Mesa sph.rs:134,169 and cla097sph.h:37-42,66-73.
    let mut words = [0u32; 32];
    words[0] = stage << 10;
    words[2] = 3 << 24;
    words[3] = 7 << 24;
    words[4] = 3;
    headers::parse(&words).unwrap()
}
fn state(topology: GeometryInput) -> GraphicsState<'static> {
    GraphicsState {
        geometry_input: Some(topology),
        tessellation: Some(TessellationState {
            domain: TessellationDomain::Triangles,
            spacing: TessellationSpacing::Equal,
            clockwise: false,
        }),
        ..GraphicsState::default()
    }
}
fn module(mut xs: Vec<Instruction>, stage: u32, state: GraphicsState) -> Vec<u32> {
    xs.push(Instruction {
        address: 0,
        predicate: None,
        kind: K::Exit(ops::Exit { cc_test: 15 }),
    });
    for (i, instruction) in xs.iter_mut().enumerate() {
        instruction.address = i as u32 * 8;
    }
    let words = emit_unoptimized_with_graphics_state(&xs, Some(&header(stage)), &state)
        .unwrap()
        .words;
    let result = spirv_validation::validate(&words);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    words
}
fn ops(words: &[u32]) -> Vec<(Op, &[u32])> {
    let mut at = 5;
    let mut result = Vec::new();
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        result.push((Op::of(words[at]), &words[at + 1..at + count]));
        at += count;
    }
    result
}

// Execute just the emitted integer/vector subset, with external builtin inputs.
// Branches follow the actual SPIR-V labels and unhandled executable ops fail.
fn run(words: &[u32], inputs: &HashMap<u32, Vec<u32>>) -> [u32; 256] {
    let instructions = ops(words);
    let labels: HashMap<_, _> = instructions
        .iter()
        .enumerate()
        .filter(|(_, (op, _))| *op == Op::Label)
        .map(|(i, (_, a))| (a[0], i))
        .collect();
    let builtins: HashMap<_, _> = instructions
        .iter()
        .filter(|(op, a)| *op == Op::Decorate && a[1] == 11)
        .map(|(_, a)| (a[0], a[2]))
        .collect();
    let mut values: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut memory: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut locals = Vec::new();
    let mut at = 0;
    for _ in 0..10_000 {
        let (op, a) = instructions[at];
        let get = |i: usize| values[&a[i]][0];
        let result = match op {
            Op::Undef => None, // Texture-only undef is declared, never used by S2R.
            Op::ConstantTrue => Some(vec![1]),
            Op::ConstantFalse => Some(vec![0]),
            Op::Constant | Op::SpecConstant => Some(vec![a[2]]),
            Op::Variable => {
                if a[2] == 7 {
                    locals.push(a[1]);
                }
                let initial = if a[2] == 1 {
                    inputs
                        .get(&builtins[&a[1]])
                        .expect("missing builtin input")
                        .clone()
                } else {
                    vec![0]
                };
                memory.insert(a[1], initial);
                None
            }
            Op::Load => Some(memory[&a[2]].clone()),
            Op::Store => {
                memory.insert(a[0], values[&a[1]].clone());
                None
            }
            Op::VectorExtractDynamic => Some(vec![values[&a[2]][get(3) as usize]]),
            Op::CompositeExtract => Some(vec![values[&a[2]][a[3] as usize]]),
            Op::Bitcast => Some(values[&a[2]].clone()),
            Op::LogicalNot => Some(vec![u32::from(get(2) == 0)]),
            Op::Select => Some(values[&a[if get(2) != 0 { 3 } else { 4 }]].clone()),
            Op::IEqual => Some(vec![u32::from(get(2) == get(3))]),
            Op::INotEqual => Some(vec![u32::from(get(2) != get(3))]),
            Op::ShiftRightLogical => Some(vec![get(2).checked_shr(get(3)).unwrap()]),
            Op::ShiftLeftLogical => Some(vec![get(2).checked_shl(get(3)).unwrap()]),
            Op::BitwiseOr => Some(vec![get(2) | get(3)]),
            Op::BitwiseAnd => Some(vec![get(2) & get(3)]),
            Op::CompositeConstruct => Some(a[2..].iter().map(|id| values[id][0]).collect()),
            Op::Branch => {
                at = labels[&a[0]];
                continue;
            }
            Op::BranchConditional => {
                at = labels[&a[if get(0) != 0 { 1 } else { 2 }]];
                continue;
            }
            Op::Switch => {
                let selector = get(0);
                let target = a[2..]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .find(|pair| pair[0] == selector)
                    .map_or(a[1], |p| p[1]);
                at = labels[&target];
                continue;
            }
            Op::Return => return std::array::from_fn(|i| memory[&locals[i]][0]),
            Op::ExtInstImport
            | Op::MemoryModel
            | Op::EntryPoint
            | Op::ExecutionMode
            | Op::Capability
            | Op::TypeVoid
            | Op::TypeBool
            | Op::TypeInt
            | Op::TypeFloat
            | Op::TypeVector
            | Op::TypeArray
            | Op::TypeStruct
            | Op::TypePointer
            | Op::TypeFunction
            | Op::Function
            | Op::FunctionEnd
            | Op::Decorate
            | Op::MemberDecorate
            | Op::LoopMerge
            | Op::SelectionMerge
            | Op::Label => None,
            other => panic!("unsupported executable opcode {}", other.number()),
        };
        if let Some(result) = result {
            values.insert(a[1], result);
        }
        at += 1;
    }
    panic!("SPIR-V did not return");
}

#[test]
fn masks_select_the_correct_word_and_match_every_lane() {
    // Ryujinx InstEmitMove.cs:149-162,183-215; IoMap.cs:35-40.
    // The builtin numbers intentionally differ from the Maxwell ordering.
    let xs = (0..5)
        .map(|i| s2r(i, 0x38 + i))
        .chain([s2r(5, 0)])
        .collect();
    let words = module(xs, 5, GraphicsState::default());
    assert!(ops(&words)
        .iter()
        .any(|(op, a)| *op == Op::Capability && *a == [64]));
    assert!(ops(&words)
        .iter()
        .any(|(op, a)| *op == Op::Capability && *a == [61]));
    assert_eq!(
        ops(&words)
            .iter()
            .filter(|(op, _)| *op == Op::VectorExtractDynamic)
            .count(),
        5
    );
    for size in [4, 8, 16, 32, 64, 128u32] {
        for lane in 0..size {
            let mut inputs = HashMap::from([(41, vec![lane])]);
            for builtin in [4416, 4417, 4418, 4419, 4420] {
                let mut mask = vec![0u32; 4];
                for bit in 0..size {
                    let set = match builtin {
                        4416 => bit == lane,
                        4417 => bit >= lane,
                        4418 => bit > lane,
                        4419 => bit <= lane,
                        4420 => bit < lane,
                        _ => unreachable!(),
                    };
                    if set {
                        mask[bit as usize / 32] |= 1 << (bit % 32);
                    }
                }
                inputs.insert(builtin, mask);
            }
            let regs = run(&words, &inputs);
            let local = lane % 32;
            let available = u32::MAX >> (32 - size.min(32));
            let lt = ((1u64 << local) - 1) as u32;
            let le = lt | 1 << local;
            assert_eq!(
                &regs[..6],
                &[1 << local, lt, le, !le & available, !lt & available, local],
                "size={size} lane={lane}"
            );
        }
    }
}

#[test]
fn packed_thread_and_cta_components_are_exact() {
    // Ryujinx InstEmitMove.cs:118-147; IoMap.cs:20,44.
    let words = module(
        [0x20, 0x21, 0x22, 0x23, 0x25, 0x26, 0x27]
            .into_iter()
            .enumerate()
            .map(|(i, index)| s2r(i as u8, index))
            .collect(),
        0,
        GraphicsState::default(),
    );
    for tid in [[0, 0, 0], [1023, 511, 63], [1, 2, 3]] {
        let regs = run(
            &words,
            &HashMap::from([(27, tid.to_vec()), (26, vec![41, 42, 43])]),
        );
        assert_eq!(
            &regs[..7],
            &[
                tid[0] | tid[1] << 16 | tid[2] << 26,
                tid[0],
                tid[1],
                tid[2],
                41,
                42,
                43
            ]
        );
    }
}

#[test]
fn invocation_info_uses_dynamic_patch_size_or_input_topology() {
    // Ryujinx InstEmitMove.cs:94-114; InputTopology.cs:26-34.
    for stage in [2, 3] {
        let words = module(vec![s2r(0, 0x1d)], stage, state(GeometryInput::Triangles));
        for count in [1, 3, 32] {
            assert_eq!(
                run(&words, &HashMap::from([(14, vec![count]), (8, vec![0])]))[0],
                count << 16
            );
        }
    }
    for stage in [1, 4] {
        for (topology, count) in [
            (GeometryInput::Points, 1),
            (GeometryInput::Lines, 2),
            (GeometryInput::LinesAdjacency, 4),
            (GeometryInput::Triangles, 3),
            (GeometryInput::TrianglesAdjacency, 6),
        ] {
            let words = module(vec![s2r(0, 0x1d)], stage, state(topology));
            assert_eq!(run(&words, &HashMap::new())[0], count << 16);
        }
    }
    for stage in [0, 5] {
        let words = module(vec![s2r(0, 0x1d)], stage, GraphicsState::default());
        assert_eq!(run(&words, &HashMap::new())[0], 0);
    }
}

#[test]
fn thread_kill_and_invocation_id_and_y_direction_keep_their_values() {
    // Ryujinx InstEmitMove.cs:82-92; CodeGenContext.cs:342-347;
    // IntermediateRepresentation/IrConsts.cs:5-6: bool true is -1.
    let words = module(
        vec![s2r(0, 0x13), s2r(1, 0x12)],
        5,
        GraphicsState::default(),
    );
    for helper in [0, 1] {
        let regs = run(&words, &HashMap::from([(23, vec![helper])]));
        assert_eq!(
            &regs[..2],
            &[if helper == 0 { 0 } else { u32::MAX }, 1.0f32.to_bits()]
        );
    }
    let words = module(vec![s2r(0, 0x13)], 0, GraphicsState::default());
    assert_eq!(run(&words, &HashMap::new())[0], 0);
    for stage in [2, 4] {
        let words = module(vec![s2r(0, 0x11)], stage, state(GeometryInput::Triangles));
        assert_eq!(run(&words, &HashMap::from([(8, vec![2])]))[0], 2);
    }
}

#[test]
fn s2r_predication_preserves_the_old_destination() {
    let mut read = s2r(0, 0x38);
    read.predicate = Some(Predicate {
        register: 0,
        inverted: false,
    });
    let words = module(
        vec![
            Instruction {
                address: 0,
                predicate: None,
                kind: K::Mov(ops::Mov {
                    quad_lanes: 15,
                    dst: Register(0),
                    src: Source::Immediate(0x1234abcd),
                }),
            },
            read,
        ],
        0,
        GraphicsState::default(),
    );
    assert_eq!(
        run(
            &words,
            &HashMap::from([(41, vec![64]), (4416, vec![0, 0, 1, 0])])
        )[0],
        0x1234abcd
    );
}

#[test]
fn reference_matched_wscale_zero_and_guards_in_every_stage() {
    // Ryujinx Instructions/InstEmitMove.cs:76-168,170: both indices reach
    // default165-167 and copy integer zero, regardless of shader stage.
    // Encoding: envytools envydis/gm107.c:148-149; Mesa sm50.rs:3220-3224.
    // This checks reference matching, not Maxwell hardware readback.
    for stage in 0..=5 {
        for index in [0x1e, 0x1f] {
            for dst in [0, 17, 254] {
                for predicate in [false, true] {
                    for guard in [None, Some(false), Some(true)] {
                        let mut read = s2r(dst, index);
                        read.predicate = guard.map(|inverted| Predicate {
                            register: 0,
                            inverted,
                        });
                        let words = module(
                            vec![
                                Instruction {
                                    address: 0,
                                    predicate: None,
                                    kind: K::Mov(ops::Mov {
                                        quad_lanes: 15,
                                        dst: Register(253),
                                        src: Source::Immediate(u32::from(predicate)),
                                    }),
                                },
                                // Set P0 explicitly: Ryujinx InstEmitMove.cs:252-269.
                                Instruction {
                                    address: 0,
                                    predicate: None,
                                    kind: K::R2P(ops::R2P {
                                        src: Register(253),
                                        mask: Source::Immediate(1),
                                        byte: 0,
                                        bank: shadowbox::ir::FlagBank::Predicates,
                                    }),
                                },
                                Instruction {
                                    address: 0,
                                    predicate: None,
                                    kind: K::Mov(ops::Mov {
                                        quad_lanes: 15,
                                        dst: Register(dst),
                                        src: Source::Immediate(0x1234abcd),
                                    }),
                                },
                                read,
                            ],
                            stage,
                            state(GeometryInput::Triangles),
                        );
                        let active = guard.is_none_or(|inverted| predicate ^ inverted);
                        let expected = if active { 0 } else { 0x1234abcd };
                        assert_eq!(
                            // Stage interfaces can declare InvocationId and
                            // PatchVertices even when this read does not use them.
                            run(&words, &HashMap::from([(8, vec![0]), (14, vec![3])]))
                                [dst as usize],
                            expected,
                            "stage{stage}, SR{index:02x}, R{dst}, P0={predicate}, guard{guard:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn unhandled_registers_and_unassigned_fields_stay_unknown() {
    // Explicit Ryujinx InstEmitMove.cs:78-163 cases plus the two
    // reference-matched W-scale defaults (165-167); no other defaults.
    for index in 0..=255 {
        let supported = matches!(index, 0 | 0x11..=0x13 | 0x1d..=0x23 | 0x25..=0x27 | 0x38..=0x3c);
        if supported {
            assert_eq!(
                s2r(0, index).kind,
                K::S2R(ops::S2R {
                    dst: Register(0),
                    idx: index
                })
            );
        } else {
            assert!(
                matches!(s2r(0, index).kind, K::Unknown(ops::Unknown { .. })),
                "{index:02x}"
            );
        }
    }
    let raw = (0xf0c8u64 << 48) | (7 << 16) | (0x38 << 20);
    for bit in (8..16).chain(28..51) {
        let changed = raw | 1 << bit;
        assert!(
            matches!(
                decode_word([changed as u32, (changed >> 32) as u32], 0).kind,
                K::Unknown(ops::Unknown { .. })
            ),
            "bit{bit}"
        );
    }
    for stage in [1, 5] {
        let xs = [s2r(0, 0x20)];
        assert!(emit_unoptimized_with_graphics_state(
            &xs,
            Some(&header(stage)),
            &GraphicsState::default()
        )
        .is_err());
    }
    assert!(emit_unoptimized_with_graphics_state(
        &[s2r(0, 0x1d)],
        Some(&header(1)),
        &GraphicsState::default()
    )
    .is_err());
}

#[test]
fn optimized_masks_keep_the_required_ballot_capability() {
    // Mesa spirv.core.grammar.json:16221-16258 requires Ballot for the
    // builtins even when no OpGroupNonUniformBallot instruction is present.
    // Render target 0 writes RGBA: the target mask is bits 576..608
    // (Mesa sph.rs:185-188).
    let mut words = [0u32; 32];
    words[0] = 5 << 10;
    words[18] = 0xf;
    let h = headers::parse(&words).unwrap();
    let xs: Vec<_> = (0..5)
        .map(|i| {
            let mut instruction = s2r(i, 0x38 + i);
            instruction.address = u32::from(i) * 8;
            instruction
        })
        .chain([Instruction {
            address: 40,
            predicate: None,
            kind: K::Exit(ops::Exit { cc_test: 15 }),
        }])
        .collect();
    let words =
        shadowbox::emitter::emit_with_graphics_state(&xs, Some(&h), &GraphicsState::default())
            .unwrap()
            .words;
    let validation = spirv_validation::validate(&words);
    assert!(
        validation.status.success(),
        "{}",
        String::from_utf8_lossy(&validation.stderr)
    );
    assert!(ops(&words)
        .iter()
        .any(|(op, a)| *op == Op::Capability && *a == [64]));
    assert!(ops(&words)
        .iter()
        .any(|(op, _)| *op == Op::VectorExtractDynamic));
}
