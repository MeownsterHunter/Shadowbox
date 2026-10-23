//! Project-owned instructions. Encodings: Mesa sm50.rs:2206-2278,2435-2529;
//! envytools gm107.c:753-838,932-972,1011-1023,1909-1910,1965-1969.
//! Semantics: Ryujinx InstEmitSurface.cs:294-458,566-795 and
//! InstEmitTexture.cs:159-323,469-561. No upstream implementation copied.
use shadowbox::ir::ops;
use shadowbox::spirv::Op;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{Instruction, InstructionKind as K, Register as R, Source as S},
};
use std::{collections::HashMap, fs, process::Command};

const PT: u64 = 7 << 16;
fn decode(raw: u64) -> Instruction {
    decode_word([raw as u32, (raw >> 32) as u32], 0)
}
fn surface(store: bool, target: u64, raw: bool, size_or_mask: u64, bound: bool) -> u64 {
    (if store { 0xeb20 } else { 0xeb00 }) << 48
        | u64::from(raw) << 52
        | u64::from(bound) << 51
        | if bound { 9 << 36 } else { 20 << 39 }
        | target << 33
        | size_or_mask << 20
        | PT
        | 2 << 8
        | 8
}
fn fetch(bound: bool, dim: u64, lod: bool, offset: bool, ms: bool) -> u64 {
    (if bound { 0xdc38 } else { 0xdd38 }) << 48
        | u64::from(lod) << 55
        | u64::from(ms) << 50
        | if bound { 9 << 36 } else { 0 }
        | u64::from(offset) << 35
        | 15 << 31
        | dim << 28
        | (if bound && !lod && !offset && !ms {
            255
        } else {
            20
        }) << 20
        | PT
        | 2 << 8
        | 8
}
fn mov(reg: u8, value: u32) -> Instruction {
    Instruction {
        address: 0,
        predicate: None,
        kind: K::Mov(ops::Mov {
            quad_lanes: 15,
            dst: R(reg),
            src: S::Immediate(value),
        }),
    }
}
fn emit(mut xs: Vec<Instruction>) -> Vec<u32> {
    xs.push(Instruction {
        address: 0,
        predicate: None,
        kind: K::Exit(ops::Exit { cc_test: 15 }),
    });
    for (i, x) in xs.iter_mut().enumerate() {
        x.address = i as u32 * 8;
    }
    emit_unoptimized_with_header(&xs, Some(&headers::parse(&[0; 32]).unwrap()))
        .unwrap()
        .words
}
fn ops(words: &[u32]) -> Vec<(Op, &[u32])> {
    let mut at = 5;
    let mut result = Vec::new();
    while at < words.len() {
        let size = (words[at] >> 16) as usize;
        result.push((Op::of(words[at]), &words[at + 1..at + size]));
        at += size;
    }
    result
}
fn validate(words: &[u32]) {
    let dir = std::env::temp_dir().join(format!(
        "surface-fetch-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    fs::create_dir_all(&dir).unwrap();
    let path = dir.join("module.spv");
    fs::write(
        &path,
        words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let check = Command::new("spirv-val")
        .args([
            "--target-env",
            if words[1] >= 0x00010600 {
                "vulkan1.3"
            } else {
                "vulkan1.2"
            },
        ])
        .arg(&path)
        .output()
        .unwrap();
    assert!(
        check.status.success(),
        "{}",
        String::from_utf8_lossy(&check.stderr)
    );
    fs::remove_dir_all(dir).unwrap();
}

// Execute the emitted control flow and integer/vector operations. Image data
// comes from a supplied typed texel. This checks observable coordinates, store
// values, fetch LOD/sample and register results; it does not emulate a GPU.
#[derive(Debug)]
struct AtomicObservation {
    opcode: Op,
    coordinates: Vec<u32>,
    operands: Vec<u32>,
    old: u32,
    value: u32,
}
#[derive(Default, Debug)]
struct Observation {
    writes: Vec<(Vec<u32>, Vec<u32>)>,
    fetches: Vec<(Vec<u32>, u32, Vec<u32>)>,
    registers: Vec<u32>,
    atomics: Vec<AtomicObservation>,
    queries: Vec<(Op, Vec<u32>)>,
}
fn observe(words: &[u32], texel: [u32; 4]) -> Observation {
    let instructions = ops(words);
    let labels: HashMap<_, _> = instructions
        .iter()
        .enumerate()
        .filter(|(_, (op, _))| *op == Op::Label)
        .map(|(i, (_, a))| (a[0], i))
        .collect();
    let mut values: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut memory: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut variables = Vec::new();
    let mut out = Observation::default();
    let mut at = 0;
    let mut cell = texel[0];
    for _ in 0..100_000 {
        let (op, a) = instructions[at];
        let scalar = |i: usize| values[&a[i]][0];
        let value = match op {
            Op::ConstantTrue => Some(vec![1]),
            Op::ConstantFalse => Some(vec![0]),
            Op::Constant => Some(vec![a[2]]),
            Op::ConstantComposite | Op::CompositeConstruct => {
                Some(a[2..].iter().flat_map(|id| values[id].clone()).collect())
            }
            Op::Variable => {
                if a[2] == 7 {
                    variables.push(a[1]);
                }
                memory.insert(a[1], vec![0]);
                None
            }
            Op::ImageTexelPointer => Some(values[&a[3]].clone()),
            Op::ExtInst if a[3] == 58 => {
                // Independent exact half values used by the packing cases.
                let half = |bits: u32| match bits {
                    0 => 0,
                    0x3f80_0000 => 0x3c00,
                    0xc000_0000 => 0xc000,
                    0x3f00_0000 => 0x3800,
                    0x4080_0000 => 0x4400,
                    _ => panic!("unexpected half input {bits:08x}"),
                };
                let pair = &values[&a[4]];
                Some(vec![half(pair[0]) | half(pair[1]) << 16])
            }
            Op::SampledImage => Some(vec![0]),
            Op::ImageQuerySizeLod | Op::ImageQuerySize => {
                out.queries
                    .push((op, a[3..].iter().map(|id| values[id][0]).collect()));
                let count = instructions
                    .iter()
                    .find(|(op, args)| *op == Op::TypeVector && args[0] == a[0])
                    .unwrap()
                    .1[2];
                Some(texel[..count as usize].to_vec())
            }
            Op::ImageQueryLod => {
                out.queries.push((op, values[&a[3]].clone()));
                Some(texel[..2].to_vec())
            }
            Op::ImageQueryLevels => Some(vec![texel[3]]),
            Op::ImageQuerySamples => {
                out.queries.push((op, vec![]));
                Some(vec![texel[2]])
            }
            Op::ConvertFToS => Some(vec![(f32::from_bits(scalar(2)) as i32) as u32]),
            Op::FMul => Some(vec![(f32::from_bits(scalar(2))
                * f32::from_bits(scalar(3)))
            .to_bits()]),
            Op::AtomicExchange
            | Op::AtomicCompareExchange
            | Op::AtomicCompareExchangeWeak
            | Op::AtomicIIncrement
            | Op::AtomicIDecrement
            | Op::AtomicIAdd
            | Op::AtomicISub
            | Op::AtomicSMin
            | Op::AtomicUMin
            | Op::AtomicSMax
            | Op::AtomicUMax
            | Op::AtomicAnd
            | Op::AtomicOr
            | Op::AtomicXor => {
                let old = cell;
                let operands: Vec<_> = a[5..].iter().map(|id| values[id][0]).collect();
                let value = operands.first().copied().unwrap_or(0);
                cell = match op {
                    Op::AtomicExchange => value,
                    Op::AtomicCompareExchange => {
                        if old == operands[2] {
                            operands[1]
                        } else {
                            old
                        }
                    }
                    Op::AtomicIIncrement => old.wrapping_add(1),
                    Op::AtomicIDecrement => old.wrapping_sub(1),
                    Op::AtomicIAdd => old.wrapping_add(value),
                    Op::AtomicSMin => (old as i32).min(value as i32) as u32,
                    Op::AtomicUMin => old.min(value),
                    Op::AtomicSMax => (old as i32).max(value as i32) as u32,
                    Op::AtomicUMax => old.max(value),
                    Op::AtomicAnd => old & value,
                    Op::AtomicOr => old | value,
                    Op::AtomicXor => old ^ value,
                    _ => panic!("unexpected atomic {}", op.number()),
                };
                out.atomics.push(AtomicObservation {
                    opcode: op,
                    coordinates: values[&a[2]].clone(),
                    operands,
                    old,
                    value: cell,
                });
                assert_eq!(scalar(3), 1); // Device scope.
                assert_eq!(scalar(4), 0); // Relaxed semantics.
                Some(vec![old])
            }
            Op::Load => Some(memory[&a[2]].clone()),
            Op::Store => {
                memory.insert(a[0], values[&a[1]].clone());
                None
            }
            Op::CompositeExtract => Some(vec![values[&a[2]][a[3] as usize]]),
            Op::Bitcast => Some(values[&a[2]].clone()),
            Op::IAdd => Some(vec![scalar(2).wrapping_add(scalar(3))]),
            Op::LogicalNot => Some(vec![u32::from(scalar(2) == 0)]),
            Op::Select => Some(values[&a[if scalar(2) != 0 { 3 } else { 4 }]].clone()),
            Op::INotEqual => Some(vec![u32::from(scalar(2) != scalar(3))]),
            Op::ShiftRightArithmetic => Some(vec![((scalar(2) as i32) >> scalar(3)) as u32]),
            Op::BitFieldSExtract => {
                let shift = 32 - scalar(4);
                Some(vec![
                    (((scalar(2) >> scalar(3)) << shift) as i32 >> shift) as u32,
                ])
            }
            Op::ImageSampleImplicitLod
            | Op::ImageSampleExplicitLod
            | Op::ImageGather
            | Op::ImageDrefGather => Some(texel.to_vec()),
            Op::ImageFetch => {
                out.fetches.push((
                    values[&a[3]].clone(),
                    a[4],
                    a[5..].iter().map(|id| values[id][0]).collect(),
                ));
                Some(texel.to_vec())
            }
            Op::ImageRead => Some(texel.to_vec()),
            Op::ImageWrite => {
                out.writes
                    .push((values[&a[1]].clone(), values[&a[2]].clone()));
                None
            }
            Op::Branch => {
                at = labels[&a[0]];
                continue;
            }
            Op::BranchConditional => {
                at = labels[&a[if scalar(0) != 0 { 1 } else { 2 }]];
                continue;
            }
            Op::Switch => {
                let target = a[2..]
                    .chunks(2)
                    .find(|c| c[0] == scalar(0))
                    .map_or(a[1], |c| c[1]);
                at = labels[&target];
                continue;
            }
            Op::Return => {
                // Optimized shaders may eliminate register locals entirely.
                out.registers = variables.iter().take(256).map(|v| memory[v][0]).collect();
                return out;
            }
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
            | Op::TypeMatrix
            | Op::TypeImage
            | Op::TypeSampler
            | Op::TypeSampledImage
            | Op::TypeArray
            | Op::TypeRuntimeArray
            | Op::TypeStruct
            | Op::TypeOpaque
            | Op::TypePointer
            | Op::TypeFunction
            | Op::Function
            | Op::FunctionEnd
            | Op::Decorate
            | Op::MemberDecorate
            | Op::LoopMerge
            | Op::SelectionMerge
            | Op::Label => None,
            other => panic!("unexpected executable SPIR-V opcode {}", other.number()),
        };
        if let Some(value) = value {
            values.insert(a[1], value);
        }
        at += 1;
    }
    panic!("emitted control flow did not terminate")
}

#[test]
fn all_surface_dimensions_raw_sizes_masks_and_handles_validate() {
    for target in 0..=5 {
        for bound in [false, true] {
            for store in [false, true] {
                for (raw, field) in (1..=15)
                    .map(|m| (false, m))
                    .chain((0..=6).map(|s| (true, s)))
                {
                    let raw_word = surface(store, target, raw, field, bound);
                    let decoded = decode(raw_word);
                    assert!(
                        matches!(decoded.kind, K::Surface(ops::Surface { target: t, .. }) if u64::from(t) == target)
                    );
                    for modifiers in [0, 1 << 24, if raw { 1 << 23 } else { 0 }] {
                        let words = emit(vec![mov(20, 0), decode(raw_word | modifiers)]);
                        validate(&words);
                        assert_eq!(
                            ops(&words)
                                .iter()
                                .filter(|(op, _)| *op
                                    == if store { Op::ImageWrite } else { Op::ImageRead })
                                .count(),
                            1
                        );
                        assert!(!store || !ops(&words).iter().any(|(op, _)| *op == Op::ImageRead));
                    }
                }
            }
        }
    }
}

#[test]
fn streaming_surface_stores_preserve_every_target_format_mask_and_byte_address() {
    // CS24..25: envytools gm107.c:825-831; Mesa sm50.rs:2424-2431.
    // Nontemporal0x4000: Mesa spirv.core.grammar.json:11990-11994.
    for target in 0..=5 {
        for bound in [false, true] {
            for (raw, field) in (1..=15)
                .map(|mask| (false, mask))
                .chain((0..=6).map(|size| (true, size)))
            {
                for byte_address in [false, true] {
                    if byte_address && !raw {
                        continue;
                    }
                    let word = surface(true, target, raw, field, bound)
                        | 2 << 24
                        | u64::from(byte_address) << 23;
                    assert!(matches!(
                        decode(word).kind,
                        K::Surface(ops::Surface {
                            store: true,
                            cache: 2,
                            ..
                        })
                    ));
                    let mut xs = vec![mov(20, 0), mov(2, (-33i32) as u32), mov(3, 7), mov(4, 11)];
                    xs.extend((8..12).map(|r| mov(r, u32::from(r) * 0x1234)));
                    xs.push(decode(word));
                    let words = emit(xs.clone());
                    validate(&words);
                    assert_eq!(words[1], 0x00010600);
                    let write = ops(&words)
                        .into_iter()
                        .find(|(op, _)| *op == Op::ImageWrite)
                        .unwrap()
                        .1;
                    assert_eq!(write.len(), 4);
                    assert_eq!(write[3], 0x4000);
                    let observed = observe(&words, [0; 4]);
                    let shift = if byte_address {
                        [0, 0, 1, 1, 2, 3, 4, 4][field as usize]
                    } else {
                        0
                    };
                    let dimensions = [1, 1, 2, 2, 3, 3][target as usize];
                    let components = if raw {
                        [1, 1, 1, 1, 1, 2, 4, 4][field as usize]
                    } else {
                        field.count_ones() as usize
                    };
                    let mut expected = vec![0; 4];
                    for (i, value) in expected.iter_mut().take(components).enumerate() {
                        *value = (8 + i as u32) * 0x1234;
                    }
                    assert_eq!(
                        observed.writes,
                        vec![(
                            vec![((-33i32) >> shift) as u32, 7, 11][..dimensions].to_vec(),
                            expected
                        )]
                    );
                    // Optimization must retain the hint, store, and interface.
                    xs.push(Instruction {
                        address: 0,
                        predicate: None,
                        kind: K::Exit(ops::Exit { cc_test: 15 }),
                    });
                    for (i, x) in xs.iter_mut().enumerate() {
                        x.address = i as u32 * 8;
                    }
                    let optimized = shadowbox::emitter::emit_with_header(
                        &xs,
                        Some(&headers::parse(&[0; 32]).unwrap()),
                    )
                    .unwrap()
                    .words;
                    validate(&optimized);
                    assert_eq!(optimized[1], 0x00010600);
                    let write = ops(&optimized)
                        .into_iter()
                        .find(|(op, _)| *op == Op::ImageWrite)
                        .unwrap()
                        .1;
                    assert_eq!(write[3], 0x4000);
                    assert!(
                        optimized.len() < words.len(),
                        "CS must not disable optimization"
                    );
                    assert_eq!(observe(&optimized, [0; 4]).writes, observed.writes);
                }
            }
        }
    }
    // A skipped CS store still upgrades the module but performs no write.
    for guard in [0, PT | 1 << 19] {
        let word = (surface(true, 3, false, 15, true) & !(15 << 16)) | guard | 2 << 24;
        let words = emit(vec![decode(word)]);
        validate(&words);
        assert!(observe(&words, [0; 4]).writes.is_empty());
    }
}

#[test]
fn surface_store_values_byte_coordinates_load_channels_and_predicates_are_exact() {
    let prefix = || {
        vec![
            mov(2, (-33i32) as u32),
            mov(3, 7),
            mov(4, 11),
            mov(8, 0x1234),
            mov(9, 0x5678),
            mov(10, 0x9abc),
            mov(11, 0xdef0),
        ]
    };
    for target in 0..=5 {
        for size in 0..=6 {
            let mut xs = prefix();
            xs.push(decode(surface(true, target, true, size, true) | 1 << 23));
            let result = observe(&emit(xs), [0; 4]);
            let shift = [0, 0, 1, 1, 2, 3, 4][size as usize];
            let count = [1, 1, 2, 2, 3, 3][target as usize];
            assert_eq!(
                result.writes[0].0,
                [((-33i32) >> shift) as u32, 7, 11][..count]
            );
            let components = [1, 1, 1, 1, 1, 2, 4][size as usize];
            let mut expected = vec![0; 4];
            expected[..components].copy_from_slice(&[0x1234, 0x5678, 0x9abc, 0xdef0][..components]);
            assert_eq!(result.writes[0].1, expected);
        }
    }
    for mask in 1..=15 {
        let mut xs = prefix();
        xs.push(decode(surface(true, 3, false, mask, true)));
        let result = observe(&emit(xs), [0; 4]);
        let count = mask.count_ones() as usize;
        let mut expected = vec![0; 4];
        expected[..count].copy_from_slice(&[0x1234, 0x5678, 0x9abc, 0xdef0][..count]);
        assert_eq!(result.writes[0].1, expected);
        let result = observe(
            &emit(vec![decode(surface(false, 3, false, mask, true))]),
            [10, 20, 30, 40],
        );
        let expected = (0..4)
            .filter(|i| mask & (1 << i) != 0)
            .map(|i| [10, 20, 30, 40][i])
            .collect::<Vec<_>>();
        assert_eq!(result.registers[8..8 + count], expected);
    }
    // P0 is false initially; inverted PT is false too. Neither may store.
    for guard in [0, PT | 1 << 19] {
        let words = emit(vec![decode(
            (surface(true, 3, false, 15, true) & !(15 << 16)) | guard,
        )]);
        validate(&words);
        assert!(observe(&words, [0; 4]).writes.is_empty());
    }
}

#[test]
fn bound_and_bindless_fetch_lod_offsets_ms_and_array_operands_are_exact() {
    for bound in [false, true] {
        for dim in 0..=4 {
            for lod in [false, true] {
                for offset in [false, true] {
                    for ms in [false, true] {
                        if ms && !matches!(dim, 2 | 3) {
                            continue;
                        }
                        let next = if bound { 20 } else { 21 };
                        let mut xs = vec![
                            mov(2, 7),
                            mov(3, 12),
                            mov(4, 19),
                            mov(20, 0),
                            mov(
                                next,
                                if lod {
                                    3
                                } else if offset {
                                    0x8f7
                                } else {
                                    5
                                },
                            ),
                        ];
                        if lod && offset {
                            xs.push(mov(next + 1, 0x8f7));
                        }
                        if ms {
                            xs.push(mov(next + u8::from(lod) + u8::from(offset), 5));
                        }
                        xs.push(decode(fetch(bound, dim, lod, offset, ms)));
                        let words = emit(xs);
                        validate(&words);
                        let result = observe(&words, [1, 2, 3, 4]);
                        let array = matches!(dim, 1 | 3);
                        let count = if dim <= 1 {
                            1
                        } else if dim <= 3 {
                            2
                        } else {
                            3
                        };
                        let coords = [7u32, 12, 19];
                        let mut expected =
                            coords[usize::from(array)..usize::from(array) + count].to_vec();
                        if offset {
                            for (v, off) in expected.iter_mut().zip([7i32, -1, -8]) {
                                *v = v.wrapping_add(off as u32);
                            }
                        }
                        if array {
                            expected.push(7);
                        }
                        let (position, mask, operands) = &result.fetches[0];
                        assert_eq!(
                            *position, expected,
                            "bound={bound} dim={dim} lod={lod} offset={offset} ms={ms}"
                        );
                        assert_eq!(*mask, if ms { 0x40 } else { 2 });
                        assert_eq!(
                            *operands,
                            vec![if ms {
                                5
                            } else if lod {
                                3
                            } else {
                                0
                            }]
                        );
                        assert_eq!(result.registers[8..12], [1, 2, 3, 4]);
                    }
                }
            }
        }
    }
}

#[test]
fn unsupported_surface_modes_and_atomics_stay_unknown() {
    let plain = surface(true, 3, false, 15, true);
    for raw in [
        plain | 1 << 49,
        plain | 2 << 49,
        plain | 3 << 49,
        surface(false, 3, false, 15, true) | 2 << 24,
        plain | 3 << 24,
        surface(false, 6, false, 15, true),
        surface(false, 7, false, 15, true),
        surface(true, 3, false, 0, true),
        fetch(true, 2, false, false, false) | 1 << 54,
        (fetch(true, 2, false, false, false) & !(255 << 20)) | 20 << 20,
        fetch(false, 6, false, false, false),
        fetch(false, 0, false, false, true),
    ] {
        assert!(
            matches!(decode(raw).kind, K::Unknown(ops::Unknown { .. })),
            "{raw:016x}"
        );
    }
    // Unsupported U64 surface atomics, with the correct size field for
    // each family (envytools gm107.c:1839-1857,1908,1911-1914).
    for (opcode, size_bit) in [
        (0xeb50, 20),
        (0xead0, 36),
        (0xea80, 51),
        (0xea70, 36),
        (0xea00, 51),
    ] {
        assert!(matches!(
            decode(opcode << 48 | 2 << size_bit | PT).kind,
            K::Unknown(ops::Unknown { .. })
        ));
    }
    assert_eq!(
        include_str!("../src/decoder/families.rs")
            .matches("x.guarded(InstructionKind::Nop(ops::Nop))")
            .count(),
        2
    );
}

#[test]
fn scalar_fetch_modes_use_the_correct_register_pairs() {
    // envytools gm107.c:932-972,1967; Ryujinx InstEmitTexture.cs:490-561.
    for (mode, coords, operand_mask, operand) in [
        (0, vec![7], 2, 0),
        (1, vec![7], 2, 19),
        (2, vec![7, 19], 2, 0),
        (4, vec![14, 11], 2, 0),
        (5, vec![7, 12], 2, 19),
        (6, vec![7, 12], 0x40, 19),
        (7, vec![7, 12, 19], 2, 0),
        (8, vec![19, 3, 7], 2, 0),
        (12, vec![14, 11], 2, 19),
    ] {
        let src_b = if mode == 0 { 255 } else { 20 };
        let raw = 0xda00 << 48
            | mode << 53
            | 4 << 50
            | 9 << 36
            | 40 << 28
            | src_b << 20
            | PT
            | 2 << 8
            | 8;
        let words = emit(vec![
            mov(2, 7),
            mov(3, 12),
            mov(20, if mode == 4 { 0x8f7 } else { 19 }),
            mov(21, if mode == 12 { 0x8f7 } else { 3 }),
            decode(raw),
        ]);
        validate(&words);
        let result = observe(&words, [1, 2, 3, 4]);
        assert_eq!(
            result.fetches,
            vec![(coords, operand_mask, vec![operand])],
            "mode={mode}"
        );
        assert_eq!(result.registers[8..10], [1, 2]);
        assert_eq!(result.registers[40..42], [3, 4]);
    }
}

#[test]
fn dynamic_fetch_offsets_add_signed_values_without_changing_array_layer_or_sample() {
    // An IADD-defined packed offset cannot be folded by handle tracking.
    // IADD encoding: Mesa sm50.rs:1291-1343; offset interpretation:
    // Ryujinx InstEmitTexture.cs:272-276. It must still translate.
    let offset = decode(0x3810 << 48 | 0x8f7 << 20 | PT | 255 << 8 | 20);
    let words = emit(vec![
        mov(2, 7),
        mov(3, 12),
        mov(4, 19),
        offset,
        mov(21, 5),
        decode(fetch(true, 3, false, true, true)),
    ]);
    validate(&words);
    assert_eq!(
        ops(&words)
            .iter()
            .filter(|(op, _)| *op == Op::BitFieldSExtract)
            .count(),
        2
    );
    assert_eq!(
        observe(&words, [0; 4]).fetches,
        vec![(vec![19, 18, 7], 0x40, vec![5])]
    );
}

#[test]
fn surface_and_fetch_descriptor_identity_and_raw_component_types_are_exact() {
    let formats = [39, 29, 38, 28, 33, 35, 30];
    for (size, format) in formats.into_iter().enumerate() {
        let words = emit(vec![decode(surface(false, 3, true, size as u64, true))]);
        let declarations = ops(&words);
        let image = declarations
            .iter()
            .find(|(op, a)| *op == Op::TypeImage && a[6] == 2)
            .unwrap()
            .1;
        assert_eq!(image[7], format);
        let scalar_type = declarations
            .iter()
            .find(|(op, a)| *op == Op::TypeInt && a[0] == image[1])
            .unwrap()
            .1;
        assert_eq!(scalar_type[1..], [32, u32::from(matches!(size, 1 | 3))]);
        assert!(declarations
            .iter()
            .any(|(op, a)| *op == Op::Decorate && a[1..] == [34, 2]));
        assert!(declarations
            .iter()
            .any(|(op, a)| *op == Op::Decorate && a[1..] == [33, 9]));
    }
    for bound in [false, true] {
        let words = emit(vec![mov(20, 0), decode(surface(true, 3, false, 15, bound))]);
        assert!(ops(&words)
            .iter()
            .any(|(op, a)| *op == Op::Decorate && a[1..] == [34, if bound { 2 } else { 0 }]));
        assert!(ops(&words)
            .iter()
            .any(|(op, a)| *op == Op::Decorate && a[1..] == [33, if bound { 9 } else { 2048 }]));
        let words = emit(vec![
            mov(20, 0),
            decode(fetch(bound, 2, false, false, false)),
        ]);
        assert!(ops(&words)
            .iter()
            .any(|(op, a)| *op == Op::Decorate && a[1..] == [34, u32::from(bound)]));
        assert!(ops(&words)
            .iter()
            .any(|(op, a)| *op == Op::Decorate && a[1..] == [33, if bound { 18 } else { 0 }]));
    }
    let cg = emit(vec![decode(surface(true, 3, false, 15, true) | 1 << 24)]);
    assert!(ops(&cg)
        .iter()
        .any(|(op, a)| *op == Op::Decorate && a[1] == 23));
}

#[test]
fn bound_gather_components_offsets_arrays_and_comparison_validate() {
    // Mesa sm50.rs:2245-2278; envytools gm107.c:1011-1023,1969;
    // Ryujinx InstEmitTexture.cs:738-776.
    for dim in [2, 3, 6, 7] {
        for component in 0..=3 {
            for offset in [false, true] {
                if offset && matches!(dim, 6 | 7) {
                    continue;
                }
                for compare in [false, true] {
                    if compare && component != 0 {
                        continue;
                    }
                    let raw = 0xc838 << 48
                        | component << 56
                        | u64::from(offset) << 54
                        | u64::from(compare) << 50
                        | 9 << 36
                        | 15 << 31
                        | dim << 28
                        | (if !offset && !compare { 255 } else { 20 }) << 20
                        | PT
                        | 2 << 8
                        | 8;
                    assert!(
                        matches!(decode(raw).kind, K::Texture(ops::Texture { bound_handle: Some(9), op: shadowbox::ir::TexOp::Gather { component: c, .. }, .. }) if u64::from(c) == component)
                    );
                    let words = emit(vec![mov(20, 0x3f01), decode(raw)]);
                    validate(&words);
                    let gather = ops(&words)
                        .into_iter()
                        .find(|(op, _)| matches!(op, Op::ImageGather | Op::ImageDrefGather))
                        .unwrap();
                    assert_eq!(
                        gather.0,
                        if compare {
                            Op::ImageDrefGather
                        } else {
                            Op::ImageGather
                        }
                    );
                    if offset {
                        assert_eq!(gather.1[5], 8);
                    }
                }
            }
        }
    }
}

// Atomic encodings: Mesa sm50.rs:2558-2591; envytools gm107.c:769-791,
// 1833-1871,1908,1911-1914. All words here are project-owned.
fn surface_atom(reduction: bool, bound: bool, signed: bool, target: u64, op: u64) -> u64 {
    let opcode = if reduction {
        0xeb50
    } else if op == 9 {
        if bound {
            0xea80
        } else {
            0xead0
        }
    } else if bound {
        0xea00
    } else {
        0xea70
    };
    let size_bit = if reduction {
        20
    } else if bound {
        51
    } else {
        36
    };
    let op_bits = if reduction {
        op << 24
    } else if op == 9 {
        7 << 30
    } else {
        op << 29
    };
    let data = if matches!(op, 3 | 4) { 255 } else { 20 };
    opcode << 48
        | u64::from(signed) << size_bit
        | target << 33
        | op_bits
        | if bound {
            9 << 36 | if reduction { 1 << 51 } else { 0 }
        } else {
            40 << 39
        }
        | if reduction { data } else { data << 20 | 8 }
        | PT
        | 2 << 8
}

fn surface_atom_pt_result(signed: bool, target: u64, op: u64, byte_address: bool) -> u64 {
    let data = if matches!(op, 3 | 4) { 255 } else { 20 };
    0xeb80_u64 << 48
        | u64::from(signed) << 36
        | target << 33
        | op << 29
        | u64::from(byte_address) << 28
        | 40 << 39
        | data << 20
        | PT
        | 2 << 8
        | 8
}

#[test]
fn integer_surface_atomics_have_exact_values_coordinates_and_resource_identity() {
    // Expected opcodes: Mesa src/compiler/spirv/spirv.h:1946-1959.
    for reduction in [false, true] {
        for bound in [false, true] {
            for signed in [false, true] {
                for target in 0..=5 {
                    for op in 0..=if reduction { 7 } else { 9 } {
                        let raw = surface_atom(reduction, bound, signed, target, op);
                        assert!(
                            matches!(
                                decode(raw).kind,
                                K::SurfaceAtomic(ops::SurfaceAtomic { .. })
                            ),
                            "{raw:016x}"
                        );
                        let words = emit(vec![
                            mov(40, 0),
                            mov(2, 28),
                            mov(3, 7),
                            mov(4, 11),
                            mov(20, 5),
                            mov(21, 13),
                            decode(raw),
                        ]);
                        validate(&words);
                        let old = if signed { (-9i32) as u32 } else { 9 };
                        let result = observe(&words, [old, 0, 0, 0]);
                        let expected = match op {
                            0 => old.wrapping_add(5),
                            1 if signed => (-9i32) as u32,
                            1 => 5,
                            2 if signed => 5,
                            2 => 9,
                            3 => old.wrapping_add(1),
                            4 => old.wrapping_sub(1),
                            5 => old & 5,
                            6 => old | 5,
                            7 => old ^ 5,
                            8 => 5,
                            9 => old,
                            _ => unreachable!(),
                        };
                        assert_eq!(result.atomics.len(), 1);
                        let atomic = &result.atomics[0];
                        let expected_opcode = match op {
                            0 => Op::AtomicIAdd,
                            1 => {
                                if signed {
                                    Op::AtomicSMin
                                } else {
                                    Op::AtomicUMin
                                }
                            }
                            2 => {
                                if signed {
                                    Op::AtomicSMax
                                } else {
                                    Op::AtomicUMax
                                }
                            }
                            3 => Op::AtomicIIncrement,
                            4 => Op::AtomicIDecrement,
                            5 => Op::AtomicAnd,
                            6 => Op::AtomicOr,
                            7 => Op::AtomicXor,
                            8 => Op::AtomicExchange,
                            9 => Op::AtomicCompareExchange,
                            _ => unreachable!(),
                        };
                        assert_eq!(atomic.opcode, expected_opcode);
                        let count = [1, 1, 2, 2, 3, 3][target as usize];
                        assert_eq!(atomic.coordinates, [28, 7, 11][..count]);
                        assert_eq!(atomic.old, old);
                        assert_eq!(atomic.value, expected);
                        if !reduction {
                            assert_eq!(result.registers[8], old);
                        }
                        let image_var = ops(&words)
                            .iter()
                            .find(|(op, _)| *op == Op::ImageTexelPointer)
                            .unwrap()
                            .1[2];
                        let decorations: Vec<_> = ops(&words)
                            .into_iter()
                            .filter(|(op, a)| *op == Op::Decorate && a[0] == image_var)
                            .map(|(_, a)| a[1..].to_vec())
                            .collect();
                        assert!(decorations.contains(&vec![34, if bound { 2 } else { 0 }]));
                        assert!(decorations.contains(&vec![33, if bound { 9 } else { 2048 }]));
                    }
                }
            }
        }
    }
}

#[test]
fn pt_result_surface_atomics_have_exact_semantics_and_boundaries() {
    use shadowbox::emitter::emit_with_header;

    let header = headers::parse(&[0; 32]).unwrap();
    for signed in [false, true] {
        for target in 0..=5 {
            for op in 0..=8 {
                for byte_address in [false, true] {
                    let raw = surface_atom_pt_result(signed, target, op, byte_address);
                    let instruction = decode(raw);
                    assert!(matches!(
                        instruction.kind,
                        K::SurfaceAtomic(ops::SurfaceAtomic {
                            dst: Some(R(8)),
                            handle: S::Register(R(40)),
                            ..
                        })
                    ));
                    let mut xs = vec![
                        mov(40, 0),
                        mov(2, if byte_address { (-33_i32) as u32 } else { 28 }),
                        mov(3, 7),
                        mov(4, 11),
                        mov(20, 5),
                        instruction,
                        Instruction {
                            address: 0,
                            predicate: None,
                            kind: K::Exit(ops::Exit { cc_test: 15 }),
                        },
                    ];
                    for (index, instruction) in xs.iter_mut().enumerate() {
                        instruction.address = index as u32 * 8;
                    }

                    let raw_module = emit_unoptimized_with_header(&xs, Some(&header)).unwrap();
                    validate(&raw_module.words);
                    let old = if signed { (-9_i32) as u32 } else { 9 };
                    let observed = observe(&raw_module.words, [old, 0, 0, 0]);
                    let expected = match op {
                        0 => old.wrapping_add(5),
                        1 if signed => (-9_i32) as u32,
                        1 => 5,
                        2 if signed => 5,
                        2 => 9,
                        3 => old.wrapping_add(1),
                        4 => old.wrapping_sub(1),
                        5 => old & 5,
                        6 => old | 5,
                        7 => old ^ 5,
                        8 => 5,
                        _ => unreachable!(),
                    };
                    let atomic = observed.atomics.first().unwrap();
                    assert_eq!(observed.atomics.len(), 1);
                    assert_eq!(atomic.old, old);
                    assert_eq!(atomic.value, expected);
                    let count = [1, 1, 2, 2, 3, 3][target as usize];
                    let x = if byte_address { (-9_i32) as u32 } else { 28 };
                    assert_eq!(atomic.coordinates, [x, 7, 11][..count]);

                    let optimized = emit_with_header(&xs, Some(&header)).unwrap();
                    validate(&optimized.words);
                    assert!(ops(&optimized.words).iter().any(|(opcode, _)| matches!(
                        opcode,
                        Op::AtomicExchange
                            | Op::AtomicIIncrement
                            | Op::AtomicIDecrement
                            | Op::AtomicIAdd
                            | Op::AtomicSMin
                            | Op::AtomicUMin
                            | Op::AtomicSMax
                            | Op::AtomicUMax
                            | Op::AtomicAnd
                            | Op::AtomicOr
                            | Op::AtomicXor
                    )));
                }
            }
        }
    }

    for raw in [
        0xeb80_0000_0007_0000,
        0xeb80_7f80_0ff7_ff00,
        0xeb80_7f80_0007_ff00,
        0xeb80_7f81_0007_ff00,
        0xeb80_0000_0000_0000,
    ] {
        assert!(
            matches!(decode(raw).kind, K::SurfaceAtomic(_)),
            "{raw:016x}"
        );
    }
    let base = surface_atom_pt_result(false, 0, 0, false);
    for raw in (1..=7)
        .map(|value| base | value << 51)
        .chain((1..=3).map(|value| base | value << 49))
        .chain((2..=7).map(|value| base | value << 36))
        .chain((6..=7).map(|value| base | value << 33))
        .chain((9..=15).map(|value| base | value << 29))
        .chain([base | 1 << 47, base | 1 << 48, 0xebc0_u64 << 48])
        .chain([surface_atom_pt_result(false, 0, 3, false) & !(255 << 20) | 20 << 20])
        .chain([surface_atom_pt_result(false, 0, 4, false) & !(255 << 20) | 20 << 20])
    {
        assert!(matches!(decode(raw).kind, K::Unknown(_)), "{raw:016x}");
    }
}

#[test]
fn image_cas_byte_address_predication_and_unknown_boundaries() {
    for bound in [false, true] {
        for reduction in [false, true] {
            let op = if reduction { 0 } else { 9 };
            let raw =
                surface_atom(reduction, bound, false, 4, op) | 1 << if reduction { 23 } else { 28 };
            let mut xs = vec![
                mov(40, 0),
                mov(2, (-33i32) as u32),
                mov(3, 7),
                mov(4, 11),
                mov(20, 5),
                mov(21, 13),
                decode(raw),
            ];
            let words = emit(xs.clone());
            validate(&words);
            let result = observe(&words, [5, 0, 0, 0]);
            assert_eq!(result.atomics[0].coordinates, [(-9i32) as u32, 7, 11]);
            assert_eq!(result.atomics[0].value, if reduction { 10 } else { 13 });
            if !reduction {
                assert_eq!(result.atomics[0].operands, [0, 13, 5]);
            }
            xs.last_mut().unwrap().predicate = Some(shadowbox::ir::Predicate {
                register: 0,
                inverted: false,
            });
            let words = emit(xs);
            validate(&words);
            assert!(observe(&words, [5, 0, 0, 0]).atomics.is_empty());
        }
    }
    for bound in [false, true] {
        let raw = surface_atom(false, bound, false, 3, 0);
        let size_bit = if bound { 51 } else { 36 };
        for size in 2..=7 {
            assert!(matches!(
                decode(raw | size << size_bit).kind,
                K::Unknown(ops::Unknown { .. })
            ));
        }
        for clamp in 1..=3 {
            assert!(matches!(
                decode(raw | clamp << 49).kind,
                K::Unknown(ops::Unknown { .. })
            ));
        }
        assert!(matches!(
            decode(surface_atom(false, bound, false, 3, 3) & !(255 << 20)).kind,
            K::Unknown(ops::Unknown { .. })
        ));
        assert!(matches!(
            decode(surface_atom(false, bound, false, 3, 9) & !(7 << 30)).kind,
            K::Unknown(ops::Unknown { .. })
        ));
    }
    let bindless = surface_atom(false, false, false, 3, 0);
    assert!(matches!(
        decode(bindless & !(1 << 52)).kind,
        K::Unknown(ops::Unknown { .. })
    ));
    assert!(matches!(
        decode(bindless | 1 << 51).kind,
        K::Unknown(ops::Unknown { .. })
    ));
}

#[test]
fn scalar_half_results_pack_selected_channels_and_only_write_the_encoded_registers() {
    // F16 selector59=0 for TEXS/TLDS, selector55=1 for gather:
    // envytools gm107.c:1960,1967-1968. Masks at 958-972.
    for opcode in [0xd000, 0xd200, 0xdf80] {
        for dst2 in [12u64, 255] {
            for selector in 0..=if dst2 == 255 { 7 } else { 4 } {
                let raw = opcode << 48
                    | if opcode == 0xdf80 {
                        0
                    } else {
                        2 << 53 | selector << 50
                    }
                    | 9 << 36
                    | dst2 << 28
                    | 20 << 20
                    | PT
                    | 2 << 8
                    | 8;
                let decoded = decode(raw);
                let K::Texture(ops::Texture {
                    mask,
                    packed_f16: true,
                    ..
                }) = decoded.kind
                else {
                    panic!("{raw:016x}")
                };
                assert!(decoded.kind.writes_register(8));
                assert!(!decoded.kind.writes_register(9));
                assert_eq!(decoded.kind.writes_register(12), dst2 == 12);
                assert!(!decoded.kind.writes_register(13));
                let words = emit(vec![
                    mov(2, 3),
                    mov(20, 7),
                    mov(9, 0xdead),
                    mov(13, 0xbeef),
                    decoded,
                ]);
                validate(&words);
                let result = observe(
                    &words,
                    [
                        1.0f32.to_bits(),
                        (-2.0f32).to_bits(),
                        0.5f32.to_bits(),
                        4.0f32.to_bits(),
                    ],
                );
                let mut expected = [0; 4];
                let mut next = 0;
                for (channel, half) in [0x3c00u32, 0xc000, 0x3800, 0x4400].into_iter().enumerate() {
                    if mask & (1 << channel) != 0 {
                        expected[next] = half;
                        next += 1;
                    }
                }
                assert_eq!(result.registers[8], expected[0] | expected[1] << 16);
                if dst2 == 12 {
                    assert_eq!(result.registers[12], expected[2] | expected[3] << 16);
                }
                assert_eq!(result.registers[9], 0xdead);
                assert_eq!(result.registers[13], 0xbeef);
            }
        }
    }
}

fn emit_fragment(mut xs: Vec<Instruction>) -> Vec<u32> {
    xs.push(Instruction {
        address: 0,
        predicate: None,
        kind: K::Exit(ops::Exit { cc_test: 15 }),
    });
    for (i, x) in xs.iter_mut().enumerate() {
        x.address = i as u32 * 8;
    }
    let mut header = [0; 32];
    header[0] = 5 << 10; // Mesa sph.rs:204-215 fragment stage.
    emit_unoptimized_with_header(&xs, Some(&headers::parse(&header).unwrap()))
        .unwrap()
        .words
}

#[test]
fn txq_dimensions_levels_and_samples_are_exact_and_unknown_kinds_remain_unknown() {
    // Mesa sm50.rs:2345-2381; Ryujinx InstEmitTexture.cs:1058-1187.
    for mask in 1..=15u64 {
        let raw = 0xdf50 << 48 | mask << 31 | 1 << 22 | PT | 2 << 8 | 8;
        let words = emit(vec![mov(2, 0), mov(3, 3), decode(raw)]);
        validate(&words);
        let result = observe(&words, [23, 17, 11, 5]);
        // A levels-only mask has no size query, matching Ryujinx
        // CodeGen/Spirv/Instructions.cs:1487-1489. Exact results stay below.
        let queries = if mask & 7 != 0 {
            vec![(Op::ImageQuerySizeLod, vec![3])]
        } else {
            vec![]
        };
        assert_eq!(result.queries, queries);
        let channels = [23, 17, if mask & 4 != 0 { 11 } else { 1 }, 5];
        let expected: Vec<_> = channels
            .into_iter()
            .enumerate()
            .filter(|(i, _)| mask & (1 << i) != 0)
            .map(|(_, v)| v)
            .collect();
        assert_eq!(result.registers[8..8 + expected.len()], expected);
    }
    let samples = 0xdf50 << 48 | 4 << 31 | 2 << 22 | PT | 2 << 8 | 8;
    let words = emit(vec![mov(2, 0), decode(samples)]);
    validate(&words);
    let result = observe(&words, [0, 0, 8, 0]);
    assert_eq!(result.registers[8], 8);
    assert_eq!(result.queries, [(Op::ImageQuerySamples, vec![])]);
    for query in [0, 5, 16, 18, 20, 22, 63] {
        assert!(matches!(
            decode(0xdf50 << 48 | query << 22 | PT).kind,
            K::Unknown(ops::Unknown { .. })
        ));
    }
    // Bound shape requires a consistent typed binding, never assume 2D.
    // Ryujinx InstEmitTexture.cs:1116-1119 supplies the driver contract.
    let bound = decode(0xdf48 << 48 | 1 << 22 | 1 << 31 | PT);
    assert!(matches!(
        bound.kind,
        K::Texture(ops::Texture {
            op: shadowbox::ir::TexOp::Query { query: 1 },
            bound_handle: Some(0),
            ..
        })
    ));
    assert!(emit_unoptimized_with_header(&[bound], None).is_err());
    // Ryujinx InstEmitTexture.cs:1161-1178 preserves selected X/Y
    // destination slots and advances to the Z sample-count destination.
    for skipped in 0..4u64 {
        let raw = samples & !(15 << 31) | (4 | skipped) << 31;
        let words = emit(vec![
            mov(2, 0),
            mov(8, 0xdead_beef),
            mov(9, 0xcafe_babe),
            mov(10, 0x8765_4321),
            decode(raw),
        ]);
        validate(&words);
        let result = observe(&words, [0, 0, 8, 0]);
        let mut expected = [0xdead_beef, 0xcafe_babe, 0x8765_4321];
        expected[skipped.count_ones() as usize] = 8;
        assert_eq!(result.registers[8..11], expected);
        assert_eq!(result.queries, [(Op::ImageQuerySamples, vec![])]);
    }
    for mask in (0..16u64).filter(|mask| !(4..=7).contains(mask)) {
        assert!(matches!(
            decode(samples & !(15 << 31) | mask << 31).kind,
            K::Unknown(ops::Unknown { .. })
        ));
    }
}

#[test]
fn tmml_reverses_lods_scales_and_truncates_once_for_both_outputs() {
    // Fields: Mesa sm50.rs:2287-2309. Values: Ryujinx
    // InstEmitTexture.cs:908-923 and CodeGen/Spirv/Instructions.cs:859-896.
    for bound in [false, true] {
        for dim in [0u64, 1, 2, 3, 6, 7] {
            let raw = (if bound { 0xdf58 } else { 0xdf60 }) << 48
                | 3 << 31
                | dim << 28
                | (if bound { 255 } else { 40 }) << 20
                | if bound { 9 << 36 } else { 0 }
                | PT
                | 2 << 8
                | 8;
            let words = emit_fragment(vec![
                mov(40, 0),
                mov(2, 1.0f32.to_bits()),
                mov(3, (-2.0f32).to_bits()),
                mov(4, 0.5f32.to_bits()),
                mov(5, 4.0f32.to_bits()),
                decode(raw),
            ]);
            validate(&words);
            let result = observe(&words, [(-1.234f32).to_bits(), 2.345f32.to_bits(), 0, 0]);
            assert_eq!(&result.registers[8..10], &[600, (-315i32) as u32]);
            assert!(matches!(
                decode(raw | 4 << 31).kind,
                K::Unknown(ops::Unknown { .. })
            ));
            assert_eq!(result.queries.len(), 1);
            let array = matches!(dim, 1 | 3 | 7);
            let count = if dim < 2 {
                1
            } else if dim < 4 {
                2
            } else {
                3
            };
            let coords = [
                1.0f32.to_bits(),
                (-2.0f32).to_bits(),
                0.5f32.to_bits(),
                4.0f32.to_bits(),
            ];
            assert_eq!(
                result.queries[0],
                (
                    Op::ImageQueryLod,
                    coords[usize::from(array)..usize::from(array) + count].to_vec()
                )
            );
            assert!(matches!(
                decode(raw | 1 << 35).kind,
                K::Unknown(ops::Unknown { .. })
            ));
        }
    }
}

#[test]
fn optimized_integer_image_atomics_preserve_the_access_when_the_result_is_discarded() {
    use shadowbox::emitter::emit_with_header;
    let header = headers::parse(&[0; 32]).unwrap();
    for signed in [false, true] {
        for op in 0..=9 {
            let raw = surface_atom(false, true, signed, 3, op) | 255;
            let mut xs = vec![
                mov(2, 28),
                mov(3, 7),
                mov(20, 5),
                mov(21, 13),
                decode(raw),
                Instruction {
                    address: 0,
                    predicate: None,
                    kind: K::Exit(ops::Exit { cc_test: 15 }),
                },
            ];
            for (i, x) in xs.iter_mut().enumerate() {
                x.address = i as u32 * 8;
            }
            let words = emit_with_header(&xs, Some(&header)).unwrap().words;
            validate(&words);
            assert_eq!(
                ops(&words)
                    .iter()
                    .filter(|(op, _)| *op == Op::ImageTexelPointer)
                    .count(),
                1
            );
            assert_eq!(
                ops(&words)
                    .iter()
                    .filter(|(op, _)| matches!(
                        op,
                        Op::AtomicExchange
                            | Op::AtomicCompareExchange
                            | Op::AtomicCompareExchangeWeak
                            | Op::AtomicIIncrement
                            | Op::AtomicIDecrement
                            | Op::AtomicIAdd
                            | Op::AtomicISub
                            | Op::AtomicSMin
                            | Op::AtomicUMin
                            | Op::AtomicSMax
                            | Op::AtomicUMax
                            | Op::AtomicAnd
                            | Op::AtomicOr
                            | Op::AtomicXor
                    ))
                    .count(),
                1
            );
        }
    }
}

#[test]
fn one_surface_handle_has_distinct_typed_views_and_shared_identical_views() {
    use shadowbox::emitter::{emit_with_header, HandleLeaf, HandleSource};
    let header = headers::parse(&[0; 32]).unwrap();
    for source in 0..3 {
        let bound = source == 0;
        let handle = if bound {
            HandleSource::Bound(9)
        } else if source == 1 {
            HandleSource::Leaf(HandleLeaf::Constant(0x1234))
        } else {
            HandleSource::Leaf(HandleLeaf::ConstantBuffer {
                bank: 3,
                offset: 16,
            })
        };
        let mut xs = vec![mov(2, 13), mov(3, 17), mov(4, 19), mov(8, 0x1234)];
        if !bound {
            xs.push(Instruction {
                address: 0,
                predicate: None,
                kind: K::Mov(ops::Mov {
                    quad_lanes: 0xf, // Mesa sm50.rs:1936,1941,1946.
                    dst: R(20),
                    src: if source == 1 {
                        S::Immediate(0x1234)
                    } else {
                        S::ConstantBuffer {
                            binding: 3,
                            offset: 16,
                        }
                    },
                }),
            });
        }
        // Raw6 and raw7 are the same typed view. Formatless .P has its own
        // view. Raw sizes: InstEmitSurface.cs:769-780; dimensions: gm107.c:759-766.
        let mut expected = Vec::new();
        for target in 0..=5 {
            for (raw, field, image_format) in [
                (false, 1, 0),
                (true, 0, 39),
                (true, 1, 29),
                (true, 2, 38),
                (true, 3, 28),
                (true, 4, 33),
                (true, 5, 35),
                (true, 6, 30),
            ] {
                xs.push(decode(surface(true, target, raw, field, bound)));
                let k = expected.len() as u32;
                expected.push(shadowbox::emitter::SurfaceDescriptor {
                    source: handle,
                    target: target as u8,
                    image_format,
                    set: if bound { 2 } else { 0 },
                    binding: if bound {
                        if k == 0 {
                            9
                        } else {
                            8192 + k
                        }
                    } else {
                        2048 + k
                    },
                });
            }
            // An identical view shares its descriptor and ORs coherence.
            xs.push(decode(surface(true, target, true, 6, bound) | 1 << 24));
        }
        xs.push(Instruction {
            address: 0,
            predicate: None,
            kind: K::Exit(ops::Exit { cc_test: 0xf }),
        });
        for (i, x) in xs.iter_mut().enumerate() {
            x.address = i as u32 * 8;
        }
        for optimized in [false, true] {
            let module = if optimized {
                emit_with_header(&xs, Some(&header))
            } else {
                emit_unoptimized_with_header(&xs, Some(&header))
            }
            .unwrap();
            assert_eq!(module.surface_descriptors, expected);
            validate(&module.words);
            let instructions = ops(&module.words);
            let decorations: HashMap<_, _> = instructions
                .iter()
                .filter(|(op, a)| *op == Op::Decorate && a[1] == 33)
                .map(|(_, a)| (a[0], a[2]))
                .collect();
            let pointer_types: HashMap<_, _> = instructions
                .iter()
                .filter(|(op, _)| *op == Op::TypePointer)
                .map(|(_, a)| (a[0], a[2]))
                .collect();
            let image_formats: HashMap<_, _> = instructions
                .iter()
                .filter(|(op, _)| *op == Op::TypeImage)
                .map(|(_, a)| (a[0], a[7]))
                .collect();
            let images: HashMap<_, _> = instructions
                .iter()
                .filter(|(op, a)| {
                    *op == Op::Variable && image_formats.contains_key(&pointer_types[&a[0]])
                })
                .map(|(_, a)| (a[1], image_formats[&pointer_types[&a[0]]]))
                .collect();
            for descriptor in &expected {
                let (&variable, &format) = images
                    .iter()
                    .find(|(var, _)| decorations[var] == descriptor.binding)
                    .unwrap();
                assert_eq!(format, descriptor.image_format);
                assert!(instructions.iter().any(
                    |(op, a)| *op == Op::Decorate && a == &&[variable, 34, descriptor.set][..]
                ));
                let coherent = descriptor.image_format == 30;
                assert_eq!(
                    instructions
                        .iter()
                        .any(|(op, a)| *op == Op::Decorate && a == &&[variable, 23][..]),
                    coherent
                );
            }
            assert_eq!(
                instructions
                    .iter()
                    .filter(|(op, _)| *op == Op::ImageWrite)
                    .count(),
                54
            );
            if source != 2 && !optimized {
                let writes = observe(&module.words, [0; 4]).writes;
                assert_eq!(writes.len(), 54);
                for (i, (coords, value)) in writes.iter().enumerate() {
                    let target = i / 9;
                    let count = [1, 1, 2, 2, 3, 3][target];
                    assert_eq!(*coords, [13, 17, 19][..count]);
                    assert_eq!(value[0], 0x1234);
                }
            }
        }
    }
}

#[test]
fn raw_stores_and_signed_unsigned_atomics_select_their_own_views() {
    use shadowbox::emitter::HandleSource;
    let header = headers::parse(&[0; 32]).unwrap();
    for bound in [false, true] {
        let mut store = decode(surface(true, 3, true, 4, bound));
        if let K::Surface(ops::Surface {
            ref mut handle,
            ref mut data,
            ..
        }) = store.kind
        {
            *data = R(12);
            if !bound {
                *handle = S::Register(R(40));
            }
        }
        // ImageAtomic formats: Ryujinx InstEmitSurface.cs:709-720. Raw
        // B32 is R32ui, so the unsigned atomic must share that view while
        // signed min uses a separate R32i descriptor.
        let mut xs = vec![
            mov(2, 13),
            mov(3, 17),
            mov(12, 0x1234),
            mov(20, 5),
            mov(40, 0),
            decode(surface_atom(false, bound, true, 3, 1)),
            decode(surface_atom(false, bound, false, 3, 0)),
            store,
            Instruction {
                address: 0,
                predicate: None,
                kind: K::Exit(ops::Exit { cc_test: 0xf }),
            },
        ];
        for (i, x) in xs.iter_mut().enumerate() {
            x.address = i as u32 * 8;
        }
        let module = emit_unoptimized_with_header(&xs, Some(&header)).unwrap();
        validate(&module.words);
        assert_eq!(module.surface_descriptors.len(), 2);
        assert_eq!(module.surface_descriptors[0].image_format, 24);
        assert_eq!(module.surface_descriptors[1].image_format, 33);
        assert_eq!(
            module.surface_descriptors[0].source,
            module.surface_descriptors[1].source
        );
        if bound {
            assert_eq!(module.surface_descriptors[0].source, HandleSource::Bound(9));
        }
        let result = observe(&module.words, [(-9i32) as u32, 0, 0, 0]);
        assert_eq!(result.writes, vec![(vec![13, 17], vec![0x1234, 0, 0, 0])]);
        assert_eq!(result.atomics.len(), 2);
        assert_eq!(result.atomics[0].opcode, Op::AtomicSMin);
        assert_eq!(result.atomics[0].value, (-9i32) as u32);
        assert_eq!(result.atomics[1].opcode, Op::AtomicIAdd);
        assert_eq!(result.atomics[1].value, (-4i32) as u32);
        assert_eq!(result.registers[8], (-9i32) as u32);
        // Check the actual variables used, not only the metadata vector.
        let instructions = ops(&module.words);
        let texel_images: Vec<_> = instructions
            .iter()
            .filter(|(op, _)| *op == Op::ImageTexelPointer)
            .map(|(_, a)| a[2])
            .collect();
        assert_ne!(texel_images[0], texel_images[1]);
        let binding = |var| {
            instructions
                .iter()
                .find(|(op, a)| *op == Op::Decorate && a[0] == var && a[1] == 33)
                .unwrap()
                .1[2]
        };
        assert_eq!(
            binding(texel_images[0]),
            module.surface_descriptors[0].binding
        );
        assert_eq!(
            binding(texel_images[1]),
            module.surface_descriptors[1].binding
        );
    }
}
