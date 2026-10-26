//! Exact register and image-operand checks on project-owned shaders. The small
//! SPIR-V executor treats the image operation as a deterministic external input;
//! it checks translation, not Vulkan filtering or GPU texture implementation.
//! SPIR-V opcodes follow 1.6 sections 3.3.9-12, 3.3.14, 3.3.17.
use shadowbox::ir::ops;
use shadowbox::spirv::Op;
use shadowbox::{
    decoder::{decode_stream, decode_word},
    emitter::emit_unoptimized_with_graphics_state,
    headers,
    interface::{BoundTexture, GraphicsState, TextureArray, TextureDimension},
    ir::{Instruction, InstructionKind, LdcAddressMode, LdcWidth, Register, Source, TexOp},
};
use std::collections::HashMap;

#[path = "support/spirv_validation.rs"]
mod spirv_validation;

const PT: u64 = 7 << 16;
const TEXEL: [u32; 4] = [0x8000_0000, 0x3e80_0000, 0x7fc1_2345, 0x7f80_0000];

fn mov(dst: u8, value: u32) -> InstructionKind {
    InstructionKind::Mov(ops::Mov {
        quad_lanes: 15,
        dst: Register(dst),
        src: Source::Immediate(value),
    })
}

fn decoded(raw: u64) -> InstructionKind {
    decode_word([raw as u32, (raw >> 32) as u32], 0).kind
}

fn emit(kinds: &[InstructionKind], bindings: &[BoundTexture]) -> Vec<u32> {
    emit_stage(kinds, bindings, true)
}

fn emit_stage(kinds: &[InstructionKind], bindings: &[BoundTexture], fragment: bool) -> Vec<u32> {
    let xs = kinds
        .iter()
        .enumerate()
        .map(|(i, kind)| Instruction {
            address: (i as u32 + 1) * 8,
            predicate: None,
            kind: *kind,
        })
        .collect::<Vec<_>>();
    let mut header_words = [0; 32];
    header_words[0] = if fragment { 5 << 10 } else { 0 };
    let header = headers::parse(&header_words).unwrap();
    let words = emit_unoptimized_with_graphics_state(
        &xs,
        Some(&header),
        &GraphicsState {
            texture_bindings: bindings,
            ..Default::default()
        },
    )
    .unwrap()
    .words;
    let check = spirv_validation::validate(&words);
    assert!(
        check.status.success(),
        "{}",
        String::from_utf8_lossy(&check.stderr)
    );
    if kinds.iter().any(|kind| {
        matches!(
            kind,
            InstructionKind::Texture(ops::Texture {
                dim: 7,
                offset: true,
                ..
            })
        )
    }) {
        let optimized = shadowbox::emitter::emit_with_graphics_state(
            &xs,
            Some(&header),
            &GraphicsState {
                texture_bindings: bindings,
                ..Default::default()
            },
        )
        .unwrap()
        .words;
        let check = spirv_validation::validate(&optimized);
        assert!(
            check.status.success(),
            "{}",
            String::from_utf8_lossy(&check.stderr)
        );
        let mut at = 5;
        let mut samples = 0;
        while at < optimized.len() {
            samples += usize::from(matches!(
                Op::of(optimized[at]),
                Op::ImageSampleImplicitLod
                    | Op::ImageSampleExplicitLod
                    | Op::ImageSampleDrefImplicitLod
                    | Op::ImageSampleDrefExplicitLod
            ));
            at += (optimized[at] >> 16) as usize;
        }
        assert_eq!(samples, 1, "optimization must preserve cube-array sampling");
    }
    words
}

#[derive(Debug, PartialEq, Eq)]
enum ImageCall {
    Sample {
        opcode: Op,
        coords: Vec<u32>,
        dref: Option<u32>,
        operands: Vec<u32>,
    },
    Size(u32),
    Levels,
}

struct Image {
    // SPIR-V Dim, Depth, Arrayed, MS and expected descriptor set/binding.
    shape: [u32; 4],
    descriptor: [u32; 2],
    sizes: Vec<u32>,
    levels: u32,
}

// This executor follows actual branches, loads and stores. Unknown executable
// operations fail, so extra lowering cannot silently disappear from the test.
fn run(words: &[u32], image: &Image) -> (Vec<u32>, Vec<ImageCall>) {
    run_with_uniform(words, image, 0)
}

fn run_with_uniform(words: &[u32], image: &Image, uniform: u32) -> (Vec<u32>, Vec<ImageCall>) {
    let mut values: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut labels = HashMap::new();
    let mut descriptors: HashMap<u32, [u32; 2]> = HashMap::new();
    let mut vectors = HashMap::new();
    let mut image_types = HashMap::new();
    let mut pointers = HashMap::new();
    let mut regs = Vec::new();
    let mut uint = 0;
    let mut calls = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        let op = Op::of(words[at]);
        let a = &words[at + 1..at + count];
        match op {
            Op::TypeInt if a[1..] == [32, 0] => uint = a[0],
            Op::TypeVector => {
                vectors.insert(a[0], a[2] as usize);
            }
            Op::TypeImage => {
                image_types.insert(a[0], [a[2], a[3], a[4], a[5]]);
            }
            Op::TypePointer => {
                pointers.insert(a[0], (a[1], a[2]));
            }
            Op::Variable if pointers[&a[0]] == (7, uint) => regs.push(a[1]),
            Op::Variable if image_types.contains_key(&pointers[&a[0]].1) => {
                assert_eq!(image_types[&pointers[&a[0]].1], image.shape);
                assert_eq!(descriptors[&a[1]], image.descriptor);
            }
            Op::Decorate if a[1] == 33 => descriptors.entry(a[0]).or_default()[1] = a[2],
            Op::Decorate if a[1] == 34 => descriptors.entry(a[0]).or_default()[0] = a[2],
            Op::Label => {
                labels.insert(a[0], at);
            }
            _ => {}
        }
        at += count;
    }
    at = 5;
    for _ in 0..10000 {
        let count = (words[at] >> 16) as usize;
        let op = Op::of(words[at]);
        let a = &words[at + 1..at + count];
        let scalar = |id| values[&id][0];
        let result = match op {
            Op::ConstantTrue => Some(vec![1]),
            Op::ConstantFalse => Some(vec![0]),
            Op::Constant => Some(vec![a[2]]),
            Op::ConstantComposite | Op::CompositeConstruct => {
                Some(a[2..].iter().flat_map(|id| values[id].clone()).collect())
            }
            Op::Variable => {
                // Handles retain variable identity; functions start at zero.
                values.insert(a[1], vec![if a[2] == 0 { a[1] } else { 0 }]);
                None
            }
            Op::Load | Op::Bitcast => Some(values[&a[2]].clone()),
            Op::Store => {
                values.insert(a[0], values[&a[1]].clone());
                None
            }
            Op::AccessChain => {
                assert_eq!((scalar(a[3]), scalar(a[4])), (0, 0));
                Some(vec![uniform; 4])
            }
            Op::CompositeExtract => Some(vec![values[&a[2]][a[3] as usize]]),
            Op::CompositeInsert => {
                let mut result = values[&a[3]].clone();
                result[a[4] as usize] = scalar(a[2]);
                Some(result)
            }
            Op::SampledImage => {
                let image_var = scalar(a[2]);
                let sampler_var = scalar(a[3]);
                let [set, binding] = descriptors[&image_var];
                assert_eq!(descriptors[&sampler_var], [set, binding + 1]);
                Some(vec![image_var])
            }
            Op::ImageSampleImplicitLod
            | Op::ImageSampleExplicitLod
            | Op::ImageSampleDrefImplicitLod
            | Op::ImageSampleDrefExplicitLod => {
                let shadow = matches!(
                    op,
                    Op::ImageSampleDrefImplicitLod | Op::ImageSampleDrefExplicitLod
                );
                let first = 4 + usize::from(shadow);
                let dref = shadow.then(|| scalar(a[4]));
                let operands = if a.len() > first {
                    std::iter::once(a[first])
                        .chain(a[first + 1..].iter().flat_map(|id| values[id].clone()))
                        .collect()
                } else {
                    Vec::new()
                };
                calls.push(ImageCall::Sample {
                    opcode: op,
                    coords: values[&a[3]].clone(),
                    dref,
                    operands,
                });
                Some(if shadow {
                    vec![TEXEL[1]]
                } else {
                    TEXEL.to_vec()
                })
            }
            Op::ImageQuerySizeLod => {
                let lod = scalar(a[3]);
                calls.push(ImageCall::Size(lod));
                let count = vectors.get(&a[0]).copied().unwrap_or(1);
                assert_eq!(count, image.sizes.len());
                Some(
                    image
                        .sizes
                        .iter()
                        .enumerate()
                        .map(|(i, size)| {
                            // Array layers are independent of mip level. Cube sizes
                            // have two spatial axes (Ryujinx Instructions.cs:1496-1501).
                            if image.shape[2] != 0 && i + 1 == count {
                                *size
                            } else {
                                (*size >> lod.min(31)).max(1)
                            }
                        })
                        .collect(),
                )
            }
            Op::ImageQueryLod => {
                // Deterministic external LOD value. Check that the array layer
                // is absent, and query the clamped mip at fractional level 2.75.
                let coords = &values[&a[3]];
                let count = if image.shape[0] == 0 {
                    1
                } else if image.shape[0] == 1 {
                    2
                } else {
                    3
                };
                assert_eq!(coords.len(), count);
                Some(vec![2.75f32.to_bits(), 9.5f32.to_bits()])
            }
            Op::ImageQueryLevels => {
                calls.push(ImageCall::Levels);
                Some(vec![image.levels])
            }
            Op::ConvertFToS => Some(vec![(f32::from_bits(scalar(a[2])) as i32) as u32]),
            Op::ConvertSToF => Some(vec![(scalar(a[2]) as i32 as f32).to_bits()]),
            Op::ConvertUToF => Some(vec![(scalar(a[2]) as f32).to_bits()]),
            Op::FAdd => Some(vec![(f32::from_bits(scalar(a[2]))
                + f32::from_bits(scalar(a[3])))
            .to_bits()]),
            Op::FDiv => Some(vec![(f32::from_bits(scalar(a[2]))
                / f32::from_bits(scalar(a[3])))
            .to_bits()]),
            Op::BitFieldSExtract => {
                let shift = scalar(a[3]);
                let width = scalar(a[4]);
                Some(vec![
                    (((scalar(a[2]) >> shift) << (32 - width)) as i32 >> (32 - width)) as u32,
                ])
            }
            Op::IAdd => Some(vec![scalar(a[2]).wrapping_add(scalar(a[3]))]),
            Op::Select => Some(values[&a[if scalar(a[2]) != 0 { 3 } else { 4 }]].clone()),
            Op::IEqual => Some(vec![u32::from(scalar(a[2]) == scalar(a[3]))]),
            Op::INotEqual => Some(vec![u32::from(scalar(a[2]) != scalar(a[3]))]),
            Op::Branch => {
                at = labels[&a[0]];
                continue;
            }
            Op::BranchConditional => {
                at = labels[&a[if scalar(a[0]) != 0 { 1 } else { 2 }]];
                continue;
            }
            Op::Switch => {
                let selector = scalar(a[0]);
                let label = a[2..]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .find(|pair| pair[0] == selector)
                    .map_or(a[1], |pair| pair[1]);
                at = labels[&label];
                continue;
            }
            Op::Return => return (regs[..256].iter().map(|r| values[r][0]).collect(), calls),
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
            other => panic!("unsupported SPIR-V executable opcode {}", other.number()),
        };
        if let Some(result) = result {
            values.insert(a[1], result);
        }
        at += count;
    }
    panic!("SPIR-V dispatch did not terminate")
}

fn shape(dim: TextureDimension, depth: bool) -> [u32; 4] {
    let (base, array) = match dim {
        TextureDimension::D1 => (0, 0),
        TextureDimension::Array1D => (0, 1),
        TextureDimension::D2 => (1, 0),
        TextureDimension::Array2D => (1, 1),
        TextureDimension::D3 => (2, 0),
        TextureDimension::Cube => (3, 0),
        TextureDimension::ArrayCube => (3, 1),
    };
    [base, u32::from(depth), array, 0]
}

#[test]
fn bound_arrays_and_cubes_preserve_exact_operands_and_channel_bits() {
    // Fields: Mesa sm50.rs:2170-2197,2105-2119. Register order: Ryujinx
    // InstEmitTexture.cs:203-258; signed layer: Spirv/Instructions.cs:1218-1225.
    for dim in [
        TextureDimension::Array1D,
        TextureDimension::Array2D,
        TextureDimension::Cube,
        TextureDimension::ArrayCube,
    ] {
        for depth in [false, true] {
            for lod in [0, 1, 2, 3, 6, 7_u64] {
                for mask in [1, 5, 10, 15_u64] {
                    let array = matches!(
                        dim,
                        TextureDimension::Array1D
                            | TextureDimension::Array2D
                            | TextureDimension::ArrayCube
                    );
                    let axes = match dim {
                        TextureDimension::Array1D => 1,
                        TextureDimension::Array2D => 2,
                        _ => 3,
                    };
                    let coords = [0x3e80_0000, 0xbf00_0000, 0x3f40_0000];
                    let mut kinds = vec![mov(8, (-2_i32) as u32), mov(20, 0x3fc0_0000)];
                    for (i, value) in coords[..axes].iter().enumerate() {
                        kinds.push(mov(8 + u8::from(array) + i as u8, *value));
                    }
                    kinds.push(mov(20 + u8::from(lod >= 2), 0x3e00_0000));
                    // Restore the LOD value after placing the optional Dref.
                    if lod >= 2 {
                        kinds.push(mov(20, 0x3fc0_0000));
                    }
                    let raw = 0xc038_u64 << 48
                        | PT
                        | lod << 55
                        | u64::from(depth) << 50
                        | 7 << 36
                        | mask << 31
                        | (dim as u64) << 28
                        | (if !depth && lod <= 1 { 255 } else { 20 }) << 20
                        | 8 << 8
                        | 8;
                    if array {
                        // IADD is intentionally outside handle constant tracking.
                        // This forces runtime signed layer conversion, so the
                        // integer-to-float instruction is also checked by value.
                        kinds.push(InstructionKind::IAdd(ops::IAdd {
                            dst: Register(8),
                            lhs: Register(8),
                            rhs: Source::Immediate(0),
                            cc: false,
                            x: false,
                            negate: [false; 2],
                        }));
                    }
                    kinds.push(decoded(raw));
                    kinds.push(InstructionKind::Exit(ops::Exit { cc_test: 15 }));
                    let image = Image {
                        shape: shape(dim, depth),
                        descriptor: [1, 14],
                        sizes: vec![],
                        levels: 0,
                    };
                    let (regs, calls) = run(&emit(&kinds, &[]), &image);
                    let mut expected_coords = coords[..axes].to_vec();
                    if array {
                        expected_coords.push((-2.0_f32).to_bits());
                    }
                    let expected_operands = match lod {
                        0 => vec![],
                        1 => vec![2, 0],
                        2 | 6 => vec![1, 0x3fc0_0000],
                        _ => vec![2, 0x3fc0_0000],
                    };
                    assert_eq!(
                        calls,
                        [ImageCall::Sample {
                            opcode: match (depth, lod) {
                                (false, 0 | 2 | 6) => Op::ImageSampleImplicitLod,
                                (false, _) => Op::ImageSampleExplicitLod,
                                (true, 0 | 2 | 6) => Op::ImageSampleDrefImplicitLod,
                                (true, _) => Op::ImageSampleDrefExplicitLod,
                            },
                            coords: expected_coords,
                            dref: depth.then_some(0x3e00_0000),
                            operands: expected_operands,
                        }]
                    );
                    let expected = TEXEL
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| mask & (1 << i) != 0)
                        .map(|(_, bits)| if depth { TEXEL[1] } else { *bits })
                        .collect::<Vec<_>>();
                    assert_eq!(&regs[8..8 + expected.len()], expected);
                }
            }
        }
    }
}

#[test]
fn standalone_bound_queries_use_driver_shape_lod_and_sparse_masks() {
    // TXQ fields: Mesa sm50.rs:2351-2381. Semantics: Ryujinx
    // InstEmitTexture.cs:1084-1089,1116-1148; Spirv/Instructions.cs:1487-1521.
    for (dim, sizes) in [
        (TextureDimension::D1, vec![96]),
        (TextureDimension::Array1D, vec![96, 7]),
        (TextureDimension::D2, vec![96, 48]),
        (TextureDimension::Array2D, vec![96, 48, 7]),
        (TextureDimension::D3, vec![96, 48, 24]),
        (TextureDimension::Cube, vec![96, 96]),
        (TextureDimension::ArrayCube, vec![96, 96, 7]),
    ] {
        for lod in [0, 2, 7] {
            for mask in 1..16_u64 {
                if mask & 7 & !((1 << sizes.len()) - 1) != 0 {
                    continue;
                }
                for depth in [false, true] {
                    if dim == TextureDimension::D3 && depth {
                        continue;
                    }
                    let binding = BoundTexture {
                        binding: 7,
                        dimension: dim,
                        depth_compare: depth,
                        multisample: false,
                    };
                    let raw = 0xdf48_u64 << 48
                        | PT
                        | 7 << 36
                        | mask << 31
                        | 1 << 22
                        | (if mask == 8 { 255 } else { 8 }) << 8
                        | 8;
                    let image = Image {
                        shape: shape(dim, depth),
                        descriptor: [1, 14],
                        sizes: sizes.clone(),
                        levels: 8,
                    };
                    let (regs, calls) = run(
                        &emit(
                            &[
                                mov(8, lod),
                                decoded(raw),
                                InstructionKind::Exit(ops::Exit { cc_test: 15 }),
                            ],
                            &[binding],
                        ),
                        &image,
                    );
                    let array = matches!(
                        dim,
                        TextureDimension::Array1D
                            | TextureDimension::Array2D
                            | TextureDimension::ArrayCube
                    );
                    let mut expected = [1; 4];
                    for (i, size) in sizes.iter().enumerate() {
                        expected[i] = if array && i + 1 == sizes.len() {
                            *size
                        } else {
                            (*size >> lod).max(1)
                        };
                    }
                    expected[3] = 8;
                    let expected = expected
                        .iter()
                        .enumerate()
                        .filter(|(i, _)| mask & (1 << i) != 0)
                        .map(|(_, v)| *v)
                        .collect::<Vec<_>>();
                    assert_eq!(&regs[8..8 + expected.len()], expected);
                    let mut expected_calls = Vec::new();
                    if mask & 7 != 0 {
                        expected_calls.push(ImageCall::Size(lod));
                    }
                    if mask & 8 != 0 {
                        expected_calls.push(ImageCall::Levels);
                    }
                    assert_eq!(calls, expected_calls);
                }
            }
        }
    }
}

#[test]
fn array_offsets_follow_lod_then_offset_then_dref_register_order() {
    // AOFFI54, dimensions28..30, binding36..48: Mesa sm50.rs:2193-2196,
    // 2173-2175. Offset signed nibble order: Ryujinx InstEmitTexture.cs:252-276.
    for (dim, coords) in [
        (TextureDimension::Array1D, vec![0x3e80_0000]),
        (TextureDimension::Array2D, vec![0x3e80_0000, 0xbf00_0000]),
    ] {
        for depth in [false, true] {
            for lod in [0, 1, 2, 3, 6, 7_u64] {
                let mut kinds = vec![mov(8, 3), mov(20, 0x3f80_0000)];
                for (i, bits) in coords.iter().enumerate() {
                    kinds.push(mov(9 + i as u8, *bits));
                }
                let offset_reg = 20 + u8::from(lod >= 2);
                kinds.push(mov(offset_reg, 0xe3));
                kinds.push(mov(offset_reg + 1, 0x3e00_0000));
                let raw = 0xc038_u64 << 48
                    | PT
                    | lod << 55
                    | 1 << 54
                    | u64::from(depth) << 50
                    | 513 << 36
                    | 15 << 31
                    | (dim as u64) << 28
                    | 20 << 20
                    | 8 << 8
                    | 8;
                kinds.extend([
                    decoded(raw),
                    InstructionKind::Exit(ops::Exit { cc_test: 15 }),
                ]);
                let image = Image {
                    shape: shape(dim, depth),
                    descriptor: [1, 1026],
                    sizes: vec![],
                    levels: 0,
                };
                let (regs, calls) = run(&emit(&kinds, &[]), &image);
                let mut expected_coords = coords.clone();
                expected_coords.push(3.0_f32.to_bits());
                let mut operands = match lod {
                    0 => vec![8],
                    1 => vec![10, 0],
                    2 | 6 => vec![9, 0x3f80_0000],
                    _ => vec![10, 0x3f80_0000],
                };
                operands.push(3);
                if coords.len() == 2 {
                    operands.push((-2_i32) as u32);
                }
                assert_eq!(
                    calls,
                    [ImageCall::Sample {
                        opcode: match (depth, lod) {
                            (false, 0 | 2 | 6) => Op::ImageSampleImplicitLod,
                            (false, _) => Op::ImageSampleExplicitLod,
                            (true, 0 | 2 | 6) => Op::ImageSampleDrefImplicitLod,
                            (true, _) => Op::ImageSampleDrefExplicitLod,
                        },
                        coords: expected_coords,
                        dref: depth.then_some(0x3e00_0000),
                        operands,
                    }]
                );
                assert_eq!(&regs[8..12], if depth { &[TEXEL[1]; 4] } else { &TEXEL });
            }
        }
    }
}

#[test]
fn standalone_bindless_queries_consume_handle_then_lod_and_infer_shape() {
    // TXQ.B source8..15 and masks31..34: Mesa sm50.rs:2366,2380.
    // Shape fallback and handle/LOD order: Ryujinx InstEmitTexture.cs:1084-1114.
    for mask in [1, 3, 8, 11, 4, 5, 7, 15_u64] {
        let dim = if mask & 4 != 0 {
            TextureDimension::D3
        } else {
            TextureDimension::D2
        };
        let sizes = if mask & 4 != 0 {
            vec![96, 48, 24]
        } else {
            vec![96, 48]
        };
        let raw = 0xdf50_u64 << 48 | PT | mask << 31 | 1 << 22 | 8 << 8 | 8;
        let image = Image {
            shape: shape(dim, false),
            descriptor: [0, 0],
            sizes,
            levels: 8,
        };
        let (regs, calls) = run(
            &emit(
                &[
                    mov(8, 0x1234),
                    mov(9, 2),
                    decoded(raw),
                    InstructionKind::Exit(ops::Exit { cc_test: 15 }),
                ],
                &[],
            ),
            &image,
        );
        let expected = [24, 12, 6, 8]
            .into_iter()
            .enumerate()
            .filter(|(i, _)| mask & (1 << i) != 0)
            .map(|(_, v)| v)
            .collect::<Vec<_>>();
        assert_eq!(&regs[8..8 + expected.len()], expected);
        let mut expected_calls = Vec::new();
        if mask & 7 != 0 {
            expected_calls.push(ImageCall::Size(2));
        }
        if mask & 8 != 0 {
            expected_calls.push(ImageCall::Levels);
        }
        assert_eq!(calls, expected_calls);
    }
}

#[test]
fn bound_metadata_conflicts_and_unproven_query_modes_are_rejected() {
    // No TXQ dimension field: Mesa nak_nir_lower_tex.c:59-64.
    // Driver shape: Ryujinx InstEmitTexture.cs:1116-1119. Query selectors:
    // Mesa sm50.rs:2368-2377; envytools gm107.c:882-891.
    let binding = BoundTexture {
        binding: 7,
        dimension: TextureDimension::D2,
        depth_compare: false,
        multisample: false,
    };
    let query = 0xdf48_u64 << 48 | PT | 7 << 36 | 3 << 31 | 1 << 22 | 8 << 8;
    let xs = |raw| {
        [
            Instruction {
                address: 8,
                predicate: None,
                kind: decoded(raw),
            },
            Instruction {
                address: 16,
                predicate: None,
                kind: InstructionKind::Exit(ops::Exit { cc_test: 15 }),
            },
        ]
    };
    for bindings in [
        vec![],
        vec![BoundTexture {
            binding: 8,
            ..binding
        }],
        vec![
            binding,
            BoundTexture {
                dimension: TextureDimension::D3,
                ..binding
            },
        ],
        vec![BoundTexture {
            multisample: true,
            ..binding
        }],
    ] {
        assert!(emit_unoptimized_with_graphics_state(
            &xs(query),
            None,
            &GraphicsState {
                texture_bindings: &bindings,
                ..Default::default()
            }
        )
        .is_err());
    }
    // A 2D image has no third size axis in Ryujinx's SPIR-V lowering.
    assert!(emit_unoptimized_with_graphics_state(
        &xs(query | 4 << 31),
        None,
        &GraphicsState {
            texture_bindings: &[binding],
            ..Default::default()
        }
    )
    .is_err());
    // The instruction's explicit shape must agree with the driver metadata.
    let sample = 0xc038_u64 << 48 | PT | 7 << 36 | 3 << 28 | 1 << 55 | 15 << 31 | 255 << 20;
    let mut instructions = xs(query).to_vec();
    instructions.insert(
        0,
        Instruction {
            address: 0,
            predicate: None,
            kind: decoded(sample),
        },
    );
    assert!(emit_unoptimized_with_graphics_state(
        &instructions,
        None,
        &GraphicsState {
            texture_bindings: &[binding],
            ..Default::default()
        }
    )
    .is_err());
    // No unexplained bindless table selector survives.
    let bindless = 0xdf50_u64 << 48 | PT | 1 << 22 | 3 << 31;
    for field in [1 << 36, 1 << 48] {
        assert!(matches!(
            decoded(bindless | field),
            InstructionKind::Unknown(ops::Unknown { .. })
        ));
    }
    for selector in (0..64).filter(|q| *q != 1) {
        assert!(matches!(
            decoded((bindless & !(63 << 22)) | selector << 22),
            InstructionKind::Unknown(ops::Unknown { .. })
        ));
    }
}

// Dynamic offsets: Ryujinx TexturePass.cs:501-525,591-650; scalar offset
// packing: InstEmitTexture.cs:272-276. Words are authored from Mesa
// sm50.rs:2171-2197 and envytools gm107.c:1970,1026-1033,875-880.
#[test]
fn dynamic_sample_offsets_use_queried_mip_and_preserve_lod_depth_and_layer() {
    for dim in [
        TextureDimension::D1,
        TextureDimension::Array1D,
        TextureDimension::D2,
        TextureDimension::Array2D,
        TextureDimension::D3,
        TextureDimension::ArrayCube,
    ] {
        let array = matches!(
            dim,
            TextureDimension::Array1D | TextureDimension::Array2D | TextureDimension::ArrayCube
        );
        let axes = match dim {
            TextureDimension::D1 | TextureDimension::Array1D => 1,
            TextureDimension::D2 | TextureDimension::Array2D => 2,
            _ => 3,
        };
        for bound in [false, true] {
            for lod in [0u64, 1, 2, 3, 6, 7] {
                for depth in [false, true] {
                    if depth && dim == TextureDimension::D3 {
                        continue;
                    }
                    let raw = (if bound { 0xc038u64 } else { 0xdeb8 }) << 48
                        | lod << if bound { 55 } else { 37 }
                        | 1 << if bound { 54 } else { 36 }
                        | if bound { 7 << 36 } else { 0 }
                        | u64::from(depth) << 50
                        | (dim as u64) << 28
                        | 15 << 31
                        | 20 << 20
                        | PT
                        | 8 << 8
                        | 32;
                    let start = 20 + u8::from(!bound);
                    let offset = start + u8::from(lod >= 2);
                    let mut kinds = vec![mov(20, 0x1234), mov(start, 1.5f32.to_bits())];
                    if !bound {
                        kinds[0] = InstructionKind::Mov(ops::Mov {
                            dst: Register(20),
                            src: Source::ConstantBuffer {
                                binding: 0,
                                offset: 0,
                            },
                            quad_lanes: 15,
                        });
                    }
                    if array {
                        kinds.push(mov(8, 7));
                    }
                    for axis in 0..axes {
                        kinds.push(mov(8 + u8::from(array) + axis, 0.5f32.to_bits()));
                    }
                    kinds.push(InstructionKind::Mov(ops::Mov {
                        dst: Register(offset),
                        src: Source::ConstantBuffer {
                            binding: 0,
                            offset: 0,
                        },
                        quad_lanes: 15,
                    }));
                    if depth {
                        kinds.push(mov(offset + 1, 0.25f32.to_bits()));
                    }
                    kinds.extend([
                        decoded(raw),
                        InstructionKind::Exit(ops::Exit { cc_test: 15 }),
                    ]);
                    let image = Image {
                        shape: shape(dim, depth),
                        descriptor: if bound { [1, 14] } else { [0, 0] },
                        sizes: (0..if dim == TextureDimension::ArrayCube {
                            2
                        } else {
                            axes
                        })
                            .map(|axis| 64u32 << axis)
                            .chain(array.then_some(12))
                            .collect(),
                        levels: 8,
                    };
                    let binding = [BoundTexture {
                        binding: 7,
                        dimension: dim,
                        depth_compare: depth,
                        multisample: false,
                    }];
                    for fragment in [false, true] {
                        let words = emit_stage(&kinds, &binding, fragment);
                        let (regs, calls) = run_with_uniform(&words, &image, 0x7f8);
                        assert_eq!(&regs[32..36], if depth { &[TEXEL[1]; 4] } else { &TEXEL });
                        let expected_coords = (0..axes)
                            .map(|axis| {
                                let offsets = [-8f32, -1., 7.];
                                // Ryujinx TexturePass.cs:623-648 queries axis2 for
                                // cube-array Z; Instructions.cs:1496-1519 returns
                                // array layers there, independent of the mip.
                                let extent = if dim == TextureDimension::ArrayCube && axis == 2 {
                                    12
                                } else {
                                    (64u32 << axis) >> if fragment { 2 } else { 0 }
                                };
                                (0.5 + offsets[axis as usize] / extent as f32).to_bits()
                            })
                            .chain(array.then_some(7f32.to_bits()))
                            .collect::<Vec<_>>();
                        let explicit = !fragment || matches!(lod, 1 | 3 | 7);
                        let operands = match lod {
                            0 if fragment => vec![],
                            0 | 1 => vec![2, 0],
                            2 | 6 if fragment => vec![1, 1.5f32.to_bits()],
                            _ => vec![2, 1.5f32.to_bits()],
                        };
                        assert_eq!(
                            calls,
                            vec![
                                ImageCall::Size(if fragment { 2 } else { 0 }),
                                ImageCall::Sample {
                                    opcode: match (depth, explicit) {
                                        (false, false) => Op::ImageSampleImplicitLod,
                                        (false, true) => Op::ImageSampleExplicitLod,
                                        (true, false) => Op::ImageSampleDrefImplicitLod,
                                        (true, true) => Op::ImageSampleDrefExplicitLod,
                                    },
                                    coords: expected_coords,
                                    dref: depth.then_some(0.25f32.to_bits()),
                                    operands,
                                }
                            ],
                            "{raw:016x}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn constant_cube_array_offsets_fail_instead_of_emitting_invalid_const_offset() {
    // TexturePass.cs:399-405 retains constant offsets. Vulkan disallows
    // Cube ConstOffset; the reference defines no replacement for this case.
    for bound in [false, true] {
        let word = (if bound { 0xc038u64 } else { 0xdeb8 }) << 48
            | 1 << if bound { 54 } else { 36 }
            | if bound { 7 << 36 } else { 0 }
            | 7 << 28
            | 15 << 31
            | 20 << 20
            | PT
            | 8 << 8
            | 32;
        let kinds = [
            mov(20, 0),
            mov(21, 0x7f8),
            decoded(word),
            InstructionKind::Exit(ops::Exit { cc_test: 15 }),
        ];
        let xs = kinds
            .into_iter()
            .enumerate()
            .map(|(i, kind)| Instruction {
                address: i as u32 * 8,
                predicate: None,
                kind,
            })
            .collect::<Vec<_>>();
        assert!(
            emit_unoptimized_with_graphics_state(&xs, None, &GraphicsState::default()).is_err()
        );
    }
}

fn texture_array_words(ldc_opcode: u64, copy_handle: bool) -> Vec<u32> {
    // MOV: Mesa sm50.rs:1931-1951. LDC: :2386-2398,2633-2648.
    // Bindless TEX.Sample: :2171-2197. EXIT/NOP: :3106-3110,3178-3184.
    let mov =
        |dst: u64, value: u32| 0x0100_u64 << 48 | u64::from(value) << 20 | 15 << 12 | PT | dst;
    let ldc = ldc_opcode << 48 | 2 << 36 | 0x20 << 20 | PT | 12 << 8 | 1;
    let sample = 0xdeb8_u64 << 48 | 15 << 31 | 2 << 28 | 1 << 20 | PT | 10 << 8;
    let mut slots = vec![mov(10, 0), mov(11, 0), mov(12, 0), ldc];
    if copy_handle {
        slots.push(0x5c98_u64 << 48 | PT | 1 << 20 | 2);
        slots.push((sample & !(0xff << 20)) | 2 << 20);
    } else {
        slots.push(sample);
    }
    slots.push(0xe300_u64 << 48 | PT | 15);
    while !slots.len().is_multiple_of(3) {
        slots.push(0x50b0_u64 << 48 | PT);
    }
    slots
        .chunks(3)
        .flat_map(|chunk| std::iter::once(0).chain(chunk.iter().copied()))
        .flat_map(|word| [word as u32, (word >> 32) as u32])
        .collect()
}

fn texture_array_header() -> headers::ShaderHeader {
    let mut words = [0; 32];
    words[0] = 5 << 10;
    words[18] = 15;
    headers::parse(&words).unwrap()
}

fn texture_array_state(arrays: &[TextureArray]) -> GraphicsState<'_> {
    GraphicsState {
        texture_arrays: arrays,
        ..Default::default()
    }
}

fn module_instructions(words: &[u32]) -> impl Iterator<Item = (Op, &[u32])> {
    let mut at = 5;
    std::iter::from_fn(move || {
        let header = *words.get(at)?;
        let count = (header >> 16) as usize;
        let op = Op::of(header);
        let operands = &words[at + 1..at + count];
        at += count;
        Some((op, operands))
    })
}

#[test]
fn dynamic_texture_array_keeps_typed_ldc_to_sample_state() {
    let code = texture_array_words(0xef94, false);
    let decoded = decode_stream(shadowbox::QuietRun, &code);
    assert!(decoded.iter().any(|instruction| matches!(
        instruction.kind,
        InstructionKind::Ldc(ops::Ldc {
            dst: Register(1),
            binding: 2,
            offset: Register(12),
            immediate: 0x20,
            mem_type: LdcWidth::B32,
            address_mode: LdcAddressMode::Default,
        })
    )));
    assert!(decoded.iter().any(|instruction| matches!(
        instruction.kind,
        InstructionKind::Texture(ops::Texture {
            op: TexOp::Sample { lod: 0 },
            src0: Register(10),
            src1: Register(1),
            dim: 2,
            mask: 15,
            bound_handle: None,
            ..
        })
    )));

    let arrays = [TextureArray {
        binding: 2,
        handle: 0,
        length: 64,
    }];
    let header = texture_array_header();
    assert!(emit_unoptimized_with_graphics_state(
        &decoded,
        Some(&header),
        &GraphicsState::default(),
    )
    .is_err());
    let mut module = emit_unoptimized_with_graphics_state(
        &decoded,
        Some(&header),
        &texture_array_state(&arrays),
    )
    .unwrap();

    let raw = spirv_validation::validate(&module.words);
    assert!(
        raw.status.success(),
        "{}",
        String::from_utf8_lossy(&raw.stderr)
    );
    let ops = module_instructions(&module.words).collect::<Vec<_>>();
    assert!(ops
        .iter()
        .any(|(op, args)| *op == Op::Capability && *args == [5301]));
    assert!(ops
        .iter()
        .any(|(op, args)| *op == Op::Capability && *args == [5307]));
    assert!(ops.iter().any(|(op, _)| *op == Op::TypeArray));
    assert!(ops.iter().any(|(op, _)| *op == Op::AccessChain));
    assert!(
        ops.iter()
            .filter(|(op, args)| *op == Op::Decorate && args.last() == Some(&5300))
            .count()
            >= 5
    );
    assert!(ops.iter().any(|(op, _)| *op == Op::ImageSampleImplicitLod));

    module.optimize();
    let optimized = spirv_validation::validate(&module.words);
    assert!(
        optimized.status.success(),
        "{}",
        String::from_utf8_lossy(&optimized.stderr)
    );
}

#[test]
fn dynamic_texture_array_index_matches_the_independent_cpu_oracle() {
    // Ryujinx BindlessToArray.cs:214-221 derives
    // (vec_index << 1) + (element_index >> 1). LDC addresses are bytes.
    for source_index in 0..80_u32 {
        let byte_address = source_index * 8;
        let vec_index = byte_address >> 4;
        let element_index = (byte_address >> 2) & 3;
        let ryujinx_index = (vec_index << 1) + (element_index >> 1);
        assert_eq!(byte_address >> 3, ryujinx_index);
        let oracle = ryujinx_index.min(63);
        assert_eq!(oracle, source_index.min(63));
    }
}

#[test]
fn dynamic_texture_array_boundary_fails_closed() {
    let header = texture_array_header();
    let arrays = [TextureArray {
        binding: 2,
        handle: 0,
        length: 64,
    }];
    // B64 is not one packed handle, and a copied handle does not carry the
    // runtime descriptor index. Both remain unsupported.
    for code in [
        texture_array_words(0xef95, false),
        texture_array_words(0xef94, true),
    ] {
        let decoded = decode_stream(shadowbox::QuietRun, &code);
        assert!(emit_unoptimized_with_graphics_state(
            &decoded,
            Some(&header),
            &texture_array_state(&arrays),
        )
        .is_err());
    }
    for length in [0, 1] {
        let invalid = [TextureArray {
            binding: 2,
            handle: 0,
            length,
        }];
        let decoded = decode_stream(shadowbox::QuietRun, &texture_array_words(0xef94, false));
        assert!(emit_unoptimized_with_graphics_state(
            &decoded,
            Some(&header),
            &texture_array_state(&invalid),
        )
        .is_err());
    }
}
