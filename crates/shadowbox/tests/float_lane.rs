//! Execute the emitted scalar SPIR-V subset on project-owned inputs.
//! Encoding citations are in decoder.rs. Operation numbers/semantics follow
//! SPIR-V 1.6 sections 3.3.11-15, 3.3.17, 3.3.24 and GLSL.std.450
//! section 2 Binary Form. spirv-val coverage remains in family_validation.rs.

use shadowbox::ir::ops;
use shadowbox::spirv::Op;
use shadowbox::{
    decoder::{decode_stream, decode_word},
    emitter::emit_unoptimized_with_header as emit_with_header,
    headers,
    ir::{FloatFlags, Instruction, InstructionKind, Register, Source},
};
use std::collections::HashMap;

fn decode(raw: u64) -> Instruction {
    decode_word([raw as u32, (raw >> 32) as u32], 40)
}

// This evaluator observes internal R1 and P0..P7, even when the shader
// exports none of them. Keep those observations on the unoptimised emitter.
// Optimiser tests compare externally observable output stores instead.
fn module(raw: u64, inputs: [u32; 4]) -> Vec<u32> {
    let c = if matches!(raw >> 56, 0x5c | 0x5d) {
        0
    } else {
        4 << 39
    };
    exact_module(raw | (7 << 16) | 1 | (2 << 8) | (3 << 20) | c, inputs)
}

fn exact_module(raw: u64, inputs: [u32; 4]) -> Vec<u32> {
    let mut instructions = inputs
        .into_iter()
        .enumerate()
        .map(|(i, value)| Instruction {
            address: (i as u32 + 1) * 8,
            predicate: None,
            kind: InstructionKind::Mov(ops::Mov {
                quad_lanes: 0xf,
                dst: Register(i as u8 + 1),
                src: Source::Immediate(value),
            }),
        })
        .collect::<Vec<_>>();
    instructions.push(decode(raw));
    instructions.push(Instruction {
        address: 48,
        predicate: None,
        kind: InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
    });
    let header = headers::parse(&[0; 32]).unwrap();
    emit_with_header(&instructions, Some(&header))
        .unwrap()
        .words
}

// Independent binary16 reference, including subnormal conversion and RTE.
fn unpack(bits: u16) -> u32 {
    let negative = bits & 0x8000 != 0;
    let exp = (bits >> 10) & 31;
    let fraction = bits & 1023;
    let f = match exp {
        0 => f32::from(fraction) * 2.0_f32.powi(-24),
        31 => {
            if fraction == 0 {
                f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => (1.0 + f32::from(fraction) / 1024.0) * 2.0_f32.powi(i32::from(exp) - 15),
    };
    (if negative { -f } else { f }).to_bits()
}

fn pack(bits: u32) -> u16 {
    let sign = ((bits >> 16) & 0x8000) as u16;
    let f = f32::from_bits(bits).abs();
    if f.is_nan() {
        return sign | 0x7e00;
    }
    if f >= 65520.0 {
        return sign | 0x7c00;
    }
    if f < 2.0_f32.powi(-14) {
        return sign | (f * 2.0_f32.powi(24)).round_ties_even() as u16;
    }
    let exponent = ((bits >> 23) & 255) as i32 - 127;
    let significand = (f * 2.0_f32.powi(10 - exponent)).round_ties_even() as u16;
    sign | ((((exponent + 15) as u16) << 10) + (significand - 1024))
}

fn run(words: &[u32], lane: u32) -> (u32, [u32; 8]) {
    run_with_fragment_inputs(words, lane, None, None)
}

fn run_with_fragment_inputs(
    words: &[u32],
    lane: u32,
    inputs: Option<([u32; 4], [u32; 4])>,
    lod: Option<([f32; 2], &[u32])>,
) -> (u32, [u32; 8]) {
    run_with_context(words, lane, inputs, None, lod)
}

fn run_with_uniform(words: &[u32], lane: u32, uniform: Option<[u32; 4]>) -> (u32, [u32; 8]) {
    run_with_context(words, lane, None, uniform, None)
}

fn run_with_context(
    words: &[u32],
    lane: u32,
    inputs: Option<([u32; 4], [u32; 4])>,
    uniform: Option<[u32; 4]>,
    lod: Option<([f32; 2], &[u32])>,
) -> (u32, [u32; 8]) {
    let mut queries = 0;
    let mut values: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut labels = HashMap::new();
    let mut uint_type = 0;
    let mut register_pointer = 0;
    let mut register_vars = Vec::new();
    let mut predicate_pointer = 0;
    let mut predicate_vars = Vec::new();
    let mut lane_variable = 0;
    let mut attribute_variable = 0;
    let mut frag_coord_variable = 0;
    let mut uniform_variable = 0;
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        let op = Op::of(words[at]);
        let a = &words[at + 1..at + count];
        match op {
            // SPIR-V decorations: Location=30, BuiltIn=11, FragCoord=15.
            Op::Decorate if a[1..] == [30, 0] => attribute_variable = a[0],
            Op::Decorate if a[1..] == [11, 15] => frag_coord_variable = a[0],
            Op::TypeBool => predicate_pointer = a[0],
            Op::TypeInt if a[1..] == [32, 0] => uint_type = a[0],
            Op::TypePointer if a[1] == 7 && a[2] == uint_type => register_pointer = a[0],
            Op::TypePointer if a[1] == 7 && a[2] == predicate_pointer => predicate_pointer = a[0],
            Op::Variable if a[2] == 2 => uniform_variable = a[1],
            Op::Variable if a[0] == register_pointer => register_vars.push(a[1]),
            Op::Variable if a[0] == predicate_pointer => predicate_vars.push(a[1]),
            Op::Decorate if a[1..] == [11, 41] => lane_variable = a[0],
            Op::Label => {
                labels.insert(a[0], at);
            }
            _ => {}
        }
        at += count;
    }
    let mut at = 5;
    for _ in 0..10000 {
        let count = (words[at] >> 16) as usize;
        let op = Op::of(words[at]);
        let a = &words[at + 1..at + count];
        let get = |index: usize| values[&a[index]][0];
        let scalar = match op {
            Op::ConstantTrue => Some(1),
            Op::ConstantFalse => Some(0),
            Op::Constant => Some(a[2]),
            Op::Variable => {
                let value = if let Some((attribute, coord)) = inputs {
                    if a[1] == attribute_variable {
                        attribute.to_vec()
                    } else if a[1] == frag_coord_variable {
                        coord.to_vec()
                    } else {
                        vec![if a[1] == lane_variable { lane } else { 0 }]
                    }
                } else {
                    vec![if a[1] == lane_variable { lane } else { 0 }]
                };
                values.insert(a[1], value);
                None
            }
            Op::Load => {
                let v = values[&a[2]].clone();
                values.insert(a[1], v);
                None
            }
            Op::Store => {
                let v = values[&a[1]].clone();
                values.insert(a[0], v);
                None
            }
            Op::AccessChain => {
                assert_eq!(a[2], uniform_variable);
                assert_eq!((get(3), get(4)), (0, 0));
                values.insert(a[1], uniform.expect("missing uniform contents").to_vec());
                None
            }
            Op::CompositeConstruct => {
                let v = a[2..].iter().map(|id| values[id][0]).collect();
                values.insert(a[1], v);
                None
            }
            Op::VectorShuffle => {
                let mut source = values[&a[2]].clone();
                source.extend_from_slice(&values[&a[3]]);
                values.insert(a[1], a[4..].iter().map(|i| source[*i as usize]).collect());
                None
            }
            Op::CompositeExtract => Some(values[&a[2]][a[3] as usize]),
            // Mock the host image result; this tests Maxwell's result
            // conversion/packing and coordinate plan, not GPU LOD math.
            Op::SampledImage => Some(0),
            Op::ImageQueryLod => {
                let (pair, expected_coords) = lod.expect("unexpected image LOD query");
                assert_eq!(values[&a[3]], expected_coords);
                queries += 1;
                values.insert(a[1], pair.map(f32::to_bits).to_vec());
                None
            }
            Op::ConvertFToS => Some((f32::from_bits(get(2)) as i32) as u32),
            Op::Bitcast => {
                values.insert(a[1], values[&a[2]].clone());
                None
            }
            Op::FNegate => {
                values.insert(
                    a[1],
                    values[&a[2]].iter().map(|v| v ^ 0x8000_0000).collect(),
                );
                None
            }
            // LogicalOr166 and UGreaterThan172 operate component-wise on
            // vectors too (Mesa vtn_alu.c:297,373,779-783,1033;
            // opcode IDs: spirv.core.grammar.json:1939-1948,2011-2020).
            // The half saturation lowering now uses both packed lanes.
            Op::IAdd
            | Op::LogicalOr
            | Op::LogicalAnd
            | Op::IEqual
            | Op::UGreaterThan
            | Op::ULessThan
            | Op::ShiftRightLogical
            | Op::BitwiseOr
            | Op::BitwiseXor
            | Op::BitwiseAnd => {
                let v = values[&a[2]]
                    .iter()
                    .zip(&values[&a[3]])
                    .map(|(x, y)| match op {
                        Op::IAdd => x.wrapping_add(*y),
                        Op::LogicalOr => u32::from(*x != 0 || *y != 0),
                        Op::LogicalAnd => u32::from(*x != 0 && *y != 0),
                        Op::IEqual => u32::from(x == y),
                        Op::ULessThan => u32::from(x < y),
                        Op::BitwiseXor => x ^ y,
                        Op::UGreaterThan => u32::from(x > y),
                        Op::ShiftRightLogical => x >> y,
                        Op::BitwiseOr => x | y,
                        Op::BitwiseAnd => x & y,
                        _ => unreachable!(),
                    })
                    .collect();
                values.insert(a[1], v);
                None
            }
            Op::FAdd | Op::FMul => {
                let v = values[&a[2]]
                    .iter()
                    .zip(&values[&a[3]])
                    .map(|(x, y)| {
                        let (x, y) = (f32::from_bits(*x), f32::from_bits(*y));
                        (if op == Op::FAdd { x + y } else { x * y }).to_bits()
                    })
                    .collect();
                values.insert(a[1], v);
                None
            }
            Op::ISub => Some(get(2).wrapping_sub(get(3))),
            Op::FDiv => Some((f32::from_bits(get(2)) / f32::from_bits(get(3))).to_bits()),

            Op::IsNan => {
                values.insert(
                    a[1],
                    values[&a[2]]
                        .iter()
                        .map(|v| u32::from(f32::from_bits(*v).is_nan()))
                        .collect(),
                );
                None
            }
            Op::LogicalNotEqual => Some(u32::from((get(2) != 0) != (get(3) != 0))),
            Op::LogicalNot => {
                values.insert(
                    a[1],
                    values[&a[2]].iter().map(|x| u32::from(*x == 0)).collect(),
                );
                None
            }
            Op::Select => {
                let cond = &values[&a[2]];
                let v = if cond.len() == 1 {
                    values[&a[if cond[0] != 0 { 3 } else { 4 }]].clone()
                } else {
                    cond.iter()
                        .enumerate()
                        .map(|(i, c)| values[&a[if *c != 0 { 3 } else { 4 }]][i])
                        .collect()
                };
                values.insert(a[1], v);
                None
            }
            Op::INotEqual => Some(u32::from(get(2) != get(3))),
            Op::SGreaterThan => Some(u32::from(get(2) as i32 > get(3) as i32)),
            Op::UGreaterThanEqual => Some(u32::from(get(2) >= get(3))),
            Op::SGreaterThanEqual => Some(u32::from(get(2) as i32 >= get(3) as i32)),
            Op::SLessThan => Some(u32::from((get(2) as i32) < get(3) as i32)),
            Op::ULessThanEqual => Some(u32::from(get(2) <= get(3))),
            Op::SLessThanEqual => Some(u32::from(get(2) as i32 <= get(3) as i32)),
            Op::FOrdEqual
            | Op::FUnordEqual
            | Op::FOrdNotEqual
            | Op::FUnordNotEqual
            | Op::FOrdLessThan
            | Op::FUnordLessThan
            | Op::FOrdGreaterThan
            | Op::FUnordGreaterThan
            | Op::FOrdLessThanEqual
            | Op::FUnordLessThanEqual
            | Op::FOrdGreaterThanEqual
            | Op::FUnordGreaterThanEqual => {
                let v = values[&a[2]]
                    .iter()
                    .zip(&values[&a[3]])
                    .map(|(x, y)| {
                        let (x, y) = (f32::from_bits(*x), f32::from_bits(*y));
                        let nan = x.is_nan() || y.is_nan();
                        let comparison = match (op.number() - 180) / 2 {
                            0 => x == y,
                            1 => x != y,
                            2 => x < y,
                            3 => x > y,
                            4 => x <= y,
                            5 => x >= y,
                            _ => unreachable!(),
                        };
                        u32::from(if op.number() & 1 != 0 {
                            nan || comparison
                        } else {
                            !nan && comparison
                        })
                    })
                    .collect();
                values.insert(a[1], v);
                None
            }
            Op::ShiftLeftLogical => Some(get(2) << get(3)),

            Op::Not => Some(!get(2)),
            Op::BitFieldInsert => {
                let mask = ((1u32 << get(5)) - 1) << get(4);
                Some((get(2) & !mask) | ((get(3) << get(4)) & mask))
            }
            Op::BitFieldSExtract | Op::BitFieldUExtract => {
                let value = get(2) >> get(3);
                let width = get(4);
                Some(if op == Op::BitFieldSExtract {
                    ((value << (32 - width)) as i32 >> (32 - width)) as u32
                } else {
                    value & ((1 << width) - 1)
                })
            }
            // SPIRV-Headers include/spirv/unified1/spirv.h:1969.
            Op::BitCount => Some(get(2).count_ones()),
            Op::ExtInst => match a[3] {
                2 => {
                    values.insert(
                        a[1],
                        values[&a[4]]
                            .iter()
                            .map(|v| f32::from_bits(*v).round_ties_even().to_bits())
                            .collect(),
                    );
                    None
                }
                4 => {
                    values.insert(
                        a[1],
                        values[&a[4]].iter().map(|v| v & 0x7fff_ffff).collect(),
                    );
                    None
                }
                43 => {
                    let v = values[&a[4]]
                        .iter()
                        .zip(&values[&a[5]])
                        .zip(&values[&a[6]])
                        .map(|((x, low), high)| {
                            f32::from_bits(*x)
                                .clamp(f32::from_bits(*low), f32::from_bits(*high))
                                .to_bits()
                        })
                        .collect();
                    values.insert(a[1], v);
                    None
                }
                50 => {
                    let v = values[&a[4]]
                        .iter()
                        .zip(&values[&a[5]])
                        .zip(&values[&a[6]])
                        .map(|((x, y), z)| {
                            f32::from_bits(*x)
                                .mul_add(f32::from_bits(*y), f32::from_bits(*z))
                                .to_bits()
                        })
                        .collect();
                    values.insert(a[1], v);
                    None
                }
                13 => Some(f32::from_bits(get(4)).sin().to_bits()),
                14 => Some(f32::from_bits(get(4)).cos().to_bits()),
                29 => Some(f32::from_bits(get(4)).exp2().to_bits()),
                30 => Some(f32::from_bits(get(4)).log2().to_bits()),
                31 => Some(f32::from_bits(get(4)).sqrt().to_bits()),
                32 => Some((1.0 / f32::from_bits(get(4)).sqrt()).to_bits()),
                81 => Some(
                    f32::from_bits(get(4))
                        .max(f32::from_bits(get(5)))
                        .min(f32::from_bits(get(6)))
                        .to_bits(),
                ),
                58 => {
                    let v = &values[&a[4]];
                    Some(u32::from(pack(v[0])) | (u32::from(pack(v[1])) << 16))
                }
                62 => {
                    values.insert(
                        a[1],
                        vec![unpack(get(4) as u16), unpack((get(4) >> 16) as u16)],
                    );
                    None
                }
                other => panic!("unimplemented extended opcode {other}"),
            },
            Op::GroupNonUniformShuffle => Some(0x100 + get(4)), // Project-owned lane contents.
            Op::Branch => {
                at = labels[&a[0]];
                continue;
            }
            Op::BranchConditional => {
                at = labels[&a[if values[&a[0]][0] != 0 { 1 } else { 2 }]];
                continue;
            }
            Op::Switch => {
                let selector = values[&a[0]][0];
                let label = a[2..]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .find(|pair| pair[0] == selector)
                    .map_or(a[1], |pair| pair[1]);
                at = labels[&label];
                continue;
            }
            Op::Return => {
                if lod.is_some() {
                    assert_eq!(queries, 1);
                }
                let predicates = std::array::from_fn(|p| values[&predicate_vars[p]][0]);
                return (values[&register_vars[1]][0], predicates);
            }
            Op::Extension
            | Op::ExtInstImport
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
            | Op::TypeEvent
            | Op::TypeDeviceEvent
            | Op::TypeReserveId
            | Op::TypeQueue
            | Op::TypePipe
            | Op::TypeForwardPointer
            | Op::Function
            | Op::FunctionEnd
            | Op::Decorate
            | Op::MemberDecorate
            | Op::LoopMerge
            | Op::SelectionMerge
            | Op::Label => None,
            other => panic!("unimplemented opcode {}", other.number()),
        };
        if let Some(value) = scalar {
            values.insert(a[1], vec![value]);
        }
        at += count;
    }
    panic!("SPIR-V dispatch did not terminate")
}

#[test]
fn fcmp_ftz_flushes_only_the_test_word() {
    // FTZ47 and compare48..51: Mesa nv50_ir_emit_gm107.cpp:1568-1569;
    // envytools gm107.c:2025,1411-1427. Select raw A/B: Ryujinx
    // InstEmitFloatComparison.cs:369-375. Include NaNs and normal boundaries.
    let a = 0x0000_0123; // Selected subnormal payload must not be flushed.
    let b = 0x8000_0456;
    for test in [
        1,
        0x8000_0001,
        0x007f_ffff,
        0x807f_ffff,
        0x0080_0000,
        0x8080_0000,
        0,
        0x8000_0000,
        0x7fc1_2345,
        0x7f80_0000,
    ] {
        let flushed = if test & 0x7fff_ffff < 0x0080_0000 {
            test & 0x8000_0000
        } else {
            test
        };
        for comparison in 0..16u64 {
            let raw = (0x5ba0u64 << 48) | comparison << 48;
            let expected = run(&module(raw, [0, a, b, flushed]), 0).0;
            assert_eq!(
                run(&module(raw | 1 << 47, [0, a, b, test]), 0).0,
                expected,
                "compare={comparison} test={test:08x}"
            );
            assert!([a, b].contains(&expected));
        }
    }
    // Prefix 5bad is unordered !=. A negative subnormal differs from zero
    // without FTZ and equals signed zero with FTZ.
    assert_eq!(run(&module(0x5bad << 48, [0, a, b, 0x8000_0001]), 0).0, a);
    assert_eq!(
        run(&module(0x5bad << 48 | 1 << 47, [0, a, b, 0x8000_0001]), 0).0,
        b
    );
}

#[test]
fn fcmp_selects_raw_bits_for_all_comparisons() {
    let a = 0x7fc1_2345;
    let b = 0x8000_0000;
    for value in [-1.0_f32, -0.0, 1.0, f32::NAN] {
        let lt = value < 0.0;
        let eq = value == 0.0;
        let gt = value > 0.0;
        let nan = value.is_nan();
        let expected = [
            false,
            lt,
            eq,
            lt || eq,
            gt,
            !nan && !eq,
            gt || eq,
            !nan,
            nan,
            nan || lt,
            nan || eq,
            nan || lt || eq,
            nan || gt,
            nan || !eq,
            nan || gt || eq,
            true,
        ];
        for (code, condition) in expected.into_iter().enumerate() {
            let words = module(
                (0x5ba0u64 << 48) | ((code as u64) << 48),
                [0, a, b, value.to_bits()],
            );
            assert_eq!(
                run(&words, 0).0,
                if condition { a } else { b },
                "comparison {code} test {value}"
            );
        }
    }
}

#[test]
fn fmul_scale_precedes_multiply_and_ftz_preserves_zero_sign() {
    for (scale, expected) in [
        (0, 6.0_f32),
        (1, 3.0),
        (2, 1.5),
        (3, 0.75),
        (4, 48.0),
        (5, 24.0),
        (6, 12.0),
    ] {
        let words = module(
            (0x5c68u64 << 48) | (scale << 41),
            [0, 2.0f32.to_bits(), 3.0f32.to_bits(), 0],
        );
        assert_eq!(run(&words, 0).0, expected.to_bits());
    }
    // Pre-scaling A overflows even though A*B alone would not.
    let words = module(
        (0x5c68u64 << 48) | (4 << 41),
        [0, f32::MAX.to_bits(), 0.125f32.to_bits(), 0],
    );
    assert_eq!(run(&words, 0).0, f32::INFINITY.to_bits());
    let words = module(
        (0x5c68u64 << 48) | (1 << 44),
        [0, 0x8000_0001, 2.0f32.to_bits(), 0],
    );
    assert_eq!(run(&words, 0).0, (-0.0f32).to_bits());
    let words = module(
        (0x5c68u64 << 48) | (1 << 44),
        [0, 0x8080_0000, 0.5f32.to_bits(), 0],
    );
    assert_eq!(run(&words, 0).0, (-0.0f32).to_bits());
}

#[test]
fn half_add_swizzles_output_merge_and_modifiers_execute() {
    // A = [1, -2], B = [0.5, 4], old destination = [8, 16].
    for (swizzle, expected) in [(0, 0x4000_3e00), (2, 0x4500_3e00), (3, 0x4000_be00)] {
        let words = module(
            (0x5d10u64 << 48) | (swizzle << 47),
            [0x4c00_4800, 0xc000_3c00, 0x4400_3800, 0],
        );
        assert_eq!(run(&words, 0).0, expected);
    }
    let words = module(
        (0x5d10u64 << 48) | (1 << 47) | (1 << 28) | (1 << 49),
        [0, 1.5f32.to_bits(), 0.25f32.to_bits(), 0],
    );
    assert_eq!(run(&words, 0).0, 1.75f32.to_bits());
    for (output, expected) in [(2, 0x4c00_3e00), (3, 0x4000_4800)] {
        let words = module(
            (0x5d10u64 << 48) | (output << 49),
            [0x4c00_4800, 0xc000_3c00, 0x4400_3800, 0],
        );
        assert_eq!(run(&words, 0).0, expected);
    }
    // abs(A), negate(B), saturation => [0.5, 0].
    let words = module(
        (0x5d10u64 << 48) | (1 << 44) | (1 << 31) | (1 << 32),
        [0, 0xc000_3c00, 0x4400_3800, 0],
    );
    assert_eq!(run(&words, 0).0, 0x0000_3800);
    // Half subnormal + half subnormal and a halfway rounding tie.
    let words = module(0x5d10u64 << 48, [0, 0x3c00_0001, 0x1000_0001, 0]);
    assert_eq!(run(&words, 0).0, 0x3c00_0002);
}

// Compressed half fields: envytools gm107.c:1977; Ryujinx
// InstEmitHelper.cs:105-114. Project-owned inputs, no captured instructions.
fn half_add_immediate(low: u64, high: u64) -> u64 {
    (0x7a00 << 48)
        | (low << 20)
        | ((high & 511) << 30)
        | ((high >> 9) << 56)
        | (7 << 16)
        | (2 << 8)
        | 1
}

#[test]
fn half_add_immediate_all_compressed_values_and_independent_signs_execute() {
    for value in 0..1024 {
        let opposite = value ^ 512;
        let raw = half_add_immediate(value, opposite);
        let words = exact_module(raw, [0, 0x4000_3c00, 0, 0]);
        let low = 1.0 + f32::from_bits(unpack((value << 6) as u16));
        let high = 2.0 + f32::from_bits(unpack((opposite << 6) as u16));
        let expected = u32::from(pack(low.to_bits())) | (u32::from(pack(high.to_bits())) << 16);
        assert_eq!(run(&words, 0).0, expected, "compressed half {value}");
        // Finite immediates need only A's unpack, never B's unpack/extracts.
        if value & 496 != 496 {
            let mut at = 5;
            let mut unpacks = 0;
            while at < words.len() {
                if Op::of(words[at]) == Op::ExtInst && words[at + 4] == 62 {
                    unpacks += 1;
                }
                at += (words[at] >> 16) as usize;
            }
            assert_eq!(unpacks, 1);
        }
    }
}

#[test]
fn half_add_immediate_all_swizzles_outputs_modifiers_and_guard_execute() {
    let old = 0x4c00_4800;
    for swizzle in 0..4 {
        let input = if swizzle == 1 {
            (-1.5f32).to_bits()
        } else {
            0xc000_3c00
        };
        let a = match swizzle {
            0 => [1.0f32, -2.0],
            1 => [-1.5, -1.5],
            2 => [1.0, 1.0],
            _ => [-2.0, -2.0],
        };
        for output in 0..4 {
            for absolute in 0..2 {
                for negate in 0..2 {
                    for saturate in 0..2 {
                        // B = [0.5, -4], including its independent high sign.
                        let raw = half_add_immediate(0x3800 >> 6, 0xc400 >> 6)
                            | (swizzle << 47)
                            | (output << 49)
                            | (absolute << 44)
                            | (negate << 43)
                            | (saturate << 52);
                        let mut result = a;
                        for (part, b) in result.iter_mut().zip([0.5, -4.0]) {
                            if absolute != 0 {
                                *part = part.abs();
                            }
                            if negate != 0 {
                                *part = -*part;
                            }
                            *part += b;
                            if saturate != 0 {
                                *part = part.clamp(0.0, 1.0);
                            }
                        }
                        let packed = u32::from(pack(result[0].to_bits()))
                            | (u32::from(pack(result[1].to_bits())) << 16);
                        let expected = match output {
                            0 => packed,
                            1 => result[0].to_bits(),
                            2 => (old & 0xffff_0000) | (packed & 0xffff),
                            _ => (old & 0xffff) | (packed & 0xffff_0000),
                        };
                        let words = exact_module(raw, [old, input, 0, 0]);
                        assert_eq!(run(&words, 0).0, expected, "{raw:016x}");
                        let words = exact_module(raw | (1 << 19), [old, input, 0, 0]);
                        assert_eq!(run(&words, 0).0, old);
                    }
                }
            }
        }
    }
}

#[test]
fn video_select_saturate_and_all_post_operations_execute() {
    for (post, expected) in [
        (0, 0x0003_bbaa),
        (1, 0xddcc_0003),
        (2, 0xddcc_bb03),
        (3, 0xdd03_bbaa),
        (4, 0xddcc_bbad),
        (5, 3),
        (6, 0xddcc_bbaa),
        (7, 3),
    ] {
        let raw = (0x3a00u64 << 48) | (6 << 36) | (6 << 28) | (1 << 50) | (post << 51);
        assert_eq!(run(&module(raw, [0, 7, 3, 0xddcc_bbaa]), 0).0, expected);
    }
    // signed A byte1 = -2, unsigned B half1 = 1, unsigned saturation => 0.
    let raw =
        (0x3a00u64 << 48) | (1 << 36) | (1 << 48) | (5 << 28) | (1 << 50) | (1 << 55) | (7 << 51);
    assert_eq!(run(&module(raw, [0, 0x0000_fe00, 0x0001_0000, 0]), 0).0, 0);
    // Mixed signedness follows the Ryujinx high unsigned source path.
    let raw = (0x3a00u64 << 48) | (6 << 36) | (1 << 48) | (6 << 28) | (1 << 50) | (7 << 51);
    assert_eq!(
        run(&module(raw, [0, (-1i32) as u32, 0x8000_0000, 0]), 0).0,
        0x8000_0000
    );
}

#[test]
fn shuffle_segments_validity_and_warp_base_execute() {
    // Register B/C forms, predicate output P0.
    for (mode, index, control, lane, expected, valid) in [
        (0, 5, 0x1807, 10, 0x10d, 1), // idx preserves segment bits
        (0, 7, 3, 10, 0x10a, 0),      // invalid selects original input
        (1, 3, 0x1800, 10, 0x10a, 0), // up crosses segment start
        (1, 2, 0x1800, 10, 0x108, 1),
        (2, 3, 0x181f, 10, 0x10d, 1),
        (2, 3, 0x1803, 10, 0x10a, 0),
        (3, 1, 0x181f, 42, 0x12b, 1), // second Maxwell warp
    ] {
        let raw = (0xef10u64 << 48) | (mode << 30);
        let words = module(raw, [0, 0x100 + lane, index, control]);
        let (result, predicates) = run(&words, lane);
        assert_eq!((result, predicates[0]), (expected, valid));
    }
}

#[test]
fn fmul_directed_rounding_and_scaled_ftz_translate() {
    // Mesa sm50.rs:695-700 encodes rounding and FTZ. Float64 produces exact
    // directed results. Hosts without Float64 use the nearest-even fallback.
    for raw in [
        (0x5c68u64 << 48) | (1 << 39),
        (0x5c68u64 << 48) | (3 << 39),
        (0x5c68u64 << 48) | (6 << 41) | (1 << 44),
    ] {
        assert!(
            matches!(decode(raw).kind, InstructionKind::FMul(ops::FMul { .. })),
            "{raw:016x}"
        );
    }
}

#[test]
fn unsupported_modes_remain_unknown_and_real_branches_stay_branches() {
    let fmul_cc = (0x5c68u64 << 48) | (1 << 47);
    assert!(matches!(
        decode(fmul_cc).kind,
        InstructionKind::FMul(ops::FMul {
            flags: FloatFlags::Written,
            ..
        })
    ));

    for raw in [
        (0x5c68u64 << 48) | (7 << 41), // reserved scale
        (0x5d08u64 << 48) | (3 << 39), // reserved half flush mode
        (0x3a00u64 << 48) | (1 << 47), // video CC
        (0x3a00u64 << 48) | (7 << 36), // reserved A select
        0xef18u64 << 48,               // outside SHFL table mask
    ] {
        assert!(
            matches!(
                decode(raw).kind,
                InstructionKind::Unknown(ops::Unknown { .. })
            ),
            "{raw:016x}"
        );
    }
    let raw = (0xe240u64 << 48) | (7 << 16) | 0xf; // unconditional jump to next slot
    let exit = (0xe300u64 << 48) | (7 << 16) | 0xf;
    let words = [
        0,
        0,
        raw as u32,
        (raw >> 32) as u32,
        exit as u32,
        (exit >> 32) as u32,
    ];
    assert!(matches!(
        decode_stream(shadowbox::QuietRun, &words)[0].kind,
        InstructionKind::Branch(ops::Branch { .. })
    ));
    assert_eq!(
        include_str!("../src/decoder/families.rs")
            .matches("x.guarded(InstructionKind::Nop(ops::Nop))")
            .count(),
        2
    );
}

#[test]
fn source_forms_and_modifiers_decode_without_losing_fields() {
    use shadowbox::ir::{FloatCompare, Relation};
    // envytools gm107.c:2025,2044,2096,2143 and Mesa sm50.rs:361-368:
    // cbuf offset is in words at 20..33, bank at 34..38.
    let operands = 1 | (2 << 8) | (7 << 20) | (3 << 34) | (4 << 39);
    for (opcode, expected_false, expected_test) in [
        (
            0x5ba0,
            Source::Register(Register(7)),
            Source::Register(Register(4)),
        ),
        (
            0x4ba0,
            Source::ConstantBuffer {
                binding: 3,
                offset: 28,
            },
            Source::Register(Register(4)),
        ),
        (
            0x53a0,
            Source::Register(Register(4)),
            Source::ConstantBuffer {
                binding: 3,
                offset: 28,
            },
        ),
    ] {
        let instruction = decode((opcode << 48) | (2 << 48) | operands);
        assert!(matches!(instruction.kind,
            InstructionKind::FCmp(ops::FCmp { dst: Register(1), when_true: Register(2),
                when_false, test, compare: FloatCompare::Ordered(Relation::Eq), ftz: false })
            if when_false == expected_false && test == expected_test));
    }
    let instruction = decode((0x36a0u64 << 48) | (1 << 56) | (0x12345 << 20));
    assert!(matches!(
        instruction.kind,
        InstructionKind::FCmp(ops::FCmp {
            when_false: Source::Immediate(0x9234_5000),
            ..
        })
    ));
    // Every HADD2 swizzle/output combination and every sign/saturation field.
    // envytools gm107.c:1133-1153,1987.
    for a in 0..4 {
        for b in 0..4 {
            for output in 0..4 {
                let raw = (0x5d10u64 << 48)
                    | (a << 47)
                    | (b << 28)
                    | (output << 49)
                    | (1 << 43)
                    | (1 << 44)
                    | (1 << 30)
                    | (1 << 31)
                    | (1 << 32);
                assert!(
                    matches!(decode(raw).kind, InstructionKind::HalfArithmetic(ops::HalfArithmetic {
                    swizzle, output: actual, abs: [true, true, false], negate: [true, true, false], saturate: true, ..
                }) if swizzle == [a as u8, b as u8, 0] && actual == output as u8)
                );
            }
        }
    }
    // Immediate SHFL lane/control flags and PT predicate destination.
    // Mesa sm50.rs:2046-2067.
    let raw = (0xef17u64 << 48) | (1 << 28) | (1 << 29) | (9 << 20) | (0x181f << 34);
    assert!(matches!(
        decode(raw).kind,
        InstructionKind::Shuffle(ops::Shuffle {
            lane: Source::Immediate(9),
            control: Source::Immediate(0x181f),
            in_bounds: 7,
            ..
        })
    ));
    // VMNMX signed immediate must be sign extended, with no byte extraction.
    let raw = (0x3a00u64 << 48) | (1 << 49) | (0xffff << 20);
    assert!(matches!(
        decode(raw).kind,
        InstructionKind::VideoMinMax(ops::VideoMinMax {
            rhs: Source::Immediate(u32::MAX),
            select: [0, 14],
            ..
        })
    ));
}

#[test]
fn inverted_pt_suppresses_float_video_and_shuffle_writes() {
    // envytools gm107.c:363-364,371,408-411; Ryujinx
    // Instructions/InstEmitHelper.cs:203-212. Inversion applies to PT too.
    for raw in [
        0x5ba2u64 << 48,
        0x5c68u64 << 48,
        0x5d10u64 << 48,
        0x3a00u64 << 48 | 6 << 36 | 6 << 28 | 1 << 50 | 7 << 51,
        0xef10u64 << 48,
    ] {
        let old = 0x1234_5678;
        let words = module(raw | 1 << 19, [old, 0, 0, 0]);
        assert_eq!(run(&words, 0).0, old, "inverted PT at {raw:016x}");
    }
}
// MUFU fields: envytools envydis/gm107.c:1690-1700,2062; Mesa
// src/nouveau/compiler/nak/sm50.rs:749-768. Semantics:
// Ryujinx Instructions/InstEmitMultifunction.cs:32-83.
fn mufu(function: u64, modifiers: u64) -> u64 {
    (0x5080_u64 << 48)
        | (function << 20)
        | (7 << 16)
        | (2 << 8)
        | 1
        | ((modifiers & 1) << 46)
        | (((modifiers >> 1) & 1) << 48)
        | (((modifiers >> 2) & 1) << 50)
}

#[test]
fn mufu_decode_all_fields_and_reserved_encodings() {
    for modifiers in 0..8 {
        for function in 0..16 {
            let decoded = decode(mufu(function, modifiers)).kind;
            if matches!(function, 0..=8) {
                assert!(matches!(decoded, InstructionKind::MuFu(ops::MuFu {
                    dst: Register(1), src: Register(2), op, abs, negate, saturate,
                }) if u64::from(op as u8) == function
                    && abs == (modifiers & 1 != 0)
                    && negate == (modifiers & 2 != 0)
                    && saturate == (modifiers & 4 != 0)));
            } else {
                assert!(matches!(
                    decoded,
                    InstructionKind::Unknown(ops::Unknown { .. })
                ));
            }
        }
    }
    // These bits have no field in either permitted encoder/table.
    for bit in (24..46).chain([47, 49]) {
        assert!(
            matches!(
                decode(mufu(4, 0) | 1 << bit).kind,
                InstructionKind::Unknown(ops::Unknown { .. })
            ),
            "bit {bit}"
        );
    }
}

fn mufu_module(raw: u64, input: u32) -> Vec<u32> {
    let header = headers::parse(&[0; 32]).unwrap();
    emit_with_header(
        &[
            Instruction {
                address: 8,
                predicate: None,
                kind: InstructionKind::Mov(ops::Mov {
                    quad_lanes: 0xf,
                    dst: Register(1),
                    src: Source::Immediate(0x1234),
                }),
            },
            Instruction {
                address: 16,
                predicate: None,
                kind: InstructionKind::Mov(ops::Mov {
                    quad_lanes: 0xf,
                    dst: Register(2),
                    src: Source::Immediate(input),
                }),
            },
            decode(raw),
            Instruction {
                address: 48,
                predicate: None,
                kind: InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
            },
        ],
        Some(&header),
    )
    .unwrap()
    .words
}

#[test]
#[allow(clippy::manual_clamp)]
fn mufu_results_modifiers_exceptional_inputs_and_radian_range() {
    let inputs = [
        0.0,
        -0.0,
        0.25,
        -0.25,
        1.0,
        -1.0,
        2.0,
        -2.0,
        4.0,
        -4.0,
        std::f32::consts::FRAC_PI_2,
        -std::f32::consts::PI,
        std::f32::consts::TAU,
        10.0,
        -10.0,
        100.0,
        -100.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    for function in [0, 1, 2, 3, 4, 5, 8] {
        for modifiers in 0..8 {
            for input in inputs {
                let mut x = if modifiers & 1 != 0 {
                    input.abs()
                } else {
                    input
                };
                if modifiers & 2 != 0 {
                    x = -x;
                }
                let expected = match function {
                    0 => x.cos(),
                    1 => x.sin(),
                    2 => x.exp2(),
                    3 => x.log2(),
                    4 => 1.0 / x,
                    5 => 1.0 / x.sqrt(),
                    8 => x.sqrt(),
                    _ => unreachable!(),
                };
                let expected = if modifiers & 4 != 0 {
                    expected.max(0.0).min(1.0)
                } else {
                    expected
                };
                let actual = run(&mufu_module(mufu(function, modifiers), input.to_bits()), 0).0;
                assert!(if expected.is_nan() { f32::from_bits(actual).is_nan() }
                    else { actual == expected.to_bits() },
                    "function={function} modifiers={modifiers} input={input} actual={actual:x} expected={:x}", expected.to_bits());
            }
        }
    }
    // P0 starts false; !PT is also false. Neither guard may write R1.
    for raw in [mufu(4, 4) & !(7 << 16), mufu(4, 4) | 1 << 19] {
        assert_eq!(run(&mufu_module(raw, 0), 0).0, 0x1234);
    }
    // RZ reads zero, writes are discarded, and destination/source aliasing
    // must capture the source before writing the result.
    assert_eq!(
        run(&mufu_module((mufu(4, 0) & !(255 << 8)) | 255 << 8, 0), 0).0,
        f32::INFINITY.to_bits()
    );
    assert_eq!(run(&mufu_module((mufu(4, 0) & !255) | 255, 0), 0).0, 0x1234);
    assert_eq!(
        run(&mufu_module((mufu(4, 0) & !(255 << 8)) | 1 << 8, 0), 0).0,
        (1.0 / f32::from_bits(0x1234)).to_bits()
    );
}

// Ryujinx InstEmitIntegerComparison.cs:117-131: only C participates in
// comparison; A/B remain raw bits. This CPU execution detects swapped
// operands, incorrect signedness, and wrong compare-to-zero behaviour.
#[test]
fn icmp_cpu_selects_raw_bits_for_every_compare_and_signedness() {
    let a = 0x7fc1_2345;
    let b = 0x8000_0000;
    for value in [0u32, 1, 0x7fff_ffff, 0x8000_0000, u32::MAX] {
        for signed in [false, true] {
            let test = if signed {
                i64::from(value as i32)
            } else {
                i64::from(value)
            };
            let expected = [
                false,
                test < 0,
                test == 0,
                test <= 0,
                test > 0,
                test != 0,
                test >= 0,
                true,
            ];
            for (cmp, condition) in expected.into_iter().enumerate() {
                let raw = (0x5b40u64 << 48) | ((cmp as u64) << 49) | (u64::from(signed) << 48);
                assert_eq!(
                    run(&module(raw, [0, a, b, value]), 0).0,
                    if condition { a } else { b }
                );
            }
        }
    }
    // Immediate B: sign bit 56, not bit 39 (gm107.c:253,2148).
    let raw = (0x3640u64 << 48) | (1 << 56) | (7 << 49) | (1 << 48);
    assert_eq!(run(&module(raw, [0, a, b, u32::MAX]), 0).0, a);
    let raw = raw & !(7 << 49); // Compare false selects B, not C.
    assert_eq!(run(&module(raw, [0, a, b, u32::MAX]), 0).0, 0xfff8_0003);
}

// Ryujinx InstEmitBitfield.cs:185-192, independently count bits in the
// project-owned integer inputs. R2 is deliberately different from R3.
#[test]
fn popc_cpu_counts_b_with_inversion_and_signed_immediates() {
    for value in [0u32, 1, 0x8000_0000, 0x1234_5678, u32::MAX] {
        for invert in [false, true] {
            let raw = (0x5c08u64 << 48) | (u64::from(invert) << 40);
            let expected = (if invert { !value } else { value }).count_ones();
            assert_eq!(run(&module(raw, [0, !value, value, 0]), 0).0, expected);
        }
    }
    for sign in [false, true] {
        for invert in [false, true] {
            let raw = (0x3808u64 << 48) | (u64::from(sign) << 56) | (u64::from(invert) << 40);
            let value = if sign { 0xfff8_0003u32 } else { 3 };
            let expected = (if invert { !value } else { value }).count_ones();
            assert_eq!(run(&module(raw, [0, 0, 0, 0]), 0).0, expected);
        }
    }
}

#[test]
fn integer_cbuf_forms_preserve_sources_and_descriptors() {
    use shadowbox::ir::{IntCompare, Relation, Signedness};
    // Offset is words 20..33, bank 34..38, C register 39..46:
    // gm107.c:258,302-309,2030,2046,2101,2148.
    let cbuf = Source::ConstantBuffer {
        binding: 3,
        offset: 28,
    };
    let fields = (7 << 20) | (3 << 34) | (4 << 39) | (2 << 8) | 1 | (7 << 16);
    for (opcode, when_false, test) in [
        (0x4b40u64, cbuf, Source::Register(Register(4))),
        (0x5340u64, Source::Register(Register(4)), cbuf),
    ] {
        let instruction = decode((opcode << 48) | (2 << 49) | (1 << 48) | fields);
        assert!(matches!(instruction.kind,
            InstructionKind::ICmp(ops::ICmp { when_false: b, test: c, compare: IntCompare::Relation(Relation::Eq), signedness: Signedness::Signed, .. })
            if b == when_false && c == test));
        assert!(instruction.kind.sources().contains(&cbuf));
        assert!(emit_with_header(&[instruction], None).is_ok());
    }
    let instruction = decode((0x4c08u64 << 48) | (1 << 40) | (fields & !(0xff << 39)));
    assert!(
        matches!(instruction.kind, InstructionKind::Popc(ops::Popc { src, invert: true, .. }) if src == cbuf)
    );
    assert_eq!(instruction.kind.sources(), vec![cbuf]);
    assert!(emit_with_header(&[instruction], None).is_ok());
}

#[test]
fn lop3_f8_cpu_uses_immediate_b_register_c_and_correct_truth_table() {
    // gm107.c:2108,2169; Ryujinx InstDecoders.cs:3752-3764,
    // Lop3Expression.cs:22: A | (B & C). Distinct values expose accidental
    // register reads from B's immediate payload and reversed truth inputs.
    for sign in [false, true] {
        let a = 0x0100_0100;
        let c = 0x8000_0012;
        let raw = (0x3cf8u64 << 48) | (u64::from(sign) << 56);
        let b = if sign { 0xfff8_0003 } else { 3 };
        assert_eq!(run(&module(raw, [0, a, 0xdead_beef, c]), 0).0, a | (b & c));
    }
}

// Ryujinx InstEmitPredicate.cs:16-32: independent boolean truth tables.
// These checks execute the emitted operations, including predicate inversion,
// both logical selectors, integer/float results and the instruction guard.
#[test]
fn pset_cpu_truth_tables_and_predication() {
    let op = |code, a: bool, b: bool| match code {
        0 => a && b,
        1 => a || b,
        2 => a != b,
        _ => unreachable!(),
    };
    for ab in 0..3_u64 {
        for c in 0..3_u64 {
            for inversions in 0..8_u64 {
                for predicates in 0..8_u32 {
                    for float in [false, true] {
                        let raw = (0x5088_u64 << 48)
                            | 1
                            | (1 << 29)
                            | (2 << 39)
                            | (ab << 24)
                            | (c << 45)
                            | (u64::from(float) << 44)
                            | ((inversions & 1) << 15)
                            | (((inversions >> 1) & 1) << 32)
                            | (((inversions >> 2) & 1) << 42);
                        let values: [bool; 3] = std::array::from_fn(|i| {
                            ((predicates >> i) & 1 != 0) != ((inversions >> i) & 1 != 0)
                        });
                        let condition = op(c, op(ab, values[0], values[1]), values[2]);
                        for inverted_guard in [false, true] {
                            let kinds = [
                                InstructionKind::Mov(ops::Mov {
                                    quad_lanes: 15,
                                    dst: Register(1),
                                    src: Source::Immediate(0x1234),
                                }),
                                InstructionKind::Mov(ops::Mov {
                                    quad_lanes: 15,
                                    dst: Register(2),
                                    src: Source::Immediate(predicates),
                                }),
                                InstructionKind::R2P(ops::R2P {
                                    src: Register(2),
                                    mask: Source::Immediate(7),
                                    byte: 0,
                                    bank: shadowbox::ir::FlagBank::Predicates,
                                }),
                            ];
                            let mut xs: Vec<_> = kinds
                                .into_iter()
                                .enumerate()
                                .map(|(i, kind)| Instruction {
                                    address: i as u32 * 8,
                                    predicate: None,
                                    kind,
                                })
                                .collect();
                            xs.push(decode(raw | (7 << 16) | (u64::from(inverted_guard) << 19)));
                            let header = headers::parse(&[0; 32]).unwrap();
                            let words = emit_with_header(&xs, Some(&header)).unwrap().words;
                            let expected = if inverted_guard {
                                0x1234
                            } else if condition {
                                if float {
                                    1.0_f32.to_bits()
                                } else {
                                    u32::MAX
                                }
                            } else {
                                0
                            };
                            assert_eq!(run(&words, 0).0, expected,
                                "ab={ab} c={c} inv={inversions} preds={predicates} float={float} guard={inverted_guard}");
                        }
                    }
                }
            }
        }
    }
}

// Ryujinx InstEmitAttribute.cs:167-175,208-217. Test multiplication order
// and saturation independently, for perspective and linear headers and all
// four components. The host input and FragCoord.w deliberately differ from B.
#[test]
fn ipa_saturate_cpu_reconstructs_and_multiplies_before_clamp() {
    // ShaderHeader.cs:30-48 returns Unused for an empty map, which also
    // skips reconstruction (InstEmitAttribute.cs:171). Flat and linear
    // maps skip it; only perspective uses FragCoord.w.
    for interpolation in [0_u32, 1, 2, 3] {
        let perspective = interpolation == 2;
        for component in 0..4_u64 {
            for input in [-3.0_f32, -0.0, 0.25, 0.9, 2.0, f32::INFINITY, f32::NAN] {
                for inv_w in [0.125_f32, 1.0, 4.0, -1.0] {
                    let raw = (0xe04b_u64 << 48)
                        | 1
                        | (255 << 8)
                        | (7 << 16)
                        | (2 << 20)
                        | ((0x80 + component * 4) << 28)
                        | (255 << 39)
                        | (1 << 47);
                    let xs = [
                        Instruction {
                            address: 0,
                            predicate: None,
                            kind: InstructionKind::Mov(ops::Mov {
                                quad_lanes: 15,
                                dst: Register(2),
                                src: Source::Immediate(inv_w.to_bits()),
                            }),
                        },
                        decode(raw),
                    ];
                    let mut record = [0; 32];
                    record[0] = 5 << 10;
                    record[6] = interpolation;
                    let header = headers::parse(&record).unwrap();
                    let words = emit_with_header(&xs, Some(&header)).unwrap().words;
                    let w_value = 0.75_f32;
                    let attr = if perspective { input * w_value } else { input };
                    // NClamp maps NaN and signed zero to +0, unlike Rust clamp.
                    let product = attr * inv_w;
                    let expected = if product > 0.0 { product.min(1.0) } else { 0.0 };
                    let output = run_with_fragment_inputs(
                        &words,
                        0,
                        Some(([input.to_bits(); 4], [0, 0, 0, w_value.to_bits()])),
                        None,
                    )
                    .0;
                    assert_eq!(
                        output,
                        expected.to_bits(),
                        "perspective={perspective} component={component} input={input} B={inv_w}"
                    );
                }
            }
        }
    }
}

// Ryujinx InstEmitTexture.cs:858-875,897-923 and Spirv/Instructions.cs:
// 873-895. Mocked host image values isolate coordinate layout, reversed XY,
// signed fixed-point conversion, selected-channel packing and query sharing.
#[test]
fn tmml_cpu_reverses_scales_truncates_and_packs_one_query() {
    for dim in [0_u64, 1, 2, 3, 4, 6, 7] {
        let array = matches!(dim, 1 | 3 | 7);
        let count = match dim {
            0 | 1 => 1,
            2 | 3 => 2,
            _ => 3,
        };
        let coords = [0.125_f32.to_bits(), (-2.5_f32).to_bits(), 3.0_f32.to_bits()];
        for mask in [1_u64, 2, 3] {
            for dest in if mask == 3 { vec![0_u64, 1] } else { vec![1] } {
                let raw = (0xdf58_u64 << 48)
                    | dest
                    | (4 << 8)
                    | (7 << 16)
                    | (dim << 28)
                    | (mask << 31)
                    | (3 << 36);
                let mut xs = Vec::new();
                let mut add = |kind| {
                    xs.push(Instruction {
                        address: 0,
                        predicate: None,
                        kind,
                    })
                };
                if array {
                    add(InstructionKind::Mov(ops::Mov {
                        quad_lanes: 15,
                        dst: Register(4),
                        src: Source::Immediate(17),
                    }));
                }
                for (index, bits) in coords[..count].iter().enumerate() {
                    add(InstructionKind::Mov(ops::Mov {
                        quad_lanes: 15,
                        dst: Register(4 + u8::from(array) + index as u8),
                        src: Source::Immediate(*bits),
                    }));
                }
                xs.push(decode(raw));
                let mut record = [0; 32];
                record[0] = 5 << 10;
                let header = headers::parse(&record).unwrap();
                let words = emit_with_header(&xs, Some(&header)).unwrap().words;
                for pair in [[1.234_f32, -2.345_f32], [-0.003, 0.003], [0.0, 5.0]] {
                    let expected = pair[if mask == 2 || dest == 0 { 0 } else { 1 }];
                    let expected = ((expected * 256.0) as i32) as u32;
                    assert_eq!(
                        run_with_fragment_inputs(&words, 0, None, Some((pair, &coords[..count]))).0,
                        expected,
                        "dim={dim} mask={mask} dest={dest} pair={pair:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn bounded_game_forms_reject_unimplemented_fields() {
    // IPA.MUL.SAT fields: Mesa nak/sm50.rs:2908-2934;
    // envytools gm107.c:1953-1954,859-871. Unsupported values must not
    // inherit main's simpler IPA behavior and silently lose their effects.
    let ipa = (0xe04b_u64 << 48)
        | (255 << 8)
        | (7 << 16)
        | (2 << 20)
        | (0x80 << 28)
        | (255 << 39)
        | (1 << 47)
        | 1;
    assert!(matches!(
        decode(ipa).kind,
        InstructionKind::IpaSaturate(ops::IpaSaturate { .. })
    ));
    for raw in [
        ipa ^ (1 << 47),    // unsupported predicate output
        ipa | (1 << 50),    // predicate inversion
        ipa | (1 << 38),    // indexed address
        ipa ^ (1 << 8),     // unknown address slot
        ipa & !(255 << 8),  // zero literal encodes RZ, not register R0
        ipa ^ (1 << 39),    // non-RZ offset
        ipa | (1 << 28),    // unaligned address
        ipa ^ (0x80 << 28), // unsupported fixed-function input
        ipa | (1 << 52),    // centroid
        ipa | (2 << 52),    // offset sampling
        ipa | (3 << 52),    // reserved location
        ipa ^ (1 << 54),    // unsupported saturating PASS
        ipa ^ (3 << 54),    // unsupported saturating CONSTANT
        ipa | (2 << 54),    // unsupported saturating SC
    ] {
        assert!(
            matches!(
                decode(raw).kind,
                InstructionKind::Unknown(ops::Unknown { .. })
            ),
            "{raw:016x}"
        );
    }
    // The new variants have no meaningful representation in non-fragment
    // stages. Do not produce arbitrary inputs or a made-up LOD instead.
    let tmml = (0xdf58_u64 << 48) | (2 << 28) | (3 << 31) | (7 << 16) | 1;
    for stage in 0..5_u32 {
        let mut record = [0; 32];
        record[0] = stage << 10;
        let header = headers::parse(&record).unwrap();
        for raw in [ipa, tmml] {
            assert!(emit_with_header(&[decode(raw)], Some(&header)).is_err());
        }
    }
    // PSET CC/reserved logical operations cannot be accepted by ignoring
    // their fields (gm107.c:2061,1430-1434,1683-1687). RRO.C EX2 is
    // decoded with its mode retained; see alu_semantics.rs.
    for raw in [
        (0x5088_u64 << 48) | (1 << 47),
        (0x5088_u64 << 48) | (3 << 24),
        (0x5088_u64 << 48) | (3 << 45),
        (0xdf58_u64 << 48) | (1 << 35),
        (0xdf58_u64 << 48) | (4 << 31),
        (0xdf58_u64 << 48) | (1 << 20),
    ] {
        assert!(
            matches!(
                decode(raw).kind,
                InstructionKind::Unknown(ops::Unknown { .. })
            ),
            "{raw:016x}"
        );
    }
}

#[path = "support/spirv_validation.rs"]
mod spirv_validation;

// Emit actual decoded instructions, without the operand bits supplied by
// module(). Cbuf tests supply a complete vec4 at bank 0, byte offset 0.
fn half_module(raw: u64, registers: [u32; 4]) -> Vec<u32> {
    let mut instructions = registers
        .into_iter()
        .enumerate()
        .map(|(i, value)| Instruction {
            address: (i as u32 + 1) * 8,
            predicate: None,
            kind: InstructionKind::Mov(ops::Mov {
                quad_lanes: 15,
                dst: Register(i as u8 + 1),
                src: Source::Immediate(value),
            }),
        })
        .collect::<Vec<_>>();
    instructions.push(decode(raw | 7 << 16));
    instructions.push(Instruction {
        address: 48,
        predicate: None,
        kind: InstructionKind::Exit(ops::Exit { cc_test: 15 }),
    });
    let header = headers::parse(&[0; 32]).unwrap();
    emit_with_header(&instructions, Some(&header))
        .unwrap()
        .words
}

fn validate_half(words: &[u32]) {
    let result = spirv_validation::validate(words);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}

fn signed_halves(low: u16) -> u32 {
    u32::from(low) | (u32::from(low ^ 0x8000) << 16)
}

// Independent numeric oracle. Classify inputs in their source precision,
// perform the reference's float32 operation, then classify the rounded output.
// This does not use the emitter's magnitude-bit test or FMZ operand substitution.
fn flush_reference(value: f32, enabled: bool, half: bool) -> f32 {
    let normal = if half {
        2.0_f32.powi(-14)
    } else {
        f32::MIN_POSITIVE
    };
    if enabled && value.abs() < normal {
        0.0_f32.copysign(value)
    } else {
        value
    }
}

#[test]
fn half_modes_execute_exact_values_in_every_arithmetic_form() {
    use shadowbox::ir::HalfOp;
    // Fields/forms: envytools gm107.c:1977-1983,1987-1989,2155-2157;
    // mode tables :1083-1087,1157-1168,1782-1786. No FMZ in HADD2.
    for (opcode, op, shift, modes, form) in [
        (0x5d10_u64, HalfOp::Add, 39, 2, "reg"),
        (0x7a00, HalfOp::Add, 39, 2, "short"),
        (0x7a80, HalfOp::Add, 39, 2, "cbuf"),
        (0x2c00, HalfOp::Add, 55, 2, "wide"),
        (0x5d08, HalfOp::Mul, 39, 3, "reg"),
        (0x7800, HalfOp::Mul, 39, 3, "short"),
        (0x7880, HalfOp::Mul, 39, 3, "cbuf"),
        (0x2a00, HalfOp::Mul, 55, 3, "wide"),
        (0x5d00, HalfOp::Fma, 37, 3, "reg"),
        (0x7000, HalfOp::Fma, 57, 3, "short"),
        (0x7080, HalfOp::Fma, 57, 3, "cbuf"),
        (0x6080, HalfOp::Fma, 57, 3, "rc"),
        (0x2800, HalfOp::Fma, 55, 3, "wide"),
    ] {
        for mode in 0..modes {
            let mut validated = false;
            // Short literals have six implicit zero bits, gm107.c:240-241.
            // Mixed lanes include both signs of zero, denormal, infinity, NaN.
            let inputs = [
                0_u16, 0x8000, 0x0040, 0x8040, 0x03c0, 0x0400, 0x3800, 0x3c00, 0xbc00, 0x7c00,
                0xfc00, 0x7e00,
            ];
            for a in inputs {
                for b in inputs {
                    let addends: &[u16] = if op == HalfOp::Fma {
                        &[0, 0x8000, 0x0040, 0x8040, 0x3c00, 0x7c00, 0x7e00]
                    } else {
                        &[0]
                    };
                    for &c in addends {
                        let (a_bits, b_bits, c_bits) =
                            (signed_halves(a), signed_halves(b), signed_halves(c));
                        let mut raw = opcode << 48 | mode << shift | 1 | 2 << 8;
                        raw |= match form {
                            "reg" => 3 << 20,
                            "wide" => u64::from(b_bits) << 20,
                            "short" => {
                                u64::from((b_bits >> 6) & 0x3ff) << 20
                                    | u64::from((b_bits >> 22) & 0x1ff) << 30
                                    | u64::from(b_bits >> 31) << 56
                            }
                            "cbuf" | "rc" => 0,
                            _ => unreachable!(),
                        };
                        if op == HalfOp::Fma && form != "wide" {
                            raw |= 4 << 39;
                        }
                        let uniform = if form == "rc" { unpack(c) } else { unpack(b) };
                        let words = half_module(
                            raw,
                            [
                                c_bits,
                                a_bits,
                                b_bits,
                                if form == "rc" { b_bits } else { c_bits },
                            ],
                        );
                        if !validated {
                            validate_half(&words);
                            validated = true;
                        }
                        let mut expected = 0;
                        for lane in 0..2 {
                            let a = f32::from_bits(unpack((a_bits >> (lane * 16)) as u16));
                            let b = f32::from_bits(if form == "cbuf" {
                                uniform
                            } else {
                                unpack((b_bits >> (lane * 16)) as u16)
                            });
                            let c = f32::from_bits(if form == "rc" {
                                uniform
                            } else {
                                unpack((c_bits >> (lane * 16)) as u16)
                            });
                            let a = flush_reference(a, mode != 0, true);
                            let b = flush_reference(b, mode != 0, form != "cbuf");
                            let c = flush_reference(c, mode != 0, form != "rc");
                            let zero_product = mode == 2 && (a == 0.0 || b == 0.0);
                            let product_zero = if a.is_sign_negative() ^ b.is_sign_negative() {
                                -0.0
                            } else {
                                0.0
                            };
                            let result = match op {
                                HalfOp::Add => a + b,
                                HalfOp::Mul if zero_product => product_zero,
                                HalfOp::Mul => a * b,
                                HalfOp::Fma if zero_product => product_zero + c,
                                HalfOp::Fma => a.mul_add(b, c),
                            };
                            let half = pack(result.to_bits());
                            let half = if mode != 0 && half & 0x7c00 == 0 {
                                half & 0x8000
                            } else {
                                half
                            };
                            expected |= u32::from(half) << (lane * 16);
                        }
                        let actual = run_with_uniform(&words, 0, Some([uniform; 4])).0;
                        assert_eq!(
                            actual, expected,
                            "op={opcode:04x} mode={mode} a={a:04x} b={b:04x} c={c:04x}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn half_ftz_fmz_fixed_bits_rounding_merges_and_f32_precision() {
    // Fixed answers establish signs, output flushing, NaN and FMZ exceptions.
    // Register operand/output fields: envytools gm107.c:1987-1989,1133-1175.
    for (opcode, shift, a, b, c, expected) in [
        (
            0x5d10_u64,
            39,
            0x8040_0040,
            0x8000_0000,
            0,
            [0x8040_0040, 0x8000_0000, 0],
        ),
        (
            0x5d08,
            39,
            0x8400_0400,
            0x3800_3800,
            0,
            [0x8200_0200, 0x8000_0000, 0x8000_0000],
        ),
        (
            0x5d08,
            39,
            0x8000_0000,
            0x7c00_7c00,
            0,
            [0xfe00_fe00, 0xfe00_fe00, 0x8000_0000],
        ),
        (
            0x5d08,
            39,
            0x8000_0000,
            0x7e00_7e00,
            0,
            [0x7e00_7e00, 0x7e00_7e00, 0x8000_0000],
        ),
        (
            0x5d00,
            37,
            0x8000_0000,
            0x7c00_7c00,
            0x8000_3c00,
            [0xfe00_fe00, 0xfe00_fe00, 0x8000_3c00],
        ),
        // Input denormal becomes zero even when multiplication could normalize it.
        (
            0x5d08,
            39,
            0x8040_0040,
            0x5c00_5c00,
            0,
            [0x9400_1400, 0x8000_0000, 0x8000_0000],
        ),
    ] {
        let modes = if opcode == 0x5d10 { 2 } else { 3 };
        for mode in 0..modes {
            let raw = opcode << 48
                | mode << shift
                | 1
                | 2 << 8
                | 3 << 20
                | if opcode == 0x5d00 { 4 << 39 } else { 0 };
            assert_eq!(
                run(&half_module(raw, [0, a, b, c]), 0).0,
                expected[mode as usize]
            );
        }
    }
    // F32 source/output flush at 2^-126, not the half normal boundary.
    let raw = 0x5d08_u64 << 48 | 1 << 47 | 1 << 28 | 1 << 49 | 1 | 2 << 8 | 3 << 20;
    for mode in 0..3 {
        for (a, b, expected) in [
            (
                0x8000_0001,
                1.0_f32.to_bits(),
                if mode == 0 { 0x8000_0001 } else { 0x8000_0000 },
            ),
            (
                2.0_f32.powi(-20).to_bits(),
                1.0_f32.to_bits(),
                2.0_f32.powi(-20).to_bits(),
            ),
            (
                f32::MIN_POSITIVE.to_bits(),
                0.5_f32.to_bits(),
                if mode == 0 { 0x0040_0000 } else { 0 },
            ),
        ] {
            let words = half_module(raw | mode << 39, [0, a, b, 0]);
            validate_half(&words);
            assert_eq!(run(&words, 0).0, expected);
        }
    }
    // A normal F32 broadcast just below min-normal16 rounds upward before FTZ.
    let raw = 0x5d10_u64 << 48 | 1 << 47 | 1 << 28 | 1 << 39 | 1 | 2 << 8 | 3 << 20;
    assert_eq!(
        run(&half_module(raw, [0, 0x387f_f000, 0, 0]), 0).0,
        0x0400_0400
    );
    // Merge preserves the old denormal lane rather than flushing the destination.
    for (output, expected) in [(2_u64, 0x8040_0000), (3, 0x8000_0040)] {
        let raw = 0x5d08_u64 << 48 | 1 << 39 | output << 49 | 1 | 2 << 8 | 3 << 20;
        assert_eq!(
            run(
                &half_module(raw, [0x8040_0040, 0x8400_0400, 0x3800_3800, 0]),
                0
            )
            .0,
            expected
        );
    }
}

#[test]
fn half_comparison_ftz_exact_register_predicate_and_cbuf_results() {
    // Forms, compare and FTZ bits: envytools gm107.c:1973-1976,1985-1986,
    // 1113-1129,1063-1079. Ryujinx InstEmitFloatComparison.cs:456-564
    // defines comparison/accumulator results. Cbuf is F32, not packed half.
    for (opcode, ftz_bit, compare_bit, form, predicates) in [
        (0x5d18_u64, 50, 35, "reg", false),
        (0x7c00, 54, 49, "short", false),
        (0x7c80, 54, 49, "cbuf", false),
        (0x5d20, 6, 35, "reg", true),
        (0x7e00, 6, 49, "short", true),
        (0x7e80, 6, 49, "cbuf", true),
    ] {
        for ftz in [false, true] {
            for selector in [1_u64, 2, 7, 8, 10] {
                let mut validated = false;
                let inputs = [
                    0_u16, 0x8000, 0x0040, 0x8040, 0x0400, 0x3c00, 0x7c00, 0xfc00, 0x7e00,
                ];
                for a in inputs {
                    for b in inputs {
                        let (a_bits, b_bits) = (signed_halves(a), signed_halves(b));
                        let mut raw = opcode << 48
                            | u64::from(ftz) << ftz_bit
                            | selector << compare_bit
                            | 1
                            | 2 << 8
                            | 7 << 39;
                        raw |= match form {
                            "reg" => 3 << 20,
                            "cbuf" => 0,
                            "short" => {
                                u64::from((b_bits >> 6) & 0x3ff) << 20
                                    | u64::from((b_bits >> 22) & 0x1ff) << 30
                                    | u64::from(b_bits >> 31) << 56
                            }
                            _ => unreachable!(),
                        };
                        let words = half_module(raw, [0xdead_beef, a_bits, b_bits, 0]);
                        if !validated {
                            validate_half(&words);
                            validated = true;
                        }
                        let mut expected = [0_u32; 2];
                        for lane in 0..2 {
                            let x = f32::from_bits(unpack((a_bits >> (16 * lane)) as u16));
                            let y = f32::from_bits(unpack(if form == "cbuf" {
                                b
                            } else {
                                (b_bits >> (16 * lane)) as u16
                            }));
                            let x = flush_reference(x, ftz, true);
                            let y = flush_reference(y, ftz, form != "cbuf");
                            let nan = x.is_nan() || y.is_nan();
                            expected[lane as usize] = u32::from(match selector {
                                1 => !nan && x < y,
                                2 => !nan && x == y,
                                7 => !nan,
                                8 => nan,
                                10 => nan || x == y,
                                _ => unreachable!(),
                            });
                        }
                        let (register, ps) = run_with_uniform(&words, 0, Some([unpack(b); 4]));
                        if predicates {
                            assert_eq!(
                                [ps[0], ps[1]],
                                expected,
                                "opcode={opcode:04x} ftz={ftz} cmp={selector} a={a:04x} b={b:04x}"
                            );
                            assert_eq!(register, 0xdead_beef);
                        } else {
                            assert_eq!(
                                register,
                                (expected[0] * 0xffff) | ((expected[1] * 0xffff) << 16),
                                "opcode={opcode:04x} ftz={ftz} cmp={selector} a={a:04x} b={b:04x}"
                            );
                        }
                    }
                }
            }
            // F32 cbuf denormals flush before comparison; positive and negative
            // half denormals on A already compare equal to zero under FTZ.
            if form == "cbuf" {
                let raw = opcode << 48
                    | u64::from(ftz) << ftz_bit
                    | 2 << compare_bit
                    | 1
                    | 2 << 8
                    | 7 << 39;
                let words = half_module(raw, [0, 0, 0, 0]);
                for uniform in [1, 0x8000_0001, 0x007f_ffff, 0x807f_ffff] {
                    let (register, ps) = run_with_uniform(&words, 0, Some([uniform; 4]));
                    if predicates {
                        assert_eq!([ps[0], ps[1]], [u32::from(ftz); 2]);
                    } else {
                        assert_eq!(register, if ftz { u32::MAX } else { 0 });
                    }
                }
            }
        }
    }
}

#[test]
fn directed_fmul_uses_float64_only_with_host_support() {
    use shadowbox::{emitter::emit_unoptimized_with_options, TranslationOptions};
    // Rounding39..40 and B20..27: Mesa sm50.rs:679-695.
    // Host capability gate: Ryujinx SpirvGenerator.cs:58-60.
    let instruction = decode((0x5c68u64 << 48) | (1 << 39) | (7 << 16) | (3 << 20) | (2 << 8) | 1);
    let exit = Instruction {
        address: 48,
        predicate: None,
        kind: InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
    };
    let header = headers::parse(&[0; 32]).unwrap();
    for float64 in [
        shadowbox::HostFloat64::Unsupported,
        shadowbox::HostFloat64::Supported,
    ] {
        let supported = float64 == shadowbox::HostFloat64::Supported;
        let result = emit_unoptimized_with_options(
            &[instruction, exit],
            Some(&header),
            TranslationOptions {
                float64,
                ..Default::default()
            },
        );
        let module = result.unwrap();
        validate_half(&module.words);
        let mut has_float64 = false;
        let mut converts = 0;
        let mut at = 5;
        while at < module.words.len() {
            let word = module.words[at];
            let count = (word >> 16) as usize;
            let args = &module.words[at + 1..at + count];
            match Op::of(word) {
                Op::Capability if args == [10] => has_float64 = true,
                Op::FConvert => converts += 1,
                _ => {}
            }
            at += count;
        }
        assert_eq!(has_float64, supported);
        assert_eq!(converts, if supported { 4 } else { 0 });
    }
}

#[test]
fn half_predicate_outputs_share_the_entry_guard() {
    // HSETP2 predicates0/3, H_AND49, cmp35..38, accumulator39:
    // envytools gm107.c:1985,363-365,1113-1129. Guard16..19:408-411.
    // Ryujinx Translator.cs:359-393 guards the complete instruction;
    // InstEmitFloatComparison.cs:494-523 writes both predicates in order.
    // Both outputs alias P0. !P0 initially executes. First output is true;
    // second is false and must still execute after P0 changes.
    let raw = 0x5d20_u64 << 48 | 1 << 19 | 15 << 35 | 7 << 39 | 1 << 49;
    let words = exact_module(raw, [0; 4]);
    validate_half(&words);
    assert_eq!(run(&words, 0).1[0], 0);
}
