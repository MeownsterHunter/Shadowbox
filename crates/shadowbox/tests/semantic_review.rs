//! Binary opcode/operand checks plus an independent evaluator for the emitted
//! integer, memory and branch subset.
use shadowbox::ir::ops;
use shadowbox::spirv::Op;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header as emit_with_header,
    headers,
    ir::{
        AtomicOp, BranchTarget, BrxBase, Instruction, InstructionKind as K, LocalStoreCache,
        LogicOp, LsSize2, Predicate, Register as R, Signedness, Source as S, VideoElement,
        VideoSelection,
    },
};
use std::collections::HashMap;
#[path = "support/spirv_validation.rs"]
mod spirv_validation;

fn decoded(raw: u64) -> K {
    decoded_at(raw, 0)
}

fn decoded_at(raw: u64, address: u32) -> K {
    decode_word([raw as u32, (raw >> 32) as u32], address).kind
}

fn module(kinds: Vec<(K, Option<Predicate>)>) -> Vec<u32> {
    let mut xs: Vec<_> = kinds
        .into_iter()
        .enumerate()
        .map(|(i, (kind, predicate))| Instruction {
            address: i as u32 * 8,
            kind,
            predicate,
        })
        .collect();
    xs.push(Instruction {
        address: xs.len() as u32 * 8,
        predicate: None,
        kind: K::Exit(ops::Exit { cc_test: 0xf }),
    });
    emit_with_header(&xs, Some(&headers::parse(&[0; 32]).unwrap()))
        .unwrap()
        .words
}

fn mov(dst: u8, value: u32) -> (K, Option<Predicate>) {
    (
        K::Mov(ops::Mov {
            quad_lanes: 0xf,
            dst: R(dst),
            src: S::Immediate(value),
        }),
        None,
    )
}

fn vmad_whole_a(dst: u8, lhs: u8, rhs: u8, addend: u8, signed: bool) -> u64 {
    (0x5f00_u64 << 48)
        | u64::from(dst)
        | (u64::from(lhs) << 8)
        | (7 << 16)
        | (u64::from(rhs) << 20)
        | (6 << 36)
        | (u64::from(addend) << 39)
        | (u64::from(signed) << 48)
        | (1 << 50)
}

#[test]
fn vmad_whole_word_selector_is_typed_exact_and_validates() {
    for (signed, expected_selection, lhs, expected) in [
        (
            false,
            VideoSelection {
                element: VideoElement::Whole,
                signedness: Signedness::Unsigned,
            },
            0xffff_fff0,
            0xffff_ffd5,
        ),
        (
            true,
            VideoSelection {
                element: VideoElement::Whole,
                signedness: Signedness::Signed,
            },
            0xffff_fff0,
            (-43_i32) as u32,
        ),
    ] {
        let raw = vmad_whole_a(2, 2, 4, 5, signed);
        let K::VideoMultiplyAdd(ops::VideoMultiplyAdd { select, .. }) = decoded(raw) else {
            panic!("whole-word VMAD must decode concretely");
        };
        assert_eq!(select[0], expected_selection);

        let kinds = vec![mov(2, lhs), mov(4, 3), mov(5, 5), (decoded(raw), None)];
        let words = module(kinds);
        assert!(spirv_validation::validate(&words).status.success());
        let widened = if signed {
            lhs as i32 as i64 as u64
        } else {
            u64::from(lhs)
        };
        assert_eq!(widened.wrapping_mul(3).wrapping_add(5) as u32, expected);
    }

    let reserved = vmad_whole_a(1, 2, 4, 5, false) | (1 << 36);
    assert!(matches!(decoded(reserved), K::Unknown(_)));
}

/// LD/ST sizes by encoding, Ryujinx Decoders/InstDecoders.cs (LsSize2).
const SIZES: [LsSize2; 7] = [
    LsSize2::U8,
    LsSize2::S8,
    LsSize2::U16,
    LsSize2::S16,
    LsSize2::B32,
    LsSize2::B64,
    LsSize2::B128,
];

fn memory(store: bool, shared: bool, size: u8, data: u8, offset: i32) -> K {
    let size = SIZES[usize::from(size)];
    match (store, shared) {
        (true, false) => K::LocalStore(ops::LocalStore {
            data: R(data),
            address: R(10),
            offset,
            mem_type: size,
            cache: LocalStoreCache::Default,
        }),
        (true, true) => K::SharedStore(ops::SharedStore {
            data: R(data),
            address: R(10),
            offset,
            mem_type: size,
        }),
        (false, false) => K::LocalLoad(ops::LocalLoad {
            dst: R(data),
            address: R(10),
            offset,
            mem_type: size,
        }),
        (false, true) => K::SharedLoad(ops::SharedLoad {
            dst: R(data),
            address: R(10),
            offset,
            mem_type: size,
        }),
    }
}

fn ops(words: &[u32]) -> Vec<(Op, &[u32])> {
    let mut result = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        result.push((Op::of(words[at]), &words[at + 1..at + count]));
        at += count;
    }
    result
}

fn run(words: &[u32]) -> [u32; 256] {
    execute(words, 0).registers
}

struct Execution {
    registers: [u32; 256],
    outputs: HashMap<u32, Vec<u32>>,
    discarded: bool,
}

fn execute(words: &[u32], cbuf: u32) -> Execution {
    let instructions = ops(words);
    let labels: HashMap<_, _> = instructions
        .iter()
        .enumerate()
        .filter(|(_, (op, _))| *op == Op::Label)
        .map(|(i, (_, a))| (a[0], i))
        .collect();
    let mut uint = 0;
    let mut double = 0;
    let mut reg_pointers = Vec::new();
    let mut regs = Vec::new();
    for &(op, a) in &instructions {
        match op {
            Op::TypeInt if a[1..] == [32, 0] => uint = a[0],
            Op::TypeFloat if a[1] == 64 => double = a[0],
            Op::TypePointer if a[1..] == [7, uint] => reg_pointers.push(a[0]),
            Op::Variable if reg_pointers.contains(&a[0]) => regs.push(a[1]),
            _ => {}
        }
    }
    // Scalar values, composite values, and pointer paths are kept separately.
    let mut values: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut pointers: HashMap<u32, (u32, u32)> = HashMap::new();
    let mut memory: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    let uniform_vars: Vec<_> = instructions
        .iter()
        .filter(|(op, a)| *op == Op::Variable && a[2] == 2)
        .map(|(_, a)| a[1])
        .collect();
    let output_vars: Vec<_> = instructions
        .iter()
        .filter(|(op, a)| *op == Op::Variable && a[2] == 3)
        .map(|(_, a)| a[1])
        .collect();
    let mut outputs = HashMap::new();
    let mut demoted = false;
    let mut at = 0;
    let mut current = 0;
    let mut previous = 0;
    for _ in 0..100_000 {
        let (op, a) = instructions[at];
        let get = |i: usize| values[&a[i]][0];
        let scalar = match op {
            Op::ConstantTrue => Some(1),
            Op::ConstantFalse => Some(0),
            Op::Constant => Some(a[2]),
            Op::Variable => {
                pointers.insert(a[1], (a[1], 0));
                memory.entry((a[1], 0)).or_insert(vec![0]);
                None
            }
            Op::AccessChain => {
                let (root, base) = pointers[&a[2]];
                pointers.insert(a[1], (root, base + get(3)));
                None
            }
            Op::Load => {
                let pointer = pointers[&a[2]];
                let value = if uniform_vars.contains(&pointer.0) {
                    vec![cbuf; 4]
                } else {
                    memory.get(&pointer).cloned().unwrap_or(vec![0])
                };
                values.insert(a[1], value);
                None
            }
            Op::Store => {
                let pointer = pointers[&a[0]];
                if output_vars.contains(&pointer.0) && !demoted {
                    outputs.insert(pointer.0, values[&a[1]].clone());
                }
                memory.insert(pointer, values[&a[1]].clone());
                None
            }
            Op::CompositeConstruct => {
                let composite = a[2..].iter().map(|id| values[id][0]).collect();
                values.insert(a[1], composite);
                None
            }
            Op::CompositeExtract => Some(values[&a[2]][a[3] as usize]),
            Op::ConvertFToU => Some(f32::from_bits(get(2)) as u32),
            Op::ConvertFToS => Some((f32::from_bits(get(2)) as i32) as u32),
            Op::IsNan => Some(u32::from(f32::from_bits(get(2)).is_nan())),
            Op::FOrdGreaterThan => Some(u32::from(f32::from_bits(get(2)) > f32::from_bits(get(3)))),
            Op::ExtInst if a[3] != 5 => Some(match a[3] {
                2 => f32::from_bits(get(4)).round_ties_even().to_bits(),
                3 => f32::from_bits(get(4)).trunc().to_bits(),
                8 => f32::from_bits(get(4)).floor().to_bits(),
                9 => f32::from_bits(get(4)).ceil().to_bits(),
                43 => f32::from_bits(get(4))
                    .clamp(f32::from_bits(get(5)), f32::from_bits(get(6)))
                    .to_bits(),
                other => panic!("unexpected GLSL.std.450 instruction {other}"),
            }),
            Op::ConvertSToF | Op::ConvertUToF if a[0] == double => {
                let number = if op == Op::ConvertSToF {
                    (get(2) as i32) as f64
                } else {
                    get(2) as f64
                };
                let bits = number.to_bits();
                values.insert(a[1], vec![bits as u32, (bits >> 32) as u32]);
                None
            }
            Op::ConvertSToF => Some(((get(2) as i32) as f32).to_bits()),
            Op::ConvertUToF => Some((get(2) as f32).to_bits()),
            Op::SNegate => Some(get(2).wrapping_neg()),
            Op::ExtInst if a[3] == 5 => Some((get(4) as i32).wrapping_abs() as u32),
            Op::Bitcast => {
                values.insert(a[1], values[&a[2]].clone());
                None
            }
            Op::IAdd => Some(get(2).wrapping_add(get(3))),
            Op::ISub => Some(get(2).wrapping_sub(get(3))),
            Op::IAddCarry => {
                let sum = u64::from(get(2)) + u64::from(get(3));
                values.insert(a[1], vec![sum as u32, (sum >> 32) as u32]);
                None
            }
            Op::LogicalEqual | Op::IEqual => Some(u32::from(get(2) == get(3))),
            Op::UGreaterThanEqual => Some(u32::from(get(2) >= get(3))),
            Op::LogicalNotEqual | Op::INotEqual => Some(u32::from(get(2) != get(3))),
            Op::LogicalOr => Some(u32::from(get(2) != 0 || get(3) != 0)),
            Op::LogicalAnd => Some(u32::from(get(2) != 0 && get(3) != 0)),
            Op::LogicalNot => Some(u32::from(get(2) == 0)),
            Op::Select => Some(get(if get(2) != 0 { 3 } else { 4 })),
            // OpUGreaterThan: Mesa spirv.core.grammar.json:2011-2018.
            Op::UGreaterThan => Some(u32::from(get(2) > get(3))),
            Op::ULessThan => Some(u32::from(get(2) < get(3))),
            Op::ShiftRightLogical => {
                Some(get(2).checked_shr(get(3)).expect("undefined logical shift"))
            }
            Op::ShiftRightArithmetic => Some(
                (get(2) as i32)
                    .checked_shr(get(3))
                    .expect("undefined arithmetic shift") as u32,
            ),
            Op::ShiftLeftLogical => Some(get(2).checked_shl(get(3)).expect("undefined left shift")),
            Op::BitwiseOr => Some(get(2) | get(3)),
            Op::BitwiseXor => Some(get(2) ^ get(3)),
            Op::BitwiseAnd => Some(get(2) & get(3)),
            Op::Not => Some(!get(2)),
            Op::BitFieldInsert => {
                let width = get(5);
                let mask = ((1u64 << width) - 1) as u32;
                Some(
                    ((u64::from(get(2)) & !(u64::from(mask) << get(4)))
                        | ((u64::from(get(3)) & u64::from(mask)) << get(4)))
                        as u32,
                )
            }
            Op::BitFieldSExtract | Op::BitFieldUExtract => {
                let width = get(4);
                let shift = 32 - width;
                let extracted = get(2).checked_shr(get(3)).unwrap() << shift;
                Some(if op == Op::BitFieldSExtract {
                    ((extracted as i32) >> shift) as u32
                } else {
                    extracted >> shift
                })
            }
            Op::AtomicLoad => Some(memory.get(&pointers[&a[2]]).map_or(0, |v| v[0])),
            Op::AtomicCompareExchange => {
                let pointer = pointers[&a[2]];
                let old = memory.get(&pointer).map_or(0, |v| v[0]);
                if old == get(7) {
                    memory.insert(pointer, vec![get(6)]);
                }
                Some(old)
            }
            Op::AtomicExchange
            | Op::AtomicIAdd
            | Op::AtomicSMin
            | Op::AtomicUMin
            | Op::AtomicSMax
            | Op::AtomicUMax
            | Op::AtomicAnd
            | Op::AtomicOr
            | Op::AtomicXor => {
                let pointer = pointers[&a[2]];
                let old = memory.get(&pointer).map_or(0, |v| v[0]);
                let value = get(5);
                let updated = match op {
                    Op::AtomicExchange => value,
                    Op::AtomicIAdd => old.wrapping_add(value),
                    Op::AtomicSMin => (old as i32).min(value as i32) as u32,
                    Op::AtomicUMin => old.min(value),
                    Op::AtomicSMax => (old as i32).max(value as i32) as u32,
                    Op::AtomicUMax => old.max(value),
                    Op::AtomicAnd => old & value,
                    Op::AtomicOr => old | value,
                    Op::AtomicXor => old ^ value,
                    _ => unreachable!(),
                };
                memory.insert(pointer, vec![updated]);
                Some(old)
            }
            Op::Label => {
                current = a[0];
                // SPIR-V core grammar spirv.core.grammar.json:2788-2796.
                // Snapshot all inputs before writing loop-carried phi results.
                let mut phis = Vec::new();
                while instructions.get(at + 1).is_some_and(|i| i.0 == Op::Phi) {
                    at += 1;
                    let phi = instructions[at].1;
                    let source = phi[2..]
                        .as_chunks::<2>()
                        .0
                        .iter()
                        .find(|pair| pair[1] == previous)
                        .expect("phi must have the executed predecessor")[0];
                    phis.push((phi[1], values[&source].clone()));
                }
                values.extend(phis);
                None
            }
            Op::Branch => {
                previous = current;
                at = labels[&a[0]];
                continue;
            }
            Op::BranchConditional => {
                previous = current;
                at = labels[&a[if get(0) != 0 { 1 } else { 2 }]];
                continue;
            }
            Op::Switch => {
                previous = current;
                let label = a[2..]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .find(|p| p[0] == get(0))
                    .map_or(a[1], |p| p[1]);
                at = labels[&label];
                continue;
            }
            Op::DemoteToHelperInvocation => {
                demoted = true;
                outputs.clear();
                None
            }
            Op::IsHelperInvocationEXT => Some(u32::from(demoted)),
            Op::Kill | Op::Return => {
                return Execution {
                    registers: std::array::from_fn(|i| {
                        regs.get(i)
                            .and_then(|&r| memory.get(&(r, 0)))
                            .map_or(0, |v| v[0])
                    }),
                    outputs,
                    discarded: demoted || op == Op::Kill,
                }
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
            | Op::SelectionMerge => None,
            other => panic!("unexpected opcode {}", other.number()),
        };
        if let Some(value) = scalar {
            values.insert(a[1], vec![value]);
        }
        at += 1;
    }
    panic!("dispatcher did not terminate");
}

#[test]
fn brx_uses_byte_displacements_bundle_targets_and_predication() {
    // Ryujinx InstEmitFlowControl.cs:38-40 adds the loaded byte displacement
    // to the PC-relative base. Decoder.cs:265-270 skips control words.
    // Translator.cs:359-393 guards the entire BRX instruction. Duplicate
    // table entries describe one runtime displacement, not distinct cases.
    for base in [-8, 0, 8, 16] {
        for (selector, expected) in [(8_i32 - base, 0x1111), (48 - base, 0x2222)] {
            for predicate in [
                None,
                Some(Predicate {
                    register: 0,
                    inverted: false,
                }),
                Some(Predicate {
                    register: 0,
                    inverted: true,
                }),
            ] {
                let mut targets = [0; 32];
                targets[..3].copy_from_slice(&[
                    (8 - base) as u32,
                    (8 - base) as u32,
                    (48 - base) as u32,
                ]);
                let xs = [
                    Instruction {
                        address: 8,
                        predicate: None,
                        kind: mov(2, selector as u32).0,
                    },
                    Instruction {
                        address: 16,
                        predicate,
                        kind: K::brx(
                            R(2),
                            BrxBase::decode(16, base as u32),
                            targets[0],
                            &targets[1..3],
                        ),
                    },
                    Instruction {
                        address: 24,
                        predicate: None,
                        kind: K::Exit(ops::Exit { cc_test: 15 }),
                    },
                    Instruction {
                        address: 40,
                        predicate: None,
                        kind: mov(10, 0x1111).0,
                    },
                    Instruction {
                        address: 48,
                        predicate: None,
                        kind: K::Exit(ops::Exit { cc_test: 15 }),
                    },
                    Instruction {
                        address: 56,
                        predicate: None,
                        kind: K::Nop(ops::Nop),
                    },
                    Instruction {
                        address: 72,
                        predicate: None,
                        kind: mov(10, 0x2222).0,
                    },
                    Instruction {
                        address: 80,
                        predicate: None,
                        kind: K::Exit(ops::Exit { cc_test: 15 }),
                    },
                    Instruction {
                        address: 88,
                        predicate: None,
                        kind: K::Nop(ops::Nop),
                    },
                ];
                let spirv = emit_with_header(&xs, Some(&headers::parse(&[0; 32]).unwrap()))
                    .unwrap()
                    .words;
                assert_eq!(
                    run(&spirv)[10],
                    if predicate.is_some_and(|p| !p.inverted) {
                        0
                    } else {
                        expected
                    }
                );
                let output = spirv_validation::validate(&spirv);
                assert!(
                    output.status.success(),
                    "{}",
                    String::from_utf8_lossy(&output.stderr)
                );
            }
        }
    }
}

#[test]
fn lop3_immediate_fields_and_all_truth_tables() {
    let inputs = [0x596ac35a, 0xffff_81a5, 0xa623e741];
    for lut in 0..=255u64 {
        let raw = (0x3c00u64 << 48)
            | (lut << 48)
            | (1 << 56)
            | (7 << 16)
            | 1
            | (2 << 8)
            | (0x781a5 << 20)
            | (3 << 39);
        let kind = decoded(raw);
        assert!(
            !matches!(kind, K::Unknown(ops::Unknown { .. })),
            "LUT {lut:02x} rejected"
        );
        let words = module(vec![mov(2, inputs[0]), mov(3, inputs[2]), (kind, None)]);
        let expected = (0..32).fold(0, |result, bit| {
            let index = (((inputs[0] >> bit) & 1) << 2)
                | (((inputs[1] >> bit) & 1) << 1)
                | ((inputs[2] >> bit) & 1);
            result | ((((lut >> index) & 1) as u32) << bit)
        });
        assert_eq!(run(&words)[1], expected, "LUT {lut:02x}");
    }
}

#[test]
fn narrow_stores_preserve_neighbouring_bytes() {
    for shared in [false, true] {
        for (size, offset, expected) in [(0, 1, 0x1122_ab44), (2, 2, 0xabcd_3344)] {
            let words = module(vec![
                mov(10, 0),
                mov(1, 0x1122_3344),
                (memory(true, shared, 4, 1, 0), None),
                mov(2, if size == 0 { 0xab } else { 0xabcd }),
                (memory(true, shared, size, 2, offset), None),
                (memory(false, shared, 4, 3, 0), None),
            ]);
            assert_eq!(run(&words)[3], expected, "shared={shared} size={size}");
        }
    }
}

#[test]
fn b128_transfers_all_four_words() {
    for shared in [false, true] {
        let words = module(vec![
            mov(10, 0),
            mov(1, 11),
            mov(2, 22),
            mov(3, 33),
            mov(4, 44),
            (memory(true, shared, 6, 1, 0), None),
            (memory(false, shared, 6, 5, 0), None),
        ]);
        assert_eq!(run(&words)[5..9], [11, 22, 33, 44]);
    }
}

#[test]
fn false_predicates_suppress_memory_side_effects() {
    let false_pred = Some(Predicate {
        register: 0,
        inverted: false,
    });
    for shared in [false, true] {
        let words = module(vec![
            mov(10, 0),
            mov(1, 123),
            (memory(true, shared, 4, 1, 0), None),
            mov(2, 999),
            (memory(true, shared, 4, 2, 0), false_pred),
            (memory(false, shared, 4, 3, 0), None),
        ]);
        assert_eq!(run(&words)[3], 123);
    }
    let words = module(vec![
        mov(10, 0),
        mov(1, 123),
        (memory(true, true, 4, 1, 0), None),
        mov(2, 999),
        (
            K::Atomic(ops::Atomic {
                dst: R(4),
                address: R(10),
                offset: 0,
                data: R(2),
                op: AtomicOp::Exchange,
                signed: false,
            }),
            false_pred,
        ),
        (memory(false, true, 4, 3, 0), None),
    ]);
    assert_eq!(run(&words)[3], 123);
}

fn condition(test: u8, z: bool, s: bool, c: bool, o: bool) -> bool {
    match test {
        0 => false,
        1 => (s && !z) ^ o,
        2 => !s && z,
        3 => s ^ (z || o),
        4 => !((s ^ o) || z),
        5 => !z,
        6 => !(s ^ o),
        7 => !(s && z),
        8 => s && z,
        9 => s ^ o,
        10 => z,
        11 => (s ^ o) || z,
        12 => !s ^ (o || z),
        13 => s || !z,
        14 => (!s || z) ^ o,
        15 => true,
        16 => !o,
        17 => !c,
        18 => !s,
        19 => z || !c,
        20 => c && !z,
        21 => s,
        22 => c,
        23 => o,
        // Ryujinx Decoders/InstDecoders.cs:104-111 assigns RLE/RGT
        // to30/31. Selectors24..29 are CSM modes and remain unsupported.
        30 => s || z,
        31 => !(s || z),
        _ => unreachable!(),
    }
}

#[test]
fn bra_uses_condition_even_with_pt_and_matches_flag_formulas() {
    // Reach all attainable IADD flag states, then separately vary Z/N with LOP.CC.
    for (a, b) in [
        (0, 0),
        (1, 0),
        (u32::MAX, 1),
        (0x7fff_ffff, 1),
        (0x8000_0000, 0x8000_0000),
        (u32::MAX, u32::MAX),
    ] {
        for zn in [0, 1, 0x8000_0000] {
            for test in (0..=23).chain([30, 31]) {
                for guard in [
                    None,
                    Some(Predicate {
                        register: 7,
                        inverted: false,
                    }),
                ] {
                    let sum = u64::from(a) + u64::from(b);
                    let d = sum as u32;
                    let o = ((d ^ a) & !(a ^ b)) >> 31 != 0;
                    let expected = condition(test, zn == 0, zn >> 31 != 0, sum >> 32 != 0, o);
                    let words = module(vec![
                        mov(2, a),
                        mov(3, b),
                        (
                            K::IAdd(ops::IAdd {
                                dst: R(4),
                                lhs: R(2),
                                rhs: S::Register(R(3)),
                                cc: true,
                                x: false,
                                negate: [false; 2],
                            }),
                            None,
                        ),
                        (
                            K::Lop(ops::Lop {
                                dst: R(5),
                                lhs: R(2),
                                rhs: S::Immediate(zn),
                                op: LogicOp::PassB,
                                invert: [false; 2],
                                predicate_output: None,
                                extended: false,
                                cc: true,
                            }),
                            None,
                        ),
                        (
                            K::Branch(ops::Branch {
                                target: BranchTarget::reaching(32, 48),
                                cc_test: test,
                            }),
                            guard,
                        ),
                        mov(1, 1),
                    ]);
                    assert_eq!(
                        run(&words)[1],
                        u32::from(!expected),
                        "cc={test:x} a={a:x} b={b:x} zn={zn:x} guard={guard:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn bra_csm_selectors_take_the_reference_default_without_reclassifying_flags() {
    for selector in 24..=29_u8 {
        let raw = (0xe240_u64 << 48) | (16 << 20) | (7 << 16) | u64::from(selector);
        assert!(matches!(
            decoded(raw),
            K::Branch(ops::Branch { cc_test, .. }) if cc_test == selector
        ));
        for seed in [0_u32, 1, 0x8000_0000, u32::MAX] {
            let words = module(vec![
                mov(2, seed),
                mov(3, !seed),
                (
                    K::IAdd(ops::IAdd {
                        dst: R(4),
                        lhs: R(2),
                        rhs: S::Register(R(3)),
                        cc: true,
                        x: false,
                        negate: [false; 2],
                    }),
                    None,
                ),
                (
                    K::Branch(ops::Branch {
                        target: BranchTarget::reaching(24, 40),
                        cc_test: selector,
                    }),
                    None,
                ),
                mov(1, 1),
            ]);
            assert_eq!(run(&words)[1], 0, "selector={selector} seed={seed:08x}");
        }
    }
}

#[test]
fn extended_add_preserves_previous_zero_flag() {
    let words = module(vec![
        mov(2, 1),
        mov(3, 0),
        (
            K::IAdd(ops::IAdd {
                dst: R(4),
                lhs: R(2),
                rhs: S::Register(R(3)),
                cc: true,
                x: false,
                negate: [false; 2],
            }),
            None,
        ),
        (
            K::IAdd(ops::IAdd {
                dst: R(4),
                lhs: R(3),
                rhs: S::Register(R(3)),
                cc: true,
                x: true,
                negate: [false; 2],
            }),
            None,
        ),
        (
            K::Branch(ops::Branch {
                target: BranchTarget::reaching(32, 48),
                cc_test: 10,
            }),
            None,
        ),
        mov(1, 1),
    ]);
    assert_eq!(run(&words)[1], 1);
}

#[test]
fn ldl_decodes_all_widths_and_rejects_unimplemented_options() {
    for size in 0..=6u64 {
        let raw = (0xef40u64 << 48) | (size << 48) | (7 << 16);
        assert!(
            matches!(decoded(raw), K::LocalLoad(ops::LocalLoad { mem_type, .. }) if mem_type as u64 == size)
        );
    }
    let inverted_pt = decode_word([((7 << 16) | (1 << 19)) as u32, 0x50b00000], 0);
    assert_eq!(
        inverted_pt.predicate,
        Some(Predicate {
            register: 7,
            inverted: true
        })
    );
}

#[test]
fn stl_cache_five_decode_typed() {
    let cases = [
        (0xef50_1000_0007_0814, LsSize2::U8, LocalStoreCache::Cg),
        (0xef51_2000_0007_0814, LsSize2::S8, LocalStoreCache::Cs),
        (0xef54_3000_0007_0814, LsSize2::B32, LocalStoreCache::Wt),
        (0xef55_1000_0007_0814, LsSize2::B64, LocalStoreCache::Cg),
        (0xef56_2000_0007_0814, LsSize2::B128, LocalStoreCache::Cs),
    ];
    for (raw, size, cache) in cases {
        let kind = decoded(raw);
        assert!(matches!(
            kind,
            K::LocalStore(ops::LocalStore {
                data: R(20),
                address: R(8),
                offset: 0,
                mem_type,
                cache: actual,
            }) if mem_type == size && actual == cache
        ));
        let instruction = Instruction {
            address: 0,
            kind,
            predicate: None,
        };
        assert!(instruction
            .to_string()
            .starts_with(&format!("STL{} ", cache.suffix())));
        let mut instructions = vec![instruction];
        instructions.push(Instruction {
            address: 8,
            kind: K::Exit(ops::Exit { cc_test: 0xf }),
            predicate: None,
        });
        let mut emitted =
            emit_with_header(&instructions, Some(&headers::parse(&[0; 32]).unwrap())).unwrap();
        assert!(spirv_validation::validate(&emitted.words).status.success());
        emitted.optimize();
        assert!(spirv_validation::validate(&emitted.words).status.success());
    }
}

#[test]
fn stl_cache_next_five_decode_typed() {
    let cases = [
        (0xef50_2000_0007_0814, LsSize2::U8, LocalStoreCache::Cs),
        (0xef50_3000_0007_0814, LsSize2::U8, LocalStoreCache::Wt),
        (0xef51_1000_0007_0814, LsSize2::S8, LocalStoreCache::Cg),
        (0xef51_3000_0007_0814, LsSize2::S8, LocalStoreCache::Wt),
        (0xef52_1000_0007_0814, LsSize2::U16, LocalStoreCache::Cg),
    ];
    for (raw, size, cache) in cases {
        let kind = decoded(raw);
        assert!(matches!(
            kind,
            K::LocalStore(ops::LocalStore {
                data: R(20),
                address: R(8),
                offset: 0,
                mem_type,
                cache: actual,
            }) if mem_type == size && actual == cache
        ));
        let instruction = Instruction {
            address: 0,
            kind,
            predicate: None,
        };
        assert!(instruction
            .to_string()
            .starts_with(&format!("STL{} ", cache.suffix())));
        let mut instructions = vec![instruction];
        instructions.push(Instruction {
            address: 8,
            kind: K::Exit(ops::Exit { cc_test: 0xf }),
            predicate: None,
        });
        let mut emitted =
            emit_with_header(&instructions, Some(&headers::parse(&[0; 32]).unwrap())).unwrap();
        assert!(spirv_validation::validate(&emitted.words).status.success());
        emitted.optimize();
        assert!(spirv_validation::validate(&emitted.words).status.success());
    }
}

#[test]
fn stl_cache_third_five_decode_typed() {
    let cases = [
        (0xef52_2000_0007_0814, LsSize2::U16, LocalStoreCache::Cs),
        (0xef52_3000_0007_0814, LsSize2::U16, LocalStoreCache::Wt),
        (0xef53_1000_0007_0814, LsSize2::S16, LocalStoreCache::Cg),
        (0xef53_2000_0007_0814, LsSize2::S16, LocalStoreCache::Cs),
        (0xef53_3000_0007_0814, LsSize2::S16, LocalStoreCache::Wt),
    ];
    for (raw, size, cache) in cases {
        let kind = decoded(raw);
        assert!(matches!(
            kind,
            K::LocalStore(ops::LocalStore {
                data: R(20),
                address: R(8),
                offset: 0,
                mem_type,
                cache: actual,
            }) if mem_type == size && actual == cache
        ));
        let instruction = Instruction {
            address: 0,
            kind,
            predicate: None,
        };
        assert!(instruction
            .to_string()
            .starts_with(&format!("STL{} ", cache.suffix())));
        let mut instructions = vec![instruction];
        instructions.push(Instruction {
            address: 8,
            kind: K::Exit(ops::Exit { cc_test: 0xf }),
            predicate: None,
        });
        let mut emitted =
            emit_with_header(&instructions, Some(&headers::parse(&[0; 32]).unwrap())).unwrap();
        assert!(spirv_validation::validate(&emitted.words).status.success());
        emitted.optimize();
        assert!(spirv_validation::validate(&emitted.words).status.success());
    }
}

#[test]
fn stl_cache_wide_five_decode_typed() {
    let cases = [
        (0xef54_1000_0007_0814, LsSize2::B32, LocalStoreCache::Cg),
        (0xef54_2000_0007_0814, LsSize2::B32, LocalStoreCache::Cs),
        (0xef55_2000_0007_0814, LsSize2::B64, LocalStoreCache::Cs),
        (0xef55_3000_0007_0814, LsSize2::B64, LocalStoreCache::Wt),
        (0xef56_1000_0007_0814, LsSize2::B128, LocalStoreCache::Cg),
    ];
    for (raw, size, cache) in cases {
        let kind = decoded(raw);
        assert!(matches!(
            kind,
            K::LocalStore(ops::LocalStore {
                data: R(20),
                address: R(8),
                offset: 0,
                mem_type,
                cache: actual,
            }) if mem_type == size && actual == cache
        ));
        let instruction = Instruction {
            address: 0,
            kind,
            predicate: None,
        };
        assert!(instruction
            .to_string()
            .starts_with(&format!("STL{} ", cache.suffix())));
        let mut instructions = vec![instruction];
        instructions.push(Instruction {
            address: 8,
            kind: K::Exit(ops::Exit { cc_test: 0xf }),
            predicate: None,
        });
        let mut emitted =
            emit_with_header(&instructions, Some(&headers::parse(&[0; 32]).unwrap())).unwrap();
        assert!(spirv_validation::validate(&emitted.words).status.success());
        emitted.optimize();
        assert!(spirv_validation::validate(&emitted.words).status.success());
    }
}

#[test]
fn atoms_opcode_signedness_address_and_value_operands() {
    for (operation, signed, opcode) in [
        (AtomicOp::Add, false, Op::AtomicIAdd),
        (AtomicOp::Exchange, false, Op::AtomicExchange),
        (AtomicOp::And, false, Op::AtomicAnd),
        (AtomicOp::Or, false, Op::AtomicOr),
        (AtomicOp::Xor, false, Op::AtomicXor),
        (AtomicOp::Min, false, Op::AtomicUMin),
        (AtomicOp::Min, true, Op::AtomicSMin),
        (AtomicOp::Max, false, Op::AtomicUMax),
        (AtomicOp::Max, true, Op::AtomicSMax),
    ] {
        for offset in [0u32, 8, (-4i32) as u32] {
            let words = module(vec![
                mov(10, 12),
                mov(1, 0x8000_0012),
                (memory(true, true, 4, 1, offset as i32), None),
                mov(2, 7),
                (
                    K::Atomic(ops::Atomic {
                        dst: R(3),
                        address: R(10),
                        offset,
                        data: R(2),
                        op: operation,
                        signed,
                    }),
                    None,
                ),
                (memory(false, true, 4, 4, offset as i32), None),
            ]);
            let atomic = ops(&words)
                .into_iter()
                .find(|(op, _)| *op == opcode)
                .unwrap()
                .1;
            assert_eq!(atomic.len(), 6);
            assert_eq!(run(&words)[3], 0x8000_0012);
            let expected = match operation {
                AtomicOp::Add => 0x8000_0019,
                AtomicOp::Exchange => 7,
                AtomicOp::And => 2,
                AtomicOp::Or => 0x8000_0017,
                AtomicOp::Xor => 0x8000_0015,
                AtomicOp::Min if signed => 0x8000_0012,
                AtomicOp::Min => 7,
                AtomicOp::Max if signed => 7,
                AtomicOp::Max => 0x8000_0012,
                // old >= data wraps to 0; old > data reloads data.
                AtomicOp::Inc => 0,
                AtomicOp::Dec => 7,
            };
            assert_eq!(run(&words)[4], expected);
        }
    }
}

#[test]
fn shared_subword_stores_use_atomic_cas_with_replacement_before_comparator() {
    for size in 0..=3 {
        let words = module(vec![
            mov(10, 0),
            mov(1, 0x1122_3344),
            (memory(true, true, 4, 1, 0), None),
            mov(2, 0xab),
            (memory(true, true, size, 2, 0), None),
            (memory(false, true, 4, 3, 0), None),
        ]);
        let instructions = ops(&words);
        let cas = instructions
            .iter()
            .find(|(op, _)| *op == Op::AtomicCompareExchange)
            .expect("shared subword store needs CAS")
            .1;
        assert_eq!(cas.len(), 8);
        assert!(instructions
            .iter()
            .any(|(op, a)| *op == Op::AtomicLoad && a[1] == cas[7]));
        assert!(instructions
            .iter()
            .any(|(op, a)| *op == Op::BitwiseOr && a[1] == cas[6]));
        let expected = if size < 2 { 0x1122_33ab } else { 0x1122_00ab };
        assert_eq!(run(&words)[3], expected);
    }
}

#[test]
fn narrow_loads_extend_the_shifted_word_at_every_byte_offset() {
    for shared in [false, true] {
        for size in 0..=3 {
            for offset in 0..4 {
                let words = module(vec![
                    mov(10, 0),
                    mov(1, 0x80ff_7f81),
                    (memory(true, shared, 4, 1, 0), None),
                    (memory(false, shared, size, 2, offset), None),
                ]);
                let shifted = 0x80ff_7f81u32 >> (offset * 8);
                let expected = match size {
                    0 => shifted & 255,
                    1 => (shifted as i8) as u32,
                    2 => shifted & 65535,
                    3 => (shifted as i16) as u32,
                    _ => unreachable!(),
                };
                assert_eq!(run(&words)[2], expected);
            }
        }
    }
}

#[test]
fn rz_stays_zero_after_flag_writes_and_vector_loads() {
    for shared in [false, true] {
        let words = module(vec![
            mov(10, 0),
            mov(1, 42),
            mov(2, 99),
            (memory(true, shared, 5, 1, 0), None),
            (memory(false, shared, 5, 254, 0), None),
            (
                K::IAdd(ops::IAdd {
                    dst: R(255),
                    lhs: R(1),
                    rhs: S::Immediate(1),
                    cc: true,
                    x: false,
                    negate: [false; 2],
                }),
                None,
            ),
            (memory(true, shared, 4, 255, 8), None),
            (memory(false, shared, 4, 3, 8), None),
        ]);
        assert_eq!(run(&words)[254], 42);
        assert_eq!(run(&words)[255], 0);
        assert_eq!(run(&words)[3], 0);
    }
}

// Add an observation buffer only to test modules. The production entry point,
// instruction bodies and memory layouts are retained byte for byte.
// SPIR-V 1.6: TypeArray28, TypeStruct30, TypePointer32, AccessChain65,
// Block2, ArrayStride6, Binding33, DescriptorSet34, Offset35;
// /usr/include/spirv/unified1/spirv.hpp:461-496,1798-1802,1833.
fn observed_module(words: &[u32]) -> Vec<u32> {
    fn instruction(op: Op, args: &[u32]) -> Vec<u32> {
        let mut words = vec![((args.len() as u32 + 1) << 16) | u32::from(op)];
        words.extend_from_slice(args);
        words
    }
    let mut next = words[3];
    let mut id = || {
        let value = next;
        next += 1;
        value
    };
    let array = id();
    let structure = id();
    let pointer = id();
    let element_pointer = id();
    let output = id();
    let mut uint = 0;
    let mut register_pointers = Vec::new();
    let mut registers = Vec::new();
    let mut constants = HashMap::new();
    for (op, a) in ops(words) {
        match op {
            Op::TypeInt if a[1..] == [32, 0] => uint = a[0],
            Op::TypePointer if a[1..] == [7, uint] => register_pointers.push(a[0]),
            Op::Variable if register_pointers.contains(&a[0]) => registers.push(a[1]),
            Op::Constant if a[0] == uint => {
                constants.insert(a[2], a[1]);
            }
            _ => {}
        }
    }
    let mut globals = Vec::new();
    for value in 0..=256 {
        constants.entry(value).or_insert_with(|| {
            let result = id();
            globals.extend(instruction(Op::Constant, &[uint, result, value]));
            result
        });
    }
    globals.extend(instruction(Op::TypeArray, &[array, uint, constants[&256]]));
    globals.extend(instruction(Op::TypeStruct, &[structure, array]));
    globals.extend(instruction(Op::TypePointer, &[pointer, 12, structure]));
    globals.extend(instruction(Op::TypePointer, &[element_pointer, 12, uint]));
    globals.extend(instruction(Op::Variable, &[pointer, output, 12]));
    let annotations: Vec<u32> = [
        instruction(Op::Decorate, &[array, 6, 4]),
        instruction(Op::Decorate, &[structure, 2]),
        instruction(Op::MemberDecorate, &[structure, 0, 35, 0]),
        instruction(Op::Decorate, &[output, 34, 0]),
        instruction(Op::Decorate, &[output, 33, 0]),
    ]
    .concat();
    let mut result = words[..5].to_vec();
    let mut annotated = false;
    let mut declared = false;
    for (op, a) in ops(words) {
        if !annotated
            && matches!(
                op,
                Op::TypeVoid
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
            )
        {
            result.extend_from_slice(&annotations);
            annotated = true;
        }
        if !declared && op == Op::Function {
            result.extend_from_slice(&globals);
            declared = true;
        }
        if op == Op::Return {
            for (index, &register) in registers[..256].iter().enumerate() {
                let loaded = id();
                let destination = id();
                result.extend(instruction(Op::Load, &[uint, loaded, register]));
                result.extend(instruction(
                    Op::AccessChain,
                    &[
                        element_pointer,
                        destination,
                        output,
                        constants[&0],
                        constants[&(index as u32)],
                    ],
                ));
                result.extend(instruction(Op::Store, &[destination, loaded]));
            }
        }
        result.extend(instruction(op, a));
    }
    result[3] = next;
    result
}

#[test]
#[ignore = "requires Amber and a working Vulkan device; performs exact uint comparisons"]
fn exact_vulkan_memory_lop3_and_branch_comparisons() {
    use std::{fs, process::Command};
    let work = std::env::temp_dir().join("shadowbox-semantic-review-gpu");
    fs::create_dir_all(&work).unwrap();
    let mut cases = Vec::new();
    let conversions_only = std::env::var_os("GPU_CONVERSIONS_ONLY").is_some();
    if !conversions_only {
        for shared in [false, true] {
            for size in 0..=6 {
                let words = module(vec![
                    mov(10, 0),
                    mov(1, 0x80ff_7f81),
                    mov(2, 0xface_abcd),
                    mov(3, 33),
                    mov(4, 44),
                    (memory(true, shared, 6, 1, 0), None),
                    (
                        memory(true, shared, size, 2, if size < 2 { 1 } else { 0 }),
                        None,
                    ),
                    (
                        memory(false, shared, size, 5, if size < 2 { 1 } else { 0 }),
                        None,
                    ),
                    (memory(false, shared, 6, 20, 0), None),
                ]);
                cases.push(words);
            }
        }
        for (op, signed) in [
            (AtomicOp::Add, false),
            (AtomicOp::Min, false),
            (AtomicOp::Min, true),
            (AtomicOp::Max, false),
            (AtomicOp::Max, true),
            (AtomicOp::And, false),
            (AtomicOp::Or, false),
            (AtomicOp::Xor, false),
            (AtomicOp::Exchange, false),
        ] {
            for pred in [
                None,
                Some(Predicate {
                    register: 0,
                    inverted: false,
                }),
            ] {
                cases.push(module(vec![
                    mov(10, 12),
                    mov(1, 0x8000_0012),
                    (memory(true, true, 4, 1, -4), None),
                    mov(2, 7),
                    (
                        K::Atomic(ops::Atomic {
                            dst: R(3),
                            address: R(10),
                            offset: (-4i32) as u32,
                            data: R(2),
                            op,
                            signed,
                        }),
                        pred,
                    ),
                    (memory(false, true, 4, 4, -4), None),
                ]));
            }
        }
        for lut in 0..=255u64 {
            cases.push(module(vec![
                mov(2, 0x596a_c35a),
                mov(3, 0xa623_e741),
                (
                    decoded(
                        (0x3c00u64 << 48)
                            | (lut << 48)
                            | (1 << 56)
                            | (7 << 16)
                            | 1
                            | (2 << 8)
                            | (0x781a5 << 20)
                            | (3 << 39),
                    ),
                    None,
                ),
            ]));
        }
        for test in (0..=23).chain([30, 31]) {
            for (a, b) in [(0, 0), (u32::MAX, 1), (0x7fff_ffff, 1), (1, 0)] {
                cases.push(module(vec![
                    mov(2, a),
                    mov(3, b),
                    (
                        K::IAdd(ops::IAdd {
                            dst: R(255),
                            lhs: R(2),
                            rhs: S::Register(R(3)),
                            cc: true,
                            x: false,
                            negate: [false; 2],
                        }),
                        None,
                    ),
                    (
                        K::Branch(ops::Branch {
                            target: BranchTarget::reaching(24, 40),
                            cc_test: test,
                        }),
                        None,
                    ),
                    mov(1, 42),
                ]));
            }
        }
    }
    for signed in [false, true] {
        for round in 0..4u64 {
            for value in [
                1u32,
                0x8000_0001,
                0,
                0x8000_0000,
                0x3fc0_0000,
                0xbfc0_0000,
                0x7fc0_0123,
                0x7f80_0000,
                0xff80_0000,
                0x4f00_0000,
                0xcf00_0000,
            ] {
                cases.push(module(vec![
                    mov(2, value),
                    (
                        decoded(
                            (0x5cb0u64 << 48)
                                | (0xa << 8)
                                | (u64::from(signed) << 12)
                                | (round << 39)
                                | (2 << 20)
                                | (7 << 16)
                                | 1,
                        ),
                        None,
                    ),
                ]));
            }
        }
    }
    for size in 0..=2u64 {
        for modifiers in 0..4u64 {
            cases.push(module(vec![
                mov(2, 0x8000_807f),
                (
                    decoded(
                        (0x5cb8u64 << 48)
                            | (size << 10)
                            | (2 << 8)
                            | (1 << 13)
                            | ((modifiers & 1) << 49)
                            | ((modifiers >> 1) << 45)
                            | (2 << 20)
                            | (7 << 16)
                            | 1,
                    ),
                    None,
                ),
            ]));
        }
    }
    let mut script = String::from("#!amber\n");
    for (i, words) in cases.iter().enumerate() {
        let expected = run(words);
        let observed = observed_module(words);
        let path = work.join(format!("case-{i}.spv"));
        fs::write(
            &path,
            observed
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let validation = Command::new("spirv-val")
            .args(["--target-env", "vulkan1.2"])
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            validation.status.success(),
            "case {i}: {}",
            String::from_utf8_lossy(&validation.stderr)
        );
        script.push_str(&format!("SHADER compute s{i} SPIRV-BIN TARGET_ENV vulkan1.2 FILE case-{i}.spv\n\
            BUFFER out{i} DATA_TYPE uint32 SIZE 256 FILL 0\n\
            BUFFER ref{i} DATA_TYPE uint32 DATA {} END\n\
            PIPELINE compute p{i}\nATTACH s{i}\nBIND BUFFER out{i} AS storage DESCRIPTOR_SET 0 BINDING 0\nEND\n\
            RUN p{i} 1 1 1\nEXPECT out{i} EQ_BUFFER ref{i}\n",
            expected.iter().map(u32::to_string).collect::<Vec<_>>().join(" ")));
    }
    fs::write(work.join("review.amber"), script).unwrap();
    let amber = std::env::var("AMBER").unwrap_or_else(|_| "amber".into());
    let output = Command::new(amber)
        .current_dir(&work)
        .args(["-d", "review.amber"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    println!(
        "{} modules, {} exact uint comparisons passed on Vulkan",
        cases.len(),
        cases.len() * 256
    );
}

#[test]
fn bra_truth_table_covers_all_sixteen_flag_combinations() {
    for bits in 0..16u32 {
        for test in (0..=23).chain([30, 31]) {
            let mut words = module(vec![
                (
                    K::Branch(ops::Branch {
                        target: BranchTarget::reaching(0, 16),
                        cc_test: test,
                    }),
                    None,
                ),
                mov(1, 1),
            ]);
            let mut bool_type = 0;
            let mut bool_pointer = 0;
            let mut variables = Vec::new();
            let mut yes = 0;
            let mut no = 0;
            for (op, a) in ops(&words) {
                match op {
                    Op::TypeBool => bool_type = a[0],
                    Op::TypePointer if a[1..] == [7, bool_type] => bool_pointer = a[0],
                    Op::Variable if a[0] == bool_pointer => variables.push(a[1]),
                    Op::ConstantTrue if a[0] == bool_type => yes = a[1],
                    Op::ConstantFalse if a[0] == bool_type => no = a[1],
                    _ => {}
                }
            }
            // Seed the four CC inputs in an otherwise untouched branch module.
            // Z/N may both be set for the floating NaN flag representation.
            let mut at = 5;
            while at < words.len() {
                let count = (words[at] >> 16) as usize;
                if Op::of(words[at]) == Op::Store {
                    if let Some(index) = variables[8..12].iter().position(|id| *id == words[at + 1])
                    {
                        words[at + 2] = if bits & (1 << index) != 0 { yes } else { no };
                    }
                }
                at += count;
            }
            let expected = condition(
                test,
                bits & 1 != 0,
                bits & 2 != 0,
                bits & 4 != 0,
                bits & 8 != 0,
            );
            assert_eq!(
                run(&words)[1],
                u32::from(!expected),
                "CC={test:x} flags={bits:x}"
            );
        }
    }
}

#[test]
fn unsupported_memory_and_lop3_fields_remain_unknown() {
    for prefix in [0xef40u64, 0xef48, 0xef50, 0xef58] {
        for bit in 44..=47 {
            // Load hints decode (envytools gm107.c:602-607,1894-1895): LDL
            // cache LU/CI/CV in bits 44..45, LDS.U in bit 44. They do not
            // change the loaded value. Every other listed bit stays Unknown.
            // STL reads its cache in bits 44..45 and nvdisasm ignores 46..47
            // (Ryujinx InstTable.cs, MIT).
            let hint = (prefix == 0xef40 && bit <= 45)
                || (prefix == 0xef48 && bit == 44)
                || prefix == 0xef50;
            let kind = decoded((prefix << 48) | (4 << 48) | (1 << bit) | (7 << 16));
            if hint {
                assert!(
                    matches!(
                        kind,
                        K::LocalLoad(ops::LocalLoad { .. })
                            | K::SharedLoad(ops::SharedLoad { .. })
                            | K::LocalStore(ops::LocalStore { .. })
                    ),
                    "{prefix:x} bit {bit}"
                );
            } else {
                assert!(
                    matches!(kind, K::Unknown(ops::Unknown { .. })),
                    "{prefix:x} bit {bit}"
                );
            }
        }
        assert!(matches!(
            decoded((prefix << 48) | (7 << 48) | (7 << 16)),
            K::Unknown(ops::Unknown { .. })
        ));
    }
    assert!(matches!(
        decoded((0x3c00u64 << 48) | (1 << 57) | (7 << 16)),
        K::Unknown(ops::Unknown { .. })
    ));
}

#[test]
fn i2f_uses_integer_immediates() {
    let immediate =
        (0x38b8u64 << 48) | (1 << 56) | (0x7ffff << 20) | (1 << 13) | (0xa << 8) | (7 << 16) | 1;
    assert!(matches!(
        decoded(immediate),
        K::Convert(ops::Convert {
            src: S::Immediate(u32::MAX),
            ..
        })
    ));
    assert_eq!(
        run(&module(vec![(decoded(immediate), None)]))[1],
        (-1f32).to_bits()
    );
}

fn check_i2f_extraction(modifiers: std::ops::Range<u64>) {
    for size in 0..=2u64 {
        for byte in 0..=3u64 {
            // nvdisasm SM52: a 16-bit source selects H0 or H1.
            if byte * 8 + (8 << size) > 32 || byte % (1 << size) != 0 {
                continue;
            }
            for modifiers in modifiers.clone() {
                let raw = (0x5cb8u64 << 48)
                    | (1 << 13)
                    | (size << 10)
                    | (2 << 8)
                    | (byte << 41)
                    | ((modifiers & 1) << 49)
                    | ((modifiers >> 1) << 45)
                    | (7 << 16)
                    | (2 << 20)
                    | 1;
                let value = 0x8000_807fu32 as i32;
                let value = if modifiers & 1 != 0 {
                    value.wrapping_abs()
                } else {
                    value
                };
                let value = if modifiers & 2 != 0 {
                    value.wrapping_neg()
                } else {
                    value
                };
                let shift = 32 - (8 << size);
                let extracted = ((((value as u32) >> (byte * 8)) << shift) as i32) >> shift;
                let words = module(vec![mov(2, 0x8000_807f), (decoded(raw), None)]);
                assert_eq!(
                    run(&words)[1],
                    (extracted as f32).to_bits(),
                    "size={size} byte={byte} modifiers={modifiers}"
                );
            }
        }
    }
}

#[test]
fn i2f_uses_the_extracted_byte_or_half() {
    check_i2f_extraction(0..1);
}

#[test]
fn i2f_modifies_the_full_word_before_extracting() {
    check_i2f_extraction(1..4);
}

#[test]
fn unmerged_bfe_forms_stay_unknown() {
    // main already merged BFE. Keep all three forms and the rejection check:
    // envytools gm107.c:2017,2091,2137 encodes CC47; decoder does not model it.
    for prefix in [0x5c00u64, 0x4c00, 0x3800] {
        assert!(matches!(
            decoded((prefix << 48) | (7 << 16)),
            K::BitFieldExtract(ops::BitFieldExtract { .. })
        ));
        assert!(matches!(
            decoded((prefix << 48) | (7 << 16) | (1 << 47)),
            K::Unknown(ops::Unknown { .. })
        ));
    }
}

#[test]
fn conversions_reject_unimplemented_flags() {
    for prefix in [0x5cb8u64, 0x5cb0, 0x5ca8] {
        assert!(matches!(
            decoded((prefix << 48) | (0xa << 8) | (7 << 16) | (1 << 47)),
            K::Unknown(ops::Unknown { .. })
        ));
    }
}

#[test]
fn f2i_sanitizes_nan_before_the_float_to_integer_instruction() {
    let words = module(vec![
        mov(2, 0x7fc0_0123),
        (
            decoded((0x5cb0u64 << 48) | (0xa << 8) | (1 << 12) | (2 << 20) | (7 << 16) | 1),
            None,
        ),
    ]);
    let instructions = ops(&words);
    let cast = instructions
        .iter()
        .find(|(op, _)| *op == Op::ConvertFToS)
        .unwrap()
        .1;
    let clamp = instructions
        .iter()
        .find(|(op, a)| *op == Op::ExtInst && a[1] == cast[2])
        .unwrap()
        .1;
    let finite = instructions
        .iter()
        .find(|(op, a)| *op == Op::Select && a[1] == clamp[4])
        .expect("NaN must be selected away before FClamp and ConvertFToS");
    assert!(instructions
        .iter()
        .any(|(op, a)| *op == Op::IsNan && a[1] == finite.1[2]));
}

#[test]
fn f2i_rounding_saturation_nan_and_subnormal_endpoints() {
    for signed in [false, true] {
        for round in 0..4u64 {
            for value in [
                1u32,
                0x8000_0001,
                0x3fc0_0000,
                0xbfc0_0000,
                0x7fc0_0123,
                0x7f80_0000,
                0xff80_0000,
                0x4f00_0000,
                0xcf00_0000,
            ] {
                let raw = (0x5cb0u64 << 48)
                    | (0xa << 8)
                    | (u64::from(signed) << 12)
                    | (round << 39)
                    | (2 << 20)
                    | (7 << 16)
                    | 1;
                let f = f32::from_bits(value);
                let rounded = match round {
                    0 => f.round_ties_even(),
                    1 => f.floor(),
                    2 => f.ceil(),
                    3 => f.trunc(),
                    _ => unreachable!(),
                };
                let expected = if signed {
                    (rounded as i32) as u32
                } else {
                    rounded as u32
                };
                let words = module(vec![mov(2, value), (decoded(raw), None)]);
                assert_eq!(
                    run(&words)[1],
                    expected,
                    "signed={signed} round={round} value={value:x}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires Amber and Vulkan; uses dynamic uniform input to prevent constant folding"]
fn dynamic_conversion_subnormal_rounding_matches_exact_results() {
    use std::{fs, process::Command};
    let work = std::env::temp_dir().join("shadowbox-semantic-review-dynamic");
    fs::create_dir_all(&work).unwrap();
    let mut script = String::from("#!amber\n");
    let mut count = 0;
    for (float_result, signed, ftz) in [
        (false, false, false),
        (false, true, false),
        (false, false, true),
        (false, true, true),
        (true, false, false),
        (true, false, true),
    ] {
        for round in 0..4u64 {
            for value in [1u32, 0x8000_0001] {
                let prefix = if float_result { 0x5ca8u64 } else { 0x5cb0 };
                let K::Convert(ops::Convert { kind, .. }) = decoded(
                    (prefix << 48)
                        | (0xa << 8)
                        | (u64::from(signed) << 12)
                        | (round << 39)
                        | (u64::from(float_result) << 42)
                        | (u64::from(ftz) << 44)
                        | (7 << 16)
                        | 1,
                ) else {
                    panic!()
                };
                let words = observed_module(&module(vec![(
                    K::Convert(ops::Convert {
                        kind,
                        dst: R(1),
                        src: S::ConstantBuffer {
                            binding: 2,
                            offset: 0,
                        },
                    }),
                    None,
                )]));
                let f = f32::from_bits(if ftz { value & 0x8000_0000 } else { value });
                let f = match round {
                    0 => f.round_ties_even(),
                    1 => f.floor(),
                    2 => f.ceil(),
                    3 => f.trunc(),
                    _ => unreachable!(),
                };
                let expected = if float_result {
                    f.to_bits()
                } else if signed {
                    (f as i32) as u32
                } else {
                    f as u32
                };
                let i = count;
                let path = work.join(format!("case-{i}.spv"));
                fs::write(
                    &path,
                    words
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
                let validation = Command::new("spirv-val")
                    .args(["--target-env", "vulkan1.2"])
                    .arg(&path)
                    .output()
                    .unwrap();
                assert!(
                    validation.status.success(),
                    "{}",
                    String::from_utf8_lossy(&validation.stderr)
                );
                let expected_words: [u32; 256] =
                    std::array::from_fn(|r| if r == 1 { expected } else { 0 });
                script.push_str(&format!(
                    "SHADER compute s{i} SPIRV-BIN TARGET_ENV vulkan1.2 FILE case-{i}.spv\n\
                    BUFFER out{i} DATA_TYPE uint32 SIZE 256 FILL 0\n\
                    BUFFER ref{i} DATA_TYPE uint32 DATA {} END\n\
                    BUFFER in{i} DATA_TYPE vec4<uint32> STD140 DATA {value} 0 0 0 END\n\
                    PIPELINE compute p{i}\nATTACH s{i}\n\
                    BIND BUFFER out{i} AS storage DESCRIPTOR_SET 0 BINDING 0\n\
                    BIND BUFFER in{i} AS uniform DESCRIPTOR_SET 0 BINDING 2\nEND\n\
                    RUN p{i} 1 1 1\nEXPECT out{i} EQ_BUFFER ref{i}\n",
                    expected_words
                        .iter()
                        .map(u32::to_string)
                        .collect::<Vec<_>>()
                        .join(" ")
                ));
                count += 1;
            }
        }
    }
    fs::write(work.join("dynamic.amber"), script).unwrap();
    let amber = std::env::var("AMBER").unwrap_or_else(|_| "amber".into());
    let output = Command::new(amber)
        .current_dir(&work)
        .args(["-d", "dynamic.amber"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    println!("{count} dynamic subnormal conversions passed exact Vulkan comparisons");
}

#[test]
fn f2i_ftz_flushes_before_rounding() {
    for signed in [false, true] {
        for round in 0..4u64 {
            for value in [1u32, 0x8000_0001] {
                let raw = (0x5cb0u64 << 48)
                    | (0xa << 8)
                    | (u64::from(signed) << 12)
                    | (round << 39)
                    | (1 << 44)
                    | (2 << 20)
                    | (7 << 16)
                    | 1;
                let words = module(vec![mov(2, value), (decoded(raw), None)]);
                assert_eq!(run(&words)[1], 0);
            }
        }
    }
}

#[test]
fn inverted_pt_is_preserved_as_a_false_guard() {
    let instruction = decode_word([((7 << 16) | (1 << 19)) as u32, 0x50b0_0000], 0);
    assert_eq!(
        instruction.predicate,
        Some(Predicate {
            register: 7,
            inverted: true
        })
    );
}

#[test]
fn f2f_ftz_and_directed_rounding_preserve_signed_zero() {
    for ftz in [false, true] {
        for round in 0..4u64 {
            for value in [1u32, 0x8000_0001, 0x3fc0_0000, 0xbfc0_0000] {
                let raw = (0x5ca8u64 << 48)
                    | (0xa << 8)
                    | (round << 39)
                    | (1 << 42)
                    | (u64::from(ftz) << 44)
                    | (2 << 20)
                    | (7 << 16)
                    | 1;
                let f = if ftz && value & 0x7f80_0000 == 0 {
                    f32::from_bits(value & 0x8000_0000)
                } else {
                    f32::from_bits(value)
                };
                let expected = match round {
                    0 => f.round_ties_even(),
                    1 => f.floor(),
                    2 => f.ceil(),
                    3 => f.trunc(),
                    _ => unreachable!(),
                };
                let words = module(vec![mov(2, value), (decoded(raw), None)]);
                assert_eq!(run(&words)[1], expected.to_bits());
            }
        }
    }
}

// Independent composition case reserved after the family fixes: an address
// register is also a vector destination, then LOP3.CC writes RZ while carrying
// IADD's C into BRA.HI. References: Ryujinx InstEmitMemory.cs:308-356,
// InstEmitIntegerLogical.cs:118-136, InstEmitConditionCode.cs:76.
#[test]
fn composed_vector_lop3_rz_and_bra_preserve_address_and_carry() {
    let raw_lop3 =
        (0x3c00u64 << 48) | (0xf0 << 48) | (1 << 47) | (2 << 39) | (10 << 8) | (7 << 16) | 255;
    let words = module(vec![
        mov(10, 0),
        mov(1, 16),
        mov(2, 77),
        (memory(true, true, 5, 1, 0), None),
        (memory(false, true, 5, 10, 0), None),
        mov(3, u32::MAX),
        (
            K::IAdd(ops::IAdd {
                dst: R(4),
                lhs: R(3),
                rhs: S::Immediate(1),
                cc: true,
                x: false,
                negate: [false; 2],
            }),
            None,
        ),
        (decoded(raw_lop3), None),
        (
            K::Branch(ops::Branch {
                target: BranchTarget::reaching(64, 80),
                cc_test: 0x14,
            }),
            None,
        ),
        mov(5, 1),
    ]);
    let result = run(&words);
    assert_eq!((result[10], result[11]), (16, 77));
    assert_eq!((result[4], result[5], result[255]), (0, 0, 0));
}

#[test]
fn inactive_load_is_guarded_before_pointer_construction() {
    let words = module(vec![(
        memory(false, true, 4, 1, 0),
        Some(Predicate {
            register: 0,
            inverted: false,
        }),
    )]);
    let instructions = ops(&words);
    let mut block = 0;
    let mut access_blocks = Vec::new();
    for (opcode, args) in &instructions {
        if *opcode == Op::Label {
            block = args[0];
        }
        if *opcode == Op::AccessChain {
            access_blocks.push(block);
        }
    }
    assert!(!access_blocks.is_empty());
    for access in access_blocks {
        assert!(
            instructions
                .iter()
                .any(|(opcode, args)| { *opcode == Op::BranchConditional && args[1] == access }),
            "load pointer was formed outside its predicate body"
        );
    }
}

#[test]
fn bfi_immediate_inserts_only_the_selected_bits() {
    // Ryujinx InstEmitBitfield.cs:154-162; encoding gm107.c:2139,253.
    for offset in 0..=32u64 {
        for count in 0..=32 - offset {
            for (insert, base) in [(0, u32::MAX), (u32::MAX, 0), (0x596a_c35a, 0xa623_e741)] {
                let raw = (0x36f0u64 << 48)
                    | 1
                    | (2 << 8)
                    | (7 << 16)
                    | (offset << 20)
                    | (count << 28)
                    | (3 << 39);
                let words = module(vec![mov(2, insert), mov(3, base), (decoded(raw), None)]);
                let mask = (((1u64 << count) - 1) << offset) as u32;
                let expected = (base & !mask) | (((u64::from(insert) << offset) as u32) & mask);
                assert_eq!(run(&words)[1], expected, "offset={offset} count={count}");
                assert_eq!(
                    ops(&words)
                        .iter()
                        .filter(|(op, _)| *op == Op::BitFieldInsert)
                        .count(),
                    1
                );
                // No runtime extraction of the immediate control operand.
                assert!(!ops(&words)
                    .iter()
                    .any(|(op, _)| matches!(op, Op::BitFieldSExtract | Op::BitFieldUExtract)));
            }
        }
    }
    let plain = (0x36f0u64 << 48) | 1 | (2 << 8) | (7 << 16) | (8 << 28) | (3 << 39);
    for extra in [33 << 20, 33 << 28, 1 << 36, 1 << 47, 1 << 48, 1 << 56] {
        assert!(matches!(
            decoded(plain | extra),
            K::Unknown(ops::Unknown { .. })
        ));
    }
}

#[test]
fn csetp_matches_both_condition_outputs_and_source_aliases() {
    // Encoding gm107.c:2058,1647-1680,1430-1434,368-369,373,406.
    // Ryujinx InstEmitConditionCode.cs:35-81 computes the complement
    // before combining either output with the source predicate.
    for selector in 24..=29u64 {
        let raw = (0x50a0u64 << 48) | (selector << 8) | (7 << 16);
        assert!(matches!(
            decoded(raw),
            K::CSetP(ops::CSetP { condition, .. }) if condition.selector() == selector as u8
        ));
    }
    for (a, b) in [
        (0, 0),
        (1, 0),
        (u32::MAX, 1),
        (0x7fff_ffff, 1),
        (0x8000_0000, 0x8000_0000),
        (u32::MAX, u32::MAX),
    ] {
        for zn in [0, 1, 0x8000_0000] {
            for test in 0..=31u64 {
                for bop in 0..3u64 {
                    for initial in [false, true] {
                        for invert in [false, true] {
                            let raw = (0x50a0u64 << 48)
                                | 1
                                | (test << 8)
                                | (7 << 16)
                                | (u64::from(invert) << 42)
                                | (bop << 45);
                            // Destination P0 is also the source, P1 is the complement.
                            let sum = u64::from(a) + u64::from(b);
                            let overflow = (((sum as u32) ^ a) & !(a ^ b)) >> 31 != 0;
                            let cc = if (24..=29).contains(&test) {
                                // Ryujinx InstEmitConditionCode.cs:52-82 has no
                                // CSM case, so GetCondition retains true.
                                true
                            } else {
                                condition(
                                    // Native RLE/RGT selectors30/31, Ryujinx
                                    // Decoders/InstDecoders.cs:104-111.
                                    test as u8,
                                    zn == 0,
                                    zn >> 31 != 0,
                                    sum >> 32 != 0,
                                    overflow,
                                )
                            };
                            let acc = initial ^ invert;
                            let combine = |v| match bop {
                                0 => v && acc,
                                1 => v || acc,
                                _ => v ^ acc,
                            };
                            let words = module(vec![
                                mov(2, a),
                                mov(3, b),
                                mov(9, u32::from(initial)),
                                (
                                    K::R2P(ops::R2P {
                                        src: R(9),
                                        mask: S::Immediate(3),
                                        byte: 0,
                                        bank: shadowbox::ir::FlagBank::Predicates,
                                    }),
                                    None,
                                ),
                                (
                                    K::IAdd(ops::IAdd {
                                        dst: R(4),
                                        lhs: R(2),
                                        rhs: S::Register(R(3)),
                                        cc: true,
                                        x: false,
                                        negate: [false; 2],
                                    }),
                                    None,
                                ),
                                (
                                    K::Lop(ops::Lop {
                                        dst: R(5),
                                        lhs: R(2),
                                        rhs: S::Immediate(zn),
                                        op: LogicOp::PassB,
                                        invert: [false; 2],
                                        predicate_output: None,
                                        extended: false,
                                        cc: true,
                                    }),
                                    None,
                                ),
                                (decoded(raw), None),
                                mov(6, 1),
                                (
                                    K::Sel(ops::Sel {
                                        dst: R(10),
                                        when_true: R(6),
                                        when_false: S::Immediate(0),
                                        condition: 0,
                                        condition_negate: false,
                                    }),
                                    None,
                                ),
                                (
                                    K::Sel(ops::Sel {
                                        dst: R(11),
                                        when_true: R(6),
                                        when_false: S::Immediate(0),
                                        condition: 1,
                                        condition_negate: false,
                                    }),
                                    None,
                                ),
                            ]);
                            assert_eq!(
                                run(&words)[10..12],
                                [u32::from(combine(cc)), u32::from(combine(!cc))],
                                "test={test} bop={bop} initial={initial} invert={invert}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn csetp_snapshots_an_aliased_execution_predicate() {
    // Ryujinx Translation/Translator.cs:339-385: whole-instruction guard.
    for initial in [false, true] {
        for inverted in [false, true] {
            let raw = (0x50a0u64 << 48) | 1 | (7 << 39); // F AND PT, P0/P1
            let words = module(vec![
                mov(9, u32::from(initial)),
                (
                    K::R2P(ops::R2P {
                        src: R(9),
                        mask: S::Immediate(3),
                        byte: 0,
                        bank: shadowbox::ir::FlagBank::Predicates,
                    }),
                    None,
                ),
                (
                    decoded(raw),
                    Some(Predicate {
                        register: 0,
                        inverted,
                    }),
                ),
                mov(6, 1),
                (
                    K::Sel(ops::Sel {
                        dst: R(10),
                        when_true: R(6),
                        when_false: S::Immediate(0),
                        condition: 0,
                        condition_negate: false,
                    }),
                    None,
                ),
                (
                    K::Sel(ops::Sel {
                        dst: R(11),
                        when_true: R(6),
                        when_false: S::Immediate(0),
                        condition: 1,
                        condition_negate: false,
                    }),
                    None,
                ),
            ]);
            let expected = if initial ^ inverted {
                [0, 1]
            } else {
                [u32::from(initial), 0]
            };
            assert_eq!(run(&words)[10..12], expected);
        }
    }
}

#[test]
fn f2f_modifiers_apply_before_rounding_and_saturation() {
    // Mesa sm50.rs:1765-1794; Ryujinx InstEmitConversion.cs:132-160.
    for abs in [false, true] {
        for negate in [false, true] {
            for saturate in [false, true] {
                for ftz in [false, true] {
                    for round in 0..5u64 {
                        for value in [
                            0u32,
                            0x8000_0000,
                            1,
                            0x8000_0001,
                            0x3fc0_0000,
                            0xbfc0_0000,
                            0x3f00_0000,
                            0xbf00_0000,
                            0x7f80_0000,
                            0xff80_0000,
                        ] {
                            let raw = (0x5ca8u64 << 48)
                                | 1
                                | (0xa << 8)
                                | (7 << 16)
                                | (2 << 20)
                                | (u64::from(abs) << 49)
                                | (u64::from(negate) << 45)
                                | (u64::from(saturate) << 50)
                                | (u64::from(ftz) << 44)
                                | if round == 4 {
                                    0
                                } else {
                                    (1 << 42) | (round << 39)
                                };
                            let mut bits = if abs { value & 0x7fff_ffff } else { value };
                            if negate {
                                bits ^= 0x8000_0000;
                            }
                            if ftz && bits & 0x7f80_0000 == 0 {
                                bits &= 0x8000_0000;
                            }
                            let f = f32::from_bits(bits);
                            let expected = match round {
                                0 => f.round_ties_even(),
                                1 => f.floor(),
                                2 => f.ceil(),
                                3 => f.trunc(),
                                _ => f,
                            };
                            // Mesa nir_opcodes.py:265 and util/u_math.h:
                            // 343-371 define min/max-number. Saturated -0 is
                            // +0; Rust's bare clamp preserves -0 instead.
                            let expected = if saturate {
                                if expected.is_nan() || expected <= 0.0 {
                                    0.0
                                } else {
                                    expected.min(1.0)
                                }
                            } else {
                                expected
                            };
                            let words = module(vec![mov(2, value), (decoded(raw), None)]);
                            assert_eq!(
                                run(&words)[1],
                                expected.to_bits(),
                                "raw={raw:x} input={value:x}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn group_a_unsupported_fields_stay_unknown() {
    // VTG row/table: gm107.c:2053,1633-1637. Ryujinx InstEmit.cs:330-336
    // has no emitted operation. Undefined modes remain Unknown; the policy
    // exception for R/A/RA is tested in votevtg.rs, never as VOTE or Nop.
    for mode in 3..8u64 {
        assert!(matches!(
            decoded(((0x50e0 | mode) << 48) | (7 << 16) | (0xffffff << 20)),
            K::Unknown(ops::Unknown { .. })
        ));
    }
    let csetp = (0x50a0u64 << 48) | (7 << 16) | (7 << 39);
    for extra in [3 << 45, 1 << 47, 1 << 20, 1 << 48] {
        assert!(matches!(
            decoded(csetp | extra),
            K::Unknown(ops::Unknown { .. })
        ));
    }
    let f2f = (0x5caau64 << 48) | (7 << 16) | (0xa << 8) | (2 << 20);
    for extra in [1 << 47, 1 << 12, 1 << 28, 1 << 48] {
        assert!(matches!(
            decoded(f2f | extra),
            K::Unknown(ops::Unknown { .. })
        ));
    }
}

#[test]
fn csetp_xor_uses_one_xor_and_one_complement() {
    // Ryujinx InstEmitConditionCode.cs:40-44 has complement + two XORs.
    // Use a constant condition to count only the two-output combination.
    let raw = (0x50a0u64 << 48) | 1 | (15 << 8) | (2 << 45);
    let words = module(vec![(decoded(raw), None)]);
    assert_eq!(
        ops(&words)
            .iter()
            .filter(|(op, _)| *op == Op::LogicalNotEqual)
            .count(),
        1
    );
    assert_eq!(
        ops(&words)
            .iter()
            .filter(|(op, _)| *op == Op::LogicalNot)
            .count(),
        1
    );
}

#[test]
fn i2f_register_outputs_have_exact_half_float_and_double_bits() {
    // Encoding: envytools gm107.c:1995,1256-1280; Mesa sm50.rs:1851-1879.
    // Semantics: Ryujinx InstEmitConversion.cs:257-281,398-415.
    // These exact values distinguish signed/unsigned, nearest-even rounding,
    // half overflow, zero high half, and the two-word double destination.
    for (input, signed, half, float, double) in [
        (u32::MAX, true, 0xbc00, 0xbf800000, 0xbff0000000000000u64),
        (u32::MAX, false, 0x7c00, 0x4f800000, 0x41efffffffe00000),
        (16777217, true, 0x7c00, 0x4b800000, 0x4170000010000000),
        (32767, true, 0x7800, 0x46fffe00, 0x40dfffc000000000),
        (0, false, 0, 0, 0),
    ] {
        for size in 1..=3u64 {
            let raw = 0x5cb8u64 << 48
                | 7 << 16
                | 2 << 20
                | size << 8
                | 2 << 10
                | u64::from(signed) << 13
                | 4;
            let words = module(vec![mov(2, input), (decoded(raw), None)]);
            let regs = run(&words);
            match size {
                1 => assert_eq!(regs[4], half),
                2 => assert_eq!(regs[4], float),
                3 => assert_eq!(&regs[4..6], &[double as u32, (double >> 32) as u32]),
                _ => unreachable!(),
            }
        }
    }
}

#[test]
fn i2f_unnamed_fields_do_not_disappear() {
    // Unnamed fields in envytools gm107.c:1995,2069,2115. All named
    // modifiers are checked by the exact-value tests above and below.
    for prefix in [0x5cb8u64, 0x4cb8, 0x38b8] {
        let raw = prefix << 48 | 7 << 16 | 2 << 8 | 2 << 10;
        for bit in [12, 14, 15, 43, 44, 46, 48, 50] {
            assert!(
                matches!(decoded(raw | 1 << bit), K::Unknown(ops::Unknown { .. })),
                "{prefix:x} bit{bit}"
            );
        }
        if prefix == 0x5cb8 {
            for bit in 28..39 {
                assert!(
                    matches!(decoded(raw | 1 << bit), K::Unknown(ops::Unknown { .. })),
                    "register bit{bit}"
                );
            }
        }
    }
}

// BFI.RC: envytools gm107.c:2041,203-207,258,302-309; Mesa
// nv50_ir_emit_gm107.cpp:2314-2326. The packed control is register39,
// and the base is the cbuf operand, per Ryujinx InstEmitBitfield.cs:75-83.
fn bfi_rc(dst: u8, insert: u8, control: u8, bank: u8, offset: u16) -> u64 {
    (0x53f0u64 << 48)
        | u64::from(dst)
        | (u64::from(insert) << 8)
        | (7 << 16)
        | (u64::from(offset / 4) << 20)
        | (u64::from(bank) << 34)
        | (u64::from(control) << 39)
}

#[test]
fn bfi_rc_fields_and_exact_values() {
    let raw = bfi_rc(5, 6, 7, 19, 0xfffc);
    assert_eq!(
        decoded(raw),
        K::BitFieldInsert(ops::BitFieldInsert {
            dst: R(5),
            insert: R(6),
            base: S::ConstantBuffer {
                binding: 19,
                offset: 0xfffc
            },
            range: S::Register(R(7)),
        })
    );
    // Every unnamed/modifier bit is rejected, including CC and low prefix
    // bits48..50. Every operand field is independently represented above.
    for bit in 47..=50 {
        assert!(matches!(
            decoded(raw | (1 << bit)),
            K::Unknown(ops::Unknown { .. })
        ));
    }
    let predicated = shadowbox::decoder::decode_word(
        [
            ((raw & !(15 << 16)) | (3 << 16) | (1 << 19)) as u32,
            (raw >> 32) as u32,
        ],
        0,
    );
    assert_eq!(
        predicated.predicate,
        Some(Predicate {
            register: 3,
            inverted: true
        })
    );
    for offset in 0..=32u32 {
        for count in 0..=32 - offset {
            for (insert, base) in [(0, u32::MAX), (u32::MAX, 0), (0x596a_c35a, 0xa623_e741)] {
                // Runtime control upper16 has no meaning (Ryujinx
                // InstEmitBitfield.cs:156-158); preserve the instruction fields.
                let control = 0xdead_0000 | offset | (count << 8);
                let kind = decoded(bfi_rc(1, 2, 3, 19, 0xfffc));
                let words = module(vec![mov(2, insert), mov(3, control), (kind, None)]);
                let mask = (((1u64 << count) - 1) << offset) as u32;
                let expected = (base & !mask) | (((u64::from(insert) << offset) as u32) & mask);
                assert_eq!(
                    execute(&words, base).registers[1],
                    expected,
                    "offset={offset}, count={count}"
                );
                assert_eq!(
                    ops(&words)
                        .iter()
                        .filter(|(op, _)| *op == Op::BitFieldInsert)
                        .count(),
                    1
                );
                assert!(!ops(&words)
                    .iter()
                    .any(|(op, _)| matches!(op, Op::BitFieldSExtract | Op::BitFieldUExtract)));
            }
        }
    }
    // Destination aliases insertion or control: snapshot before the store.
    for dst in [2, 3] {
        let words = module(vec![
            mov(2, 0x1234_5678),
            mov(3, 0x0804),
            (decoded(bfi_rc(dst, 2, 3, 0, 0)), None),
        ]);
        assert_eq!(
            execute(&words, 0x8765_4321).registers[dst as usize],
            0x8765_4781
        );
    }
}

fn encoded_stream(raw: &[u64]) -> Vec<Instruction> {
    // Project-owned bundles; Mesa sm50.rs:3418-3428. Pad with actual OpNop
    // (sm50.rs:3178-3184) rather than dropping instructions.
    let mut raw = raw.to_vec();
    while !raw.len().is_multiple_of(3) {
        raw.push((0x50b0u64 << 48) | (7 << 16));
    }
    let words: Vec<_> = raw
        .chunks(3)
        .flat_map(|c| [0, c[0], c[1], c[2]])
        .flat_map(|word| [word as u32, (word >> 32) as u32])
        .collect();
    shadowbox::decoder::decode_stream(shadowbox::QuietRun, &words)
}

#[test]
fn bfi_rc_stream_proof_and_rejection() {
    // MOV32I dst0..8, value20..52: Mesa sm50.rs:681-688.
    let mov_control = |value: u64| (0x0100u64 << 48) | (15 << 12) | 3 | (7 << 16) | (value << 20);
    let raw = bfi_rc(1, 2, 3, 0, 0);
    for value in [0, 0x2000, 0x011f, 0x0020, 0xdead_0804] {
        let xs = encoded_stream(&[mov_control(value), raw]);
        assert!(matches!(
            xs[1].kind,
            K::BitFieldInsert(ops::BitFieldInsert {
                range: S::Immediate(_),
                ..
            })
        ));
        let header = headers::parse(&[0; 32]).unwrap();
        let words = shadowbox::emitter::emit_with_header(&xs, Some(&header))
            .unwrap()
            .words;
        let result = spirv_validation::validate(&words);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    for value in [33, 0x2100, 0x021f] {
        let xs = encoded_stream(&[mov_control(value), raw]);
        assert!(matches!(
            xs[1].kind,
            K::OverlongInsert(ops::OverlongInsert { control, .. }) if u64::from(control) == value
        ));
    }
    assert!(matches!(
        encoded_stream(&[raw])[0].kind,
        K::BitFieldInsert(ops::BitFieldInsert {
            range: S::Register(_),
            ..
        })
    ));
    let xs = encoded_stream(&[bfi_rc(1, 2, 255, 0, 0)]);
    assert!(matches!(
        xs[0].kind,
        K::BitFieldInsert(ops::BitFieldInsert {
            range: S::Immediate(0),
            ..
        })
    ));
}

fn builtin(words: &[u32], value: u32) -> Option<u32> {
    ops(words)
        .iter()
        .find(|(op, a)| *op == Op::Decorate && a[1..] == [11, value])
        .map(|(_, a)| a[0])
}

#[test]
fn fragment_sample_mask_register_layout_depth_and_discard() {
    // Header: Ryujinx ShaderHeader.cs:142,162-167, Mesa sph.rs:185-188;
    // cla097sph.h:628-633. Register layout: Mesa from_nir.rs:3007-3030.
    // Vulkan SampleMask VUID-04359 requires an array of 32-bit integers.
    for targets in [0u32, 0xf, 0xf0f0, 0xffff_ffff, 0x0010_0001] {
        for mask in [false, true] {
            for depth in [false, true] {
                for kill in 0..4 {
                    let mut header_words = [0u32; 32];
                    header_words[0] = 5 << 10;
                    header_words[3] = 7 << 24; // OutputTopology is geometry-only.
                    header_words[18] = targets;
                    header_words[19] = u32::from(mask) | (u32::from(depth) << 1);
                    let header = headers::parse(&header_words).unwrap();
                    let headers::ShaderHeader::Fragment(fragment) = &header else {
                        panic!("fragment header");
                    };
                    assert_eq!(fragment.writes_sample_mask(), mask);
                    assert_eq!(fragment.writes_depth(), depth);
                    let color_targets: Vec<_> =
                        (0..8).filter(|t| targets & (15 << (t * 4)) != 0).collect();
                    let mask_register = color_targets.len() * 4;
                    for bits in [0u32, 1, 0x8000_0000, 0xa5a5_5a5a, u32::MAX] {
                        let mut kinds: Vec<_> = (0..mask_register + 2)
                            .map(|r| mov(r as u8, 0x3f00_0000 + r as u32))
                            .collect();
                        kinds[mask_register] = mov(mask_register as u8, bits);
                        if kill != 0 {
                            kinds.push((
                                K::Kill(ops::Kill),
                                (kill != 3).then_some(Predicate {
                                    register: 0,
                                    inverted: kill == 2,
                                }),
                            ));
                        }
                        kinds.push((K::Exit(ops::Exit { cc_test: 0xf }), None));
                        let xs: Vec<_> = kinds
                            .into_iter()
                            .enumerate()
                            .map(|(i, (kind, predicate))| Instruction {
                                address: i as u32 * 8,
                                kind,
                                predicate,
                            })
                            .collect();
                        // Execute both shapes; optimization must retain builtin stores.
                        for words in [
                            emit_with_header(&xs, Some(&header)).unwrap().words,
                            shadowbox::emitter::emit_with_header(&xs, Some(&header))
                                .unwrap()
                                .words,
                        ] {
                            let execution = execute(&words, 0);
                            // P0 initializes false: @!P0 discards, @P0 returns.
                            assert_eq!(execution.discarded, kill >= 2);
                            let sample_mask = builtin(&words, 20);
                            let depth_output = builtin(&words, 22);
                            assert_eq!(sample_mask.is_some(), mask);
                            assert_eq!(depth_output.is_some(), depth);
                            assert_eq!(
                                ops(&words)
                                    .iter()
                                    .any(|(op, a)| *op == Op::ExecutionMode && a[1] == 12),
                                depth
                            );
                            let entry = ops(&words)
                                .into_iter()
                                .find(|(op, _)| *op == Op::EntryPoint)
                                .unwrap()
                                .1;
                            for variable in sample_mask.into_iter().chain(depth_output) {
                                assert!(entry[4..].contains(&variable));
                            }
                            if execution.discarded {
                                assert!(execution.outputs.is_empty());
                                continue;
                            }
                            if let Some(variable) = sample_mask {
                                assert_eq!(execution.outputs[&variable], [bits]);
                            }
                            if let Some(variable) = depth_output {
                                assert_eq!(
                                    execution.outputs[&variable],
                                    [0x3f00_0000 + mask_register as u32 + 1]
                                );
                            }
                            for (packed, target) in color_targets.iter().enumerate() {
                                let variable = ops(&words)
                                    .into_iter()
                                    .find(|(op, a)| *op == Op::Decorate && a[1..] == [30, *target])
                                    .unwrap()
                                    .1[0];
                                assert_eq!(
                                    execution.outputs[&variable],
                                    (0..4)
                                        .map(|c| 0x3f00_0000 + (packed * 4 + c) as u32)
                                        .collect::<Vec<_>>()
                                );
                            }
                            // Validate representative variants, including sparse omaps.
                            if bits == 0xa5a5_5a5a {
                                let result = spirv_validation::validate(&words);
                                assert!(
                                    result.status.success(),
                                    "{}",
                                    String::from_utf8_lossy(&result.stderr)
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn bfi_rc_control_guards_overwrites_and_cbuf_address() {
    let raw = bfi_rc(1, 2, 3, 19, 0xfffc);
    for inverted in [false, true] {
        let words = module(vec![
            mov(1, 0xface_cafe),
            mov(2, 0x1234_5678),
            mov(3, 0x0804),
            (
                decoded(raw),
                Some(Predicate {
                    register: 0,
                    inverted,
                }),
            ),
        ]);
        assert_eq!(
            execute(&words, 0x8765_4321).registers[1],
            if inverted { 0x8765_4781 } else { 0xface_cafe }
        );
        let constants: HashMap<_, _> = ops(&words)
            .iter()
            .filter(|(op, _)| *op == Op::Constant)
            .map(|(_, a)| (a[1], a[2]))
            .collect();
        assert!(ops(&words).iter().any(|(op, a)| *op == Op::AccessChain
            && a.len() == 5
            && constants[&a[3]] == 0
            && constants[&a[4]] == 4095));
        assert!(ops(&words)
            .iter()
            .any(|(op, a)| *op == Op::CompositeExtract && a[3] == 3));
        assert!(ops(&words)
            .iter()
            .any(|(op, a)| *op == Op::Decorate && a[1..] == [33, 19]));
    }
    // Decode complete bank/offset fields, including the sign-bit boundary.
    for bank in 0..32 {
        for offset in [0, 4, 16, 0x7ffc, 0x8000, 0xfffc] {
            assert!(matches!(decoded(bfi_rc(255, 255, 255, bank, offset)),
                K::BitFieldInsert(ops::BitFieldInsert { dst: R(255), insert: R(255),
                    base: S::ConstantBuffer { binding, offset: decoded_offset },
                    range: S::Register(R(255)) }) if binding == bank && decoded_offset == offset));
        }
    }
    let mov_control =
        |value: u64, guard: u64| (0x0100u64 << 48) | (15 << 12) | 3 | (guard << 16) | (value << 20);
    let guarded_bfi = raw & !(15 << 16);
    let matching = encoded_stream(&[mov_control(33, 7), mov_control(0x0804, 0), guarded_bfi]);
    assert!(matches!(
        matching[2].kind,
        K::BitFieldInsert(ops::BitFieldInsert {
            range: S::Immediate(0x0804),
            ..
        })
    ));
    // Unguarded use sees both controls and cannot choose one.
    let ambiguous = encoded_stream(&[mov_control(0, 7), mov_control(0x0804, 0), raw]);
    assert!(matches!(
        ambiguous[2].kind,
        K::BitFieldInsert(ops::BitFieldInsert {
            range: S::Register(_),
            ..
        })
    ));
    // A BFI that overwrites its control kills the constant proof for later use.
    let overwrite = encoded_stream(&[mov_control(0x0804, 7), bfi_rc(3, 2, 3, 0, 0), raw]);
    assert!(matches!(
        overwrite[1].kind,
        K::BitFieldInsert(ops::BitFieldInsert { .. })
    ));
    assert!(matches!(
        overwrite[2].kind,
        K::BitFieldInsert(ops::BitFieldInsert {
            range: S::Register(_),
            ..
        })
    ));
}

#[test]
fn fragment_sample_mask_early_exit_stores_executed_registers() {
    let mut h = [0u32; 32];
    h[0] = 5 << 10;
    h[19] = 3;
    let header = headers::parse(&h).unwrap();
    for inverted in [false, true] {
        let kinds = vec![
            mov(0, 0x8000_0001),
            mov(1, 0x3f00_0000),
            (
                K::Exit(ops::Exit { cc_test: 0xf }),
                Some(Predicate {
                    register: 0,
                    inverted,
                }),
            ),
            mov(0, 0xa5a5_5a5a),
            mov(1, 0x3f80_0000),
            (K::Exit(ops::Exit { cc_test: 0xf }), None),
        ];
        let xs: Vec<_> = kinds
            .into_iter()
            .enumerate()
            .map(|(i, (kind, predicate))| Instruction {
                address: i as u32 * 8,
                kind,
                predicate,
            })
            .collect();
        for words in [
            emit_with_header(&xs, Some(&header)).unwrap().words,
            shadowbox::emitter::emit_with_header(&xs, Some(&header))
                .unwrap()
                .words,
        ] {
            let execution = execute(&words, 0);
            assert!(!execution.discarded);
            assert_eq!(
                execution.outputs[&builtin(&words, 20).unwrap()],
                [if inverted { 0x8000_0001 } else { 0xa5a5_5a5a }]
            );
            assert_eq!(
                execution.outputs[&builtin(&words, 22).unwrap()],
                [if inverted { 0x3f00_0000 } else { 0x3f80_0000 }]
            );
            let result = spirv_validation::validate(&words);
            assert!(
                result.status.success(),
                "{}",
                String::from_utf8_lossy(&result.stderr)
            );
        }
    }
}

#[test]
fn bfi_rc_control_join_folds_equal_values_and_preserves_runtime_values() {
    let header = headers::parse(&[0; 32]).unwrap();
    for right in [0x0804, 0x011f] {
        let kinds = vec![
            mov(3, 0x0804),
            (
                K::Branch(ops::Branch {
                    target: BranchTarget::reaching(8, 32),
                    cc_test: 15,
                }),
                Some(Predicate {
                    register: 0,
                    inverted: false,
                }),
            ),
            mov(3, right),
            (
                K::Branch(ops::Branch {
                    target: BranchTarget::reaching(24, 40),
                    cc_test: 15,
                }),
                None,
            ),
            mov(3, 0x0804),
            (decoded(bfi_rc(1, 2, 3, 0, 0)), None),
            (K::Exit(ops::Exit { cc_test: 0xf }), None),
        ];
        let xs: Vec<_> = kinds
            .into_iter()
            .enumerate()
            .map(|(i, (kind, predicate))| Instruction {
                address: i as u32 * 8,
                kind,
                predicate,
            })
            .collect();
        let emitted = emit_with_header(&xs, Some(&header));
        let words = emitted.unwrap().words;
        assert_eq!(
            execute(&words, u32::MAX).registers[1],
            if right == 0x0804 {
                0xffff_f00f
            } else {
                0x7fff_ffff
            }
        );
        assert_eq!(
            ops(&words)
                .iter()
                .filter(|(op, _)| *op == Op::BitFieldInsert)
                .count(),
            1
        );
        assert_eq!(
            ops(&words)
                .iter()
                .any(|(op, _)| *op == Op::BitFieldUExtract),
            right != 0x0804
        );
    }
}

// BRX encoding: Mesa nv50_ir_emit_gm107.cpp:533-555; envytools
// gm107.c:1942-1943,204,254,426-457. Runtime address arithmetic:
// Ryujinx InstEmitFlowControl.cs:38-40. All words below are project-owned.
fn brx_word(register: u8, base: i32, guard: u8) -> u64 {
    (0xe250_u64 << 48)
        | (u64::from(base as u32 & 0xffffff) << 20)
        | (u64::from(guard) << 16)
        | (u64::from(register) << 8)
        | 0xf
}

#[test]
fn brx_register_fields_and_unsupported_controls() {
    for (raw, register, base) in [
        (brx_word(0, 0, 7), 0, 0),
        (brx_word(0xff, -1, 7), 0xff, -1),
        (brx_word(0xa5, -0x800000, 7), 0xa5, -0x800000),
        (brx_word(0x5a, 0x7fffff, 7), 0x5a, 0x7fffff),
    ] {
        assert_eq!(
            decoded(raw),
            K::BrxRegister(ops::BrxRegister {
                displacement: R(register),
                base: BrxBase::decode(0, base as u32),
                cc_test: 0xf,
            })
        );
    }
    for guard in 0..16 {
        let instruction = decode_word(
            [
                brx_word(3, -32, guard) as u32,
                (brx_word(3, -32, guard) >> 32) as u32,
            ],
            8,
        );
        assert_eq!(
            instruction.predicate,
            (guard != 7).then_some(Predicate {
                register: guard & 7,
                inverted: guard & 8 != 0
            })
        );
    }
    let plain = brx_word(3, 0, 7);
    for bit in [5, 6, 7, 44, 45, 46, 47, 48, 49, 50, 51] {
        assert!(
            matches!(decoded(plain ^ (1 << bit)), K::Unknown(ops::Unknown { .. })),
            "bit{bit}"
        );
    }
    for cc in 0..32_u64 {
        if cc != 15 {
            assert!(matches!(
                decoded((plain & !31) | cc),
                K::Unknown(ops::Unknown { .. })
            ));
        }
    }
    // All other audited blockers remain visible, including X57 and reserved
    // half flush mode3. A fabricated arithmetic operation is not a decode.
    for prefix in [
        0x4196, 0x3f80, 0x3f59, 0x50e3, 0x4b09, 0xbfc9, 0x38d1, 0x2b8c, 0x3f7f,
    ] {
        assert!(matches!(
            decoded((prefix << 48) | (7 << 16)),
            K::Unknown(ops::Unknown { .. })
        ));
    }
}

#[test]
fn brx_dispatch_uses_byte_displacements_and_guard() {
    for (displacement, base, guard, expected) in [
        (48, -24, 7, 0x222),
        (64, -24, 7, 0x333), // target bundle0x40 runs slot0x48
        (72, -24, 7, 0x333),
        (48, -24, 15, 0x111),      // inverted PT falls through
        (48, -24, 0, 0x111),       // P0 initially false
        (48, -24, 8, 0x222),       // inverted P0 takes the branch
        (u32::MAX - 23, 40, 7, 0), // wrapping displacement selects EXIT0x28
    ] {
        let kinds = [
            mov(0, displacement).0,
            decoded_at(brx_word(0, base, guard), 16),
            mov(1, 0x111).0,
            K::Exit(ops::Exit { cc_test: 15 }),
            mov(1, 0x222).0,
            K::Exit(ops::Exit { cc_test: 15 }),
            mov(1, 0x333).0,
            K::Exit(ops::Exit { cc_test: 15 }),
        ];
        let xs: Vec<_> = kinds
            .into_iter()
            .zip([8, 16, 24, 40, 48, 56, 72, 80])
            .enumerate()
            .map(|(i, (kind, address))| Instruction {
                kind,
                address,
                predicate: if i == 1 {
                    decode_word(
                        [
                            brx_word(0, base, guard) as u32,
                            (brx_word(0, base, guard) >> 32) as u32,
                        ],
                        address,
                    )
                    .predicate
                } else {
                    None
                },
            })
            .collect();
        let header = headers::parse(&[0; 32]).unwrap();
        let module = emit_with_header(&xs, Some(&header)).unwrap();
        assert_eq!(
            run(&module.words)[1],
            expected,
            "displacement={displacement:x}, base={base}, guard={guard}"
        );
    }
}

#[test]
fn brx_can_jump_backwards_and_read_rz() {
    let kinds = [
        mov(0, 8).0,
        K::Branch(ops::Branch {
            target: BranchTarget::reaching(16, 48),
            cc_test: 15,
        }), // from0x10 to0x30
        mov(1, 0x1234).0,
        K::Exit(ops::Exit { cc_test: 15 }),
        decoded_at(brx_word(0, -40, 7), 48), // from0x30 back to0x18
        K::Exit(ops::Exit { cc_test: 15 }),
    ];
    let xs: Vec<_> = kinds
        .into_iter()
        .zip([8, 16, 24, 40, 48, 56])
        .map(|(kind, address)| Instruction {
            kind,
            address,
            predicate: None,
        })
        .collect();
    let header = headers::parse(&[0; 32]).unwrap();
    assert_eq!(
        run(&emit_with_header(&xs, Some(&header)).unwrap().words)[1],
        0x1234
    );
    // RZ is zero even if a preceding destination was R255.
    let xs = [
        Instruction {
            address: 8,
            predicate: None,
            kind: mov(255, 0x1234).0,
        },
        Instruction {
            address: 16,
            predicate: None,
            kind: decoded_at(brx_word(255, 24, 7), 16),
        },
        Instruction {
            address: 24,
            predicate: None,
            kind: mov(1, 0xdead).0,
        },
        Instruction {
            address: 40,
            predicate: None,
            kind: K::Exit(ops::Exit { cc_test: 15 }),
        },
        Instruction {
            address: 48,
            predicate: None,
            kind: mov(1, 0xbeef).0,
        },
        Instruction {
            address: 56,
            predicate: None,
            kind: K::Exit(ops::Exit { cc_test: 15 }),
        },
    ];
    assert_eq!(
        run(&emit_with_header(&xs, Some(&header)).unwrap().words)[1],
        0xbeef
    );
}

#[test]
fn kil_keeps_private_registers_and_reports_current_helper_state() {
    // Mesa sm50.rs:3172-3174; from_nir.rs:3035-3050: demotion, not exit.
    // S2R thread_kill: Ryujinx InstEmitMove.cs:90-92 / IoMap.cs:45.
    for guard in [
        None,
        Some(Predicate {
            register: 0,
            inverted: false,
        }),
        Some(Predicate {
            register: 0,
            inverted: true,
        }),
    ] {
        let xs = [
            Instruction {
                address: 8,
                kind: K::Kill(ops::Kill),
                predicate: guard,
            },
            Instruction {
                address: 16,
                kind: mov(40, 42).0,
                predicate: None,
            },
            Instruction {
                address: 24,
                kind: K::S2R(ops::S2R {
                    dst: R(41),
                    idx: 0x13,
                }),
                predicate: None,
            },
            Instruction {
                address: 32,
                kind: K::Exit(ops::Exit { cc_test: 0xf }),
                predicate: None,
            },
        ];
        for mut module in [
            shadowbox::emitter::emit_unoptimized_with_header(&xs, None).unwrap(),
            shadowbox::emitter::emit(&xs).unwrap(),
        ] {
            module.optimize();
            let actual = execute(&module.words, 0);
            let demoted = guard.is_none_or(|p| p.inverted);
            assert_eq!(actual.registers[40], 42);
            assert_eq!(actual.registers[41], if demoted { u32::MAX } else { 0 });
            assert_eq!(actual.discarded, demoted);
        }
    }
}
