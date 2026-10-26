//! Execute emitted SPIR-V on the CPU and compare exact bits to an independent
//! rational oracle. All inputs are project generated. Encoding and semantic
//! sources are recorded in docs/reference-sources.md
use shadowbox::ir::ops;
use shadowbox::spirv::Op;
use shadowbox::{
    decoder::decode_word,
    emitter::{emit_unoptimized_with_header, emit_with_header},
    headers,
    ir::{
        Conversion, GlobalAccess, Instruction, InstructionKind as K, LsSize2, Predicate, Register,
        RoundingMode, Source,
    },
};
use std::collections::BTreeSet;

#[path = "support/spirv_validation.rs"]
mod spirv_validation;

const MODES: [RoundingMode; 4] = [
    RoundingMode::NearestEven,
    RoundingMode::NegativeInfinity,
    RoundingMode::PositiveInfinity,
    RoundingMode::Zero,
];

fn decode(raw: u64) -> Instruction {
    decode_word([raw as u32, (raw >> 32) as u32], 0)
}
fn word(src_size: u8, signed: bool, dst_size: u8, round: u8) -> u64 {
    (0x5cb8 << 48)
        | (8 << 20)
        | (7 << 16)
        | (u64::from(signed) << 13)
        | (u64::from(src_size) << 10)
        | (u64::from(dst_size) << 8)
        | (u64::from(round) << 39)
        | 2
}
fn module(raw: u64, input: u64, optimized: bool) -> Vec<u32> {
    let mov = |r, value| Instruction {
        address: 0,
        predicate: None,
        kind: K::Mov(ops::Mov {
            quad_lanes: 0xf, // Mesa sm50.rs:1936,1941,1946: every quad lane.
            dst: Register(r),
            src: Source::Immediate(value),
        }),
    };
    let mut xs = vec![
        mov(8, input as u32),
        mov(9, (input >> 32) as u32),
        mov(2, 0x12345678),
        mov(3, 0x87654321),
        decode(raw),
        decode((0xe300 << 48) | (7 << 16)),
    ];
    for (i, x) in xs.iter_mut().enumerate() {
        x.address = i as u32 * 8;
    }
    let header = headers::parse(&[0; 32]).unwrap();
    if optimized {
        emit_with_header(&xs, Some(&header))
    } else {
        emit_unoptimized_with_header(&xs, Some(&header))
    }
    .unwrap()
    .words
}

#[derive(Clone, Copy, Default)]
struct Ty {
    width: u32,
    vector: usize,
}
struct Program {
    instructions: Vec<(Op, Vec<u32>)>,
    types: Vec<Ty>,
    result_types: Vec<u32>,
    registers: Vec<u32>,
    bound: usize,
}
impl Program {
    fn new(words: &[u32]) -> Self {
        let bound = words[3] as usize;
        let mut result = Self {
            instructions: Vec::new(),
            types: vec![Ty::default(); bound],
            result_types: vec![0; bound],
            registers: Vec::new(),
            bound,
        };
        let mut at = 5;
        while at < words.len() {
            let end = at + (words[at] >> 16) as usize;
            let op = Op::of(words[at]);
            let a = words[at + 1..end].to_vec();
            match op {
                Op::TypeBool => result.types[a[0] as usize].width = 1,
                Op::TypeInt => result.types[a[0] as usize].width = a[1],
                Op::TypeFloat => {
                    result.types[a[0] as usize] = Ty {
                        width: a[1],
                        vector: 0,
                    }
                }
                Op::TypeVector => {
                    result.types[a[0] as usize] = Ty {
                        vector: a[2] as usize,
                        ..result.types[a[1] as usize]
                    }
                }
                Op::Variable if a[2] == 7 => result.registers.push(a[1]),
                _ => {}
            }
            if matches!(
                op,
                Op::ExtInst
                    | Op::ConstantTrue
                    | Op::ConstantFalse
                    | Op::Constant
                    | Op::Load
                    | Op::CompositeExtract
                    | Op::ConvertFToU
                    | Op::ConvertFToS
                    | Op::ConvertSToF
                    | Op::ConvertUToF
                    | Op::UConvert
                    | Op::SConvert
                    | Op::Bitcast
                    | Op::SNegate
                    | Op::IAdd
                    | Op::ISub
                    | Op::LogicalEqual
                    | Op::LogicalNotEqual
                    | Op::LogicalOr
                    | Op::LogicalAnd
                    | Op::LogicalNot
                    | Op::Select
                    | Op::IEqual
                    | Op::INotEqual
                    | Op::UGreaterThan
                    | Op::SGreaterThan
                    | Op::UGreaterThanEqual
                    | Op::SGreaterThanEqual
                    | Op::ULessThan
                    | Op::SLessThan
                    | Op::ShiftRightLogical
                    | Op::ShiftRightArithmetic
                    | Op::ShiftLeftLogical
                    | Op::BitwiseOr
                    | Op::BitwiseXor
                    | Op::BitwiseAnd
                    | Op::Not
                    | Op::BitFieldSExtract
                    | Op::BitFieldUExtract
            ) {
                result.result_types[a[1] as usize] = a[0];
            }
            result.instructions.push((op, a));
            at = end;
        }
        result
    }
    fn run(&self, input: u64) -> [u32; 2] {
        let mut values = vec![Vec::<u64>::new(); self.bound];
        let mut memory = vec![Vec::<u64>::new(); self.bound];
        let mask = |width: u32| -> u64 {
            if width == 64 {
                u64::MAX
            } else {
                (1u64 << width) - 1
            }
        };
        let signed = |value: u64, width| ((value << (64 - width)) as i64) >> (64 - width);
        for (op, a) in &self.instructions {
            let get = |index: usize| values[a[index] as usize][0];
            let ty = if a.len() >= 2 {
                self.types[a[0] as usize]
            } else {
                Ty::default()
            };
            let input_ty = |index: usize| self.types[self.result_types[a[index] as usize] as usize];
            let result = match op {
                Op::ConstantTrue => Some(vec![1]),
                Op::ConstantFalse => Some(vec![0]),
                Op::Constant => Some(vec![
                    u64::from(a[2])
                        | if a.len() == 4 {
                            u64::from(a[3]) << 32
                        } else {
                            0
                        },
                ]),
                Op::Variable => {
                    memory[a[1] as usize] = vec![0];
                    None
                }
                Op::Load => Some(memory[a[2] as usize].clone()),
                Op::Store => {
                    let mut value = values[a[1] as usize].clone();
                    // Inject runtime register inputs at the MOV stores, keeping
                    // a single compiled instruction shape for all exact cases.
                    if a[0] == self.registers[8] {
                        value = vec![input as u32 as u64];
                    }
                    if a[0] == self.registers[9] {
                        value = vec![input >> 32];
                    }
                    memory[a[0] as usize] = value;
                    None
                }
                Op::CompositeExtract => Some(vec![values[a[2] as usize][a[3] as usize]]),
                Op::Bitcast => {
                    let source = &values[a[2] as usize];
                    Some(if ty.vector == 2 && source.len() == 1 {
                        vec![source[0] as u32 as u64, source[0] >> 32]
                    } else if ty.width == 64 && ty.vector == 0 && source.len() == 2 {
                        vec![source[0] | (source[1] << 32)]
                    } else {
                        source.clone()
                    })
                }
                Op::UConvert => Some(vec![get(2) & mask(ty.width)]),
                Op::SConvert => Some(vec![
                    signed(get(2), input_ty(2).width) as u64 & mask(ty.width),
                ]),
                Op::ConvertFToS => {
                    let f = f32::from_bits(get(2) as u32) as f64;
                    assert!(
                        f >= i64::MIN as f64 && f < -(i64::MIN as f64),
                        "undefined back-conversion"
                    );
                    Some(vec![(f as i64) as u64 & mask(ty.width)])
                }
                Op::ConvertSToF | Op::ConvertUToF => {
                    let width = input_ty(2).width;
                    let bits = if ty.width == 64 {
                        if *op == Op::ConvertSToF {
                            (signed(get(2), width) as f64).to_bits()
                        } else {
                            (get(2) as f64).to_bits()
                        }
                    } else if *op == Op::ConvertSToF {
                        (signed(get(2), width) as f32).to_bits() as u64
                    } else {
                        (get(2) as f32).to_bits() as u64
                    };
                    Some(vec![bits])
                }
                Op::ExtInst => Some(vec![match a[3] {
                    5 => signed(get(4), input_ty(4).width).wrapping_abs() as u64 & mask(ty.width),
                    75 => {
                        if get(4) == 0 {
                            u32::MAX as u64
                        } else {
                            (63 - get(4).leading_zeros()) as u64
                        }
                    }
                    other => panic!("unsupported extended op {other}"),
                }]),
                Op::SNegate => Some(vec![get(2).wrapping_neg() & mask(ty.width)]),
                Op::IAdd => Some(vec![get(2).wrapping_add(get(3)) & mask(ty.width)]),
                Op::ISub => Some(vec![get(2).wrapping_sub(get(3)) & mask(ty.width)]),
                Op::LogicalEqual | Op::IEqual => Some(vec![u64::from(get(2) == get(3))]),
                Op::LogicalNotEqual | Op::INotEqual => Some(vec![u64::from(get(2) != get(3))]),
                Op::LogicalOr => Some(vec![u64::from(get(2) != 0 || get(3) != 0)]),
                Op::LogicalAnd => Some(vec![u64::from(get(2) != 0 && get(3) != 0)]),
                Op::LogicalNot => Some(vec![u64::from(get(2) == 0)]),
                Op::Select => Some(values[a[if get(2) != 0 { 3 } else { 4 }] as usize].clone()),
                Op::UGreaterThan => Some(vec![u64::from(get(2) > get(3))]),
                Op::SGreaterThan => Some(vec![u64::from(
                    signed(get(2), input_ty(2).width) > signed(get(3), input_ty(3).width),
                )]),
                Op::UGreaterThanEqual => Some(vec![u64::from(get(2) >= get(3))]),
                Op::ULessThan => Some(vec![u64::from(get(2) < get(3))]),
                Op::SLessThan => Some(vec![u64::from(
                    signed(get(2), input_ty(2).width) < signed(get(3), input_ty(3).width),
                )]),
                Op::ShiftRightLogical | Op::ShiftLeftLogical => {
                    assert!(get(3) < u64::from(input_ty(2).width), "undefined shift");
                    Some(vec![if *op == Op::ShiftRightLogical {
                        get(2) >> get(3)
                    } else {
                        (get(2) << get(3)) & mask(ty.width)
                    }])
                }
                Op::BitwiseOr => Some(vec![get(2) | get(3)]),
                Op::BitwiseAnd => Some(vec![get(2) & get(3)]),
                Op::BitFieldSExtract | Op::BitFieldUExtract => {
                    assert!(get(3) + get(4) <= 32);
                    let field = (get(2) >> get(3)) & mask(get(4) as u32);
                    Some(vec![if *op == Op::BitFieldSExtract {
                        signed(field, get(4) as u32) as u32 as u64
                    } else {
                        field
                    }])
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
                | Op::Label
                | Op::Branch
                | Op::BranchConditional
                | Op::Switch
                | Op::Return => None,
                other => panic!("unsupported executable SPIR-V op {}", other.number()),
            };
            if let Some(value) = result {
                values[a[1] as usize] = value;
            }
        }
        [
            memory[self.registers[2] as usize][0] as u32,
            memory[self.registers[3] as usize][0] as u32,
        ]
    }
}

// Independent oracle: binary search the floating encoding space, comparing
// exact rational values in u128. It does not convert through a host float or
// repeat the emitter's leading-bit/rounding formulas.
fn rational(bits: u64, size: u8) -> (u128, u128) {
    let (p, bias) = match size {
        1 => (10, 15),
        2 => (23, 127),
        3 => (52, 1023),
        _ => unreachable!(),
    };
    if size == 1 && bits == 0x7c00 {
        return (65536, 1);
    } // virtual overflow endpoint
    let exponent = ((bits >> p) as i32 - bias) - p;
    let significand = (bits & ((1 << p) - 1)) | (1 << p);
    if exponent >= 0 {
        (u128::from(significand) << exponent, 1)
    } else {
        (u128::from(significand), 1u128 << -exponent)
    }
}
fn expected(value: u64, src_size: u8, signed: bool, dst_size: u8, mode: RoundingMode) -> u64 {
    let width = 8 << src_size;
    let number = if signed {
        (((value << (64 - width)) as i64) >> (64 - width)) as i128
    } else {
        i128::from(
            value
                & if width == 64 {
                    u64::MAX
                } else {
                    (1u64 << width) - 1
                },
        )
    };
    if number == 0 {
        return 0;
    }
    let negative = number < 0;
    let magnitude = number.unsigned_abs();
    let (p, bias, sign_bit) = match dst_size {
        1 => (10, 15, 15),
        2 => (23, 127, 31),
        3 => (52, 1023, 63),
        _ => unreachable!(),
    };
    let mut low = (bias as u64) << p; // 1.0; inputs are nonzero integers
    let mut high = if dst_size == 1 {
        0x7bff
    } else {
        ((bias + 64) as u64) << p
    };
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        let (n, d) = rational(middle, dst_size);
        if n <= magnitude * d {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    let (n, d) = rational(low, dst_size);
    let exact = n == magnitude * d;
    let up = low + 1;
    let bits = if exact {
        low
    } else {
        match mode {
            RoundingMode::Zero => low,
            RoundingMode::NegativeInfinity => {
                if negative {
                    up
                } else {
                    low
                }
            }
            RoundingMode::PositiveInfinity => {
                if negative {
                    low
                } else {
                    up
                }
            }
            RoundingMode::NearestEven => {
                let (un, ud) = rational(up, dst_size);
                let denominator = d.max(ud);
                let twice = magnitude * denominator * 2;
                let endpoints = n * (denominator / d) + un * (denominator / ud);
                if twice > endpoints || (twice == endpoints && low & 1 != 0) {
                    up
                } else {
                    low
                }
            }
        }
    };
    bits | if negative { 1 << sign_bit } else { 0 }
}
fn cases(width: u32) -> BTreeSet<u64> {
    let max = if width == 64 {
        u64::MAX
    } else {
        (1u64 << width) - 1
    };
    let mut values = BTreeSet::from([0, 1, max, max / 2, max / 2 + 1]);
    for exponent in 0..width {
        let power = 1u64 << exponent;
        for offset in [
            0, 1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511,
            512, 513, 1023, 1024, 1025,
        ] {
            for center in [power, power + power / 2] {
                for v in [center.checked_add(offset), center.checked_sub(offset)]
                    .into_iter()
                    .flatten()
                {
                    if v <= max {
                        values.insert(v);
                        values.insert(v.wrapping_neg() & max);
                    }
                }
            }
        }
    }
    // Include both even and odd retained-significand ties at the ULP scale,
    // especially the large F32 ULPs for 64-bit inputs.
    for exponent in 1..width {
        for precision in [10, 23, 52] {
            if exponent <= precision {
                continue;
            }
            let power = 1u64 << exponent;
            let ulp = 1u64 << (exponent - precision);
            for center in [power, power + ulp] {
                for offset in [ulp / 2 - 1, ulp / 2, ulp / 2 + 1, ulp - 1, ulp, ulp + 1] {
                    for v in [center.checked_add(offset), center.checked_sub(offset)]
                        .into_iter()
                        .flatten()
                    {
                        if v <= max {
                            values.insert(v);
                            values.insert(v.wrapping_neg() & max);
                        }
                    }
                }
            }
        }
    }
    for v in 65488..=65536 {
        if v <= max {
            values.insert(v);
            values.insert(v.wrapping_neg() & max);
        }
    }
    values
}

#[test]
fn all_encodable_widths_modes_and_signs_match_exact_rationals() {
    let mut comparisons = 0;
    for src_size in 0..=3 {
        let values = cases(8 << src_size);
        for signed in [false, true] {
            for dst_size in 1..=3 {
                if src_size == 3 && dst_size == 1 {
                    continue;
                }
                for round in 0..4 {
                    let raw = word(src_size, signed, dst_size, round);
                    assert!(matches!(decode(raw).kind, K::Convert(ops::Convert { .. })));
                    let words = module(raw, 0, false);
                    let validation = spirv_validation::validate(&words);
                    assert!(
                        validation.status.success(),
                        "{raw:016x}: {}",
                        String::from_utf8_lossy(&validation.stderr)
                    );
                    let program = Program::new(&words);
                    for &input in &values {
                        let actual = program.run(input);
                        let actual = u64::from(actual[0])
                            | if dst_size == 3 {
                                u64::from(actual[1]) << 32
                            } else {
                                0
                            };
                        assert_eq!(actual, expected(input, src_size, signed, dst_size, MODES[round as usize]),
                            "src_size={src_size} signed={signed} dst_size={dst_size} round={round} input={input:016x}");
                        comparisons += 1;
                    }
                }
            }
        }
    }
    eprintln!("{comparisons} exact I2F comparisons");
}

#[test]
fn predicate_bits_guard_both_output_words() {
    for dst_size in [1, 2, 3] {
        for pred in 0..8u64 {
            for inverted in [false, true] {
                let raw = (word(2, false, dst_size, 3) & !(15 << 16))
                    | (pred << 16)
                    | (u64::from(inverted) << 19);
                assert_eq!(
                    decode(raw).predicate,
                    if pred == 7 && !inverted {
                        None
                    } else {
                        Some(Predicate {
                            register: pred as u8,
                            inverted,
                        })
                    }
                );
                let actual = Program::new(&module(raw, 0, false)).run(u32::MAX as u64);
                // P0..P6 initialize false and PT initializes true.
                if (pred == 7) != inverted {
                    let expected =
                        expected(u32::MAX as u64, 2, false, dst_size, RoundingMode::Zero);
                    assert_eq!(actual[0], expected as u32);
                    if dst_size == 3 {
                        assert_eq!(actual[1], (expected >> 32) as u32);
                    } else {
                        assert_eq!(actual[1], 0x87654321);
                    }
                } else {
                    assert_eq!(actual, [0x12345678, 0x87654321]);
                }
            }
        }
    }
}

#[test]
fn unsupported_fields_stay_unknown_and_nop_arm_count_is_two() {
    let raw = word(2, false, 2, 3);
    for bit in [12, 14, 15, 28, 38, 43, 44, 46, 47, 48, 50] {
        assert!(
            matches!(
                decode(raw | (1 << bit)).kind,
                K::Unknown(ops::Unknown { .. })
            ),
            "bit {bit}"
        );
    }
    for opcode in [0x5cb8, 0x4cb8, 0x38b8] {
        let raw = (word(3, true, 1, 0) & !(0xffff << 48)) | opcode << 48;
        assert!(matches!(decode(raw).kind, K::Unknown(ops::Unknown { .. })));
    }
    let source = include_str!("../src/decoder/families.rs");
    assert_eq!(
        source
            .matches("x.guarded(InstructionKind::Nop(ops::Nop))")
            .count(),
        2
    );
}

#[test]
fn native_nearest_even_and_exact_small_conversions_keep_short_paths() {
    for (src_size, dst_size, round) in [(2, 2, 0), (0, 2, 3), (1, 2, 1), (2, 3, 2)] {
        let program = Program::new(&module(word(src_size, true, dst_size, round), 0, false));
        assert!(!program
            .instructions
            .iter()
            .any(|(op, a)| *op == Op::Capability && a == &[11]));
        assert_eq!(
            program
                .instructions
                .iter()
                .filter(|(op, _)| *op == Op::ConvertSToF)
                .count(),
            1
        );
    }
}

#[test]
fn optimized_constant_conversions_validate() {
    for (src_size, dst_size) in [(2, 1), (2, 2), (3, 2), (3, 3)] {
        for round in 0..4 {
            for signed in [false, true] {
                let words = module(word(src_size, signed, dst_size, round), u64::MAX, true);
                let validation = spirv_validation::validate(&words);
                assert!(
                    validation.status.success(),
                    "{}",
                    String::from_utf8_lossy(&validation.stderr)
                );
            }
        }
    }
}

#[test]
fn high_word_is_a_source_operand() {
    assert_eq!(
        decode(word(3, true, 3, 3)).kind.sources(),
        vec![Source::Register(Register(8)), Source::Register(Register(9))]
    );
    assert!(matches!(
        decode(word(3, true, 3, 3)).kind,
        K::Convert(ops::Convert {
            kind: Conversion::I2F { src_size: 3, .. },
            ..
        })
    ));
}

#[test]
fn modifiers_precede_extraction_for_every_mode() {
    for src_size in 0..=3 {
        for byte in 0..=3u8 {
            // nvdisasm SM52: a 16-bit source selects H0 or H1.
            if (src_size < 3 && (byte * 8 + (8 << src_size) > 32 || byte % (1 << src_size) != 0))
                || (src_size == 3 && byte != 0)
            {
                continue;
            }
            for abs in [false, true] {
                for negate in [false, true] {
                    for signed in [false, true] {
                        for dst_size in 1..=3 {
                            if src_size == 3 && dst_size == 1 {
                                continue;
                            }
                            for round in 0..4 {
                                let raw = word(src_size, signed, dst_size, round)
                                    | (u64::from(byte) << 41)
                                    | (u64::from(abs) << 49)
                                    | (u64::from(negate) << 45);
                                let program = Program::new(&module(raw, 0, false));
                                for input in [
                                    0u64,
                                    1,
                                    u64::MAX,
                                    0x8000000080000000,
                                    0xffffff00800000ff,
                                    0x123456789abcdef0,
                                    0x0020000000000003,
                                ] {
                                    let modified = if src_size == 3 {
                                        let mut n = input as i64;
                                        if abs {
                                            n = n.wrapping_abs();
                                        }
                                        if negate {
                                            n = n.wrapping_neg();
                                        }
                                        n as u64
                                    } else {
                                        let mut n = input as i32;
                                        if abs {
                                            n = n.wrapping_abs();
                                        }
                                        if negate {
                                            n = n.wrapping_neg();
                                        }
                                        (n as u32 >> (byte * 8)) as u64
                                    };
                                    let expected = expected(
                                        modified,
                                        src_size,
                                        signed,
                                        dst_size,
                                        MODES[round as usize],
                                    );
                                    let actual = program.run(input);
                                    assert_eq!(
                                        actual[0], expected as u32,
                                        "raw={raw:016x} input={input:016x}"
                                    );
                                    if dst_size == 3 {
                                        assert_eq!(actual[1], (expected >> 32) as u32);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn directed_conversion_and_global_memory_share_int64() {
    let mut xs = vec![
        decode(word(2, true, 2, 3)),
        Instruction {
            address: 8,
            predicate: None,
            kind: K::GlobalMemory(ops::GlobalMemory {
                access: GlobalAccess::Store(LsSize2::B32),
                data: Register(2),
                address: Register(10),
                offset: 0,
                extended: true,
                cache: 0,
            }),
        },
        decode((0xe300 << 48) | (7 << 16)),
    ];
    for (i, x) in xs.iter_mut().enumerate() {
        x.address = i as u32 * 8;
    }
    let header = headers::parse(&[0; 32]).unwrap();
    for optimized in [false, true] {
        let words = if optimized {
            emit_with_header(&xs, Some(&header))
        } else {
            emit_unoptimized_with_header(&xs, Some(&header))
        }
        .unwrap()
        .words;
        let validation = spirv_validation::validate(&words);
        assert!(
            validation.status.success(),
            "{}",
            String::from_utf8_lossy(&validation.stderr)
        );
        let program = Program::new(&words);
        assert_eq!(
            program
                .instructions
                .iter()
                .filter(|(op, a)| *op == Op::TypeInt && a[1..] == [64, 0])
                .count(),
            1
        );
    }
}

#[test]
fn zero_register_is_not_extended_past_the_register_file() {
    for src in [254, 255] {
        let raw = (word(3, false, 3, 3) & !(255 << 20)) | (src << 20);
        let words = module(raw, 0, false);
        let validation = spirv_validation::validate(&words);
        assert!(
            validation.status.success(),
            "{}",
            String::from_utf8_lossy(&validation.stderr)
        );
        assert_eq!(Program::new(&words).run(u64::MAX), [0, 0]);
    }
}
