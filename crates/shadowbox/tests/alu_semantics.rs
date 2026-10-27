//! Execute the emitted straight-line ALU instructions, independently of the
//! emitter, to check values and flags as well as SPIR-V validation. This small
//! evaluator deliberately rejects unsupported executable operations. It does
//! not model branch execution, GPU rounding of transcendental functions, or
//! memory outside the one constant-buffer vector used by these cases.
use shadowbox::ir::ops;
use shadowbox::spirv::Op;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header as emit_with_header,
    headers,
    ir::{
        DoubleProductSign, DoubleSetResult, DoubleSource, DoubleSourceModifier, FloatCompare,
        FloatFlags, IMadSpType, Instruction, InstructionKind, IntCompare, IntegerCompareMode,
        MulNegation, Predicate, Register, RegisterPair, Relation, RoundingMode, SetOp, Source,
    },
};
use std::collections::HashMap;

#[path = "support/spirv_validation.rs"]
mod spirv_validation;

fn decoded(raw: u64) -> InstructionKind {
    decode_word([raw as u32, (raw >> 32) as u32], 0).kind
}

#[test]
fn fadd_cc_is_a_concrete_state_and_writes_only_zn() {
    let register = |cc: bool| {
        (0x5c58_u64 << 48) | (u64::from(cc) << 47) | (3 << 20) | (7 << 16) | (2 << 8) | 1
    };
    for (raw, written) in [(register(false), false), (register(true), true)] {
        let InstructionKind::FAdd(ops::FAdd { flags, .. }) = decoded(raw) else {
            panic!("{raw:016x} did not decode as FADD")
        };
        assert_eq!(
            flags,
            [FloatFlags::Kept, FloatFlags::Written][usize::from(written)]
        );
    }

    for (lhs, rhs, result, flags) in [
        (1.0_f32, -1.0_f32, 0.0_f32, [true, false, false, false]),
        (-2.0, 1.0, -1.0, [false, true, false, false]),
        (1.0, 2.0, 3.0, [false, false, false, false]),
        (f32::NAN, 1.0, f32::NAN, [false, false, false, false]),
    ] {
        let (registers, actual_flags) = run(
            &[
                mov(2, lhs.to_bits()),
                mov(3, rhs.to_bits()),
                decoded(register(true)),
            ],
            0,
        );
        if result.is_nan() {
            assert!(f32::from_bits(registers[1]).is_nan());
        } else {
            assert_eq!(registers[1], result.to_bits());
        }
        assert_eq!(actual_flags, flags);
    }

    // A false instruction guard preserves the destination and every flag.
    // First establish Z=true, then present a negative FADD.CC behind false P0.
    let mut xs = [
        mov(2, 1.0_f32.to_bits()),
        mov(3, (-1.0_f32).to_bits()),
        decoded(register(true)),
        mov(2, (-2.0_f32).to_bits()),
        mov(3, 1.0_f32.to_bits()),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, kind)| Instruction {
        address: i as u32 * 8,
        predicate: None,
        kind,
    })
    .collect::<Vec<_>>();
    let guarded = register(true) & !(7 << 16);
    xs.push(decode_word(
        [guarded as u32, (guarded >> 32) as u32],
        xs.len() as u32 * 8,
    ));
    let (registers, flags) = execute(&xs, 0);
    assert_eq!(registers[1], 0.0_f32.to_bits());
    assert_eq!(flags, [true, false, false, false]);

    // Every FADD operand road constructs the same closed .CC state. Directed
    // rounding remains rejected independently by the decoder boundary.
    for raw in [
        register(true),
        (0x4c58_u64 << 48) | (1 << 47) | (7 << 16) | (2 << 8) | 1,
        (0x3858_u64 << 48) | (1 << 47) | (0x3f800 << 20) | (7 << 16) | (2 << 8) | 1,
        (0x0800_u64 << 48)
            | (1 << 52)
            | (u64::from(1.0_f32.to_bits()) << 20)
            | (7 << 16)
            | (2 << 8)
            | 1,
    ] {
        assert!(matches!(
            decoded(raw),
            InstructionKind::FAdd(ops::FAdd {
                flags: FloatFlags::Written,
                ..
            })
        ));
    }
    let directed = register(true) | (1 << 39);
    assert!(matches!(decoded(directed), InstructionKind::Unknown(_)));
}

#[test]
fn fmul_cc_is_a_concrete_state_and_writes_only_zn() {
    // CC47 is shared by the register/cbuf/f20 roads; FMUL32I uses CC52.
    // envytools gm107.c:2004,2078,2124,2160; Ryujinx
    // InstDecoders.cs:2077-2141 and InstEmitFloatArithmetic.cs:435-480.
    let register = |cc: bool| {
        (0x5c68_u64 << 48) | (u64::from(cc) << 47) | (3 << 20) | (7 << 16) | (2 << 8) | 1
    };
    for (raw, written) in [(register(false), false), (register(true), true)] {
        let InstructionKind::FMul(ops::FMul { flags, .. }) = decoded(raw) else {
            panic!("{raw:016x} did not decode as FMUL")
        };
        assert_eq!(
            flags,
            [FloatFlags::Kept, FloatFlags::Written][usize::from(written)]
        );
    }

    for (lhs, rhs, result, flags) in [
        (0.0_f32, 3.0_f32, 0.0_f32, [true, false, false, false]),
        (-2.0, 2.0, -4.0, [false, true, false, false]),
        (2.0, 3.0, 6.0, [false, false, false, false]),
        (f32::NAN, 1.0, f32::NAN, [false, false, false, false]),
        (-0.0, 3.0, -0.0, [true, false, false, false]),
    ] {
        let (registers, actual_flags) = run(
            &[
                mov(2, lhs.to_bits()),
                mov(3, rhs.to_bits()),
                decoded(register(true)),
            ],
            0,
        );
        if result.is_nan() {
            assert!(f32::from_bits(registers[1]).is_nan());
        } else {
            assert_eq!(registers[1], result.to_bits());
        }
        assert_eq!(actual_flags, flags);
    }

    // A false instruction guard preserves the destination and all four flags.
    let mut xs = [
        mov(2, 0.0_f32.to_bits()),
        mov(3, 3.0_f32.to_bits()),
        decoded(register(true)),
        mov(2, (-2.0_f32).to_bits()),
        mov(3, 2.0_f32.to_bits()),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, kind)| Instruction {
        address: i as u32 * 8,
        predicate: None,
        kind,
    })
    .collect::<Vec<_>>();
    let guarded = register(true) & !(7 << 16);
    xs.push(decode_word(
        [guarded as u32, (guarded >> 32) as u32],
        xs.len() as u32 * 8,
    ));
    let (registers, flags) = execute(&xs, 0);
    assert_eq!(registers[1], 0.0_f32.to_bits());
    assert_eq!(flags, [true, false, false, false]);

    let roads = [
        register(true),
        (0x4c68_u64 << 48) | (1 << 47) | (7 << 16) | (2 << 8) | 1,
        (0x3868_u64 << 48) | (1 << 47) | (0x3f800 << 20) | (7 << 16) | (2 << 8) | 1,
        (0x1e00_u64 << 48)
            | (1 << 52)
            | (u64::from(1.0_f32.to_bits()) << 20)
            | (7 << 16)
            | (2 << 8)
            | 1,
    ];
    for raw in roads {
        assert!(matches!(
            decoded(raw),
            InstructionKind::FMul(ops::FMul {
                flags: FloatFlags::Written,
                ..
            })
        ));
    }
    for raw in [
        register(true) | (7 << 41),
        register(true) | (3 << 44),
        roads[3] | (3 << 53),
    ] {
        assert!(
            matches!(decoded(raw), InstructionKind::Unknown(_)),
            "{raw:016x}"
        );
    }

    // The exact accepted roads validate both before and after optimization.
    let header = headers::parse(&[0; 32]).unwrap();
    for raw in roads {
        let instruction = Instruction {
            address: 0,
            predicate: None,
            kind: decoded(raw),
        };
        let mut module = emit_with_header(&[instruction], Some(&header)).unwrap();
        for variant in ["raw", "optimized"] {
            let validation = spirv_validation::validate(&module.words);
            assert!(
                validation.status.success(),
                "{raw:016x} {variant}: {}",
                String::from_utf8_lossy(&validation.stderr)
            );
            module.optimize();
        }
    }
}

#[test]
fn ffma_cc_is_a_concrete_state_and_writes_only_final_zn() {
    let register = |cc: bool| {
        (0x5980_u64 << 48)
            | (u64::from(cc) << 47)
            | (4 << 39)
            | (3 << 20)
            | (7 << 16)
            | (2 << 8)
            | 1
    };
    for (raw, written) in [(register(false), false), (register(true), true)] {
        let InstructionKind::FFma(ops::FFma { flags, .. }) = decoded(raw) else {
            panic!("{raw:016x} did not decode as FFMA")
        };
        assert_eq!(
            flags,
            [FloatFlags::Kept, FloatFlags::Written][usize::from(written)]
        );
    }

    for (lhs, rhs, addend, result, flags) in [
        (
            0.0_f32,
            3.0_f32,
            0.0_f32,
            0.0_f32,
            [true, false, false, false],
        ),
        (-2.0, 2.0, 1.0, -3.0, [false, true, false, false]),
        (2.0, 3.0, 1.0, 7.0, [false, false, false, false]),
        (f32::NAN, 1.0, 0.0, f32::NAN, [false, false, false, false]),
        (-0.0, 3.0, -0.0, -0.0, [true, false, false, false]),
    ] {
        let (registers, actual_flags) = run(
            &[
                mov(2, lhs.to_bits()),
                mov(3, rhs.to_bits()),
                mov(4, addend.to_bits()),
                decoded(register(true)),
            ],
            0,
        );
        if result.is_nan() {
            assert!(f32::from_bits(registers[1]).is_nan());
        } else {
            assert_eq!(registers[1], result.to_bits());
        }
        assert_eq!(actual_flags, flags);
    }

    // Establish Z, then prove a false guard preserves both result and flags.
    let mut xs = [
        mov(2, 0.0_f32.to_bits()),
        mov(3, 3.0_f32.to_bits()),
        mov(4, 0.0_f32.to_bits()),
        decoded(register(true)),
        mov(2, (-2.0_f32).to_bits()),
        mov(3, 2.0_f32.to_bits()),
        mov(4, 1.0_f32.to_bits()),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, kind)| Instruction {
        address: i as u32 * 8,
        predicate: None,
        kind,
    })
    .collect::<Vec<_>>();
    let guarded = register(true) & !(7 << 16);
    xs.push(decode_word(
        [guarded as u32, (guarded >> 32) as u32],
        xs.len() as u32 * 8,
    ));
    let (registers, flags) = execute(&xs, 0);
    assert_eq!(registers[1], 0.0_f32.to_bits());
    assert_eq!(flags, [true, false, false, false]);

    // All five operand roads construct the same typed state and validate in
    // both raw and optimized SPIR-V. Directed rounding and mode 3 stay closed.
    let roads = [
        register(true),
        (0x4980_u64 << 48) | (1 << 47) | (4 << 39) | (7 << 16) | (2 << 8) | 1,
        (0x5180_u64 << 48) | (1 << 47) | (3 << 39) | (7 << 16) | (2 << 8) | 1,
        (0x3280_u64 << 48) | (1 << 47) | (4 << 39) | (0x3f800 << 20) | (7 << 16) | (2 << 8) | 1,
        (0x0c00_u64 << 48)
            | (1 << 52)
            | (u64::from(1.0_f32.to_bits()) << 20)
            | (7 << 16)
            | (2 << 8)
            | 1,
    ];
    let header = headers::parse(&[0; 32]).unwrap();
    for raw in roads {
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 0);
        assert!(matches!(
            instruction.kind,
            InstructionKind::FFma(ops::FFma {
                flags: FloatFlags::Written,
                ..
            })
        ));
        let mut module = emit_with_header(
            &[
                instruction,
                Instruction {
                    address: 8,
                    predicate: None,
                    kind: InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
                },
            ],
            Some(&header),
        )
        .unwrap();
        for mode in ["raw", "optimized"] {
            let validation = spirv_validation::validate(&module.words);
            assert!(
                validation.status.success(),
                "{raw:016x}/{mode}: {}",
                String::from_utf8_lossy(&validation.stderr)
            );
            module.optimize();
        }
    }
    for rejected in [register(true) | (1 << 51), register(true) | (3 << 53)] {
        assert!(matches!(decoded(rejected), InstructionKind::Unknown(_)));
    }
    let ffma32i = roads[4];
    assert!(matches!(
        decoded(ffma32i | (3 << 53)),
        InstructionKind::Unknown(_)
    ));
}

#[test]
fn fmnmx_cc_is_a_concrete_state_and_writes_only_final_zn() {
    // Ryujinx InstEmitFloatMinMax.cs:48-104 and InstEmitAluHelper.cs:124-138:
    // select minNum/maxNum first, then set ordered Z/N from the final result.
    let register = |cc: bool, min: bool| {
        (0x5c60_u64 << 48)
            | (u64::from(cc) << 47)
            | ((if min { 7 } else { 15 }) << 39)
            | (3 << 20)
            | (7 << 16)
            | (2 << 8)
            | 1
    };
    for (raw, written) in [(register(false, true), false), (register(true, true), true)] {
        let InstructionKind::FMnMx(ops::FMnMx { flags, .. }) = decoded(raw) else {
            panic!("{raw:016x} did not decode as FMNMX")
        };
        assert_eq!(
            flags,
            [FloatFlags::Kept, FloatFlags::Written][usize::from(written)]
        );
    }

    for (lhs, rhs, min, expected, flags) in [
        (0.0_f32, 1.0_f32, true, 0.0_f32, [true, false, false, false]),
        (-2.0, 1.0, true, -2.0, [false, true, false, false]),
        (-2.0, 1.0, false, 1.0, [false, false, false, false]),
        (f32::NAN, 0.0, true, 0.0, [true, false, false, false]),
        (
            f32::NAN,
            f32::NAN,
            false,
            f32::NAN,
            [false, false, false, false],
        ),
    ] {
        let (registers, actual_flags) = run(
            &[
                mov(2, lhs.to_bits()),
                mov(3, rhs.to_bits()),
                decoded(register(true, min)),
            ],
            0,
        );
        if expected.is_nan() {
            assert!(f32::from_bits(registers[1]).is_nan());
        } else {
            assert_eq!(registers[1], expected.to_bits());
        }
        assert_eq!(actual_flags, flags);
    }

    // Runtime selector P0 chooses min when true and max when false.
    let selected = (0x5c60_u64 << 48) | (1 << 47) | (3 << 20) | (7 << 16) | (2 << 8) | 1;
    for (predicate, expected, flags) in [
        (false, 1.0_f32, [false, false, false, false]),
        (true, -2.0_f32, [false, true, false, false]),
    ] {
        let (registers, actual_flags) = run(
            &[
                set_predicate(0, predicate),
                mov(2, (-2.0_f32).to_bits()),
                mov(3, 1.0_f32.to_bits()),
                decoded(selected),
            ],
            0,
        );
        assert_eq!(registers[1], expected.to_bits());
        assert_eq!(actual_flags, flags);
    }

    // Establish Z, then prove a false outer guard preserves result and flags.
    let mut xs = [
        mov(2, 0.0_f32.to_bits()),
        mov(3, 1.0_f32.to_bits()),
        decoded(register(true, true)),
        mov(2, (-2.0_f32).to_bits()),
        mov(3, 1.0_f32.to_bits()),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, kind)| Instruction {
        address: i as u32 * 8,
        predicate: None,
        kind,
    })
    .collect::<Vec<_>>();
    let guarded = register(true, true) & !(7 << 16);
    xs.push(decode_word(
        [guarded as u32, (guarded >> 32) as u32],
        xs.len() as u32 * 8,
    ));
    let (registers, flags) = execute(&xs, 0);
    assert_eq!(registers[1], 0.0_f32.to_bits());
    assert_eq!(flags, [true, false, false, false]);

    let exact: [u64; 6] = [
        0x5c60_8000_0007_0000,
        0x4c60_8000_0007_0000,
        0x3860_8000_0007_0000,
        0x3860_fa4f_51d7_a623,
        0x3862_ca04_ecf7_de9e,
        0x3863_e0ce_4567_8242,
    ];
    let header = headers::parse(&[0; 32]).unwrap();
    for raw in exact {
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 0);
        assert!(matches!(
            instruction.kind,
            InstructionKind::FMnMx(ops::FMnMx {
                flags: FloatFlags::Written,
                ..
            }) | InstructionKind::FMnMxSelect(ops::FMnMxSelect {
                flags: FloatFlags::Written,
                ..
            })
        ));
        let mut module = emit_with_header(&[instruction], Some(&header)).unwrap();
        for mode in ["raw", "optimized"] {
            let validation = spirv_validation::validate(&module.words);
            assert!(
                validation.status.success(),
                "{raw:016x}/{mode}: {}",
                String::from_utf8_lossy(&validation.stderr)
            );
            module.optimize();
        }
    }
}

#[test]
fn dfma_cc_is_concrete_fused_guarded_and_writes_only_final_zn() {
    let register = |cc: bool, guard: u64| {
        (0x5b70_u64 << 48)
            | (u64::from(cc) << 47)
            | (8 << 39)
            | (6 << 20)
            | (guard << 16)
            | (4 << 8)
            | 10
    };
    let mov64 = |register: u8, value: f64| {
        let bits = value.to_bits();
        [
            mov(register, bits as u32),
            mov(register + 1, (bits >> 32) as u32),
        ]
    };
    let execute_case = |lhs: f64, multiplier: f64, addend: f64| {
        let mut instructions = Vec::new();
        instructions.extend(mov64(4, lhs));
        instructions.extend(mov64(6, multiplier));
        instructions.extend(mov64(8, addend));
        instructions.push(decoded(register(true, 7)));
        let (registers, flags) = run(&instructions, 0);
        let bits = u64::from(registers[10]) | u64::from(registers[11]) << 32;
        (f64::from_bits(bits), flags)
    };

    for (lhs, multiplier, addend, expected, flags) in [
        (0.0, 2.0, 0.0, 0.0, [true, false, false, false]),
        (-2.0, 2.0, 1.0, -3.0, [false, true, false, false]),
        (2.0, 3.0, 1.0, 7.0, [false, false, false, false]),
        (f64::NAN, 1.0, 0.0, f64::NAN, [false, false, false, false]),
        (
            f64::INFINITY,
            2.0,
            0.0,
            f64::INFINITY,
            [false, false, false, false],
        ),
        (
            f64::NEG_INFINITY,
            2.0,
            0.0,
            f64::NEG_INFINITY,
            [false, true, false, false],
        ),
        (-0.0, 3.0, -0.0, -0.0, [true, false, false, false]),
    ] {
        let (actual, actual_flags) = execute_case(lhs, multiplier, addend);
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        assert_eq!(actual_flags, flags);
    }

    // A separately rounded multiply loses this residual; DFMA keeps it.
    let delta = 2.0_f64.powi(-27);
    let (fused, _) = execute_case(1.0 + delta, 1.0 - delta, -1.0);
    assert_eq!(fused, -2.0_f64.powi(-54));
    assert_ne!(fused, ((1.0 + delta) * (1.0 - delta)) - 1.0);

    let iadd_cc = (0x5c10_u64 << 48) | (1 << 47) | (3 << 20) | (7 << 16) | (2 << 8) | 1;
    for (lhs, rhs, expected_cv) in [
        (u32::MAX, 1, [true, false]),
        (0x7fff_ffff, 1, [false, true]),
    ] {
        let mut instructions = vec![mov(2, lhs), mov(3, rhs), decoded(iadd_cc)];
        instructions.extend(mov64(4, 2.0));
        instructions.extend(mov64(6, 3.0));
        instructions.extend(mov64(8, 1.0));
        instructions.push(decoded(register(true, 7)));
        let (_, flags) = run(&instructions, 0);
        assert_eq!([flags[2], flags[3]], expected_cv);
    }

    // A false outer guard preserves both the destination pair and all flags.
    let sentinel = 0x0123_4567_89ab_cdef_u64;
    let mut guarded = vec![mov(2, u32::MAX), mov(3, 1), decoded(iadd_cc)];
    guarded.extend(mov64(4, 2.0));
    guarded.extend(mov64(6, 3.0));
    guarded.extend(mov64(8, 1.0));
    guarded.push(mov(10, sentinel as u32));
    guarded.push(mov(11, (sentinel >> 32) as u32));
    let mut guarded = guarded
        .into_iter()
        .enumerate()
        .map(|(i, kind)| Instruction {
            address: i as u32 * 8,
            predicate: None,
            kind,
        })
        .collect::<Vec<_>>();
    let guarded_raw = register(true, 0);
    guarded.push(decode_word(
        [guarded_raw as u32, (guarded_raw >> 32) as u32],
        guarded.len() as u32 * 8,
    ));
    let (registers, flags) = execute(&guarded, 0);
    assert_eq!(
        u64::from(registers[10]) | u64::from(registers[11]) << 32,
        sentinel
    );
    assert_eq!(flags, [true, false, true, false]);

    let exact = [
        0x5b70_8000_0007_0000,
        0x5370_8000_0007_0000,
        0x4b70_8000_0007_0000,
        0x3670_8000_0007_0000,
        0x3673_debb_4e77_4fc0,
    ];
    let header = headers::parse(&[0; 32]).unwrap();
    for raw in exact {
        assert!(matches!(
            decoded(raw),
            InstructionKind::DoubleFma(ops::DoubleFma {
                flags: FloatFlags::Written,
                ..
            })
        ));
        assert!(matches!(
            decoded(raw & !(1 << 47)),
            InstructionKind::DoubleFma(ops::DoubleFma {
                flags: FloatFlags::Kept,
                ..
            })
        ));
        for rounding in 1..=3 {
            assert!(matches!(
                decoded(raw | rounding << 50),
                InstructionKind::Unknown(_)
            ));
        }
        let instruction = Instruction {
            address: 0,
            predicate: None,
            kind: decoded(raw),
        };
        let mut module = emit_with_header(&[instruction], Some(&header)).unwrap();
        for variant in ["raw", "optimized"] {
            let validation = spirv_validation::validate(&module.words);
            assert!(
                validation.status.success(),
                "{raw:016x} {variant}: {}",
                String::from_utf8_lossy(&validation.stderr)
            );
            module.optimize();
        }
    }
}

#[test]
fn dset_cc_result_modes_write_their_own_zn_and_preserve_cv() {
    let mov64 = |register: u8, value: f64| {
        let bits = value.to_bits();
        [
            mov(register, bits as u32),
            mov(register + 1, (bits >> 32) as u32),
        ]
    };
    let dset = |result, compare| {
        InstructionKind::DoubleSet(ops::DoubleSet {
            dst: Register(1),
            lhs: RegisterPair(Register(2)),
            rhs: DoubleSource::RegisterPair(RegisterPair(Register(4))),
            compare,
            set_op: SetOp::And,
            accum: Predicate {
                register: 7,
                inverted: false,
            },
            abs: [false; 2],
            negate: [false; 2],
            result,
        })
    };
    let execute_case = |lhs, rhs, result, compare| {
        let mut xs = Vec::new();
        xs.extend(mov64(2, lhs));
        xs.extend(mov64(4, rhs));
        xs.push(dset(result, compare));
        run(&xs, 0)
    };

    for (lhs, rhs, result, compare, expected, flags) in [
        (
            1.0,
            1.0,
            DoubleSetResult::Mask(FloatFlags::Written),
            FloatCompare::Ordered(Relation::Eq),
            u32::MAX,
            [false, true, false, false],
        ),
        (
            1.0,
            2.0,
            DoubleSetResult::Mask(FloatFlags::Written),
            FloatCompare::Ordered(Relation::Eq),
            0,
            [true, false, false, false],
        ),
        (
            -0.0,
            0.0,
            DoubleSetResult::FloatOne(FloatFlags::Written),
            FloatCompare::Ordered(Relation::Eq),
            1.0_f32.to_bits(),
            [false, false, false, false],
        ),
        (
            f64::INFINITY,
            f64::NEG_INFINITY,
            DoubleSetResult::FloatOne(FloatFlags::Written),
            FloatCompare::Ordered(Relation::Gt),
            1.0_f32.to_bits(),
            [false, false, false, false],
        ),
        (
            f64::NAN,
            1.0,
            DoubleSetResult::FloatOne(FloatFlags::Written),
            FloatCompare::Nan,
            1.0_f32.to_bits(),
            [false, false, false, false],
        ),
    ] {
        let (registers, actual_flags) = execute_case(lhs, rhs, result, compare);
        assert_eq!(registers[1], expected);
        assert_eq!(actual_flags, flags);
    }

    // Seed C/V independently, then prove DSET.CC changes only Z/N.
    let iadd_cc = (0x5c10_u64 << 48) | (1 << 47) | (3 << 20) | (7 << 16) | (2 << 8) | 1;
    let mut xs = vec![mov(2, u32::MAX), mov(3, 1), decoded(iadd_cc)];
    xs.extend(mov64(2, 1.0));
    xs.extend(mov64(4, 1.0));
    xs.push(dset(
        DoubleSetResult::Mask(FloatFlags::Written),
        FloatCompare::Ordered(Relation::Eq),
    ));
    assert_eq!(run(&xs, 0).1, [false, true, true, false]);

    // A false outer guard preserves both the destination and all four flags.
    let mut guarded = xs[..3]
        .iter()
        .enumerate()
        .map(|(i, kind)| Instruction {
            address: i as u32 * 8,
            predicate: None,
            kind: *kind,
        })
        .collect::<Vec<_>>();
    guarded.push(Instruction {
        address: guarded.len() as u32 * 8,
        predicate: None,
        kind: mov(1, 0x1234_5678),
    });
    guarded.push(Instruction {
        address: guarded.len() as u32 * 8,
        predicate: Some(Predicate {
            register: 0,
            inverted: false,
        }),
        kind: dset(
            DoubleSetResult::FloatOne(FloatFlags::Written),
            FloatCompare::True,
        ),
    });
    let (registers, flags) = execute(&guarded, 0);
    assert_eq!(registers[1], 0x1234_5678);
    assert_eq!(flags, [true, false, true, false]);
}

#[test]
fn double_add_multiply_rn_cc_are_closed_guarded_and_write_only_zn() {
    let dadd = InstructionKind::DoubleAdd(ops::DoubleAdd {
        dst: RegisterPair(Register(10)),
        lhs: RegisterPair(Register(2)),
        rhs: DoubleSource::RegisterPair(RegisterPair(Register(4))),
        modifiers: [DoubleSourceModifier::Plain; 2],
        round: RoundingMode::NearestEven,
        flags: FloatFlags::Written,
    });
    let dmul = InstructionKind::DoubleMultiply(ops::DoubleMultiply {
        dst: RegisterPair(Register(10)),
        lhs: RegisterPair(Register(2)),
        rhs: DoubleSource::RegisterPair(RegisterPair(Register(4))),
        sign: DoubleProductSign::Positive,
        flags: FloatFlags::Written,
    });
    let mov64 = |register: u8, value: f64| {
        let bits = value.to_bits();
        [
            mov(register, bits as u32),
            mov(register + 1, (bits >> 32) as u32),
        ]
    };
    let execute_binary = |kind: InstructionKind, lhs: f64, rhs: f64| {
        let mut instructions = Vec::new();
        instructions.extend(mov64(2, lhs));
        instructions.extend(mov64(4, rhs));
        instructions.push(kind);
        let (registers, flags) = run(&instructions, 0);
        let bits = u64::from(registers[10]) | u64::from(registers[11]) << 32;
        (f64::from_bits(bits), flags)
    };

    for (lhs, rhs, expected, flags) in [
        (0.0, 0.0, 0.0, [true, false, false, false]),
        (-0.0, -0.0, -0.0, [true, false, false, false]),
        (-3.0, 1.0, -2.0, [false, true, false, false]),
        (f64::NAN, 1.0, f64::NAN, [false, false, false, false]),
        (
            f64::INFINITY,
            1.0,
            f64::INFINITY,
            [false, false, false, false],
        ),
    ] {
        let (actual, actual_flags) = execute_binary(dadd, lhs, rhs);
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        assert_eq!(actual_flags, flags);
    }
    for (lhs, rhs, expected, flags) in [
        (0.0, 3.0, 0.0, [true, false, false, false]),
        (-0.0, 3.0, -0.0, [true, false, false, false]),
        (-2.0, 3.0, -6.0, [false, true, false, false]),
        (f64::NAN, 1.0, f64::NAN, [false, false, false, false]),
        (
            f64::INFINITY,
            2.0,
            f64::INFINITY,
            [false, false, false, false],
        ),
    ] {
        let (actual, actual_flags) = execute_binary(dmul, lhs, rhs);
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
        assert_eq!(actual_flags, flags);
    }

    // C/V survive both operations, and a false outer guard preserves the
    // destination pair and all four flags.
    let iadd_cc = (0x5c10_u64 << 48) | (1 << 47) | (3 << 20) | (7 << 16) | (2 << 8) | 1;
    for kind in [dadd, dmul] {
        let mut instructions = vec![mov(2, u32::MAX), mov(3, 1), decoded(iadd_cc)];
        instructions.extend(mov64(2, -2.0));
        instructions.extend(mov64(4, 3.0));
        instructions.push(kind);
        assert_eq!(run(&instructions, 0).1[2..], [true, false]);
    }
    let sentinel = 0x0123_4567_89ab_cdef_u64;
    for kind in [dadd, dmul] {
        let mut instructions = vec![mov(2, u32::MAX), mov(3, 1), decoded(iadd_cc)];
        instructions.extend(mov64(2, -2.0));
        instructions.extend(mov64(4, 3.0));
        instructions.extend(mov64(10, f64::from_bits(sentinel)));
        let mut instructions = instructions
            .into_iter()
            .enumerate()
            .map(|(i, kind)| Instruction {
                address: i as u32 * 8,
                predicate: None,
                kind,
            })
            .collect::<Vec<_>>();
        instructions.push(Instruction {
            address: instructions.len() as u32 * 8,
            predicate: Some(Predicate {
                register: 0,
                inverted: false,
            }),
            kind,
        });
        let (registers, flags) = execute(&instructions, 0);
        assert_eq!(
            u64::from(registers[10]) | u64::from(registers[11]) << 32,
            sentinel
        );
        assert_eq!(flags, [true, false, true, false]);
    }

    let exact = [
        0x4c76_fc03_2b36_9161,
        0x3872_9a45_25e5_1fed,
        0x3971_d03d_ef95_759d,
        0x3976_fc48_5c55_51ee,
        0x5c86_da71_bf40_b4f2,
        0x3986_c87c_3b2e_5d0d,
        0x5c80_8000_0007_0000,
        0x5c81_8000_0007_0000,
        0x5c80_c82a_d4c7_25a4,
        0x5c81_c82a_d4c7_25a4,
        0x5c80_dc03_b1f7_0950,
    ];
    let header = headers::parse(&[0; 32]).unwrap();
    for raw in exact {
        let flags = match decoded(raw) {
            InstructionKind::DoubleAdd(ops::DoubleAdd { flags, .. })
            | InstructionKind::DoubleMultiply(ops::DoubleMultiply { flags, .. }) => flags,
            kind => panic!("{raw:016x} decoded as {kind:?}"),
        };
        assert_eq!(flags, FloatFlags::Written);
        for bit in (0..64).filter(|bit| *bit != 47) {
            if exact.contains(&(raw ^ (1_u64 << bit))) {
                continue;
            }
            assert!(
                !matches!(
                    decoded(raw ^ (1_u64 << bit)),
                    InstructionKind::DoubleAdd(ops::DoubleAdd {
                        flags: FloatFlags::Written,
                        ..
                    }) | InstructionKind::DoubleMultiply(ops::DoubleMultiply {
                        flags: FloatFlags::Written,
                        ..
                    })
                ),
                "{raw:016x} one-bit neighbor {bit} escaped the exact whitelist"
            );
        }
        assert!(matches!(
            decoded(raw & !(1 << 47)),
            InstructionKind::DoubleAdd(ops::DoubleAdd {
                flags: FloatFlags::Kept,
                ..
            }) | InstructionKind::DoubleMultiply(ops::DoubleMultiply {
                flags: FloatFlags::Kept,
                ..
            })
        ));
        for rounding in 1..=3 {
            assert!(matches!(
                decoded(raw | rounding << 39),
                InstructionKind::Unknown(_)
            ));
        }
        let instruction = Instruction {
            address: 0,
            predicate: None,
            kind: decoded(raw),
        };
        let mut module = emit_with_header(&[instruction], Some(&header)).unwrap();
        for variant in ["raw", "optimized"] {
            let validation = spirv_validation::validate(&module.words);
            assert!(
                validation.status.success(),
                "{raw:016x} {variant}: {}",
                String::from_utf8_lossy(&validation.stderr)
            );
            module.optimize();
        }
    }
    for raw in [
        0x5c70_8000_0007_0000,
        0x4c70_8000_0007_0000,
        0x3870_8000_0007_0001,
        0x4c80_8000_0007_0000,
        0x3880_8000_0007_0000,
    ] {
        assert!(
            matches!(decoded(raw), InstructionKind::Unknown(_)),
            "{raw:016x} unexpectedly decoded"
        );
    }
}

fn mov(dst: u8, value: u32) -> InstructionKind {
    InstructionKind::Mov(ops::Mov {
        quad_lanes: 0xf,
        dst: Register(dst),
        src: Source::Immediate(value),
    })
}

fn mov_i20(payload: u32, mask: u8) -> u64 {
    (0x3898_u64 << 48)
        | (u64::from(payload >> 19) << 56)
        | (u64::from(payload & 0x7ffff) << 20)
        | (u64::from(mask) << 39)
        | (7 << 16)
        | 1
}

#[test]
fn mov_i20_is_typed_zero_extended_and_exact_bounded() {
    for payload in [0, 1, 0x7ffff, 0x80000, 0xfffff] {
        let raw = mov_i20(payload, 0xf);
        let InstructionKind::Mov(ops::Mov {
            dst,
            src: Source::Immediate(immediate),
            quad_lanes,
        }) = decoded(raw)
        else {
            panic!("{raw:016x} did not decode as compact MOV.I20")
        };
        assert_eq!(dst, Register(1));
        assert_eq!(immediate, payload);
        assert_eq!(quad_lanes, 0xf);
        assert_eq!(run(&[decoded(raw)], 0).0[1], payload);
    }

    for mask in 0..=0xf {
        let InstructionKind::Mov(ops::Mov { quad_lanes, .. }) = decoded(mov_i20(0x80001, mask))
        else {
            panic!("mask {mask:x} did not decode as compact MOV.I20")
        };
        assert_eq!(quad_lanes, mask);
    }

    // Adjacent RRO and SEL opcode rows retain their own meanings.
    for raw in [0x3890_u64 << 48, 0x38a0_u64 << 48] {
        assert!(
            !matches!(decoded(raw), InstructionKind::Mov(_)),
            "{raw:016x}"
        );
    }

    let header = headers::parse(&[0; 32]).unwrap();
    for payload in [0, 1, 0x7ffff, 0x80000, 0xfffff] {
        let raw = mov_i20(payload, 0xf);
        let instruction = Instruction {
            address: 0,
            predicate: None,
            kind: decoded(raw),
        };
        let mut module = emit_with_header(&[instruction], Some(&header)).unwrap();
        for variant in ["raw", "optimized"] {
            let validation = spirv_validation::validate(&module.words);
            assert!(
                validation.status.success(),
                "{raw:016x} {variant}: {}",
                String::from_utf8_lossy(&validation.stderr)
            );
            module.optimize();
        }
    }
}

fn run(kinds: &[InstructionKind], cbuf: u32) -> (Vec<u32>, [bool; 4]) {
    let xs = kinds
        .iter()
        .enumerate()
        .map(|(i, kind)| Instruction {
            address: i as u32 * 8,
            predicate: None,
            kind: *kind,
        })
        .collect::<Vec<_>>();
    execute(&xs, cbuf)
}

fn execute(xs: &[Instruction], cbuf: u32) -> (Vec<u32>, [bool; 4]) {
    let (regs, cc, _) = run_instructions(xs, cbuf);
    (regs, cc)
}

fn run_instructions(xs: &[Instruction], cbuf: u32) -> (Vec<u32>, [bool; 4], [bool; 8]) {
    let header = headers::parse(&[0; 32]).unwrap();
    let words = emit_with_header(xs, Some(&header)).unwrap().words;
    let mut values: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut memory: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut variables = Vec::new();
    let mut labels = HashMap::new();
    let mut at = 5;
    while at < words.len() {
        if Op::of(words[at]) == Op::Label {
            labels.insert(words[at + 1], at);
        }
        at += (words[at] >> 16) as usize;
    }
    at = 5;
    let mut steps = 0;
    while at < words.len() {
        steps += 1;
        assert!(steps < 100_000, "unexpected evaluator loop");
        let count = (words[at] >> 16) as usize;
        let opcode = Op::of(words[at]);
        let a = &words[at + 1..at + count];
        let scalar = |id| values[&id][0];
        let f = |id| f32::from_bits(scalar(id));
        let wide = |id| u64::from(values[&id][0]) | u64::from(values[&id][1]) << 32;
        let result = match opcode {
            Op::ConstantTrue => Some(vec![1]),
            Op::ConstantFalse => Some(vec![0]),
            Op::Constant => Some(a[2..].to_vec()),
            Op::Variable => {
                if a[2] == 7 {
                    variables.push(a[1]);
                }
                memory.insert(a[1], if a[2] == 2 { vec![cbuf; 4] } else { vec![0] });
                None
            }
            Op::Load => Some(memory[&a[2]].clone()),
            Op::Store => {
                memory.insert(a[0], values[&a[1]].clone());
                None
            }
            Op::AccessChain => {
                // These cases address c[0][0] only. Track the emitted access
                // chain rather than replacing the source in the IR.
                assert_eq!(scalar(a[3]), 0);
                assert_eq!(scalar(a[4]), 0);
                memory.insert(a[1], memory[&a[2]].clone());
                None
            }
            Op::CompositeExtract => Some(vec![values[&a[2]][a[3] as usize]]),
            Op::CompositeConstruct => Some(
                a[2..]
                    .iter()
                    .flat_map(|id| values[id].iter().copied())
                    .collect(),
            ),
            Op::Bitcast => Some(values[&a[2]].clone()),
            Op::FNegate => Some(if values[&a[2]].len() == 2 {
                vec![values[&a[2]][0], values[&a[2]][1] ^ 0x8000_0000]
            } else {
                vec![scalar(a[2]) ^ 0x8000_0000]
            }),
            Op::IAdd => Some(vec![scalar(a[2]).wrapping_add(scalar(a[3]))]),
            Op::ISub => Some(vec![scalar(a[2]).wrapping_sub(scalar(a[3]))]),
            Op::IMul => Some(vec![scalar(a[2]).wrapping_mul(scalar(a[3]))]),
            Op::UMulExtended => {
                let product = u64::from(scalar(a[2])) * u64::from(scalar(a[3]));
                Some(vec![product as u32, (product >> 32) as u32])
            }
            Op::SMulExtended => {
                let product = i64::from(scalar(a[2]) as i32) * i64::from(scalar(a[3]) as i32);
                let product = product as u64;
                Some(vec![product as u32, (product >> 32) as u32])
            }
            Op::FDiv => {
                assert_ne!(f(a[3]), 0.0, "division input was not sanitized");
                Some(vec![(f(a[2]) / f(a[3])).to_bits()])
            }
            Op::FAdd | Op::FMul if values[&a[2]].len() == 2 => {
                let lhs = f64::from_bits(wide(a[2]));
                let rhs = f64::from_bits(wide(a[3]));
                let bits = if opcode == Op::FAdd {
                    (lhs + rhs).to_bits()
                } else {
                    (lhs * rhs).to_bits()
                };
                Some(vec![bits as u32, (bits >> 32) as u32])
            }
            Op::FAdd => Some(vec![(f(a[2]) + f(a[3])).to_bits()]),
            Op::FMul => Some(vec![(f(a[2]) * f(a[3])).to_bits()]),
            Op::IAddCarry => {
                let sum = u64::from(scalar(a[2])) + u64::from(scalar(a[3]));
                Some(vec![sum as u32, (sum >> 32) as u32])
            }
            Op::LogicalNotEqual => {
                Some(vec![u32::from((scalar(a[2]) != 0) != (scalar(a[3]) != 0))])
            }
            Op::LogicalOr => Some(vec![u32::from(scalar(a[2]) != 0 || scalar(a[3]) != 0)]),
            Op::LogicalAnd => Some(vec![u32::from(scalar(a[2]) != 0 && scalar(a[3]) != 0)]),
            Op::LogicalNot => Some(vec![u32::from(scalar(a[2]) == 0)]),
            Op::Select => Some(values[&if scalar(a[2]) != 0 { a[3] } else { a[4] }].clone()),
            Op::IEqual => Some(vec![u32::from(scalar(a[2]) == scalar(a[3]))]),
            Op::INotEqual => Some(vec![u32::from(scalar(a[2]) != scalar(a[3]))]),
            // Mesa spirv.core.grammar.json:2011-2025, OpUGreaterThan.
            Op::UGreaterThan => Some(vec![u32::from(scalar(a[2]) > scalar(a[3]))]),
            Op::ULessThan => Some(vec![u32::from(scalar(a[2]) < scalar(a[3]))]),
            Op::SGreaterThan => Some(vec![u32::from(scalar(a[2]) as i32 > scalar(a[3]) as i32)]),
            Op::UGreaterThanEqual => Some(vec![u32::from(scalar(a[2]) >= scalar(a[3]))]),
            Op::SGreaterThanEqual => {
                Some(vec![u32::from(scalar(a[2]) as i32 >= scalar(a[3]) as i32)])
            }
            Op::SLessThan => Some(vec![u32::from((scalar(a[2]) as i32) < scalar(a[3]) as i32)]),
            Op::ULessThanEqual => Some(vec![u32::from(scalar(a[2]) <= scalar(a[3]))]),
            Op::SLessThanEqual => Some(vec![u32::from(scalar(a[2]) as i32 <= scalar(a[3]) as i32)]),
            Op::BitFieldSExtract | Op::BitFieldUExtract => {
                let offset = scalar(a[3]);
                let width = scalar(a[4]);
                let bits = scalar(a[2]) >> offset;
                Some(vec![if opcode == Op::BitFieldSExtract {
                    ((bits << (32 - width)) as i32 >> (32 - width)) as u32
                } else {
                    bits & (u32::MAX >> (32 - width))
                }])
            }
            Op::IsNan => Some(vec![u32::from(if values[&a[2]].len() == 2 {
                f64::from_bits(wide(a[2])).is_nan()
            } else {
                f(a[2]).is_nan()
            })]),
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
                let double = values[&a[2]].len() == 2;
                let lhs = if double {
                    f64::from_bits(wide(a[2]))
                } else {
                    f64::from(f(a[2]))
                };
                let rhs = if double {
                    f64::from_bits(wide(a[3]))
                } else {
                    f64::from(f(a[3]))
                };
                let nan = lhs.is_nan() || rhs.is_nan();
                let relation = match opcode.number() & !1 {
                    180 => lhs == rhs,
                    182 => lhs != rhs,
                    184 => lhs < rhs,
                    186 => lhs > rhs,
                    188 => lhs <= rhs,
                    190 => lhs >= rhs,
                    _ => unreachable!(),
                };
                Some(vec![u32::from(if opcode.number() & 1 == 1 {
                    nan || relation
                } else {
                    !nan && relation
                })])
            }
            // OpBitFieldInsert (Mesa spirv.h:1924); source semantics:
            // Ryujinx InstEmitBitfield.cs:154-162. Mask in u64 handles
            // offset32/count0 and full-width insertion without host overflow.
            Op::BitFieldInsert => {
                let offset = scalar(a[4]);
                let count = scalar(a[5]);
                assert!(offset <= 32 && count <= 32 - offset);
                let mask = (((1u64 << count) - 1) << offset) as u32;
                Some(vec![
                    (scalar(a[2]) & !mask) | ((u64::from(scalar(a[3])) << offset) as u32 & mask),
                ])
            }
            Op::ShiftRightLogical => Some(vec![scalar(a[2]) >> scalar(a[3])]),
            Op::ShiftRightArithmetic => Some(vec![((scalar(a[2]) as i32) >> scalar(a[3])) as u32]),
            Op::ShiftLeftLogical => Some(vec![scalar(a[2]) << scalar(a[3])]),
            Op::BitwiseOr => Some(vec![scalar(a[2]) | scalar(a[3])]),
            Op::BitwiseXor => Some(vec![scalar(a[2]) ^ scalar(a[3])]),
            Op::Not => Some(vec![!scalar(a[2])]),
            Op::BitwiseAnd => Some(vec![scalar(a[2]) & scalar(a[3])]),
            Op::ExtInst => {
                let double = values[&a[4]].len() == 2;
                if double {
                    assert_eq!(a[3], 50, "unexpected binary64 GLSL instruction");
                    let result = f64::from_bits(wide(a[4]))
                        .mul_add(f64::from_bits(wide(a[5])), f64::from_bits(wide(a[6])))
                        .to_bits();
                    Some(vec![result as u32, (result >> 32) as u32])
                } else {
                    let x = f(a[4]);
                    let result = match a[3] {
                        4 => x.abs().to_bits(),
                        13 => x.sin().to_bits(),
                        14 => x.cos().to_bits(),
                        29 => x.exp2().to_bits(),
                        30 => {
                            assert!(x > 0.0 || x.is_nan(), "log input was not sanitized");
                            x.log2().to_bits()
                        }
                        31 => {
                            assert!(x >= 0.0 || x.is_nan());
                            x.sqrt().to_bits()
                        }
                        32 => {
                            assert!(x > 0.0 || x.is_nan());
                            (1.0 / x.sqrt()).to_bits()
                        }
                        // GLSL.std.450 UMin/SMin/UMax/SMax.
                        38 => scalar(a[4]).min(scalar(a[5])),
                        39 => (scalar(a[4]) as i32).min(scalar(a[5]) as i32) as u32,
                        41 => scalar(a[4]).max(scalar(a[5])),
                        42 => (scalar(a[4]) as i32).max(scalar(a[5]) as i32) as u32,
                        50 => x.mul_add(f(a[5]), f(a[6])).to_bits(),
                        79 => {
                            let y = f(a[5]);
                            if x.is_nan() {
                                y.to_bits()
                            } else if y.is_nan() {
                                x.to_bits()
                            } else {
                                x.min(y).to_bits()
                            }
                        }
                        80 => {
                            let y = f(a[5]);
                            if x.is_nan() {
                                y.to_bits()
                            } else if y.is_nan() {
                                x.to_bits()
                            } else {
                                x.max(y).to_bits()
                            }
                        }
                        81 => x.max(f(a[5])).min(f(a[6])).to_bits(),
                        code => panic!("unexpected GLSL.std.450 instruction {code}"),
                    };
                    Some(vec![result])
                }
            }
            Op::Branch => {
                at = labels[&a[0]];
                continue;
            }
            Op::BranchConditional => {
                at = labels[&if scalar(a[0]) != 0 { a[1] } else { a[2] }];
                continue;
            }
            Op::Switch => {
                let selector = scalar(a[0]);
                let target = a[2..]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .find(|case| case[0] == selector)
                    .map_or(a[1], |case| case[1]);
                at = labels[&target];
                continue;
            }
            Op::Return => break,
            // Declarations and the emitter's dispatch scaffolding. These
            // inputs have no Maxwell branches and execute once in order.
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
            op => panic!("unexpected SPIR-V opcode {}", op.number()),
        };
        if let Some(result) = result {
            values.insert(a[1], result);
        }
        at += count;
    }
    // emitter.rs declares R0..R255, P0..P7, then Z/S/C/V, then dispatch PC.
    assert_eq!(variables.len(), 269);
    let regs = variables[..256].iter().map(|id| memory[id][0]).collect();
    let cc = std::array::from_fn(|i| memory[&variables[264 + i]][0] != 0);
    let predicates = std::array::from_fn(|i| memory[&variables[256 + i]][0] != 0);
    (regs, cc, predicates)
}

#[test]
// min/max-number maps NaN to zero. Rust clamp preserves NaN.
#[allow(clippy::manual_clamp)]
fn mufu_modifiers_and_exceptional_inputs() {
    // Encodings: envytools envydis/gm107.c:1691-1699,2062. Semantics:
    // Ryujinx Instructions/InstEmitMultifunction.cs:36-83, and GLSL.std.450
    // section 2. Only exact, simple ordinary results are compared here.
    for op in [0, 1, 2, 3, 4, 5, 8] {
        for input in [
            0.0_f32,
            -0.0,
            1.0,
            -1.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
        ] {
            // Skip trig at infinity, whose result is outside our exact cases.
            if op <= 1 && !input.is_finite() {
                continue;
            }
            for abs in [false, true] {
                for neg in [false, true] {
                    for sat in [false, true] {
                        let raw = (0x5080_u64 << 48)
                            | 7 << 16
                            | 2 << 8
                            | 1
                            | op << 20
                            | u64::from(abs) << 46
                            | u64::from(neg) << 48
                            | u64::from(sat) << 50;
                        let (regs, _) = run(&[mov(2, input.to_bits()), decoded(raw)], 0);
                        let x = if abs { input.abs() } else { input };
                        let x = if neg { -x } else { x };
                        let expected = match op {
                            0 => x.cos(),
                            1 => x.sin(),
                            2 => x.exp2(),
                            3 => x.log2(),
                            4 => 1.0 / x,
                            5 => 1.0 / x.sqrt(),
                            8 => x.sqrt(),
                            _ => unreachable!(),
                        };
                        let expected = if sat {
                            expected.max(0.0).min(1.0)
                        } else {
                            expected
                        };
                        let actual = f32::from_bits(regs[1]);
                        assert!(actual.is_nan() && expected.is_nan() || actual.to_bits() == expected.to_bits(),
                            "op {op}, input {input}, abs {abs}, neg {neg}, sat {sat}: {actual:?} != {expected:?}");
                    }
                }
            }
        }
    }
}

#[test]
fn cbuf_integer_modifiers_and_flags() {
    // envytools envydis/gm107.c:2088-2089. Ryujinx
    // Instructions/InstEmitIntegerArithmetic.cs:326-336,469-476,682-698.
    for opcode in [0x4c10_u64, 0x4c18] {
        for shift in [0, 5, 31] {
            if opcode == 0x4c10 && shift != 0 {
                continue;
            }
            for mode in 0..=2 {
                for a in [0_u32, 1, u32::MAX, 0x8000_0000] {
                    for b in [0_u32, 1, u32::MAX, 0x8000_0000] {
                        let raw = opcode << 48
                            | 7 << 16
                            | 2 << 8
                            | 1
                            | mode << 48
                            | 1 << 47
                            | shift << 39;
                        let (regs, cc) = run(&[mov(2, a), decoded(raw)], b);
                        let a = a << shift;
                        let a = if mode == 2 { a.wrapping_neg() } else { a };
                        let b = if mode == 1 { b.wrapping_neg() } else { b };
                        let result = a.wrapping_add(b);
                        assert_eq!(regs[1], result);
                        assert_eq!(
                            cc,
                            [
                                result == 0,
                                result >> 31 != 0,
                                u64::from(a) + u64::from(b) > u64::from(u32::MAX),
                                ((result ^ a) & !(a ^ b)) >> 31 != 0
                            ]
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn iadd_extended_carry_and_zero_chain() {
    // Ryujinx InstEmitIntegerArithmetic.cs:331-336,672-680 and
    // InstEmitAluHelper.cs:102-114. .X includes carry and chains zero.
    let add_cc = decoded(0x5c10_u64 << 48 | 7 << 16 | 2 << 8 | 3 << 20 | 1 << 47 | 1);
    let add_x_cc = decoded(0x5c10_u64 << 48 | 7 << 16 | 4 << 8 | 5 << 20 | 1 << 43 | 1 << 47 | 6);
    for (a, b, old_zero, carry) in [
        (u32::MAX, 1, true, true),
        (u32::MAX, 2, false, true),
        (1, 1, false, false),
        (0, 0, true, false),
    ] {
        let (regs, cc) = run(
            &[
                mov(2, a),
                mov(3, b),
                add_cc,
                mov(4, u32::MAX),
                mov(5, u32::from(!carry)),
                add_x_cc,
            ],
            0,
        );
        assert_eq!(regs[6], 0);
        assert_eq!(cc, [old_zero, false, true, false]);
    }
}

#[test]
fn iadd32i_carry_overflow_negation_and_extended_edge_matrix() {
    // Encoding: envytools gm107.c:2162. Semantics: Ryujinx
    // InstEmitIntegerArithmetic.cs:43-51,317-341,665-699 and
    // InstEmitAluHelper.cs:95-158. These expectations are calculated here
    // from the integer operands, independently of the emitted SPIR-V.
    for (lhs, immediate, expected, flags) in [
        (u32::MAX, 1, 0, [true, false, true, false]),
        (0x7fff_ffff, 1, 0x8000_0000, [false, true, false, true]),
        (
            0x8000_0000,
            u32::MAX,
            0x7fff_ffff,
            [false, false, true, true],
        ),
    ] {
        let raw = (0x1c00_u64 << 48)
            | (u64::from(immediate) << 20)
            | (1 << 52)
            | (7 << 16)
            | (2 << 8)
            | 1;
        let (registers, actual_flags) = run(&[mov(2, lhs), decoded(raw)], 0);
        assert_eq!(registers[1], expected, "{raw:016x}");
        assert_eq!(actual_flags, flags, "{raw:016x}");
    }

    // The register negate (bit 56) is a concrete decoded state. Bits 55..56
    // are nvdisasm SM52's PSign32: value 1 (bit 55 alone, Ryujinx's NegB)
    // is "undefined value 0x1 for table 'PSign32'", so it stays Unknown.
    let raw = (0x1c00_u64 << 48) | (2_u64 << 20) | (1 << 56) | (7 << 16) | (2 << 8) | 1;
    assert_eq!(
        run(&[mov(2, 5), decoded(raw)], 0).0[1],
        3_u32.wrapping_neg()
    );
    let raw = (0x1c00_u64 << 48) | (2_u64 << 20) | (1 << 55) | (7 << 16) | (2 << 8) | 1;
    assert!(matches!(decoded(raw), InstructionKind::Unknown(_)));

    let establish_cc = |a: u8, b: u8| {
        decoded(
            (0x5c10_u64 << 48)
                | (1 << 47)
                | (u64::from(b) << 20)
                | (7 << 16)
                | (u64::from(a) << 8)
                | 9,
        )
    };
    let extended = decoded((0x1c00_u64 << 48) | (1 << 53) | (1 << 52) | (7 << 16) | (4 << 8) | 6);
    for (lower_a, lower_b, upper, expected, flags) in [
        (u32::MAX, 1, u32::MAX, 0, [true, false, true, false]),
        (1, 1, 0, 0, [false, false, false, false]),
        (
            u32::MAX,
            2,
            0x7fff_ffff,
            0x8000_0000,
            [false, true, false, true],
        ),
    ] {
        let (registers, actual_flags) = run(
            &[
                mov(2, lower_a),
                mov(3, lower_b),
                establish_cc(2, 3),
                mov(4, upper),
                extended,
            ],
            0,
        );
        assert_eq!(registers[6], expected);
        assert_eq!(actual_flags, flags);
    }
}

#[test]
fn xmad_cc_and_imnmx_cc_exact_state_matrices() {
    // XMAD.CC observes the pre-MRG product+addend (Ryujinx
    // InstEmitIntegerArithmetic.cs:261-274,665-699).
    let xmad = |merge: bool| {
        decoded(
            (0x5b00_u64 << 48)
                | (1 << 47)
                | (u64::from(merge) << 37)
                | (4 << 39)
                | (3 << 20)
                | (7 << 16)
                | (2 << 8)
                | 1,
        )
    };
    for (a, b, c, expected, flags) in [
        (0xffff, 0xffff, 0x1ffff, 0, [true, false, true, false]),
        (
            0x8000,
            1,
            0x7fff_8000,
            0x8000_0000,
            [false, true, false, true],
        ),
    ] {
        for merge in [false, true] {
            let (registers, actual_flags) = run(&[mov(2, a), mov(3, b), mov(4, c), xmad(merge)], 0);
            let destination = if merge {
                expected & 0xffff | b << 16
            } else {
                expected
            };
            assert_eq!(registers[1], destination);
            assert_eq!(actual_flags, flags, "merge {merge}");
        }
    }

    // Each XMAD operand form admits the same concrete .CC state.
    for (raw, b, c, cbuf) in [
        (
            (0x5b00_u64 << 48) | (1 << 47) | (4 << 39) | (3 << 20),
            0xffff,
            2,
            0,
        ),
        (
            (0x3600_u64 << 48) | (1 << 47) | (4 << 39) | (0xffff << 20),
            0,
            2,
            0,
        ),
        ((0x4e00_u64 << 48) | (1 << 47) | (4 << 39), 0, 2, 0xffff),
        ((0x5100_u64 << 48) | (1 << 47) | (4 << 39), 0, 0xffff, 2),
    ] {
        let raw = raw | (7 << 16) | (2 << 8) | 1;
        let kind = decoded(raw);
        assert!(
            matches!(kind, InstructionKind::Xmad(_)),
            "{raw:016x}: {kind:?}"
        );
        let (registers, flags) = run(&[mov(2, 0xffff), mov(3, b), mov(4, c), kind], cbuf);
        assert_eq!(registers[1], 0xfffe_0003, "{raw:016x}");
        assert_eq!(flags, [false, true, false, false], "{raw:016x}");
    }

    // IMNMX.CC writes only Z/N; establish C=true/V=false first and prove
    // they survive both ordinary and runtime-selector roads.
    let establish_carry =
        decoded((0x5c10_u64 << 48) | (1 << 47) | (3 << 20) | (7 << 16) | (2 << 8) | 9);
    for (raw, a, b, expected) in [
        (
            (0x5c20_u64 << 48) | (1 << 47) | (7 << 39) | (4 << 20),
            0,
            1,
            0,
        ),
        (
            (0x5c21_u64 << 48) | (1 << 47) | (7 << 39) | (4 << 20),
            0x8000_0000,
            1,
            0x8000_0000,
        ),
        ((0x5c20_u64 << 48) | (1 << 47) | (4 << 20), 7, 9, 7),
    ] {
        let raw = raw | (7 << 16) | (2 << 8) | 1;
        let (registers, flags) = run(
            &[
                mov(2, u32::MAX),
                mov(3, 1),
                establish_carry,
                mov(2, a),
                mov(4, b),
                set_predicate(0, true),
                decoded(raw),
            ],
            0,
        );
        assert_eq!(registers[1], expected, "{raw:016x}");
        assert_eq!(flags, [expected == 0, expected >> 31 != 0, true, false]);
    }
}

fn multiply_word(a: u32, b: u32, high: bool, signed: [bool; 2]) -> u32 {
    let extend = |value: u32, signed: bool| {
        if signed {
            i128::from(value as i32)
        } else {
            i128::from(value)
        }
    };
    let product = (extend(a, signed[0]) * extend(b, signed[1])) as u64;
    if high {
        (product >> 32) as u32
    } else {
        product as u32
    }
}

#[test]
fn integer_multiply_add_concrete_state_edge_matrix() {
    // The host expression is the independent CPU oracle for the closed IR
    // states constructed by the decoder. It covers high/low, all mixed-sign
    // combinations, both negations, an addend alias, immediate, and cbuf.
    let cases = [
        (
            0_u32,
            u32::MAX,
            7_u32,
            false,
            [false, false],
            MulNegation::None,
        ),
        (
            u32::MAX,
            u32::MAX,
            1,
            true,
            [false, false],
            MulNegation::None,
        ),
        (u32::MAX, 2, 3, true, [true, false], MulNegation::None),
        (2, u32::MAX, 4, true, [false, true], MulNegation::None),
        (
            0x8000_0000,
            u32::MAX,
            5,
            true,
            [true, true],
            MulNegation::None,
        ),
        (
            0x8000_0000,
            3,
            9,
            false,
            [true, false],
            MulNegation::Multiplier,
        ),
        (7, u32::MAX, 11, false, [false, true], MulNegation::Addend),
    ];
    for (a, mut b, mut c, high, signed, negate) in cases {
        let kinds = [
            mov(2, a),
            mov(3, b),
            mov(4, c),
            InstructionKind::IntegerMul(ops::IntegerMul {
                dst: Register(4),
                lhs: Register(2),
                rhs: Source::Register(Register(3)),
                addend: Source::Register(Register(4)),
                high,
                signed,
                negate,
            }),
        ];
        match negate {
            MulNegation::None => {}
            MulNegation::Multiplier => b = b.wrapping_neg(),
            MulNegation::Addend => c = c.wrapping_neg(),
        }
        let expected = multiply_word(a, b, high, signed).wrapping_add(c);
        assert_eq!(run(&kinds, 0).0[4], expected, "{a:08x} {b:08x} {c:08x}");
    }

    let immediate = InstructionKind::IntegerMul(ops::IntegerMul {
        dst: Register(1),
        lhs: Register(2),
        rhs: Source::Immediate(u32::MAX),
        addend: Source::ConstantBuffer {
            binding: 0,
            offset: 0,
        },
        high: true,
        signed: [true, false],
        negate: MulNegation::None,
    });
    assert_eq!(
        run(&[mov(2, 0x8000_0000), immediate], 13).0[1],
        multiply_word(0x8000_0000, u32::MAX, true, [true, false]).wrapping_add(13)
    );
}

fn extract_imadsp(value: u32, ty: IMadSpType) -> u32 {
    match ty {
        IMadSpType::U32 | IMadSpType::S32 => value,
        IMadSpType::U24 => value & 0x00ff_ffff,
        IMadSpType::S24 => ((value << 8) as i32 >> 8) as u32,
        IMadSpType::U16Lo => value & 0xffff,
        IMadSpType::S16Lo => (value as i16 as i32) as u32,
        IMadSpType::U16Hi => value >> 16,
        IMadSpType::S16Hi => ((value as i32) >> 16) as u32,
    }
}

#[test]
fn imadsp_extraction_and_addend_state_matrix() {
    let values = [0x80ff_7f01_u32, 0xff80_8001, 0x807f_ff01];
    let types = [
        IMadSpType::U32,
        IMadSpType::S32,
        IMadSpType::U24,
        IMadSpType::S24,
        IMadSpType::U16Lo,
        IMadSpType::S16Lo,
        IMadSpType::U16Hi,
        IMadSpType::S16Hi,
    ];
    for position in 0..3 {
        for ty in types {
            let mut selected = [IMadSpType::U32; 3];
            selected[position] = ty;
            let kind = InstructionKind::IMadSp(ops::IMadSp {
                dst: Register(1),
                a: Register(2),
                b: Source::Register(Register(3)),
                c: Source::Register(Register(4)),
                types: selected,
            });
            let extracted: [u32; 3] =
                std::array::from_fn(|i| extract_imadsp(values[i], selected[i]));
            let expected = extracted[0]
                .wrapping_mul(extracted[1])
                .wrapping_add(extracted[2]);
            assert_eq!(
                run(
                    &[
                        mov(2, values[0]),
                        mov(3, values[1]),
                        mov(4, values[2]),
                        kind
                    ],
                    0
                )
                .0[1],
                expected,
                "operand {position}, type {ty:?}"
            );
        }
    }
}

#[test]
fn integer_translator_new_paths_validate_raw_and_optimized() {
    if !std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success())
    {
        eprintln!("spirv-val unavailable");
        return;
    }
    let raws = [
        (0x1c00_u64 << 48) | (1 << 53) | (1 << 52) | (1 << 20) | (7 << 16) | (2 << 8) | 1,
        (0x5c38_u64 << 48) | (1 << 39) | (1 << 40) | (3 << 20) | (7 << 16) | (2 << 8) | 1,
        (0x1f00_u64 << 48) | (1 << 53) | (1 << 54) | (3 << 20) | (7 << 16) | (2 << 8) | 1,
        (0x5a00_u64 << 48)
            | (1 << 54)
            | (2 << 51)
            | (4 << 39)
            | (3 << 20)
            | (7 << 16)
            | (2 << 8)
            | 1,
        (0x1000_u64 << 48) | (1 << 53) | (1 << 55) | (3 << 20) | (7 << 16) | (2 << 8) | 1,
        (0x5a80_u64 << 48)
            | (6 << 48)
            | (2 << 53)
            | (2 << 51)
            | (4 << 39)
            | (3 << 20)
            | (7 << 16)
            | (2 << 8)
            | 1,
        (0x5280_u64 << 48) | (3 << 39) | (7 << 16) | (2 << 8) | 1,
        (0x5b00_u64 << 48) | (1 << 47) | (4 << 39) | (3 << 20) | (7 << 16) | (2 << 8) | 1,
        (0x5c20_u64 << 48) | (1 << 47) | (7 << 39) | (4 << 20) | (7 << 16) | (2 << 8) | 1,
    ];
    let header = headers::parse(&[0; 32]).unwrap();
    for raw in raws {
        let kind = decoded(raw);
        assert!(!matches!(kind, InstructionKind::Unknown(_)), "{raw:016x}");
        let instruction = Instruction {
            address: 0,
            predicate: None,
            kind,
        };
        let mut module = emit_with_header(&[instruction], Some(&header)).unwrap();
        for variant in ["raw", "optimized"] {
            let validation = spirv_validation::validate(&module.words);
            assert!(
                validation.status.success(),
                "{raw:016x} {variant}: {}",
                String::from_utf8_lossy(&validation.stderr)
            );
            module.optimize();
        }
    }
}

// Encoding: Mesa nak/sm50.rs:725-739; semantics: Ryujinx
// InstEmitMultifunction.cs:25-29,86-94. Exercise actual cbuf loads, not
// substituted immediate IR, and preserve NaN payloads and signed zero.
#[test]
fn rro_cbuf_cpu_abs_then_negate_preserves_bits() {
    for value in [
        0,
        0x8000_0000,
        0x3f80_0000,
        0xbf80_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc1_2345,
        0xffc1_2345,
    ] {
        for abs in [false, true] {
            for negate in [false, true] {
                let raw = (0x4c90_u64 << 48)
                    | (u64::from(abs) << 49)
                    | (u64::from(negate) << 45)
                    | (7 << 16)
                    | 1;
                let expected = (if abs { value & 0x7fff_ffff } else { value })
                    ^ if negate { 0x8000_0000 } else { 0 };
                assert_eq!(run(&[decoded(raw)], value).0[1], expected);
            }
        }
    }
}

// PSET source/op fields: envytools gm107.c:2061,404-406,1430-1434,
// 1683-1687. ISETP predicate setup: Mesa sm50.rs:1487-1550.
fn set_predicate(p: u8, value: bool) -> InstructionKind {
    decoded(
        (0x5b60u64 << 48)
            | ((u64::from(value) * 7) << 49)
            | (7 << 39)
            | (255 << 20)
            | (7 << 16)
            | (255 << 8)
            | (u64::from(p) << 3)
            | 7,
    )
}

#[test]
fn isetp_extended_compare_decodes_five_operand_forms() {
    // ISETP.X: envytools gm107.c:2028,2099,2146 and 1445-1453;
    // Mesa nak/sm50.rs:1494-1541. Bit43 extends relational compares with
    // the previous Z/C flags. Keep the five encodings independently visible.
    let forms = [
        ("register U32", 0x5b64_u64, 3_u64 << 20),
        ("register S32", 0x5b65_u64, 3_u64 << 20),
        (
            "constant-buffer U32",
            0x4b64_u64,
            (7_u64 << 34) | (5_u64 << 20),
        ),
        ("positive immediate U32", 0x3664_u64, 5_u64 << 20),
        ("negative immediate S32", 0x3765_u64, 0x7ffff_u64 << 20),
    ];
    for (name, opcode, rhs) in forms {
        let raw = opcode << 48 | 1 << 43 | 7 << 39 | rhs | 7 << 16 | 2 << 8 | 7;
        assert!(
            matches!(
                decoded(raw),
                InstructionKind::ISetP(ops::ISetP {
                    mode: IntegerCompareMode::Extended,
                    compare: IntCompare::Relation(Relation::Eq),
                    ..
                })
            ),
            "{name}: {raw:016x}"
        );
    }
}

#[test]
fn isetp_extended_compare_executes_five_operand_forms() {
    let forms = [
        ("register U32", 0x5b64_u64, 3_u64 << 20, 5, 5, 0),
        (
            "register S32",
            0x5b65_u64,
            3_u64 << 20,
            u32::MAX,
            u32::MAX,
            0,
        ),
        ("constant-buffer U32", 0x4b64_u64, 0, 5, 0, 5),
        ("positive immediate U32", 0x3664_u64, 5_u64 << 20, 5, 0, 0),
        (
            "negative immediate S32",
            0x3765_u64,
            0x7ffff_u64 << 20,
            u32::MAX,
            0,
            0,
        ),
    ];
    for (name, opcode, rhs, lhs_value, rhs_value, cbuf) in forms {
        let raw = opcode << 48 | 1 << 43 | 7 << 39 | rhs | 7 << 16 | 2 << 8 | 7;
        let kinds = [
            InstructionKind::ShiftAdd(ops::ShiftAdd {
                dst: Register(4),
                lhs: Register(255),
                rhs: Source::Immediate(0),
                shift: 0,
                negate: [false; 2],
                cc: true,
            }),
            mov(2, lhs_value),
            mov(3, rhs_value),
            decoded(raw),
        ];
        let instructions = kinds
            .iter()
            .enumerate()
            .map(|(at, kind)| Instruction {
                address: at as u32 * 8,
                predicate: None,
                kind: *kind,
            })
            .collect::<Vec<_>>();
        let (_, _, predicates) = run_instructions(&instructions, cbuf);
        assert!(predicates[0], "{name}: {raw:016x}");
    }
}

#[test]
fn isetp_extended_compare_five_forms_validate_raw_and_optimized() {
    if !std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success())
    {
        eprintln!("spirv-val unavailable");
        return;
    }
    let forms = [
        ("register U32", 0x5b64_u64, 3_u64 << 20),
        ("register S32", 0x5b65_u64, 3_u64 << 20),
        ("constant-buffer U32", 0x4b64_u64, 0),
        ("positive immediate U32", 0x3664_u64, 5_u64 << 20),
        ("negative immediate S32", 0x3765_u64, 0x7ffff_u64 << 20),
    ];
    let header = headers::parse(&[0; 32]).unwrap();
    for (name, opcode, rhs) in forms {
        let raw = opcode << 48 | 1 << 43 | 7 << 39 | rhs | 7 << 16 | 2 << 8 | 7;
        let instruction = Instruction {
            address: 0,
            predicate: None,
            kind: decoded(raw),
        };
        let mut module = emit_with_header(&[instruction], Some(&header)).unwrap();
        for variant in ["raw", "optimized"] {
            let validation = spirv_validation::validate(&module.words);
            assert!(
                validation.status.success(),
                "{name} {variant}: {}",
                String::from_utf8_lossy(&validation.stderr)
            );
            module.optimize();
        }
    }
}

fn pset_word(dst: u8, ops: [u8; 2], negate: u8, bool_float: bool) -> u64 {
    (0x5088u64 << 48)
        | (u64::from(ops[0]) << 24)
        | (u64::from(ops[1]) << 45)
        | (u64::from(bool_float) << 44)
        | (1 << 29)
        | (2 << 39)
        | (u64::from(negate & 1) << 15)
        | (u64::from((negate >> 1) & 1) << 32)
        | (u64::from((negate >> 2) & 1) << 42)
        | (7 << 16)
        | u64::from(dst)
}

#[test]
fn pset_all_boolean_ops_negates_and_output_bits() {
    // Ryujinx InstEmitPredicate.cs:16-31. Exhaust every Boolean input,
    // negate combination, both ops and both output representations.
    let logic = |op, a, b| match op {
        0 => a && b,
        1 => a || b,
        2 => a != b,
        _ => unreachable!(),
    };
    for inputs in 0u8..8 {
        for negate in 0u8..8 {
            for op0 in 0..3 {
                for op1 in 0..3 {
                    for bf in [false, true] {
                        let bits = inputs ^ negate;
                        let result =
                            logic(op1, logic(op0, bits & 1 != 0, bits & 2 != 0), bits & 4 != 0);
                        let expected = if result {
                            if bf {
                                1.0_f32.to_bits()
                            } else {
                                u32::MAX
                            }
                        } else {
                            0
                        };
                        let kinds = [
                            set_predicate(0, inputs & 1 != 0),
                            set_predicate(1, inputs & 2 != 0),
                            set_predicate(2, inputs & 4 != 0),
                            decoded(pset_word(5, [op0, op1], negate, bf)),
                        ];
                        assert_eq!(run(&kinds, 0).0[5], expected);
                    }
                }
            }
        }
    }
    for raw in [
        pset_word(5, [3, 0], 0, false),
        pset_word(5, [0, 3], 0, false),
        pset_word(5, [0, 0], 0, false) | 1 << 47,
    ] {
        assert!(matches!(
            decoded(raw),
            InstructionKind::Unknown(ops::Unknown { .. })
        ));
    }
}

#[test]
fn pset_opcode_span_forms_execute_and_validate() {
    // These are the six PSET
    // opcode forms 0x5089..0x508e. Envytools gm107.c:2061 assigns the
    // complete 0x5088/0xfff8 row to PSET, so 0x508f belongs to the same
    // closed family while the adjacent opcodes do not.
    let forms = [
        (0x5089_0b54_58f7_25c7, Register(199), [2, 2, 6]),
        (0x508a_0986_a887_3188, Register(136), [3, 5, 3]),
        (0x508b_0304_4c67_7bf2, Register(242), [7, 2, 6]),
        (0x508c_0952_2047_1f77, Register(119), [1, 1, 2]),
        (0x508d_0174_1437_72b6, Register(182), [7, 0, 2]),
        (0x508e_0278_50b7_4a91, Register(145), [4, 2, 4]),
        // The pinned 0xfff8 source mask also includes 0x508f. Reuse the
        // 0x508e word's fields so the otherwise-unseeded span endpoint is
        // exercised through CPU execution and SPIR-V validation.
        (0x508f_0278_50b7_4a91, Register(145), [4, 2, 4]),
    ];

    for &(raw, dst, src) in &forms {
        assert_eq!(
            decoded(raw),
            InstructionKind::PSet(ops::PSet {
                dst,
                src,
                set_op: [SetOp::And; 2],
                negate: [false; 3],
                bool_float: false,
            }),
            "{raw:016x}"
        );

        for pattern in 0_u8..128 {
            let mut kinds = (0..7)
                .map(|p| set_predicate(p, pattern & (1 << p) != 0))
                .collect::<Vec<_>>();
            kinds.push(decoded(raw));
            let value = |p: u8| p == 7 || pattern & (1 << p) != 0;
            let expected = if src.into_iter().all(value) {
                u32::MAX
            } else {
                0
            };
            assert_eq!(run(&kinds, 0).0[dst.0 as usize], expected, "{raw:016x}");
        }
    }

    for opcode in 0x5088_u64..=0x508f {
        let raw = opcode << 48 | 7 << 39 | 7 << 29 | 7 << 16 | 7 << 12 | 1;
        assert!(
            matches!(decoded(raw), InstructionKind::PSet(_)),
            "{raw:016x}"
        );
        assert!(matches!(
            decoded(raw | 1 << 47),
            InstructionKind::Unknown(_)
        ));
        assert!(matches!(
            decoded(raw | 3 << 24),
            InstructionKind::Unknown(_)
        ));
    }
    for opcode in [0x5087_u64, 0x5090] {
        assert!(!matches!(decoded(opcode << 48), InstructionKind::PSet(_)));
    }

    let header = headers::parse(&[0; 32]).unwrap();
    for &(raw, _, _) in &forms {
        let mut module = emit_with_header(
            &[Instruction {
                address: 0,
                predicate: None,
                kind: decoded(raw),
            }],
            Some(&header),
        )
        .unwrap();
        for variant in ["raw", "optimized"] {
            let validation = spirv_validation::validate(&module.words);
            assert!(
                validation.status.success(),
                "{raw:016x} {variant}: {}",
                String::from_utf8_lossy(&validation.stderr)
            );
            module.optimize();
        }
    }
}

#[test]
fn register_lop3_every_lut_and_predicate_mode() {
    // envytools gm107.c:2020,1391-1396. Independent per-bit LUT oracle,
    // Ryujinx Lop3Expression.cs:13-31; InstEmitIntegerLogical.cs:118-162.
    let (a, b, c) = (0xf0f0_1234u32, 0xcccc_5678u32, 0xaaaa_9abcu32);
    for lut in 0u64..256 {
        let expected = (0..32).fold(0u32, |out, bit| {
            let index = (((a >> bit) & 1) << 2) | (((b >> bit) & 1) << 1) | ((c >> bit) & 1);
            out | (((lut as u32 >> index) & 1) << bit)
        });
        for mode in 0u64..4 {
            let raw = (0x5be1u64 << 48)
                | (1 << 47)
                | (4 << 39)
                | (mode << 36)
                | (lut << 28)
                | (3 << 20)
                | (7 << 16)
                | (2 << 8)
                | 1;
            let probe = (0x5088u64 << 48) | (7 << 39) | (7 << 29) | (1 << 12) | (7 << 16) | 5;
            let (regs, cc) = run(
                &[
                    mov(2, a),
                    mov(3, b),
                    mov(4, c),
                    decoded(raw),
                    decoded(probe),
                ],
                0,
            );
            let predicate = match mode {
                0 => false,
                1 => true,
                2 => expected == 0,
                3 => expected != 0,
                _ => unreachable!(),
            };
            assert_eq!(regs[1], expected, "LUT {lut:02x}");
            assert_eq!(regs[5], if predicate { u32::MAX } else { 0 });
            assert_eq!(cc[..2], [expected == 0, expected & 0x8000_0000 != 0]);
            assert!(matches!(
                decoded(raw | 1 << 38),
                InstructionKind::Lop3(ops::Lop3 {
                    extended: true,
                    cc: true,
                    ..
                })
            ));
            // Envytools gm107.c:2020; X requires an observable CC output.
            assert!(matches!(
                decoded((raw | 1 << 38) & !(1 << 47)),
                InstructionKind::Unknown(ops::Unknown { .. })
            ));
        }
    }
}

#[test]
fn pset_and_lop3_preserve_whole_instruction_guard() {
    // Guard16..19: envytools gm107.c:408-411. Ryujinx Translator.cs:339-385.
    for active in [false, true] {
        let lop3 =
            (0x5be1u64 << 48) | (1 << 47) | (1 << 36) | (0xff << 28) | (3 << 20) | (2 << 8) | 1;
        let pset = pset_word(5, [1, 1], 0, false) & !(7 << 16);
        let kinds = [
            set_predicate(0, active),
            set_predicate(1, false),
            set_predicate(2, true),
            mov(1, 0x1234),
            mov(5, 0x5678),
        ];
        let mut xs = kinds
            .into_iter()
            .enumerate()
            .map(|(i, kind)| Instruction {
                address: i as u32 * 8,
                predicate: None,
                kind,
            })
            .collect::<Vec<_>>();
        xs.push(decode_word([lop3 as u32, (lop3 >> 32) as u32], 40));
        xs.push(decode_word([pset as u32, (pset >> 32) as u32], 48));
        let probe = (0x5088u64 << 48) | (7 << 39) | (7 << 29) | (1 << 12) | (7 << 16) | 6;
        xs.push(decode_word([probe as u32, (probe >> 32) as u32], 56));
        let (regs, cc) = execute(&xs, 0);
        assert_eq!(regs[6], if active { u32::MAX } else { 0 });
        assert_eq!(cc[..2], [false, active]);
        assert_eq!(regs[1], if active { u32::MAX } else { 0x1234 });
        assert_eq!(regs[5], if active { u32::MAX } else { 0x5678 });
    }
}

#[test]
fn cbuf_rro_modifiers_preserve_exact_bits() {
    // Mesa sm50.rs:725-739,361-368; Ryujinx InstEmitMultifunction.cs:86-94.
    for input in [0, 0x8000_0000, 0x7fc1_2345, 0xffc1_2345, 1.25_f32.to_bits()] {
        for abs in [false, true] {
            for neg in [false, true] {
                let raw = (0x4c90u64 << 48)
                    | (u64::from(abs) << 49)
                    | (u64::from(neg) << 45)
                    | (7 << 16)
                    | 5;
                let expected = (if abs { input & 0x7fff_ffff } else { input })
                    ^ if neg { 0x8000_0000 } else { 0 };
                assert_eq!(run(&[decoded(raw)], input).0[5], expected);
                // EX2 mode39 is the same reference lowering
                // (Ryujinx InstEmitMultifunction.cs:25-29,86-94).
                assert!(matches!(
                    decoded(raw | 1 << 39),
                    InstructionKind::Rro(ops::Rro { ex2: true, .. })
                ));
                assert_eq!(run(&[decoded(raw | 1 << 39)], input).0[5], expected);
            }
        }
    }
    let raw = (0x4c90u64 << 48) | (3 << 34) | (7 << 20) | (7 << 16) | 5;
    assert_eq!(
        decoded(raw).sources(),
        vec![Source::ConstantBuffer {
            binding: 3,
            offset: 28
        }]
    );
}

#[test]
fn blocker_fcmp_forms_select_payload_bits_with_and_without_ftz() {
    // envytools gm107.c:2143,2025,1419-1425; Mesa older emitter
    // nv50_ir_emit_gm107.cpp:1537-1571. Ryujinx InstEmitFloatComparison.cs:369-375.
    for prefix in [0x36a9u64, 0x36ab, 0x5bae] {
        for test in [-1.0_f32, -0.0, 1.0, f32::NAN] {
            let immediate = prefix != 0x5bae;
            let b = if immediate { 0x8000_3000 } else { 0xffc1_2345 };
            let raw = (prefix << 48)
                | (if immediate {
                    1 << 56 | 3 << 20
                } else {
                    3 << 20
                })
                | (4 << 39)
                | (7 << 16)
                | (2 << 8)
                | 1;
            let condition = test.is_nan()
                || match prefix {
                    0x36a9 => test < 0.0,
                    0x36ab => test <= 0.0,
                    _ => test >= 0.0,
                };
            let a = 0x7fc1_5678;
            assert_eq!(
                run(
                    &[mov(2, a), mov(3, b), mov(4, test.to_bits()), decoded(raw)],
                    0
                )
                .0[1],
                if condition { a } else { b }
            );
            assert_eq!(
                run(
                    &[
                        mov(2, a),
                        mov(3, b),
                        mov(4, test.to_bits()),
                        decoded(raw | 1 << 47)
                    ],
                    0
                )
                .0[1],
                if condition { a } else { b }
            );
        }
    }
}

#[test]
fn surface_reduction_has_real_ir_and_unproven_ipa_stays_unknown() {
    // Fields: envytools gm107.c:1908,1953-1954; Mesa sm50.rs:2905-2934.
    // SUATOM/SURED semantics now have value tests in surface_fetch.rs.
    // envytools gm107.c:1908,769-791; Mesa sm50.rs:2558-2591.
    for payload in [0u64, 15 << 31 | 7 << 39] {
        assert_eq!(
            decoded(0xeb50u64 << 48 | 7 << 16 | payload),
            InstructionKind::SurfaceAtomic(ops::SurfaceAtomic {
                op: 0,
                signed: false,
                target: ((payload >> 33) & 7) as u8,
                handle: Source::Register(Register(((payload >> 39) & 255) as u8)),
                coord: Register(0),
                data: Register(0),
                dst: None,
                byte_address: false,
            })
        );
    }
    assert!(matches!(
        decoded(0xeb50u64 << 48 | 7 << 16 | 255 << 8 | 255 << 20),
        InstructionKind::Unknown(ops::Unknown { .. })
    ));
    for prefix in [0xe04bu64] {
        for payload in [0u64, 255 << 8 | 255 << 20, 15 << 31 | 7 << 39] {
            assert!(matches!(
                decoded(prefix << 48 | 7 << 16 | payload),
                InstructionKind::Unknown(ops::Unknown { .. })
            ));
        }
    }
}

#[test]
fn heldout_aliasing_and_inverted_pt_keep_exact_values() {
    // Fresh final milestone: alias A with the destination, then independently
    // apply the hardware always-false guard. Field citations are in decoder.rs.
    let raw = (0x5be1u64 << 48) | (4 << 39) | (0xf8 << 28) | (3 << 20) | (7 << 16) | (2 << 8) | 2;
    let inputs = [
        mov(2, 0x1020_3040),
        mov(3, 0x0f0f_0f0f),
        mov(4, 0x8765_4321),
    ];
    assert_eq!(
        run(&[inputs[0], inputs[1], inputs[2], decoded(raw)], 0).0[2],
        0x1725_3341
    );
    let mut xs = inputs
        .into_iter()
        .enumerate()
        .map(|(i, kind)| Instruction {
            address: i as u32 * 8,
            predicate: None,
            kind,
        })
        .collect::<Vec<_>>();
    let guarded = raw | 1 << 19;
    xs.push(decode_word([guarded as u32, (guarded >> 32) as u32], 24));
    assert_eq!(execute(&xs, 0).0[2], 0x1020_3040);
    let rro = (0x4c90u64 << 48) | (1 << 49) | (1 << 45) | (1 << 19) | (7 << 16) | 2;
    xs[3] = decode_word([rro as u32, (rro >> 32) as u32], 24);
    assert_eq!(execute(&xs, 0xffc1_2345).0[2], 0x1020_3040);
}

#[test]
fn lop3_predicate_destination_can_alias_its_guard_without_changing_cc() {
    // Ryujinx Translator.cs:339-385 guards the entire instruction. Updating
    // the guard predicate must not change whether that instruction writes CC.
    for inverted in [false, true] {
        let raw = (0x5be1u64 << 48)
            | (1 << 47)
            | (u64::from(inverted) << 36)
            | (0xff << 28)
            | (3 << 20)
            | (u64::from(inverted) << 19)
            | (1 << 16)
            | (2 << 8)
            | 1;
        let mut xs = [set_predicate(1, !inverted), mov(1, 0x1234)]
            .into_iter()
            .enumerate()
            .map(|(i, kind)| Instruction {
                address: i as u32 * 8,
                predicate: None,
                kind,
            })
            .collect::<Vec<_>>();
        xs.push(decode_word([raw as u32, (raw >> 32) as u32], 16));
        let (regs, cc) = execute(&xs, 0);
        assert_eq!(regs[1], u32::MAX);
        assert_eq!(cc[..2], [false, true]);
    }
}

#[test]
fn register_lop3_truth_tables_predicates_and_flags() {
    // gm107.c:2020,1391-1396; Ryujinx InstEmitIntegerLogical.cs:118-163.
    // Bit j of each byte encodes (A,B,C)=j, so each output byte is the LUT.
    for lut in 0..=255_u64 {
        for mode in 0..4_u64 {
            let raw = 0x5be1_u64 << 48
                | 7 << 16
                | 1
                | 2 << 8
                | 3 << 20
                | 4 << 39
                | lut << 28
                | mode << 36
                | 1 << 47;
            let kinds = [
                mov(2, 0xf0f0_f0f0),
                mov(3, 0xcccc_cccc),
                mov(4, 0xaaaa_aaaa),
                decoded(raw),
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
            let (regs, cc, predicates) = run_instructions(&xs, 0);
            let expected = lut as u32 * 0x0101_0101;
            assert_eq!(regs[1], expected, "LUT={lut:02x}");
            assert_eq!(
                predicates[1],
                match mode {
                    0 => false,
                    1 => true,
                    2 => expected == 0,
                    _ => expected != 0,
                }
            );
            assert_eq!(cc, [expected == 0, expected >> 31 != 0, false, false]);
        }
    }
}

#[test]
fn register_lop3_extended_zero_and_aliased_guard() {
    // Ryujinx InstEmitAluHelper.cs:95-121: X propagates Z, preserves C/V.
    // Translator.cs:339-385: the instruction guard uses the old predicate.
    for old_zero in [false, true] {
        for active in [false, true] {
            for inverted in [false, true] {
                for lut in [0_u64, 255] {
                    let raw = 0x5be1_u64 << 48
                        | 1 << 16
                        | u64::from(inverted) << 19
                        | 1
                        | 2 << 8
                        | 3 << 20
                        | 4 << 39
                        | lut << 28
                        | 2 << 36
                        | 1 << 38
                        | 1 << 47;
                    let mut xs = [
                        mov(1, 0x1234),
                        mov(2, u32::from(!old_zero)),
                        decoded(0x5c10_u64 << 48 | 7 << 16 | 5 | 2 << 8 | 255 << 20 | 1 << 47),
                        mov(6, u32::from(active != inverted) << 1),
                        InstructionKind::R2P(ops::R2P {
                            src: Register(6),
                            mask: Source::Immediate(2),
                            byte: 0,
                            bank: shadowbox::ir::FlagBank::Predicates,
                        }),
                    ]
                    .into_iter()
                    .enumerate()
                    .map(|(i, kind)| Instruction {
                        address: i as u32 * 8,
                        predicate: None,
                        kind,
                    })
                    .collect::<Vec<_>>();
                    xs.push(decode_word([raw as u32, (raw >> 32) as u32], 40));
                    let (regs, cc, predicates) = run_instructions(&xs, 0);
                    let result = if lut == 0 { 0 } else { u32::MAX };
                    assert_eq!(regs[1], if active { result } else { 0x1234 });
                    assert_eq!(
                        cc,
                        if active {
                            [old_zero && result == 0, result >> 31 != 0, false, false]
                        } else {
                            [old_zero, false, false, false]
                        }
                    );
                    assert_eq!(
                        predicates[1],
                        if active {
                            result == 0
                        } else {
                            active != inverted
                        }
                    );
                }
            }
        }
    }
}

#[test]
fn r2p_concrete_banks_apply_byte_masks_and_preserve_unselected_state() {
    use shadowbox::ir::FlagBank;

    for bank in [FlagBank::Predicates, FlagBank::ConditionCodes] {
        let count = match bank {
            FlagBank::Predicates => 7,
            FlagBank::ConditionCodes => 4,
        };
        let full_mask = (1_u32 << count) - 1;
        for byte in 0..4_u8 {
            for mask in [0_u32, 1, 0b0101, full_mask] {
                for bits in [0_u32, 1, 0b1010, full_mask] {
                    let source = bits << (u32::from(byte) * 8);
                    let kinds = [
                        mov(2, full_mask),
                        InstructionKind::R2P(ops::R2P {
                            src: Register(2),
                            mask: Source::Immediate(full_mask),
                            byte: 0,
                            bank,
                        }),
                        mov(3, source),
                        InstructionKind::R2P(ops::R2P {
                            src: Register(3),
                            mask: Source::Immediate(mask),
                            byte,
                            bank,
                        }),
                    ];
                    let (_, cc, predicates) = run_instructions(
                        &kinds
                            .into_iter()
                            .enumerate()
                            .map(|(i, kind)| Instruction {
                                address: i as u32 * 8,
                                predicate: None,
                                kind,
                            })
                            .collect::<Vec<_>>(),
                        0,
                    );
                    let expected = |at: usize| mask & (1 << at) == 0 || bits & (1 << at) != 0;
                    match bank {
                        FlagBank::Predicates => {
                            for (at, &actual) in predicates[..count].iter().enumerate() {
                                assert_eq!(
                                    actual,
                                    expected(at),
                                    "byte={byte} mask={mask:02x} bits={bits:02x} P{at}"
                                );
                            }
                        }
                        FlagBank::ConditionCodes => {
                            for (at, &actual) in cc[..count].iter().enumerate() {
                                assert_eq!(
                                    actual,
                                    expected(at),
                                    "byte={byte} mask={mask:02x} bits={bits:02x} CC{at}"
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
fn vote_and_r2p_closed_states_decode_and_validate_raw_and_optimized() {
    use shadowbox::ir::{FlagBank, VoteOp};

    let vote = |mode: u64| (0x50d8_u64 | mode) << 48 | 2 << 45 | 3 << 39 | 7 << 16 | 1;
    for (mode, op) in [(0, VoteOp::All), (1, VoteOp::Any), (2, VoteOp::Eq)] {
        assert!(matches!(
            decoded(vote(mode)),
            InstructionKind::Vote(ops::Vote { op: actual, .. }) if actual == op
        ));
    }
    assert!(matches!(decoded(vote(3)), InstructionKind::Unknown(_)));

    let r2p = [
        0x5cf0_u64 << 48 | 1 << 40 | 3 << 20 | 7 << 16 | 2 << 8,
        0x4cf0_u64 << 48 | 1 << 40 | 1 << 20 | 7 << 16 | 2 << 8,
        0x38f0_u64 << 48 | 1 << 40 | 0xf << 20 | 7 << 16 | 2 << 8,
    ];
    for raw in r2p {
        assert!(matches!(
            decoded(raw),
            InstructionKind::R2P(ops::R2P {
                bank: FlagBank::ConditionCodes,
                ..
            })
        ));
        assert!(matches!(
            decoded(raw ^ (1 << 40)),
            InstructionKind::R2P(ops::R2P {
                bank: FlagBank::Predicates,
                ..
            })
        ));
    }

    if !std::process::Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success())
    {
        eprintln!("spirv-val unavailable");
        return;
    }
    let header = headers::parse(&[0; 32]).unwrap();
    for raw in [vote(0), vote(1), vote(2)]
        .into_iter()
        .chain(r2p)
        .chain(r2p.map(|raw| raw ^ (1 << 40)))
    {
        let instruction = Instruction {
            address: 0,
            predicate: None,
            kind: decoded(raw),
        };
        let mut module = emit_with_header(&[instruction], Some(&header)).unwrap();
        for variant in ["raw", "optimized"] {
            let validation = spirv_validation::validate(&module.words);
            assert!(
                validation.status.success(),
                "{raw:016x} {variant}: {}",
                String::from_utf8_lossy(&validation.stderr)
            );
            module.optimize();
        }
    }
}

#[test]
fn register_lop3_cc_preserves_carry_and_overflow() {
    // Ryujinx InstEmitAluHelper.cs:95-121 only changes Z/N for LOP3.CC.
    for (a, b) in [(u32::MAX, 1), (0x7fff_ffff, 1), (0x8000_0000, 0x8000_0000)] {
        let add = decoded(0x5c10_u64 << 48 | 7 << 16 | 5 | 2 << 8 | 3 << 20 | 1 << 47);
        let prefix = [mov(2, a), mov(3, b), add];
        let (_, before) = run(&prefix, 0);
        for lut in [0_u64, 255] {
            for extended in [false, true] {
                let raw = 0x5be1_u64 << 48
                    | 7 << 16
                    | 1
                    | 2 << 8
                    | 3 << 20
                    | 4 << 39
                    | lut << 28
                    | u64::from(extended) << 38
                    | 1 << 47;
                let mut kinds = prefix.to_vec();
                kinds.push(decoded(raw));
                let (regs, after) = run(&kinds, 0);
                assert_eq!(regs[1], if lut == 0 { 0 } else { u32::MAX });
                assert_eq!(
                    after,
                    [
                        lut == 0 && (!extended || before[0]),
                        lut != 0,
                        before[2],
                        before[3]
                    ]
                );
            }
        }
    }
}

#[test]
fn cbuf_lop3_all_truth_tables_execute_the_encoded_source() {
    // Encoding: envytools gm107.c:2169; Ryujinx InstDecoders.cs:3771-3780.
    // Per-bit oracle: Ryujinx Instructions/Lop3Expression.cs:13-31.
    let (a, b, c) = (0xf0f0_1234u32, 0xcccc_5678u32, 0xaaaa_9abcu32);
    for lut in 0u64..=255 {
        for cc in [false, true] {
            let raw = (0x0200u64 << 48)
                | (lut << 48)
                | (u64::from(cc) << 47)
                | (4 << 39)
                | (7 << 16)
                | (2 << 8)
                | 1;
            let expected = (0..32).fold(0u32, |out, bit| {
                let index = (((a >> bit) & 1) << 2) | (((b >> bit) & 1) << 1) | ((c >> bit) & 1);
                out | (((lut as u32 >> index) & 1) << bit)
            });
            let (regs, flags) = run(&[mov(2, a), mov(4, c), decoded(raw)], b);
            assert_eq!(regs[1], expected, "LUT={lut:02x} CC={cc}");
            assert_eq!(flags[..2], [cc && expected == 0, cc && expected >> 31 != 0]);
            let extended = raw | (1 << 56);
            assert!(matches!(
                decoded(extended),
                InstructionKind::Lop3(ops::Lop3 { extended: true, .. })
            ));
            let (regs, flags) = run(&[mov(2, a), mov(4, c), decoded(extended)], b);
            assert_eq!(regs[1], expected, "LUT={lut:02x} CC={cc} X");
            assert_eq!(flags[..2], [false, cc && expected >> 31 != 0]);
        }
    }
    let raw = (0x02f8u64 << 48) | (31 << 34) | (0x3fff << 20) | (254 << 39) | (7 << 16);
    assert!(matches!(
        decoded(raw),
        InstructionKind::Lop3(ops::Lop3 {
            rhs: Source::ConstantBuffer {
                binding: 31,
                offset: 0xfffc
            },
            third: Register(254),
            lut: 0xf8,
            ..
        })
    ));
}

#[test]
fn cbuf_lop3_x_discovery_words_validate_raw_and_optimized_spirv() {
    let header = headers::parse(&[0; 32]).unwrap();
    for raw in [
        0x0300_0600_0001_0804_u64,
        0x0300_8600_0001_0804,
        0x0300_7f80_0007_ffff,
        0x0300_ff80_0007_ffff,
    ] {
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 0);
        assert!(matches!(
            instruction.kind,
            InstructionKind::Lop3(ops::Lop3 { extended: true, .. })
        ));
        let mut module = emit_with_header(&[instruction], Some(&header)).unwrap();
        for variant in ["raw", "optimized"] {
            let validation = spirv_validation::validate(&module.words);
            assert!(
                validation.status.success(),
                "{raw:016x} {variant}: {}",
                String::from_utf8_lossy(&validation.stderr)
            );
            module.optimize();
        }
    }
}

#[test]
fn lea_register_all_shifts_negation_and_guards_are_exact() {
    // envytools gm107.c:2022; Ryujinx InstDecoders.cs:3585-3594 and
    // InstEmitIntegerArithmetic.cs:481-490. Shift then negate then add.
    for a in [0u32, 1, 0x8000_0001, u32::MAX] {
        for b in [0u32, 1, 0x1234_5678, u32::MAX] {
            for shift in 0..32u64 {
                for negate in [false, true] {
                    let raw = (0x5bd7u64 << 48)
                        | (u64::from(negate) << 45)
                        | (shift << 39)
                        | (3 << 20)
                        | (7 << 16)
                        | (2 << 8)
                        | 1;
                    let shifted = a.wrapping_shl(shift as u32);
                    let expected = b.wrapping_add(if negate {
                        shifted.wrapping_neg()
                    } else {
                        shifted
                    });
                    let (regs, _) = run(&[mov(2, a), mov(3, b), decoded(raw)], 0);
                    assert_eq!(regs[1], expected);
                }
            }
        }
    }
    let raw = (0x5bd7u64 << 48) | (3 << 20) | (7 << 16) | (2 << 8) | 1;
    for bit in (28..39).chain([44, 46, 47]) {
        assert!(
            matches!(
                decoded(raw | (1 << bit)),
                InstructionKind::Unknown(ops::Unknown { .. })
            ),
            "bit {bit}"
        );
    }
    for pred in 0..7u64 {
        assert!(matches!(
            decoded((raw & !(7 << 48)) | pred << 48),
            InstructionKind::Unknown(ops::Unknown { .. })
        ));
    }
    let mut xs = vec![
        Instruction {
            address: 0,
            predicate: None,
            kind: mov(1, 99),
        },
        Instruction {
            address: 8,
            predicate: Some(shadowbox::ir::Predicate {
                register: 0,
                inverted: false,
            }),
            kind: decoded(raw),
        },
    ];
    assert_eq!(execute(&xs, 0).0[1], 99);
    xs[1].predicate.as_mut().unwrap().inverted = true;
    assert_eq!(execute(&xs, 0).0[1], 0);
}

#[test]
fn fset_cbuf_accumulator_modifiers_and_destination_bits_are_exact() {
    use shadowbox::ir::SetOp;
    // Ryujinx InstEmitFloatComparison.cs:378-418, gm107.c:2106 and
    // old Mesa nv50_ir_emit_gm107.cpp:1597-1620. No FTZ assumption.
    for selector in 0..16u64 {
        for bop in 0..3u64 {
            for pt in [false, true] {
                for p in [false, true] {
                    for inverted in [false, true] {
                        for bf in [false, true] {
                            for (a, b) in [
                                (1.0f32, 2.0f32),
                                (2.0, 1.0),
                                (-1.0, -1.0),
                                (f32::NAN, 0.0),
                                (0.0, f32::NAN),
                            ] {
                                let raw = (0x4800u64 << 48)
                                    | (selector << 48)
                                    | (u64::from(bf) << 52)
                                    | (1 << 47)
                                    | (bop << 45)
                                    | (u64::from(inverted) << 42)
                                    | (if pt { 7 } else { 0 } << 39)
                                    | (7 << 16)
                                    | (2 << 8)
                                    | 65;
                                let nan = a.is_nan() || b.is_nan();
                                let comparison = match selector {
                                    0 => false,
                                    7 => !nan,
                                    8 => nan,
                                    15 => true,
                                    x => {
                                        let unordered = x >= 9;
                                        let relation = match if unordered { x - 8 } else { x } {
                                            1 => a < b,
                                            2 => a == b,
                                            3 => a <= b,
                                            4 => a > b,
                                            5 => a != b,
                                            6 => a >= b,
                                            _ => unreachable!(),
                                        };
                                        if unordered {
                                            nan || relation
                                        } else {
                                            !nan && relation
                                        }
                                    }
                                };
                                let pred = (pt || p) != inverted;
                                let truth = match bop {
                                    0 => comparison && pred,
                                    1 => comparison || pred,
                                    2 => comparison != pred,
                                    _ => unreachable!(),
                                };
                                let expected = if truth {
                                    if bf {
                                        1.0f32.to_bits()
                                    } else {
                                        u32::MAX
                                    }
                                } else {
                                    0
                                };
                                let (regs, flags) = run(
                                    &[
                                        mov(2, a.to_bits()),
                                        InstructionKind::PSetP(ops::PSetP {
                                            dst: [0, 7],
                                            src: [7; 3],
                                            set_op: [SetOp::And; 2],
                                            negate: [!p, false, false],
                                        }),
                                        decoded(raw),
                                    ],
                                    b.to_bits(),
                                );
                                assert_eq!(regs[65], expected, "selector={selector} bop={bop} pt={pt} p={p} inv={inverted} bf={bf}");
                                assert_eq!(flags[..2], [expected == 0, expected >> 31 != 0]);
                            }
                        }
                    }
                }
            }
        }
    }
    let base = (0x4811u64 << 48) | (7 << 39) | (7 << 16) | (2 << 8) | 65;
    for abs_a in [false, true] {
        for abs_b in [false, true] {
            for neg_a in [false, true] {
                for neg_b in [false, true] {
                    let raw = base
                        | (u64::from(abs_a) << 54)
                        | (u64::from(abs_b) << 44)
                        | (u64::from(neg_a) << 43)
                        | (u64::from(neg_b) << 53);
                    let modified = |x: f32, abs: bool, neg: bool| {
                        let x = if abs { x.abs() } else { x };
                        if neg {
                            -x
                        } else {
                            x
                        }
                    };
                    let expected = modified(-1.0, abs_a, neg_a) < modified(-2.0, abs_b, neg_b);
                    assert_eq!(
                        run(
                            &[mov(2, (-1.0f32).to_bits()), decoded(raw)],
                            (-2.0f32).to_bits()
                        )
                        .0[65],
                        if expected { 1.0f32.to_bits() } else { 0 }
                    );
                }
            }
        }
    }
    assert!(matches!(
        decoded(base | 3 << 45),
        InstructionKind::Unknown(ops::Unknown { .. })
    ));
    // Bit56 selects DSET cbuf, not an FSET operand: gm107.c:2105 and
    // Ryujinx InstTable.cs:80,119. nvdisasm SM52 prints
    // DSET.BF.LT.AND R65, R2, c[0x0][0x0], PT.
    assert!(matches!(
        decoded(base | 1 << 56),
        InstructionKind::DoubleSet(ops::DoubleSet { .. })
    ));
}

#[test]
fn fset_ftz_inputs_and_output_flags_are_exact_in_all_three_forms() {
    // Encodings: Mesa sm50.rs:839-862; envytools gm107.c:2036,2106,2154.
    // Modifiers/compare/output: Ryujinx InstEmitFloatComparison.cs:397-419.
    // FTZ consumers: Mesa ir.rs:8949 and opt_copy_prop.rs:323-325.
    let inputs = [
        (1u32, 0u32),
        (0x8000_0001, 0),
        (0, 0x0000_1000),
        (0, 0x8000_1000),
        (0x007f_ffff, 0x007f_f000),
        (0x0080_0000, 0x007f_f000),
        (0x8080_0000, 0x8000_1000),
        (0x8000_0000, 0),
        (0x3f80_0000, 0x4000_0000),
        (0x7f80_0000, 0xff80_0000),
        (0x7fc1_2345, 0),
        (0, 0xffc1_2000),
    ];
    for prefix in [0x5800u64, 0x4800, 0x3000] {
        for ftz in [false, true] {
            for selector in 0..16u64 {
                for bf in [false, true] {
                    for (a, b) in inputs {
                        let operand = match prefix {
                            0x5800 => 3 << 20,
                            0x4800 => 0,
                            _ => {
                                (u64::from(b & 0x7fff_ffff) >> 12) << 20
                                    | (u64::from(b >> 31) << 56)
                            }
                        };
                        let raw = (prefix << 48)
                            | (selector << 48)
                            | (u64::from(ftz) << 55)
                            | (u64::from(bf) << 52)
                            | (1 << 47)
                            | (7 << 39)
                            | operand
                            | (7 << 16)
                            | (2 << 8)
                            | 65;
                        let float = |bits: u32| {
                            // Independent oracle: zero exponent means zero
                            // or subnormal; preserve the sign when flushing.
                            let bits = if ftz && bits & 0x7f80_0000 == 0 {
                                bits & 0x8000_0000
                            } else {
                                bits
                            };
                            f32::from_bits(bits)
                        };
                        let (lhs, rhs) = (float(a), float(b));
                        let nan = lhs.is_nan() || rhs.is_nan();
                        let truth = match selector {
                            0 => false,
                            7 => !nan,
                            8 => nan,
                            15 => true,
                            x => {
                                let unordered = x >= 9;
                                let relation = match if unordered { x - 8 } else { x } {
                                    1 => lhs < rhs,
                                    2 => lhs == rhs,
                                    3 => lhs <= rhs,
                                    4 => lhs > rhs,
                                    5 => lhs != rhs,
                                    6 => lhs >= rhs,
                                    _ => unreachable!(),
                                };
                                if unordered {
                                    nan || relation
                                } else {
                                    !nan && relation
                                }
                            }
                        };
                        let expected = if truth {
                            if bf {
                                1.0f32.to_bits()
                            } else {
                                u32::MAX
                            }
                        } else {
                            0
                        };
                        let kind = decoded(raw);
                        assert!(
                            matches!(kind, InstructionKind::FSet(ops::FSet { ftz: decoded_ftz, .. }) if decoded_ftz == ftz)
                        );
                        let (regs, flags) = run(&[mov(2, a), mov(3, b), kind], b);
                        assert_eq!(regs[65], expected, "{raw:016x} A={a:08x} B={b:08x}");
                        assert_eq!(flags[..2], [expected == 0, expected >> 31 != 0]);
                    }
                }
            }
        }
    }
}

#[test]
fn field_probe_signed_sel_preserves_selected_payload() {
    // sm50.rs:2010-2029; gm107.c:2118,253,373,406;
    // Ryujinx InstDecoders.cs:4292-4302, InstEmitMove.cs:230-238,271-276.
    for immediate in [-524288i32, -1, -17, 0, 1, 524287] {
        for invert in [false, true] {
            let bits = immediate as u32;
            let raw = 0x38a0u64 << 48
                | u64::from(bits & 0x7ffff) << 20
                | u64::from((bits >> 19) & 1) << 56
                | 7 << 39
                | u64::from(invert) << 42
                | 7 << 16
                | 2 << 8
                | 1;
            for payload in [0u32, 0x80000000, 0x7fc12345, 0x7f800000, u32::MAX] {
                let (regs, _) = run(&[mov(2, payload), decoded(raw)], 0);
                assert_eq!(regs[1], if invert { bits } else { payload }, "{raw:016x}");
            }
        }
    }
}

#[test]
fn field_probe_lea_immediate_and_cbuf_are_wrapping_shift_add() {
    // gm107.c:2093,2140,253,302-309; Ryujinx InstDecoders.cs:3597-3627
    // and InstEmitIntegerArithmetic.cs:231-249,481-490.
    for b in [-524288i32, -1, 0, 1, 524287] {
        let bits = b as u32;
        for shift in 0..32u64 {
            for negate in [false, true] {
                for a in [0u32, 1, 0x80000001, u32::MAX] {
                    let shifted = a.wrapping_shl(shift as u32);
                    let expected = bits.wrapping_add(if negate {
                        shifted.wrapping_neg()
                    } else {
                        shifted
                    });
                    let common = 7 << 16 | 2 << 8 | 1 | shift << 39 | u64::from(negate) << 45;
                    for raw in [
                        0x36d7u64 << 48
                            | common
                            | u64::from(bits & 0x7ffff) << 20
                            | u64::from((bits >> 19) & 1) << 56,
                        0x4bd7u64 << 48 | common,
                    ] {
                        let (regs, _) = run(&[mov(2, a), decoded(raw)], bits);
                        assert_eq!(regs[1], expected, "{raw:016x}");
                    }
                }
            }
        }
    }
}

#[test]
fn field_probe_bfi_register_inserts_exact_bits_for_every_valid_range() {
    // gm107.c:2019,203-207; Mesa nv50_ir_emit_gm107.cpp:2293-2298,
    // 2312,2324-2326; Ryujinx InstDecoders.cs:1030-1040,
    // InstEmitBitfield.cs:42-50,154-162.
    let raw = 0x5bf0u64 << 48 | 7 << 16 | 4 << 39 | 3 << 20 | 2 << 8 | 1;
    for offset in 0..=32u32 {
        for count in 0..=32 - offset {
            for (insert, base) in [(0x12345678u32, 0xdeadbeef), (u32::MAX, 0), (0, u32::MAX)] {
                let (regs, _) = run(
                    &[
                        mov(2, insert),
                        mov(3, offset | count << 8),
                        mov(4, base),
                        decoded(raw),
                    ],
                    0,
                );
                let mask = (((1u64 << count) - 1) << offset) as u32;
                let expected = (base & !mask) | ((u64::from(insert) << offset) as u32 & mask);
                assert_eq!(regs[1], expected, "offset {offset}, count {count}");
            }
        }
    }
}

// P2R: envytools gm107.c:1992,2066,2112,1194-1204; semantics:
// Ryujinx InstEmitPredicate.cs:86-114 and RegisterConsts.cs:6-7.
#[test]
fn p2r_merges_predicates_and_flags_for_each_form_and_byte() {
    for cc in [false, true] {
        for byte in 0..4u64 {
            for mask in [0u32, 1, 0x55, 0x7f, 0x80, 0xff, 0xabcde] {
                for pattern in [0u32, 1, 0x2a, 0x55, 0x7f] {
                    let mut setup = (0..7)
                        .map(|p| set_predicate(p, pattern & (1 << p) != 0))
                        .collect::<Vec<_>>();
                    // Establish nontrivial Z/S/C/V with an actual CC writer.
                    setup.extend([
                        mov(2, u32::MAX),
                        mov(3, 1),
                        decoded(0x5c10u64 << 48 | 1 << 47 | 3 << 20 | 7 << 16 | 2 << 8 | 4),
                    ]);
                    setup.extend([mov(1, 0x5aa5_9669), mov(2, 0xa596_69c3), mov(3, mask)]);
                    let state = run_instructions(
                        &setup
                            .iter()
                            .enumerate()
                            .map(|(i, kind)| Instruction {
                                address: i as u32 * 8,
                                predicate: None,
                                kind: *kind,
                            })
                            .collect::<Vec<_>>(),
                        mask,
                    );
                    let packed = if cc {
                        state
                            .1
                            .into_iter()
                            .enumerate()
                            .fold(0, |v, (i, flag)| v | u32::from(flag) << i)
                    } else {
                        pattern
                    };
                    let shifted = (mask & 255) << (byte * 8);
                    let expected = (0xa596_69c3 & !shifted) | ((packed << (byte * 8)) & shifted);
                    for (prefix, encoded_mask) in [
                        (0x5ce8u64, 3u64),
                        (0x4ce8, 0),
                        (0x38e8, u64::from(mask & 0x7ffff)),
                    ] {
                        let raw = prefix << 48
                            | u64::from(cc) << 40
                            | byte << 41
                            | encoded_mask << 20
                            | 7 << 16
                            | 2 << 8
                            | 1;
                        let mut input = setup.clone();
                        input.push(decoded(raw));
                        assert!(matches!(
                            input.last(),
                            Some(InstructionKind::P2R(ops::P2R { .. }))
                        ));
                        assert_eq!(run(&input, mask).0[1], expected, "{raw:016x}, {mask:x}");
                    }
                }
            }
        }
    }
}

// VSETP: envytools gm107.c:2052,1588-1628; semantic reference:
// Ryujinx InstEmitVideoMinMax.cs:122-180. Independent host integer oracle.
#[test]
fn vsetp_extracts_signed_lanes_compares_and_combines_both_outputs() {
    let extend = |bits: u32, sel: u64| -> i64 {
        let (offset, width) = match sel & 7 {
            byte @ 0..=3 => (byte * 8, 8),
            4 => (0, 16),
            5 => (16, 16),
            6 => (0, 32),
            _ => unreachable!(),
        };
        let bits = bits >> offset;
        if sel & 8 != 0 {
            i64::from((bits << (32 - width)) as i32 >> (32 - width))
        } else {
            i64::from(bits & (u32::MAX >> (32 - width)))
        }
    };
    for sa in (0..7u64).chain(8..15) {
        for sb in (0..7u64).chain(8..15) {
            let mixed32 = (sa ^ sb) & 8 != 0 && (sa & 7 == 6 || sb & 7 == 6);
            for cmp in 0..8u64 {
                let raw = 0x50f4u64 << 48
                    | (sa & 8) << 45
                    | (sb & 8) << 46
                    | (sa & 7) << 36
                    | (sb & 7) << 28
                    | (cmp & 3) << 43
                    | (cmp & 4) << 45
                    | 7 << 39
                    | 3 << 20
                    | 7 << 16
                    | 2 << 8
                    | 1;
                if mixed32 {
                    assert!(matches!(
                        decoded(raw),
                        InstructionKind::Unknown(ops::Unknown { .. })
                    ));
                    continue;
                }
                let a = extend(0x807f_ff01, sa);
                let b = extend(0xff80_017f, sb);
                let expected =
                    [false, a < b, a == b, a <= b, a > b, a != b, a >= b, true][cmp as usize];
                for bop in 0..3u64 {
                    for acc in [false, true] {
                        for inv in [false, true] {
                            let raw = raw & !(7 << 39) | bop << 45 | 2 << 39 | u64::from(inv) << 42;
                            let (_, _, predicates) = run_instructions(
                                &[
                                    Instruction {
                                        address: 0,
                                        predicate: None,
                                        kind: mov(2, 0x807f_ff01),
                                    },
                                    Instruction {
                                        address: 8,
                                        predicate: None,
                                        kind: mov(3, 0xff80_017f),
                                    },
                                    Instruction {
                                        address: 16,
                                        predicate: None,
                                        kind: set_predicate(2, acc),
                                    },
                                    decode_word([raw as u32, (raw >> 32) as u32], 24),
                                ],
                                0,
                            );
                            let combine = |v| match bop {
                                0 => v && (acc != inv),
                                1 => v || (acc != inv),
                                _ => v != (acc != inv),
                            };
                            assert_eq!(
                                &predicates[..2],
                                &[combine(expected), combine(!expected)],
                                "{raw:016x}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn vsetp_immediate_sign_and_aliasing_guard_use_old_predicate_values() {
    for signed in [false, true] {
        let raw = 0x50f0u64 << 48
            | u64::from(signed) << 49
            | 0xff80 << 20
            | 2 << 36
            | 2 << 43
            | 7 << 39
            | 7 << 16
            | 2 << 8
            | 1;
        // A byte2 is 0x80, signed B is -128 or unsigned B is 65408.
        let (_, _, ps) = run_instructions(
            &[
                Instruction {
                    address: 0,
                    predicate: None,
                    kind: mov(2, 0x0080_0000),
                },
                decode_word([raw as u32, (raw >> 32) as u32], 8),
            ],
            0,
        );
        assert_eq!(&ps[..2], &[false, true]);
    }
    // Guard P0=true, A<B=false: P0 clears but P1 must still be written.
    let raw = 0x50f4u64 << 48 | 6 << 36 | 6 << 28 | 1 << 43 | 3 << 20 | 2 << 8 | 1;
    let (_, _, ps) = run_instructions(
        &[
            Instruction {
                address: 0,
                predicate: None,
                kind: set_predicate(0, true),
            },
            Instruction {
                address: 8,
                predicate: None,
                kind: mov(2, 9),
            },
            Instruction {
                address: 16,
                predicate: None,
                kind: mov(3, 3),
            },
            decode_word([raw as u32, (raw >> 32) as u32], 24),
        ],
        0,
    );
    assert_eq!(&ps[..2], &[false, true]);
}

#[test]
fn rro_ex2_mode_emits_the_same_module_as_sincos_in_both_forms() {
    // Mode39: envytools gm107.c:1320-1323. Ryujinx
    // InstEmitMultifunction.cs:11-29,86-94 lowers both modes identically
    // for the register (0x5c90) and constant-buffer (0x4c90) forms.
    let header = headers::parse(&[0; 32]).unwrap();
    for opcode in [0x5c90u64, 0x4c90] {
        for modifiers in [0, 1 << 45, 1 << 49, 1 << 45 | 1 << 49] {
            let sincos = (opcode << 48) | modifiers | (3 << 34) | (7 << 20) | (7 << 16) | 5;
            let ex2 = sincos | (1 << 39);
            let module = |raw: u64| {
                let x = Instruction {
                    address: 0,
                    predicate: None,
                    kind: decoded(raw),
                };
                emit_with_header(&[x], Some(&header)).unwrap().words
            };
            assert!(matches!(
                decoded(sincos),
                InstructionKind::Rro(ops::Rro { ex2: false, .. })
            ));
            assert!(matches!(
                decoded(ex2),
                InstructionKind::Rro(ops::Rro { ex2: true, .. })
            ));
            assert_eq!(module(ex2), module(sincos), "{opcode:x} {modifiers:x}");
            // The mode stays visible for audit.
            let text = |raw: u64| {
                Instruction {
                    address: 0,
                    predicate: None,
                    kind: decoded(raw),
                }
                .to_string()
            };
            assert!(text(ex2).contains("RRO.EX2"));
            assert!(text(sincos).contains("RRO.SINCOS"));
        }
    }
    // nvdisasm ignores opcode bits 48 and 50 of the cbuf form, and
    // Ryujinx InstTable.cs (MIT) maps 0x4c90-0x4c97 to RRO.
    let plain = (0x4c90u64 << 48) | (1 << 39) | (7 << 16) | 5;
    for bit in [48, 50] {
        assert_eq!(decoded(plain | (1 << bit)), decoded(plain));
    }
}
