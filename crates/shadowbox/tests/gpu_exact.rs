#![cfg(feature = "gpu-check")]
// Exact GPU comparisons, independent of the software SPIR-V evaluator.
// Keep uncovered families and disagreements visible.
#[path = "support/compute.rs"]
mod compute;
#[path = "support/control_reference.rs"]
mod control_reference;
#[path = "support/divergent_compute.rs"]
mod divergent_compute;
#[path = "support/families.rs"]
mod families;
#[path = "support/lane_reference.rs"]
mod lane_reference;
#[path = "support/memory_reference.rs"]
mod memory_reference;
#[path = "support/reference.rs"]
mod reference;
#[path = "support/spirv_validation.rs"]
mod spirv_validation;
#[path = "support/surface_reference.rs"]
mod surface_reference;
#[path = "support/texture_reference.rs"]
mod texture_reference;
#[path = "support/texture_tail2.rs"]
mod texture_tail2;

use compute::{Compute, WORDS};
use families::{Family, FAMILIES};
use reference::{field, normalise, reference, Expected};
use shadowbox::ir::ops;
use shadowbox::spirv::Op;
use shadowbox::{
    decoder::decode_stream,
    emitter::{emit_unoptimized_with_header, emit_unoptimized_with_options},
    headers,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

const SEED: u64 = 0x5eed_5eed_5eed_5eed;
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        // Same xorshift64* and initial seed as fuzz_validation.rs:24-33.
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}
fn program(raw: u64) -> Vec<u32> {
    // sm50.rs:3418-3428,3438-3468: scheduling word + three instructions.
    // EXIT gm107.c:1925; NOP sm50.rs:3178-3184. No invented no-ops.
    [
        0,
        raw,
        0xe300_u64 << 48 | 7 << 16 | 15,
        0x50b0_u64 << 48 | 7 << 16,
    ]
    .into_iter()
    .flat_map(|w| [w as u32, (w >> 32) as u32])
    .collect()
}
fn texture_program(raw: u64, dynamic: bool) -> Vec<u32> {
    // Immediate MOV, all quad lanes: Mesa sm50.rs:1938-1941.
    // Runtime observer values equal the setup values in both references.
    let mut instructions = if surface_reference::is_surface(raw) {
        surface_reference::setup(raw)
    } else if dynamic {
        texture_reference::runtime_setup(raw)
    } else {
        texture_reference::setup(raw)
    }
    .unwrap_or_default()
    .into_iter()
    .map(|(register, value)| {
        0x0100_u64 << 48 | u64::from(value) << 20 | 15 << 12 | 7 << 16 | u64::from(register)
    })
    .collect::<Vec<_>>();
    instructions.extend([raw, 0xe300_u64 << 48 | 7 << 16 | 15]);
    while !instructions.len().is_multiple_of(3) {
        instructions.push(0x50b0_u64 << 48 | 7 << 16);
    }
    instructions
        .chunks(3)
        .flat_map(|chunk| std::iter::once(0).chain(chunk.iter().copied()))
        .flat_map(|word| [word as u32, (word >> 32) as u32])
        .collect()
}
fn texture_array_program(raw: u64) -> Vec<u32> {
    // Project-authored fixture from Mesa sm50.rs:1938-1941,2171-2197,
    // 2386-2398,2633-2648. R12 supplies a byte offset into cbuf2; LDC reads
    // the packed bindless handle and carries its descriptor-array index.
    let mov =
        |dst: u64, value: u32| 0x0100_u64 << 48 | u64::from(value) << 20 | 15 << 12 | 7 << 16 | dst;
    let ldc = 0xef94_u64 << 48 | 2 << 36 | 0x20 << 20 | 7 << 16 | 12 << 8 | 1;
    let mut instructions = vec![
        mov(10, 0),
        mov(11, 0),
        ldc,
        raw,
        0xe300_u64 << 48 | 7 << 16 | 15,
    ];
    while !instructions.len().is_multiple_of(3) {
        instructions.push(0x50b0_u64 << 48 | 7 << 16);
    }
    instructions
        .chunks(3)
        .flat_map(|chunk| std::iter::once(0).chain(chunk.iter().copied()))
        .flat_map(|word| [word as u32, (word >> 32) as u32])
        .collect()
}
fn representative(family: Family) -> u64 {
    // Same operand setup as family_validation.rs::program.
    // gm107.c:203-218,408-411; conversion sizes sm50.rs:1831-1833,1873-1875,1913-1916.
    let mut raw = u64::from(family.opcode) << 48 | 7 << 16 | 1;
    // CSETP stores its condition at8..12; S2R has no A operand.
    // envytools gm107.c:2058,1647-1680; Mesa sm50.rs:3220-3224.
    raw |= if matches!(family.opcode, 0xef98 | 0x50a0 | 0xf0c8) {
        0
    } else {
        2 << 8
    };
    if matches!(
        family.opcode & 0xfef8,
        0x5ce0 | 0x4ce0 | 0x38e0 | 0x5cb0 | 0x4cb0 | 0x38b0 | 0x5cb8 | 0x4cb8 | 0x38b8
    ) && !family.name.starts_with("I2F_F64")
    {
        raw |= 0xa00;
    }
    if family.name.starts_with("BRA_EXACT_")
        || family.name.starts_with("EXIT_EXACT_")
        || family.name == "SSY_SYNC_EXACT"
    {
        raw &= !0xff;
    }
    raw | family.fields
}
fn inputs(raw: u64) -> Vec<[u32; WORDS]> {
    let mut rng = Rng(SEED ^ raw | 1);
    let edges = [
        0,
        0x8000_0000,
        1,
        u32::MAX,
        31,
        32,
        63,
        64,
        0x7fff_ffff,
        0x0080_0000,
        0x007f_ffff,
        0x8000_0001,
        0x3f00_0000,
        0x3fc0_0000,
        0xbfc0_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc1_2345,
        0xffa0_0042,
        0x4eff_ffff,
        0x4f00_0000,
        0x4f7f_ffff,
        0x4f80_0000,
        0xbe00_3e00,
        0x7bff_7c00,
        0xfc00_7e01,
    ];
    (0..48)
        .map(|case| {
            let mut state = [0; WORDS];
            for (register, word) in state[..255].iter_mut().enumerate() {
                *word = if case < edges.len() {
                    edges[(case + register) % edges.len()]
                } else {
                    rng.next() as u32
                };
            }
            for word in &mut state[256..268] {
                *word = (rng.next() & 1) as u32;
            }
            for (word, bits) in state[268..].iter_mut().enumerate() {
                *bits = (rng.next() as u32).rotate_left(word as u32);
            }
            state[255] = 0;
            state[263] = 1;
            // BFE register range20..27: Mesa sm50.rs:1168-1170.
            // Ryujinx InstEmitBitfield.cs:141-147 extracts offset/count bytes.
            // Preserve every previously defined case. Replace only invalid
            // active ranges with boundaries and interior ranges in the domain.
            if (raw >> 48) & 0xfff8 == 0x5c00 {
                let range = field(raw, 20, 8) as usize;
                let value = state[range];
                let offset = value & 255;
                let count = value >> 8 & 255;
                if offset > 32 || count > 32 - offset {
                    let offset = case as u32 % 33;
                    let count = case as u32 % (33 - offset);
                    state[range] = value & !65535 | count << 8 | offset;
                }
            }
            control_reference::prepare(raw, case, &mut state);
            // Each input set covers true and false guards and all four initial CCs.
            state
        })
        .collect()
}

fn is_f2i64(raw: u64) -> bool {
    matches!(
        raw,
        0x5cb0_0000_0027_1e01
            | 0x5cb0_0000_0027_0e01
            | 0x5cb0_0080_0027_1e01
            | 0x5cb0_0100_0027_1e01
            | 0x5cb0_0180_0027_1e01
            | 0x5cb0_0000_0027_1d01
            | 0x5cb0_0000_0027_0d01
            | 0x5cb0_0080_0027_1d01
            | 0x5cb0_0100_0027_1d01
            | 0x5cb0_0180_0027_1d01
            | 0x5cb0_0080_0027_0e01
            | 0x5cb0_0100_0027_0e01
            | 0x5cb0_0180_0027_0e01
            | 0x5cb0_0080_0027_0d01
            | 0x5cb0_0100_0027_0d01
            | 0x5cb0_0180_0027_0d01
    )
}

fn is_f2i64_source_modifier(raw: u64) -> bool {
    matches!(
        raw,
        0x5cb0_2000_0027_1e01
            | 0x5cb2_0000_0027_1e01
            | 0x5cb2_2000_0027_1e01
            | 0x5cb0_2180_0027_1e01
            | 0x5cb2_0080_0027_0e01
            | 0x5cb0_2180_0027_0e01
    )
}

fn is_dadd_five(raw: u64) -> bool {
    matches!(
        raw,
        0x5c70_0000_0047_0206
            | 0x5c70_4000_0047_0206
            | 0x5c71_0000_0047_0206
            | 0x5c72_0000_0047_0206
            | 0x5c70_2000_0047_0206
            | 0x4c70_0000_0047_0206
            | 0x4c70_4000_0047_0206
            | 0x4c71_0000_0047_0206
            | 0x4c72_0000_0047_0206
            | 0x4c70_2000_0047_0206
            | 0x3870_003f_f007_0206
            | 0x3970_003f_f007_0206
            | 0x3870_403f_e007_0206
            | 0x3871_0040_0007_0206
            | 0x3971_4040_0007_0206
    )
}

fn is_dmul_exact(raw: u64) -> bool {
    matches!(
        raw,
        0x5c80_0000_0047_0206
            | 0x5c81_0000_0047_0206
            | 0x4c80_0000_0047_0206
            | 0x4c81_0000_0047_0206
            | 0x3880_003f_f007_0206
            | 0x5c80_0000_0007_0000
            | 0x5c80_0000_0007_0001
            | 0x5c80_0000_0007_0200
            | 0x5c80_0000_0047_0000
            | 0x5c81_0000_0007_0000
    )
}

fn is_dfma_six(raw: u64) -> bool {
    matches!(
        raw,
        0x5b70_0000_0007_0000
            | 0x5b71_0000_0007_0000
            | 0x5b72_0000_0007_0000
            | 0x5b70_0000_0007_0200
            | 0x5b70_0000_0047_0000
            | 0x5b70_0200_0007_0000
    )
}

fn prepare_dadd_cases(states: &mut [[u32; WORDS]]) {
    // Finite exact and deliberately inexact binary64 additions exercise sign,
    // magnitude, cancellation and modifier-sensitive inputs without relying
    // on implementation-defined NaN payload selection.
    let values = [
        (0.0_f64, 0.0_f64),
        (-0.0, 0.0),
        (1.0, 2.0),
        (-1.0, 2.0),
        (1.5, -0.5),
        (-1.5, -0.5),
        (f64::MIN_POSITIVE, f64::MIN_POSITIVE),
        (f64::MAX / 2.0, f64::MAX / 4.0),
        (1.0, f64::EPSILON),
        (1.0, f64::EPSILON / 2.0),
        (2.0_f64.powi(500), 2.0_f64.powi(448)),
        (-2.0_f64.powi(500), 2.0_f64.powi(499)),
    ];
    for (case, state) in states.iter_mut().enumerate() {
        let (lhs, rhs) = values[case % values.len()];
        let lhs = lhs.to_bits();
        let rhs = rhs.to_bits();
        state[2] = lhs as u32;
        state[3] = (lhs >> 32) as u32;
        state[4] = rhs as u32;
        state[5] = (rhs >> 32) as u32;
    }
}

const FADD_FTZ_RM_BASES: [u64; 11] = [
    0x5c58_1080_0000_0000,
    0x5c58_1080_0007_0000,
    0x5c58_10c1_e777_1d6e,
    0x5c58_1286_3767_da24,
    0x5c58_14bd_d8e7_972e,
    0x5c58_14f1_e8b7_f5ce,
    0x5c58_18d7_7b67_7d3d,
    0x5c58_1ad8_99f7_42aa,
    0x5c58_1ccf_af87_30a7,
    0x5c58_1e8d_1517_72c3,
    0x5c58_1e90_6e97_1225,
];
const FADD_FTZ_RM_VARIABLE_BITS: [u32; 6] = [19, 45, 46, 47, 48, 49];

fn fadd_ftz_rm_words() -> impl Iterator<Item = u64> {
    FADD_FTZ_RM_BASES.into_iter().flat_map(|base| {
        (0_u8..64).map(move |selector| {
            FADD_FTZ_RM_VARIABLE_BITS
                .iter()
                .enumerate()
                .fold(base, |raw, (index, bit)| {
                    raw | u64::from(selector >> index & 1) << bit
                })
        })
    })
}

fn is_fadd_ftz_rm(raw: u64) -> bool {
    const MASK: u64 = 0xfffc_1fff_fff7_ffff;
    FADD_FTZ_RM_BASES.contains(&(raw & MASK))
}

fn prepare_fadd_ftz_rm_cases(raw: u64, states: &mut [[u32; WORDS]]) {
    let lhs = field(raw, 8, 8) as usize;
    let rhs = field(raw, 20, 8) as usize;
    let values = [
        (0.0_f32, -0.0_f32),
        (1.0, 2.0),
        (-1.0, f32::from_bits(1)),
        (1.0, f32::EPSILON / 2.0),
        (f32::MIN_POSITIVE, f32::from_bits(1)),
        (f32::from_bits(1), f32::from_bits(1)),
        (f32::MAX / 2.0, f32::MAX / 4.0),
        (f32::MAX, f32::MAX),
        (-f32::MAX, -f32::MAX),
        (2.0_f32.powi(60), 2.0_f32.powi(35)),
        (-2.0_f32.powi(60), 2.0_f32.powi(59)),
        (1.0, -1.0),
    ];
    for (case, state) in states.iter_mut().enumerate() {
        let (a, b) = values[case % values.len()];
        state[lhs] = a.to_bits();
        state[rhs] = b.to_bits();
        state[255] = 0;
    }
}

const DADD_IMMEDIATE_RN_CC_BASES: [u64; 8] = [
    0x3870_8000_0007_0000,
    0x3870_824d_7ec7_7914,
    0x3870_8c27_1987_c17e,
    0x3870_8e43_ba37_5b2b,
    0x3870_9866_f617_7807,
    0x3870_9e00_0ff7_ffff,
    0x3870_9e04_6767_b14c,
    0x3870_9e7a_5ea7_cf2a,
];
const DADD_IMMEDIATE_RN_CC_VARIABLE_BITS: [u32; 7] = [19, 45, 46, 48, 49, 50, 56];

fn dadd_immediate_rn_cc_words() -> impl Iterator<Item = u64> {
    DADD_IMMEDIATE_RN_CC_BASES.into_iter().flat_map(|base| {
        (0_u8..128).map(move |selector| {
            DADD_IMMEDIATE_RN_CC_VARIABLE_BITS
                .iter()
                .enumerate()
                .fold(base, |raw, (index, bit)| {
                    raw | u64::from(selector >> index & 1) << bit
                })
        })
    })
}

fn is_dadd_immediate_rn_cc(raw: u64) -> bool {
    const MASK: u64 = 0xfef8_9fff_fff7_ffff;
    DADD_IMMEDIATE_RN_CC_BASES.contains(&(raw & MASK))
}

fn prepare_dadd_immediate_rn_cc_cases(raw: u64, states: &mut [[u32; WORDS]]) {
    let lhs = field(raw, 8, 8) as usize;
    let values = [
        0.0_f64.to_bits(),
        (-0.0_f64).to_bits(),
        1.0_f64.to_bits(),
        (-1.0_f64).to_bits(),
        (1.0_f64.to_bits() - 1),
        (1.0_f64.to_bits() + 1),
        f64::from_bits(1).to_bits(),
        f64::MIN_POSITIVE.to_bits(),
        (f64::MAX / 2.0).to_bits(),
        (-(f64::MAX / 2.0)).to_bits(),
        f64::MAX.to_bits(),
        (-f64::MAX).to_bits(),
    ];
    for (state, value) in states.iter_mut().zip(values.into_iter().cycle()) {
        state[lhs] = value as u32;
        state[lhs | 1] = (value >> 32) as u32;
        if lhs & 1 == 1 {
            state[lhs] = (value >> 32) as u32;
        }
        state[255] = 0;
    }
}

const DADD_IMMEDIATE_RM_BASES: [u64; 8] = [
    0x3870_0080_0007_0000,
    0x3870_02cd_7ec7_7914,
    0x3870_0ca7_1987_c17e,
    0x3870_0ec3_ba37_5b2b,
    0x3870_18e6_f617_7807,
    0x3870_1e80_0ff7_ffff,
    0x3870_1e84_6767_b14c,
    0x3870_1efa_5ea7_cf2a,
];
const DADD_IMMEDIATE_RM_VARIABLE_BITS: [u32; 8] = [19, 45, 46, 47, 48, 49, 50, 56];

fn dadd_immediate_rm_words() -> impl Iterator<Item = u64> {
    DADD_IMMEDIATE_RM_BASES.into_iter().flat_map(|base| {
        (0_u16..256).map(move |selector| {
            DADD_IMMEDIATE_RM_VARIABLE_BITS
                .iter()
                .enumerate()
                .fold(base, |raw, (index, bit)| {
                    raw | u64::from(selector >> index & 1) << bit
                })
        })
    })
}

fn is_dadd_immediate_rm(raw: u64) -> bool {
    const MASK: u64 = 0xfef8_1fff_fff7_ffff;
    DADD_IMMEDIATE_RM_BASES.contains(&(raw & MASK))
}

fn prepare_dadd_immediate_rm_cases(raw: u64, states: &mut [[u32; WORDS]]) {
    let lhs = field(raw, 8, 8) as usize;
    let values = [
        0.0_f64.to_bits(),
        (-0.0_f64).to_bits(),
        1.0_f64.to_bits(),
        (-1.0_f64).to_bits(),
        (1.0_f64.to_bits() - 1),
        (1.0_f64.to_bits() + 1),
        f64::from_bits(1).to_bits(),
        f64::MIN_POSITIVE.to_bits(),
        (f64::MAX / 2.0).to_bits(),
        (-(f64::MAX / 2.0)).to_bits(),
        f64::MAX.to_bits(),
        (-f64::MAX).to_bits(),
    ];
    for (state, value) in states.iter_mut().zip(values.into_iter().cycle()) {
        state[lhs] = value as u32;
        state[lhs | 1] = (value >> 32) as u32;
        if lhs & 1 == 1 {
            state[lhs] = (value >> 32) as u32;
        }
        state[255] = 0;
    }
}

const DADD_REGISTER_RM_BASES: [u64; 8] = [
    0x5c70_0080_0007_0000,
    0x5c70_0080_a427_440b,
    0x5c70_02e1_8077_a0d1,
    0x5c70_06dc_cc47_3d36,
    0x5c70_0cb2_f917_817b,
    0x5c70_0cb4_1db7_aa3a,
    0x5c70_1ca4_8b47_e19b,
    0x5c70_1caa_6b67_2c35,
];
const DADD_REGISTER_RM_VARIABLE_BITS: [u32; 7] = [19, 45, 46, 47, 48, 49, 50];

fn dadd_register_rm_words() -> impl Iterator<Item = u64> {
    DADD_REGISTER_RM_BASES.into_iter().flat_map(|base| {
        (0_u8..128).map(move |selector| {
            DADD_REGISTER_RM_VARIABLE_BITS
                .iter()
                .enumerate()
                .fold(base, |raw, (index, bit)| {
                    raw | u64::from(selector >> index & 1) << bit
                })
        })
    })
}

fn is_dadd_register_rm(raw: u64) -> bool {
    const MASK: u64 = 0xfff8_1fff_fff7_ffff;
    DADD_REGISTER_RM_BASES.contains(&(raw & MASK))
}

fn prepare_dadd_register_rm_cases(raw: u64, states: &mut [[u32; WORDS]]) {
    let lhs = field(raw, 8, 8) as usize;
    let rhs = field(raw, 20, 8) as usize;
    let values = [
        (0.0_f64, -0.0_f64),
        (1.0, 2.0),
        (-1.0, f64::from_bits(1)),
        (1.0, f64::EPSILON / 2.0),
        (f64::MIN_POSITIVE, f64::from_bits(1)),
        (f64::from_bits(1), f64::from_bits(1)),
        (f64::MAX / 2.0, f64::MAX / 4.0),
        (f64::MAX, f64::MAX),
        (-f64::MAX, -f64::MAX),
        (2.0_f64.powi(500), 2.0_f64.powi(448)),
        (-2.0_f64.powi(500), 2.0_f64.powi(499)),
        (1.0, -1.0),
    ];
    let set_pair = |state: &mut [u32; WORDS], register: usize, value: u64| {
        state[register] = value as u32;
        state[register | 1] = (value >> 32) as u32;
        if register & 1 == 1 {
            state[register] = (value >> 32) as u32;
        }
    };
    for (case, state) in states.iter_mut().enumerate() {
        let (lhs_value, rhs_value) = values[case % values.len()];
        set_pair(state, lhs, lhs_value.to_bits());
        set_pair(state, rhs, rhs_value.to_bits());
        state[255] = 0;
    }
}

const NOP_REGIONS: &[(u64, &[u32])] = &[
    (0x50b1_0000_0007_0000, &[8, 9, 10, 11, 12, 19, 49, 50]),
    (0x50b1_28de_18a7_00f2, &[8, 9, 10, 11, 12, 19, 49, 50]),
    (0x50b1_6f7a_8a07_00b2, &[8, 9, 10, 11, 12, 19, 49, 50]),
    (0x50b1_953c_3f97_8072, &[8, 9, 10, 11, 12, 19, 49, 50]),
    (0x50b1_9607_1787_0070, &[8, 9, 10, 11, 12, 19, 49, 50]),
    (0x50b1_87f3_87f7_8061, &[8, 10, 11, 12, 19, 49, 50]),
    (0x50b1_f38f_1097_40df, &[8, 9, 10, 12, 19, 49, 50]),
];
fn nop_words() -> impl Iterator<Item = u64> {
    NOP_REGIONS.iter().flat_map(|&(base, bits)| {
        (0..(1_u16 << bits.len())).map(move |selector| {
            bits.iter().enumerate().fold(base, |raw, (i, bit)| {
                raw | u64::from((selector >> i) & 1) << bit
            })
        })
    })
}

const NOP_TAIL_REGIONS: &[(u64, &[u32])] = &[
    (0x50b2_0000_0007_0000, &[8, 9, 10, 11, 12, 19, 50]),
    (0x50b2_28de_18a7_00f2, &[8, 9, 10, 11, 12, 19, 50]),
    (0x50b2_6f7a_8a07_00b2, &[8, 9, 10, 11, 12, 19, 50]),
    (0x50b2_953c_3f97_8072, &[8, 9, 10, 11, 12, 19, 50]),
    (0x50b2_9607_1787_0070, &[8, 9, 10, 11, 12, 19, 50]),
    (0x50b2_87f3_87f7_8061, &[8, 10, 11, 12, 19, 50]),
    (0x50b2_f38f_1097_40df, &[8, 9, 10, 12, 19, 50]),
    (0x50b4_0000_0007_0000, &[8, 9, 10, 11, 12, 19]),
    (0x50b4_28de_18a7_00f2, &[8, 9, 10, 11, 12, 19]),
    (0x50b4_6f7a_8a07_00b2, &[8, 9, 10, 11, 12, 19]),
    (0x50b4_953c_3f97_8072, &[8, 9, 10, 11, 12, 19]),
    (0x50b4_9607_1787_0070, &[8, 9, 10, 11, 12, 19]),
    (0x50b4_87f3_87f7_8061, &[8, 10, 11, 12, 19]),
    (0x50b4_f38f_1097_40df, &[8, 9, 10, 12, 19]),
];
fn nop_tail_words() -> impl Iterator<Item = u64> {
    NOP_TAIL_REGIONS.iter().flat_map(|&(base, bits)| {
        (0..(1_u16 << bits.len())).map(move |selector| {
            bits.iter().enumerate().fold(base, |raw, (index, bit)| {
                raw | u64::from((selector >> index) & 1) << bit
            })
        })
    })
}

const DADD_CBUF_RM_BASES: [u64; 9] = [
    0x4c70_0080_0000_0000,
    0x4c70_0080_0007_0000,
    0x4c70_0ade_fac7_4812,
    0x4c70_0e83_90c7_921f,
    0x4c70_16d1_9b07_39ac,
    0x4c70_16ec_8347_824a,
    0x4c70_18fd_a827_4194,
    0x4c70_1ce6_b537_0757,
    0x4c70_1e80_0ff7_ffff,
];
const DADD_CBUF_RM_VARIABLE_BITS: [u32; 7] = [19, 45, 46, 47, 48, 49, 50];

fn dadd_cbuf_rm_words() -> impl Iterator<Item = u64> {
    DADD_CBUF_RM_BASES.into_iter().flat_map(|base| {
        (0_u8..128).map(move |selector| {
            DADD_CBUF_RM_VARIABLE_BITS
                .iter()
                .enumerate()
                .fold(base, |raw, (index, bit)| {
                    raw | u64::from(selector >> index & 1) << bit
                })
        })
    })
}

fn is_dadd_cbuf_rm(raw: u64) -> bool {
    const MASK: u64 = 0xfff8_1fff_fff7_ffff;
    DADD_CBUF_RM_BASES.contains(&(raw & MASK))
}

fn prepare_dadd_cbuf_rm_cases(raw: u64, states: &mut [[u32; WORDS]]) {
    let lhs = field(raw, 8, 8) as usize;
    let values = [
        0.0_f64.to_bits(),
        (-0.0_f64).to_bits(),
        1.0_f64.to_bits(),
        (-1.0_f64).to_bits(),
        (1.0_f64.to_bits() - 1),
        (1.0_f64.to_bits() + 1),
        f64::from_bits(1).to_bits(),
        f64::MIN_POSITIVE.to_bits(),
        (f64::MAX / 2.0).to_bits(),
        (-(f64::MAX / 2.0)).to_bits(),
        f64::MAX.to_bits(),
        (-f64::MAX).to_bits(),
    ];
    for (state, value) in states.iter_mut().zip(values.into_iter().cycle()) {
        state[lhs] = value as u32;
        state[lhs | 1] = (value >> 32) as u32;
        if lhs & 1 == 1 {
            state[lhs] = (value >> 32) as u32;
        }
        state[255] = 0;
    }
}

const DADD_CBUF_RP_BASES: [u64; 9] = [
    0x4c70_0100_0000_0000,
    0x4c70_0100_0007_0000,
    0x4c70_0b5e_fac7_4812,
    0x4c70_0f03_90c7_921f,
    0x4c70_1751_9b07_39ac,
    0x4c70_176c_8347_824a,
    0x4c70_197d_a827_4194,
    0x4c70_1d66_b537_0757,
    0x4c70_1f00_0ff7_ffff,
];
const DADD_CBUF_RP_VARIABLE_BITS: [u32; 7] = [19, 45, 46, 47, 48, 49, 50];

fn dadd_cbuf_rp_words() -> impl Iterator<Item = u64> {
    DADD_CBUF_RP_BASES.into_iter().flat_map(|base| {
        (0_u8..128).map(move |selector| {
            DADD_CBUF_RP_VARIABLE_BITS
                .iter()
                .enumerate()
                .fold(base, |raw, (index, bit)| {
                    raw | u64::from(selector >> index & 1) << bit
                })
        })
    })
}

fn is_dadd_cbuf_rp(raw: u64) -> bool {
    const MASK: u64 = 0xfff8_1fff_fff7_ffff;
    DADD_CBUF_RP_BASES.contains(&(raw & MASK))
}

const DADD_CBUF_RZ_BASES: [u64; 9] = [
    0x4c70_0180_0000_0000,
    0x4c70_0180_0007_0000,
    0x4c70_0bde_fac7_4812,
    0x4c70_0f83_90c7_921f,
    0x4c70_17d1_9b07_39ac,
    0x4c70_17ec_8347_824a,
    0x4c70_19fd_a827_4194,
    0x4c70_1de6_b537_0757,
    0x4c70_1f80_0ff7_ffff,
];
const DADD_CBUF_RZ_VARIABLE_BITS: [u32; 7] = [19, 45, 46, 47, 48, 49, 50];

fn dadd_cbuf_rz_words() -> impl Iterator<Item = u64> {
    DADD_CBUF_RZ_BASES.into_iter().flat_map(|base| {
        (0_u8..128).map(move |selector| {
            DADD_CBUF_RZ_VARIABLE_BITS
                .iter()
                .enumerate()
                .fold(base, |raw, (index, bit)| {
                    raw | u64::from(selector >> index & 1) << bit
                })
        })
    })
}

fn is_dadd_cbuf_rz(raw: u64) -> bool {
    const MASK: u64 = 0xfff8_1fff_fff7_ffff;
    DADD_CBUF_RZ_BASES.contains(&(raw & MASK))
}

const DADD_REGISTER_RP_BASES: [u64; 8] = [
    0x5c70_0100_0007_0000,
    0x5c70_0100_a427_440b,
    0x5c70_0361_8077_a0d1,
    0x5c70_075c_cc47_3d36,
    0x5c70_0d32_f917_817b,
    0x5c70_0d34_1db7_aa3a,
    0x5c70_1d24_8b47_e19b,
    0x5c70_1d2a_6b67_2c35,
];
const DADD_REGISTER_RP_VARIABLE_BITS: [u32; 7] = [19, 45, 46, 47, 48, 49, 50];

fn dadd_register_rp_words() -> impl Iterator<Item = u64> {
    DADD_REGISTER_RP_BASES.into_iter().flat_map(|base| {
        (0_u8..128).map(move |selector| {
            DADD_REGISTER_RP_VARIABLE_BITS
                .iter()
                .enumerate()
                .fold(base, |raw, (index, bit)| {
                    raw | u64::from(selector >> index & 1) << bit
                })
        })
    })
}

fn is_dadd_register_rp(raw: u64) -> bool {
    const MASK: u64 = 0xfff8_1fff_fff7_ffff;
    DADD_REGISTER_RP_BASES.contains(&(raw & MASK))
}

const DADD_REGISTER_RZ_BASES: [u64; 8] = [
    0x5c70_0180_0007_0000,
    0x5c70_0180_a427_440b,
    0x5c70_03e1_8077_a0d1,
    0x5c70_07dc_cc47_3d36,
    0x5c70_0db2_f917_817b,
    0x5c70_0db4_1db7_aa3a,
    0x5c70_1da4_8b47_e19b,
    0x5c70_1daa_6b67_2c35,
];
const DADD_REGISTER_RZ_VARIABLE_BITS: [u32; 7] = [19, 45, 46, 47, 48, 49, 50];

fn dadd_register_rz_words() -> impl Iterator<Item = u64> {
    DADD_REGISTER_RZ_BASES.into_iter().flat_map(|base| {
        (0_u8..128).map(move |selector| {
            DADD_REGISTER_RZ_VARIABLE_BITS
                .iter()
                .enumerate()
                .fold(base, |raw, (index, bit)| {
                    raw | u64::from(selector >> index & 1) << bit
                })
        })
    })
}

fn is_dadd_register_rz(raw: u64) -> bool {
    const MASK: u64 = 0xfff8_1fff_fff7_ffff;
    DADD_REGISTER_RZ_BASES.contains(&(raw & MASK))
}

const DADD_IMMEDIATE_RP_BASES: [u64; 8] = [
    0x3870_0100_0007_0000,
    0x3870_034d_7ec7_7914,
    0x3870_0d27_1987_c17e,
    0x3870_0f43_ba37_5b2b,
    0x3870_1966_f617_7807,
    0x3870_1f00_0ff7_ffff,
    0x3870_1f04_6767_b14c,
    0x3870_1f7a_5ea7_cf2a,
];
const DADD_IMMEDIATE_RP_VARIABLE_BITS: [u32; 8] = [19, 45, 46, 47, 48, 49, 50, 56];

fn dadd_immediate_rp_words() -> impl Iterator<Item = u64> {
    DADD_IMMEDIATE_RP_BASES.into_iter().flat_map(|base| {
        (0_u16..256).map(move |selector| {
            DADD_IMMEDIATE_RP_VARIABLE_BITS
                .iter()
                .enumerate()
                .fold(base, |raw, (index, bit)| {
                    raw | u64::from(selector >> index & 1) << bit
                })
        })
    })
}

fn is_dadd_immediate_rp(raw: u64) -> bool {
    const MASK: u64 = 0xfef8_1fff_fff7_ffff;
    DADD_IMMEDIATE_RP_BASES.contains(&(raw & MASK))
}

fn prepare_dadd_immediate_rp_cases(raw: u64, states: &mut [[u32; WORDS]]) {
    let lhs = field(raw, 8, 8) as usize;
    let values = [
        0.0_f64.to_bits(),
        (-0.0_f64).to_bits(),
        1.0_f64.to_bits(),
        (-1.0_f64).to_bits(),
        (1.0_f64.to_bits() - 1),
        (1.0_f64.to_bits() + 1),
        f64::from_bits(1).to_bits(),
        f64::MIN_POSITIVE.to_bits(),
        (f64::MAX / 2.0).to_bits(),
        (-(f64::MAX / 2.0)).to_bits(),
        f64::MAX.to_bits(),
        (-f64::MAX).to_bits(),
    ];
    for (state, value) in states.iter_mut().zip(values.into_iter().cycle()) {
        state[lhs] = value as u32;
        state[lhs | 1] = (value >> 32) as u32;
        if lhs & 1 == 1 {
            state[lhs] = (value >> 32) as u32;
        }
        state[255] = 0;
    }
}

const DADD_IMMEDIATE_RZ_BASES: [u64; 8] = [
    0x3870_0180_0007_0000,
    0x3870_03cd_7ec7_7914,
    0x3870_0da7_1987_c17e,
    0x3870_0fc3_ba37_5b2b,
    0x3870_19e6_f617_7807,
    0x3870_1f80_0ff7_ffff,
    0x3870_1f84_6767_b14c,
    0x3870_1ffa_5ea7_cf2a,
];
const DADD_IMMEDIATE_RZ_VARIABLE_BITS: [u32; 8] = [19, 45, 46, 47, 48, 49, 50, 56];

fn dadd_immediate_rz_words() -> impl Iterator<Item = u64> {
    DADD_IMMEDIATE_RZ_BASES.into_iter().flat_map(|base| {
        (0_u16..256).map(move |selector| {
            DADD_IMMEDIATE_RZ_VARIABLE_BITS
                .iter()
                .enumerate()
                .fold(base, |raw, (index, bit)| {
                    raw | u64::from(selector >> index & 1) << bit
                })
        })
    })
}

fn is_dadd_immediate_rz(raw: u64) -> bool {
    const MASK: u64 = 0xfef8_1fff_fff7_ffff;
    DADD_IMMEDIATE_RZ_BASES.contains(&(raw & MASK))
}

fn prepare_dadd_immediate_rz_cases(raw: u64, states: &mut [[u32; WORDS]]) {
    let lhs = field(raw, 8, 8) as usize;
    let values = [
        0.0_f64.to_bits(),
        (-0.0_f64).to_bits(),
        1.0_f64.to_bits(),
        (-1.0_f64).to_bits(),
        (1.0_f64.to_bits() - 1),
        (1.0_f64.to_bits() + 1),
        f64::from_bits(1).to_bits(),
        f64::MIN_POSITIVE.to_bits(),
        (f64::MAX / 2.0).to_bits(),
        (-(f64::MAX / 2.0)).to_bits(),
        f64::MAX.to_bits(),
        (-f64::MAX).to_bits(),
    ];
    for (state, value) in states.iter_mut().zip(values.into_iter().cycle()) {
        state[lhs] = value as u32;
        state[lhs | 1] = (value >> 32) as u32;
        if lhs & 1 == 1 {
            state[lhs] = (value >> 32) as u32;
        }
        state[255] = 0;
    }
}

fn is_fmul_fmz(raw: u64) -> bool {
    matches!(
        raw,
        0x5c68_2000_0037_0201
            | 0x4c69_2000_0017_0201
            | 0x386c_203f_c007_0201
            | 0x5c68_2200_0037_0201
            | 0x1e43_fc00_0007_0201
            | 0x1ec3_fc00_0007_0201
    )
}

fn prepare_fmul_fmz_cases(states: &mut [[u32; WORDS]]) {
    // FMZ: a +/-0 or flushed-subnormal factor against Inf, NaN, finite and
    // zero values in either operand, written into the last (random) states.
    let values = [
        (0, 0x7f80_0000),
        (0x8000_0000, 0x7fc1_2345),
        (0x7f80_0000, 0x8000_0000),
        (0xffa0_0042, 0),
        (0x0000_0001, 0xff80_0000),
        (0x7f80_0000, 0x807f_ffff),
        (0, 0xbfc0_0000),
        (0x8000_0000, 0x3fc0_0000),
        (0x3fc0_0000, 0x8000_0000),
        (0x807f_ffff, 0x7fc1_2345),
        (0x8000_0000, 0x8000_0000),
        (0x3f00_0000, 0x0000_0001),
    ];
    for (state, (lhs, rhs)) in states.iter_mut().rev().zip(values) {
        state[2] = lhs;
        state[3] = rhs;
    }
}

fn prepare_fsetp_immediate_cases(states: &mut [[u32; WORDS]]) {
    // R2 meets each immediate magnitude with both signs, both signed zeros,
    // infinities and quiet NaNs of either sign so abs/negate on B decide
    // every equality and ordering edge the six words can reach.
    let values = [
        0x3f80_0000,
        0xbf80_0000,
        0x4000_0000,
        0xc000_0000,
        0x3f00_0000,
        0xbf00_0000,
        0,
        0x8000_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0042,
        0xffc0_1337,
    ];
    for (state, value) in states.iter_mut().zip(values.into_iter().cycle()) {
        state[2] = value;
    }
}

fn prepare_fsetp_second_destination_cases(states: &mut [[u32; WORDS]]) {
    // R2 and R4 pair signed zeros, finite ordering, equal values,
    // infinities and quiet NaNs of either sign. Subnormals are excluded:
    // the existing FSETP comparison is a float compare, which RADV flushes.
    let values = [
        0x3f80_0000,
        0xbf80_0000,
        0x4000_0000,
        0xc000_0000,
        0x3f00_0000,
        0xbf00_0000,
        0,
        0x8000_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0042,
        0xffc0_1337,
    ];
    for (case, state) in states.iter_mut().enumerate() {
        if let (Some(&lhs), Some(&rhs)) = (
            values.get(case % values.len()),
            values.get(case * 7 % values.len()),
        ) {
            state[2] = lhs;
            state[4] = rhs;
        }
    }
}

fn prepare_isetp_second_destination_cases(states: &mut [[u32; WORDS]]) {
    // R2 and R4 pair zero, the immediates 7 and -5 with their neighbors,
    // the signed and unsigned extremes, and the constant-buffer word at
    // c[0][0x10]. Every third case makes R4 equal to R2 so the equality
    // edge of each relation runs on the register forms.
    let values = [
        0,
        1,
        6,
        7,
        8,
        -6_i32 as u32,
        -5_i32 as u32,
        -4_i32 as u32,
        0x7fff_ffff,
        0x8000_0000,
        u32::MAX,
        compute::cbuf(0, 4),
    ];
    for (case, state) in states.iter_mut().enumerate() {
        if let (Some(&lhs), Some(&rhs)) = (
            values.get(case % values.len()),
            values.get(case * 7 % values.len()),
        ) {
            state[2] = lhs;
            state[4] = if case % 3 == 0 { lhs } else { rhs };
        }
    }
}

fn prepare_shr_brev_cases(states: &mut [[u32; WORDS]]) {
    // R2 takes bit patterns whose reversal differs from themselves and
    // whose reversed sign bit is both clear and set; R4 takes shift amounts
    // across the 31/32 clamp edge, the wrap edge at 32/33 and huge values.
    let values = [
        0,
        1,
        0x8000_0000,
        0x0000_00f1,
        0x1234_5678,
        0xf0f0_0f0e,
        0x7fff_fffe,
        u32::MAX,
    ];
    let shifts = [0, 1, 7, 31, 32, 33, 63, 255, u32::MAX];
    for (case, state) in states.iter_mut().enumerate() {
        if let (Some(&lhs), Some(&shift)) = (
            values.get(case % values.len()),
            shifts.get(case % shifts.len()),
        ) {
            state[2] = lhs;
            state[4] = shift;
        }
    }
}

fn prepare_dset_cases(states: &mut [[u32; WORDS]]) {
    // DSET comparison cases deliberately cover both signed zeros, finite
    // ordering, infinities, one NaN in either position, and two NaNs with
    // distinct signs and payloads. Comparisons never depend on payload choice.
    let values = [
        (0.0_f64.to_bits(), (-0.0_f64).to_bits()),
        ((-0.0_f64).to_bits(), 0.0_f64.to_bits()),
        (1.0_f64.to_bits(), 2.0_f64.to_bits()),
        (2.0_f64.to_bits(), 1.0_f64.to_bits()),
        (f64::NEG_INFINITY.to_bits(), f64::INFINITY.to_bits()),
        (f64::INFINITY.to_bits(), f64::NEG_INFINITY.to_bits()),
        (0x7ff8_0000_0000_0042, 1.0_f64.to_bits()),
        (1.0_f64.to_bits(), 0x7ff8_0000_0000_1337),
        (0xfff8_0000_0000_00a5, (-1.0_f64).to_bits()),
        ((-1.0_f64).to_bits(), 0xfff8_0000_0000_00a5),
        (0x7ff8_0000_0000_cafe, 0xfff8_0000_0000_beef),
        (f64::MIN_POSITIVE.to_bits(), (-f64::MIN_POSITIVE).to_bits()),
    ];
    for (state, (lhs, rhs)) in states.iter_mut().zip(values.into_iter().cycle()) {
        state[2] = lhs as u32;
        state[3] = (lhs >> 32) as u32;
        state[4] = rhs as u32;
        state[5] = (rhs >> 32) as u32;
    }
}

fn prepare_dset_cc_cases(raw: u64, states: &mut [[u32; WORDS]]) {
    let values = [
        (0.0_f64.to_bits(), (-0.0_f64).to_bits()),
        ((-0.0_f64).to_bits(), 0.0_f64.to_bits()),
        (1.0_f64.to_bits(), 2.0_f64.to_bits()),
        (2.0_f64.to_bits(), 1.0_f64.to_bits()),
        (f64::NEG_INFINITY.to_bits(), f64::INFINITY.to_bits()),
        (f64::INFINITY.to_bits(), f64::NEG_INFINITY.to_bits()),
        (0x7ff8_0000_0000_0042, 1.0_f64.to_bits()),
        (1.0_f64.to_bits(), 0x7ff8_0000_0000_1337),
        (0x7ff8_0000_0000_cafe, 0xfff8_0000_0000_beef),
    ];
    for (case, state) in states.iter_mut().enumerate() {
        let (lhs, rhs) = values[case % values.len()];
        let lhs_register = field(raw, 8, 8) as usize;
        state[lhs_register] = if lhs_register & 1 == 1 {
            (lhs >> 32) as u32
        } else {
            lhs as u32
        };
        state[lhs_register | 1] = (lhs >> 32) as u32;
        if raw >> 60 == 5 {
            let rhs_register = field(raw, 20, 8) as usize;
            state[rhs_register] = if rhs_register & 1 == 1 {
                (rhs >> 32) as u32
            } else {
                rhs as u32
            };
            state[rhs_register | 1] = (rhs >> 32) as u32;
        }
    }
}

fn prepare_dmnmx_cases(raw: u64, states: &mut [[u32; WORDS]]) {
    // GLSL FMin/FMax cases preserve and compare exact payload bits rather than
    // normalizing exceptional values. Applying the six raw words' modifiers
    // to this table also covers modifier-created signed zero and
    // modifier-changed NaN signs/payloads.
    let qnan_a = 0x7ff8_0000_0000_0042_u64;
    let qnan_b = 0x7ff8_0000_0000_1337_u64;
    let qnan_negative = 0xfff8_0000_0000_00a5_u64;
    let modifier_preimage = |target: u64, absolute: bool, negate: bool| {
        let before_negate = if negate { target ^ 1_u64 << 63 } else { target };
        assert!(!absolute || before_negate >> 63 == 0);
        before_negate
    };
    let both_nan_a = 0x7ff8_0000_0000_cafe_u64;
    let both_nan_b = 0x7ff8_0000_0000_beef_u64;
    let lhs_both_a = modifier_preimage(both_nan_a, field(raw, 46, 1) != 0, field(raw, 48, 1) != 0);
    let rhs_both_a = modifier_preimage(both_nan_a, field(raw, 49, 1) != 0, field(raw, 45, 1) != 0);
    let lhs_both_b = modifier_preimage(both_nan_b, field(raw, 46, 1) != 0, field(raw, 48, 1) != 0);
    let rhs_both_b = modifier_preimage(both_nan_b, field(raw, 49, 1) != 0, field(raw, 45, 1) != 0);
    let cases = [
        (0.0_f64.to_bits(), 0.0_f64.to_bits()),
        ((-0.0_f64).to_bits(), (-0.0_f64).to_bits()),
        (0.0_f64.to_bits(), (-0.0_f64).to_bits()),
        ((-0.0_f64).to_bits(), 0.0_f64.to_bits()),
        (qnan_a, 1.0_f64.to_bits()),
        (1.0_f64.to_bits(), qnan_b),
        (qnan_negative, (-1.0_f64).to_bits()),
        ((-1.0_f64).to_bits(), qnan_negative),
        (lhs_both_a, rhs_both_a),
        (lhs_both_b, rhs_both_b),
        (f64::INFINITY.to_bits(), f64::NEG_INFINITY.to_bits()),
        (f64::NEG_INFINITY.to_bits(), f64::INFINITY.to_bits()),
        ((-0.0_f64).to_bits(), qnan_negative),
        (qnan_negative, (-0.0_f64).to_bits()),
        (2.0_f64.to_bits(), (-3.0_f64).to_bits()),
        ((-3.0_f64).to_bits(), 2.0_f64.to_bits()),
    ];
    for (state, (lhs, rhs)) in states.iter_mut().zip(cases.into_iter().cycle()) {
        state[2] = lhs as u32;
        state[3] = (lhs >> 32) as u32;
        state[4] = rhs as u32;
        state[5] = (rhs >> 32) as u32;
    }
}

fn is_dmnmx_source_six(raw: u64) -> bool {
    matches!(
        raw,
        0x4c50_0380_0047_0208
            | 0x4c50_0780_0047_0208
            | 0x4c53_0784_0087_0208
            | 0x3850_03bf_f007_0208
            | 0x3950_07c0_0007_0208
            | 0x3850_63bf_e007_0208
    )
}

const DMNMX_IMMEDIATE_CC_BASES: [u64; 7] = [
    0x3850_8000_0007_0000,
    0x3850_805d_3db7_5ce2,
    0x3850_822e_ade7_8146,
    0x3850_82e3_d857_c6ee,
    0x3850_83b0_d857_31fe,
    0x3850_885c_2af7_8254,
    0x3850_8a5a_3f57_c9ab,
];
const DMNMX_IMMEDIATE_CC_VARIABLE_BITS: [u32; 8] = [19, 42, 45, 46, 48, 49, 50, 56];

fn dmnmx_immediate_cc_words() -> impl Iterator<Item = u64> {
    DMNMX_IMMEDIATE_CC_BASES.into_iter().flat_map(|base| {
        (0_u16..256).map(move |selector| {
            DMNMX_IMMEDIATE_CC_VARIABLE_BITS
                .iter()
                .enumerate()
                .fold(base, |raw, (index, bit)| {
                    raw | u64::from(selector >> index & 1) << bit
                })
        })
    })
}

fn is_dmnmx_immediate_cc(raw: u64) -> bool {
    const MASK: u64 = 0xfef8_9bff_fff7_ffff;
    DMNMX_IMMEDIATE_CC_BASES.contains(&(raw & MASK))
}

fn prepare_dmnmx_immediate_cases(raw: u64, states: &mut [[u32; WORDS]]) {
    let lhs = field(raw, 8, 8) as usize;
    let values = [
        1.0_f64.to_bits(),
        (-1.0_f64).to_bits(),
        2.0_f64.to_bits(),
        (-2.0_f64).to_bits(),
        f64::MIN_POSITIVE.to_bits(),
        (-f64::MIN_POSITIVE).to_bits(),
        (f64::MAX / 2.0).to_bits(),
        (-f64::MAX / 2.0).to_bits(),
        f64::INFINITY.to_bits(),
        f64::NEG_INFINITY.to_bits(),
    ];
    for (case, state) in states.iter_mut().enumerate() {
        let value = values[case % values.len()];
        state[lhs] = value as u32;
        state[lhs | 1] = (value >> 32) as u32;
        if lhs & 1 == 1 {
            state[lhs] = (value >> 32) as u32;
        }
    }
}

fn dmnmx_immediate_case_is_defined(raw: u64, state: &[u32; WORDS]) -> bool {
    let guard = field(raw, 16, 3) as usize;
    if !((state[256 + guard] != 0) ^ (field(raw, 19, 1) != 0)) {
        return true;
    }
    let modified = |bits: u64, absolute: u32, negate: u32| {
        let bits = if field(raw, absolute, 1) != 0 {
            bits & !(1_u64 << 63)
        } else {
            bits
        };
        bits ^ (u64::from(field(raw, negate, 1)) << 63)
    };
    let lhs_register = field(raw, 8, 8) as usize;
    let lhs = u64::from(state[lhs_register]) | u64::from(state[lhs_register | 1]) << 32;
    let lhs = modified(lhs, 46, 48);
    let rhs = u64::from(field(raw, 20, 19) << 12 | field(raw, 56, 1) << 31) << 32;
    let rhs = modified(rhs, 49, 45);
    let (lhs_value, rhs_value) = (f64::from_bits(lhs), f64::from_bits(rhs));
    if lhs_value.is_nan() || rhs_value.is_nan() {
        return false;
    }
    let predicate = field(raw, 39, 3) as usize;
    let minimum = (state[256 + predicate] != 0) ^ (field(raw, 42, 1) != 0);
    let result = if minimum {
        if rhs_value < lhs_value {
            rhs
        } else {
            lhs
        }
    } else if lhs_value < rhs_value {
        rhs
    } else {
        lhs
    };
    // GLSL.std.450 FMin/FMax does not require NaN choice or signed-zero
    // preservation. Keep the RADV comparison exact by excluding those
    // results instead of fitting the independent oracle to one driver.
    result & 0x7fff_ffff_ffff_ffff != 0
}

fn prepare_dmnmx_source_cases(raw: u64, states: &mut [[u32; WORDS]]) {
    // The right operand is the word's fixed binary64 value (the cbuf pair or
    // the D20 immediate) after its modifier. The R2 targets sit on, beside
    // and opposite that value, plus signed zeros, infinities and NaNs; each
    // is pre-negated when the word negates R2.
    let constant = if raw >> 60 == 3 {
        u64::from(field(raw, 20, 19) << 12 | field(raw, 56, 1) << 31) << 32
    } else {
        let (bank, word) = (field(raw, 34, 5), field(raw, 20, 14));
        u64::from(compute::cbuf(bank, word)) | u64::from(compute::cbuf(bank, word + 1)) << 32
    };
    let constant = if field(raw, 49, 1) != 0 {
        constant & !(1_u64 << 63)
    } else {
        constant
    };
    let rhs = if field(raw, 45, 1) != 0 {
        constant ^ 1_u64 << 63
    } else {
        constant
    };
    let targets = [
        rhs,
        rhs ^ 1_u64 << 63,
        rhs + 1,
        rhs - 1,
        0.0_f64.to_bits(),
        (-0.0_f64).to_bits(),
        0x7ff8_0000_0000_0042,
        0xfff8_0000_0000_00a5,
        f64::INFINITY.to_bits(),
        f64::NEG_INFINITY.to_bits(),
    ];
    let negate = field(raw, 48, 1) != 0;
    for (state, target) in states.iter_mut().zip(targets.into_iter().cycle()) {
        let lhs = if negate { target ^ 1_u64 << 63 } else { target };
        state[2] = lhs as u32;
        state[3] = (lhs >> 32) as u32;
    }
}

fn prepare_dfma_cases(raw: u64, states: &mut [[u32; WORDS]]) {
    // Finite triples include cases where a separately rounded multiply plus
    // add differs from the required single-rounding fused result.
    let delta = 2.0_f64.powi(-27);
    let values = [
        (0.0, 0.0, 0.0),
        (-0.0, 2.0, 0.0),
        (1.0, 2.0, 3.0),
        (-1.0, 2.0, 3.0),
        (1.5, -0.5, 0.25),
        (1.0 + delta, 1.0 - delta, -1.0),
        (2.0_f64.powi(500), 2.0_f64.powi(-500), -1.0),
        (f64::MIN_POSITIVE, 2.0, -f64::MIN_POSITIVE),
        (f64::MAX / 4.0, 2.0, -f64::MAX / 4.0),
        (-3.0, -4.0, -12.0),
        (0.5, 0.5, -0.25),
        (-0.5, 0.5, 0.25),
    ];
    for (case, state) in states.iter_mut().enumerate() {
        let (lhs, multiplier, addend) = values[case % values.len()];
        for (register, value) in [
            (field(raw, 8, 8) as usize, lhs),
            (field(raw, 20, 8) as usize, multiplier),
            (field(raw, 39, 8) as usize, addend),
        ] {
            let bits = value.to_bits();
            state[register] = bits as u32;
            state[register + 1] = (bits >> 32) as u32;
        }
    }
}

fn is_dfma_cc(raw: u64) -> bool {
    matches!(
        raw,
        0x5b70_8000_0007_0000
            | 0x5370_8000_0007_0000
            | 0x4b70_8000_0007_0000
            | 0x3670_8000_0007_0000
            | 0x3673_debb_4e77_4fc0
    )
}

fn prepare_dfma_cc_cases(raw: u64, states: &mut [[u32; WORDS]]) {
    let values = [
        0.0_f64,
        -0.0,
        -2.0,
        2.0,
        f64::NAN,
        3.0,
        -3.0,
        f64::MIN_POSITIVE,
        f64::MAX / 4.0,
        1.0 + 2.0_f64.powi(-27),
    ];
    let pair = |state: &mut [u32; WORDS], register: usize, value: f64| {
        let bits = value.to_bits();
        state[register] = bits as u32;
        state[register | 1] = (bits >> 32) as u32;
    };
    for (case, state) in states.iter_mut().enumerate() {
        let value = values[case % values.len()];
        let lhs = field(raw, 8, 8) as usize;
        pair(state, lhs, value);
        match ((raw >> 48) as u16 & 0xfff0, (raw >> 48) as u16 & 0xfef0) {
            (0x5b70, _) => {
                // The exact base form aliases all three sources at R0:R1.
                pair(state, field(raw, 20, 8) as usize, value);
                pair(state, field(raw, 39, 8) as usize, value);
            }
            (0x4b70, _) => pair(state, field(raw, 39, 8) as usize, value),
            (0x5370, _) => pair(state, field(raw, 39, 8) as usize, value),
            (_, 0x3670) => {
                let immediate =
                    f64::from_bits(u64::from(field(raw, 20, 19) | field(raw, 56, 1) << 19) << 44);
                let addend = if case % values.len() == 9 {
                    -(value * immediate)
                } else {
                    value
                };
                pair(state, field(raw, 39, 8) as usize, addend);
            }
            _ => unreachable!(),
        }
    }
}

fn is_double_rn_cc(raw: u64) -> bool {
    matches!(
        raw,
        0x4c76_fc03_2b36_9161
            | 0x3872_9a45_25e5_1fed
            | 0x3971_d03d_ef95_759d
            | 0x3976_fc48_5c55_51ee
            | 0x5c86_da71_bf40_b4f2
            | 0x3986_c87c_3b2e_5d0d
            | 0x5c80_8000_0007_0000
            | 0x5c81_8000_0007_0000
            | 0x5c80_c82a_d4c7_25a4
            | 0x5c81_c82a_d4c7_25a4
            | 0x5c80_dc03_b1f7_0950
    )
}

fn prepare_double_rn_cc_cases(raw: u64, states: &mut [[u32; WORDS]]) {
    let values = [
        0.0_f64,
        -0.0,
        -2.0,
        2.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::MIN_POSITIVE,
        f64::MAX / 4.0,
        1.0,
        -1.0,
        3.0,
    ];
    let pair = |state: &mut [u32; WORDS], register: usize, value: f64| {
        let bits = value.to_bits();
        state[register] = bits as u32;
        state[register | 1] = (bits >> 32) as u32;
    };
    for (case, state) in states.iter_mut().enumerate() {
        pair(
            state,
            field(raw, 8, 8) as usize,
            values[case % values.len()],
        );
        if raw >> 60 == 5 {
            pair(
                state,
                field(raw, 20, 8) as usize,
                values[(case + 3) % values.len()],
            );
        }
    }
}

fn is_dfma_source_six(raw: u64) -> bool {
    matches!(
        raw,
        0x4b70_0200_0047_0206
            | 0x4b72_0200_0047_0206
            | 0x5370_0200_0047_0206
            | 0x5371_0200_0047_0206
            | 0x3670_023f_f007_0206
            | 0x3772_0240_0007_0206
    )
}

fn prepare_dfma_source_cases(raw: u64, states: &mut [[u32; WORDS]]) {
    // R2 and R4 pairs around the word's fixed binary64 operand k (the
    // c[0x0][0x10] pair or the D20 immediate). The (1 +/- delta) rows cancel
    // the rounded product against k or the addend, so a separately rounded
    // multiply plus add differs from the single-rounding fused result.
    let constant = if raw >> 60 == 3 {
        f64::from_bits(u64::from(field(raw, 20, 19) | field(raw, 56, 1) << 19) << 44)
    } else {
        let (bank, word) = (field(raw, 34, 5), field(raw, 20, 14));
        f64::from_bits(
            u64::from(compute::cbuf(bank, word)) | u64::from(compute::cbuf(bank, word + 1)) << 32,
        )
    };
    let delta = 2.0_f64.powi(-27);
    let values = if (raw >> 48) & 0xfff0 == 0x5370 {
        [
            (0.0, 0.0),
            (-0.0, 0.0),
            (1.0, 2.0),
            (-3.0, 0.5),
            (1.0 + delta, -constant * (1.0 - delta)),
            (1.0 - delta, constant * (1.0 + delta)),
            (2.0, -constant / 2.0),
            (f64::INFINITY, 1.0),
            (0.5, -constant),
            (-1.5, 0.25),
        ]
    } else {
        [
            (0.0, 0.0),
            (-0.0, 0.0),
            (1.0, 2.0),
            (-1.5, 0.25),
            (1.0 + delta, -((1.0 + delta) * constant)),
            (1.0 - delta, (1.0 - delta) * constant),
            (3.0, -(3.0 * constant)),
            (f64::INFINITY, 1.0),
            (-2.0, 1.0e300),
            (1.0 + delta, -(1.0 - delta)),
        ]
    };
    for (state, (lhs, other)) in states.iter_mut().zip(values.into_iter().cycle()) {
        for (register, value) in [(2, lhs), (4, other)] {
            let bits = f64::to_bits(value);
            state[register] = bits as u32;
            state[register + 1] = (bits >> 32) as u32;
        }
    }
}

fn prepare_f2i64_cases(raw: u64, states: &mut [[u32; WORDS]]) {
    // Deliberate binary64 cases. The triples at each clamp boundary are the
    // immediately adjacent representable values below, at, and above it.
    let (signed_min, signed_max, unsigned_max) = if field(raw, 8, 2) == 1 {
        (
            f64::from(i16::MIN).to_bits(),
            f64::from(i16::MAX).to_bits(),
            f64::from(u16::MAX).to_bits(),
        )
    } else {
        (
            f64::from(i32::MIN).to_bits(),
            f64::from(i32::MAX).to_bits(),
            f64::from(u32::MAX).to_bits(),
        )
    };
    let values = [
        (-0.5_f64).to_bits(),
        0.5_f64.to_bits(),
        (-1.5_f64).to_bits(),
        1.5_f64.to_bits(),
        (-2.5_f64).to_bits(),
        2.5_f64.to_bits(),
        f64::NEG_INFINITY.to_bits(),
        f64::INFINITY.to_bits(),
        0x7ff8_0000_0000_0042, // positive quiet NaN
        0xfff8_0000_0000_0042, // negative quiet NaN
        signed_min + 1,        // next below signed minimum
        signed_min,
        signed_min - 1, // next above signed minimum
        signed_max - 1, // next below signed maximum
        signed_max,
        signed_max + 1,        // next above signed maximum
        0x8000_0000_0000_0001, // next below zero
        0,
        1, // next above zero
        unsigned_max - 1,
        unsigned_max,
        unsigned_max + 1,
    ];
    for (state, value) in states.iter_mut().zip(values) {
        state[2] = value as u32;
        state[3] = (value >> 32) as u32;
    }
}
fn bytes(words: impl IntoIterator<Item = u32>) -> Vec<u8> {
    words.into_iter().flat_map(u32::to_le_bytes).collect()
}
fn validate(words: &[u32], path: &Path) {
    let result = spirv_validation::validate(words);
    if !result.status.success() {
        fs::write(path, bytes(words.iter().copied())).unwrap();
    }
    assert!(
        result.status.success(),
        "{}: {}",
        path.display(),
        String::from_utf8_lossy(&result.stderr)
    );
}
fn work() -> PathBuf {
    let dir = std::env::var_os("SHADOWBOX_GPU_REPORT_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("CARGO_TARGET_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(std::env::temp_dir)
                .join("gpu-exact")
        });
    fs::create_dir_all(&dir).unwrap();
    dir
}

// MUFU uses a GLSL equivalent because transcendental approximation is a host
// operation in Ryujinx, not a bit-exact Rust libm promise.
// Encodings: Mesa sm50.rs:749-768, predicate gm107.c:408-411.
// Semantics: Ryujinx InstEmitMultifunction.cs:32-83; Spirv/Instructions.cs:
// 491-494,531-534,901-904,1039-1042,1120-1128.
fn glsl_reference(
    gpu: &Compute,
    raw: u64,
    cases: &[[u32; WORDS]],
    dir: &Path,
    artifacts: &mut BTreeMap<PathBuf, Vec<u8>>,
    dynamic: bool,
    family: &str,
) -> (Vec<Expected>, Vec<u8>) {
    let d = field(raw, 0, 8) as usize;
    let a = field(raw, 8, 8);
    let p = field(raw, 16, 3) as usize;
    let inverted = field(raw, 19, 1) != 0;
    let o = (raw >> 48) as u16;
    let minmax = matches!(o & 0xfffc, 0x5c60 | 0x4c60) || o & 0xfefc == 0x3860;
    let dsetp = matches!(o & 0xfff0, 0x5b80 | 0x4b80) || o & 0xfef0 == 0x3680;
    let f2i64 = is_f2i64(raw);
    let body = if texture_reference::is_texture(raw)
        || surface_reference::is_surface(raw)
        || lane_reference::is_system(raw)
    {
        String::new()
    } else if f2i64 {
        let rounded = match field(raw, 39, 2) {
            0 => "roundEven(x)",
            1 => "floor(x)",
            2 => "ceil(x)",
            _ => "trunc(x)",
        };
        let converted = match (field(raw, 8, 2), field(raw, 12, 1)) {
            (1, 1) => "uint(int(clamp(rounded,-32768.0lf,32767.0lf)))",
            (1, 0) => "uint(clamp(rounded,0.0lf,65535.0lf))",
            (_, 1) => "uint(int(clamp(rounded,-2147483648.0lf,2147483647.0lf)))",
            (_, 0) => "uint(clamp(rounded,0.0lf,4294967295.0lf))",
            _ => unreachable!(),
        };
        format!(
            "double x=packDouble2x32(uvec2(input_bits[base+2u],input_bits[base+3u]));
             double rounded={rounded};
             uint result=isnan(x) ? 0u : {converted};"
        )
    } else if dsetp {
        let (b_low, b_high) = match raw >> 60 {
            4 => {
                let bank = field(raw, 34, 5);
                let word = field(raw, 20, 14);
                (
                    format!("{}u", compute::cbuf(bank, word)),
                    format!("{}u", compute::cbuf(bank, word + 1)),
                )
            }
            3 => (
                "0u".to_owned(),
                format!("{}u", (field(raw, 20, 19) | field(raw, 56, 1) << 19) << 12),
            ),
            _ => {
                let b = field(raw, 20, 8);
                if b == 255 {
                    ("0u".to_owned(), "0u".to_owned())
                } else {
                    (
                        format!("input_bits[base+{b}u]"),
                        format!("input_bits[base+{}u]", b + 1),
                    )
                }
            }
        };
        let comparison = match field(raw, 48, 4) {
            0 => "false",
            1 => "x<y",
            2 => "x==y",
            3 => "x<=y",
            4 => "x>y",
            5 => "!isnan(x)&&!isnan(y)&&x!=y",
            6 => "x>=y",
            7 => "!isnan(x)&&!isnan(y)",
            8 => "isnan(x)||isnan(y)",
            9 => "isnan(x)||isnan(y)||x<y",
            10 => "isnan(x)||isnan(y)||x==y",
            11 => "isnan(x)||isnan(y)||x<=y",
            12 => "isnan(x)||isnan(y)||x>y",
            13 => "isnan(x)||isnan(y)||x!=y",
            14 => "isnan(x)||isnan(y)||x>=y",
            _ => "true",
        };
        let combine = match field(raw, 45, 2) {
            0 => "compared&&accumulator",
            1 => "compared||accumulator",
            _ => "compared!=accumulator",
        };
        format!(
            "uint alo=input_bits[base+{a}u]; uint ahi=input_bits[base+{}u];
             uint blo={b_low}; uint bhi={b_high};
             ahi=(ahi & {aa}u) ^ {an}u; bhi=(bhi & {ba}u) ^ {bn}u;
             double x=packDouble2x32(uvec2(alo,ahi));
             double y=packDouble2x32(uvec2(blo,bhi));
             bool compared={comparison};
             bool accumulator=(input_bits[base+{}u]!=0u)!={};
             bool first={combine}; compared=!compared; bool second={combine};",
            a + 1,
            256 + field(raw, 39, 3),
            field(raw, 42, 1) != 0,
            aa = if field(raw, 7, 1) != 0 {
                0x7fff_ffff_u32
            } else {
                u32::MAX
            },
            an = field(raw, 43, 1) << 31,
            ba = if field(raw, 44, 1) != 0 {
                0x7fff_ffff_u32
            } else {
                u32::MAX
            },
            bn = field(raw, 6, 1) << 31,
        )
    } else if minmax {
        // sm50.rs:605-637, InstEmitFloatMinMax.cs:78-108.
        // Ryujinx Spirv/Instructions.cs:954-956,970-972 uses FMin/FMax.
        // Use the same host operators, including implementation-dependent
        // NaN/zero selection, rather than assuming NMin CPU semantics.
        let b = match raw >> 60 {
            4 => format!("{}u", compute::cbuf(field(raw, 34, 5), field(raw, 20, 14))),
            3 => format!("{}u", (field(raw, 20, 19) | field(raw, 56, 1) << 19) << 12),
            _ => format!("input_bits[base+{}u]", field(raw, 20, 8)),
        };
        let min_pred = 256 + field(raw, 39, 3);
        format!(
            "uint ab=input_bits[base+{a}u]; uint bb={b};
             ab=(ab & {aa}u) ^ {an}u; bb=(bb & {ba}u) ^ {bn}u;
             {ftz}
             float x=uintBitsToFloat(ab); float y=uintBitsToFloat(bb);
             precise float value=(input_bits[base+{min_pred}u]!=0u)!={pinv} ? min(x,y) : max(x,y);
             uint result=floatBitsToUint(value);",
            aa = if field(raw, 46, 1) != 0 {
                0x7fff_ffff_u32
            } else {
                u32::MAX
            },
            an = field(raw, 48, 1) << 31,
            ba = if field(raw, 49, 1) != 0 {
                0x7fff_ffff_u32
            } else {
                u32::MAX
            },
            bn = field(raw, 45, 1) << 31,
            pinv = field(raw, 42, 1) != 0,
            ftz = if field(raw, 44, 1) != 0 {
                "if((ab & 0x7fffffffu)<0x00800000u) ab &= 0x80000000u;
                 if((bb & 0x7fffffffu)<0x00800000u) bb &= 0x80000000u;"
            } else {
                ""
            }
        )
    } else {
        let selector = field(raw, 20, 4);
        let operation = match selector {
            0 => "cos(x)",
            1 => "sin(x)",
            2 => "exp2(x)",
            3 => "log2(x)",
            4 => "1.0/x",
            5 => "inversesqrt(x)",
            6 | 7 => "x",
            8 => "sqrt(x)",
            _ => unreachable!(),
        };
        let calculation = if matches!(selector, 6 | 7) {
            format!(
                "double x64=packDouble2x32(uvec2(0u,bits));
                 double value64={};
                 uint result=unpackDouble2x32(value64).y;
                 float value=uintBitsToFloat(result);",
                if selector == 6 {
                    "1.0lf/x64"
                } else {
                    "inversesqrt(x64)"
                }
            )
        } else {
            format!(
                "float x=uintBitsToFloat(bits); precise float value={operation};
                 uint result=floatBitsToUint(value);"
            )
        };
        format!(
            "uint bits=input_bits[base+{a}u];
             bits=bits & {abs_mask}u; bits=bits ^ {neg_mask}u;
             {calculation}
             {saturate}
             result=floatBitsToUint(value);",
            // Mesa nir_opcodes.py:265; util/u_math.h:343-371 defines
            // min/max-number saturation, including NaN and signed zero.
            saturate = if field(raw, 50, 1) != 0 {
                "value=(isnan(value) || value<=0.0) ? 0.0 : min(value,1.0);"
            } else {
                ""
            },
            abs_mask = if field(raw, 46, 1) != 0 {
                0x7fff_ffff_u32
            } else {
                u32::MAX
            },
            neg_mask = field(raw, 48, 1) << 31
        )
    };
    let write_result = if dsetp {
        let first = field(raw, 3, 3);
        let second = field(raw, 0, 3);
        format!(
            "if((input_bits[base+{predicate}u]!=0u)!={inverted}){{
               if({first}u!=7u) output_bits[base+{}u]=uint(first);
               if({second}u!=7u) output_bits[base+{}u]=uint(second);
             }}",
            256 + first,
            256 + second,
            predicate = 256 + p,
        )
    } else {
        format!(
            "if((input_bits[base+{}u]!=0u)!={inverted} && {d}u!=255u)
               output_bits[base+{d}u]=result;",
            256 + p
        )
    };
    let source = format!(
        "#version 450
layout(local_size_x=1) in;
layout(set=3,binding=0,std430) readonly buffer In {{uint input_bits[];}};
layout(set=3,binding=1,std430) writeonly buffer Out {{uint output_bits[];}};
void main(){{
 uint base=gl_GlobalInvocationID.x*{WORDS}u;
 for(uint i=0u;i<{WORDS}u;i++) output_bits[base+i]=input_bits[base+i];
 {body}
 {write_result}
}}"
    );
    let source_path = dir.join(format!("{raw:016x}.reference.comp"));
    let spirv_path = source_path.with_extension("spv");
    let source = if lane_reference::is_system(raw) {
        lane_reference::system_source(raw).unwrap()
    } else if surface_reference::is_surface(raw) {
        surface_reference::source(raw).unwrap()
    } else if texture_reference::is_texture(raw) {
        if dynamic {
            texture_reference::source_dynamic(raw)
        } else {
            texture_reference::source(raw)
        }
        .unwrap()
    } else {
        source
    };
    // glslc --help: "-" is stdin and "-o -" is stdout. Keep successful
    // compilation in memory; persist the source only when compilation fails.
    fs::write(
        dir.join("current-case.txt"),
        format!("{family}\t{raw:016x}\treference-compile\n"),
    )
    .unwrap();
    let mut child = Command::new("glslc")
        .args([
            "--target-env=vulkan1.2",
            "-fshader-stage=compute",
            "-",
            "-o",
            "-",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(source.as_bytes())
        .unwrap();
    let result = child.wait_with_output().unwrap();
    if !result.status.success() {
        fs::write(&source_path, &source).unwrap();
    }
    artifacts.insert(source_path, source.into_bytes());
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(result.stdout.len().is_multiple_of(4));
    let words = result
        .stdout
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect::<Vec<_>>();
    artifacts.insert(spirv_path.clone(), bytes(words.iter().copied()));
    validate(&words, &spirv_path);
    fs::write(
        dir.join("current-case.txt"),
        format!("{family}\t{raw:016x}\treference-pipeline-and-dispatch\n"),
    )
    .unwrap();
    let (states, image) = gpu.run_images(&words, cases, false);
    let expected = states
        .into_iter()
        .zip(cases)
        .map(|(words, state)| {
            let mut nan = [0; WORDS];
            if !texture_reference::is_texture(raw)
                && !lane_reference::is_system(raw)
                && !surface_reference::is_surface(raw)
                && !dsetp
                && (state[256 + p] != 0) ^ inverted
                && d != 255
            {
                nan[d] = 1;
            }
            Expected { words, nan }
        })
        .collect();
    (expected, image)
}

struct Report {
    dir: PathBuf,
    gpu: Compute,
    rows: String,
    defects: String,
    artifacts: BTreeMap<PathBuf, Vec<u8>>,
    attempted: usize,
    executed: usize,
    mismatches: usize,
    gaps: BTreeSet<String>,
}
impl Report {
    fn new(label: &str) -> Self {
        let filter = std::env::var("SHADOWBOX_GPU_FILTER").unwrap_or_default();
        let suffix = filter
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect::<String>();
        let dir = work().join(if suffix.is_empty() {
            label.to_owned()
        } else {
            format!("{label}-{suffix}")
        });
        fs::create_dir_all(&dir).unwrap();
        let completion = dir.join("complete.txt");
        if completion.exists() {
            fs::remove_file(completion).unwrap();
        }
        fs::write(
            dir.join("coverage.tsv"),
            "family\traw\tstatus\tcases\texcluded\tdetail\n",
        )
        .unwrap();
        fs::write(
            dir.join("defects.tsv"),
            "family\traw\tvariant\tcase\tslot\texpected\tactual\tpolicy\n",
        )
        .unwrap();
        fs::write(dir.join("timings.tsv"), "family\traw\tseconds\n").unwrap();
        fs::write(dir.join("current-case.txt"), "resource-initialization\n").unwrap();
        let gpu = Compute::new();
        let fixtures = FAMILIES
            .iter()
            .map(|f| format!("{}\t{}\n", f.name, f.required))
            .collect::<String>();
        fs::write(dir.join("fixture-requirements.tsv"), fixtures).unwrap();
        fs::write(dir.join("device.txt"), &gpu.description).unwrap();
        Self {
            dir,
            gpu,
            rows: "family\traw\tstatus\tcases\texcluded\tdetail\n".into(),
            defects: "family\traw\tvariant\tcase\tslot\texpected\tactual\tpolicy\n".into(),
            artifacts: BTreeMap::new(),
            attempted: 0,
            executed: 0,
            mismatches: 0,
            gaps: BTreeSet::new(),
        }
    }
    fn phase(&self, family: &str, raw: u64, phase: &str) {
        fs::write(
            self.dir.join("current-case.txt"),
            format!("{family}\t{raw:016x}\t{phase}\n"),
        )
        .unwrap();
    }
    fn check(&mut self, family: &str, raw: u64, required: bool) {
        let started = std::time::Instant::now();
        self.phase(family, raw, "translate");
        let dynamic = family.contains("_dynamic_");
        let layer_edge = family.contains("_layer_edge_");
        self.attempted += 1;
        if self.attempted.is_multiple_of(16) {
            fs::write(self.dir.join("coverage.tsv"), &self.rows).unwrap();
            fs::write(self.dir.join("defects.tsv"), &self.defects).unwrap();
            eprintln!(
                "{}: {} encodings checked, {} differing words",
                self.dir.display(),
                self.attempted - 1,
                self.mismatches
            );
        }
        let code = if family == "TEX_SAMPLE_CBUF_ARRAY" {
            texture_array_program(raw)
        } else if family == "BRA_LOOP_EXACT" {
            control_reference::loop_program(raw)
        } else if memory_reference::is_barrier(raw) {
            memory_reference::barrier_program(raw)
        } else if control_reference::is_control(raw) {
            control_reference::program(raw)
        } else if texture_reference::is_texture(raw) || surface_reference::is_surface(raw) {
            texture_program(raw, dynamic)
        } else {
            program(raw)
        };
        let decoded = decode_stream(shadowbox::QuietRun, &code);
        let header = headers::parse(&[0; 32]).unwrap();
        let emitted = if family == "TEX_SAMPLE_CBUF_ARRAY" {
            let array = shadowbox::interface::TextureArray {
                binding: 2,
                handle: 0,
                length: 64,
            };
            shadowbox::emitter::emit_unoptimized_with_graphics_state(
                &decoded,
                Some(&header),
                &shadowbox::interface::GraphicsState {
                    texture_arrays: &[array],
                    ..Default::default()
                },
            )
        } else if family == "TXQ_BOUND_DIMENSION_WITH_SHAPE" {
            // TXQ needs driver sampler metadata, Ryujinx InstEmitTexture.cs:
            // 1116-1119. The fixture owns a 3D image with two levels.
            let binding = shadowbox::interface::BoundTexture {
                binding: field(raw, 36, 13) as u16,
                dimension: shadowbox::interface::TextureDimension::D3,
                depth_compare: false,
                multisample: false,
            };
            shadowbox::emitter::emit_unoptimized_with_graphics_state(
                &decoded,
                Some(&header),
                &shadowbox::interface::GraphicsState {
                    texture_bindings: &[binding],
                    ..Default::default()
                },
            )
        } else {
            emit_unoptimized_with_options(
                &decoded,
                Some(&header),
                shadowbox::TranslationOptions {
                    float64: if self.gpu.supports_shader_float64 {
                        shadowbox::HostFloat64::Supported
                    } else {
                        shadowbox::HostFloat64::Unsupported
                    },
                    ..Default::default()
                },
            )
        };
        let mut module = match emitted {
            Ok(module) => module,
            Err(error) => {
                if required {
                    self.gaps
                        .insert(format!("{family}: required representative is Unknown"));
                }
                self.rows
                    .push_str(&format!("{family}\t{raw:016x}\tUnknown\t0\t0\t{error}\n"));
                return;
            }
        };
        let mut cases = Vec::new();
        let mut expected = Vec::new();
        let o = (raw >> 48) as u16;
        let glsl = texture_reference::is_texture(raw)
            || lane_reference::is_system(raw)
            || surface_reference::is_surface(raw)
            || (raw >> 48) & 0xfff8 == 0x5080
            || is_f2i64(raw)
            || matches!(o & 0xfff0, 0x5b80 | 0x4b80)
            || o & 0xfef0 == 0x3680
            || matches!(o & 0xfffc, 0x5c60 | 0x4c60)
            || o & 0xfefc == 0x3860;
        let texture = texture_reference::is_texture(raw);
        let surface = surface_reference::is_surface(raw);
        let lane = lane_reference::is_lane(raw);
        let memory = memory_reference::is_memory(raw);
        let barrier = memory_reference::is_barrier(raw);
        let mut states = inputs(raw);
        if is_dadd_five(raw) || is_dmul_exact(raw) {
            prepare_dadd_cases(&mut states);
        }
        if is_fadd_ftz_rm(raw) {
            prepare_fadd_ftz_rm_cases(raw, &mut states);
        }
        if is_dadd_immediate_rn_cc(raw) {
            prepare_dadd_immediate_rn_cc_cases(raw, &mut states);
        }
        if is_dadd_immediate_rm(raw) {
            prepare_dadd_immediate_rm_cases(raw, &mut states);
        }
        if is_dadd_register_rm(raw) {
            prepare_dadd_register_rm_cases(raw, &mut states);
        }
        if is_dadd_cbuf_rm(raw) {
            prepare_dadd_cbuf_rm_cases(raw, &mut states);
        }
        if is_dadd_cbuf_rp(raw) {
            prepare_dadd_cbuf_rm_cases(raw, &mut states);
        }
        if is_dadd_cbuf_rz(raw) {
            prepare_dadd_cbuf_rm_cases(raw, &mut states);
        }
        if is_dadd_register_rp(raw) {
            prepare_dadd_register_rm_cases(raw, &mut states);
        }
        if is_dadd_register_rz(raw) {
            prepare_dadd_register_rm_cases(raw, &mut states);
        }
        if is_dadd_immediate_rp(raw) {
            prepare_dadd_immediate_rp_cases(raw, &mut states);
        }
        if is_dadd_immediate_rz(raw) {
            prepare_dadd_immediate_rz_cases(raw, &mut states);
        }
        if matches!(
            raw,
            0x5902_0380_0047_0201
                | 0x4901_0380_0047_0201
                | 0x3203_03bf_f007_0201
                | 0x5907_0380_0047_0201
                | 0x5908_0380_0047_0201
                | 0x590d_0380_0047_0201
                | 0x5914_0380_0047_0201
                | 0x5906_2400_0047_0201
                | 0x5909_4080_0047_0201
                | 0x5901_1b80_0047_0201
                | 0x596a_0380_0047_0201
                | 0x491e_0b80_0047_0201
                | 0x5b84_2400_0047_0201
                | 0x5b89_4080_0047_0201
                | 0x5b81_1b80_0047_0201
                | 0x5b8d_1b80_0047_02c1
                | 0x4b8e_2d00_0047_0201
                | 0x368c_4cbf_f007_0281
        ) {
            prepare_dset_cases(&mut states);
        }
        if matches!(
            raw,
            0x36b4_03bf_8007_0247
                | 0x37b1_13c0_0007_0207
                | 0x37b6_13bf_0007_0247
                | 0x36b2_24bf_8007_0247
                | 0x36bd_0bbf_8007_02c7
                | 0x37b3_5140_0007_0287
        ) {
            prepare_fsetp_immediate_cases(&mut states);
        }
        if matches!(
            raw,
            0x5bb4_0380_0047_0201
                | 0x5bb9_2400_0047_020a
                | 0x5bb3_5900_0047_0298
                | 0x4bb5_0380_0047_0255
                | 0x36b6_04bf_8007_0203
                | 0x37bd_21c0_0007_02a6
        ) {
            prepare_fsetp_second_destination_cases(&mut states);
        }
        if matches!(
            raw,
            0x5b62_0380_0047_0201
                | 0x5b6d_2400_0047_020a
                | 0x5b6b_4900_0047_0218
                | 0x4b68_0380_0047_0215
                | 0x3767_04ff_ffb7_0203
                | 0x3664_2180_0077_023e
        ) {
            prepare_isetp_second_destination_cases(&mut states);
        }
        if matches!(
            raw,
            0x5c28_0100_0047_0203
                | 0x5c29_0180_0047_0203
                | 0x4c28_0180_0047_0203
                | 0x4c29_0100_0047_0203
                | 0x3828_0100_0077_0203
                | 0x3929_017f_fff7_0203
        ) {
            prepare_shr_brev_cases(&mut states);
        }
        if (raw >> 48) & 0xfff8 == 0x5c50 {
            prepare_dmnmx_cases(raw, &mut states);
        }
        if is_dmnmx_source_six(raw) {
            prepare_dmnmx_source_cases(raw, &mut states);
        }
        if is_dmnmx_immediate_cc(raw) {
            prepare_dmnmx_immediate_cases(raw, &mut states);
        }
        if is_dfma_six(raw) {
            prepare_dfma_cases(raw, &mut states);
        }
        if is_dfma_source_six(raw) {
            prepare_dfma_source_cases(raw, &mut states);
        }
        if is_dfma_cc(raw) {
            prepare_dfma_cc_cases(raw, &mut states);
        }
        if is_double_rn_cc(raw) {
            prepare_double_rn_cc_cases(raw, &mut states);
        }
        if is_f2i64(raw) || is_f2i64_source_modifier(raw) {
            prepare_f2i64_cases(raw, &mut states);
        }
        if is_fmul_fmz(raw) {
            prepare_fmul_fmz_cases(&mut states);
        }
        if family == "BRA_LOOP_EXACT" {
            for (case, state) in states.iter_mut().enumerate() {
                state[240] = (-(case as i32 % 8 + 1)) as u32;
            }
        }
        if family == "BRX_CC_LT_B8E0511D302035A9D" {
            for (case, state) in states.iter_mut().enumerate() {
                control_reference::prepare_lt(raw, case, state);
            }
        }
        if lane {
            states.truncate(32);
            for (i, state) in states.iter_mut().enumerate() {
                // Keep all source lanes active for defined shuffle results.
                // Alternate uniform true/false instruction guards by encoding.
                if (raw >> 48) & 0xfff8 == 0xef10 {
                    state[256 + field(raw, 16, 3) as usize] = 1;
                    if field(raw, 28, 1) == 0 && field(raw, 20, 8) != 255 {
                        state[field(raw, 20, 8) as usize] = (i % 32) as u32;
                    }
                    if field(raw, 29, 1) == 0 && field(raw, 39, 8) != 255 {
                        state[field(raw, 39, 8) as usize] = 31 | ((i as u32 % 4) << 8);
                    }
                }
            }
            let result = if lane_reference::is_system(raw) {
                lane_reference::system_source(raw).map(|_| Vec::new())
            } else {
                lane_reference::reference(raw, &states)
            };
            match result {
                Ok(results) => {
                    cases = states;
                    expected = results;
                }
                Err(reason) => {
                    self.gaps.insert(family.to_owned());
                    self.rows.push_str(&format!(
                        "{family}\t{raw:016x}\toracle-missing\t0\t32\t{reason}\n"
                    ));
                    return;
                }
            }
        } else {
            for (case, mut state) in states.into_iter().enumerate() {
                let case = if memory_reference::is_flat(raw) || barrier {
                    cases.len()
                } else {
                    case
                };
                if barrier
                    && memory_reference::prepare(
                        memory_reference::barrier_memory(),
                        case,
                        &mut state,
                    )
                    .is_err()
                {
                    continue;
                }
                if memory && memory_reference::prepare(raw, case, &mut state).is_err() {
                    continue;
                }
                if surface {
                    surface_reference::prepare(raw, case, &mut state).unwrap();
                }
                if texture {
                    if let Err(reason) = if layer_edge {
                        texture_reference::prepare_layer_edge(raw, case, &mut state)
                    } else if dynamic {
                        texture_reference::prepare_dynamic(raw, case, &mut state)
                    } else {
                        texture_reference::prepare(raw, case, &mut state)
                    } {
                        self.gaps.insert(family.to_owned());
                        self.rows.push_str(&format!(
                            "{family}\t{raw:016x}\toracle-missing\t0\t0\t{reason}\n"
                        ));
                        return;
                    }
                    if let Err(reason) = texture_reference::source(raw) {
                        self.gaps.insert(family.to_owned());
                        self.rows.push_str(&format!(
                            "{family}\t{raw:016x}\toracle-missing\t0\t0\t{reason}\n"
                        ));
                        return;
                    }
                    if family == "TEX_SAMPLE_CBUF_ARRAY" {
                        // The translated setup writes zero coordinates. Vary
                        // the LDC byte address across in-range and clamped
                        // descriptor indices while the reference samples the
                        // identical image installed in every array element.
                        state[10] = 0.0f32.to_bits();
                        state[11] = 0.0f32.to_bits();
                        let source_index = if case < 32 {
                            case as u32
                        } else {
                            60 + (case as u32 - 32)
                        };
                        state[12] = source_index * 8;
                    }
                }
                if is_dmnmx_immediate_cc(raw) && !dmnmx_immediate_case_is_defined(raw, &state) {
                    continue;
                }
                if glsl {
                    cases.push(state);
                    continue;
                }
                let result = if barrier {
                    memory_reference::barrier_reference(raw, case, &state)
                } else if memory {
                    memory_reference::reference(raw, case, &state)
                } else if control_reference::is_control(raw) {
                    control_reference::reference(&code, &state)
                } else {
                    reference(raw, &state)
                };
                match result {
                    Ok(result) => {
                        cases.push(state);
                        expected.push(result);
                    }
                    Err(reason) => {
                        // Defined-domain exclusions are counted, never compared
                        // against an arbitrary result of undefined SPIR-V.
                        if reason.starts_with("undefined ") {
                            continue;
                        }
                        self.gaps.insert(family.to_owned());
                        self.rows.push_str(&format!(
                            "{family}\t{raw:016x}\toracle-missing\t0\t0\t{reason}\n"
                        ));
                        return;
                    }
                }
            }
        }
        if cases.is_empty() {
            self.gaps.insert(family.to_owned());
            self.rows.push_str(&format!(
                "{family}\t{raw:016x}\tdomain-empty\t0\t48\tall inputs outside defined reference domain\n"
            ));
            return;
        }
        let mut expected_image = Vec::new();
        if glsl {
            self.phase(family, raw, "reference");
            (expected, expected_image) = glsl_reference(
                &self.gpu,
                raw,
                &cases,
                &self.dir,
                &mut self.artifacts,
                dynamic,
                family,
            );
        }
        assert_eq!(
            expected.len(),
            cases.len(),
            "reference state count mismatch"
        );
        compute::observe_group(&mut module, if lane { 32 } else { 1 });
        let original = module.words.clone();
        self.phase(family, raw, "optimize");
        module.optimize();
        let mut defect = false;
        for (variant, words) in [("raw", &original), ("optimized", &module.words)] {
            self.phase(family, raw, &format!("{variant}-validate"));
            validate(words, &self.dir.join("validate.spv"));
            self.phase(family, raw, &format!("{variant}-pipeline-and-dispatch"));
            let cbuf_override = ((raw >> 48) & 0xfff8 == 0x4c00).then(|| {
                let bank = field(raw, 34, 5);
                let word = field(raw, 20, 14);
                (bank, word, compute::bfe_cbuf(bank, word))
            });
            let (actual, image) = if memory_reference::fixed_address(raw) {
                // A fixed RZ address cannot select per-invocation allocations.
                // Submit each state separately at the same guest address,
                // keeping all active loads/stores and every observed word.
                let actual = cases
                    .iter()
                    .flat_map(|state| {
                        self.gpu
                            .run_images_config(
                                words,
                                std::slice::from_ref(state),
                                true,
                                None,
                                memory_reference::guest_base(raw),
                                false,
                            )
                            .0
                    })
                    .collect::<Vec<_>>();
                (actual, Vec::new())
            } else {
                self.gpu.run_images_config(
                    words,
                    &cases,
                    memory_reference::is_flat(raw) || barrier,
                    cbuf_override,
                    compute::GUEST_BASE,
                    false,
                )
            };
            assert_eq!(actual.len(), cases.len(), "GPU state count mismatch");
            assert_eq!(
                image.len(),
                expected_image.len(),
                "storage transport mismatch"
            );
            for (at, (&byte, &expected_byte)) in image.iter().zip(&expected_image).enumerate() {
                if byte != expected_byte {
                    defect = true;
                    self.mismatches += 1;
                    self.defects.push_str(&format!("{family}\t{raw:016x}\t{variant}\t0\t{}\t{expected_byte:08x}\t{byte:08x}\timage-byte\n",WORDS+at));
                }
            }
            if image != expected_image {
                let stem = self.dir.join(format!("{raw:016x}-{variant}"));
                fs::write(stem.with_extension("actual-image.bin"), &image).unwrap();
                fs::write(stem.with_extension("expected-image.bin"), &expected_image).unwrap();
            }
            for (case, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                for (slot, &actual_word) in actual.iter().enumerate() {
                    if normalise(actual_word, expected.nan[slot])
                        != normalise(expected.words[slot], expected.nan[slot])
                    {
                        defect = true;
                        self.mismatches += 1;
                        self.defects.push_str(&format!(
                            "{family}\t{raw:016x}\t{variant}\t{case}\t{slot}\t{:08x}\t{:08x}\t{}\n",
                            expected.words[slot], actual_word, expected.nan[slot]
                        ));
                    }
                }
            }
            if defect {
                let stem = self.dir.join(format!("{raw:016x}-{variant}"));
                self.artifacts
                    .insert(stem.with_extension("spv"), bytes(words.iter().copied()));
                self.artifacts.insert(
                    stem.with_extension("inputs.bin"),
                    bytes(cases.iter().flatten().copied()),
                );
                self.artifacts.insert(
                    stem.with_extension("actual.bin"),
                    bytes(actual.iter().flatten().copied()),
                );
                self.artifacts.insert(
                    stem.with_extension("expected.bin"),
                    bytes(expected.iter().flat_map(|e| e.words)),
                );
            }
        }
        self.phase(family, raw, "compared");
        writeln!(
            fs::OpenOptions::new()
                .append(true)
                .open(self.dir.join("timings.tsv"))
                .unwrap(),
            "{family}\t{raw:016x}\t{:.6}",
            started.elapsed().as_secs_f64()
        )
        .unwrap();
        self.executed += cases.len() * 2;
        self.rows.push_str(&format!(
            "{family}\t{raw:016x}\t{}\t{}\t{}\t{}\n",
            if defect { "DEFECT" } else { "MATCH" },
            cases.len() * 2,
            if lane { 32 } else { 48 } - cases.len(),
            if glsl {
                "GLSL equivalent"
            } else {
                "independent CPU reference"
            }
        ));
    }
    fn finish(self) {
        assert!(
            self.attempted != 0,
            "no fixtures matched the requested filter"
        );
        fs::write(self.dir.join("coverage.tsv"), &self.rows).unwrap();
        fs::write(self.dir.join("defects.tsv"), &self.defects).unwrap();
        eprintln!(
            "{}; {} executions; {} differing words; missing oracles: {:?}",
            self.gpu.description, self.executed, self.mismatches, self.gaps
        );
        eprintln!("Evidence: {}", self.dir.display());
        assert!(
            self.executed != 0,
            "no translated encoding was compared; see coverage.tsv"
        );
        assert_eq!(
            self.mismatches, 0,
            "GPU disagrees with the independent reference; see defects.tsv"
        );
        assert!(
            self.gaps.is_empty(),
            "GPU coverage has missing oracles: {:?}",
            self.gaps
        );
        fs::write(
            self.dir.join("complete.txt"),
            format!("{}\t{}\t0\n", self.attempted, self.executed),
        )
        .unwrap();
    }
}

impl Drop for Report {
    fn drop(&mut self) {
        // Preserve a reviewable checkpoint if validation or dispatch aborts.
        // Completion is reported only by finish(), never by these files alone.
        fs::write(self.dir.join("coverage.tsv"), &self.rows).unwrap();
        fs::write(self.dir.join("defects.tsv"), &self.defects).unwrap();
        // Buffer artifacts during dispatch so shared disk throttling cannot
        // interrupt comparisons. Every reference and differing-word dump is
        // persisted on normal completion and on a Rust panic.
        for (path, data) in &self.artifacts {
            fs::write(path, data).unwrap();
        }
    }
}

#[test]
fn imnmx_fixtures_name_the_instruction_they_execute() {
    let shr = FAMILIES
        .iter()
        .find(|family| family.opcode == 0x5c28 && family.fields == 0)
        .expect("0x5c28 register representative");
    assert_eq!(shr.name, "SHR register 0x5c28");

    let imnmx = FAMILIES
        .iter()
        .find(|family| family.name == "IMNMX ordinary register")
        .expect("ordinary register IMNMX representative");
    assert_eq!(imnmx.opcode, 0x5c20);
    assert_eq!(imnmx.fields, 7 << 39);
    assert!(imnmx.required);
}

#[test]
fn imnmx_reference_distinguishes_ordinary_and_extended_states() {
    let mut state = [0; WORDS];
    state[263] = 1;
    state[2] = 0x8000_0000;
    state[3] = 1;
    let ordinary = 0x5c20_u64 << 48 | 3 << 20 | 7 << 39 | 7 << 16 | 2 << 8 | 1;
    assert_eq!(reference(ordinary, &state).unwrap().words[1], 1);
    assert_eq!(
        reference(ordinary | 1 << 42, &state).unwrap().words[1],
        0x8000_0000
    );
    assert_eq!(
        reference(ordinary | 1 << 48, &state).unwrap().words[1],
        0x8000_0000
    );
    assert_eq!(
        reference(ordinary | 1 << 48 | 1 << 42, &state)
            .unwrap()
            .words[1],
        1
    );

    for mode in 1..=3 {
        let extended = ordinary | mode << 43;
        assert!(matches!(
            shadowbox::decoder::decode_word([extended as u32, (extended >> 32) as u32], 0).kind,
            shadowbox::ir::InstructionKind::Unknown(ops::Unknown { .. })
        ));
        assert!(matches!(
            reference(extended, &state),
            Err("IMNMX extended mode needs a CC oracle")
        ));
    }
}

#[test]
#[ignore = "RADV hardware; directed ordinary IMNMX"]
fn gpu_imnmx_ordinary_exact() {
    let mut report = Report::new("imnmx-ordinary");
    for (name, opcode, selector) in [
        ("IMNMX_U32_MIN", 0x5c20_u64, 7_u64),
        ("IMNMX_U32_MAX", 0x5c20, 15),
        ("IMNMX_S32_MIN", 0x5c21, 7),
        ("IMNMX_S32_MAX", 0x5c21, 15),
    ] {
        // Modes43..44 are zero. selector 7 is PT (min); selector 15 is
        // inverted PT (max). The values are supplied at runtime.
        let raw = opcode << 48 | selector << 39 | 3 << 20 | 7 << 16 | 2 << 8 | 1;
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; six IMNMX runtime-selector translations"]
fn gpu_imnmx_select_six_exact() {
    let mut report = Report::new("imnmx-select-six");
    for (name, raw) in [
        ("IMNMX_U32_REGISTER_P0", 0x5c20_0000_0047_0201),
        ("IMNMX_S32_REGISTER_NOT_P1", 0x5c21_0480_0047_0201),
        ("IMNMX_S32_REGISTER_P3", 0x5c21_0180_0047_0201),
        ("IMNMX_U32_REGISTER_NOT_P6", 0x5c20_0700_0047_0201),
        ("IMNMX_U32_CBUF_NOT_P0", 0x4c20_0400_0047_0201),
        ("IMNMX_S32_NEGATIVE_IMMEDIATE_P4", 0x3921_0200_0047_0201),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; directed XMAD.CC and IMNMX.CC"]
fn gpu_integer_cc_exact() {
    let mut report = Report::new("integer-cc");
    for (name, raw) in [
        (
            "XMAD_CC_REGISTER",
            (0x5b00_u64 << 48) | (1 << 47) | (4 << 39) | (3 << 20) | (7 << 16) | (2 << 8) | 1,
        ),
        (
            "XMAD_CC_IMMEDIATE",
            (0x3600_u64 << 48) | (1 << 47) | (4 << 39) | (3 << 20) | (7 << 16) | (2 << 8) | 1,
        ),
        (
            "XMAD_CC_CBUF_B",
            (0x4e00_u64 << 48) | (1 << 47) | (4 << 39) | (7 << 16) | (2 << 8) | 1,
        ),
        (
            "XMAD_CC_CBUF_C",
            (0x5100_u64 << 48) | (1 << 47) | (4 << 39) | (7 << 16) | (2 << 8) | 1,
        ),
        (
            "IMNMX_CC_U32_MIN",
            (0x5c20_u64 << 48) | (1 << 47) | (7 << 39) | (4 << 20) | (7 << 16) | (2 << 8) | 1,
        ),
        (
            "IMNMX_CC_U32_MAX",
            (0x5c20_u64 << 48) | (1 << 47) | (15 << 39) | (4 << 20) | (7 << 16) | (2 << 8) | 1,
        ),
        (
            "IMNMX_CC_S32_MIN",
            (0x5c21_u64 << 48) | (1 << 47) | (7 << 39) | (4 << 20) | (7 << 16) | (2 << 8) | 1,
        ),
        (
            "IMNMX_CC_S32_MAX",
            (0x5c21_u64 << 48) | (1 << 47) | (15 << 39) | (4 << 20) | (7 << 16) | (2 << 8) | 1,
        ),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; FADD.CC Z/N through CSET observers"]
fn gpu_fadd_cc_exact() {
    let gpu = Compute::new();
    let header = headers::parse(&[0; 32]).unwrap();
    let fadd_cc = (0x5c58_u64 << 48) | (1 << 47) | (3 << 20) | (7 << 16) | (2 << 8) | 1;
    let cases = [
        (1.0_f32, -1.0_f32, true, false),
        (-2.0, 1.0, false, true),
        (1.0, 2.0, false, false),
        (f32::NAN, 1.0, false, false),
    ];
    let states = cases
        .iter()
        .map(|(lhs, rhs, _, _)| {
            let mut state = [0_u32; WORDS];
            state[2] = lhs.to_bits();
            state[3] = rhs.to_bits();
            state
        })
        .collect::<Vec<_>>();
    let mut comparisons = 0;
    // CSET.EQ observes Z; CSET.LT observes N. Both AND PT and write R4.
    for (name, cset, flag) in [
        ("Z", 0x5098_0380_0007_0204_u64, 0_usize),
        ("N", 0x5098_0380_0007_0104_u64, 1_usize),
    ] {
        let code = [0, fadd_cc, cset, 0xe300_u64 << 48 | 7 << 16 | 15]
            .into_iter()
            .flat_map(|word| [word as u32, (word >> 32) as u32])
            .collect::<Vec<_>>();
        let decoded = decode_stream(shadowbox::QuietRun, &code);
        let mut module = emit_unoptimized_with_header(&decoded, Some(&header)).unwrap();
        compute::observe_group(&mut module, 1);
        let raw = module.words.clone();
        module.optimize();
        for (mode, words) in [("raw", &raw), ("optimized", &module.words)] {
            let validation = spirv_validation::validate(words);
            assert!(
                validation.status.success(),
                "{name}/{mode}: {}",
                String::from_utf8_lossy(&validation.stderr)
            );
            for (case, actual) in gpu.run(words, &states).iter().enumerate() {
                let expected = [cases[case].2, cases[case].3][flag];
                assert_eq!(
                    actual[4],
                    if expected { u32::MAX } else { 0 },
                    "{name}/{mode}/case{case}"
                );
                comparisons += 1;
            }
        }
    }
    eprintln!(
        "{}; FADD.CC: {comparisons} exact Z/N observations, 0 differences",
        gpu.description
    );
}

#[test]
#[ignore = "RADV hardware; FMUL.CC Z/N through CSET observers"]
fn gpu_fmul_cc_exact() {
    let gpu = Compute::new();
    let header = headers::parse(&[0; 32]).unwrap();
    let fmul_cc = (0x5c68_u64 << 48) | (1 << 47) | (3 << 20) | (7 << 16) | (2 << 8) | 1;
    let cases = [
        (0.0_f32, 3.0_f32, true, false),
        (-2.0, 2.0, false, true),
        (2.0, 3.0, false, false),
        (f32::NAN, 1.0, false, false),
        (-0.0, 3.0, true, false),
    ];
    let states = cases
        .iter()
        .map(|(lhs, rhs, _, _)| {
            let mut state = [0_u32; WORDS];
            state[2] = lhs.to_bits();
            state[3] = rhs.to_bits();
            state
        })
        .collect::<Vec<_>>();
    let mut comparisons = 0;
    for (name, cset, flag) in [
        ("Z", 0x5098_0380_0007_0204_u64, 0_usize),
        ("N", 0x5098_0380_0007_0104_u64, 1_usize),
    ] {
        let code = [0, fmul_cc, cset, 0xe300_u64 << 48 | 7 << 16 | 15]
            .into_iter()
            .flat_map(|word| [word as u32, (word >> 32) as u32])
            .collect::<Vec<_>>();
        let decoded = decode_stream(shadowbox::QuietRun, &code);
        let mut module = emit_unoptimized_with_header(&decoded, Some(&header)).unwrap();
        compute::observe_group(&mut module, 1);
        let raw = module.words.clone();
        module.optimize();
        for (mode, words) in [("raw", &raw), ("optimized", &module.words)] {
            let validation = spirv_validation::validate(words);
            assert!(
                validation.status.success(),
                "{name}/{mode}: {}",
                String::from_utf8_lossy(&validation.stderr)
            );
            for (case, actual) in gpu.run(words, &states).iter().enumerate() {
                let expected = [cases[case].2, cases[case].3][flag];
                assert_eq!(
                    actual[4],
                    if expected { u32::MAX } else { 0 },
                    "{name}/{mode}/case{case}"
                );
                comparisons += 1;
            }
        }
    }
    eprintln!(
        "{}; FMUL.CC: {comparisons} exact Z/N observations, 0 differences",
        gpu.description
    );
}

#[test]
#[ignore = "RADV hardware; FFMA.CC final-result Z/N through CSET observers"]
fn gpu_ffma_cc_exact() {
    let gpu = Compute::new();
    let header = headers::parse(&[0; 32]).unwrap();
    let ffma_cc = (0x5980_u64 << 48) | (1 << 47) | (4 << 39) | (3 << 20) | (7 << 16) | (2 << 8) | 1;
    let cases = [
        (0.0_f32, 3.0_f32, 0.0_f32, true, false),
        (-2.0, 2.0, 1.0, false, true),
        (2.0, 3.0, 1.0, false, false),
        (f32::NAN, 1.0, 0.0, false, false),
        (-0.0, 3.0, -0.0, true, false),
    ];
    let states = cases
        .iter()
        .map(|(lhs, rhs, addend, _, _)| {
            let mut state = [0_u32; WORDS];
            state[2] = lhs.to_bits();
            state[3] = rhs.to_bits();
            state[4] = addend.to_bits();
            state
        })
        .collect::<Vec<_>>();
    let mut comparisons = 0;
    for (name, cset, flag) in [
        ("Z", 0x5098_0380_0007_0205_u64, 0_usize),
        ("N", 0x5098_0380_0007_0105_u64, 1_usize),
    ] {
        let code = [0, ffma_cc, cset, 0xe300_u64 << 48 | 7 << 16 | 15]
            .into_iter()
            .flat_map(|word| [word as u32, (word >> 32) as u32])
            .collect::<Vec<_>>();
        let decoded = decode_stream(shadowbox::QuietRun, &code);
        let mut module = emit_unoptimized_with_header(&decoded, Some(&header)).unwrap();
        compute::observe_group(&mut module, 1);
        let raw = module.words.clone();
        module.optimize();
        for (mode, words) in [("raw", &raw), ("optimized", &module.words)] {
            let validation = spirv_validation::validate(words);
            assert!(
                validation.status.success(),
                "{name}/{mode}: {}",
                String::from_utf8_lossy(&validation.stderr)
            );
            for (case, actual) in gpu.run(words, &states).iter().enumerate() {
                let expected = [cases[case].3, cases[case].4][flag];
                assert_eq!(
                    actual[5],
                    if expected { u32::MAX } else { 0 },
                    "{name}/{mode}/case{case}"
                );
                comparisons += 1;
            }
        }
    }
    eprintln!(
        "{}; FFMA.CC: {comparisons} exact Z/N observations, 0 differences",
        gpu.description
    );
}

#[test]
#[ignore = "RADV hardware; FMNMX.CC final-result Z/N through CSET observers"]
fn gpu_fmnmx_cc_exact() {
    let gpu = Compute::new();
    let header = headers::parse(&[0; 32]).unwrap();
    let fmnmx_cc =
        (0x5c60_u64 << 48) | (1 << 47) | (7 << 39) | (3 << 20) | (7 << 16) | (2 << 8) | 1;
    let cases = [
        (0.0_f32, 1.0_f32, true, false),
        (-2.0, 1.0, false, true),
        (2.0, 3.0, false, false),
        (f32::NAN, 0.0, true, false),
        (f32::NAN, f32::NAN, false, false),
        (-0.0, 1.0, true, false),
    ];
    let states = cases
        .iter()
        .map(|(lhs, rhs, _, _)| {
            let mut state = [0_u32; WORDS];
            state[2] = lhs.to_bits();
            state[3] = rhs.to_bits();
            state
        })
        .collect::<Vec<_>>();
    let mut comparisons = 0;
    for (name, cset, flag) in [
        ("Z", 0x5098_0380_0007_0204_u64, 0_usize),
        ("N", 0x5098_0380_0007_0104_u64, 1_usize),
    ] {
        let code = [0, fmnmx_cc, cset, 0xe300_u64 << 48 | 7 << 16 | 15]
            .into_iter()
            .flat_map(|word| [word as u32, (word >> 32) as u32])
            .collect::<Vec<_>>();
        let decoded = decode_stream(shadowbox::QuietRun, &code);
        let mut module = emit_unoptimized_with_header(&decoded, Some(&header)).unwrap();
        compute::observe_group(&mut module, 1);
        let raw = module.words.clone();
        module.optimize();
        for (mode, words) in [("raw", &raw), ("optimized", &module.words)] {
            let validation = spirv_validation::validate(words);
            assert!(
                validation.status.success(),
                "{name}/{mode}: {}",
                String::from_utf8_lossy(&validation.stderr)
            );
            for (case, actual) in gpu.run(words, &states).iter().enumerate() {
                let expected = [cases[case].2, cases[case].3][flag];
                assert_eq!(
                    actual[4],
                    if expected { u32::MAX } else { 0 },
                    "{name}/{mode}/case{case}"
                );
                comparisons += 1;
            }
        }
    }
    eprintln!(
        "{}; FMNMX.CC: {comparisons} exact Z/N observations, 0 differences",
        gpu.description
    );
}

#[test]
#[ignore = "RADV hardware; directed integer carry and multiply"]
fn gpu_integer_translate_exact() {
    let pt = 7_u64 << 16;
    let operands = (3_u64 << 20) | pt | (2_u64 << 8) | 1;
    let imad_operands = operands | (4_u64 << 39);
    let mut report = Report::new("integer-translate");
    for (name, raw) in [
        (
            "IADD32I_CC",
            (0x1c00_u64 << 48) | (1 << 52) | (1 << 20) | pt | (2 << 8) | 1,
        ),
        (
            "IADD32I_X_CC",
            (0x1c00_u64 << 48) | (1 << 53) | (1 << 52) | pt | (2 << 8) | 1,
        ),
        (
            "IADD32I_NEGA_CC",
            (0x1c00_u64 << 48) | (1 << 56) | (1 << 52) | (2 << 20) | pt | (2 << 8) | 1,
        ),
        (
            "IADD32I_NEGB_CC",
            (0x1c00_u64 << 48) | (1 << 55) | (1 << 52) | (2 << 20) | pt | (2 << 8) | 1,
        ),
        ("IMUL_HI_UU", (0x5c38_u64 << 48) | (1 << 39) | operands),
        (
            "IMUL_HI_SU",
            (0x5c38_u64 << 48) | (1 << 40) | (1 << 39) | operands,
        ),
        (
            "IMUL_HI_US",
            (0x5c38_u64 << 48) | (1 << 41) | (1 << 39) | operands,
        ),
        (
            "IMUL_HI_SS",
            (0x5c38_u64 << 48) | (3 << 40) | (1 << 39) | operands,
        ),
        (
            "IMUL32I_HI_SU",
            (0x1f00_u64 << 48)
                | (1 << 54)
                | (1 << 53)
                | (0xffff_fffd_u64 << 20)
                | pt
                | (2 << 8)
                | 1,
        ),
        (
            "IMUL32I_HI_US",
            (0x1f00_u64 << 48)
                | (1 << 55)
                | (1 << 53)
                | (0x8000_0001_u64 << 20)
                | pt
                | (2 << 8)
                | 1,
        ),
        (
            "IMAD_HI_SU",
            (0x5a00_u64 << 48) | (1 << 54) | (1 << 48) | imad_operands,
        ),
        (
            "IMAD_HI_US",
            (0x5a00_u64 << 48) | (1 << 54) | (1 << 53) | imad_operands,
        ),
        (
            "IMAD_NEG_MULTIPLIER",
            (0x5a00_u64 << 48) | (2 << 51) | imad_operands,
        ),
        (
            "IMAD_NEG_ADDEND",
            (0x5a00_u64 << 48) | (1 << 51) | imad_operands,
        ),
        (
            "IMAD_ADDEND_ALIAS",
            (0x5a00_u64 << 48) | (4 << 39) | (3 << 20) | pt | (2 << 8) | 4,
        ),
        (
            "IMAD_CBUF_B_HI_SS",
            (0x4a00_u64 << 48) | (1 << 54) | (1 << 53) | (1 << 48) | (4 << 39) | pt | (2 << 8) | 1,
        ),
        (
            "IMAD_IMMEDIATE_NEG_ADDEND",
            (0x3400_u64 << 48) | (1 << 51) | (4 << 39) | (3 << 20) | pt | (2 << 8) | 1,
        ),
        (
            "IMAD_RC_NEG_MULTIPLIER",
            (0x5200_u64 << 48) | (2 << 51) | (4 << 39) | pt | (2 << 8) | 1,
        ),
        (
            "IMAD32I_HI_SU",
            (0x1000_u64 << 48) | (1 << 54) | (1 << 53) | (3 << 20) | pt | (2 << 8) | 1,
        ),
        (
            "IMAD32I_HI_US_NEG_ADDEND",
            (0x1000_u64 << 48)
                | (1 << 57)
                | (1 << 55)
                | (1 << 53)
                | (0xffff_fffd_u64 << 20)
                | pt
                | (2 << 8)
                | 1,
        ),
        (
            "IMADSP_REGISTER_U32_U24_U32",
            (0x5a80_u64 << 48) | imad_operands,
        ),
        (
            "IMADSP_REGISTER_S24_S24_S24",
            (0x5a80_u64 << 48) | (3 << 48) | (1 << 53) | (1 << 51) | imad_operands,
        ),
        (
            "IMADSP_REGISTER_U16HI_U16LO_U16HI",
            (0x5a80_u64 << 48) | (6 << 48) | (2 << 53) | (2 << 51) | imad_operands,
        ),
        (
            "IMADSP_REGISTER_S16HI_S16LO_S16HI",
            (0x5a80_u64 << 48) | (7 << 48) | (3 << 53) | (2 << 51) | imad_operands,
        ),
        (
            "IMADSP_IMMEDIATE",
            (0x3480_u64 << 48) | (2 << 53) | (4 << 39) | (0x7fff << 20) | pt | (2 << 8) | 1,
        ),
        (
            "IMADSP_CBUF_B",
            (0x4a80_u64 << 48) | (1 << 53) | (4 << 39) | pt | (2 << 8) | 1,
        ),
        (
            "IMADSP_CBUF_C",
            (0x5280_u64 << 48) | (2 << 53) | (4 << 39) | pt | (2 << 8) | 1,
        ),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; six FMNMX runtime-selector translations"]
fn gpu_fmnmx_select_six_exact() {
    let mut report = Report::new("fmnmx-select-six");
    for (name, raw) in [
        ("FMNMX_REGISTER_P0", 0x5c60_0000_0037_0201),
        ("FMNMX_REGISTER_NOT_P1", 0x5c60_0480_0037_0201),
        ("FMNMX_REGISTER_ABS_A_NEG_B_P3", 0x5c60_6180_0037_0201),
        ("FMNMX_REGISTER_NEG_A_ABS_B_NOT_P6", 0x5c63_0700_0037_0201),
        ("FMNMX_CBUF_NOT_P0", 0x4c60_0400_0017_0201),
        ("FMNMX_IMMEDIATE_P4", 0x3860_023f_c007_0201),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; five MUFU FP64 high-word translations"]
fn gpu_mufu64h_exact() {
    let mut report = Report::new("mufu64h-five");
    for (name, selector, modifiers) in [
        ("MUFU_RCP64H", 6_u64, 0_u64),
        ("MUFU_RCP64H_ABS", 6, 1 << 46),
        ("MUFU_RCP64H_NEG", 6, 1 << 48),
        ("MUFU_RSQ64H", 7, 0),
        ("MUFU_RSQ64H_SAT", 7, 1 << 50),
    ] {
        let raw = 0x5080_u64 << 48 | selector << 20 | modifiers | 7 << 16 | 2 << 8 | 1;
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; directed extended ISETP"]
fn gpu_isetp_extended_five_forms_exact() {
    let mut report = Report::new("isetp-extended-five-forms");
    for (name, opcode, operand) in [
        ("ISETP_X_REGISTER_U32", 0x5b64_u64, 3_u64 << 20),
        ("ISETP_X_REGISTER_S32", 0x5b65, 3 << 20),
        ("ISETP_X_CBUF_U32", 0x4b64, 0),
        ("ISETP_X_POSITIVE_I20_U32", 0x3664, 5 << 20),
        ("ISETP_X_NEGATIVE_I20_S32", 0x3765, 0x7ffff << 20),
    ] {
        // EQ.X, P0 destination, PT accumulator and guard. The reference and
        // emitted shader both consume the entry Z/C flags supplied per case.
        let raw = opcode << 48 | operand | 1 << 43 | 2 << 49 | 7 << 39 | 7 << 16 | 2 << 8 | 7;
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; five DSETP translations"]
fn gpu_dsetp_five_exact() {
    let mut report = Report::new("dsetp-five");
    let common = 7_u64 << 39 | 7 << 16 | 2 << 8 | 4 << 20 | 1;
    for (name, raw) in [
        ("DSETP_REGISTER_EQ", 0x5b80_u64 << 48 | 2 << 48 | common),
        ("DSETP_CBUF_LT", 0x4b80_u64 << 48 | 1 << 48 | common),
        (
            "DSETP_IMMEDIATE_LE",
            0x3680_u64 << 48 | 3 << 48 | 0x3ff00 << 20 | common & !(0xff << 20),
        ),
        (
            "DSETP_REGISTER_EQ_LHS_ABS",
            0x5b80_u64 << 48 | 2 << 48 | common | 1 << 7,
        ),
        (
            "DSETP_REGISTER_EQ_RHS_NEG",
            0x5b80_u64 << 48 | 2 << 48 | common | 1 << 6,
        ),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; six exact DSET translations"]
fn gpu_dset_six_exact() {
    let mut report = Report::new("dset-six");
    for (name, raw) in [
        ("DSET_REGISTER_EQ", 0x5902_0380_0047_0201),
        ("DSET_CBUF_LT", 0x4901_0380_0047_0201),
        ("DSET_IMMEDIATE_LE", 0x3203_03bf_f007_0201),
        ("DSET_REGISTER_NUM", 0x5907_0380_0047_0201),
        ("DSET_REGISTER_NAN", 0x5908_0380_0047_0201),
        ("DSET_REGISTER_UNORDERED_NE", 0x590d_0380_0047_0201),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; seven exact DSET.CC results through CSET observers"]
fn gpu_dset_cc_exact() {
    let gpu = Compute::new();
    assert!(gpu.supports_shader_float64, "DSET.CC needs shaderFloat64");
    let header = headers::parse(&[0; 32]).unwrap();
    let forms = [
        ("DSET_CC_MASK_F_OR", 0x5940_bf2f_80e1_2c1a_u64),
        ("DSET_CC_MASK_LT_XOR", 0x5961_cbc8_e1e6_78bd),
        ("DSET_CC_MASK_NAN_OR", 0x3248_b199_07bb_bb0d),
        ("DSET_CC_MASK_EQU_XOR", 0x334a_d6b1_3595_21a1),
        ("DSET_CC_BF_T_AND", 0x593f_94c6_cf78_22b3),
        ("DSET_CC_BF_NE_XOR", 0x3255_cb90_b738_8bba),
        ("DSET_CC_BF_LE_OR", 0x3313_b2ae_d9bc_64e7),
    ];
    let mut comparisons = 0;
    for (family, raw) in forms {
        let mut states = inputs(raw);
        prepare_dset_cc_cases(raw, &mut states);
        for (name, cset, flag) in [
            ("Z", 0x5098_0380_0007_0afa_u64, 264_usize),
            ("N", 0x5098_0380_0007_15fa_u64, 265_usize),
        ] {
            let code = [0, raw, cset, 0xe300_u64 << 48 | 7 << 16 | 15]
                .into_iter()
                .flat_map(|word| [word as u32, (word >> 32) as u32])
                .collect::<Vec<_>>();
            let decoded = decode_stream(shadowbox::QuietRun, &code);
            let mut module = emit_unoptimized_with_options(
                &decoded,
                Some(&header),
                shadowbox::TranslationOptions {
                    float64: shadowbox::HostFloat64::Supported,
                    ..Default::default()
                },
            )
            .unwrap();
            compute::observe_group(&mut module, 1);
            let raw_module = module.words.clone();
            module.optimize();
            for (variant, words) in [("raw", &raw_module), ("optimized", &module.words)] {
                validate(
                    words,
                    &work().join(format!("dset-cc-{family}-{name}-{variant}.spv")),
                );
                for (case, actual) in gpu.run(words, &states).iter().enumerate() {
                    let expected = reference(raw, &states[case]).unwrap();
                    assert_eq!(
                        actual[field(raw, 0, 8) as usize],
                        expected.words[field(raw, 0, 8) as usize],
                        "{family}/{name}/{variant}/case{case}/result P3={} lhs={:08x}{:08x}",
                        states[case][259],
                        states[case][field(raw, 8, 8) as usize + 1],
                        states[case][field(raw, 8, 8) as usize]
                    );
                    assert_eq!(
                        actual[flag], expected.words[flag],
                        "{family}/{name}/{variant}/case{case}/direct-flag"
                    );
                    assert_eq!(
                        actual[250],
                        if expected.words[flag] != 0 {
                            u32::MAX
                        } else {
                            0
                        },
                        "{family}/{name}/{variant}/case{case}/CSET"
                    );
                    comparisons += 1;
                }
            }
        }
    }
    eprintln!(
        "{}; DSET.CC: {comparisons} exact CSET Z/N observations, 0 differences",
        gpu.description
    );
}

#[test]
#[ignore = "RADV shaderFloat64; six exact DSETP field translations"]
fn gpu_dsetp_fields_six_exact() {
    let mut report = Report::new("dsetp-fields-six");
    for (name, raw) in [
        ("DSETP_REGISTER_GT_OR_NOT_P0", 0x5b84_2400_0047_0201),
        ("DSETP_REGISTER_LTU_XOR_P1", 0x5b89_4080_0047_0201),
        ("DSETP_REGISTER_LT_NEG_A_ABS_B", 0x5b81_1b80_0047_0201),
        (
            "DSETP_REGISTER_NEU_NEG_ABS_A_NEG_ABS_B",
            0x5b8d_1b80_0047_02c1,
        ),
        ("DSETP_CBUF_GEU_NEG_A_OR_NOT_P2", 0x4b8e_2d00_0047_0201),
        (
            "DSETP_IMMEDIATE_GTU_NEG_ABS_A_XOR_NOT_P1",
            0x368c_4cbf_f007_0281,
        ),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; six exact DSET field translations"]
fn gpu_dset_fields_six_exact() {
    let mut report = Report::new("dset-fields-six");
    for (name, raw) in [
        ("DSET_REGISTER_GT_BF", 0x5914_0380_0047_0201),
        ("DSET_REGISTER_GE_OR_NOT_P0", 0x5906_2400_0047_0201),
        ("DSET_REGISTER_LTU_XOR_P1", 0x5909_4080_0047_0201),
        ("DSET_REGISTER_LT_NEG_A_ABS_B", 0x5901_1b80_0047_0201),
        ("DSET_REGISTER_EQU_ABS_A_NEG_B", 0x596a_0380_0047_0201),
        ("DSET_CBUF_GEU_NEG_A_BF", 0x491e_0b80_0047_0201),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV; six exact FSETP immediate field translations"]
fn gpu_fsetp_immediate_fields_six_exact() {
    let mut report = Report::new("fsetp-immediate-fields-six");
    for (name, raw) in [
        ("FSETP_IMMEDIATE_GT_NEG_B", 0x36b4_03bf_8007_0247),
        ("FSETP_IMMEDIATE_LT_ABS_B", 0x37b1_13c0_0007_0207),
        ("FSETP_IMMEDIATE_GE_NEG_ABS_B", 0x37b6_13bf_0007_0247),
        ("FSETP_IMMEDIATE_EQ_NEG_B_OR_NOT_P1", 0x36b2_24bf_8007_0247),
        ("FSETP_IMMEDIATE_NEU_NEG_ABS_A_NEG_B", 0x36bd_0bbf_8007_02c7),
        (
            "FSETP_IMMEDIATE_LE_ABS_A_ABS_B_XOR_P2",
            0x37b3_5140_0007_0287,
        ),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; six exact ISCADD32I translations"]
fn gpu_iscadd32i_six_exact() {
    let mut report = Report::new("iscadd32i-six");
    for (name, raw) in [
        ("ISCADD32I_SHIFT0_0X10", 0x1400_0000_0107_0201),
        ("ISCADD32I_SHIFT1_0X3", 0x1420_0000_0037_0201),
        ("ISCADD32I_SHIFT4_NEG_0X10", 0x148f_ffff_ff07_0201),
        ("ISCADD32I_SHIFT31_0X12345678", 0x17e1_2345_6787_0201),
        ("ISCADD32I_SHIFT16_NEG_0X80000000", 0x1608_0000_0007_0201),
        ("ISCADD32I_SHIFT2_0X7FFFFFFF_CC", 0x1457_ffff_fff7_0201),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV; six exact FFMA32I translations"]
fn gpu_ffma32i_six_exact() {
    let mut report = Report::new("ffma32i-six");
    for (name, raw) in [
        ("FFMA32I_PLAIN", 0x0c03_fc00_0007_0201),
        ("FFMA32I_NEG_A", 0x0d0b_f400_0007_0201),
        ("FFMA32I_NEG_C", 0x0e04_0000_0007_0201),
        ("FFMA32I_SAT", 0x0c83_f000_0007_0201),
        ("FFMA32I_FTZ", 0x0c24_0400_0007_0201),
        ("FFMA32I_FMZ", 0x0c40_0000_0007_0201),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; six exact one-axis TXD translations"]
fn gpu_txd_one_axis_six_exact() {
    let mut report = Report::new("txd-one-axis-six");
    for (name, raw) in [
        ("TXD_1D_MASK15", 0xde38_0017_8107_0820),
        ("TXD_NODEP_1D_MASK5", 0xde3a_0022_8107_0820),
        ("TXD_ARRAY1D_MASK15", 0xde38_0037_9107_0820),
        ("TXD_AOFFI_1D_MASK15", 0xde38_001f_8107_0820),
        ("TXD_AOFFI_ARRAY1D_MASK11", 0xde38_004d_9107_0820),
        ("TXD_NOT_P2_ARRAY1D_MASK2", 0xde38_0051_110a_0820),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV; six exact XMAD.CSFU translations"]
fn gpu_xmad_csfu_six_exact() {
    let mut report = Report::new("xmad-csfu-six");
    for (name, raw) in [
        ("XMAD_CSFU_REGISTER_S16_S16", 0x5b0f_0200_0027_0103),
        ("XMAD_CSFU_REGISTER_H1_P1", 0x5b2d_0208_0021_0103),
        ("XMAD_CSFU_REGISTER_PSL_NOT_P0", 0x5b0e_0210_0028_0103),
        ("XMAD_CSFU_IMMEDIATE_8001", 0x360f_0208_0017_0103),
        ("XMAD_CSFU_CBUF_B_H1", 0x4e1f_0200_0027_0103),
        ("XMAD_CSFU_CBUF_C_NOT_P0", 0x512f_0200_0018_0103),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV; six exact RRO immediate translations"]
fn gpu_rro_immediate_six_exact() {
    let mut report = Report::new("rro-immediate-six");
    for (name, raw) in [
        ("RRO_IMMEDIATE_SINCOS_ONE", 0x3890_003f_8007_0003),
        ("RRO_IMMEDIATE_EX2_MINUS_2_5", 0x3990_00c0_2007_0003),
        ("RRO_IMMEDIATE_SINCOS_ABS_MINUS_3", 0x3992_0040_4007_0003),
        ("RRO_IMMEDIATE_EX2_NEG_HALF", 0x3890_20bf_0007_0003),
        ("RRO_IMMEDIATE_SINCOS_NEG_ABS_INF", 0x3892_207f_8007_0003),
        ("RRO_IMMEDIATE_EX2_NEG_ABS_MINUS_NAN", 0x3992_20ff_c007_0003),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV; six exact CC-tested SYNC translations"]
fn gpu_sync_cc_six_exact() {
    let mut report = Report::new("sync-cc-six");
    for (name, raw) in [
        ("SYNC_CC_LT", 0xf0f8_0000_0007_0001),
        ("SYNC_CC_EQ_P0", 0xf0f8_0000_0000_0002),
        ("SYNC_CC_NE_NOT_P2", 0xf0f8_0000_000a_0005),
        ("SYNC_CC_LO", 0xf0f8_0000_0007_0011),
        ("SYNC_CC_HS_P3", 0xf0f8_0000_0003_0016),
        ("SYNC_CC_RLE_NOT_P1", 0xf0f8_0000_0009_001e),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV; six more exact CC-tested SYNC translations"]
fn gpu_sync_cc_tests_six_exact() {
    let mut report = Report::new("sync-cc-tests-six");
    for (name, raw) in [
        ("SYNC_CC_LE_P1", 0xf0f8_0000_0001_0003),
        ("SYNC_CC_GT", 0xf0f8_0000_0007_0004),
        ("SYNC_CC_GE_NOT_P0", 0xf0f8_0000_0008_0006),
        ("SYNC_CC_NUM", 0xf0f8_0000_0007_0007),
        ("SYNC_CC_NAN_P2", 0xf0f8_0000_0002_0008),
        ("SYNC_CC_HI", 0xf0f8_0000_0007_0014),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; sixteen SSY opcode aliases"]
fn gpu_ssy_alias_sixteen_exact() {
    let mut report = Report::new("ssy-alias-sixteen");
    for opcode in 0xe290_u64..=0xe29f {
        report.check("SSY_ALIAS", opcode << 48 | 0x20, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV; six exact FFMA32I modifier and register combinations"]
fn gpu_ffma32i_combinations_six_exact() {
    let mut report = Report::new("ffma32i-combinations-six");
    for (name, raw) in [
        ("FFMA32I_NEG_A_NEG_C", 0x0f03_fc00_0007_0201),
        ("FFMA32I_SAT_NEG_A", 0x0d84_0000_0007_0201),
        ("FFMA32I_SAT_NEG_C", 0x0e8b_fc00_0007_0201),
        ("FFMA32I_SAT_NEG_A_NEG_C", 0x0f83_e800_0007_0201),
        ("FFMA32I_R5_R3", 0x0c04_0800_0007_0305),
        ("FFMA32I_R7_R6_NEG_ZERO", 0x0c08_0000_0007_0607),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; six exact CSET translations"]
fn gpu_cset_six_exact() {
    let mut report = Report::new("cset-six");
    for (name, raw) in [
        ("CSET_LT_AND_PT", 0x5098_0380_0007_0101),
        ("CSET_GT_AND_PT_BF", 0x5098_1380_0007_0401),
        ("CSET_EQ_OR_NOT_P0", 0x5098_2400_0007_0201),
        ("CSET_HI_XOR_P1", 0x5098_4080_0007_1401),
        ("CSET_NEU_OR_NOT_P3_BF", 0x5098_3580_0007_0d01),
        ("CSET_RGT_AND_PT", 0x5098_0380_0007_1f01),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV; six exact FSETP second-destination translations"]
fn gpu_fsetp_second_destination_six_exact() {
    let mut report = Report::new("fsetp-second-destination-six");
    for (name, raw) in [
        ("FSETP_REGISTER_GT_P0_P1", 0x5bb4_0380_0047_0201),
        ("FSETP_REGISTER_LTU_OR_NOT_P0_P1_P2", 0x5bb9_2400_0047_020a),
        ("FSETP_REGISTER_LE_XOR_P2_P3_P0", 0x5bb3_5900_0047_0298),
        ("FSETP_CBUF_NE_NEG_B_P2_P5", 0x4bb5_0380_0047_0255),
        ("FSETP_IMMEDIATE_GE_NOT_P1_P0_P3", 0x36b6_04bf_8007_0203),
        ("FSETP_IMMEDIATE_NEU_OR_P3_P4_P6", 0x37bd_21c0_0007_02a6),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; six exact CSET selector translations"]
fn gpu_cset_selectors_six_exact() {
    let mut report = Report::new("cset-selectors-six");
    for (name, raw) in [
        ("CSET_LE_AND_P2", 0x5098_0100_0007_0301),
        ("CSET_NE_OR_PT_BF", 0x5098_3380_0007_0501),
        ("CSET_GE_XOR_NOT_P1", 0x5098_4480_0007_0601),
        ("CSET_LTU_AND_PT_BF", 0x5098_1380_0007_0901),
        ("CSET_LO_OR_NOT_P0", 0x5098_2400_0007_1101),
        ("CSET_RLE_AND_P5", 0x5098_0280_0007_1e01),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV; six exact FMUL/FMUL32I FMZ translations"]
fn gpu_fmul_fmz_six_exact() {
    let mut report = Report::new("fmul-fmz-six");
    for (name, raw) in [
        ("FMUL_REGISTER_FMZ", 0x5c68_2000_0037_0201),
        ("FMUL_CBUF_FMZ_NEG_B", 0x4c69_2000_0017_0201),
        ("FMUL_IMMEDIATE_FMZ_SAT", 0x386c_203f_c007_0201),
        ("FMUL_REGISTER_FMZ_D2", 0x5c68_2200_0037_0201),
        ("FMUL32I_FMZ", 0x1e43_fc00_0007_0201),
        ("FMUL32I_FMZ_SAT", 0x1ec3_fc00_0007_0201),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; six exact LOP .X and LOP32I PASS_B translations"]
fn gpu_lop_extended_six_exact() {
    let mut report = Report::new("lop-extended-six");
    for (name, raw) in [
        ("LOP_AND_X_CC_R", 0x5c47_8800_0037_0201),
        ("LOP_OR_X_CC_C", 0x4c47_8a04_0027_0201),
        ("LOP_XOR_X_CC_NOT_A_I", 0x3847_8c92_3457_0201),
        ("LOP_PASS_B_X_CC_Z_P0", 0x5c40_af00_0037_0201),
        ("LOP32I_PASS_B", 0x0461_2345_6787_0201),
        ("LOP32I_AND_X_CC_NOT_B", 0x0710_000f_fff7_0201),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV; six exact ISETP second-destination translations"]
fn gpu_isetp_second_destination_six_exact() {
    let mut report = Report::new("isetp-second-destination-six");
    for (name, raw) in [
        ("ISETP_REGISTER_U32_LT_P0_P1", 0x5b62_0380_0047_0201),
        (
            "ISETP_REGISTER_S32_GE_OR_NOT_P0_P1_P2",
            0x5b6d_2400_0047_020a,
        ),
        (
            "ISETP_REGISTER_S32_NE_X_XOR_P2_P3_P0",
            0x5b6b_4900_0047_0218,
        ),
        ("ISETP_CBUF_U32_GT_P2_P5", 0x4b68_0380_0047_0215),
        ("ISETP_IMMEDIATE_S32_LE_NOT_P1_P0_P3", 0x3767_04ff_ffb7_0203),
        ("ISETP_IMMEDIATE_U32_EQ_OR_P3_PT_P6", 0x3664_2180_0077_023e),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV; six exact SHR.BREV translations"]
fn gpu_shr_brev_six_exact() {
    let mut report = Report::new("shr-brev-six");
    for (name, raw) in [
        ("SHR_BREV_REGISTER_U32", 0x5c28_0100_0047_0203),
        ("SHR_BREV_REGISTER_S32_W", 0x5c29_0180_0047_0203),
        ("SHR_BREV_CBUF_U32_W", 0x4c28_0180_0047_0203),
        ("SHR_BREV_CBUF_S32", 0x4c29_0100_0047_0203),
        ("SHR_BREV_IMMEDIATE_U32_7", 0x3828_0100_0077_0203),
        ("SHR_BREV_IMMEDIATE_S32_MINUS_1", 0x3929_017f_fff7_0203),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV; VMAD whole-word A selector U32/S32"]
fn gpu_vmad_whole_a_exact() {
    let base = (0x5f00_u64 << 48)
        | 1
        | (2 << 8)
        | (7 << 16)
        | (4 << 20)
        | (6 << 36)
        | (5 << 39)
        | (1 << 50);
    let mut report = Report::new("vmad-whole-a");
    for (name, raw) in [
        ("VMAD_U32_A", base),
        ("VMAD_S32_A", base | (1 << 48)),
        ("VMAD_S32_A_SHIFT7", base | (1 << 48) | (1 << 51)),
        ("VMAD_U32_A_SAT", base | (1 << 55)),
        ("VMAD_S32_A_SAT", base | (1 << 48) | (1 << 55)),
        ("VMAD_U32_A_DST_LHS_ALIAS", (base & !0xff) | 2),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; five DADD translations"]
fn gpu_dadd_five_exact() {
    let mut report = Report::new("dadd-five");
    for (name, raw) in [
        ("DADD_REGISTER", 0x5c70_0000_0047_0206),
        ("DADD_REGISTER_LHS_ABS", 0x5c70_4000_0047_0206),
        ("DADD_REGISTER_LHS_NEG", 0x5c71_0000_0047_0206),
        ("DADD_REGISTER_RHS_ABS", 0x5c72_0000_0047_0206),
        ("DADD_REGISTER_RHS_NEG", 0x5c70_2000_0047_0206),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; six exact DMNMX register translations"]
fn gpu_dmnmx_six_exact() {
    let mut report = Report::new("dmnmx-six");
    for (name, raw) in [
        ("DMNMX_MINIMUM", 0x5c50_0380_0047_0208),
        ("DMNMX_MAXIMUM", 0x5c50_0780_0047_0208),
        ("DMNMX_LHS_ABS", 0x5c50_4380_0047_0208),
        ("DMNMX_LHS_NEG", 0x5c51_0380_0047_0208),
        ("DMNMX_RHS_ABS", 0x5c52_0380_0047_0208),
        ("DMNMX_RHS_NEG", 0x5c50_2380_0047_0208),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; six exact DMNMX constant-buffer and immediate translations"]
fn gpu_dmnmx_source_six_exact() {
    let mut report = Report::new("dmnmx-source-six");
    for (name, raw) in [
        ("DMNMX_CBUF_MINIMUM", 0x4c50_0380_0047_0208),
        ("DMNMX_CBUF_MAXIMUM", 0x4c50_0780_0047_0208),
        ("DMNMX_CBUF_MAXIMUM_LHS_NEG_RHS_ABS", 0x4c53_0784_0087_0208),
        ("DMNMX_IMMEDIATE_MINIMUM_ONE", 0x3850_03bf_f007_0208),
        (
            "DMNMX_IMMEDIATE_MAXIMUM_NEGATIVE_TWO",
            0x3950_07c0_0007_0208,
        ),
        (
            "DMNMX_IMMEDIATE_MINIMUM_LHS_ABS_RHS_NEG",
            0x3850_63bf_e007_0208,
        ),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; fifteen exact DMNMX.CC forms with CSET Z/N observers"]
fn gpu_dmnmx_cc_fifteen_exact() {
    let gpu = Compute::new();
    assert!(gpu.supports_shader_float64, "DMNMX.CC needs shaderFloat64");
    let header = headers::parse(&[0; 32]).unwrap();
    let forms = [
        0x5c50_8000_0007_0000_u64,
        0x5c51_e26d_1df0_3060,
        0x5c53_abe9_f630_8c0c,
        0x5c54_8037_73ec_86fe,
        0x5c57_d1ca_73fe_9cd5,
        0x4c50_8000_0007_0000,
        0x4c50_f2c6_4fc6_7370,
        0x4c54_8129_8a24_86bd,
        0x4c54_df5c_97d5_b24a,
        0x4c57_9e13_2e27_584f,
        0x3853_858d_815e_e051,
        0x3855_b18f_0e65_7ed7,
        0x3952_81b4_453c_4204,
        0x3952_c50d_ef49_2290,
        0x3956_f5db_a153_81c0,
    ];
    let mut comparisons = 0;
    for raw in forms {
        let mut states = inputs(raw);
        // Keep arbitrary register aliases deterministic. Exceptional numeric
        // behavior is covered by the independent CPU matrix; this live path
        // establishes every exact word's final-result Z/N write through CSET.
        for state in &mut states {
            state[..256].fill(0);
        }
        for (name, cset, flag) in [
            ("Z", 0x5098_0380_0007_0afa_u64, 264_usize),
            ("N", 0x5098_0380_0007_15fa_u64, 265_usize),
        ] {
            let code = [0, raw, cset, 0xe300_u64 << 48 | 7 << 16 | 15]
                .into_iter()
                .flat_map(|word| [word as u32, (word >> 32) as u32])
                .collect::<Vec<_>>();
            let decoded = decode_stream(shadowbox::QuietRun, &code);
            let mut module = emit_unoptimized_with_options(
                &decoded,
                Some(&header),
                shadowbox::TranslationOptions {
                    float64: shadowbox::HostFloat64::Supported,
                    ..Default::default()
                },
            )
            .unwrap();
            compute::observe_group(&mut module, 1);
            let raw_module = module.words.clone();
            module.optimize();
            for (variant, words) in [("raw", &raw_module), ("optimized", &module.words)] {
                validate(
                    words,
                    &work().join(format!("dmnmx-cc-{raw:016x}-{name}-{variant}.spv")),
                );
                for (case, actual) in gpu.run(words, &states).iter().enumerate() {
                    let expected = reference(raw, &states[case]).unwrap();
                    assert_eq!(
                        actual[flag], expected.words[flag],
                        "{raw:016x}/{name}/{variant}/case{case}/direct-flag",
                    );
                    assert_eq!(
                        actual[250],
                        if expected.words[flag] != 0 {
                            u32::MAX
                        } else {
                            0
                        },
                        "{raw:016x}/{name}/{variant}/case{case}/CSET"
                    );
                    assert_eq!(
                        actual[266], states[case][266],
                        "{raw:016x}/{variant}/case{case}/C preserved"
                    );
                    assert_eq!(
                        actual[267], states[case][267],
                        "{raw:016x}/{variant}/case{case}/V preserved"
                    );
                    comparisons += 1;
                }
            }
        }
    }
    eprintln!(
        "{}; DMNMX.CC: {comparisons} exact CSET Z/N observations, 0 differences",
        gpu.description
    );
}

#[test]
#[ignore = "RADV shaderFloat64; all 1,792 DMNMX immediate CC words"]
fn gpu_dmnmx_immediate_cc_exact() {
    let mut report = Report::new("dmnmx-immediate-cc");
    assert!(
        report.gpu.supports_shader_float64,
        "DMNMX.CC needs shaderFloat64"
    );
    for raw in dmnmx_immediate_cc_words() {
        report.check("DMNMX_IMMEDIATE_CC_B3E90B402E51EB95B", raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; five DADD constant-buffer translations"]
fn gpu_dadd_cbuf_five_exact() {
    let mut report = Report::new("dadd-cbuf-five");
    for (name, raw) in [
        ("DADD_CBUF", 0x4c70_0000_0047_0206),
        ("DADD_CBUF_LHS_ABS", 0x4c70_4000_0047_0206),
        ("DADD_CBUF_LHS_NEG", 0x4c71_0000_0047_0206),
        ("DADD_CBUF_RHS_ABS", 0x4c72_0000_0047_0206),
        ("DADD_CBUF_RHS_NEG", 0x4c70_2000_0047_0206),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; five DADD immediate translations"]
fn gpu_dadd_immediate_five_exact() {
    let mut report = Report::new("dadd-immediate-five");
    for (name, raw) in [
        ("DADD_IMMEDIATE_POSITIVE_ONE", 0x3870_003f_f007_0206),
        ("DADD_IMMEDIATE_NEGATIVE_ONE", 0x3970_003f_f007_0206),
        ("DADD_IMMEDIATE_HALF_LHS_ABS", 0x3870_403f_e007_0206),
        ("DADD_IMMEDIATE_TWO_LHS_NEG", 0x3871_0040_0007_0206),
        (
            "DADD_IMMEDIATE_NEGATIVE_TWO_LHS_ABS_NEG",
            0x3971_4040_0007_0206,
        ),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; all 704 FADD.FTZ.RM register words"]
fn gpu_fadd_ftz_rm_exact() {
    let mut report = Report::new("fadd-ftz-rm");
    assert!(
        report.gpu.supports_shader_float64,
        "FADD.FTZ.RM needs shaderFloat64"
    );
    for raw in fadd_ftz_rm_words() {
        report.check("FADD_FTZ_RM_B931C88442C4F9F4A", raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; all 1,024 DADD.RN.CC immediate words"]
fn gpu_dadd_immediate_rn_cc_exact() {
    let mut report = Report::new("dadd-immediate-rn-cc");
    assert!(
        report.gpu.supports_shader_float64,
        "DADD.RN.CC needs shaderFloat64"
    );
    for raw in dadd_immediate_rn_cc_words() {
        report.check("DADD_IMMEDIATE_RN_CC_B998CE0179F0C887C", raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; all 2,048 DADD.RM immediate words"]
fn gpu_dadd_immediate_rm_exact() {
    let mut report = Report::new("dadd-immediate-rm");
    assert!(
        report.gpu.supports_shader_float64,
        "DADD.RM needs shaderFloat64"
    );
    for raw in dadd_immediate_rm_words() {
        report.check("DADD_IMMEDIATE_RM_B52EEA4C54C38D8B5", raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; all 1,024 DADD.RM register words"]
fn gpu_dadd_register_rm_exact() {
    let mut report = Report::new("dadd-register-rm");
    assert!(
        report.gpu.supports_shader_float64,
        "DADD.RM needs shaderFloat64"
    );
    for raw in dadd_register_rm_words() {
        report.check("DADD_REGISTER_RM_BCF4E4433B82E5605", raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; all 1,152 DADD.RM constant-buffer words"]
fn gpu_dadd_cbuf_rm_exact() {
    let mut report = Report::new("dadd-cbuf-rm");
    assert!(
        report.gpu.supports_shader_float64,
        "DADD.RM needs shaderFloat64"
    );
    for raw in dadd_cbuf_rm_words() {
        report.check("DADD_CBUF_RM_B6A7D64E7F031AD87", raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV; all 1,536 NOP words"]
fn gpu_nop_exact() {
    let mut report = Report::new("nop");
    for raw in nop_words() {
        report.check("NOP_B4B1706D67C322E39", raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV; all 1,152 NOP tail words"]
fn gpu_nop_tail_exact() {
    let mut report = Report::new("nop-tail");
    for raw in nop_tail_words() {
        report.check("NOP_TAIL_B0C2D009E25FBC74A_BCC6BE917D64B5C02", raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; all 1,152 DADD.RP constant-buffer words"]
fn gpu_dadd_cbuf_rp_exact() {
    let mut report = Report::new("dadd-cbuf-rp");
    assert!(
        report.gpu.supports_shader_float64,
        "DADD.RP needs shaderFloat64"
    );
    for raw in dadd_cbuf_rp_words() {
        report.check("DADD_CBUF_RP_B6F062091C6E76F1B", raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; all 1,152 DADD.RZ constant-buffer words"]
fn gpu_dadd_cbuf_rz_exact() {
    let mut report = Report::new("dadd-cbuf-rz");
    assert!(
        report.gpu.supports_shader_float64,
        "DADD.RZ needs shaderFloat64"
    );
    for raw in dadd_cbuf_rz_words() {
        report.check("DADD_CBUF_RZ_BE1E786EB176D6782", raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; all 1,024 DADD.RP register words"]
fn gpu_dadd_register_rp_exact() {
    let mut report = Report::new("dadd-register-rp");
    assert!(
        report.gpu.supports_shader_float64,
        "DADD.RP needs shaderFloat64"
    );
    for raw in dadd_register_rp_words() {
        report.check("DADD_REGISTER_RP_B62BBBCF898DEFF98", raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; all 1,024 DADD.RZ register words"]
fn gpu_dadd_register_rz_exact() {
    let mut report = Report::new("dadd-register-rz");
    assert!(
        report.gpu.supports_shader_float64,
        "DADD.RZ needs shaderFloat64"
    );
    for raw in dadd_register_rz_words() {
        report.check("DADD_REGISTER_RZ_B13D614303751D81C", raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; all 2,048 DADD.RP immediate words"]
fn gpu_dadd_immediate_rp_exact() {
    let mut report = Report::new("dadd-immediate-rp");
    assert!(
        report.gpu.supports_shader_float64,
        "DADD.RP needs shaderFloat64"
    );
    for raw in dadd_immediate_rp_words() {
        report.check("DADD_IMMEDIATE_RP_B96A2BAB5FA689E7D", raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; all 2,048 DADD.RZ immediate words"]
fn gpu_dadd_immediate_rz_exact() {
    let mut report = Report::new("dadd-immediate-rz");
    assert!(
        report.gpu.supports_shader_float64,
        "DADD.RZ needs shaderFloat64"
    );
    for raw in dadd_immediate_rz_words() {
        report.check("DADD_IMMEDIATE_RZ_BAAD61EDEC9687E55", raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; five DMUL translations"]
fn gpu_dmul_five_exact() {
    let mut report = Report::new("dmul-five");
    for (name, raw) in [
        ("DMUL_REGISTER", 0x5c80_0000_0047_0206),
        ("DMUL_REGISTER_NEGATED", 0x5c81_0000_0047_0206),
        ("DMUL_CBUF", 0x4c80_0000_0047_0206),
        ("DMUL_CBUF_NEGATED", 0x4c81_0000_0047_0206),
        ("DMUL_IMMEDIATE_POSITIVE_ONE", 0x3880_003f_f007_0206),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; five exact DMUL register-operand translations"]
fn gpu_dmul_register_operands_five_exact() {
    let mut report = Report::new("dmul-register-operands-five");
    for (name, raw) in [
        ("DMUL_REGISTER_BASELINE", 0x5c80_0000_0007_0000),
        ("DMUL_REGISTER_DST_R1", 0x5c80_0000_0007_0001),
        ("DMUL_REGISTER_LHS_R2", 0x5c80_0000_0007_0200),
        ("DMUL_REGISTER_RHS_R4", 0x5c80_0000_0047_0000),
        ("DMUL_REGISTER_BASELINE_NEGATED", 0x5c81_0000_0007_0000),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; six exact DFMA translations"]
fn gpu_dfma_six_exact() {
    let mut report = Report::new("dfma-six");
    for (name, raw) in [
        ("DFMA_REGISTER", 0x5b70_0000_0007_0000),
        ("DFMA_PRODUCT_NEGATED", 0x5b71_0000_0007_0000),
        ("DFMA_ADDEND_NEGATED", 0x5b72_0000_0007_0000),
        ("DFMA_LHS_R2", 0x5b70_0000_0007_0200),
        ("DFMA_MULTIPLIER_R4", 0x5b70_0000_0047_0000),
        ("DFMA_ADDEND_R4", 0x5b70_0200_0007_0000),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; six exact DFMA C/RC/I translations"]
fn gpu_dfma_source_six_exact() {
    let mut report = Report::new("dfma-source-six");
    for (name, raw) in [
        ("DFMA_CBUF_MULTIPLIER", 0x4b70_0200_0047_0206),
        ("DFMA_CBUF_MULTIPLIER_ADDEND_NEGATED", 0x4b72_0200_0047_0206),
        ("DFMA_CBUF_ADDEND", 0x5370_0200_0047_0206),
        ("DFMA_CBUF_ADDEND_PRODUCT_NEGATED", 0x5371_0200_0047_0206),
        ("DFMA_IMMEDIATE_POSITIVE_ONE", 0x3670_023f_f007_0206),
        (
            "DFMA_IMMEDIATE_NEGATIVE_TWO_ADDEND_NEGATED",
            0x3772_0240_0007_0206,
        ),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; DFMA.CC final-result Z/N through CSET observers"]
fn gpu_dfma_cc_exact() {
    let gpu = Compute::new();
    assert!(gpu.supports_shader_float64, "DFMA.CC needs shaderFloat64");
    let header = headers::parse(&[0; 32]).unwrap();
    let forms = [
        ("DFMA_CC_REGISTER", 0x5b70_8000_0007_0000_u64),
        ("DFMA_CC_REGISTER_CBUF", 0x5370_8000_0007_0000),
        ("DFMA_CC_CBUF_REGISTER", 0x4b70_8000_0007_0000),
        ("DFMA_CC_IMMEDIATE_ZERO", 0x3670_8000_0007_0000),
        ("DFMA_CC_IMMEDIATE_MODIFIED", 0x3673_debb_4e77_4fc0),
    ];
    let mut comparisons = 0;
    for (family, raw) in forms {
        let mut states = inputs(raw);
        prepare_dfma_cc_cases(raw, &mut states);
        for (name, cset, flag) in [
            ("Z", 0x5098_0380_0007_0afa_u64, 264_usize),
            ("N", 0x5098_0380_0007_15fa_u64, 265_usize),
        ] {
            let code = [0, raw, cset, 0xe300_u64 << 48 | 7 << 16 | 15]
                .into_iter()
                .flat_map(|word| [word as u32, (word >> 32) as u32])
                .collect::<Vec<_>>();
            let decoded = decode_stream(shadowbox::QuietRun, &code);
            let mut module = emit_unoptimized_with_options(
                &decoded,
                Some(&header),
                shadowbox::TranslationOptions {
                    float64: shadowbox::HostFloat64::Supported,
                    ..Default::default()
                },
            )
            .unwrap();
            compute::observe_group(&mut module, 1);
            let raw_module = module.words.clone();
            module.optimize();
            for (variant, words) in [("raw", &raw_module), ("optimized", &module.words)] {
                validate(
                    words,
                    &work().join(format!("dfma-cc-{family}-{name}-{variant}.spv")),
                );
                for (case, actual) in gpu.run(words, &states).iter().enumerate() {
                    let expected = reference(raw, &states[case]).unwrap();
                    assert_eq!(
                        actual[flag], expected.words[flag],
                        "{family}/{name}/{variant}/case{case}/direct-flag result={:08x}{:08x} expected={:08x}{:08x}",
                        actual[field(raw, 0, 8) as usize + 1],
                        actual[field(raw, 0, 8) as usize],
                        expected.words[field(raw, 0, 8) as usize + 1],
                        expected.words[field(raw, 0, 8) as usize],
                    );
                    assert_eq!(
                        actual[250],
                        if expected.words[flag] != 0 {
                            u32::MAX
                        } else {
                            0
                        },
                        "{family}/{name}/{variant}/case{case}"
                    );
                    comparisons += 1;
                }
            }
        }
    }
    eprintln!(
        "{}; DFMA.CC: {comparisons} exact CSET Z/N observations, 0 differences",
        gpu.description
    );
}

#[test]
#[ignore = "RADV shaderFloat64; DADD/DMUL RN.CC final-result Z/N through CSET observers"]
fn gpu_double_rn_cc_exact() {
    let gpu = Compute::new();
    assert!(
        gpu.supports_shader_float64,
        "double RN.CC needs shaderFloat64"
    );
    let header = headers::parse(&[0; 32]).unwrap();
    let forms = [
        ("DADD_CC_CBUF", 0x4c76_fc03_2b36_9161_u64),
        ("DADD_CC_IMMEDIATE_A", 0x3872_9a45_25e5_1fed),
        ("DADD_CC_IMMEDIATE_B", 0x3971_d03d_ef95_759d),
        ("DADD_CC_IMMEDIATE_C", 0x3976_fc48_5c55_51ee),
        ("DMUL_CC_REGISTER", 0x5c86_da71_bf40_b4f2),
        ("DMUL_CC_IMMEDIATE", 0x3986_c87c_3b2e_5d0d),
        ("DMUL_CC_ZERO", 0x5c80_8000_0007_0000),
        ("DMUL_CC_ZERO_NEG", 0x5c81_8000_0007_0000),
        ("DMUL_CC_VARIED_A", 0x5c80_c82a_d4c7_25a4),
        ("DMUL_CC_VARIED_A_NEG", 0x5c81_c82a_d4c7_25a4),
        ("DMUL_CC_VARIED_B", 0x5c80_dc03_b1f7_0950),
    ];
    let mut comparisons = 0;
    for (family, raw) in forms {
        let mut states = inputs(raw);
        prepare_double_rn_cc_cases(raw, &mut states);
        for (name, cset, flag) in [
            ("Z", 0x5098_0380_0007_0afa_u64, 264_usize),
            ("N", 0x5098_0380_0007_15fa_u64, 265_usize),
        ] {
            let code = [0, raw, cset, 0xe300_u64 << 48 | 7 << 16 | 15]
                .into_iter()
                .flat_map(|word| [word as u32, (word >> 32) as u32])
                .collect::<Vec<_>>();
            let decoded = decode_stream(shadowbox::QuietRun, &code);
            let mut module = emit_unoptimized_with_options(
                &decoded,
                Some(&header),
                shadowbox::TranslationOptions {
                    float64: shadowbox::HostFloat64::Supported,
                    ..Default::default()
                },
            )
            .unwrap();
            compute::observe_group(&mut module, 1);
            let raw_module = module.words.clone();
            module.optimize();
            for (variant, words) in [("raw", &raw_module), ("optimized", &module.words)] {
                validate(
                    words,
                    &work().join(format!("double-rn-cc-{family}-{name}-{variant}.spv")),
                );
                for (case, actual) in gpu.run(words, &states).iter().enumerate() {
                    let expected = reference(raw, &states[case]).unwrap();
                    assert_eq!(
                        actual[flag], expected.words[flag],
                        "{family}/{name}/{variant}/case{case}/direct-flag"
                    );
                    assert_eq!(
                        actual[250],
                        if expected.words[flag] != 0 {
                            u32::MAX
                        } else {
                            0
                        },
                        "{family}/{name}/{variant}/case{case}"
                    );
                    comparisons += 1;
                }
            }
        }
    }
    eprintln!(
        "{}; double RN.CC: {comparisons} exact CSET Z/N observations, 0 differences",
        gpu.description
    );
}

#[test]
#[ignore = "RADV shared-memory compare-exchange; five exact ATOMS.CAS translations"]
fn gpu_shared_cas_five_exact() {
    let mut report = Report::new("shared-cas-five");
    for (name, raw) in [
        ("ATOMS_CAS_B32_OFFSET_0", 0xee40_0000_0047_0201),
        ("ATOMS_CAS_B32_OFFSET_4", 0xee40_0000_4087_0307),
        ("ATOMS_CAS_B32_OFFSET_12", 0xee40_0000_c107_050d),
        ("ATOMS_CAS_B32_OFFSET_28", 0xee40_0001_c187_0715),
        ("ATOMS_CAS_B32_OFFSET_60", 0xee40_0003_c207_091f),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shared-memory compare-exchange; five alias/offset ATOMS.CAS translations"]
fn gpu_shared_cas_alias_offset_five_exact() {
    let mut report = Report::new("shared-cas-alias-offset-five");
    for (name, raw) in [
        ("ATOMS_CAS_B32_DST_COMPARE_ALIAS", 0xee40_0000_0047_0204),
        ("ATOMS_CAS_B32_DST_ADDRESS_ALIAS", 0xee40_0000_4087_0303),
        ("ATOMS_CAS_B32_COMPARE_ADDRESS_ALIAS", 0xee40_0000_c067_060c),
        ("ATOMS_CAS_B32_HIGH_COMPARATOR_PAIR", 0xee40_0001_cfd7_0814),
        ("ATOMS_CAS_B32_NEGATIVE_OFFSET", 0xee4f_ffff_c287_0a1e),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shared-memory compare-exchange; six guard/RZ/offset-edge ATOMS.CAS translations"]
fn gpu_shared_cas_guard_rz_edge_six_exact() {
    let mut report = Report::new("shared-cas-guard-rz-edge-six");
    for (name, raw) in [
        ("ATOMS_CAS_B32_GUARD_P0", 0xee40_0000_80a0_0306),
        ("ATOMS_CAS_B32_GUARD_NOT_P2", 0xee40_0001_00ea_050c),
        (
            "ATOMS_CAS_B32_GUARD_P1_DST_REPLACEMENT_ALIAS",
            0xee40_0000_4081_0209,
        ),
        ("ATOMS_CAS_B32_RZ_DESTINATION", 0xee40_0001_4087_04ff),
        ("ATOMS_CAS_B32_MAX_OFFSET", 0xee47_ffff_c0c7_0602),
        ("ATOMS_CAS_B32_MIN_OFFSET", 0xee48_0000_0107_0703),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderBufferInt64Atomics; six 64-bit global ATOM/RED translations"]
fn gpu_global_atomic64_six_exact() {
    let mut report = Report::new("global-atomic64-six");
    for (name, raw) in [
        ("ATOM_E_ADD_U64", 0xed05_0001_0067_0204),
        ("ATOM_E_MIN_S64_NEGATIVE_OFFSET", 0xed1b_ffff_80a7_0208),
        ("ATOM_E_MAX_U64", 0xed25_0001_80e7_020c),
        (
            "ATOM_E_OR_U64_GUARD_P0_DST_DATA_ALIAS",
            0xed65_0000_8060_0206,
        ),
        ("ATOM_EXCH_U64_A32", 0xed84_0002_0127_0310),
        ("RED_E_ADD_U64", 0xebf9_0001_0027_0206),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderBufferInt64Atomics; next six 64-bit global ATOM/RED translations"]
fn gpu_global_atomic64_next_six_exact() {
    let mut report = Report::new("global-atomic64-next-six");
    for (name, raw) in [
        ("ATOM_E_MIN_U64", 0xed15_0001_0067_0204),
        ("ATOM_E_MAX_S64_NEGATIVE_OFFSET", 0xed2b_ffff_80a7_0208),
        ("ATOM_E_AND_U64", 0xed55_0001_80e7_020c),
        ("ATOM_XOR_U64_A32", 0xed74_0002_0127_0310),
        ("RED_E_MIN_S64", 0xebf9_0001_00d7_0206),
        (
            "RED_E_XOR_U64_GUARD_P1_NEGATIVE_OFFSET",
            0xebf9_ffff_83a1_0208,
        ),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderBufferInt64Atomics; six 64-bit global RED translations"]
fn gpu_global_red64_six_exact() {
    let mut report = Report::new("global-red64-six");
    for (name, raw) in [
        ("RED_E_MIN_U64", 0xebf9_0001_00a7_0206),
        ("RED_E_MAX_U64_NEGATIVE_OFFSET", 0xebf9_ffff_8127_020a),
        ("RED_E_MAX_S64", 0xebf9_0001_8157_020e),
        ("RED_E_AND_U64", 0xebf9_0001_02a7_0208),
        ("RED_E_OR_U64_GUARD_P0", 0xebf9_0000_8320_0206),
        ("RED_ADD_U64_A32", 0xebf8_0002_0027_0312),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; five F2I.F64 translations"]
fn gpu_f2i64_five_exact() {
    let mut report = Report::new("f2i64-five");
    for (name, raw) in [
        ("F2I_F64_S32_RNE", 0x5cb0_0000_0027_1e01),
        ("F2I_F64_U32_RNE", 0x5cb0_0000_0027_0e01),
        ("F2I_F64_S32_FLOOR", 0x5cb0_0080_0027_1e01),
        ("F2I_F64_S32_CEIL", 0x5cb0_0100_0027_1e01),
        ("F2I_F64_S32_TRUNC", 0x5cb0_0180_0027_1e01),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; five F2I.F64 to 16-bit translations"]
fn gpu_f2i64_sixteen_five_exact() {
    let mut report = Report::new("f2i64-sixteen-five");
    for (name, raw) in [
        ("F2I_F64_S16_RNE", 0x5cb0_0000_0027_1d01),
        ("F2I_F64_U16_RNE", 0x5cb0_0000_0027_0d01),
        ("F2I_F64_S16_FLOOR", 0x5cb0_0080_0027_1d01),
        ("F2I_F64_S16_CEIL", 0x5cb0_0100_0027_1d01),
        ("F2I_F64_S16_TRUNC", 0x5cb0_0180_0027_1d01),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; six unsigned directed F2I.F64 translations"]
fn gpu_f2i64_unsigned_directed_six_exact() {
    let mut report = Report::new("f2i64-unsigned-directed-six");
    for (name, raw) in [
        ("F2I_F64_U32_FLOOR", 0x5cb0_0080_0027_0e01),
        ("F2I_F64_U32_CEIL", 0x5cb0_0100_0027_0e01),
        ("F2I_F64_U32_TRUNC", 0x5cb0_0180_0027_0e01),
        ("F2I_F64_U16_FLOOR", 0x5cb0_0080_0027_0d01),
        ("F2I_F64_U16_CEIL", 0x5cb0_0100_0027_0d01),
        ("F2I_F64_U16_TRUNC", 0x5cb0_0180_0027_0d01),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; six F2I.F64 source-modifier translations"]
fn gpu_f2i64_source_modifier_six_exact() {
    let mut report = Report::new("f2i64-source-modifier-six");
    for (name, raw) in [
        ("F2I_F64_S32_RNE_NEG", 0x5cb0_2000_0027_1e01),
        ("F2I_F64_S32_RNE_ABS", 0x5cb2_0000_0027_1e01),
        ("F2I_F64_S32_RNE_ABS_NEG", 0x5cb2_2000_0027_1e01),
        ("F2I_F64_S32_TRUNC_NEG", 0x5cb0_2180_0027_1e01),
        ("F2I_F64_U32_FLOOR_ABS", 0x5cb2_0080_0027_0e01),
        ("F2I_F64_U32_TRUNC_NEG", 0x5cb0_2180_0027_0e01),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV shaderFloat64; six F2I.F64 constant-buffer translations"]
fn gpu_f2i64_cbuf_six_exact() {
    // The constant-buffer pairs are fixed project values, so each word has
    // one source value; the states vary only the untouched registers.
    let mut report = Report::new("f2i64-cbuf-six");
    for (name, raw) in [
        ("F2I_F64_S32_RNE_CBUF", 0x4cb0_0000_0127_1e01),
        ("F2I_F64_U32_TRUNC_CBUF", 0x4cb0_01e4_06e7_0e01),
        ("F2I_F64_S16_FLOOR_CBUF", 0x4cb0_00b8_09a7_1d01),
        ("F2I_F64_U16_CEIL_CBUF", 0x4cb0_0164_0ac7_0d01),
        ("F2I_F64_S32_RNE_CBUF_ABS_NEG", 0x4cb2_2034_0407_1e01),
        ("F2I_F64_U32_FLOOR_CBUF_ABS", 0x4cb2_00a4_05e7_0e01),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; five exact STL cache-policy translations"]
fn gpu_stl_cache_five_exact() {
    let mut report = Report::new("stl-cache-five");
    for (name, raw) in [
        ("STL_CG_U8", 0xef50_1000_0007_0814),
        ("STL_CS_S8", 0xef51_2000_0007_0814),
        ("STL_WT_B32", 0xef54_3000_0007_0814),
        ("STL_CG_B64", 0xef55_1000_0007_0814),
        ("STL_CS_B128", 0xef56_2000_0007_0814),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; next five exact STL cache-policy translations"]
fn gpu_stl_cache_next_five_exact() {
    let mut report = Report::new("stl-cache-next-five");
    for (name, raw) in [
        ("STL_CS_U8", 0xef50_2000_0007_0814),
        ("STL_WT_U8", 0xef50_3000_0007_0814),
        ("STL_CG_S8", 0xef51_1000_0007_0814),
        ("STL_WT_S8", 0xef51_3000_0007_0814),
        ("STL_CG_U16", 0xef52_1000_0007_0814),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; third five exact STL cache-policy translations"]
fn gpu_stl_cache_third_five_exact() {
    let mut report = Report::new("stl-cache-third-five");
    for (name, raw) in [
        ("STL_CS_U16", 0xef52_2000_0007_0814),
        ("STL_WT_U16", 0xef52_3000_0007_0814),
        ("STL_CG_S16", 0xef53_1000_0007_0814),
        ("STL_CS_S16", 0xef53_2000_0007_0814),
        ("STL_WT_S16", 0xef53_3000_0007_0814),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; five exact wide STL cache-policy translations"]
fn gpu_stl_cache_wide_five_exact() {
    let mut report = Report::new("stl-cache-wide-five");
    for (name, raw) in [
        ("STL_CG_B32", 0xef54_1000_0007_0814),
        ("STL_CS_B32", 0xef54_2000_0007_0814),
        ("STL_CS_B64", 0xef55_2000_0007_0814),
        ("STL_WT_B64", 0xef55_3000_0007_0814),
        ("STL_CG_B128", 0xef56_1000_0007_0814),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; six exact LDC.IL translations"]
fn gpu_ldc_il_six_exact() {
    let mut report = Report::new("ldc-il-six");
    for (name, raw) in [
        ("LDC_IL_U8", 0xef90_1000_0007_0000),
        ("LDC_IL_S8", 0xef91_1000_0007_0000),
        ("LDC_IL_U16", 0xef92_1000_0007_0000),
        ("LDC_IL_S16", 0xef93_1000_0007_0000),
        ("LDC_IL_B32", 0xef94_1000_0007_0000),
        ("LDC_IL_B64", 0xef95_1000_0007_0000),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; five exact LDC.IL operand translations"]
fn gpu_ldc_il_operands_five_exact() {
    let mut report = Report::new("ldc-il-operands-five");
    for (name, raw) in [
        ("LDC_IL_U8_OPERANDS", 0xef90_1000_0037_0201),
        ("LDC_IL_S8_OPERANDS", 0xef91_1000_0017_0304),
        ("LDC_IL_U16_OPERANDS", 0xef92_1000_0027_0406),
        ("LDC_IL_S16_OPERANDS", 0xef93_1000_0067_0508),
        ("LDC_IL_B32_OPERANDS", 0xef94_1000_0087_060a),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; next five exact LDC.IL operand translations"]
fn gpu_ldc_il_operands_next_five_exact() {
    let mut report = Report::new("ldc-il-operands-next-five");
    for (name, raw) in [
        ("LDC_IL_U8_OPERANDS_NEXT", 0xef90_1000_0047_070c),
        ("LDC_IL_S8_OPERANDS_NEXT", 0xef91_1000_0057_080e),
        ("LDC_IL_U16_OPERANDS_NEXT", 0xef92_1000_0067_0910),
        ("LDC_IL_S16_OPERANDS_NEXT", 0xef93_1000_0077_0a12),
        ("LDC_IL_B64_OPERANDS_NEXT", 0xef95_1000_0087_0b14),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; five exact LDC.IS segmented-address translations"]
fn gpu_ldc_is_five_exact() {
    let mut report = Report::new("ldc-is-five");
    for (name, raw) in [
        ("LDC_IS_U8", 0xef90_2000_0037_0201),
        ("LDC_IS_S8", 0xef91_2000_0017_0304),
        ("LDC_IS_U16", 0xef92_2000_0027_0406),
        ("LDC_IS_B32", 0xef94_2000_0087_060a),
        ("LDC_IS_B64", 0xef95_2000_0087_0b14),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; six exact LDC.ISL packed-address translations"]
fn gpu_ldc_isl_six_exact() {
    let mut report = Report::new("ldc-isl-six");
    for (name, raw) in [
        ("LDC_ISL_U8", 0xef90_3000_0007_0000),
        ("LDC_ISL_S8", 0xef91_3000_0007_0000),
        ("LDC_ISL_U16", 0xef92_3000_0007_0000),
        ("LDC_ISL_S16", 0xef93_3000_0007_0000),
        ("LDC_ISL_B32", 0xef94_3000_0007_0000),
        ("LDC_ISL_B64", 0xef95_3000_0007_0000),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; five exact LDC.ISL operand translations"]
fn gpu_ldc_isl_operands_five_exact() {
    let mut report = Report::new("ldc-isl-operands-five");
    for (name, raw) in [
        ("LDC_ISL_U8_OPERANDS_A", 0xef90_7b7b_c947_9d94),
        ("LDC_ISL_U8_OPERANDS_B", 0xef90_b3f4_86f7_3aae),
        ("LDC_ISL_S8_OPERANDS", 0xef91_7b7b_c947_9d94),
        ("LDC_ISL_B32_OPERANDS", 0xef94_7b7b_c947_9d94),
        ("LDC_ISL_B64_OPERANDS", 0xef95_b3f4_86f7_3aae),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; six SHL opcode aliases"]
fn gpu_shl_alias_six_exact() {
    let mut report = Report::new("shl-alias-six");
    for (name, raw) in [
        ("SHL_ALIAS_REGISTER_CLAMP", 0x5c4a_0733_d3f7_4209),
        ("SHL_ALIAS_REGISTER_WRAP", 0x5c4a_07b3_d3f7_4209),
        ("SHL_ALIAS_CBUF_CLAMP", 0x4c4a_037a_0367_2672),
        ("SHL_ALIAS_CBUF_WRAP", 0x4c4a_03fa_0367_2672),
        ("SHL_ALIAS_IMMEDIATE_CLAMP", 0x3849_4406_e117_07a6),
        ("SHL_ALIAS_IMMEDIATE_WRAP", 0x3849_4486_e117_07a6),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; cbuf PCNT aliases"]
fn gpu_pcnt_cbuf_exact() {
    let mut report = Report::new("pcnt-cbuf");
    for alias in 0..16_u64 {
        report.check(
            "PCNT_CBUF_P007061",
            0xe2b0_0000_0000_0020 | alias << 48,
            true,
        );
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; direct PCNT aliases"]
fn gpu_pcnt_direct_aliases_exact() {
    let mut report = Report::new("pcnt-direct");
    for alias in 1..16_u64 {
        report.check(
            "PCNT_DIRECT_P007061_P007064",
            0xe2b0_0000_0000_0000 | alias << 48,
            true,
        );
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; compact MOV.I20 payload and quad-mask matrix"]
fn gpu_mov_i20_exact() {
    let mut report = Report::new("mov-i20-exact");
    for payload in [0_u32, 1, 0x7ffff, 0x80000, 0xfffff] {
        for mask in 0_u64..=0xf {
            let raw = (0x3898_u64 << 48)
                | (u64::from(payload >> 19) << 56)
                | (u64::from(payload & 0x7ffff) << 20)
                | (mask << 39)
                | (7 << 16)
                | 1;
            report.check("MOV_I20", raw, true);
        }
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware and spirv-val required; writes exact comparison evidence"]
fn gpu_family_exact() {
    let names = FAMILIES
        .iter()
        .map(|family| family.name)
        .collect::<BTreeSet<_>>();
    assert_eq!(names.len(), FAMILIES.len(), "duplicate family labels");
    let samples = std::env::var("SHADOWBOX_GPU_SAMPLES").map_or(8, |s| s.parse::<usize>().unwrap());
    let filter = std::env::var("SHADOWBOX_GPU_FILTER").ok();
    let prefix = std::env::var("SHADOWBOX_GPU_PREFIX").ok();
    let replay = std::env::var("SHADOWBOX_GPU_RAW")
        .ok()
        .map(|s| u64::from_str_radix(s.trim_start_matches("0x"), 16).unwrap());
    let mut report = Report::new("families");
    for (index, family) in FAMILIES.iter().enumerate() {
        if filter.as_ref().is_some_and(|f| family.name != f) {
            continue;
        }
        if prefix.as_ref().is_some_and(|p| !family.name.starts_with(p)) {
            continue;
        }
        let base = representative(*family);
        report.check(family.name, replay.unwrap_or(base), family.required);
        if replay.is_some() {
            break;
        }
        // Preserve the full opcode prefix and randomise all 48 low bits,
        // exactly as the fuzz gate does. Inputs remain runtime values.
        let mut rng = Rng(SEED ^ base | 1);
        for _ in 0..samples {
            report.check(
                family.name,
                base & 0xffff_0000_0000_0000 | rng.next() & 0x0000_ffff_ffff_ffff,
                false,
            );
        }
        if filter.is_none() && (index + 1) % 64 == 0 {
            eprintln!("checked {}/{} families", index + 1, FAMILIES.len());
        }
    }
    report.finish();
}

#[test]
#[ignore = "RADV scalar floating controls and directed rounding"]
fn gpu_float_modes_exact() {
    let mut report = Report::new("float-modes");
    // FMUL fields: Mesa sm50.rs:467-475,679-700;
    // envytools gm107.c:1331-1345. Cross all scale and rounding modes,
    // operand forms, saturation and FTZ. Keep the original runtime inputs.
    for (form, opcode) in [("R", 0x5c68_u64), ("C", 0x4c68), ("I", 0x3868)] {
        for round in 0..4_u64 {
            for scale in 0..7_u64 {
                for ftz in 0..2_u64 {
                    for sat in 0..2_u64 {
                        let operand = if form == "I" { 0x3f000 } else { 3 };
                        let raw = opcode << 48
                            | 7 << 16
                            | 2 << 8
                            | 1
                            | operand << 20
                            | round << 39
                            | scale << 41
                            | ftz << 44
                            | sat << 50;
                        report.check(
                            &format!("FMUL_{form}_{round}_{scale}_{ftz}_{sat}"),
                            raw,
                            true,
                        );
                    }
                }
            }
        }
    }
    // FTZ and SAT must affect both immediate arithmetic and fused results.
    // Mesa sm50.rs:496-527,558-602,665-677 places their fields.
    for (form, opcode) in [
        ("FADD", 0x5c58_u64),
        ("FFMA", 0x5980),
        ("FADD32I", 0x0800),
        ("FMUL32I", 0x1e00),
    ] {
        for ftz in 0..2_u64 {
            for sat in 0..2_u64 {
                if form == "FADD32I" && sat != 0 {
                    continue;
                }
                let raw = opcode << 48
                    | 7 << 16
                    | 2 << 8
                    | 1
                    | if form.ends_with("32I") {
                        0x3f00_0000_u64 << 20
                    } else {
                        3 << 20
                    }
                    | if form == "FFMA" { 4 << 39 } else { 0 }
                    | ftz
                        << if form == "FFMA" || form == "FMUL32I" {
                            53
                        } else if form == "FADD32I" {
                            55
                        } else {
                            44
                        }
                    | sat << if form == "FMUL32I" { 55 } else { 50 };
                report.check(&format!("{form}_{ftz}_{sat}"), raw, true);
            }
        }
    }
    // FFMA DNZ product: Mesa sm50.rs:601-602; nir_opcodes.py:1213-1225.
    // Mode3 is reserved (envytools gm107.c:1516-1520), checked by decoder tests.
    for sat in 0..2_u64 {
        let raw = 0x5980_u64 << 48 | 7 << 16 | 2 << 8 | 1 | 3 << 20 | 4 << 39 | 1 << 54 | sat << 50;
        report.check(&format!("FFMA_DNZ_{sat}"), raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV conversion matrix; uses the shared exact compute runner"]
fn gpu_conversion_exact() {
    let filter = std::env::var("SHADOWBOX_GPU_FILTER").ok();
    let mut report = Report::new("conversions");
    // Fields: sm50.rs:1806-1922; gm107.c:1200-1225,1256-1307.
    // Every implemented size/sign/rounding/modifier value is represented.
    // Byte selection cycles over valid offsets. This is not the Cartesian
    // product of byte selections, source values and all modifier combinations.
    for (name, register, cb, immediate) in [
        ("I2I", 0x5ce0_u64, 0x4ce0, 0x38e0),
        ("I2F", 0x5cb8, 0x4cb8, 0x38b8),
        ("F2I", 0x5cb0, 0x4cb0, 0x38b0),
    ] {
        if filter.as_ref().is_some_and(|f| name != f) {
            continue;
        }
        for (form, prefix) in [("R", register), ("C", cb), ("I", immediate)] {
            for src in 0..=2_u64 {
                for dst in 0..=2_u64 {
                    if name != "I2I" && (dst == 0 || name == "F2I" && src == 0) {
                        continue;
                    }
                    for signs in 0..4_u64 {
                        if name != "I2I" && signs >= 2 {
                            continue;
                        }
                        for modifiers in 0..if name == "I2I" {
                            16
                        } else if name == "I2F" {
                            4
                        } else {
                            8
                        } {
                            for rounding in 0..if name == "F2I" { 4 } else { 1 } {
                                let byte = modifiers % (5 - (1 << src));
                                let mut raw = prefix << 48
                                    | 7 << 16
                                    | 1
                                    | dst << 8
                                    | src << 10
                                    | (modifiers & 1) << 49
                                    | ((modifiers >> 1) & 1) << 45;
                                if name == "I2I" {
                                    raw |= signs << 12
                                        | byte << 41
                                        | ((modifiers >> 2) & 1) << 50
                                        | ((modifiers >> 3) & 1) << 47;
                                } else if name == "I2F" {
                                    raw |= signs << 13 | byte << 41;
                                } else {
                                    raw |= signs << 12 | rounding << 39;
                                    if src == 1 {
                                        raw |= ((modifiers >> 2) & 1) << 41;
                                    } else {
                                        raw |= ((modifiers >> 2) & 1) << 44;
                                    }
                                }
                                // Register source R3; cbuf source word3;
                                // immediate source encodes 3, with both signs.
                                raw |= 3 << 20;
                                report.check(&format!("{name}_{form}"), raw, false);
                            }
                        }
                    }
                }
            }
        }
    }
    report.finish();
}

#[test]
#[ignore = "check the observation transport before family comparisons"]
fn gpu_observer_preserves_bits_and_guards() {
    let mut report = Report::new("transport");
    for raw in [
        0x50b0_u64 << 48 | 7 << 16,
        0x5c98_u64 << 48 | 3 << 20 | 15 << 39 | 7 << 16 | 1,
        0x5c98_u64 << 48 | 3 << 20 | 15 << 39 | 7 << 16 | 1 << 19 | 1,
        0x0100_u64 << 48 | 0x7fc1_2345 << 20 | 15 << 12 | 7 << 16 | 1,
    ] {
        report.check("transport", raw, true);
    }
    report.finish();
}

#[test]
fn independent_memory_control_and_lane_anchors() {
    // These anchors are calculated from the cited raw-bit semantics, without
    // running the decoder, translator, optimiser or GPU.
    let mut state = [0; WORDS];
    state[263] = 1;
    state[268 + 8] = 0x80ff_7f01;
    // LDS.S16 at byte32, sm50.rs:2595-2615,2386-2397.
    let load = 0xef4b_u64 << 48 | 7 << 16 | 2 << 8 | 1;
    memory_reference::prepare(load, 0, &mut state).unwrap();
    assert_eq!(
        memory_reference::reference(load, 0, &state).unwrap().words[1],
        0x7f01
    );
    // STS.S8 writes only one byte, sm50.rs:2652-2669.
    state[1] = 0xab;
    let store = 0xef59_u64 << 48 | 7 << 16 | 2 << 8 | 1;
    memory_reference::prepare(store, 0, &mut state).unwrap();
    let stored = memory_reference::reference(store, 0, &state).unwrap();
    assert_eq!(stored.words[268 + 8], 0x80ff_7fab);
    assert_eq!(stored.words[268 + 9], state[268 + 9]);
    // ATOMS.XOR returns the old value and writes the new memory word.
    // sm50.rs:2800-2834, InstEmitMemory.cs:202-305.
    state[3] = 0xffff;
    let atom = 0xec70_u64 << 48 | 3 << 20 | 7 << 16 | 2 << 8 | 1;
    memory_reference::prepare(atom, 0, &mut state).unwrap();
    let result = memory_reference::reference(atom, 0, &state).unwrap();
    assert_eq!(result.words[1], 0x80ff_7f01);
    assert_eq!(result.words[268 + 8], 0x80ff_80fe);
    // BRA to the second diamond arm and conditional EXIT.F fallthrough.
    // sm50.rs:3020-3030,3101-3109; InstEmitFlowControl.cs:153-186,279-310.
    let branch = 0xe240_u64 << 48 | 24 << 20 | 7 << 16 | 15;
    assert_eq!(
        control_reference::reference(&control_reference::program(branch), &state)
            .unwrap()
            .words[240],
        2
    );
    let exit = 0xe300_u64 << 48 | 7 << 16;
    assert_eq!(
        control_reference::reference(&control_reference::program(exit), &state)
            .unwrap()
            .words[240],
        1
    );
    // A 32-lane downward shuffle with immediate index1 and clamp31.
    // sm50.rs:2043-2080; HelperFunctionManager.cs:272-297.
    let states = (0..32)
        .map(|lane| {
            let mut state = state;
            state[2] = lane;
            state
        })
        .collect::<Vec<_>>();
    let shuffle = 0xef10_u64 << 48 | 31 << 34 | 2 << 30 | 3 << 28 | 1 << 20 | 7 << 16 | 2 << 8 | 1;
    let result = lane_reference::reference(shuffle, &states).unwrap();
    assert_eq!(result[0].words[1], 1);
    assert_eq!(result[0].words[256], 1);
    assert_eq!(result[31].words[1], 31);
    assert_eq!(result[31].words[256], 0);
    // Ryujinx InstDecoders.cs:163, InstEmitMove.cs:86-88: YDirection
    // is raw float1, not an integer or compute-specific zero value.
    let y_direction = 0xf0c8_u64 << 48 | 0x12 << 20 | 7 << 16 | 1;
    let result = lane_reference::reference(y_direction, &states).unwrap();
    assert!(result.iter().all(|result| result.words[1] == 0x3f80_0000));
    // Four enabled quad positions, single-lane mask. sm50.rs:1926-1953.
    let mov = 0x5c98_u64 << 48 | 1 << 39 | 2 << 20 | 7 << 16 | 1;
    let result = lane_reference::reference(mov, &states).unwrap();
    assert_eq!(result[4].words[1], 4);
    assert_eq!(result[5].words[1], states[5][1]);

    // Ordinary VOTE is one Maxwell 32-lane warp. Its ballot register and
    // predicate result come from the same active-lane state.
    for ballot in [0_u32, 1, 0x5555_5555, 0x7fff_ffff, u32::MAX] {
        let states = (0..32)
            .map(|lane| {
                let mut state = state;
                state[256 + 3] = (ballot >> lane) & 1;
                state
            })
            .collect::<Vec<_>>();
        for mode in 0..3_u64 {
            let raw = (0x50d8_u64 | mode) << 48 | 2 << 45 | 3 << 39 | 7 << 16 | 1;
            let result = lane_reference::reference(raw, &states).unwrap();
            let vote = match mode {
                0 => ballot == u32::MAX,
                1 => ballot != 0,
                2 => ballot == 0 || ballot == u32::MAX,
                _ => unreachable!(),
            };
            assert!(result.iter().all(
                |result| result.words[1] == ballot && result.words[256 + 2] == u32::from(vote)
            ));
        }
    }
}

#[test]
fn independent_fcmp_ftz_preserves_selected_payload_anchors() {
    // A/B/C fields and FTZ47: Mesa nv50_ir_emit_gm107.cpp:1537-1571,
    // with FTZ defined at :474-477. Ryujinx InstEmitFloatComparison.cs:
    // 369-375 selects raw A/B, but :163-204 omits the encoded FTZ modifier.
    // Mesa nv50_ir_peephole.cpp:2009-2014 also preserves equal A/B as MOV.
    let mut state = [0; WORDS];
    state[263] = 1;
    state[2] = 0x0000_0123;
    state[3] = 0x8000_0456;
    state[4] = 0x8000_0789;
    for (opcode, operand, false_bits) in [
        (0x5ba0_u64, 3_u64, state[3]),
        (0x4ba0, 0, 0x0000_1000), // c[0][0], compute.rs cbuf fixture.
        (0x36a0, 0x12345, 0x1234_5000),
        (0x53a0, 0, state[4]),
    ] {
        state[4] = 0x8000_0789;
        for (test, eq_without_ftz, eq_with_ftz) in [
            (0x0000_0001, false, true),
            (0x8000_0001, false, true),
            (0x007f_ffff, false, true),
            (0x807f_ffff, false, true),
            (0x0080_0000, false, false),
            (0x8080_0000, false, false),
            (0x0000_0000, true, true),
            (0x8000_0000, true, true),
            (0x7fc1_2345, false, false),
        ] {
            if opcode != 0x53a0 {
                state[4] = test;
            }
            // Swapped form compares c[0][0] = 0x1000, while B remains R4.
            let (eq_without_ftz, eq_with_ftz) = if opcode == 0x53a0 {
                (false, true)
            } else {
                (eq_without_ftz, eq_with_ftz)
            };
            for (comparison, equal) in [(2_u64, true), (13, false)] {
                for (ftz, is_zero) in [(0_u64, eq_without_ftz), (1, eq_with_ftz)] {
                    let raw = (opcode | comparison) << 48
                        | ftz << 47
                        | 4 << 39
                        | operand << 20
                        | 7 << 16
                        | 2 << 8
                        | 1;
                    let result = reference::reference(raw, &state).unwrap();
                    let expected = if is_zero == equal {
                        state[2]
                    } else {
                        false_bits
                    };
                    assert_eq!(result.words[1], expected, "raw={raw:016x} C={test:08x}");
                    assert_eq!(result.nan[1], 0);
                }
            }
        }
    }
}

#[test]
fn independent_exit_selector_anchors() {
    // Ryujinx Decoders/InstDecoders.cs:104-111 puts CSM at24..29,
    // RLE at30 and RGT at31. InstEmitConditionCode.cs:80-81 tests N/Z.
    let mut state = [0; WORDS];
    state[263] = 1;
    state[240] = 0x1234;
    state[264] = 1;
    let result = |selector| {
        let raw = 0xe300_u64 << 48 | 7 << 16 | selector;
        control_reference::reference(&control_reference::program(raw), &state)
    };
    assert_eq!(result(30).unwrap().words[240], 0x1234);
    assert_eq!(result(31).unwrap().words[240], 1);
    assert!(result(24).is_err());
    assert!(result(25).is_err());
}

#[test]
#[ignore = "RADV hardware; focused field and predicate-alias comparisons"]
fn gpu_fix_a_semantic_edges() {
    let mut report = Report::new("fix-a-edges");
    // sm50.rs:1931-1951: each MOV form and all quad masks. These hold
    // register2 / cbuf word3 / literal bits constant while masks vary.
    for mask in 0..16_u64 {
        for (label, raw) in [
            (
                "MOV_EDGE_R",
                0x5c98_u64 << 48 | mask << 39 | 2 << 20 | 7 << 16 | 1,
            ),
            (
                "MOV_EDGE_C",
                0x4c98_u64 << 48 | mask << 39 | 2 << 34 | 3 << 20 | 7 << 16 | 1,
            ),
            (
                "MOV_EDGE_I",
                0x0100_u64 << 48 | 0x7fc12345_u64 << 20 | 7 << 16 | mask << 12 | 1,
            ),
        ] {
            report.check(label, raw, true);
        }
    }
    // PSETP fields: sm50.rs:2090-2100; fold ir.rs:5326-5330.
    // Guard P0 aliases the first destination, or P1 aliases the second.
    // All AB/C logical modes include the AND/OR complement distinction.
    for ab in 0..3_u64 {
        for c in 0..3_u64 {
            for guard in 0..2_u64 {
                for inverted in 0..2_u64 {
                    let raw = 0x5090_u64 << 48
                        | c << 45
                        | 4 << 39
                        | 3 << 29
                        | ab << 24
                        | inverted << 19
                        | guard << 16
                        | 2 << 12
                        | 1;
                    report.check("PSETP_GUARD_ALIAS", raw, true);
                }
            }
        }
    }
    // VOTE fields sm50.rs:3233-3244; Translator.cs:359-393 applies
    // the guard before collecting lanes. The result overwrites its guard.
    for mode in 0..3_u64 {
        for inverted in 0..2_u64 {
            let raw = (0x50d8_u64 | mode) << 48 | 1 << 39 | inverted << 19 | 1;
            report.check("VOTE_GUARD_ALIAS", raw, true);
        }
    }
    for selector in [30_u64, 31] {
        report.check("EXIT_RLE_RGT", 0xe300_u64 << 48 | 7 << 16 | selector, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; complete PSET opcode span"]
fn gpu_pset_opcode_span_exact() {
    let mut report = Report::new("pset-opcode-span");
    for (name, raw) in [
        ("PSET_5089_P016605", 0x5089_0b54_58f7_25c7),
        ("PSET_508A_P016606", 0x508a_0986_a887_3188),
        ("PSET_508B_P016607", 0x508b_0304_4c67_7bf2),
        ("PSET_508C_P016608", 0x508c_0952_2047_1f77),
        ("PSET_508D_P016609", 0x508d_0174_1437_72b6),
        ("PSET_508E_P016610", 0x508e_0278_50b7_4a91),
        ("PSET_508F_SOURCE_ROW_ENDPOINT", 0x508f_0278_50b7_4a91),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; exact PBK cbuf words with no matching BRK"]
fn gpu_pbk_cbuf_exact() {
    let mut report = Report::new("pbk-cbuf");
    for alias in 0_u64..16 {
        report.check(
            "PBK_CBUF_P007066",
            0xe2a0_0000_0000_0020 | alias << 48,
            true,
        );
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; exact VOTE warp and R2P bank comparisons"]
fn gpu_vote_r2p_exact() {
    let mut report = Report::new("vote-r2p-exact");
    for mode in 0..3_u64 {
        for inverted in 0..2_u64 {
            let raw = (0x50d8_u64 | mode) << 48 | 2 << 45 | 3 << 39 | inverted << 42 | 7 << 16 | 1;
            report.check("VOTE_WARP", raw, true);
        }
    }
    let predicate_forms = [
        0x5cf0_u64 << 48 | 3 << 20 | 7 << 16 | 2 << 8,
        0x4cf0_u64 << 48 | 1 << 20 | 7 << 16 | 2 << 8,
        0x38f0_u64 << 48 | 0x7f << 20 | 7 << 16 | 2 << 8,
    ];
    for (form, raw) in predicate_forms.into_iter().enumerate() {
        report.check("R2P_PREDICATES", raw | (form as u64 % 4) << 41, true);
        for byte in [0_u64, 3] {
            report.check("R2P_CC", raw | 1 << 40 | byte << 41, true);
        }
    }
    report.finish();
}

// Exhaustive accepted scalar modes and masks, and vector dimensions/LOD/masks.
// This matrix is generated from the cited encoding fields, never the decoder.
#[test]
#[ignore = "RADV hardware; bounded TEX exact matrix"]
fn gpu_texture_tex_exact() {
    gpu_texture_forms_exact("TEX");
}

#[test]
#[ignore = "RADV hardware; dynamic cbuf texture-array index"]
fn gpu_tex_sample_cbuf_array_exact() {
    let mut report = Report::new("tex-sample-cbuf-array");
    let raw = 0xdeb8_u64 << 48 | 15 << 31 | 2 << 28 | 1 << 20 | 7 << 16 | 10 << 8;
    report.check("TEX_SAMPLE_CBUF_ARRAY", raw, true);
    report.finish();
}

#[test]
#[ignore = "RADV hardware; bounded TLD exact matrix"]
fn gpu_texture_tld_exact() {
    gpu_texture_forms_exact("TLD");
}

#[test]
#[ignore = "RADV hardware; bounded TLD4 exact matrix"]
fn gpu_texture_tld4_exact() {
    gpu_texture_forms_exact("TLD4");
}

#[test]
#[ignore = "RADV hardware; bounded TEXS exact matrix"]
fn gpu_texture_texs_exact() {
    gpu_texture_forms_exact("TEXS");
}

#[test]
#[ignore = "RADV hardware; bounded TLDS exact matrix"]
fn gpu_texture_tlds_exact() {
    gpu_texture_forms_exact("TLDS");
}

#[test]
#[ignore = "RADV hardware; bounded TLD4S exact matrix"]
fn gpu_texture_tld4s_exact() {
    gpu_texture_forms_exact("TLD4S");
}

#[test]
#[ignore = "RADV hardware; bounded TXQ exact matrix"]
fn gpu_texture_txq_exact() {
    gpu_texture_forms_exact("TXQ");
}

fn gpu_texture_forms_exact(family: &str) {
    let mut report = Report::new(&format!("texture-forms-{family}"));
    let filter = std::env::var("SHADOWBOX_GPU_FILTER").ok();
    let mut check = |name: String, raw| {
        if name.split('_').next() == Some(family)
            && filter.as_ref().is_none_or(|f| name.contains(f))
        {
            report.check(&name, raw, true);
        }
    };
    // Mesa sm50.rs:2105-2149,2169-2197; TEX.B, A8/B16/R32.
    for dim in [0u64, 1, 2, 3, 4, 6, 7] {
        for lod in 0..4 {
            for mask in 1..16 {
                let raw =
                    0xdeb8_u64 << 48 | 8 << 8 | 16 << 20 | 32 | dim << 28 | mask << 31 | lod << 37;
                check(format!("TEX_dim{dim}_lod{lod}_mask{mask}"), raw);
            }
            if dim <= 4 {
                let raw = 0xdeb8_u64 << 48
                    | 8 << 8
                    | 16 << 20
                    | 32
                    | dim << 28
                    | 15 << 31
                    | lod << 37
                    | 1 << 36;
                check(format!("TEX_offset_dim{dim}_lod{lod}"), raw);
            }
            if dim != 4 {
                let raw = 0xdeb8_u64 << 48
                    | 8 << 8
                    | 16 << 20
                    | 32
                    | dim << 28
                    | 15 << 31
                    | lod << 37
                    | 1 << 50;
                check(format!("TEX_depth_dim{dim}_lod{lod}"), raw);
            }
        }
        if dim <= 4 {
            // Bound/bindless TLD, sm50.rs:2206-2236. RZ B for unused LZ.
            for bound in [false, true] {
                for lod in [false, true] {
                    for offset in [false, true] {
                        for mask in 1..16 {
                            let b = if bound && !lod && !offset { 255 } else { 16 };
                            let raw = (if bound { 0xdc38_u64 } else { 0xdd38 }) << 48
                                | 8 << 8
                                | b << 20
                                | 32
                                | dim << 28
                                | mask << 31
                                | u64::from(lod) << 55
                                | u64::from(offset) << 35;
                            check(
                                format!(
                                    "TLD_dim{dim}_bound{bound}_lod{lod}_offset{offset}_mask{mask}"
                                ),
                                raw,
                            );
                        }
                    }
                }
            }
        }
        if matches!(dim, 2 | 3 | 6 | 7) {
            // Bound/bindless gather, sm50.rs:2245-2278.
            for bound in [false, true] {
                for component in 0..4 {
                    for offset in [false, true] {
                        if offset && dim >= 6 {
                            continue;
                        }
                        let b = if bound && !offset { 255 } else { 16 };
                        let raw = (if bound { 0xc838_u64 } else { 0xdef8 }) << 48
                            | 8 << 8
                            | b << 20
                            | 32
                            | dim << 28
                            | 15 << 31
                            | component << (if bound { 56 } else { 38 })
                            | u64::from(offset) << (if bound { 54 } else { 36 });
                        check(
                            format!(
                                "TLD4_dim{dim}_bound{bound}_offset{offset}_component{component}"
                            ),
                            raw,
                        );
                    }
                }
                let raw = (if bound { 0xc838_u64 } else { 0xdef8 }) << 48
                    | 8 << 8
                    | 16 << 20
                    | 32
                    | dim << 28
                    | 15 << 31
                    | 1 << 50;
                check(format!("TLD4_depth_dim{dim}_bound{bound}"), raw);
            }
        }
    }
    // envytools gm107.c:932-1008,1967-1968, scalar mode and both mask tables.
    for fetch in [false, true] {
        for mode in if fetch {
            vec![0u64, 1, 2, 4, 5, 7, 8, 12]
        } else {
            (0u64..14).collect()
        } {
            for half in [false, true] {
                for pair in [false, true] {
                    for selector in 0..if pair { 5 } else { 8 } {
                        let b = if mode == 0 { 255 } else { 16 };
                        let raw = (if fetch { 0xd200_u64 } else { 0xd000 }) << 48
                            | u64::from(!half) << 59
                            | mode << 53
                            | selector << 50
                            | (if pair { 40 } else { 255 }) << 28
                            | b << 20
                            | 8 << 8
                            | 32;
                        check(
                            format!(
                                "{}_mode{mode}_half{half}_pair{pair}_mask{selector}",
                                if fetch { "TLDS" } else { "TEXS" }
                            ),
                            raw,
                        );
                    }
                }
            }
        }
    }
    // TLD4S encodings: envytools gm107.c:1960,893-898. Output mask 3 or 15.
    for half in [false, true] {
        for pair in [false, true] {
            for offset in [false, true] {
                for depth in [false, true] {
                    for component in 0..if depth { 1 } else { 4 } {
                        let raw = 0xdf00_u64 << 48
                            | u64::from(half) << 55
                            | component << 52
                            | u64::from(offset) << 51
                            | u64::from(depth) << 50
                            | (if pair { 40 } else { 255 }) << 28
                            | 16 << 20
                            | 8 << 8
                            | 32;
                        check(format!("TLD4S_half{half}_pair{pair}_offset{offset}_depth{depth}_component{component}"),raw);
                    }
                }
            }
        }
    }
    // TXQ query 1 mask selects image shape. sm50.rs:2345-2381;
    // Ryujinx InstEmitTexture.cs:1105-1114,1124-1146.
    for mask in 1u64..16 {
        check(
            format!("TXQ_dimension_mask{mask}"),
            0xdf50_u64 << 48 | 1 << 22 | mask << 31 | 8 << 8 | 32,
        );
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; bounded SULD exact matrix"]
fn gpu_surface_suld_exact() {
    gpu_surface_forms_exact("SULD");
}

#[test]
#[ignore = "RADV hardware; bounded SUST exact matrix"]
fn gpu_surface_sust_exact() {
    gpu_surface_forms_exact("SUST");
}

#[test]
#[ignore = "RADV hardware; bounded SUATOM exact matrix"]
fn gpu_surface_suatom_exact() {
    gpu_surface_forms_exact("SUATOM");
}

#[test]
#[ignore = "RADV hardware; SUATOM register handle with PT result matrix"]
fn gpu_surface_suatom_pt_exact() {
    gpu_surface_forms_exact("SUATOMPT");
}

#[test]
#[ignore = "RADV hardware; bounded SURED exact matrix"]
fn gpu_surface_sured_exact() {
    gpu_surface_forms_exact("SURED");
}

fn gpu_surface_forms_exact(family: &str) {
    let mut report = Report::new(&format!("surface-forms-{family}"));
    let filter = std::env::var("SHADOWBOX_GPU_FILTER").ok();
    let mut check = |name: String, raw| {
        if name.split('_').next() == Some(family)
            && filter.as_ref().is_none_or(|f| name.contains(f))
        {
            report.check(&name, raw, true);
        }
    };
    // SULD/SUST fields: envytools gm107.c:794-838,1909-1910;
    // Mesa sm50.rs:2471-2529. Bounded unique X per case avoids write races.
    for target in 0u64..6 {
        for store in [false, true] {
            for bound in [false, true] {
                for raw in [false, true] {
                    for selector in if raw { 0..8 } else { 1..16 } {
                        for byte in [false, true] {
                            if !raw && byte {
                                continue;
                            }
                            let instruction = (if store { 0xeb20_u64 } else { 0xeb00 }) << 48
                                | u64::from(raw) << 52
                                | u64::from(bound) << 51
                                | (if bound { 9 << 36 } else { 200 << 39 })
                                | target << 33
                                | selector << 20
                                | u64::from(byte) << 23
                                | 8 << 8
                                | 32;
                            check(format!("{}_dim{target}_bound{bound}_raw{raw}_selector{selector}_byte{byte}",
                                if store{"SUST"}else{"SULD"}),instruction);
                        }
                    }
                }
            }
        }
        // SUATOM bound/flexible register forms, sizes U32/S32 and CAS.
        // Mesa sm50.rs:2558-2591; envytools gm107.c:1833-1871,1911-1914.
        for bound in [false, true] {
            for signed in [false, true] {
                for byte in [false, true] {
                    for op in 0u64..10 {
                        let opcode = match (bound, op == 9) {
                            (true, false) => 0xea00u64,
                            (true, true) => 0xea80,
                            (false, false) => 0xea60,
                            (false, true) => 0xeac0,
                        };
                        let data = if matches!(op, 3 | 4) { 255 } else { 32 };
                        let raw = opcode << 48
                            | u64::from(signed) << (if bound { 51 } else { 36 })
                            | (if bound { 9 << 36 } else { 1 << 52 | 200 << 39 })
                            | target << 33
                            | (if op == 9 { 7 << 30 } else { op << 29 })
                            | u64::from(byte) << 28
                            | data << 20
                            | 8 << 8
                            | 40;
                        check(
                            format!(
                                "SUATOM_dim{target}_bound{bound}_signed{signed}_byte{byte}_op{op}"
                            ),
                            raw,
                        );
                    }
                }
            }
        }
        // Ryujinx InstTable.cs:281 and InstDecoders.cs:4611-4627: the
        // 0xeb80 B2 form has a register handle and predicate result. The
        // admitted slice encodes PT in bits51..53 and IGN in bits49..50.
        for signed in [false, true] {
            for byte in [false, true] {
                for op in 0u64..9 {
                    let data = if matches!(op, 3 | 4) { 255 } else { 32 };
                    let raw = 0xeb80_u64 << 48
                        | u64::from(signed) << 36
                        | 200 << 39
                        | target << 33
                        | op << 29
                        | u64::from(byte) << 28
                        | data << 20
                        | 8 << 8
                        | 40;
                    check(
                        format!("SUATOMPT_dim{target}_signed{signed}_byte{byte}_op{op}"),
                        raw,
                    );
                }
            }
        }
        // SURED's data is R0 field, size20..23, op24..27, BA23.
        // envytools gm107.c:769-791,1908; Ryujinx InstDecoders.cs:4740-4758.
        for bound in [false, true] {
            for signed in [false, true] {
                for byte in [false, true] {
                    for op in 0u64..8 {
                        let raw = 0xeb40_u64 << 48
                            | 1 << 52
                            | u64::from(bound) << 51
                            | (if bound { 9 << 36 } else { 200 << 39 })
                            | target << 33
                            | op << 24
                            | u64::from(signed) << 20
                            | u64::from(byte) << 23
                            | 8 << 8
                            | if matches!(op, 3 | 4) { 255 } else { 32 };
                        check(
                            format!(
                                "SURED_dim{target}_bound{bound}_signed{signed}_byte{byte}_op{op}"
                            ),
                            raw,
                        );
                    }
                }
            }
        }
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; bounded TLD exact matrix"]
fn gpu_texture_ms_tld_exact() {
    gpu_texture_multisample_exact("TLD");
}

#[test]
#[ignore = "RADV hardware; bounded TLDS exact matrix"]
fn gpu_texture_ms_tlds_exact() {
    gpu_texture_multisample_exact("TLDS");
}

#[test]
#[ignore = "RADV hardware; bounded TXQ exact matrix"]
fn gpu_texture_ms_txq_exact() {
    gpu_texture_multisample_exact("TXQ");
}

fn gpu_texture_multisample_exact(family: &str) {
    let mut report = Report::new(&format!("texture-ms-{family}"));
    let filter = std::env::var("SHADOWBOX_GPU_FILTER").ok();
    let mut check = |name: &str, raw, required| {
        if name.split('_').next() == Some(family)
            && filter.as_ref().is_none_or(|f| name.contains(f))
        {
            report.check(name, raw, required);
        }
    };
    // MS50, LOD55, TOFF35, dim28 and mask31: sm50.rs:2226-2236.
    for dim in [2u64, 3] {
        for bound in [false, true] {
            for lod in [false, true] {
                for offset in [false, true] {
                    for mask in 1u64..16 {
                        let raw = (if bound { 0xdc38_u64 } else { 0xdd38 }) << 48
                            | 1 << 50
                            | u64::from(lod) << 55
                            | u64::from(offset) << 35
                            | dim << 28
                            | mask << 31
                            | 16 << 20
                            | 8 << 8
                            | 32;
                        check(
                            &format!(
                                "TLD_MS_dim{dim}_bound{bound}_lod{lod}_offset{offset}_mask{mask}"
                            ),
                            raw,
                            true,
                        );
                    }
                }
            }
        }
    }
    // Scalar MS mode6: envytools gm107.c:938,951,1967.
    for half in [false, true] {
        for pair in [false, true] {
            for selector in 0u64..if pair { 5 } else { 8 } {
                let raw = 0xd200_u64 << 48
                    | u64::from(!half) << 59
                    | 6 << 53
                    | selector << 50
                    | (if pair { 40 } else { 255 }) << 28
                    | 16 << 20
                    | 8 << 8
                    | 32;
                check(
                    &format!("TLDS_MS_half{half}_pair{pair}_mask{selector}"),
                    raw,
                    true,
                );
            }
        }
    }
    // TXQ query2 defines sample count in Z only, InstEmitTexture.cs:1152-1182.
    check(
        "TXQ_samples",
        0xdf50_u64 << 48 | 2 << 22 | 4 << 31 | 8 << 8 | 32,
        true,
    );
    report.finish();
}

#[test]
#[ignore = "RADV hardware; packed offsets stay runtime register values"]
fn gpu_texture_runtime_offsets_exact() {
    let mut report = Report::new("texture-runtime");
    // Encodings: sm50.rs:2206-2236,2245-2278. Prefix sets only the handle,
    // leaving offsets as runtime input bits. Each axis visits -1/0/+1.
    for dim in 0u64..5 {
        for bound in [false, true] {
            for lod in [false, true] {
                let raw = (if bound { 0xdc38_u64 } else { 0xdd38 }) << 48
                    | u64::from(lod) << 55
                    | 1 << 35
                    | dim << 28
                    | 15 << 31
                    | 16 << 20
                    | 8 << 8
                    | 32;
                report.check(
                    &format!("TLD_dynamic_dim{dim}_bound{bound}_lod{lod}"),
                    raw,
                    true,
                );
            }
        }
    }
    for dim in [2u64, 3] {
        for bound in [false, true] {
            for depth in [false, true] {
                for component in 0..if depth { 1 } else { 4 } {
                    let raw = (if bound { 0xc838_u64 } else { 0xdef8 }) << 48
                        | component << (if bound { 56 } else { 38 })
                        | 1 << (if bound { 54 } else { 36 })
                        | u64::from(depth) << 50
                        | dim << 28
                        | 15 << 31
                        | 16 << 20
                        | 8 << 8
                        | 32;
                    report.check(
                        &format!(
                            "TLD4_dynamic_dim{dim}_bound{bound}_depth{depth}_component{component}"
                        ),
                        raw,
                        true,
                    );
                }
            }
            for lod in [false, true] {
                let raw = (if bound { 0xdc38_u64 } else { 0xdd38 }) << 48
                    | u64::from(lod) << 55
                    | 1 << 50
                    | 1 << 35
                    | dim << 28
                    | 15 << 31
                    | 16 << 20
                    | 8 << 8
                    | 32;
                report.check(
                    &format!("TLD_MS_dynamic_dim{dim}_bound{bound}_lod{lod}"),
                    raw,
                    true,
                );
            }
        }
    }
    for half in [false, true] {
        for mode in [4u64, 12] {
            let raw = 0xd200_u64 << 48
                | u64::from(!half) << 59
                | mode << 53
                | 4 << 50
                | 40 << 28
                | 16 << 20
                | 8 << 8
                | 32;
            report.check(&format!("TLDS_dynamic_half{half}_mode{mode}"), raw, true);
        }
        for pair in [false, true] {
            for depth in [false, true] {
                let raw = 0xdf00_u64 << 48
                    | u64::from(half) << 55
                    | 1 << 51
                    | u64::from(depth) << 50
                    | (if pair { 40 } else { 255 }) << 28
                    | 16 << 20
                    | 8 << 8
                    | 32;
                report.check(
                    &format!("TLD4S_dynamic_half{half}_pair{pair}_depth{depth}"),
                    raw,
                    true,
                );
            }
        }
    }
    report.finish();
}

#[test]
#[ignore = "Uncovered TMML fragment derivatives; required compute fixtures remain failing"]
fn gpu_texture_derivative_gap() {
    let mut report = Report::new("texture-derivative-gap");
    // TMML fields: Mesa sm50.rs:2287-2309. This compute runner cannot
    // establish fragment derivatives. Keep the required fixture visibly failing.
    for dim in [0u64, 1, 2, 3, 6, 7] {
        for bound in [false, true] {
            let raw = (if bound { 0xdf58_u64 } else { 0xdf60 }) << 48
                | dim << 28
                | 3 << 31
                | (if bound { 255 } else { 16 }) << 20
                | 8 << 8
                | 32;
            report.check(&format!("TMML_dim{dim}_bound{bound}"), raw, true);
        }
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; signed array layers and clamping"]
fn gpu_texture_array_layer_edges() {
    let mut report = Report::new("texture-layers");
    // Vector array layer precedes coordinates, InstEmitTexture.cs:203,247-249.
    // Conversion to float is signed, Spirv/Instructions.cs:1218-1225.
    for dim in [1u64, 3, 7] {
        for lod in 0u64..4 {
            for depth in [false, true] {
                let raw = 0xdeb8_u64 << 48
                    | lod << 37
                    | u64::from(depth) << 50
                    | dim << 28
                    | 15 << 31
                    | 16 << 20
                    | 8 << 8
                    | 32;
                report.check(
                    &format!("TEX_layer_edge_dim{dim}_lod{lod}_depth{depth}"),
                    raw,
                    true,
                );
            }
        }
    }
    // Scalar array modes and layouts: envytools gm107.c:983-985;
    // InstEmitTexture.cs:394-411.
    for mode in [7u64, 8, 9] {
        for half in [false, true] {
            let raw = 0xd000_u64 << 48
                | u64::from(!half) << 59
                | mode << 53
                | 4 << 50
                | 40 << 28
                | 16 << 20
                | 8 << 8
                | 32;
            report.check(&format!("TEXS_layer_edge_mode{mode}_half{half}"), raw, true);
        }
    }
    for dim in [3u64, 7] {
        for bound in [false, true] {
            for depth in [false, true] {
                let raw = (if bound { 0xc838_u64 } else { 0xdef8 }) << 48
                    | u64::from(depth) << 50
                    | dim << 28
                    | 15 << 31
                    | (if bound && !depth { 255 } else { 16 }) << 20
                    | 8 << 8
                    | 32;
                report.check(
                    &format!("TLD4_layer_edge_dim{dim}_bound{bound}_depth{depth}"),
                    raw,
                    true,
                );
            }
        }
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; every vector gather destination mask"]
fn gpu_texture_gather_masks_exact() {
    let mut report = Report::new("texture-gather-masks");
    // Dimensions/mask: Mesa sm50.rs:2105-2149,2245-2278.
    // Compact selected results: Ryujinx InstEmitTexture.cs:779-800.
    // Depth gather omits a component operand: InstEmitTexture.cs:773-776.
    // Use canonical component zero so no encoded field is ignored.
    for dim in [2u64, 3, 6, 7] {
        for bound in [false, true] {
            for depth in [false, true] {
                for mask in 0..16 {
                    let raw = (if bound { 0xc838_u64 } else { 0xdef8 }) << 48
                        | 8 << 8
                        | (if bound && !depth { 255 } else { 16 }) << 20
                        | 32
                        | dim << 28
                        | mask << 31
                        | u64::from(depth) << 50
                        | (if depth { 0 } else { 2 }) << (if bound { 56 } else { 38 });
                    report.check(
                        &format!("TLD4_mask_dim{dim}_bound{bound}_depth{depth}_mask{mask}"),
                        raw,
                        true,
                    );
                }
            }
        }
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; five exact TLD4 PTP forms"]
fn gpu_tld4_ptp_five_exact() {
    let mut report = Report::new("tld4-ptp-five");
    for (name, raw) in [
        ("bindless_2d_r", 0xdef8_0027_a107_0820),
        ("bindless_2da_g", 0xdef8_0067_b107_0820),
        ("bound_2d_b", 0xcab8_0097_a107_0820),
        ("bound_2da_a", 0xcbb8_0097_b107_0820),
        ("bindless_2d_a", 0xdef8_00e7_a107_0820),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; six more exact TLD4 PTP forms"]
fn gpu_tld4_ptp_six_exact() {
    let mut report = Report::new("tld4-ptp-six");
    for (name, raw) in [
        ("TLD4_B_BINDLESS_PTP_2D_MASK15", 0xdef8_00a7_a107_0820),
        ("TLD4_R_BINDLESS_PTP_ARRAY2D_MASK3", 0xdef8_0021_b107_0820),
        ("TLD4_R_PTP_2D_MASK5", 0xc8b8_0012_a107_0820),
        ("TLD4_G_NOT_P2_PTP_ARRAY2D_MASK15", 0xc9b8_0037_b10a_0820),
        ("TLD4_G_P3_BINDLESS_PTP_2D_MASK8", 0xdef8_0064_2103_0820),
        ("TLD4_A_PTP_ARRAY2D_MASK10", 0xcbb8_0025_3107_0820),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
fn zero_gather_masks_retain_the_image_operation() {
    // Maxwell dimensions/mask and handle: Mesa sm50.rs:2105-2149,2245-2278.
    // SPIR-V image gather opcodes: Mesa src/compiler/spirv/spirv.h:1824-1825.
    // Even with no destination channels, emission must retain the instruction.
    for dim in [2u64, 3, 6, 7] {
        for bound in [false, true] {
            for depth in [false, true] {
                let raw = (if bound { 0xc838_u64 } else { 0xdef8 }) << 48
                    | 8 << 8
                    | (if bound && !depth { 255 } else { 16 }) << 20
                    | 32
                    | dim << 28
                    | u64::from(depth) << 50;
                let code = texture_program(raw, false);
                let header = headers::parse(&[0; 32]).unwrap();
                let module = emit_unoptimized_with_header(
                    &decode_stream(shadowbox::QuietRun, &code),
                    Some(&header),
                )
                .unwrap();
                let mut at = 5;
                let mut retained = false;
                while at < module.words.len() {
                    let word = module.words[at];
                    retained |= Op::of(word)
                        == if depth {
                            Op::ImageDrefGather
                        } else {
                            Op::ImageGather
                        };
                    at += (word >> 16) as usize;
                }
                assert!(retained, "gather was dropped: {raw:016x}");
            }
        }
    }
}

#[test]
fn independent_main_family_oracle_anchors() {
    // BFI immediate and C base: Mesa gm107 emitter:2304-2312;
    // Ryujinx InstEmitBitfield.cs:53-61,154-162.
    let mut state = [0; WORDS];
    state[263] = 1;
    state[2] = 0x12345678;
    state[3] = 0xabcdef01;
    let bfi = 0x36f0u64 << 48 | 1 | 2 << 8 | 7 << 16 | 4 << 20 | 8 << 28 | 3 << 39;
    assert_eq!(reference(bfi, &state).unwrap().words[1], 0xabcde781);

    // RLE/RGT30/31 and separate complement: Ryujinx
    // InstDecoders.cs:104-111; InstEmitConditionCode.cs:35-47.
    state[264] = 1;
    state[256] = 1;
    let csetp = 0x50a0u64 << 48 | 1 | 30 << 8 | 7 << 16;
    let expected = reference(csetp, &state).unwrap();
    assert_eq!(&expected.words[256..258], &[1, 0]);

    // F2F ABS49/NEG45/SAT50, integer42/round39..40:
    // Mesa sm50.rs:1765-1796; Ryujinx InstEmitConversion.cs:121-162.
    state[3] = (-1.75f32).to_bits();
    let f2f = 0x5caa_u64 << 48 | 1 | 0xa << 8 | 7 << 16 | 3 << 20 | 1 << 42 | 3 << 39 | 1 << 49;
    assert_eq!(reference(f2f, &state).unwrap().words[1], 1.0f32.to_bits());
    state[3] = f32::NAN.to_bits();
    assert_eq!(reference(f2f | 1 << 50, &state).unwrap().words[1], 0);
}

#[test]
fn directed_fmul_without_float64_uses_nearest_fallback() {
    // Round39..40: Mesa src/nouveau/compiler/nak/sm50.rs:695-700.
    // Host Float64 gate: Ryujinx CodeGen/Spirv/SpirvGenerator.cs:58-60.
    // Nearest fallback: Ryujinx InstEmitFloatArithmetic.cs:435-480.
    let header = headers::parse(&[0; 32]).unwrap();
    for mode in 1..4_u64 {
        let raw = 0x5c68_u64 << 48 | mode << 39 | 3 << 20 | 7 << 16 | 2 << 8 | 1;
        let result = emit_unoptimized_with_options(
            &decode_stream(shadowbox::QuietRun, &program(raw)),
            Some(&header),
            shadowbox::TranslationOptions::default(),
        );
        let module = result.unwrap();
        let nearest = emit_unoptimized_with_options(
            &decode_stream(shadowbox::QuietRun, &program(raw & !(3 << 39))),
            Some(&header),
            shadowbox::TranslationOptions::default(),
        )
        .unwrap();
        assert_eq!(module.words, nearest.words);
        validate(&module.words, &work().join("nearest-validation.spv"));
    }
}

#[test]
#[ignore = "RADV checks F2F saturation and raw modifier bits"]
fn gpu_f2f_saturation_exact() {
    let mut report = Report::new("f2f-saturation");
    // Mesa sm50.rs:1765-1796 and envytools gm107.c:1997,2071:
    // sizes8..11, B20..27/cbuf20..38, integer42/round39..40,
    // FTZ44, NEG45, ABS49, SAT50. Ryujinx InstEmitConversion.cs:121-162.
    for opcode in [0x5ca8_u64, 0x4ca8] {
        for rounding in [None, Some(0), Some(1), Some(2), Some(3)] {
            for ftz in [false, true] {
                let raw = opcode << 48
                    | 1 << 50
                    | u64::from(ftz) << 44
                    | 3 << 20
                    | 7 << 16
                    | 0xa << 8
                    | 1
                    | rounding.map_or(0, |mode| 1 << 42 | mode << 39);
                report.check(&format!("F2F_SAT_{opcode:x}_{rounding:?}_{ftz}"), raw, true);
            }
        }
    }
    report.finish();
}

#[test]
#[ignore = "RADV checks six exact F2F immediate translations"]
fn gpu_f2f_immediate_six_exact() {
    let mut report = Report::new("f2f-immediate-six");
    for (name, raw) in [
        ("F2F_IMMEDIATE_PASS", 0x38a8_003f_c007_0a01),
        ("F2F_IMMEDIATE_FLOOR_NEGATIVE", 0x39a8_04c0_2007_0a01),
        ("F2F_IMMEDIATE_CEIL_NEG", 0x38a8_2540_2007_0a01),
        ("F2F_IMMEDIATE_ROUND_ABS", 0x39aa_0440_2007_0a01),
        ("F2F_IMMEDIATE_TRUNC_SAT", 0x38ac_05bf_c007_0a01),
        ("F2F_IMMEDIATE_FTZ_SUBNORMAL", 0x38a8_1000_4007_0a01),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV checks six exact F2F.F32.F16 translations"]
fn gpu_f2f_half_source_six_exact() {
    let mut report = Report::new("f2f-half-source-six");
    for (name, raw) in [
        ("F2F_F32_F16_H0", 0x5ca8_0000_0027_0601),
        ("F2F_F32_F16_NEG", 0x5ca8_2000_0027_0601),
        ("F2F_F32_F16_ABS", 0x5caa_0000_0027_0601),
        ("F2F_F32_F16_ABS_NEG", 0x5caa_2000_0027_0601),
        ("F2F_F32_F16_SAT", 0x5cac_0000_0027_0601),
        ("F2F_F32_F16_CBUF_NEG", 0x4ca8_2000_0037_0601),
    ] {
        report.check(name, raw, true);
    }
    report.finish();
}

#[test]
#[ignore = "RADV round-to-nearest fallback without declared Float64"]
fn gpu_fmul_nearest_fallback() {
    // Rounding39..40: Mesa sm50.rs:695-700. Nearest fallback:
    // Ryujinx InstEmitFloatArithmetic.cs:435-480.
    let gpu = Compute::new();
    let header = headers::parse(&[0; 32]).unwrap();
    let mut executions = 0;
    for mode in 1..4_u64 {
        let raw = 0x5c68_u64 << 48 | mode << 39 | 3 << 20 | 7 << 16 | 2 << 8 | 1;
        let cases = inputs(raw);
        let expected = cases
            .iter()
            .map(|state| reference(raw & !(3 << 39), state).unwrap())
            .collect::<Vec<_>>();
        let mut module = emit_unoptimized_with_options(
            &decode_stream(shadowbox::QuietRun, &program(raw)),
            Some(&header),
            shadowbox::TranslationOptions::default(),
        )
        .unwrap();
        compute::observe_group(&mut module, 1);
        let original = module.words.clone();
        module.optimize();
        for words in [&original, &module.words] {
            validate(words, &work().join("nearest-validation.spv"));
            let actual = gpu.run_images(words, &cases, false).0;
            for (case, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                for (slot, &bits) in actual.iter().enumerate() {
                    assert_eq!(
                        normalise(bits, expected.nan[slot]),
                        normalise(expected.words[slot], expected.nan[slot]),
                        "mode {mode}, case {case}, slot {slot}"
                    );
                }
            }
            executions += cases.len();
        }
    }
    eprintln!(
        "{}; {executions} nearest-fallback executions; 0 differing words",
        gpu.description
    );
}

#[test]
#[ignore = "RADV predicate alias and FMZ signed-zero replays"]
fn gpu_semantic_edges() {
    // Project-owned mutation replays from the first corrected family sweep.
    // HSETP2 outputs/guard: envytools gm107.c:1985,1973-1974,363-365,408-411;
    // semantics: Ryujinx Translator.cs:359-393, InstEmitFloatComparison.cs:494-523.
    // FMZ modes: envytools gm107.c:1157-1168; source swizzles/modifiers:
    // Ryujinx InstEmitHelper.cs:139-158, InstEmitAluHelper.cs:75-91.
    // Zero-product signs: docs/reference-sources.md.
    let mut report = Report::new("semantic-edges");
    for (family, raw) in [
        ("HSETP2_REG_1", 0x5d2281ee890cb164),
        ("HSETP2_IMM_1", 0x7e24133baf64c360),
        ("HSETP2_CBUF_1", 0x7ea4091e1e34aa65),
        ("HMUL2_REG_SWZ0_OUT0", 0x5d083f4ec379781c),
        ("HMUL2_REG_SWZ0_OUT2", 0x5d0ce554ce10901c),
        ("HMUL2_REG_SWZ1_OUT0", 0x5d081d4edee359ad),
        ("HMUL2_REG_SWZ1_OUT3", 0x5d0ee7128a17e71c),
        ("HMUL2_REG_SWZ2_OUT1", 0x5d0bf50ce5c92894),
        ("HMUL2_REG_SWZ3_OUT1", 0x5d0b457eabfe1342),
        ("HMUL2_REG_SWZ3_OUT2", 0x5d0d1d0aa7d94b34),
        ("HFMA2_REG_SWZ1_OUT2", 0x5d04c5c4c9353f9c),
    ] {
        report.check(family, raw, true);
    }
    report.finish();
}

// LEA low fields: envytools envydis/gm107.c:2022,2093,2140;
// Ryujinx Decoders/InstDecoders.cs:3581-3627. Independent result equation:
// Instructions/InstEmitIntegerArithmetic.cs:481-488, B +/- (A << shift).
#[test]
fn lea_low_exact_on_radv() {
    let gpu = Compute::new();
    let header = headers::parse(&[0; 32]).unwrap();
    let edges = [
        0,
        1,
        0x7fff_ffff,
        0x8000_0000,
        u32::MAX,
        0x1234_5678,
        0xaaaa_aaaa,
        0x5555_5555,
    ];
    let mut checked = 0;
    for shift in 0..32u64 {
        for negate in [false, true] {
            for form in 0..3 {
                let imm = [-524288i32, -1, 0, 1, 524287][shift as usize % 5] as u32;
                let encoded = match form {
                    0 => 0x5bd7u64 << 48 | 3 << 20,
                    1 => 0x4bd7u64 << 48 | 2 << 34 | 19 << 20,
                    _ => {
                        0x36d7u64 << 48
                            | u64::from(imm & 0x7ffff) << 20
                            | u64::from((imm >> 19) & 1) << 56
                    }
                };
                for guard in [7u64, 0, 8, 15] {
                    let raw =
                        encoded | guard << 16 | shift << 39 | u64::from(negate) << 45 | 2 << 8 | 1;
                    let xs = [
                        shadowbox::decoder::decode_word([raw as u32, (raw >> 32) as u32], 8),
                        shadowbox::decoder::decode_word([0x0007_000f, 0xe300_0000], 16),
                    ];
                    assert!(matches!(
                        xs[0].kind,
                        shadowbox::ir::InstructionKind::Lea(ops::Lea { .. })
                    ));
                    let mut module = emit_unoptimized_with_header(&xs, Some(&header)).unwrap();
                    compute::observe_group(&mut module, 1);
                    let validation = spirv_validation::validate(&module.words);
                    assert!(
                        validation.status.success(),
                        "{}",
                        String::from_utf8_lossy(&validation.stderr)
                    );
                    let cases: Vec<_> = (0..64)
                        .map(|i| {
                            let mut state = [0; WORDS];
                            state[1] = 0xdead_beef;
                            state[2] = edges[i % 8];
                            state[3] = edges[i / 8];
                            state[256] = (i % 2) as u32;
                            state[263] = 1; // PT is always true, envytools gm107.c:408-411.
                            for (j, flag) in state[264..268].iter_mut().enumerate() {
                                *flag = ((i >> j) & 1) as u32;
                            }
                            state
                        })
                        .collect();
                    let actual = gpu.run_images(&module.words, &cases, false).0;
                    for (input, output) in cases.iter().zip(actual) {
                        let b = match form {
                            0 => input[3],
                            1 => compute::cbuf(2, 19),
                            _ => imm,
                        };
                        let a = input[2].wrapping_shl(shift as u32);
                        let active = match guard {
                            7 => true,
                            0 => input[256] != 0,
                            8 => input[256] == 0,
                            _ => false,
                        };
                        let expected = if !active {
                            input[1]
                        } else if negate {
                            b.wrapping_sub(a)
                        } else {
                            b.wrapping_add(a)
                        };
                        assert_eq!(
                            output[1], expected,
                            "{raw:016x} A={:08x} B={b:08x}",
                            input[2]
                        );
                        assert_eq!(output[2], input[2]);
                        assert_eq!(output[3], input[3]);
                        assert_eq!(
                            &output[256..268],
                            &input[256..268],
                            "LEA must preserve predicates and CC"
                        );
                        checked += 1;
                    }
                }
            }
        }
    }
    eprintln!(
        "LEA low: {checked} exact register comparisons on {}",
        gpu.description
    );
}

// Mesa sm50.rs:2171-2236; envytools gm107.c:1962,1965-1966,1970.
// Operand order follows Ryujinx InstEmitTexture.cs:203-299. Outside fragment,
// lod/bias uses explicit LOD (CodeGen/Spirv/Instructions.cs:1339-1348).
#[test]
fn texture_vector_exact_on_radv() {
    use shadowbox::ir::{Instruction, InstructionKind as K, Register, Source};
    let gpu = Compute::new();
    let header = headers::parse(&[0; 32]).unwrap();
    let mut checked = 0;
    for fetch in [false, true] {
        for bound in [false, true] {
            for lod in 0..if fetch { 2u64 } else { 4u64 } {
                for offset in [false, true] {
                    for mask in 1..16u64 {
                        let b = if bound { 20 } else { 21 };
                        let has_lod = if fetch { lod == 1 } else { lod >= 2 };
                        let offset_reg = b + u8::from(has_lod);
                        let mut xs = Vec::new();
                        if !bound {
                            xs.push(Instruction {
                                address: 0,
                                predicate: None,
                                kind: K::Mov(ops::Mov {
                                    dst: Register(20),
                                    src: Source::Immediate(1),
                                    quad_lanes: 15,
                                }),
                            });
                        }
                        if offset && !fetch {
                            // Signed four-bit X=-1,Y=+1, gm107.c:1965 and
                            // InstEmitTexture.cs:274-276. Kept constant only
                            // for sampling; fetch also has a dynamic GPU check.
                            xs.push(Instruction {
                                address: xs.len() as u32 * 8,
                                predicate: None,
                                kind: K::Mov(ops::Mov {
                                    dst: Register(offset_reg),
                                    src: Source::Immediate(0x1f),
                                    quad_lanes: 15,
                                }),
                            });
                        }
                        let opcode = match (fetch, bound) {
                            (false, false) => 0xdeb8u64,
                            (false, true) => 0xc038,
                            (true, false) => 0xdd38,
                            (true, true) => 0xdc38,
                        };
                        // Bound implicit/LZ without offsets consumes no B:
                        // Ryujinx InstEmitTexture.cs:207-210,252-258.
                        let src_b = if bound && !has_lod && !offset {
                            255
                        } else {
                            20
                        };
                        let raw = opcode << 48
                            | 7 << 16
                            | mask << 31
                            | 2 << 28
                            | src_b << 20
                            | 2 << 8
                            | 8
                            | if bound { 9 << 36 } else { 0 }
                            | if fetch {
                                lod << 55 | u64::from(offset) << 35
                            } else if bound {
                                lod << 55 | u64::from(offset) << 54
                            } else {
                                lod << 37 | u64::from(offset) << 36
                            };
                        xs.push(shadowbox::decoder::decode_word(
                            [raw as u32, (raw >> 32) as u32],
                            xs.len() as u32 * 8,
                        ));
                        xs.push(shadowbox::decoder::decode_word(
                            [0x0007_000f, 0xe300_0000],
                            xs.len() as u32 * 8,
                        ));
                        let mut module = emit_unoptimized_with_header(&xs, Some(&header)).unwrap();
                        compute::observe_group(&mut module, 1);
                        let validation = spirv_validation::validate(&module.words);
                        assert!(
                            validation.status.success(),
                            "{}",
                            String::from_utf8_lossy(&validation.stderr)
                        );
                        let cases: Vec<_> = (0..32)
                            .map(|i| {
                                let mut state = [0; WORDS];
                                state[8..12].fill(0xdead_beef);
                                let level = if has_lod { i % 2 } else { 0 };
                                let size = 4 >> level;
                                let x = if offset { 1 } else { (i / 2) % size };
                                let y = if offset { 0 } else { (i / 8) % size };
                                state[2] = if fetch {
                                    x
                                } else {
                                    ((x as f32 + 0.5) / size as f32).to_bits()
                                };
                                state[3] = if fetch {
                                    y
                                } else {
                                    ((y as f32 + 0.5) / size as f32).to_bits()
                                };
                                state[b as usize] = if fetch {
                                    level
                                } else {
                                    (level as f32).to_bits()
                                };
                                if offset {
                                    state[offset_reg as usize] = 0x1f;
                                }
                                state[263] = 1;
                                state
                            })
                            .collect();
                        let actual = gpu.run_images(&module.words, &cases, false).0;
                        for (i, (input, output)) in cases.iter().zip(actual).enumerate() {
                            let level = if has_lod { i as u32 % 2 } else { 0 };
                            let size = 4 >> level;
                            let x = if offset { 0 } else { (i as u32 / 2) % size };
                            let y = if offset { 1 } else { (i as u32 / 8) % size };
                            let mut dst = 8;
                            for channel in 0..4 {
                                if mask & (1 << channel) != 0 {
                                    // Immutable image uploaded by support/texture.rs, no
                                    // production decoder or IR used in this value equation.
                                    let expected = ((1 + level * 128 + y * 32 + x * 4 + channel)
                                        as f32
                                        / 256.0)
                                        .to_bits();
                                    assert_eq!(
                                        output[dst], expected,
                                        "{raw:016x}, case={i}, channel={channel}"
                                    );
                                    dst += 1;
                                    checked += 1;
                                }
                            }
                            assert_eq!(&output[dst..12], &input[dst..12]);
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "TEX/TLD vector: {checked} exact channel comparisons on {}",
        gpu.description
    );
}

#[test]
fn independent_gap_oracle_anchors() {
    // ICMP signed GE: envytools gm107.c:2148,1365-1368,1452;
    // Ryujinx InstEmitIntegerComparison.cs:117-131 selects raw A/B bits.
    let mut state = [0; WORDS];
    state[263] = 1;
    state[2] = 0x7fc1_2345;
    state[3] = 0xdead_beef;
    let icmp = 0x5b4d_u64 << 48 | 4 << 39 | 3 << 20 | 7 << 16 | 2 << 8 | 1;
    assert_eq!(reference(icmp, &state).unwrap().words[1], 0x7fc1_2345);
    state[4] = u32::MAX;
    assert_eq!(reference(icmp, &state).unwrap().words[1], 0xdead_beef);
    let immediate = 0x364d_u64 << 48 | 4 << 39 | 3 << 20 | 7 << 16 | 2 << 8 | 1;
    assert_eq!(reference(immediate, &state).unwrap().words[1], 3);

    // BFI.RC swaps C into the range and cbuf into the base:
    // envytools gm107.c:2041; Ryujinx InstEmitBitfield.cs:75-83,154-162.
    state[4] = 0x0808;
    state[2] = 0x1234;
    let bfi = 0x53f0_u64 << 48 | 4 << 39 | 1 << 34 | 7 << 16 | 2 << 8 | 1;
    assert_eq!(reference(bfi, &state).unwrap().words[1], 0x0001_3401);
    let bfi_rz = bfi & !(255 << 39) | 255 << 39;
    assert_eq!(reference(bfi_rz, &state).unwrap().words[1], 0x0001_1001);

    // BRX next-PC + displacement: Ryujinx InstEmitFlowControl.cs:38-40.
    let brx = 0xe250_u64 << 48 | 7 << 16 | 2 << 8 | 15;
    for case in 0..2 {
        control_reference::prepare(brx, case, &mut state);
        assert_eq!(
            control_reference::reference(&control_reference::program(brx), &state)
                .unwrap()
                .words[240],
            case as u32 + 1
        );
    }
    // Fixed RZ LDG uses its signed immediate and a relocated allocation:
    // Mesa sm50.rs:2402-2410; Ryujinx InstEmitMemory.cs:518-541.
    let load = 0xeed4_47d5_0452_ff02;
    state[258] = 1;
    state[268 + 8] = 0x1234_5678;
    assert!(memory_reference::fixed_address(load));
    memory_reference::prepare(load, 0, &mut state).unwrap();
    assert_eq!(
        memory_reference::reference(load, 0, &state).unwrap().words[2],
        0x1234_5678
    );
}

#[test]
#[ignore = "RADV hardware; exact CC.F slice of the register BRX cubes"]
fn gpu_brx_condition_f_slice_exact() {
    const CUBES: [(u64, u64); 2] = [
        (0xe250_0000_0007_0000, 0x000f_0000_0008_0000),
        (0xe250_5bb1_b767_1200, 0x000f_0000_0008_0000),
    ];
    let mut report = Report::new("brx-condition-f");
    for (base, variable) in CUBES {
        let bits: Vec<u32> = (0..64).filter(|bit| variable & (1 << bit) != 0).collect();
        for selector in 0..(1_u16 << bits.len()) {
            let raw = bits.iter().enumerate().fold(base, |word, (index, bit)| {
                word | u64::from((selector >> index) & 1) << bit
            });
            report.check("BRX_CC_F_B8E0511D302035A9D", raw, true);
        }
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; exact CC.LT slice of the register BRX cubes"]
fn gpu_brx_condition_lt_slice_exact() {
    const CUBES: [(u64, u64); 2] = [
        (0xe250_0000_0007_0001, 0x000f_0000_0008_0000),
        (0xe250_5bb1_b767_1201, 0x000f_0000_0008_0000),
    ];
    let mut report = Report::new("brx-condition-lt");
    for (base, variable) in CUBES {
        let bits: Vec<u32> = (0..64).filter(|bit| variable & (1 << bit) != 0).collect();
        for selector in 0..(1_u16 << bits.len()) {
            let raw = bits.iter().enumerate().fold(base, |word, (index, bit)| {
                word | u64::from((selector >> index) & 1) << bit
            });
            report.check("BRX_CC_LT_B8E0511D302035A9D", raw, true);
        }
    }
    report.finish();
}

#[test]
#[ignore = "RADV hardware; resolved BRX target regions from strict-cache case 03"]
fn gpu_brx_resolved_reconvergence_exact() {
    use shadowbox::{
        decoder::decode_word,
        ir::{BrxBase, Instruction, InstructionKind, Register},
    };

    const SITE: u32 = 0x1c90;
    const RAW: u64 = 0xe250_0ffe_3687_3f0f;
    const LOADED: [u32; 3] = [0x1c98, 0x1da0, 0x1ea0];
    const TARGETS: [u32; 3] = [0x1c98, 0x1da8, 0x1ea8];
    let move_immediate = |address: u32, value: u32| {
        let raw = 0x0100_u64 << 48 | u64::from(value) << 20 | 15 << 12 | 7 << 16 | 240;
        decode_word([raw as u32, (raw >> 32) as u32], address)
    };
    let vote = |address: u32| {
        let raw = 0x50d8_u64 << 48 | 2 << 45 | 1 << 39 | 7 << 16 | 50;
        decode_word([raw as u32, (raw >> 32) as u32], address)
    };
    let exit = |address: u32| {
        let raw = 0xe300_u64 << 48 | 7 << 16 | 15;
        decode_word([raw as u32, (raw >> 32) as u32], address)
    };

    let base = BrxBase::decode(SITE, ((RAW >> 20) & 0xff_ffff) as u32);
    let mut xs = vec![Instruction {
        address: SITE,
        predicate: None,
        kind: InstructionKind::brx(Register(63), base, LOADED[0], &LOADED[1..]),
    }];
    for (index, target) in TARGETS.into_iter().enumerate() {
        xs.push(move_immediate(target, index as u32 + 1));
        xs.push(vote(target + 8));
        xs.push(exit(target + 16));
    }
    xs.sort_by_key(|instruction| instruction.address);

    let header = headers::parse(&[0; 32]).unwrap();
    let mut module = emit_unoptimized_with_header(&xs, Some(&header)).unwrap();
    assert!(module.requires_maximal_reconvergence);
    compute::observe_group(&mut module, 1);
    let raw_module = module.words.clone();
    module.optimize();

    let mut states = inputs(RAW);
    for (case, state) in states.iter_mut().enumerate() {
        state[63] = LOADED[case % LOADED.len()];
        state[257] = u32::from(case.is_multiple_of(2));
        state[263] = 1;
    }
    let gpu = Compute::new();
    let report = work().join("brx-resolved-reconvergence");
    fs::create_dir_all(&report).unwrap();
    let mut executions = 0;
    for (variant, words) in [("raw", &raw_module), ("optimized", &module.words)] {
        validate(words, &report.join(format!("{variant}.spv")));
        for (case, actual) in gpu.run(words, &states).iter().enumerate() {
            assert_eq!(actual[240], case as u32 % 3 + 1, "{variant}/case{case}");
            executions += 1;
        }
    }
    fs::write(
        report.join("coverage.tsv"),
        "site\tword\traw\toptimized\texecutions\tdifferences\tmissing\n\
         0x1c90\te2500ffe36873f0f\tPASS\tPASS\t96\t0\t0\n",
    )
    .unwrap();
    fs::write(report.join("complete.txt"), format!("1\t{executions}\t0\n")).unwrap();
    assert_eq!(executions, 96);
}

#[test]
#[ignore = "RADV hardware; oracle gaps and defined input domains"]
fn gpu_gap_semantic_edges() {
    let mut report = Report::new("gap-edges");
    for family in FAMILIES {
        if matches!(
            family.name,
            "BRX_REGISTER"
                | "BFI RC base c[7][0x14] control RZ"
                | "ICMP_IMM_SIGNED_GE"
                | "BFE register"
                | "BFE constant-buffer"
                | "LDG_CI"
                | "TXQ_BOUND_DIMENSION_WITH_SHAPE"
        ) || family.name.starts_with("TEXS_")
            || family.name.starts_with("TLDS_")
            || family.name.starts_with("TLD4S_")
            || family.name.starts_with("TEX_BOUND_")
        {
            let base = representative(*family);
            report.check(family.name, base, family.required);
            let mut rng = Rng(SEED ^ base | 1);
            for _ in 0..8 {
                report.check(
                    family.name,
                    base & 0xffff_0000_0000_0000 | rng.next() & 0x0000_ffff_ffff_ffff,
                    false,
                );
            }
        }
    }
    report.finish();
}

#[test]
fn bfe_cbuf_domains_preserve_defined_values_and_exercise_extraction() {
    // Range bytes: Ryujinx InstEmitBitfield.cs:141-147; Mesa sm50.rs:1174-1179.
    // These properties cover the whole supplied cbuf input domain, independent
    // of the generator's particular offset/count construction.
    for bank in 0..32 {
        for word in 0..16384 {
            let old = compute::cbuf(bank, word);
            let new = compute::bfe_cbuf(bank, word);
            let old_offset = old & 255;
            let old_width = old >> 8 & 255;
            let offset = new & 255;
            let width = new >> 8 & 255;
            assert!(offset + width <= 32);
            assert_eq!(new & !65535, old & !65535);
            if old_offset + old_width <= 32 {
                assert_eq!(new, old);
            } else {
                assert!(width > 0);
            }
        }
    }
}

// Runtime BFI controls: Mesa nv50_ir_emit_gm107.cpp:2293-2326;
// envytools gm107.c:2019,2092,2041,203-207. Ryujinx
// Instructions/InstEmitBitfield.cs:42-83,154-162 defines the packed bytes.
#[test]
fn bfi_runtime_controls_exact_on_radv() {
    use shadowbox::ir::InstructionKind as K;
    let gpu = Compute::new();
    let header = headers::parse(&[0; 32]).unwrap();
    let controls = (0..=32u32)
        .flat_map(|offset| (0..=32 - offset).map(move |count| 0xdead_0000 | count << 8 | offset))
        .collect::<Vec<_>>();
    assert_eq!(controls.len(), 561);
    let edges = [
        0,
        1,
        u32::MAX,
        0x8000_0000,
        0x7fff_ffff,
        0x1234_5678,
        0xaaaa_aaaa,
        0x5555_5555,
    ];
    let mut checked = 0;
    for form in 0..3 {
        // R has range R3 and base R4; RC swaps cbuf into base and R3 into
        // control39..46. C has range c[19][0xfffc] and base R4.
        // Mesa gm107 emitter:2297-2302,2312,2315-2317.
        let encoding = match form {
            0 => 0x5bf0u64 << 48 | 3 << 20 | 4 << 39,
            1 => 0x53f0u64 << 48 | 19 << 34 | 0x3fff << 20 | 3 << 39,
            _ => 0x4bf0u64 << 48 | 19 << 34 | 0x3fff << 20 | 4 << 39,
        };
        for (dst, guard, exhaustive) in [
            (1u64, 0u64, true),
            (2, 7, false),
            (3, 8, false),
            (4, 7, false),
            (255, 15, false),
        ] {
            let raw = encoding | guard << 16 | 2 << 8 | dst;
            let xs = decode_stream(shadowbox::QuietRun, &program(raw));
            assert!(
                matches!(xs[0].kind, K::BitFieldInsert(ops::BitFieldInsert { .. })),
                "{raw:016x}"
            );
            let mut module = emit_unoptimized_with_header(&xs, Some(&header)).unwrap();
            compute::observe_group(&mut module, 1);
            let original = module.words.clone();
            module.optimize();
            let chosen = if exhaustive {
                controls.clone()
            } else {
                vec![0, 0xdead_2000, 0xdead_0804, 0xdead_011f, 0xdead_0020]
            };
            for words in [&original, &module.words] {
                validate(words, &work().join("bfi-validation.spv"));
                let make_state = |control: u32, case: usize| {
                    let mut state = [0; WORDS];
                    // Distinct initial words reveal every unintended write,
                    // even when D aliases an input or is RZ.
                    for (slot, value) in state.iter_mut().enumerate() {
                        *value = (slot as u32)
                            .wrapping_mul(0x9e37_79b9)
                            .rotate_left(case as u32);
                    }
                    state[2] = edges[case];
                    state[3] = control;
                    state[4] = edges[7 - case];
                    state[255] = 0;
                    for (p, value) in state[256..263].iter_mut().enumerate() {
                        *value = ((case >> (p % 3)) & 1) as u32;
                    }
                    state[263] = 1;
                    for (flag, value) in state[264..268].iter_mut().enumerate() {
                        *value = ((case >> (flag % 3)) & 1) as u32;
                    }
                    state
                };
                let batches = if form == 2 {
                    // Constant-buffer contents change between dispatches,
                    // without changing the translated instruction.
                    chosen
                        .iter()
                        .map(|&c| (vec![c], Some((19, 0x3fff, c))))
                        .collect::<Vec<_>>()
                } else {
                    chosen.chunks(8).map(|c| (c.to_vec(), None)).collect()
                };
                for (batch, cbuf_override) in batches {
                    let cases = batch
                        .iter()
                        .flat_map(|&c| (0..8).map(move |case| make_state(c, case)))
                        .collect::<Vec<_>>();
                    let actual = gpu
                        .run_images_config(
                            words,
                            &cases,
                            false,
                            cbuf_override,
                            compute::GUEST_BASE,
                            false,
                        )
                        .0;
                    for (input, output) in cases.iter().zip(actual) {
                        let control = input[3];
                        let offset = control & 255;
                        let count = control >> 8 & 255;
                        // Independent 64-bit mask equation, not decoded IR
                        // or the emitted SPIR-V operation. All 561 defined
                        // offset/count pairs are included without exclusions.
                        let mask = (((1u64 << count) - 1) << offset) as u32;
                        let base = if form == 1 {
                            compute::cbuf(19, 0x3fff)
                        } else {
                            input[4]
                        };
                        let value =
                            (base & !mask) | ((u64::from(input[2]) << offset) as u32 & mask);
                        let mut expected = *input;
                        let active = match guard {
                            0 => input[256] != 0,
                            7 => true,
                            8 => input[256] == 0,
                            _ => false,
                        };
                        if active && dst != 255 {
                            expected[dst as usize] = value;
                        }
                        assert_eq!(output, expected, "{raw:016x}, control={control:08x}");
                        checked += WORDS;
                    }
                }
            }
        }
    }
    eprintln!("BFI runtime R/RC/C: {checked} exact state words, all 561 defined ranges, raw and optimized, on {}", gpu.description);
}

// TXQ: Mesa sm50.rs:2351-2381; envytools gm107.c:1958,882-891.
// Driver shape and no-LOD MS size query: Ryujinx InstEmitTexture.cs:1116-1146
// and CodeGen/Spirv/Instructions.cs:1493-1515.
#[test]
fn txq_multisample_dimensions_exact_on_radv() {
    use shadowbox::{
        emitter::emit_unoptimized_with_graphics_state,
        interface::{BoundTexture, GraphicsState, TextureDimension},
    };
    let gpu = Compute::new();
    let header = headers::parse(&[0; 32]).unwrap();
    let mut checked = 0;
    for arrayed in [false, true] {
        let bindings = [BoundTexture {
            binding: 9,
            dimension: if arrayed {
                TextureDimension::Array2D
            } else {
                TextureDimension::D2
            },
            depth_compare: false,
            multisample: true,
        }];
        let state = GraphicsState {
            texture_bindings: &bindings,
            ..Default::default()
        };
        for mask in 1..if arrayed { 8u64 } else { 4u64 } {
            for guard in [0u64, 7, 8, 15] {
                for nodep in [false, true] {
                    let raw = 0xdf48u64 << 48
                        | 9 << 36
                        | mask << 31
                        | 1 << 22
                        | 255 << 8
                        | guard << 16
                        | u64::from(nodep) << 49
                        | 32;
                    let mut module = emit_unoptimized_with_graphics_state(
                        &decode_stream(shadowbox::QuietRun, &program(raw)),
                        Some(&header),
                        &state,
                    )
                    .unwrap();
                    compute::observe_group(&mut module, 1);
                    let original = module.words.clone();
                    module.optimize();
                    let cases = inputs(raw);
                    for words in [&original, &module.words] {
                        validate(words, &work().join("txq-ms-validation.spv"));
                        let actual = gpu.run_images(words, &cases, false).0;
                        for (input, output) in cases.iter().zip(actual) {
                            let mut expected = *input;
                            let active = match guard {
                                0 => input[256] != 0,
                                7 => true,
                                8 => input[256] == 0,
                                _ => false,
                            };
                            if active {
                                // These dimensions come from the Vulkan image
                                // allocation in support/texture_ms.rs:16,23-29,
                                // independently of the decoder and emitter.
                                let mut dst = 32;
                                for (axis, size) in [4, 4, 3].into_iter().enumerate() {
                                    if mask & (1 << axis) != 0 {
                                        expected[dst] = size;
                                        dst += 1;
                                    }
                                }
                            }
                            assert_eq!(output, expected, "{raw:016x}, array={arrayed}");
                            checked += WORDS;
                        }
                    }
                }
            }
        }
        // Level counts, missing axes and noncanonical unused A remain
        // rejected. These had no accepted MS dimension path on main.
        for (mask, src) in [(8u64, 255u64), (3, 8)] {
            let raw = 0xdf48u64 << 48 | 9 << 36 | mask << 31 | 1 << 22 | src << 8 | 7 << 16 | 32;
            assert!(emit_unoptimized_with_graphics_state(
                &decode_stream(shadowbox::QuietRun, &program(raw)),
                Some(&header),
                &state
            )
            .is_err());
        }
    }
    eprintln!(
        "TXQ MS dimensions: {checked} exact state words, raw and optimized, on {}",
        gpu.description
    );
}

// Query2 selected X/Y registers retain their values; selected Z follows them.
// Ryujinx InstEmitTexture.cs:1152-1178; fields Mesa sm50.rs:2351-2381,
// envytools gm107.c:1958-1959,882-891.
#[test]
fn txq_sample_count_skipped_destinations_exact_on_radv() {
    use shadowbox::{
        emitter::emit_unoptimized_with_graphics_state,
        interface::{BoundTexture, GraphicsState, TextureDimension},
    };
    let gpu = Compute::new();
    let header = headers::parse(&[0; 32]).unwrap();
    let mut checked = 0;
    for (bound, arrayed) in [(false, false), (true, false), (true, true)] {
        let bindings = [BoundTexture {
            binding: 9,
            dimension: if arrayed {
                TextureDimension::Array2D
            } else {
                TextureDimension::D2
            },
            depth_compare: false,
            multisample: true,
        }];
        let state = GraphicsState {
            texture_bindings: &bindings,
            ..Default::default()
        };
        for mask in 4..8u64 {
            for guard in [0u64, 7, 8, 15] {
                for dst in [32u64, 254, 255] {
                    let raw = (if bound { 0xdf48u64 } else { 0xdf50 }) << 48
                        | (if bound { 9 << 36 } else { 0 })
                        | mask << 31
                        | 2 << 22
                        | (if bound { 255 } else { 8 }) << 8
                        | guard << 16
                        | dst;
                    let code = texture_program(raw, false);
                    let mut module = emit_unoptimized_with_graphics_state(
                        &decode_stream(shadowbox::QuietRun, &code),
                        Some(&header),
                        &state,
                    )
                    .unwrap();
                    assert!(
                        compute::ops(&module.words).any(|(op, _)| op == Op::ImageQuerySamples),
                        "query must execute, even at RZ"
                    );
                    compute::observe_group(&mut module, 1);
                    let original = module.words.clone();
                    module.optimize();
                    let mut cases = inputs(raw);
                    if !bound {
                        for input in &mut cases {
                            input[8] = 1;
                        }
                    }
                    for words in [&original, &module.words] {
                        validate(words, &work().join("txq-samples-validation.spv"));
                        let actual = gpu.run_images(words, &cases, false).0;
                        for (input, output) in cases.iter().zip(actual) {
                            let mut expected = *input;
                            let active = match guard {
                                0 => input[256] != 0,
                                7 => true,
                                8 => input[256] == 0,
                                _ => false,
                            };
                            let z = dst + (mask & 3).count_ones() as u64;
                            // The fixture independently creates a four-sample
                            // image, support/texture_ms.rs:30.
                            if active && z < 255 {
                                expected[z as usize] = 4;
                            }
                            assert_eq!(output, expected, "{raw:016x}, array={arrayed}");
                            checked += WORDS;
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "TXQ sample masks: {checked} exact state words, raw and optimized, on {}",
        gpu.description
    );
}
