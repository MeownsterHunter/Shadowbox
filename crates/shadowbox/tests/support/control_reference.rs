// Bounded raw-instruction execution for branch/exit/reconvergence observations.
// Encoding: Mesa src/nouveau/compiler/nak/sm50.rs:3004-3109;
// envytools envydis/gm107.c:1877,1925,1933-1945.
use super::compute::WORDS;
use super::reference::{field, reference as alu_reference, Expected};

pub fn is_control(raw: u64) -> bool {
    matches!(
        raw >> 48,
        0xe240 | 0xe250..=0xe25f | 0xe290..=0xe29f | 0xe300 | 0xf0f8
    )
}
pub fn condition(code: u32, state: &[u32; WORDS]) -> Result<bool, &'static str> {
    // Ryujinx Instructions/InstEmitConditionCode.cs:52-84.
    let z = state[264] != 0;
    let n = state[265] != 0;
    let c = state[266] != 0;
    let v = state[267] != 0;
    Ok(match code {
        0 => false,
        1 => (n && !z) ^ v,
        2 => !n && z,
        3 => n ^ (z || v),
        4 => !((n ^ v) || z),
        5 => !z,
        6 => !(n ^ v),
        7 => !(n && z),
        8 => n && z,
        9 => n ^ v,
        10 => z,
        11 => (n ^ v) || z,
        12 => !n ^ (v || z),
        13 => n || !z,
        14 => (!n || z) ^ v,
        15 => true,
        16 => !v,
        17 => !c,
        18 => !n,
        19 => z || !c,
        20 => c && !z,
        21 => n,
        22 => c,
        23 => v,
        // Numeric selectors: Ryujinx Decoders/InstDecoders.cs:104-111.
        // 24..29 are CSM modes; RLE/RGT are30/31.
        30 => n || z,
        31 => !(n || z),
        _ => return Err("control oracle missing condition code"),
    })
}
pub fn program(raw: u64) -> Vec<u32> {
    let mov = |value: u64| 0x0100_u64 << 48 | value << 20 | 15 << 12 | 7 << 16 | 240;
    let exit = 0xe300_u64 << 48 | 7 << 16 | 15;
    let nop = 0x50b0_u64 << 48 | 7 << 16;
    let slots = if matches!(raw >> 48, 0xe240 | 0xe250..=0xe25f) {
        vec![
            raw,
            mov(1),
            0xe240_u64 << 48 | 16 << 20 | 7 << 16 | 15,
            mov(2),
            exit,
            nop,
        ]
    } else if (0xe290..=0xe29f).contains(&(raw >> 48)) && raw & 0x20 != 0 {
        vec![raw, mov(1), exit, nop, nop, nop]
    } else if raw >> 48 == 0xe290 {
        vec![raw, 0xf0f8_u64 << 48 | 7 << 16 | 15, mov(1), exit, nop, nop]
    } else if raw >> 48 == 0xf0f8 {
        // SSY to slot4 (PC 0x30 = 0x10 + 0x20; BTARG20..43 relative to
        // next PC, gm107.c:297-299), then the CC-tested SYNC under test.
        // Taken writes 2; fallthrough writes 1.
        let ssy = 0xe290_u64 << 48 | 0x20 << 20 | 7 << 16 | 15;
        vec![ssy, raw, mov(1), exit, mov(2), exit]
    } else {
        vec![raw, mov(1), exit, exit, nop, nop]
    };
    slots
        .chunks(3)
        .flat_map(|s| [0, s[0], s[1], s[2]])
        .flat_map(|w| [w as u32, (w >> 32) as u32])
        .collect()
}
pub fn prepare(raw: u64, case: usize, state: &mut [u32; WORDS]) {
    if (0xe250..=0xe25f).contains(&(raw >> 48)) && field(raw, 8, 8) != 255 {
        // BRX adds the signed displacement register to next PC + offset.
        // Mesa nv50_ir_emit_gm107.cpp:533-555; Ryujinx
        // InstEmitFlowControl.cs:38-40. Alternate the two diamond paths.
        let target = if case.is_multiple_of(2) { 16_i32 } else { 40 };
        let offset = (field(raw, 20, 24) << 8) as i32 >> 8;
        state[field(raw, 8, 8) as usize] = target.wrapping_sub(16 + offset) as u32;
    }
}

/// The twelve Z/N/C/V states reachable after integer flag writers. Z and N
/// cannot both hold; IADD supplies all four C/V pairs and LOP.CC can select
/// zero, positive, or negative Z/N without changing C/V.
pub const REACHABLE_FLAGS: [[u32; 4]; 12] = [
    [1, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 0, 1, 0],
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [0, 0, 0, 1],
    [0, 1, 0, 1],
    [1, 0, 1, 1],
    [0, 0, 1, 1],
    [0, 1, 1, 1],
];

/// Give every reachable flag state both targets twice in the 48-case RADV
/// transport. The observer writes these bits into the emitter's typed flag
/// variables before the program starts.
pub fn prepare_lt(raw: u64, case: usize, state: &mut [u32; WORDS]) {
    state[264..268].copy_from_slice(&REACHABLE_FLAGS[(case / 2) % REACHABLE_FLAGS.len()]);
    prepare(raw, case % 2, state);
}
pub fn reference(code: &[u32], state: &[u32; WORDS]) -> Result<Expected, &'static str> {
    let mut out = Expected::new(state);
    let mut pc = 8_i64;
    let mut stack = Vec::new();
    // Ryujinx InstEmitFlowControl.cs:12-17,153-186,239-310. Push/pop
    // target semantics are executed here, never treated as scheduling no-ops.
    for _ in 0..64 {
        if pc < 0 || pc as usize + 8 > code.len() * 4 || pc % 32 == 0 {
            return Err("undefined control-flow target");
        }
        let at = pc as usize / 4;
        let raw = u64::from(code[at]) | u64::from(code[at + 1]) << 32;
        let f = |a, b| field(raw, a, b);
        let op = raw >> 48;
        let guard = (out.words[256 + f(16, 3) as usize] != 0) ^ (f(19, 1) != 0);
        let offset = (f(20, 24) << 8) as i32 >> 8;
        let next = pc + 8;
        match op {
            0xe240 if guard && condition(f(0, 5), &out.words)? => pc = next + i64::from(offset),
            // BRX source8..15 and offset20..43: envytools gm107.c:1943,254.
            // Ryujinx InstEmitFlowControl.cs:38-40 adds both to next PC.
            0xe250..=0xe25f if guard && condition(f(0, 5), &out.words)? => {
                pc = next + i64::from(offset) + i64::from(out.words[f(8, 8) as usize] as i32);
            }
            0xe300 if guard && condition(f(0, 5), &out.words)? => return Ok(out),
            0xe290..=0xe29f => {
                stack.push(next + i64::from(offset));
                pc = next;
            }
            // SYNC tests guard16..19 and CC0..4 (gm107.c:1877,426-458);
            // Ryujinx InstEmitFlowControl.cs:232-237,279-313.
            0xf0f8 if guard && condition(f(0, 5), &out.words)? => {
                pc = stack.pop().ok_or("control oracle missing sync target")?
            }
            0x0100..=0x01ff => {
                if guard && f(0, 8) != 255 {
                    out.words[f(0, 8) as usize] = f(20, 32);
                }
                pc = next;
            }
            // The loop context uses IADD.CC, whose independent raw-bit
            // arithmetic oracle cites sm50.rs:1265-1286 and Mesa ir.rs:
            // 3846-3857. It updates the flags tested by the back edge.
            0x3810 => {
                out = alu_reference(raw, &out.words)?;
                pc = next;
            }
            0xe240 | 0xe250..=0xe25f | 0xe300 | 0xf0f8 | 0x50b0 => pc = next,
            _ => return Err("control oracle missing instruction"),
        }
        if pc % 32 == 0 {
            pc += 8;
        }
    }
    Err("undefined nonterminating control program")
}

pub fn loop_program(raw: u64) -> Vec<u32> {
    // IADD.CC R240,R240,1 then BRA back to that add; EXIT on fallthrough.
    // sm50.rs:1265-1286,3020-3030,3101-3109.
    let add = 0x3810_u64 << 48 | 1 << 47 | 1 << 20 | 7 << 16 | 240 << 8 | 240;
    let exit = 0xe300_u64 << 48 | 7 << 16 | 15;
    [0, add, raw, exit]
        .into_iter()
        .flat_map(|w| [w as u32, (w >> 32) as u32])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CC_F_CUBES: [(u64, u64); 2] = [
        (0xe250_0000_0007_0000, 0x000f_0000_0008_0000),
        (0xe250_5bb1_b767_1200, 0x000f_0000_0008_0000),
    ];
    const CC_LT_CUBES: [(u64, u64); 2] = [
        (0xe250_0000_0007_0001, 0x000f_0000_0008_0000),
        (0xe250_5bb1_b767_1201, 0x000f_0000_0008_0000),
    ];

    fn words(cubes: [(u64, u64); 2]) -> impl Iterator<Item = u64> {
        cubes.into_iter().flat_map(|(base, variable)| {
            let bits: Vec<u32> = (0..64).filter(|bit| variable & (1 << bit) != 0).collect();
            (0..(1_u16 << bits.len())).map(move |selector| {
                bits.iter().enumerate().fold(base, |word, (index, bit)| {
                    word | u64::from((selector >> index) & 1) << bit
                })
            })
        })
    }

    #[test]
    fn brx_cc_f_slice_has_a_cpu_oracle_for_every_word() {
        let mut checked = 0;
        for raw in words(CC_F_CUBES) {
            let code = program(raw);
            for flags in 0_u32..16 {
                let mut state = [0; WORDS];
                state[263] = 1;
                for bit in 0..4 {
                    state[264 + bit] = flags >> bit & 1;
                }
                for target_case in 0..2 {
                    prepare(raw, target_case, &mut state);
                    assert_eq!(
                        reference(&code, &state).unwrap().words[240],
                        1,
                        "{raw:016x}, flags={flags:#x}, target={target_case}"
                    );
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 2_048);
    }

    #[test]
    fn brx_cc_lt_slice_has_a_cpu_oracle_for_every_word() {
        let mut checked = 0;
        for raw in words(CC_LT_CUBES) {
            let code = program(raw);
            for flags in REACHABLE_FLAGS {
                let mut state = [0; WORDS];
                state[263] = 1;
                state[264..268].copy_from_slice(&flags);
                for target_case in 0..2 {
                    prepare(raw, target_case, &mut state);
                    let executes = field(raw, 19, 1) == 0;
                    let taken = executes && condition(1, &state).unwrap();
                    let expected = if taken && target_case == 1 { 2 } else { 1 };
                    assert_eq!(
                        reference(&code, &state).unwrap().words[240],
                        expected,
                        "{raw:016x}, flags={flags:?}, target={target_case}"
                    );
                    checked += 1;
                }
            }
        }
        assert_eq!(checked, 1_536);
    }
}
