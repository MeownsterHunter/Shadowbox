//! Project-owned Maxwell control flow, shared by the regular gate and fuzzing.
//! No instruction semantics are added. Each encoding field is cited below.

// Mesa nak/sm50.rs:204-209 RZ=255 and PT=7; :225-235 predicate fields.
const PT: u64 = 7 << 16;

// Mesa nak/sm50.rs:225-235 predicate register 16..19 and inversion 19.
fn pred(register: u64, inverted: bool) -> u64 {
    register << 16 | u64::from(inverted) << 19
}

// Mesa nak/sm50.rs:1938-1941 MOV32I, immediate 20..52, quad lanes 12..16;
// :321-328 destination 0..8. All four lanes are enabled.
fn mov(dst: u64, bits: u32) -> u64 {
    0x0100 << 48 | u64::from(bits) << 20 | PT | 0xf << 12 | dst
}

// Mesa nak/sm50.rs:1505-1509 ISETP register form, src1 20..28;
// :1523-1541 dst1 0..3=PT, dst0 3..6, src0 8..16, accumulator
// 39..42=PT, invert42=0, .X43=0, AND45..47=0, unsigned48=0,
// compare49..52=EQ. AND and EQ values: :798-817.
// RZ=255 and PT=7: Mesa :204-209; Ryujinx Decoders/RegisterConsts.cs:5-11.
fn equal_zero(predicate: u64, register: u64) -> u64 {
    0x5b64 << 48 | 7 << 39 | 255 << 20 | PT | register << 8 | predicate << 3 | 7
}

// Mesa nak/sm50.rs:1255-1263 IADD32I, destination 0..8,
// src0 8..16, immediate 20..52, .CC52=0, .X53=0, negate56=0.
// envytools envydis/gm107.c:2162 saturation54=0.
fn decrement(register: u64) -> u64 {
    0x1c00 << 48 | u64::from(u32::MAX) << 20 | PT | register << 8 | register
}

// Mesa nak/sm50.rs:3025-3074 BRA/SSY/SYNC/PBK/BRK opcodes and CC.T=15;
// :3106-3110 EXIT and CC.T=15. Predicate: :225-235.
fn flow(opcode: u64, predicate: u64) -> u64 {
    opcode << 48 | predicate | 15
}

// Mesa nak/sm50.rs:3418-3428,3438-3468 schedule word + three slots.
// :3005-3016 relative target = target_ip - ip - 8, :3027,3039,3069
// rel24 at 20..44. Only branch/push slots appear in the fixup list.
// :3178-3187 OpNop padding, condition field 8..12=15.
fn assemble(mut slots: Vec<u64>, fixups: &[(usize, usize)]) -> Vec<u8> {
    let address = |slot: usize| ((slot / 3 * 4 + slot % 3 + 1) * 8) as i64;
    for &(from, to) in fixups {
        let relative = address(to) - address(from) - 8;
        slots[from] |= (relative as u64 & 0xff_ffff) << 20;
    }
    while !slots.len().is_multiple_of(3) {
        slots.push(0x50b0 << 48 | PT | 15 << 8);
    }
    slots
        .chunks(3)
        .flat_map(|s| [0, s[0], s[1], s[2]])
        .flat_map(u64::to_le_bytes)
        .collect()
}

pub fn programs(seed: u64) -> Vec<(&'static str, Vec<u8>)> {
    // Alternate a folded input and a dynamic c[0][0] input. Mesa
    // nak/sm50.rs:1943-1951 MOV cbuf, quad lanes39..43=15, dst0..8;
    // :361-368 offset20..34=0 and binding34..39=0.
    let input = if seed & 1 == 0 {
        mov(0, (seed >> 1) as u32 & 1)
    } else {
        0x4c98 << 48 | 15 << 39 | PT
    };
    let p0 = pred(0, seed & 4 != 0);
    let p1 = pred(1, seed & 8 != 0);
    let common = [
        input,
        equal_zero(0, 0),
        mov(1, (seed >> 4) as u32 & 1),
        equal_zero(1, 1),
    ];
    let color = (seed as u32).wrapping_mul(0x9e37_79b9);
    let cases = [
        (
            "nested_if_else",
            vec![
                flow(0xe240, p0),
                flow(0xe240, p1),
                mov(0, color),
                flow(0xe240, PT),
                mov(0, !color),
                flow(0xe240, PT),
                mov(0, color.rotate_left(7)),
                flow(0xe300, PT),
            ],
            vec![(4, 10), (5, 8), (7, 11), (9, 11)],
        ),
        (
            "nested_predicated_exit",
            vec![
                flow(0xe300, p0),
                flow(0xe300, p1),
                mov(0, color),
                flow(0xe300, PT),
            ],
            vec![],
        ),
        (
            "nested_ssy_predicated_sync",
            vec![
                flow(0xe290, PT),
                flow(0xe290, PT),
                flow(0xf0f8, p0),
                flow(0xf0f8, PT),
                flow(0xf0f8, p1),
                flow(0xf0f8, PT),
                mov(0, color),
                flow(0xe300, PT),
            ],
            vec![(4, 10), (5, 8)],
        ),
        (
            "ssy_early_exit",
            vec![
                flow(0xe290, PT),
                flow(0xe300, p0),
                flow(0xf0f8, p1),
                mov(0, color),
                flow(0xf0f8, PT),
                flow(0xe300, PT),
            ],
            vec![(4, 9)],
        ),
        (
            "loop_counter_brk",
            vec![
                mov(1, ((seed >> 4) as u32 & 1) + 1),
                flow(0xe2a0, PT),
                decrement(1),
                equal_zero(1, 1),
                flow(0xe340, pred(1, false)),
                flow(0xe300, p0),
                mov(0, color),
                flow(0xe240, PT),
                flow(0xe300, PT),
            ],
            vec![(5, 12), (11, 6)],
        ),
        (
            "loop_predicated_brk_exit",
            vec![
                flow(0xe2a0, PT),
                flow(0xe340, p0),
                flow(0xe300, p1),
                mov(0, color),
                flow(0xe240, pred(7, true)),
                flow(0xe340, PT),
                flow(0xe300, PT),
            ],
            vec![(4, 10), (8, 5)],
        ),
    ];
    cases
        .into_iter()
        .map(|(name, body, fixups)| {
            (
                name,
                assemble(common.into_iter().chain(body).collect(), &fixups),
            )
        })
        .collect()
}
