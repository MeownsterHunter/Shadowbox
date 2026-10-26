// Independent raw-bit memory oracle. No decoded IR or emitted operations.
// Encoding paths: Mesa src/nouveau/compiler/nak/sm50.rs; envytools envydis/gm107.c.
// Semantics paths: Ryujinx.Graphics.Shader/Instructions/InstEmitMemory.cs.
use super::compute::{cbuf, GUEST_BASE, MEMORY_WORDS, STATE_WORDS, WORDS};
use super::reference::{field, Expected};

pub fn is_memory(raw: u64) -> bool {
    let op = (raw >> 48) as u16;
    matches!(
        op & 0xfff8,
        0xef90 | 0xef40 | 0xef48 | 0xef50 | 0xef58 | 0xeed0 | 0xeed8 | 0xebf8
    ) || matches!(op & 0xff00, 0xec00 | 0xed00)
        || matches!(op & 0xfff0, 0xee40 | 0xeef0)
}
pub fn is_flat(raw: u64) -> bool {
    let op = (raw >> 48) as u16;
    matches!(op & 0xfff8, 0xeed0 | 0xeed8 | 0xebf8)
        || op & 0xff00 == 0xed00
        || op & 0xfff0 == 0xeef0
}
fn is_atomic(raw: u64) -> bool {
    let op = (raw >> 48) as u16;
    op & 0xff00 == 0xec00
        || op & 0xff00 == 0xed00
        || op & 0xfff8 == 0xebf8
        || matches!(op & 0xfff0, 0xee40 | 0xeef0)
}
fn signed(raw: u64, at: u32, width: u32) -> i32 {
    (field(raw, at, width) << (32 - width)) as i32 >> (32 - width)
}
fn offset(raw: u64) -> i32 {
    let op = (raw >> 48) as u16;
    if op & 0xfff8 == 0xef90 {
        signed(raw, 20, 16)
    } else if op & 0xff00 == 0xec00 || op & 0xfff0 == 0xee40 {
        signed(raw, 30, 22).wrapping_mul(4)
    } else if op & 0xff00 == 0xed00 || op & 0xfff8 == 0xebf8 || op & 0xfff0 == 0xeef0 {
        signed(raw, 28, 20)
    } else {
        signed(raw, 20, 24)
    }
}
fn extended(raw: u64) -> bool {
    // A64 LD/ST bit45 (sm50.rs:2402-2410), ATOM/RED E48 (:2781-2789).
    field(
        raw,
        if matches!((raw >> 48) as u16 & 0xfff8, 0xeed0 | 0xeed8) {
            45
        } else {
            48
        },
        1,
    ) != 0
}
pub fn fixed_address(raw: u64) -> bool {
    is_flat(raw) && field(raw, 8, 8) == 255
}
pub fn guest_base(raw: u64) -> u64 {
    if fixed_address(raw) {
        // RZ has no high register: Ryujinx InstEmitMemory.cs:518-527.
        // Place the encoded address at byte32 of one observation allocation.
        // The signed immediate remains unchanged, including its carry/borrow.
        (i64::from(offset(raw)) as u64).wrapping_sub(32)
    } else {
        GUEST_BASE
    }
}
pub fn prepare(raw: u64, case: usize, state: &mut [u32; WORDS]) -> Result<(), &'static str> {
    let addr = field(raw, 8, 8) as usize;
    if fixed_address(raw) {
        return Ok(());
    }
    if addr == 255 || is_flat(raw) && extended(raw) && addr >= 254 {
        return Err("undefined bounded memory address register");
    }
    // Keep offsets, signs and widths in the actual encoding. Adjust runtime
    // addresses, not the instruction, to stay in allocated test memory.
    let ty = field(raw, 48, 3);
    let subword = if !is_atomic(raw) && ty < 2 {
        case as u32 % 4
    } else if !is_atomic(raw) && ty < 4 {
        (case as u32 % 2) * 2
    } else {
        0
    };
    let ldc_segmented = (raw >> 48) as u16 & 0xfff8 == 0xef90 && matches!(field(raw, 44, 2), 2 | 3);
    // Segmented LDC adds a signed 16-bit immediate to the low 16 address
    // bits. Keep the final byte near the middle of the bank so both large
    // positive and negative encoded offsets remain in the same segment.
    let byte = if ldc_segmented { 32768 } else { 32 } + (case as u32 % 4) * 16 + subword;
    let target = if is_flat(raw) {
        GUEST_BASE + (case * MEMORY_WORDS * 4) as u64 + u64::from(byte)
    } else {
        u64::from(byte)
    };
    let address = target.wrapping_sub(i64::from(offset(raw)) as u64);
    state[addr] = if ldc_segmented {
        // Exercise runtime bank selection independently of the decoder/IR.
        // Keep the packed runtime segment plus the encoded bank inside the
        // five-bit constant-buffer domain. Bank 31 necessarily uses segment
        // zero; smaller encoded banks exercise a nonzero runtime segment.
        let encoded_bank = field(raw, 36, 5);
        let remaining = 31 - encoded_bank;
        let bank = if remaining == 0 {
            0
        } else {
            (case as u32 % remaining) + 1
        };
        bank << 16 | address as u32 & 0xffff
    } else {
        address as u32
    };
    if is_flat(raw) && extended(raw) {
        state[addr + 1] = (address >> 32) as u32;
    }
    // Exercise both packed-CAS branches, not only overwhelmingly unequal
    // random comparators. CAS compare/data layout: sm50.rs:2733-2761.
    if matches!((raw >> 48) as u16 & 0xfff0, 0xee40 | 0xeef0) && case.is_multiple_of(2) {
        state[STATE_WORDS + byte as usize / 4] = state[field(raw, 20, 8) as usize];
    }
    Ok(())
}
pub fn reference(raw: u64, case: usize, state: &[u32; WORDS]) -> Result<Expected, &'static str> {
    let op = (raw >> 48) as u16;
    let f = |a, b| field(raw, a, b);
    let mut out = Expected::new(state);
    // Common predicate bits16..19: gm107.c:408-411.
    if (state[256 + f(16, 3) as usize] != 0) == (f(19, 1) != 0) {
        return Ok(out);
    }
    let packed_addr = state[f(8, 8) as usize];
    let ldc_segmented = op & 0xfff8 == 0xef90 && matches!(f(44, 2), 2 | 3);
    let addr = if ldc_segmented {
        packed_addr & 0xffff
    } else {
        packed_addr
    };
    let byte = if is_flat(raw) {
        // Ryujinx InstEmitMemory.cs:511-542 constructs the 40-bit Maxwell
        // address, including signed immediate carry/borrow.
        let high = if extended(raw) && f(8, 8) != 255 {
            u64::from(state[(f(8, 8) as usize + 1).min(255)] & 255)
        } else {
            0
        };
        let absolute = (high << 32 | u64::from(addr)).wrapping_add(i64::from(offset(raw)) as u64);
        let case_offset = if fixed_address(raw) {
            0
        } else {
            (case * MEMORY_WORDS * 4) as u64
        };
        absolute.wrapping_sub(guest_base(raw).wrapping_add(case_offset)) as usize
    } else {
        addr.wrapping_add(offset(raw) as u32) as usize
    };
    let atom = op & 0xff00 == 0xec00
        || op & 0xff00 == 0xed00
        || op & 0xfff8 == 0xebf8
        || matches!(op & 0xfff0, 0xee40 | 0xeef0);
    if atom {
        // ATOMS dst0,addr8,data20,type28,op52,offset30/4: sm50.rs:2800-2834.
        // ATOM/RED/CAS fields: sm50.rs:2717-2789; gm107.c:1897,1903-1905.
        // Ryujinx InstEmitMemory.cs:202-305 returns the old word.
        if byte >= MEMORY_WORDS * 4 || !byte.is_multiple_of(4) {
            return Err("undefined atomic address");
        }
        let red = op & 0xfff8 == 0xebf8;
        let cas = matches!(op & 0xfff0, 0xee40 | 0xeef0);
        let shared = op & 0xff00 == 0xec00 || op & 0xfff0 == 0xee40;
        let data = f(if red { 0 } else { 20 }, 8) as usize;
        let value = state[data];
        let code = if cas {
            15
        } else {
            f(if red { 23 } else { 52 }, if red { 3 } else { 4 })
        };
        let signed = f(
            if shared {
                28
            } else if red {
                20
            } else {
                49
            },
            if shared { 2 } else { 3 },
        ) == 1;
        // 64-bit ATOM/RED type 2 U64 / 5 S64: gm107.c:688-694,731-737;
        // Mesa nak/from_nir.rs:2129-2173. Low word at the lower address and
        // in the even register (pairs as for the E address above).
        let ty = f(if red { 20 } else { 49 }, 3);
        if !shared && !cas && matches!(ty, 2 | 5) {
            if !byte.is_multiple_of(8) || byte + 8 > MEMORY_WORDS * 4 || data >= 255 {
                return Err("undefined 64-bit atomic address or data pair");
            }
            let at = STATE_WORDS + byte / 4;
            let old = u64::from(state[at]) | u64::from(state[at + 1]) << 32;
            let value = u64::from(state[data]) | u64::from(state[data + 1]) << 32;
            let s64 = ty == 5;
            let new = match code {
                0 => old.wrapping_add(value),
                1 if s64 => (old as i64).min(value as i64) as u64,
                1 => old.min(value),
                2 if s64 => (old as i64).max(value as i64) as u64,
                2 => old.max(value),
                5 => old & value,
                6 => old | value,
                7 => old ^ value,
                8 => value,
                _ => return Err("memory oracle missing 64-bit atomic operation"),
            };
            out.words[at] = new as u32;
            out.words[at + 1] = (new >> 32) as u32;
            let dst = f(0, 8) as usize;
            if !red && dst < 255 {
                out.words[dst] = old as u32;
                out.words[dst + 1] = (old >> 32) as u32;
            }
            return Ok(out);
        }
        let at = STATE_WORDS + byte / 4;
        let old = state[at];
        out.words[at] = match code {
            0 => old.wrapping_add(value),
            1 if signed => (old as i32).min(value as i32) as u32,
            1 => old.min(value),
            2 if signed => (old as i32).max(value as i32) as u32,
            2 => old.max(value),
            5 => old & value,
            6 => old | value,
            7 => old ^ value,
            8 => value,
            15 => {
                if old == value {
                    state[(data + 1).min(255)]
                } else {
                    old
                }
            }
            _ => return Err("memory oracle missing atomic operation"),
        };
        if !red && f(0, 8) != 255 {
            out.words[f(0, 8) as usize] = old;
        }
        return Ok(out);
    }
    let ty = f(48, 3);
    let count = if ty >= 6 {
        4
    } else if ty == 5 {
        2
    } else {
        1
    };
    let width = if ty < 2 {
        8
    } else if ty < 4 {
        16
    } else {
        32
    };
    let ldc = op & 0xfff8 == 0xef90;
    if byte + count * 4 > if ldc { 65536 } else { MEMORY_WORDS * 4 }
        || !byte.is_multiple_of((width / 8) as usize)
    {
        return Err("undefined memory range or alignment");
    }
    let store = matches!(op & 0xfff8, 0xef50 | 0xef58 | 0xeed8);
    for index in 0..count {
        let register = (f(0, 8) as usize + index).min(255);
        let at = byte / 4 + index;
        // LDC bank36..41, signed offset20..36, size48..51: sm50.rs:2618-2649.
        // Loads: InstEmitMemory.cs:59-105,461-510,548-574.
        // Stores: InstEmitMemory.cs:337-377,577-598.
        let old = if ldc {
            let bank = f(36, 5) + if ldc_segmented { packed_addr >> 16 } else { 0 };
            if bank >= 32 {
                return Err("undefined constant-buffer bank");
            }
            cbuf(bank, at as u32)
        } else {
            state[STATE_WORDS + at]
        };
        let shift = (byte & 3) * 8;
        if store {
            out.words[STATE_WORDS + at] = if width == 32 {
                state[register]
            } else {
                let mask = ((1_u32 << width) - 1) << shift;
                old & !mask | (state[register] << shift) & mask
            };
        } else if register != 255 {
            let value = old >> shift;
            out.words[register] = if width == 32 {
                value
            } else if ty & 1 != 0 {
                ((value << (32 - width)) as i32 >> (32 - width)) as u32
            } else {
                value & ((1 << width) - 1)
            };
        }
    }
    Ok(out)
}

pub fn is_barrier(raw: u64) -> bool {
    raw >> 48 == 0xef98
}
pub fn barrier_memory() -> u64 {
    // STG.B32 R240,[R2], then LDG.B32 R241,[R2]. Encodings:
    // sm50.rs:2595-2615,2652-2669; B32 size4, sm50.rs:2386-2397.
    0xeedc_u64 << 48 | 2 << 8 | 7 << 16 | 240
}
pub fn barrier_program(raw: u64) -> Vec<u32> {
    let load = 0xeed4_u64 << 48 | 2 << 8 | 7 << 16 | 241;
    let exit = 0xe300_u64 << 48 | 7 << 16 | 15;
    let nop = 0x50b0_u64 << 48 | 7 << 16;
    [0, barrier_memory(), raw, load, 0, exit, nop, nop]
        .into_iter()
        .flat_map(|w| [w as u32, (w >> 32) as u32])
        .collect()
}
pub fn barrier_reference(
    raw: u64,
    case: usize,
    state: &[u32; WORDS],
) -> Result<Expected, &'static str> {
    // Ryujinx InstEmitMemory.cs:142-147,337-377 supplies the store/load
    // behavior. InstEmitBarrier.cs:32-44 maps MEMBAR to host memory barriers.
    // The single-invocation context has no competing writer.
    let stored = reference(barrier_memory(), case, state)?;
    let _ = raw; // Scope affects ordering, not the value in this defined context.
    reference(
        0xeed4_u64 << 48 | 2 << 8 | 7 << 16 | 241,
        case,
        &stored.words,
    )
}

/// Raw-bit interpreter for a bounded descriptor allocation, independent of the
/// translator's cbuf reaching proof and selected SPIR-V pointers.
/// LDG/STG fields: Mesa nak/sm50.rs:2386-2431,2603-2614,2659-2669.
/// Address arithmetic: Ryujinx InstEmitMemory.cs:512-541; relative descriptor
/// addressing: Translation/Optimizations/GlobalToStorage.cs:305-357,409-445.
pub fn descriptor_reference(
    raw: u64,
    state: &[u32; WORDS],
    bases: &[u64],
    backing: &mut [u32],
) -> Result<Expected, &'static str> {
    let op = (raw >> 48) as u16;
    if !matches!(op & 0xfff8, 0xeed0 | 0xeed8) || field(raw, 48, 3) != 4 || !extended(raw) {
        return Err("descriptor oracle only implements LDG/STG.E.B32");
    }
    let mut result = Expected::new(state);
    // Mesa sm50.rs:225-235: check the instruction's runtime predicate before
    // dereferencing. This includes false predicates on out-of-range addresses.
    if (state[256 + field(raw, 16, 3) as usize] != 0) == (field(raw, 19, 1) != 0) {
        return Ok(result);
    }
    let ra = field(raw, 8, 8) as usize;
    if ra >= 254 {
        return Err("descriptor oracle requires an address register pair");
    }
    let address = (u64::from(state[ra + 1] & 255) << 32 | u64::from(state[ra]))
        .wrapping_add(i64::from(offset(raw)) as u64);
    let matches = bases
        .iter()
        .filter_map(|base| {
            let byte = address.wrapping_sub(*base);
            (byte < (backing.len() * 4) as u64).then_some(byte as usize)
        })
        .collect::<Vec<_>>();
    if matches.len() != 1 || matches[0] % 4 != 0 {
        return Err("descriptor oracle cannot resolve a unique aligned allocation");
    }
    let at = matches[0] / 4;
    let rd = field(raw, 0, 8) as usize;
    if op & 0xfff8 == 0xeed8 {
        // RZ is zero, even if an observer accidentally supplied another value.
        // Ryujinx RegisterConsts.cs:5-11, InstEmitMemory.cs:478-486.
        backing[at] = if rd == 255 { 0 } else { state[rd] };
    } else if rd != 255 {
        result.words[rd] = backing[at];
    }
    Ok(result)
}
