//! Project-authored BRK words and frozen CC truth tables.
//! Fields: Mesa sm50.rs:217-235,3032-3074,3418-3428; envytools
//! envydis/gm107.c:1920-1921,426-458. Semantics: Ryujinx
//! Instructions/InstEmitFlowControl.cs:254-262,295-324 and
//! Instructions/InstEmitConditionCode.cs:56-81.
#![allow(dead_code)]
pub const CONDITIONS: [u8; 26] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 30, 31,
];
pub fn address(slot: usize) -> u32 {
    (slot / 3 * 32 + slot % 3 * 8 + 8) as u32
}
fn op(code: u64) -> u64 {
    code << 48 | 7 << 16
}
pub fn push(from: usize, to: usize) -> u64 {
    op(0xe2a0) | u64::from(address(to).wrapping_sub(address(from) + 8) & 0xff_ffff) << 20 | 15
}
pub fn brk(cc: u8, predicate: u8) -> u64 {
    op(0xe340) & !(15 << 16) | u64::from(predicate) << 16 | u64::from(cc)
}
pub fn mov(dst: u8, value: u32) -> u64 {
    // MOV32I: Mesa sm50.rs:1938-1941,1951: immediate20..51,
    // quad lanes12..15 and destination0..7.
    op(0x0100) | 15 << 12 | u64::from(value) << 20 | u64::from(dst)
}
pub fn exit() -> u64 {
    op(0xe300) | 15
}
pub fn words(slots: &[u64]) -> Vec<u32> {
    let mut words = Vec::new();
    for bundle in slots.chunks(3) {
        words.extend([0, 0]);
        for &raw in bundle {
            words.extend([raw as u32, (raw >> 32) as u32]);
        }
    }
    words
}
pub fn program(cc: u8, predicate: u8) -> Vec<u32> {
    program_for_stack(cc, predicate, false)
}
pub fn program_for_stack(cc: u8, predicate: u8, continuation: bool) -> Vec<u32> {
    words(&[
        push(0, 4) ^ if continuation { 0x10 << 48 } else { 0 },
        brk(cc, predicate) ^ if continuation { 0x10 << 48 } else { 0 },
        mov(8, 1),
        exit(),
        mov(8, 2),
        exit(),
    ])
}
// Bit f is the expected condition for the four-bit flag state f.
const TRUTH: [u16; 32] = [
    0x0000, 0xbb44, 0x2222, 0x3366, 0x4411, 0x5555, 0xcc33, 0x7777, 0x8888, 0x33cc, 0xaaaa, 0xbbee,
    0xcc99, 0xdddd, 0x44bb, 0xffff, 0x00ff, 0x0f0f, 0x3333, 0xafaf, 0x5050, 0xcccc, 0xf0f0, 0xff00,
    0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0x0000, 0xeeee, 0x1111,
];
pub fn taken(cc: u8, flags: u8) -> bool {
    TRUTH[usize::from(cc)] & (1 << flags) != 0
}
