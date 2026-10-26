//! Memory-class lowering shared by LDC, LDL/LDS/STL/STS and LDG/STG:
//! sub-word extraction and the register each stored component reads.
//! Adapted from Ryujinx Instructions/InstEmitMemory.cs under the MIT
//! notice at the emitter's helpers (Copyright (c) Ryujinx Team and
//! Contributors).

use super::alu;
use crate::ir::Register;
use crate::spirv::Op;

/// Sign or zero extension of a sub-word, indexed by its signedness.
const EXTEND: [Op; 2] = [Op::BitFieldUExtract, Op::BitFieldSExtract];

/// The sub-word of `word` at byte `byte & 3`, extended to 32 bits.
/// Ryujinx InstEmitMemory.cs:544-577 (GetBitOffset, ExtractSmallInt):
/// shift the word right by the byte's bit offset, then zero or sign
/// extend `width` bits. The right shift is never wider than 24 bits, so a
/// half-word at byte 3 reads only the word's top byte, as in Ryujinx.
#[allow(clippy::too_many_arguments)]
pub(super) fn extract_small_int(
    w: &mut Vec<u32>,
    id: &mut u32,
    uint: u32,
    zero: u32,
    [three, width_constant]: [u32; 2],
    signed: bool,
    word: u32,
    byte: u32,
) -> u32 {
    let byte_in_word = alu(w, id, Op::BitwiseAnd, uint, &[byte, three]);
    let shift = alu(w, id, Op::ShiftLeftLogical, uint, &[byte_in_word, three]);
    let shifted = alu(w, id, Op::ShiftRightLogical, uint, &[word, shift]);
    alu(
        w,
        id,
        EXTEND[usize::from(signed)],
        uint,
        &[shifted, zero, width_constant],
    )
}

/// The register component `component` of a store reads. Ryujinx
/// InstEmitMemory.cs:419-421 (EmitStore) and :480-482 (EmitStg): a
/// component at or past RZ reads the first data register instead, so a
/// vector whose data is RZ stores zeroes and one that runs into RZ
/// repeats its first register.
pub(crate) fn stored_register(data: Register, component: u8) -> Register {
    let index = u16::from(data.0) + u16::from(component);
    [Register(index as u8), data][usize::from(index >= 255)]
}
