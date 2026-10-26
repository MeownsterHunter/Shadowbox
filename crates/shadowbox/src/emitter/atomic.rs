//! Atomic updates SPIR-V has no single instruction for: ATOM/RED/ATOMS
//! INC and DEC (U32), and ATOM/RED ADD.F32.FTZ.RN.
//!
//! Ryujinx InstEmitMemory.cs:202-305 has no mapping for these (it logs and
//! writes zero). Their meaning comes from Mesa nouveau, the approved second
//! reference: codegen nv50_ir_from_nir.cpp:641-670 maps NIR inc_wrap,
//! dec_wrap and fadd onto the hardware INC, DEC and ADD sub-ops
//! (nv50_ir_emit_gm107.cpp:2642-2735 encodes them), and NAK
//! from_nir.rs:2129-2173 and sm50.rs:2694-2760 type float add as F32. NIR
//! inc_wrap and dec_wrap are typed uint (nir.h:1326-1332):
//!
//! - INC: new = old >= data ? 0 : old + 1
//! - DEC: new = (old == 0 || old > data) ? data : old - 1
//! - ADD.F32.FTZ.RN: new = ftz(ftz(old) + ftz(data)), rounded to nearest
//!   even; FTZ flushes a denormal to its signed zero, as Mesa NAK
//!   from_nir.rs:1313-1320 does.
//!
//! Each is a compare-exchange loop on the 32-bit word: load, compute the
//! new value, swap it in only if the word still holds the loaded value, and
//! go again otherwise. The old value the swap saw is the result.
//!
//! Mesa nouveau files used above:
//! Copyright © 2022-2023 Collabora, Ltd.; Copyright 2014 Red Hat Inc.
//! SPDX-License-Identifier: MIT
//!
//! Permission is hereby granted, free of charge, to any person obtaining a
//! copy of this software and associated documentation files (the
//! "Software"), to deal in the Software without restriction, including
//! without limitation the rights to use, copy, modify, merge, publish,
//! distribute, sublicense, and/or sell copies of the Software, and to permit
//! persons to whom the Software is furnished to do so, subject to the
//! following conditions:
//!
//! The above copyright notice and this permission notice shall be included
//! in all copies or substantial portions of the Software.
//!
//! THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//! OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//! MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
//! NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
//! DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
//! OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
//! USE OR OTHER DEALINGS IN THE SOFTWARE.

use super::inst;
use crate::ir::AtomicOp;
use crate::spirv::Op;

/// The new word an update computes from the old word and the data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Update {
    Increment,
    Decrement,
    FloatAddFtz,
}

/// How one ATOMS operation lowers: one SPIR-V atomic, or a swap loop.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum SharedForm {
    Direct(Op),
    Loop(Update),
}

/// ATOMS lowering by operation and signedness. Ryujinx
/// InstEmitMemory.cs:202-305 for the direct ones.
pub(crate) fn shared_form(op: AtomicOp, signed: bool) -> SharedForm {
    match (op, signed) {
        (AtomicOp::Add, _) => SharedForm::Direct(Op::AtomicIAdd),
        (AtomicOp::Exchange, _) => SharedForm::Direct(Op::AtomicExchange),
        (AtomicOp::And, _) => SharedForm::Direct(Op::AtomicAnd),
        (AtomicOp::Or, _) => SharedForm::Direct(Op::AtomicOr),
        (AtomicOp::Xor, _) => SharedForm::Direct(Op::AtomicXor),
        (AtomicOp::Min, true) => SharedForm::Direct(Op::AtomicSMin),
        (AtomicOp::Min, false) => SharedForm::Direct(Op::AtomicUMin),
        (AtomicOp::Max, true) => SharedForm::Direct(Op::AtomicSMax),
        (AtomicOp::Max, false) => SharedForm::Direct(Op::AtomicUMax),
        (AtomicOp::Inc, _) => SharedForm::Loop(Update::Increment),
        (AtomicOp::Dec, _) => SharedForm::Loop(Update::Decrement),
    }
}

/// The 32-bit uint constants the updates use.
#[derive(Clone, Copy)]
pub(crate) struct Constants {
    pub zero: u32,
    pub one: u32,
    /// 0x7fff_ffff
    pub magnitude: u32,
    /// 0x0080_0000, the smallest normal magnitude.
    pub smallest_normal: u32,
    /// 0x8000_0000
    pub sign: u32,
}

impl Constants {
    /// Builds each constant with `make(value)`.
    pub(crate) fn new(mut make: impl FnMut(u32) -> u32) -> Self {
        Self {
            zero: make(0),
            one: make(1),
            magnitude: make(0x7fff_ffff),
            smallest_normal: make(0x0080_0000),
            sign: make(0x8000_0000),
        }
    }
}

/// One swap loop on a 32-bit word.
#[derive(Clone, Copy)]
pub(crate) struct Swap {
    pub uint: u32,
    pub boolt: u32,
    pub float: u32,
    pub pointer: u32,
    pub scope: u32,
    /// Semantics of the load and of a failed compare (no Release).
    pub acquire: u32,
    /// Semantics of a successful compare-exchange.
    pub equal: u32,
    pub value: u32,
    pub constants: Constants,
}

fn fresh(id: &mut u32) -> u32 {
    let value = *id;
    *id += 1;
    value
}

fn alu(w: &mut Vec<u32>, id: &mut u32, op: Op, ty: u32, operands: &[u32]) -> u32 {
    let result = fresh(id);
    let mut words = vec![ty, result];
    words.extend_from_slice(operands);
    inst(w, op, &words);
    result
}

/// A denormal's bits become its signed zero; others stay.
fn flush(w: &mut Vec<u32>, id: &mut u32, s: &Swap, bits: u32) -> u32 {
    let c = s.constants;
    let magnitude = alu(w, id, Op::BitwiseAnd, s.uint, &[bits, c.magnitude]);
    let tiny = alu(
        w,
        id,
        Op::ULessThan,
        s.boolt,
        &[magnitude, c.smallest_normal],
    );
    let sign = alu(w, id, Op::BitwiseAnd, s.uint, &[bits, c.sign]);
    alu(w, id, Op::Select, s.uint, &[tiny, sign, bits])
}

fn update(w: &mut Vec<u32>, id: &mut u32, s: &Swap, update: Update, old: u32) -> u32 {
    let c = s.constants;
    match update {
        Update::Increment => {
            let wraps = alu(w, id, Op::UGreaterThanEqual, s.boolt, &[old, s.value]);
            let next = alu(w, id, Op::IAdd, s.uint, &[old, c.one]);
            alu(w, id, Op::Select, s.uint, &[wraps, c.zero, next])
        }
        Update::Decrement => {
            let empty = alu(w, id, Op::IEqual, s.boolt, &[old, c.zero]);
            let above = alu(w, id, Op::UGreaterThan, s.boolt, &[old, s.value]);
            let wraps = alu(w, id, Op::LogicalOr, s.boolt, &[empty, above]);
            let next = alu(w, id, Op::ISub, s.uint, &[old, c.one]);
            alu(w, id, Op::Select, s.uint, &[wraps, s.value, next])
        }
        Update::FloatAddFtz => {
            let a = flush(w, id, s, old);
            let b = flush(w, id, s, s.value);
            let a = alu(w, id, Op::Bitcast, s.float, &[a]);
            let b = alu(w, id, Op::Bitcast, s.float, &[b]);
            let sum = alu(w, id, Op::FAdd, s.float, &[a, b]);
            let bits = alu(w, id, Op::Bitcast, s.uint, &[sum]);
            flush(w, id, s, bits)
        }
    }
}

/// Emits the loop and returns the old value. The current block ends; the
/// loop's merge block is current afterwards.
pub(crate) fn swap_loop(w: &mut Vec<u32>, id: &mut u32, s: &Swap, how: Update) -> u32 {
    let header = fresh(id);
    let body = fresh(id);
    let next = fresh(id);
    let merge = fresh(id);
    inst(w, Op::Branch, &[header]);
    inst(w, Op::Label, &[header]);
    inst(w, Op::LoopMerge, &[merge, next, 0]);
    inst(w, Op::Branch, &[body]);
    inst(w, Op::Label, &[body]);
    let old = alu(
        w,
        id,
        Op::AtomicLoad,
        s.uint,
        &[s.pointer, s.scope, s.acquire],
    );
    let new = update(w, id, s, how, old);
    let seen = alu(
        w,
        id,
        Op::AtomicCompareExchange,
        s.uint,
        &[s.pointer, s.scope, s.equal, s.acquire, new, old],
    );
    let done = alu(w, id, Op::IEqual, s.boolt, &[seen, old]);
    inst(w, Op::BranchConditional, &[done, merge, next]);
    inst(w, Op::Label, &[next]);
    inst(w, Op::Branch, &[header]);
    inst(w, Op::Label, &[merge]);
    seen
}
