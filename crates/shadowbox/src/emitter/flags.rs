//! Condition-code writes adapted from Ryujinx
//! Instructions/InstEmitAluHelper.cs:95-158
//! (SetZnFlags, SetFPZnFlags, AddWithCarry) and
//! InstEmitIntegerArithmetic.cs:665-699 (SetIaddFlags), pinned in
//! `docs/reference-sources.md`.
//!
//! The four flags are boolean Function variables (Z, N, C, V). A guarded
//! instruction writes each flag only where its guard holds, as every other
//! destination does (Ryujinx Translator.cs:339-393 wraps the whole op).

// Copyright (c) Ryujinx Team and Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#![allow(dead_code)] // Each instruction class adds its users; see the module doc.

use super::{alu, alu_when, inst};
use crate::spirv::Op;

/// The condition-code variables: zero, sign (N), carry, overflow.
#[derive(Clone, Copy)]
pub(super) struct CcFlags {
    pub zero: u32,
    pub sign: u32,
    pub carry: u32,
    pub overflow: u32,
}

/// Whether a flag write continues a multi-word operation (.X): the zero
/// flag then also requires the previous words' zero flag
/// (InstEmitAluHelper.cs:102-114).
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub(super) enum Words {
    Single = 0,
    Extended = 1,
}

/// An integer sum and its unsigned carry out (a bool).
#[derive(Clone, Copy)]
pub(super) struct Carried {
    pub sum: u32,
    pub carry: u32,
}

/// Writes condition codes for one instruction.
pub(super) struct CcWriter<'a> {
    pub w: &'a mut Vec<u32>,
    pub id: &'a mut u32,
    pub uint: u32,
    pub sint: u32,
    pub boolt: u32,
    /// The uint constants 0, 1 and 31.
    pub zero: u32,
    pub one: u32,
    pub shift31: u32,
    pub flags: CcFlags,
    /// The guard predicate variable and whether it is inverted.
    pub pred: Option<(u32, bool)>,
}

impl CcWriter<'_> {
    fn load(&mut self, variable: u32) -> u32 {
        alu(self.w, self.id, Op::Load, self.boolt, &[variable])
    }

    /// Stores `value` to a flag, keeping the old value where the guard is
    /// false.
    fn store(&mut self, flag: u32, value: u32) {
        let value = self
            .pred
            .into_iter()
            .fold(value, |value, (guard, inverted)| {
                let guard = alu(self.w, self.id, Op::Load, self.boolt, &[guard]);
                let guard = alu_when(
                    self.w,
                    self.id,
                    inverted,
                    Op::LogicalNot,
                    self.boolt,
                    &[guard],
                    guard,
                );
                let old = alu(self.w, self.id, Op::Load, self.boolt, &[flag]);
                alu(
                    self.w,
                    self.id,
                    Op::Select,
                    self.boolt,
                    &[guard, value, old],
                )
            });
        inst(self.w, Op::Store, &[flag, value]);
    }

    /// The carry flag as the word 0 or 1 (`GetCF() & 1`).
    pub fn carry_word(&mut self) -> u32 {
        let carry = self.load(self.flags.carry);
        alu(
            self.w,
            self.id,
            Op::Select,
            self.uint,
            &[carry, self.one, self.zero],
        )
    }

    /// SetZnFlags (InstEmitAluHelper.cs:95-122): Z is result == 0, ANDed
    /// with the previous Z for an extended word; N is result < 0 signed.
    pub fn set_zn(&mut self, result: u32, words: Words) {
        let is_zero = alu(
            self.w,
            self.id,
            Op::IEqual,
            self.boolt,
            &[result, self.zero],
        );
        let is_zero = match words {
            Words::Single => is_zero,
            Words::Extended => {
                let previous = self.load(self.flags.zero);
                alu(
                    self.w,
                    self.id,
                    Op::LogicalAnd,
                    self.boolt,
                    &[is_zero, previous],
                )
            }
        };
        self.store(self.flags.zero, is_zero);
        let negative = alu(
            self.w,
            self.id,
            Op::SLessThan,
            self.boolt,
            &[result, self.zero],
        );
        self.store(self.flags.sign, negative);
    }

    /// SetFPZnFlags (InstEmitAluHelper.cs:124-138): Z is result == +0 and
    /// N is result < +0, both ordered (Ryujinx CodeGen/Spirv/
    /// Instructions.cs:349-377 emits FOrdEqual/FOrdLessThan), so NaN
    /// clears both. `float_zero` is +0 in the result's binary32 or
    /// binary64 type.
    pub fn set_fp_zn(&mut self, result: u32, float_zero: u32) {
        let is_zero = alu(
            self.w,
            self.id,
            Op::FOrdEqual,
            self.boolt,
            &[result, float_zero],
        );
        self.store(self.flags.zero, is_zero);
        let negative = alu(
            self.w,
            self.id,
            Op::FOrdLessThan,
            self.boolt,
            &[result, float_zero],
        );
        self.store(self.flags.sign, negative);
    }

    /// AddWithCarry (InstEmitAluHelper.cs:149-156): the 32-bit sum and its
    /// carry out, sum < lhs unsigned. Writes no flag.
    pub fn add_with_carry(&mut self, lhs: u32, rhs: u32) -> Carried {
        let sum = alu(self.w, self.id, Op::IAdd, self.uint, &[lhs, rhs]);
        let carry = alu(self.w, self.id, Op::ULessThan, self.boolt, &[sum, lhs]);
        Carried { sum, carry }
    }

    /// SetIaddFlags (InstEmitIntegerArithmetic.cs:665-699) for
    /// `result = lhs + rhs (+ C for an extended word)`: C is result < lhs,
    /// or for an extended word (result == lhs && C) || result < lhs; V is
    /// the sign of (result ^ lhs) & ~(lhs ^ rhs); then SetZnFlags.
    pub fn set_add(&mut self, result: u32, lhs: u32, rhs: u32, words: Words) {
        let below = alu(self.w, self.id, Op::ULessThan, self.boolt, &[result, lhs]);
        let carry = match words {
            Words::Single => below,
            Words::Extended => {
                let same = alu(self.w, self.id, Op::IEqual, self.boolt, &[result, lhs]);
                let carry_in = self.load(self.flags.carry);
                let through = alu(
                    self.w,
                    self.id,
                    Op::LogicalAnd,
                    self.boolt,
                    &[same, carry_in],
                );
                alu(
                    self.w,
                    self.id,
                    Op::LogicalOr,
                    self.boolt,
                    &[through, below],
                )
            }
        };
        let changed = alu(self.w, self.id, Op::BitwiseXor, self.sint, &[result, lhs]);
        let differ = alu(self.w, self.id, Op::BitwiseXor, self.sint, &[lhs, rhs]);
        let alike = alu(self.w, self.id, Op::Not, self.sint, &[differ]);
        let both = alu(
            self.w,
            self.id,
            Op::BitwiseAnd,
            self.sint,
            &[changed, alike],
        );
        let overflow = alu(
            self.w,
            self.id,
            Op::SLessThan,
            self.boolt,
            &[both, self.zero],
        );
        // The previous Z and C are read above, before any flag store.
        self.set_zn(result, words);
        self.store(self.flags.carry, carry);
        self.store(self.flags.overflow, overflow);
    }
}
