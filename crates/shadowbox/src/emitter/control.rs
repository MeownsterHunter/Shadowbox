//! Control-class lowering: warp votes and flag transfers.
//!
//! VOTE adapts Ryujinx Instructions/InstEmitWarp.cs:48-143 and R2P's flag
//! bank Instructions/InstEmitMove.cs:254-270, both under the notice below.
//!
//! Copyright (c) Ryujinx Team and Contributors
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

use super::alu;
use crate::fast_hash::HashMap;
use crate::ir::VoteOp;
use crate::spirv::Op;

/// What VOTE writes: the vote predicate and the ballot register.
pub(super) struct Vote {
    pub vote: u32,
    pub ballot: u32,
}

pub(super) struct WarpEmitter<'a> {
    pub w: &'a mut Vec<u32>,
    pub id: &'a mut u32,
    pub uint: u32,
    pub boolt: u32,
    pub uint_vec4: u32,
    /// The Subgroup scope constant.
    pub subgroup: u32,
    /// The bool true constant.
    pub yes: u32,
    /// The uint zero constant.
    pub zero: u32,
    pub constants: &'a HashMap<u32, u32>,
}

impl WarpEmitter<'_> {
    /// The 32-bit word of `condition`'s ballot that holds this lane's
    /// Maxwell warp. Ryujinx EmitBallot (InstEmitWarp.cs:110-143) selects
    /// word lane >> 5 for every host subgroup size; a dynamic extract does
    /// the same selection with no compare chain, and is word 0 on hosts of
    /// at most 32 lanes.
    fn ballot(&mut self, condition: u32, lane: u32) -> u32 {
        let mask = alu(
            self.w,
            self.id,
            Op::GroupNonUniformBallot,
            self.uint_vec4,
            &[self.subgroup, condition],
        );
        let word = alu(
            self.w,
            self.id,
            Op::ShiftRightLogical,
            self.uint,
            &[lane, self.constants[&5]],
        );
        alu(
            self.w,
            self.id,
            Op::VectorExtractDynamic,
            self.uint,
            &[mask, word],
        )
    }

    /// VOTE over this lane's 32-lane warp. Ryujinx EmitVote
    /// (InstEmitWarp.cs:73-108) votes from ballot words on hosts wider than
    /// 32 lanes: all is ballot(p) == ballot(true), any is ballot(p) != 0,
    /// and equal is all or none. On narrower hosts the same words give the
    /// subgroup vote it uses there, so one lowering serves every size.
    pub fn vote(&mut self, op: VoteOp, source: u32, lane: u32) -> Vote {
        let ballot = self.ballot(source, lane);
        let none = alu(
            self.w,
            self.id,
            Op::IEqual,
            self.boolt,
            &[ballot, self.zero],
        );
        let vote = match op {
            VoteOp::Any => alu(self.w, self.id, Op::LogicalNot, self.boolt, &[none]),
            VoteOp::All => self.all(ballot, lane),
            VoteOp::Eq => {
                let all = self.all(ballot, lane);
                alu(self.w, self.id, Op::LogicalOr, self.boolt, &[all, none])
            }
        };
        Vote { vote, ballot }
    }

    fn all(&mut self, ballot: u32, lane: u32) -> u32 {
        let active = self.ballot(self.yes, lane);
        alu(self.w, self.id, Op::IEqual, self.boolt, &[ballot, active])
    }
}
