//! Integer lowering with 32-bit operations only.
//!
//! Opcode names and numbers: crate::spirv (SPIR-V 1.6 sections
//! 3.3.13-3.3.15).
//! Every shift operand is below 32 before emitting a SPIR-V shift.

use super::{fresh, inst};
use crate::fast_hash::HashMap;
use crate::ir::{
    FunnelType, HalfSelect, IMadSpType, MulNegation, PartialSumShift, Relation, Signedness,
};
use crate::spirv::Op;

pub(super) struct IntegerEmitter<'a> {
    pub w: &'a mut Vec<u32>,
    pub id: &'a mut u32,
    pub uint: u32,
    pub boolt: u32,
    pub constants: &'a HashMap<u32, u32>,
}

impl IntegerEmitter<'_> {
    pub fn binary(&mut self, op: Op, a: u32, b: u32) -> u32 {
        let result = fresh(self.id);
        inst(self.w, op, &[self.uint, result, a, b]);
        result
    }
    pub fn logical(&mut self, op: Op, a: u32, b: u32) -> u32 {
        let result = fresh(self.id);
        inst(self.w, op, &[self.boolt, result, a, b]);
        result
    }
    pub fn unary(&mut self, op: Op, a: u32) -> u32 {
        let result = fresh(self.id);
        inst(self.w, op, &[self.uint, result, a]);
        result
    }
    pub fn select(&mut self, condition: u32, a: u32, b: u32) -> u32 {
        let result = fresh(self.id);
        inst(self.w, Op::Select, &[self.uint, result, condition, a, b]);
        result
    }
    pub fn not_bool(&mut self, a: u32) -> u32 {
        let result = fresh(self.id);
        inst(self.w, Op::LogicalNot, &[self.boolt, result, a]);
        result
    }
    pub fn k(&self, value: u32) -> u32 {
        self.constants[&value]
    }

    fn shift(&mut self, shift: Shift, a: u32, count: u32) -> u32 {
        // SPIR-V 3.3.14 makes oversized shift counts undefined. Mask the
        // actual shift, then select the Maxwell endpoint explicitly.
        let safe = self.binary(Op::BitwiseAnd, count, self.k(31));
        let shifted = self.binary(shift.op, a, safe);
        let endpoint = (shift.endpoint)(self, a);
        let in_range = self.logical(Op::ULessThan, count, self.k(32));
        self.select(in_range, shifted, endpoint)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn funnel(
        &mut self,
        low: u32,
        high: u32,
        count: u32,
        right: bool,
        dst_high: bool,
        ty: FunnelType,
        wrap: bool,
    ) -> u32 {
        // Ryujinx Instructions/InstEmitShift.cs:107-175 establishes the
        // funnel direction and word split. Mesa nak/ir.rs:4501-4543 defines
        // Maxwell .HI and saturation endpoints which Ryujinx omits: clamp
        // to 32/64 rather than return zero at the maximum. Use two 32-bit
        // words to avoid Int64 capability and 64-bit arithmetic.
        let bits = FUNNEL_WIDTH[ty as usize];
        let count = FUNNEL_COUNT[usize::from(wrap)](self, count, bits);
        FUNNEL[usize::from(right)][usize::from(dst_high)](
            self,
            low,
            high,
            count,
            FUNNEL_RIGHT[ty as usize],
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn funnel_immediate(
        &mut self,
        low: u32,
        high: u32,
        count: u32,
        right: bool,
        dst_high: bool,
        ty: FunnelType,
        wrap: bool,
    ) -> u32 {
        // Same Mesa nak/ir.rs:4501-4543 semantics as funnel(), with a known
        // count. Fold clamp/wrap, the word choice, and boundary selects at
        // translation time. The common immediate funnel needs three ops.
        let width = FUNNEL_WIDTH[ty as usize];
        let count = [count.min(width), count & (width - 1)][usize::from(wrap)];
        FUNNEL_IMMEDIATE[usize::from(right)][usize::from(dst_high)][count_shape(count)](
            self,
            low,
            high,
            count,
            FUNNEL_RIGHT[ty as usize],
        )
    }

    pub fn lea_hi(&mut self, low: u32, high: u32, addend: u32, shift: u8, negate: bool) -> u32 {
        // Ryujinx Instructions/InstEmitIntegerArithmetic.cs:493-519:
        // shift the pair, optionally negate the pair, then add to its high
        // word. The carry in a 64-bit negation is exactly (low == 0), so
        // use a compare/select instead of the reference's low-word add.
        // The 5-bit shift is zero (no shift emitted) or 1..31.
        let shifted = usize::from((shift + 31) >> 5);
        let shift = u32::from(shift);
        let upper = LEA_HI_UPPER[shifted](self, low, high, shift);
        let upper = LEA_HI_NEGATE[usize::from(negate)](self, low, upper, shift, shifted);
        self.binary(Op::IAdd, upper, addend)
    }

    /// IMUL/IMAD: Ryujinx Instructions/InstEmitIntegerArithmetic.cs:409-457
    /// (EmitImad). NegA negates B, NegB negates C; .HI takes the high word
    /// of the 64-bit product, with the reference's mixed-sign correction.
    /// Ryujinx skips the add of a constant-zero C (IMUL); so does this.
    /// `pair` is the {uint, uint} struct for OpUMulExtended/OpSMulExtended
    /// (SPIR-V 3.52.13; Ryujinx CodeGen/Spirv/Instructions.cs:990-1012).
    pub fn mul_add(
        &mut self,
        [a, b, c]: [u32; 3],
        high: bool,
        signed: [bool; 2],
        negate: MulNegation,
        zero_addend: bool,
        pair: u32,
    ) -> u32 {
        let [b, c] = MUL_NEGATE[negate as usize](self, b, c);
        let product = MUL_WORD[usize::from(high)](self, a, b, signed, pair);
        MUL_ADDEND[usize::from(zero_addend)](self, product, c)
    }

    /// IMADSP: Mesa nak/ir.rs OpIMadSp fold (MIT), the low word of
    /// cast(a) * cast(b) + cast(c). The low 32 bits of the 64-bit result
    /// are the 32-bit wrapping multiply-add of the extended operands.
    pub fn mad_sp(&mut self, [a, b, c]: [u32; 3], types: [IMadSpType; 3]) -> u32 {
        let [a, b, c] = [(a, types[0]), (b, types[1]), (c, types[2])]
            .map(|(value, ty)| self.extract_sp(value, ty));
        let product = self.binary(Op::IMul, a, b);
        self.binary(Op::IAdd, product, c)
    }

    /// IMadSpSrcType::cast: shift the field to the top, then back down,
    /// arithmetically for a signed type.
    fn extract_sp(&mut self, value: u32, ty: IMadSpType) -> u32 {
        let (left, right) = IMADSP_SHIFTS[ty as usize >> 1];
        let top = self.binary(Op::ShiftLeftLogical, value, self.k(left));
        let down = [Op::ShiftRightLogical, Op::ShiftRightArithmetic][ty as usize & 1];
        self.binary(down, top, self.k(right))
    }

    pub fn add3(
        &mut self,
        operands: [u32; 3],
        halves: [HalfSelect; 3],
        shift: PartialSumShift,
        negate: [bool; 3],
    ) -> u32 {
        // Ryujinx Instructions/InstEmitIntegerArithmetic.cs:343-404.
        // Half extraction is unsigned and occurs before negation; the
        // 16-bit shift applies to A+B, before C is added.
        let mut values = operands;
        for i in 0..3 {
            values[i] = HALF[halves[i] as usize](self, values[i]);
            values[i] = NEGATE[usize::from(negate[i])](self, values[i]);
        }
        let ab = self.binary(Op::IAdd, values[0], values[1]);
        let ab = PARTIAL_SUM_SHIFT[shift as usize](self, ab);
        self.binary(Op::IAdd, ab, values[2])
    }

    pub fn bfe(&mut self, base: u32, range: u32, signed: bool, reverse: bool) -> u32 {
        // Ryujinx Instructions/InstEmitBitfield.cs:128-151: range's low
        // two bytes are position/count. No invented wrapping or clamping
        // for the reference's unresolved runtime corner cases.
        let offset = self.binary(Op::BitwiseAnd, range, self.k(255));
        let size = self.binary(Op::ShiftRightLogical, range, self.k(8));
        let size = self.binary(Op::BitwiseAnd, size, self.k(255));
        self.extract(base, offset, size, signed, reverse)
    }

    pub fn bfe_immediate(&mut self, base: u32, range: u32, signed: bool, reverse: bool) -> u32 {
        // Fold the byte extraction in InstEmitBitfield.cs:141-143 at
        // translation time: one extract instead of four instructions.
        self.extract(
            base,
            self.k(range & 255),
            self.k((range >> 8) & 255),
            signed,
            reverse,
        )
    }

    fn extract(&mut self, base: u32, offset: u32, size: u32, signed: bool, reverse: bool) -> u32 {
        // Ryujinx CodeGen/Spirv/Instructions.cs:265-282,2139-2152.
        // Reverse the base before extracting the field, with signed or
        // unsigned extension (SPIR-V 3.3.14).
        let base = REVERSE[usize::from(reverse)](self, base);
        let result = fresh(self.id);
        inst(
            self.w,
            EXTRACT[usize::from(signed)],
            &[self.uint, result, base, offset, size],
        );
        result
    }

    pub fn extended_compare(
        &mut self,
        compare: Relation,
        signedness: Signedness,
        a: u32,
        b: u32,
        z: u32,
        carry: u32,
    ) -> u32 {
        // Ryujinx Instructions/InstEmitIntegerComparison.cs:207-266.
        // Lexicographic high-word compare, with low-word equality/ordering
        // from Z/C. Avoid its dead subtract-and-borrow expression.
        let eq = self.logical(Op::IEqual, a, b);
        EXTENDED_COMPARE[compare as usize](
            self,
            ExtendedCompare {
                eq,
                z,
                carry,
                a,
                b,
                signedness,
            },
        )
    }

    /// Lexicographic tail: the high words order strictly, or tie with the
    /// low-word relation `lower`.
    fn ordered_or_tie(&mut self, c: ExtendedCompare, lower: u32, strict: Relation) -> u32 {
        let tie = self.logical(Op::LogicalAnd, c.eq, lower);
        let ordered = self.logical(super::int_compare_op(strict, c.signedness), c.a, c.b);
        self.logical(Op::LogicalOr, ordered, tie)
    }
}

/// A SPIR-V shift with the Maxwell result for counts of 32 or more.
#[derive(Clone, Copy)]
struct Shift {
    op: Op,
    endpoint: fn(&mut IntegerEmitter<'_>, u32) -> u32,
}

/// Logical shifts move every bit out: the endpoint is zero.
fn zero_endpoint(e: &mut IntegerEmitter<'_>, _: u32) -> u32 {
    e.k(0)
}
/// An arithmetic right shift leaves only the sign.
fn sign_endpoint(e: &mut IntegerEmitter<'_>, a: u32) -> u32 {
    e.binary(Op::ShiftRightArithmetic, a, e.k(31))
}

const LEFT: Shift = Shift {
    op: Op::ShiftLeftLogical,
    endpoint: zero_endpoint,
};
const LOGICAL_RIGHT: Shift = Shift {
    op: Op::ShiftRightLogical,
    endpoint: zero_endpoint,
};
const ARITHMETIC_RIGHT: Shift = Shift {
    op: Op::ShiftRightArithmetic,
    endpoint: sign_endpoint,
};

/// Per FunnelType (Bits32, Unsigned64, Signed64): pair width and right shift.
const FUNNEL_WIDTH: [u32; 3] = [32, 64, 64];
const FUNNEL_RIGHT: [Shift; 3] = [LOGICAL_RIGHT, LOGICAL_RIGHT, ARITHMETIC_RIGHT];

type CountBound = fn(&mut IntegerEmitter<'_>, u32, u32) -> u32;
/// By wrap: clamp the runtime count to the width, or wrap it.
const FUNNEL_COUNT: [CountBound; 2] = [clamp_count, wrap_count];
fn clamp_count(e: &mut IntegerEmitter<'_>, count: u32, bits: u32) -> u32 {
    let below = e.logical(Op::ULessThan, count, e.k(bits));
    e.select(below, count, e.k(bits))
}
fn wrap_count(e: &mut IntegerEmitter<'_>, count: u32, bits: u32) -> u32 {
    e.binary(Op::BitwiseAnd, count, e.k(bits - 1))
}

type Funnel = fn(&mut IntegerEmitter<'_>, u32, u32, u32, Shift) -> u32;
/// By [right][dst_high]. A left funnel always produces the high word.
const FUNNEL: [[Funnel; 2]; 2] = [
    [funnel_left, funnel_left],
    [funnel_right, funnel_right_high],
];
fn funnel_left(e: &mut IntegerEmitter<'_>, low: u32, high: u32, count: u32, _: Shift) -> u32 {
    let complement = e.binary(Op::ISub, e.k(32), count);
    let a = e.shift(LEFT, high, count);
    let b = e.shift(LOGICAL_RIGHT, low, complement);
    let joined = e.binary(Op::BitwiseOr, a, b);
    let count_high = e.binary(Op::ISub, count, e.k(32));
    let other = e.shift(LEFT, low, count_high);
    let first_word = e.logical(Op::ULessThan, count, e.k(32));
    e.select(first_word, joined, other)
}
fn funnel_right(e: &mut IntegerEmitter<'_>, low: u32, high: u32, count: u32, right: Shift) -> u32 {
    let complement = e.binary(Op::ISub, e.k(32), count);
    let a = e.shift(LOGICAL_RIGHT, low, count);
    let b = e.shift(LEFT, high, complement);
    let joined = e.binary(Op::BitwiseOr, a, b);
    let count_high = e.binary(Op::ISub, count, e.k(32));
    let other = e.shift(right, high, count_high);
    let first_word = e.logical(Op::ULessThan, count, e.k(32));
    e.select(first_word, joined, other)
}
fn funnel_right_high(
    e: &mut IntegerEmitter<'_>,
    _: u32,
    high: u32,
    count: u32,
    right: Shift,
) -> u32 {
    e.shift(right, high, count)
}

/// The shape of a known count in 0..=64: 0, 1..31, 32, 33..63 or 64.
/// Whole words give the pair, a nonzero remainder the odd member.
fn count_shape(count: u32) -> usize {
    ((count >> 5) * 2 + (((count & 31) + 31) >> 5)) as usize
}

/// By [right][dst_high][count_shape].
const FUNNEL_IMMEDIATE: [[[Funnel; 5]; 2]; 2] = [
    [LEFT_IMMEDIATE, LEFT_IMMEDIATE],
    [
        [
            take_low,
            right_joined,
            take_high,
            right_high_past_word,
            right_endpoint,
        ],
        [
            take_high,
            right_high_within_word,
            right_endpoint,
            right_endpoint,
            right_endpoint,
        ],
    ],
];
const LEFT_IMMEDIATE: [Funnel; 5] = [
    take_high,
    left_joined,
    take_low,
    left_low_past_word,
    left_zero,
];
fn take_low(_: &mut IntegerEmitter<'_>, low: u32, _: u32, _: u32, _: Shift) -> u32 {
    low
}
fn take_high(_: &mut IntegerEmitter<'_>, _: u32, high: u32, _: u32, _: Shift) -> u32 {
    high
}
fn left_zero(e: &mut IntegerEmitter<'_>, _: u32, _: u32, _: u32, _: Shift) -> u32 {
    e.k(0)
}
fn right_endpoint(e: &mut IntegerEmitter<'_>, _: u32, high: u32, _: u32, right: Shift) -> u32 {
    (right.endpoint)(e, high)
}
fn right_high_within_word(
    e: &mut IntegerEmitter<'_>,
    _: u32,
    high: u32,
    count: u32,
    right: Shift,
) -> u32 {
    e.binary(right.op, high, e.k(count))
}
fn right_high_past_word(
    e: &mut IntegerEmitter<'_>,
    _: u32,
    high: u32,
    count: u32,
    right: Shift,
) -> u32 {
    e.binary(right.op, high, e.k(count - 32))
}
fn left_low_past_word(e: &mut IntegerEmitter<'_>, low: u32, _: u32, count: u32, _: Shift) -> u32 {
    e.binary(Op::ShiftLeftLogical, low, e.k(count - 32))
}
fn right_joined(e: &mut IntegerEmitter<'_>, low: u32, high: u32, count: u32, _: Shift) -> u32 {
    let a = e.binary(Op::ShiftRightLogical, low, e.k(count));
    let b = e.binary(Op::ShiftLeftLogical, high, e.k(32 - count));
    e.binary(Op::BitwiseOr, a, b)
}
fn left_joined(e: &mut IntegerEmitter<'_>, low: u32, high: u32, count: u32, _: Shift) -> u32 {
    let a = e.binary(Op::ShiftLeftLogical, high, e.k(count));
    let b = e.binary(Op::ShiftRightLogical, low, e.k(32 - count));
    e.binary(Op::BitwiseOr, a, b)
}

/// Builds the LEA.HI upper word from low, high and shift.
type LeaHiUpper = fn(&mut IntegerEmitter<'_>, u32, u32, u32) -> u32;
/// Applies LEA.HI negation to low, upper, shift and the shift shape.
type LeaHiNegate = fn(&mut IntegerEmitter<'_>, u32, u32, u32, usize) -> u32;

/// LEA.HI upper word by shift shape: unshifted, or funnelled 1..31.
const LEA_HI_UPPER: [LeaHiUpper; 2] = [
    |_, _, high, _| high,
    |e, low, high, shift| {
        let a = e.binary(Op::ShiftRightLogical, low, e.k(32 - shift));
        let b = e.binary(Op::ShiftLeftLogical, high, e.k(shift));
        e.binary(Op::BitwiseOr, a, b)
    },
];
/// LEA.HI lower word by shift shape, used only by the negated pair.
const LEA_HI_LOWER: [fn(&mut IntegerEmitter<'_>, u32, u32) -> u32; 2] = [
    |_, low, _| low,
    |e, low, shift| e.binary(Op::ShiftLeftLogical, low, e.k(shift)),
];
/// LEA.HI by negate: keep the upper word, or negate the 64-bit pair.
const LEA_HI_NEGATE: [LeaHiNegate; 2] = [
    |_, _, upper, _, _| upper,
    |e, low, upper, shift, shifted| {
        let lower = LEA_HI_LOWER[shifted](e, low, shift);
        let carry = e.logical(Op::IEqual, lower, e.k(0));
        let carry = e.select(carry, e.k(1), e.k(0));
        let complement = e.unary(Op::Not, upper);
        e.binary(Op::IAdd, complement, carry)
    },
];

/// By MulNegation (None, Multiplier, Addend): B and C after AvgMode.
type MulNegate = fn(&mut IntegerEmitter<'_>, u32, u32) -> [u32; 2];
const MUL_NEGATE: [MulNegate; 3] = [
    |_, b, c| [b, c],
    |e, b, c| [e.binary(Op::ISub, e.k(0), b), c],
    |e, b, c| [b, e.binary(Op::ISub, e.k(0), c)],
];
type MulWord = fn(&mut IntegerEmitter<'_>, u32, u32, [bool; 2], u32) -> u32;
/// By high: the low word (independent of signedness), or the high word.
const MUL_WORD: [MulWord; 2] = [|e, a, b, _, _| e.binary(Op::IMul, a, b), mul_high];
/// Ryujinx InstEmitIntegerArithmetic.cs:424-446: both signed is a signed
/// high multiply; otherwise the unsigned high word, plus the other
/// operand times the signed operand's sign word (0 or -1).
fn mul_high(e: &mut IntegerEmitter<'_>, a: u32, b: u32, signed: [bool; 2], pair: u32) -> u32 {
    let both = usize::from(signed[0] & signed[1]);
    let product = fresh(e.id);
    inst(
        e.w,
        [Op::UMulExtended, Op::SMulExtended][both],
        &[pair, product, a, b],
    );
    let word = fresh(e.id);
    inst(e.w, Op::CompositeExtract, &[e.uint, word, product, 1]);
    // signedA alone corrects by B, else signedB alone by A (an if/else
    // if in the reference: never both, since both took the signed path).
    let corrections = [
        (signed[0] & !signed[1], a, b),
        (signed[1] & !signed[0], b, a),
    ];
    corrections
        .into_iter()
        .filter(|&(applies, _, _)| applies)
        .fold(word, |word, (_, sign_of, times)| {
            let sign = e.binary(Op::ShiftRightArithmetic, sign_of, e.k(31));
            let correction = e.binary(Op::IMul, times, sign);
            e.binary(Op::IAdd, word, correction)
        })
}
/// By zero_addend: add C, or keep the product (Ryujinx :448-451).
const MUL_ADDEND: [fn(&mut IntegerEmitter<'_>, u32, u32) -> u32; 2] =
    [|e, p, c| e.binary(Op::IAdd, p, c), |_, p, _| p];

type Unary = fn(&mut IntegerEmitter<'_>, u32) -> u32;
/// By HalfSelect (Full, Low, High).
const HALF: [Unary; 3] = [
    |_, v| v,
    |e, v| e.binary(Op::BitwiseAnd, v, e.k(0xffff)),
    |e, v| e.binary(Op::ShiftRightLogical, v, e.k(16)),
];
/// By negate.
const NEGATE: [Unary; 2] = [|_, v| v, |e, v| e.binary(Op::ISub, e.k(0), v)];
/// By PartialSumShift (None, Right, Left).
const PARTIAL_SUM_SHIFT: [Unary; 3] = [
    |_, v| v,
    |e, v| e.binary(Op::ShiftRightLogical, v, e.k(16)),
    |e, v| e.binary(Op::ShiftLeftLogical, v, e.k(16)),
];
/// By reverse.
const REVERSE: [Unary; 2] = [|_, v| v, |e, v| e.unary(Op::BitReverse, v)];
/// By signed.
const EXTRACT: [Op; 2] = [Op::BitFieldUExtract, Op::BitFieldSExtract];

/// The operands of an extended (.X) compare after its high-word equality.
#[derive(Clone, Copy)]
struct ExtendedCompare {
    eq: u32,
    z: u32,
    carry: u32,
    a: u32,
    b: u32,
    signedness: Signedness,
}

/// By Relation (Lt, Eq, Le, Gt, Ne, Ge).
const EXTENDED_COMPARE: [fn(&mut IntegerEmitter<'_>, ExtendedCompare) -> u32; 6] = [
    |e, c| {
        let lower = e.not_bool(c.carry);
        e.ordered_or_tie(c, lower, Relation::Lt)
    },
    |e, c| e.logical(Op::LogicalAnd, c.eq, c.z),
    |e, c| {
        let borrow = e.not_bool(c.carry);
        let lower = e.logical(Op::LogicalOr, c.z, borrow);
        e.ordered_or_tie(c, lower, Relation::Lt)
    },
    |e, c| {
        let nz = e.not_bool(c.z);
        let lower = e.logical(Op::LogicalAnd, nz, c.carry);
        e.ordered_or_tie(c, lower, Relation::Gt)
    },
    |e, c| {
        let same = e.logical(Op::LogicalAnd, c.eq, c.z);
        e.not_bool(same)
    },
    |e, c| e.ordered_or_tie(c, c.carry, Relation::Gt),
];

#[cfg(test)]
mod tests;

/// (left, right) shifts per IMADSP width: 32, 24, low 16, high 16.
const IMADSP_SHIFTS: [(u32, u32); 4] = [(0, 0), (8, 8), (16, 16), (0, 16)];
