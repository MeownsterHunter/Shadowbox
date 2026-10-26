//! Exact I2F rounding. Encodings: Mesa nak/sm50.rs:1848-1879 and
//! envytools envydis/gm107.c:1256-1280,1995. Operand order and output packing:
//! Ryujinx Instructions/InstEmitConversion.cs:257-281,398-415. Ryujinx does
//! not apply RoundMode (its call at :73 omits it) and has no S/U64 (:255).
//! These independently written integer formulas extend those semantics.
use super::{alu, fresh, inst};
use crate::ir::RoundingMode;
use crate::spirv::Op;

pub(super) struct ConversionEmitter<'a> {
    pub w: &'a mut Vec<u32>,
    pub globals: &'a mut Vec<u32>,
    pub id: &'a mut u32,
    pub uint: u32,
    pub boolt: u32,
    pub glsl: u32,
    /// uint32 for half conversion, uint64 for 64-bit sources/back-conversion.
    pub integer: u32,
    pub wide: bool,
}

impl ConversionEmitter<'_> {
    fn k(&mut self, value: u64) -> u32 {
        let result = fresh(self.id);
        CONSTANT[usize::from(self.wide)](self.globals, self.integer, result, value);
        result
    }
    fn binary(&mut self, op: Op, a: u32, b: u32) -> u32 {
        alu(self.w, self.id, op, self.integer, &[a, b])
    }
    fn compare(&mut self, op: Op, a: u32, b: u32) -> u32 {
        alu(self.w, self.id, op, self.boolt, &[a, b])
    }
    fn select(&mut self, condition: u32, a: u32, b: u32) -> u32 {
        alu(
            self.w,
            self.id,
            Op::Select,
            self.integer,
            &[condition, a, b],
        )
    }
    fn word(&mut self, op: Op, a: u32, b: u32) -> u32 {
        alu(self.w, self.id, op, self.uint, &[a, b])
    }
    pub fn join(&mut self, low: u32, high: u32) -> u32 {
        // Mesa nak/ir.rs:352-353,394-397 defines contiguous register
        // components. Low/high order follows from_nir.rs:751-754,1842-1848.
        let low = alu(self.w, self.id, Op::UConvert, self.integer, &[low]);
        let high = alu(self.w, self.id, Op::UConvert, self.integer, &[high]);
        let shift = self.k(32);
        let high = self.binary(Op::ShiftLeftLogical, high, shift);
        self.binary(Op::BitwiseOr, low, high)
    }
    pub fn modifiers(&mut self, a: u32, abs: bool, negate: bool) -> u32 {
        // Mesa sm50.rs:1852 encodes integer negation on the complete source;
        // nv50_ir_emit_gm107.cpp:908-910 also encodes abs49 then neg45.
        // Ryujinx InstEmitConversion.cs:257 uses signed abs before negation.
        // Unsigned subtraction handles INT64_MIN without signed overflow.
        let zero = self.k(0);
        let a = ABSOLUTE[usize::from(abs)](self, a, zero);
        NEGATE[usize::from(negate)](self, a, zero)
    }
    pub fn directed_f32(&mut self, a: u32, f: u32, signed: bool, mode: RoundingMode) -> u32 {
        // Ryujinx CodeGen/Spirv/Instructions.cs:463-488 maps the native
        // conversion to Convert[S/U]ToF. Mesa sm50.rs:467-475 defines the
        // directed modes. For any 32-bit input, the rounded float is in
        // [-2^31, 2^32], so signed 64-bit back-conversion is always defined.
        // Compare integers, then move one ULP with integer bit arithmetic:
        // no float arithmetic, driver rounding decorations or branch needed.
        // Signedness and mode fold at translation time: RZ needs only a
        // decrement, and RM/RP emit just their relevant integer comparison.
        // Nearest-even is the native conversion's own result.
        DIRECTED_F32[mode as usize][usize::from(signed)](self, a, f)
    }

    /// The source widened to the integer type, its back-conversion and zero.
    fn directed_start(&mut self, a: u32, f: u32, convert: Op) -> (u32, u32, u32) {
        let original = alu(self.w, self.id, convert, self.integer, &[a]);
        let back = alu(self.w, self.id, Op::ConvertFToS, self.integer, &[f]);
        let zero = self.k(0);
        (original, back, zero)
    }

    /// Step the float's bits one ULP where `adjust` holds.
    fn directed_finish(
        &mut self,
        f: u32,
        adjust: u32,
        step: impl FnOnce(&mut Self, u32, u32) -> u32,
    ) -> u32 {
        let raw = alu(self.w, self.id, Op::Bitcast, self.uint, &[f]);
        let one = super::late_constant(self.globals, self.id, self.uint, 1);
        let stepped = step(self, raw, one);
        alu(
            self.w,
            self.id,
            Op::Select,
            self.uint,
            &[adjust, stepped, raw],
        )
    }

    /// A signed source steps away from or toward zero by its sign.
    fn signed_step(&mut self, negative: u32, raw: u32, one: u32, positive: Op, other: Op) -> u32 {
        let positive = self.word(positive, raw, one);
        let neg = self.word(other, raw, one);
        alu(
            self.w,
            self.id,
            Op::Select,
            self.uint,
            &[negative, neg, positive],
        )
    }

    pub fn rounded_bits(&mut self, a: u32, signed: bool, dst_size: u8, mode: RoundingMode) -> u32 {
        // Integer-only construction avoids double rounding through F32 for
        // half and avoids undefined back-conversion of rounded UINT64_MAX.
        // IEEE layouts: Mesa src/util/half_float.c:38-44,60-62,105-131;
        // src/util/u_math.h:91-100; src/util/softfloat.c:127-148,885,1372-1374.
        // Keep p+1 significand bits, inspect the discarded remainder, and
        // increment only for the requested direction. Carry into the exponent
        // happens in the same add. Every evaluated shift is below its width.
        let format = FORMAT[usize::from(dst_size) - 1];
        let zero = self.k(0);
        let one = self.k(1);
        let negative = NEGATIVE[usize::from(signed)](self, a, zero);
        let negated = self.binary(Op::ISub, zero, a);
        let magnitude = self.select(negative, negated, a);
        // GLSL.std.450 FindUMsb75 is Mesa compiler/spirv/GLSL.std.450.h:102.
        // Scan only 32-bit operands so no extended-instruction Int64 support
        // is required; zero's sentinel is replaced before using it as a shift.
        let low = LOW_WORD[usize::from(self.wide)](self, magnitude);
        let low_msb = alu(
            self.w,
            self.id,
            Op::ExtInst,
            self.uint,
            &[self.glsl, 75, low],
        );
        let exponent = MSB[usize::from(self.wide)](self, magnitude, low_msb);
        let is_zero = self.compare(Op::IEqual, magnitude, zero);
        let exponent = self.select(is_zero, zero, exponent);
        let precision = self.k(format.precision);
        let large = self.compare(Op::UGreaterThan, exponent, precision);
        let right = self.binary(Op::ISub, exponent, precision);
        let right = self.select(large, right, zero);
        let left = self.binary(Op::ISub, precision, exponent);
        let left = self.select(large, zero, left);
        let down = self.binary(Op::ShiftRightLogical, magnitude, right);
        let up = self.binary(Op::ShiftLeftLogical, magnitude, left);
        let significand = self.select(large, down, up);
        let unit = self.binary(Op::ShiftLeftLogical, one, right);
        let mask = self.binary(Op::ISub, unit, one);
        let remainder = self.binary(Op::BitwiseAnd, magnitude, mask);
        let inexact = self.compare(Op::INotEqual, remainder, zero);
        let discard = Discard {
            inexact,
            negative,
            zero,
            one,
            unit,
            remainder,
            significand,
        };
        let increment = INCREMENT[mode as usize](self, discard);
        let fraction_mask = self.k((1 << format.precision) - 1);
        let fraction = self.binary(Op::BitwiseAnd, significand, fraction_mask);
        let bias = self.k(format.bias);
        let biased = self.binary(Op::IAdd, exponent, bias);
        let base = self.binary(Op::ShiftLeftLogical, biased, precision);
        let base = self.binary(Op::BitwiseOr, base, fraction);
        let bits = self.binary(Op::IAdd, base, increment);
        let bits = (format.overflow)(self, bits, negative, mode);
        let bits = self.select(is_zero, zero, bits);
        let sign = self.k(1 << format.sign_bit);
        let sign = self.select(negative, sign, zero);
        self.binary(Op::BitwiseOr, bits, sign)
    }
}

/// Declares integer constant `result` of `value`.
type IntegerConstant = fn(&mut Vec<u32>, u32, u32, u64);
/// By wide: a 32-bit or a 64-bit integer constant.
const CONSTANT: [IntegerConstant; 2] = [
    |globals, integer, result, value| inst(globals, Op::Constant, &[integer, result, value as u32]),
    |globals, integer, result, value| {
        inst(
            globals,
            Op::Constant,
            &[integer, result, value as u32, (value >> 32) as u32],
        )
    },
];

type Modifier = fn(&mut ConversionEmitter<'_>, u32, u32) -> u32;
/// By abs: keep the source, or take its signed magnitude.
const ABSOLUTE: [Modifier; 2] = [
    |_, a, _| a,
    |e, a, zero| {
        let negative = e.compare(Op::SLessThan, a, zero);
        let magnitude = e.binary(Op::ISub, zero, a);
        e.select(negative, magnitude, a)
    },
];
/// By negate.
const NEGATE: [Modifier; 2] = [|_, a, _| a, |e, a, zero| e.binary(Op::ISub, zero, a)];

type Directed = fn(&mut ConversionEmitter<'_>, u32, u32) -> u32;
/// By [RoundingMode (NearestEven, NegativeInfinity, PositiveInfinity, Zero)][signed].
const DIRECTED_F32: [[Directed; 2]; 4] = [
    [native_bits, native_bits],
    [
        |e, a, f| {
            let (original, back, _) = e.directed_start(a, f, Op::UConvert);
            let adjust = e.compare(Op::SGreaterThan, back, original);
            e.directed_finish(f, adjust, |e, raw, one| e.word(Op::ISub, raw, one))
        },
        |e, a, f| {
            let (original, back, zero) = e.directed_start(a, f, Op::SConvert);
            let negative = e.compare(Op::SLessThan, original, zero);
            let adjust = e.compare(Op::SGreaterThan, back, original);
            e.directed_finish(f, adjust, |e, raw, one| {
                e.signed_step(negative, raw, one, Op::ISub, Op::IAdd)
            })
        },
    ],
    [
        |e, a, f| {
            let (original, back, _) = e.directed_start(a, f, Op::UConvert);
            let adjust = e.compare(Op::SLessThan, back, original);
            e.directed_finish(f, adjust, |e, raw, one| e.word(Op::IAdd, raw, one))
        },
        |e, a, f| {
            let (original, back, zero) = e.directed_start(a, f, Op::SConvert);
            let negative = e.compare(Op::SLessThan, original, zero);
            let adjust = e.compare(Op::SLessThan, back, original);
            e.directed_finish(f, adjust, |e, raw, one| {
                e.signed_step(negative, raw, one, Op::IAdd, Op::ISub)
            })
        },
    ],
    [
        |e, a, f| {
            let (original, back, _) = e.directed_start(a, f, Op::UConvert);
            let adjust = e.compare(Op::SGreaterThan, back, original);
            e.directed_finish(f, adjust, |e, raw, one| e.word(Op::ISub, raw, one))
        },
        |e, a, f| {
            let (original, back, zero) = e.directed_start(a, f, Op::SConvert);
            let negative = e.compare(Op::SLessThan, original, zero);
            let above = e.compare(Op::SGreaterThan, back, original);
            let below = e.compare(Op::SLessThan, back, original);
            let adjust = alu(e.w, e.id, Op::Select, e.boolt, &[negative, below, above]);
            e.directed_finish(f, adjust, |e, raw, one| e.word(Op::ISub, raw, one))
        },
    ],
];
/// The native conversion already rounds to nearest even: its bits are the result.
fn native_bits(e: &mut ConversionEmitter<'_>, _: u32, f: u32) -> u32 {
    alu(e.w, e.id, Op::Bitcast, e.uint, &[f])
}

/// A binary float destination: binary16, binary32 or binary64 (sizes 1..=3).
#[derive(Clone, Copy)]
struct Format {
    precision: u64,
    bias: u64,
    sign_bit: u64,
    overflow: fn(&mut ConversionEmitter<'_>, u32, u32, RoundingMode) -> u32,
}
const FORMAT: [Format; 3] = [
    Format {
        precision: 10,
        bias: 15,
        sign_bit: 15,
        overflow: half_overflow,
    },
    Format {
        precision: 23,
        bias: 127,
        sign_bit: 31,
        overflow: |_, bits, _, _| bits,
    },
    Format {
        precision: 52,
        bias: 1023,
        sign_bit: 63,
        overflow: |_, bits, _, _| bits,
    },
];
/// Binary16 has the narrowest range: a carry past 0x7bff is infinity or
/// the mode's largest finite value.
fn half_overflow(
    e: &mut ConversionEmitter<'_>,
    bits: u32,
    negative: u32,
    mode: RoundingMode,
) -> u32 {
    let infinity = e.k(0x7c00);
    let max = e.k(0x7bff);
    let overflow = e.compare(Op::UGreaterThanEqual, bits, infinity);
    let endpoint = HALF_ENDPOINT[mode as usize](e, negative, infinity, max);
    e.select(overflow, endpoint, bits)
}
/// Picks the overflow endpoint from the sign, infinity and max.
type HalfEndpoint = fn(&mut ConversionEmitter<'_>, u32, u32, u32) -> u32;
/// By RoundingMode (NearestEven, NegativeInfinity, PositiveInfinity, Zero).
const HALF_ENDPOINT: [HalfEndpoint; 4] = [
    |_, _, infinity, _| infinity,
    |e, negative, infinity, max| e.select(negative, infinity, max),
    |e, negative, infinity, max| e.select(negative, max, infinity),
    |_, _, _, max| max,
];

/// By signed: an unsigned source is never negative.
const NEGATIVE: [Modifier; 2] = [
    |e, _, zero| e.compare(Op::INotEqual, zero, zero),
    |e, a, zero| e.compare(Op::SLessThan, a, zero),
];
/// By wide: the low 32-bit word of the magnitude.
const LOW_WORD: [fn(&mut ConversionEmitter<'_>, u32) -> u32; 2] = [
    |_, magnitude| magnitude,
    |e, magnitude| alu(e.w, e.id, Op::UConvert, e.uint, &[magnitude]),
];
/// By wide: the magnitude's most significant bit from its 32-bit words.
const MSB: [fn(&mut ConversionEmitter<'_>, u32, u32) -> u32; 2] = [
    |_, _, low_msb| low_msb,
    |e, magnitude, low_msb| {
        let shift = e.k(32);
        let high = e.binary(Op::ShiftRightLogical, magnitude, shift);
        let high = alu(e.w, e.id, Op::UConvert, e.uint, &[high]);
        let high_msb = alu(e.w, e.id, Op::ExtInst, e.uint, &[e.glsl, 75, high]);
        let thirty_two = super::late_constant(e.globals, e.id, e.uint, 32);
        let high_msb = alu(e.w, e.id, Op::IAdd, e.uint, &[high_msb, thirty_two]);
        let zero32 = super::late_constant(e.globals, e.id, e.uint, 0);
        let high_nonzero = e.compare(Op::INotEqual, high, zero32);
        let msb = alu(
            e.w,
            e.id,
            Op::Select,
            e.uint,
            &[high_nonzero, high_msb, low_msb],
        );
        alu(e.w, e.id, Op::UConvert, e.integer, &[msb])
    },
];

/// The bits a rounding discards, with the values its increment needs.
#[derive(Clone, Copy)]
struct Discard {
    inexact: u32,
    negative: u32,
    zero: u32,
    one: u32,
    unit: u32,
    remainder: u32,
    significand: u32,
}
/// By RoundingMode (NearestEven, NegativeInfinity, PositiveInfinity, Zero).
const INCREMENT: [fn(&mut ConversionEmitter<'_>, Discard) -> u32; 4] = [
    |e, d| {
        // Above half rounds up; exactly half rounds to the even significand.
        let half = e.binary(Op::ShiftRightLogical, d.unit, d.one);
        let above = e.compare(Op::UGreaterThan, d.remainder, half);
        let tie = e.compare(Op::IEqual, d.remainder, half);
        let low = e.binary(Op::BitwiseAnd, d.significand, d.one);
        let odd = e.compare(Op::INotEqual, low, d.zero);
        let tie_odd = alu(e.w, e.id, Op::LogicalAnd, e.boolt, &[tie, odd]);
        let up = alu(e.w, e.id, Op::LogicalOr, e.boolt, &[above, tie_odd]);
        let bump = alu(e.w, e.id, Op::LogicalAnd, e.boolt, &[d.inexact, up]);
        e.select(bump, d.one, d.zero)
    },
    |e, d| directed_increment(e, d, d.negative),
    |e, d| {
        let positive = alu(e.w, e.id, Op::LogicalNot, e.boolt, &[d.negative]);
        directed_increment(e, d, positive)
    },
    |_, d| d.zero,
];
fn directed_increment(e: &mut ConversionEmitter<'_>, d: Discard, direction: u32) -> u32 {
    let bump = alu(e.w, e.id, Op::LogicalAnd, e.boolt, &[d.inexact, direction]);
    e.select(bump, d.one, d.zero)
}
