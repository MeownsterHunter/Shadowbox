//! Half-precision arithmetic and compare, one constructor per opcode form.
//! The opcode and the form fields read below already name the operation
//! (FMA, ADD, MUL, SET, SETP) and the B operand's encoding (register, 32-bit
//! literal, short literal, constant buffer, or the RC form's exchanged B/C),
//! so each constructor builds its instruction from fixed field positions.
//!
//! Half arithmetic fields/forms: envytools envydis/gm107.c:1977-1989,
//! 2155-2157. dst 0..7, A 8..15, register B 20..27, C 39..46
//! (register definitions :203-218); output 49..50 and A/B swizzles
//! 47..48/28..29 (:1133-1153), register FMA C swizzle 35..36
//! (:1171-1175), other C/A swizzles 53..54 (:1774-1778).
//! Reserved FTZ/FMZ selector 3 (:1157-1168,1083-1087,1782-1786) stays Unknown.

use super::*;

/// The one-bit field at `at`.
fn bit(r: u64, at: u32) -> bool {
    b(r, at, 1) != 0
}
/// The two-bit field at `at`.
fn two(r: u64, at: u32) -> u8 {
    b(r, at, 2) as u8
}
fn lhs(r: u64) -> Source {
    Source::Register(Register(b(r, 8, 8) as u8))
}
fn dst(r: u64) -> Register {
    Register(b(r, 0, 8) as u8)
}

/// The fields every half arithmetic form shares, given in the order the
/// form's constructor reads them.
struct Half {
    op: HalfOp,
    flush: HalfFlush,
    src: [Source; 3],
    swizzle: [u8; 3],
    abs: [bool; 3],
    negate: [bool; 3],
    output: u8,
    saturate: bool,
}

impl Half {
    fn arrive(self, x: Word) -> Instruction {
        x.guarded(InstructionKind::HalfArithmetic(ops::HalfArithmetic {
            op: self.op,
            flush: self.flush,
            dst: dst(x.r),
            src: self.src,
            swizzle: self.swizzle,
            abs: self.abs,
            negate: self.negate,
            output: self.output,
            saturate: self.saturate,
        }))
    }
}

const ZERO: Source = Source::Immediate(0);

// envytools :1981-1983,1989,2157: an FMA's product negation belongs on
// B; C is negated at 30 (reg), 51 (short/cbuf), 52 (32i).
fn half_fma_register(x: Word) -> Instruction {
    let r = x.r;
    Half {
        op: HalfOp::Fma,
        flush: HALF_FLUSH[b(r, 37, 2) as usize],
        src: [lhs(r), register_b(r), register_c(r)],
        swizzle: [two(r, 47), two(r, 28), two(r, 35)],
        abs: [false; 3],
        negate: [false, bit(r, 31), bit(r, 30)],
        output: two(r, 49),
        saturate: bit(r, 32),
    }
    .arrive(x)
}
const HALF_FMA_REGISTER_FIELDS: Fields<1> = Fields([Field::bits(37, 2)]);
const fn half_fma_register_rule(r: u64) -> u8 {
    (b(r, 37, 2) < 3) as u8
}
const HALF_FMA_REGISTER_FORMS: [u8; HALF_FMA_REGISTER_FIELDS.len()] =
    forms!(HALF_FMA_REGISTER_FIELDS, half_fma_register_rule);
pub(super) fn read_half_fma_register(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, half_fma_register];
    paths[(HALF_FMA_REGISTER_FORMS[HALF_FMA_REGISTER_FIELDS.index(x.r)]) as usize](x)
}
/// HFMA2_32I: C is the destination register itself.
fn half_fma_long(x: Word) -> Instruction {
    let r = x.r;
    Half {
        op: HalfOp::Fma,
        flush: HALF_FLUSH[b(r, 55, 2) as usize],
        src: [lhs(r), long_b(r), Source::Register(dst(r))],
        swizzle: [two(r, 53), 0, 0],
        abs: [false; 3],
        negate: [false, false, bit(r, 52)],
        output: 0,
        saturate: false,
    }
    .arrive(x)
}
fn half_fma_short(x: Word) -> Instruction {
    let r = x.r;
    Half {
        op: HalfOp::Fma,
        flush: HALF_FLUSH[b(r, 57, 2) as usize],
        src: [lhs(r), short_b(r), register_c(r)],
        swizzle: [two(r, 47), 0, two(r, 53)],
        abs: [false; 3],
        negate: [false, false, bit(r, 51)],
        output: two(r, 49),
        saturate: bit(r, 52),
    }
    .arrive(x)
}
fn half_fma_constant(x: Word) -> Instruction {
    let r = x.r;
    Half {
        op: HalfOp::Fma,
        flush: HALF_FLUSH[b(r, 57, 2) as usize],
        src: [lhs(r), constant_b(r), register_c(r)],
        swizzle: [two(r, 47), 1, two(r, 53)],
        abs: [false; 3],
        negate: [false, bit(r, 56), bit(r, 51)],
        output: two(r, 49),
        saturate: bit(r, 52),
    }
    .arrive(x)
}
/// The RC form exchanges B and C, swizzles included, while the negations
/// stay in place (envytools :1983; Ryujinx InstEmitFloatArithmetic.cs:
/// 324-332).
fn half_fma_constant_c(x: Word) -> Instruction {
    let r = x.r;
    Half {
        op: HalfOp::Fma,
        flush: HALF_FLUSH[b(r, 57, 2) as usize],
        src: [lhs(r), register_c(r), constant_b(r)],
        swizzle: [two(r, 47), two(r, 53), 1],
        abs: [false; 3],
        negate: [false, bit(r, 56), bit(r, 51)],
        output: two(r, 49),
        saturate: bit(r, 52),
    }
    .arrive(x)
}
/// HFMA2 short/cbuf (0x7000..0x77ff, cbuf at bit 55) and RC (0x6x80, bit
/// 60 clear, always cbuf): flush 57..58.
const HALF_FMA_SHORT_FIELDS: Fields<3> =
    Fields([Field::bits(57, 2), Field::bits(55, 1), Field::bits(60, 1)]);
const fn half_fma_short_rule(r: u64) -> u8 {
    (b(r, 57, 2) < 3) as u8 * [[0, 1], [2, 3]][b(r, 60, 1) as usize][b(r, 55, 1) as usize]
}
const HALF_FMA_SHORT_FORMS: [u8; HALF_FMA_SHORT_FIELDS.len()] =
    forms!(HALF_FMA_SHORT_FIELDS, half_fma_short_rule);
pub(super) fn read_half_fma_short(x: Word) -> Instruction {
    let paths: [Decode; 4] = [
        unknown,
        half_fma_constant_c,
        half_fma_short,
        half_fma_constant,
    ];
    paths[(HALF_FMA_SHORT_FORMS[HALF_FMA_SHORT_FIELDS.index(x.r)]) as usize](x)
}

// envytools :1977-1980,1987-1988,2155-2156: A abs44, A neg43 (short),
// B abs30/54 and neg31/56. Register HMUL2's product sign is carried by B,
// not A.
pub(super) fn half_add_register(x: Word) -> Instruction {
    let r = x.r;
    Half {
        op: HalfOp::Add,
        flush: HALF_FLUSH[b(r, 39, 1) as usize],
        src: [lhs(r), register_b(r), ZERO],
        swizzle: [two(r, 47), two(r, 28), 0],
        abs: [bit(r, 44), bit(r, 30), false],
        negate: [bit(r, 43), bit(r, 31), false],
        output: two(r, 49),
        saturate: bit(r, 32),
    }
    .arrive(x)
}
pub(super) fn half_add_long(x: Word) -> Instruction {
    let r = x.r;
    Half {
        op: HalfOp::Add,
        flush: HALF_FLUSH[b(r, 55, 1) as usize],
        src: [lhs(r), long_b(r), ZERO],
        swizzle: [two(r, 53), 0, 0],
        abs: [false; 3],
        negate: [bit(r, 56), false, false],
        output: 0,
        saturate: bit(r, 52),
    }
    .arrive(x)
}
fn half_add_short(x: Word) -> Instruction {
    let r = x.r;
    Half {
        op: HalfOp::Add,
        flush: HALF_FLUSH[b(r, 39, 1) as usize],
        src: [lhs(r), short_b(r), ZERO],
        swizzle: [two(r, 47), 0, 0],
        abs: [bit(r, 44), false, false],
        negate: [bit(r, 43), false, false],
        output: two(r, 49),
        saturate: bit(r, 52),
    }
    .arrive(x)
}
pub(super) fn half_add_constant(x: Word) -> Instruction {
    let r = x.r;
    Half {
        op: HalfOp::Add,
        flush: HALF_FLUSH[b(r, 39, 1) as usize],
        src: [lhs(r), constant_b(r), ZERO],
        swizzle: [two(r, 47), 1, 0],
        abs: [bit(r, 44), bit(r, 54), false],
        negate: [bit(r, 43), bit(r, 56), false],
        output: two(r, 49),
        saturate: bit(r, 52),
    }
    .arrive(x)
}
const HALF_ADD_SHORT_FIELDS: Fields<4> = Fields([
    Field::zero(40, 3),
    Field::zero(45, 2),
    Field::zero(51, 1),
    Field::zero(53, 2),
]);
const fn half_add_short_rule(r: u64) -> u8 {
    (b(r, 40, 3) == 0 && b(r, 45, 2) == 0 && b(r, 51, 1) == 0 && b(r, 53, 2) == 0) as u8
}
const HALF_ADD_SHORT_FORMS: [u8; HALF_ADD_SHORT_FIELDS.len()] =
    forms!(HALF_ADD_SHORT_FIELDS, half_add_short_rule);
pub(super) fn read_half_add_short(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, half_add_short];
    paths[(HALF_ADD_SHORT_FORMS[HALF_ADD_SHORT_FIELDS.index(x.r)]) as usize](x)
}

fn half_mul_register(x: Word) -> Instruction {
    let r = x.r;
    Half {
        op: HalfOp::Mul,
        flush: HALF_FLUSH[b(r, 39, 2) as usize],
        src: [lhs(r), register_b(r), ZERO],
        swizzle: [two(r, 47), two(r, 28), 0],
        abs: [bit(r, 44), bit(r, 30), false],
        negate: [false, bit(r, 31), false],
        output: two(r, 49),
        saturate: bit(r, 32),
    }
    .arrive(x)
}
const HALF_MUL_REGISTER_FIELDS: Fields<1> = Fields([Field::bits(39, 2)]);
const fn half_mul_register_rule(r: u64) -> u8 {
    (b(r, 39, 2) < 3) as u8
}
const HALF_MUL_REGISTER_FORMS: [u8; HALF_MUL_REGISTER_FIELDS.len()] =
    forms!(HALF_MUL_REGISTER_FIELDS, half_mul_register_rule);
pub(super) fn read_half_mul_register(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, half_mul_register];
    paths[(HALF_MUL_REGISTER_FORMS[HALF_MUL_REGISTER_FIELDS.index(x.r)]) as usize](x)
}
fn half_mul_long(x: Word) -> Instruction {
    let r = x.r;
    Half {
        op: HalfOp::Mul,
        flush: HALF_FLUSH[b(r, 55, 2) as usize],
        src: [lhs(r), long_b(r), ZERO],
        swizzle: [two(r, 53), 0, 0],
        abs: [false; 3],
        negate: [false; 3],
        output: 0,
        saturate: bit(r, 52),
    }
    .arrive(x)
}
/// HFMA2_32I (0x2800, bit 57 clear) and HMUL2_32I (0x2a00, bit 57 set):
/// flush 55..56.
const HALF_LONG_FIELDS: Fields<2> = Fields([Field::bits(55, 2), Field::bits(57, 1)]);
const fn half_long_rule(r: u64) -> u8 {
    (b(r, 55, 2) < 3) as u8 * (1 + b(r, 57, 1) as u8)
}
const HALF_LONG_FORMS: [u8; HALF_LONG_FIELDS.len()] = forms!(HALF_LONG_FIELDS, half_long_rule);
pub(super) fn read_half_long(x: Word) -> Instruction {
    let paths: [Decode; 3] = [unknown, half_fma_long, half_mul_long];
    paths[(HALF_LONG_FORMS[HALF_LONG_FIELDS.index(x.r)]) as usize](x)
}
fn half_mul_short(x: Word) -> Instruction {
    let r = x.r;
    Half {
        op: HalfOp::Mul,
        flush: HALF_FLUSH[b(r, 39, 2) as usize],
        src: [lhs(r), short_b(r), ZERO],
        swizzle: [two(r, 47), 0, 0],
        abs: [bit(r, 44), false, false],
        negate: [bit(r, 43), false, false],
        output: two(r, 49),
        saturate: bit(r, 52),
    }
    .arrive(x)
}
/// envytools :1980 lists cbuf HMUL2 negation on A; Ryujinx
/// InstEmitFloatArithmetic.cs:366-373 and Decoders/InstDecoders.cs:
/// 2511-2529 identify the product negation at 43, carried by B.
fn half_mul_constant(x: Word) -> Instruction {
    let r = x.r;
    Half {
        op: HalfOp::Mul,
        flush: HALF_FLUSH[b(r, 39, 2) as usize],
        src: [lhs(r), constant_b(r), ZERO],
        swizzle: [two(r, 47), 1, 0],
        abs: [bit(r, 44), bit(r, 54), false],
        negate: [false, bit(r, 43), false],
        output: two(r, 49),
        saturate: bit(r, 52),
    }
    .arrive(x)
}
/// HMUL2 short and cbuf (cbuf at bit 55): flush 39..40.
const HALF_MUL_SHORT_FIELDS: Fields<2> = Fields([Field::bits(39, 2), Field::bits(55, 1)]);
const fn half_mul_short_rule(r: u64) -> u8 {
    (b(r, 39, 2) < 3) as u8 * (1 + b(r, 55, 1) as u8)
}
const HALF_MUL_SHORT_FORMS: [u8; HALF_MUL_SHORT_FIELDS.len()] =
    forms!(HALF_MUL_SHORT_FIELDS, half_mul_short_rule);
pub(super) fn read_half_mul_short(x: Word) -> Instruction {
    let paths: [Decode; 3] = [unknown, half_mul_short, half_mul_constant];
    paths[(HALF_MUL_SHORT_FORMS[HALF_MUL_SHORT_FIELDS.index(x.r)]) as usize](x)
}

// HSET2/HSETP2 register/short/cbuf: envytools envydis/gm107.c:
// 1973-1976,1985-1986. Registers 0/8/20 and predicates 0/3/39:
// :203-218,363-365; guard16..19 :408-411. A swizzle47..48,
// B28..29 :1141-1153; cmp35..38 (register) /49..52 (other)
// :1113-1129,1063-1079; boolean op45..46 :1430-1434.
// Neg/abs A43/44, B31/30 (reg), B56/54 (cbuf HSETP2),
// accum invert42; BF/H_AND49 (reg) /53 (other). FTZ50/54
// (HSET2) or6 (HSETP2). Cbuf HSET2 bit54 is
// FTZ, not ABS: Ryujinx Decoders/InstDecoders.cs:2587-2606;
// its semantics have no ABS B (InstEmitFloatComparison.cs:329-336).

/// The fields every half compare form shares.
struct HalfCompare {
    ftz: bool,
    dst: crate::ir::HalfSetDestination,
    rhs: Source,
    swizzle: u8,
    abs: bool,
    negate: bool,
    selector: u32,
}

impl HalfCompare {
    fn arrive(self, x: Word) -> Instruction {
        let r = x.r;
        x.guarded(InstructionKind::HalfSet(ops::HalfSet {
            ftz: self.ftz,
            dst: self.dst,
            src: [lhs(r), self.rhs],
            swizzle: [two(r, 47), self.swizzle],
            abs: [bit(r, 44), self.abs],
            negate: [bit(r, 43), self.negate],
            compare: fcmp(self.selector),
            accum: b(r, 39, 3) as u8,
            accum_negate: bit(r, 42),
            set_op: SET_OPS[b(r, 45, 2) as usize],
        }))
    }
}

fn set_register(r: u64, at: u32) -> crate::ir::HalfSetDestination {
    crate::ir::HalfSetDestination::Register {
        dst: dst(r),
        bool_float: bit(r, at),
    }
}
fn set_predicates(r: u64, at: u32) -> crate::ir::HalfSetDestination {
    crate::ir::HalfSetDestination::Predicates {
        dst: [b(r, 3, 3) as u8, b(r, 0, 3) as u8],
        horizontal: bit(r, at),
    }
}

fn half_set_register(x: Word) -> Instruction {
    let r = x.r;
    HalfCompare {
        ftz: bit(r, 50),
        dst: set_register(r, 49),
        rhs: register_b(r),
        swizzle: two(r, 28),
        abs: bit(r, 30),
        negate: bit(r, 31),
        selector: b(r, 35, 4),
    }
    .arrive(x)
}
fn half_setp_register(x: Word) -> Instruction {
    let r = x.r;
    HalfCompare {
        ftz: bit(r, 6),
        dst: set_predicates(r, 49),
        rhs: register_b(r),
        swizzle: two(r, 28),
        abs: bit(r, 30),
        negate: bit(r, 31),
        selector: b(r, 35, 4),
    }
    .arrive(x)
}
fn half_set_short(x: Word) -> Instruction {
    let r = x.r;
    HalfCompare {
        ftz: bit(r, 54),
        dst: set_register(r, 53),
        rhs: short_b(r),
        swizzle: 0,
        abs: false,
        negate: false,
        selector: b(r, 49, 4),
    }
    .arrive(x)
}
fn half_set_constant(x: Word) -> Instruction {
    let r = x.r;
    HalfCompare {
        ftz: bit(r, 54),
        dst: set_register(r, 53),
        rhs: constant_b(r),
        swizzle: 1,
        abs: false,
        negate: bit(r, 56),
        selector: b(r, 49, 4),
    }
    .arrive(x)
}
fn half_setp_short(x: Word) -> Instruction {
    let r = x.r;
    HalfCompare {
        ftz: bit(r, 6),
        dst: set_predicates(r, 53),
        rhs: short_b(r),
        swizzle: 0,
        abs: false,
        negate: false,
        selector: b(r, 49, 4),
    }
    .arrive(x)
}
fn half_setp_constant(x: Word) -> Instruction {
    let r = x.r;
    HalfCompare {
        ftz: bit(r, 6),
        dst: set_predicates(r, 53),
        rhs: constant_b(r),
        swizzle: 1,
        abs: bit(r, 54),
        negate: bit(r, 56),
        selector: b(r, 49, 4),
    }
    .arrive(x)
}
/// HSET2 (0x5d18, bit 53 clear) and HSETP2 (0x5d20, bit 53 set) register
/// forms; boolean op 3 stays Unknown.
const HALF_SET_REGISTER_FIELDS: Fields<2> = Fields([Field::bits(45, 2), Field::bits(53, 1)]);
const fn half_set_register_rule(r: u64) -> u8 {
    (b(r, 45, 2) < 3) as u8 * (1 + b(r, 53, 1) as u8)
}
const HALF_SET_REGISTER_FORMS: [u8; HALF_SET_REGISTER_FIELDS.len()] =
    forms!(HALF_SET_REGISTER_FIELDS, half_set_register_rule);
pub(super) fn read_half_set_register(x: Word) -> Instruction {
    let paths: [Decode; 3] = [unknown, half_set_register, half_setp_register];
    paths[(HALF_SET_REGISTER_FORMS[HALF_SET_REGISTER_FIELDS.index(x.r)]) as usize](x)
}
/// HSET2 (0x7c00, bit 57 clear) and HSETP2 (0x7e00, bit 57 set) short and
/// cbuf (bit 55) forms; boolean op 3 stays Unknown.
const HALF_SET_SHORT_FIELDS: Fields<3> =
    Fields([Field::bits(45, 2), Field::bits(55, 1), Field::bits(57, 1)]);
const fn half_set_short_rule(r: u64) -> u8 {
    (b(r, 45, 2) < 3) as u8 * (1 + b(r, 55, 1) as u8 + 2 * b(r, 57, 1) as u8)
}
const HALF_SET_SHORT_FORMS: [u8; HALF_SET_SHORT_FIELDS.len()] =
    forms!(HALF_SET_SHORT_FIELDS, half_set_short_rule);
pub(super) fn read_half_set_short(x: Word) -> Instruction {
    let paths: [Decode; 5] = [
        unknown,
        half_set_short,
        half_set_constant,
        half_setp_short,
        half_setp_constant,
    ];
    paths[(HALF_SET_SHORT_FORMS[HALF_SET_SHORT_FIELDS.index(x.r)]) as usize](x)
}
