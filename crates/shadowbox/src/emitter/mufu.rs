//! MUFU lowering: Ryujinx Instructions/InstEmitMultifunction.cs:32-83
//! applies source abs then neg, host math, then saturation. The MIT notice
//! at the emitter's helpers covers this adapted mapping. Each op is its
//! own path, indexed by its encoded value.

use super::{alu, fresh, inst, saturate_f32, sign_modifiers};
use crate::fast_hash::HashMap;
use crate::ir::MufuOp;
use crate::spirv::Op;

/// The module declares no binary64 types, so RCP64H and RSQ64H have no
/// lowering.
#[derive(Clone, Copy)]
pub(super) struct Undeclared;

pub(super) struct MufuEmitter<'a> {
    pub w: &'a mut Vec<u32>,
    pub id: &'a mut u32,
    pub uint: u32,
    pub float: u32,
    pub boolt: u32,
    pub glsl_std_450: u32,
    pub one_float: u32,
    pub zero_float: u32,
    /// The uint zero constant.
    pub zero: u32,
    pub constants: &'a HashMap<u32, u32>,
    pub late_globals: &'a mut Vec<u32>,
    /// The binary64 scalar, its uvec2 word view and 1.0, when declared.
    pub double: Result<(u32, u32, u32), Undeclared>,
}

/// The source after its sign modifiers: bits, float view, and whether its
/// magnitude is zero.
#[derive(Clone, Copy)]
struct Operand {
    input: u32,
    value: u32,
    is_zero: u32,
}

impl MufuEmitter<'_> {
    /// The MUFU result bits of `input` register bits under `op`.
    pub fn emit(
        &mut self,
        op: MufuOp,
        input: u32,
        abs: bool,
        negate: bool,
        saturate: bool,
    ) -> Result<u32, Undeclared> {
        // Sign-bit operations avoid float abs/neg bitcasts and keep signed
        // zero and NaN payloads intact (InstEmitMultifunction.cs:36).
        let input = sign_modifiers(
            self.w,
            self.id,
            input,
            abs,
            negate,
            self.uint,
            self.constants,
        );
        let value = self.alu(Op::Bitcast, self.float, &[input]);
        // Preserve batch 4 defined-domain evaluation before selecting IEEE
        // limits. One safe operand avoids branches and repeated math.
        let magnitude = self.alu(
            Op::BitwiseAnd,
            self.uint,
            &[input, self.constants[&0x7fff_ffff]],
        );
        let is_zero = self.alu(Op::IEqual, self.boolt, &[magnitude, self.constants[&0]]);
        MUFU_PATHS[op as usize](
            self,
            Operand {
                input,
                value,
                is_zero,
            },
            usize::from(saturate),
        )
    }

    fn alu(&mut self, op: Op, ty: u32, operands: &[u32]) -> u32 {
        alu(self.w, self.id, op, ty, operands)
    }

    fn ext(&mut self, ty: u32, instruction: u32, operand: u32) -> u32 {
        self.alu(Op::ExtInst, ty, &[self.glsl_std_450, instruction, operand])
    }

    fn saturated(&mut self, bits: u32) -> u32 {
        saturate_f32(
            self.w,
            self.id,
            bits,
            (self.uint, self.boolt),
            self.constants,
            self.late_globals,
        )
    }

    /// COS, SIN and EX2: the GLSL.std.450 instruction on the source.
    /// SIN/COS use radians without a pi scaling or a modulo pass; RRO
    /// already defers reduction (InstEmitMultifunction.cs:86-94).
    fn plain(&mut self, operand: Operand, instruction: u32, saturate: usize) -> u32 {
        let result = self.ext(self.float, instruction, operand.value);
        let bits = self.alu(Op::Bitcast, self.uint, &[result]);
        SATURATE[saturate](self, bits)
    }

    /// LG2, RCP, RSQ and SQRT: host math on a safe operand, then the IEEE
    /// limits selected explicitly. Ryujinx InstEmitMultifunction.cs:52-75
    /// defines them; GLSL.std.450 section 2 limits their domains, and
    /// Vulkan 1.2 does not require extinst preservation of signed zero,
    /// infinity or NaN (docs/reference-sources.md).
    fn limited<D: Domain>(
        &mut self,
        operand: Operand,
        saturate: usize,
        math: fn(&mut Self, u32) -> u32,
        infinity: fn(&mut Self, Operand) -> u32,
    ) -> u32 {
        let domain = D::open(self, operand);
        let exceptional = domain.exceptional(self, operand.is_zero);
        let safe = self.alu(
            Op::Select,
            self.float,
            &[exceptional, self.one_float, operand.value],
        );
        let result = math(self, safe);
        // Select register bits, not floats: infinity and NaN constants
        // need no bitcast.
        let bits = self.alu(Op::Bitcast, self.uint, &[result]);
        let infinity = infinity(self, operand);
        let bits = self.alu(Op::Select, self.uint, &[operand.is_zero, infinity, bits]);
        let bits = domain.nan(self, bits);
        LIMITED_SATURATE[saturate](self, bits)
    }

    /// RCP64H and RSQ64H: Ryujinx InstEmitMultifunction.cs:58-71 packs zero
    /// as the low word and the modified source as the high word, performs
    /// FP64 math, then returns the result high word.
    fn wide(
        &mut self,
        operand: Operand,
        saturate: usize,
        math: fn(&mut Self, u32, u32, u32) -> u32,
    ) -> Result<u32, Undeclared> {
        let declared = self.double;
        declared.map(|(double, words_type, one)| {
            let words = fresh(self.id);
            inst(
                self.w,
                Op::CompositeConstruct,
                &[words_type, words, self.zero, operand.input],
            );
            let value64 = self.alu(Op::Bitcast, double, &[words]);
            let result64 = math(self, double, one, value64);
            let result_words = self.alu(Op::Bitcast, words_type, &[result64]);
            let bits = self.alu(Op::CompositeExtract, self.uint, &[result_words, 1]);
            SATURATE[saturate](self, bits)
        })
    }

    /// RCP and RSQ: zero gives infinity of the source's sign.
    fn signed_infinity(&mut self, operand: Operand) -> u32 {
        let sign = self.alu(
            Op::BitwiseAnd,
            self.uint,
            &[operand.input, self.constants[&0x8000_0000]],
        );
        self.alu(
            Op::BitwiseOr,
            self.uint,
            &[sign, self.constants[&0x7f80_0000]],
        )
    }
}

type Saturation = fn(&mut MufuEmitter, u32) -> u32;
/// By saturate: the bits as they are, or clamped to [0, 1].
const SATURATE: [Saturation; 2] = [|_, bits| bits, |e, bits| e.saturated(bits)];
/// By saturate, after the IEEE limits: a saturated result passes back
/// through its float view before the clamp.
const LIMITED_SATURATE: [Saturation; 2] = [
    |_, bits| bits,
    |e, bits| {
        let result = e.alu(Op::Bitcast, e.float, &[bits]);
        let bits = e.alu(Op::Bitcast, e.uint, &[result]);
        e.saturated(bits)
    },
];

/// Where a limited op's operand leaves its defined domain.
trait Domain: Copy {
    fn open(e: &mut MufuEmitter, operand: Operand) -> Self;
    /// Whether the operand takes the safe 1.0 instead.
    fn exceptional(self, e: &mut MufuEmitter, is_zero: u32) -> u32;
    /// The result bits once a negative operand gives NaN.
    fn nan(self, e: &mut MufuEmitter, bits: u32) -> u32;
}

/// RCP: only zero is exceptional.
#[derive(Clone, Copy)]
struct Nonzero;
impl Domain for Nonzero {
    fn open(_: &mut MufuEmitter, _: Operand) -> Self {
        Nonzero
    }
    fn exceptional(self, _: &mut MufuEmitter, is_zero: u32) -> u32 {
        is_zero
    }
    fn nan(self, _: &mut MufuEmitter, bits: u32) -> u32 {
        bits
    }
}

/// LG2, RSQ and SQRT: zero and negative operands are exceptional, and a
/// negative one gives the default NaN.
#[derive(Clone, Copy)]
struct NonNegative {
    negative: u32,
}
impl Domain for NonNegative {
    fn open(e: &mut MufuEmitter, operand: Operand) -> Self {
        let negative = e.alu(Op::FOrdLessThan, e.boolt, &[operand.value, e.zero_float]);
        NonNegative { negative }
    }
    fn exceptional(self, e: &mut MufuEmitter, is_zero: u32) -> u32 {
        e.alu(Op::LogicalOr, e.boolt, &[is_zero, self.negative])
    }
    fn nan(self, e: &mut MufuEmitter, bits: u32) -> u32 {
        e.alu(
            Op::Select,
            e.uint,
            &[self.negative, e.constants[&0x7fc0_0000], bits],
        )
    }
}

type MufuPath = fn(&mut MufuEmitter, Operand, usize) -> Result<u32, Undeclared>;
/// By MufuOp. GLSL.std.450 Cos 14, Sin 13, Exp2 29, Log2 30, Sqrt 31 and
/// InverseSqrt 32 (Ryujinx Spirv/Instructions.cs:491-494,901-904,
/// 1039-1042,1120-1123; Translation/EmitterContextInsts.cs:394-401).
const MUFU_PATHS: [MufuPath; 9] = [
    |e, x, s| Ok(e.plain(x, 14, s)),
    |e, x, s| Ok(e.plain(x, 13, s)),
    |e, x, s| Ok(e.plain(x, 29, s)),
    |e, x, s| {
        Ok(e.limited::<NonNegative>(
            x,
            s,
            |e, safe| e.ext(e.float, 30, safe),
            |e, _| e.constants[&0xff80_0000],
        ))
    },
    |e, x, s| {
        Ok(e.limited::<Nonzero>(
            x,
            s,
            |e, safe| e.alu(Op::FDiv, e.float, &[e.one_float, safe]),
            MufuEmitter::signed_infinity,
        ))
    },
    |e, x, s| {
        Ok(e.limited::<NonNegative>(
            x,
            s,
            |e, safe| e.ext(e.float, 32, safe),
            MufuEmitter::signed_infinity,
        ))
    },
    |e, x, s| {
        e.wide(x, s, |e, double, one, value| {
            e.alu(Op::FDiv, double, &[one, value])
        })
    },
    |e, x, s| e.wide(x, s, |e, double, _, value| e.ext(double, 32, value)),
    |e, x, s| {
        Ok(e.limited::<NonNegative>(x, s, |e, safe| e.ext(e.float, 31, safe), |_, x| x.input))
    },
];
