pub(crate) mod atomic;
mod control;
mod conversion;
mod flags;
mod integer;
pub(crate) mod memory;
mod mufu;
mod shader_stage;
mod stages;
mod surface;
mod texture;
use crate::fast_hash::{HashMap, HashSet};
use crate::headers::{Interface, Interpolation, ShaderHeader, Stage};
use crate::interface::{NoLayout, Stopped};
use crate::ir::ops;
use crate::ir::FlagBank;
use crate::ir::{
    AttributeBank, CSetPCondition, Conversion, Denormals, DirectedRounding, DoubleMinMaxOperation,
    DoubleSetResult, DoubleSource, DoubleSourceModifier, F2fDestination, F2fSource, FAddModifiers,
    FMulModifiers, FMulRounding, FMulScale, FloatCompare, FloatFlags, Flush, GatherOffsets,
    GradOffset, HalfFlush, HalfOp, Instruction, InstructionKind, IntCompare, IntegerCompareMode,
    IntegerWidth, InterpFreq, InterpLoc, LdcWidth, LegacyInterpolation, LogicOp, LsSize2,
    PredicateOp, PredicateOutput, Register, RegisterPair, Relation, RoundingMode, Saturation,
    SetOp, SetResult, ShuffleMode, Signedness, Source, SourceAbs, SourceNegation, TexOp,
    VideoElement, VideoSelection, XmadCop,
};
use crate::ir::{Requirements, Translation};
use crate::reaching::ReachingDefinitions;
use crate::spirv::{self, Op};
use conversion::ConversionEmitter;
use flags::{CcFlags, CcWriter, Words};
use integer::IntegerEmitter;
use shader_stage::{OutputStores, OutputTypes, Outputs as _, ShaderStage, StagePath};
use surface::{
    atomic_opcode, coordinate_count, surface_format, surface_image_shape, texel_bytes_log2,
    CachePolicy,
};
use texture::{
    implicit_mip, offset_coordinates, texture_sources, CoordinateTypes, Coordinates, OffsetTypes,
    OffsetWord, RuntimeOffset, TextureResults, EVERY_RESULT,
};

pub struct SpirvModule {
    pub words: Vec<u32>,
    /// Storage image views the host must bind, including aliases of one handle.
    pub surface_descriptors: Vec<SurfaceDescriptor>,
    /// Enable VK_KHR_shader_maximal_reconvergence before installing this module.
    pub requires_maximal_reconvergence: bool,
    /// Enable shaderDemoteToHelperInvocation before installing a KIL module.
    pub requires_demote_to_helper_invocation: bool,
}

/// A storage image view of a Maxwell handle. Separate formats need separate
/// Vulkan views (Ryujinx Translation/ResourceManager.cs:263-308).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceDescriptor {
    pub source: HandleSource,
    /// Maxwell SuDim, envytools envydis/gm107.c:759-766 (field33..35).
    pub target: u8,
    /// SPIR-V ImageFormat; Mesa src/compiler/spirv/spirv.h:291-335.
    pub image_format: u32,
    pub set: u32,
    pub binding: u32,
}

impl SpirvModule {
    /// Run the same conservative optimisation used by production emission.
    /// GPU tests add observable stores before this pass so it cannot remove
    /// the instruction under test as an unused compute result.
    pub fn optimize(&mut self) {
        crate::optimizer::optimize(&mut self.words);
    }
}

#[derive(Clone, Copy)]
enum Attribute {
    Generic {
        location: u16,
        component: u8,
    },
    FragCoord {
        component: u8,
    },
    /// a[0x3fc]: Ryujinx AttributeMap.cs:81.
    FrontFacing,
}

// Mesa's attribute map is nak_private.h:45-57: POSITION_X..W are 0x70..0x7c,
// and generic attributes use 0x80 + location * 16 + component * 4.
fn attribute(addr: u16) -> Option<Attribute> {
    match addr {
        0x70..=0x7c if addr.is_multiple_of(4) => Some(Attribute::FragCoord {
            component: ((addr - 0x70) / 4) as u8,
        }),
        0x3fc => Some(Attribute::FrontFacing),
        0x80.. if addr.is_multiple_of(4) => {
            let relative = addr - 0x80;
            Some(Attribute::Generic {
                location: relative / 16,
                component: ((relative % 16) / 4) as u8,
            })
        }
        _ => None,
    }
}

pub(crate) fn inst(w: &mut Vec<u32>, op: Op, a: &[u32]) {
    w.push(((a.len() as u32 + 1) << 16) | u32::from(op));
    w.extend_from_slice(a);
}

// Opcode names and numbers: crate::spirv (SPIR-V 1.6 section 3.52).
// Each helper emits a typed result and allocates its ID in one place.
fn alu(w: &mut Vec<u32>, id: &mut u32, op: Op, ty: u32, operands: &[u32]) -> u32 {
    let result = fresh(id);
    // Write the same words directly, avoiding an allocation per ALU result.
    w.push(((operands.len() as u32 + 3) << 16) | u32::from(op));
    w.extend_from_slice(&[ty, result]);
    w.extend_from_slice(operands);
    result
}

fn double_source_operation(
    w: &mut Vec<u32>,
    id: &mut u32,
    double: u32,
    glsl_std_450: u32,
    value: u32,
    modifier: DoubleSourceModifier,
) -> u32 {
    DOUBLE_SOURCE[modifier as usize](w, id, double, glsl_std_450, value)
}
/// Applies one source modifier to `value`, typed `double`, using GLSL.std.450.
type DoubleModifier = fn(&mut Vec<u32>, &mut u32, u32, u32, u32) -> u32;
/// By DoubleSourceModifier (Plain, Absolute, Negated, NegatedAbsolute).
const DOUBLE_SOURCE: [DoubleModifier; 4] = [
    |_, _, _, _, value| value,
    |w, id, double, glsl, value| alu(w, id, Op::ExtInst, double, &[glsl, 4, value]),
    |w, id, double, _, value| alu(w, id, Op::FNegate, double, &[value]),
    |w, id, double, glsl, value| {
        let absolute = alu(w, id, Op::ExtInst, double, &[glsl, 4, value]);
        alu(w, id, Op::FNegate, double, &[absolute])
    },
];
fn precise_f64(
    w: &mut Vec<u32>,
    id: &mut u32,
    op: Op,
    double: u32,
    operands: &[u32],
    annotations: &mut Vec<u32>,
) -> u32 {
    let value = alu(w, id, op, double, operands);
    inst(annotations, Op::Decorate, &[value, 42]); // NoContraction
    value
}

/// Binary64 addition rounded to nearest, ties to even. SPIR-V does not require
/// an implementation to preserve the sign of an exact zero result, so restore
/// the IEEE sign rule explicitly: only -0 + -0 produces -0; every other exact
/// zero sum produces +0.
#[allow(clippy::too_many_arguments)]
fn dadd_rn(
    w: &mut Vec<u32>,
    id: &mut u32,
    operands: [u32; 2],
    (double, words, uint, boolt): (u32, u32, u32, u32),
    late_globals: &mut Vec<u32>,
    annotations: &mut Vec<u32>,
) -> u32 {
    let sum = precise_f64(w, id, Op::FAdd, double, &operands, annotations);
    let bits = alu(w, id, Op::Bitcast, words, &[sum]);
    let low = alu(w, id, Op::CompositeExtract, uint, &[bits, 0]);
    let high = alu(w, id, Op::CompositeExtract, uint, &[bits, 1]);
    let zero = late_constant(late_globals, id, uint, 0);
    let sign_mask = late_constant(late_globals, id, uint, 0x8000_0000);
    let magnitude_mask = late_constant(late_globals, id, uint, 0x7fff_ffff);

    let low_zero = alu(w, id, Op::IEqual, boolt, &[low, zero]);
    let high_magnitude = alu(w, id, Op::BitwiseAnd, uint, &[high, magnitude_mask]);
    let high_zero = alu(w, id, Op::IEqual, boolt, &[high_magnitude, zero]);
    let sum_zero = alu(w, id, Op::LogicalAnd, boolt, &[low_zero, high_zero]);

    let operand_is_negative_zero = |w: &mut Vec<u32>, id: &mut u32, value| {
        let value_bits = alu(w, id, Op::Bitcast, words, &[value]);
        let value_low = alu(w, id, Op::CompositeExtract, uint, &[value_bits, 0]);
        let value_high = alu(w, id, Op::CompositeExtract, uint, &[value_bits, 1]);
        let low_zero = alu(w, id, Op::IEqual, boolt, &[value_low, zero]);
        let high_negative_zero = alu(w, id, Op::IEqual, boolt, &[value_high, sign_mask]);
        alu(
            w,
            id,
            Op::LogicalAnd,
            boolt,
            &[low_zero, high_negative_zero],
        )
    };
    let a_negative_zero = operand_is_negative_zero(w, id, operands[0]);
    let b_negative_zero = operand_is_negative_zero(w, id, operands[1]);
    let both_negative_zero = alu(
        w,
        id,
        Op::LogicalAnd,
        boolt,
        &[a_negative_zero, b_negative_zero],
    );
    let zero_high = alu(
        w,
        id,
        Op::Select,
        uint,
        &[both_negative_zero, sign_mask, zero],
    );
    let rounded_low = alu(w, id, Op::Select, uint, &[sum_zero, zero, low]);
    let rounded_high = alu(w, id, Op::Select, uint, &[sum_zero, zero_high, high]);
    let rounded_bits = alu(
        w,
        id,
        Op::CompositeConstruct,
        words,
        &[rounded_low, rounded_high],
    );
    alu(w, id, Op::Bitcast, double, &[rounded_bits])
}

/// Binary64 addition rounded toward negative infinity. The default RTE sum
/// and Knuth TwoSum residual identify whether RTE lies above the exact sum;
/// that case steps one binary64 encoding downward. Finite positive overflow
/// steps from +Inf to MAX, and exact zero follows IEEE's RM sign rule.
#[allow(clippy::too_many_arguments)]
fn dadd_rm(
    w: &mut Vec<u32>,
    id: &mut u32,
    operands: [u32; 2],
    (double, words, uint, boolt): (u32, u32, u32, u32),
    late_globals: &mut Vec<u32>,
    annotations: &mut Vec<u32>,
) -> u32 {
    let [a, b] = operands;
    let sum = precise_f64(w, id, Op::FAdd, double, &[a, b], annotations);
    let virtual_b = precise_f64(w, id, Op::FSub, double, &[sum, a], annotations);
    let virtual_a = precise_f64(w, id, Op::FSub, double, &[sum, virtual_b], annotations);
    let a_error = precise_f64(w, id, Op::FSub, double, &[a, virtual_a], annotations);
    let b_error = precise_f64(w, id, Op::FSub, double, &[b, virtual_b], annotations);
    let error = precise_f64(w, id, Op::FAdd, double, &[a_error, b_error], annotations);

    let zero_f64 = late_double(late_globals, id, double, 0.0);
    let error_negative = alu(w, id, Op::FOrdLessThan, boolt, &[error, zero_f64]);
    inst(annotations, Op::Decorate, &[error_negative, 42]);

    let bits = alu(w, id, Op::Bitcast, words, &[sum]);
    let low = alu(w, id, Op::CompositeExtract, uint, &[bits, 0]);
    let high = alu(w, id, Op::CompositeExtract, uint, &[bits, 1]);
    let zero = late_constant(late_globals, id, uint, 0);
    let one = late_constant(late_globals, id, uint, 1);
    let sign_mask = late_constant(late_globals, id, uint, 0x8000_0000);
    let magnitude_mask = late_constant(late_globals, id, uint, 0x7fff_ffff);
    let infinity_high = late_constant(late_globals, id, uint, 0x7ff0_0000);

    let negative = alu(w, id, Op::UGreaterThan, boolt, &[high, magnitude_mask]);
    let low_zero = alu(w, id, Op::IEqual, boolt, &[low, zero]);
    let high_magnitude = alu(w, id, Op::BitwiseAnd, uint, &[high, magnitude_mask]);
    let high_zero = alu(w, id, Op::IEqual, boolt, &[high_magnitude, zero]);
    let sum_zero = alu(w, id, Op::LogicalAnd, boolt, &[low_zero, high_zero]);

    let incremented_low = alu(w, id, Op::IAdd, uint, &[low, one]);
    let carry = alu(w, id, Op::IEqual, boolt, &[incremented_low, zero]);
    let carry_word = alu(w, id, Op::Select, uint, &[carry, one, zero]);
    let incremented_high = alu(w, id, Op::IAdd, uint, &[high, carry_word]);
    let decremented_low = alu(w, id, Op::ISub, uint, &[low, one]);
    let borrow_word = alu(w, id, Op::Select, uint, &[low_zero, one, zero]);
    let decremented_high = alu(w, id, Op::ISub, uint, &[high, borrow_word]);
    let down_low = alu(
        w,
        id,
        Op::Select,
        uint,
        &[negative, incremented_low, decremented_low],
    );
    let down_high = alu(
        w,
        id,
        Op::Select,
        uint,
        &[negative, incremented_high, decremented_high],
    );

    let operand_is_finite = |w: &mut Vec<u32>, id: &mut u32, value| {
        let value_bits = alu(w, id, Op::Bitcast, words, &[value]);
        let value_high = alu(w, id, Op::CompositeExtract, uint, &[value_bits, 1]);
        let magnitude = alu(w, id, Op::BitwiseAnd, uint, &[value_high, magnitude_mask]);
        alu(w, id, Op::ULessThan, boolt, &[magnitude, infinity_high])
    };
    let a_finite = operand_is_finite(w, id, a);
    let b_finite = operand_is_finite(w, id, b);
    let finite_operands = alu(w, id, Op::LogicalAnd, boolt, &[a_finite, b_finite]);
    let high_is_inf = alu(w, id, Op::IEqual, boolt, &[high, infinity_high]);
    let positive_infinity = alu(w, id, Op::LogicalAnd, boolt, &[low_zero, high_is_inf]);
    let positive_overflow = alu(
        w,
        id,
        Op::LogicalAnd,
        boolt,
        &[finite_operands, positive_infinity],
    );
    let adjust = alu(
        w,
        id,
        Op::LogicalOr,
        boolt,
        &[error_negative, positive_overflow],
    );
    let rounded_low = alu(w, id, Op::Select, uint, &[adjust, down_low, low]);
    let rounded_high = alu(w, id, Op::Select, uint, &[adjust, down_high, high]);

    let operand_is_positive_zero = |w: &mut Vec<u32>, id: &mut u32, value| {
        let value_bits = alu(w, id, Op::Bitcast, words, &[value]);
        let value_low = alu(w, id, Op::CompositeExtract, uint, &[value_bits, 0]);
        let value_high = alu(w, id, Op::CompositeExtract, uint, &[value_bits, 1]);
        let low_zero = alu(w, id, Op::IEqual, boolt, &[value_low, zero]);
        let high_zero = alu(w, id, Op::IEqual, boolt, &[value_high, zero]);
        alu(w, id, Op::LogicalAnd, boolt, &[low_zero, high_zero])
    };
    let a_positive_zero = operand_is_positive_zero(w, id, a);
    let b_positive_zero = operand_is_positive_zero(w, id, b);
    let both_positive_zero = alu(
        w,
        id,
        Op::LogicalAnd,
        boolt,
        &[a_positive_zero, b_positive_zero],
    );
    let not_both_positive = alu(w, id, Op::LogicalNot, boolt, &[both_positive_zero]);
    let negative_zero = alu(w, id, Op::LogicalAnd, boolt, &[sum_zero, not_both_positive]);
    let rounded_low = alu(w, id, Op::Select, uint, &[negative_zero, zero, rounded_low]);
    let rounded_high = alu(
        w,
        id,
        Op::Select,
        uint,
        &[negative_zero, sign_mask, rounded_high],
    );
    let rounded_bits = alu(
        w,
        id,
        Op::CompositeConstruct,
        words,
        &[rounded_low, rounded_high],
    );
    alu(w, id, Op::Bitcast, double, &[rounded_bits])
}

/// Binary64 addition rounded toward positive infinity. The default RTE sum
/// and Knuth TwoSum residual identify whether RTE lies below the exact sum;
/// that case steps one binary64 encoding upward. Finite negative overflow
/// steps from -Inf to -MAX, and exact zero follows IEEE's RP sign rule.
#[allow(clippy::too_many_arguments)]
fn dadd_rp(
    w: &mut Vec<u32>,
    id: &mut u32,
    operands: [u32; 2],
    (double, words, uint, boolt): (u32, u32, u32, u32),
    late_globals: &mut Vec<u32>,
    annotations: &mut Vec<u32>,
) -> u32 {
    let [a, b] = operands;
    let sum = precise_f64(w, id, Op::FAdd, double, &[a, b], annotations);
    let virtual_b = precise_f64(w, id, Op::FSub, double, &[sum, a], annotations);
    let virtual_a = precise_f64(w, id, Op::FSub, double, &[sum, virtual_b], annotations);
    let a_error = precise_f64(w, id, Op::FSub, double, &[a, virtual_a], annotations);
    let b_error = precise_f64(w, id, Op::FSub, double, &[b, virtual_b], annotations);
    let error = precise_f64(w, id, Op::FAdd, double, &[a_error, b_error], annotations);

    let zero_f64 = late_double(late_globals, id, double, 0.0);
    let error_positive = alu(w, id, Op::FOrdGreaterThan, boolt, &[error, zero_f64]);
    inst(annotations, Op::Decorate, &[error_positive, 42]);

    let bits = alu(w, id, Op::Bitcast, words, &[sum]);
    let low = alu(w, id, Op::CompositeExtract, uint, &[bits, 0]);
    let high = alu(w, id, Op::CompositeExtract, uint, &[bits, 1]);
    let zero = late_constant(late_globals, id, uint, 0);
    let one = late_constant(late_globals, id, uint, 1);
    let sign_mask = late_constant(late_globals, id, uint, 0x8000_0000);
    let magnitude_mask = late_constant(late_globals, id, uint, 0x7fff_ffff);
    let infinity_high = late_constant(late_globals, id, uint, 0x7ff0_0000);

    let negative = alu(w, id, Op::UGreaterThan, boolt, &[high, magnitude_mask]);
    let low_zero = alu(w, id, Op::IEqual, boolt, &[low, zero]);
    let high_magnitude = alu(w, id, Op::BitwiseAnd, uint, &[high, magnitude_mask]);
    let high_zero = alu(w, id, Op::IEqual, boolt, &[high_magnitude, zero]);
    let sum_zero = alu(w, id, Op::LogicalAnd, boolt, &[low_zero, high_zero]);

    let incremented_low = alu(w, id, Op::IAdd, uint, &[low, one]);
    let carry = alu(w, id, Op::IEqual, boolt, &[incremented_low, zero]);
    let carry_word = alu(w, id, Op::Select, uint, &[carry, one, zero]);
    let incremented_high = alu(w, id, Op::IAdd, uint, &[high, carry_word]);
    let decremented_low = alu(w, id, Op::ISub, uint, &[low, one]);
    let borrow_word = alu(w, id, Op::Select, uint, &[low_zero, one, zero]);
    let decremented_high = alu(w, id, Op::ISub, uint, &[high, borrow_word]);
    let operand_is_finite = |w: &mut Vec<u32>, id: &mut u32, value| {
        let value_bits = alu(w, id, Op::Bitcast, words, &[value]);
        let value_high = alu(w, id, Op::CompositeExtract, uint, &[value_bits, 1]);
        let magnitude = alu(w, id, Op::BitwiseAnd, uint, &[value_high, magnitude_mask]);
        alu(w, id, Op::ULessThan, boolt, &[magnitude, infinity_high])
    };
    let a_finite = operand_is_finite(w, id, a);
    let b_finite = operand_is_finite(w, id, b);
    let finite_operands = alu(w, id, Op::LogicalAnd, boolt, &[a_finite, b_finite]);
    let negative_infinity_high = alu(w, id, Op::BitwiseOr, uint, &[infinity_high, sign_mask]);
    let high_is_negative_inf = alu(w, id, Op::IEqual, boolt, &[high, negative_infinity_high]);
    let negative_infinity = alu(
        w,
        id,
        Op::LogicalAnd,
        boolt,
        &[low_zero, high_is_negative_inf],
    );
    let negative_overflow = alu(
        w,
        id,
        Op::LogicalAnd,
        boolt,
        &[finite_operands, negative_infinity],
    );
    let adjust = alu(
        w,
        id,
        Op::LogicalOr,
        boolt,
        &[error_positive, negative_overflow],
    );
    let up_low = alu(
        w,
        id,
        Op::Select,
        uint,
        &[negative, decremented_low, incremented_low],
    );
    let up_high = alu(
        w,
        id,
        Op::Select,
        uint,
        &[negative, decremented_high, incremented_high],
    );
    let rounded_low = alu(w, id, Op::Select, uint, &[adjust, up_low, low]);
    let rounded_high = alu(w, id, Op::Select, uint, &[adjust, up_high, high]);

    let operand_is_negative_zero = |w: &mut Vec<u32>, id: &mut u32, value| {
        let value_bits = alu(w, id, Op::Bitcast, words, &[value]);
        let value_low = alu(w, id, Op::CompositeExtract, uint, &[value_bits, 0]);
        let value_high = alu(w, id, Op::CompositeExtract, uint, &[value_bits, 1]);
        let low_zero = alu(w, id, Op::IEqual, boolt, &[value_low, zero]);
        let high_negative_zero = alu(w, id, Op::IEqual, boolt, &[value_high, sign_mask]);
        alu(
            w,
            id,
            Op::LogicalAnd,
            boolt,
            &[low_zero, high_negative_zero],
        )
    };
    let a_negative_zero = operand_is_negative_zero(w, id, a);
    let b_negative_zero = operand_is_negative_zero(w, id, b);
    let both_negative_zero = alu(
        w,
        id,
        Op::LogicalAnd,
        boolt,
        &[a_negative_zero, b_negative_zero],
    );
    let zero_high = alu(
        w,
        id,
        Op::Select,
        uint,
        &[both_negative_zero, sign_mask, zero],
    );
    let rounded_low = alu(w, id, Op::Select, uint, &[sum_zero, zero, rounded_low]);
    let rounded_high = alu(
        w,
        id,
        Op::Select,
        uint,
        &[sum_zero, zero_high, rounded_high],
    );
    let rounded_bits = alu(
        w,
        id,
        Op::CompositeConstruct,
        words,
        &[rounded_low, rounded_high],
    );
    alu(w, id, Op::Bitcast, double, &[rounded_bits])
}

/// Binary64 addition rounded toward zero. The default RTE sum and Knuth
/// TwoSum residual identify whether RTE lies away from zero, then the result
/// steps one binary64 encoding toward zero. Finite overflow steps from either
/// infinity to the corresponding maximum finite value.
#[allow(clippy::too_many_arguments)]
fn dadd_rz(
    w: &mut Vec<u32>,
    id: &mut u32,
    operands: [u32; 2],
    (double, words, uint, boolt): (u32, u32, u32, u32),
    late_globals: &mut Vec<u32>,
    annotations: &mut Vec<u32>,
) -> u32 {
    let [a, b] = operands;
    let sum = precise_f64(w, id, Op::FAdd, double, &[a, b], annotations);
    let virtual_b = precise_f64(w, id, Op::FSub, double, &[sum, a], annotations);
    let virtual_a = precise_f64(w, id, Op::FSub, double, &[sum, virtual_b], annotations);
    let a_error = precise_f64(w, id, Op::FSub, double, &[a, virtual_a], annotations);
    let b_error = precise_f64(w, id, Op::FSub, double, &[b, virtual_b], annotations);
    let error = precise_f64(w, id, Op::FAdd, double, &[a_error, b_error], annotations);

    let zero_f64 = late_double(late_globals, id, double, 0.0);
    let error_negative = alu(w, id, Op::FOrdLessThan, boolt, &[error, zero_f64]);
    let error_positive = alu(w, id, Op::FOrdGreaterThan, boolt, &[error, zero_f64]);
    inst(annotations, Op::Decorate, &[error_negative, 42]);
    inst(annotations, Op::Decorate, &[error_positive, 42]);

    let bits = alu(w, id, Op::Bitcast, words, &[sum]);
    let low = alu(w, id, Op::CompositeExtract, uint, &[bits, 0]);
    let high = alu(w, id, Op::CompositeExtract, uint, &[bits, 1]);
    let zero = late_constant(late_globals, id, uint, 0);
    let one = late_constant(late_globals, id, uint, 1);
    let sign_mask = late_constant(late_globals, id, uint, 0x8000_0000);
    let magnitude_mask = late_constant(late_globals, id, uint, 0x7fff_ffff);
    let infinity_high = late_constant(late_globals, id, uint, 0x7ff0_0000);

    let negative = alu(w, id, Op::UGreaterThan, boolt, &[high, magnitude_mask]);
    let nonnegative = alu(w, id, Op::LogicalNot, boolt, &[negative]);
    let low_zero = alu(w, id, Op::IEqual, boolt, &[low, zero]);
    let high_magnitude = alu(w, id, Op::BitwiseAnd, uint, &[high, magnitude_mask]);
    let high_zero = alu(w, id, Op::IEqual, boolt, &[high_magnitude, zero]);
    let sum_zero = alu(w, id, Op::LogicalAnd, boolt, &[low_zero, high_zero]);

    let incremented_low = alu(w, id, Op::IAdd, uint, &[low, one]);
    let carry = alu(w, id, Op::IEqual, boolt, &[incremented_low, zero]);
    let carry_word = alu(w, id, Op::Select, uint, &[carry, one, zero]);
    let incremented_high = alu(w, id, Op::IAdd, uint, &[high, carry_word]);
    let decremented_low = alu(w, id, Op::ISub, uint, &[low, one]);
    let borrow_word = alu(w, id, Op::Select, uint, &[low_zero, one, zero]);
    let decremented_high = alu(w, id, Op::ISub, uint, &[high, borrow_word]);

    let down_low = alu(
        w,
        id,
        Op::Select,
        uint,
        &[negative, incremented_low, decremented_low],
    );
    let down_high = alu(
        w,
        id,
        Op::Select,
        uint,
        &[negative, incremented_high, decremented_high],
    );
    let up_low = alu(
        w,
        id,
        Op::Select,
        uint,
        &[negative, decremented_low, incremented_low],
    );
    let up_high = alu(
        w,
        id,
        Op::Select,
        uint,
        &[negative, decremented_high, incremented_high],
    );
    let toward_zero_low = alu(w, id, Op::Select, uint, &[negative, up_low, down_low]);
    let toward_zero_high = alu(w, id, Op::Select, uint, &[negative, up_high, down_high]);

    let operand_is_finite = |w: &mut Vec<u32>, id: &mut u32, value| {
        let value_bits = alu(w, id, Op::Bitcast, words, &[value]);
        let value_high = alu(w, id, Op::CompositeExtract, uint, &[value_bits, 1]);
        let magnitude = alu(w, id, Op::BitwiseAnd, uint, &[value_high, magnitude_mask]);
        alu(w, id, Op::ULessThan, boolt, &[magnitude, infinity_high])
    };
    let a_finite = operand_is_finite(w, id, a);
    let b_finite = operand_is_finite(w, id, b);
    let finite_operands = alu(w, id, Op::LogicalAnd, boolt, &[a_finite, b_finite]);
    let high_is_positive_inf = alu(w, id, Op::IEqual, boolt, &[high, infinity_high]);
    let positive_infinity = alu(
        w,
        id,
        Op::LogicalAnd,
        boolt,
        &[low_zero, high_is_positive_inf],
    );
    let negative_infinity_high = alu(w, id, Op::BitwiseOr, uint, &[infinity_high, sign_mask]);
    let high_is_negative_inf = alu(w, id, Op::IEqual, boolt, &[high, negative_infinity_high]);
    let negative_infinity = alu(
        w,
        id,
        Op::LogicalAnd,
        boolt,
        &[low_zero, high_is_negative_inf],
    );
    let infinity = alu(
        w,
        id,
        Op::LogicalOr,
        boolt,
        &[positive_infinity, negative_infinity],
    );
    let overflow = alu(w, id, Op::LogicalAnd, boolt, &[finite_operands, infinity]);
    let positive_away = alu(w, id, Op::LogicalAnd, boolt, &[nonnegative, error_negative]);
    let negative_away = alu(w, id, Op::LogicalAnd, boolt, &[negative, error_positive]);
    let away = alu(w, id, Op::LogicalOr, boolt, &[positive_away, negative_away]);
    let adjust = alu(w, id, Op::LogicalOr, boolt, &[away, overflow]);
    let rounded_low = alu(w, id, Op::Select, uint, &[adjust, toward_zero_low, low]);
    let rounded_high = alu(w, id, Op::Select, uint, &[adjust, toward_zero_high, high]);

    let operand_is_negative_zero = |w: &mut Vec<u32>, id: &mut u32, value| {
        let value_bits = alu(w, id, Op::Bitcast, words, &[value]);
        let value_low = alu(w, id, Op::CompositeExtract, uint, &[value_bits, 0]);
        let value_high = alu(w, id, Op::CompositeExtract, uint, &[value_bits, 1]);
        let low_zero = alu(w, id, Op::IEqual, boolt, &[value_low, zero]);
        let high_negative_zero = alu(w, id, Op::IEqual, boolt, &[value_high, sign_mask]);
        alu(
            w,
            id,
            Op::LogicalAnd,
            boolt,
            &[low_zero, high_negative_zero],
        )
    };
    let a_negative_zero = operand_is_negative_zero(w, id, a);
    let b_negative_zero = operand_is_negative_zero(w, id, b);
    let both_negative_zero = alu(
        w,
        id,
        Op::LogicalAnd,
        boolt,
        &[a_negative_zero, b_negative_zero],
    );
    let zero_high = alu(
        w,
        id,
        Op::Select,
        uint,
        &[both_negative_zero, sign_mask, zero],
    );
    let rounded_low = alu(w, id, Op::Select, uint, &[sum_zero, zero, rounded_low]);
    let rounded_high = alu(
        w,
        id,
        Op::Select,
        uint,
        &[sum_zero, zero_high, rounded_high],
    );
    let rounded_bits = alu(
        w,
        id,
        Op::CompositeConstruct,
        words,
        &[rounded_low, rounded_high],
    );
    alu(w, id, Op::Bitcast, double, &[rounded_bits])
}
/// Both lanes of a bool vector2, lane 0 first.
fn bool_lanes(w: &mut Vec<u32>, id: &mut u32, boolt: u32, vector: u32) -> [u32; 2] {
    [
        alu(w, id, Op::CompositeExtract, boolt, &[vector, 0]),
        alu(w, id, Op::CompositeExtract, boolt, &[vector, 1]),
    ]
}

// Ryujinx Instructions/Lop3Expression.cs:13-31 defines LUT indexing:
// A has weight4, B2, C1. Shannon expansion computes the same truth table
// without searching permutations or expanding eight separate minterms.
// Equal cofactors and constant cofactors fold at translation time.
fn lop3_lut(
    w: &mut Vec<u32>,
    id: &mut u32,
    uint: u32,
    inputs: &[u32],
    table: u8,
    zero: u32,
    ones: u32,
) -> u32 {
    let entries = 1u32 << inputs.len();
    let full = ((1u16 << entries) - 1) as u8;
    if table == 0 {
        return zero;
    }
    if table == full {
        return ones;
    }
    let half = entries / 2;
    let mask = ((1u16 << half) - 1) as u8;
    let low = table & mask;
    let high = table >> half;
    if low == high {
        return lop3_lut(w, id, uint, &inputs[1..], low, zero, ones);
    }
    let input = inputs[0];
    let a = lop3_lut(w, id, uint, &inputs[1..], low, zero, ones);
    let b = lop3_lut(w, id, uint, &inputs[1..], high, zero, ones);
    match (low, high) {
        (0, h) if h == mask => input,
        (l, 0) if l == mask => alu(w, id, Op::Not, uint, &[input]),
        (0, _) => alu(w, id, Op::BitwiseAnd, uint, &[input, b]),
        (_, 0) => {
            let not = alu(w, id, Op::Not, uint, &[input]);
            alu(w, id, Op::BitwiseAnd, uint, &[not, a])
        }
        (l, _) if l == mask => {
            let not = alu(w, id, Op::Not, uint, &[input]);
            alu(w, id, Op::BitwiseOr, uint, &[not, b])
        }
        (_, h) if h == mask => alu(w, id, Op::BitwiseOr, uint, &[input, a]),
        _ => {
            let delta = alu(w, id, Op::BitwiseXor, uint, &[a, b]);
            let selected = alu(w, id, Op::BitwiseAnd, uint, &[input, delta]);
            alu(w, id, Op::BitwiseXor, uint, &[a, selected])
        }
    }
}

// Ryujinx InstEmitConversion.cs:136-142,187-194 requires mathematical
// rounding. GLSL.std.450.h:21-28 assigns RoundEven2, Trunc3, Floor8, Ceil9.
// Use binary32 bits for directed subnormal endpoints, avoiding Vulkan's
// default denormal flushing without adding float-control device features.
// Mesa src/util/u_math.h:91-100 defines exponent extraction.
fn round_f32(
    w: &mut Vec<u32>,
    id: &mut u32,
    bits: u32,
    (mode, ftz): (RoundingMode, bool),
    (uint, float, boolt, glsl): (u32, u32, u32, u32),
    constants: &HashMap<u32, u32>,
    globals: &mut Vec<u32>,
) -> (u32, u32) {
    // Preserve semantic-review A's pre-round FTZ boundary. Mesa sm50.rs:
    // 1793,1837 encodes FTZ44; Ryujinx InstEmitConversion.cs:136-142,187-194
    // defines rounding. Flush integer bits first so floor cannot turn a
    // negative subnormal into -1.
    let bits = FLUSH_F32[usize::from(ftz)](w, id, bits, (uint, boolt), constants);
    let f = alu(w, id, Op::Bitcast, float, &[bits]);
    let r = alu(
        w,
        id,
        Op::ExtInst,
        float,
        &[glsl, ROUND_OP[mode as usize], f],
    );
    let rounded = ROUNDED_ENDPOINT[usize::from(ftz)][mode as usize](
        w,
        id,
        (bits, r),
        (uint, float, boolt),
        constants,
        globals,
    );
    (f, rounded)
}
type FlushF32 = fn(&mut Vec<u32>, &mut u32, u32, (u32, u32), &HashMap<u32, u32>) -> u32;
/// By ftz: keep the bits, or flush a denormal to its signed zero.
const FLUSH_F32: [FlushF32; 2] = [|_, _, bits, _, _| bits, flush_f32];
/// GLSL.std.450 rounding by RoundingMode (NearestEven, NegativeInfinity,
/// PositiveInfinity, Zero): RoundEven2, Floor8, Ceil9, Trunc3.
const ROUND_OP: [u32; 4] = [2, 8, 9, 3];
type RoundedEndpoint = fn(
    &mut Vec<u32>,
    &mut u32,
    (u32, u32),
    (u32, u32, u32),
    &HashMap<u32, u32>,
    &mut Vec<u32>,
) -> u32;
/// By [ftz][RoundingMode]. Without FTZ, floor and ceil of a nonzero
/// subnormal take their directed endpoint: -1 or -0, +0 or 1.
const ROUNDED_ENDPOINT: [[RoundedEndpoint; 4]; 2] = [
    [
        |_, _, (_, r), _, _, _| r,
        subnormal_endpoint::<0xbf80_0000, 0>,
        subnormal_endpoint::<0x8000_0000, 0x3f80_0000>,
        |_, _, (_, r), _, _, _| r,
    ],
    [|_, _, (_, r), _, _, _| r; 4],
];
fn subnormal_endpoint<const NEG: u32, const POS: u32>(
    w: &mut Vec<u32>,
    id: &mut u32,
    (bits, r): (u32, u32),
    (uint, float, boolt): (u32, u32, u32),
    constants: &HashMap<u32, u32>,
    globals: &mut Vec<u32>,
) -> u32 {
    let mask = late_constant(globals, id, uint, 0x7f80_0000);
    let exponent = alu(w, id, Op::BitwiseAnd, uint, &[bits, mask]);
    let exp_zero = alu(w, id, Op::IEqual, boolt, &[exponent, constants[&0]]);
    let magnitude = alu(
        w,
        id,
        Op::BitwiseAnd,
        uint,
        &[bits, constants[&0x7fff_ffff]],
    );
    let nonzero = alu(w, id, Op::INotEqual, boolt, &[magnitude, constants[&0]]);
    let tiny = alu(w, id, Op::LogicalAnd, boolt, &[exp_zero, nonzero]);
    let sign = alu(
        w,
        id,
        Op::BitwiseAnd,
        uint,
        &[bits, constants[&0x8000_0000]],
    );
    let negative = alu(w, id, Op::INotEqual, boolt, &[sign, constants[&0]]);
    let neg = late_constant(globals, id, float, NEG);
    let pos = late_constant(globals, id, float, POS);
    let endpoint = alu(w, id, Op::Select, float, &[negative, neg, pos]);
    alu(w, id, Op::Select, float, &[tiny, endpoint, r])
}

// The half, video, shuffle, MUFU and condition-code formulas below are adapted
// from Ryujinx's InstEmitConditionCode.cs:56-81 and
// InstEmitHelper.cs:139-185, InstEmitFloatArithmetic.cs:251-383,483-528,
// InstEmitFloatComparison.cs:309-366,456-568,
// InstEmitAluHelper.cs:75-91,
// InstEmitVideoMinMax.cs:12-119, InstEmitVideoArithmetic.cs:15-112,
// HelperFunctionManager.cs:245-361,
// and InstEmitMultifunction.cs:32-83.
// The compare/select and population-count mappings also follow
// InstEmitIntegerComparison.cs:117-131 and InstEmitBitfield.cs:185-192.
//
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

// Adapted from Ryujinx Instructions/InstEmitConditionCode.cs:56-81 under
// the MIT notice above. Unlike loading all flags and their complements for
// every condition, build only the chosen formula. CSETP shares that result
// between both predicate outputs. Selector values follow Ryujinx
// Decoders/InstDecoders.cs:78-111. CSETP handles its typed CSM state at the
// call site because Ryujinx retains GetCondition's true default for it.
/// The flags a condition formula reads, and where it writes.
struct ConditionFlags<'a> {
    w: &'a mut Vec<u32>,
    id: &'a mut u32,
    /// Zero, sign, carry, overflow.
    flags: [u32; 4],
    boolt: u32,
    yes: u32,
    no: u32,
}

impl ConditionFlags<'_> {
    fn flag(&mut self, flag: usize) -> u32 {
        alu(self.w, self.id, Op::Load, self.boolt, &[self.flags[flag]])
    }
    fn z(&mut self) -> u32 {
        self.flag(0)
    }
    fn s(&mut self) -> u32 {
        self.flag(1)
    }
    fn c(&mut self) -> u32 {
        self.flag(2)
    }
    fn o(&mut self) -> u32 {
        self.flag(3)
    }
    fn not(&mut self, v: u32) -> u32 {
        alu(self.w, self.id, Op::LogicalNot, self.boolt, &[v])
    }
    fn bin(&mut self, op: Op, a: u32, b: u32) -> u32 {
        alu(self.w, self.id, op, self.boolt, &[a, b])
    }
    /// S != O.
    fn sign_differs(&mut self) -> u32 {
        let s = self.s();
        let o = self.o();
        self.bin(Op::LogicalNotEqual, s, o)
    }
    /// S || Z.
    fn sign_or_zero(&mut self) -> u32 {
        let s = self.s();
        let z = self.z();
        self.bin(Op::LogicalOr, s, z)
    }
}

/// The value of a condition, computed from the flags it reads.
trait ConditionTest {
    fn test(self, flags: &mut ConditionFlags) -> u32;
}

impl ConditionTest for CSetPCondition {
    fn test(self, flags: &mut ConditionFlags) -> u32 {
        CONDITION[usize::from(self.selector())](flags)
    }
}

/// By selector. CSM selectors 24-29 read true: Ryujinx retains
/// GetCondition's true default for them.
const CONDITION: [fn(&mut ConditionFlags) -> u32; 32] = [
    // 0x0 F
    |f| f.no,
    // 0x1 LT: (S && !Z) != O
    |f| {
        let s = f.s();
        let z = f.z();
        let nz = f.not(z);
        let a = f.bin(Op::LogicalAnd, s, nz);
        let o = f.o();
        f.bin(Op::LogicalNotEqual, a, o)
    },
    // 0x2 EQ: !S && Z
    |f| {
        let s = f.s();
        let ns = f.not(s);
        let z = f.z();
        f.bin(Op::LogicalAnd, ns, z)
    },
    // 0x3 LE: S != (Z || O)
    |f| {
        let s = f.s();
        let z = f.z();
        let o = f.o();
        let zo = f.bin(Op::LogicalOr, z, o);
        f.bin(Op::LogicalNotEqual, s, zo)
    },
    // 0x4 GT: !((S != O) || Z)
    |f| {
        let d = f.sign_differs();
        let z = f.z();
        let a = f.bin(Op::LogicalOr, d, z);
        f.not(a)
    },
    // 0x5 NE: !Z
    |f| {
        let z = f.z();
        f.not(z)
    },
    // 0x6 GE: !(S != O)
    |f| {
        let d = f.sign_differs();
        f.not(d)
    },
    // 0x7 NUM: !(S && Z)
    |f| {
        let s = f.s();
        let z = f.z();
        let a = f.bin(Op::LogicalAnd, s, z);
        f.not(a)
    },
    // 0x8 NAN: S && Z
    |f| {
        let s = f.s();
        let z = f.z();
        f.bin(Op::LogicalAnd, s, z)
    },
    // 0x9 LTU: S != O
    |f| f.sign_differs(),
    // 0xa EQU: Z
    |f| f.z(),
    // 0xb LEU: (S != O) || Z
    |f| {
        let d = f.sign_differs();
        let z = f.z();
        f.bin(Op::LogicalOr, d, z)
    },
    // 0xc GTU: !S != (O || Z)
    |f| {
        let s = f.s();
        let ns = f.not(s);
        let o = f.o();
        let z = f.z();
        let oz = f.bin(Op::LogicalOr, o, z);
        f.bin(Op::LogicalNotEqual, ns, oz)
    },
    // 0xd NEU: S || !Z
    |f| {
        let s = f.s();
        let z = f.z();
        let nz = f.not(z);
        f.bin(Op::LogicalOr, s, nz)
    },
    // 0xe GEU: (!S || Z) != O
    |f| {
        let s = f.s();
        let ns = f.not(s);
        let z = f.z();
        let a = f.bin(Op::LogicalOr, ns, z);
        let o = f.o();
        f.bin(Op::LogicalNotEqual, a, o)
    },
    // 0xf T
    |f| f.yes,
    // 0x10 OFF: !O
    |f| {
        let o = f.o();
        f.not(o)
    },
    // 0x11 LO: !C
    |f| {
        let c = f.c();
        f.not(c)
    },
    // 0x12 SFF: !S
    |f| {
        let s = f.s();
        f.not(s)
    },
    // 0x13 LS: Z || !C
    |f| {
        let z = f.z();
        let c = f.c();
        let nc = f.not(c);
        f.bin(Op::LogicalOr, z, nc)
    },
    // 0x14 HI: C && !Z
    |f| {
        let c = f.c();
        let z = f.z();
        let nz = f.not(z);
        f.bin(Op::LogicalAnd, c, nz)
    },
    // 0x15 SFT: S
    |f| f.s(),
    // 0x16 HS: C
    |f| f.c(),
    // 0x17 OFT: O
    |f| f.o(),
    |f| f.yes,
    |f| f.yes,
    |f| f.yes,
    |f| f.yes,
    |f| f.yes,
    |f| f.yes,
    // 0x1e RLE: S || Z
    |f| f.sign_or_zero(),
    // 0x1f RGT: !(S || Z)
    |f| {
        let a = f.sign_or_zero();
        f.not(a)
    },
];

/// Exact finite binary16 to binary32 conversion for immediate HADD2 operands.
/// Ryujinx InstEmitHelper.cs:105-125 converts each half before arithmetic;
/// Mesa src/util/half_float.c:157-169 specifies exponent adjustment and sign.
/// Fold finite halves into constants, removing an unpack and two extracts.
/// Keep Inf/NaN on the shared unpack path, preserving its representation.
fn half_immediate(bits: u32) -> Option<[u32; 2]> {
    let convert = |half: u32| {
        let exponent = (half >> 10) & 31;
        let fraction = half & 1023;
        let magnitude = match exponent {
            0 => (fraction as f32 * f32::from_bits(0x3380_0000)).to_bits(),
            31 => return None,
            _ => ((exponent + 112) << 23) | (fraction << 13),
        };
        Some(magnitude | ((half & 0x8000) << 16))
    };
    Some([convert(bits & 0xffff)?, convert(bits >> 16)?])
}

fn half_pair(
    w: &mut Vec<u32>,
    id: &mut u32,
    bits: u32,
    swizzle: u8,
    abs: bool,
    negate: bool,
    (float, vec2, glsl): (u32, u32, u32),
) -> u32 {
    // Ryujinx InstEmitHelper.cs:139-159, InstEmitAluHelper.cs:75-91.
    // Keep lanes in a vector: one unpack, one arithmetic operation and one
    // pack replace separate extracts, scalar arithmetic and reconstruction.
    // GLSL.std.450 section 2 UnpackHalf2x16=62, FAbs=4;
    // SPIR-V 1.6 sections 3.3.11-13 Bitcast=124, CompositeConstruct=80,
    // VectorShuffle=79, FNegate=127.
    let types = (float, vec2, glsl);
    let pair = HALF_SWIZZLE[usize::from(swizzle)](w, id, bits, types);
    let pair = HALF_ABS[usize::from(abs)](w, id, pair, types);
    HALF_NEGATE[usize::from(negate)](w, id, pair, types)
}
type HalfLanes = fn(&mut Vec<u32>, &mut u32, u32, (u32, u32, u32)) -> u32;
/// By swizzle (H1_H0, F32, H0_H0, H1_H1): the source's two lanes.
const HALF_SWIZZLE: [HalfLanes; 4] = [
    |w, id, bits, (_, vec2, glsl)| alu(w, id, Op::ExtInst, vec2, &[glsl, 62, bits]),
    |w, id, bits, (float, vec2, _)| {
        let f = alu(w, id, Op::Bitcast, float, &[bits]);
        alu(w, id, Op::CompositeConstruct, vec2, &[f, f])
    },
    half_lane::<0>,
    half_lane::<1>,
];
/// One unpacked lane broadcast to both lanes.
fn half_lane<const LANE: u32>(
    w: &mut Vec<u32>,
    id: &mut u32,
    bits: u32,
    (_, vec2, glsl): (u32, u32, u32),
) -> u32 {
    let pair = alu(w, id, Op::ExtInst, vec2, &[glsl, 62, bits]);
    alu(w, id, Op::VectorShuffle, vec2, &[pair, pair, LANE, LANE])
}
/// By abs: GLSL.std.450 FAbs on both lanes.
const HALF_ABS: [HalfLanes; 2] = [
    |_, _, pair, _| pair,
    |w, id, pair, (_, vec2, glsl)| alu(w, id, Op::ExtInst, vec2, &[glsl, 4, pair]),
];
/// By negate.
const HALF_NEGATE: [HalfLanes; 2] = [
    |_, _, pair, _| pair,
    |w, id, pair, (_, vec2, _)| alu(w, id, Op::FNegate, vec2, &[pair]),
];

fn half_sign_bits(
    w: &mut Vec<u32>,
    id: &mut u32,
    bits: u32,
    swizzle: u8,
    (absolute, negative): (bool, bool),
    uint: u32,
    globals: &mut Vec<u32>,
) -> u32 {
    // Ryujinx InstEmitHelper.cs:139-158 defines F16/F32/H0/H1 sources;
    // InstEmitAluHelper.cs:75-91 applies absolute then negate. Binary16
    // sign15 and binary32 sign31: Mesa src/util/half_float.c:60-67,168-169.
    // FMZ exposes NaN operand signs as signed zero (the accepted contract
    // in docs/reference-sources.md). Floating NaN sign is not
    // compared by the GPU exact tests, but a zero sign is. Use encoded
    // signs so NaN conversion/arithmetic cannot change the FMZ zero sign.
    // Integer shifts/OR/XOR/AND: SPIRV-Headers
    // include/spirv/unified1/spirv.core.grammar.json:2251-2321.
    // Keep both signs packed in a scalar until their final F32 vector.
    HALF_SIGN[usize::from(absolute)](w, id, bits, swizzle, negative, uint, globals)
}
type HalfSign = fn(&mut Vec<u32>, &mut u32, u32, u8, bool, u32, &mut Vec<u32>) -> u32;
/// By absolute. Both modifier bits are static: absolute folds with negate
/// into a constant; otherwise read the encoded signs, then negate them.
const HALF_SIGN: [HalfSign; 2] = [
    |w, id, bits, swizzle, negative, uint, globals| {
        let mask = late_constant(globals, id, uint, HALF_SIGN_MASK[usize::from(swizzle)]);
        let sign = alu(w, id, Op::BitwiseAnd, uint, &[bits, mask]);
        let sign = HALF_SIGN_SPREAD[usize::from(swizzle)](w, id, sign, uint, globals);
        HALF_SIGN_NEGATE[usize::from(negative)](w, id, sign, uint, globals)
    },
    |_, id, _, _, negative, uint, globals| {
        late_constant(globals, id, uint, [0, 0x8000_8000][usize::from(negative)])
    },
];
/// The sign bits each swizzle reads (H1_H0, F32, H0_H0, H1_H1).
const HALF_SIGN_MASK: [u32; 4] = [0x8000_8000, 0x8000_0000, 0x0000_8000, 0x8000_0000];
type SignBits = fn(&mut Vec<u32>, &mut u32, u32, u32, &mut Vec<u32>) -> u32;
/// By swizzle: a broadcast source copies its one sign into the other lane.
const HALF_SIGN_SPREAD: [SignBits; 4] = [
    |_, _, sign, _, _| sign,
    |w, id, sign, uint, globals| half_sign_copy(w, id, sign, uint, globals, Op::ShiftRightLogical),
    |w, id, sign, uint, globals| half_sign_copy(w, id, sign, uint, globals, Op::ShiftLeftLogical),
    |w, id, sign, uint, globals| half_sign_copy(w, id, sign, uint, globals, Op::ShiftRightLogical),
];
fn half_sign_copy(
    w: &mut Vec<u32>,
    id: &mut u32,
    sign: u32,
    uint: u32,
    globals: &mut Vec<u32>,
    shift_op: Op,
) -> u32 {
    let shift = late_constant(globals, id, uint, 16);
    let other = alu(w, id, shift_op, uint, &[sign, shift]);
    alu(w, id, Op::BitwiseOr, uint, &[sign, other])
}
/// By negative.
const HALF_SIGN_NEGATE: [SignBits; 2] = [
    |_, _, sign, _, _| sign,
    |w, id, sign, uint, globals| {
        let mask = late_constant(globals, id, uint, 0x8000_8000);
        alu(w, id, Op::BitwiseXor, uint, &[sign, mask])
    },
];

fn flush_half_pair(
    w: &mut Vec<u32>,
    id: &mut u32,
    pair: u32,
    f32_format: bool,
    (uint, uvec2, vec2, bool_vec2): (u32, u32, u32, u32),
    globals: &mut Vec<u32>,
) -> u32 {
    // Ryujinx InstEmitHelper.cs:139-158 distinguishes F32 broadcast from
    // binary16 lanes; Mesa src/util/half_float.c:96-104 gives the half normal
    // boundary 2^-14. Mesa nak/from_nir.rs:1313-1320 preserves the zero sign.
    // The FTZ/FMZ contract flushes inputs and outputs to signed zero. Ryujinx InstEmitFloatArithmetic.cs:483-528 does not
    // implement this mode. See docs/reference-sources.md
    // Integer magnitude comparisons preserve NaNs and zero signs and avoid
    // driver-dependent float denormal comparisons. Both lanes share one test.
    let mut splat = |bits| {
        let c = late_constant(globals, id, uint, bits);
        alu(w, id, Op::CompositeConstruct, uvec2, &[c, c])
    };
    let magnitude_mask = splat(0x7fff_ffff);
    let sign_mask = splat(0x8000_0000);
    let normal = splat([0x3880_0000, 0x0080_0000][usize::from(f32_format)]);
    let bits = alu(w, id, Op::Bitcast, uvec2, &[pair]);
    let magnitude = alu(w, id, Op::BitwiseAnd, uvec2, &[bits, magnitude_mask]);
    let sign = alu(w, id, Op::BitwiseAnd, uvec2, &[bits, sign_mask]);
    let subnormal = alu(w, id, Op::ULessThan, bool_vec2, &[magnitude, normal]);
    let flushed = alu(w, id, Op::Select, uvec2, &[subnormal, sign, bits]);
    alu(w, id, Op::Bitcast, vec2, &[flushed])
}

fn flush_f32(
    w: &mut Vec<u32>,
    id: &mut u32,
    bits: u32,
    (uint, boolt): (u32, u32),
    constants: &HashMap<u32, u32>,
) -> u32 {
    // Mesa from_nir.rs:1313-1320 flushes a denormal to its correctly signed
    // zero; opt_copy_prop.rs:323-325 prevents propagation across FTZ consumers.
    // SPIR-V 1.6 section 3.2.7 DenormFlushToZero defines both input and result
    // flushing. Use integer magnitude < smallest normal instead of a float
    // comparison, preserving NaN/Inf bits and avoiding host denorm behavior.
    let magnitude = alu(
        w,
        id,
        Op::BitwiseAnd,
        uint,
        &[bits, constants[&0x7fff_ffff]],
    );
    let tiny = alu(
        w,
        id,
        Op::ULessThan,
        boolt,
        &[magnitude, constants[&0x0080_0000]],
    );
    let zero = alu(
        w,
        id,
        Op::BitwiseAnd,
        uint,
        &[bits, constants[&0x8000_0000]],
    );
    alu(w, id, Op::Select, uint, &[tiny, zero, bits])
}

// Mesa nir_opcodes.py:265 and util/u_math.h:343-371 define saturation as
// min(max-number(x,+0),1). Integer ordering makes NaN and -0 deterministic
// and preserves positive denormals without float min/max instructions.
// UGreaterThan172: Mesa spirv.core.grammar.json:2011-2025.
fn saturate_f32(
    w: &mut Vec<u32>,
    id: &mut u32,
    bits: u32,
    (uint, boolt): (u32, u32),
    constants: &HashMap<u32, u32>,
    globals: &mut Vec<u32>,
) -> u32 {
    let magnitude = alu(
        w,
        id,
        Op::BitwiseAnd,
        uint,
        &[bits, constants[&0x7fff_ffff]],
    );
    let nan = alu(
        w,
        id,
        Op::UGreaterThan,
        boolt,
        &[magnitude, constants[&0x7f80_0000]],
    );
    let negative = alu(
        w,
        id,
        Op::UGreaterThan,
        boolt,
        &[bits, constants[&0x7fff_ffff]],
    );
    let zero = alu(w, id, Op::LogicalOr, boolt, &[nan, negative]);
    let one = late_constant(globals, id, uint, 0x3f80_0000);
    let large = alu(w, id, Op::UGreaterThan, boolt, &[bits, one]);
    let capped = alu(w, id, Op::Select, uint, &[large, one, bits]);
    alu(w, id, Op::Select, uint, &[zero, constants[&0], capped])
}

/// The float constants 2^power that FMUL's scale multiplies A by, one per
/// FMulScale that scales.
#[derive(Clone, Copy)]
struct ScaleFactors {
    divide8: u32,
    divide4: u32,
    divide2: u32,
    multiply2: u32,
    multiply4: u32,
    multiply8: u32,
}
/// The module state that FADD and FMUL and their modifiers write through.
struct FloatOps<'a> {
    w: &'a mut Vec<u32>,
    id: &'a mut u32,
    uint: u32,
    float: u32,
    boolt: u32,
    glsl_std_450: u32,
    constants: &'a HashMap<u32, u32>,
    scale_factors: ScaleFactors,
    annotations: &'a mut Vec<u32>,
    late_globals: &'a mut Vec<u32>,
}
impl FloatOps<'_> {
    /// Both operand bit patterns, flushed per `denormals`, as floats.
    fn operands(&mut self, denormals: Denormals, a: u32, b: u32) -> (u32, u32) {
        let a = denormals.flush_bits(self, a);
        let b = denormals.flush_bits(self, b);
        let af = fresh(self.id);
        let bf = fresh(self.id);
        inst(self.w, Op::Bitcast, &[self.float, af, a]);
        inst(self.w, Op::Bitcast, &[self.float, bf, b]);
        (af, bf)
    }
    /// The binary32 sum, kept uncontracted.
    fn sum(&mut self, a: u32, b: u32) -> u32 {
        let f = alu(self.w, self.id, Op::FAdd, self.float, &[a, b]);
        inst(self.annotations, Op::Decorate, &[f, 42]); // NoContraction
        f
    }
    /// The nearest-even binary32 product, kept uncontracted.
    fn product(&mut self, a: u32, b: u32) -> u32 {
        let f = alu(self.w, self.id, Op::FMul, self.float, &[a, b]);
        inst(self.annotations, Op::Decorate, &[f, 42]); // NoContraction
        f
    }
    /// `a` times the constant `factor`, kept uncontracted.
    fn scaled(&mut self, a: u32, factor: u32) -> u32 {
        let f = alu(self.w, self.id, Op::FMul, self.float, &[a, factor]);
        inst(self.annotations, Op::Decorate, &[f, 42]);
        f
    }
    /// A float result's bits, flushed per `denormals`.
    fn bits(&mut self, f: u32, denormals: Denormals) -> u32 {
        let f = denormals.flush_float(self, f);
        alu(self.w, self.id, Op::Bitcast, self.uint, &[f])
    }
}
/// How a float source modifier reads a source.
trait ReadFloat {
    fn read(self, ops: &mut FloatOps, f: u32) -> u32;
}
/// How FTZ reads inputs and writes a result.
trait FlushDenormals {
    fn flush_bits(self, ops: &mut FloatOps, bits: u32) -> u32;
    fn flush_float(self, ops: &mut FloatOps, f: u32) -> u32;
}
/// How a result modifier writes result bits.
trait WriteFloat {
    fn write(self, ops: &mut FloatOps, bits: u32) -> u32;
}
/// How an FMUL rounding multiplies.
trait MultiplyFloat {
    fn multiply(self, ops: &mut FloatOps, a: u32, b: u32, binary64: Binary64Product) -> u32;
}
/// How an FADD rounding adds two already-modified float sources.
trait RoundFloatAdd {
    fn add(self, ops: &mut FloatOps, a: u32, b: u32, binary64: Binary64Product) -> u32;
}
/// What an instruction, or an FMUL's modifiers or rounding, contributes
/// to the module's FMUL product lane.
trait FMulLaneContribution {
    fn fmul_lane(self, lane: FMulProductLane) -> FMulProductLane;
}
/// How an FADD form adds its sources.
trait EmitFloatAdd {
    fn sum(
        self,
        ops: &mut FloatOps,
        a: u32,
        b: u32,
        round: RoundingMode,
        binary64: Binary64Product,
    ) -> u32;
}
/// How an FMUL form multiplies its sources.
trait EmitFloatMultiply {
    fn product(self, ops: &mut FloatOps, a: u32, b: u32, binary64: Binary64Product) -> u32;
}
impl FlushDenormals for Denormals {
    /// Bits as FTZ reads them.
    fn flush_bits(self, ops: &mut FloatOps, bits: u32) -> u32 {
        FLUSH_BITS[self as usize](ops, bits)
    }
    /// A float as FTZ writes it.
    fn flush_float(self, ops: &mut FloatOps, f: u32) -> u32 {
        FLUSH_FLOAT[self as usize](ops, f)
    }
}
type FloatStep = fn(&mut FloatOps, u32) -> u32;
/// By Denormals (Preserved, FlushedToZero).
const FLUSH_BITS: [FloatStep; 2] = [
    |_, bits| bits,
    |ops, bits| flush_f32(ops.w, ops.id, bits, (ops.uint, ops.boolt), ops.constants),
];
/// By Denormals (Preserved, FlushedToZero).
const FLUSH_FLOAT: [FloatStep; 2] = [
    |_, f| f,
    |ops, f| {
        let bits = alu(ops.w, ops.id, Op::Bitcast, ops.uint, &[f]);
        let bits = flush_f32(ops.w, ops.id, bits, (ops.uint, ops.boolt), ops.constants);
        alu(ops.w, ops.id, Op::Bitcast, ops.float, &[bits])
    },
];
/// How FMUL's denormal mode reads its operands and writes its product.
trait FlushProduct {
    fn denormals(self) -> Denormals;
    fn zero_product(self, ops: &mut FloatOps, a: u32, b: u32, bits: u32) -> u32;
}
impl FlushProduct for Flush {
    /// FTZ and DNZ both flush; DNZ implies FTZ (Mesa from_nir.rs:1280-1282).
    fn denormals(self) -> Denormals {
        FLUSH_DENORMALS[self as usize]
    }
    /// Product bits as DNZ writes them: +0 when the flushed float `a` or
    /// `b` is +/-0, even against Inf or NaN (Mesa nir_opcodes.py:783-793).
    fn zero_product(self, ops: &mut FloatOps, a: u32, b: u32, bits: u32) -> u32 {
        ZERO_PRODUCT[self as usize](ops, a, b, bits)
    }
}
/// By Flush (None, Ftz, Dnz).
const FLUSH_DENORMALS: [Denormals; 3] = [
    Denormals::Preserved,
    Denormals::FlushedToZero,
    Denormals::FlushedToZero,
];
type ProductStep = fn(&mut FloatOps, u32, u32, u32) -> u32;
/// By Flush (None, Ftz, Dnz): only DNZ replaces a zero-operand product.
const ZERO_PRODUCT: [ProductStep; 3] = [
    |_, _, _, bits| bits,
    |_, _, _, bits| bits,
    |ops, a, b, bits| {
        let mut zero = |f: u32| {
            let v = alu(ops.w, ops.id, Op::Bitcast, ops.uint, &[f]);
            let v = alu(
                ops.w,
                ops.id,
                Op::BitwiseAnd,
                ops.uint,
                &[v, ops.constants[&0x7fff_ffff]],
            );
            alu(
                ops.w,
                ops.id,
                Op::IEqual,
                ops.boolt,
                &[v, ops.constants[&0]],
            )
        };
        let az = zero(a);
        let bz = zero(b);
        let any = alu(ops.w, ops.id, Op::LogicalOr, ops.boolt, &[az, bz]);
        alu(
            ops.w,
            ops.id,
            Op::Select,
            ops.uint,
            &[any, ops.constants[&0], bits],
        )
    },
];
impl WriteFloat for Saturation {
    /// Result bits as SAT writes them.
    fn write(self, ops: &mut FloatOps, bits: u32) -> u32 {
        SATURATE[self as usize](ops, bits)
    }
}
/// By Saturation (Unclamped, Saturated).
const SATURATE: [FloatStep; 2] = [
    |_, bits| bits,
    |ops, bits| {
        saturate_f32(
            ops.w,
            ops.id,
            bits,
            (ops.uint, ops.boolt),
            ops.constants,
            ops.late_globals,
        )
    },
];
impl ReadFloat for SourceAbs {
    /// A float source as abs reads it: GLSL.std.450 FAbs (4).
    fn read(self, ops: &mut FloatOps, f: u32) -> u32 {
        SOURCE_ABS[self as usize](ops, f)
    }
}
/// By SourceAbs (Plain, Absolute).
const SOURCE_ABS: [FloatStep; 2] = [
    |_, f| f,
    |ops, f| {
        let v = fresh(ops.id);
        inst(ops.w, Op::ExtInst, &[ops.float, v, ops.glsl_std_450, 4, f]);
        v
    },
];
impl ReadFloat for SourceNegation {
    /// A float source as negation reads it.
    fn read(self, ops: &mut FloatOps, f: u32) -> u32 {
        SOURCE_NEGATION[self as usize](ops, f)
    }
}
/// By SourceNegation (Plain, Negated).
const SOURCE_NEGATION: [FloatStep; 2] = [
    |_, f| f,
    |ops, f| {
        let v = fresh(ops.id);
        inst(ops.w, Op::FNegate, &[ops.float, v, f]);
        v
    },
];
impl ReadFloat for FMulScale {
    /// A scaled by its power of two; Ryujinx InstEmitFloatArithmetic.cs:450-476
    /// scales A BEFORE multiplying B. Keep that order for overflow and
    /// rounding. No algebraic reassociation or FMA.
    fn read(self, ops: &mut FloatOps, a: u32) -> u32 {
        FMUL_SCALE[self as usize](ops, a)
    }
}
/// By FMulScale (Unscaled, Divide2, Divide4, Divide8, Multiply8,
/// Multiply4, Multiply2).
const FMUL_SCALE: [FloatStep; 7] = [
    |_, a| a,
    |ops, a| ops.scaled(a, ops.scale_factors.divide2),
    |ops, a| ops.scaled(a, ops.scale_factors.divide4),
    |ops, a| ops.scaled(a, ops.scale_factors.divide8),
    |ops, a| ops.scaled(a, ops.scale_factors.multiply8),
    |ops, a| ops.scaled(a, ops.scale_factors.multiply4),
    |ops, a| ops.scaled(a, ops.scale_factors.multiply2),
];
/// Whether the module declared the binary64 type that directed FMUL
/// rounding multiplies in. Without it (no host Float64 support), a directed
/// mode takes the nearest-even product: Ryujinx SpirvGenerator.cs:58-60
/// gates Float64, InstEmitFloatArithmetic.cs:435-480 is the fallback.
#[derive(Clone, Copy)]
enum Binary64Product {
    Declared { double: u32 },
    Undeclared,
}
/// How the module's binary64 declaration multiplies in a directed mode.
trait DirectedMultiply {
    fn directed(self, ops: &mut FloatOps, a: u32, b: u32, mode: DirectedRounding) -> u32;
}
impl DirectedMultiply for Binary64Product {
    fn directed(self, ops: &mut FloatOps, a: u32, b: u32, mode: DirectedRounding) -> u32 {
        match self {
            Binary64Product::Declared { double } => directed_fmul_f32(
                ops.w,
                ops.id,
                a,
                b,
                mode,
                (ops.uint, ops.float, double, ops.boolt),
                ops.constants,
                ops.late_globals,
                ops.annotations,
            ),
            Binary64Product::Undeclared => ops.product(a, b),
        }
    }
}
impl RoundFloatAdd for RoundingMode {
    fn add(self, ops: &mut FloatOps, a: u32, b: u32, binary64: Binary64Product) -> u32 {
        let mode = match self {
            RoundingMode::NearestEven => return ops.sum(a, b),
            RoundingMode::NegativeInfinity => DirectedRounding::NegativeInfinity,
            RoundingMode::PositiveInfinity => DirectedRounding::PositiveInfinity,
            RoundingMode::Zero => DirectedRounding::Zero,
        };
        match binary64 {
            Binary64Product::Declared { double } => directed_fadd_f32(
                ops.w,
                ops.id,
                a,
                b,
                mode,
                (ops.uint, ops.float, double, ops.boolt),
                ops.constants,
                ops.late_globals,
                ops.annotations,
            ),
            Binary64Product::Undeclared => ops.sum(a, b),
        }
    }
}
impl MultiplyFloat for FMulRounding {
    /// The product in this rounding.
    fn multiply(self, ops: &mut FloatOps, a: u32, b: u32, binary64: Binary64Product) -> u32 {
        match self {
            FMulRounding::NearestEven => ops.product(a, b),
            FMulRounding::Directed(mode) => binary64.directed(ops, a, b, mode),
        }
    }
}
impl FMulLaneContribution for FMulRounding {
    /// A directed mode takes the binary64 product.
    fn fmul_lane(self, lane: FMulProductLane) -> FMulProductLane {
        match self {
            FMulRounding::NearestEven => lane,
            FMulRounding::Directed(_) => FMulProductLane::Binary64,
        }
    }
}
impl EmitFloatAdd for FAddModifiers {
    /// The FADD result bits of float sources `a` and `b` under this form's
    /// modifiers: FTZ, abs, then negation, the sum, FTZ and SAT.
    fn sum(
        self,
        ops: &mut FloatOps,
        a: u32,
        b: u32,
        round: RoundingMode,
        binary64: Binary64Product,
    ) -> u32 {
        match self {
            FAddModifiers::Operand {
                lhs_abs,
                rhs_abs,
                lhs_negation,
                rhs_negation,
                saturation,
                denormals,
            } => {
                let (af, bf) = ops.operands(denormals, a, b);
                let af = lhs_abs.read(ops, af);
                let bf = rhs_abs.read(ops, bf);
                let af = lhs_negation.read(ops, af);
                let bf = rhs_negation.read(ops, bf);
                let f = round.add(ops, af, bf, binary64);
                let bits = ops.bits(f, denormals);
                saturation.write(ops, bits)
            }
            FAddModifiers::Immediate32 {
                lhs_abs,
                rhs_abs,
                lhs_negation,
                rhs_negation,
                denormals,
            } => {
                let (af, bf) = ops.operands(denormals, a, b);
                let af = lhs_abs.read(ops, af);
                let bf = rhs_abs.read(ops, bf);
                let af = lhs_negation.read(ops, af);
                let bf = rhs_negation.read(ops, bf);
                let f = round.add(ops, af, bf, binary64);
                ops.bits(f, denormals)
            }
        }
    }
}
impl EmitFloatMultiply for FMulModifiers {
    /// The FMUL result bits of float sources `a` and `b` under this form's
    /// modifiers: FTZ, B negation, A scale, the product in its rounding,
    /// FTZ and SAT.
    fn product(self, ops: &mut FloatOps, a: u32, b: u32, binary64: Binary64Product) -> u32 {
        match self {
            FMulModifiers::Operand {
                rhs_negation,
                saturation,
                flush,
                rounding,
                scale,
            } => {
                let denormals = flush.denormals();
                let (af, bf) = ops.operands(denormals, a, b);
                let bf = rhs_negation.read(ops, bf);
                let af = scale.read(ops, af);
                let f = rounding.multiply(ops, af, bf, binary64);
                let bits = ops.bits(f, denormals);
                let bits = flush.zero_product(ops, af, bf, bits);
                saturation.write(ops, bits)
            }
            FMulModifiers::Immediate32 { saturation, flush } => {
                let denormals = flush.denormals();
                let (af, bf) = ops.operands(denormals, a, b);
                let f = ops.product(af, bf);
                let bits = ops.bits(f, denormals);
                let bits = flush.zero_product(ops, af, bf, bits);
                saturation.write(ops, bits)
            }
        }
    }
}
/// The product lane the module's FMULs multiply in: binary32 when every
/// FMUL rounds to nearest-even, binary64 when one rounds in a directed mode
/// (Round39..40, Mesa sm50.rs:695-700). Ryujinx SpirvGenerator.cs:58-60
/// gates Float64; without host support a directed mode takes the
/// nearest-even product (InstEmitFloatArithmetic.cs:435-480).
#[derive(Clone, Copy)]
enum FMulProductLane {
    Binary32,
    Binary64,
}
/// How the module's FMUL product lane shapes the module.
trait FMulLaneDeclaration {
    fn hosted(self, float64: crate::HostFloat64) -> FMulProductLane;
    fn needs_float64(self, conversions_need_float64: bool) -> bool;
    fn declare_rounding_mode(self, w: &mut Vec<u32>, fun: u32);
    fn binary64(self, double_types: Option<(u32, u32)>) -> Binary64Product;
}
impl FMulLaneDeclaration for FMulProductLane {
    /// The lane on this host: binary64 needs host Float64 support.
    fn hosted(self, float64: crate::HostFloat64) -> FMulProductLane {
        match float64 {
            crate::HostFloat64::Unsupported => FMulProductLane::Binary32,
            crate::HostFloat64::Supported => self,
        }
    }
    /// The Float64 capability: the binary64 lane needs it, and otherwise
    /// the conversions decide.
    fn needs_float64(self, conversions_need_float64: bool) -> bool {
        [conversions_need_float64, true][self as usize]
    }
    /// The binary64 lane's preserved signed-zero/Inf/NaN execution mode
    /// (Mesa grammar:13168-13195).
    fn declare_rounding_mode(self, w: &mut Vec<u32>, fun: u32) {
        [
            |_, _| (),
            |w, fun| inst(w, Op::ExecutionMode, &[fun, 4461, 64]),
        ][self as usize](w, fun)
    }
    /// The binary64 type the directed product multiplies in; the binary64
    /// lane declared it through needs_float64.
    fn binary64(self, double_types: Option<(u32, u32)>) -> Binary64Product {
        match self {
            FMulProductLane::Binary32 => Binary64Product::Undeclared,
            FMulProductLane::Binary64 => double_types
                .map_or(Binary64Product::Undeclared, |(double, _)| {
                    Binary64Product::Declared { double }
                }),
        }
    }
}
// FADD rounding39..40: Mesa sm50.rs:525. Two binary32 operands and their
// exact sum fit in binary64. Convert once to RTE binary32, compare with the
// exact sum, then step one encoding in the directed mode when needed.
#[allow(clippy::too_many_arguments)]
fn directed_fadd_f32(
    w: &mut Vec<u32>,
    id: &mut u32,
    a: u32,
    b: u32,
    mode: DirectedRounding,
    (uint, float, double, boolt): (u32, u32, u32, u32),
    constants: &HashMap<u32, u32>,
    globals: &mut Vec<u32>,
    annotations: &mut Vec<u32>,
) -> u32 {
    let a32 = a;
    let b32 = b;
    let a = alu(w, id, Op::FConvert, double, &[a]);
    let b = alu(w, id, Op::FConvert, double, &[b]);
    let exact = precise_f64(w, id, Op::FAdd, double, &[a, b], annotations);
    let rounded = directed_f32(
        w,
        id,
        exact,
        mode,
        (uint, float, double, boolt),
        constants,
        globals,
        annotations,
    );

    // IEEE 754 exact-zero sign: RM produces -0 unless both inputs are +0;
    // RP and RZ retain -0 only for -0 + -0.
    let zero = constants[&0];
    let sign = late_constant(globals, id, uint, 0x8000_0000);
    let zero_f64 = late_double(globals, id, double, 0.0);
    let exact_zero = alu(w, id, Op::FOrdEqual, boolt, &[exact, zero_f64]);
    let a_bits = alu(w, id, Op::Bitcast, uint, &[a32]);
    let b_bits = alu(w, id, Op::Bitcast, uint, &[b32]);
    let a_positive = alu(w, id, Op::IEqual, boolt, &[a_bits, zero]);
    let b_positive = alu(w, id, Op::IEqual, boolt, &[b_bits, zero]);
    let both_positive = alu(w, id, Op::LogicalAnd, boolt, &[a_positive, b_positive]);
    let a_negative = alu(w, id, Op::IEqual, boolt, &[a_bits, sign]);
    let b_negative = alu(w, id, Op::IEqual, boolt, &[b_bits, sign]);
    let both_negative = alu(w, id, Op::LogicalAnd, boolt, &[a_negative, b_negative]);
    let negative_zero = match mode {
        DirectedRounding::NegativeInfinity => {
            let not_both_positive = alu(w, id, Op::LogicalNot, boolt, &[both_positive]);
            alu(
                w,
                id,
                Op::LogicalAnd,
                boolt,
                &[exact_zero, not_both_positive],
            )
        }
        DirectedRounding::PositiveInfinity | DirectedRounding::Zero => {
            alu(w, id, Op::LogicalAnd, boolt, &[exact_zero, both_negative])
        }
    };
    let bits = alu(w, id, Op::Bitcast, uint, &[rounded]);
    let bits = alu(w, id, Op::Select, uint, &[negative_zero, sign, bits]);
    alu(w, id, Op::Bitcast, float, &[bits])
}

// FMUL rounding39..40: Mesa sm50.rs:467-475,695. Ryujinx's
// InstEmitFloatArithmetic.cs:476 multiplies, but omits directed rounding.
// A binary32 product has at most 48 significant bits and exponents -298..256,
// so binary64 represents every finite product exactly. Convert once to RTE,
// compare with that exact product, then step one binary32 encoding if needed.
// This is branchless and avoids a software multiply or significand loop.
#[allow(clippy::too_many_arguments)]
fn directed_fmul_f32(
    w: &mut Vec<u32>,
    id: &mut u32,
    a: u32,
    b: u32,
    mode: DirectedRounding,
    (uint, float, double, boolt): (u32, u32, u32, u32),
    constants: &HashMap<u32, u32>,
    globals: &mut Vec<u32>,
    annotations: &mut Vec<u32>,
) -> u32 {
    // Mesa spirv.core.grammar.json:1343-1353,1550-1561,2155-2201:
    // FConvert115, FMul133, FOrdLessThan184 and FOrdGreaterThan186.
    let a = alu(w, id, Op::FConvert, double, &[a]);
    let b = alu(w, id, Op::FConvert, double, &[b]);
    let exact = alu(w, id, Op::FMul, double, &[a, b]);
    inst(annotations, Op::Decorate, &[exact, 42]); // NoContraction, Ryujinx CodeGen/Spirv/Instructions.cs:2030.
    directed_f32(
        w,
        id,
        exact,
        mode,
        (uint, float, double, boolt),
        constants,
        globals,
        annotations,
    )
}
/// The binary64 value `exact` rounded to binary32 in a directed mode: the
/// nearest-even conversion, stepped one encoding when it lies on the
/// wrong side of `exact`. F2F.F32.F64 narrows its source the same way.
#[allow(clippy::too_many_arguments)]
fn directed_f32(
    w: &mut Vec<u32>,
    id: &mut u32,
    exact: u32,
    mode: DirectedRounding,
    (uint, float, double, boolt): (u32, u32, u32, u32),
    constants: &HashMap<u32, u32>,
    globals: &mut Vec<u32>,
    annotations: &mut Vec<u32>,
) -> u32 {
    let nearest = alu(w, id, Op::FConvert, float, &[exact]);
    // Mesa nir_opt_algebraic.py:1263-1264 can collapse imprecise
    // conversion chains. Keep this binary32 rounding checkpoint precise.
    inst(annotations, Op::Decorate, &[nearest, 42]);
    let widened = alu(w, id, Op::FConvert, double, &[nearest]);
    inst(annotations, Op::Decorate, &[widened, 42]);
    let bits = alu(w, id, Op::Bitcast, uint, &[nearest]);
    let negative = alu(
        w,
        id,
        Op::UGreaterThan,
        boolt,
        &[bits, constants[&0x7fff_ffff]],
    );
    let magnitude = alu(
        w,
        id,
        Op::BitwiseAnd,
        uint,
        &[bits, constants[&0x7fff_ffff]],
    );
    let zero = alu(w, id, Op::IEqual, boolt, &[magnitude, constants[&0]]);
    let incremented = alu(w, id, Op::IAdd, uint, &[bits, constants[&1]]);
    let decremented = alu(w, id, Op::ISub, uint, &[bits, constants[&1]]);
    let product = DirectedProduct {
        bits,
        negative,
        zero,
        incremented,
        decremented,
        widened,
        exact,
    };
    DIRECTED_FMUL[mode as usize](
        w,
        id,
        product,
        (uint, float, boolt),
        constants,
        globals,
        annotations,
    )
}
/// The nearest-even binary32 product and the values its directed step reads.
#[derive(Clone, Copy)]
struct DirectedProduct {
    bits: u32,
    negative: u32,
    zero: u32,
    incremented: u32,
    decremented: u32,
    widened: u32,
    exact: u32,
}
type DirectedFMul = fn(
    &mut Vec<u32>,
    &mut u32,
    DirectedProduct,
    (u32, u32, u32),
    &HashMap<u32, u32>,
    &mut Vec<u32>,
    &mut Vec<u32>,
) -> u32;
/// By DirectedRounding (NegativeInfinity, PositiveInfinity, Zero).
const DIRECTED_FMUL: [DirectedFMul; 3] = [
    |w, id, p, (uint, float, boolt), _, globals, annotations| {
        let step = alu(
            w,
            id,
            Op::Select,
            uint,
            &[p.negative, p.incremented, p.decremented],
        );
        let zero_step = late_constant(globals, id, uint, 0x8000_0001);
        directed_fmul_step(
            w,
            id,
            p,
            (uint, float, boolt),
            (Op::FOrdGreaterThan, step, zero_step),
            annotations,
        )
    },
    |w, id, p, (uint, float, boolt), constants, _, annotations| {
        let step = alu(
            w,
            id,
            Op::Select,
            uint,
            &[p.negative, p.decremented, p.incremented],
        );
        directed_fmul_step(
            w,
            id,
            p,
            (uint, float, boolt),
            (Op::FOrdLessThan, step, constants[&1]),
            annotations,
        )
    },
    |w, id, p, (uint, float, boolt), _, _, annotations| {
        // Toward zero always reduces magnitude. Choose the comparison
        // from the exact product sign, even when RTE produced signed zero.
        let below = alu(w, id, Op::FOrdLessThan, boolt, &[p.widened, p.exact]);
        let above = alu(w, id, Op::FOrdGreaterThan, boolt, &[p.widened, p.exact]);
        inst(annotations, Op::Decorate, &[below, 42]);
        inst(annotations, Op::Decorate, &[above, 42]);
        let adjust = alu(w, id, Op::Select, boolt, &[p.negative, below, above]);
        let result = alu(w, id, Op::Select, uint, &[adjust, p.decremented, p.bits]);
        alu(w, id, Op::Bitcast, float, &[result])
    },
];
/// RM and RP step away from the nearest product when it lies on the wrong
/// side of the exact one; a zero product steps to the mode's smallest value.
fn directed_fmul_step(
    w: &mut Vec<u32>,
    id: &mut u32,
    p: DirectedProduct,
    (uint, float, boolt): (u32, u32, u32),
    (opcode, step, zero_step): (Op, u32, u32),
    annotations: &mut Vec<u32>,
) -> u32 {
    let step = alu(w, id, Op::Select, uint, &[p.zero, zero_step, step]);
    let adjust = alu(w, id, opcode, boolt, &[p.widened, p.exact]);
    inst(annotations, Op::Decorate, &[adjust, 42]);
    let result = alu(w, id, Op::Select, uint, &[adjust, step, p.bits]);
    alu(w, id, Op::Bitcast, float, &[result])
}

fn video_extend(
    w: &mut Vec<u32>,
    id: &mut u32,
    value: u32,
    select: u8,
    uint: u32,
    constants: &HashMap<u32, u32>,
) -> u32 {
    // Ryujinx InstEmitAluHelper.cs:75-91. One bitfield extract replaces
    // its shift + mask/sign-extension chain.
    // The decoder established select & 7 in 0..=6.
    VIDEO_SELECT[usize::from(select & 7)](
        w,
        id,
        value,
        EXTRACT_OP[usize::from((select >> 3) & 1)],
        uint,
        constants,
    )
}
fn video_extend_selection(
    w: &mut Vec<u32>,
    id: &mut u32,
    value: u32,
    selection: VideoSelection,
    uint: u32,
    constants: &HashMap<u32, u32>,
) -> u32 {
    let extract = match selection.signedness {
        Signedness::Unsigned => Op::BitFieldUExtract,
        Signedness::Signed => Op::BitFieldSExtract,
    };
    match selection.element {
        VideoElement::Byte0 => video_field::<0, 8>(w, id, value, extract, uint, constants),
        VideoElement::Byte1 => video_field::<8, 8>(w, id, value, extract, uint, constants),
        VideoElement::Byte2 => video_field::<16, 8>(w, id, value, extract, uint, constants),
        VideoElement::Byte3 => video_field::<24, 8>(w, id, value, extract, uint, constants),
        VideoElement::Half0 => video_field::<0, 16>(w, id, value, extract, uint, constants),
        VideoElement::Half1 => video_field::<16, 16>(w, id, value, extract, uint, constants),
        VideoElement::Whole => value,
    }
}
/// By signedness bit: unsigned or signed bitfield extract.
const EXTRACT_OP: [Op; 2] = [Op::BitFieldUExtract, Op::BitFieldSExtract];
type VideoSelect = fn(&mut Vec<u32>, &mut u32, u32, Op, u32, &HashMap<u32, u32>) -> u32;
/// By select: bytes 0..3, halves 0..1, then the whole word.
const VIDEO_SELECT: [VideoSelect; 7] = [
    video_field::<0, 8>,
    video_field::<8, 8>,
    video_field::<16, 8>,
    video_field::<24, 8>,
    video_field::<0, 16>,
    video_field::<16, 16>,
    |_, _, value, _, _, _| value,
];
fn video_field<const OFFSET: u32, const WIDTH: u32>(
    w: &mut Vec<u32>,
    id: &mut u32,
    value: u32,
    extract: Op,
    uint: u32,
    constants: &HashMap<u32, u32>,
) -> u32 {
    alu(
        w,
        id,
        extract,
        uint,
        &[value, constants[&OFFSET], constants[&WIDTH]],
    )
}

fn video_min_max(
    w: &mut Vec<u32>,
    id: &mut u32,
    a: u32,
    b: u32,
    max: bool,
    signed: bool,
    (uint, boolt): (u32, u32),
) -> u32 {
    // SPIR-V 1.6 section 3.3.15: SLessThan 177 / ULessThan 176.
    // Compare + select avoids introducing a duplicate signed integer type;
    // signed comparison accepts integer operands of either Signedness.
    let less = alu(
        w,
        id,
        [Op::ULessThan, Op::SLessThan][usize::from(signed)],
        boolt,
        &[a, b],
    );
    let operands = [[less, a, b], [less, b, a]][usize::from(max)];
    alu(w, id, Op::Select, uint, &operands)
}

#[allow(clippy::too_many_arguments)]
fn emit_video_min_max(
    w: &mut Vec<u32>,
    id: &mut u32,
    [a, b, c]: [u32; 3],
    select: [u8; 2],
    max: bool,
    signed_output: bool,
    saturate: bool,
    op: u8,
    types: (u32, u32),
    constants: &HashMap<u32, u32>,
) -> u32 {
    let (uint, _) = types;
    let a = video_extend(w, id, a, select[0], uint, constants);
    let b = video_extend(w, id, b, select[1], uint, constants);
    let signed_a = usize::from((select[0] >> 3) & 1);
    let signed_b = usize::from((select[1] >> 3) & 1);
    // Equal signedness compares in it; mixed signedness compares signed for
    // min and unsigned for max.
    let result_signed = [[false, !max], [!max, true]][signed_a][signed_b];
    let value = video_min_max(w, id, a, b, max, result_signed, types);
    // Match the shipping translator's mixed-sign path exactly, including
    // its selection of the unsigned source when its top bit is set
    // (Ryujinx InstEmitVideoMinMax.cs:40-61).
    let value = VIDEO_MIXED_SIGN[signed_a][signed_b](w, id, value, [a, b], types, constants);
    // Saturate the FIRST min/max, then apply C (upstream :82-115).
    let value = VIDEO_SATURATE[usize::from(saturate)][usize::from(signed_output)]
        [usize::from(result_signed)](w, id, value, types, constants);
    // The three-bit operation field.
    VIDEO_POST[usize::from(op)](w, id, value, c, signed_output, types, constants)
}
type VideoMixedSign =
    fn(&mut Vec<u32>, &mut u32, u32, [u32; 2], (u32, u32), &HashMap<u32, u32>) -> u32;
/// By [signed A][signed B]: the unsigned source wins when its top bit is set.
const VIDEO_MIXED_SIGN: [[VideoMixedSign; 2]; 2] = [
    [
        |_, _, value, _, _, _| value,
        |w, id, value, [a, _], types, constants| {
            video_unsigned_high(w, id, value, a, types, constants)
        },
    ],
    [
        |w, id, value, [_, b], types, constants| {
            video_unsigned_high(w, id, value, b, types, constants)
        },
        |_, _, value, _, _, _| value,
    ],
];
fn video_unsigned_high(
    w: &mut Vec<u32>,
    id: &mut u32,
    value: u32,
    unsigned: u32,
    (uint, boolt): (u32, u32),
    constants: &HashMap<u32, u32>,
) -> u32 {
    let high = alu(w, id, Op::SLessThan, boolt, &[unsigned, constants[&0]]);
    alu(w, id, Op::Select, uint, &[high, unsigned, value])
}
type VideoClamp = fn(&mut Vec<u32>, &mut u32, u32, (u32, u32), &HashMap<u32, u32>) -> u32;
/// By [saturate][signed output][signed result]: a signed output clamps an
/// unsigned result to INT_MAX, an unsigned output a signed result to zero.
const VIDEO_SATURATE: [[[VideoClamp; 2]; 2]; 2] = [
    [[|_, _, value, _, _| value; 2]; 2],
    [
        [
            |_, _, value, _, _| value,
            |w, id, value, types, constants| {
                video_min_max(w, id, value, constants[&0], true, true, types)
            },
        ],
        [
            |w, id, value, types, constants| {
                video_min_max(w, id, value, constants[&0x7fff_ffff], false, false, types)
            },
            |_, _, value, _, _| value,
        ],
    ],
];
type VideoPost = fn(&mut Vec<u32>, &mut u32, u32, u32, bool, (u32, u32), &HashMap<u32, u32>) -> u32;
/// By operation: merge into a half or byte of C, add C, min or max with C,
/// or none (envytools gm107.c:1560 preserves the min/max).
const VIDEO_POST: [VideoPost; 8] = [
    video_merge::<16, 16>,
    video_merge::<0, 16>,
    video_merge::<0, 8>,
    video_merge::<16, 8>,
    |w, id, value, c, _, (uint, _), _| alu(w, id, Op::IAdd, uint, &[value, c]),
    |w, id, value, c, signed_output, types, _| {
        video_min_max(w, id, value, c, false, signed_output, types)
    },
    |w, id, value, c, signed_output, types, _| {
        video_min_max(w, id, value, c, true, signed_output, types)
    },
    |_, _, value, _, _, _, _| value,
];
fn video_merge<const OFFSET: u32, const WIDTH: u32>(
    w: &mut Vec<u32>,
    id: &mut u32,
    value: u32,
    c: u32,
    _: bool,
    (uint, _): (u32, u32),
    constants: &HashMap<u32, u32>,
) -> u32 {
    alu(
        w,
        id,
        Op::BitFieldInsert,
        uint,
        &[c, value, constants[&OFFSET], constants[&WIDTH]],
    )
}

#[allow(clippy::too_many_arguments)]
fn emit_shuffle(
    w: &mut Vec<u32>,
    id: &mut u32,
    value: u32,
    index: u32,
    control: u32,
    physical_lane: u32,
    mode: ShuffleMode,
    (uint, boolt, subgroup): (u32, u32, u32),
    constants: &HashMap<u32, u32>,
) -> (u32, u32) {
    // Ryujinx Translation/HelperFunctionManager.cs:245-361. Inline its helper
    // to avoid a function call and select a safe source BEFORE the collective
    // so an invalid lane never feeds SPIR-V an out-of-range invocation.
    let clamp = alu(w, id, Op::BitwiseAnd, uint, &[control, constants[&31]]);
    let segment = alu(
        w,
        id,
        Op::ShiftRightLogical,
        uint,
        &[control, constants[&8]],
    );
    let segment = alu(w, id, Op::BitwiseAnd, uint, &[segment, constants[&31]]);
    let lane = alu(
        w,
        id,
        Op::BitwiseAnd,
        uint,
        &[physical_lane, constants[&31]],
    );
    let min_lane = alu(w, id, Op::BitwiseAnd, uint, &[lane, segment]);
    let not_segment = alu(w, id, Op::Not, uint, &[segment]);
    let limit = alu(w, id, Op::BitwiseAnd, uint, &[clamp, not_segment]);
    let max_lane = alu(w, id, Op::BitwiseOr, uint, &[limit, min_lane]);
    let lanes = ShuffleLanes {
        lane,
        index,
        min_lane,
        max_lane,
        not_segment,
    };
    let (target, valid) = SHUFFLE_TARGET[mode as usize](w, id, lanes, (uint, boolt));
    let safe_lane = alu(w, id, Op::Select, uint, &[valid, target, lane]);
    // Keep independent Maxwell warps when the host subgroup has >32 lanes.
    let warp_base = alu(
        w,
        id,
        Op::BitwiseAnd,
        uint,
        &[physical_lane, constants[&!31]],
    );
    let target = alu(w, id, Op::BitwiseOr, uint, &[warp_base, safe_lane]);
    let shuffled = alu(
        w,
        id,
        Op::GroupNonUniformShuffle,
        uint,
        &[subgroup, value, target],
    );
    (
        alu(w, id, Op::Select, uint, &[valid, shuffled, value]),
        valid,
    )
}
/// The lanes a shuffle's target and its validity derive from.
#[derive(Clone, Copy)]
struct ShuffleLanes {
    lane: u32,
    index: u32,
    min_lane: u32,
    max_lane: u32,
    not_segment: u32,
}
type ShuffleTarget = fn(&mut Vec<u32>, &mut u32, ShuffleLanes, (u32, u32)) -> (u32, u32);
/// By ShuffleMode (Idx, Up, Down, Bfly): the target lane, then whether it
/// stays in the segment (UP reads below, the others above).
const SHUFFLE_TARGET: [ShuffleTarget; 4] = [
    |w, id, l, (uint, boolt)| {
        let index = alu(w, id, Op::BitwiseAnd, uint, &[l.index, l.not_segment]);
        let target = alu(w, id, Op::BitwiseOr, uint, &[index, l.min_lane]);
        (target, shuffle_below_max(w, id, target, l, boolt))
    },
    |w, id, l, (uint, boolt)| {
        let target = alu(w, id, Op::ISub, uint, &[l.lane, l.index]);
        let valid = alu(w, id, Op::SGreaterThanEqual, boolt, &[target, l.min_lane]);
        (target, valid)
    },
    |w, id, l, (uint, boolt)| {
        let target = alu(w, id, Op::IAdd, uint, &[l.lane, l.index]);
        (target, shuffle_below_max(w, id, target, l, boolt))
    },
    |w, id, l, (uint, boolt)| {
        let target = alu(w, id, Op::BitwiseXor, uint, &[l.lane, l.index]);
        (target, shuffle_below_max(w, id, target, l, boolt))
    },
];
fn shuffle_below_max(
    w: &mut Vec<u32>,
    id: &mut u32,
    target: u32,
    l: ShuffleLanes,
    boolt: u32,
) -> u32 {
    alu(w, id, Op::ULessThanEqual, boolt, &[target, l.max_lane])
}
/// SPIR-V shifts are undefined for amounts >= 32, so shift by the amount
/// masked to 0..31 and patch the >= 32 case: wrap keeps the masked shift;
/// otherwise the result is 0, or x >> 31 for a signed right shift (Mesa
/// OpShl/OpShr fold).
fn shift(
    w: &mut Vec<u32>,
    id: &mut u32,
    (opcode, signed, wrap): (Op, bool, bool),
    a: u32,
    b: u32,
    (uint, boolt, zero): (u32, u32, u32),
    constants: &HashMap<u32, u32>,
) -> u32 {
    let masked = alu(w, id, Op::BitwiseAnd, uint, &[b, constants[&31]]);
    let shifted = alu(w, id, opcode, uint, &[a, masked]);
    SHIFT_RANGE[usize::from(wrap)](w, id, signed, a, b, shifted, (uint, boolt, zero), constants)
}
/// By signed: the right shift that fills with zero, or with the sign bit.
const SHIFT_RIGHT: [Op; 2] = [Op::ShiftRightLogical, Op::ShiftRightArithmetic];
type ShiftRange =
    fn(&mut Vec<u32>, &mut u32, bool, u32, u32, u32, (u32, u32, u32), &HashMap<u32, u32>) -> u32;
/// By wrap: shifts of 32 or more give the overflow value, or the masked shift.
const SHIFT_RANGE: [ShiftRange; 2] = [
    |w, id, signed, a, b, shifted, (uint, boolt, zero), constants| {
        let large = alu(w, id, Op::UGreaterThanEqual, boolt, &[b, constants[&32]]);
        let overflow = alu_when(
            w,
            id,
            signed,
            Op::ShiftRightArithmetic,
            uint,
            &[a, constants[&31]],
            zero,
        );
        alu(w, id, Op::Select, uint, &[large, overflow, shifted])
    },
    |_, _, _, _, _, shifted, _, _| shifted,
];

/// By SetOp (And, Or, Xor).
const SET_OP_CODE: [Op; 3] = [Op::LogicalAnd, Op::LogicalOr, Op::LogicalNotEqual];

type ConditionComplement = fn(&mut Vec<u32>, &mut u32, Op, u32, u32, u32, u32) -> u32;
/// By SetOp: CSETP's second output, !cc combined with the accumulator.
/// Ryujinx InstEmitConditionCode.cs:40-44 computes !cc and two logical ops.
/// For XOR, (!cc XOR acc) = !(cc XOR acc): one XOR and one NOT replace its
/// three Boolean operations.
const CONDITION_COMPLEMENT: [ConditionComplement; 3] = [
    |w, id, op, boolt, cc, acc, _| {
        let inverse = alu(w, id, Op::LogicalNot, boolt, &[cc]);
        alu(w, id, op, boolt, &[inverse, acc])
    },
    |w, id, op, boolt, cc, acc, _| {
        let inverse = alu(w, id, Op::LogicalNot, boolt, &[cc]);
        alu(w, id, op, boolt, &[inverse, acc])
    },
    |w, id, _, boolt, _, _, first| alu(w, id, Op::LogicalNot, boolt, &[first]),
];

/// Writes each predicate output under the guard snapshot.
fn write_predicate_outputs(
    w: &mut Vec<u32>,
    id: &mut u32,
    ps: &[u32],
    guard: Option<u32>,
    outputs: impl IntoIterator<Item = (u8, u32)>,
    boolt: u32,
) {
    for (destination, value) in outputs {
        PREDICATE_OUTPUT[usize::from(destination)](
            w,
            id,
            ps[usize::from(destination)],
            guard,
            value,
            boolt,
        );
    }
}
type PredicateWrite = fn(&mut Vec<u32>, &mut u32, u32, Option<u32>, u32, u32);
/// A guarded write keeps the old value where the guard is false.
const STORE_PREDICATE: PredicateWrite = |w, id, pointer, guard, value, boolt| {
    let value = guard.into_iter().fold(value, |value, active| {
        let old = alu(w, id, Op::Load, boolt, &[pointer]);
        alu(w, id, Op::Select, boolt, &[active, value, old])
    });
    inst(w, Op::Store, &[pointer, value]);
};
/// By destination predicate: P0-P6 store; PT discards.
const PREDICATE_OUTPUT: [PredicateWrite; 8] = [
    STORE_PREDICATE,
    STORE_PREDICATE,
    STORE_PREDICATE,
    STORE_PREDICATE,
    STORE_PREDICATE,
    STORE_PREDICATE,
    STORE_PREDICATE,
    |_, _, _, _, _, _| {},
];

type LopPredicate = fn(&mut Vec<u32>, &mut u32, u32, (u32, u32, u32, u32)) -> u32;
/// By PredicateOp (F, T, Z, NZ): LOP's predicate output (Ryujinx
/// InstEmitIntegerLogical.cs:145-160).
const LOP_PREDICATE: [LopPredicate; 4] = [
    |_, _, _, (_, _, _, no)| no,
    |_, _, _, (_, _, yes, _)| yes,
    |w, id, v, (boolt, zero, _, _)| {
        let nonzero = alu(w, id, Op::INotEqual, boolt, &[v, zero]);
        alu(w, id, Op::LogicalNot, boolt, &[nonzero])
    },
    |w, id, v, (boolt, zero, _, _)| alu(w, id, Op::INotEqual, boolt, &[v, zero]),
];

fn bad(x: &Instruction) -> Stopped {
    Stopped::Unbuilt(Box::new(*x))
}
/// SPIR-V opcode for an integer compare (spirv.h), by signedness.
fn int_compare_op(rel: Relation, signedness: Signedness) -> Op {
    INT_COMPARE[rel as usize][signedness as usize]
}
/// SPIR-V opcode for an ordered float compare (spirv.h).
fn ordered_compare_op(rel: Relation) -> Op {
    ORDERED_COMPARE[rel as usize]
}
/// SPIR-V opcode for an unordered float compare (spirv.h).
fn unordered_compare_op(rel: Relation) -> Op {
    UNORDERED_COMPARE[rel as usize]
}
/// By [Relation (Lt, Eq, Le, Gt, Ne, Ge)][Signedness (Unsigned, Signed)].
const INT_COMPARE: [[Op; 2]; 6] = [
    [Op::ULessThan, Op::SLessThan],
    [Op::IEqual, Op::IEqual],
    [Op::ULessThanEqual, Op::SLessThanEqual],
    [Op::UGreaterThan, Op::SGreaterThan],
    [Op::INotEqual, Op::INotEqual],
    [Op::UGreaterThanEqual, Op::SGreaterThanEqual],
];
/// By Relation (Lt, Eq, Le, Gt, Ne, Ge).
const ORDERED_COMPARE: [Op; 6] = [
    Op::FOrdLessThan,
    Op::FOrdEqual,
    Op::FOrdLessThanEqual,
    Op::FOrdGreaterThan,
    Op::FOrdNotEqual,
    Op::FOrdGreaterThanEqual,
];
/// By Relation (Lt, Eq, Le, Gt, Ne, Ge).
const UNORDERED_COMPARE: [Op; 6] = [
    Op::FUnordLessThan,
    Op::FUnordEqual,
    Op::FUnordLessThanEqual,
    Op::FUnordGreaterThan,
    Op::FUnordNotEqual,
    Op::FUnordGreaterThanEqual,
];

pub fn emit(xs: &[Instruction]) -> Result<SpirvModule, Stopped> {
    emit_with_header(xs, None)
}

/// Emit and optimise a module with the selected stage interface.
pub fn emit_with_header(
    xs: &[Instruction],
    header: Option<&ShaderHeader>,
) -> Result<SpirvModule, Stopped> {
    emit_with_options(xs, header, crate::TranslationOptions::default())
}

pub fn emit_with_options(
    xs: &[Instruction],
    header: Option<&ShaderHeader>,
    options: crate::TranslationOptions,
) -> Result<SpirvModule, Stopped> {
    emit_with_options_and_graphics_state(
        xs,
        header,
        options,
        &crate::interface::GraphicsState::default(),
    )
}

/// Emit with driver-supplied geometry input topology or tessellation state.
pub fn emit_with_graphics_state(
    xs: &[Instruction],
    header: Option<&ShaderHeader>,
    state: &crate::interface::GraphicsState,
) -> Result<SpirvModule, Stopped> {
    emit_with_options_and_graphics_state(xs, header, crate::TranslationOptions::default(), state)
}

pub fn emit_with_options_and_graphics_state(
    xs: &[Instruction],
    header: Option<&ShaderHeader>,
    options: crate::TranslationOptions,
    state: &crate::interface::GraphicsState,
) -> Result<SpirvModule, Stopped> {
    emit_with_scratch(
        xs,
        crate::structure::GraphStore::default().build(xs),
        match header {
            Some(header) => Interface::Header(header),
            None => Interface::Headerless,
        },
        options,
        state,
        &mut Scratch::default(),
        &mut crate::optimizer::Scratch::default(),
    )
}

pub(crate) fn emit_with_scratch(
    xs: &[Instruction],
    graph: Result<&crate::structure::ControlFlowGraph, Stopped>,
    interface: Interface<'_>,
    options: crate::TranslationOptions,
    state: &crate::interface::GraphicsState,
    scratch: &mut Scratch,
    optimizer: &mut crate::optimizer::Scratch,
) -> Result<SpirvModule, Stopped> {
    let words = optimizer.take_arena();
    let mut module = emit_unoptimized_into(xs, graph, interface, options, state, scratch, words)?;
    crate::optimizer::optimize_with_scratch(&mut module.words, optimizer);
    Ok(module)
}

/// Emit the original module for differential checks and offline benchmarks.
pub fn emit_unoptimized_with_header(
    xs: &[Instruction],
    header: Option<&ShaderHeader>,
) -> Result<SpirvModule, Stopped> {
    emit_unoptimized_with_graphics_state(xs, header, &crate::interface::GraphicsState::default())
}

/// Emit with explicit host capabilities, retaining the module before optimization.
pub fn emit_unoptimized_with_options(
    xs: &[Instruction],
    header: Option<&ShaderHeader>,
    options: crate::TranslationOptions,
) -> Result<SpirvModule, Stopped> {
    emit_unoptimized_with_options_and_graphics_state(
        xs,
        header,
        options,
        &crate::interface::GraphicsState::default(),
    )
}

pub fn emit_unoptimized_with_graphics_state(
    xs: &[Instruction],
    header: Option<&ShaderHeader>,
    state: &crate::interface::GraphicsState,
) -> Result<SpirvModule, Stopped> {
    emit_unoptimized_with_options_and_graphics_state(
        xs,
        header,
        crate::TranslationOptions::default(),
        state,
    )
}

// Buffers are owned by one translator. Clear contents between programs and
// preserve capacity. Ryujinx SpirvGenerator.cs:19-28,38-42 pools emission storage.
#[derive(Default)]
pub(crate) struct Scratch {
    annotations: Vec<u32>,
    late_globals: Vec<u32>,
    constants: HashMap<u32, u32>,
    float_constants: HashMap<u32, u32>,
    cb_types: HashMap<u8, (u32, u32, u32, u32)>,
    cb_vars: HashMap<u8, u32>,
    cb_bindings: Vec<u8>,
    labels: HashMap<u32, u32>,
    block_starts: HashSet<u32>,
    blocks: Vec<u32>,
    case_of: HashMap<u32, u32>,
    switch: Vec<u32>,
    reaching: ReachingDefinitions,
    reach_scratch: crate::reaching::Scratch,
}

fn emit_unoptimized_with_options_and_graphics_state(
    xs: &[Instruction],
    header: Option<&ShaderHeader>,
    options: crate::TranslationOptions,
    state: &crate::interface::GraphicsState,
) -> Result<SpirvModule, Stopped> {
    emit_unoptimized_into(
        xs,
        crate::structure::GraphStore::default().build(xs),
        match header {
            Some(header) => Interface::Header(header),
            None => Interface::Headerless,
        },
        options,
        state,
        &mut Scratch::default(),
        Vec::new(),
    )
}

/// By whether the program is subgroup-sensitive: per-lane ALU and memory
/// need no token scheduling; only a collective observes the region's
/// active set (Ryujinx InstEmitWarp.cs:24-45,48-101).
type AnalyzeRegions = fn(
    &[Instruction],
    &crate::structure::ControlFlowGraph,
) -> Result<crate::reconvergence::Regions, Stopped>;
const REGIONS: [AnalyzeRegions; 2] = [
    |_, _| Ok(crate::reconvergence::Regions::default()),
    crate::reconvergence::Regions::analyze,
];
/// TXQ has no shape field. Ryujinx InstEmitTexture.cs:1116-1119 asks
/// the driver for the sampler type. Accept explicit metadata or a typed
/// operation on the same binding, and reject conflicting evidence.
fn resolve_queries<'a>(
    xs: &'a [Instruction],
    state: &crate::interface::GraphicsState,
    resolved_queries: &'a mut Vec<Instruction>,
) -> Result<&'a [Instruction], Stopped> {
    *resolved_queries = xs
        .iter()
        .map(|x| {
            let mut result = *x;
            if let InstructionKind::Texture(ops::Texture {
                op: TexOp::Query { .. },
                bound_handle: Some(binding),
                ref mut dim,
                ref mut depth_compare,
                ref mut multisample,
                ..
            }) = result.kind
            {
                let mut shapes = xs.iter().filter_map(|decl| match decl.kind {
                    InstructionKind::Texture(ops::Texture {
                        op,
                        bound_handle: Some(b),
                        dim,
                        depth_compare,
                        multisample,
                        ..
                    }) if b == binding && !matches!(op, TexOp::Query { .. }) => {
                        Some((dim, depth_compare, multisample))
                    }
                    _ => None,
                });
                let mut supplied = state
                    .texture_bindings
                    .iter()
                    .filter(|t| t.binding == binding);
                let explicit = supplied
                    .next()
                    .map(|t| (t.dimension as u8, t.depth_compare, t.multisample));
                let shape = explicit.or_else(|| shapes.next()).ok_or_else(|| bad(x))?;
                if shapes.any(|other| other != shape)
                    || supplied
                        .any(|t| (t.dimension as u8, t.depth_compare, t.multisample) != shape)
                {
                    return Err(bad(x));
                }
                (*dim, *depth_compare, *multisample) = shape;
            }
            Ok(result)
        })
        .collect::<Result<Vec<_>, Stopped>>()?;
    Ok(resolved_queries.as_slice())
}

/// Rejects the first word the decoder could not name.
struct FirstUnknown<'a>(&'a Instruction);
impl Translation for FirstUnknown<'_> {
    type Out = Result<(), Stopped>;
    fn unknown(self) -> Self::Out {
        Err(Stopped::Unknown(Box::new(*self.0)))
    }
    fn padding(self) -> Self::Out {
        Ok(())
    }
    fn unbuilt(self) -> Self::Out {
        Err(Stopped::Unbuilt(Box::new(*self.0)))
    }
    fn unwanted(self) -> Self::Out {
        Err(Stopped::Unwanted(Box::new(*self.0)))
    }
    fn unresolved_stack(self) -> Self::Out {
        Ok(())
    }
    fn unproven_stream(self) -> Self::Out {
        Ok(())
    }
    fn known(self) -> Self::Out {
        Ok(())
    }
}

/// Rejects an unresolved stack instruction, and padding control reaches.
/// Unresolved SSY/PBK/PCNT and pops require control-stack semantics.
/// Single-target pushes affect the branch lowered by decode_stream. Mesa
/// sm50.rs:3032-3041; envytools gm107.c:1930-1933. Do not silently drop
/// these pushes.
struct Admission<'a> {
    x: &'a Instruction,
    live: bool,
}
impl Translation for Admission<'_> {
    type Out = Result<(), Stopped>;
    fn unknown(self) -> Self::Out {
        Err(Stopped::Unknown(Box::new(*self.x)))
    }
    fn padding(self) -> Self::Out {
        if self.live {
            Err(Stopped::LivePadding(Box::new(*self.x)))
        } else {
            Ok(())
        }
    }
    fn unbuilt(self) -> Self::Out {
        Err(Stopped::Unbuilt(Box::new(*self.x)))
    }
    fn unwanted(self) -> Self::Out {
        Err(Stopped::Unwanted(Box::new(*self.x)))
    }
    fn unresolved_stack(self) -> Self::Out {
        Err(Stopped::UnresolvedStack(Box::new(*self.x)))
    }
    fn unproven_stream(self) -> Self::Out {
        Err(Stopped::UnprovenStream(Box::new(*self.x)))
    }
    fn known(self) -> Self::Out {
        Ok(())
    }
}

/// Padding in the dispatch loop: every padding word left is dead, so its
/// case gets a terminator rather than a translation.
struct DeadPadding<'a> {
    w: &'a mut Vec<u32>,
    reachable: &'a mut bool,
}
impl Translation for DeadPadding<'_> {
    type Out = std::ops::ControlFlow<()>;
    fn unknown(self) -> Self::Out {
        std::ops::ControlFlow::Continue(())
    }
    fn padding(self) -> Self::Out {
        if *self.reachable {
            // This dispatch case has no path from entry. Give its label
            // a terminator rather than translating padding as an op.
            // OpUnreachable has no operands or results. The CFG has no live incoming edge.
            inst(self.w, Op::Unreachable, &[]);
            *self.reachable = false;
        }
        std::ops::ControlFlow::Break(())
    }
    fn unbuilt(self) -> Self::Out {
        std::ops::ControlFlow::Continue(())
    }
    fn unwanted(self) -> Self::Out {
        std::ops::ControlFlow::Continue(())
    }
    fn unresolved_stack(self) -> Self::Out {
        std::ops::ControlFlow::Continue(())
    }
    fn unproven_stream(self) -> Self::Out {
        std::ops::ControlFlow::Continue(())
    }
    fn known(self) -> Self::Out {
        std::ops::ControlFlow::Continue(())
    }
}

/// What the module is made with: everything the program's instructions
/// require, told by each instruction itself.
struct Required {
    demote: bool,
    subgroup: bool,
    preserve_float32: bool,
    float64: bool,
    int64: bool,
    surface_store: bool,
    surface_load: bool,
    surface_extended: bool,
    signed_storage: bool,
    nontemporal: bool,
    sampled_1d: bool,
    cube_array: bool,
    texture_offsets: bool,
    image_1d: bool,
    image_buffer: bool,
    image_query: bool,
    images: bool,
    shuffle: bool,
    vote: bool,
    lane_id: bool,
    output_lanes: bool,
    frag_coord: bool,
    nodep: bool,
    dynamic_input: bool,
    half: bool,
    half_uvec2: bool,
    ldc_indexed: bool,
    shared: bool,
    local: bool,
    register_brx: bool,
    y_direction: bool,
    helper: bool,
    invocation_info: bool,
    local_invocation_id: bool,
    workgroup_id: bool,
    /// The lane-mask system registers read, by index.
    lane_masks: std::collections::BTreeSet<u8>,
}

impl Required {
    /// What `xs` requires of the module.
    fn of(xs: &[Instruction]) -> Self {
        let mut required = Self {
            demote: false,
            subgroup: false,
            preserve_float32: false,
            float64: false,
            int64: false,
            surface_store: false,
            surface_load: false,
            surface_extended: false,
            signed_storage: false,
            nontemporal: false,
            sampled_1d: false,
            cube_array: false,
            texture_offsets: false,
            image_1d: false,
            image_buffer: false,
            image_query: false,
            images: false,
            shuffle: false,
            vote: false,
            lane_id: false,
            output_lanes: false,
            frag_coord: false,
            nodep: false,
            dynamic_input: false,
            half: false,
            half_uvec2: false,
            ldc_indexed: false,
            shared: false,
            local: false,
            register_brx: false,
            y_direction: false,
            helper: false,
            invocation_info: false,
            local_invocation_id: false,
            workgroup_id: false,
            lane_masks: std::collections::BTreeSet::new(),
        };
        xs.iter().for_each(|x| x.kind.require(&mut required));
        required
    }
}

macro_rules! required {
    ($($need:ident)*) => {
        impl Requirements for Required {
            $(fn $need(&mut self) {
                self.$need = true;
            })*
            fn lane_mask(&mut self, index: u8) {
                self.lane_masks.insert(index);
            }
        }
    };
}
required! {
    demote subgroup preserve_float32 float64 int64 surface_store surface_load
    surface_extended signed_storage nontemporal sampled_1d cube_array
    texture_offsets image_1d image_buffer image_query images shuffle vote
    lane_id output_lanes frag_coord nodep dynamic_input half half_uvec2
    ldc_indexed shared local register_brx y_direction helper invocation_info
    local_invocation_id workgroup_id
}

type SwizzleLaneTail =
    fn(&mut Vec<u32>, &mut u32, &mut Vec<u32>, (u32, u32), [u32; 3], u32, u32) -> u32;
/// By SWZADD source: A keeps A*0 for mode 3 (NaN and signed zero make
/// deleting it incorrect); B is the selected value as it is.
const SWIZZLE_LANE_TAIL: [SwizzleLaneTail; 2] = [
    |w, id, annotations, (boolt, float), [op, three, zero], value, selected| {
        let omit = alu(w, id, Op::IEqual, boolt, &[op, three]);
        let fzero = alu(w, id, Op::Bitcast, float, &[zero]);
        let product = alu(w, id, Op::FMul, float, &[value, fzero]);
        inst(annotations, Op::Decorate, &[product, 42]);
        alu(w, id, Op::Select, float, &[omit, product, selected])
    },
    |_, _, _, _, _, _, selected| selected,
];

/// XMAD half selection by high: the low half shifted to the top, or the
/// high half where it is (a shift by zero keeps the emitted sequence).
const XMAD_HALF: [(Op, u32); 2] = [(Op::ShiftLeftLogical, 16), (Op::ShiftRightLogical, 0)];
/// XMAD widening by signed.
const XMAD_WIDEN: [Op; 2] = [Op::ShiftRightLogical, Op::ShiftRightArithmetic];

/// `op` of `operands` when `apply`, otherwise `value` as it is.
fn alu_when(
    w: &mut Vec<u32>,
    id: &mut u32,
    apply: bool,
    op: Op,
    ty: u32,
    operands: &[u32],
    value: u32,
) -> u32 {
    ALU_WHEN[usize::from(apply)](w, id, op, ty, operands, value)
}
/// ISCADD by (negate B, negate A without .CC): the add, or which side subtracts.
const SHIFT_ADD: [(Op, usize); 4] = [(Op::IAdd, 0), (Op::ISub, 1), (Op::ISub, 0), (Op::ISub, 0)];
/// LDC by width: the narrow field's (bits, signed), and the extra words read.
const LDC_SHAPE: [(u32, bool, usize); 6] = [
    (8, false, 0),
    (8, true, 0),
    (16, false, 0),
    (16, true, 0),
    (32, false, 0),
    (32, false, 1),
];
type XmadMerge = fn(&mut Vec<u32>, &mut u32, u32, u32, u32, [u32; 2]) -> u32;
/// XMAD by .MRG: the sum as it is, or its low half under B's low half.
const XMAD_MERGE: [XmadMerge; 2] = [
    |_, _, _, result, _, _| result,
    |w, id, uint, result, b_full, [low_mask, sixteen]| {
        let low = alu(w, id, Op::BitwiseAnd, uint, &[result, low_mask]);
        let high = alu(w, id, Op::ShiftLeftLogical, uint, &[b_full, sixteen]);
        alu(w, id, Op::BitwiseOr, uint, &[low, high])
    },
];
type AluWhen = fn(&mut Vec<u32>, &mut u32, Op, u32, &[u32], u32) -> u32;
/// By apply: the value as it is, or the operation's result.
const ALU_WHEN: [AluWhen; 2] = [
    |_, _, _, _, _, value| value,
    |w, id, op, ty, operands, _| alu(w, id, op, ty, operands),
];

/// Emits `op` once per `times`: once for a need that is met, else never.
fn inst_times(w: &mut Vec<u32>, op: Op, operands: &[u32], times: usize) {
    std::iter::repeat_n(operands, times).for_each(|operands| inst(w, op, operands));
}

/// A SPIR-V literal string: little-endian bytes, nul-terminated, padded.
const fn literal<const N: usize>(bytes: &[u8]) -> [u32; N] {
    let mut words = [0u32; N];
    let mut at = 0;
    while at < bytes.len() {
        words[at / 4] |= (bytes[at] as u32) << (8 * (at % 4));
        at += 1;
    }
    words
}
const FLOAT_CONTROLS: [u32; 6] = literal(b"SPV_KHR_float_controls\0");
const DEMOTE_TO_HELPER: [u32; 9] = literal(b"SPV_EXT_demote_to_helper_invocation\0");
const MAXIMAL_RECONVERGENCE: [u32; 8] = literal(b"SPV_KHR_maximal_reconvergence\0");
const DESCRIPTOR_INDEXING: [u32; 7] = literal(b"SPV_EXT_descriptor_indexing\0");

/// Mesa nak_nir_lower_vtg_io.c / Ryujinx Decoders/Decoder.cs:501-517:
/// which instructions a walk of the CFG from entry reaches.
fn live_instructions(len: usize, cfg: &crate::structure::ControlFlowGraph) -> Vec<bool> {
    let mut live = vec![false; len];
    let mut visited = vec![false; cfg.blocks.len()];
    let mut pending = (0..cfg.blocks.len().min(1)).collect::<Vec<usize>>();
    while let Some(block) = pending.pop() {
        if std::mem::replace(&mut visited[block], true) {
            continue;
        }
        live[cfg.blocks[block].instructions.clone()].fill(true);
        pending.extend(cfg.successors(&cfg.blocks[block]));
    }
    live
}

fn emit_unoptimized_into(
    xs: &[Instruction],
    graph: Result<&crate::structure::ControlFlowGraph, Stopped>,
    interface: Interface<'_>,
    options: crate::TranslationOptions,
    state: &crate::interface::GraphicsState,
    scratch: &mut Scratch,
    w: Vec<u32>,
) -> Result<SpirvModule, Stopped> {
    interface.route(Emission {
        xs,
        graph,
        options,
        state,
        scratch,
        w,
    })
}

/// A program's emission, waiting for its stage.
struct Emission<'a, 'g> {
    xs: &'a [Instruction],
    graph: Result<&'g crate::structure::ControlFlowGraph, Stopped>,
    options: crate::TranslationOptions,
    state: &'a crate::interface::GraphicsState<'a>,
    scratch: &'a mut Scratch,
    w: Vec<u32>,
}

impl StagePath for Emission<'_, '_> {
    type Output = Result<SpirvModule, Stopped>;
    fn follow<S: ShaderStage>(self, stage: S) -> Self::Output {
        emit_staged(
            stage,
            self.xs,
            self.graph,
            self.options,
            self.state,
            self.scratch,
            self.w,
        )
    }
}

fn emit_staged<S: ShaderStage>(
    track: S,
    xs: &[Instruction],
    graph: Result<&crate::structure::ControlFlowGraph, Stopped>,
    options: crate::TranslationOptions,
    state: &crate::interface::GraphicsState,
    scratch: &mut Scratch,
    mut w: Vec<u32>,
) -> Result<SpirvModule, Stopped> {
    // A cbuf PBK target is runtime data. It is inert when no BRK can consume
    // it, but it must not let stack resolution bind a later BRK to an older
    // immediate PBK. Conservatively reject any program containing both.
    for push in xs.iter().filter(|x| {
        matches!(
            x.kind,
            InstructionKind::CbufPushTarget(ops::CbufPushTarget {
                stack: crate::ir::TargetStack::Break,
                ..
            })
        )
    }) {
        let InstructionKind::CbufPushTarget(ops::CbufPushTarget { stack, .. }) = push.kind else {
            unreachable!()
        };
        if xs.iter().any(|x| {
            matches!(x.kind, InstructionKind::PopJump(ops::PopJump { stack: candidate, .. }) if candidate == stack)
                || matches!(x.kind, InstructionKind::Reconverge(ops::Reconverge { stack: candidate, .. }) if candidate == stack)
        }) {
            return Err(Stopped::UnresolvedStack(Box::new(*push)));
        }
    }
    // Unknown opcodes prevent a complete stack proof (Ryujinx
    // Decoders/Decoder.cs:800-850 needs every push/pop and successor).
    // Report the opcode first, rather than hiding it behind an unresolved SYNC.
    xs.iter()
        .try_for_each(|x| x.kind.translation(FirstUnknown(x)))?;
    // TXQ has no shape field. Ryujinx InstEmitTexture.cs:1116-1119 asks
    // the driver for the sampler type. Accept explicit metadata or a typed
    // operation on the same binding, and reject conflicting evidence.
    let mut resolved_queries = Vec::new();
    let xs = resolve_queries(xs, state, &mut resolved_queries)?;
    let cfg = graph?;
    let mut resolved_addresses = Vec::new();
    let stage = S::STAGE;
    let resolved = S::resolve_addresses(xs, cfg, &mut resolved_addresses);
    let required = Required::of(resolved);
    // Mesa from_nir.rs:3035-3050 lowers demote to KIL. Ryujinx
    // InstEmitFlowControl.cs:190-195 / Spirv/Instructions.cs:506-509 uses
    // ambiguous OpKill; explicit demotion preserves Mesa's helper execution.
    let uses_demote = required.demote;
    let subgroup_sensitive = required.subgroup;
    // Per-lane ALU and memory need no token scheduling overhead. Only a
    // collective observes the region's active set (Ryujinx InstEmitWarp.cs:
    // 24-45,48-101). Keep the existing dispatcher for other programs.
    let regions = REGIONS[usize::from(subgroup_sensitive)](xs, cfg)?;
    // Vulkan-Docs v1.4.351 shaders.adoc:3818-3820 requires helpers to stay
    // active for the quad lifetime with MaximallyReconvergesKHR. This does
    // not define a source taking a different branch. See
    // docs/reference-sources.md
    let requires_maximal_reconvergence =
        regions.is_enabled() || (uses_demote && subgroup_sensitive);
    let xs = resolved;
    // Zero has no Maxwell instruction entry (gm107.c:2171 is OOPS).
    // Preserve it as Unknown in decode_stream. Only a CFG proof that the
    // word cannot execute permits treating it as padding. Ryujinx
    // Decoders/Decoder.cs:501-517 distinguishes unconditional EXIT/BRA.
    let live = live_instructions(xs.len(), cfg);
    xs.iter()
        .zip(live)
        .try_for_each(|(x, live)| x.kind.translation(Admission { x, live }))?;
    let Scratch {
        annotations,
        late_globals,
        constants,
        float_constants,
        cb_types,
        cb_vars,
        cb_bindings,
        labels,
        block_starts,
        blocks,
        case_of,
        switch,
        reaching,
        reach_scratch,
    } = scratch;
    annotations.clear();
    late_globals.clear();
    constants.clear();
    float_constants.clear();
    cb_types.clear();
    cb_vars.clear();
    cb_bindings.clear();
    labels.clear();
    block_starts.clear();
    blocks.clear();
    case_of.clear();
    switch.clear();
    reaching.analyze_into(xs, cfg, reach_scratch, !state.storage_buffers.is_empty());
    let handles = texture_handles(xs, reaching)?;
    let mut texture_array_lengths = HashMap::default();
    for (index, source) in &handles {
        let HandleSource::ConstantBufferArray { bank, handle } = source else {
            continue;
        };
        let mut declarations = state
            .texture_arrays
            .iter()
            .filter(|array| array.binding == *bank && array.handle == *handle);
        let declaration = declarations.next().ok_or_else(|| bad(&xs[*index]))?;
        if declaration.length < 2 || declarations.any(|array| array.length != declaration.length) {
            return Err(bad(&xs[*index]));
        }
        texture_array_lengths.insert(*source, declaration.length);
    }
    let dynamic_texture_arrays = !texture_array_lengths.is_empty();
    let storage_plan = crate::storage_memory::plan(xs, reaching, state.storage_buffers)?;
    let physical_xs = if state.storage_buffers.is_empty() {
        xs
    } else {
        &[]
    };
    let uses_global = physical_xs.iter().any(crate::global_memory::is_global);
    w.clear();
    w.reserve(2048 + xs.len() * 32);
    w.extend_from_slice(&[spirv::MAGIC, spirv::VERSION_1_3, 0, 1, 0]);
    let mut id = 1u32;
    let mut n = || {
        let x = id;
        id += 1;
        x
    };
    let mut stage_io = track.stage_io(state, xs, &mut n)?;
    let void = n();
    let boolt = n();
    let uint = n();
    let float = n();
    let vec4 = n();
    let fnty = n();
    let pu = n();
    let pb = n();
    let po = n();
    let entry = n();
    let fun = n();
    let output = n();
    let zero = n();
    let no = n();
    let yes = n();
    let glsl_std_450 = n();
    inst(&mut w, Op::Capability, &[1]);
    // Dynamically indexed sampled-image arrays require the descriptor-indexing
    // capability pair (Ryujinx Spirv/Instructions.cs:590-603,1910-1960).
    inst_times(
        &mut w,
        Op::Capability,
        &[5301],
        usize::from(dynamic_texture_arrays),
    );
    inst_times(
        &mut w,
        Op::Capability,
        &[5307],
        usize::from(dynamic_texture_arrays),
    );
    // Mesa spirv.h:1272-1273: DemoteToHelperInvocationEXT capability.
    inst_times(
        &mut w,
        Op::Capability,
        &[5379],
        usize::from(required.demote),
    );
    // Ryujinx InstEmitFloatArithmetic.cs:404,428,473-476 and
    // InstEmitFloatComparison.cs:371 require ordinary binary32 arithmetic.
    // Mesa spirv_to_nir.c:6061-6078 lowers these controls to preservation.
    // FTZ is per instruction (Mesa sm50.rs:526,601,697), so implement it
    // explicitly inside a preserving module. Vulkan hosts must support the
    // corresponding Float32 properties.
    // Half operations use F32 broadcasts and intermediates, not only
    // unpacked F16 values. Preserve denormals and signed zero before
    // packing: Ryujinx InstEmitHelper.cs:139-184 and
    // InstEmitFloatArithmetic.cs:483-528. F2F without FTZ also preserves raw
    // subnormals (InstEmitConversion.cs:121-162).
    let preserve_float32 = required.preserve_float32;
    let float32 = usize::from(preserve_float32);
    // Mesa spirv.core.grammar.json:17530-17545: capability values.
    inst_times(&mut w, Op::Capability, &[4464], float32);
    inst_times(&mut w, Op::Capability, &[4466], float32);
    // Mesa grammar:17548-17551: exact binary32 conversion rounding.
    inst_times(&mut w, Op::Capability, &[4467], float32);
    crate::global_memory::capabilities(&mut w, physical_xs);
    // Round39..40: Mesa sm50.rs:525,695-700. Binary64 exact arithmetic
    // implements directed FADD and FMUL when host Float64 support is declared. Ryujinx
    // SpirvGenerator.cs:58-60 gates Float64; InstEmitFloatArithmetic.cs:435-480
    // supplies the round-to-nearest multiply fallback without that support.
    let fmul_lane = xs.iter().fold(FMulProductLane::Binary32, |lane, x| {
        x.kind.product_rounding().fmul_lane(lane)
    });
    let f32_exact_lane = if xs.iter().any(|x| {
        matches!(
            x.kind,
            InstructionKind::FAdd(ops::FAdd {
                round: RoundingMode::NegativeInfinity
                    | RoundingMode::PositiveInfinity
                    | RoundingMode::Zero,
                ..
            })
        )
    }) {
        FMulProductLane::Binary64
    } else {
        fmul_lane
    }
    .hosted(options.float64);
    let needs_float64 = f32_exact_lane.needs_float64(required.float64);
    // SPIR-V 3.2.1: Float64 capability10. Ryujinx
    // CodeGen/Spirv/SpirvGenerator.cs:58-60 adds this capability.
    inst_times(&mut w, Op::Capability, &[10], usize::from(needs_float64));
    let needs_int64 = required.int64;
    // Mesa src/compiler/spirv/spirv.h:1085: Int64 capability11.
    inst_times(
        &mut w,
        Op::Capability,
        &[11],
        usize::from(needs_int64 & !uses_global),
    );
    // Ryujinx SpirvGenerator.cs:82-95: Geometry/Tessellation capabilities.
    for &capability in S::CAPABILITIES {
        inst(&mut w, Op::Capability, &[capability]);
    }
    // Mesa spirv.core.grammar.json: ClipDistance capability 32; Ryujinx
    // CodeGen/Spirv/IoMap.cs:19,58 declares the BuiltIn array.
    inst_times(
        &mut w,
        Op::Capability,
        &[32],
        usize::from(S::writes_clip_distances(xs)),
    );
    if stage_io.as_ref().is_some_and(|io| io.uses_streams()) {
        // Mesa spirv.core.grammar.json:17226-17229: GeometryStreams = 54.
        inst(&mut w, Op::Capability, &[54]);
    }
    // Storage images without a declared format: StorageImageWriteWithoutFormat
    // (56) and StorageImageReadWithoutFormat (55).
    // SPIR-V StorageImageExtendedFormats: Mesa src/compiler/spirv/spirv.h:1121.
    // Sampled1D (43) for 1D textures, SampledCubeArray (45) for cube arrays
    // (Mesa set_tex_dim values 0/1 and 7).
    // ImageGatherExtended (25): non-constant Offset image operands.
    // SPIR-V Image1D capability 44 is required for t1d/a1d images
    // (spirv.h SpvCapabilityImage1D); ImageBuffer 47 for b1d images.
    [
        (required.surface_store, 56),
        (required.surface_load, 55),
        (required.surface_extended, 49),
        (required.sampled_1d, 43),
        (required.cube_array, 45),
        (required.texture_offsets, 25),
        (required.image_1d, 44),
        (required.image_buffer, 47),
    ]
    .into_iter()
    .for_each(|(required, capability)| {
        inst_times(&mut w, Op::Capability, &[capability], usize::from(required))
    });
    let image_query = required.image_query
        || xs.iter().enumerate().any(|(index, x)| match x.kind {
            InstructionKind::Texture(
                texture @ ops::Texture {
                    op: TexOp::Sample { .. },
                    offset: true,
                    ..
                },
            ) => texture_sources(&texture)
                .and_then(|sources| sources.offset)
                .is_some_and(|register| reaching.constant(index, register.0).is_none()),
            // TXD run-time offsets query the image size (and, in fragments,
            // its LOD) like TEX.
            InstructionKind::Texture(ops::Texture {
                op:
                    TexOp::Grad {
                        offset: GradOffset::Packed { word, .. },
                        ..
                    },
                dim,
                ..
            }) => matches!(
                OffsetWord::of(reaching, index, word, dim),
                OffsetWord::RunTime
            ),
            _ => false,
        });
    // Dynamic sample offsets need ImageQuerySizeLod (Mesa spirv.h:1831).
    // ImageQuery capability: Mesa spirv.h:1122. Literal offsets keep the
    // existing module byte for byte, with no unused capability.
    inst_times(&mut w, Op::Capability, &[50], usize::from(image_query));
    // SPIR-V 1.6 section 3.2.30: GroupNonUniformShuffle = 65.
    // All Maxwell modes use an absolute index after segment handling.
    inst_times(&mut w, Op::Capability, &[65], usize::from(required.shuffle));
    // GroupNonUniformBallot is capability 64 in SPIR-V; Maxwell VOTE
    // lowers to ballot words alone (emitter/control.rs).
    // Ryujinx CodeGen/Spirv/IoMap.cs:35-40 uses uvec4 mask builtins.
    // Mesa src/compiler/spirv/spirv.core.grammar.json:16221-16258,
    // 17285-17288: these require GroupNonUniformBallot (64).
    inst_times(
        &mut w,
        Op::Capability,
        &[64],
        usize::from(!required.lane_masks.is_empty() | required.vote),
    );
    // Mesa nak_nir_lower_vtg_io.c:10-49: in tessellation control an ALD.O
    // with Rb names the lane holding the control point it reads. Decided
    // once here; the read arms take the result through `control_points`.
    let reads_output_lanes = S::reads_output_lanes(&required);
    let needs_lane_id = reads_output_lanes || required.lane_id || !required.lane_masks.is_empty();
    // SPIR-V SubgroupLocalInvocationId is Maxwell LANEID's faithful
    // Vulkan equivalent (envytools/envydis/gm107.c:121; Mesa
    // src/nouveau/compiler/nak/sm50.rs:3215-3224).
    inst_times(&mut w, Op::Capability, &[61], usize::from(needs_lane_id));
    // Ordered dispatch uses subgroup UMin. Mesa spirv.h:1133,1135,2064.
    let ordered = usize::from(regions.is_enabled());
    inst_times(&mut w, Op::Capability, &[61], ordered);
    inst_times(&mut w, Op::Capability, &[63], ordered);
    // SPIR-V 1.3 needs this extension (Mesa grammar:13168-13195).
    inst_times(&mut w, Op::Extension, &FLOAT_CONTROLS, float32);
    // SPV_EXT_demote_to_helper_invocation, Extension Name; Mesa
    // spirv.h:2310-2312 defines demote5380 and current-helper5381.
    inst_times(
        &mut w,
        Op::Extension,
        &DEMOTE_TO_HELPER,
        usize::from(uses_demote),
    );
    // Nested regions can start with only part of a subgroup active.
    // SPV_KHR_maximal_reconvergence, Maximal Reconvergence: loop merges
    // reunite that entering set, including in nonuniform control flow.
    // Mesa src/compiler/spirv/spirv.h:210 defines execution mode 6023.
    inst_times(
        &mut w,
        Op::Extension,
        &MAXIMAL_RECONVERGENCE,
        usize::from(requires_maximal_reconvergence),
    );
    inst_times(
        &mut w,
        Op::Extension,
        &DESCRIPTOR_INDEXING,
        usize::from(dynamic_texture_arrays),
    );
    crate::global_memory::extensions(&mut w, physical_xs);
    inst(
        &mut w,
        Op::ExtInstImport,
        &[glsl_std_450, 0x4c534c47, 0x6474732e, 0x3035342e, 0],
    );
    // spirv.h:101,104: Logical / PhysicalStorageBuffer64, GLSL450 (:112).
    inst(
        &mut w,
        Op::MemoryModel,
        &[[0, 5348][usize::from(uses_global)], 1],
    );
    let reads_frag_coord = required.frag_coord;
    // Mesa nak_nir_lower_tex.c:454 maps NODEP to skip_helpers;
    // nir.h:2704-2708 permits skipped helper results. Ryujinx
    // InstEmitTexture.cs:28-39 emits the same sample for active lanes, and
    // CodeGen/Spirv/IoMap.cs:45 maps helper state to HelperInvocation.
    let helper_invocation = S::helper_invocation(&required, &mut n);
    let helper_undefined = S::helper_undefined(&required, &mut n);
    // By NODEP: what each texture result becomes.
    let texture_results = [
        EVERY_RESULT,
        helper_invocation.zip(helper_undefined).into_iter().fold(
            EVERY_RESULT,
            |_, (variable, undefined)| {
                TextureResults::skip_helpers(variable, undefined, uses_demote)
            },
        ),
    ];
    let input_type = n();
    let mut input_decorations = Vec::new();
    let mut input_by_location = HashMap::default();
    let mut system_inputs = HashMap::default();
    let mut system_input_decorations = Vec::new();
    let frag_coord = S::frag_coord(reads_frag_coord, &mut n);
    let front_facing = S::front_facing(xs, &mut n);
    // Vertex shaders: one vec4 Input and Output per generic location the
    // header marks (vertex input and output maps).
    let mut vertex_outputs = Vec::new();
    let has_dynamic_input = stage_io.is_none() && required.dynamic_input;
    let dynamic_input = has_dynamic_input.then(&mut n);
    track.declare_inputs(
        shader_stage::Inputs {
            xs,
            state,
            has_dynamic_input,
            decorations: &mut input_decorations,
            by_location: &mut input_by_location,
            vertex_outputs: &mut vertex_outputs,
            system: &mut system_inputs,
            system_decorations: &mut system_input_decorations,
        },
        &mut n,
    );
    let needs_local_invocation_id = required.local_invocation_id;
    let needs_workgroup_id = required.workgroup_id;
    // Ryujinx CodeGen/Spirv/IoMap.cs:20,44: these builtins are compute
    // inputs. Reject other stages rather than emitting an invalid module.
    S::compute_registers(needs_local_invocation_id, needs_workgroup_id)?;
    // Ryujinx InstEmitMove.cs:94-114: patch count is dynamic in tessellation;
    // other graphics stages use the driver's input primitive vertex count
    // (Ryujinx InputTopology.cs:26-34).
    let needs_invocation_info = required.invocation_info;
    let patch_vertices = S::patch_vertices(needs_invocation_info, &mut n);
    let primitive_vertices = S::primitive_vertices(needs_invocation_info, state)?;
    let needs_y_direction = required.y_direction;
    let outputs = track.outputs(xs, output, &mut n);
    let pfloat_out = n();
    let mut entry_interfaces = vec![S::EXECUTION_MODEL, fun, 0x6e69616d, 0];
    outputs.entry(&mut entry_interfaces);
    entry_interfaces.extend(input_decorations.iter().map(|(input, _, _)| *input));
    entry_interfaces.extend(dynamic_input);
    entry_interfaces.extend(helper_invocation);
    entry_interfaces.extend(vertex_outputs.iter().map(|(_, v)| *v));
    entry_interfaces.extend(system_inputs.values().copied());
    entry_interfaces.extend(frag_coord);
    entry_interfaces.extend(front_facing);
    let local_invocation_id = needs_local_invocation_id.then_some(n());
    let workgroup_id = needs_workgroup_id.then_some(n());
    let lane_id = needs_lane_id.then_some(n());
    // The LANEID and InvocationId inputs a lane-indexed output read uses,
    // present exactly when `reads_output_lanes` holds.
    let control_points = lane_id
        .filter(|_| reads_output_lanes)
        .zip(stage_io.as_ref().and_then(stages::StageIo::invocation_id));
    // SReg ordering Eq/Lt/Le/Gt/Ge: envytools gm107.c:173-177.
    // BuiltIn numbers: Mesa spirv.core.grammar.json:16221-16258.
    const MASK_BUILTIN: [u32; 5] = [4416, 4420, 4419, 4418, 4417];
    let mask_inputs: Vec<_> = required
        .lane_masks
        .iter()
        .map(|&idx| (idx, MASK_BUILTIN[usize::from(idx - 0x38)], n()))
        .collect();
    entry_interfaces.extend(mask_inputs.iter().map(|(_, _, variable)| *variable));
    entry_interfaces.extend(patch_vertices);
    let vec3 = n();
    entry_interfaces.extend(local_invocation_id);
    entry_interfaces.extend(workgroup_id);
    entry_interfaces.extend(lane_id);
    for io in stage_io.iter() {
        entry_interfaces.extend(io.interfaces());
    }
    inst(&mut w, Op::EntryPoint, &entry_interfaces);
    inst_times(
        &mut w,
        Op::ExecutionMode,
        &[fun, 6023],
        usize::from(requires_maximal_reconvergence),
    );
    for io in stage_io.iter() {
        io.modes(&mut w, fun);
    }
    S::execution_modes(&outputs, &mut w, fun);
    if preserve_float32 {
        // Mesa grammar:13168-13195: target width follows each mode.
        inst(&mut w, Op::ExecutionMode, &[fun, 4459, 32]);
        inst(&mut w, Op::ExecutionMode, &[fun, 4461, 32]);
        inst(&mut w, Op::ExecutionMode, &[fun, 4462, 32]);
        f32_exact_lane.declare_rounding_mode(&mut w, fun);
    }
    crate::global_memory::debug_hints(&mut w, &mut n, xs);
    let annotation_at = w.len();
    for (location, variable) in &vertex_outputs {
        inst(annotations, Op::Decorate, &[*variable, 30, *location]);
    }
    {
        for (variable, builtin) in &system_input_decorations {
            inst(annotations, Op::Decorate, &[*variable, 11, *builtin]);
        }
        for input in frag_coord.into_iter() {
            // Mesa nak_nir_lower_fs_inputs.c:209-226 lowers load_frag_coord to
            // IPA.PASS on 0x70..0x7c, so FragCoord.w is passed directly.
            inst(annotations, Op::Decorate, &[input, 11, 15]);
        }
        for (input, location, interpolation) in &input_decorations {
            inst(annotations, Op::Decorate, &[*input, 30, *location]);
            match interpolation {
                Interpolation::Constant => inst(annotations, Op::Decorate, &[*input, 14]),
                Interpolation::ScreenLinear => inst(annotations, Op::Decorate, &[*input, 13]),
                Interpolation::Perspective | Interpolation::Unused => {}
            }
        }
    }
    outputs.decorate(annotations);
    let iadd_carry_type = n();
    inst(&mut w, Op::TypeVoid, &[void]);
    inst(&mut w, Op::TypeBool, &[boolt]);
    inst(&mut w, Op::TypeInt, &[uint, 32, 0]);
    // Mesa models IADD carry as a two-component result: the low sum and the
    // unsigned overflow carry (ir.rs:3792-3857; sm50.rs:1249-1286).
    inst(&mut w, Op::TypeStruct, &[iadd_carry_type, uint, uint]);
    inst(&mut w, Op::TypeFloat, &[float, 32]);
    let double_types = if needs_float64 {
        let double = n();
        let words = n();
        inst(&mut w, Op::TypeFloat, &[double, 64]);
        inst(&mut w, Op::TypeVector, &[words, uint, 2]);
        Some((double, words))
    } else {
        None
    };
    let double_one = double_types.map(|(double, _)| {
        let one = n();
        // IEEE-754 binary64 1.0, literal words low then high.
        inst(&mut w, Op::Constant, &[double, one, 0, 0x3ff0_0000]);
        one
    });
    // The binary64 type for directed binary32 arithmetic.
    let binary64 = f32_exact_lane.binary64(double_types);
    let mut integer64 = if needs_int64 && !uses_global {
        let ty = n();
        inst(&mut w, Op::TypeInt, &[ty, 64, 0]);
        Some(ty)
    } else {
        None
    };
    inst(&mut w, Op::TypeVector, &[vec4, float, 4]);
    inst(&mut w, Op::TypeVector, &[vec3, uint, 3]);
    inst(&mut w, Op::TypePointer, &[input_type, 1, vec4]);
    // Numeric SPIR-V fields: Mesa src/compiler/spirv/spirv.h:728
    // (HelperInvocation=23).
    // BuiltIn FrontFacing=17 (spirv.h:722); Ryujinx CodeGen/Spirv/IoMap.cs:24
    // declares it as a Bool input.
    for variable in front_facing.into_iter() {
        let pointer = n();
        inst(&mut w, Op::TypePointer, &[pointer, 1, boolt]);
        inst(&mut w, Op::Variable, &[pointer, variable, 1]);
        inst(annotations, Op::Decorate, &[variable, 11, 17]);
    }
    for variable in helper_invocation.into_iter() {
        let pointer = n();
        inst(&mut w, Op::TypePointer, &[pointer, 1, boolt]);
        inst(&mut w, Op::Variable, &[pointer, variable, 1]);
        inst(annotations, Op::Decorate, &[variable, 11, 23]);
        for undefined in helper_undefined.into_iter() {
            inst(&mut w, Op::Undef, &[uint, undefined]);
        }
    }
    let input_array_type = n();
    let input_array_ptr = n();
    let input_array_elem_ptr = n();
    let input_array_scalar_ptr = has_dynamic_input.then(&mut n);
    for (scalar_ptr, dynamic_input) in input_array_scalar_ptr.zip(dynamic_input).into_iter() {
        // Ryujinx InstEmitAttribute.cs:43-51 treats the physical register as
        // a byte address: (R - 0x80) >> 4 selects a vec4 and ((R - 0x80)
        // >> 2) & 3 selects its component. AttributeConsts.cs:31-33 bounds
        // the bank to 32 locations. Mesa sm50.rs:2864-2876 encodes the
        // scalar physical form; static loads share the same Input bank.
        let array_len = n();
        inst(&mut w, Op::Constant, &[uint, array_len, 32]);
        inst(&mut w, Op::TypeArray, &[input_array_type, vec4, array_len]);
        inst(
            &mut w,
            Op::TypePointer,
            &[input_array_ptr, 1, input_array_type],
        );
        inst(&mut w, Op::TypePointer, &[input_array_elem_ptr, 1, vec4]);
        inst(&mut w, Op::TypePointer, &[scalar_ptr, 1, float]);
        inst(&mut w, Op::Variable, &[input_array_ptr, dynamic_input, 1]);
        inst(annotations, Op::Decorate, &[dynamic_input, 30, 0]);
    }
    let input_uint_ptr =
        (!system_inputs.is_empty() || lane_id.is_some() || patch_vertices.is_some()).then(&mut n);
    for ty in input_uint_ptr.into_iter() {
        inst(&mut w, Op::TypePointer, &[ty, 1, uint]);
    }
    for ty in input_uint_ptr.into_iter() {
        for &variable in system_inputs.values() {
            inst(&mut w, Op::Variable, &[ty, variable, 1]);
        }
    }
    let vec2 = n();
    inst(&mut w, Op::TypeVector, &[vec2, float, 2]);
    // The vector half paths need a bool2 type. Avoid allocating an ID for
    // other shaders, preserving main's frozen output hashes and eliminating
    // an unused declaration from those modules.
    let bool_vec2 = if required.half {
        let ty = n();
        inst(&mut w, Op::TypeVector, &[ty, boolt, 2]);
        ty
    } else {
        0
    };
    let uint_vec4 = n();
    inst(&mut w, Op::TypeVector, &[uint_vec4, uint, 4]);
    for (variable, ty) in patch_vertices.zip(input_uint_ptr).into_iter() {
        // Ryujinx IoMap.cs:30; Mesa spirv.core.grammar.json:16029-16032.
        inst(&mut w, Op::Variable, &[ty, variable, 1]);
        inst(annotations, Op::Decorate, &[variable, 11, 14]);
    }
    if !mask_inputs.is_empty() {
        let pointer = n();
        inst(&mut w, Op::TypePointer, &[pointer, 1, uint_vec4]);
        for &(_, builtin, variable) in &mask_inputs {
            inst(&mut w, Op::Variable, &[pointer, variable, 1]);
            inst(annotations, Op::Decorate, &[variable, 11, builtin]);
            if stage == Stage::Fragment {
                inst(annotations, Op::Decorate, &[variable, 14]);
            }
        }
    }
    let y_direction = if needs_y_direction {
        let value = n();
        inst(&mut w, Op::SpecConstant, &[float, value, 0x3f800000]);
        inst(
            annotations,
            Op::Decorate,
            &[value, 1, crate::SPEC_ID_Y_DIRECTION],
        );
        Some(value)
    } else {
        None
    };
    inst(&mut w, Op::TypeFunction, &[fnty, void]);
    inst(&mut w, Op::TypePointer, &[pu, 7, uint]);
    inst(&mut w, Op::TypePointer, &[pb, 7, boolt]);
    inst(&mut w, Op::TypePointer, &[po, 3, vec4]);
    for (input, _, _) in &input_decorations {
        inst(&mut w, Op::Variable, &[input_type, *input, 1]);
    }
    for input in frag_coord.into_iter() {
        inst(&mut w, Op::Variable, &[input_type, input, 1]);
    }
    for variable in local_invocation_id.into_iter() {
        let pointer = n();
        inst(&mut w, Op::TypePointer, &[pointer, 1, vec3]);
        inst(&mut w, Op::Variable, &[pointer, variable, 1]);
        inst(annotations, Op::Decorate, &[variable, 11, 27]);
    }
    for variable in workgroup_id.into_iter() {
        let pointer = n();
        inst(&mut w, Op::TypePointer, &[pointer, 1, vec3]);
        inst(&mut w, Op::Variable, &[pointer, variable, 1]);
        inst(annotations, Op::Decorate, &[variable, 11, 26]);
    }
    for (variable, ty) in lane_id.zip(input_uint_ptr).into_iter() {
        inst(&mut w, Op::Variable, &[ty, variable, 1]);
        inst(annotations, Op::Decorate, &[variable, 11, 41]);
        if stage == Stage::Fragment {
            // SPIR-V 1.6 section 3.2.19: Flat = 14. Vulkan standalone
            // VUID-StandaloneSpirv-Flat-04744 requires integer fragment inputs
            // to have Flat, including the lane-ID builtin.
            inst(annotations, Op::Decorate, &[variable, 14]);
        }
    }
    let subgroup = n();
    inst(&mut w, Op::Constant, &[uint, subgroup, 3]);
    let one_float = n();
    let zero_float = n();
    inst(&mut w, Op::Constant, &[uint, zero, 0]);
    inst(&mut w, Op::Constant, &[float, one_float, 0x3f800000]);
    inst(&mut w, Op::Constant, &[float, zero_float, 0]);
    inst(&mut w, Op::ConstantFalse, &[boolt, no]);
    inst(&mut w, Op::ConstantTrue, &[boolt, yes]);
    constants.insert(0, zero);
    cb_bindings.extend(
        xs.iter()
            .flat_map(|x| x.kind.source_operands())
            .filter_map(|source| match source {
                Source::ConstantBuffer { binding, .. } => Some(binding),
                _ => None,
            })
            .chain(xs.iter().filter_map(|x| match x.kind {
                InstructionKind::Ldc(ops::Ldc { binding, .. }) => Some(binding),
                _ => None,
            })),
    );
    // LDC.IS chooses the constant-buffer bank from the address register at
    // runtime. The Vulkan interface therefore exposes every bank that the
    // closed five-bit Maxwell bank domain can select.
    cb_bindings.extend(0..[0, 32][usize::from(required.ldc_indexed)]);
    cb_bindings.sort_unstable();
    cb_bindings.dedup();
    for binding in cb_bindings.iter().copied() {
        if cb_types.contains_key(&binding) {
            continue;
        }
        let length = n();
        let array = n();
        let structure = n();
        let pointer = n();
        let element_pointer = n();
        let variable = n();
        // A Maxwell constant buffer holds up to 64 KiB: 4096 vec4.
        inst(&mut w, Op::Constant, &[uint, length, 4096]);
        inst(&mut w, Op::TypeArray, &[array, vec4, length]);
        inst(&mut w, Op::TypeStruct, &[structure, array]);
        inst(&mut w, Op::TypePointer, &[pointer, 2, structure]);
        inst(&mut w, Op::TypePointer, &[element_pointer, 2, vec4]);
        inst(annotations, Op::Decorate, &[structure, 2]);
        inst(annotations, Op::MemberDecorate, &[structure, 0, 35, 0]);
        inst(annotations, Op::Decorate, &[array, 6, 16]);
        inst(annotations, Op::Decorate, &[variable, 34, 0]);
        inst(
            annotations,
            Op::Decorate,
            &[variable, 33, u32::from(binding)],
        );
        inst(&mut w, Op::Variable, &[pointer, variable, 2]);
        cb_types.insert(binding, (array, structure, pointer, element_pointer));
        cb_vars.insert(binding, variable);
    }
    // Bindless textures use a register handle (Mesa sm50.rs:2180-2183). Each
    // distinct handle source (a constant, or a constant-buffer word the
    // program loads it from) gets its own texture and sampler binding,
    // 2k and 2k + 1 in set 0, so a host can resolve the handle at draw time.
    // Bindings are numbered in program order so translation is deterministic.
    let mut handle_keys: Vec<HandleSource> = Vec::new();
    let mut handle_indices = HashMap::default();
    let mut ordered = handles.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|(index, _)| **index);
    for (index, key) in ordered {
        // Surface handles get storage-image bindings below, not textures.
        if matches!(
            xs[*index].kind,
            InstructionKind::Texture(ops::Texture { .. })
        ) {
            if let std::collections::hash_map::Entry::Vacant(entry) = handle_indices.entry(*key) {
                entry.insert(handle_keys.len());
                handle_keys.push(*key);
            }
        }
    }
    // Bound indices use set 1: image 2*t and sampler 2*t+1 for
    // table index t. Set 0 retains the existing bindless descriptor scheme.
    // This preserves U13_36 even with a single scalar instruction, and
    // avoids collisions with set 0 constant buffers and storage images.
    // Ryujinx InstEmitTexture.cs:612-650 retains the bound table identity.
    // One uvec2 type shared by texture fetches, offsets and surfaces.
    let uses_images = required.images;
    // Reuse the double split's uvec2 when images and I2F.F64 coexist.
    // SPIR-V forbids duplicate non-aggregate type declarations.
    let uvec2 = double_types.map_or_else(&mut n, |(_, words)| words);
    // vec3 is already the uint vector of 3; SPIR-V forbids a duplicate.
    let uvec3 = vec3;
    inst_times(
        &mut w,
        Op::TypeVector,
        &[uvec2, uint, 2],
        usize::from((uses_images || uses_global || required.half_uvec2) & double_types.is_none()),
    );
    let storage = crate::storage_memory::StorageMemory::declare(
        &mut w,
        annotations,
        &mut n,
        &storage_plan,
        uint,
    );
    let global = uses_global.then(|| {
        crate::global_memory::GlobalMemoryEmitter::declare(
            &mut w,
            annotations,
            &mut n,
            xs,
            options,
            uint,
            uvec2,
            uint_vec4,
            float,
        )
    });
    for global in global.iter() {
        integer64 = Some(global.integer_type());
    }
    let mut texture_bindings = Vec::new();
    let mut texture_types = None;
    // Each handle binding takes the image type of the instructions that use
    // it; a handle used with two dimensions has no single type (None).
    let mut key_dims: Vec<Option<(u8, bool, bool)>> = vec![None; handle_keys.len()];
    let mut key_conflict = vec![false; handle_keys.len()];
    for (index, key) in &handles {
        if let InstructionKind::Texture(ops::Texture {
            dim,
            depth_compare,
            multisample,
            ..
        }) = xs[*index].kind
        {
            let dim = (dim, depth_compare, multisample);
            if let Some(k) = handle_indices.get(key).copied() {
                match key_dims[k] {
                    None => key_dims[k] = Some(dim),
                    Some(d) if d != dim => key_conflict[k] = true,
                    _ => {}
                }
            }
        }
    }
    let fvec3 = n();
    inst_times(
        &mut w,
        Op::TypeVector,
        &[fvec3, float, 3],
        usize::from(!handle_keys.is_empty() || stage == Stage::TessellationEvaluation),
    );
    let uvec4 = uint_vec4;
    if !handle_keys.is_empty() {
        let sampler = n();
        let sampler_ptr = n();
        inst(&mut w, Op::TypeSampler, &[sampler]);
        inst(&mut w, Op::TypePointer, &[sampler_ptr, 0, sampler]);
        // One image type per distinct Mesa dimension (set_tex_dim, sm50.rs:
        // 2105-2119): 0 1D, 1 1D array, 2 2D, 3 2D array, 4 3D, 6 cube,
        // 7 cube array -> SPIR-V Dim and Arrayed.
        let mut dim_types: HashMap<(u8, bool, bool), (u32, u32, u32)> = HashMap::default();
        for shape @ (dim, depth, ms) in key_dims.iter().flatten() {
            if dim_types.contains_key(shape) || !matches!(dim, 0 | 1 | 2 | 3 | 4 | 6 | 7) {
                continue;
            }
            let (spirv_dim, arrayed) = match dim {
                0 => (0, 0),
                1 => (0, 1),
                2 => (1, 0),
                3 => (1, 1),
                4 => (2, 0),
                6 => (3, 0),
                _ => (3, 1),
            };
            let image = n();
            let sampled = n();
            let image_ptr = n();
            // Depth and MS are part of the descriptor type. Ryujinx
            // CodeGen/Spirv/Declarations.cs:176-183; Instructions.cs:1394-1436.
            inst(
                &mut w,
                Op::TypeImage,
                &[
                    image,
                    float,
                    spirv_dim,
                    u32::from(*depth),
                    arrayed,
                    u32::from(*ms),
                    1,
                    0,
                ],
            );
            inst(&mut w, Op::TypeSampledImage, &[sampled, image]);
            inst(&mut w, Op::TypePointer, &[image_ptr, 0, image]);
            dim_types.insert(*shape, (image, sampled, image_ptr));
        }
        for (k, key) in handle_keys.iter().enumerate() {
            let (set, pair) = match key {
                HandleSource::Bound(t) => (1, u32::from(*t)),
                _ => (0, k as u32),
            };
            let image_var = n();
            let sampler_var = n();
            let image_shape = key_dims[k]
                .and_then(|d| dim_types.get(&d))
                .map(|t| (t.0, t.2));
            inst(annotations, Op::Decorate, &[image_var, 34, set]);
            inst(annotations, Op::Decorate, &[image_var, 33, 2 * pair]);
            inst(annotations, Op::Decorate, &[sampler_var, 34, set]);
            inst(annotations, Op::Decorate, &[sampler_var, 33, 2 * pair + 1]);
            let array_pointers = if let Some(&length) = texture_array_lengths.get(key) {
                let (image, image_ptr) = image_shape.expect("admitted texture array dimension");
                let length_id = if let Some(&constant) = constants.get(&length) {
                    constant
                } else {
                    let constant = n();
                    inst(&mut w, Op::Constant, &[uint, constant, length]);
                    constants.insert(length, constant);
                    constant
                };
                if let std::collections::hash_map::Entry::Vacant(entry) =
                    constants.entry(length - 1)
                {
                    let constant = n();
                    inst(&mut w, Op::Constant, &[uint, constant, length - 1]);
                    entry.insert(constant);
                }
                let image_array = n();
                let image_array_ptr = n();
                let sampler_array = n();
                let sampler_array_ptr = n();
                inst(&mut w, Op::TypeArray, &[image_array, image, length_id]);
                inst(&mut w, Op::TypePointer, &[image_array_ptr, 0, image_array]);
                inst(&mut w, Op::TypeArray, &[sampler_array, sampler, length_id]);
                inst(
                    &mut w,
                    Op::TypePointer,
                    &[sampler_array_ptr, 0, sampler_array],
                );
                inst(&mut w, Op::Variable, &[image_array_ptr, image_var, 0]);
                inst(&mut w, Op::Variable, &[sampler_array_ptr, sampler_var, 0]);
                Some((image_ptr, sampler_ptr))
            } else {
                if let Some((_, image_ptr)) = image_shape {
                    inst(&mut w, Op::Variable, &[image_ptr, image_var, 0]);
                }
                inst(&mut w, Op::Variable, &[sampler_ptr, sampler_var, 0]);
                None
            };
            texture_bindings.push((image_var, sampler_var, array_pointers));
        }
        texture_types = Some((dim_types, sampler, uvec2));
    }
    // Surface handles use their own storage-image descriptor range. The
    // handle tracing is shared with textures because Mesa emits the same
    // constant, constant-buffer, and LOP handle construction for both
    // classes (Mesa nv50_ir_lowering_nvc0.cpp:2640-2649).
    let mut storage_keys = Vec::new();
    let mut storage_shapes = Vec::new();
    let mut storage_coherent = Vec::new();
    let mut storage_indices = HashMap::default();
    let mut storage_uses = HashMap::default();
    for (index, x) in xs.iter().enumerate() {
        let spec = match x.kind.legacy_form() {
            InstructionKind::Surface(ops::Surface {
                store,
                target,
                format,
                size,
                cache,
                ..
            }) => Some((
                target,
                surface_format(format, size).ok_or_else(|| bad(x))?,
                CachePolicy::of(store, cache) == CachePolicy::Coherent,
            )),
            // R32i=24 / R32ui=33: Mesa src/compiler/spirv/spirv.h:316,325.
            // Ryujinx InstEmitSurface.cs:709-720 requires matching raw views.
            InstructionKind::SurfaceAtomic(ops::SurfaceAtomic { target, signed, .. }) => {
                Some((target, [33, 24][usize::from(signed)], false))
            }
            _ => None,
        };
        for (target, image_format, coherent) in spec.into_iter() {
            let shape = (target, image_format);
            let key = handles[&index];
            // Resource identity includes the format in Ryujinx
            // Translation/ResourceManager.cs:263-308. Track target too:
            // OpTypeImage must match the instruction's coordinate shape
            // (CodeGen/Spirv/Declarations.cs:232-241). Static descriptor
            // selection avoids any runtime branch or handle arithmetic.
            let k = match storage_indices.entry((key, shape)) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let k = storage_keys.len();
                    entry.insert(k);
                    storage_keys.push(key);
                    storage_shapes.push(shape);
                    storage_coherent.push(coherent);
                    k
                }
                std::collections::hash_map::Entry::Occupied(entry) => {
                    let k = *entry.get();
                    storage_coherent[k] |= coherent;
                    k
                }
            };
            storage_uses.insert(index, k);
        }
    }
    let mut storage_bindings = Vec::new();
    let mut storage_images = Vec::new();
    let mut surface_descriptors = Vec::new();
    let mut bound_surface_handles = HashSet::default();
    // Allocate early signed types only for raw signed storage views. Other
    // shaders retain their declaration/ID order and existing output hashes.
    let storage_signed_types = required.signed_storage.then(|| {
        let scalar = n();
        let vector = n();
        inst(&mut w, Op::TypeInt, &[scalar, 32, 1]);
        inst(&mut w, Op::TypeVector, &[vector, scalar, 4]);
        (scalar, vector)
    });
    let mut storage_types: HashMap<(u32, u32, u32), (u32, u32)> = HashMap::default();
    if !storage_keys.is_empty() {
        for (k, &(target, image_format)) in storage_shapes.iter().enumerate() {
            // Mesa emits x from coord, then y and z/layer from consecutive
            // registers in nv50_ir_lowering_nvc0.cpp:2625-2668; target values
            // and their dimensions are emitted by emitSUTarget at
            // nv50_ir_emit_gm107.cpp:3261-3275.
            let (dim, arrayed) =
                surface_image_shape(target).ok_or(Stopped::SurfaceTarget(target))?;
            // One image type per shape: SPIR-V forbids duplicate types.
            let (image, image_ptr) = *storage_types
                .entry((dim, arrayed, image_format))
                .or_insert_with(|| {
                    let image = n();
                    let image_ptr = n();
                    let component = match image_format {
                        0 => float,
                        24 | 28 | 29 => storage_signed_types.expect("signed surface types").0,
                        _ => uint,
                    };
                    inst(
                        &mut w,
                        Op::TypeImage,
                        &[image, component, dim, 0, arrayed, 0, 2, image_format],
                    );
                    inst(&mut w, Op::TypePointer, &[image_ptr, 0, image]);
                    (image, image_ptr)
                });
            let variable = n();
            let (set, binding) = match storage_keys[k] {
                // Keep the first bound view at table index t. Aliases use
                // 8192+k, above every U13_36 index (gm107.c:788-791).
                // Bindless views retain 2048+k. These are project-owned
                // descriptor conventions, exported below for the host.
                HandleSource::Bound(t) if bound_surface_handles.insert(t) => (2, u32::from(t)),
                HandleSource::Bound(_) => (2, 8192 + k as u32),
                _ => (0, 2048 + k as u32),
            };
            inst(annotations, Op::Decorate, &[variable, 34, set]);
            inst(annotations, Op::Decorate, &[variable, 33, binding]);
            if storage_coherent[k] {
                // Ryujinx CodeGen/Spirv/Declarations.cs:265-267.
                inst(annotations, Op::Decorate, &[variable, 23]);
            }
            inst(&mut w, Op::Variable, &[image_ptr, variable, 0]);
            storage_bindings.push(variable);
            storage_images.push(image);
            surface_descriptors.push(SurfaceDescriptor {
                source: storage_keys[k],
                target,
                image_format,
                set,
                binding,
            });
        }
    }
    // Image storage-class scalar pointers for OpImageTexelPointer. Declare
    // each type once. Ryujinx CodeGen/Spirv/Instructions.cs:625-626,661;
    // Mesa src/compiler/spirv/spirv.h:234,1790.
    let mut atomic_pointers = HashMap::default();
    for x in xs {
        if let InstructionKind::SurfaceAtomic(ops::SurfaceAtomic { signed, .. }) =
            x.kind.legacy_form()
        {
            atomic_pointers.entry(signed).or_insert_with(|| {
                let pointer = n();
                let component = if signed {
                    storage_signed_types.expect("signed image").0
                } else {
                    uint
                };
                inst(&mut w, Op::TypePointer, &[pointer, 11, component]);
                pointer
            });
        }
    }
    for x in xs {
        for src in x.kind.source_operands() {
            let value = match src {
                Source::Immediate(v) => Some(v),
                Source::ConstantBuffer { offset, .. } => Some(u32::from(offset) / 16),
                Source::Register(_) => None,
            };
            for v in value.into_iter() {
                if let std::collections::hash_map::Entry::Vacant(entry) = constants.entry(v) {
                    let q = n();
                    inst(&mut w, Op::Constant, &[uint, q, v]);
                    entry.insert(q);
                }
            }
        }
    }
    // BRX consumes a byte displacement loaded from the table, not its slot
    // index. Ryujinx InstEmitFlowControl.cs:38-40,101-109 compares addresses.
    for x in xs {
        if let InstructionKind::Brx(ops::Brx { table, .. }) = x.kind {
            for target in table.targets() {
                let offset = target.loaded();
                if let std::collections::hash_map::Entry::Vacant(entry) = constants.entry(offset) {
                    let q = n();
                    inst(&mut w, Op::Constant, &[uint, q, offset]);
                    entry.insert(q);
                }
            }
        }
    }
    labels.extend(xs.iter().map(|x| (x.address, n())));
    let loop_exit = n();
    block_starts.extend(xs.first().map(|x| x.address));
    for (index, x) in xs.iter().enumerate() {
        x.kind
            .mark_block_starts(&mut *block_starts, &xs[index + 1..]);
    }
    block_starts.extend(regions.boundaries());
    let register_brx = required.register_brx;
    block_starts.extend(
        xs.iter()
            .take(xs.len() * usize::from(register_brx))
            .map(|x| x.address),
    );
    // Control flow is emitted as a dispatch loop: every basic block is a case
    // of an OpSwitch on a program-counter variable, and each block ends by
    // storing its successor. Register BRX uses byte addresses as case values.
    // This replaces Ryujinx's address comparisons (InstEmitFlowControl.cs:
    // 93-115) with address arithmetic and the existing switch. No target
    // table is guessed. Token regions below add nested loop merges for
    // collective participation; the flat loop alone only models per-lane PCs.
    blocks.extend(block_starts.iter().copied());
    blocks.sort_unstable();
    case_of.extend(blocks.iter().enumerate().map(|(index, address)| {
        (
            *address,
            [index as u32, *address][usize::from(register_brx)],
        )
    }));
    const EXIT_CASE: u32 = u32::MAX;
    for value in blocks
        .iter()
        .map(|address| case_of[address])
        .chain([EXIT_CASE])
        .chain(xs.iter().flat_map(|x| {
            let table: &[crate::ir::BrxEntry] = match &x.kind {
                InstructionKind::Brx(ops::Brx { table, .. }) => table.targets(),
                _ => &[],
            };
            table.iter().map(|target| target.loaded())
        }))
    {
        if let std::collections::hash_map::Entry::Vacant(entry) = constants.entry(value) {
            let q = n();
            inst(&mut w, Op::Constant, &[uint, q, value]);
            entry.insert(q);
        }
    }
    // Saturation bounds for F2I (largest floats below 2^31 and 2^32, and
    // -2^31) and the saturated integer results.
    for bits in [0xcf00_0000u32, 0x4eff_ffff, 0x4f7f_ffff, 0] {
        let q = n();
        inst(&mut w, Op::Constant, &[float, q, bits]);
        float_constants.insert(bits, q);
    }
    // FMUL scale factors 2^-3..2^3 without 2^0 (envytools gm107.c:1331-1345).
    let [divide8, divide4, divide2, multiply2, multiply4, multiply8] = [-3_i32, -2, -1, 1, 2, 3]
        .map(|power| {
            let bits = (2.0_f32.powi(power)).to_bits();
            let q = n();
            inst(&mut w, Op::Constant, &[float, q, bits]);
            float_constants.insert(bits, q);
            q
        });
    let scale_factors = ScaleFactors {
        divide8,
        divide4,
        divide2,
        multiply2,
        multiply4,
        multiply8,
    };
    for value in (0..=32u32).chain([
        0x7fff_ffff,
        0x8000_0000,
        u32::MAX,
        63,
        64,
        128,
        255,
        0xffff,
        !3,
        !31,
        0x0080_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
    ]) {
        if let std::collections::hash_map::Entry::Vacant(entry) = constants.entry(value) {
            let q = n();
            inst(&mut w, Op::Constant, &[uint, q, value]);
            entry.insert(q);
        }
    }
    let pc_var = n();
    let header = n();
    let dispatch = n();
    let switch_merge = n();
    let continue_block = n();
    let default_block = n();
    let regs: [u32; 256] = std::array::from_fn(|_| n());
    let texture_array_indices: Option<[u32; 256]> =
        dynamic_texture_arrays.then(|| std::array::from_fn(|_| n()));
    let ps: [u32; 8] = std::array::from_fn(|_| n());
    let cc_zero = n();
    let cc_sign = n();
    let cc_carry = n();
    let cc_overflow = n();
    // Maxwell shared memory is Workgroup storage.  Use the documented 48 KiB
    // fallback (docs/reference-sources.md) as 12288 uint words.
    // Declared only for compute shaders that touch shared memory (Workgroup
    // storage is compute-only in Vulkan).
    let uses_shared = required.shared;
    let uses_local = required.local;
    S::shared_memory(uses_shared)?;
    let shared_len = n();
    let shared_array = n();
    let shared_ptr_type = n();
    let shared_elem_ptr = n();
    let shared_var = n();
    if uses_shared {
        inst(&mut w, Op::Constant, &[uint, shared_len, 12288]);
        inst(&mut w, Op::TypeArray, &[shared_array, uint, shared_len]);
        inst(&mut w, Op::TypePointer, &[shared_ptr_type, 4, shared_array]);
        inst(&mut w, Op::TypePointer, &[shared_elem_ptr, 4, uint]);
        inst(&mut w, Op::Variable, &[shared_ptr_type, shared_var, 4]);
    }
    // Mesa writes the local-memory size into SPHV3_T1's low/high fields
    // (mesa/src/nouveau/compiler/nak/sph.rs:270-281), but compute headers are
    // currently emitted as zeroes (same file:494-495). Until the header parser
    // carries those fields, use the documented maximum fallback of 64 KiB,
    // represented as 16384 uint words. Function storage is private per
    // invocation, so this declaration needs no Vulkan storage capability.
    let local_array = n();
    let local_ptr_type = n();
    let local_elem_ptr = n();
    let local_var = n();
    if uses_local {
        let local_len = n();
        inst(&mut w, Op::Constant, &[uint, local_len, 16384]);
        inst(&mut w, Op::TypeArray, &[local_array, uint, local_len]);
        inst(&mut w, Op::TypePointer, &[local_ptr_type, 7, local_array]);
        inst(&mut w, Op::TypePointer, &[local_elem_ptr, 7, uint]);
    }
    outputs.declare(
        &mut w,
        OutputTypes {
            po,
            pfloat_out,
            float,
            uint,
            constants,
            vertex_outputs: &vertex_outputs,
        },
    );
    if let Some(io) = &mut stage_io {
        io.declare(
            &mut w,
            annotations,
            &mut n,
            uint,
            float,
            vec4,
            fvec3,
            input_type,
            po,
            pfloat_out,
        );
    }
    // Constants discovered while emitting the body (constant texture
    // offsets) are spliced in here, before the function.
    let function_at = w.len();
    let signed_int = storage_signed_types.map_or_else(&mut n, |(scalar, _)| scalar);
    let ivec2 = n();
    let ivec3 = n();

    let mut late_types_declared = false;
    if storage_signed_types.is_none() {
        inst(late_globals, Op::TypeInt, &[signed_int, 32, 1]);
    }
    // The signed int type is always declared (several forms use it), so
    // the ivec declarations below must not repeat it.
    inst(&mut w, Op::Function, &[void, fun, 0, fnty]);
    inst(&mut w, Op::Label, &[entry]);
    for &r in &regs {
        inst(&mut w, Op::Variable, &[pu, r, 7]);
    }
    for indices in texture_array_indices.iter() {
        for &index in indices {
            inst(&mut w, Op::Variable, &[pu, index, 7]);
        }
    }
    for &p in &ps {
        inst(&mut w, Op::Variable, &[pb, p, 7]);
    }
    inst(&mut w, Op::Variable, &[pb, cc_zero, 7]);
    inst(&mut w, Op::Variable, &[pb, cc_sign, 7]);
    inst(&mut w, Op::Variable, &[pb, cc_carry, 7]);
    inst(&mut w, Op::Variable, &[pb, cc_overflow, 7]);
    inst(&mut w, Op::Variable, &[pu, pc_var, 7]);
    // Function-storage variables belong at the top of the entry block.
    inst_times(
        &mut w,
        Op::Variable,
        &[local_ptr_type, local_var, 7],
        usize::from(uses_local),
    );
    for &r in &regs {
        inst(&mut w, Op::Store, &[r, zero]);
    }
    for indices in texture_array_indices.iter() {
        for &index in indices {
            inst(&mut w, Op::Store, &[index, zero]);
        }
    }
    // P7 is PT, always true; P0..P6 start false.
    for (index, &p) in ps.iter().enumerate() {
        inst(&mut w, Op::Store, &[p, [no, yes][usize::from(index == 7)]]);
    }
    inst(&mut w, Op::Store, &[cc_zero, no]);
    inst(&mut w, Op::Store, &[cc_sign, no]);
    inst(&mut w, Op::Store, &[cc_carry, no]);
    inst(&mut w, Op::Store, &[cc_overflow, no]);
    fn load_var(w: &mut Vec<u32>, id: &mut u32, t: u32, p: u32) -> u32 {
        let x = *id;
        *id += 1;
        inst(w, Op::Load, &[t, x, p]);
        x
    }
    #[allow(clippy::too_many_arguments)]
    fn source(
        w: &mut Vec<u32>,
        id: &mut u32,
        s: Source,
        rs: &[u32],
        t: u32,
        cs: &HashMap<u32, u32>,
        cb_types: &HashMap<u8, (u32, u32, u32, u32)>,
        cb_vars: &HashMap<u8, u32>,
        vec4: u32,
        float: u32,
        zero: u32,
    ) -> u32 {
        match s {
            // Ryujinx IntermediateRepresentation/OperandHelper.cs:43-52;
            // envytools gm107.c:116. RZ is a constant even after a discard.
            Source::Register(Register(255)) => zero,
            Source::Register(Register(r)) => load_var(w, id, t, rs[r as usize]),
            Source::Immediate(v) => cs[&v],
            Source::ConstantBuffer { binding, offset } => {
                let (array, structure, _pointer, element_pointer) = cb_types[&binding];
                let bank = cb_vars[&binding];
                let index = cs[&(u32::from(offset) / 16)];
                let element = *id;
                *id += 1;
                inst(
                    w,
                    Op::AccessChain,
                    &[element_pointer, element, bank, zero, index],
                );
                let component_float = *id;
                *id += 1;
                inst(w, Op::Load, &[vec4, component_float, element]);
                let component = *id;
                *id += 1;
                inst(
                    w,
                    Op::CompositeExtract,
                    &[float, component, component_float, u32::from(offset / 4 % 4)],
                );
                let component_uint = *id;
                *id += 1;
                inst(w, Op::Bitcast, &[t, component_uint, component]);
                let _ = (array, structure);
                component_uint
            }
        }
    }
    /// A binary64 operand's low and high words: a pair's two registers
    /// (RZ reads zero for both), a cbuf operand's two words, or a D20
    /// immediate's high word over the declared zero low word.
    #[allow(clippy::too_many_arguments)]
    fn double_words(
        w: &mut Vec<u32>,
        id: &mut u32,
        s: DoubleSource,
        rs: &[u32],
        t: u32,
        cs: &HashMap<u32, u32>,
        cb_types: &HashMap<u8, (u32, u32, u32, u32)>,
        cb_vars: &HashMap<u8, u32>,
        vec4: u32,
        float: u32,
        zero: u32,
    ) -> (u32, u32) {
        let read = |w: &mut Vec<u32>, id: &mut u32, s| {
            source(w, id, s, rs, t, cs, cb_types, cb_vars, vec4, float, zero)
        };
        match s {
            DoubleSource::Immediate(high) => (zero, read(w, id, Source::Immediate(high))),
            DoubleSource::RegisterPair(_) | DoubleSource::ConstantBuffer { .. } => {
                let [low, high] = s.words();
                let low = read(w, id, low);
                (low, read(w, id, high))
            }
        }
    }
    /// Emits an integer compare of two uint-typed values and returns a bool id.
    fn int_compare(
        w: &mut Vec<u32>,
        id: &mut u32,
        c: IntCompare,
        signedness: Signedness,
        a: u32,
        b: u32,
        (boolt, yes, no): (u32, u32, u32),
    ) -> u32 {
        match c {
            IntCompare::False => no,
            IntCompare::True => yes,
            IntCompare::Relation(rel) => {
                let v = *id;
                *id += 1;
                inst(w, int_compare_op(rel, signedness), &[boolt, v, a, b]);
                v
            }
        }
    }
    /// Emits a comparison of two values already carrying the same floating
    /// type and returns a bool id.
    fn typed_float_compare(
        w: &mut Vec<u32>,
        id: &mut u32,
        c: FloatCompare,
        a: u32,
        b: u32,
        (boolt, yes, no): (u32, u32, u32),
    ) -> u32 {
        let op = match c {
            FloatCompare::False => return no,
            FloatCompare::True => return yes,
            // Either operand NaN, from OpIsNan (156); OpOrdered and
            // OpUnordered need the Kernel capability, not allowed in
            // Vulkan. Num is the negation.
            FloatCompare::Nan => return either_nan(w, id, a, b, boolt),
            FloatCompare::Num => {
                let either = either_nan(w, id, a, b, boolt);
                return alu(w, id, Op::LogicalNot, boolt, &[either]);
            }
            FloatCompare::Ordered(rel) => ordered_compare_op(rel),
            FloatCompare::Unordered(rel) => unordered_compare_op(rel),
        };
        let v = *id;
        *id += 1;
        inst(w, op, &[boolt, v, a, b]);
        v
    }
    /// Emits a float compare of two uint-typed values (bitcast to float
    /// first) and returns a bool id.
    fn float_compare(
        w: &mut Vec<u32>,
        id: &mut u32,
        c: FloatCompare,
        a: u32,
        b: u32,
        (float, boolt, yes, no): (u32, u32, u32, u32),
    ) -> u32 {
        let a_float = *id;
        inst(w, Op::Bitcast, &[float, a_float, a]);
        let b_float = a_float + 1;
        inst(w, Op::Bitcast, &[float, b_float, b]);
        *id += 2;
        typed_float_compare(w, id, c, a_float, b_float, (boolt, yes, no))
    }
    fn either_nan(w: &mut Vec<u32>, id: &mut u32, a: u32, b: u32, boolt: u32) -> u32 {
        let nan_a = *id;
        let nan_b = *id + 1;
        let either = *id + 2;
        *id += 3;
        inst(w, Op::IsNan, &[boolt, nan_a, a]);
        inst(w, Op::IsNan, &[boolt, nan_b, b]);
        inst(w, Op::LogicalOr, &[boolt, either, nan_a, nan_b]);
        either
    }
    fn set_op_code(op: SetOp) -> Op {
        SET_OP_CODE[op as usize]
    }
    fn put_var(
        w: &mut Vec<u32>,
        id: &mut u32,
        p: u32,
        v: u32,
        pred: Option<(u32, bool)>,
        t: u32,
        bt: u32,
    ) {
        // A guarded write keeps the old value where the guard is false.
        let v = pred.into_iter().fold(v, |v, (q, inv)| {
            let q = load_var(w, id, bt, q);
            let q = alu_when(w, id, inv, Op::LogicalNot, bt, &[q], q);
            let old = load_var(w, id, t, p);
            alu(w, id, Op::Select, t, &[q, v, old])
        });
        inst(w, Op::Store, &[p, v]);
    }
    #[allow(clippy::too_many_arguments)]
    fn write_cc(
        w: &mut Vec<u32>,
        id: &mut u32,
        result: u32,
        pred: Option<(u32, bool)>,
        z: u32,
        s: u32,
        uint: u32,
        boolt: u32,
        zero: u32,
        shift31: u32,
    ) {
        let nz = *id;
        *id += 1;
        inst(w, Op::INotEqual, &[boolt, nz, result, zero]);
        let zval = *id;
        *id += 1;
        inst(w, Op::LogicalNot, &[boolt, zval, nz]);
        put_var(w, id, z, zval, pred, boolt, boolt);
        let bits = *id;
        *id += 1;
        inst(w, Op::ShiftRightLogical, &[uint, bits, result, shift31]);
        let sval = *id;
        *id += 1;
        inst(w, Op::INotEqual, &[boolt, sval, bits, zero]);
        put_var(w, id, s, sval, pred, boolt, boolt);
    }
    #[allow(clippy::too_many_arguments)]
    fn write_iadd_overflow(
        w: &mut Vec<u32>,
        id: &mut u32,
        result: u32,
        lhs: u32,
        rhs: u32,
        pred: Option<(u32, bool)>,
        overflow: u32,
        sint: u32,
        boolt: u32,
        zero: u32,
        shift31: u32,
    ) {
        // Ryujinx InstEmitIntegerArithmetic.cs:665-698: V is the sign bit of
        // ((result ^ lhs) & ~(lhs ^ rhs)); this is signed-add overflow.
        let x = *id;
        *id += 1;
        inst(w, Op::BitwiseXor, &[sint, x, result, lhs]);
        let y = *id;
        *id += 1;
        inst(w, Op::BitwiseXor, &[sint, y, lhs, rhs]);
        let ny = *id;
        *id += 1;
        inst(w, Op::Not, &[sint, ny, y]); // OpNot: bitwise, not OpLogicalNot (168)
        let v = *id;
        *id += 1;
        inst(w, Op::BitwiseAnd, &[sint, v, x, ny]);
        let bits = *id;
        *id += 1;
        inst(w, Op::ShiftRightLogical, &[sint, bits, v, shift31]);
        let flag = *id;
        *id += 1;
        inst(w, Op::INotEqual, &[boolt, flag, bits, zero]);
        put_var(w, id, overflow, flag, pred, boolt, boolt);
    }
    // Special registers remain constants (Ryujinx
    // IntermediateRepresentation/OperandHelper.cs:43-52; envytools
    // gm107.c:116,363-364). A discarded destination must not change later
    // operands, but arithmetic and condition-code effects still execute.
    let load = |w: &mut Vec<u32>, id: &mut u32, t, p| {
        if p == regs[255] {
            zero
        } else if p == ps[7] {
            yes
        } else {
            load_var(w, id, t, p)
        }
    };
    let put = |w: &mut Vec<u32>, id: &mut u32, p, v, pred, t, bt| {
        if p != regs[255] && p != ps[7] {
            put_var(w, id, p, v, pred, t, bt);
        }
    };
    // A binary64 result: the high word, then the low word where the pair
    // has a register of its own for it. Ryujinx InstEmitHelper.cs:215-226
    // writes D, then D | 1, so an odd pair keeps only the high word.
    let put_pair =
        |w: &mut Vec<u32>, id: &mut u32, pair: RegisterPair, [low, high]: [u32; 2], pred, t, bt| {
            put(w, id, regs[pair.high().0 as usize], high, pred, t, bt);
            if let Some(register) = pair.low_alone() {
                put(w, id, regs[register.0 as usize], low, pred, t, bt);
            }
        };
    let global = global.map(|g| g.load_delta(&mut w, &mut id, zero));
    let first_case = xs
        .first()
        .map(|x| constants[&case_of[&x.address]])
        .unwrap_or(constants[&EXIT_CASE]);
    inst(&mut w, Op::Store, &[pc_var, first_case]);
    let dispatch_at = w.len();
    inst(&mut w, Op::Branch, &[header]);
    inst(&mut w, Op::Label, &[header]);
    inst(&mut w, Op::LoopMerge, &[loop_exit, continue_block, 0]);
    inst(&mut w, Op::Branch, &[dispatch]);
    inst(&mut w, Op::Label, &[dispatch]);
    let selector = load(&mut w, &mut id, uint, pc_var);
    inst(&mut w, Op::SelectionMerge, &[switch_merge, 0]);
    switch.extend([selector, default_block]);
    for address in blocks.iter() {
        switch.push(case_of[address]);
        switch.push(labels[address]);
    }
    inst(&mut w, Op::Switch, switch);
    // LDL/LDS and STL/STS address a uint word array: the per-invocation
    // local array or the workgroup shared array (Ryujinx
    // InstEmitMemory.cs:110-131,146-167). Each kind passes its own array.
    let local_words = WordArray {
        elem_ptr: local_elem_ptr,
        variable: local_var,
    };
    let shared_words = WordArray {
        elem_ptr: shared_elem_ptr,
        variable: shared_var,
    };
    let word_load = |w: &mut Vec<u32>,
                     id: &mut u32,
                     late_globals: &mut Vec<u32>,
                     words: WordArray,
                     dst: Register,
                     address: Register,
                     offset: i32,
                     mem_type: LsSize2,
                     pred: Option<(u32, bool)>| {
        let addr = load(w, id, uint, regs[address.0 as usize]);
        let off = fresh(id);
        inst(late_globals, Op::Constant, &[uint, off, offset as u32]);
        let WordArray { elem_ptr, variable } = words;
        let word_at = |w: &mut Vec<u32>, id: &mut u32, byte: u32| {
            let word = fresh(id);
            inst(w, Op::ShiftRightLogical, &[uint, word, byte, constants[&2]]);
            let ptr = fresh(id);
            inst(w, Op::AccessChain, &[elem_ptr, ptr, variable, word]);
            let value = fresh(id);
            inst(w, Op::Load, &[uint, value, ptr]);
            value
        };
        let byte = fresh(id);
        inst(w, Op::IAdd, &[uint, byte, addr, off]);
        let first = word_at(w, id, byte);
        let value = if let Some((width, signed)) = mem_type.subword() {
            // Ryujinx InstEmitMemory.cs:552-574: shift then extend.
            // Even unaligned half loads at byte3 have safe shifts.
            let byte_in_word = alu(w, id, Op::BitwiseAnd, uint, &[byte, constants[&3]]);
            let shift = alu(
                w,
                id,
                Op::ShiftLeftLogical,
                uint,
                &[byte_in_word, constants[&3]],
            );
            let shifted = alu(w, id, Op::ShiftRightLogical, uint, &[first, shift]);
            alu(
                w,
                id,
                if signed {
                    Op::BitFieldSExtract
                } else {
                    Op::BitFieldUExtract
                },
                uint,
                &[shifted, zero, constants[&width]],
            )
        } else {
            first
        };
        put(w, id, regs[dst.0 as usize], value, pred, uint, boolt);
        for component in 1..mem_type.registers() {
            let next = fresh(id);
            inst(
                w,
                Op::IAdd,
                &[uint, next, byte, constants[&(u32::from(component) * 4)]],
            );
            let word = word_at(w, id, next);
            put(
                w,
                id,
                regs[(dst.0 as usize + component as usize).min(255)],
                word,
                pred,
                uint,
                boolt,
            );
        }
    };
    // A local subword store merges with a plain load and store.
    let local_merge = |w: &mut Vec<u32>, id: &mut u32, ptr: u32, clear: u32, shifted: u32| {
        let old = fresh(id);
        inst(w, Op::Load, &[uint, old, ptr]);
        let kept = fresh(id);
        inst(w, Op::BitwiseAnd, &[uint, kept, old, clear]);
        let merged = fresh(id);
        inst(w, Op::BitwiseOr, &[uint, merged, kept, shifted]);
        inst(w, Op::Store, &[ptr, merged]);
    };
    // Ryujinx HelperFunctionManager.cs:456-469 implements shared subword
    // stores with CAS so adjacent byte/half writers cannot lose each
    // other's changes. Use an atomic load and hoist the invariant
    // mask/shift.
    let shared_merge = |w: &mut Vec<u32>, id: &mut u32, ptr: u32, clear: u32, shifted: u32| {
        let head = fresh(id);
        let body = fresh(id);
        let cont = fresh(id);
        let merge = fresh(id);
        inst(w, Op::Branch, &[head]);
        inst(w, Op::Label, &[head]);
        inst(w, Op::LoopMerge, &[merge, cont, 0]);
        inst(w, Op::Branch, &[body]);
        inst(w, Op::Label, &[body]);
        let old = fresh(id);
        // OpAtomicLoad227, Workgroup2, Relaxed0;
        // spirv.hpp:1978-1981, Ryujinx Instructions.cs:1639-1642.
        inst(w, Op::AtomicLoad, &[uint, old, ptr, constants[&2], zero]);
        let kept = fresh(id);
        inst(w, Op::BitwiseAnd, &[uint, kept, old, clear]);
        let merged = fresh(id);
        inst(w, Op::BitwiseOr, &[uint, merged, kept, shifted]);
        let previous = alu(
            w,
            id,
            Op::AtomicCompareExchange,
            uint,
            &[ptr, constants[&2], zero, zero, merged, old],
        );
        let failed = alu(w, id, Op::INotEqual, boolt, &[previous, old]);
        inst(w, Op::Branch, &[cont]);
        inst(w, Op::Label, &[cont]);
        inst(w, Op::BranchConditional, &[failed, head, merge]);
        inst(w, Op::Label, &[merge]);
    };
    let word_store = |w: &mut Vec<u32>,
                      id: &mut u32,
                      late_globals: &mut Vec<u32>,
                      words: WordArray,
                      merge: SubwordMerge,
                      data: Register,
                      address: Register,
                      offset: i32,
                      mem_type: LsSize2| {
        let addr = load(w, id, uint, regs[address.0 as usize]);
        let off = fresh(id);
        inst(late_globals, Op::Constant, &[uint, off, offset as u32]);
        let WordArray { elem_ptr, variable } = words;
        let byte = fresh(id);
        inst(w, Op::IAdd, &[uint, byte, addr, off]);
        let store_word = |w: &mut Vec<u32>, id: &mut u32, byte: u32, value: u32| {
            let word = fresh(id);
            inst(w, Op::ShiftRightLogical, &[uint, word, byte, constants[&2]]);
            let ptr = fresh(id);
            inst(w, Op::AccessChain, &[elem_ptr, ptr, variable, word]);
            inst(w, Op::Store, &[ptr, value]);
        };
        let value = load(w, id, uint, regs[data.0 as usize]);
        if mem_type == LsSize2::B32 {
            store_word(w, id, byte, value);
        } else if let Some((width, _)) = mem_type.subword() {
            let aligned = fresh(id);
            inst(w, Op::BitwiseAnd, &[uint, aligned, byte, constants[&!3u32]]);
            let word = fresh(id);
            inst(
                w,
                Op::ShiftRightLogical,
                &[uint, word, aligned, constants[&2]],
            );
            let ptr = fresh(id);
            inst(w, Op::AccessChain, &[elem_ptr, ptr, variable, word]);

            let byte_in_word = fresh(id);
            inst(
                w,
                Op::BitwiseAnd,
                &[uint, byte_in_word, byte, constants[&3]],
            );
            let shift = fresh(id);
            inst(
                w,
                Op::ShiftLeftLogical,
                &[uint, shift, byte_in_word, constants[&3]],
            );
            let mask = (1u32 << width) - 1;
            let bits = fresh(id);
            let mask_value = late_constant(late_globals, id, uint, mask);
            inst(w, Op::BitwiseAnd, &[uint, bits, value, mask_value]);
            let shifted = fresh(id);
            inst(w, Op::ShiftLeftLogical, &[uint, shifted, bits, shift]);
            let clear = fresh(id);
            // Clear exactly the inserted bits (Ryujinx
            // InstEmitMemory.cs:579-600, BitFieldInsert).
            let shifted_mask = alu(w, id, Op::ShiftLeftLogical, uint, &[mask_value, shift]);
            inst(w, Op::Not, &[uint, clear, shifted_mask]);
            // Each space merges the subword its own way.
            merge(w, id, ptr, clear, shifted);
        } else {
            for component in 0..mem_type.registers() {
                let address = if component == 0 {
                    byte
                } else {
                    let next = fresh(id);
                    inst(
                        w,
                        Op::IAdd,
                        &[uint, next, byte, constants[&(u32::from(component) * 4)]],
                    );
                    next
                };
                // Ryujinx InstEmitMemory.cs:419-421: past RZ, the first
                // data register again.
                let component_value = load(
                    w,
                    id,
                    uint,
                    regs[usize::from(memory::stored_register(data, component).0)],
                );
                store_word(w, id, address, component_value);
            }
        }
    };
    let cases_at = w.len();
    let mut reachable = false;
    for (index, x) in xs.iter().enumerate() {
        let is_block_start = index == 0 || block_starts.contains(&x.address);
        if is_block_start {
            if reachable {
                // Fall through into this block.
                inst(
                    &mut w,
                    Op::Store,
                    &[pc_var, constants[&case_of[&x.address]]],
                );
                inst(&mut w, Op::Branch, &[switch_merge]);
            }
            inst(&mut w, Op::Label, &[labels[&x.address]]);
            reachable = true;
        }
        if x.kind
            .translation(DeadPadding {
                w: &mut w,
                reachable: &mut reachable,
            })
            .is_break()
        {
            continue;
        }
        if !reachable {
            continue;
        }
        let mut pred = x.predicate.map(|p| (ps[p.register as usize], p.inverted));
        // Ryujinx Translation/Translator.cs:339-385 guards the instruction
        // itself. Inactive accesses must neither dereference nor store.
        let memory_merge = pred
            .take_if(|_| x.kind.guards_whole())
            .map(|(predicate, inverted)| {
                let condition = load(&mut w, &mut id, boolt, predicate);
                let condition = alu_when(
                    &mut w,
                    &mut id,
                    inverted,
                    Op::LogicalNot,
                    boolt,
                    &[condition],
                    condition,
                );
                let body = fresh(&mut id);
                let merge = fresh(&mut id);
                inst(&mut w, Op::SelectionMerge, &[merge, 0]);
                inst(&mut w, Op::BranchConditional, &[condition, body, merge]);
                inst(&mut w, Op::Label, &[body]);
                merge
            });
        // A static ALD's lanes through the stage interface (Ryujinx
        // InstEmitAttribute.cs:34-41 stops at the RZ lane), then the
        // guard's merge.
        macro_rules! stage_lanes {
            ($io:expr, $dst:expr, $address:expr, $lanes:expr, $patch:expr, $vertex:expr) => {{
                let (io, dst, address, lanes, vertex) = ($io, $dst, $address, $lanes, $vertex);
                for (c, (a, lane)) in lanes.iter(address).enumerate() {
                    let destination = usize::from(dst.0) + c;
                    if destination >= 255 {
                        break;
                    }
                    // Adapted from Ryujinx AttributeMap.cs:129-141 under
                    // the MIT notice above: a static lane with no
                    // attribute in this stage logs and reads 0.
                    if !lane.reads_in(stage) {
                        eprintln!("shadowbox: {}", lane.zero_reason(a, stage));
                        put(&mut w, &mut id, regs[destination], zero, pred, uint, boolt);
                        continue;
                    }
                    let (pointer, ty) = io.pointer(
                        &mut w,
                        &mut id,
                        a,
                        $patch,
                        lane.bank,
                        vertex,
                        None,
                        &regs,
                        late_globals,
                    )?;
                    let value = load(&mut w, &mut id, ty, pointer);
                    let value = alu_when(
                        &mut w,
                        &mut id,
                        ty == float,
                        Op::Bitcast,
                        uint,
                        &[value],
                        value,
                    );
                    put(&mut w, &mut id, regs[destination], value, pred, uint, boolt);
                }
                for merge in memory_merge.into_iter() {
                    inst(&mut w, Op::Branch, &[merge]);
                    inst(&mut w, Op::Label, &[merge]);
                }
                continue;
            }};
        }
        // A physical ALD through the stage interface: one generic word at
        // the byte address in Ra (InstEmitAttribute.cs:43-51).
        macro_rules! stage_physical {
            ($io:expr, $dst:expr, $bank:expr, $vertex:expr, $offset:expr) => {{
                let (io, dst, vertex) = ($io, $dst, $vertex);
                if dst.0 != 255 {
                    let (pointer, ty) = io.pointer(
                        &mut w,
                        &mut id,
                        0,
                        false,
                        $bank,
                        vertex,
                        Some($offset),
                        &regs,
                        late_globals,
                    )?;
                    let value = load(&mut w, &mut id, ty, pointer);
                    let value = alu_when(
                        &mut w,
                        &mut id,
                        ty == float,
                        Op::Bitcast,
                        uint,
                        &[value],
                        value,
                    );
                    put(
                        &mut w,
                        &mut id,
                        regs[dst.0 as usize],
                        value,
                        pred,
                        uint,
                        boolt,
                    );
                }
                for merge in memory_merge.into_iter() {
                    inst(&mut w, Op::Branch, &[merge]);
                    inst(&mut w, Op::Label, &[merge]);
                }
                continue;
            }};
        }
        // Mesa nak_nir_lower_vtg_io.c:10-49: Rb of a tessellation control
        // ALD.O is lane - invocation + vertex, so the control point is
        // Rb - lane + invocation, snapshot before any destination is
        // written (Ryujinx InstEmitAttribute.cs:32). LANEID is
        // SubgroupLocalInvocationId here as for S2R. Elsewhere there are
        // no lane-indexed outputs and Rb is not read (AttributeMap.cs:
        // 323-328).
        macro_rules! control_point {
            ($rb:expr) => {
                control_points.map(|(lane, invocation)| {
                    let rb = load(&mut w, &mut id, uint, regs[usize::from($rb.0)]);
                    let lane = load(&mut w, &mut id, uint, lane);
                    let invocation = load(&mut w, &mut id, uint, invocation);
                    let first = alu(&mut w, &mut id, Op::ISub, uint, &[rb, lane]);
                    alu(&mut w, &mut id, Op::IAdd, uint, &[first, invocation])
                })
            };
        }
        // A static ALD's lanes in vertex, fragment and compute.
        macro_rules! fixed_lanes {
            ($dst:expr, $address:expr, $lanes:expr) => {{
                let (dst, address, lanes) = ($dst, $address, $lanes);
                // Static ALD in vertex, fragment and compute. Ryujinx
                // AttributeMap.cs:323-333 gives these stages no primitive
                // vertex, so the Rb vertex operand is ignored. Each lane
                // was decoded once (InstEmitAttribute.cs:34-55 advances the
                // byte address per lane); Position and the 32 generics are
                // vec4 variables (AttributeMap.cs:67-68, GetIoVariable
                // :297-304 maps fragment Position to FragCoord). Read each
                // touched vec4 once, rather than emitting the reference's
                // scalar load per lane, with identical component bits.
                let mut cached = None;
                for (c, (a, lane)) in lanes.iter(address).enumerate() {
                    // InstEmitAttribute.cs:34-41 stops at the RZ lane.
                    let destination = usize::from(dst.0) + c;
                    if destination >= 255 {
                        break;
                    }
                    let value = match lane.attribute {
                        // Adapted from Ryujinx AttributeMap.cs:129-141 under
                        // the MIT notice above: a lane with no attribute in
                        // this stage logs and reads 0.
                        _ if !lane.reads_in(stage) => {
                            eprintln!("shadowbox: {}", lane.zero_reason(a, stage));
                            zero
                        }
                        crate::ir::Attribute::InstanceId | crate::ir::Attribute::VertexId => {
                            let input = system_inputs.get(&a).copied().ok_or_else(|| bad(x))?;
                            load(&mut w, &mut id, uint, input)
                        }
                        crate::ir::Attribute::Position | crate::ir::Attribute::Generic => {
                            let position = lane.attribute == crate::ir::Attribute::Position;
                            let location = if position { 0 } else { (a - 0x80) / 16 };
                            let key = (position, location);
                            let vector = if let Some((_, value)) =
                                cached.filter(|(cached_key, _)| *cached_key == key)
                            {
                                value
                            } else {
                                let variable = if lane.bank == AttributeBank::Output {
                                    if position {
                                        output
                                    } else {
                                        vertex_outputs
                                            .iter()
                                            .find(|(l, _)| *l == u32::from(location))
                                            .map(|(_, v)| *v)
                                            .ok_or_else(|| bad(x))?
                                    }
                                } else if position {
                                    frag_coord.ok_or_else(|| bad(x))?
                                } else if has_dynamic_input {
                                    alu(
                                        &mut w,
                                        &mut id,
                                        Op::AccessChain,
                                        input_array_elem_ptr,
                                        &[
                                            dynamic_input.ok_or_else(|| bad(x))?,
                                            constants[&u32::from(location)],
                                        ],
                                    )
                                } else {
                                    input_by_location
                                        .get(&location)
                                        .copied()
                                        .ok_or_else(|| bad(x))?
                                };
                                let value = load(&mut w, &mut id, vec4, variable);
                                cached = Some((key, value));
                                value
                            };
                            let component = if position {
                                (a - 0x70) / 4
                            } else {
                                (a - 0x80) % 16 / 4
                            };
                            let value = alu(
                                &mut w,
                                &mut id,
                                Op::CompositeExtract,
                                float,
                                &[vector, u32::from(component)],
                            );
                            alu(&mut w, &mut id, Op::Bitcast, uint, &[value])
                        }
                        // Attributes Ryujinx maps in this stage that have no
                        // declaration here yet.
                        _ => return Err(bad(x)),
                    };
                    put(&mut w, &mut id, regs[destination], value, pred, uint, boolt);
                }
            }};
        }
        // The condition-code writer for this instruction (emitter/flags.rs).
        macro_rules! cc_writer {
            () => {
                CcWriter {
                    w: &mut w,
                    id: &mut id,
                    uint,
                    sint: signed_int,
                    boolt,
                    zero,
                    one: constants[&1],
                    shift31: constants[&31],
                    flags: CcFlags {
                        zero: cc_zero,
                        sign: cc_sign,
                        carry: cc_carry,
                        overflow: cc_overflow,
                    },
                    pred,
                }
            };
        }
        match x.kind.legacy_form() {
            InstructionKind::Vote(ops::Vote {
                op,
                ballot,
                vote,
                pred: source_pred,
            }) => {
                let source = load(&mut w, &mut id, boolt, ps[source_pred.register as usize]);
                let source = alu_when(
                    &mut w,
                    &mut id,
                    source_pred.inverted,
                    Op::LogicalNot,
                    boolt,
                    &[source],
                    source,
                );
                let lane = load(&mut w, &mut id, uint, lane_id.unwrap());
                let result = control::WarpEmitter {
                    w: &mut w,
                    id: &mut id,
                    uint,
                    boolt,
                    uint_vec4,
                    subgroup,
                    yes,
                    zero,
                    constants,
                }
                .vote(op, source, lane);
                put(
                    &mut w,
                    &mut id,
                    ps[vote as usize],
                    result.vote,
                    pred,
                    boolt,
                    boolt,
                );
                put(
                    &mut w,
                    &mut id,
                    regs[ballot.0 as usize],
                    result.ballot,
                    pred,
                    uint,
                    boolt,
                );
            }
            InstructionKind::Shuffle(ops::Shuffle {
                dst,
                src,
                lane,
                mode,
                control,
                in_bounds,
            }) => {
                let value = load(&mut w, &mut id, uint, regs[src.0 as usize]);
                let index = source(
                    &mut w, &mut id, lane, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let control = source(
                    &mut w, &mut id, control, &regs, uint, constants, cb_types, cb_vars, vec4,
                    float, zero,
                );
                let physical_lane = load(&mut w, &mut id, uint, lane_id.unwrap());
                let (result, valid) = emit_shuffle(
                    &mut w,
                    &mut id,
                    value,
                    index,
                    control,
                    physical_lane,
                    mode,
                    (uint, boolt, subgroup),
                    constants,
                );
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    result,
                    pred,
                    uint,
                    boolt,
                );
                // PT discards the in-bounds result.
                put(
                    &mut w,
                    &mut id,
                    ps[in_bounds as usize],
                    valid,
                    pred,
                    boolt,
                    boolt,
                );
            }
            InstructionKind::SwizzleAdd(ops::SwizzleAdd {
                dst,
                src,
                mask,
                ftz,
                flags,
            }) => {
                // Mesa from_nir.rs:2463-2492,2524-2553 emits neighbor A,
                // original B and masks99/a5 for positive fine derivatives.
                // Thus coefficients for A are [1,1,-1,0], B [1,-1,1,1].
                // Ryujinx Translation/Optimizations/Optimizer.cs:240-286 agrees through its derivative
                // peephole. Its fallback CodeGen/Spirv/Instructions.cs:1150-1151
                // reverses these two subtraction modes, giving -1 for
                // dFdx(gl_FragCoord.x). The exact quad fixture proves why
                // we follow Mesa's lowering for these modes.
                // Negation replaces multiplication by -1 and identity replaces
                // multiplication by +1. Keep A*0 for mode3: NaNs and signed
                // zero make deleting it incorrect. This avoids two dynamic
                // vector lookups and the nonzero coefficient multiplications.
                let lane = load(&mut w, &mut id, uint, lane_id.unwrap());
                let quad = alu(
                    &mut w,
                    &mut id,
                    Op::BitwiseAnd,
                    uint,
                    &[lane, constants[&3]],
                );
                let shift = alu(
                    &mut w,
                    &mut id,
                    Op::ShiftLeftLogical,
                    uint,
                    &[quad, constants[&1]],
                );
                let op = alu(
                    &mut w,
                    &mut id,
                    Op::ShiftRightLogical,
                    uint,
                    &[constants[&u32::from(mask)], shift],
                );
                let op = alu(&mut w, &mut id, Op::BitwiseAnd, uint, &[op, constants[&3]]);
                let mut values = [0; 2];
                for (i, reg) in src.iter().enumerate() {
                    let bits = source(
                        &mut w,
                        &mut id,
                        Source::Register(*reg),
                        &regs,
                        uint,
                        constants,
                        cb_types,
                        cb_vars,
                        vec4,
                        float,
                        zero,
                    );
                    // Mesa sm50.rs:942, ir.rs:8952: instruction FTZ.
                    // flush_f32 preserves the sign and leaves NaN/Inf intact.
                    let bits = FLUSH_F32[usize::from(ftz)](
                        &mut w,
                        &mut id,
                        bits,
                        (uint, boolt),
                        constants,
                    );
                    let value = alu(&mut w, &mut id, Op::Bitcast, float, &[bits]);
                    let negate = alu(
                        &mut w,
                        &mut id,
                        Op::IEqual,
                        boolt,
                        &[op, constants[&(2 - i as u32)]],
                    );
                    let negative = alu(&mut w, &mut id, Op::FNegate, float, &[value]);
                    let selected = alu(
                        &mut w,
                        &mut id,
                        Op::Select,
                        float,
                        &[negate, negative, value],
                    );
                    values[i] = SWIZZLE_LANE_TAIL[i](
                        &mut w,
                        &mut id,
                        annotations,
                        (boolt, float),
                        [op, constants[&3], zero],
                        value,
                        selected,
                    );
                }
                let sum = alu(&mut w, &mut id, Op::FAdd, float, &values);
                inst(annotations, Op::Decorate, &[sum, 42]);
                let result = alu(&mut w, &mut id, Op::Bitcast, uint, &[sum]);
                let result =
                    FLUSH_F32[usize::from(ftz)](&mut w, &mut id, result, (uint, boolt), constants);
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    result,
                    pred,
                    uint,
                    boolt,
                );
                // Ryujinx InstEmitWarp.cs:20: SetFPZnFlags on the written
                // result.
                if flags == FloatFlags::Written {
                    let value = alu(&mut w, &mut id, Op::Bitcast, float, &[result]);
                    let float_zero = late_constant(late_globals, &mut id, float, 0);
                    cc_writer!().set_fp_zn(value, float_zero);
                }
            }
            InstructionKind::Mov(ops::Mov {
                dst,
                src: s,
                quad_lanes,
            }) => {
                let v = source(
                    &mut w, &mut id, s, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                // Mesa sm50.rs:1936,1941,1946 and ir.rs:4963-4972 retain
                // one enable per quad position. Mesa codegen
                // nv50_ir_lowering_nvc0.cpp:1290-1303 sets bit l for lane l. Ryujinx InstEmitMove.cs:12-37
                // copies raw bits but omits its decoded QuadMask. Keep the
                // old destination where that enable is clear. Constant masks
                // 0/F fold here; partial masks use one shift and one bit test.
                let v = match quad_lanes {
                    0 => load(&mut w, &mut id, uint, regs[dst.0 as usize]),
                    15 => v,
                    mask => {
                        let lane = load(&mut w, &mut id, uint, lane_id.unwrap());
                        let position = alu(
                            &mut w,
                            &mut id,
                            Op::BitwiseAnd,
                            uint,
                            &[lane, constants[&3]],
                        );
                        let enables = alu(
                            &mut w,
                            &mut id,
                            Op::ShiftRightLogical,
                            uint,
                            &[constants[&u32::from(mask)], position],
                        );
                        let bit = alu(
                            &mut w,
                            &mut id,
                            Op::BitwiseAnd,
                            uint,
                            &[enables, constants[&1]],
                        );
                        let enabled = alu(&mut w, &mut id, Op::INotEqual, boolt, &[bit, zero]);
                        let old = load(&mut w, &mut id, uint, regs[dst.0 as usize]);
                        alu(&mut w, &mut id, Op::Select, uint, &[enabled, v, old])
                    }
                };
                put(&mut w, &mut id, regs[dst.0 as usize], v, pred, uint, boolt)
            }
            InstructionKind::IAdd(ops::IAdd {
                dst,
                lhs,
                rhs,
                cc,
                x,
                negate,
            }) => {
                let mut a = load(&mut w, &mut id, uint, regs[lhs.0 as usize]);
                let mut b = source(
                    &mut w, &mut id, rhs, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                a = alu_when(&mut w, &mut id, negate[0], Op::ISub, uint, &[zero, a], a);
                b = alu_when(&mut w, &mut id, negate[1], Op::ISub, uint, &[zero, b], b);
                let v = id;
                id += 1;
                let carry = if x || cc {
                    // Mesa's OpIAdd2X folds the carry-in into the unsigned sum and
                    // sets carry_out on sum >= 2^32 (ir.rs:3846-3857). OpIAddCarry
                    // (149) takes two operands, so a carry-in is a second add;
                    // OR of the two carry outputs gives the final carry-out.
                    let pair = id;
                    id += 1;
                    inst(&mut w, Op::IAddCarry, &[iadd_carry_type, pair, a, b]);
                    let carry_u = id;
                    id += 1;
                    if x {
                        let carry_in = load(&mut w, &mut id, boolt, cc_carry);
                        let low = id;
                        let carry_in_u = id + 1;
                        let pair2 = id + 2;
                        let carry1 = id + 3;
                        let carry2 = id + 4;
                        id += 5;
                        inst(&mut w, Op::CompositeExtract, &[uint, low, pair, 0]);
                        inst(
                            &mut w,
                            Op::Select,
                            &[uint, carry_in_u, carry_in, constants[&1], zero],
                        );
                        inst(
                            &mut w,
                            Op::IAddCarry,
                            &[iadd_carry_type, pair2, low, carry_in_u],
                        );
                        inst(&mut w, Op::CompositeExtract, &[uint, v, pair2, 0]);
                        inst(&mut w, Op::CompositeExtract, &[uint, carry1, pair, 1]);
                        inst(&mut w, Op::CompositeExtract, &[uint, carry2, pair2, 1]);
                        inst(&mut w, Op::BitwiseOr, &[uint, carry_u, carry1, carry2]);
                    } else {
                        inst(&mut w, Op::CompositeExtract, &[uint, v, pair, 0]);
                        inst(&mut w, Op::CompositeExtract, &[uint, carry_u, pair, 1]);
                    }
                    let carry = id;
                    id += 1;
                    inst(&mut w, Op::INotEqual, &[boolt, carry, carry_u, zero]);
                    Some(carry)
                } else {
                    inst(&mut w, Op::IAdd, &[uint, v, a, b]);
                    None
                };
                put(&mut w, &mut id, regs[dst.0 as usize], v, pred, uint, boolt);
                if cc {
                    // Ryujinx InstEmitAluHelper.cs:102-118: extended Z
                    // accumulates zero flags of all earlier words.
                    let previous_zero = x.then(|| load(&mut w, &mut id, boolt, cc_zero));
                    put(
                        &mut w,
                        &mut id,
                        cc_carry,
                        carry.unwrap(),
                        pred,
                        boolt,
                        boolt,
                    );
                    write_cc(
                        &mut w,
                        &mut id,
                        v,
                        pred,
                        cc_zero,
                        cc_sign,
                        uint,
                        boolt,
                        zero,
                        constants[&31],
                    );
                    // Keep the semantic-review chain and predicated writes;
                    // Ryujinx InstEmitAluHelper.cs:102-114 defines .X zero.
                    for previous_zero in previous_zero.into_iter() {
                        let this_zero = load(&mut w, &mut id, boolt, cc_zero);
                        let combined = alu(
                            &mut w,
                            &mut id,
                            Op::LogicalAnd,
                            boolt,
                            &[this_zero, previous_zero],
                        );
                        put(&mut w, &mut id, cc_zero, combined, pred, boolt, boolt);
                    }
                    write_iadd_overflow(
                        &mut w,
                        &mut id,
                        v,
                        a,
                        b,
                        pred,
                        cc_overflow,
                        signed_int,
                        boolt,
                        zero,
                        constants[&31],
                    );
                }
            }
            InstructionKind::Xmad(ops::Xmad {
                dst,
                a,
                b,
                c,
                signed,
                high,
                psl,
                mrg,
                cmode,
                cc,
            }) => {
                let a_full = load(&mut w, &mut id, uint, regs[a.0 as usize]);
                let b_full = source(
                    &mut w, &mut id, b, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let c_full = source(
                    &mut w, &mut id, c, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                // Select a 16-bit half and widen it: shift the half to the
                // top, then shift back arithmetically (signed) or logically.
                let mut half = |w: &mut Vec<u32>, value: u32, high: bool, signed: bool| {
                    let (op, amount) = XMAD_HALF[usize::from(high)];
                    let top = alu(w, &mut id, op, uint, &[value, constants[&amount]]);
                    alu(
                        w,
                        &mut id,
                        XMAD_WIDEN[usize::from(signed)],
                        uint,
                        &[[top, value][usize::from(high)], constants[&16]],
                    )
                };
                let ah = half(&mut w, a_full, high[0], signed[0]);
                let bh = half(&mut w, b_full, high[1], signed[1]);
                let product = alu(&mut w, &mut id, Op::IMul, uint, &[ah, bh]);
                let product = alu_when(
                    &mut w,
                    &mut id,
                    psl,
                    Op::ShiftLeftLogical,
                    uint,
                    &[product, constants[&16]],
                    product,
                );
                let addend = match cmode {
                    XmadCop::Cfull => c_full,
                    XmadCop::Clo => {
                        let v = fresh(&mut id);
                        inst(
                            &mut w,
                            Op::BitwiseAnd,
                            &[uint, v, c_full, constants[&0xffff]],
                        );
                        v
                    }
                    XmadCop::Chi => {
                        let v = fresh(&mut id);
                        inst(
                            &mut w,
                            Op::ShiftRightLogical,
                            &[uint, v, c_full, constants[&16]],
                        );
                        v
                    }
                    XmadCop::Csfu => {
                        // Ryujinx InstEmitIntegerArithmetic.cs:629-634: the
                        // sign bit of each widened half, moved to bit 16.
                        let mut adjust = |w: &mut Vec<u32>, widened: u32| {
                            let sign = fresh(&mut id);
                            inst(
                                w,
                                Op::ShiftRightLogical,
                                &[uint, sign, widened, constants[&31]],
                            );
                            let moved = fresh(&mut id);
                            inst(
                                w,
                                Op::ShiftLeftLogical,
                                &[uint, moved, sign, constants[&16]],
                            );
                            moved
                        };
                        let sa = adjust(&mut w, ah);
                        let sb = adjust(&mut w, bh);
                        let both = fresh(&mut id);
                        inst(&mut w, Op::IAdd, &[uint, both, sa, sb]);
                        let v = fresh(&mut id);
                        inst(&mut w, Op::ISub, &[uint, v, c_full, both]);
                        v
                    }
                    XmadCop::Cbcc => {
                        let shifted = fresh(&mut id);
                        inst(
                            &mut w,
                            Op::ShiftLeftLogical,
                            &[uint, shifted, b_full, constants[&16]],
                        );
                        let v = fresh(&mut id);
                        inst(&mut w, Op::IAdd, &[uint, v, c_full, shifted]);
                        v
                    }
                };
                let result = alu(&mut w, &mut id, Op::IAdd, uint, &[product, addend]);
                if cc {
                    // Ryujinx InstEmitIntegerArithmetic.cs:261-274,665-699:
                    // SetIaddFlags(sum, product, addend) before MRG.
                    cc_writer!().set_add(result, product, addend, Words::Single);
                }
                let result = XMAD_MERGE[usize::from(mrg)](
                    &mut w,
                    &mut id,
                    uint,
                    result,
                    b_full,
                    [constants[&0xffff], constants[&16]],
                );
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    result,
                    pred,
                    uint,
                    boolt,
                )
            }
            InstructionKind::Ldc(ops::Ldc {
                dst,
                binding,
                offset,
                immediate,
                mem_type,
                address_mode,
            }) => {
                macro_rules! emit_ldc_result {
                    ($word_at:ident, $address:ident) => {
                        match mem_type {
                            LdcWidth::B32 | LdcWidth::B64 => {
                                let first = $word_at(&mut w, &mut id, $address);
                                put(
                                    &mut w,
                                    &mut id,
                                    regs[dst.0 as usize],
                                    first,
                                    pred,
                                    uint,
                                    boolt,
                                );
                                std::iter::repeat_n((), LDC_SHAPE[mem_type as usize].2).for_each(
                                    |()| {
                                        let next = alu(
                                            &mut w,
                                            &mut id,
                                            Op::IAdd,
                                            uint,
                                            &[$address, constants[&4]],
                                        );
                                        let second = $word_at(&mut w, &mut id, next);
                                        put(
                                            &mut w,
                                            &mut id,
                                            regs[(dst.0 as usize + 1).min(255)],
                                            second,
                                            pred,
                                            uint,
                                            boolt,
                                        );
                                    },
                                );
                            }
                            LdcWidth::U8 | LdcWidth::S8 | LdcWidth::U16 | LdcWidth::S16 => {
                                let aligned = fresh(&mut id);
                                inst(
                                    &mut w,
                                    Op::BitwiseAnd,
                                    &[
                                        uint,
                                        aligned,
                                        $address,
                                        late_constant(late_globals, &mut id, uint, !3),
                                    ],
                                );
                                let word = $word_at(&mut w, &mut id, aligned);
                                // Ryujinx InstEmitMemory.cs:97-100: the
                                // word's sub-word at the address's byte.
                                let (width, signed, _) = LDC_SHAPE[mem_type as usize];
                                let value = memory::extract_small_int(
                                    &mut w,
                                    &mut id,
                                    uint,
                                    zero,
                                    [constants[&3], constants[&width]],
                                    signed,
                                    word,
                                    $address,
                                );
                                put(
                                    &mut w,
                                    &mut id,
                                    regs[dst.0 as usize],
                                    value,
                                    pred,
                                    uint,
                                    boolt,
                                );
                            }
                        }
                    };
                }

                match address_mode {
                    crate::ir::LdcAddressMode::Default | crate::ir::LdcAddressMode::Il => {
                        // Keep this path byte-for-byte equivalent to the
                        // established static-bank emission order.
                        let (_, _, _, element_pointer) = cb_types[&binding];
                        let bank = cb_vars[&binding];
                        let base = load(&mut w, &mut id, uint, regs[offset.0 as usize]);
                        let imm = late_constant(late_globals, &mut id, uint, immediate as u32);
                        let address = fresh(&mut id);
                        inst(&mut w, Op::IAdd, &[uint, address, base, imm]);
                        if mem_type == LdcWidth::B32
                            && texture_array_lengths.contains_key(
                                &HandleSource::ConstantBufferArray {
                                    bank: binding,
                                    handle: 0,
                                },
                            )
                        {
                            // Ryujinx BindlessToArray.cs:214-221 turns the
                            // cbuf vec/component address into one packed
                            // handle index, which is exactly byte_address / 8.
                            let descriptor_index = fresh(&mut id);
                            inst(
                                &mut w,
                                Op::ShiftRightLogical,
                                &[uint, descriptor_index, address, constants[&3]],
                            );
                            let indices = texture_array_indices
                                .as_ref()
                                .expect("dynamic texture index variables");
                            put(
                                &mut w,
                                &mut id,
                                indices[dst.0 as usize],
                                descriptor_index,
                                pred,
                                uint,
                                boolt,
                            );
                        }
                        let word_at = |w: &mut Vec<u32>, id: &mut u32, byte: u32| {
                            let index = fresh(id);
                            inst(
                                w,
                                Op::ShiftRightLogical,
                                &[uint, index, byte, constants[&4]],
                            );
                            let ptr = fresh(id);
                            inst(
                                w,
                                Op::AccessChain,
                                &[element_pointer, ptr, bank, zero, index],
                            );
                            let row = fresh(id);
                            inst(w, Op::Load, &[vec4, row, ptr]);
                            let word_index = fresh(id);
                            inst(
                                w,
                                Op::ShiftRightLogical,
                                &[uint, word_index, byte, constants[&2]],
                            );
                            let component = fresh(id);
                            inst(
                                w,
                                Op::BitwiseAnd,
                                &[uint, component, word_index, constants[&3]],
                            );
                            let f = fresh(id);
                            inst(w, Op::VectorExtractDynamic, &[float, f, row, component]);
                            let value = fresh(id);
                            inst(w, Op::Bitcast, &[uint, value, f]);
                            value
                        };
                        emit_ldc_result!(word_at, address);
                    }
                    crate::ir::LdcAddressMode::IndexedSegmented
                    | crate::ir::LdcAddressMode::IndexedSegmentedLocal => {
                        let base = load(&mut w, &mut id, uint, regs[offset.0 as usize]);
                        let bank_delta = fresh(&mut id);
                        inst(
                            &mut w,
                            Op::ShiftRightLogical,
                            &[uint, bank_delta, base, constants[&16]],
                        );
                        let selected = fresh(&mut id);
                        inst(
                            &mut w,
                            Op::IAdd,
                            &[
                                uint,
                                selected,
                                bank_delta,
                                late_constant(late_globals, &mut id, uint, u32::from(binding)),
                            ],
                        );
                        let low = fresh(&mut id);
                        inst(
                            &mut w,
                            Op::BitwiseAnd,
                            &[uint, low, base, constants[&0xffff]],
                        );
                        let imm = late_constant(late_globals, &mut id, uint, immediate as u32);
                        let address = fresh(&mut id);
                        inst(&mut w, Op::IAdd, &[uint, address, low, imm]);
                        let bank_constants = (0_u32..32)
                            .map(|candidate| late_constant(late_globals, &mut id, uint, candidate))
                            .collect::<Vec<_>>();
                        let word_at = |w: &mut Vec<u32>, id: &mut u32, byte: u32| {
                            let index = fresh(id);
                            inst(
                                w,
                                Op::ShiftRightLogical,
                                &[uint, index, byte, constants[&4]],
                            );
                            let word_index = fresh(id);
                            inst(
                                w,
                                Op::ShiftRightLogical,
                                &[uint, word_index, byte, constants[&2]],
                            );
                            let component = fresh(id);
                            inst(
                                w,
                                Op::BitwiseAnd,
                                &[uint, component, word_index, constants[&3]],
                            );
                            let load_bank = |w: &mut Vec<u32>, id: &mut u32, binding: u8| {
                                let (_, _, _, element_pointer) = cb_types[&binding];
                                let bank = cb_vars[&binding];
                                let ptr = fresh(id);
                                inst(
                                    w,
                                    Op::AccessChain,
                                    &[element_pointer, ptr, bank, zero, index],
                                );
                                let row = fresh(id);
                                inst(w, Op::Load, &[vec4, row, ptr]);
                                let f = fresh(id);
                                inst(w, Op::VectorExtractDynamic, &[float, f, row, component]);
                                let value = fresh(id);
                                inst(w, Op::Bitcast, &[uint, value, f]);
                                value
                            };
                            // Ryujinx InstEmitMemory.cs:181-196: a
                            // selector naming no bank reads zero.
                            let mut value = zero;
                            for candidate in 0_u8..32 {
                                let is_candidate = fresh(id);
                                inst(
                                    w,
                                    Op::IEqual,
                                    &[
                                        boolt,
                                        is_candidate,
                                        selected,
                                        bank_constants[usize::from(candidate)],
                                    ],
                                );
                                let candidate_value = load_bank(w, id, candidate);
                                let chosen = fresh(id);
                                inst(
                                    w,
                                    Op::Select,
                                    &[uint, chosen, is_candidate, candidate_value, value],
                                );
                                value = chosen;
                            }
                            value
                        };
                        emit_ldc_result!(word_at, address);
                    }
                }
            }
            InstructionKind::ShiftAdd(ops::ShiftAdd {
                dst,
                lhs,
                rhs,
                shift,
                negate,
                cc,
            }) => {
                let a = load(&mut w, &mut id, uint, regs[lhs.0 as usize]);
                let b = source(
                    &mut w, &mut id, rhs, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let shifted = fresh(&mut id);
                inst(
                    &mut w,
                    Op::ShiftLeftLogical,
                    &[uint, shifted, a, constants[&u32::from(shift)]],
                );
                // Ryujinx InstEmitIntegerArithmetic.cs:469-476 negates the
                // shifted A / B before adding and uses modified A for CF.
                // Subtraction folds B's negation into the add. Without .CC,
                // swapping operands also folds A's negation into one subtract.
                let a = alu_when(
                    &mut w,
                    &mut id,
                    negate[0] & (cc | negate[1]),
                    Op::ISub,
                    uint,
                    &[zero, shifted],
                    shifted,
                );
                let (op, swap) =
                    SHIFT_ADD[usize::from(negate[1]) * 2 + usize::from(negate[0] & !cc)];
                let v = alu(&mut w, &mut id, op, uint, &[[a, b], [b, a]][swap]);
                put(&mut w, &mut id, regs[dst.0 as usize], v, pred, uint, boolt);
                if cc {
                    // CF = result < modified A (Ryujinx
                    // InstEmitIntegerArithmetic.cs:682-685). OpULessThan avoids building a carry struct.
                    let carry = fresh(&mut id);
                    inst(&mut w, Op::ULessThan, &[boolt, carry, v, a]);
                    put(&mut w, &mut id, cc_carry, carry, pred, boolt, boolt);
                    let flag_rhs =
                        alu_when(&mut w, &mut id, negate[1], Op::ISub, uint, &[zero, b], b);
                    // Ryujinx InstEmitIntegerArithmetic.cs:476,688-696:
                    // ISCADD writes overflow from the modified operands.
                    write_iadd_overflow(
                        &mut w,
                        &mut id,
                        v,
                        a,
                        flag_rhs,
                        pred,
                        cc_overflow,
                        signed_int,
                        boolt,
                        zero,
                        constants[&31],
                    );
                    write_cc(
                        &mut w,
                        &mut id,
                        v,
                        pred,
                        cc_zero,
                        cc_sign,
                        uint,
                        boolt,
                        zero,
                        constants[&31],
                    );
                }
            }
            InstructionKind::IAdd3(ops::IAdd3 {
                dst,
                lhs,
                rhs,
                addend,
                halves,
                shift,
                negate,
            }) => {
                let a = load(&mut w, &mut id, uint, regs[lhs.0 as usize]);
                let b = source(
                    &mut w, &mut id, rhs, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let c = load(&mut w, &mut id, uint, regs[addend.0 as usize]);
                let v = IntegerEmitter {
                    w: &mut w,
                    id: &mut id,
                    uint,
                    boolt,
                    constants,
                }
                .add3([a, b, c], halves, shift, negate);
                put(&mut w, &mut id, regs[dst.0 as usize], v, pred, uint, boolt);
            }
            InstructionKind::Lea(ops::Lea {
                dst,
                lhs,
                rhs,
                shift,
                negate,
            }) => {
                let a = load(&mut w, &mut id, uint, regs[lhs.0 as usize]);
                let b = source(
                    &mut w, &mut id, rhs, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                // Ryujinx InstEmitIntegerArithmetic.cs:481-488 shifts, negates
                // and adds. B - shifted A folds negation into the addition,
                // saving an instruction. Shift zero needs no shift instruction.
                let a = if shift == 0 {
                    a
                } else {
                    alu(
                        &mut w,
                        &mut id,
                        Op::ShiftLeftLogical,
                        uint,
                        &[a, constants[&u32::from(shift)]],
                    )
                };
                let value = alu(
                    &mut w,
                    &mut id,
                    [Op::IAdd, Op::ISub][usize::from(negate)],
                    uint,
                    &[[a, b], [b, a]][usize::from(negate)],
                );
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    value,
                    pred,
                    uint,
                    boolt,
                );
            }
            InstructionKind::IMadSp(ops::IMadSp {
                dst,
                a,
                b,
                c,
                types,
            }) => {
                let a = load(&mut w, &mut id, uint, regs[a.0 as usize]);
                let [b, c] = [b, c].map(|operand| {
                    source(
                        &mut w, &mut id, operand, &regs, uint, constants, cb_types, cb_vars, vec4,
                        float, zero,
                    )
                });
                let v = IntegerEmitter {
                    w: &mut w,
                    id: &mut id,
                    uint,
                    boolt,
                    constants,
                }
                .mad_sp([a, b, c], types);
                put(&mut w, &mut id, regs[dst.0 as usize], v, pred, uint, boolt);
            }
            InstructionKind::LeaHi(ops::LeaHi {
                dst,
                low,
                addend,
                high,
                shift,
                negate,
            }) => {
                let a = load(&mut w, &mut id, uint, regs[low.0 as usize]);
                let b = source(
                    &mut w, &mut id, addend, &regs, uint, constants, cb_types, cb_vars, vec4,
                    float, zero,
                );
                let c = load(&mut w, &mut id, uint, regs[high.0 as usize]);
                let v = IntegerEmitter {
                    w: &mut w,
                    id: &mut id,
                    uint,
                    boolt,
                    constants,
                }
                .lea_hi(a, c, b, shift, negate);
                put(&mut w, &mut id, regs[dst.0 as usize], v, pred, uint, boolt);
            }
            InstructionKind::FunnelShift(ops::FunnelShift {
                dst,
                low,
                high,
                shift,
                right,
                dst_high,
                ty,
                wrap,
                cc,
            }) => {
                let a = load(&mut w, &mut id, uint, regs[low.0 as usize]);
                let b = source(
                    &mut w, &mut id, shift, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let c = load(&mut w, &mut id, uint, regs[high.0 as usize]);
                let mut ints = IntegerEmitter {
                    w: &mut w,
                    id: &mut id,
                    uint,
                    boolt,
                    constants,
                };
                let v = if let Source::Immediate(count) = shift {
                    ints.funnel_immediate(a, c, count, right, dst_high, ty, wrap)
                } else {
                    ints.funnel(a, c, b, right, dst_high, ty, wrap)
                };
                put(&mut w, &mut id, regs[dst.0 as usize], v, pred, uint, boolt);
                if cc {
                    write_cc(
                        &mut w,
                        &mut id,
                        v,
                        pred,
                        cc_zero,
                        cc_sign,
                        uint,
                        boolt,
                        zero,
                        constants[&31],
                    );
                }
            }
            InstructionKind::BitFieldInsert(ops::BitFieldInsert {
                dst,
                insert,
                base,
                range,
            }) => {
                // Ryujinx Instructions/InstEmitBitfield.cs:154-162 and
                // CodeGen/Spirv/Instructions.cs:2155-2171 use the low two
                // control bytes at runtime. Fold proven constants to save
                // both extracts; otherwise retain one OpBitFieldInsert.
                // Mesa src/compiler/spirv/spirv.h:1922,1924,1926 defines
                // BitwiseAnd, BitFieldInsert and BitFieldUExtract.
                // Runtime out-of-range controls have no defined reference
                // result. Do not clamp, wrap, or fabricate one.
                let insert = load(&mut w, &mut id, uint, regs[insert.0 as usize]);
                let constant = match range {
                    Source::Immediate(control) => Some(control),
                    Source::Register(register) => reaching.constant(index, register.0),
                    Source::ConstantBuffer { .. } => None,
                };
                let (offset, count) = if let Some(control) = constant {
                    let offset = control & 255;
                    let count = (control >> 8) & 255;
                    if offset > 32 || count > 32 - offset {
                        return Err(bad(x));
                    }
                    (constants[&offset], constants[&count])
                } else {
                    let control = source(
                        &mut w, &mut id, range, &regs, uint, constants, cb_types, cb_vars, vec4,
                        float, zero,
                    );
                    let offset = alu(
                        &mut w,
                        &mut id,
                        Op::BitwiseAnd,
                        uint,
                        &[control, constants[&255]],
                    );
                    let count = alu(
                        &mut w,
                        &mut id,
                        Op::BitFieldUExtract,
                        uint,
                        &[control, constants[&8], constants[&8]],
                    );
                    (offset, count)
                };
                let base = source(
                    &mut w, &mut id, base, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let result = alu(
                    &mut w,
                    &mut id,
                    Op::BitFieldInsert,
                    uint,
                    &[base, insert, offset, count],
                );
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    result,
                    pred,
                    uint,
                    boolt,
                );
            }
            InstructionKind::BitFieldExtract(ops::BitFieldExtract {
                dst,
                base,
                range,
                signed,
                reverse,
            }) => {
                let a = load(&mut w, &mut id, uint, regs[base.0 as usize]);
                let b = source(
                    &mut w, &mut id, range, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let mut ints = IntegerEmitter {
                    w: &mut w,
                    id: &mut id,
                    uint,
                    boolt,
                    constants,
                };
                let v = if let Source::Immediate(range) = range {
                    ints.bfe_immediate(a, range, signed, reverse)
                } else {
                    ints.bfe(a, b, signed, reverse)
                };
                put(&mut w, &mut id, regs[dst.0 as usize], v, pred, uint, boolt);
            }
            InstructionKind::IntegerMul(ops::IntegerMul {
                dst,
                lhs,
                rhs,
                addend,
                high,
                signed,
                negate,
            }) => {
                // Ryujinx InstEmitIntegerArithmetic.cs:409-457 (EmitImad).
                let a = load(&mut w, &mut id, uint, regs[lhs.0 as usize]);
                let b = source(
                    &mut w, &mut id, rhs, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let c = source(
                    &mut w, &mut id, addend, &regs, uint, constants, cb_types, cb_vars, vec4,
                    float, zero,
                );
                let value = IntegerEmitter {
                    w: &mut w,
                    id: &mut id,
                    uint,
                    boolt,
                    constants,
                }
                .mul_add(
                    [a, b, c],
                    high,
                    signed,
                    negate,
                    addend == Source::Immediate(0),
                    iadd_carry_type,
                );
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    value,
                    pred,
                    uint,
                    boolt,
                )
            }
            InstructionKind::FAdd(ops::FAdd {
                dst,
                lhs,
                rhs,
                modifiers,
                round,
                flags,
            }) => {
                let a = load(&mut w, &mut id, uint, regs[lhs.0 as usize]);
                let b = source(
                    &mut w, &mut id, rhs, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let v = modifiers.sum(
                    &mut FloatOps {
                        w: &mut w,
                        id: &mut id,
                        uint,
                        float,
                        boolt,
                        glsl_std_450,
                        constants,
                        scale_factors,
                        annotations,
                        late_globals,
                    },
                    a,
                    b,
                    round,
                    binary64,
                );
                put(&mut w, &mut id, regs[dst.0 as usize], v, pred, uint, boolt);
                if flags == FloatFlags::Written {
                    let result = alu(&mut w, &mut id, Op::Bitcast, float, &[v]);
                    let float_zero = late_constant(late_globals, &mut id, float, 0);
                    cc_writer!().set_fp_zn(result, float_zero);
                }
            }
            InstructionKind::FMul(ops::FMul {
                dst,
                lhs,
                rhs,
                modifiers,
                flags,
            }) => {
                let a = load(&mut w, &mut id, uint, regs[lhs.0 as usize]);
                let b = source(
                    &mut w, &mut id, rhs, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let v = modifiers.product(
                    &mut FloatOps {
                        w: &mut w,
                        id: &mut id,
                        uint,
                        float,
                        boolt,
                        glsl_std_450,
                        constants,
                        scale_factors,
                        annotations,
                        late_globals,
                    },
                    a,
                    b,
                    binary64,
                );
                put(&mut w, &mut id, regs[dst.0 as usize], v, pred, uint, boolt);
                if flags == FloatFlags::Written {
                    let result = alu(&mut w, &mut id, Op::Bitcast, float, &[v]);
                    let float_zero = late_constant(late_globals, &mut id, float, 0);
                    cc_writer!().set_fp_zn(result, float_zero);
                }
            }
            InstructionKind::FCmp(ops::FCmp {
                dst,
                when_true,
                when_false,
                test,
                compare: c,
                ftz,
            }) => {
                let a = load(&mut w, &mut id, uint, regs[when_true.0 as usize]);
                let b = source(
                    &mut w, &mut id, when_false, &regs, uint, constants, cb_types, cb_vars, vec4,
                    float, zero,
                );
                let test = source(
                    &mut w, &mut id, test, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                // Ryujinx selects the raw A/B words (InstEmitFloatComparison.cs:
                // 369-375). Its FcmpR/I/C/Rc paths (:163-205) omit FTZ. Mesa
                // nv50_ir_emit_gm107.cpp:1569,474-477 explicitly encodes FTZ;
                // flush C alone, preserving selected subnormal A/B payloads.
                // Mesa nv50_ir_peephole.cpp:2009-2014 replaces SLCT(a,a,C)
                // with MOV, independently confirming unmodified selected bits.
                // The existing integer flush avoids host denorm behavior.
                let test =
                    FLUSH_F32[usize::from(ftz)](&mut w, &mut id, test, (uint, boolt), constants);
                // Ryujinx InstEmitFloatComparison.cs:369-375. Copyright (c)
                // Ryujinx Team and Contributors; MIT notice above applies.
                // Compare only C; OpSelect on uint retains the chosen A/B's
                // exact bits.
                let condition =
                    float_compare(&mut w, &mut id, c, test, zero, (float, boolt, yes, no));
                let value = alu(&mut w, &mut id, Op::Select, uint, &[condition, a, b]);
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    value,
                    pred,
                    uint,
                    boolt,
                );
            }
            InstructionKind::ICmp(ops::ICmp {
                dst,
                when_true,
                when_false,
                test,
                compare: c,
                signedness,
            }) => {
                let a = load(&mut w, &mut id, uint, regs[when_true.0 as usize]);
                let b = source(
                    &mut w, &mut id, when_false, &regs, uint, constants, cb_types, cb_vars, vec4,
                    float, zero,
                );
                let test = source(
                    &mut w, &mut id, test, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                // Ryujinx InstEmitIntegerComparison.cs:117-131. Copyright (c)
                // Ryujinx Team and Contributors; MIT notice above applies.
                // Compare only C; OpSelect on uint retains the chosen A/B's
                // exact bits.
                let condition =
                    int_compare(&mut w, &mut id, c, signedness, test, zero, (boolt, yes, no));
                let value = alu(&mut w, &mut id, Op::Select, uint, &[condition, a, b]);
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    value,
                    pred,
                    uint,
                    boolt,
                );
            }
            InstructionKind::HalfArithmetic(ops::HalfArithmetic {
                op,
                flush,
                dst,
                src,
                swizzle,
                output,
                abs,
                negate,
                saturate,
            }) => {
                let types = (float, vec2, glsl_std_450);
                let mut pairs = [0; 3];
                let mut fmz_signs = [0; 2];
                for index in 0..if op == HalfOp::Fma { 3 } else { 2 } {
                    if let Source::Immediate(bits) = src[index] {
                        if swizzle[index] == 0 && !abs[index] && !negate[index] {
                            if let Some(parts) = half_immediate(bits) {
                                if flush == HalfFlush::Fmz && index < 2 {
                                    // Finite immediate signs fold without runtime operations.
                                    fmz_signs[index] = late_constant(
                                        late_globals,
                                        &mut id,
                                        uint,
                                        bits & 0x8000_8000,
                                    );
                                }
                                let parts = parts.map(|value| {
                                    late_constant(late_globals, &mut id, float, value)
                                });
                                pairs[index] =
                                    alu(&mut w, &mut id, Op::CompositeConstruct, vec2, &parts);
                                continue;
                            }
                        }
                    }
                    let bits = source(
                        &mut w, &mut id, src[index], &regs, uint, constants, cb_types, cb_vars,
                        vec4, float, zero,
                    );
                    if flush == HalfFlush::Fmz && index < 2 {
                        fmz_signs[index] = half_sign_bits(
                            &mut w,
                            &mut id,
                            bits,
                            swizzle[index],
                            (abs[index], negate[index]),
                            uint,
                            late_globals,
                        );
                    }
                    pairs[index] = half_pair(
                        &mut w,
                        &mut id,
                        bits,
                        swizzle[index],
                        abs[index],
                        negate[index],
                        types,
                    );
                }
                if flush != HalfFlush::Preserve {
                    for index in 0..if op == HalfOp::Fma { 3 } else { 2 } {
                        pairs[index] = flush_half_pair(
                            &mut w,
                            &mut id,
                            pairs[index],
                            swizzle[index] == 1,
                            (uint, uvec2, vec2, bool_vec2),
                            late_globals,
                        );
                    }
                }
                if flush == HalfFlush::Fmz {
                    // The FMZ contract makes zero * anything a signed
                    // zero, even for infinity/NaN. Test after input flushing.
                    // Sanitize just those lanes to signed-zero * +1. This
                    // needs one arithmetic operation, preserves ordinary FMA
                    // fusion, and avoids a separate product/add/result select.
                    // Sign XOR follows IEEE product signs, including modifiers
                    // (Ryujinx InstEmitHelper.cs:80-102,129-158).
                    let zero_pair = alu(
                        &mut w,
                        &mut id,
                        Op::CompositeConstruct,
                        vec2,
                        &[zero_float, zero_float],
                    );
                    let one_pair = alu(
                        &mut w,
                        &mut id,
                        Op::CompositeConstruct,
                        vec2,
                        &[one_float, one_float],
                    );
                    let a_zero = alu(
                        &mut w,
                        &mut id,
                        Op::FOrdEqual,
                        bool_vec2,
                        &[pairs[0], zero_pair],
                    );
                    let b_zero = alu(
                        &mut w,
                        &mut id,
                        Op::FOrdEqual,
                        bool_vec2,
                        &[pairs[1], zero_pair],
                    );
                    let any_zero =
                        alu(&mut w, &mut id, Op::LogicalOr, bool_vec2, &[a_zero, b_zero]);
                    let sign = alu(&mut w, &mut id, Op::BitwiseXor, uint, &fmz_signs);
                    let shift = late_constant(late_globals, &mut id, uint, 16);
                    let low = alu(&mut w, &mut id, Op::ShiftLeftLogical, uint, &[sign, shift]);
                    let mask = late_constant(late_globals, &mut id, uint, 0x8000_0000);
                    let high = alu(&mut w, &mut id, Op::BitwiseAnd, uint, &[sign, mask]);
                    let signs = alu(&mut w, &mut id, Op::CompositeConstruct, uvec2, &[low, high]);
                    let signed_zero = alu(&mut w, &mut id, Op::Bitcast, vec2, &[signs]);
                    pairs[0] = alu(
                        &mut w,
                        &mut id,
                        Op::Select,
                        vec2,
                        &[any_zero, signed_zero, pairs[0]],
                    );
                    pairs[1] = alu(
                        &mut w,
                        &mut id,
                        Op::Select,
                        vec2,
                        &[any_zero, one_pair, pairs[1]],
                    );
                }
                // Ryujinx InstEmitFloatArithmetic.cs:483-528 computes in
                // float32 before packing, including F32 broadcast and output.
                // Native Float16 arithmetic would round those forms too early.
                // GLSL.std.450 section 2 Fma=50. One vector FMA preserves the
                // single rounding of both lanes with no intermediate product.
                let mut pair = match op {
                    HalfOp::Add | HalfOp::Mul => {
                        let value = alu(
                            &mut w,
                            &mut id,
                            if op == HalfOp::Add {
                                Op::FAdd
                            } else {
                                Op::FMul
                            },
                            vec2,
                            &pairs[..2],
                        );
                        inst(annotations, Op::Decorate, &[value, 42]); // NoContraction
                        value
                    }
                    HalfOp::Fma => alu(
                        &mut w,
                        &mut id,
                        Op::ExtInst,
                        vec2,
                        &[glsl_std_450, 50, pairs[0], pairs[1], pairs[2]],
                    ),
                };
                if saturate {
                    // Mesa nir_opcodes.py:265 and util/u_math.h:343-371
                    // require NaN -> +0. Host FClamp returned 1 for NaNs on
                    // RADV. Integer vector ordering also makes -0 -> +0
                    // deterministic, with half the operations of two scalar
                    // saturate_f32 calls. Ryujinx EmitterContextInsts.cs:
                    // 419-423 applies saturation before half packing.
                    let mut splat = |bits| {
                        let value = late_constant(late_globals, &mut id, uint, bits);
                        alu(
                            &mut w,
                            &mut id,
                            Op::CompositeConstruct,
                            uvec2,
                            &[value, value],
                        )
                    };
                    let sign_bound = splat(0x7fff_ffff);
                    let infinity = splat(0x7f80_0000);
                    let one = splat(0x3f80_0000);
                    let zero_pair = splat(0);
                    let bits = alu(&mut w, &mut id, Op::Bitcast, uvec2, &[pair]);
                    let magnitude =
                        alu(&mut w, &mut id, Op::BitwiseAnd, uvec2, &[bits, sign_bound]);
                    let nan = alu(
                        &mut w,
                        &mut id,
                        Op::UGreaterThan,
                        bool_vec2,
                        &[magnitude, infinity],
                    );
                    let negative = alu(
                        &mut w,
                        &mut id,
                        Op::UGreaterThan,
                        bool_vec2,
                        &[bits, sign_bound],
                    );
                    let zero = alu(&mut w, &mut id, Op::LogicalOr, bool_vec2, &[nan, negative]);
                    let large = alu(&mut w, &mut id, Op::UGreaterThan, bool_vec2, &[bits, one]);
                    let capped = alu(&mut w, &mut id, Op::Select, uvec2, &[large, one, bits]);
                    let saturated = alu(
                        &mut w,
                        &mut id,
                        Op::Select,
                        uvec2,
                        &[zero, zero_pair, capped],
                    );
                    pair = alu(&mut w, &mut id, Op::Bitcast, vec2, &[saturated]);
                }
                // F32 outputs have their own normal boundary. Packed outputs
                // flush after nearest-even quantization, so a value rounding
                // up to the smallest half normal is retained.
                if output == 1 && flush != HalfFlush::Preserve {
                    pair = flush_half_pair(
                        &mut w,
                        &mut id,
                        pair,
                        true,
                        (uint, uvec2, vec2, bool_vec2),
                        late_globals,
                    );
                }
                let value = if output == 1 {
                    let low = alu(&mut w, &mut id, Op::CompositeExtract, float, &[pair, 0]);
                    alu(&mut w, &mut id, Op::Bitcast, uint, &[low])
                } else {
                    // Mesa src/util/half_float.c:38-44,96-131 specifies
                    // nearest-even. Native PackHalf2x16 on RADV GFX1201
                    // truncates overflowing HMUL2 to 0x7bff and loses
                    // HFMA2's subnormal result.
                    // Quantize to exactly representable binary16 values
                    // before the native pack, preventing both losses.
                    let mut splat = |ty, bits| {
                        let value = late_constant(
                            late_globals,
                            &mut id,
                            if ty == uvec2 { uint } else { float },
                            bits,
                        );
                        alu(&mut w, &mut id, Op::CompositeConstruct, ty, &[value, value])
                    };
                    let low = splat(vec2, 0xbf80_0000);
                    let high = splat(vec2, 0x3f80_0000);
                    let up = splat(vec2, 0x4b80_0000); // 2^24
                    let down = splat(vec2, 0x3380_0000); // 2^-24
                    let normal = splat(vec2, 0x3880_0000); // 2^-14
                    let overflow_bound = splat(vec2, 0x477f_f000); // 65520
                    let shift = splat(uvec2, 13);
                    let one = splat(uvec2, 1);
                    let bias = splat(uvec2, 4095);
                    let mask = splat(uvec2, 0xffff_e000);
                    let sign_mask = splat(uvec2, 0x8000_0000);
                    let infinity = splat(uvec2, 0x7f80_0000);
                    // Round normal significands by adding 4095 + retained
                    // LSB, then clearing the 13 discarded bits. Carry into
                    // the exponent is intentional (Mesa :116-128).
                    let bits = alu(&mut w, &mut id, Op::Bitcast, uvec2, &[pair]);
                    let retained = alu(
                        &mut w,
                        &mut id,
                        Op::ShiftRightLogical,
                        uvec2,
                        &[bits, shift],
                    );
                    let lsb = alu(&mut w, &mut id, Op::BitwiseAnd, uvec2, &[retained, one]);
                    let round_bias = alu(&mut w, &mut id, Op::IAdd, uvec2, &[bias, lsb]);
                    let rounded_bits = alu(&mut w, &mut id, Op::IAdd, uvec2, &[bits, round_bias]);
                    let rounded_bits = alu(
                        &mut w,
                        &mut id,
                        Op::BitwiseAnd,
                        uvec2,
                        &[rounded_bits, mask],
                    );
                    let rounded = alu(&mut w, &mut id, Op::Bitcast, vec2, &[rounded_bits]);
                    let magnitude =
                        alu(&mut w, &mut id, Op::ExtInst, vec2, &[glsl_std_450, 4, pair]);
                    let overflow = alu(
                        &mut w,
                        &mut id,
                        Op::FOrdGreaterThanEqual,
                        bool_vec2,
                        &[magnitude, overflow_bound],
                    );
                    let sign = alu(&mut w, &mut id, Op::BitwiseAnd, uvec2, &[bits, sign_mask]);
                    let inf = alu(&mut w, &mut id, Op::BitwiseOr, uvec2, &[sign, infinity]);
                    let inf = alu(&mut w, &mut id, Op::Bitcast, vec2, &[inf]);
                    let rounded = alu(&mut w, &mut id, Op::Select, vec2, &[overflow, inf, rounded]);
                    // Signed roundEven(value * 2^24) * 2^-24 handles
                    // subnormals and keeps signed zero. Bound the temporary
                    // so unrelated large/NaN lanes cannot overflow scaling.
                    let bounded = alu(
                        &mut w,
                        &mut id,
                        Op::ExtInst,
                        vec2,
                        &[glsl_std_450, 43, pair, low, high],
                    );
                    let scaled = alu(&mut w, &mut id, Op::FMul, vec2, &[bounded, up]);
                    inst(annotations, Op::Decorate, &[scaled, 42]);
                    let sub = alu(
                        &mut w,
                        &mut id,
                        Op::ExtInst,
                        vec2,
                        &[glsl_std_450, 2, scaled],
                    );
                    let sub = alu(&mut w, &mut id, Op::FMul, vec2, &[sub, down]);
                    inst(annotations, Op::Decorate, &[sub, 42]);
                    let is_sub = alu(
                        &mut w,
                        &mut id,
                        Op::FOrdLessThan,
                        bool_vec2,
                        &[magnitude, normal],
                    );
                    let rounded = alu(&mut w, &mut id, Op::Select, vec2, &[is_sub, sub, rounded]);
                    // Retain the original NaN for the pack operation; the
                    // integer rounding bias must never alter its payload.
                    let nan = alu(&mut w, &mut id, Op::IsNan, bool_vec2, &[pair]);
                    pair = alu(&mut w, &mut id, Op::Select, vec2, &[nan, pair, rounded]);
                    if flush != HalfFlush::Preserve {
                        pair = flush_half_pair(
                            &mut w,
                            &mut id,
                            pair,
                            false,
                            (uint, uvec2, vec2, bool_vec2),
                            late_globals,
                        );
                    }
                    if output >= 2 {
                        // Ryujinx InstEmitHelper.cs:173-184 unpacks/repacks
                        // the preserved lane too, including its NaN conversion.
                        let old = load(&mut w, &mut id, uint, regs[dst.0 as usize]);
                        let old = half_pair(&mut w, &mut id, old, 0, false, false, types);
                        let lanes = [[2, 1], [0, 3]][usize::from(output == 2)];
                        pair = alu(
                            &mut w,
                            &mut id,
                            Op::VectorShuffle,
                            vec2,
                            &[pair, old, lanes[0], lanes[1]],
                        );
                    }
                    // GLSL.std.450 section 2 PackHalf2x16=58.
                    alu(
                        &mut w,
                        &mut id,
                        Op::ExtInst,
                        uint,
                        &[glsl_std_450, 58, pair],
                    )
                };
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    value,
                    pred,
                    uint,
                    boolt,
                );
            }
            InstructionKind::HalfSet(ops::HalfSet {
                ftz,
                dst,
                src,
                swizzle,
                abs,
                negate,
                compare: c,
                accum,
                accum_negate,
                set_op,
            }) => {
                // Ryujinx InstEmitFloatComparison.cs:309-366,456-523.
                // Compare both lanes together. Constant conditions fold here;
                // BF selects the binary16 bit pattern 0x3c00 directly, avoiding
                // float selects and PackHalf2x16 for known boolean values.
                let mut pairs = [0; 2];
                for i in 0..2 {
                    let bits = source(
                        &mut w, &mut id, src[i], &regs, uint, constants, cb_types, cb_vars, vec4,
                        float, zero,
                    );
                    pairs[i] = half_pair(
                        &mut w,
                        &mut id,
                        bits,
                        swizzle[i],
                        abs[i],
                        negate[i],
                        (float, vec2, glsl_std_450),
                    );
                }
                if ftz {
                    for i in 0..2 {
                        pairs[i] = flush_half_pair(
                            &mut w,
                            &mut id,
                            pairs[i],
                            swizzle[i] == 1,
                            (uint, uvec2, vec2, bool_vec2),
                            late_globals,
                        );
                    }
                }
                let mut results = match c {
                    FloatCompare::False => [no; 2],
                    FloatCompare::True => [yes; 2],
                    FloatCompare::Num | FloatCompare::Nan => {
                        let a = alu(&mut w, &mut id, Op::IsNan, bool_vec2, &[pairs[0]]);
                        let b = alu(&mut w, &mut id, Op::IsNan, bool_vec2, &[pairs[1]]);
                        let nan = alu(&mut w, &mut id, Op::LogicalOr, bool_vec2, &[a, b]);
                        let vector = alu_when(
                            &mut w,
                            &mut id,
                            c == FloatCompare::Num,
                            Op::LogicalNot,
                            bool_vec2,
                            &[nan],
                            nan,
                        );
                        bool_lanes(&mut w, &mut id, boolt, vector)
                    }
                    FloatCompare::Ordered(rel) => {
                        let op = ordered_compare_op(rel);
                        let vector = alu(&mut w, &mut id, op, bool_vec2, &pairs);
                        bool_lanes(&mut w, &mut id, boolt, vector)
                    }
                    FloatCompare::Unordered(rel) => {
                        let op = unordered_compare_op(rel);
                        let vector = alu(&mut w, &mut id, op, bool_vec2, &pairs);
                        bool_lanes(&mut w, &mut id, boolt, vector)
                    }
                };
                if matches!(
                    dst,
                    crate::ir::HalfSetDestination::Predicates {
                        horizontal: true,
                        ..
                    }
                ) {
                    results[0] = alu(&mut w, &mut id, Op::LogicalAnd, boolt, &results);
                    results[1] = alu(&mut w, &mut id, Op::LogicalNot, boolt, &[results[0]]);
                }
                // Read the accumulator before either predicate destination is
                // overwritten, including aliasing accumulator/destination.
                let mut acc = load(&mut w, &mut id, boolt, ps[accum as usize]);
                acc = alu_when(
                    &mut w,
                    &mut id,
                    accum_negate,
                    Op::LogicalNot,
                    boolt,
                    &[acc],
                    acc,
                );
                let op = set_op_code(set_op);
                for result in &mut results {
                    *result = alu(&mut w, &mut id, op, boolt, &[*result, acc]);
                }
                match dst {
                    crate::ir::HalfSetDestination::Predicates { dst, .. } => {
                        for i in 0..2 {
                            put(
                                &mut w,
                                &mut id,
                                ps[dst[i] as usize],
                                results[i],
                                pred,
                                boolt,
                                boolt,
                            );
                        }
                    }
                    crate::ir::HalfSetDestination::Register { dst, bool_float } => {
                        let mask = [0xffff, 0x3c00][usize::from(bool_float)];
                        let low_true = late_constant(late_globals, &mut id, uint, mask);
                        let high_true = late_constant(late_globals, &mut id, uint, mask << 16);
                        let low = alu(
                            &mut w,
                            &mut id,
                            Op::Select,
                            uint,
                            &[results[0], low_true, zero],
                        );
                        let high = alu(
                            &mut w,
                            &mut id,
                            Op::Select,
                            uint,
                            &[results[1], high_true, zero],
                        );
                        let value = alu(&mut w, &mut id, Op::BitwiseOr, uint, &[low, high]);
                        put(
                            &mut w,
                            &mut id,
                            regs[dst.0 as usize],
                            value,
                            pred,
                            uint,
                            boolt,
                        );
                    }
                }
            }
            InstructionKind::VideoMultiplyAdd(ops::VideoMultiplyAdd {
                dst,
                lhs,
                rhs,
                addend,
                select,
                mode,
                shift,
                saturate,
            }) => {
                // Ryujinx InstEmitVideoArithmetic.cs:15-112. Extended integer
                // multiply supplies both words without enabling shaderInt64.
                // Signed correction is high -= unsigned_other for a negative
                // signed operand, replacing multiply by its sign extension.
                let a = load(&mut w, &mut id, uint, regs[lhs.0 as usize]);
                let b = source(
                    &mut w, &mut id, rhs, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let a = video_extend_selection(&mut w, &mut id, a, select[0], uint, constants);
                let b = video_extend_selection(&mut w, &mut id, b, select[1], uint, constants);
                let signed_result = select.iter().any(|selection| {
                    matches!(
                        selection,
                        VideoSelection {
                            element: VideoElement::Whole,
                            signedness: Signedness::Signed,
                        }
                    )
                }) || matches!(mode, 1 | 2);
                let product = alu(&mut w, &mut id, Op::UMulExtended, iadd_carry_type, &[a, b]);
                let mut low = alu(&mut w, &mut id, Op::CompositeExtract, uint, &[product, 0]);
                let mut high = alu(&mut w, &mut id, Op::CompositeExtract, uint, &[product, 1]);
                for (selection, value, other) in [(select[0], a, b), (select[1], b, a)] {
                    if selection.signedness == Signedness::Signed {
                        let negative = alu(&mut w, &mut id, Op::SLessThan, boolt, &[value, zero]);
                        let correction =
                            alu(&mut w, &mut id, Op::Select, uint, &[negative, other, zero]);
                        high = alu(&mut w, &mut id, Op::ISub, uint, &[high, correction]);
                    }
                }
                if mode == 1 {
                    let carry = alu(&mut w, &mut id, Op::IEqual, boolt, &[low, zero]);
                    let carry = alu(
                        &mut w,
                        &mut id,
                        Op::Select,
                        uint,
                        &[carry, constants[&1], zero],
                    );
                    low = alu(&mut w, &mut id, Op::ISub, uint, &[zero, low]);
                    high = alu(&mut w, &mut id, Op::Not, uint, &[high]);
                    high = alu(&mut w, &mut id, Op::IAdd, uint, &[high, carry]);
                }
                let mut c = load(&mut w, &mut id, uint, regs[addend.0 as usize]);
                if mode == 2 {
                    c = alu(&mut w, &mut id, Op::ISub, uint, &[zero, c]);
                }
                let sum = alu(&mut w, &mut id, Op::IAddCarry, iadd_carry_type, &[low, c]);
                low = alu(&mut w, &mut id, Op::CompositeExtract, uint, &[sum, 0]);
                let carry = alu(&mut w, &mut id, Op::CompositeExtract, uint, &[sum, 1]);
                high = alu(&mut w, &mut id, Op::IAdd, uint, &[high, carry]);
                if mode == 3 {
                    let sum = alu(
                        &mut w,
                        &mut id,
                        Op::IAddCarry,
                        iadd_carry_type,
                        &[low, constants[&1]],
                    );
                    low = alu(&mut w, &mut id, Op::CompositeExtract, uint, &[sum, 0]);
                    let carry = alu(&mut w, &mut id, Op::CompositeExtract, uint, &[sum, 1]);
                    high = alu(&mut w, &mut id, Op::IAdd, uint, &[high, carry]);
                }
                if shift != 0 {
                    let right = constants[&u32::from(shift)];
                    let left = constants[&(32 - u32::from(shift))];
                    low = alu(&mut w, &mut id, Op::ShiftRightLogical, uint, &[low, right]);
                    let upper = alu(&mut w, &mut id, Op::ShiftLeftLogical, uint, &[high, left]);
                    low = alu(&mut w, &mut id, Op::BitwiseOr, uint, &[low, upper]);
                    high = alu(
                        &mut w,
                        &mut id,
                        if signed_result {
                            Op::ShiftRightArithmetic
                        } else {
                            Op::ShiftRightLogical
                        },
                        uint,
                        &[high, right],
                    );
                }
                if saturate {
                    let negative = alu(&mut w, &mut id, Op::SLessThan, boolt, &[high, zero]);
                    let (overflow, clamp) = if signed_result {
                        let sign = alu(
                            &mut w,
                            &mut id,
                            Op::ShiftRightArithmetic,
                            uint,
                            &[low, constants[&31]],
                        );
                        let overflow = alu(&mut w, &mut id, Op::INotEqual, boolt, &[high, sign]);
                        let clamp = alu(
                            &mut w,
                            &mut id,
                            Op::Select,
                            uint,
                            &[negative, constants[&0x8000_0000], constants[&0x7fff_ffff]],
                        );
                        (overflow, clamp)
                    } else {
                        let overflow = alu(&mut w, &mut id, Op::INotEqual, boolt, &[high, zero]);
                        let clamp = alu(
                            &mut w,
                            &mut id,
                            Op::Select,
                            uint,
                            &[negative, zero, constants[&u32::MAX]],
                        );
                        (overflow, clamp)
                    };
                    low = alu(&mut w, &mut id, Op::Select, uint, &[overflow, clamp, low]);
                }
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    low,
                    pred,
                    uint,
                    boolt,
                );
            }
            InstructionKind::VideoMinMax(ops::VideoMinMax {
                dst,
                lhs,
                rhs,
                addend,
                select,
                max,
                signed_output,
                saturate,
                op,
            }) => {
                let a = load(&mut w, &mut id, uint, regs[lhs.0 as usize]);
                let b = source(
                    &mut w, &mut id, rhs, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let c = load(&mut w, &mut id, uint, regs[addend.0 as usize]);
                let value = emit_video_min_max(
                    &mut w,
                    &mut id,
                    [a, b, c],
                    select,
                    max,
                    signed_output,
                    saturate,
                    op,
                    (uint, boolt),
                    constants,
                );
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    value,
                    pred,
                    uint,
                    boolt,
                );
            }
            InstructionKind::FFma(ops::FFma {
                dst,
                lhs,
                rhs,
                addend,
                flags,
                modifiers,
            }) => {
                let flushes = match modifiers.flush {
                    Flush::Keep => false,
                    Flush::Ftz | Flush::Dnz => true,
                };
                let a_uint = load(&mut w, &mut id, uint, regs[lhs.0 as usize]);
                let a_uint = FLUSH_F32[usize::from(flushes)](
                    &mut w,
                    &mut id,
                    a_uint,
                    (uint, boolt),
                    constants,
                );
                let mut a = id;
                id += 1;
                inst(&mut w, Op::Bitcast, &[float, a, a_uint]);
                let b_uint = source(
                    &mut w, &mut id, rhs, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let b_uint = FLUSH_F32[usize::from(flushes)](
                    &mut w,
                    &mut id,
                    b_uint,
                    (uint, boolt),
                    constants,
                );
                let mut b = id;
                id += 1;
                inst(&mut w, Op::Bitcast, &[float, b, b_uint]);
                let c_uint = source(
                    &mut w, &mut id, addend, &regs, uint, constants, cb_types, cb_vars, vec4,
                    float, zero,
                );
                let c_uint = FLUSH_F32[usize::from(flushes)](
                    &mut w,
                    &mut id,
                    c_uint,
                    (uint, boolt),
                    constants,
                );
                let mut c = id;
                id += 1;
                inst(&mut w, Op::Bitcast, &[float, c, c_uint]);
                a = alu_when(
                    &mut w,
                    &mut id,
                    modifiers.product_negate,
                    Op::FNegate,
                    float,
                    &[a],
                    a,
                );
                c = alu_when(
                    &mut w,
                    &mut id,
                    modifiers.addend_negate,
                    Op::FNegate,
                    float,
                    &[c],
                    c,
                );
                if modifiers.flush == Flush::Dnz {
                    // Mesa sm50.rs:601-602, from_nir.rs:1167-1180 and
                    // nir_opcodes.py:1213-1225: DNZ implies FTZ; a zero
                    // multiplicand produces +0 even when the other is NaN/Inf.
                    // Select (0,1) before FMA so its product is exactly +0.
                    // This keeps fusion and needs no separate add or branch.
                    let am = alu(
                        &mut w,
                        &mut id,
                        Op::BitwiseAnd,
                        uint,
                        &[a_uint, constants[&0x7fff_ffff]],
                    );
                    let bm = alu(
                        &mut w,
                        &mut id,
                        Op::BitwiseAnd,
                        uint,
                        &[b_uint, constants[&0x7fff_ffff]],
                    );
                    let az = alu(&mut w, &mut id, Op::IEqual, boolt, &[am, zero]);
                    let bz = alu(&mut w, &mut id, Op::IEqual, boolt, &[bm, zero]);
                    let any_zero = alu(&mut w, &mut id, Op::LogicalOr, boolt, &[az, bz]);
                    a = alu(
                        &mut w,
                        &mut id,
                        Op::Select,
                        float,
                        &[any_zero, zero_float, a],
                    );
                    b = alu(
                        &mut w,
                        &mut id,
                        Op::Select,
                        float,
                        &[any_zero, one_float, b],
                    );
                }
                let f = id;
                id += 1;
                inst(&mut w, Op::ExtInst, &[float, f, glsl_std_450, 50, a, b, c]);
                // Keep the FMA fused: Ryujinx CodeGen/Spirv/Instructions.cs:
                // 2106-2112 decorates it; Mesa vtn_alu.c:466-467 makes it exact.
                inst(annotations, Op::Decorate, &[f, 42]); // NoContraction
                let f = if flushes {
                    let bits = alu(&mut w, &mut id, Op::Bitcast, uint, &[f]);
                    let bits = flush_f32(&mut w, &mut id, bits, (uint, boolt), constants);
                    alu(&mut w, &mut id, Op::Bitcast, float, &[bits])
                } else {
                    f
                };
                let v = alu(&mut w, &mut id, Op::Bitcast, uint, &[f]);
                let v = if modifiers.saturate {
                    saturate_f32(&mut w, &mut id, v, (uint, boolt), constants, late_globals)
                } else {
                    v
                };
                put(&mut w, &mut id, regs[dst.0 as usize], v, pred, uint, boolt);
                if flags == FloatFlags::Written {
                    let result = alu(&mut w, &mut id, Op::Bitcast, float, &[v]);
                    let float_zero = late_constant(late_globals, &mut id, float, 0);
                    cc_writer!().set_fp_zn(result, float_zero);
                }
            }
            InstructionKind::FMnMx(ops::FMnMx {
                dst,
                lhs,
                rhs,
                min,
                flags,
                abs,
                negate,
            }) => {
                let a = load(&mut w, &mut id, uint, regs[lhs.0 as usize]);
                let b = source(
                    &mut w, &mut id, rhs, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let mut operands = [0; 2];
                for (i, raw) in [a, b].into_iter().enumerate() {
                    let mut f = fresh(&mut id);
                    inst(&mut w, Op::Bitcast, &[float, f, raw]);
                    f = alu_when(
                        &mut w,
                        &mut id,
                        abs[i],
                        Op::ExtInst,
                        float,
                        &[glsl_std_450, 4, f],
                        f,
                    );
                    f = alu_when(&mut w, &mut id, negate[i], Op::FNegate, float, &[f], f);
                    operands[i] = f;
                }
                // GLSL.std.450 NMin (79) / NMax (80): a NaN operand yields
                // the other operand, as FMNMX does.
                let f = fresh(&mut id);
                inst(
                    &mut w,
                    Op::ExtInst,
                    &[
                        float,
                        f,
                        glsl_std_450,
                        if min { 79 } else { 80 },
                        operands[0],
                        operands[1],
                    ],
                );
                let v = id;
                id += 1;
                inst(&mut w, Op::Bitcast, &[uint, v, f]);
                put(&mut w, &mut id, regs[dst.0 as usize], v, pred, uint, boolt);
                if flags == FloatFlags::Written {
                    let float_zero = late_constant(late_globals, &mut id, float, 0);
                    cc_writer!().set_fp_zn(f, float_zero);
                }
            }
            // GLSL.std.450 NMin (79) / NMax (80) over the modified sources,
            // selected by the runtime predicate (Ryujinx
            // InstEmitFloatMinMax.cs:78-104).
            InstructionKind::FMnMxSelect(ops::FMnMxSelect {
                dst,
                lhs,
                rhs,
                min_when,
                flags,
                abs,
                negate,
            }) => {
                let a = load(&mut w, &mut id, uint, regs[lhs.0 as usize]);
                let b = source(
                    &mut w, &mut id, rhs, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let mut operands = [0; 2];
                for (i, raw) in [a, b].into_iter().enumerate() {
                    let mut f = alu(&mut w, &mut id, Op::Bitcast, float, &[raw]);
                    f = alu_when(
                        &mut w,
                        &mut id,
                        abs[i],
                        Op::ExtInst,
                        float,
                        &[glsl_std_450, 4, f],
                        f,
                    );
                    f = alu_when(&mut w, &mut id, negate[i], Op::FNegate, float, &[f], f);
                    operands[i] = f;
                }
                let [x, y] = operands;
                let min = alu(
                    &mut w,
                    &mut id,
                    Op::ExtInst,
                    float,
                    &[glsl_std_450, 79, x, y],
                );
                let max = alu(
                    &mut w,
                    &mut id,
                    Op::ExtInst,
                    float,
                    &[glsl_std_450, 80, x, y],
                );
                let c = load(&mut w, &mut id, boolt, ps[min_when.register as usize]);
                // An inverted selector picks the maximum when the predicate holds.
                let [when_true, when_false] =
                    [[min, max], [max, min]][usize::from(min_when.inverted)];
                let f = alu(
                    &mut w,
                    &mut id,
                    Op::Select,
                    float,
                    &[c, when_true, when_false],
                );
                let v = alu(&mut w, &mut id, Op::Bitcast, uint, &[f]);
                put(&mut w, &mut id, regs[dst.0 as usize], v, pred, uint, boolt);
                if flags == FloatFlags::Written {
                    let float_zero = late_constant(late_globals, &mut id, float, 0);
                    cc_writer!().set_fp_zn(f, float_zero);
                }
            }
            InstructionKind::Flo(ops::Flo {
                dst,
                src,
                signed,
                invert,
                shift,
            }) => {
                if signed && shift {
                    return Err(bad(x));
                }
                let v = if let Source::Immediate(value) = src {
                    // Fold the complete immediate at translation time. Ryujinx
                    // InstEmitBitfield.cs:165-182 emits runtime bit operations.
                    let value = [value, !value][usize::from(invert)];
                    let result = if shift {
                        if value == 0 {
                            u32::MAX
                        } else {
                            value.leading_zeros()
                        }
                    } else {
                        let value = [value, !value][usize::from(signed && (value as i32) < 0)];
                        31u32.wrapping_sub(value.leading_zeros())
                    };
                    late_constant(late_globals, &mut id, uint, result)
                } else {
                    let a = source(
                        &mut w, &mut id, src, &regs, uint, constants, cb_types, cb_vars, vec4,
                        float, zero,
                    );
                    let a = alu_when(&mut w, &mut id, invert, Op::Not, uint, &[a], a);
                    // Ryujinx InstEmitBitfield.cs:171-179, Spirv/Instructions.cs:
                    // 556-571. Unsigned SH is reverse + FindILsb, including
                    // the all-zero sentinel -1 (Mesa nir_opcodes.py:583-591).
                    // GLSL.std.450 73/74/75.
                    if shift {
                        let a = alu(&mut w, &mut id, Op::BitReverse, uint, &[a]);
                        alu(&mut w, &mut id, Op::ExtInst, uint, &[glsl_std_450, 73, a])
                    } else {
                        // Use native FindSMsb on a bitcast signed input,
                        // rather than sign-normalizing with shift/XOR.
                        let a = alu_when(&mut w, &mut id, signed, Op::Bitcast, signed_int, &[a], a);
                        alu(
                            &mut w,
                            &mut id,
                            Op::ExtInst,
                            uint,
                            &[glsl_std_450, [75, 74][usize::from(signed)], a],
                        )
                    }
                };
                put(&mut w, &mut id, regs[dst.0 as usize], v, pred, uint, boolt);
            }
            InstructionKind::Popc(ops::Popc { dst, src, invert }) => {
                // Ryujinx InstEmitBitfield.cs:185-192 and
                // CodeGen/Spirv/Instructions.cs:260-263. Copyright (c)
                // Ryujinx Team and Contributors; MIT notice above applies.
                let a = source(
                    &mut w, &mut id, src, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                // SPIRV-Headers include/spirv/unified1/spirv.h:1969 (205),
                // :1897 (ISub 130).
                let count = alu(&mut w, &mut id, Op::BitCount, uint, &[a]);
                // popcount(!x) = 32 - popcount(x): same result, without a
                // complemented temporary. Fold immediate inputs through the
                // normal optimizer. Ryujinx's result is a 32-bit integer.
                let v = alu_when(
                    &mut w,
                    &mut id,
                    invert,
                    Op::ISub,
                    uint,
                    &[constants[&32], count],
                    count,
                );
                put(&mut w, &mut id, regs[dst.0 as usize], v, pred, uint, boolt)
            }
            InstructionKind::ISet(ops::ISet {
                dst,
                lhs,
                rhs,
                compare: c,
                signedness,
                bool_float,
                cc,
                x: extended,
                accum,
                accum_negate,
                set_op,
            }) => {
                // ISET is Mesa's register destination form of the integer
                // compare (sm50.rs:1494-1519); true is 1.0f when BF is set.
                let a = load(&mut w, &mut id, uint, regs[lhs.0 as usize]);
                let b = source(
                    &mut w, &mut id, rhs, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                // Capture old Z before .X/.CC updates it.
                let old_z = if extended {
                    Some(load(&mut w, &mut id, boolt, cc_zero))
                } else {
                    None
                };
                let q = match (old_z, c) {
                    (Some(old_z), IntCompare::Relation(rel)) => {
                        let carry = load(&mut w, &mut id, boolt, cc_carry);
                        IntegerEmitter {
                            w: &mut w,
                            id: &mut id,
                            uint,
                            boolt,
                            constants,
                        }
                        .extended_compare(rel, signedness, a, b, old_z, carry)
                    }
                    (None, _) | (Some(_), IntCompare::False | IntCompare::True) => {
                        int_compare(&mut w, &mut id, c, signedness, a, b, (boolt, yes, no))
                    }
                };
                let accum = load(&mut w, &mut id, boolt, ps[accum as usize]);
                let mut ints = IntegerEmitter {
                    w: &mut w,
                    id: &mut id,
                    uint,
                    boolt,
                    constants,
                };
                let accum = if accum_negate {
                    ints.not_bool(accum)
                } else {
                    accum
                };
                let q = ints.logical(set_op_code(set_op), q, accum);
                // True is 1.0f with .BF, else all ones (Ryujinx
                // InstEmitIntegerComparison.cs Iset), as uint bit patterns.
                let one = late_constant(
                    late_globals,
                    &mut id,
                    uint,
                    if bool_float { 0x3f80_0000 } else { u32::MAX },
                );
                let v = id;
                id += 1;
                inst(&mut w, Op::Select, &[uint, v, q, one, zero]);
                put(&mut w, &mut id, regs[dst.0 as usize], v, pred, uint, boolt);
                if cc {
                    write_cc(
                        &mut w,
                        &mut id,
                        v,
                        pred,
                        cc_zero,
                        cc_sign,
                        uint,
                        boolt,
                        zero,
                        constants[&31],
                    );
                    // Ryujinx InstEmitAluHelper.cs:95-121: extended integer
                    // Z is new Z AND old Z. .BF uses FP flags without chaining
                    // (InstEmitIntegerComparison.cs:154-166).
                    if extended && !bool_float {
                        let new_z = fresh(&mut id);
                        inst(&mut w, Op::IEqual, &[boolt, new_z, v, zero]);
                        let combined = fresh(&mut id);
                        inst(
                            &mut w,
                            Op::LogicalAnd,
                            &[boolt, combined, new_z, old_z.unwrap()],
                        );
                        put(&mut w, &mut id, cc_zero, combined, pred, boolt, boolt);
                    }
                }
            }
            InstructionKind::P2R(ops::P2R {
                dst,
                src,
                mask,
                byte,
                cc,
            }) => {
                // Adapted semantics: Ryujinx InstEmitPredicate.cs:86-114,
                // under the MIT notice above. Fold byte positions and skip
                // unselected bits for immediate masks, avoiding its full pack.
                let a = load(&mut w, &mut id, uint, regs[src.0 as usize]);
                let m = source(
                    &mut w, &mut id, mask, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let m = alu(&mut w, &mut id, Op::BitwiseAnd, uint, &[m, constants[&255]]);
                let shift = constants[&(u32::from(byte) * 8)];
                let shifted = alu(&mut w, &mut id, Op::ShiftLeftLogical, uint, &[m, shift]);
                let inverse = alu(&mut w, &mut id, Op::Not, uint, &[shifted]);
                let preserved = alu(&mut w, &mut id, Op::BitwiseAnd, uint, &[a, inverse]);
                // Flag order: Ryujinx Instructions/InstEmitHelper.cs:12-29.
                let flags = [cc_zero, cc_sign, cc_carry, cc_overflow];
                let count = [7, 4][usize::from(cc)];
                let mut insert = zero;
                for bit in 0..count {
                    if matches!(mask, Source::Immediate(m) if m & (1 << bit) == 0) {
                        continue;
                    }
                    let condition = load(
                        &mut w,
                        &mut id,
                        boolt,
                        if cc { flags[bit] } else { ps[bit] },
                    );
                    let position = late_constant(
                        late_globals,
                        &mut id,
                        uint,
                        1 << (bit as u32 + u32::from(byte) * 8),
                    );
                    let value = alu(
                        &mut w,
                        &mut id,
                        Op::Select,
                        uint,
                        &[condition, position, zero],
                    );
                    insert = alu(&mut w, &mut id, Op::BitwiseOr, uint, &[insert, value]);
                }
                let insert = alu(&mut w, &mut id, Op::BitwiseAnd, uint, &[insert, shifted]);
                let value = alu(&mut w, &mut id, Op::BitwiseOr, uint, &[preserved, insert]);
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    value,
                    pred,
                    uint,
                    boolt,
                );
            }
            InstructionKind::VideoSetP(ops::VideoSetP {
                dst,
                lhs,
                rhs,
                select,
                compare: c,
                signedness,
                set_op,
                accum,
            }) => {
                // Ryujinx InstEmitVideoMinMax.cs:122-180, under the MIT notice
                // above. video_extend emits one extract per small operand;
                // the comparison and its complement share one accumulator load.
                let a = load(&mut w, &mut id, uint, regs[lhs.0 as usize]);
                let b = source(
                    &mut w, &mut id, rhs, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let a = video_extend(&mut w, &mut id, a, select[0], uint, constants);
                let b = video_extend(&mut w, &mut id, b, select[1], uint, constants);
                let value = int_compare(&mut w, &mut id, c, signedness, a, b, (boolt, yes, no));
                let complement = alu(&mut w, &mut id, Op::LogicalNot, boolt, &[value]);
                let accumulator = load(&mut w, &mut id, boolt, ps[accum.register as usize]);
                let accumulator = alu_when(
                    &mut w,
                    &mut id,
                    accum.inverted,
                    Op::LogicalNot,
                    boolt,
                    &[accumulator],
                    accumulator,
                );
                let operation = set_op_code(set_op);
                // Both results are computed before either store, including
                // aliasing accumulator/destination predicates.
                let results = [value, complement]
                    .map(|v| alu(&mut w, &mut id, operation, boolt, &[v, accumulator]));
                for (destination, result) in dst.into_iter().zip(results) {
                    put(
                        &mut w,
                        &mut id,
                        ps[destination as usize],
                        result,
                        pred,
                        boolt,
                        boolt,
                    );
                }
            }
            InstructionKind::R2P(ops::R2P {
                src,
                mask,
                byte,
                bank,
            }) => {
                // Pi = mask bit i ? bit i of the selected byte : Pi (i < 7).
                // R2P.CC writes Z, N, C, V the same way (Ryujinx
                // InstEmitMove.cs:254-270; flag order InstEmitHelper.cs:12-29).
                let cc_flags = [cc_zero, cc_sign, cc_carry, cc_overflow];
                let targets: &[u32] = match bank {
                    FlagBank::Predicates => &ps[..7],
                    FlagBank::ConditionCodes => &cc_flags,
                };
                let a = load(&mut w, &mut id, uint, regs[src.0 as usize]);
                let bits = fresh(&mut id);
                inst(
                    &mut w,
                    Op::ShiftRightLogical,
                    &[uint, bits, a, constants[&(u32::from(byte) * 8)]],
                );
                let m = source(
                    &mut w, &mut id, mask, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                for (p, &target) in targets.iter().enumerate() {
                    let flag = constants[&(1 << p)];
                    let bit = fresh(&mut id);
                    inst(&mut w, Op::BitwiseAnd, &[uint, bit, bits, flag]);
                    let value = fresh(&mut id);
                    inst(&mut w, Op::INotEqual, &[boolt, value, bit, zero]);
                    let selected = fresh(&mut id);
                    inst(&mut w, Op::BitwiseAnd, &[uint, selected, m, flag]);
                    let enabled = fresh(&mut id);
                    inst(&mut w, Op::INotEqual, &[boolt, enabled, selected, zero]);
                    let old = load(&mut w, &mut id, boolt, target);
                    let new = fresh(&mut id);
                    inst(&mut w, Op::Select, &[boolt, new, enabled, value, old]);
                    put(&mut w, &mut id, target, new, pred, boolt, boolt);
                }
            }
            InstructionKind::ISetP(ops::ISetP {
                dst,
                lhs,
                rhs,
                compare: c,
                signedness,
                mode,
                set_op,
                accum,
                accum_negate,
                abs,
                negate,
            }) => {
                // dst = cmp SETOP accum (Mesa OpISetP; PT is true).
                let a = load(&mut w, &mut id, uint, regs[lhs.0 as usize]);
                let b = source(
                    &mut w, &mut id, rhs, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let a = sign_modifiers(&mut w, &mut id, a, abs[0], negate[0], uint, constants);
                let b = sign_modifiers(&mut w, &mut id, b, abs[1], negate[1], uint, constants);
                let v = match (mode, c) {
                    (IntegerCompareMode::Extended, IntCompare::Relation(relation)) => {
                        // ISETP.X consumes the condition-code state established
                        // by the preceding instruction, exactly like ISET.X.
                        let old_z = load(&mut w, &mut id, boolt, cc_zero);
                        let carry = load(&mut w, &mut id, boolt, cc_carry);
                        IntegerEmitter {
                            w: &mut w,
                            id: &mut id,
                            uint,
                            boolt,
                            constants,
                        }
                        .extended_compare(relation, signedness, a, b, old_z, carry)
                    }
                    (IntegerCompareMode::Ordinary, _)
                    | (IntegerCompareMode::Extended, IntCompare::False | IntCompare::True) => {
                        int_compare(&mut w, &mut id, c, signedness, a, b, (boolt, yes, no))
                    }
                };
                let acc = load(&mut w, &mut id, boolt, ps[accum as usize]);
                let acc = alu_when(
                    &mut w,
                    &mut id,
                    accum_negate,
                    Op::LogicalNot,
                    boolt,
                    &[acc],
                    acc,
                );
                let q = id;
                id += 1;
                inst(&mut w, set_op_code(set_op), &[boolt, q, v, acc]);
                match dst {
                    [first, 7] => {
                        put(&mut w, &mut id, ps[first as usize], q, pred, boolt, boolt);
                    }
                    destinations => {
                        // Ryujinx EmitIsetp (InstEmitIntegerComparison.cs:
                        // 183-191): the second destination receives the
                        // complement combined with the same accumulator.
                        // Snapshot the guard before either write replaces it.
                        let guard = pred.map(|(pointer, inverted)| {
                            let value = load(&mut w, &mut id, boolt, pointer);
                            alu_when(
                                &mut w,
                                &mut id,
                                inverted,
                                Op::LogicalNot,
                                boolt,
                                &[value],
                                value,
                            )
                        });
                        let complement = alu(&mut w, &mut id, Op::LogicalNot, boolt, &[v]);
                        let complement = alu(
                            &mut w,
                            &mut id,
                            set_op_code(set_op),
                            boolt,
                            &[complement, acc],
                        );
                        write_predicate_outputs(
                            &mut w,
                            &mut id,
                            &ps,
                            guard,
                            destinations.into_iter().zip([q, complement]),
                            boolt,
                        );
                    }
                }
            }
            InstructionKind::DoubleAdd(ops::DoubleAdd {
                dst,
                lhs,
                rhs,
                modifiers,
                round,
                flags,
            }) => {
                let (double, words) = double_types.ok_or_else(|| bad(x))?;
                let mut operands = [zero; 2];
                for (slot, source_value) in [DoubleSource::RegisterPair(lhs), rhs]
                    .into_iter()
                    .enumerate()
                {
                    let (low, high) = double_words(
                        &mut w,
                        &mut id,
                        source_value,
                        &regs,
                        uint,
                        constants,
                        cb_types,
                        cb_vars,
                        vec4,
                        float,
                        zero,
                    );
                    let bits = alu(&mut w, &mut id, Op::CompositeConstruct, words, &[low, high]);
                    let value = alu(&mut w, &mut id, Op::Bitcast, double, &[bits]);
                    operands[slot] = double_source_operation(
                        &mut w,
                        &mut id,
                        double,
                        glsl_std_450,
                        value,
                        modifiers[slot],
                    );
                }
                let sum = match round {
                    RoundingMode::NearestEven => dadd_rn(
                        &mut w,
                        &mut id,
                        operands,
                        (double, words, uint, boolt),
                        late_globals,
                        annotations,
                    ),
                    RoundingMode::NegativeInfinity => dadd_rm(
                        &mut w,
                        &mut id,
                        operands,
                        (double, words, uint, boolt),
                        late_globals,
                        annotations,
                    ),
                    RoundingMode::PositiveInfinity => dadd_rp(
                        &mut w,
                        &mut id,
                        operands,
                        (double, words, uint, boolt),
                        late_globals,
                        annotations,
                    ),
                    RoundingMode::Zero => dadd_rz(
                        &mut w,
                        &mut id,
                        operands,
                        (double, words, uint, boolt),
                        late_globals,
                        annotations,
                    ),
                };
                let result = alu(&mut w, &mut id, Op::Bitcast, words, &[sum]);
                let low = alu(&mut w, &mut id, Op::CompositeExtract, uint, &[result, 0]);
                let high = alu(&mut w, &mut id, Op::CompositeExtract, uint, &[result, 1]);
                put_pair(&mut w, &mut id, dst, [low, high], pred, uint, boolt);
                if flags == FloatFlags::Written {
                    let double_zero = late_double(late_globals, &mut id, double, 0.0);
                    cc_writer!().set_fp_zn(sum, double_zero);
                }
            }
            InstructionKind::DoubleMinMax(ops::DoubleMinMax {
                dst,
                lhs,
                rhs,
                operation,
                modifiers,
                flags,
            }) => {
                let (double, words) = double_types.ok_or_else(|| bad(x))?;
                let mut operands = [0; 2];
                for (slot, source_value) in [DoubleSource::RegisterPair(lhs), rhs]
                    .into_iter()
                    .enumerate()
                {
                    // Ryujinx InstEmitHelper.cs:42-66: a cbuf operand packs
                    // word offset (low) with offset + 1 (high); a D20
                    // immediate is the high word over a zero low word.
                    // The immediate's zero low word is the declared zero.
                    let [low, high] = source_value.words().map(|word| {
                        source(
                            &mut w, &mut id, word, &regs, uint, constants, cb_types, cb_vars, vec4,
                            float, zero,
                        )
                    });
                    let bits = alu(&mut w, &mut id, Op::CompositeConstruct, words, &[low, high]);
                    let value = alu(&mut w, &mut id, Op::Bitcast, double, &[bits]);
                    operands[slot] = double_source_operation(
                        &mut w,
                        &mut id,
                        double,
                        glsl_std_450,
                        value,
                        modifiers[slot],
                    );
                }
                // Ryujinx's SPIR-V backend lowers FPMinimum/FPMaximum to
                // GLSL.std.450 FMin/FMax (37/40), not NMin/NMax (79/80).
                let extreme = |w: &mut Vec<u32>, id: &mut u32, operation| {
                    alu(
                        w,
                        id,
                        Op::ExtInst,
                        double,
                        &[glsl_std_450, operation, operands[0], operands[1]],
                    )
                };
                let value = match operation {
                    DoubleMinMaxOperation::Minimum => extreme(&mut w, &mut id, 37),
                    DoubleMinMaxOperation::Maximum => extreme(&mut w, &mut id, 40),
                    // Ryujinx InstEmitFloatMinMax.cs:78-104: ConditionalSelect
                    // on the source predicate, minimum where it holds.
                    DoubleMinMaxOperation::Selected(predicate) => {
                        let minimum = extreme(&mut w, &mut id, 37);
                        let maximum = extreme(&mut w, &mut id, 40);
                        let held = load(&mut w, &mut id, boolt, ps[predicate.register as usize]);
                        let held = alu_when(
                            &mut w,
                            &mut id,
                            predicate.inverted,
                            Op::LogicalNot,
                            boolt,
                            &[held],
                            held,
                        );
                        alu(
                            &mut w,
                            &mut id,
                            Op::Select,
                            double,
                            &[held, minimum, maximum],
                        )
                    }
                };
                let result = alu(&mut w, &mut id, Op::Bitcast, words, &[value]);
                let low = alu(&mut w, &mut id, Op::CompositeExtract, uint, &[result, 0]);
                let high = alu(&mut w, &mut id, Op::CompositeExtract, uint, &[result, 1]);
                put_pair(&mut w, &mut id, dst, [low, high], pred, uint, boolt);
                if flags == FloatFlags::Written {
                    let double_zero = late_double(late_globals, &mut id, double, 0.0);
                    cc_writer!().set_fp_zn(value, double_zero);
                }
            }
            InstructionKind::DoubleFma(ops::DoubleFma {
                dst,
                lhs,
                multiplier,
                addend,
                product_sign,
                addend_sign,
                flags,
            }) => {
                let (double, words) = double_types.ok_or_else(|| bad(x))?;
                let mut operands = [zero; 3];
                for (slot, source_value) in [DoubleSource::RegisterPair(lhs), multiplier, addend]
                    .into_iter()
                    .enumerate()
                {
                    let (low, high) = double_words(
                        &mut w,
                        &mut id,
                        source_value,
                        &regs,
                        uint,
                        constants,
                        cb_types,
                        cb_vars,
                        vec4,
                        float,
                        zero,
                    );
                    let bits = alu(&mut w, &mut id, Op::CompositeConstruct, words, &[low, high]);
                    operands[slot] = alu(&mut w, &mut id, Op::Bitcast, double, &[bits]);
                }
                // By sign: the operand as read, or negated.
                operands[0] = ALU_WHEN[product_sign as usize](
                    &mut w,
                    &mut id,
                    Op::FNegate,
                    double,
                    &[operands[0]],
                    operands[0],
                );
                // By sign: the operand as read, or negated.
                operands[2] = ALU_WHEN[addend_sign as usize](
                    &mut w,
                    &mut id,
                    Op::FNegate,
                    double,
                    &[operands[2]],
                    operands[2],
                );
                let fused = alu(
                    &mut w,
                    &mut id,
                    Op::ExtInst,
                    double,
                    &[glsl_std_450, 50, operands[0], operands[1], operands[2]],
                );
                // Keep the binary64 FMA fused: Ryujinx CodeGen/Spirv/
                // Instructions.cs:2095-2102 decorates it; Mesa vtn_alu.c:
                // 466-467 makes it exact. Without it RADV split the FMA
                // when an operand came from a constant buffer.
                inst(annotations, Op::Decorate, &[fused, 42]); // NoContraction
                let result = alu(&mut w, &mut id, Op::Bitcast, words, &[fused]);
                let low = alu(&mut w, &mut id, Op::CompositeExtract, uint, &[result, 0]);
                let high = alu(&mut w, &mut id, Op::CompositeExtract, uint, &[result, 1]);
                put_pair(&mut w, &mut id, dst, [low, high], pred, uint, boolt);
                if flags == FloatFlags::Written {
                    let double_zero = late_double(late_globals, &mut id, double, 0.0);
                    cc_writer!().set_fp_zn(fused, double_zero);
                }
            }
            InstructionKind::DoubleMultiply(ops::DoubleMultiply {
                dst,
                lhs,
                rhs,
                sign,
                flags,
            }) => {
                let (double, words) = double_types.ok_or_else(|| bad(x))?;
                let mut operands = [zero; 2];
                for (slot, source_value) in [DoubleSource::RegisterPair(lhs), rhs]
                    .into_iter()
                    .enumerate()
                {
                    let (low, high) = double_words(
                        &mut w,
                        &mut id,
                        source_value,
                        &regs,
                        uint,
                        constants,
                        cb_types,
                        cb_vars,
                        vec4,
                        float,
                        zero,
                    );
                    let bits = alu(&mut w, &mut id, Op::CompositeConstruct, words, &[low, high]);
                    operands[slot] = alu(&mut w, &mut id, Op::Bitcast, double, &[bits]);
                }
                // By sign: the operand as read, or negated.
                operands[0] = ALU_WHEN[sign as usize](
                    &mut w,
                    &mut id,
                    Op::FNegate,
                    double,
                    &[operands[0]],
                    operands[0],
                );
                let product = alu(&mut w, &mut id, Op::FMul, double, &operands);
                let result = alu(&mut w, &mut id, Op::Bitcast, words, &[product]);
                let low = alu(&mut w, &mut id, Op::CompositeExtract, uint, &[result, 0]);
                let high = alu(&mut w, &mut id, Op::CompositeExtract, uint, &[result, 1]);
                put_pair(&mut w, &mut id, dst, [low, high], pred, uint, boolt);
                if flags == FloatFlags::Written {
                    let double_zero = late_double(late_globals, &mut id, double, 0.0);
                    cc_writer!().set_fp_zn(product, double_zero);
                }
            }
            InstructionKind::DoubleSetP(ops::DoubleSetP {
                dst,
                lhs,
                rhs,
                compare,
                set_op,
                accum,
                abs,
                negate,
            }) => {
                let (double, words) = double_types.ok_or_else(|| bad(x))?;
                // The instruction guard and accumulator are input state. Read
                // both before either predicate destination can replace them.
                let guard = pred.map(|(pointer, inverted)| {
                    let value = load(&mut w, &mut id, boolt, pointer);
                    alu_when(
                        &mut w,
                        &mut id,
                        inverted,
                        Op::LogicalNot,
                        boolt,
                        &[value],
                        value,
                    )
                });
                let mut accumulator = load(&mut w, &mut id, boolt, ps[accum.register as usize]);
                accumulator = alu_when(
                    &mut w,
                    &mut id,
                    accum.inverted,
                    Op::LogicalNot,
                    boolt,
                    &[accumulator],
                    accumulator,
                );

                let mut values = [zero; 2];
                for (slot, source_value) in [DoubleSource::RegisterPair(lhs), rhs]
                    .into_iter()
                    .enumerate()
                {
                    // A binary64 immediate carries its high word in the
                    // instruction; its low word is zero. Register and cbuf
                    // operands occupy the aligned adjacent word.
                    let (low, high) = double_words(
                        &mut w,
                        &mut id,
                        source_value,
                        &regs,
                        uint,
                        constants,
                        cb_types,
                        cb_vars,
                        vec4,
                        float,
                        zero,
                    );
                    let high = sign_modifiers(
                        &mut w,
                        &mut id,
                        high,
                        abs[slot],
                        negate[slot],
                        uint,
                        constants,
                    );
                    let bits = alu(&mut w, &mut id, Op::CompositeConstruct, words, &[low, high]);
                    values[slot] = alu(&mut w, &mut id, Op::Bitcast, double, &[bits]);
                }

                let compared = typed_float_compare(
                    &mut w,
                    &mut id,
                    compare,
                    values[0],
                    values[1],
                    (boolt, yes, no),
                );
                let complement = alu(&mut w, &mut id, Op::LogicalNot, boolt, &[compared]);
                let operation = set_op_code(set_op);
                let outputs = [compared, complement]
                    .map(|value| alu(&mut w, &mut id, operation, boolt, &[value, accumulator]));
                write_predicate_outputs(
                    &mut w,
                    &mut id,
                    &ps,
                    guard,
                    dst.into_iter().zip(outputs),
                    boolt,
                );
            }
            InstructionKind::DoubleSet(ops::DoubleSet {
                dst,
                lhs,
                rhs,
                compare,
                set_op,
                accum,
                abs,
                negate,
                result,
            }) => {
                let (double, words) = double_types.ok_or_else(|| bad(x))?;
                let mut accumulator = load(&mut w, &mut id, boolt, ps[accum.register as usize]);
                accumulator = alu_when(
                    &mut w,
                    &mut id,
                    accum.inverted,
                    Op::LogicalNot,
                    boolt,
                    &[accumulator],
                    accumulator,
                );
                let mut values = [zero; 2];
                for (slot, source_value) in [DoubleSource::RegisterPair(lhs), rhs]
                    .into_iter()
                    .enumerate()
                {
                    let (low, high) = double_words(
                        &mut w,
                        &mut id,
                        source_value,
                        &regs,
                        uint,
                        constants,
                        cb_types,
                        cb_vars,
                        vec4,
                        float,
                        zero,
                    );
                    let high = sign_modifiers(
                        &mut w,
                        &mut id,
                        high,
                        abs[slot],
                        negate[slot],
                        uint,
                        constants,
                    );
                    let bits = alu(&mut w, &mut id, Op::CompositeConstruct, words, &[low, high]);
                    values[slot] = alu(&mut w, &mut id, Op::Bitcast, double, &[bits]);
                }
                let compared = typed_float_compare(
                    &mut w,
                    &mut id,
                    compare,
                    values[0],
                    values[1],
                    (boolt, yes, no),
                );
                let selected = alu(
                    &mut w,
                    &mut id,
                    set_op_code(set_op),
                    boolt,
                    &[compared, accumulator],
                );
                let (when_true, flags) = match result {
                    DoubleSetResult::Mask(flags) => {
                        (alu(&mut w, &mut id, Op::Not, uint, &[zero]), flags)
                    }
                    DoubleSetResult::FloatOne(flags) => {
                        (alu(&mut w, &mut id, Op::Bitcast, uint, &[one_float]), flags)
                    }
                };
                let value = alu(
                    &mut w,
                    &mut id,
                    Op::Select,
                    uint,
                    &[selected, when_true, zero],
                );
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    value,
                    pred,
                    uint,
                    boolt,
                );
                if flags == FloatFlags::Written {
                    match result {
                        DoubleSetResult::Mask(_) => cc_writer!().set_zn(value, Words::Single),
                        DoubleSetResult::FloatOne(_) => {
                            let result = alu(&mut w, &mut id, Op::Bitcast, float, &[value]);
                            let float_zero = late_constant(late_globals, &mut id, float, 0);
                            cc_writer!().set_fp_zn(result, float_zero);
                        }
                    }
                }
            }
            InstructionKind::FSetP(ops::FSetP {
                dst,
                lhs,
                rhs,
                compare: c,
                set_op,
                accum,
                accum_negate,
                abs,
                negate,
            }) => {
                // dst = cmp SETOP accum (Mesa OpFSetP; PT is true).
                let a = load(&mut w, &mut id, uint, regs[lhs.0 as usize]);
                let b = source(
                    &mut w, &mut id, rhs, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let a = sign_modifiers(&mut w, &mut id, a, abs[0], negate[0], uint, constants);
                let b = sign_modifiers(&mut w, &mut id, b, abs[1], negate[1], uint, constants);
                let v = float_compare(&mut w, &mut id, c, a, b, (float, boolt, yes, no));
                let acc = load(&mut w, &mut id, boolt, ps[accum as usize]);
                let acc = alu_when(
                    &mut w,
                    &mut id,
                    accum_negate,
                    Op::LogicalNot,
                    boolt,
                    &[acc],
                    acc,
                );
                let q = id;
                id += 1;
                inst(&mut w, set_op_code(set_op), &[boolt, q, v, acc]);
                match dst {
                    [first, 7] => {
                        put(&mut w, &mut id, ps[first as usize], q, pred, boolt, boolt);
                    }
                    destinations => {
                        // Ryujinx EmitFsetp (InstEmitFloatComparison.cs:
                        // 445-453): the second destination receives the
                        // complement combined with the same accumulator.
                        // Snapshot the guard before either write replaces it.
                        let guard = pred.map(|(pointer, inverted)| {
                            let value = load(&mut w, &mut id, boolt, pointer);
                            alu_when(
                                &mut w,
                                &mut id,
                                inverted,
                                Op::LogicalNot,
                                boolt,
                                &[value],
                                value,
                            )
                        });
                        let complement = alu(&mut w, &mut id, Op::LogicalNot, boolt, &[v]);
                        let complement = alu(
                            &mut w,
                            &mut id,
                            set_op_code(set_op),
                            boolt,
                            &[complement, acc],
                        );
                        write_predicate_outputs(
                            &mut w,
                            &mut id,
                            &ps,
                            guard,
                            destinations.into_iter().zip([q, complement]),
                            boolt,
                        );
                    }
                }
            }
            InstructionKind::CSetP(ops::CSetP {
                dst,
                condition: csetp_condition,
                set_op,
                accum,
            }) => {
                // Ryujinx InstEmitConditionCode.cs:39-47. Read source before
                // either write so aliased source/destinations stay correct.
                // Translator.cs:339-385 predicates the whole instruction.
                // Snapshot its guard before either output can overwrite it.
                let guard = pred.map(|(pointer, inverted)| {
                    let value = load(&mut w, &mut id, boolt, pointer);
                    alu_when(
                        &mut w,
                        &mut id,
                        inverted,
                        Op::LogicalNot,
                        boolt,
                        &[value],
                        value,
                    )
                });
                let cc = csetp_condition.test(&mut ConditionFlags {
                    w: &mut w,
                    id: &mut id,
                    flags: [cc_zero, cc_sign, cc_carry, cc_overflow],
                    boolt,
                    yes,
                    no,
                });
                let acc = load(&mut w, &mut id, boolt, ps[accum.register as usize]);
                let acc = alu_when(
                    &mut w,
                    &mut id,
                    accum.inverted,
                    Op::LogicalNot,
                    boolt,
                    &[acc],
                    acc,
                );
                let op = set_op_code(set_op);
                let first = alu(&mut w, &mut id, op, boolt, &[cc, acc]);
                let second = CONDITION_COMPLEMENT[set_op as usize](
                    &mut w, &mut id, op, boolt, cc, acc, first,
                );
                write_predicate_outputs(
                    &mut w,
                    &mut id,
                    &ps,
                    guard,
                    dst.into_iter().zip([first, second]),
                    boolt,
                );
            }
            InstructionKind::CSet(ops::CSet {
                dst,
                condition: cset_condition,
                set_op,
                accum,
                result,
            }) => {
                // Ryujinx InstEmitConditionCode.cs:12-33: one condition,
                // one Boolean operation with the accumulator, then the mask
                // or 1.0f. The GPR write keeps the instruction guard.
                let cc = cset_condition.test(&mut ConditionFlags {
                    w: &mut w,
                    id: &mut id,
                    flags: [cc_zero, cc_sign, cc_carry, cc_overflow],
                    boolt,
                    yes,
                    no,
                });
                let mut accumulator = load(&mut w, &mut id, boolt, ps[accum.register as usize]);
                accumulator = alu_when(
                    &mut w,
                    &mut id,
                    accum.inverted,
                    Op::LogicalNot,
                    boolt,
                    &[accumulator],
                    accumulator,
                );
                let selected = alu(
                    &mut w,
                    &mut id,
                    set_op_code(set_op),
                    boolt,
                    &[cc, accumulator],
                );
                let when_true = match result {
                    SetResult::Mask => alu(&mut w, &mut id, Op::Not, uint, &[zero]),
                    SetResult::FloatOne => alu(&mut w, &mut id, Op::Bitcast, uint, &[one_float]),
                };
                let value = alu(
                    &mut w,
                    &mut id,
                    Op::Select,
                    uint,
                    &[selected, when_true, zero],
                );
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    value,
                    pred,
                    uint,
                    boolt,
                );
            }
            InstructionKind::PSet(ops::PSet {
                dst,
                src,
                set_op,
                negate,
                bool_float,
            }) => {
                // Independently implement Ryujinx InstEmitPredicate.cs:16-32.
                // Keep predicates boolean until one final uint OpSelect,
                // avoiding integer masks and intermediate bitcasts.
                let mut values = [0; 3];
                for (index, value) in values.iter_mut().enumerate() {
                    *value = load(&mut w, &mut id, boolt, ps[src[index] as usize]);
                    if negate[index] {
                        *value = alu(&mut w, &mut id, Op::LogicalNot, boolt, &[*value]);
                    }
                }
                let ab = alu(&mut w, &mut id, set_op_code(set_op[0]), boolt, &values[..2]);
                let condition = alu(
                    &mut w,
                    &mut id,
                    set_op_code(set_op[1]),
                    boolt,
                    &[ab, values[2]],
                );
                let true_bits = [u32::MAX, 0x3f80_0000][usize::from(bool_float)];
                let true_value = fresh(&mut id);
                inst(late_globals, Op::Constant, &[uint, true_value, true_bits]);
                let value = alu(
                    &mut w,
                    &mut id,
                    Op::Select,
                    uint,
                    &[condition, true_value, zero],
                );
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    value,
                    pred,
                    uint,
                    boolt,
                );
            }
            InstructionKind::PSetP(ops::PSetP {
                dst,
                src,
                set_op,
                negate,
            }) => {
                // Mesa OpPSetP fold (ir.rs:5318-5334):
                // dst0 = op1(c, op0(a, b)), dst1 = op1(c, op0(!a, b)).
                let mut value = [0; 3];
                for index in 0..3 {
                    let v = load(&mut w, &mut id, boolt, ps[src[index] as usize]);
                    value[index] = if negate[index] {
                        let n = id;
                        id += 1;
                        inst(&mut w, Op::LogicalNot, &[boolt, n, v]);
                        n
                    } else {
                        v
                    };
                }
                let op0 = set_op_code(set_op[0]);
                let op1 = set_op_code(set_op[1]);
                let not_a = id;
                id += 1;
                inst(&mut w, Op::LogicalNot, &[boolt, not_a, value[0]]);
                let mut results = [0; 2];
                for (slot, first) in [value[0], not_a].into_iter().enumerate() {
                    let t = id;
                    id += 1;
                    inst(&mut w, op0, &[boolt, t, first, value[1]]);
                    let r = id;
                    id += 1;
                    inst(&mut w, op1, &[boolt, r, value[2], t]);
                    results[slot] = r;
                }
                for slot in 0..2 {
                    put(
                        &mut w,
                        &mut id,
                        ps[dst[slot] as usize],
                        results[slot],
                        pred,
                        boolt,
                        boolt,
                    );
                }
            }
            InstructionKind::FSet(ops::FSet {
                dst,
                lhs,
                rhs,
                compare: c,
                bool_float,
                abs,
                negate,
                cc,
                accum,
                set_op,
                ftz,
            }) => {
                // FSET writes 1.0f (bool-float, bit 52) or 0xffffffff for true
                // and 0 for false (Mesa OpFSet always sets bool float).
                let a = load(&mut w, &mut id, uint, regs[lhs.0 as usize]);
                let b = source(
                    &mut w, &mut id, rhs, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let a = sign_modifiers(&mut w, &mut id, a, abs[0], negate[0], uint, constants);
                let b = sign_modifiers(&mut w, &mut id, b, abs[1], negate[1], uint, constants);
                // Ryujinx InstEmitFloatComparison.cs:397-403 specifies the
                // comparison order but omits decoded FTZ (InstDecoders.cs:
                // 2161,2184,2206). Mesa ir.rs:2910,8949 and opt_copy_prop.rs:
                // 323-325 make FSET an FTZ consumer. Flush both inputs using
                // integer bits, avoiding float-control modes and branches;
                // no result flush is needed for zero/one/all-ones outputs.
                let (a, b) = if ftz {
                    (
                        flush_f32(&mut w, &mut id, a, (uint, boolt), constants),
                        flush_f32(&mut w, &mut id, b, (uint, boolt), constants),
                    )
                } else {
                    (a, b)
                };
                let q = float_compare(&mut w, &mut id, c, a, b, (float, boolt, yes, no));
                // Ryujinx InstEmitFloatComparison.cs:400-403: combine the
                // comparison with the (possibly inverted) predicate. Fold
                // PT at translation time, avoiding a load and Boolean op.
                let q = if accum.register == 7 {
                    match (set_op, accum.inverted) {
                        (SetOp::And, false) | (SetOp::Or | SetOp::Xor, true) => q,
                        (SetOp::And, true) => no,
                        (SetOp::Or, false) => yes,
                        (SetOp::Xor, false) => alu(&mut w, &mut id, Op::LogicalNot, boolt, &[q]),
                    }
                } else {
                    let p = load(&mut w, &mut id, boolt, ps[accum.register as usize]);
                    let p = alu_when(
                        &mut w,
                        &mut id,
                        accum.inverted,
                        Op::LogicalNot,
                        boolt,
                        &[p],
                        p,
                    );
                    alu(&mut w, &mut id, set_op_code(set_op), boolt, &[q, p])
                };
                let t = id;
                id += 1;
                if bool_float {
                    inst(&mut w, Op::Bitcast, &[uint, t, one_float]);
                } else {
                    inst(&mut w, Op::Not, &[uint, t, zero]);
                }
                let v = id;
                id += 1;
                inst(&mut w, Op::Select, &[uint, v, q, t, zero]);
                put(&mut w, &mut id, regs[dst.0 as usize], v, pred, uint, boolt);
                if cc {
                    write_cc(
                        &mut w,
                        &mut id,
                        v,
                        pred,
                        cc_zero,
                        cc_sign,
                        uint,
                        boolt,
                        zero,
                        constants[&31],
                    );
                }
            }
            InstructionKind::Sel(ops::Sel {
                dst,
                when_true,
                when_false,
                condition,
                condition_negate,
            }) => {
                let c = load(&mut w, &mut id, boolt, ps[condition as usize]);
                let a = load(&mut w, &mut id, uint, regs[when_true.0 as usize]);
                let b = source(
                    &mut w, &mut id, when_false, &regs, uint, constants, cb_types, cb_vars, vec4,
                    float, zero,
                );
                let v = id;
                id += 1;
                // A negated condition selects the other operand.
                let (a, b) = if condition_negate { (b, a) } else { (a, b) };
                inst(&mut w, Op::Select, &[uint, v, c, a, b]);
                put(&mut w, &mut id, regs[dst.0 as usize], v, pred, uint, boolt)
            }
            InstructionKind::DoubleToInt(ops::DoubleToInt {
                dst,
                src,
                modifier,
                signedness,
                width,
                round,
            }) => {
                let (double, words) = double_types.ok_or_else(|| bad(x))?;
                // Ryujinx InstEmitHelper.cs:42-78: a pair, a cbuf word and
                // the next, or a D20 immediate's high word.
                let (low, high) = double_words(
                    &mut w, &mut id, src, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let bits = alu(&mut w, &mut id, Op::CompositeConstruct, words, &[low, high]);
                let value = alu(&mut w, &mut id, Op::Bitcast, double, &[bits]);
                // Ryujinx InstEmitConversion.cs:185: FPAbsNeg precedes the
                // rounding. Neither modifier changes whether the value is NaN.
                let value =
                    double_source_operation(&mut w, &mut id, double, glsl_std_450, value, modifier);
                // Ryujinx InstEmitConversion.cs:185-194: integral rounding
                // precedes the float-to-integer conversion. GLSL.std.450
                // supplies the same operations for binary64.
                let rounding = match round {
                    RoundingMode::NearestEven => 2,
                    RoundingMode::NegativeInfinity => 8,
                    RoundingMode::PositiveInfinity => 9,
                    RoundingMode::Zero => 3,
                };
                let rounded = alu(
                    &mut w,
                    &mut id,
                    Op::ExtInst,
                    double,
                    &[glsl_std_450, rounding, value],
                );

                // SPIR-V leaves out-of-range floating casts undefined. Keep
                // the established conversion contract: NaN becomes zero and
                // finite/infinite values clamp to the destination bounds.
                let zero_double = late_double(late_globals, &mut id, double, 0.0);
                let nan = alu(&mut w, &mut id, Op::IsNan, boolt, &[value]);
                let finite = alu(
                    &mut w,
                    &mut id,
                    Op::Select,
                    double,
                    &[nan, zero_double, rounded],
                );
                let lower = late_double(
                    late_globals,
                    &mut id,
                    double,
                    match (width, signedness) {
                        (IntegerWidth::Bits16, Signedness::Signed) => f64::from(i16::MIN),
                        (IntegerWidth::Bits32, Signedness::Signed) => f64::from(i32::MIN),
                        (_, Signedness::Unsigned) => 0.0,
                    },
                );
                let upper = late_double(
                    late_globals,
                    &mut id,
                    double,
                    match (width, signedness) {
                        (IntegerWidth::Bits16, Signedness::Signed) => f64::from(i16::MAX),
                        (IntegerWidth::Bits16, Signedness::Unsigned) => f64::from(u16::MAX),
                        (IntegerWidth::Bits32, Signedness::Signed) => f64::from(i32::MAX),
                        (IntegerWidth::Bits32, Signedness::Unsigned) => f64::from(u32::MAX),
                    },
                );
                let clamped = alu(
                    &mut w,
                    &mut id,
                    Op::ExtInst,
                    double,
                    &[glsl_std_450, 43, finite, lower, upper],
                );
                let result = alu(
                    &mut w,
                    &mut id,
                    match signedness {
                        Signedness::Signed => Op::ConvertFToS,
                        Signedness::Unsigned => Op::ConvertFToU,
                    },
                    uint,
                    &[clamped],
                );
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    result,
                    pred,
                    uint,
                    boolt,
                );
            }
            InstructionKind::DoubleToFloat(ops::DoubleToFloat {
                dst,
                src,
                modifier,
                round,
                saturate,
            }) => {
                let (double, words) = double_types.ok_or_else(|| bad(x))?;
                let (low, high) = double_words(
                    &mut w, &mut id, src, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let bits = alu(&mut w, &mut id, Op::CompositeConstruct, words, &[low, high]);
                let value = alu(&mut w, &mut id, Op::Bitcast, double, &[bits]);
                // Ryujinx InstEmitConversion.cs:132-162: FPAbsNeg, the
                // narrowing conversion, then FPSaturate in binary32.
                let value =
                    double_source_operation(&mut w, &mut id, double, glsl_std_450, value, modifier);
                // FConvert115 rounds to nearest even; a directed mode
                // steps from it as the directed FMUL product does.
                let narrowed = match round {
                    FMulRounding::NearestEven => {
                        alu(&mut w, &mut id, Op::FConvert, float, &[value])
                    }
                    FMulRounding::Directed(mode) => directed_f32(
                        &mut w,
                        &mut id,
                        value,
                        mode,
                        (uint, float, double, boolt),
                        constants,
                        late_globals,
                        annotations,
                    ),
                };
                let bits = alu(&mut w, &mut id, Op::Bitcast, uint, &[narrowed]);
                let bits = match saturate {
                    Saturation::Unclamped => bits,
                    Saturation::Saturated => saturate_f32(
                        &mut w,
                        &mut id,
                        bits,
                        (uint, boolt),
                        constants,
                        late_globals,
                    ),
                };
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    bits,
                    pred,
                    uint,
                    boolt,
                );
            }
            InstructionKind::DoubleRound(ops::DoubleRound {
                dst,
                src,
                modifier,
                integer_round,
                saturate,
            }) => {
                let (double, words) = double_types.ok_or_else(|| bad(x))?;
                let (low, high) = double_words(
                    &mut w, &mut id, src, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let bits = alu(&mut w, &mut id, Op::CompositeConstruct, words, &[low, high]);
                let value = alu(&mut w, &mut id, Op::Bitcast, double, &[bits]);
                // Ryujinx InstEmitConversion.cs:132-145: FPAbsNeg, then the
                // same-type integral rounding (GLSL.std.450 RoundEven2,
                // Trunc3, Floor8, Ceil9), then FPSaturate.
                let value =
                    double_source_operation(&mut w, &mut id, double, glsl_std_450, value, modifier);
                let value = match integer_round {
                    None => value,
                    Some(mode) => {
                        let operation = match mode {
                            RoundingMode::NearestEven => 2,
                            RoundingMode::NegativeInfinity => 8,
                            RoundingMode::PositiveInfinity => 9,
                            RoundingMode::Zero => 3,
                        };
                        alu(
                            &mut w,
                            &mut id,
                            Op::ExtInst,
                            double,
                            &[glsl_std_450, operation, value],
                        )
                    }
                };
                let value = match saturate {
                    Saturation::Unclamped => value,
                    // As saturate_f32: NaN and every negative value,
                    // -0 included, become +0; above 1.0 becomes 1.0.
                    Saturation::Saturated => {
                        let zero_double = late_double(late_globals, &mut id, double, 0.0);
                        let one_double = late_double(late_globals, &mut id, double, 1.0);
                        let pair = alu(&mut w, &mut id, Op::Bitcast, words, &[value]);
                        let sign = alu(&mut w, &mut id, Op::CompositeExtract, uint, &[pair, 1]);
                        let negative = alu(
                            &mut w,
                            &mut id,
                            Op::UGreaterThan,
                            boolt,
                            &[sign, constants[&0x7fff_ffff]],
                        );
                        let nan = alu(&mut w, &mut id, Op::IsNan, boolt, &[value]);
                        let cleared = alu(&mut w, &mut id, Op::LogicalOr, boolt, &[nan, negative]);
                        let large = alu(
                            &mut w,
                            &mut id,
                            Op::FOrdGreaterThan,
                            boolt,
                            &[value, one_double],
                        );
                        let capped = alu(
                            &mut w,
                            &mut id,
                            Op::Select,
                            double,
                            &[large, one_double, value],
                        );
                        alu(
                            &mut w,
                            &mut id,
                            Op::Select,
                            double,
                            &[cleared, zero_double, capped],
                        )
                    }
                };
                let result = alu(&mut w, &mut id, Op::Bitcast, words, &[value]);
                let low = alu(&mut w, &mut id, Op::CompositeExtract, uint, &[result, 0]);
                let high = alu(&mut w, &mut id, Op::CompositeExtract, uint, &[result, 1]);
                put_pair(&mut w, &mut id, dst, [low, high], pred, uint, boolt);
            }
            InstructionKind::Convert(ops::Convert { kind, dst, src: s }) => {
                let a = source(
                    &mut w, &mut id, s, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                // SPIR-V 1.6 sections 3.3.11 (109..112,124), 3.3.12
                // (80,81), 3.3.13 (126), 3.3.14 (202,203), 3.3.15
                // (156,169,170,186), BitwiseAnd199 in section 3.3.14.
                // Numeric opcodes checked against
                // /usr/include/spirv/unified1/spirv.h:1874-1964.
                // GLSL.std.450 section 2: RoundEven2, Trunc3, Floor8, Ceil9,
                // FAbs4, SAbs5, FClamp43, UClamp44, SClamp45, PackHalf2x16
                // 58 and UnpackHalf2x16 62 (GLSL.std.450.h:21-29,66-68,83-87).
                let round_op = |mode: RoundingMode| match mode {
                    RoundingMode::NearestEven => 2,
                    RoundingMode::NegativeInfinity => 8,
                    RoundingMode::PositiveInfinity => 9,
                    RoundingMode::Zero => 3,
                };
                let result = match kind {
                    Conversion::I2I {
                        src_signed,
                        dst_signed,
                        src_size,
                        dst_size,
                        byte,
                        abs,
                        negate,
                        saturate,
                        ..
                    } => {
                        // Ryujinx Instructions/InstEmitConversion.cs:318-341.
                        let a = conversion_extract(
                            &mut w,
                            &mut id,
                            a,
                            uint,
                            constants,
                            (src_size, byte, src_signed),
                        );
                        let a = conversion_abs_neg(
                            &mut w,
                            &mut id,
                            a,
                            uint,
                            glsl_std_450,
                            (abs, negate),
                        );
                        if saturate {
                            let (low, high) = integer_bounds(dst_size, dst_signed);
                            let low = late_constant(late_globals, &mut id, uint, low);
                            let high = late_constant(late_globals, &mut id, uint, high);
                            let v = fresh(&mut id);
                            inst(
                                &mut w,
                                Op::ExtInst,
                                &[
                                    uint,
                                    v,
                                    glsl_std_450,
                                    if dst_signed { 45 } else { 44 },
                                    a,
                                    low,
                                    high,
                                ],
                            );
                            v
                        } else {
                            // Ryujinx:339 writes all 32 bits, even for an
                            // 8/16-bit destination. No truncation is applied.
                            a
                        }
                    }
                    Conversion::I2F {
                        signed,
                        dst_size,
                        src_size,
                        round,
                        byte,
                        abs,
                        negate,
                    } => {
                        if !matches!(dst_size, 1..=3)
                            || src_size > 3
                            || (src_size < 3 && u32::from(byte) * 8 + (8 << src_size) > 32)
                            || (src_size == 3 && (byte != 0 || dst_size == 1))
                        {
                            return Err(bad(x));
                        }
                        let a = if src_size == 3 {
                            let Source::Register(register) = s else {
                                return Err(bad(x));
                            };
                            // Mesa nak/ir.rs:4834-4838 requires GPR for 64-bit
                            // inputs. RegRef components are contiguous (:394-397).
                            // Mesa sm50.rs:204-205: R255 is the zero register.
                            let high = load(
                                &mut w,
                                &mut id,
                                uint,
                                regs[(usize::from(register.0) + 1).min(255)],
                            );
                            let mut conversion = ConversionEmitter {
                                w: &mut w,
                                globals: late_globals,
                                id: &mut id,
                                uint,
                                boolt,
                                glsl: glsl_std_450,
                                integer: integer64.ok_or_else(|| bad(x))?,
                                wide: true,
                            };
                            let a = conversion.join(a, high);
                            conversion.modifiers(a, abs, negate)
                        } else {
                            // Ryujinx InstEmitConversion.cs:257-265 applies
                            // signed abs/neg to the full word BEFORE extraction.
                            let a = conversion_abs_neg(
                                &mut w,
                                &mut id,
                                a,
                                uint,
                                glsl_std_450,
                                (abs, negate),
                            );
                            conversion_extract(
                                &mut w,
                                &mut id,
                                a,
                                uint,
                                constants,
                                (src_size, byte, signed),
                            )
                        };
                        let rounded_bits = if round != RoundingMode::NearestEven
                            && (src_size == 3 || dst_size == 1)
                        {
                            Some(
                                ConversionEmitter {
                                    w: &mut w,
                                    globals: late_globals,
                                    id: &mut id,
                                    uint,
                                    boolt,
                                    glsl: glsl_std_450,
                                    integer: if src_size == 3 {
                                        integer64.unwrap()
                                    } else {
                                        uint
                                    },
                                    wide: src_size == 3,
                                }
                                .rounded_bits(a, signed, dst_size, round),
                            )
                        } else {
                            None
                        };
                        if dst_size == 3 {
                            // Ryujinx InstEmitConversion.cs:268-272,410-415:
                            // low/high words to consecutive registers. 32-bit
                            // inputs are exact; 64-bit inputs use the chosen mode.
                            // One vector bitcast
                            // shares the split instead of unpacking twice.
                            let (double, words_type) = double_types.ok_or_else(|| bad(x))?;
                            let f = if let Some(bits) = rounded_bits {
                                bits
                            } else {
                                alu(
                                    &mut w,
                                    &mut id,
                                    if signed {
                                        Op::ConvertSToF
                                    } else {
                                        Op::ConvertUToF
                                    },
                                    double,
                                    &[a],
                                )
                            };
                            let words = alu(&mut w, &mut id, Op::Bitcast, words_type, &[f]);
                            let low = alu(&mut w, &mut id, Op::CompositeExtract, uint, &[words, 0]);
                            let high =
                                alu(&mut w, &mut id, Op::CompositeExtract, uint, &[words, 1]);
                            put(
                                &mut w,
                                &mut id,
                                regs[dst.0.saturating_add(1) as usize],
                                high,
                                pred,
                                uint,
                                boolt,
                            );
                            low
                        } else if let Some(bits) = rounded_bits {
                            if src_size == 3 {
                                alu(&mut w, &mut id, Op::UConvert, uint, &[bits])
                            } else {
                                bits
                            }
                        } else {
                            let f = fresh(&mut id);
                            inst(
                                &mut w,
                                if signed {
                                    Op::ConvertSToF
                                } else {
                                    Op::ConvertUToF
                                },
                                &[float, f, a],
                            );
                            if dst_size == 1 {
                                // Ryujinx InstEmitConversion.cs:398-408 writes
                                // the converted half in the low 16 bits and zero
                                // in the high half. Use explicit nearest-even bits.
                                integer_float_to_half(
                                    &mut w,
                                    late_globals,
                                    &mut id,
                                    f,
                                    (uint, boolt),
                                    constants,
                                )
                            } else if round != RoundingMode::NearestEven && src_size == 2 {
                                ConversionEmitter {
                                    w: &mut w,
                                    globals: late_globals,
                                    id: &mut id,
                                    uint,
                                    boolt,
                                    glsl: glsl_std_450,
                                    integer: integer64.unwrap(),
                                    wide: true,
                                }
                                .directed_f32(a, f, signed, round)
                            } else {
                                alu(&mut w, &mut id, Op::Bitcast, uint, &[f])
                            }
                        }
                    }
                    // Ryujinx InstEmitConversion.cs:128-162.
                    // round_f32 preserves subnormal floor/ceil and signed zero
                    // with integer operations instead of device FP controls.
                    Conversion::F2F {
                        source,
                        destination,
                        integer_round,
                        ftz,
                        abs,
                        negate,
                        saturate,
                    } => {
                        // Ryujinx InstEmitConversion.cs:344-378 widens the
                        // low binary16 half exactly before the modifiers.
                        // GLSL.std.450 UnpackHalf2x16 is ExtInst 62.
                        let half = |w: &mut Vec<u32>, id: &mut u32, lane: u32| {
                            let pair = fresh(id);
                            inst(w, Op::ExtInst, &[vec2, pair, glsl_std_450, 62, a]);
                            let value = fresh(id);
                            inst(w, Op::CompositeExtract, &[float, value, pair, lane]);
                            alu(w, id, Op::Bitcast, uint, &[value])
                        };
                        let a = match source {
                            F2fSource::Float32 => a,
                            F2fSource::Float16Low => half(&mut w, &mut id, 0),
                            F2fSource::Float16High => half(&mut w, &mut id, 1),
                        };
                        // Ryujinx InstEmitConversion.cs:132-160. Combine
                        // abs+neg in the sign-bit helper before rounding.
                        let a = sign_modifiers(&mut w, &mut id, a, abs, negate, uint, constants);
                        let bits = FLUSH_F32[usize::from(integer_round.is_none() && ftz)](
                            &mut w,
                            &mut id,
                            a,
                            (uint, boolt),
                            constants,
                        );
                        let rounded = integer_round.map(|mode| {
                            let (_, rounded) = round_f32(
                                &mut w,
                                &mut id,
                                a,
                                (mode, ftz),
                                (uint, float, boolt, glsl_std_450),
                                constants,
                                late_globals,
                            );
                            rounded
                        });
                        let bits =
                            rounded.map_or(bits, |f| alu(&mut w, &mut id, Op::Bitcast, uint, &[f]));
                        let bits = if saturate {
                            // SAT50: envytools gm107.c:1997,2071; Mesa's
                            // gm107 emitter:823. Mesa nir_opcodes.py:265 and
                            // util/u_math.h:343-371 define min/max-number.
                            // Integer saturation preserves positive subnormals,
                            // NaN-to-zero and +0 without extra FP control modes.
                            saturate_f32(
                                &mut w,
                                &mut id,
                                bits,
                                (uint, boolt),
                                constants,
                                late_globals,
                            )
                        } else {
                            bits
                        };
                        match destination {
                            F2fDestination::Float32 => bits,
                            F2fDestination::Float16Low => {
                                // Ryujinx InstEmitConversion.cs:406-409 packs
                                // the result with 0 in the high half.
                                // GLSL.std.450 PackHalf2x16=58.
                                let f = alu(&mut w, &mut id, Op::Bitcast, float, &[bits]);
                                let pair = alu(
                                    &mut w,
                                    &mut id,
                                    Op::CompositeConstruct,
                                    vec2,
                                    &[f, zero_float],
                                );
                                alu(
                                    &mut w,
                                    &mut id,
                                    Op::ExtInst,
                                    uint,
                                    &[glsl_std_450, 58, pair],
                                )
                            }
                            F2fDestination::Float64 => {
                                // Ryujinx InstEmitConversion.cs:149-152,410-416:
                                // exact widening, low/high words to D and D+1.
                                // Saturating before widening is the same value.
                                let (double, words_type) = double_types.ok_or_else(|| bad(x))?;
                                let f = alu(&mut w, &mut id, Op::Bitcast, float, &[bits]);
                                let f = alu(&mut w, &mut id, Op::FConvert, double, &[f]);
                                let words = alu(&mut w, &mut id, Op::Bitcast, words_type, &[f]);
                                let high =
                                    alu(&mut w, &mut id, Op::CompositeExtract, uint, &[words, 1]);
                                put(
                                    &mut w,
                                    &mut id,
                                    regs[dst.0.saturating_add(1) as usize],
                                    high,
                                    pred,
                                    uint,
                                    boolt,
                                );
                                alu(&mut w, &mut id, Op::CompositeExtract, uint, &[words, 0])
                            }
                        }
                    }
                    Conversion::F2I {
                        signed,
                        dst_size,
                        src_size,
                        round,
                        high,
                        ftz,
                        abs,
                        negate,
                    } => {
                        // Ryujinx InstEmitConversion.cs:344-352,380-388:
                        // unpack the selected half, or reinterpret float32.
                        // Prefer the shared FTZ helper.
                        // Mesa sm50.rs:1837 encodes FTZ44; Ryujinx
                        // InstEmitConversion.cs:185-194 defines modifier and
                        // rounding order. The magnitude test uses existing
                        // constants and preserves NaN/Inf and signed zero.
                        let a = FLUSH_F32[usize::from(ftz)](
                            &mut w,
                            &mut id,
                            a,
                            (uint, boolt),
                            constants,
                        );
                        let mut f = fresh(&mut id);
                        if src_size == 1 {
                            let pair = fresh(&mut id);
                            inst(&mut w, Op::ExtInst, &[vec2, pair, glsl_std_450, 62, a]);
                            inst(
                                &mut w,
                                Op::CompositeExtract,
                                &[float, f, pair, u32::from(high)],
                            );
                        } else {
                            inst(&mut w, Op::Bitcast, &[float, f, a]);
                        }
                        // Ryujinx:185-194: FP abs, neg, then integer rounding.
                        f = alu_when(
                            &mut w,
                            &mut id,
                            abs,
                            Op::ExtInst,
                            float,
                            &[glsl_std_450, 4, f],
                            f,
                        );
                        if negate {
                            let v = fresh(&mut id);
                            // FNegate127: SPIR-V 1.6 section 3.3.13;
                            // spirv.h:1891.
                            inst(&mut w, Op::FNegate, &[float, v, f]);
                            f = v;
                        }
                        let mut r = fresh(&mut id);
                        inst(
                            &mut w,
                            Op::ExtInst,
                            &[float, r, glsl_std_450, round_op(round), f],
                        );
                        if src_size == 2
                            && !ftz
                            && matches!(
                                round,
                                RoundingMode::NegativeInfinity | RoundingMode::PositiveInfinity
                            )
                        {
                            // Mesa src/util/u_math.h:91-100: exponent-zero,
                            // nonzero-mantissa binary32 values are subnormal.
                            // Ryujinx InstEmitConversion.cs:187-194 requires
                            // mathematical floor/ceil. Some Vulkan devices
                            // flush subnormals in FP arithmetic by default.
                            // Compute this case from the original bits, with
                            // abs/neg affecting only its sign. No FP-control
                            // device feature or global execution mode is needed.
                            // SPIR-V 3.3.15: LogicalAnd167, LogicalNot168,
                            // INotEqual171; spirv.h:1930-1934.
                            let exp_mask = late_constant(late_globals, &mut id, uint, 0x7f80_0000);
                            let exponent = fresh(&mut id);
                            inst(&mut w, Op::BitwiseAnd, &[uint, exponent, a, exp_mask]);
                            let exp_zero = fresh(&mut id);
                            inst(&mut w, Op::IEqual, &[boolt, exp_zero, exponent, zero]);
                            let magnitude = fresh(&mut id);
                            inst(
                                &mut w,
                                Op::BitwiseAnd,
                                &[uint, magnitude, a, constants[&0x7fff_ffff]],
                            );
                            let nonzero = fresh(&mut id);
                            inst(&mut w, Op::INotEqual, &[boolt, nonzero, magnitude, zero]);
                            let tiny = fresh(&mut id);
                            inst(&mut w, Op::LogicalAnd, &[boolt, tiny, exp_zero, nonzero]);
                            let mut negative = if abs {
                                no
                            } else {
                                let sign = fresh(&mut id);
                                inst(
                                    &mut w,
                                    Op::BitwiseAnd,
                                    &[uint, sign, a, constants[&0x8000_0000]],
                                );
                                let negative = fresh(&mut id);
                                inst(&mut w, Op::INotEqual, &[boolt, negative, sign, zero]);
                                negative
                            };
                            negative = alu_when(
                                &mut w,
                                &mut id,
                                negate,
                                Op::LogicalNot,
                                boolt,
                                &[negative],
                                negative,
                            );
                            let (when_negative, when_positive) =
                                if matches!(round, RoundingMode::NegativeInfinity) {
                                    (
                                        late_constant(
                                            late_globals,
                                            &mut id,
                                            float,
                                            (-1.0_f32).to_bits(),
                                        ),
                                        float_constants[&0],
                                    )
                                } else {
                                    (
                                        float_constants[&0],
                                        late_constant(
                                            late_globals,
                                            &mut id,
                                            float,
                                            1.0_f32.to_bits(),
                                        ),
                                    )
                                };
                            let rounded_tiny = fresh(&mut id);
                            inst(
                                &mut w,
                                Op::Select,
                                &[float, rounded_tiny, negative, when_negative, when_positive],
                            );
                            let result = fresh(&mut id);
                            inst(&mut w, Op::Select, &[float, result, tiny, rounded_tiny, r]);
                            r = result;
                        }
                        // SPIR-V 3.3.11 leaves out-of-range float casts
                        // undefined. Sanitize NaN BEFORE conversion so no
                        // undefined value ever reaches the conversion.
                        let nan = fresh(&mut id);
                        inst(&mut w, Op::IsNan, &[boolt, nan, f]);
                        let finite = fresh(&mut id);
                        inst(
                            &mut w,
                            Op::Select,
                            &[float, finite, nan, float_constants[&0], r],
                        );
                        // Clamp and NaN-to-zero contract is documented in
                        // docs/reference-sources.md
                        // Use destination bounds directly for 16-bit results,
                        // saving the second integer clamp in Ryujinx:217-224.
                        // These bounds are exactly representable as float32.
                        let (low, high, top) = if dst_size == 1 {
                            let (low, top) = integer_bounds(dst_size, signed);
                            let low_float = [low as f32, (low as i32) as f32][usize::from(signed)];
                            (low_float.to_bits(), (top as f32).to_bits(), top)
                        } else if signed {
                            (0xcf00_0000u32, 0x4eff_ffffu32, 0x7fff_ffffu32)
                        } else {
                            (0, 0x4f7f_ffff, 0xffff_ffff)
                        };
                        let low = late_constant(late_globals, &mut id, float, low);
                        let high = late_constant(late_globals, &mut id, float, high);
                        let top = late_constant(late_globals, &mut id, uint, top);
                        let clamped = fresh(&mut id);
                        inst(
                            &mut w,
                            Op::ExtInst,
                            &[float, clamped, glsl_std_450, 43, finite, low, high],
                        );
                        let converted = fresh(&mut id);
                        inst(
                            &mut w,
                            if signed {
                                Op::ConvertFToS
                            } else {
                                Op::ConvertFToU
                            },
                            &[uint, converted, clamped],
                        );
                        if dst_size == 1 {
                            converted
                        } else {
                            // Largest safe float32 is below INT/UINT_MAX.
                            // Patch values above it to the exact integer max.
                            let above = fresh(&mut id);
                            inst(&mut w, Op::FOrdGreaterThan, &[boolt, above, finite, high]);
                            let v = fresh(&mut id);
                            inst(&mut w, Op::Select, &[uint, v, above, top, converted]);
                            v
                        }
                    }
                };
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    result,
                    pred,
                    uint,
                    boolt,
                );
                if matches!(kind, Conversion::I2I { cc: true, .. }) {
                    // Ryujinx InstEmitConversion.cs:341 and
                    // InstEmitAluHelper.cs:95-121 updates only Z and N.
                    write_cc(
                        &mut w,
                        &mut id,
                        result,
                        pred,
                        cc_zero,
                        cc_sign,
                        uint,
                        boolt,
                        zero,
                        constants[&31],
                    );
                }
            }
            InstructionKind::IpaSaturate(ops::IpaSaturate { dst, addr, inv_w }) => {
                S::fragment_only(x)?;
                let location = (addr - 0x80) / 16;
                let input = *input_by_location.get(&location).ok_or_else(|| bad(x))?;
                let vector = load(&mut w, &mut id, vec4, input);
                let component = u32::from((addr & 15) / 4);
                let mut value = alu(
                    &mut w,
                    &mut id,
                    Op::CompositeExtract,
                    float,
                    &[vector, component],
                );
                // Ryujinx InstEmitAttribute.cs:167-175,208-217: undo host
                // perspective interpolation with FragCoord.w, apply B, then
                // saturate. Keep both multiplies: combining them would alter
                // FP32 rounding. Linear/flat inputs do not need the undo.
                // Use the guest header, not the fallback host decoration:
                // all-Unused maps also skip reconstruction in Ryujinx
                // InstEmitAttribute.cs:171 and ShaderHeader.cs:30-48.
                let perspective =
                    track.input_mode(usize::from(location)) == Interpolation::Perspective;
                if perspective {
                    let coord = load(&mut w, &mut id, vec4, frag_coord.ok_or_else(|| bad(x))?);
                    let w_value = alu(&mut w, &mut id, Op::CompositeExtract, float, &[coord, 3]);
                    value = alu(&mut w, &mut id, Op::FMul, float, &[value, w_value]);
                }
                let inv_w = source(
                    &mut w, &mut id, inv_w, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let inv_w = alu(&mut w, &mut id, Op::Bitcast, float, &[inv_w]);
                value = alu(&mut w, &mut id, Op::FMul, float, &[value, inv_w]);
                // Same NaN-aware saturation as MUFU: Ryujinx
                // EmitterContextInsts.cs:419-423; Mesa nir_opcodes.py:265.
                value = alu(
                    &mut w,
                    &mut id,
                    Op::ExtInst,
                    float,
                    &[glsl_std_450, 81, value, zero_float, one_float],
                );
                let bits = alu(&mut w, &mut id, Op::Bitcast, uint, &[value]);
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    bits,
                    pred,
                    uint,
                    boolt,
                );
            }
            InstructionKind::LegacyVaryingLoad(ops::LegacyVaryingLoad {
                dst,
                varying,
                interpolation,
                inv_w,
            }) => {
                S::fragment_only(x)?;
                let flat = match interpolation {
                    LegacyInterpolation::Pass => false,
                    LegacyInterpolation::State => state
                        .legacy_color_flat_shading
                        .ok_or(Stopped::NoLayout(NoLayout::LegacyColorShading))?,
                };
                let location = varying
                    .location(track.legacy_input_locations())
                    .ok_or(Stopped::NoLayout(NoLayout::LegacyVaryingLocations))?;
                let input = input_by_location
                    .get(&location)
                    .copied()
                    .ok_or_else(|| bad(x))?;
                let value = load(&mut w, &mut id, vec4, input);
                let component = alu(
                    &mut w,
                    &mut id,
                    Op::CompositeExtract,
                    float,
                    &[value, u32::from(varying.component())],
                );
                let mut component = if flat {
                    component
                } else {
                    let coord = load(&mut w, &mut id, vec4, frag_coord.ok_or_else(|| bad(x))?);
                    let reciprocal_w =
                        alu(&mut w, &mut id, Op::CompositeExtract, float, &[coord, 3]);
                    let value = alu(&mut w, &mut id, Op::FMul, float, &[component, reciprocal_w]);
                    inst(annotations, Op::Decorate, &[value, 42]);
                    value
                };
                if interpolation == LegacyInterpolation::State && !flat {
                    let multiplier = load(&mut w, &mut id, uint, regs[usize::from(inv_w.0)]);
                    let multiplier = alu(&mut w, &mut id, Op::Bitcast, float, &[multiplier]);
                    component = alu(&mut w, &mut id, Op::FMul, float, &[component, multiplier]);
                    inst(annotations, Op::Decorate, &[component, 42]);
                }
                let value = alu(&mut w, &mut id, Op::Bitcast, uint, &[component]);
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    value,
                    pred,
                    uint,
                    boolt,
                );
            }
            InstructionKind::IpaPass(ops::IpaPass {
                dst,
                addr,
                inv_w,
                offset,
                loc,
                freq,
                saturate,
            })
            | InstructionKind::IpaPassMulW(ops::IpaPassMulW {
                dst,
                addr,
                inv_w,
                offset,
                loc,
                freq,
                saturate,
            }) => {
                // The public IR must obey the same bounds as the decoder.
                S::fragment_only(x)?;
                if loc != InterpLoc::Default
                    || offset != Source::Register(Register(255))
                    || !matches!(inv_w, Source::Register(_))
                    || (matches!(x.kind, InstructionKind::IpaPass(ops::IpaPass { .. }))
                        && (!matches!(freq, InterpFreq::Pass | InterpFreq::Constant)
                            || inv_w != Source::Register(Register(255))))
                    || (matches!(
                        x.kind,
                        InstructionKind::IpaPassMulW(ops::IpaPassMulW { .. })
                    ) && freq != InterpFreq::PassMulW)
                {
                    return Err(bad(x));
                }
                if let Some(Attribute::FrontFacing) = attribute(addr) {
                    // FrontFacing is a Bool input (Ryujinx IoMap.cs:24) that
                    // a register receives as -1 or 0 (CodeGenContext.cs:
                    // 342-348, IrConsts.True). It is flat: nothing to
                    // interpolate or multiply, and SAT never reaches it.
                    if freq == InterpFreq::PassMulW {
                        return Err(bad(x));
                    }
                    let facing = load(&mut w, &mut id, boolt, front_facing.ok_or_else(|| bad(x))?);
                    let ones = late_constant(late_globals, &mut id, uint, u32::MAX);
                    let value = alu(&mut w, &mut id, Op::Select, uint, &[facing, ones, zero]);
                    put(
                        &mut w,
                        &mut id,
                        regs[dst.0 as usize],
                        value,
                        pred,
                        uint,
                        boolt,
                    );
                } else {
                    let interpolation = match attribute(addr) {
                        Some(Attribute::Generic { location, .. }) => {
                            track.input_mode(usize::from(location))
                        }
                        _ => Interpolation::Unused,
                    };
                    // CONSTANT needs a flat linked interface. Its operation cannot
                    // change a perspective declaration into a provoking-vertex load.
                    // Ryujinx AttributeMap.cs:68; InstEmitAttribute.cs:167-175;
                    // Translation/ShaderHeader.cs:30-48,162-167 selects the map.
                    if freq == InterpFreq::Constant && interpolation != Interpolation::Constant {
                        return Err(bad(x));
                    }
                    let (input, component_index) = match attribute(addr) {
                        Some(Attribute::Generic {
                            location,
                            component,
                        }) => (
                            input_by_location
                                .get(&location)
                                .copied()
                                .ok_or_else(|| bad(x))?,
                            component,
                        ),
                        Some(Attribute::FragCoord { component }) => {
                            (frag_coord.ok_or_else(|| bad(x))?, component)
                        }
                        Some(Attribute::FrontFacing) | None => return Err(bad(x)),
                    };
                    let value = load(&mut w, &mut id, vec4, input);
                    let component = id;
                    id += 1;
                    inst(
                        &mut w,
                        Op::CompositeExtract,
                        &[float, component, value, u32::from(component_index)],
                    );
                    // Ryujinx InstEmitAttribute.cs:167-175,208-217 first undoes
                    // host perspective interpolation, then MUL applies B. Keep
                    // separate FP32 operations and forbid contraction to preserve
                    // rounding. A general B need not be the reciprocal of W.
                    // NoContraction=42: Mesa src/compiler/spirv/
                    // spirv.core.grammar.json:15033-15036; Ryujinx
                    // CodeGen/Spirv/Instructions.cs:2035-2044.
                    let mut component = component;
                    if interpolation == Interpolation::Perspective {
                        let coord = load(&mut w, &mut id, vec4, frag_coord.ok_or_else(|| bad(x))?);
                        let reciprocal_w =
                            alu(&mut w, &mut id, Op::CompositeExtract, float, &[coord, 3]);
                        component =
                            alu(&mut w, &mut id, Op::FMul, float, &[component, reciprocal_w]);
                        inst(annotations, Op::Decorate, &[component, 42]); // NoContraction
                    }
                    if freq == InterpFreq::PassMulW {
                        let multiplier = source(
                            &mut w, &mut id, inv_w, &regs, uint, constants, cb_types, cb_vars,
                            vec4, float, zero,
                        );
                        let multiplier = alu(&mut w, &mut id, Op::Bitcast, float, &[multiplier]);
                        component = alu(&mut w, &mut id, Op::FMul, float, &[component, multiplier]);
                        inst(annotations, Op::Decorate, &[component, 42]); // NoContraction
                    }
                    if saturate {
                        // Ryujinx InstEmitAttribute.cs:215 saturates the final
                        // value; NaN-aware NClamp as for IpaSaturate above.
                        component = alu(
                            &mut w,
                            &mut id,
                            Op::ExtInst,
                            float,
                            &[glsl_std_450, 81, component, zero_float, one_float],
                        );
                    }
                    let value = id;
                    id += 1;
                    inst(&mut w, Op::Bitcast, &[uint, value, component]);
                    put(
                        &mut w,
                        &mut id,
                        regs[dst.0 as usize],
                        value,
                        pred,
                        uint,
                        boolt,
                    )
                }
            }
            InstructionKind::GeometryOut(ops::GeometryOut { emit, cut, stream }) => {
                stage_io.as_ref().ok_or_else(|| bad(x))?.emit_vertex(
                    &mut w,
                    &mut id,
                    emit,
                    cut,
                    stream,
                    pred,
                    boolt,
                    late_globals,
                )?;
            }
            InstructionKind::GeometryOutHandle(ops::GeometryOutHandle {
                emit,
                cut,
                stream,
                dst: _,
                handle: _,
            }) => {
                stage_io.as_ref().ok_or_else(|| bad(x))?.emit_vertex(
                    &mut w,
                    &mut id,
                    emit,
                    cut,
                    stream,
                    pred,
                    boolt,
                    late_globals,
                )?;
            }
            InstructionKind::AttributeAddress(ops::AttributeAddress {
                dst,
                offset,
                address,
            }) => {
                // Ryujinx InstEmitAttribute.cs:12-16 keeps the byte address
                // as source + immediate. No hardware handle is needed.
                let value = load(&mut w, &mut id, uint, regs[offset.0 as usize]);
                let address = late_constant(late_globals, &mut id, uint, u32::from(address));
                let value = alu(&mut w, &mut id, Op::IAdd, uint, &[value, address]);
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    value,
                    pred,
                    uint,
                    boolt,
                );
            }
            InstructionKind::IsbeRead(ops::IsbeRead { dst, offset }) => {
                // Ryujinx InstEmitAttribute.cs:220-229 maps the default input
                // ISBE offset to the vertex index within the current primitive,
                // copying Ra to Rd in every stage.
                let value = load(&mut w, &mut id, uint, regs[offset.0 as usize]);
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    value,
                    pred,
                    uint,
                    boolt,
                );
            }
            InstructionKind::UnmappedAttributeLoad(ops::UnmappedAttributeLoad {
                dst,
                address,
                components,
            }) => {
                // Adapted from Ryujinx Instructions/AttributeMap.cs:129-133
                // and InstEmitAttribute.cs:34-41 under the MIT notice above
                // (Copyright (c) Ryujinx Team and Contributors): no lane of
                // the offset has an attribute, so each lane up to RZ logs
                // its offset and reads 0. No interface variable is touched,
                // so the guard needs no merge block.
                for (lane, register) in (usize::from(dst.0)..255)
                    .take(usize::from(components))
                    .enumerate()
                {
                    let offset = address + lane as u16 * 4;
                    eprintln!("shadowbox: Attribute offset 0x{offset:X} is not valid.");
                    put(&mut w, &mut id, regs[register], zero, pred, uint, boolt);
                }
            }
            InstructionKind::IndexedPatchLoad(ops::IndexedPatchLoad {
                dst,
                base,
                index,
                components,
                bank,
                lanes,
            }) => {
                // StageIo decided whether this stage has the bank.
                let Some((io, words)) = stage_io
                    .as_ref()
                    .and_then(|io| io.patch_words(bank).map(|words| (io, words)))
                else {
                    // No patch bank in this stage: whatever Ra holds, each
                    // lane is the static lane Ryujinx reads for this word,
                    // which has no attribute here. Adapted from Ryujinx
                    // AttributeMap.cs:129-141 under the MIT notice above:
                    // log and read 0, stopping at RZ (InstEmitAttribute.cs:
                    // 34-41).
                    for ((address, lane), register) in lanes.iter(base).zip(usize::from(dst.0)..255)
                    {
                        eprintln!("shadowbox: {}", lane.zero_reason(address, stage));
                        put(&mut w, &mut id, regs[register], zero, pred, uint, boolt);
                    }
                    for merge in memory_merge.into_iter() {
                        inst(&mut w, Op::Branch, &[merge]);
                        inst(&mut w, Op::Label, &[merge]);
                    }
                    continue;
                };
                // Mesa nak_nir_lower_vtg_io.c:203,208-211: the hardware
                // address is the immediate plus Ra (never RZ: decode). Snapshot
                // Ra before any destination is written, as for the vertex
                // operand (InstEmitAttribute.cs:32). Each lane compares its
                // address with every declared word of the bank; an address
                // naming none reads 0, as an unmapped static lane does
                // (AttributeMap.cs:129-141).
                let offset = load(&mut w, &mut id, uint, regs[usize::from(index.0)]);
                for (lane, register) in (usize::from(dst.0)..255)
                    .take(usize::from(components))
                    .enumerate()
                {
                    let lane_base = late_constant(
                        late_globals,
                        &mut id,
                        uint,
                        u32::from(base) + lane as u32 * 4,
                    );
                    let address = alu(&mut w, &mut id, Op::IAdd, uint, &[lane_base, offset]);
                    let mut value = zero;
                    for &word in &words {
                        let (pointer, ty) = io.pointer(
                            &mut w,
                            &mut id,
                            word,
                            true,
                            bank,
                            None,
                            None,
                            &regs,
                            late_globals,
                        )?;
                        let loaded = load(&mut w, &mut id, ty, pointer);
                        let loaded = alu_when(
                            &mut w,
                            &mut id,
                            ty == float,
                            Op::Bitcast,
                            uint,
                            &[loaded],
                            loaded,
                        );
                        let word = late_constant(late_globals, &mut id, uint, u32::from(word));
                        let hit = alu(&mut w, &mut id, Op::IEqual, boolt, &[address, word]);
                        value = alu(&mut w, &mut id, Op::Select, uint, &[hit, loaded, value]);
                    }
                    put(&mut w, &mut id, regs[register], value, pred, uint, boolt);
                }
                for merge in memory_merge.into_iter() {
                    inst(&mut w, Op::Branch, &[merge]);
                    inst(&mut w, Op::Label, &[merge]);
                }
            }
            InstructionKind::AttributeLoad(ops::AttributeLoad {
                dst,
                vertex,
                address,
                lanes,
                ..
            }) => {
                if let Some(io) = &stage_io {
                    // Ryujinx InstEmitAttribute.cs:32 snapshots the vertex
                    // before writing any destination. ALD may overwrite its
                    // own vertex source with its first component.
                    let vertex =
                        vertex.snapshot(|r| load(&mut w, &mut id, uint, regs[r.0 as usize]));
                    stage_lanes!(io, dst, address, lanes, false, vertex);
                }
                fixed_lanes!(dst, address, lanes);
            }
            InstructionKind::OutputAttributeLoad(ops::OutputAttributeLoad {
                dst,
                vertex,
                address,
                lanes,
                ..
            }) => {
                if let Some(io) = &stage_io {
                    let vertex = vertex.snapshot(|rb| control_point!(rb));
                    stage_lanes!(io, dst, address, lanes, false, vertex);
                }
                fixed_lanes!(dst, address, lanes);
            }
            InstructionKind::PatchAttributeLoad(ops::PatchAttributeLoad {
                dst,
                address,
                lanes,
                ..
            }) => {
                if let Some(io) = &stage_io {
                    stage_lanes!(io, dst, address, lanes, true, None);
                }
                fixed_lanes!(dst, address, lanes);
            }
            InstructionKind::PhysicalAttributeLoad(ops::PhysicalAttributeLoad {
                dst,
                vertex,
                offset,
            }) => {
                if let Some(io) = &stage_io {
                    // InstEmitAttribute.cs:32 snapshots the vertex first.
                    let vertex =
                        vertex.snapshot(|r| load(&mut w, &mut id, uint, regs[r.0 as usize]));
                    stage_physical!(io, dst, AttributeBank::Input, vertex, offset);
                }
                // Ryujinx AttributeMap.cs:63-81 has only graphics attributes;
                // compute has no physical input bank. Vertex and fragment
                // have no primitive vertex (AttributeMap.cs:323-333), so a
                // physical read there ignores Rb.
                if stage == Stage::Compute {
                    return Err(bad(x));
                }
                // Ryujinx InstEmitAttribute.cs:43-51: physical input ALD
                // loads the byte address register before decomposing it.
                // A scalar access chain avoids loading/extracting a vec4.
                let byte_address = load(&mut w, &mut id, uint, regs[offset.0 as usize]);
                let base = late_constant(late_globals, &mut id, uint, 0x80);
                let relative = alu(&mut w, &mut id, Op::ISub, uint, &[byte_address, base]);
                let location = alu(
                    &mut w,
                    &mut id,
                    Op::ShiftRightLogical,
                    uint,
                    &[relative, constants[&4]],
                );
                let component = alu(
                    &mut w,
                    &mut id,
                    Op::ShiftRightLogical,
                    uint,
                    &[relative, constants[&2]],
                );
                let component = alu(
                    &mut w,
                    &mut id,
                    Op::BitwiseAnd,
                    uint,
                    &[component, constants[&3]],
                );
                let pointer = alu(
                    &mut w,
                    &mut id,
                    Op::AccessChain,
                    input_array_scalar_ptr.ok_or_else(|| bad(x))?,
                    &[dynamic_input.ok_or_else(|| bad(x))?, location, component],
                );
                let value = load(&mut w, &mut id, float, pointer);
                let value = alu(&mut w, &mut id, Op::Bitcast, uint, &[value]);
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    value,
                    pred,
                    uint,
                    boolt,
                );
            }
            InstructionKind::PhysicalOutputAttributeLoad(ops::PhysicalOutputAttributeLoad {
                dst,
                vertex,
                offset,
            }) => {
                if let Some(io) = &stage_io {
                    let vertex = vertex.snapshot(|rb| control_point!(rb));
                    stage_physical!(io, dst, AttributeBank::Output, vertex, offset);
                }
                // Output physical access outside the stage interface needs a
                // shared output bank and remains unsupported here rather
                // than reading Input.
                return Err(bad(x));
            }
            InstructionKind::LegacyVaryingStore(ops::LegacyVaryingStore { data, varying }) => {
                S::vertex_only(x)?;
                let location = varying
                    .location(track.legacy_output_locations())
                    .ok_or(Stopped::NoLayout(NoLayout::LegacyVaryingLocations))?;
                let variable = vertex_outputs
                    .iter()
                    .find(|(candidate, _)| *candidate == u32::from(location))
                    .map(|(_, variable)| *variable)
                    .ok_or_else(|| bad(x))?;
                let pointer = fresh(&mut id);
                inst(
                    &mut w,
                    Op::AccessChain,
                    &[
                        pfloat_out,
                        pointer,
                        variable,
                        constants[&u32::from(varying.component())],
                    ],
                );
                let bits = load(&mut w, &mut id, uint, regs[usize::from(data.0)]);
                let value = alu(&mut w, &mut id, Op::Bitcast, float, &[bits]);
                let value = if let Some((predicate, inverted)) = pred {
                    let condition = load(&mut w, &mut id, boolt, predicate);
                    let old = load(&mut w, &mut id, float, pointer);
                    let (true_value, false_value) =
                        if inverted { (old, value) } else { (value, old) };
                    alu(
                        &mut w,
                        &mut id,
                        Op::Select,
                        float,
                        &[condition, true_value, false_value],
                    )
                } else {
                    value
                };
                inst(&mut w, Op::Store, &[pointer, value]);
            }
            InstructionKind::PointSizeStore(ops::PointSizeStore { data }) => {
                S::vertex_only(x)?;
                let pointer = outputs.point_size().ok_or_else(|| bad(x))?;
                let bits = load(&mut w, &mut id, uint, regs[usize::from(data.0)]);
                let value = alu(&mut w, &mut id, Op::Bitcast, float, &[bits]);
                put(&mut w, &mut id, pointer, value, pred, float, boolt);
            }
            InstructionKind::AttributeStore(ops::AttributeStore {
                data,
                address,
                components,
                patch,
                offset,
            }) => {
                if let Some(io) = &stage_io {
                    for c in 0..components {
                        // Ryujinx InstEmitAttribute.cs:109-114 stops when the
                        // source register index goes beyond RZ, not at RZ.
                        if usize::from(data.0) + usize::from(c) > 255 {
                            break;
                        }
                        let (pointer, ty) = io.pointer(
                            &mut w,
                            &mut id,
                            address + u16::from(c) * 4,
                            patch,
                            AttributeBank::Output,
                            None,
                            offset,
                            &regs,
                            late_globals,
                        )?;
                        // Ryujinx OperandHelper.cs:43-47: reading RZ yields
                        // zero even if an earlier instruction named it as dst.
                        let source = usize::from(data.0) + usize::from(c);
                        let value = load(&mut w, &mut id, uint, regs[source]);
                        // PrimitiveId is already a 32-bit integer bit pattern
                        // (Ryujinx AttributeMap.cs:63, Spirv/IoMap.cs:34).
                        let value = if ty == uint {
                            value
                        } else {
                            alu(&mut w, &mut id, Op::Bitcast, ty, &[value])
                        };
                        put(&mut w, &mut id, pointer, value, pred, ty, boolt);
                    }
                    for merge in memory_merge.into_iter() {
                        inst(&mut w, Op::Branch, &[merge]);
                        inst(&mut w, Op::Label, &[merge]);
                    }
                    continue;
                }
                // Ryujinx AttributeMap.cs:67-68 permits these AST outputs in
                // VTG stages only. Fragment exports come from terminal GPRs.
                S::vertex_only(x)?;
                if offset.is_some() || patch || !(1..=4).contains(&components) {
                    return Err(bad(x));
                }
                // Store each component when the AST executes: 0x70..0x7c is
                // gl_Position, 0x80 + location * 16 + component * 4 a generic
                // output (Mesa nak_private.h attribute map).
                if address % 4 != 0 {
                    return Err(bad(x));
                }
                for component in 0..u16::from(components) {
                    // Ryujinx InstEmitAttribute.cs:109-114 bounds the GPR range.
                    let source = usize::from(data.0) + usize::from(component);
                    if source > 255 {
                        break;
                    }
                    let a = address + component * 4;
                    let (variable, index) = if (0x70..0x80).contains(&a) {
                        (output, u32::from((a - 0x70) / 4))
                    } else if (0x2c0..0x2e0).contains(&a) {
                        // Ryujinx AttributeMap.cs (ClipDistance0..7, VTG
                        // outputs) and CodeGen/Spirv/IoMap.cs:19.
                        let variable = outputs.clip_distance().ok_or_else(|| bad(x))?;
                        (variable, u32::from((a - 0x2c0) / 4))
                    } else if (0x80..0x280).contains(&a) {
                        let location = u32::from((a - 0x80) / 16);
                        let variable = vertex_outputs
                            .iter()
                            .find(|(l, _)| *l == location)
                            .map(|(_, v)| *v)
                            .ok_or_else(|| bad(x))?;
                        (variable, u32::from((a - 0x80) % 16 / 4))
                    } else {
                        return Err(bad(x));
                    };
                    let pointer = fresh(&mut id);
                    inst(
                        &mut w,
                        Op::AccessChain,
                        &[pfloat_out, pointer, variable, constants[&index]],
                    );
                    // Ryujinx OperandHelper.cs:43-47: RZ always reads zero.
                    let u = load(&mut w, &mut id, uint, regs[source]);
                    let value = fresh(&mut id);
                    inst(&mut w, Op::Bitcast, &[float, value, u]);
                    let value = if let Some((predicate, inverted)) = pred {
                        // A predicated store keeps the old value where the
                        // predicate does not hold.
                        let condition = load(&mut w, &mut id, boolt, predicate);
                        let old = load(&mut w, &mut id, float, pointer);
                        let v = fresh(&mut id);
                        let (t, f) = if inverted { (old, value) } else { (value, old) };
                        inst(&mut w, Op::Select, &[float, v, condition, t, f]);
                        v
                    } else {
                        value
                    };
                    inst(&mut w, Op::Store, &[pointer, value]);
                }
            }
            InstructionKind::GeometryAttributeStore(ops::GeometryAttributeStore {
                data,
                address,
                handle: _,
            }) => {
                S::geometry_only(x)?;
                let io = stage_io.as_ref().ok_or_else(|| bad(x))?;
                let (pointer, ty) = io.pointer(
                    &mut w,
                    &mut id,
                    address,
                    false,
                    AttributeBank::Output,
                    None,
                    None,
                    &regs,
                    late_globals,
                )?;
                let value = load(&mut w, &mut id, uint, regs[usize::from(data.0)]);
                let value = if ty == uint {
                    value
                } else {
                    alu(&mut w, &mut id, Op::Bitcast, ty, &[value])
                };
                put(&mut w, &mut id, pointer, value, pred, ty, boolt);
            }
            InstructionKind::MuFu(ops::MuFu {
                dst,
                src,
                op,
                abs,
                negate,
                saturate,
            }) => {
                let input = load(&mut w, &mut id, uint, regs[src.0 as usize]);
                let result = mufu::MufuEmitter {
                    w: &mut w,
                    id: &mut id,
                    uint,
                    float,
                    boolt,
                    glsl_std_450,
                    one_float,
                    zero_float,
                    zero,
                    constants,
                    late_globals: &mut *late_globals,
                    double: double_types
                        .zip(double_one)
                        .map(|((double, words), one)| (double, words, one))
                        .ok_or(mufu::Undeclared),
                }
                .emit(op, input, abs, negate, saturate)
                .map_err(|mufu::Undeclared| bad(x))?;
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    result,
                    pred,
                    uint,
                    boolt,
                )
            }
            InstructionKind::Rro(ops::Rro {
                dst,
                src,
                abs,
                negate,
                // Both modes: Ryujinx InstEmitMultifunction.cs:86-94.
                ex2: _,
            }) => {
                // Ryujinx Instructions/InstEmitMultifunction.cs:86-94 defers
                // range reduction to MUFU. Apply FP abs then negation directly
                // to the sign bit, preserving payload bits with fewer ops.
                let mut value = source(
                    &mut w, &mut id, src, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let mut ints = IntegerEmitter {
                    w: &mut w,
                    id: &mut id,
                    uint,
                    boolt,
                    constants,
                };
                if abs {
                    value = ints.binary(Op::BitwiseAnd, value, constants[&0x7fff_ffff]);
                }
                if negate {
                    value = ints.binary(Op::BitwiseXor, value, constants[&0x8000_0000]);
                }
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    value,
                    pred,
                    uint,
                    boolt,
                )
            }
            InstructionKind::IMinMax(ops::IMinMax {
                dst,
                lhs,
                rhs,
                min,
                signed,
                cc,
            }) => {
                let ty = uint;
                let a = load(&mut w, &mut id, ty, regs[lhs.0 as usize]);
                let b = source(
                    &mut w, &mut id, rhs, &regs, ty, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let c = id;
                id += 1;
                inst(
                    &mut w,
                    Op::ExtInst,
                    &[
                        ty,
                        c,
                        glsl_std_450,
                        if signed {
                            if min {
                                39
                            } else {
                                42
                            }
                        } else {
                            if min {
                                38
                            } else {
                                41
                            }
                        },
                        a,
                        b,
                    ],
                );
                put(&mut w, &mut id, regs[dst.0 as usize], c, pred, uint, boolt);
                // Ryujinx InstEmitIntegerMinMax.cs:66: Z and N of the result.
                if cc {
                    cc_writer!().set_zn(c, Words::Single);
                }
            }
            // GLSL.std.450 SMin/UMin 39/38 and SMax/UMax 42/41, selected by
            // the runtime predicate (Ryujinx InstEmitIntegerMinMax.cs:45-73).
            InstructionKind::IMinMaxSelect(ops::IMinMaxSelect {
                dst,
                lhs,
                rhs,
                min_when,
                signed,
                cc,
            }) => {
                let a = load(&mut w, &mut id, uint, regs[lhs.0 as usize]);
                let b = source(
                    &mut w, &mut id, rhs, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let (min_op, max_op) = if signed { (39, 42) } else { (38, 41) };
                let min = alu(
                    &mut w,
                    &mut id,
                    Op::ExtInst,
                    uint,
                    &[glsl_std_450, min_op, a, b],
                );
                let max = alu(
                    &mut w,
                    &mut id,
                    Op::ExtInst,
                    uint,
                    &[glsl_std_450, max_op, a, b],
                );
                let c = load(&mut w, &mut id, boolt, ps[min_when.register as usize]);
                // An inverted selector picks the maximum when the predicate holds.
                let [when_true, when_false] =
                    [[min, max], [max, min]][usize::from(min_when.inverted)];
                let v = alu(
                    &mut w,
                    &mut id,
                    Op::Select,
                    uint,
                    &[c, when_true, when_false],
                );
                put(&mut w, &mut id, regs[dst.0 as usize], v, pred, uint, boolt);
                // Ryujinx InstEmitIntegerMinMax.cs:66: Z and N of the result.
                if cc {
                    cc_writer!().set_zn(v, Words::Single);
                }
            }
            InstructionKind::Lop(ops::Lop {
                dst,
                lhs,
                rhs,
                op,
                invert,
                predicate_output,
                extended,
                cc,
            }) => {
                let mut a = load(&mut w, &mut id, uint, regs[lhs.0 as usize]);
                let mut b = source(
                    &mut w, &mut id, rhs, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                // OpNot (200) for each inverted source.
                a = alu_when(&mut w, &mut id, invert[0], Op::Not, uint, &[a], a);
                b = alu_when(&mut w, &mut id, invert[1], Op::Not, uint, &[b], b);
                // PASS_B is srcs[1] after its inversion.
                let mut bitwise = |op| {
                    let v = fresh(&mut id);
                    inst(&mut w, op, &[uint, v, a, b]);
                    v
                };
                let v = match op {
                    LogicOp::And => bitwise(Op::BitwiseAnd),
                    LogicOp::Or => bitwise(Op::BitwiseOr),
                    LogicOp::Xor => bitwise(Op::BitwiseXor),
                    LogicOp::PassB => b,
                };
                put(&mut w, &mut id, regs[dst.0 as usize], v, pred, uint, boolt);
                if cc && extended {
                    // .X: Z is the result's zero AND the entry Z (Ryujinx
                    // InstEmitAluHelper.cs:102-114); N is the sign bit.
                    let old_zero = load(&mut w, &mut id, boolt, cc_zero);
                    let result_zero = alu(&mut w, &mut id, Op::IEqual, boolt, &[v, zero]);
                    let z = alu(
                        &mut w,
                        &mut id,
                        Op::LogicalAnd,
                        boolt,
                        &[old_zero, result_zero],
                    );
                    put(&mut w, &mut id, cc_zero, z, pred, boolt, boolt);
                    let sign = alu(
                        &mut w,
                        &mut id,
                        Op::ShiftRightLogical,
                        uint,
                        &[v, constants[&31]],
                    );
                    let negative = alu(&mut w, &mut id, Op::INotEqual, boolt, &[sign, zero]);
                    put(&mut w, &mut id, cc_sign, negative, pred, boolt, boolt);
                } else if cc {
                    write_cc(
                        &mut w,
                        &mut id,
                        v,
                        pred,
                        cc_zero,
                        cc_sign,
                        uint,
                        boolt,
                        zero,
                        constants[&31],
                    );
                }
                // Written last: the guard may be this predicate.
                // Predicate output modes F, T, Z, NZ (Ryujinx
                // InstEmitIntegerLogical.cs:145-160).
                for PredicateOutput { op, dst: p } in predicate_output.into_iter() {
                    let value =
                        LOP_PREDICATE[op as usize](&mut w, &mut id, v, (boolt, zero, yes, no));
                    put(&mut w, &mut id, ps[p as usize], value, pred, boolt, boolt);
                }
            }
            InstructionKind::Lop3(ops::Lop3 {
                dst,
                lhs,
                rhs,
                third,
                lut,
                predicate_output,
                extended,
                cc,
            }) => {
                let a = load(&mut w, &mut id, uint, regs[lhs.0 as usize]);
                let b = source(
                    &mut w, &mut id, rhs, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let c = load(&mut w, &mut id, uint, regs[third.0 as usize]);
                let value = lop3_lut(
                    &mut w,
                    &mut id,
                    uint,
                    &[a, b, c],
                    lut,
                    zero,
                    constants[&u32::MAX],
                );
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    value,
                    pred,
                    uint,
                    boolt,
                );
                // Ryujinx Translator.cs:339-385 guards the whole instruction.
                // Write CC before the predicate: its destination may alias
                // the guard, which must retain its entry value for every write.
                // Share the zero comparison with predicate output, and
                // compute extended Z directly instead of overwriting Z twice.
                // Ryujinx InstEmitAluHelper.cs:95-121 and
                // InstEmitIntegerLogical.cs:141-163 define these outputs.
                let result_zero = (cc
                    || predicate_output.is_some_and(|output| {
                        matches!(output.op, PredicateOp::Z | PredicateOp::Nz)
                    }))
                .then(|| alu(&mut w, &mut id, Op::IEqual, boolt, &[value, zero]));
                if cc {
                    let mut z = result_zero.unwrap();
                    if extended {
                        let old_zero = load(&mut w, &mut id, boolt, cc_zero);
                        z = alu(&mut w, &mut id, Op::LogicalAnd, boolt, &[old_zero, z]);
                    }
                    put(&mut w, &mut id, cc_zero, z, pred, boolt, boolt);
                    let sign = alu(
                        &mut w,
                        &mut id,
                        Op::ShiftRightLogical,
                        uint,
                        &[value, constants[&31]],
                    );
                    let negative = alu(&mut w, &mut id, Op::INotEqual, boolt, &[sign, zero]);
                    put(&mut w, &mut id, cc_sign, negative, pred, boolt, boolt);
                }
                // Write this last. The guard may alias the destination, so
                // all GPR/flag writes must read its original value first.
                if let Some(PredicateOutput { op, dst: p }) = predicate_output {
                    let condition = match op {
                        PredicateOp::F => no,
                        PredicateOp::T => yes,
                        PredicateOp::Z => result_zero.unwrap(),
                        PredicateOp::Nz => alu(
                            &mut w,
                            &mut id,
                            Op::LogicalNot,
                            boolt,
                            &[result_zero.unwrap()],
                        ),
                    };
                    put(
                        &mut w,
                        &mut id,
                        ps[p as usize],
                        condition,
                        pred,
                        boolt,
                        boolt,
                    );
                }
            }
            InstructionKind::Shl(ops::Shl {
                dst,
                lhs,
                rhs,
                wrap,
            }) => {
                let a = load(&mut w, &mut id, uint, regs[lhs.0 as usize]);
                let b = source(
                    &mut w, &mut id, rhs, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let v = shift(
                    &mut w,
                    &mut id,
                    (Op::ShiftLeftLogical, false, wrap),
                    a,
                    b,
                    (uint, boolt, zero),
                    constants,
                );
                put(&mut w, &mut id, regs[dst.0 as usize], v, pred, uint, boolt)
            }
            InstructionKind::Shr(ops::Shr {
                dst,
                lhs,
                rhs,
                wrap,
                signed,
                reverse,
            }) => {
                let a = load(&mut w, &mut id, uint, regs[lhs.0 as usize]);
                // SHR.BREV reverses A first; every later use, including the
                // clamped x >> 31, sees the reversed value
                // (Ryujinx InstEmitShift.cs:211-214).
                let a = alu_when(&mut w, &mut id, reverse, Op::BitReverse, uint, &[a], a);
                let b = source(
                    &mut w, &mut id, rhs, &regs, uint, constants, cb_types, cb_vars, vec4, float,
                    zero,
                );
                let v = shift(
                    &mut w,
                    &mut id,
                    (SHIFT_RIGHT[usize::from(signed)], signed, wrap),
                    a,
                    b,
                    (uint, boolt, zero),
                    constants,
                );
                put(&mut w, &mut id, regs[dst.0 as usize], v, pred, uint, boolt)
            }
            InstructionKind::Prmt(ops::Prmt {
                dst,
                lhs,
                rhs,
                selector,
            }) => {
                // Index mode (Mesa OpPrmt fold, ir.rs:5133-5150; PrmtSelByte
                // ir.rs:4997-5035): destination byte i takes the selector
                // nibble n = sel >> 4i; source (n >> 2) & 1, byte n & 3, and
                // n & 8 replicates that byte's sign bit.
                let sources = [
                    load(&mut w, &mut id, uint, regs[lhs.0 as usize]),
                    load(&mut w, &mut id, uint, regs[rhs.0 as usize]),
                ];
                let sel = source(
                    &mut w, &mut id, selector, &regs, uint, constants, cb_types, cb_vars, vec4,
                    float, zero,
                );
                let mut result = zero;
                for byte in 0..4u32 {
                    let mut next = || {
                        let v = id;
                        id += 1;
                        v
                    };
                    let nibble = next();
                    inst(
                        &mut w,
                        Op::ShiftRightLogical,
                        &[uint, nibble, sel, constants[&(byte * 4)]],
                    );
                    let n = next();
                    inst(&mut w, Op::BitwiseAnd, &[uint, n, nibble, constants[&15]]);
                    let which = next();
                    inst(&mut w, Op::BitwiseAnd, &[uint, which, n, constants[&4]]);
                    let use_second = next();
                    inst(&mut w, Op::INotEqual, &[boolt, use_second, which, zero]);
                    let src = next();
                    inst(
                        &mut w,
                        Op::Select,
                        &[uint, src, use_second, sources[1], sources[0]],
                    );
                    let index = next();
                    inst(&mut w, Op::BitwiseAnd, &[uint, index, n, constants[&3]]);
                    let bits = next();
                    inst(&mut w, Op::IMul, &[uint, bits, index, constants[&8]]);
                    let shifted = next();
                    inst(&mut w, Op::ShiftRightLogical, &[uint, shifted, src, bits]);
                    let value = next();
                    inst(
                        &mut w,
                        Op::BitwiseAnd,
                        &[uint, value, shifted, constants[&255]],
                    );
                    let msb_flag = next();
                    inst(&mut w, Op::BitwiseAnd, &[uint, msb_flag, n, constants[&8]]);
                    let replicate = next();
                    inst(&mut w, Op::INotEqual, &[boolt, replicate, msb_flag, zero]);
                    let sign = next();
                    inst(
                        &mut w,
                        Op::BitwiseAnd,
                        &[uint, sign, value, constants[&128]],
                    );
                    let negative = next();
                    inst(&mut w, Op::INotEqual, &[boolt, negative, sign, zero]);
                    let filled = next();
                    inst(
                        &mut w,
                        Op::Select,
                        &[uint, filled, negative, constants[&255], zero],
                    );
                    let chosen = next();
                    inst(
                        &mut w,
                        Op::Select,
                        &[uint, chosen, replicate, filled, value],
                    );
                    let placed = next();
                    inst(
                        &mut w,
                        Op::ShiftLeftLogical,
                        &[uint, placed, chosen, constants[&(byte * 8)]],
                    );
                    let merged = next();
                    inst(&mut w, Op::BitwiseOr, &[uint, merged, result, placed]);
                    result = merged;
                }
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    result,
                    pred,
                    uint,
                    boolt,
                )
            }
            InstructionKind::Branch(ops::Branch { cc_test, .. })
            | InstructionKind::Reconverge(ops::Reconverge { cc_test, .. })
            | InstructionKind::Exit(ops::Exit { cc_test }) => {
                // EXIT uses the same CC truth table as BRA, with the return
                // case as successor (Ryujinx InstEmitFlowControl.cs:153-186).
                let taken = match x.kind {
                    InstructionKind::Branch(ops::Branch { target, .. })
                    | InstructionKind::Reconverge(ops::Reconverge { target, .. }) => {
                        constants[&case_of[&target.slot()]]
                    }
                    _ => constants[&EXIT_CASE],
                };
                let next_case = xs
                    .get(index + 1)
                    .map(|next| constants[&case_of[&next.address]])
                    .unwrap_or(constants[&EXIT_CASE]);
                macro_rules! cc {
                    ($v:expr) => {
                        load(&mut w, &mut id, boolt, $v)
                    };
                }
                macro_rules! not {
                    ($v:expr) => {{
                        let value = $v;
                        let n = id;
                        id += 1;
                        inst(&mut w, Op::LogicalNot, &[boolt, n, value]);
                        n
                    }};
                }
                macro_rules! bin {
                    ($op:expr, $a:expr, $b:expr) => {{
                        let left = $a;
                        let right = $b;
                        let n = id;
                        id += 1;
                        inst(&mut w, $op, &[boolt, n, left, right]);
                        n
                    }};
                }
                // F/T are encoded constants; folding them here avoids four
                // flag loads and their Boolean intermediates.
                let cc = match cc_test {
                    0 => no,
                    15 => yes,
                    // Reference-matched, hardware unproven: Ryujinx
                    // InstEmitFlowControl.cs:153-181 evaluates EXIT through
                    // GetCondition(.., IrConsts.False), and
                    // InstEmitConditionCode.cs:52-82 has no CSM24..29 case,
                    // so the reference never takes this EXIT.
                    24..=29 if matches!(x.kind, InstructionKind::Exit(ops::Exit { .. })) => no,
                    // Reference-matched, hardware unproven: BRA and the
                    // resolved pops reach EmitBranch's GetCondition(ccc)
                    // (InstEmitFlowControl.cs:295-297), whose true default
                    // (InstEmitConditionCode.cs:52-82) takes them, as CSETP.
                    24..=29 => yes,
                    _ => {
                        let z = cc!(cc_zero);
                        let s = cc!(cc_sign);
                        let c = cc!(cc_carry);
                        let o = cc!(cc_overflow);
                        let so = bin!(Op::LogicalNotEqual, s, o);
                        let ns = not!(s);
                        let nz = not!(z);
                        let nc = not!(c);
                        let no_o = not!(o);
                        // Adapted from Ryujinx InstEmitConditionCode.cs:56-81,
                        // under the MIT notice above. Bool XOR165, AND167, OR166,
                        // NOT168 (spirv.hpp:1923-1927).
                        match cc_test {
                            0x10 => no_o,
                            0x1 => bin!(Op::LogicalNotEqual, bin!(Op::LogicalAnd, s, nz), o),
                            0x2 => bin!(Op::LogicalAnd, ns, z),
                            0xa => z,
                            0x3 => bin!(Op::LogicalNotEqual, s, bin!(Op::LogicalOr, z, o)),
                            0x4 => not!(bin!(Op::LogicalOr, so, z)),
                            0x5 => nz,
                            0xd => bin!(Op::LogicalOr, s, nz),
                            0x6 => not!(so),
                            0x7 => not!(bin!(Op::LogicalAnd, s, z)),
                            0x8 => bin!(Op::LogicalAnd, s, z),
                            0x9 => so,
                            0xb => bin!(Op::LogicalOr, so, z),
                            0xc => bin!(Op::LogicalNotEqual, not!(s), bin!(Op::LogicalOr, o, z)),
                            0xe => bin!(Op::LogicalNotEqual, bin!(Op::LogicalOr, ns, z), o),
                            0x11 => nc,
                            0x12 => not!(s),
                            0x13 => bin!(Op::LogicalOr, z, nc),
                            0x14 => bin!(Op::LogicalAnd, c, nz),
                            0x15 => s,
                            0x16 => c,
                            0x17 => o,
                            // Ryujinx Decoders/InstDecoders.cs:110-111.
                            0x1e => bin!(Op::LogicalOr, s, z),
                            0x1f => not!(bin!(Op::LogicalOr, s, z)),
                            _ => return Err(bad(x)),
                        }
                    }
                };
                let value = if let Some((predicate, inverted)) = pred {
                    let p = load(&mut w, &mut id, boolt, predicate);
                    let p = alu_when(&mut w, &mut id, inverted, Op::LogicalNot, boolt, &[p], p);
                    let condition = {
                        let v = id;
                        id += 1;
                        inst(&mut w, Op::LogicalAnd, &[boolt, v, p, cc]);
                        v
                    };
                    let (when_true, when_false) = (taken, next_case);
                    let v = id;
                    id += 1;
                    inst(
                        &mut w,
                        Op::Select,
                        &[uint, v, condition, when_true, when_false],
                    );
                    v
                } else {
                    // PT still evaluates CC (Ryujinx InstEmitFlowControl.cs:295-312).
                    alu(&mut w, &mut id, Op::Select, uint, &[cc, taken, next_case])
                };
                inst(&mut w, Op::Store, &[pc_var, value]);
                inst(&mut w, Op::Branch, &[switch_merge]);
                reachable = false;
            }
            InstructionKind::BrxRegister(ops::BrxRegister {
                displacement,
                base,
                cc_test,
            }) => {
                // Address arithmetic: Ryujinx InstEmitFlowControl.cs:38-40.
                // Bundle targets run slot0 (Mesa sm50.rs:3418-3422;
                // nv50_ir_emit_gm107.cpp:551-555). Normalize only bundle
                // addresses, never a misaligned or missing instruction.
                let offset = load(&mut w, &mut id, uint, regs[displacement.0 as usize]);
                let address_base = base.runtime();
                let base_id = late_constant(late_globals, &mut id, uint, address_base);
                let target = alu(&mut w, &mut id, Op::IAdd, uint, &[offset, base_id]);
                let bundle_offset = alu(
                    &mut w,
                    &mut id,
                    Op::BitwiseAnd,
                    uint,
                    &[target, constants[&31]],
                );
                let is_bundle = alu(&mut w, &mut id, Op::IEqual, boolt, &[bundle_offset, zero]);
                let first_slot = alu(&mut w, &mut id, Op::IAdd, uint, &[target, constants[&8]]);
                let target = alu(
                    &mut w,
                    &mut id,
                    Op::Select,
                    uint,
                    &[is_bundle, first_slot, target],
                );
                let next_case = xs
                    .get(index + 1)
                    .map(|next| constants[&case_of[&next.address]])
                    .unwrap_or(constants[&EXIT_CASE]);
                // BRX uses GetCondition's true default for CSM24..29, like
                // BRA (Ryujinx InstEmitFlowControl.cs:26-42,295-312).
                let test = *CONDITION.get(usize::from(cc_test)).ok_or_else(|| bad(x))?;
                let cc = test(&mut ConditionFlags {
                    w: &mut w,
                    id: &mut id,
                    flags: [cc_zero, cc_sign, cc_carry, cc_overflow],
                    boolt,
                    yes,
                    no,
                });
                // Whole-instruction predication runs before the condition
                // test; either false input preserves the fallthrough.
                let condition = if let Some((predicate, inverted)) = pred {
                    let p = load(&mut w, &mut id, boolt, predicate);
                    let p = alu_when(&mut w, &mut id, inverted, Op::LogicalNot, boolt, &[p], p);
                    alu(&mut w, &mut id, Op::LogicalAnd, boolt, &[p, cc])
                } else {
                    cc
                };
                let value = alu(
                    &mut w,
                    &mut id,
                    Op::Select,
                    uint,
                    &[condition, target, next_case],
                );
                inst(&mut w, Op::Store, &[pc_var, value]);
                inst(&mut w, Op::Branch, &[switch_merge]);
                reachable = false;
            }
            InstructionKind::Brx(ops::Brx {
                index: selector_reg,
                base: _,
                table,
            }) => {
                // Ryujinx InstEmitFlowControl.cs:38-40,93-119 compares the
                // loaded byte displacement plus base with each target. Fold
                // the base into the constants and select the dispatch PC:
                // this removes its runtime add and per-target branches, and
                // never branches between sibling dispatch cases. Normalize
                // bundle destinations without changing the loaded offsets
                // (Ryujinx Decoder.cs:265-270). Duplicate offsets need one
                // comparison. Mesa src/compiler/spirv/spirv.h:1894-1895
                // defines Select169 and IEqual170.
                let selector = load(&mut w, &mut id, uint, regs[selector_reg.0 as usize]);
                let mut seen = HashSet::default();
                let mut value = constants[&EXIT_CASE];
                for target in table.targets() {
                    if !seen.insert(target.loaded()) {
                        continue;
                    }
                    let address = target.slot();
                    let matches = alu(
                        &mut w,
                        &mut id,
                        Op::IEqual,
                        boolt,
                        &[selector, constants[&target.loaded()]],
                    );
                    value = alu(
                        &mut w,
                        &mut id,
                        Op::Select,
                        uint,
                        &[matches, constants[&case_of[&address]], value],
                    );
                }
                // Whole-instruction predication follows Ryujinx
                // Translation/Translator.cs:359-393; a skipped BRX uses its
                // fallthrough block, just as a skipped BRA does.
                for (predicate, inverted) in pred.into_iter() {
                    let condition = load(&mut w, &mut id, boolt, predicate);
                    let condition = alu_when(
                        &mut w,
                        &mut id,
                        inverted,
                        Op::LogicalNot,
                        boolt,
                        &[condition],
                        condition,
                    );
                    let next_case = xs
                        .get(index + 1)
                        .map(|next| constants[&case_of[&next.address]])
                        .unwrap_or(constants[&EXIT_CASE]);
                    value = alu(
                        &mut w,
                        &mut id,
                        Op::Select,
                        uint,
                        &[condition, value, next_case],
                    );
                }
                inst(&mut w, Op::Store, &[pc_var, value]);
                inst(&mut w, Op::Branch, &[switch_merge]);
                reachable = false;
            }
            // Mesa from_nir.rs:3035-3050: KIL demotes without changing the PC.
            InstructionKind::Kill(ops::Kill) => {
                // Discard exists only in fragment shaders (execution model 4).
                S::fragment_only(x)?;
                if let Some((predicate, inverted)) = pred {
                    // @P KILL: discard only where the predicate holds.
                    let condition = load(&mut w, &mut id, boolt, predicate);
                    let condition = alu_when(
                        &mut w,
                        &mut id,
                        inverted,
                        Op::LogicalNot,
                        boolt,
                        &[condition],
                        condition,
                    );
                    let kill = fresh(&mut id);
                    let resume = fresh(&mut id);
                    inst(&mut w, Op::SelectionMerge, &[resume, 0]);
                    inst(&mut w, Op::BranchConditional, &[condition, kill, resume]);
                    inst(&mut w, Op::Label, &[kill]);
                    inst(&mut w, Op::DemoteToHelperInvocation, &[]);
                    inst(&mut w, Op::Branch, &[resume]);
                    inst(&mut w, Op::Label, &[resume]);
                } else {
                    inst(&mut w, Op::DemoteToHelperInvocation, &[]);
                }
            }
            InstructionKind::S2R(ops::S2R { dst, idx }) => {
                if idx == 0x11 {
                    // envytools gm107.c:135; Mesa sm50.rs:3215-3224.
                    // Ryujinx InstEmitMove.cs:82-84 loads InvocationId.
                    let variable = stage_io
                        .as_ref()
                        .and_then(|io| io.invocation())
                        .ok_or_else(|| bad(x))?;
                    let value = load(&mut w, &mut id, uint, variable);
                    put(
                        &mut w,
                        &mut id,
                        regs[dst.0 as usize],
                        value,
                        pred,
                        uint,
                        boolt,
                    );
                    continue;
                }
                if idx == 0 {
                    // Ryujinx InstEmitMove.cs:173-180: a scalar lane index,
                    // masked to Maxwell's 32-lane warp. The mask is identity
                    // on subgroups of at most 32 lanes, so no host branch.
                    let value = load(&mut w, &mut id, uint, lane_id.ok_or_else(|| bad(x))?);
                    let value = alu(
                        &mut w,
                        &mut id,
                        Op::BitwiseAnd,
                        uint,
                        &[value, constants[&31]],
                    );
                    put(
                        &mut w,
                        &mut id,
                        regs[dst.0 as usize],
                        value,
                        pred,
                        uint,
                        boolt,
                    );
                    continue;
                }
                if idx == 0x12 {
                    let value = id;
                    id += 1;
                    // Mesa identifies SR_YDIRECTION as the render-target Y
                    // orientation input (`envytools/envydis/gm107.c:136`;
                    // Mesa NAK emits this S2R through `sm50.rs:3215-3224`).
                    // It is a float specialization value, bitcast to the
                    // Maxwell register's uint representation.
                    inst(&mut w, Op::Bitcast, &[uint, value, y_direction.unwrap()]);
                    put(
                        &mut w,
                        &mut id,
                        regs[dst.0 as usize],
                        value,
                        pred,
                        uint,
                        boolt,
                    );
                    continue;
                }
                let value = match idx {
                    0x38..=0x3c => {
                        // Ryujinx InstEmitMove.cs:183-215 selects the 32-bit
                        // word containing this physical lane. A dynamic vector
                        // extract replaces its three comparisons and selects,
                        // working for every subgroup size up to 128.
                        let variable = mask_inputs
                            .iter()
                            .find(|(index, _, _)| *index == idx)
                            .map(|(_, _, variable)| *variable)
                            .ok_or_else(|| bad(x))?;
                        let vector = load(&mut w, &mut id, uint_vec4, variable);
                        let lane = load(&mut w, &mut id, uint, lane_id.unwrap());
                        let element = alu(
                            &mut w,
                            &mut id,
                            Op::ShiftRightLogical,
                            uint,
                            &[lane, constants[&5]],
                        );
                        // OpVectorExtractDynamic: Mesa spirv.core.grammar.json:888-898.
                        alu(
                            &mut w,
                            &mut id,
                            Op::VectorExtractDynamic,
                            uint,
                            &[vector, element],
                        )
                    }
                    0x13 if stage == Stage::Fragment => {
                        // Ryujinx InstEmitMove.cs:90-92 and IoMap.cs:45:
                        // HelperInvocation is a bool; true is integer -1
                        // (CodeGen/Spirv/CodeGenContext.cs:342-347; IrConsts.cs:5-6).
                        // SPIRV-Registry extensions/EXT/
                        // SPV_EXT_demote_to_helper_invocation.asciidoc:117-120:
                        // HelperInvocation is undefined after demotion. Query
                        // current state directly, with no software shadow flag.
                        let helper = if uses_demote {
                            alu(&mut w, &mut id, Op::IsHelperInvocationEXT, boolt, &[])
                        } else {
                            load(&mut w, &mut id, boolt, helper_invocation.unwrap())
                        };
                        alu(
                            &mut w,
                            &mut id,
                            Op::Select,
                            uint,
                            &[helper, constants[&u32::MAX], zero],
                        )
                    }
                    0x13 => zero,
                    // Reference-matched, not hardware-proven: wscalefactor_xy/z
                    // (envytools envydis/gm107.c:148-149) reach Ryujinx's
                    // Instructions/InstEmitMove.cs:165-167 default in every
                    // stage (switch76-168, write170). Reuse the uint constant
                    // directly: no runtime load, packing or conversion needed.
                    0x1e | 0x1f => zero,
                    0x1d => {
                        if let Some(variable) = patch_vertices {
                            let count = load(&mut w, &mut id, uint, variable);
                            alu(
                                &mut w,
                                &mut id,
                                Op::ShiftLeftLogical,
                                uint,
                                &[count, constants[&16]],
                            )
                        } else if let Some(count) = primitive_vertices {
                            // Fold the topology shift at translation time.
                            late_constant(late_globals, &mut id, uint, count << 16)
                        } else {
                            zero
                        }
                    }
                    0x20 => {
                        // Ryujinx InstEmitMove.cs:118-126: x | y<<16 | z<<26,
                        // with no extra masks. Share one vector load.
                        let vector = load(&mut w, &mut id, vec3, local_invocation_id.unwrap());
                        let parts: [u32; 3] = std::array::from_fn(|component| {
                            alu(
                                &mut w,
                                &mut id,
                                Op::CompositeExtract,
                                uint,
                                &[vector, component as u32],
                            )
                        });
                        let y = alu(
                            &mut w,
                            &mut id,
                            Op::ShiftLeftLogical,
                            uint,
                            &[parts[1], constants[&16]],
                        );
                        let z = alu(
                            &mut w,
                            &mut id,
                            Op::ShiftLeftLogical,
                            uint,
                            &[parts[2], constants[&26]],
                        );
                        let yz = alu(&mut w, &mut id, Op::BitwiseOr, uint, &[y, z]);
                        alu(&mut w, &mut id, Op::BitwiseOr, uint, &[parts[0], yz])
                    }
                    0x21..=0x23 | 0x25..=0x27 => {
                        // Ryujinx InstEmitMove.cs:129-147, IoMap.cs:20,44.
                        let (variable, component) = if idx <= 0x23 {
                            (local_invocation_id, idx - 0x21)
                        } else {
                            (workgroup_id, idx - 0x25)
                        };
                        let vector = load(&mut w, &mut id, vec3, variable.ok_or_else(|| bad(x))?);
                        alu(
                            &mut w,
                            &mut id,
                            Op::CompositeExtract,
                            uint,
                            &[vector, u32::from(component)],
                        )
                    }
                    _ => return Err(bad(x)),
                };
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    value,
                    pred,
                    uint,
                    boolt,
                );
            }
            // Decode resolved each consumer to its target. The region pass
            // wraps used pushes in nested dispatch loops; unused pushes need
            // no consumer tags (Ryujinx InstEmitFlowControl.cs:239-262).
            // Unresolved pushes fail above.
            InstructionKind::ResolvedSchedule(ops::ResolvedSchedule { .. })
            | InstructionKind::ResolvedPushTarget(ops::ResolvedPushTarget { .. })
            | InstructionKind::CbufPushTarget(ops::CbufPushTarget { .. }) => {}
            // Explicit policy exception for this encoding alone. Ryujinx
            // Decoders/InstTable.cs:320 has no destinations; InstEmit.cs:
            // 330-336 emits nothing. No predicate evaluation or SPIR-V is
            // needed for its exact reference result. Hardware is unproven.
            // Immediate20..47: Ryujinx Decoders/InstDecoders.cs:5231 (Imm28).
            InstructionKind::VoteVtg(ops::VoteVtg { .. }) => {}
            InstructionKind::Nop(ops::Nop) => {}
            // Mesa nak/sm50.rs:2937-2980 (OpCCtl): cache maintenance with
            // no data effect; Vulkan memory is coherent at each access.
            InstructionKind::CacheControl(ops::CacheControl { .. }) => {}
            InstructionKind::Texture(
                texture @ ops::Texture {
                    op,
                    dst,
                    src0,
                    src1,
                    dim,
                    mask,
                    offset,
                    scalar,
                    packed_f16,
                    lod_clamp,
                    multisample,
                    depth_compare,
                    nodep,
                    scalar_dst,
                    bound_handle,
                },
            ) => {
                // TLD.CL is not sampling MinLod: Ryujinx decodes CL but
                // does not implement it (InstDecoders.cs:4922;
                // InstEmitTexture.cs:65-72). Its semantics are unproven here.
                // Invalid Vulkan dimensions/modes must remain Unknown.
                // NODEP is Mesa's skip_helpers (nak_nir_lower_tex.c:454):
                // results unused by helper invocations. Only fragment
                // shaders have helpers, so elsewhere the flag has no effect.
                if (op == TexOp::Lod && stage != Stage::Fragment)
                    || lod_clamp
                    // 1D.LZ consumes only A, Ryujinx InstEmitTexture.cs:
                    // 417-428,492-508. Require unused B to be canonical RZ;
                    // an unexplained source field must not be ignored.
                    || (scalar && dim == 0 && matches!(op,
                        TexOp::Sample { lod: 1 } | TexOp::Fetch { lod: false })
                        && src1.0 != 255)
                    || (multisample && !matches!(dim, 2 | 3))
                    || (depth_compare && (dim == 4 || matches!(op, TexOp::Fetch { .. })))
                    || (matches!(op, TexOp::Gather { .. }) && !matches!(dim, 2 | 3 | 6 | 7))
                {
                    return Err(bad(x));
                }
                let results = texture_results[usize::from(nodep)].begin(&mut w, &mut id, boolt);
                let (dim_types, sampler_type, uvec2) =
                    texture_types.as_ref().expect("texture declarations");
                let uvec2 = *uvec2;
                let sampler_type = *sampler_type;
                let key = handles.get(&index).ok_or_else(|| bad(x))?;
                let k = handle_indices[key];
                if key_conflict[k] {
                    return Err(bad(x));
                }
                let (image_type, sampled, _) = *dim_types
                    .get(&(dim, depth_compare, multisample))
                    .ok_or_else(|| bad(x))?;
                let (image_var, sampler_var, array_pointers) = texture_bindings[k];
                let descriptor_index = array_pointers.map(|_| {
                    let register = if matches!(op, TexOp::Query { .. }) {
                        src0
                    } else {
                        src1
                    };
                    let indices = texture_array_indices
                        .as_ref()
                        .expect("dynamic texture index variables");
                    let raw = load(&mut w, &mut id, uint, indices[register.0 as usize]);
                    let last = constants[&(texture_array_lengths[key] - 1)];
                    alu(
                        &mut w,
                        &mut id,
                        Op::ExtInst,
                        uint,
                        &[glsl_std_450, 38, raw, last],
                    )
                });
                let image = if let (Some((image_ptr, _)), Some(index)) =
                    (array_pointers, descriptor_index)
                {
                    let pointer = fresh(&mut id);
                    inst(
                        &mut w,
                        Op::AccessChain,
                        &[image_ptr, pointer, image_var, index],
                    );
                    inst(annotations, Op::Decorate, &[pointer, 5300]);
                    let image = load(&mut w, &mut id, image_type, pointer);
                    inst(annotations, Op::Decorate, &[image, 5300]);
                    image
                } else {
                    load(&mut w, &mut id, image_type, image_var)
                };
                let sources = texture_sources(&texture).ok_or_else(|| bad(x))?;
                let reg = |r: u8, i: u8| regs[(r as usize + i as usize).min(255)];
                // Query operands are LOD, not coordinates (Ryujinx
                // InstEmitTexture.cs:1084-1089): TXQ loads no coordinates.
                let coordinates = CoordinateTypes {
                    reaching,
                    index,
                    regs: &regs,
                    uint,
                    float,
                    sampled: [float, vec2, fvec3, vec4],
                    fetched: [uint, uvec2, vec3, uvec4],
                };
                let offset_types = OffsetTypes {
                    uint,
                    uvec2,
                    uvec3,
                    float,
                    vec2,
                    fvec3,
                    vec4,
                    signed_int,
                };
                let coords = &sources.coords[..sources.coord_count];
                let mut operands_mask = 0u32;
                let mut operands = [0; 3];
                let mut operand_count = 0;
                let result_vec4 = match op {
                    TexOp::Sample { lod } => {
                        let mut coord = Coordinates {
                            w: &mut w,
                            globals: late_globals,
                            id: &mut id,
                            at: coordinates,
                        }
                        .sampled(coords, sources.layer);
                        let sampler = if let (Some((_, sampler_ptr)), Some(index)) =
                            (array_pointers, descriptor_index)
                        {
                            let pointer = fresh(&mut id);
                            inst(
                                &mut w,
                                Op::AccessChain,
                                &[sampler_ptr, pointer, sampler_var, index],
                            );
                            inst(annotations, Op::Decorate, &[pointer, 5300]);
                            let sampler = load(&mut w, &mut id, sampler_type, pointer);
                            inst(annotations, Op::Decorate, &[sampler, 5300]);
                            sampler
                        } else {
                            load(&mut w, &mut id, sampler_type, sampler_var)
                        };
                        let combined = fresh(&mut id);
                        inst(
                            &mut w,
                            Op::SampledImage,
                            &[sampled, combined, image, sampler],
                        );
                        if descriptor_index.is_some() {
                            inst(annotations, Op::Decorate, &[combined, 5300]);
                        }
                        // Ryujinx CodeGen/Spirv/Instructions.cs:1339-1348:
                        // non-fragment implicit sampling uses explicit zero
                        // or the encoded bias. Vulkan forbids implicit LOD
                        // outside fragment shaders.
                        // Ryujinx InstEmitTexture.cs:263-269,287-291 maps LLA7
                        // to level and LBA6 to bias. The same SPIR-V operation
                        // serves each pair without runtime mode comparisons.
                        let explicit = matches!(lod, 1 | 3 | 7) || stage != Stage::Fragment;
                        let lod_value = match lod {
                            0 => {
                                if explicit {
                                    Some(zero_float)
                                } else {
                                    None
                                }
                            }
                            1 => Some(zero_float),
                            _ => Some({
                                let value = load(
                                    &mut w,
                                    &mut id,
                                    uint,
                                    regs[sources.lod.ok_or_else(|| bad(x))?.0 as usize],
                                );
                                let f = fresh(&mut id);
                                inst(&mut w, Op::Bitcast, &[float, f, value]);
                                f
                            }),
                        };
                        if let Some(v) = lod_value {
                            // Bias (0x1) for modes2/6, Lod (0x2) otherwise.
                            operands_mask |= if matches!(lod, 2 | 6) && !explicit {
                                0x1
                            } else {
                                0x2
                            };
                            operands[operand_count] = v;
                            operand_count += 1;
                        }
                        if offset && dim == 6 {
                            return Err(bad(x));
                        }
                        if offset {
                            let register = sources.offset.ok_or_else(|| bad(x))?.0;
                            let count = texture_offset_components(dim).ok_or_else(|| bad(x))?;
                            if let Some(packed) = reaching.constant(index, register) {
                                if dim == 7 {
                                    // Ryujinx TexturePass.cs:399-405 leaves constant
                                    // offsets intact. Vulkan forbids cube ConstOffset;
                                    // do not invent a face-coordinate transformation.
                                    return Err(bad(x));
                                }
                                // Fold signed four-bit offsets, Ryujinx
                                // InstEmitTexture.cs:272-276. Keep the existing
                                // ConstOffset path without a runtime size query.
                                let cv = constant_texture_offset(
                                    late_globals,
                                    &mut id,
                                    &mut late_types_declared,
                                    (signed_int, ivec2, ivec3),
                                    packed,
                                    (count, 4, 4),
                                );
                                operands_mask |= 0x8;
                                operands[operand_count] = cv;
                                operand_count += 1;
                            } else {
                                // The reference uses mip zero for every
                                // non-fragment mode, including bias/explicit LOD
                                // (TexturePass.cs:618-620). The encoded LOD still
                                // controls the sample operand above.
                                let mip = if stage == Stage::Fragment {
                                    let spatial = if sources.layer.is_some() {
                                        Coordinates {
                                            w: &mut w,
                                            globals: late_globals,
                                            id: &mut id,
                                            at: coordinates,
                                        }
                                        .sampled(coords, None)
                                    } else {
                                        coord
                                    };
                                    implicit_mip(&mut w, &mut id, offset_types, combined, spatial)
                                } else {
                                    zero
                                };
                                let array = u8::from(sources.layer.is_some());
                                let packed = load(&mut w, &mut id, uint, regs[register as usize]);
                                coord = offset_coordinates(
                                    &mut w,
                                    &mut id,
                                    late_globals,
                                    offset_types,
                                    image,
                                    coord,
                                    RuntimeOffset {
                                        packed,
                                        first_bit: 0,
                                        axes: count,
                                        sizes: [count + array, 3][usize::from(dim == 7)],
                                        coordinates: count + array,
                                        mip,
                                    },
                                );
                            }
                        }
                        let v = id;
                        id += 1;
                        let mut args = [
                            if depth_compare { float } else { vec4 },
                            v,
                            combined,
                            coord,
                            0,
                            0,
                            0,
                            0,
                        ];
                        let mut arg_count = 4;
                        if depth_compare {
                            let bits = load(
                                &mut w,
                                &mut id,
                                uint,
                                regs[sources.dref.ok_or_else(|| bad(x))?.0 as usize],
                            );
                            let dref = fresh(&mut id);
                            inst(&mut w, Op::Bitcast, &[float, dref, bits]);
                            args[arg_count] = dref;
                            arg_count += 1;
                        }
                        if operands_mask != 0 {
                            args[arg_count] = operands_mask;
                            arg_count += 1;
                            args[arg_count..arg_count + operand_count]
                                .copy_from_slice(&operands[..operand_count]);
                            arg_count += operand_count;
                        }
                        // Ryujinx CodeGen/Spirv/Instructions.cs:1400-1436 uses a
                        // scalar Dref result. Reuse it for each selected channel.
                        inst(
                            &mut w,
                            match (depth_compare, explicit) {
                                (true, true) => Op::ImageSampleDrefExplicitLod,
                                (true, false) => Op::ImageSampleDrefImplicitLod,
                                (false, true) => Op::ImageSampleExplicitLod,
                                (false, false) => Op::ImageSampleImplicitLod,
                            },
                            &args[..arg_count],
                        );
                        v
                    }
                    TexOp::Fetch { lod } => {
                        let coord = Coordinates {
                            w: &mut w,
                            globals: late_globals,
                            id: &mut id,
                            at: coordinates,
                        }
                        .fetched_by(coords, sources.layer, sources.offset);
                        // texelFetch has no cube form.
                        if matches!(dim, 6 | 7) {
                            return Err(bad(x));
                        }
                        if !multisample {
                            let lod_value = if lod {
                                load(
                                    &mut w,
                                    &mut id,
                                    uint,
                                    regs[sources.lod.ok_or_else(|| bad(x))?.0 as usize],
                                )
                            } else {
                                zero
                            };
                            operands_mask |= 0x2;
                            operands[operand_count] = lod_value;
                            operand_count += 1;
                        }
                        if multisample {
                            let sample = load(
                                &mut w,
                                &mut id,
                                uint,
                                regs[sources.sample.ok_or_else(|| bad(x))?.0 as usize],
                            );
                            // Sample is 0x40, not Offset (0x10). Mesa
                            // nak_nir_lower_tex.c:259-262,362-375; Ryujinx
                            // CodeGen/Spirv/Instructions.cs:1369-1397.
                            operands_mask |= 0x40;
                            operands[operand_count] = sample;
                            operand_count += 1;
                        }
                        let v = id;
                        id += 1;
                        let mut args = [vec4, v, image, coord, operands_mask, 0, 0, 0];
                        let mut arg_count = 5;
                        args[arg_count..arg_count + operand_count]
                            .copy_from_slice(&operands[..operand_count]);
                        arg_count += operand_count;
                        inst(&mut w, Op::ImageFetch, &args[..arg_count]);
                        v
                    }
                    TexOp::Gather { component, offsets } => {
                        let coord = Coordinates {
                            w: &mut w,
                            globals: late_globals,
                            id: &mut id,
                            at: coordinates,
                        }
                        .sampled(coords, sources.layer);
                        let sampler = load(&mut w, &mut id, sampler_type, sampler_var);
                        let combined = fresh(&mut id);
                        inst(
                            &mut w,
                            Op::SampledImage,
                            &[sampled, combined, image, sampler],
                        );
                        // Gathers exist for 2D, 2D arrays and cubes only.
                        if !matches!(dim, 2 | 3 | 6 | 7)
                            || (offsets != GatherOffsets::Zero && !matches!(dim, 2 | 3))
                        {
                            return Err(bad(x));
                        }
                        // Dref gathers take the reference in place of the
                        // component, which Ryujinx ignores: InstEmitTexture.cs:
                        // 774-777 (TLD4) and 591-600 (TLD4S).
                        let gather =
                            [Op::ImageGather, Op::ImageDrefGather][usize::from(depth_compare)];
                        let selector = if depth_compare {
                            let bits = load(
                                &mut w,
                                &mut id,
                                uint,
                                regs[sources.dref.ok_or_else(|| bad(x))?.0 as usize],
                            );
                            let dref = fresh(&mut id);
                            inst(&mut w, Op::Bitcast, &[float, dref, bits]);
                            dref
                        } else {
                            constants[&(component as u32)]
                        };
                        if offsets == GatherOffsets::Four {
                            // PTP supplies four independently packed offsets.
                            // Ryujinx InstEmitTexture.cs:739-779 extracts eight
                            // signed six-bit components. Four ordinary gathers
                            // with dynamic Offset reproduce ConstOffsets; the W
                            // lane is the (0,0) texel chosen by Ryujinx's fallback.
                            // Dref gathers compare the same four texels.
                            let first = load(
                                &mut w,
                                &mut id,
                                uint,
                                regs[sources.offset.ok_or_else(|| bad(x))?.0 as usize],
                            );
                            let second = load(
                                &mut w,
                                &mut id,
                                uint,
                                regs[sources.offset2.ok_or_else(|| bad(x))?.0 as usize],
                            );
                            let packed = gather4_offsets(
                                &mut w, &mut id, first, second, uvec2, uint, constants,
                            );
                            let mut parts = [0; 4];
                            for (part, offset) in parts.iter_mut().zip(packed) {
                                let gathered = fresh(&mut id);
                                inst(
                                    &mut w,
                                    gather,
                                    &[vec4, gathered, combined, coord, selector, 0x10, offset],
                                );
                                *part = alu(
                                    &mut w,
                                    &mut id,
                                    Op::CompositeExtract,
                                    float,
                                    &[gathered, 3],
                                );
                            }
                            let v = fresh(&mut id);
                            inst(
                                &mut w,
                                Op::CompositeConstruct,
                                &[vec4, v, parts[0], parts[1], parts[2], parts[3]],
                            );
                            v
                        } else {
                            let v = id;
                            id += 1;
                            let mut args = [vec4, v, combined, coord, selector, 0, 0];
                            let mut arg_count = 5;
                            if offsets == GatherOffsets::One {
                                // TLD4 uses signed 6-bit offsets at 8-bit strides,
                                // Ryujinx InstEmitTexture.cs:756-760.
                                if !matches!(dim, 2 | 3) {
                                    return Err(bad(x));
                                }
                                let register = sources.offset.ok_or_else(|| bad(x))?.0;
                                if let Some(packed) = reaching.constant(index, register) {
                                    // Folding eliminates a register load and four
                                    // shifts. Ryujinx InstEmitTexture.cs:756-760
                                    // defines signed 6-bit values at 8-bit strides.
                                    args[arg_count] = 0x8;
                                    arg_count += 1;
                                    args[arg_count] = constant_texture_offset(
                                        late_globals,
                                        &mut id,
                                        &mut late_types_declared,
                                        (signed_int, ivec2, ivec3),
                                        packed,
                                        (2, 6, 8),
                                    );
                                    arg_count += 1;
                                } else {
                                    let packed =
                                        load(&mut w, &mut id, uint, regs[register as usize]);
                                    args[arg_count] = 0x10;
                                    arg_count += 1;
                                    args[arg_count] = unpack_offset(
                                        &mut w, &mut id, packed, uvec2, uint, constants,
                                    );
                                    arg_count += 1;
                                }
                            }
                            inst(&mut w, gather, &args[..arg_count]);
                            v
                        }
                    }
                    TexOp::Grad {
                        gradients,
                        offset: aoffi,
                    } => {
                        let spatial = Coordinates {
                            w: &mut w,
                            globals: late_globals,
                            id: &mut id,
                            at: coordinates,
                        }
                        .sampled(coords, None);
                        let sampler = load(&mut w, &mut id, sampler_type, sampler_var);
                        let combined = alu(
                            &mut w,
                            &mut id,
                            Op::SampledImage,
                            sampled,
                            &[image, sampler],
                        );
                        let count = sources.coord_count as u8;
                        let array = u8::from(sources.packed_layer.is_some());
                        // Spirv/Instructions.cs:1273-1279 assembles dPdx from the
                        // first coordinate-count B registers and dPdy from the
                        // next (InstEmitTexture.cs:987-996).
                        let gradient = gradients.0;
                        let mut derivative = |first: u8| {
                            let mut parts = [0; 3];
                            for (i, part) in (0..count).zip(&mut parts) {
                                let bits = load(&mut w, &mut id, uint, reg(gradient, first + i));
                                *part = alu(&mut w, &mut id, Op::Bitcast, float, &[bits]);
                            }
                            match count {
                                1 => parts[0],
                                _ => alu(
                                    &mut w,
                                    &mut id,
                                    Op::CompositeConstruct,
                                    offset_types.floats(count),
                                    &parts[..usize::from(count)],
                                ),
                            }
                        };
                        let dx = derivative(0);
                        let dy = derivative(count);
                        let mut coord = spatial;
                        let mut args = [vec4, 0, combined, 0, 0x4, dx, dy, 0];
                        let mut arg_count = 7;
                        match aoffi {
                            GradOffset::Absent => (),
                            // InstEmitTexture.cs:998-1011: signed nibbles from
                            // bit 16 of the packed A word.
                            GradOffset::Packed { word, axes } => {
                                match OffsetWord::of(reaching, index, word, dim) {
                                    OffsetWord::Constant(packed) => {
                                        args[4] |= 0x8;
                                        args[arg_count] = constant_texture_offset(
                                            late_globals,
                                            &mut id,
                                            &mut late_types_declared,
                                            (signed_int, ivec2, ivec3),
                                            packed >> 16,
                                            (axes, 4, 4),
                                        );
                                        arg_count += 1;
                                    }
                                    // Run-time offsets move the coordinates, as for
                                    // TEX: Ryujinx TexturePass.cs:501-525,591-620.
                                    OffsetWord::RunTime => {
                                        let mip = if stage == Stage::Fragment {
                                            implicit_mip(
                                                &mut w,
                                                &mut id,
                                                offset_types,
                                                combined,
                                                spatial,
                                            )
                                        } else {
                                            zero
                                        };
                                        let packed =
                                            load(&mut w, &mut id, uint, regs[word.0 as usize]);
                                        coord = offset_coordinates(
                                            &mut w,
                                            &mut id,
                                            late_globals,
                                            offset_types,
                                            image,
                                            coord,
                                            RuntimeOffset {
                                                packed,
                                                first_bit: 16,
                                                axes,
                                                sizes: [axes + array, 3][usize::from(dim == 7)],
                                                coordinates: count,
                                                mip,
                                            },
                                        );
                                    }
                                }
                            }
                        }
                        // Ryujinx InstEmitTexture.cs:976-985 masks the layer
                        // to bits 0..15; Spirv/Instructions.cs:1218-1225
                        // converts it as signed. It follows the coordinates.
                        for register in sources.packed_layer.into_iter() {
                            let packed = load(&mut w, &mut id, uint, regs[register.0 as usize]);
                            let low = late_constant(late_globals, &mut id, uint, 0xffff);
                            let layer = alu(&mut w, &mut id, Op::BitwiseAnd, uint, &[packed, low]);
                            let layer = alu(&mut w, &mut id, Op::ConvertSToF, float, &[layer]);
                            coord = alu(
                                &mut w,
                                &mut id,
                                Op::CompositeConstruct,
                                offset_types.floats(count + 1),
                                &[coord, layer],
                            );
                        }
                        let v = fresh(&mut id);
                        args[1] = v;
                        args[3] = coord;
                        inst(&mut w, Op::ImageSampleExplicitLod, &args[..arg_count]);
                        v
                    }
                    TexOp::Lod => {
                        // LOD derivatives are implemented only for fragments;
                        // Ryujinx Spirv/Instructions.cs:859-895 is the shape.
                        // Other stages retain an unsupported-instruction error.
                        if stage != Stage::Fragment {
                            return Err(bad(x));
                        }
                        let coord = Coordinates {
                            w: &mut w,
                            globals: late_globals,
                            id: &mut id,
                            at: coordinates,
                        }
                        .sampled(coords, sources.layer);
                        let sampler = load(&mut w, &mut id, sampler_type, sampler_var);
                        let combined = alu(
                            &mut w,
                            &mut id,
                            Op::SampledImage,
                            sampled,
                            &[image, sampler],
                        );
                        // One vector query replaces Ryujinx's per-component
                        // Lod operations (InstEmitTexture.cs:897-923; Spirv/
                        // Instructions.cs:859-898). This shares the query and
                        // retains reversed XY, *256, and truncation to S32.
                        let pair =
                            alu(&mut w, &mut id, Op::ImageQueryLod, vec2, &[combined, coord]);
                        let scale = fresh(&mut id);
                        inst(late_globals, Op::Constant, &[float, scale, 0x4380_0000]);
                        let mut written = 0;
                        for channel in 0..4 {
                            if mask & (1 << channel) != 0 {
                                // Z and W are not LOD components: Ryujinx
                                // InstEmitTexture.cs:907-913 writes zero.
                                let bits = match channel {
                                    0 | 1 => {
                                        let value = alu(
                                            &mut w,
                                            &mut id,
                                            Op::CompositeExtract,
                                            float,
                                            &[pair, channel ^ 1],
                                        );
                                        let value =
                                            alu(&mut w, &mut id, Op::FMul, float, &[value, scale]);
                                        let value = alu(
                                            &mut w,
                                            &mut id,
                                            Op::ConvertFToS,
                                            signed_int,
                                            &[value],
                                        );
                                        alu(&mut w, &mut id, Op::Bitcast, uint, &[value])
                                    }
                                    _ => constants[&0],
                                };
                                let bits = results.result(&mut w, &mut id, uint, bits);
                                put(
                                    &mut w,
                                    &mut id,
                                    reg(dst.0, written),
                                    bits,
                                    pred,
                                    uint,
                                    boolt,
                                );
                                written += 1;
                            }
                        }
                        continue;
                    }
                    TexOp::Query { query } => {
                        let mut channels = [constants[&1]; 4];
                        // Ryujinx InstEmitTexture.cs:1152-1182 selects samples
                        // into Z; Mesa spirv.h:1835 defines ImageQuerySamples.
                        if query == 2 && matches!(mask, 4..=7) && multisample {
                            channels[2] =
                                alu(&mut w, &mut id, Op::ImageQuerySamples, uint, &[image]);
                        } else {
                            // Ryujinx CodeGen/Spirv/Instructions.cs:1487-1521:
                            // cube queries have two axes; arrays append their layer
                            // count; channel 3 is levels. One size query serves every
                            // selected axis, instead of querying each axis separately.
                            let count = match dim {
                                0 => 1,
                                1 | 2 | 6 => 2,
                                3 | 4 | 7 => 3,
                                _ => return Err(bad(x)),
                            };
                            // Ryujinx CodeGen/Spirv/Instructions.cs:1493-1515
                            // queries multisample dimensions without LOD. Missing
                            // axes and multisample level counts remain unproven.
                            // A is unused for MS (InstEmitTexture.cs:1084-1089);
                            // require RZ instead of silently ignoring its field.
                            if query != 1
                                || (multisample && (mask & 8 != 0 || src0.0 != 255))
                                || mask == 0
                                || mask & 7 & !((1 << count) - 1) != 0
                            {
                                return Err(bad(x));
                            }
                            if mask & 7 != 0 {
                                let lod = if multisample {
                                    zero
                                } else {
                                    load(
                                        &mut w,
                                        &mut id,
                                        uint,
                                        reg(src0.0, u8::from(bound_handle.is_none())),
                                    )
                                };
                                let size = fresh(&mut id);
                                let size_type = match count {
                                    1 => uint,
                                    2 => uvec2,
                                    _ => uvec3,
                                };
                                if multisample {
                                    // Mesa src/compiler/spirv/spirv.h:1832:
                                    // OpImageQuerySize has no LOD operand.
                                    inst(&mut w, Op::ImageQuerySize, &[size_type, size, image]);
                                } else {
                                    inst(
                                        &mut w,
                                        Op::ImageQuerySizeLod,
                                        &[size_type, size, image, lod],
                                    );
                                }
                                for channel in 0..count {
                                    if mask & (1 << channel) != 0 {
                                        channels[channel as usize] = if count == 1 {
                                            size
                                        } else {
                                            alu(
                                                &mut w,
                                                &mut id,
                                                Op::CompositeExtract,
                                                uint,
                                                &[size, channel],
                                            )
                                        };
                                    }
                                }
                            }
                            if mask & 8 != 0 {
                                channels[3] =
                                    alu(&mut w, &mut id, Op::ImageQueryLevels, uint, &[image]);
                            }
                        }
                        let mut written = 0u8;
                        for (channel, value) in channels.into_iter().enumerate() {
                            if mask & (1 << channel) != 0 {
                                // Ryujinx InstEmitTexture.cs:1161-1178 skips
                                // selected X/Y destinations for sample count.
                                // Preserve their bits and still advance D.
                                if query == 2 && channel != 2 {
                                    written += 1;
                                    continue;
                                }
                                let value = results.result(&mut w, &mut id, uint, value);
                                put(
                                    &mut w,
                                    &mut id,
                                    reg(dst.0, written),
                                    value,
                                    pred,
                                    uint,
                                    boolt,
                                );
                                written += 1;
                            }
                        }
                        continue;
                    }
                };
                // Dref sampling returns one scalar for every selected
                // channel (Ryujinx Instructions.cs:1400-1402,1429-1436). Bitcast it
                // once instead of repeating an identical cast per channel.
                let depth_scalar = depth_compare && matches!(op, TexOp::Sample { .. });
                let depth_bits = if depth_scalar && !packed_f16 {
                    let bits = fresh(&mut id);
                    inst(&mut w, Op::Bitcast, &[uint, bits, result_vec4]);
                    Some(bits)
                } else {
                    None
                };
                if packed_f16 {
                    // Ryujinx InstEmitTexture.cs:609-655 collects selected
                    // channels in order, zero fills odd pairs, and packs each
                    // pair to one destination. Extract directly from the image
                    // result to avoid its temporary locals and copies. Dref
                    // samples reuse one float value for every selected channel.
                    let mut parts = [zero_float; 4];
                    let mut count = 0;
                    for channel in 0..4u32 {
                        if mask & (1 << channel) != 0 {
                            parts[count] = if depth_scalar {
                                result_vec4
                            } else {
                                alu(
                                    &mut w,
                                    &mut id,
                                    Op::CompositeExtract,
                                    float,
                                    &[result_vec4, channel],
                                )
                            };
                            count += 1;
                        }
                    }
                    for (pair, destination) in [dst, scalar_dst.ok_or_else(|| bad(x))?]
                        .into_iter()
                        .enumerate()
                    {
                        if destination.0 == 255 {
                            continue;
                        }
                        let vector = alu(
                            &mut w,
                            &mut id,
                            Op::CompositeConstruct,
                            vec2,
                            &parts[pair * 2..pair * 2 + 2],
                        );
                        // Ryujinx CodeGen/Spirv/Instructions.cs:1029-1036;
                        // GLSL.std.450 PackHalf2x16 (58), no Float16 capability.
                        let bits = alu(
                            &mut w,
                            &mut id,
                            Op::ExtInst,
                            uint,
                            &[glsl_std_450, 58, vector],
                        );
                        let bits = results.result(&mut w, &mut id, uint, bits);
                        put(
                            &mut w,
                            &mut id,
                            regs[destination.0 as usize],
                            bits,
                            pred,
                            uint,
                            boolt,
                        );
                    }
                    continue;
                }
                let mut written = 0u8;
                for channel in 0..4u32 {
                    if mask & (1 << channel) == 0 {
                        continue;
                    }
                    let u = if let Some(bits) = depth_bits {
                        bits
                    } else {
                        let f = fresh(&mut id);
                        inst(
                            &mut w,
                            Op::CompositeExtract,
                            &[float, f, result_vec4, channel],
                        );
                        let bits = fresh(&mut id);
                        inst(&mut w, Op::Bitcast, &[uint, bits, f]);
                        bits
                    };
                    let u = results.result(&mut w, &mut id, uint, u);
                    let target = if scalar && written >= 2 {
                        reg(scalar_dst.ok_or_else(|| bad(x))?.0, written - 2)
                    } else {
                        reg(dst.0, written)
                    };
                    // RZ discards the encoded result, but the texture op is
                    // still emitted. Ryujinx InstEmitTexture.cs:634-641.
                    put(&mut w, &mut id, target, u, pred, uint, boolt);
                    written += 1;
                }
            }
            InstructionKind::GlobalMemory(ops::GlobalMemory { .. })
            | InstructionKind::GlobalAtomic(ops::GlobalAtomic { .. }) => {
                if let Some(access) = storage_plan.get(&index) {
                    storage.emit(&mut w, &mut id, access, &regs, uint, boolt, pred);
                    continue;
                }
                global
                    .as_ref()
                    .expect("global memory types declared")
                    .emit(
                        &mut w,
                        late_globals,
                        &mut id,
                        x,
                        &regs,
                        uint,
                        uvec2,
                        signed_int,
                        boolt,
                        zero,
                        pred,
                    )?;
            }
            InstructionKind::LocalLoad(ops::LocalLoad {
                dst,
                address,
                offset,
                mem_type,
            }) => word_load(
                &mut w,
                &mut id,
                late_globals,
                local_words,
                dst,
                address,
                offset,
                mem_type,
                pred,
            ),
            InstructionKind::SharedLoad(ops::SharedLoad {
                dst,
                address,
                offset,
                mem_type,
            }) => word_load(
                &mut w,
                &mut id,
                late_globals,
                shared_words,
                dst,
                address,
                offset,
                mem_type,
                pred,
            ),
            InstructionKind::LocalStore(ops::LocalStore {
                data,
                address,
                offset,
                mem_type,
                cache: _,
            }) => word_store(
                &mut w,
                &mut id,
                late_globals,
                local_words,
                &local_merge,
                data,
                address,
                offset,
                mem_type,
            ),
            InstructionKind::SharedStore(ops::SharedStore {
                data,
                address,
                offset,
                mem_type,
            }) => word_store(
                &mut w,
                &mut id,
                late_globals,
                shared_words,
                &shared_merge,
                data,
                address,
                offset,
                mem_type,
            ),
            InstructionKind::Atomic(ops::Atomic {
                dst,
                address,
                offset,
                data,
                op,
                signed,
            }) => {
                // Shared memory is an array of uint words: the byte address
                // (register plus offset) shifted right by 2 is the index, as
                // for shared loads and stores.
                let addr = load(&mut w, &mut id, uint, regs[address.0 as usize]);
                let byte = fresh(&mut id);
                let offset_value = late_constant(late_globals, &mut id, uint, offset);
                inst(&mut w, Op::IAdd, &[uint, byte, addr, offset_value]);
                let word = fresh(&mut id);
                inst(
                    &mut w,
                    Op::ShiftRightLogical,
                    &[uint, word, byte, constants[&2]],
                );
                let ptr = fresh(&mut id);
                inst(
                    &mut w,
                    Op::AccessChain,
                    &[shared_elem_ptr, ptr, shared_var, word],
                );
                let value = load(&mut w, &mut id, uint, regs[data.0 as usize]);
                let semantics = late_constant(late_globals, &mut id, uint, 0x108);
                let result = match atomic::shared_form(op, signed) {
                    atomic::SharedForm::Direct(opcode) => {
                        let result = fresh(&mut id);
                        inst(
                            &mut w,
                            opcode,
                            &[uint, result, ptr, constants[&2], semantics, value],
                        );
                        result
                    }
                    atomic::SharedForm::Loop(how) => {
                        let swap = atomic::Swap {
                            uint,
                            boolt,
                            float,
                            pointer: ptr,
                            scope: constants[&2],
                            // A failed compare writes nothing: no Release.
                            acquire: late_constant(late_globals, &mut id, uint, 0x102),
                            equal: semantics,
                            value,
                            constants: atomic::Constants::new(|v| {
                                late_constant(late_globals, &mut id, uint, v)
                            }),
                        };
                        atomic::swap_loop(&mut w, &mut id, &swap, how)
                    }
                };
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    result,
                    pred,
                    uint,
                    boolt,
                );
            }
            InstructionKind::SharedCompareExchange(ops::SharedCompareExchange {
                dst,
                address,
                offset,
                compare,
                replacement,
            }) => {
                let addr = load(&mut w, &mut id, uint, regs[address.0 as usize]);
                let byte = fresh(&mut id);
                let offset_value = late_constant(late_globals, &mut id, uint, offset as u32);
                inst(&mut w, Op::IAdd, &[uint, byte, addr, offset_value]);
                let word = fresh(&mut id);
                inst(
                    &mut w,
                    Op::ShiftRightLogical,
                    &[uint, word, byte, constants[&2]],
                );
                let ptr = fresh(&mut id);
                inst(
                    &mut w,
                    Op::AccessChain,
                    &[shared_elem_ptr, ptr, shared_var, word],
                );
                let comparator = load(&mut w, &mut id, uint, regs[compare.0 as usize]);
                let replacement = load(&mut w, &mut id, uint, regs[replacement.0 as usize]);
                let result = fresh(&mut id);
                let scope = constants[&2];
                let equal_semantics = late_constant(late_globals, &mut id, uint, 0x108);
                // No write occurs on comparison failure, so SPIR-V forbids
                // Release or AcquireRelease semantics on that path.
                let unequal_semantics = late_constant(late_globals, &mut id, uint, 0x102);
                inst(
                    &mut w,
                    Op::AtomicCompareExchange,
                    &[
                        uint,
                        result,
                        ptr,
                        scope,
                        equal_semantics,
                        unequal_semantics,
                        replacement,
                        comparator,
                    ],
                );
                put(
                    &mut w,
                    &mut id,
                    regs[dst.0 as usize],
                    result,
                    pred,
                    uint,
                    boolt,
                );
            }
            // Ryujinx CodeGen/Spirv/Instructions.cs:584-587,964-967: the
            // group barrier uses SPIR-V Workgroup (2), the memory barrier
            // Device (1). Only compute (GLCompute, 5) has workgroups;
            // elsewhere a group barrier widens to Device and drops
            // WorkgroupMemory. Semantics: AcquireRelease | UniformMemory |
            // ImageMemory, plus WorkgroupMemory in compute.
            InstructionKind::GroupMemoryBarrier(ops::GroupMemoryBarrier) => {
                let (memory_scope, semantics) = S::GROUP_MEMORY_BARRIER;
                inst(
                    &mut w,
                    Op::MemoryBarrier,
                    &[
                        constants[&memory_scope],
                        late_constant(late_globals, &mut id, uint, semantics),
                    ],
                );
            }
            InstructionKind::MemoryBarrier(ops::MemoryBarrier) => {
                let semantics = S::MEMORY_BARRIER_SEMANTICS;
                inst(
                    &mut w,
                    Op::MemoryBarrier,
                    &[
                        constants[&1],
                        late_constant(late_globals, &mut id, uint, semantics),
                    ],
                );
            }
            InstructionKind::ControlBarrier(ops::ControlBarrier) => {
                let (scope, semantics) = S::control_barrier(x)?;
                if x.predicate.is_some() {
                    return Err(bad(x));
                }
                // Ryujinx Spirv/Instructions.cs:252-255 supplies compute's
                // Workgroup scopes and AcquireRelease | WorkgroupMemory.
                // TCS under GLSL450 implicitly orders Output through the
                // execution barrier (SPIR-V 1.6, OpControlBarrier). Vulkan
                // VUID-StandaloneSpirv-ExecutionModel-07320 forbids Workgroup
                // memory scope for TCS; use Invocation and no memory semantics.
                // Vulkan-Docs appendices/memorymodel.adoc:1286-1311, commit
                // e4e53e4b31e13eeaee1ad99fb940aa72b2ec1b14.
                // Scope values: Mesa spirv.core.grammar.json:16816-16828.
                inst(
                    &mut w,
                    Op::ControlBarrier,
                    &[
                        constants[&2],
                        constants[&scope],
                        late_constant(late_globals, &mut id, uint, semantics),
                    ],
                );
            }
            InstructionKind::SurfaceAtomic(ops::SurfaceAtomic {
                op,
                signed,
                target,
                coord,
                data,
                dst,
                byte_address,
                ..
            }) => {
                // Follow Ryujinx CodeGen/Spirv/Instructions.cs:607-684: image
                // variable -> texel pointer -> device-scope relaxed atomic.
                // Atomic opcodes: Mesa src/compiler/spirv/spirv.h:1946-1959.
                let k = storage_uses[&index];
                let count = coordinate_count(target).ok_or_else(|| bad(x))?;
                let mut parts = Vec::new();
                for i in 0..count {
                    let value = load(
                        &mut w,
                        &mut id,
                        uint,
                        regs[coord.0.saturating_add(i) as usize],
                    );
                    // InstEmitSurface.cs:253-257,532-536,693-698: signed byte X / 4.
                    let value = alu_when(
                        &mut w,
                        &mut id,
                        i == 0 && byte_address,
                        Op::ShiftRightArithmetic,
                        uint,
                        &[value, constants[&2]],
                        value,
                    );
                    parts.push(value);
                }
                let position = if count == 1 {
                    parts[0]
                } else {
                    let position = fresh(&mut id);
                    let mut args = vec![if count == 2 { uvec2 } else { uvec3 }, position];
                    args.extend(parts);
                    inst(&mut w, Op::CompositeConstruct, &args);
                    position
                };
                let component = [uint, signed_int][usize::from(signed)];
                let pointer = fresh(&mut id);
                inst(
                    &mut w,
                    Op::ImageTexelPointer,
                    &[
                        atomic_pointers[&signed],
                        pointer,
                        storage_bindings[k],
                        position,
                        zero,
                    ],
                );
                let opcode = atomic_opcode(op, signed).ok_or_else(|| bad(x))?;
                let result = fresh(&mut id);
                let mut args = vec![component, result, pointer, constants[&1], zero];
                if !matches!(op, 3 | 4) {
                    let bits = load(&mut w, &mut id, uint, regs[data.0 as usize]);
                    let value = alu_when(
                        &mut w,
                        &mut id,
                        signed,
                        Op::Bitcast,
                        component,
                        &[bits],
                        bits,
                    );
                    if op == 9 {
                        // Compare B, replacement B+1: InstEmitSurface.cs:265-270;
                        // Instructions.cs:680 supplies replacement before comparator.
                        let bits = load(
                            &mut w,
                            &mut id,
                            uint,
                            regs[data.0.saturating_add(1) as usize],
                        );
                        let replacement = alu_when(
                            &mut w,
                            &mut id,
                            signed,
                            Op::Bitcast,
                            component,
                            &[bits],
                            bits,
                        );
                        args.extend([zero, replacement, value]);
                    } else {
                        args.push(value);
                    }
                }
                inst(&mut w, opcode, &args);
                if let Some(dst) = dst.filter(|r| r.0 != 255) {
                    let bits = alu_when(
                        &mut w,
                        &mut id,
                        signed,
                        Op::Bitcast,
                        uint,
                        &[result],
                        result,
                    );
                    put(
                        &mut w,
                        &mut id,
                        regs[dst.0 as usize],
                        bits,
                        pred,
                        uint,
                        boolt,
                    );
                }
            }
            InstructionKind::Surface(ops::Surface {
                store,
                atomic,
                target,
                format,
                mask,
                coord,
                data,
                size,
                clamp,
                byte_address,
                cache,
                ..
            }) => {
                if atomic
                    || clamp != 0
                    || (cache > 1 && !(store && cache == 2))
                    || format > 1
                    || mask == 0
                    || (format == 0 && (size != 0 || byte_address))
                {
                    return Err(bad(x));
                }
                let k = storage_uses[&index];
                let image = load(&mut w, &mut id, storage_images[k], storage_bindings[k]);
                let count = coordinate_count(target).ok_or_else(|| bad(x))?;
                let mut parts = Vec::new();
                for i in 0..count {
                    let value = load(
                        &mut w,
                        &mut id,
                        uint,
                        regs[coord.0.saturating_add(i) as usize],
                    );
                    let value = if i == 0 && byte_address {
                        // Ryujinx InstEmitSurface.cs:404-408,654-658,753-764:
                        // arithmetic shift of signed X by log2(texel bytes).
                        // Shift the raw bits directly; a bitcast adds no value.
                        alu(
                            &mut w,
                            &mut id,
                            Op::ShiftRightArithmetic,
                            uint,
                            &[value, constants[&texel_bytes_log2(size)]],
                        )
                    } else {
                        value
                    };
                    parts.push(value);
                }
                let position = if count == 1 {
                    parts[0]
                } else {
                    let position = fresh(&mut id);
                    let mut args = vec![if count == 2 { uvec2 } else { uvec3 }, position];
                    args.extend(parts);
                    inst(&mut w, Op::CompositeConstruct, &args);
                    position
                };
                let component = if format == 0 {
                    float
                } else if matches!(size, 1 | 3) {
                    signed_int
                } else {
                    uint
                };
                let vector = if format == 0 {
                    vec4
                } else if matches!(size, 1 | 3) {
                    storage_signed_types.ok_or_else(|| bad(x))?.1
                } else {
                    uvec4
                };
                if store {
                    // Ryujinx InstEmitSurface.cs:637-668 collects consecutive
                    // Rb values for selected components. Instructions.cs:810-831
                    // writes that compact sequence, padded with zero. No image
                    // read/merge is needed, saving one memory operation.
                    let padding = if format == 0 {
                        zero_float
                    } else if component == uint {
                        zero
                    } else {
                        late_constant(late_globals, &mut id, signed_int, 0)
                    };
                    let mut parts = [padding; 4];
                    for i in 0..mask.count_ones() as u8 {
                        let bits = load(
                            &mut w,
                            &mut id,
                            uint,
                            regs[data.0.saturating_add(i) as usize],
                        );
                        parts[i as usize] = if component == uint {
                            bits
                        } else {
                            alu(&mut w, &mut id, Op::Bitcast, component, &[bits])
                        };
                    }
                    let value = fresh(&mut id);
                    inst(
                        &mut w,
                        Op::CompositeConstruct,
                        &[vector, value, parts[0], parts[1], parts[2], parts[3]],
                    );
                    if CachePolicy::of(store, cache) == CachePolicy::Streaming {
                        // Nontemporal image operand: Mesa spirv.h:435;
                        // grammar:11990-11994 requires 1.6.
                        inst(&mut w, Op::ImageWrite, &[image, position, value, 0x4000]);
                    } else {
                        inst(&mut w, Op::ImageWrite, &[image, position, value]);
                    }
                } else {
                    let value = fresh(&mut id);
                    inst(&mut w, Op::ImageRead, &[vector, value, image, position]);
                    let mut dest = data.0;
                    for i in 0..4u32 {
                        if mask & (1 << i) == 0 {
                            continue;
                        }
                        let part = alu(
                            &mut w,
                            &mut id,
                            Op::CompositeExtract,
                            component,
                            &[value, i],
                        );
                        let bits = alu_when(
                            &mut w,
                            &mut id,
                            component != uint,
                            Op::Bitcast,
                            uint,
                            &[part],
                            part,
                        );
                        // Typed R8/R16 image reads already perform the same
                        // sign/zero extension as Ryujinx InstEmitSurface.cs:445-458.
                        put(
                            &mut w,
                            &mut id,
                            regs[dest as usize],
                            bits,
                            pred,
                            uint,
                            boolt,
                        );
                        dest = dest.saturating_add(1);
                    }
                }
            }
            _ => return Err(bad(x)),
        }
        for merge in memory_merge.into_iter() {
            inst(&mut w, Op::Branch, &[merge]);
            inst(&mut w, Op::Label, &[merge]);
        }
    }
    if reachable {
        inst(&mut w, Op::Store, &[pc_var, constants[&EXIT_CASE]]);
        inst(&mut w, Op::Branch, &[switch_merge]);
    }
    let cases_end = w.len();
    inst(&mut w, Op::Label, &[default_block]);
    inst(&mut w, Op::Store, &[pc_var, constants[&EXIT_CASE]]);
    inst(&mut w, Op::Branch, &[switch_merge]);
    inst(&mut w, Op::Label, &[switch_merge]);
    inst(&mut w, Op::Branch, &[continue_block]);
    inst(&mut w, Op::Label, &[continue_block]);
    let state = load(&mut w, &mut id, uint, pc_var);
    let running = id;
    id += 1;
    inst(
        &mut w,
        Op::INotEqual,
        &[boolt, running, state, constants[&EXIT_CASE]],
    );
    inst(&mut w, Op::BranchConditional, &[running, header, loop_exit]);
    inst(&mut w, Op::Label, &[loop_exit]);
    if regions.is_enabled() {
        // Preserve each case's instruction body and replace only dispatcher
        // edges. Nested loop merges implement token reconvergence, following
        // Ryujinx CodeGen/Spirv/SpirvGenerator.cs:334-344,356-369.
        let addresses = labels
            .iter()
            .filter(|(address, _)| block_starts.contains(address))
            .map(|(address, label)| (*label, *address))
            .collect::<HashMap<_, _>>();
        let mut cases = HashMap::default();
        let mut at = cases_at;
        let mut begin = cases_at;
        let mut address = None;
        while at < cases_end {
            let count = (w[at] >> 16) as usize;
            if Op::of(w[at]) == Op::Label {
                if let Some(&next) = addresses.get(&w[at + 1]) {
                    for previous in address.into_iter() {
                        cases.insert(previous, w[begin..at].to_vec());
                    }
                    address = Some(next);
                    begin = at;
                }
            }
            at += count;
        }
        for address in address.into_iter() {
            cases.insert(address, w[begin..cases_end].to_vec());
        }
        w.truncate(dispatch_at);
        regions.emit(
            &mut w,
            &mut id,
            &cases,
            &crate::reconvergence::Dispatch {
                uint,
                boolt,
                pc: pc_var,
                old_merge: switch_merge,
                exit: constants[&EXIT_CASE],
                case_of,
                labels,
                constants,
            },
        );
    }
    outputs.store(
        &mut w,
        &mut id,
        OutputStores {
            uint,
            float,
            vec4,
            regs: &regs,
            load: &load,
        },
    );
    for io in stage_io.iter() {
        io.emit_passthrough(&mut w, &mut id, late_globals);
    }
    w.splice(function_at..function_at, late_globals.drain(..));
    w.splice(annotation_at..annotation_at, annotations.drain(..));
    inst(&mut w, Op::Return, &[]);
    inst(&mut w, Op::FunctionEnd, &[]);
    w[3] = id;
    [|_: &mut Vec<u32>| {}, upgrade_image_nontemporal_module][usize::from(required.nontemporal)](
        &mut w,
    );
    Ok(SpirvModule {
        words: w,
        surface_descriptors,
        requires_maximal_reconvergence,
        requires_demote_to_helper_invocation: uses_demote,
    })
}

/// Nontemporal images require SPIR-V 1.6 (Mesa
/// src/compiler/spirv/spirv.core.grammar.json:11990-11994). Starting with
/// 1.4, OpEntryPoint must list every global variable used by the entry point
/// (Mesa spirv_to_nir.c:8030-8033; vtn_variables.c:2816-2826).
/// This emitter has one entry point; listing all globals also covers late
/// declarations without a second use scan.
/// Hosts consuming these modules need Vulkan 1.3 / SPIR-V 1.6 support
/// (Khronos Vulkan-Docs appendices/spirvenv.adoc:31-34, commit
/// e4e53e4b31e13eeaee1ad99fb940aa72b2ec1b14).
fn upgrade_image_nontemporal_module(words: &mut Vec<u32>) {
    words[1] = spirv::VERSION_1_6;
    let mut globals = Vec::new();
    let mut entry = 0;
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        match Op::of(words[at]) {
            // OpVariable result2/storage3: Mesa spirv.core.grammar.json:679-687.
            Op::Variable if words[at + 3] != spirv::STORAGE_CLASS_FUNCTION => {
                globals.push(words[at + 2])
            }
            // OpEntryPoint: Mesa spirv.core.grammar.json:260-269.
            Op::EntryPoint => entry = at,
            _ => {}
        }
        at += count;
    }
    let count = (words[entry] >> 16) as usize;
    // The emitter's entry point name is the two-word string "main".
    let interfaces = &words[entry + 5..entry + count];
    globals.retain(|variable| !interfaces.contains(variable));
    words[entry] += (globals.len() as u32) << 16;
    words.splice(entry + count..entry + count, globals);
}

/// A value the host can compute at draw time: a constant or a word of a
/// bound constant buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HandleLeaf {
    Constant(u32),
    ConstantBuffer { bank: u8, offset: u16 },
}

/// Where a bindless texture handle comes from: a leaf, or two leaves
/// combined by a logic op (games OR a texture word with a sampler word;
/// AND, OR or XOR as LOP encodes it).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum HandleSource {
    /// Bound texture/surface table index (envytools gm107.c:788-791,
    /// 1960,1966-1969). The instruction determines its descriptor class.
    Bound(u16),
    Leaf(HandleLeaf),
    Combined {
        op: LogicOp,
        a: HandleLeaf,
        b: HandleLeaf,
    },
    /// A packed handle loaded through an indexed constant-buffer access.
    /// The runtime descriptor index is the byte address divided by eight.
    ConstantBufferArray {
        bank: u8,
        handle: u16,
    },
}

/// Require one source for all definitions reaching an executed handle use.
fn texture_handles(
    xs: &[Instruction],
    reaching: &ReachingDefinitions,
) -> Result<HashMap<usize, HandleSource>, Stopped> {
    let mut handles = HashMap::default();
    for (index, x) in xs.iter().enumerate() {
        let kind = x.kind.legacy_form();
        if let InstructionKind::Texture(ops::Texture {
            bound_handle: Some(binding),
            ..
        }) = kind
        {
            handles.insert(index, HandleSource::Bound(binding));
            continue;
        }
        if let InstructionKind::Surface(ops::Surface {
            handle: Source::Immediate(binding),
            ..
        })
        | InstructionKind::SurfaceAtomic(ops::SurfaceAtomic {
            handle: Source::Immediate(binding),
            ..
        }) = kind
        {
            // Bound handles are table indices, not bindless handle bits.
            // Ryujinx InstEmitSurface.cs:682-688; gm107.c:788-791.
            if binding > 0x1fff {
                return Err(bad(x));
            }
            handles.insert(index, HandleSource::Bound(binding as u16));
            continue;
        }
        let handle = match kind {
            InstructionKind::Texture(ops::Texture { op, src0, src1, .. }) => {
                Source::Register(if matches!(op, TexOp::Query { .. }) {
                    src0
                } else {
                    src1
                })
            }
            InstructionKind::Surface(ops::Surface { handle, .. })
            | InstructionKind::SurfaceAtomic(ops::SurfaceAtomic { handle, .. }) => handle,
            _ => continue,
        };
        let source = reaching.source(index, handle).ok_or_else(|| bad(x))?;
        if matches!(source, HandleSource::ConstantBufferArray { .. })
            && !matches!(kind, InstructionKind::Texture(_))
        {
            return Err(bad(x));
        }
        handles.insert(index, source);
    }
    Ok(handles)
}

fn texture_offset_components(dim: u8) -> Option<u8> {
    match dim {
        0 | 1 => Some(1),
        2 | 3 => Some(2),
        4 => Some(3),
        // Ryujinx InstEmitTexture.cs:230-235,272-276 extracts three
        // offset axes. Only cube arrays provide all three size-query axes
        // in TexturePass.cs:623-648 / Spirv/Instructions.cs:1496-1519.
        7 => Some(3),
        _ => None,
    }
}

#[cfg(test)]
mod texture_tests {
    use super::texture_offset_components;
    use crate::{decoder::decode_word, ir::Register};

    #[test]
    fn bound_tex_auxiliaries_begin_at_b0_and_bindless_at_b1() {
        for bound in [false, true] {
            for lod in [0, 1, 2, 3, 6, 7] {
                for offset in 0..2 {
                    for depth in 0..2 {
                        // Mesa sm50.rs:2171-2197 and older Mesa
                        // nv50_ir_emit_gm107.cpp:2904-2924.
                        let raw = [
                            (0xdeb8u64 << 48) | (lod << 37) | (offset << 36),
                            (0xc038u64 << 48) | (lod << 55) | (offset << 54),
                        ][usize::from(bound)]
                            | (depth << 50)
                            | (3 << 28)
                            | (1 << 31)
                            | (([10, 255]
                                [usize::from(bound && lod < 2 && offset == 0 && depth == 0)])
                                << 20)
                            | (8 << 8)
                            | (7 << 16);
                        let x = decode_word([raw as u32, (raw >> 32) as u32], 0);
                        let sources = super::texture::decoded_sources(x);
                        assert_eq!(sources.layer, Some(Register(8)));
                        assert_eq!(
                            sources.coords[..sources.coord_count],
                            [Register(9), Register(10)]
                        );
                        let mut b = [11, 10][usize::from(bound)];
                        assert_eq!(sources.lod, (lod >= 2).then_some(Register(b)));
                        b += u8::from(lod >= 2);
                        assert_eq!(sources.offset, (offset != 0).then_some(Register(b)));
                        b += offset as u8;
                        assert_eq!(sources.dref, (depth != 0).then_some(Register(b)));
                    }
                }
            }
        }
    }

    #[test]
    fn offsets_match_non_array_coordinate_dimensions() {
        assert_eq!(texture_offset_components(0), Some(1));
        assert_eq!(texture_offset_components(1), Some(1));
        assert_eq!(texture_offset_components(2), Some(2));
        assert_eq!(texture_offset_components(3), Some(2));
        assert_eq!(texture_offset_components(4), Some(3));
        assert_eq!(texture_offset_components(6), None);
        assert_eq!(texture_offset_components(7), Some(3));
    }
}

/// Ryujinx Instructions/InstEmitMultifunction.cs:40-75, and
/// Translation/EmitterContextInsts.cs:394-401: RCP is 1/x, RSQ is inverse sqrt.
/// CodeGen/Spirv/Instructions.cs:901-903,1039-1041 selects Log2/InverseSqrt.
/// Explicit zero/domain results avoid Vulkan 1.2's undefined math inputs.
/// GLSL.std.450 section 2: Cos 14, Sin 13, Exp2 29, Log2 30, Sqrt 31,
/// InverseSqrt 32. SPIR-V 1.6 sections 3.3.4 (ExtInst 12), 3.3.11
/// (Bitcast 124), 3.3.13 (FDiv 136), 3.3.14 (And 199, Or 197),
/// 3.3.15 (IEqual 170, FOrdLessThan 184, LogicalOr 166, Select 169).
/// Float absolute value and negation on raw bits: clear, then flip, the
/// sign bit (exactly IEEE abs and neg).
fn sign_modifiers(
    w: &mut Vec<u32>,
    id: &mut u32,
    value: u32,
    abs: bool,
    negate: bool,
    uint: u32,
    constants: &HashMap<u32, u32>,
) -> u32 {
    let v = SIGN_ABS[usize::from(abs)](w, id, value, uint, constants);
    SIGN_NEGATE[usize::from(negate)](w, id, v, uint, constants)
}
type SignBit = fn(&mut Vec<u32>, &mut u32, u32, u32, &HashMap<u32, u32>) -> u32;
/// By abs: clear the float sign bit.
const SIGN_ABS: [SignBit; 2] = [
    |_, _, v, _, _| v,
    |w, id, v, uint, constants| {
        let r = fresh(id);
        inst(w, Op::BitwiseAnd, &[uint, r, v, constants[&0x7fff_ffff]]);
        r
    },
];
/// By negate: flip the float sign bit.
const SIGN_NEGATE: [SignBit; 2] = [
    |_, _, v, _, _| v,
    |w, id, v, uint, constants| {
        let r = fresh(id);
        inst(w, Op::BitwiseXor, &[uint, r, v, constants[&0x8000_0000]]);
        r
    },
];

/// A uint constant declared among the late globals (before the function).
/// The uint word array an LDL/LDS or STL/STS addresses.
#[derive(Clone, Copy)]
struct WordArray {
    elem_ptr: u32,
    variable: u32,
}

/// Writes a subword into the word at `ptr`: given `clear` (the kept bits)
/// and `shifted` (the new bits), it merges them into the stored word.
type SubwordMerge<'a> = &'a dyn Fn(&mut Vec<u32>, &mut u32, u32, u32, u32);

fn late_constant(late: &mut Vec<u32>, id: &mut u32, uint: u32, value: u32) -> u32 {
    let c = fresh(id);
    inst(late, Op::Constant, &[uint, c, value]);
    c
}

/// A binary64 constant declared among the late globals. SPIR-V encodes the
/// literal low word before the high word (SPIR-V 1.6 section 3.3.7).
fn late_double(late: &mut Vec<u32>, id: &mut u32, double: u32, value: f64) -> u32 {
    let c = fresh(id);
    let bits = value.to_bits();
    inst(
        late,
        Op::Constant,
        &[double, c, bits as u32, (bits >> 32) as u32],
    );
    c
}

fn fresh(id: &mut u32) -> u32 {
    let v = *id;
    *id += 1;
    v
}

/// Fold the documented offset fields into a SPIR-V constant. Sampling and
/// fetch: signed 4 bits at strides of 4 (Ryujinx InstEmitTexture.cs:272-276,
/// 559-561). Gather: signed 6 bits at strides of 8 (same file:584-586,756-760).
/// Scalar offsets return OpConstant directly; a one-component
/// OpConstantComposite would be invalid SPIR-V.
fn constant_texture_offset(
    globals: &mut Vec<u32>,
    id: &mut u32,
    declared: &mut bool,
    (signed_int, ivec2, ivec3): (u32, u32, u32),
    packed: u32,
    (count, bits, stride): (u8, u32, u32),
) -> u32 {
    if !*declared {
        inst(globals, Op::TypeVector, &[ivec2, signed_int, 2]);
        inst(globals, Op::TypeVector, &[ivec3, signed_int, 3]);
        *declared = true;
    }
    let mut components = [0; 3];
    for i in 0..u32::from(count) {
        let value = ((packed << (32 - bits - stride * i)) as i32) >> (32 - bits);
        let component = fresh(id);
        inst(
            globals,
            Op::Constant,
            &[signed_int, component, value as u32],
        );
        components[i as usize] = component;
    }
    if count == 1 {
        return components[0];
    }
    let result = fresh(id);
    globals.push(((u32::from(count) + 3) << 16) | u32::from(Op::ConstantComposite));
    globals.extend_from_slice(&[if count == 2 { ivec2 } else { ivec3 }, result]);
    globals.extend_from_slice(&components[..usize::from(count)]);
    result
}

/// Gather offset: signed 6-bit x at 0 and y at 8, Ryujinx
/// InstEmitTexture.cs:584-586,756-760. Two shifts avoid an extended opcode.
fn unpack_offset(
    w: &mut Vec<u32>,
    id: &mut u32,
    packed: u32,
    uvec2: u32,
    uint: u32,
    constants: &HashMap<u32, u32>,
) -> u32 {
    let mut parts = [0; 2];
    for (i, part) in parts.iter_mut().enumerate() {
        let shifted = fresh(id);
        inst(
            w,
            Op::ShiftLeftLogical,
            &[uint, shifted, packed, constants[&(26 - 8 * i as u32)]],
        );
        let value = fresh(id);
        inst(
            w,
            Op::ShiftRightArithmetic,
            &[uint, value, shifted, constants[&26]],
        );
        *part = value;
    }
    let v = fresh(id);
    inst(w, Op::CompositeConstruct, &[uvec2, v, parts[0], parts[1]]);
    v
}

/// Decode the two PTP registers into four signed 2D gather offsets. Each
/// register contains two offset pairs in four signed six-bit byte lanes.
fn gather4_offsets(
    w: &mut Vec<u32>,
    id: &mut u32,
    first: u32,
    second: u32,
    uvec2: u32,
    uint: u32,
    constants: &HashMap<u32, u32>,
) -> [u32; 4] {
    let upper = |w: &mut Vec<u32>, id: &mut u32, packed| {
        alu(
            w,
            id,
            Op::ShiftRightLogical,
            uint,
            &[packed, constants[&16]],
        )
    };
    let first_upper = upper(w, id, first);
    let second_upper = upper(w, id, second);
    [
        unpack_offset(w, id, first, uvec2, uint, constants),
        unpack_offset(w, id, first_upper, uvec2, uint, constants),
        unpack_offset(w, id, second, uvec2, uint, constants),
        unpack_offset(w, id, second_upper, uvec2, uint, constants),
    ]
}

#[cfg(test)]
mod tests {
    use super::{attribute, emit, Attribute, Op};
    use crate::{decoder::decode_word, ir::ops, ir::InstructionKind, SPEC_ID_Y_DIRECTION};

    #[test]
    fn scalar_texture_register_layout_matches_ryujinx() {
        // Encoding modes: envytools gm107.c:932-955,1967. Independent
        // register-layout oracle: Ryujinx InstEmitTexture.cs:492-561.
        // Sources start at R8 and R20, destinations at R0 and R12; the
        // selector is RGBA and the bound table index is 9 (gm107.c:971,1967).
        // Register fields: gm107.c:203-206; PT encoding: gm107.c:409;
        // bound index bits 36..48: Mesa nv50_ir_emit_gm107.cpp:2958.
        for (mode, coords, layer, lod, offset, sample) in [
            (0, &[8][..], None, None, None, None),
            (1, &[8][..], None, Some(20), None, None),
            (2, &[8, 20][..], None, None, None, None),
            (4, &[8, 9][..], None, None, Some(20), None),
            (5, &[8, 9][..], None, Some(20), None, None),
            (6, &[8, 9][..], None, None, None, Some(20)),
            (7, &[8, 9, 20][..], None, None, None, None),
            (8, &[20, 21][..], Some(8), None, None, None),
            (12, &[8, 9][..], None, Some(20), Some(21), None),
        ] {
            let raw = 0xda00u64 << 48
                | mode << 53
                | 4 << 50
                | 9 << 36
                | 12 << 28
                | ([20, 255][usize::from(mode == 0)]) << 20
                | 7 << 16
                | 8 << 8;
            let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
            let sources = super::texture::decoded_sources(instruction);
            assert_eq!(
                sources.coords[..sources.coord_count]
                    .iter()
                    .map(|r| r.0)
                    .collect::<Vec<_>>(),
                coords,
                "mode {mode}"
            );
            assert_eq!(sources.layer.map(|r| r.0), layer, "mode {mode}");
            assert_eq!(sources.lod.map(|r| r.0), lod, "mode {mode}");
            assert_eq!(sources.offset.map(|r| r.0), offset, "mode {mode}");
            assert_eq!(sources.sample.map(|r| r.0), sample, "mode {mode}");
            assert_eq!(sources.dref, None);
        }
        // Gather source order: Ryujinx InstEmitTexture.cs:568-593;
        // modifier fields: envytools gm107.c:1960. With both modifiers,
        // offsets consume B before comparison, so Dref is R21, not R20.
        for (offset, dc, coords, dref) in [
            (false, false, &[8, 20][..], None),
            (true, false, &[8, 9][..], None),
            (false, true, &[8, 9][..], Some(20)),
            (true, true, &[8, 9][..], Some(21)),
        ] {
            let raw = 0xdf02u64 << 48
                | u64::from(offset) << 51
                | u64::from(dc) << 50
                | 9 << 36
                | 12 << 28
                | 20 << 20
                | 7 << 16
                | 8 << 8;
            let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
            let sources = super::texture::decoded_sources(instruction);
            assert_eq!(
                sources.coords[..sources.coord_count]
                    .iter()
                    .map(|r| r.0)
                    .collect::<Vec<_>>(),
                coords
            );
            assert_eq!(sources.offset.map(|r| r.0), offset.then_some(20));
            assert_eq!(sources.dref.map(|r| r.0), dref);
            assert_eq!(sources.layer, None);
            assert_eq!(sources.lod, None);
            assert_eq!(sources.sample, None);
        }
    }

    #[test]
    fn discard_destinations_do_not_poison_zero_register_operands() {
        use crate::ir::{HalfSelect, PartialSumShift, Register, Source};
        fn at(
            address: u32,
            predicate: Option<crate::ir::Predicate>,
            kind: InstructionKind,
        ) -> crate::ir::Instruction {
            crate::ir::Instruction {
                address,
                predicate,
                kind,
            }
        }
        let xs = [
            at(
                0,
                None,
                InstructionKind::Mov(ops::Mov {
                    quad_lanes: 0xf,
                    dst: Register(255),
                    src: Source::Immediate(u32::MAX),
                }),
            ),
            at(
                8,
                None,
                InstructionKind::IAdd3(ops::IAdd3 {
                    dst: Register(1),
                    lhs: Register(255),
                    rhs: Source::Immediate(17),
                    addend: Register(255),
                    halves: [HalfSelect::Full; 3],
                    shift: PartialSumShift::Unshifted,
                    negate: [false; 3],
                }),
            ),
            at(16, None, InstructionKind::Exit(ops::Exit { cc_test: 0xf })),
        ];
        // This check inspects the original arithmetic operands. The optimiser
        // is allowed to fold the additions; observable equivalence is checked
        // separately by optimizer/tests.rs and the Vulkan comparisons.
        let module = super::emit_unoptimized_with_header(&xs, None).unwrap();
        let mut uint_type = 0;
        let mut constants = crate::fast_hash::HashMap::default();
        let mut adds = Vec::new();
        let mut offset = 5;
        while offset < module.words.len() {
            let count = (module.words[offset] >> 16) as usize;
            let op = Op::of(module.words[offset]);
            let args = &module.words[offset + 1..offset + count];
            match op {
                Op::TypeInt if args[1..] == [32, 0] => uint_type = args[0],
                Op::Constant if args[0] == uint_type => {
                    constants.insert(args[2], args[1]);
                }
                Op::IAdd => adds.push(args.to_vec()),
                _ => {}
            }
            offset += count;
        }
        let zero = constants[&0];
        assert_eq!(adds.len(), 2);
        assert_eq!(adds[0][2..], [zero, constants[&17]]);
        assert_eq!(adds[1][2..], [adds[0][1], zero]);
    }

    #[test]
    fn mesa_attribute_addresses_decode_components() {
        for (addr, location, component) in [
            (0x80, 0, 0),
            (0x84, 0, 1),
            (0x88, 0, 2),
            (0x8c, 0, 3),
            (0x90, 1, 0),
            (0x94, 1, 1),
            (0x98, 1, 2),
            (0x9c, 1, 3),
        ] {
            assert!(matches!(
                attribute(addr),
                Some(Attribute::Generic { location: l, component: c }) if l == location && c == component
            ));
        }
        for (addr, component) in [(0x70, 0), (0x74, 1), (0x78, 2), (0x7c, 3)] {
            assert!(matches!(
                attribute(addr),
                Some(Attribute::FragCoord { component: c }) if c == component
            ));
        }
    }

    #[test]
    fn y_direction_is_a_specialization_constant() {
        // PT keeps the specialization value observable in R1. Decoder
        // predicate fields and PT are documented at decoder.rs:140-149.
        let instruction = decode_word([1 | (7 << 16) | (0x12 << 20), 0xf0c8 << 16], 0);
        assert!(matches!(
            instruction.kind,
            InstructionKind::S2R(ops::S2R { idx: 0x12, .. })
        ));
        let words = emit(&[instruction]).unwrap().words;
        let mut has_spec_constant = false;
        let mut has_spec_id = false;
        let mut at = 5;
        while at < words.len() {
            let count = (words[at] >> 16) as usize;
            let op = Op::of(words[at]);
            if op == Op::SpecConstant {
                has_spec_constant = true;
            }
            if op == Op::Decorate && words[at + 2] == 1 && words[at + 3] == SPEC_ID_Y_DIRECTION {
                has_spec_id = true;
            }
            at += count;
        }
        assert!(has_spec_constant);
        assert!(has_spec_id);
        let path =
            std::env::temp_dir().join(format!("shadowbox-y-direction-{}.spv", std::process::id()));
        std::fs::write(
            &path,
            words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let validation = std::process::Command::new("spirv-val")
            .args(["--target-env", "vulkan1.2"])
            .arg(&path)
            .output()
            .unwrap();
        std::fs::remove_file(path).unwrap();
        assert!(
            validation.status.success(),
            "{}",
            String::from_utf8_lossy(&validation.stderr)
        );
    }
}

#[cfg(test)]
mod compare_tests {
    use super::{int_compare_op, ordered_compare_op, unordered_compare_op, Op};
    use crate::ir::{Relation, Signedness};

    #[test]
    fn compare_opcodes_match_spirv() {
        assert_eq!(
            int_compare_op(Relation::Lt, Signedness::Unsigned),
            Op::ULessThan
        );
        assert_eq!(
            int_compare_op(Relation::Lt, Signedness::Signed),
            Op::SLessThan
        );
        assert_eq!(
            int_compare_op(Relation::Gt, Signedness::Unsigned),
            Op::UGreaterThan
        );
        assert_eq!(
            int_compare_op(Relation::Ge, Signedness::Signed),
            Op::SGreaterThanEqual
        );
        assert_eq!(ordered_compare_op(Relation::Lt), Op::FOrdLessThan);
        assert_eq!(unordered_compare_op(Relation::Ne), Op::FUnordNotEqual);
        assert_eq!(ordered_compare_op(Relation::Ge), Op::FOrdGreaterThanEqual);
    }
}

#[cfg(test)]
mod guard_tests {
    use crate::ir::{ops, Instruction, InstructionKind, Predicate, Register, Source};
    use crate::reaching::ReachingDefinitions;
    use crate::structure::ControlFlowGraph;

    fn at(address: u32, predicate: Option<(u8, bool)>, kind: InstructionKind) -> Instruction {
        Instruction {
            address,
            predicate: predicate.map(|(register, inverted)| Predicate { register, inverted }),
            kind,
        }
    }

    fn mov(dst: u8, src: Source) -> InstructionKind {
        InstructionKind::Mov(ops::Mov {
            quad_lanes: 0xf,
            dst: Register(dst),
            src,
        })
    }

    // Keep main's guard scenarios on the new analysis, rather than
    // retaining the deleted block-local tracer. Ryujinx Translation/
    // Optimizations/Utils.cs:115-150 matches predicate definitions.
    fn guard_holds(xs: &[Instruction], writer: usize, user: usize) -> bool {
        let InstructionKind::Mov(ops::Mov { dst, .. }) = xs[writer].kind else {
            unreachable!()
        };
        let mut program = vec![at(0, None, mov(2, Source::Immediate(17)))];
        program.extend_from_slice(xs);
        let use_index = program.len();
        program.push(Instruction {
            address: 0,
            predicate: xs[user].predicate,
            kind: InstructionKind::Surface(ops::Surface {
                store: true,
                atomic: false,
                target: 0,
                format: 0,
                mask: 1,
                handle: Source::Register(dst),
                coord: Register(8),
                data: Register(9),
                size: 0,
                byte_address: false,
                cache: 0,
                clamp: 0,
            }),
        });
        program.push(at(
            0,
            None,
            InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
        ));
        for (index, instruction) in program.iter_mut().enumerate() {
            instruction.address = index as u32 * 8;
        }
        let cfg = ControlFlowGraph::build(&program).unwrap();
        ReachingDefinitions::analyze(&program, &cfg)
            .source(use_index, Source::Register(dst))
            .is_some()
    }

    #[test]
    fn same_guard_definition_is_accepted() {
        let xs = [
            at(0, Some((0, false)), mov(1, Source::Register(Register(2)))),
            at(8, None, mov(3, Source::Immediate(5))),
            at(16, Some((0, false)), mov(4, Source::Register(Register(1)))),
        ];
        assert!(guard_holds(&xs, 0, 2));
        assert!(guard_holds(&xs, 1, 2));
    }

    #[test]
    fn different_or_rewritten_guard_is_rejected() {
        let user = |p| at(16, p, mov(4, Source::Register(Register(1))));
        let writer = at(0, Some((0, false)), mov(1, Source::Register(Register(2))));
        let plain = at(8, None, mov(3, Source::Immediate(5)));
        // The user runs unguarded or on the opposite predicate value.
        assert!(!guard_holds(&[writer, plain, user(None)], 0, 2));
        assert!(!guard_holds(&[writer, plain, user(Some((0, true)))], 0, 2));
        assert!(!guard_holds(&[writer, plain, user(Some((1, false)))], 0, 2));
        // Something in between may rewrite the predicate.
        let unknown = at(
            8,
            None,
            InstructionKind::Unknown(ops::Unknown { raw: [0, 0] }),
        );
        assert!(!guard_holds(
            &[writer, unknown, user(Some((0, false)))],
            0,
            2
        ));
    }
}

// Integer conversion sizes/signs: Ryujinx Instructions/InstEmitAluHelper.cs:
// 36-60. The size encoding is envytools envydis/gm107.c:1208-1225.
fn integer_bounds(size: u8, signed: bool) -> (u32, u32) {
    INTEGER_BOUNDS[usize::from(signed)](8_u32 << size)
}
/// By signed: the range of a `bits`-wide integer as 32-bit patterns.
const INTEGER_BOUNDS: [fn(u32) -> (u32, u32); 2] = [
    |bits| (0, u32::MAX >> (32 - bits)),
    |bits| {
        let high = u32::MAX >> (33 - bits);
        (!high, high)
    },
];

fn conversion_extract(
    w: &mut Vec<u32>,
    id: &mut u32,
    a: u32,
    uint: u32,
    constants: &HashMap<u32, u32>,
    (size, byte, signed): (u8, u8, bool),
) -> u32 {
    // Byte and half sources extract their field; a word is itself.
    CONVERSION_SOURCE[usize::from(size)](w, id, a, uint, constants, (size, byte, signed))
}
type ConversionSource =
    fn(&mut Vec<u32>, &mut u32, u32, u32, &HashMap<u32, u32>, (u8, u8, bool)) -> u32;
/// By source size (byte, half, word).
const CONVERSION_SOURCE: [ConversionSource; 3] =
    [conversion_field, conversion_field, |_, _, a, _, _, _| a];
fn conversion_field(
    w: &mut Vec<u32>,
    id: &mut u32,
    a: u32,
    uint: u32,
    constants: &HashMap<u32, u32>,
    (size, byte, signed): (u8, u8, bool),
) -> u32 {
    let v = fresh(id);
    // SPIR-V 1.6 section 3.3.14 OpBitField[S/U]Extract:
    // Base and Result Type match. Signedness comes from the opcode.
    inst(
        w,
        EXTRACT_OP[usize::from(signed)],
        &[
            uint,
            v,
            a,
            constants[&(u32::from(byte) * 8)],
            constants[&(8 << size)],
        ],
    );
    v
}

fn conversion_abs_neg(
    w: &mut Vec<u32>,
    id: &mut u32,
    a: u32,
    uint: u32,
    glsl: u32,
    (abs, negate): (bool, bool),
) -> u32 {
    // Ryujinx Translation/EmitterContextInsts.cs:466-483 uses signed
    // integer abs then neg. GLSL.std.450 section 2 SAbs5 and SPIR-V 1.6
    // section 3.3.13 SNegate126 interpret integer bits as signed even with
    // our uint register representation. No additional signed type is needed.
    let a = INTEGER_ABS[usize::from(abs)](w, id, a, uint, glsl);
    INTEGER_NEGATE[usize::from(negate)](w, id, a, uint, glsl)
}
type IntegerModifier = fn(&mut Vec<u32>, &mut u32, u32, u32, u32) -> u32;
/// By abs: GLSL.std.450 SAbs.
const INTEGER_ABS: [IntegerModifier; 2] = [
    |_, _, a, _, _| a,
    |w, id, a, uint, glsl| {
        let v = fresh(id);
        inst(w, Op::ExtInst, &[uint, v, glsl, 5, a]);
        v
    },
];
/// By negate: SNegate.
const INTEGER_NEGATE: [IntegerModifier; 2] = [
    |_, _, a, _, _| a,
    |w, id, a, uint, _| {
        let v = fresh(id);
        inst(w, Op::SNegate, &[uint, v, a]);
        v
    },
];

fn integer_float_to_half(
    w: &mut Vec<u32>,
    globals: &mut Vec<u32>,
    id: &mut u32,
    f: u32,
    (uint, boolt): (u32, u32),
    constants: &HashMap<u32, u32>,
) -> u32 {
    // Mesa src/util/half_float.c:38-44,60-62,96-131 specifies nearest-even,
    // exponent biases 127/15, 13 discarded mantissa bits and half packing.
    // Integers converted to float are zero or have magnitude >=1. This
    // closed form needs no subnormal/NaN path, no float16 type/capability,
    // and no branch. It also avoids runtime PackHalf2x16 rounding differing
    // from constant folding, observed on RADV.
    // Add 4095 + retained LSB before dropping 13 bits to round ties to even.
    let mask = late_constant(globals, id, uint, 0x7fff_ffff);
    let bias = late_constant(globals, id, uint, 4095);
    let exp_bias = late_constant(globals, id, uint, (127 - 15) << 10);
    let infinity = late_constant(globals, id, uint, 0x7c00);
    let sign_mask = late_constant(globals, id, uint, 0x8000);
    let raw = fresh(id);
    inst(w, Op::Bitcast, &[uint, raw, f]);
    let magnitude = fresh(id);
    inst(w, Op::BitwiseAnd, &[uint, magnitude, raw, mask]);
    let retained = fresh(id);
    inst(
        w,
        Op::ShiftRightLogical,
        &[uint, retained, magnitude, constants[&13]],
    );
    let lsb = fresh(id);
    inst(w, Op::BitwiseAnd, &[uint, lsb, retained, constants[&1]]);
    let round_bias = fresh(id);
    inst(w, Op::IAdd, &[uint, round_bias, bias, lsb]);
    let rounded = fresh(id);
    inst(w, Op::IAdd, &[uint, rounded, magnitude, round_bias]);
    let shifted = fresh(id);
    inst(
        w,
        Op::ShiftRightLogical,
        &[uint, shifted, rounded, constants[&13]],
    );
    let half = fresh(id);
    inst(w, Op::ISub, &[uint, half, shifted, exp_bias]);
    let overflow = fresh(id);
    inst(w, Op::UGreaterThanEqual, &[boolt, overflow, half, infinity]);
    let saturated = fresh(id);
    inst(w, Op::Select, &[uint, saturated, overflow, infinity, half]);
    let is_zero = fresh(id);
    inst(w, Op::IEqual, &[boolt, is_zero, magnitude, constants[&0]]);
    let value = fresh(id);
    inst(
        w,
        Op::Select,
        &[uint, value, is_zero, constants[&0], saturated],
    );
    let shifted_sign = fresh(id);
    inst(
        w,
        Op::ShiftRightLogical,
        &[uint, shifted_sign, raw, constants[&16]],
    );
    let sign = fresh(id);
    inst(w, Op::BitwiseAnd, &[uint, sign, shifted_sign, sign_mask]);
    let result = fresh(id);
    inst(w, Op::BitwiseOr, &[uint, result, sign, value]);
    result
}
