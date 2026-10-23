//! The instruction families of the incoming edge, one constructor per
//! encoding shape. Each family's form table, built before runtime from the
//! encoding rules, sends a word to its constructor or to the next family
//! sharing its opcode.

use super::edge::{forms, unknown, Decode, Field, Fields, Word};
use super::*;
use crate::ir::{
    DoubleSetResult, FlagBank, IMadSpType, LegacyInterpolation, LegacyVarying, MulNegation, Named,
};

mod half;
use half::*;

/// Boolean combine field (envytools gm107.c:1430-1434, 1683-1687):
/// 0 AND, 1 OR, 2 XOR; 3 is OOPS and the form tables leave it Unknown.
const SET_OPS: [SetOp; 3] = [SetOp::And, SetOp::Or, SetOp::Xor];
/// XMAD c mode (Ryujinx Decoders/InstDecoders.cs:814-821, MIT; CSFU is
/// emitted by InstEmitIntegerArithmetic.cs:629-634). nvdisasm prints the
/// undefined register/immediate values 5..7 as ???5..???7; they stay Unknown.
const XMAD_COPS: [XmadCop; 5] = [
    XmadCop::Cfull,
    XmadCop::Clo,
    XmadCop::Chi,
    XmadCop::Csfu,
    XmadCop::Cbcc,
];
/// FMUL scale field 41..43; 7 stays Unknown.
const FMUL_SCALES: [FMulScale; 7] = [
    FMulScale::Unscaled,
    FMulScale::Divide2,
    FMulScale::Divide4,
    FMulScale::Divide8,
    FMulScale::Multiply8,
    FMulScale::Multiply4,
    FMulScale::Multiply2,
];
/// MUFU function field (Ryujinx Decoders/InstDecoders.cs:494-505).
const MUFU_OPS: [MufuOp; 9] = [
    MufuOp::Cos,
    MufuOp::Sin,
    MufuOp::Ex2,
    MufuOp::Lg2,
    MufuOp::Rcp,
    MufuOp::Rsq,
    MufuOp::Rcp64h,
    MufuOp::Rsq64h,
    MufuOp::Sqrt,
];
/// LDL/LDS/STL/STS/STG size field (Ryujinx Decoders/InstDecoders.cs:438-447);
/// 7 is unassigned (envytools gm107.c:583-591) and stays Unknown.
const LS_SIZES2: [LsSize2; 7] = [
    LsSize2::U8,
    LsSize2::S8,
    LsSize2::U16,
    LsSize2::S16,
    LsSize2::B32,
    LsSize2::B64,
    LsSize2::B128,
];
/// LDC size field. All and only the six architecture-defined widths become
/// IR states here; later stages never interpret this field again.
const LDC_WIDTHS: [LdcWidth; 6] = [
    LdcWidth::U8,
    LdcWidth::S8,
    LdcWidth::U16,
    LdcWidth::S16,
    LdcWidth::B32,
    LdcWidth::B64,
];
// Half mode fields: envytools envydis/gm107.c:1987,1977-1983,2155-2157;
// tables :1157-1168 (mul39..40/FMA37..38), :1083-1087 (FMA57..58),
// :1782-1786 (wide mul/FMA55..56). HADD2 has only FTZ39 or wide FTZ55.
// Mesa nak/ir.rs:3461-3463,3590-3592 distinguishes FTZ and DNZ (FMZ);
// nv50_ir_emit_gm107.cpp:474-476 encodes dnz << 1 | ftz. Mode 3 is OOPS.
const HALF_FLUSH: [HalfFlush; 3] = [HalfFlush::Preserve, HalfFlush::Ftz, HalfFlush::Fmz];

// BRX register + signed24: Mesa nv50_ir_emit_gm107.cpp:533-555;
// envytools gm107.c:1943,254 (offset),204,210,217 (register),
// 371,408-411 (guard),426-458 (CC). Bit5 selects cbuf, bit6
// LMT (gm107.c:1942-1943). Require CC.T and canonical unused bits;
// cbuf, LMT and other CC tests remain Unknown, never a guessed BRA.
// Ryujinx InstDecoders.cs:1114-1125; InstEmitFlowControl.cs:38-40.
fn brx_register(x: Word) -> Instruction {
    let r = x.r;
    let address = x.address;
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::BrxRegister(ops::BrxRegister {
        displacement: l,
        base: BrxBase::decode(address, b(r, 20, 24)),
        cc_test: b(r, 0, 5) as u8,
    }))
}
const BRX_REGISTER_FIELDS: Fields<2> = Fields([Field::bits(0, 8), Field::zero(44, 4)]);
const fn brx_register_rule(r: u64) -> u8 {
    (b(r, 0, 8) == 0x0f && b(r, 44, 4) == 0) as u8
}
const BRX_REGISTER_FORMS: [u8; BRX_REGISTER_FIELDS.len()] =
    forms!(BRX_REGISTER_FIELDS, brx_register_rule);
fn read_brx_register(x: Word) -> Instruction {
    // Qualified semantic slices from hybrid unit B8e0511d302035a9d: exact
    // register BRX cubes from H000007/P002526 and H000010/P002527, with
    // LMT fixed clear. No neighboring condition or control form enters.
    const QUALIFIED_CUBES: [(u64, u64); 4] = [
        (0xe250_0000_0007_0000, 0x000f_0000_0008_0000),
        (0xe250_5bb1_b767_1200, 0x000f_0000_0008_0000),
        (0xe250_0000_0007_0001, 0x000f_0000_0008_0000),
        (0xe250_5bb1_b767_1201, 0x000f_0000_0008_0000),
    ];
    if QUALIFIED_CUBES
        .iter()
        .any(|&(base, variable)| x.r & !variable == base)
    {
        return brx_register(x);
    }
    if x.r >> 48 != 0xe250 {
        return unknown(x);
    }
    let paths: [Decode; 2] = [unknown, brx_register];
    paths[(BRX_REGISTER_FORMS[BRX_REGISTER_FIELDS.index(x.r)]) as usize](x)
}
// SM50 FADD/FMUL register opcodes carry source/sign and saturate
// modifiers in the opcode field (sm50.rs:507-530, 680-700).
fn fadd_register(x: Word) -> Instruction {
    let r = x.r;
    x.guarded(fadd(r, src(r, false, true), fadd_operand(r)))
}
/// Ordinary FADD forms remain round-to-nearest. Directed forms enter only
/// through their exact source-locked packet boundaries below.
const FADD_FIELDS: Fields<1> = Fields([Field::zero(39, 2)]);
const fn fadd_rule(r: u64) -> u8 {
    (b(r, 39, 2) == 0) as u8
}
const FADD_FORMS: [u8; FADD_FIELDS.len()] = forms!(FADD_FIELDS, fadd_rule);
fn fadd_form(x: Word, path: Decode) -> Instruction {
    let paths: [Decode; 2] = [unknown, path];
    paths[FADD_FORMS[FADD_FIELDS.index(x.r)] as usize](x)
}
const FADD_FTZ_RM_PACKET_MASK: u64 = 0xfffc_1fff_fff7_ffff;
const FADD_FTZ_RM_PACKET_BASES: [u64; 11] = [
    0x5c58_1080_0000_0000,
    0x5c58_1080_0007_0000,
    0x5c58_10c1_e777_1d6e,
    0x5c58_1286_3767_da24,
    0x5c58_14bd_d8e7_972e,
    0x5c58_14f1_e8b7_f5ce,
    0x5c58_18d7_7b67_7d3d,
    0x5c58_1ad8_99f7_42aa,
    0x5c58_1ccf_af87_30a7,
    0x5c58_1e8d_1517_72c3,
    0x5c58_1e90_6e97_1225,
];
fn fadd_ftz_rm_is_source_locked(r: u64) -> bool {
    FADD_FTZ_RM_PACKET_BASES.contains(&(r & FADD_FTZ_RM_PACKET_MASK))
}
fn read_fadd_register(x: Word) -> Instruction {
    // Hybrid bundle B931c88442c4f9f4a, packets P003589..P003599. The
    // compact cubes vary only guard inversion, source modifiers, .CC and
    // opcode aliases; FTZ and round-toward-negative-infinity stay fixed.
    if fadd_ftz_rm_is_source_locked(x.r) {
        return fadd_register(x);
    }
    fadd_form(x, fadd_register)
}
fn read_fadd_constant(x: Word) -> Instruction {
    fadd_form(x, fadd_constant)
}
fn read_fadd_immediate(x: Word) -> Instruction {
    fadd_form(x, fadd_immediate)
}
// FCMP: envytools/envydis/gm107.c:2025,2044,2096,2143 and
// :1410-1426. A is 8..15, B is by form, C is 39..46 (B/C
// exchange for 0x53a0), compare 48..51, FTZ 47.
// Mesa nv50_ir_emit_gm107.cpp:1537-1571 confirms these fields.
// Ryujinx Instructions/InstEmitFloatComparison.cs:369-375 compares
// C with +0 and selects the original bits of A or B.
// 0x5bad is register FCMP.NEU, not another ALU family.
// envytools gm107.c:2025,1424. FTZ flushes only C before comparison; A/B remain raw selections.
// Mesa nv50_ir_emit_gm107.cpp:474-477 identifies bit47 as FTZ.
fn fcmp_of(x: Word, when_false: Source, test: Source) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::FCmp(ops::FCmp {
        dst: Register(b(r, 0, 8) as u8),
        when_true: Register(b(r, 8, 8) as u8),
        when_false,
        test,
        compare: fcmp(b(r, 48, 4)),
        ftz: b(r, 47, 1) != 0,
    }))
}
fn fcmp_register(x: Word) -> Instruction {
    fcmp_of(x, register_b(x.r), register_c(x.r))
}
fn fcmp_constant(x: Word) -> Instruction {
    fcmp_of(x, constant_b(x.r), register_c(x.r))
}
fn fcmp_immediate(x: Word) -> Instruction {
    fcmp_of(x, imm32(x.r), register_c(x.r))
}
/// 0x53a0 exchanges B and C: C is the selection, the constant the test.
fn fcmp_constant_c(x: Word) -> Instruction {
    fcmp_of(x, register_c(x.r), constant_b(x.r))
}
// ICMP encodings: envytools/envydis/gm107.c:2030,2046,2101,2148;
// compare 49..51 (1445-1453), signed 48 (1365-1367), A 8..15,
// B register 20..27 / cbuf / signed i20, C 39..46. 0x5340 swaps
// B/C (Mesa nv50_ir_emit_gm107.cpp:2028-2072). Register fields:
// gm107.c:203-220; cbuf offset/bank:258,302-309; signed i20:253.
// Destination 0..7 and guard 16..19:203,371,408-411.
// Ryujinx Instructions/InstEmitIntegerComparison.cs:13-54,117-131:
// compare C with integer zero and select A/B without conversion.
// 0x364d is the signed immediate GE form: signed48=1 and compare
// 49..51=6 (envytools/envydis/gm107.c:2148,1365-1368,1452).
fn icmp_of(x: Word, when_false: Source, test: Source) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::ICmp(ops::ICmp {
        dst: Register(b(r, 0, 8) as u8),
        when_true: Register(b(r, 8, 8) as u8),
        when_false,
        test,
        compare: cmp(b(r, 49, 3)),
        signedness: signedness(r),
    }))
}
fn icmp_register(x: Word) -> Instruction {
    icmp_of(x, register_b(x.r), register_c(x.r))
}
fn icmp_constant(x: Word) -> Instruction {
    icmp_of(x, constant_b(x.r), register_c(x.r))
}
fn icmp_immediate(x: Word) -> Instruction {
    icmp_of(x, signed_immediate20(x.r), register_c(x.r))
}
/// 0x5340 exchanges B and C: C is the selection, the constant the test.
fn icmp_constant_c(x: Word) -> Instruction {
    icmp_of(x, register_c(x.r), constant_b(x.r))
}
// VMAD: envytools envydis/gm107.c:1984,1090-1110,1600-1628.
// dst0..7, A8..15, C39..46 (:203-218), A byte36..37 / size38
// / sign48; B register20..27 / select28..30 / sign49, flag50
// or signed/unsigned imm16 at20..35. SHR51..52: 0/7/15, sat55,
// CC47 stays Unknown (Ryujinx InstEmitVideoArithmetic.cs:114).
// AvgMode53..54 is absent from envytools's table, but is decoded
// by Ryujinx Decoders/InstDecoders.cs:5182-5187 and implemented
// in InstEmitVideoArithmetic.cs:19,56-67. A selector 6 is the
// whole word (U32/S32 by sign48): nvdisasm prints VMAD.U32/.S32,
// and Ryujinx Extend passes it through (InstEmitAluHelper.cs:91,
// S32 signed result InstEmitVideoArithmetic.cs:70). Selector 7
// prints INVALIDSEL6 and stays Unknown.
/// Video immediate B: imm16 at 20..35, sign-extended when bit 49 is set.
fn video_immediate_b(r: u64) -> Source {
    Source::Immediate([b(r, 20, 16), signed(r, 20, 16) as u32][b(r, 49, 1) as usize])
}
fn video_selection(element: u32, signed: u32) -> VideoSelection {
    let element = match element {
        0 => VideoElement::Byte0,
        1 => VideoElement::Byte1,
        2 => VideoElement::Byte2,
        3 => VideoElement::Byte3,
        4 => VideoElement::Half0,
        5 => VideoElement::Half1,
        6 => VideoElement::Whole,
        _ => unreachable!("the VMAD form table rejects selector 7"),
    };
    let signedness = match signed {
        0 => Signedness::Unsigned,
        1 => Signedness::Signed,
        _ => unreachable!("signedness is one bit"),
    };
    VideoSelection {
        element,
        signedness,
    }
}
/// VMAD with B and its encoded selector, given by the B form.
fn video_multiply_add_of(x: Word, rhs: Source, select: VideoSelection) -> Instruction {
    let r = x.r;
    let mode = b(r, 53, 2) as u8;
    let a = video_selection(b(r, 36, 3), b(r, 48, 1));
    x.guarded(InstructionKind::VideoMultiplyAdd(ops::VideoMultiplyAdd {
        dst: Register(b(r, 0, 8) as u8),
        lhs: Register(b(r, 8, 8) as u8),
        rhs,
        addend: Register(b(r, 39, 8) as u8),
        select: [a, select],
        mode,
        shift: [0, 7, 15][b(r, 51, 2) as usize],
        saturate: b(r, 55, 1) != 0,
    }))
}
/// VMAD register B (flag 50 set): B 20..27, selector 28..30 with sign 49.
fn video_multiply_add_register(x: Word) -> Instruction {
    let r = x.r;
    video_multiply_add_of(x, register_b(r), video_selection(b(r, 28, 3), b(r, 49, 1)))
}
/// VMAD immediate B (flag 50 clear): selector 6 with the immediate's sign.
fn video_multiply_add_immediate(x: Word) -> Instruction {
    let r = x.r;
    video_multiply_add_of(x, video_immediate_b(r), video_selection(6, b(r, 49, 1)))
}
const VIDEO_MULTIPLY_ADD_FIELDS: Fields<5> = Fields([
    Field::bits(47, 1),
    Field::bits(36, 3),
    Field::bits(51, 2),
    Field::bits(50, 1),
    Field::bits(28, 3),
]);
const fn video_multiply_add_rule(r: u64) -> u8 {
    (b(r, 47, 1) == 0
        && b(r, 36, 3) <= 6
        && b(r, 51, 2) < 3
        && (b(r, 50, 1) == 0 || b(r, 28, 3) <= 6)) as u8
        * (1 + b(r, 50, 1) as u8)
}
const VIDEO_MULTIPLY_ADD_FORMS: [u8; VIDEO_MULTIPLY_ADD_FIELDS.len()] =
    forms!(VIDEO_MULTIPLY_ADD_FIELDS, video_multiply_add_rule);
fn read_video_multiply_add(x: Word) -> Instruction {
    let paths: [Decode; 3] = [
        unknown,
        video_multiply_add_immediate,
        video_multiply_add_register,
    ];
    paths[(VIDEO_MULTIPLY_ADD_FORMS[VIDEO_MULTIPLY_ADD_FIELDS.index(x.r)]) as usize](x)
}
// VMNMX: envytools/envydis/gm107.c:2109,1532-1560,1600-1628,
// :1105-1109. A 8..15, C 39..46; B 20..27 or imm16 20..35;
// A/B select 36..38/28..30, sign 48/49, B register flag 50,
// post-op 51..53, signed output 54, sat 55, max 56, CC 47.
// Ryujinx Decoders/InstDecoders.cs:5194-5208 also defines A's
// U32/S32 select; :717-732 excludes selector 7. CC is unmodeled.
/// VMNMX with B and its B selector, given by the B form.
fn video_min_max_of(x: Word, rhs: Source, select: u32) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::VideoMinMax(ops::VideoMinMax {
        dst: Register(b(r, 0, 8) as u8),
        lhs: Register(b(r, 8, 8) as u8),
        rhs,
        addend: Register(b(r, 39, 8) as u8),
        select: [
            (b(r, 36, 3) | (b(r, 48, 1) << 3)) as u8,
            (select | (b(r, 49, 1) << 3)) as u8,
        ],
        max: b(r, 56, 1) != 0,
        signed_output: b(r, 54, 1) != 0,
        saturate: b(r, 55, 1) != 0,
        op: b(r, 51, 3) as u8,
    }))
}
fn video_min_max_register(x: Word) -> Instruction {
    video_min_max_of(x, register_b(x.r), b(x.r, 28, 3))
}
/// Immediate B is already extended, so its selector is 6.
fn video_min_max_immediate(x: Word) -> Instruction {
    video_min_max_of(x, video_immediate_b(x.r), 6)
}
const VIDEO_MIN_MAX_FIELDS: Fields<4> = Fields([
    Field::bits(47, 1),
    Field::bits(36, 3),
    Field::bits(50, 1),
    Field::bits(28, 3),
]);
const fn video_min_max_rule(r: u64) -> u8 {
    (b(r, 47, 1) == 0 && b(r, 36, 3) != 7 && (b(r, 50, 1) == 0 || b(r, 28, 3) != 7)) as u8
        * (1 + b(r, 50, 1) as u8)
}
const VIDEO_MIN_MAX_FORMS: [u8; VIDEO_MIN_MAX_FIELDS.len()] =
    forms!(VIDEO_MIN_MAX_FIELDS, video_min_max_rule);
fn read_video_min_max(x: Word) -> Instruction {
    let paths: [Decode; 3] = [unknown, video_min_max_immediate, video_min_max_register];
    paths[(VIDEO_MIN_MAX_FORMS[VIDEO_MIN_MAX_FIELDS.index(x.r)]) as usize](x)
}
// FMUL (Ryujinx InstTable.cs:113-115, MIT: 0x5c68-0x5c6f,
// 0x4c68-0x4c6f, 0x3868-0x386f, 0x3968-0x396f; InstDecoders.cs
// InstFmulR/I/C): scale 41..43, round 39..40, Fmz 44..45 (1 FTZ,
// 2 FMZ, 3 invalid), CC 47, B negate 48, sat 50. nvdisasm ignores
// bits 46 and 49. Scale 7 and Fmz 3 are invalid. .CC47 writes Z/N from
// the final result and keeps C/V (envytools gm107.c:2004,2078,2124;
// Ryujinx InstEmitFloatArithmetic.cs:435-480 and InstEmitAluHelper.cs:
// 124-138). Ryujinx scales A before the multiply.
fn fmul(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);
    let scale = FMUL_SCALES[b(r, 41, 3) as usize];
    x.guarded(InstructionKind::FMul(ops::FMul {
        dst: d,
        lhs: l,
        rhs: float_form_src(r),
        modifiers: FMulModifiers::Operand {
            rhs_negation: source_negation(r, 48),
            saturation: saturation(r, 50),
            flush: flush(r, 44),
            rounding: fmul_rounding(r),
            scale,
        },
        flags: float_flags(r, 47),
    }))
}
const FMUL_FIELDS: Fields<2> = Fields([Field::bits(44, 2), Field::bits(41, 3)]);
const fn fmul_rule(r: u64) -> u8 {
    (b(r, 44, 2) != 3 && b(r, 41, 3) < 7) as u8
}
const FMUL_FORMS: [u8; FMUL_FIELDS.len()] = forms!(FMUL_FIELDS, fmul_rule);
fn read_fmul(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, fmul];
    paths[(FMUL_FORMS[FMUL_FIELDS.index(x.r)]) as usize](x)
}
// Mesa sm50.rs:1931-1951; envytools gm107.c:1999,2073.
// Destination0..7, register20..27 or cbuf20..38, enables39..42.
fn mov_register(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);

    x.guarded(InstructionKind::Mov(ops::Mov {
        quad_lanes: b(r, 39, 4) as u8,
        dst: d,
        src: src(r, false, true),
    }))
}

/// 0x4c98: B is the constant buffer.
fn mov_constant(x: Word) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::Mov(ops::Mov {
        quad_lanes: b(r, 39, 4) as u8,
        dst: Register(b(r, 0, 8) as u8),
        src: constant_b(r),
    }))
}
// Compact MOV.I20: envytools gm107.c:2119 (MIT) fixes destination0..7,
// immediate low19 at20..38, immediate bit19 at56, and quad mask39..42.
// Ryubing InstDecoders.cs:3820-3829 and InstEmitMove.cs:19-24 (MIT)
// independently establish that the immediate is a zero-extended 20-bit
// payload. The decoder constructs that concrete source once.
fn mov_immediate20(x: Word) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::Mov(ops::Mov {
        quad_lanes: b(r, 39, 4) as u8,
        dst: Register(b(r, 0, 8) as u8),
        src: Source::Immediate(b(r, 20, 19) | (b(r, 56, 1) << 19)),
    }))
}
// LDC (Ryujinx InstTable.cs, MIT: 0xef90-0xef97; InstDecoders.cs
// InstLdc): destination 0..7, register 8..15, signed 16-bit byte
// offset 20..35, bank 36..40, address mode 44..45 (0 default, 1 IL,
// 2 IS, 3 ISL), size 48..50. Sizes 6 and 7 are invalid. nvdisasm ignores
// bits 41..43, 46..47. ISL has its own concrete state, but only the qualified
// exact words below are admitted.
const LDC_MODES: [LdcAddressMode; 4] = [
    LdcAddressMode::Default,
    LdcAddressMode::Il,
    LdcAddressMode::IndexedSegmented,
    LdcAddressMode::IndexedSegmentedLocal,
];
fn ldc(x: Word) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::Ldc(ops::Ldc {
        dst: Register(b(r, 0, 8) as u8),
        binding: b(r, 36, 5) as u8,
        offset: Register(b(r, 8, 8) as u8),
        immediate: ((b(r, 20, 16) << 16) as i32) >> 16,
        mem_type: LDC_WIDTHS[b(r, 48, 3) as usize],
        address_mode: LDC_MODES[b(r, 44, 2) as usize],
    }))
}
const LDC_FIELDS: Fields<2> = Fields([Field::bits(44, 2), Field::bits(48, 3)]);
const fn ldc_rule(r: u64) -> u8 {
    (b(r, 44, 2) != 3 && b(r, 48, 3) < 6) as u8
}
const LDC_FORMS: [u8; LDC_FIELDS.len()] = forms!(LDC_FIELDS, ldc_rule);
fn read_ldc(x: Word) -> Instruction {
    if matches!(
        x.r,
        0xef90_3000_0007_0000
            | 0xef91_3000_0007_0000
            | 0xef92_3000_0007_0000
            | 0xef93_3000_0007_0000
            | 0xef94_3000_0007_0000
            | 0xef95_3000_0007_0000
            | 0xef90_7b7b_c947_9d94
            | 0xef90_b3f4_86f7_3aae
            | 0xef91_7b7b_c947_9d94
            | 0xef94_7b7b_c947_9d94
            | 0xef95_b3f4_86f7_3aae
    ) {
        return ldc(x);
    }
    let paths: [Decode; 2] = [unknown, ldc];
    paths[LDC_FORMS[LDC_FIELDS.index(x.r)] as usize](x)
}
// Mesa sm50.rs:1938-1941,1951; envytools gm107.c:2170.
// MOV32I has literal20..51 and quad enables12..15.
fn mov32i(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);

    x.guarded(InstructionKind::Mov(ops::Mov {
        quad_lanes: b(r, 12, 4) as u8,
        dst: d,
        src: src(r, true, false),
    }))
}
// PRMT (sm50.rs OpPrmt encode; envytools gm107.c:2023,2141): selector
// by form at 20.., srcs[0] 8..16, srcs[1] 39..47, mode 48..51; only
// index mode (0) is modeled.
fn prmt(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::Prmt(ops::Prmt {
        dst: d,
        lhs: l,
        rhs: Register(b(r, 39, 8) as u8),
        selector: int_form_src(r),
    }))
}
// IADD register/cbuf/immediate: envytools envydis/gm107.c:2015,
// 2089,2135, masks fff8/fff8/fef8. Neg A/B 49/48, SAT 50,
// .X 43 and .CC 47; Mesa nak/sm50.rs:1265-1286.
// Ryujinx InstEmitIntegerArithmetic.cs:326-338 implements negation
// and carry but leaves SAT as TODO. Reject SAT rather than invent
// unsigned saturation. AvgMode=3 is PlusOne, not two negations
// (Ryujinx Decoders/InstDecoders.cs:354-359); keep it Unknown.
fn iadd(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::IAdd(ops::IAdd {
        dst: d,
        lhs: l,
        rhs: int_form_src(r),
        cc: b(r, 47, 1) == 1,
        x: b(r, 43, 1) == 1,
        negate: [b(r, 49, 1) == 1, b(r, 48, 1) == 1],
    }))
}
const IADD_FIELDS: Fields<2> = Fields([Field::bits(50, 1), Field::bits(48, 2)]);
const fn iadd_rule(r: u64) -> u8 {
    (b(r, 50, 1) == 0 && b(r, 48, 2) != 3) as u8
}
const IADD_FORMS: [u8; IADD_FIELDS.len()] = forms!(IADD_FIELDS, iadd_rule);
fn read_iadd(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, iadd];
    paths[(IADD_FORMS[IADD_FIELDS.index(x.r)]) as usize](x)
}
// XMAD (nouveau nv50_ir_emit_gm107.cpp:1937-1993; envytools gm107.c
// rows 0x5b00, 0x3600, 0x4e00, 0x5100): signed a/b at 48/49, cmode at
// 50.., h1 of a at 53, .CC 47 (Ryujinx InstDecoders.cs:5312-5395).
// .X stays Unknown: Ryujinx InstEmitIntegerArithmetic.cs:263-267 drops
// the addend C under .X, which is no semantics to port.
fn xmad_register(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);
    let cmode = XMAD_COPS[b(r, 50, 3) as usize];
    x.guarded(InstructionKind::Xmad(ops::Xmad {
        dst: d,
        a: l,
        b: Source::Register(Register(b(r, 20, 8) as u8)),
        c: Source::Register(Register(b(r, 39, 8) as u8)),
        signed: [b(r, 48, 1) == 1, b(r, 49, 1) == 1],
        high: [b(r, 53, 1) == 1, b(r, 35, 1) == 1],
        psl: b(r, 36, 1) == 1,
        mrg: b(r, 37, 1) == 1,
        cmode,
        cc: b(r, 47, 1) == 1,
    }))
}
const XMAD_REGISTER_FIELDS: Fields<2> = Fields([Field::bits(38, 1), Field::bits(50, 3)]);
const fn xmad_register_rule(r: u64) -> u8 {
    (b(r, 38, 1) == 0 && b(r, 50, 3) < 5) as u8
}
const XMAD_REGISTER_FORMS: [u8; XMAD_REGISTER_FIELDS.len()] =
    forms!(XMAD_REGISTER_FIELDS, xmad_register_rule);
fn read_xmad_register(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, xmad_register];
    paths[(XMAD_REGISTER_FORMS[XMAD_REGISTER_FIELDS.index(x.r)]) as usize](x)
}

fn xmad_immediate(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);
    let cmode = XMAD_COPS[b(r, 50, 3) as usize];
    x.guarded(InstructionKind::Xmad(ops::Xmad {
        dst: d,
        a: l,
        b: Source::Immediate(b(r, 20, 16)),
        c: Source::Register(Register(b(r, 39, 8) as u8)),
        signed: [b(r, 48, 1) == 1, b(r, 49, 1) == 1],
        high: [b(r, 53, 1) == 1, false],
        psl: b(r, 36, 1) == 1,
        mrg: b(r, 37, 1) == 1,
        cmode,
        cc: b(r, 47, 1) == 1,
    }))
}
const XMAD_IMMEDIATE_FIELDS: Fields<2> = Fields([Field::bits(38, 1), Field::bits(50, 3)]);
const fn xmad_immediate_rule(r: u64) -> u8 {
    (b(r, 38, 1) == 0 && b(r, 50, 3) < 5) as u8
}
const XMAD_IMMEDIATE_FORMS: [u8; XMAD_IMMEDIATE_FIELDS.len()] =
    forms!(XMAD_IMMEDIATE_FIELDS, xmad_immediate_rule);
fn read_xmad_immediate(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, xmad_immediate];
    paths[(XMAD_IMMEDIATE_FORMS[XMAD_IMMEDIATE_FIELDS.index(x.r)]) as usize](x)
}

fn xmad_constant(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);
    let cmode = XMAD_COPS[b(r, 50, 2) as usize];
    x.guarded(InstructionKind::Xmad(ops::Xmad {
        dst: d,
        a: l,
        b: Source::ConstantBuffer {
            binding: b(r, 34, 5) as u8,
            offset: (b(r, 20, 14) * 4) as u16,
        },
        c: Source::Register(Register(b(r, 39, 8) as u8)),
        signed: [b(r, 48, 1) == 1, b(r, 49, 1) == 1],
        high: [b(r, 53, 1) == 1, b(r, 52, 1) == 1],
        psl: b(r, 55, 1) == 1,
        mrg: b(r, 56, 1) == 1,
        cmode,
        cc: b(r, 47, 1) == 1,
    }))
}
const XMAD_CONSTANT_FIELDS: Fields<1> = Fields([Field::bits(54, 1)]);
const fn xmad_constant_rule(r: u64) -> u8 {
    (b(r, 54, 1) == 0) as u8
}
const XMAD_CONSTANT_FORMS: [u8; XMAD_CONSTANT_FIELDS.len()] =
    forms!(XMAD_CONSTANT_FIELDS, xmad_constant_rule);
fn read_xmad_constant(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, xmad_constant];
    paths[(XMAD_CONSTANT_FORMS[XMAD_CONSTANT_FIELDS.index(x.r)]) as usize](x)
}

fn xmad_constant_addend(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);
    let cmode = XMAD_COPS[b(r, 50, 2) as usize];
    x.guarded(InstructionKind::Xmad(ops::Xmad {
        dst: d,
        a: l,
        b: Source::Register(Register(b(r, 39, 8) as u8)),
        c: Source::ConstantBuffer {
            binding: b(r, 34, 5) as u8,
            offset: (b(r, 20, 14) * 4) as u16,
        },
        signed: [b(r, 48, 1) == 1, b(r, 49, 1) == 1],
        high: [b(r, 53, 1) == 1, b(r, 52, 1) == 1],
        psl: false,
        mrg: false,
        cmode,
        cc: b(r, 47, 1) == 1,
    }))
}
const XMAD_CONSTANT_ADDEND_FIELDS: Fields<1> = Fields([Field::bits(54, 1)]);
const fn xmad_constant_addend_rule(r: u64) -> u8 {
    (b(r, 54, 1) == 0) as u8
}
const XMAD_CONSTANT_ADDEND_FORMS: [u8; XMAD_CONSTANT_ADDEND_FIELDS.len()] =
    forms!(XMAD_CONSTANT_ADDEND_FIELDS, xmad_constant_addend_rule);
fn read_xmad_constant_addend(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, xmad_constant_addend];
    paths[(XMAD_CONSTANT_ADDEND_FORMS[XMAD_CONSTANT_ADDEND_FIELDS.index(x.r)]) as usize](x)
}
// ISCADD (envytools envydis/gm107.c:2014,2088,2134): srcs[0]
// negate 49, srcs[1] negate 48, shift bits 39..43, .CC 47.
// AvgMode=3 is not two negations (Ryujinx InstDecoders.cs:354-359).
fn shift_add(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::ShiftAdd(ops::ShiftAdd {
        dst: d,
        lhs: l,
        rhs: int_form_src(r),
        shift: b(r, 39, 5) as u8,
        negate: [b(r, 49, 1) == 1, b(r, 48, 1) == 1],
        cc: b(r, 47, 1) == 1,
    }))
}
const SHIFT_ADD_FIELDS: Fields<2> = Fields([Field::bits(48, 2), Field::bits(50, 1)]);
const fn shift_add_rule(r: u64) -> u8 {
    (b(r, 48, 2) != 3 && b(r, 50, 1) == 0) as u8
}
const SHIFT_ADD_FORMS: [u8; SHIFT_ADD_FIELDS.len()] = forms!(SHIFT_ADD_FIELDS, shift_add_rule);
fn read_shift_add(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, shift_add];
    paths[(SHIFT_ADD_FORMS[SHIFT_ADD_FIELDS.index(x.r)]) as usize](x)
}
// ISCADD32I: envytools gm107.c:2164 places CC52, D0, A8, imm32 20..51
// and shift53..57; Ryujinx InstDecoders.cs:3341-3352 (MIT) agrees, and
// Iscadd32i/EmitIscadd (InstEmitIntegerArithmetic.cs:211-219,460-479)
// adds A << shift to the immediate with no negation. Every field value
// has a meaning, so every ISCADD32I word decodes.
fn shift_add32i(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);
    let shift = b(r, 53, 5) as u8;
    x.guarded({
        InstructionKind::ShiftAdd(ops::ShiftAdd {
            dst: d,
            lhs: l,
            rhs: Source::Immediate(b(r, 20, 32)),
            shift,
            negate: [false; 2],
            cc: b(r, 52, 1) == 1,
        })
    })
}
// IADD3: gm107.c:1994,2068,2114; neg A/B/C 51/50/49,
// .X 48, .CC 47. Register-only halves A/B/C 35/33/31, each 2 bits;
// partial-sum shift 37..39 (gm107.c:1228-1253). Reserved selections
// stay Unknown. Ryujinx InstEmitIntegerArithmetic.cs:343-406 defines
// halves, negate and shift, but explicitly leaves .X/.CC unfinished.
const IADD3_REGISTER_FIELDS: Fields<5> = Fields([
    Field::bits(47, 2),
    Field::bits(37, 2),
    Field::bits(35, 2),
    Field::bits(33, 2),
    Field::bits(31, 2),
]);
const fn iadd3_register_rule(r: u64) -> u8 {
    (b(r, 47, 2) == 0
        && b(r, 37, 2) != 3
        && b(r, 35, 2) != 3
        && b(r, 33, 2) != 3
        && b(r, 31, 2) != 3) as u8
}
const IADD3_REGISTER_FORMS: [u8; IADD3_REGISTER_FIELDS.len()] =
    forms!(IADD3_REGISTER_FIELDS, iadd3_register_rule);
fn read_iadd3_register(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, iadd3_register];
    paths[(IADD3_REGISTER_FORMS[IADD3_REGISTER_FIELDS.index(x.r)]) as usize](x)
}
fn iadd3_of(x: Word, rhs: Source, halves: [HalfSelect; 3], shift: PartialSumShift) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::IAdd3(ops::IAdd3 {
        dst: Register(b(r, 0, 8) as u8),
        lhs: Register(b(r, 8, 8) as u8),
        rhs,
        addend: Register(b(r, 39, 8) as u8),
        halves,
        shift,
        negate: [b(r, 51, 1) != 0, b(r, 50, 1) != 0, b(r, 49, 1) != 0],
    }))
}
/// IADD3 register half select field: 1 low, 2 high, otherwise full.
const IADD3_HALVES: [HalfSelect; 4] = [
    HalfSelect::Full,
    HalfSelect::Low,
    HalfSelect::High,
    HalfSelect::Full,
];
/// IADD3 register partial-sum shift 37..38: 1 right, 2 left, otherwise none.
const IADD3_SHIFTS: [PartialSumShift; 4] = [
    PartialSumShift::Unshifted,
    PartialSumShift::Right,
    PartialSumShift::Left,
    PartialSumShift::Unshifted,
];
/// Register IADD3 (0x5cc0): per-source halves 35/33/31 and a shift.
fn iadd3_register(x: Word) -> Instruction {
    let r = x.r;
    let half = |at| IADD3_HALVES[b(r, at, 2) as usize];
    iadd3_of(
        x,
        register_b(r),
        [half(35), half(33), half(31)],
        IADD3_SHIFTS[b(r, 37, 2) as usize],
    )
}
/// Constant-buffer and immediate IADD3: full sources, no shift.
fn iadd3(x: Word) -> Instruction {
    let r = x.r;
    iadd3_of(
        x,
        int_form_src(r),
        [HalfSelect::Full; 3],
        PartialSumShift::Unshifted,
    )
}
const IADD3_FIELDS: Fields<1> = Fields([Field::bits(47, 2)]);
const fn iadd3_rule(r: u64) -> u8 {
    (b(r, 47, 2) == 0) as u8
}
const IADD3_FORMS: [u8; IADD3_FIELDS.len()] = forms!(IADD3_FIELDS, iadd3_rule);
fn read_iadd3(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, iadd3];
    paths[(IADD3_FORMS[IADD3_FIELDS.index(x.r)]) as usize](x)
}
// LEA low: envytools envydis/gm107.c:2022,2093,2140:
// D0..7, A8..15, B20..27 / cbuf20..38 / signed20-bit immediate,
// shift39..43, negate45, X46, CC47, predicate destination48..50.
// Ryujinx Decoders/InstDecoders.cs:3581-3627 agrees; arithmetic is
// InstEmitIntegerArithmetic.cs:481-490. X/CC and predicate outputs
// are unimplemented. Unnamed bits28..38 in register form and bit44
// have no semantics in these references, so nonzero stays Unknown.
const LEA_REGISTER_FIELDS: Fields<4> = Fields([
    Field::bits(46, 2),
    Field::bits(48, 3),
    Field::bits(44, 1),
    Field::zero(28, 11),
]);
const fn lea_register_rule(r: u64) -> u8 {
    (b(r, 46, 2) == 0 && b(r, 48, 3) == 7 && b(r, 44, 1) == 0 && b(r, 28, 11) == 0) as u8
}
const LEA_REGISTER_FORMS: [u8; LEA_REGISTER_FIELDS.len()] =
    forms!(LEA_REGISTER_FIELDS, lea_register_rule);
fn read_lea_register(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, lea];
    paths[(LEA_REGISTER_FORMS[LEA_REGISTER_FIELDS.index(x.r)]) as usize](x)
}
fn lea(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::Lea(ops::Lea {
        dst: d,
        lhs: l,
        rhs: int_form_src(r),
        shift: b(r, 39, 5) as u8,
        negate: b(r, 45, 1) != 0,
    }))
}
const LEA_FIELDS: Fields<3> = Fields([Field::bits(46, 2), Field::bits(48, 3), Field::bits(44, 1)]);
const fn lea_rule(r: u64) -> u8 {
    (b(r, 46, 2) == 0 && b(r, 48, 3) == 7 && b(r, 44, 1) == 0) as u8
}
const LEA_FORMS: [u8; LEA_FIELDS.len()] = forms!(LEA_FIELDS, lea_rule);
fn read_lea(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, lea];
    paths[(LEA_FORMS[LEA_FIELDS.index(x.r)]) as usize](x)
}
// LEA.HI register: gm107.c:2021, dst 0, low 8, addend 20, high
// 39, shift 28..33, neg 37, .X 38, .CC 47, predicate output 48..51.
// LEA.HI cbuf: gm107.c:2163, cbuf offset 20..34/bank 34..39,
// shift 51..56, neg 56, .X 57. This is NOT an immediate addend.
// Ryujinx InstEmitIntegerArithmetic.cs:493-523 leaves .X/.CC
// unfinished; predicate carry output is also unimplemented.
fn lea_hi(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::LeaHi(ops::LeaHi {
        dst: d,
        low: l,
        addend: Source::Register(Register(b(r, 20, 8) as u8)),
        high: Register(b(r, 39, 8) as u8),
        shift: b(r, 28, 5) as u8,
        negate: b(r, 37, 1) != 0,
    }))
}
const LEA_HI_FIELDS: Fields<3> =
    Fields([Field::bits(38, 1), Field::bits(47, 1), Field::bits(48, 3)]);
const fn lea_hi_rule(r: u64) -> u8 {
    (b(r, 38, 1) == 0 && b(r, 47, 1) == 0 && b(r, 48, 3) == 7) as u8
}
const LEA_HI_FORMS: [u8; LEA_HI_FIELDS.len()] = forms!(LEA_HI_FIELDS, lea_hi_rule);
fn read_lea_hi(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, lea_hi];
    paths[(LEA_HI_FORMS[LEA_HI_FIELDS.index(x.r)]) as usize](x)
}

fn lea_hi_immediate(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::LeaHi(ops::LeaHi {
        dst: d,
        low: l,
        addend: Source::ConstantBuffer {
            binding: b(r, 34, 5) as u8,
            offset: (b(r, 20, 14) * 4) as u16,
        },
        high: Register(b(r, 39, 8) as u8),
        shift: b(r, 51, 5) as u8,
        negate: b(r, 56, 1) != 0,
    }))
}
const LEA_HI_IMMEDIATE_FIELDS: Fields<3> =
    Fields([Field::bits(57, 1), Field::bits(47, 1), Field::bits(48, 3)]);
const fn lea_hi_immediate_rule(r: u64) -> u8 {
    (b(r, 57, 1) == 0 && b(r, 47, 1) == 0 && b(r, 48, 3) == 7) as u8
}
const LEA_HI_IMMEDIATE_FORMS: [u8; LEA_HI_IMMEDIATE_FIELDS.len()] =
    forms!(LEA_HI_IMMEDIATE_FIELDS, lea_hi_immediate_rule);
fn read_lea_hi_immediate(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, lea_hi_immediate];
    paths[(LEA_HI_IMMEDIATE_FORMS[LEA_HI_IMMEDIATE_FIELDS.index(x.r)]) as usize](x)
}
// SHF: Mesa sm50.rs:1653-1691; gm107.c:1990,2018,2110,2138.
// Low 8..16, shift reg 20..28 / immediate 20..26 (Ryujinx
// InstDecoders.cs:4365-4394), high 39..47; type 37..39 (0,2,3),
// dst high 48 (right only), .X 49, wrap 50, .CC 47.
// No cbuf encoding exists (sm50.rs:1664; gm107.c:2064-2102).
// .X and invalid left .HI stay Unknown (sm50.rs:1684-1690).
/// The immediate (0x3...) and register (0x5...) forms differ at bit 62.
const FUNNEL_SHIFT_RIGHT_FIELDS: Fields<3> =
    Fields([Field::bits(37, 2), Field::bits(49, 1), Field::bits(62, 1)]);
const fn funnel_shift_right_rule(r: u64) -> u8 {
    (b(r, 37, 2) != 1 && b(r, 49, 1) == 0) as u8 * (1 + b(r, 62, 1) as u8)
}
const FUNNEL_SHIFT_RIGHT_FORMS: [u8; FUNNEL_SHIFT_RIGHT_FIELDS.len()] =
    forms!(FUNNEL_SHIFT_RIGHT_FIELDS, funnel_shift_right_rule);
fn read_funnel_shift_right(x: Word) -> Instruction {
    let paths: [Decode; 3] = [
        unknown,
        funnel_shift_right_immediate,
        funnel_shift_right_register,
    ];
    paths[(FUNNEL_SHIFT_RIGHT_FORMS[FUNNEL_SHIFT_RIGHT_FIELDS.index(x.r)]) as usize](x)
}
/// SHF type 37..38: 0 32-bit, 2 unsigned 64-bit, 3 signed 64-bit; 1 stays
/// Unknown in the form tables.
const FUNNEL_TYPES: [FunnelType; 4] = [
    FunnelType::Bits32,
    FunnelType::Bits32,
    FunnelType::Unsigned64,
    FunnelType::Signed64,
];
fn funnel_shift_of(x: Word, shift: Source, right: bool, dst_high: bool) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::FunnelShift(ops::FunnelShift {
        dst: Register(b(r, 0, 8) as u8),
        low: Register(b(r, 8, 8) as u8),
        high: Register(b(r, 39, 8) as u8),
        shift,
        right,
        dst_high,
        ty: FUNNEL_TYPES[b(r, 37, 2) as usize],
        wrap: b(r, 50, 1) != 0,
        cc: b(r, 47, 1) != 0,
    }))
}
fn shift_immediate(r: u64) -> Source {
    Source::Immediate(b(r, 20, 6))
}
fn funnel_shift_right_register(x: Word) -> Instruction {
    funnel_shift_of(x, register_b(x.r), true, b(x.r, 48, 1) != 0)
}
fn funnel_shift_right_immediate(x: Word) -> Instruction {
    funnel_shift_of(x, shift_immediate(x.r), true, b(x.r, 48, 1) != 0)
}
fn funnel_shift_left_register(x: Word) -> Instruction {
    funnel_shift_of(x, register_b(x.r), false, true)
}
fn funnel_shift_left_immediate(x: Word) -> Instruction {
    funnel_shift_of(x, shift_immediate(x.r), false, true)
}
const FUNNEL_SHIFT_FIELDS: Fields<4> = Fields([
    Field::bits(37, 2),
    Field::bits(49, 1),
    Field::bits(48, 1),
    Field::bits(62, 1),
]);
const fn funnel_shift_rule(r: u64) -> u8 {
    (b(r, 37, 2) != 1 && b(r, 49, 1) == 0 && b(r, 48, 1) == 0) as u8 * (1 + b(r, 62, 1) as u8)
}
const FUNNEL_SHIFT_FORMS: [u8; FUNNEL_SHIFT_FIELDS.len()] =
    forms!(FUNNEL_SHIFT_FIELDS, funnel_shift_rule);
fn read_funnel_shift(x: Word) -> Instruction {
    let paths: [Decode; 3] = [
        unknown,
        funnel_shift_left_immediate,
        funnel_shift_left_register,
    ];
    paths[(FUNNEL_SHIFT_FORMS[FUNNEL_SHIFT_FIELDS.index(x.r)]) as usize](x)
}
// Register BFI: envytools gm107.c:2019,203-207 and Mesa
// nv50_ir_emit_gm107.cpp:2293-2298,2312,2324-2326:
// dst0..7, insert8..15, packed range20..27, base39..46, CC47.
// Ryujinx InstDecoders.cs:1030-1040 and InstEmitBitfield.cs:42-50,
// 154-162 use low8/next8 of range as offset/count. The existing
// reaching-definition proof folds constant ranges; dynamic ranges use
// the same offset/count extraction as Ryujinx. Known invalid ranges
// stay Unknown, without inventing out-of-domain results.
// Reject CC and unnamed28..38/48..50 rather than ignore them.
fn bfi_register(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::BitFieldInsert(ops::BitFieldInsert {
        dst: d,
        insert: l,
        base: Source::Register(Register(b(r, 39, 8) as u8)),
        range: Source::Register(Register(b(r, 20, 8) as u8)),
    }))
}
const BFI_REGISTER_FIELDS: Fields<2> = Fields([Field::zero(28, 11), Field::zero(47, 4)]);
const fn bfi_register_rule(r: u64) -> u8 {
    (b(r, 28, 11) == 0 && b(r, 47, 4) == 0) as u8
}
const BFI_REGISTER_FORMS: [u8; BFI_REGISTER_FIELDS.len()] =
    forms!(BFI_REGISTER_FIELDS, bfi_register_rule);
fn read_bfi_register(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, bfi_register];
    paths[(BFI_REGISTER_FORMS[BFI_REGISTER_FIELDS.index(x.r)]) as usize](x)
}
// BFI.C: Mesa src/gallium/drivers/nouveau/codegen/nv50_ir_emit_gm107.cpp:
// 2300-2302,2312,2324-2326; envytools envydis/gm107.c:2092,203-207,
// 302-309. Packed control is cbuf bank34..38, word offset20..33;
// base39..46, insert8..15, dst0..7. CC47 and unnamed48..50
// stay Unknown. Ryujinx InstEmitBitfield.cs:64-72,154-162.
fn bfi_constant(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::BitFieldInsert(ops::BitFieldInsert {
        dst: d,
        insert: l,
        base: Source::Register(Register(b(r, 39, 8) as u8)),
        range: Source::ConstantBuffer {
            binding: b(r, 34, 5) as u8,
            offset: (b(r, 20, 14) * 4) as u16,
        },
    }))
}
const BFI_CONSTANT_FIELDS: Fields<1> = Fields([Field::zero(47, 4)]);
const fn bfi_constant_rule(r: u64) -> u8 {
    (b(r, 47, 4) == 0) as u8
}
const BFI_CONSTANT_FORMS: [u8; BFI_CONSTANT_FIELDS.len()] =
    forms!(BFI_CONSTANT_FIELDS, bfi_constant_rule);
fn read_bfi_constant(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, bfi_constant];
    paths[(BFI_CONSTANT_FORMS[BFI_CONSTANT_FIELDS.index(x.r)]) as usize](x)
}
// BFI immediate: envytools envydis/gm107.c:2139,203-207,253:
// dst0..8, insert8..16, signed imm20..39/sign56, base39..47, CC47.
// Mesa nv50_ir_emit_gm107.cpp:2304-2312 confirms immediate and base.
// Ryujinx InstEmitBitfield.cs:154-162 uses low8 as offset, next8 as
// count. Fold both at translation time, saving its AND and extract.
// Reject CC, unused high immediate bits, and undefined SPIR-V ranges.
fn bfi_immediate(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::BitFieldInsert(ops::BitFieldInsert {
        dst: d,
        insert: l,
        base: Source::Register(Register(b(r, 39, 8) as u8)),
        range: Source::Immediate(b(r, 20, 16)),
    }))
}
const BFI_IMMEDIATE_FIELDS0: Fields<3> =
    Fields([Field::zero(47, 4), Field::zero(36, 3), Field::bits(56, 1)]);
const fn bfi_immediate_rule0(r: u64) -> u8 {
    (b(r, 47, 4) == 0 && b(r, 36, 3) == 0 && b(r, 56, 1) == 0) as u8
}
const BFI_IMMEDIATE_FORMS0: [u8; BFI_IMMEDIATE_FIELDS0.len()] =
    forms!(BFI_IMMEDIATE_FIELDS0, bfi_immediate_rule0);
const BFI_IMMEDIATE_FIELDS1: Fields<2> =
    Fields([Field::capped(20, 8, 33), Field::capped(28, 8, 33)]);
const fn bfi_immediate_rule1(r: u64) -> u8 {
    (b(r, 20, 8) <= 32 && b(r, 28, 8) <= 32 - b(r, 20, 8)) as u8
}
const BFI_IMMEDIATE_FORMS1: [u8; BFI_IMMEDIATE_FIELDS1.len()] =
    forms!(BFI_IMMEDIATE_FIELDS1, bfi_immediate_rule1);
fn read_bfi_immediate(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, bfi_immediate];
    paths[(BFI_IMMEDIATE_FORMS0[BFI_IMMEDIATE_FIELDS0.index(x.r)]
        & BFI_IMMEDIATE_FORMS1[BFI_IMMEDIATE_FIELDS1.index(x.r)]) as usize](x)
}
// BFI.RC: envytools envydis/gm107.c:2041,203-207,258,302-309:
// dst0..8, insert8..16, control39..47, cbuf offset20..34 in
// words and bank34..39. Mesa nv50_ir_emit_gm107.cpp:2314-2326
// confirms the swapped control/base order and CC47.
// Bits48..50 have no semantics in that row; reject them and CC.
// Ryujinx InstEmitBitfield.cs:75-83,154-162 uses C as base,
// A as insert, and register39 as packed offset/count.
fn bfi_constant_insert(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::BitFieldInsert(ops::BitFieldInsert {
        dst: d,
        insert: l,
        base: Source::ConstantBuffer {
            binding: b(r, 34, 5) as u8,
            offset: (b(r, 20, 14) * 4) as u16,
        },
        range: Source::Register(Register(b(r, 39, 8) as u8)),
    }))
}
const BFI_CONSTANT_INSERT_FIELDS: Fields<1> = Fields([Field::zero(47, 4)]);
const fn bfi_constant_insert_rule(r: u64) -> u8 {
    (b(r, 47, 4) == 0) as u8
}
const BFI_CONSTANT_INSERT_FORMS: [u8; BFI_CONSTANT_INSERT_FIELDS.len()] =
    forms!(BFI_CONSTANT_INSERT_FIELDS, bfi_constant_insert_rule);
fn read_bfi_constant_insert(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, bfi_constant_insert];
    paths[(BFI_CONSTANT_INSERT_FORMS[BFI_CONSTANT_INSERT_FIELDS.index(x.r)]) as usize](x)
}
// BFE: Mesa sm50.rs:1166-1195; gm107.c:2017,2091,2137.
// Base 8..16, range by form 20..39/sign 56, signed 48, .BREV
// 40, .CC 47. Ryujinx InstEmitBitfield.cs:128-151 leaves .CC
// unfinished, so it stays Unknown rather than discarding the flags.
fn bfe_of(x: Word, range: Source) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded({
        InstructionKind::BitFieldExtract(ops::BitFieldExtract {
            dst: d,
            base: l,
            // Mesa sm50.rs:1174-1175: only the bottom 16 bits matter.
            // Reject statically invalid SPIR-V ranges rather than invent
            // the corner cases left TODO in InstEmitBitfield.cs:151.
            range,
            signed: b(r, 48, 1) != 0,
            reverse: b(r, 40, 1) != 0,
        })
    })
}
/// Register and constant-buffer BFE.
fn bfe(x: Word) -> Instruction {
    bfe_of(x, int_form_src(x.r))
}
/// Immediate BFE: the range is the low 16 bits of the literal.
fn bfe_immediate(x: Word) -> Instruction {
    bfe_of(x, Source::Immediate(b(x.r, 20, 16)))
}
const BFE_FIELDS: Fields<1> = Fields([Field::bits(47, 1)]);
const fn bfe_rule(r: u64) -> u8 {
    (b(r, 47, 1) == 0) as u8
}
const BFE_FORMS: [u8; BFE_FIELDS.len()] = forms!(BFE_FIELDS, bfe_rule);
fn read_bfe(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, bfe];
    paths[(BFE_FORMS[BFE_FIELDS.index(x.r)]) as usize](x)
}
const BFE_IMMEDIATE_FIELDS0: Fields<1> = Fields([Field::bits(47, 1)]);
const fn bfe_immediate_rule0(r: u64) -> u8 {
    (b(r, 47, 1) == 0) as u8
}
const BFE_IMMEDIATE_FORMS0: [u8; BFE_IMMEDIATE_FIELDS0.len()] =
    forms!(BFE_IMMEDIATE_FIELDS0, bfe_immediate_rule0);
const BFE_IMMEDIATE_FIELDS1: Fields<2> =
    Fields([Field::capped(20, 8, 33), Field::capped(28, 8, 33)]);
const fn bfe_immediate_rule1(r: u64) -> u8 {
    (b(r, 20, 8) <= 32 && b(r, 28, 8) <= 32 - b(r, 20, 8)) as u8
}
const BFE_IMMEDIATE_FORMS1: [u8; BFE_IMMEDIATE_FIELDS1.len()] =
    forms!(BFE_IMMEDIATE_FIELDS1, bfe_immediate_rule1);
fn read_bfe_immediate(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, bfe_immediate];
    paths[(BFE_IMMEDIATE_FORMS0[BFE_IMMEDIATE_FIELDS0.index(x.r)]
        & BFE_IMMEDIATE_FORMS1[BFE_IMMEDIATE_FIELDS1.index(x.r)]) as usize](x)
}

/// 0x4c58: B is the constant buffer.
fn fadd_constant(x: Word) -> Instruction {
    let r = x.r;
    x.guarded(fadd(r, constant_b(r), fadd_operand(r)))
}
// Float immediate form: 20-bit immediate holding the top bits of an
// f32, sign at bit 56 (set_src_imm_f20(20..39, 56), sm50.rs).
fn fadd_immediate(x: Word) -> Instruction {
    let r = x.r;
    x.guarded(fadd(r, imm32(r), fadd_immediate20(r)))
}
// FADD32I (envytools gm107.c:2167): 32-bit immediate 20..52, .CC 52,
// srcs[1] neg 53 abs 57, srcs[0] abs 54 neg 56, ftz 55.
fn fadd32i(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::FAdd(ops::FAdd {
        dst: d,
        lhs: l,
        rhs: src(r, true, false),
        modifiers: FAddModifiers::Immediate32 {
            lhs_abs: source_abs(r, 54),
            rhs_abs: source_abs(r, 57),
            lhs_negation: source_negation(r, 56),
            rhs_negation: source_negation(r, 53),
            denormals: denormals(r, 55),
        },
        round: RoundingMode::NearestEven,
        flags: float_flags(r, 52),
    }))
}
fn read_fadd32i(x: Word) -> Instruction {
    fadd32i(x)
}
// FMUL32I (Ryujinx InstTable.cs:116, MIT: 0x1e00-0x1eff;
// InstDecoders.cs:2129-2141): 32-bit immediate 20..51, CC 52, Fmz
// 53..54 (1 FTZ, 2 FMZ, 3 invalid), sat 55. .CC52 has the same concrete
// Z/N state as the other FMUL roads. Fmz 3 stays Unknown.
fn fmul32i(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::FMul(ops::FMul {
        dst: d,
        lhs: l,
        rhs: src(r, true, false),
        modifiers: FMulModifiers::Immediate32 {
            saturation: saturation(r, 55),
            flush: flush(r, 53),
        },
        flags: float_flags(r, 52),
    }))
}
const FMUL32I_FIELDS: Fields<1> = Fields([Field::bits(53, 2)]);
const fn fmul32i_rule(r: u64) -> u8 {
    (b(r, 53, 2) != 3) as u8
}
const FMUL32I_FORMS: [u8; FMUL32I_FIELDS.len()] = forms!(FMUL32I_FIELDS, fmul32i_rule);
fn read_fmul32i(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, fmul32i];
    paths[(FMUL32I_FORMS[FMUL32I_FIELDS.index(x.r)]) as usize](x)
}
// FFMA32I (Ryujinx InstTable.cs:106, MIT: 0x0c00-0x0fff;
// InstDecoders.cs:1960-1974): 32-bit immediate 20..51, CC 52, Fmz
// 53..54 (1 FTZ, 2 FMZ, 3 invalid), sat 55, negate A 56, negate C 57;
// the addend is the destination register (InstEmitFloatArithmetic.cs:
// 200-209,411-433). CC52 is concrete Z/N state; Fmz 3 stays Unknown.
fn ffma32i(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);
    let flush = flush(r, 53);
    x.guarded(InstructionKind::FFma(ops::FFma {
        dst: d,
        lhs: l,
        rhs: src(r, true, false),
        addend: Source::Register(d),
        flags: float_flags(r, 52),
        modifiers: FFmaModifiers {
            product_negate: b(r, 56, 1) != 0,
            addend_negate: b(r, 57, 1) != 0,
            saturate: b(r, 55, 1) != 0,
            flush,
        },
    }))
}
const FFMA32I_FIELDS: Fields<1> = Fields([Field::bits(53, 2)]);
const fn ffma32i_rule(r: u64) -> u8 {
    (b(r, 53, 2) != 3) as u8
}
const FFMA32I_FORMS: [u8; FFMA32I_FIELDS.len()] = forms!(FFMA32I_FIELDS, ffma32i_rule);
fn read_ffma32i(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, ffma32i];
    paths[(FFMA32I_FORMS[FFMA32I_FIELDS.index(x.r)]) as usize](x)
}
// Mesa's OpFFma immediate path uses opcode 0x3280 and
// set_src_imm_f20 (sm50.rs:558-566); 0x32a0 is the same
// immediate family with the producer's modifier bit set.
/// FFMA directed rounding 51..52 and FTZ+DNZ remain Unknown.
const FFMA_FIELDS: Fields<2> = Fields([Field::bits(53, 2), Field::zero(51, 2)]);
const fn ffma_rule(r: u64) -> u8 {
    (b(r, 53, 2) != 3 && b(r, 51, 2) == 0) as u8
}
const FFMA_FORMS: [u8; FFMA_FIELDS.len()] = forms!(FFMA_FIELDS, ffma_rule);
fn ffma_form(x: Word, path: Decode) -> Instruction {
    let paths: [Decode; 2] = [unknown, path];
    paths[FFMA_FORMS[FFMA_FIELDS.index(x.r)] as usize](x)
}
fn read_ffma_immediate(x: Word) -> Instruction {
    ffma_form(x, ffma_immediate)
}
fn read_ffma_constant_addend(x: Word) -> Instruction {
    ffma_form(x, ffma_constant_addend)
}
fn read_ffma_constant(x: Word) -> Instruction {
    ffma_form(x, ffma_constant)
}
fn ffma_immediate(x: Word) -> Instruction {
    let r = x.r;

    x.guarded(ffma(
        r,
        imm32(r),
        Source::Register(Register(b(r, 39, 8) as u8)),
    ))
}
// 0x5180: srcs[1] register 39..47, srcs[2] constant buffer 20..39
// (sm50.rs OpFFma encode).
fn ffma_constant_addend(x: Word) -> Instruction {
    let r = x.r;

    x.guarded(ffma(
        r,
        Source::Register(Register(b(r, 39, 8) as u8)),
        Source::ConstantBuffer {
            binding: b(r, 34, 5) as u8,
            offset: (b(r, 20, 14) * 4) as u16,
        },
    ))
}
// 0x5980: srcs[1] register 20..28; 0x4980: srcs[1] constant buffer
// 20..39; srcs[2] register 39..47 in both (sm50.rs OpFFma encode).
fn ffma_constant(x: Word) -> Instruction {
    let r = x.r;

    x.guarded(ffma(
        r,
        float_form_src(r),
        Source::Register(Register(b(r, 39, 8) as u8)),
    ))
}
// FMNMX with a runtime selector predicate 39..41, invert 42:
// InstEmitFloatMinMax.cs:78-104 (Ryujinx, MIT) selects minNum when the
// predicate holds, else maxNum.
fn fmnmx_select(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);
    let register = b(r, 39, 3) as u8;
    x.guarded(InstructionKind::FMnMxSelect(ops::FMnMxSelect {
        dst: d,
        lhs: l,
        rhs: float_form_src(r),
        min_when: Predicate {
            register,
            inverted: b(r, 42, 1) == 1,
        },
        flags: [FloatFlags::Kept, FloatFlags::Written][b(r, 47, 1) as usize],
        abs: [b(r, 46, 1) == 1, b(r, 49, 1) == 1],
        negate: [b(r, 48, 1) == 1, b(r, 45, 1) == 1],
    }))
}
// Mesa OpFMnMx: min is a predicate source at 39..42 with negate 42
// (PT = min, !PT = max); a non-constant predicate takes fmnmx_select.
// FMNMX (envytools gm107.c:2005, 2079, 2125): srcs[0] neg 48 abs 46,
// srcs[1] neg 45 abs 49, ftz 44, .CC 47, min predicate 39..42.
fn fmnmx(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::FMnMx(ops::FMnMx {
        dst: d,
        lhs: l,
        rhs: float_form_src(r),
        min: b(r, 42, 1) == 0,
        flags: [FloatFlags::Kept, FloatFlags::Written][b(r, 47, 1) as usize],
        abs: [b(r, 46, 1) == 1, b(r, 49, 1) == 1],
        negate: [b(r, 48, 1) == 1, b(r, 45, 1) == 1],
    }))
}
// FMNMX (Ryujinx InstTable.cs, MIT: 0x5c60-0x5c67, 0x4c60-0x4c67,
// 0x3860-0x3867, 0x3960-0x3967; InstDecoders.cs:2019-2074): PT (7)
// is a constant min/max, any other predicate selects at run time.
// FTZ 44 is ignored as Ryujinx does; CC 47 is closed FloatFlags state.
const FMNMX_FIELDS: Fields<1> = Fields([Field::bits(39, 3)]);
const fn fmnmx_rule(r: u64) -> u8 {
    1 + (b(r, 39, 3) != 7) as u8
}
const FMNMX_FORMS: [u8; FMNMX_FIELDS.len()] = forms!(FMNMX_FIELDS, fmnmx_rule);
fn read_fmnmx(x: Word) -> Instruction {
    let paths: [Decode; 3] = [unknown, fmnmx, fmnmx_select];
    paths[(FMNMX_FORMS[FMNMX_FIELDS.index(x.r)]) as usize](x)
}
// SHFL: Mesa sm50.rs:2043-2080; envytools/envydis/gm107.c:1896,
// :610-623. dst 0..7, src 8..15, lane 20..27 or imm5 20..24,
// lane immediate flag 28, control immediate flag 29, mode 30..31,
// control register 39..46 or imm13 34..46, predicate dst 48..50.
fn shuffle(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::Shuffle(ops::Shuffle {
        dst: d,
        src: l,
        lane: [
            Source::Register(Register(b(r, 20, 8) as u8)),
            Source::Immediate(b(r, 20, 5)),
        ][b(r, 28, 1) as usize],
        control: [
            Source::Register(Register(b(r, 39, 8) as u8)),
            Source::Immediate(b(r, 34, 13)),
        ][b(r, 29, 1) as usize],
        in_bounds: b(r, 48, 3) as u8,
        mode: [
            ShuffleMode::Idx,
            ShuffleMode::Up,
            ShuffleMode::Down,
            ShuffleMode::Bfly,
        ][b(r, 30, 2) as usize],
    }))
}
// FSWZADD is two-source, per-quad-lane arithmetic, not a derivative
// by itself. Ryujinx Instructions/InstEmitWarp.cs:11-21 and
// CodeGen/Spirv/Instructions.cs:1140-1168 use both sources and mask.
// Mesa nak/sm50.rs:913-943: dst0..8, A8..16, B20..28,
// four two-bit ops28..36, rounding39..41, NDV38, FTZ44, CC47.
// envytools envydis/gm107.c:2051 agrees. Only default rounding,
// NDV=0, CC=0 and zero unnamed fields are implemented. FTZ changes
// input and result denormals (Mesa ir.rs:8944-8952).
fn swizzle_add(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);

    x.guarded(InstructionKind::SwizzleAdd(ops::SwizzleAdd {
        dst: d,
        src: [Register(b(r, 8, 8) as u8), Register(b(r, 20, 8) as u8)],
        mask: b(r, 28, 8) as u8,
        ftz: b(r, 44, 1) != 0,
    }))
}
const SWIZZLE_ADD_FIELDS: Fields<2> = Fields([Field::zero(36, 8), Field::zero(45, 3)]);
const fn swizzle_add_rule(r: u64) -> u8 {
    (b(r, 36, 8) == 0 && b(r, 45, 3) == 0) as u8
}
const SWIZZLE_ADD_FORMS: [u8; SWIZZLE_ADD_FIELDS.len()] =
    forms!(SWIZZLE_ADD_FIELDS, swizzle_add_rule);
fn read_swizzle_add(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, swizzle_add];
    paths[(SWIZZLE_ADD_FORMS[SWIZZLE_ADD_FIELDS.index(x.r)]) as usize](x)
}
// RRO register: sm50.rs:714-739, gm107.c:2000. Source 20..28,
// abs 49 then neg 45. Both operation modes at bit 39 have the same
// Vulkan lowering (Ryujinx InstEmitMultifunction.cs:86-94): the
// paired MUFU performs the host operation without range reduction.
fn rro_register(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);

    x.guarded(InstructionKind::Rro(ops::Rro {
        dst: d,
        src: Source::Register(Register(b(r, 20, 8) as u8)),
        abs: b(r, 49, 1) != 0,
        negate: b(r, 45, 1) != 0,
        ex2: b(r, 39, 1) != 0,
    }))
}
// RRO constant buffer and f20 immediate, not MOV (Ryujinx
// InstTable.cs, MIT: 0x4c90-0x4c97, 0x3890-0x3897, 0x3990-0x3997;
// InstDecoders.cs:4226-4237 InstRroC/InstRroI): dst 0..7, B by form
// at 20 (f20 sign 56), neg 45, mode 39 (SINCOS/EX2), abs 49.
// InstEmitMultifunction.cs:18-29,86-94 applies abs then negation and
// defers range reduction to MUFU for both modes, like the register
// form above. nvdisasm ignores bits 48 and 50.
fn rro_operand(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);

    x.guarded(InstructionKind::Rro(ops::Rro {
        dst: d,
        src: float_form_src(r),
        abs: b(r, 49, 1) != 0,
        negate: b(r, 45, 1) != 0,
        ex2: b(r, 39, 1) != 0,
    }))
}
// Signedness is bit 48, so it varies the low opcode bit; the
// immediate's sign is bit 56 (envytools gm107.c:2013 and siblings).
// Extended mode43..44: envytools gm107.c:1383-1388; Mesa
// nv50_ir_emit_gm107.cpp:2020. Ryujinx
// InstEmitIntegerMinMax.cs:68 leaves X flags unfinished. Ordinary
// min/max cannot implement XLO/XMED/XHI, so retain Unknown.
fn imnmx(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded({
        InstructionKind::IMinMax(ops::IMinMax {
            // Mesa's OpIMnMx immediate path selects 0x3820; min is a predicate
            // source at 39..42 with negate 42 (PT = min, !PT = max).
            dst: d,
            lhs: l,
            rhs: int_form_src(r),
            min: b(r, 42, 1) == 0,
            signed: b(r, 48, 1) != 0,
            cc: b(r, 47, 1) == 1,
        })
    })
}
const IMNMX_FIELDS: Fields<2> = Fields([Field::bits(39, 3), Field::zero(43, 2)]);
const fn imnmx_rule(r: u64) -> u8 {
    (b(r, 43, 2) == 0) as u8 * (1 + (b(r, 39, 3) != 7) as u8)
}
const IMNMX_FORMS: [u8; IMNMX_FIELDS.len()] = forms!(IMNMX_FIELDS, imnmx_rule);
// IMNMX (Ryujinx InstTable.cs, MIT: 0x5c20-0x5c27, 0x4c20-0x4c27,
// 0x3820-0x3827, 0x3920-0x3927; InstDecoders.cs:3155-3201): PT (7) is
// a constant min/max, any other predicate selects at run time. CC 47
// sets Z/N from the result (InstEmitIntegerMinMax.cs:66); the extended
// modes 43..44 are a Ryujinx TODO (:68) and stay Unknown;
// nvdisasm ignores bits 49..50.
fn read_imnmx(x: Word) -> Instruction {
    let paths: [Decode; 3] = [unknown, imnmx, imnmx_select];
    paths[(IMNMX_FORMS[IMNMX_FIELDS.index(x.r)]) as usize](x)
}
// IMNMX with a runtime selector predicate 39..41, invert 42: Mesa
// nak/ir.rs:4093-4105 and Ryujinx InstEmitIntegerMinMax.cs:45-73 take
// the minimum when the predicate holds, else the maximum.
fn imnmx_select(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);
    let register = b(r, 39, 3) as u8;
    x.guarded(InstructionKind::IMinMaxSelect(ops::IMinMaxSelect {
        dst: d,
        lhs: l,
        rhs: int_form_src(r),
        min_when: Predicate {
            register,
            inverted: b(r, 42, 1) == 1,
        },
        signed: b(r, 48, 1) == 1,
        cc: b(r, 47, 1) == 1,
    }))
}
// IADD32I (envytools gm107.c:2162, mask 0xfe80): signed 32-bit
// immediate at 20..51, source negate at 56, .CC at 52. Ryujinx
// Iadd32i/EmitIadd (InstEmitIntegerArithmetic.cs:43-51,317-341) without
// .X sets flags exactly as EmitIscadd does with shift 0 (460-479; both
// SetIaddFlags(.., false), 665-699), so .CC is ShiftAdd's. AvgMode is
// 55..56 (InstDecoders.cs InstIadd32i): NegB (55) negates the immediate,
// NegA (56) the register; PlusOne has no Ryujinx semantics. .X 53 is
// EmitIadd's carry-in add, IADD's IR. Saturation is a Ryujinx TODO
// (:338); it and PlusOne remain Unknown.
fn iscadd32i(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded({
        InstructionKind::ShiftAdd(ops::ShiftAdd {
            dst: d,
            lhs: l,
            rhs: Source::Immediate(b(r, 20, 32)),
            shift: 0,
            // envytools lists the negate before REG_08: it negates srcs[0].
            negate: [b(r, 56, 1) == 1, b(r, 55, 1) == 1],
            cc: b(r, 52, 1) == 1,
        })
    })
}
/// IADD32I.X: Ryujinx Iadd32i/EmitIadd with extended (carry in).
fn iadd32i_extended(x: Word) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::IAdd(ops::IAdd {
        dst: Register(b(r, 0, 8) as u8),
        lhs: Register(b(r, 8, 8) as u8),
        rhs: Source::Immediate(b(r, 20, 32)),
        cc: b(r, 52, 1) == 1,
        x: true,
        negate: [b(r, 56, 1) == 1, b(r, 55, 1) == 1],
    }))
}
const ISCADD32I_FIELDS: Fields<3> =
    Fields([Field::bits(53, 1), Field::zero(54, 1), Field::bits(55, 2)]);
const fn iscadd32i_rule(r: u64) -> u8 {
    (b(r, 54, 1) == 0 && b(r, 55, 2) != 3) as u8 * (1 + b(r, 53, 1) as u8)
}
const ISCADD32I_FORMS: [u8; ISCADD32I_FIELDS.len()] = forms!(ISCADD32I_FIELDS, iscadd32i_rule);
fn read_iscadd32i(x: Word) -> Instruction {
    let paths: [Decode; 3] = [unknown, iscadd32i, iadd32i_extended];
    paths[(ISCADD32I_FORMS[ISCADD32I_FIELDS.index(x.r)]) as usize](x)
}
// Mesa's SM50 OpIMul uses 0x5c38/0x4c38/0x3838 and puts high and
// per-source signedness in bits 39..41 (sm50.rs:1405-1451). Ryujinx
// ImulR/I/C (InstEmitIntegerArithmetic.cs:148-168) add the constant 0.
fn imul(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::IntegerMul(ops::IntegerMul {
        dst: d,
        lhs: l,
        rhs: int_form_src(r),
        addend: Source::Immediate(0),
        high: b(r, 39, 1) != 0,
        signed: [b(r, 40, 1) != 0, b(r, 41, 1) != 0],
        negate: MulNegation::None,
    }))
}
// IMUL32I (Ryujinx InstTable.cs 0x1f00-0x1fff; InstDecoders.cs
// InstImul32i): imm32 20..51, CC 52, Hilo 53, ASigned 54, BSigned 55.
// Imul32i (InstEmitIntegerArithmetic.cs:171-178) adds the constant 0.
// CC is a Ryujinx TODO (:455) and stays Unknown.
fn imul32i(x: Word) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::IntegerMul(ops::IntegerMul {
        dst: Register(b(r, 0, 8) as u8),
        lhs: Register(b(r, 8, 8) as u8),
        rhs: Source::Immediate(b(r, 20, 32)),
        addend: Source::Immediate(0),
        high: b(r, 53, 1) != 0,
        signed: [b(r, 54, 1) != 0, b(r, 55, 1) != 0],
        negate: MulNegation::None,
    }))
}
const IMUL32I_FIELDS: Fields<1> = Fields([Field::bits(52, 1)]);
const fn imul32i_rule(r: u64) -> u8 {
    (b(r, 52, 1) == 0) as u8
}
const IMUL32I_FORMS: [u8; IMUL32I_FIELDS.len()] = forms!(IMUL32I_FIELDS, imul32i_rule);
fn read_imul32i(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, imul32i];
    paths[(IMUL32I_FORMS[IMUL32I_FIELDS.index(x.r)]) as usize](x)
}
/// IMAD AvgMode 51..52 (or 55..56 for IMAD32I) by value: NoNeg, NegB
/// (the addend), NegA (the multiplier). PlusOne (3) never decodes.
const IMAD_NEGATION: [MulNegation; 4] = [
    MulNegation::None,
    MulNegation::Addend,
    MulNegation::Multiplier,
    MulNegation::None,
];
const IMUL_FIELDS: Fields<1> = Fields([Field::bits(47, 1)]);
const fn imul_rule(r: u64) -> u8 {
    (b(r, 47, 1) == 0) as u8
}
const IMUL_FORMS: [u8; IMUL_FIELDS.len()] = forms!(IMUL_FIELDS, imul_rule);
fn read_imul(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, imul];
    paths[(IMUL_FORMS[IMUL_FIELDS.index(x.r)]) as usize](x)
}
// Mesa's SM50 OpIMad uses 0x5a00/0x4a00/0x3400 and encodes the
// signedness and modifiers in bits 48,49,50,51,52,53,54
// (sm50.rs:1347-1403; envytools gm107.c:2033,2084,2130). Ryujinx
// InstDecoders.cs:2999-3076: CC 47, ASigned 48, X 49, Sat 50, AvgMode
// 51..52, BSigned 53, Hilo 54. ImadR/I/C (InstEmitIntegerArithmetic.cs:
// 84-116,409-457) implement Hilo, signedness and NegA/NegB; CC, X and
// Sat are TODO there (:455) and PlusOne has no semantics: Unknown.
fn imul_hi(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::IntegerMul(ops::IntegerMul {
        dst: d,
        lhs: l,
        rhs: int_form_src(r),
        addend: Source::Register(Register(b(r, 39, 8) as u8)),
        high: b(r, 54, 1) != 0,
        signed: [b(r, 48, 1) != 0, b(r, 53, 1) != 0],
        negate: IMAD_NEGATION[b(r, 51, 2) as usize],
    }))
}
const IMUL_HI_FIELDS: Fields<3> =
    Fields([Field::zero(47, 1), Field::zero(49, 2), Field::bits(51, 2)]);
const fn imul_hi_rule(r: u64) -> u8 {
    (b(r, 47, 1) == 0 && b(r, 49, 2) == 0 && b(r, 51, 2) != 3) as u8
}
const IMUL_HI_FORMS: [u8; IMUL_HI_FIELDS.len()] = forms!(IMUL_HI_FIELDS, imul_hi_rule);
fn read_imul_hi(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, imul_hi];
    paths[(IMUL_HI_FORMS[IMUL_HI_FIELDS.index(x.r)]) as usize](x)
}
// IMAD.RC (Ryujinx InstTable.cs 0x5200-0x527f; InstDecoders.cs
// InstImadRc; ImadRc, InstEmitIntegerArithmetic.cs:118-127): the
// multiplier is register 39..46 and the addend the cbuf word at bank
// 34..38, word offset 20..33. Modifiers as the other IMAD forms.
fn imad_constant_addend(x: Word) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::IntegerMul(ops::IntegerMul {
        dst: Register(b(r, 0, 8) as u8),
        lhs: Register(b(r, 8, 8) as u8),
        rhs: Source::Register(Register(b(r, 39, 8) as u8)),
        addend: constant_b(r),
        high: b(r, 54, 1) != 0,
        signed: [b(r, 48, 1) != 0, b(r, 53, 1) != 0],
        negate: IMAD_NEGATION[b(r, 51, 2) as usize],
    }))
}
fn read_imad_constant_addend(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, imad_constant_addend];
    paths[(IMUL_HI_FORMS[IMUL_HI_FIELDS.index(x.r)]) as usize](x)
}
// IMADSP (Ryujinx InstTable.cs 0x5a80/0x3480/0x4a80/0x5280 forms and
// InstDecoders.cs InstImadspR/I/C/Rc, MIT; Ryujinx only logs "not
// implemented"). Semantics: Mesa nak/ir.rs OpIMadSp fold (MIT). A select
// 48..50, B select 53..54 (U24, S24, U16H0, S16H0), C width 51..52
// (32, 24, 16H0). The nvdisasm answer store shows C signed when A or B
// is (Mesa's "src2 signedness is tied" limit), not Ryujinx's bit-48-only
// CSelect; it shows no .CC for bit 47 and no answer for C width 3,
// which stays Unknown.
const IMADSP_TYPES: [IMadSpType; 8] = [
    IMadSpType::U32,
    IMadSpType::S32,
    IMadSpType::U24,
    IMadSpType::S24,
    IMadSpType::U16Lo,
    IMadSpType::S16Lo,
    IMadSpType::U16Hi,
    IMadSpType::S16Hi,
];
const fn imadsp_types(r: u64) -> [IMadSpType; 3] {
    let signed = b(r, 48, 1) | b(r, 53, 1);
    [
        IMADSP_TYPES[b(r, 48, 3) as usize],
        IMADSP_TYPES[b(r, 53, 2) as usize + 2],
        IMADSP_TYPES[(b(r, 51, 2) * 2 + signed) as usize],
    ]
}
/// Register, immediate and constant-buffer B (by form); C register 39.
fn imadsp(x: Word) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::IMadSp(ops::IMadSp {
        dst: Register(b(r, 0, 8) as u8),
        a: Register(b(r, 8, 8) as u8),
        b: int_form_src(r),
        c: Source::Register(Register(b(r, 39, 8) as u8)),
        types: imadsp_types(r),
    }))
}
/// 0x5280: B register 39, C the constant buffer.
fn imadsp_constant_addend(x: Word) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::IMadSp(ops::IMadSp {
        dst: Register(b(r, 0, 8) as u8),
        a: Register(b(r, 8, 8) as u8),
        b: Source::Register(Register(b(r, 39, 8) as u8)),
        c: constant_b(r),
        types: imadsp_types(r),
    }))
}
const IMADSP_FIELDS: Fields<1> = Fields([Field::bits(51, 2)]);
const fn imadsp_rule(r: u64) -> u8 {
    (b(r, 51, 2) != 3) as u8
}
const IMADSP_FORMS: [u8; IMADSP_FIELDS.len()] = forms!(IMADSP_FIELDS, imadsp_rule);
fn read_imadsp(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, imadsp];
    paths[(IMADSP_FORMS[IMADSP_FIELDS.index(x.r)]) as usize](x)
}
fn read_imadsp_constant_addend(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, imadsp_constant_addend];
    paths[(IMADSP_FORMS[IMADSP_FIELDS.index(x.r)]) as usize](x)
}
// IMAD32I (Ryujinx InstTable.cs 0x1000-0x13ff; InstDecoders.cs
// InstImad32i): imm32 20..51, CC 52, Hilo 53, ASigned 54, AvgMode
// 55..56, BSigned 57. Imad32i (InstEmitIntegerArithmetic.cs:129-138)
// adds the destination register. CC and PlusOne stay Unknown.
fn imad32i(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    x.guarded(InstructionKind::IntegerMul(ops::IntegerMul {
        dst: d,
        lhs: Register(b(r, 8, 8) as u8),
        rhs: Source::Immediate(b(r, 20, 32)),
        addend: Source::Register(d),
        high: b(r, 53, 1) != 0,
        signed: [b(r, 54, 1) != 0, b(r, 57, 1) != 0],
        negate: IMAD_NEGATION[b(r, 55, 2) as usize],
    }))
}
const IMAD32I_FIELDS: Fields<2> = Fields([Field::bits(52, 1), Field::bits(55, 2)]);
const fn imad32i_rule(r: u64) -> u8 {
    (b(r, 52, 1) == 0 && b(r, 55, 2) != 3) as u8
}
const IMAD32I_FORMS: [u8; IMAD32I_FIELDS.len()] = forms!(IMAD32I_FIELDS, imad32i_rule);
fn read_imad32i(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, imad32i];
    paths[(IMAD32I_FORMS[IMAD32I_FIELDS.index(x.r)]) as usize](x)
}
// MUFU: envytools envydis/gm107.c:2062,1690-1700; Mesa
// src/nouveau/compiler/nak/sm50.rs:749-768. dst 0..8, src 8..16,
// function 20..24, abs 46, neg 48, sat 50. Mask includes neg/sat;
// bit 49 and every undefined operand bit must be zero.
// Functions 6/7 pack the source as the high word of an FP64 value,
// apply reciprocal/inverse-square-root, and return the high word
// (Ryujinx InstEmitMultifunction.cs:58-71).
fn mufu(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);
    let op = MUFU_OPS[b(r, 20, 4) as usize];
    x.guarded(InstructionKind::MuFu(ops::MuFu {
        dst: d,
        src: l,
        op,
        abs: b(r, 46, 1) != 0,
        negate: b(r, 48, 1) != 0,
        saturate: b(r, 50, 1) != 0,
    }))
}
const MUFU_FIELDS: Fields<4> = Fields([
    Field::zero(24, 22),
    Field::zero(47, 1),
    Field::zero(49, 1),
    Field::bits(20, 4),
]);
const fn mufu_rule(r: u64) -> u8 {
    (b(r, 24, 22) == 0 && b(r, 47, 1) == 0 && b(r, 49, 1) == 0 && b(r, 20, 4) < 9) as u8
}
const MUFU_FORMS: [u8; MUFU_FIELDS.len()] = forms!(MUFU_FIELDS, mufu_rule);
fn read_mufu(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, mufu];
    paths[(MUFU_FORMS[MUFU_FIELDS.index(x.r)]) as usize](x)
}
// FLO register/cbuf/immediate: envytools envydis/gm107.c:2011,
// 2085,2131; signed48:1365-1368, SH41, INV40, CC47. Mesa
// nak/sm50.rs:1203-1225 confirms sources20..28/20..39, dst0..8,
// immediate sign56 and modifiers. CC remains Unknown.
// Signed SH remains Unknown: Ryujinx InstEmitBitfield.cs:171-179
// ignores signedness in SH mode, but Mesa nak/ir.rs:3738-3747
// counts leading sign bits. Do not silently discard signed48.
fn flo(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);

    x.guarded(InstructionKind::Flo(ops::Flo {
        dst: d,
        src: int_form_src(r),
        signed: b(r, 48, 1) != 0,
        invert: b(r, 40, 1) != 0,
        shift: b(r, 41, 1) != 0,
    }))
}
const FLO_FIELDS: Fields<3> = Fields([Field::bits(47, 1), Field::bits(48, 1), Field::bits(41, 1)]);
const fn flo_rule(r: u64) -> u8 {
    (b(r, 47, 1) == 0 && !(b(r, 48, 1) != 0 && b(r, 41, 1) != 0)) as u8
}
const FLO_FORMS: [u8; FLO_FIELDS.len()] = forms!(FLO_FIELDS, flo_rule);
fn read_flo(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, flo];
    paths[(FLO_FORMS[FLO_FIELDS.index(x.r)]) as usize](x)
}
// POPC: Mesa src/nouveau/compiler/nak/sm50.rs:1616-1640 and
// envytools/envydis/gm107.c:2016,2090,2136. B is register 20..27
// (gm107.c:205,218), cbuf offset 20..33 and bank 34..38 (:258,
// 302,305,309), or signed i20 bits 20..38 plus sign 56 (:253).
// Invert is bit 40; destination 0..7 and guard 16..19 (:203,371,
// 408-411). Ryujinx InstEmitBitfield.cs:107-125,185-192 counts
// the bits of B after optional bitwise inversion.
fn popc(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);

    x.guarded(InstructionKind::Popc(ops::Popc {
        dst: d,
        src: int_form_src(r),
        invert: b(r, 40, 1) != 0,
    }))
}
// EXIT condition0..4 and keeprefcount5: envytools gm107.c:1925,
// 426-454; Mesa sm50.rs:3101-3110 emits CC.T. Ryujinx
// InstEmitFlowControl.cs:153-186 tests CC before returning.
// Keep-reference-count mode stays Unknown. CSM selectors24..29
// (envytools gm107.c:451-456; Ryujinx Decoders/InstDecoders.cs:
// 102-107) decode here; the emitter matches the reference's
// never-taken EXIT for them. RLE/RGT are30/31 (:108-109).
fn exit(x: Word) -> Instruction {
    let r = x.r;

    x.guarded(InstructionKind::Exit(ops::Exit {
        cc_test: b(r, 0, 5) as u8,
    }))
}
const EXIT_FIELDS: Fields<1> = Fields([Field::bits(5, 1)]);
const fn exit_rule(r: u64) -> u8 {
    (b(r, 5, 1) == 0) as u8
}
const EXIT_FORMS: [u8; EXIT_FIELDS.len()] = forms!(EXIT_FIELDS, exit_rule);
fn read_exit(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, exit];
    paths[(EXIT_FORMS[EXIT_FIELDS.index(x.r)]) as usize](x)
}
/// IPA.SAT on PASS, MUL or CONSTANT (Ryujinx InstDecoders.cs:3265-3282,
/// InstIpa.Sat bit 51; InstEmitAttribute.cs:146-218 saturates the result
/// last, line 215; position MUL keeps B, lines 176-193, 208-217).
fn ipa_saturated(x: Word) -> Instruction {
    IPA_SATURATED[b(x.r, 54, 2) as usize](x)
}
/// PASS (mode 0 and 3), PASS.MUL by 1/W (mode 1) or CONSTANT (mode 2).
const IPA_SATURATED: [Decode; 4] = [
    ipa_pass::<true>,
    ipa_mul_w::<true>,
    ipa_constant::<true>,
    ipa_pass::<true>,
];
const IPA_PLAIN: [Decode; 4] = [
    ipa_pass::<false>,
    ipa_mul_w::<false>,
    ipa_constant::<false>,
    ipa_pass::<false>,
];
fn ipa_pass_of<const SATURATE: bool>(x: Word, freq: InterpFreq) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::IpaPass(ops::IpaPass {
        dst: Register(b(r, 0, 8) as u8),
        addr: b(r, 28, 10) as u16,
        inv_w: Source::Register(Register(b(r, 20, 8) as u8)),
        offset: Source::Register(Register(255)),
        loc: InterpLoc::Default,
        freq,
        saturate: SATURATE,
    }))
}
fn ipa_pass<const SATURATE: bool>(x: Word) -> Instruction {
    ipa_pass_of::<SATURATE>(x, InterpFreq::Pass)
}
fn ipa_constant<const SATURATE: bool>(x: Word) -> Instruction {
    ipa_pass_of::<SATURATE>(x, InterpFreq::Constant)
}
fn ipa_mul_w<const SATURATE: bool>(x: Word) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::IpaPassMulW(ops::IpaPassMulW {
        dst: Register(b(r, 0, 8) as u8),
        addr: b(r, 28, 10) as u16,
        inv_w: Source::Register(Register(b(r, 20, 8) as u8)),
        offset: Source::Register(Register(255)),
        loc: InterpLoc::Default,
        freq: InterpFreq::PassMulW,
        saturate: SATURATE,
    }))
}
// IPA.MUL.SAT: envytools/envydis/gm107.c:1953,859-871;
// Mesa nak/sm50.rs:2905-2934: dst0..7, address28..37, IDX38,
// inv_w20..27, offset39..46, predicate47..49 + inversion50,
// SAT51, location52..53, frequency54..55. Only immediate generic
// aligned addresses, default location, no predicate output, and
// canonical unused address/offset RZ slots are implemented.
// NAK's 0.into() at sm50.rs:2909 is SrcRef::Zero (ir.rs:701-709),
// encoded as RZ=255 (sm50.rs:204-206,255-265), matching gm107.c:1953.
// Ryujinx InstEmitAttribute.cs:167-175,208-217 reconstructs the
// perspective input, multiplies by inv_w, then saturates.
fn ipa_saturate(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);

    x.guarded(InstructionKind::IpaSaturate(ops::IpaSaturate {
        dst: d,
        addr: b(r, 28, 10) as u16,
        inv_w: Source::Register(Register(b(r, 20, 8) as u8)),
    }))
}
// IPA (Ryujinx InstTable.cs, MIT: 0xe000-0xe0ff; InstDecoders.cs:
// 3265-3282; Mesa nak/sm50.rs:2905-2934; envytools gm107.c:859-871,
// 1953): dst 0..7, B 20..27, address 28..37, IDX 38, offset 39..46,
// predicate 47..49 + inversion 50, SAT 51, location 52..53, mode
// 54..55. Implemented: immediate aligned Position or generic addresses,
// or FrontFacing a[0x3fc] (Ryujinx AttributeMap.cs:81, a flat Bool
// input, never saturated); address register RZ, no IDX, offset RZ,
// predicate PT not inverted, default location. Centroid, offset,
// indexed, SC (mode 3) and predicate forms have no lowering and stay
// Unknown. PASS/CONSTANT require the unused B RZ; MUL keeps its B
// register (InstEmitAttribute.cs:208-217).
fn ipa(x: Word) -> Instruction {
    IPA_PLAIN[b(x.r, 54, 2) as usize](x)
}

fn legacy_texcoord_pass(x: Word) -> Instruction {
    x.guarded(InstructionKind::LegacyVaryingLoad(ops::LegacyVaryingLoad {
        dst: Register(b(x.r, 0, 8) as u8),
        varying: LegacyVarying::decode(b(x.r, 28, 10) as u16)
            .expect("exact legacy TexCoord0 admission"),
        interpolation: LegacyInterpolation::Pass,
        inv_w: Register(255),
    }))
}

fn legacy_front_color_sc(x: Word) -> Instruction {
    x.guarded(InstructionKind::LegacyVaryingLoad(ops::LegacyVaryingLoad {
        dst: Register(b(x.r, 0, 8) as u8),
        varying: LegacyVarying::decode(b(x.r, 28, 10) as u16)
            .expect("exact FrontColorDiffuse admission"),
        interpolation: LegacyInterpolation::State,
        inv_w: Register(b(x.r, 20, 8) as u8),
    }))
}
/// 0 Unknown, 1 plain, 2 SAT PASS/CONSTANT, 3 SAT MUL.
const IPA_FIELDS0: Fields<8> = Fields([
    Field::ones(8, 8),
    Field::bits(38, 1),
    Field::ones(39, 8),
    Field::bits(47, 3),
    Field::bits(50, 2),
    Field::zero(52, 2),
    Field::bits(54, 2),
    Field::ones(20, 8),
]);
const fn ipa_rule0(r: u64) -> u8 {
    (b(r, 8, 8) == 255
        && b(r, 38, 1) == 0
        && b(r, 39, 8) == 255
        && b(r, 47, 3) == 7
        && b(r, 50, 1) == 0
        && b(r, 52, 2) == 0
        && b(r, 54, 2) <= 2
        && (b(r, 54, 2) == 1 || b(r, 20, 8) == 255)) as u8
        * (1 + b(r, 51, 1) as u8 * (1 + (b(r, 54, 2) == 1) as u8))
}
const IPA_FORMS0: [u8; IPA_FIELDS0.len()] = forms!(IPA_FIELDS0, ipa_rule0);
/// 0 Unknown, 1 Position a[0x70..0x7c], 2 generic a[0x80..0x27c],
/// 3 FrontFacing a[0x3fc]; word aligned.
const IPA_FIELDS1: Fields<1> = Fields([Field::bits(28, 10)]);
const fn ipa_rule1(r: u64) -> u8 {
    let a = b(r, 28, 10);
    if !a.is_multiple_of(4) {
        0
    } else if a >= 0x70 && a < 0x80 {
        1
    } else if a >= 0x80 && a < 0x280 {
        2
    } else {
        3 * (a == 0x3fc) as u8
    }
}
const IPA_FORMS1: [u8; IPA_FIELDS1.len()] = forms!(IPA_FIELDS1, ipa_rule1);
fn read_ipa(x: Word) -> Instruction {
    if matches!(
        x.r,
        0xe0c3_ffa8_0047_ff00
            | 0xe0c3_ffa8_4047_ff01
            | 0xe0c3_ffa8_8047_ff02
            | 0xe0c3_ffa8_c047_ff03
    ) {
        return legacy_front_color_sc(x);
    }
    if matches!(
        x.r,
        0xe003_ffb0_0ff7_ff00
            | 0xe003_ffb0_0ff7_ff04
            | 0xe003_ffb0_4ff7_ff01
            | 0xe003_ffb0_8ff7_ff02
            | 0xe003_ffb0_cff7_ff04
            | 0xe003_ffb0_4ff7_ff05
    ) {
        return legacy_texcoord_pass(x);
    }
    // Rows: IPA_FORMS0; columns: IPA_FORMS1.
    let paths: [Decode; 16] = [
        unknown,
        unknown,
        unknown,
        unknown,
        unknown,
        ipa,
        ipa,
        ipa,
        unknown,
        ipa_saturated,
        ipa_saturated,
        unknown,
        unknown,
        ipa_saturated,
        ipa_saturate,
        unknown,
    ];
    paths[(IPA_FORMS0[IPA_FIELDS0.index(x.r)] * 4 + IPA_FORMS1[IPA_FIELDS1.index(x.r)]) as usize](x)
}
// Mesa's OpNop encodes as 0x50b0 (sm50.rs:3178-3184). Nothing else may
// be mapped to Nop: MUFU (0x5080), BRA (0xe240), KILL (0xe330),
// FSWZADD (0x50f8) and POPC (0x5c08) have data or control effects.
fn nop(x: Word) -> Instruction {
    x.guarded(InstructionKind::Nop(ops::Nop))
}
const NOP_PACKET_REGIONS: [(u64, u64); 7] = [
    (0xfff9_ffff_fff7_e0ff, 0x50b1_0000_0007_0000),
    (0xfff9_ffff_fff7_e0ff, 0x50b1_28de_18a7_00f2),
    (0xfff9_ffff_fff7_e0ff, 0x50b1_6f7a_8a07_00b2),
    (0xfff9_ffff_fff7_e0ff, 0x50b1_953c_3f97_8072),
    (0xfff9_ffff_fff7_e0ff, 0x50b1_9607_1787_0070),
    (0xfff9_ffff_fff7_e2ff, 0x50b1_87f3_87f7_8061),
    (0xfff9_ffff_fff7_e8ff, 0x50b1_f38f_1097_40df),
];
const NOP_TAIL_PACKET_REGIONS: [(u64, u64); 14] = [
    (0xfffb_ffff_fff7_e0ff, 0x50b2_0000_0007_0000),
    (0xfffb_ffff_fff7_e0ff, 0x50b2_28de_18a7_00f2),
    (0xfffb_ffff_fff7_e0ff, 0x50b2_6f7a_8a07_00b2),
    (0xfffb_ffff_fff7_e0ff, 0x50b2_953c_3f97_8072),
    (0xfffb_ffff_fff7_e0ff, 0x50b2_9607_1787_0070),
    (0xfffb_ffff_fff7_e2ff, 0x50b2_87f3_87f7_8061),
    (0xfffb_ffff_fff7_e8ff, 0x50b2_f38f_1097_40df),
    (0xffff_ffff_fff7_e0ff, 0x50b4_0000_0007_0000),
    (0xffff_ffff_fff7_e0ff, 0x50b4_28de_18a7_00f2),
    (0xffff_ffff_fff7_e0ff, 0x50b4_6f7a_8a07_00b2),
    (0xffff_ffff_fff7_e0ff, 0x50b4_953c_3f97_8072),
    (0xffff_ffff_fff7_e0ff, 0x50b4_9607_1787_0070),
    (0xffff_ffff_fff7_e2ff, 0x50b4_87f3_87f7_8061),
    (0xffff_ffff_fff7_e8ff, 0x50b4_f38f_1097_40df),
];
fn read_nop(x: Word) -> Instruction {
    if b(x.r, 48, 3) == 0
        || NOP_PACKET_REGIONS
            .iter()
            .any(|&(mask, value)| x.r & mask == value)
        || NOP_TAIL_PACKET_REGIONS
            .iter()
            .any(|&(mask, value)| x.r & mask == value)
    {
        nop(x)
    } else {
        unknown(x)
    }
}
// VOTE.VTG row: envytools envydis/gm107.c:2053, opcode 0x50e0,
// mask 0xfff8 (bits51..63). Mode48..49 R/A/RA, with 3 OOPS:
// :1633-1637. Guard16..18/invert19: :371,408-411, decoded above.
// Immediate20..47 follows the primary reference, Ryujinx
// Decoders/InstDecoders.cs:5224-5231 (Imm28); envytools :248,284
// (U24_20) names only bits20..43, and real programs set bits44..47.
// Neither source assigns bits0..15 or 50: require zero rather than
// inventing fields. Mode and immediate remain visible in IR/display.
// Only this row has operator-authorized no-effect semantics matching
// Ryujinx Decoders/InstTable.cs:320 and Instructions/InstEmit.cs:
// 330-336. It remains distinct from Nop and ordinary VOTE.
fn vote_vtg(x: Word) -> Instruction {
    let r = x.r;

    x.guarded(InstructionKind::VoteVtg(ops::VoteVtg {
        mode: [
            VoteVtgMode::R,
            VoteVtgMode::A,
            VoteVtgMode::Ra,
            VoteVtgMode::Ra,
        ][b(r, 48, 2) as usize],
        immediate: U28::low_bits(r >> 20),
    }))
}
const VOTE_VTG_FIELDS: Fields<3> =
    Fields([Field::bits(48, 2), Field::zero(0, 16), Field::bits(50, 1)]);
const fn vote_vtg_rule(r: u64) -> u8 {
    (b(r, 48, 2) <= 2 && b(r, 0, 16) == 0 && b(r, 50, 1) == 0) as u8
}
const VOTE_VTG_FORMS: [u8; VOTE_VTG_FIELDS.len()] = forms!(VOTE_VTG_FIELDS, vote_vtg_rule);
fn read_vote_vtg(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, vote_vtg];
    paths[(VOTE_VTG_FORMS[VOTE_VTG_FIELDS.index(x.r)]) as usize](x)
}
// Mesa OpVote encodes opcode 0x50d8, ballot REG_00, vote PRED45,
// source T(pred39) with negation at bit 42, and mode T(50d8_0)
// at bits 48..50 (mesa/src/nouveau/compiler/nak/sm50.rs:3227-3247;
// mesa/src/nouveau/compiler/nak/ir.rs:7531-7544; envytools/envydis/gm107.c:2054).
fn vote(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);

    x.guarded(InstructionKind::Vote(ops::Vote {
        op: [VoteOp::All, VoteOp::Any, VoteOp::Eq, VoteOp::Eq][b(r, 48, 2) as usize],
        ballot: d,
        vote: b(r, 45, 3) as u8,
        pred: Predicate {
            register: b(r, 39, 3) as u8,
            inverted: b(r, 42, 1) != 0,
        },
    }))
}
const VOTE_FIELDS: Fields<1> = Fields([Field::bits(48, 2)]);
const fn vote_rule(r: u64) -> u8 {
    (b(r, 48, 2) <= 2) as u8
}
const VOTE_FORMS: [u8; VOTE_FIELDS.len()] = forms!(VOTE_FIELDS, vote_rule);
fn read_vote(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, vote];
    paths[(VOTE_FORMS[VOTE_FIELDS.index(x.r)]) as usize](x)
}
// Mesa OpBra encodes as 0xe240 with a relative target
// (sm50.rs:3005-3027).
// BRA's condition-code test is T(f0f8_0), bits 0..4
// (envytools/envydis/gm107.c:1944-1945). Preserve every test in IR;
// the emitter must reject tests whose four CC flags are not modeled.
fn branch(x: Word) -> Instruction {
    let r = x.r;
    let address = x.address;

    x.guarded({
        let cc_test = b(r, 0, 5) as u8;
        match BranchTarget::named(address, signed(r, 20, 24)) {
            Named::Slot(target) => InstructionKind::Branch(ops::Branch { target, cc_test }),
            Named::Misaligned(target) => {
                InstructionKind::MisalignedBranch(ops::MisalignedBranch { target, cc_test })
            }
            Named::Outside(target) => {
                InstructionKind::OutsideBranch(ops::OutsideBranch { target, cc_test })
            }
        }
    })
}
const BRANCH_FIELDS: Fields<1> = Fields([Field::bits(5, 1)]);
const fn branch_rule(r: u64) -> u8 {
    (b(r, 5, 1) == 0) as u8
}
const BRANCH_FORMS: [u8; BRANCH_FIELDS.len()] = forms!(BRANCH_FIELDS, branch_rule);
fn read_branch(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, branch];
    paths[(BRANCH_FORMS[BRANCH_FIELDS.index(x.r)]) as usize](x)
}
// The zero word is padding. Any other word of opcode 0x0000 has no
// modeled meaning.
fn padding(x: Word) -> Instruction {
    x.guarded(InstructionKind::Padding(ops::Padding))
}
const PADDING_FIELDS: Fields<1> = Fields([Field::zero(0, 48)]);
const fn padding_rule(r: u64) -> u8 {
    (b(r, 0, 48) != 0) as u8
}
const PADDING_FORMS: [u8; PADDING_FIELDS.len()] = forms!(PADDING_FIELDS, padding_rule);
fn read_padding(x: Word) -> Instruction {
    let paths: [Decode; 2] = [padding, unknown];
    paths[(PADDING_FORMS[PADDING_FIELDS.index(x.r)]) as usize](x)
}
// Relative CAL: envytools gm107.c:1941,297-299 defines opcode mask
// 0xfff0000000000020 and a signed 24-bit target at bit 20, relative to
// address + 8. Bit 5 set has no CAL meaning.
fn call(x: Word) -> Instruction {
    let r = x.r;
    let address = x.address;

    x.guarded(match BranchTarget::named(address, signed(r, 20, 24)) {
        Named::Slot(target) => InstructionKind::Call(ops::Call { target }),
        Named::Misaligned(target) => {
            InstructionKind::MisalignedCall(ops::MisalignedCall { target })
        }
        Named::Outside(target) => InstructionKind::OutsideCall(ops::OutsideCall { target }),
    })
}
const CALL_FIELDS: Fields<1> = Fields([Field::bits(5, 1)]);
const fn call_rule(r: u64) -> u8 {
    (b(r, 5, 1) == 0) as u8
}
const CALL_FORMS: [u8; CALL_FIELDS.len()] = forms!(CALL_FIELDS, call_rule);
fn read_call(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, call];
    paths[(CALL_FORMS[CALL_FIELDS.index(x.r)]) as usize](x)
}
// Mesa OpSSy (0xe290), OpPBk (0xe2a0) and OpPCnt (0xe2b0) push the
// target that OpSync (0xf0f8), OpBrk (0xe340) and OpCont (0xe350) jump
// to (sm50.rs:3032-3080; from_nir.rs:4215-4240 emits SYNC/BRK/CONT
// with the pushed label as target). Mesa sets the condition field
// 0..5 to 0xF (always). BRK/CONT also accept the modeled CC tests
// from envytools gm107.c:1920-1921,426-458. Preserve the test through
// stack resolution (Ryujinx InstEmitFlowControl.cs:254-262,295-312).
// SSY/PBK/PCNT have no predicate or condition test; bit 5 must be
// clear (envytools gm107.c:1933-1935, mask 0xfff0000000000020).
fn schedule(x: Word) -> Instruction {
    let r = x.r;
    let address = x.address;

    x.unguarded(match BranchTarget::named(address, signed(r, 20, 24)) {
        Named::Slot(target) => InstructionKind::Schedule(ops::Schedule { target }),
        Named::Misaligned(target) => {
            InstructionKind::MisalignedSchedule(ops::MisalignedSchedule { target })
        }
        Named::Outside(target) => InstructionKind::OutsideSchedule(ops::OutsideSchedule { target }),
    })
}
// Hybrid conveyor bundle B10006b0187fe13c6 contains the complete
// source-locked e290..e29f alias run with otherwise identical bits. The
// permitted envytools SSY row masks the low opcode nibble, and the oracle
// identifies every word as SSY. Admit only these whole words; the generic
// decoder below keeps the established canonical e290 field boundary.
const QUALIFIED_SSY_ALIAS_WORDS: [u64; 16] = [
    0xe290_0000_0000_0020,
    0xe291_0000_0000_0020,
    0xe292_0000_0000_0020,
    0xe293_0000_0000_0020,
    0xe294_0000_0000_0020,
    0xe295_0000_0000_0020,
    0xe296_0000_0000_0020,
    0xe297_0000_0000_0020,
    0xe298_0000_0000_0020,
    0xe299_0000_0000_0020,
    0xe29a_0000_0000_0020,
    0xe29b_0000_0000_0020,
    0xe29c_0000_0000_0020,
    0xe29d_0000_0000_0020,
    0xe29e_0000_0000_0020,
    0xe29f_0000_0000_0020,
];
const SCHEDULE_FIELDS: Fields<1> = Fields([Field::bits(5, 1)]);
const fn schedule_rule(r: u64) -> u8 {
    (b(r, 5, 1) == 0) as u8
}
const SCHEDULE_FORMS: [u8; SCHEDULE_FIELDS.len()] = forms!(SCHEDULE_FIELDS, schedule_rule);
fn read_schedule(x: Word) -> Instruction {
    if QUALIFIED_SSY_ALIAS_WORDS.contains(&x.r) {
        return schedule(x);
    }
    if b(x.r, 48, 4) != 0 {
        return unknown(x);
    }
    let paths: [Decode; 2] = [unknown, schedule];
    paths[(SCHEDULE_FORMS[SCHEDULE_FIELDS.index(x.r)]) as usize](x)
}

/// PBK 0xe2a0 / BRK 0xe340 clear opcode bit 52; PCNT 0xe2b0 / CONT
/// 0xe350 set it.
const STACKS: [TargetStack; 2] = [TargetStack::Break, TargetStack::Continue];
fn push_target(x: Word) -> Instruction {
    let r = x.r;
    let address = x.address;

    x.unguarded({
        let stack = STACKS[b(r, 52, 1) as usize];
        match BranchTarget::named(address, signed(r, 20, 24)) {
            Named::Slot(target) => InstructionKind::PushTarget(ops::PushTarget { target, stack }),
            Named::Misaligned(target) => {
                InstructionKind::MisalignedPushTarget(ops::MisalignedPushTarget { target, stack })
            }
            Named::Outside(target) => {
                InstructionKind::OutsidePushTarget(ops::OutsidePushTarget { target, stack })
            }
        }
    })
}
fn cbuf_push_target(x: Word) -> Instruction {
    let r = x.r;
    if matches!(r >> 52, 0xe2a | 0xe2b) && r & 0x0000_ffff_ffff_ffff == 0x20 {
        x.unguarded(InstructionKind::CbufPushTarget(ops::CbufPushTarget {
            binding: b(r, 36, 5) as u8,
            offset: b(r, 20, 16) as u16 as i16,
            stack: STACKS[b(r, 52, 1) as usize],
        }))
    } else {
        unknown(x)
    }
}
const PUSH_TARGET_FIELDS: Fields<2> = Fields([Field::bits(5, 1), Field::bits(48, 4)]);
const fn push_target_rule(r: u64) -> u8 {
    if b(r, 5, 1) == 0 && b(r, 48, 4) == 0 {
        1
    } else if b(r, 5, 1) == 1 {
        2
    } else {
        0
    }
}
const PUSH_TARGET_FORMS: [u8; PUSH_TARGET_FIELDS.len()] =
    forms!(PUSH_TARGET_FIELDS, push_target_rule);
fn read_push_target(x: Word) -> Instruction {
    let paths: [Decode; 3] = [unknown, push_target, cbuf_push_target];
    paths[(PUSH_TARGET_FORMS[PUSH_TARGET_FIELDS.index(x.r)]) as usize](x)
}

fn read_pcnt_alias(x: Word) -> Instruction {
    match x.r & 0x0000_ffff_ffff_ffff {
        0 => push_target(x),
        0x20 => cbuf_push_target(x),
        _ => unknown(x),
    }
}

fn pop_jump(x: Word) -> Instruction {
    let r = x.r;

    x.guarded(InstructionKind::PopJump(ops::PopJump {
        cc_test: b(r, 0, 5) as u8,
        stack: STACKS[b(r, 52, 1) as usize],
    }))
}
const POP_JUMP_FIELDS: Fields<1> = Fields([Field::bits(0, 5)]);
const fn pop_jump_rule(r: u64) -> u8 {
    (b(r, 0, 5) <= 23 || b(r, 0, 5) >= 30) as u8
}
const POP_JUMP_FORMS: [u8; POP_JUMP_FIELDS.len()] = forms!(POP_JUMP_FIELDS, pop_jump_rule);
fn read_pop_jump(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, pop_jump];
    paths[(POP_JUMP_FORMS[POP_JUMP_FIELDS.index(x.r)]) as usize](x)
}

// SYNC (Ryujinx InstTable.cs:294, MIT: 0xf0f8-0xf0ff; InstDecoders.cs:
// 4823-4830): guard 16..19, CC test 0..4 (envytools gm107.c:426-458).
// InstEmitFlowControl.cs:232-237,254-262,279-313 branches to the pushed
// SSY target when the guard and GetCondition (InstEmitConditionCode.cs:
// 52-83) hold. As for the other pop jumps, the CSM tests 24..29 have no
// lowering and stay Unknown; nvdisasm ignores bits 5..15 and 20..47.
fn sync(x: Word) -> Instruction {
    x.guarded(InstructionKind::PopJump(ops::PopJump {
        stack: TargetStack::Sync,
        cc_test: b(x.r, 0, 5) as u8,
    }))
}
fn read_sync(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, sync];
    paths[(POP_JUMP_FORMS[POP_JUMP_FIELDS.index(x.r)]) as usize](x)
}
// Mesa sm50.rs:3172-3174: opcode48..63=e330, CC0..4=0xf;
// sm50.rs:225-235: predicate16..19. Refuse every unnamed bit.
// from_nir.rs:3035-3050 maps demote to KIL, so it falls through.
fn kill(x: Word) -> Instruction {
    x.guarded(InstructionKind::Kill(ops::Kill))
}
const KILL_FIELDS: Fields<3> = Fields([Field::bits(0, 5), Field::zero(5, 11), Field::zero(20, 28)]);
const fn kill_rule(r: u64) -> u8 {
    (b(r, 0, 5) == 0xf && b(r, 5, 11) == 0 && b(r, 20, 28) == 0) as u8
}
const KILL_FORMS: [u8; KILL_FIELDS.len()] = forms!(KILL_FIELDS, kill_rule);
fn read_kill(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, kill];
    paths[(KILL_FORMS[KILL_FIELDS.index(x.r)]) as usize](x)
}
// S2R: Mesa src/nouveau/compiler/nak/sm50.rs:3220-3224:
// opcode f0c8, dst0..7, index20..27. Pred16/not19 is common
// (envytools envydis/gm107.c:371,408-419). Reject unnamed bits.
// Indices: gm107.c:121,135-137,147-157,173-177;
// Ryujinx Decoders/InstDecoders.cs:148,162-164,174-184,196-200.
// Explicit InstEmitMove.cs:78-163 cases plus reference-matched
// wscalefactor_xy/z (gm107.c:148-149): zero in every stage through
// Ryujinx Instructions/InstEmitMove.cs:76-167 (default165-167).
// This is project policy, not hardware-proven packing/readback;
// see docs/reference-matched.md and provenance/0024-s2r-wscale.md.
// Other unimplemented indices, including clocks, stay Unknown.
fn s2r(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);

    x.guarded(InstructionKind::S2R(ops::S2R {
        dst: d,
        idx: b(r, 20, 8) as u8,
    }))
}
const S2R_FIELDS: Fields<3> = Fields([Field::bits(20, 8), Field::zero(8, 8), Field::zero(28, 20)]);
const fn s2r_rule(r: u64) -> u8 {
    ((b(r, 20, 8) == 0
        || (b(r, 20, 8) >= 0x11 && b(r, 20, 8) <= 0x13)
        || (b(r, 20, 8) >= 0x1d && b(r, 20, 8) <= 0x23)
        || (b(r, 20, 8) >= 0x25 && b(r, 20, 8) <= 0x27)
        || (b(r, 20, 8) >= 0x38 && b(r, 20, 8) <= 0x3c))
        && b(r, 8, 8) == 0
        && b(r, 28, 20) == 0) as u8
}
const S2R_FORMS: [u8; S2R_FIELDS.len()] = forms!(S2R_FIELDS, s2r_rule);
fn read_s2r(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, s2r];
    paths[(S2R_FORMS[S2R_FIELDS.index(x.r)]) as usize](x)
}
// ISET gm107.c:2029,2100,2147: compare 49..52 (1445-1453),
// signed 48 (1365-1367), .X 43, .BF 44, .CC 47, accumulator
// 39..42/negate 42, combine 45..47 (1430-1434). Reserved combine
// stays Unknown. Integer immediate sign is bit 56 (Mesa
// sm50.rs:335-346). Semantics: Ryujinx InstEmitIntegerComparison.cs:133-167.
fn iset(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);
    let set_op = SET_OPS[b(r, 45, 2) as usize];
    x.guarded(InstructionKind::ISet(ops::ISet {
        dst: d,
        lhs: l,
        rhs: int_form_src(r),
        compare: cmp(b(r, 49, 3)),
        signedness: signedness(r),
        bool_float: b(r, 44, 1) != 0,
        cc: b(r, 47, 1) != 0,
        x: b(r, 43, 1) != 0,
        accum: b(r, 39, 3) as u8,
        accum_negate: b(r, 42, 1) != 0,
        set_op,
    }))
}
const ISET_FIELDS: Fields<1> = Fields([Field::bits(45, 2)]);
const fn iset_rule(r: u64) -> u8 {
    (b(r, 45, 2) < 3) as u8
}
const ISET_FORMS: [u8; ISET_FIELDS.len()] = forms!(ISET_FIELDS, iset_rule);
fn read_iset(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, iset];
    paths[(ISET_FORMS[ISET_FIELDS.index(x.r)]) as usize](x)
}
// P2R: envytools gm107.c:1992,2066,2112. D0..7/A8..15
// (:203-205), B20..27 or cbuf20..38 (:258,302-309), or signed
// imm20..38 + sign56 (:253). PR/CC40 and byte41..42 (:1194-1204).
// Ryujinx InstEmitPredicate.cs:86-114 merges only the selected byte;
// undefined fields stay Unknown, including register-form bits28..38.
const P2R_REGISTER_FIELDS: Fields<3> =
    Fields([Field::bits(39, 1), Field::zero(43, 8), Field::zero(28, 11)]);
const fn p2r_register_rule(r: u64) -> u8 {
    (b(r, 39, 1) == 0 && b(r, 43, 8) == 0 && b(r, 28, 11) == 0) as u8
}
const P2R_REGISTER_FORMS: [u8; P2R_REGISTER_FIELDS.len()] =
    forms!(P2R_REGISTER_FIELDS, p2r_register_rule);
fn read_p2r_register(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, p2r];
    paths[(P2R_REGISTER_FORMS[P2R_REGISTER_FIELDS.index(x.r)]) as usize](x)
}
fn p2r(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::P2R(ops::P2R {
        dst: d,
        src: l,
        mask: int_form_src(r),
        byte: b(r, 41, 2) as u8,
        cc: b(r, 40, 1) != 0,
    }))
}
const P2R_FIELDS: Fields<2> = Fields([Field::bits(39, 1), Field::zero(43, 8)]);
const fn p2r_rule(r: u64) -> u8 {
    (b(r, 39, 1) == 0 && b(r, 43, 8) == 0) as u8
}
const P2R_FORMS: [u8; P2R_FIELDS.len()] = forms!(P2R_FIELDS, p2r_rule);
fn read_p2r(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, p2r];
    paths[(P2R_FORMS[P2R_FIELDS.index(x.r)]) as usize](x)
}
// VSETP (0x50f4 is B register mode, never FSWZADD or NOP):
// envytools gm107.c:2052,1588-1628. A select36..38/sign48,
// B select28..30/sign49, register flag50; A8..15, B20..27 or
// imm16 20..35. A byte selectors: :1538-1542,1105-1109;
// U32/S32 selector6: Ryujinx InstDecoders.cs:5261-5262,717-732.
// Compare43..44 +47 (:1588-1596), boolean45..46 (:1430-1434),
// accumulator39..41/not42 (:412-415), D3..5/Dinv0..2 (:363-364).
// Ryujinx InstEmitVideoMinMax.cs:150-164 has no semantics for
// mixed signedness with a 32-bit operand; keep those forms Unknown.
fn video_setp_of(x: Word, rhs: Source, select: u32) -> Instruction {
    let r = x.r;
    let l = Register(b(r, 8, 8) as u8);
    let set_op = SET_OPS[b(r, 45, 2) as usize];
    x.guarded({
        InstructionKind::VideoSetP(ops::VideoSetP {
            dst: [b(r, 3, 3) as u8, b(r, 0, 3) as u8],
            lhs: l,
            rhs,
            select: [
                (b(r, 36, 3) | (b(r, 48, 1) << 3)) as u8,
                (select | (b(r, 49, 1) << 3)) as u8,
            ],
            compare: cmp(b(r, 43, 2) | (b(r, 47, 1) << 2)),
            // Either source's signed bit (48, 49) makes the compare
            // signed: equal bits compare with that signedness, and the
            // guard above admits mixed bits only for two extended small
            // operands, which compare signed (Ryujinx
            // InstEmitVideoMinMax.cs:147-169).
            signedness: [Signedness::Unsigned, Signedness::Signed]
                [(b(r, 48, 1) | b(r, 49, 1)) as usize],
            set_op,
            accum: Predicate {
                register: b(r, 39, 3) as u8,
                inverted: b(r, 42, 1) != 0,
            },
        })
    })
}
fn video_setp_register(x: Word) -> Instruction {
    video_setp_of(x, register_b(x.r), b(x.r, 28, 3))
}
fn video_setp_immediate(x: Word) -> Instruction {
    video_setp_of(x, video_immediate_b(x.r), 6)
}
const VIDEO_SETP_FIELDS0: Fields<2> = Fields([Field::zero(6, 2), Field::bits(45, 2)]);
const fn video_setp_rule0(r: u64) -> u8 {
    (b(r, 6, 2) == 0 && b(r, 45, 2) < 3) as u8 * 3
}
const VIDEO_SETP_FORMS0: [u8; VIDEO_SETP_FIELDS0.len()] =
    forms!(VIDEO_SETP_FIELDS0, video_setp_rule0);
const VIDEO_SETP_FIELDS1: Fields<6> = Fields([
    Field::bits(36, 3),
    Field::bits(50, 1),
    Field::bits(28, 3),
    Field::zero(31, 5),
    Field::bits(48, 1),
    Field::bits(49, 1),
]);
const fn video_setp_rule1(r: u64) -> u8 {
    (b(r, 36, 3) != 7
        && (b(r, 50, 1) == 0 || (b(r, 28, 3) != 7 && b(r, 31, 5) == 0))
        && (b(r, 48, 1) == b(r, 49, 1)
            || (b(r, 36, 3) != 6 && (b(r, 50, 1) == 0 || b(r, 28, 3) != 6)))) as u8
        * (1 + b(r, 50, 1) as u8)
}
const VIDEO_SETP_FORMS1: [u8; VIDEO_SETP_FIELDS1.len()] =
    forms!(VIDEO_SETP_FIELDS1, video_setp_rule1);
fn read_video_setp(x: Word) -> Instruction {
    let paths: [Decode; 3] = [unknown, video_setp_immediate, video_setp_register];
    paths[(VIDEO_SETP_FORMS0[VIDEO_SETP_FIELDS0.index(x.r)]
        & VIDEO_SETP_FORMS1[VIDEO_SETP_FIELDS1.index(x.r)]) as usize](x)
}
// R2P (envytools gm107.c:1991,2065,2111): bits 0..6 select P0..P6,
// bits 41..42 select the byte, and bit 40 selects the four condition
// code flags Z/N/C/V instead (Ryujinx Decoders/InstDecoders.cs InstR2pR/
// I/C Ccpr bit 40; Instructions/InstEmitMove.cs:254-270, MIT).
// PT is not selected by this seven-bit mask (gm107.c:1194-1207).
// R2P: source 8..16, mask operand by form at 20, byte 41..43.
const FLAG_BANKS: [FlagBank; 2] = [FlagBank::Predicates, FlagBank::ConditionCodes];
fn r2p(x: Word) -> Instruction {
    let r = x.r;
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::R2P(ops::R2P {
        src: l,
        mask: int_form_src(r),
        byte: b(r, 41, 2) as u8,
        bank: FLAG_BANKS[b(r, 40, 1) as usize],
    }))
}
// ISETP (Ryujinx InstTable.cs Isetp R/I/C, MIT; InstDecoders.cs
// InstIsetpR/I/C; envytools gm107.c:2028,2099,2146): dst 3..6, the
// complement destination 0..3 (EmitIsetp,
// InstEmitIntegerComparison.cs:170-192), accumulator 39..42 with negate 42,
// .X 43 extends relational comparisons through the previous Z/C
// condition-code state, set_op 45..47, signed 48, compare 49..52.
// 0x5b00 is XMAD and 0x5a00 IMAD (gm107.c:2031,2033), not compares.
fn isetp(x: Word) -> Instruction {
    let r = x.r;
    let l = Register(b(r, 8, 8) as u8);
    let set_op = SET_OPS[b(r, 45, 2) as usize];
    x.guarded({
        InstructionKind::ISetP(ops::ISetP {
            dst: [b(r, 3, 3) as u8, b(r, 0, 3) as u8],
            lhs: l,
            rhs: int_form_src(r),
            compare: cmp(b(r, 49, 3)),
            signedness: signedness(r),
            mode: [IntegerCompareMode::Ordinary, IntegerCompareMode::Extended]
                [b(r, 43, 1) as usize],
            set_op,
            accum: b(r, 39, 3) as u8,
            accum_negate: b(r, 42, 1) == 1,
            abs: [false; 2],
            negate: [false; 2],
        })
    })
}
const ISETP_FIELDS: Fields<1> = Fields([Field::bits(45, 2)]);
const fn isetp_rule(r: u64) -> u8 {
    (b(r, 45, 2) < 3) as u8
}
const ISETP_FORMS: [u8; ISETP_FIELDS.len()] = forms!(ISETP_FIELDS, isetp_rule);
fn read_isetp(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, isetp];
    paths[(ISETP_FORMS[ISETP_FIELDS.index(x.r)]) as usize](x)
}
// Binary64 operands, Ryujinx InstEmitHelper.cs:37-78: a register pair
// (the high word at the register with bit 0 set), a cbuf word and the
// next (C34_RZ_O14_20), or a D20 immediate (20..38, sign 56) as the
// high word over a zero low word.
/// The register pair at `start`.
fn pair(r: u64, start: u32) -> RegisterPair {
    RegisterPair(Register(b(r, start, 8) as u8))
}
/// Binary64 B by the opcode's top nibble: 0x4 constant buffer, 0x3 the
/// high word of an f20 immediate, every other nibble a register pair.
const DOUBLE_FORMS: [fn(u64) -> DoubleSource; 16] = {
    let mut t = [double_register_b as fn(u64) -> DoubleSource; 16];
    t[3] = double_immediate_b;
    t[4] = double_constant_b;
    t
};
fn double_register_b(r: u64) -> DoubleSource {
    DoubleSource::RegisterPair(pair(r, 20))
}
fn double_immediate_b(r: u64) -> DoubleSource {
    DoubleSource::Immediate((b(r, 20, 19) << 12) | (b(r, 56, 1) << 31))
}
fn double_constant_b(r: u64) -> DoubleSource {
    DoubleSource::ConstantBuffer {
        binding: b(r, 34, 5) as u8,
        offset: (b(r, 20, 14) * 4) as u16,
    }
}
fn double_b(r: u64) -> DoubleSource {
    DOUBLE_FORMS[b(r, 60, 4) as usize](r)
}
const DOUBLE_PRODUCT_SIGN: [DoubleProductSign; 2] =
    [DoubleProductSign::Positive, DoubleProductSign::Negated];
// DFMA: Ryujinx InstDecoders.cs:1292-1356 (InstDfmaR/I/C/Rc): D 0..7,
// A 8..15, CC 47, NegA 48 (the product), NegC 49, Round 50..51; the R
// form's B 20..27 and C 39..46, the I and C forms' B by form with C at
// 39..46, the RC form's B at 39..46 and C the constant (envytools
// gm107.c:2027,2045,2098,2145). InstEmitFloatArithmetic.cs:53-84 fuses
// in binary64. CC47 is closed FloatFlags state. Directed rounding stays
// Unknown. nvdisasm SM52 ignores the R form's 28..38.
const DFMA_FIELDS: Fields<1> = Fields([Field::zero(50, 2)]);
const fn dfma_rule(r: u64) -> u8 {
    (b(r, 50, 2) == 0) as u8
}
const DFMA_FORMS: [u8; DFMA_FIELDS.len()] = forms!(DFMA_FIELDS, dfma_rule);
/// The multiplier and addend by bits 59..61: 0x5b R (3), 0x4b C (1),
/// 0x36/0x37 I (6), 0x53 RC (2).
const DFMA_OPERANDS: [fn(u64) -> [DoubleSource; 2]; 8] = {
    let mut t = [dfma_register as fn(u64) -> [DoubleSource; 2]; 8];
    t[1] = dfma_by_form;
    t[2] = dfma_constant_addend;
    t[6] = dfma_by_form;
    t
};
fn dfma_register(r: u64) -> [DoubleSource; 2] {
    [
        double_register_b(r),
        DoubleSource::RegisterPair(pair(r, 39)),
    ]
}
fn dfma_by_form(r: u64) -> [DoubleSource; 2] {
    [double_b(r), DoubleSource::RegisterPair(pair(r, 39))]
}
fn dfma_constant_addend(r: u64) -> [DoubleSource; 2] {
    [
        DoubleSource::RegisterPair(pair(r, 39)),
        double_constant_b(r),
    ]
}
fn dfma(x: Word) -> Instruction {
    let r = x.r;
    let [multiplier, addend] = DFMA_OPERANDS[b(r, 59, 3) as usize](r);
    x.guarded(InstructionKind::DoubleFma(ops::DoubleFma {
        dst: pair(r, 0),
        lhs: pair(r, 8),
        multiplier,
        addend,
        product_sign: DOUBLE_PRODUCT_SIGN[b(r, 48, 1) as usize],
        addend_sign: [DoubleAddendSign::Added, DoubleAddendSign::Subtracted][b(r, 49, 1) as usize],
        flags: [FloatFlags::Kept, FloatFlags::Written][b(r, 47, 1) as usize],
    }))
}
fn read_dfma(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, dfma];
    paths[DFMA_FORMS[DFMA_FIELDS.index(x.r)] as usize](x)
}
// DMUL: Ryujinx InstDecoders.cs:1413-1455 (InstDmulR/I/C): D 0..7, A
// 8..15, B by form, Round 39..40, CC 47, NegA 48; envytools
// gm107.c:2002,2076,2122 agree. InstEmitFloatArithmetic.cs:86-114
// negates A and multiplies in binary64. CC47 is closed FloatFlags state for
// the six source-locked DADD/DMUL words; every other CC word and directed
// rounding stay Unknown. nvdisasm SM52 ignores the
// unassigned 41..46 and 49..50 and the register form's 28..38.
const DMUL_FIELDS: Fields<1> = Fields([Field::zero(39, 2)]);
const fn dmul_rule(r: u64) -> u8 {
    (b(r, 39, 2) == 0) as u8
}
const fn double_rn_cc_word(r: u64) -> bool {
    matches!(
        r,
        0x4c76_fc03_2b36_9161
            | 0x3872_9a45_25e5_1fed
            | 0x3971_d03d_ef95_759d
            | 0x3976_fc48_5c55_51ee
            | 0x5c86_da71_bf40_b4f2
            | 0x3986_c87c_3b2e_5d0d
            | 0x5c80_8000_0007_0000
            | 0x5c81_8000_0007_0000
            | 0x5c80_c82a_d4c7_25a4
            | 0x5c81_c82a_d4c7_25a4
            | 0x5c80_dc03_b1f7_0950
    )
}
const DMUL_FORMS: [u8; DMUL_FIELDS.len()] = forms!(DMUL_FIELDS, dmul_rule);
fn dmul(x: Word) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::DoubleMultiply(ops::DoubleMultiply {
        dst: pair(r, 0),
        lhs: pair(r, 8),
        rhs: double_b(r),
        sign: DOUBLE_PRODUCT_SIGN[b(r, 48, 1) as usize],
        flags: [FloatFlags::Kept, FloatFlags::Written][b(r, 47, 1) as usize],
    }))
}
fn read_dmul(x: Word) -> Instruction {
    if b(x.r, 47, 1) == 1 && !double_rn_cc_word(x.r) {
        return unknown(x);
    }
    let paths: [Decode; 2] = [unknown, dmul];
    paths[DMUL_FORMS[DMUL_FIELDS.index(x.r)] as usize](x)
}
// DADD: Ryujinx InstDecoders.cs:1228-1284 (InstDaddR/I/C): D 0..7, A
// 8..15, B by form, Round 39..40, NegB 45, AbsA 46, CC 47, NegA 48,
// AbsB 49; envytools gm107.c:2003,2077,2123 agree. InstEmitFloat
// Arithmetic.cs:12-51 applies abs, then negation, and adds in binary64.
// CC47 is closed FloatFlags state for the exact source-locked words and
// packets below; every other CC word and directed rounding stay Unknown.
// nvdisasm SM52 ignores unassigned 41..44 and 50 and the register form's
// 28..38.
const DADD_FIELDS: Fields<1> = Fields([Field::zero(39, 2)]);
const fn dadd_rule(r: u64) -> u8 {
    (b(r, 39, 2) == 0) as u8
}
const DADD_FORMS: [u8; DADD_FIELDS.len()] = forms!(DADD_FIELDS, dadd_rule);
fn dadd(x: Word) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::DoubleAdd(ops::DoubleAdd {
        dst: pair(r, 0),
        lhs: pair(r, 8),
        rhs: double_b(r),
        modifiers: [
            double_source_modifier(r, 46, 48),
            double_source_modifier(r, 49, 45),
        ],
        round: rounding(b(r, 39, 2)),
        flags: [FloatFlags::Kept, FloatFlags::Written][b(r, 47, 1) as usize],
    }))
}
// Hybrid bundle B52eea4c54c38d8b5 is eight complete 256-word immediate
// cubes. Each cube varies the outer guard inversion, the five source/CC
// bits 45..49, ignored bit 50 and D20 sign while fixing DADD.RM and every
// other bit. The exact mask/value boundary admits only packets P006461-68.
const DADD_IMMEDIATE_RM_PACKET_MASK: u64 = 0xfef8_1fff_fff7_ffff;
const DADD_IMMEDIATE_RM_PACKET_VALUES: [u64; 8] = [
    0x3870_0080_0007_0000,
    0x3870_02cd_7ec7_7914,
    0x3870_0ca7_1987_c17e,
    0x3870_0ec3_ba37_5b2b,
    0x3870_18e6_f617_7807,
    0x3870_1e80_0ff7_ffff,
    0x3870_1e84_6767_b14c,
    0x3870_1efa_5ea7_cf2a,
];
fn dadd_rm_is_source_locked(r: u64) -> bool {
    DADD_IMMEDIATE_RM_PACKET_VALUES.contains(&(r & DADD_IMMEDIATE_RM_PACKET_MASK))
}
// Hybrid bundle Bcf4e4433b82e5605 is eight complete 128-word register
// cubes. Each cube varies the outer guard inversion, four source-modifier
// bits, ignored bit 50 and CC while fixing DADD.RM and every other bit.
// The exact mask/value boundary admits only packets P003585-92.
const DADD_REGISTER_RM_PACKET_MASK: u64 = 0xfff8_1fff_fff7_ffff;
const DADD_REGISTER_RM_PACKET_VALUES: [u64; 8] = [
    0x5c70_0080_0007_0000,
    0x5c70_0080_a427_440b,
    0x5c70_02e1_8077_a0d1,
    0x5c70_06dc_cc47_3d36,
    0x5c70_0cb2_f917_817b,
    0x5c70_0cb4_1db7_aa3a,
    0x5c70_1ca4_8b47_e19b,
    0x5c70_1caa_6b67_2c35,
];
fn dadd_register_rm_is_source_locked(r: u64) -> bool {
    DADD_REGISTER_RM_PACKET_VALUES.contains(&(r & DADD_REGISTER_RM_PACKET_MASK))
}
// Hybrid bundle B6a7d64e7f031ad87 is nine complete 128-word constant-buffer
// cubes. Each cube varies the outer guard inversion, four source-modifier
// bits, ignored bit 50 and CC while fixing DADD.RM and every other bit.
// The exact mask/value boundary admits only packets P005263-71.
const DADD_CBUF_RM_PACKET_MASK: u64 = 0xfff8_1fff_fff7_ffff;
const DADD_CBUF_RM_PACKET_VALUES: [u64; 9] = [
    0x4c70_0080_0000_0000,
    0x4c70_0080_0007_0000,
    0x4c70_0ade_fac7_4812,
    0x4c70_0e83_90c7_921f,
    0x4c70_16d1_9b07_39ac,
    0x4c70_16ec_8347_824a,
    0x4c70_18fd_a827_4194,
    0x4c70_1ce6_b537_0757,
    0x4c70_1e80_0ff7_ffff,
];
fn dadd_cbuf_rm_is_source_locked(r: u64) -> bool {
    DADD_CBUF_RM_PACKET_VALUES.contains(&(r & DADD_CBUF_RM_PACKET_MASK))
}
// Hybrid bundle B6f062091c6e76f1b is nine complete 128-word constant-buffer
// cubes. Each cube varies the outer guard inversion, four source-modifier
// bits, ignored bit 50 and CC while fixing DADD.RP and every other bit.
// The exact mask/value boundary admits only packets P005272-80.
const DADD_CBUF_RP_PACKET_MASK: u64 = 0xfff8_1fff_fff7_ffff;
const DADD_CBUF_RP_PACKET_VALUES: [u64; 9] = [
    0x4c70_0100_0000_0000,
    0x4c70_0100_0007_0000,
    0x4c70_0b5e_fac7_4812,
    0x4c70_0f03_90c7_921f,
    0x4c70_1751_9b07_39ac,
    0x4c70_176c_8347_824a,
    0x4c70_197d_a827_4194,
    0x4c70_1d66_b537_0757,
    0x4c70_1f00_0ff7_ffff,
];
fn dadd_cbuf_rp_is_source_locked(r: u64) -> bool {
    DADD_CBUF_RP_PACKET_VALUES.contains(&(r & DADD_CBUF_RP_PACKET_MASK))
}
// Hybrid bundle Be1e786eb176d6782 is nine complete 128-word constant-buffer
// cubes. Each cube varies the outer guard inversion, four source-modifier
// bits, ignored bit 50 and CC while fixing DADD.RZ and every other bit.
// The exact mask/value boundary admits only packets P005281-89.
const DADD_CBUF_RZ_PACKET_MASK: u64 = 0xfff8_1fff_fff7_ffff;
const DADD_CBUF_RZ_PACKET_VALUES: [u64; 9] = [
    0x4c70_0180_0000_0000,
    0x4c70_0180_0007_0000,
    0x4c70_0bde_fac7_4812,
    0x4c70_0f83_90c7_921f,
    0x4c70_17d1_9b07_39ac,
    0x4c70_17ec_8347_824a,
    0x4c70_19fd_a827_4194,
    0x4c70_1de6_b537_0757,
    0x4c70_1f80_0ff7_ffff,
];
fn dadd_cbuf_rz_is_source_locked(r: u64) -> bool {
    DADD_CBUF_RZ_PACKET_VALUES.contains(&(r & DADD_CBUF_RZ_PACKET_MASK))
}
// Hybrid bundle B96a2bab5fa689e7d is eight complete 256-word immediate
// cubes. Each cube varies the outer guard inversion, the five source/CC
// bits 45..49, ignored bit 50 and D20 sign while fixing DADD.RP and every
// other bit. The exact mask/value boundary admits only packets P006469-76.
const DADD_IMMEDIATE_RP_PACKET_MASK: u64 = 0xfef8_1fff_fff7_ffff;
const DADD_IMMEDIATE_RP_PACKET_VALUES: [u64; 8] = [
    0x3870_0100_0007_0000,
    0x3870_034d_7ec7_7914,
    0x3870_0d27_1987_c17e,
    0x3870_0f43_ba37_5b2b,
    0x3870_1966_f617_7807,
    0x3870_1f00_0ff7_ffff,
    0x3870_1f04_6767_b14c,
    0x3870_1f7a_5ea7_cf2a,
];
fn dadd_rp_is_source_locked(r: u64) -> bool {
    DADD_IMMEDIATE_RP_PACKET_VALUES.contains(&(r & DADD_IMMEDIATE_RP_PACKET_MASK))
}
// Hybrid bundle B62bbbcf898deff98 is eight complete 128-word register
// cubes. Each cube varies the outer guard inversion, four source-modifier
// bits, ignored bit 50 and CC while fixing DADD.RP and every other bit.
// The exact mask/value boundary admits only packets P003593-600.
const DADD_REGISTER_RP_PACKET_MASK: u64 = 0xfff8_1fff_fff7_ffff;
const DADD_REGISTER_RP_PACKET_VALUES: [u64; 8] = [
    0x5c70_0100_0007_0000,
    0x5c70_0100_a427_440b,
    0x5c70_0361_8077_a0d1,
    0x5c70_075c_cc47_3d36,
    0x5c70_0d32_f917_817b,
    0x5c70_0d34_1db7_aa3a,
    0x5c70_1d24_8b47_e19b,
    0x5c70_1d2a_6b67_2c35,
];
fn dadd_register_rp_is_source_locked(r: u64) -> bool {
    DADD_REGISTER_RP_PACKET_VALUES.contains(&(r & DADD_REGISTER_RP_PACKET_MASK))
}
// Hybrid bundle Baad61edec9687e55 is eight complete 256-word immediate
// cubes. Each cube varies the outer guard inversion, the five source/CC
// bits 45..49, ignored bit 50 and D20 sign while fixing DADD.RZ and every
// other bit. The exact mask/value boundary admits only packets P006477-84.
const DADD_IMMEDIATE_RZ_PACKET_MASK: u64 = 0xfef8_1fff_fff7_ffff;
const DADD_IMMEDIATE_RZ_PACKET_VALUES: [u64; 8] = [
    0x3870_0180_0007_0000,
    0x3870_03cd_7ec7_7914,
    0x3870_0da7_1987_c17e,
    0x3870_0fc3_ba37_5b2b,
    0x3870_19e6_f617_7807,
    0x3870_1f80_0ff7_ffff,
    0x3870_1f84_6767_b14c,
    0x3870_1ffa_5ea7_cf2a,
];
fn dadd_rz_is_source_locked(r: u64) -> bool {
    DADD_IMMEDIATE_RZ_PACKET_VALUES.contains(&(r & DADD_IMMEDIATE_RZ_PACKET_MASK))
}
// Hybrid bundle B13d614303751d81c is eight complete 128-word register
// cubes. Each cube varies the outer guard inversion, four source-modifier
// bits, ignored bit 50 and CC while fixing DADD.RZ and every other bit.
// The exact mask/value boundary admits only packets P003601-08.
const DADD_REGISTER_RZ_PACKET_MASK: u64 = 0xfff8_1fff_fff7_ffff;
const DADD_REGISTER_RZ_PACKET_VALUES: [u64; 8] = [
    0x5c70_0180_0007_0000,
    0x5c70_0180_a427_440b,
    0x5c70_03e1_8077_a0d1,
    0x5c70_07dc_cc47_3d36,
    0x5c70_0db2_f917_817b,
    0x5c70_0db4_1db7_aa3a,
    0x5c70_1da4_8b47_e19b,
    0x5c70_1daa_6b67_2c35,
];
fn dadd_register_rz_is_source_locked(r: u64) -> bool {
    DADD_REGISTER_RZ_PACKET_VALUES.contains(&(r & DADD_REGISTER_RZ_PACKET_MASK))
}
// Hybrid bundle B998ce0179f0c887c is eight complete 128-word immediate
// cubes. Each cube varies the outer guard inversion, four source-modifier
// bits, ignored bit 50 and D20 sign while fixing RN and CC. The exact
// mask/value boundary admits only packets P006453-60.
const DADD_IMMEDIATE_RN_CC_PACKET_MASK: u64 = 0xfef8_9fff_fff7_ffff;
const DADD_IMMEDIATE_RN_CC_PACKET_VALUES: [u64; 8] = [
    0x3870_8000_0007_0000,
    0x3870_824d_7ec7_7914,
    0x3870_8c27_1987_c17e,
    0x3870_8e43_ba37_5b2b,
    0x3870_9866_f617_7807,
    0x3870_9e00_0ff7_ffff,
    0x3870_9e04_6767_b14c,
    0x3870_9e7a_5ea7_cf2a,
];
fn dadd_rn_cc_is_source_locked(r: u64) -> bool {
    DADD_IMMEDIATE_RN_CC_PACKET_VALUES.contains(&(r & DADD_IMMEDIATE_RN_CC_PACKET_MASK))
}
fn read_dadd(x: Word) -> Instruction {
    if dadd_rm_is_source_locked(x.r)
        || dadd_register_rm_is_source_locked(x.r)
        || dadd_cbuf_rm_is_source_locked(x.r)
        || dadd_cbuf_rp_is_source_locked(x.r)
        || dadd_cbuf_rz_is_source_locked(x.r)
        || dadd_rp_is_source_locked(x.r)
        || dadd_register_rp_is_source_locked(x.r)
        || dadd_rz_is_source_locked(x.r)
        || dadd_register_rz_is_source_locked(x.r)
        || dadd_rn_cc_is_source_locked(x.r)
    {
        return dadd(x);
    }
    if b(x.r, 47, 1) == 1 && !double_rn_cc_word(x.r) {
        return unknown(x);
    }
    let paths: [Decode; 2] = [unknown, dadd];
    paths[DADD_FORMS[DADD_FIELDS.index(x.r)] as usize](x)
}
// DMNMX: Ryujinx InstDecoders.cs:1358-1411 (InstDmnmxR/I/C): D 0..7, A
// 8..15, B by form, SrcPred 39..41, SrcPredInv 42, NegB 45, AbsA 46, CC
// 47, NegA 48, AbsB 49; envytools gm107.c:2007,2081,2127 agree.
// InstEmitFloatMinMax.cs:23-43,78-104 takes the minimum where the
// predicate holds, else the maximum. nvdisasm SM52
// ignores the unassigned 43..44 and 50 and the register form's 28..38.
const DMNMX_FIELDS: Fields<1> = Fields([Field::zero(47, 1)]);
const fn dmnmx_rule(r: u64) -> u8 {
    (b(r, 47, 1) == 0) as u8
}
const DMNMX_FORMS: [u8; DMNMX_FIELDS.len()] = forms!(DMNMX_FIELDS, dmnmx_rule);
/// PT is the constant minimum and !PT the constant maximum; any other
/// predicate selects between them at run time.
fn dmnmx_operation(r: u64) -> DoubleMinMaxOperation {
    match (b(r, 39, 3), b(r, 42, 1)) {
        (7, 0) => DoubleMinMaxOperation::Minimum,
        (7, _) => DoubleMinMaxOperation::Maximum,
        (register, inverted) => DoubleMinMaxOperation::Selected(Predicate {
            register: register as u8,
            inverted: inverted == 1,
        }),
    }
}
fn dmnmx(x: Word) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::DoubleMinMax(ops::DoubleMinMax {
        dst: pair(r, 0),
        lhs: pair(r, 8),
        rhs: double_b(r),
        operation: dmnmx_operation(r),
        modifiers: [
            double_source_modifier(r, 46, 48),
            double_source_modifier(r, 49, 45),
        ],
        flags: [FloatFlags::Kept, FloatFlags::Written][b(r, 47, 1) as usize],
    }))
}
const DMNMX_CC_EXACT: [u64; 15] = [
    0x5c50_8000_0007_0000,
    0x5c51_e26d_1df0_3060,
    0x5c53_abe9_f630_8c0c,
    0x5c54_8037_73ec_86fe,
    0x5c57_d1ca_73fe_9cd5,
    0x4c50_8000_0007_0000,
    0x4c50_f2c6_4fc6_7370,
    0x4c54_8129_8a24_86bd,
    0x4c54_df5c_97d5_b24a,
    0x4c57_9e13_2e27_584f,
    0x3853_858d_815e_e051,
    0x3855_b18f_0e65_7ed7,
    0x3952_81b4_453c_4204,
    0x3952_c50d_ef49_2290,
    0x3956_f5db_a153_81c0,
];
// Hybrid bundle B3e90b402e51eb95b is seven complete 256-word cubes. Each
// cube varies the outer guard inversion, source-predicate inversion, four
// source-modifier bits, ignored bit 50 and D20 sign while fixing every other
// bit. Keeping the mask and seven values here admits exactly those 1,792
// source-locked words without broadening the rest of the immediate row.
const DMNMX_IMMEDIATE_CC_PACKET_MASK: u64 = 0xfef8_9bff_fff7_ffff;
const DMNMX_IMMEDIATE_CC_PACKET_VALUES: [u64; 7] = [
    0x3850_8000_0007_0000,
    0x3850_805d_3db7_5ce2,
    0x3850_822e_ade7_8146,
    0x3850_82e3_d857_c6ee,
    0x3850_83b0_d857_31fe,
    0x3850_885c_2af7_8254,
    0x3850_8a5a_3f57_c9ab,
];
fn dmnmx_cc_is_source_locked(r: u64) -> bool {
    DMNMX_CC_EXACT.contains(&r)
        || DMNMX_IMMEDIATE_CC_PACKET_VALUES.contains(&(r & DMNMX_IMMEDIATE_CC_PACKET_MASK))
}
fn read_dmnmx(x: Word) -> Instruction {
    if dmnmx_cc_is_source_locked(x.r) {
        return dmnmx(x);
    }
    let paths: [Decode; 2] = [unknown, dmnmx];
    paths[DMNMX_FORMS[DMNMX_FIELDS.index(x.r)] as usize](x)
}
// DSET: Ryujinx InstDecoders.cs:1456-1518 (InstDsetR/I/C): D 0..7, A
// 8..15, B by form, SrcPred 39..41, SrcPredInv 42, NegA 43, AbsB 44,
// BoolOp 45..46, CC 47, FComp 48..51, BVal 52, NegB 53, AbsA 54;
// envytools gm107.c:2035,2105,2153,1411-1434 agree, and EmitFset
// (InstEmitFloatComparison.cs:378-419) applies them. BoolOp 3 is
// reserved. CC47 is admitted only for the seven source-locked words below;
// result representation and flag behavior form one closed IR state. nvdisasm
// SM52 ignores the register
// form's 28..38.
const DSET_FIELDS: Fields<2> = Fields([Field::bits(47, 1), Field::bits(45, 2)]);
const fn dset_cc_word(r: u64) -> bool {
    matches!(
        r,
        0x5940_bf2f_80e1_2c1a
            | 0x5961_cbc8_e1e6_78bd
            | 0x3248_b199_07bb_bb0d
            | 0x334a_d6b1_3595_21a1
            | 0x593f_94c6_cf78_22b3
            | 0x3255_cb90_b738_8bba
            | 0x3313_b2ae_d9bc_64e7
    )
}
const fn dset_rule(r: u64) -> u8 {
    (b(r, 45, 2) < 3) as u8
}
const DSET_FORMS: [u8; DSET_FIELDS.len()] = forms!(DSET_FIELDS, dset_rule);
fn dset(x: Word) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::DoubleSet(ops::DoubleSet {
        dst: Register(b(r, 0, 8) as u8),
        lhs: pair(r, 8),
        rhs: double_b(r),
        compare: fcmp(b(r, 48, 4)),
        set_op: SET_OPS[b(r, 45, 2) as usize],
        accum: Predicate {
            register: b(r, 39, 3) as u8,
            inverted: b(r, 42, 1) == 1,
        },
        abs: [b(r, 54, 1) == 1, b(r, 44, 1) == 1],
        negate: [b(r, 43, 1) == 1, b(r, 53, 1) == 1],
        result: match (b(r, 52, 1), b(r, 47, 1)) {
            (0, 0) => DoubleSetResult::Mask(FloatFlags::Kept),
            (0, 1) => DoubleSetResult::Mask(FloatFlags::Written),
            (1, 0) => DoubleSetResult::FloatOne(FloatFlags::Kept),
            (1, 1) => DoubleSetResult::FloatOne(FloatFlags::Written),
            _ => unreachable!(),
        },
    }))
}
fn read_dset(x: Word) -> Instruction {
    if b(x.r, 47, 1) == 1 && !dset_cc_word(x.r) {
        return unknown(x);
    }
    let paths: [Decode; 2] = [unknown, dset];
    paths[DSET_FORMS[DSET_FIELDS.index(x.r)] as usize](x)
}
// DSETP: Ryujinx InstDecoders.cs:1520-1579 (InstDsetpR/I/C): DestPredInv
// 0..2, DestPred 3..5, NegB 6, AbsA 7, A 8..15, B by form, SrcPred
// 39..41, SrcPredInv 42, NegA 43, AbsB 44, BoolOp 45..46, FComp 48..51;
// envytools gm107.c:2026,2097,2144,406,1411-1434 agree, and EmitFsetp
// (InstEmitFloatComparison.cs:88-161,423-454) applies them. BoolOp 3 is
// reserved. nvdisasm SM52 ignores the unassigned 47 and the register
// form's 28..38.
const DSETP_FIELDS: Fields<1> = Fields([Field::bits(45, 2)]);
const fn dsetp_rule(r: u64) -> u8 {
    (b(r, 45, 2) < 3) as u8
}
const DSETP_FORMS: [u8; DSETP_FIELDS.len()] = forms!(DSETP_FIELDS, dsetp_rule);
fn dsetp(x: Word) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::DoubleSetP(ops::DoubleSetP {
        dst: [b(r, 3, 3) as u8, b(r, 0, 3) as u8],
        lhs: pair(r, 8),
        rhs: double_b(r),
        compare: fcmp(b(r, 48, 4)),
        set_op: SET_OPS[b(r, 45, 2) as usize],
        accum: Predicate {
            register: b(r, 39, 3) as u8,
            inverted: b(r, 42, 1) == 1,
        },
        abs: [b(r, 7, 1) == 1, b(r, 44, 1) == 1],
        negate: [b(r, 43, 1) == 1, b(r, 6, 1) == 1],
    }))
}
fn read_dsetp(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, dsetp];
    paths[DSETP_FORMS[DSETP_FIELDS.index(x.r)] as usize](x)
}
// FSETP (Ryujinx InstTable.cs, MIT: 0x5bb0-0x5bbf, 0x4bb0-0x4bbf,
// 0x36b0-0x36bf, 0x37b0-0x37bf; InstDecoders.cs:2210-2276): dst 3..5,
// second destination 0..2, A abs 7 neg 43, B abs 44 neg 6 (the f20
// immediate too: FsetpI/EmitFsetp, InstEmitFloatComparison.cs:261-283,
// 423-454, apply abs then negate to it), accumulator 39..41 with
// negate 42, Boolean 45..46 (3 is invalid), FTZ 47, compare 48..51.
// EmitFsetp writes the complement combined with the accumulator to the
// second destination. FTZ only affects denormal inputs.
fn fsetp(x: Word) -> Instruction {
    let r = x.r;
    let l = Register(b(r, 8, 8) as u8);
    let set_op = SET_OPS[b(r, 45, 2) as usize];
    x.guarded({
        InstructionKind::FSetP(ops::FSetP {
            dst: [b(r, 3, 3) as u8, b(r, 0, 3) as u8],
            lhs: l,
            rhs: float_form_src(r),
            compare: fcmp(b(r, 48, 4)),
            set_op,
            accum: b(r, 39, 3) as u8,
            accum_negate: b(r, 42, 1) == 1,
            abs: [b(r, 7, 1) == 1, b(r, 44, 1) == 1],
            negate: [b(r, 43, 1) == 1, b(r, 6, 1) == 1],
        })
    })
}
const FSETP_FIELDS: Fields<1> = Fields([Field::bits(45, 2)]);
const fn fsetp_rule(r: u64) -> u8 {
    (b(r, 45, 2) < 3) as u8
}
const FSETP_FORMS: [u8; FSETP_FIELDS.len()] = forms!(FSETP_FIELDS, fsetp_rule);
fn read_fsetp(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, fsetp];
    paths[(FSETP_FORMS[FSETP_FIELDS.index(x.r)]) as usize](x)
}
// CSETP, not ICMP: envytools envydis/gm107.c:2058; condition
// T(50b0_0)8..13:1647-1680, Boolean T(5bb0_1)45..47:1430-1434,
// PRED03/PRED00:368-369, source pred39/not42:373,406.
// Ryujinx InstEmitConditionCode.cs:35-47 writes condition OP pred
// and !condition OP pred. GetCondition's missing CSM cases 24..29
// retain its true default. Unsupported CC47 and unassigned bits reject.
fn csetp(x: Word) -> Instruction {
    let r = x.r;
    let set_op = SET_OPS[b(r, 45, 2) as usize];
    x.guarded(InstructionKind::CSetP(ops::CSetP {
        dst: [b(r, 3, 3) as u8, b(r, 0, 3) as u8],
        condition: CSetPCondition::decode(b(r, 8, 5) as u8),
        set_op,
        accum: Predicate {
            register: b(r, 39, 3) as u8,
            inverted: b(r, 42, 1) != 0,
        },
    }))
}
const CSETP_FIELDS: Fields<6> = Fields([
    Field::zero(6, 2),
    Field::zero(13, 3),
    Field::zero(20, 19),
    Field::zero(43, 2),
    Field::bits(47, 1),
    Field::bits(45, 2),
]);
const fn csetp_rule(r: u64) -> u8 {
    (b(r, 6, 2) == 0
        && b(r, 13, 3) == 0
        && b(r, 20, 19) == 0
        && b(r, 43, 2) == 0
        && b(r, 47, 1) == 0
        && b(r, 45, 2) < 3) as u8
}
const CSETP_FORMS: [u8; CSETP_FIELDS.len()] = forms!(CSETP_FIELDS, csetp_rule);
fn read_csetp(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, csetp];
    paths[(CSETP_FORMS[CSETP_FIELDS.index(x.r)]) as usize](x)
}
// CSET (Ryujinx InstTable.cs, MIT: 0x5098-0x509f; InstDecoders.cs:
// 1188-1201 InstCset; envytools gm107.c:2059): D 0..7, condition 8..12
// (InstDecoders.cs:104-111, flag formulas InstEmitConditionCode.cs:
// 52-84), accumulator 39..41 with invert 42, BF 44, Boolean 45..46 (3 is
// invalid), CC 47. InstEmitConditionCode.cs:12-33 combines the condition
// once with the accumulator. CC is not modeled and stays Unknown;
// nvdisasm ignores bits 13..15, 20..38, 43 and 48..50.
fn cset(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let set_op = SET_OPS[b(r, 45, 2) as usize];
    let selector = b(r, 8, 5) as u8;
    let register = b(r, 39, 3) as u8;
    x.guarded(InstructionKind::CSet(ops::CSet {
        dst: d,
        condition: CSetPCondition::decode(selector),
        set_op,
        accum: Predicate {
            register,
            inverted: b(r, 42, 1) == 1,
        },
        result: [SetResult::Mask, SetResult::FloatOne][b(r, 44, 1) as usize],
    }))
}
const CSET_FIELDS: Fields<2> = Fields([Field::bits(45, 2), Field::bits(47, 1)]);
const fn cset_rule(r: u64) -> u8 {
    (b(r, 45, 2) < 3 && b(r, 47, 1) == 0) as u8
}
const CSET_FORMS: [u8; CSET_FIELDS.len()] = forms!(CSET_FIELDS, cset_rule);
fn read_cset(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, cset];
    paths[(CSET_FORMS[CSET_FIELDS.index(x.r)]) as usize](x)
}
// Mesa sm50.rs:2090-2100,798-807 and envytools gm107.c:
// 1430-1435,1683-1688 assign AND/OR/XOR; logical mode3 is
// reserved and stays Unknown rather than a partially decoded PSETP.
fn psetp(x: Word) -> Instruction {
    let r = x.r;
    let ab = SET_OPS[b(r, 24, 2) as usize];
    let c = SET_OPS[b(r, 45, 2) as usize];
    x.guarded({
        InstructionKind::PSetP(ops::PSetP {
            dst: [b(r, 3, 3) as u8, b(r, 0, 3) as u8],
            src: [b(r, 12, 3) as u8, b(r, 29, 3) as u8, b(r, 39, 3) as u8],
            set_op: [ab, c],
            // Source negates at 15, 32 and 42 (sm50.rs OpPSetP encode).
            negate: [b(r, 15, 1) == 1, b(r, 32, 1) == 1, b(r, 42, 1) == 1],
        })
    })
}
const PSETP_FIELDS: Fields<2> = Fields([Field::bits(24, 2), Field::bits(45, 2)]);
const fn psetp_rule(r: u64) -> u8 {
    (b(r, 24, 2) < 3 && b(r, 45, 2) < 3) as u8
}
const PSETP_FORMS: [u8; PSETP_FIELDS.len()] = forms!(PSETP_FIELDS, psetp_rule);
fn read_psetp(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, psetp];
    paths[(PSETP_FORMS[PSETP_FIELDS.index(x.r)]) as usize](x)
}
// PSET, not MUFU: envytools/envydis/gm107.c:2061. dst0..7
// (:203), predicates12..14/29..31/39..41 (:370,372-373), inversion
// 15/32/42 (:404-406), opAB24..25 (:1683-1687), opC45..46
// (:1430-1434), float boolean44 and CC47 (:2061). CC and reserved
// logical selectors remain Unknown. Ryujinx InstEmitPredicate.cs:
// 12-32 combines three predicates and writes all ones or 1.0f.
fn pset(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let ab = SET_OPS[b(r, 24, 2) as usize];
    let c = SET_OPS[b(r, 45, 2) as usize];
    x.guarded(InstructionKind::PSet(ops::PSet {
        dst: d,
        src: [b(r, 12, 3) as u8, b(r, 29, 3) as u8, b(r, 39, 3) as u8],
        set_op: [ab, c],
        negate: [b(r, 15, 1) != 0, b(r, 32, 1) != 0, b(r, 42, 1) != 0],
        bool_float: b(r, 44, 1) != 0,
    }))
}
const PSET_FIELDS: Fields<3> = Fields([Field::bits(47, 1), Field::bits(24, 2), Field::bits(45, 2)]);
const fn pset_rule(r: u64) -> u8 {
    (b(r, 47, 1) == 0 && b(r, 24, 2) < 3 && b(r, 45, 2) < 3) as u8
}
const PSET_FORMS: [u8; PSET_FIELDS.len()] = forms!(PSET_FIELDS, pset_rule);
fn read_pset(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, pset];
    paths[(PSET_FORMS[PSET_FIELDS.index(x.r)]) as usize](x)
}
// FSET register/cbuf/immediate: envytools gm107.c:2036,2106,2154;
// dst0..7, A8..15, B20..27 or cbuf20..38 or immediate20..38/sign56;
// compare48..51, BF52, absA54/absB44, negA43/negB53, CC47,
// accumulator39..41/invert42, Boolean45..46 (:1430-1434).
// Mesa sm50.rs:839-862 confirms the base encoding but uses bit6
// for cbuf negB, overlapping the destination. The old Mesa emitter
// nv50_ir_emit_gm107.cpp:1575-1620 and Ryujinx InstDecoders.cs:
// 2169-2185 agree with envytools on bit53, used here.
// Semantics: InstEmitFloatComparison.cs:378-419. Mesa sm50.rs:861
// encodes FTZ55; preserve it for input flushing in the emitter.
// No reference above requires register28..38 to be zero.
// Cbuf uses main's ff00 mask: Ryujinx InstTable.cs:119. Envytools'
// broader fe00 row (:2106) follows DSET4900/ff80 (:2105); accepting
// cbuf bit56 here would misdecode that double comparison as FSET.
// Boolean3 is explicitly OOPS in gm107.c:1430-1434, so it remains
// Unknown even though main 252cf73 accepted it with a PT accumulator.
fn fset(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);
    let set_op = SET_OPS[b(r, 45, 2) as usize];
    x.guarded(InstructionKind::FSet(ops::FSet {
        dst: d,
        lhs: l,
        rhs: float_form_src(r),
        compare: fcmp(b(r, 48, 4)),
        bool_float: b(r, 52, 1) == 1,
        cc: b(r, 47, 1) == 1,
        abs: [b(r, 54, 1) == 1, b(r, 44, 1) == 1],
        negate: [b(r, 43, 1) == 1, b(r, 53, 1) == 1],
        ftz: b(r, 55, 1) != 0,
        accum: Predicate {
            register: b(r, 39, 3) as u8,
            inverted: b(r, 42, 1) != 0,
        },
        set_op,
    }))
}
const FSET_FIELDS: Fields<1> = Fields([Field::bits(45, 2)]);
const fn fset_rule(r: u64) -> u8 {
    (b(r, 45, 2) < 3) as u8
}
const FSET_FORMS: [u8; FSET_FIELDS.len()] = forms!(FSET_FIELDS, fset_rule);
fn read_fset(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, fset];
    paths[(FSET_FORMS[FSET_FIELDS.index(x.r)]) as usize](x)
}
// SEL: Mesa sm50.rs:2010-2029; envytools gm107.c:1998,2072,
// 2118,253,373,406; Ryujinx InstDecoders.cs:4279-4316 and
// InstEmitMove.cs:219-249,271-276. dst0..7, A8..15, B by form,
// condition39..41/invert42. Signed immediate20..38/sign56 makes
// the negative prefix 0x39a0. form_src preserves its exact bits.
// Keep every previously accepted word; new signed forms reject
// unnamed43..50. Selection needs one integer OpSelect, preserving
// payloads without the reference translator's extra IR copies.
fn sel(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::Sel(ops::Sel {
        dst: d,
        when_true: l,
        when_false: int_form_src(r),
        condition: b(r, 39, 3) as u8,
        condition_negate: b(r, 42, 1) == 1,
    }))
}
const SEL_IMMEDIATE_FIELDS: Fields<1> = Fields([Field::zero(43, 8)]);
const fn sel_immediate_rule(r: u64) -> u8 {
    (b(r, 43, 8) == 0) as u8
}
const SEL_IMMEDIATE_FORMS: [u8; SEL_IMMEDIATE_FIELDS.len()] =
    forms!(SEL_IMMEDIATE_FIELDS, sel_immediate_rule);
fn read_sel_immediate(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, sel];
    paths[(SEL_IMMEDIATE_FORMS[SEL_IMMEDIATE_FIELDS.index(x.r)]) as usize](x)
}
// I2F register/cbuf/immediate families: envytools envydis/gm107.c:
// 1995,2069,2115. Size/sign:1256-1272; rounding:1275-1280;
// byte:1200-1205; abs49/neg45/CC47:1995. Mesa sm50.rs:1848-1879
// confirms the fields and immediate sign56. Ryujinx Instructions/
// InstEmitConversion.cs:234-283 applies abs/neg BEFORE extraction.
// Predicate16..18/inversion19: Mesa sm50.rs:225-235; envytools
// gm107.c:371,408-411; Ryujinx InstDecoders.cs:2685-2686. These
// bits are consumed by the common execution guard above.
// All round39..40 modes are implemented. 64-bit sources require
// GPR (Mesa ir.rs:4834-4838); F16 cannot span 32 bits
// (sm50.rs:1868-1872). CC remains Unknown (Ryujinx:283).
// A 16-bit source selects H0 or H1 (byte 0 or 2): nvdisasm SM52 prints
// byte 1 and 3 as INVALIDSIZE1/INVALIDSIZE3, for I2I as well.
const I2F_REGISTER_FIELDS0: Fields<3> =
    Fields([Field::bits(8, 2), Field::bits(10, 2), Field::bits(41, 2)]);
const fn i2f_register_rule0(r: u64) -> u8 {
    (b(r, 8, 2) >= 1
        && (b(r, 10, 2) < 3 || b(r, 8, 2) >= 2)
        && (b(r, 10, 2) == 0
            || b(r, 10, 2) == 1 && b(r, 41, 1) == 0
            || b(r, 10, 2) >= 2 && b(r, 41, 2) == 0)) as u8
}
const I2F_REGISTER_FORMS0: [u8; I2F_REGISTER_FIELDS0.len()] =
    forms!(I2F_REGISTER_FIELDS0, i2f_register_rule0);
const I2F_REGISTER_FIELDS1: Fields<8> = Fields([
    Field::zero(12, 1),
    Field::zero(14, 2),
    Field::zero(43, 2),
    Field::zero(46, 1),
    Field::zero(47, 1),
    Field::zero(48, 1),
    Field::zero(50, 1),
    Field::zero(28, 11),
]);
const fn i2f_register_rule1(r: u64) -> u8 {
    (b(r, 12, 1) == 0
        && b(r, 14, 2) == 0
        && b(r, 43, 2) == 0
        && b(r, 46, 1) == 0
        && b(r, 47, 1) == 0
        && b(r, 48, 1) == 0
        && b(r, 50, 1) == 0
        && b(r, 28, 11) == 0) as u8
}
const I2F_REGISTER_FORMS1: [u8; I2F_REGISTER_FIELDS1.len()] =
    forms!(I2F_REGISTER_FIELDS1, i2f_register_rule1);
fn read_i2f_register(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, i2f];
    paths[(I2F_REGISTER_FORMS0[I2F_REGISTER_FIELDS0.index(x.r)]
        & I2F_REGISTER_FORMS1[I2F_REGISTER_FIELDS1.index(x.r)]) as usize](x)
}
fn i2f(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);

    x.guarded(InstructionKind::Convert(ops::Convert {
        kind: Conversion::I2F {
            signed: b(r, 13, 1) == 1,
            dst_size: b(r, 8, 2) as u8,
            src_size: b(r, 10, 2) as u8,
            round: rounding(b(r, 39, 2)),
            byte: b(r, 41, 2) as u8,
            abs: b(r, 49, 1) != 0,
            negate: b(r, 45, 1) != 0,
        },
        dst: d,
        src: int_form_src(r),
    }))
}
const I2F_FIELDS0: Fields<3> = Fields([Field::bits(8, 2), Field::bits(10, 2), Field::bits(41, 2)]);
const fn i2f_rule0(r: u64) -> u8 {
    (b(r, 8, 2) >= 1
        && b(r, 10, 2) < 3
        && (b(r, 10, 2) == 0
            || b(r, 10, 2) == 1 && b(r, 41, 1) == 0
            || b(r, 10, 2) >= 2 && b(r, 41, 2) == 0)) as u8
}
const I2F_FORMS0: [u8; I2F_FIELDS0.len()] = forms!(I2F_FIELDS0, i2f_rule0);
const I2F_FIELDS1: Fields<7> = Fields([
    Field::zero(12, 1),
    Field::zero(14, 2),
    Field::zero(43, 2),
    Field::zero(46, 1),
    Field::zero(47, 1),
    Field::zero(48, 1),
    Field::zero(50, 1),
]);
const fn i2f_rule1(r: u64) -> u8 {
    (b(r, 12, 1) == 0
        && b(r, 14, 2) == 0
        && b(r, 43, 2) == 0
        && b(r, 46, 1) == 0
        && b(r, 47, 1) == 0
        && b(r, 48, 1) == 0
        && b(r, 50, 1) == 0) as u8
}
const I2F_FORMS1: [u8; I2F_FIELDS1.len()] = forms!(I2F_FIELDS1, i2f_rule1);
fn read_i2f(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, i2f];
    paths[(I2F_FORMS0[I2F_FIELDS0.index(x.r)] & I2F_FORMS1[I2F_FIELDS1.index(x.r)]) as usize](x)
}
// F2I: envytools envydis/gm107.c:1996,2070,2116; destination
// size/sign:1283-1292; float size:1295-1299; round:1302-1307.
// Mesa sm50.rs:1806-1838: half select41, FTZ44, CC47, abs49,
// neg45. Sources are register20..28, cbuf20..39, float immediate
// 20..39/sign56. Ryujinx InstEmitConversion.cs:165-231 supports
// 16/32-bit destinations and float16/32 here. CC and 64-bit
// destinations stay Unknown. Half select is only valid for float16.
// FTZ44 is supported for float32; float16 FTZ stays Unknown.
fn f2i(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);

    x.guarded(InstructionKind::Convert(ops::Convert {
        kind: Conversion::F2I {
            signed: b(r, 12, 1) == 1,
            dst_size: b(r, 8, 2) as u8,
            src_size: b(r, 10, 2) as u8,
            round: rounding(b(r, 39, 2)),
            high: b(r, 41, 1) != 0,
            ftz: b(r, 44, 1) != 0,
            abs: b(r, 49, 1) != 0,
            negate: b(r, 45, 1) != 0,
        },
        dst: d,
        src: float_form_src(r),
    }))
}
const F2I_FIELDS: Fields<5> = Fields([
    Field::bits(8, 2),
    Field::bits(10, 2),
    Field::bits(41, 1),
    Field::bits(44, 1),
    Field::bits(47, 1),
]);
/// 0 Unknown, 1 a float16/32 source, 2 a float64 source (SrcFmt 3).
const fn f2i_rule(r: u64) -> u8 {
    let admitted = (b(r, 8, 2) == 1 || b(r, 8, 2) == 2)
        && b(r, 10, 2) != 0
        && (b(r, 10, 2) == 1 || b(r, 41, 1) == 0)
        && (b(r, 10, 2) == 2 || b(r, 44, 1) == 0)
        && b(r, 47, 1) == 0;
    admitted as u8 * (1 + (b(r, 10, 2) == 3) as u8)
}
const F2I_FORMS: [u8; F2I_FIELDS.len()] = forms!(F2I_FIELDS, f2i_rule);
fn read_f2i(x: Word) -> Instruction {
    let paths: [Decode; 3] = [unknown, f2i, f2i_double];
    paths[(F2I_FORMS[F2I_FIELDS.index(x.r)]) as usize](x)
}
// F2I.F64: the binary64 source by form (Ryujinx InstEmitConversion.cs:
// 344-396 reads the pair, cbuf words or the D20 high word), abs49 and
// neg45 applied before rounding39..40 (:185-194), then the 16/32-bit
// clamp. Sh41, FTZ44 and CC47 stay clear.
fn f2i_double(x: Word) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::DoubleToInt(ops::DoubleToInt {
        dst: Register(b(r, 0, 8) as u8),
        src: double_b(r),
        modifier: double_source_modifier(r, 49, 45),
        signedness: [Signedness::Unsigned, Signedness::Signed][b(r, 12, 1) as usize],
        // The form table admits only 1 (16-bit) and 2 (32-bit).
        width: [IntegerWidth::Bits16, IntegerWidth::Bits32][b(r, 9, 1) as usize],
        round: rounding(b(r, 39, 2)),
    }))
}
// F2F register/cbuf rows: envytools envydis/gm107.c:1997,2071.
// Mesa sm50.rs:1765-1778: source20..28 or cbuf20..39, abs49/neg45;
// sizes8..12, rounding39..41, half41, integer42, FTZ44:1787-1794.
// SAT50/CC47: gm107.c:1997; round table1310-1317. F32 to F32 (sizes
// 0xa) and F16 to F16 (sizes 0x5) with nearest conversion rounding
// or explicit integer rounding are supported; each size pair is its
// own constructor. Other sizes, CC, reserved and unassigned bits
// stay Unknown. Ryujinx InstEmitConversion.cs:121-162 applies
// abs/neg, integer rounding, then saturation in that order.
const F2F_REGISTER_FIELDS0: Fields<5> = Fields([
    Field::bits(8, 4),
    Field::bits(42, 1),
    Field::bits(39, 2),
    Field::bits(41, 1),
    Field::bits(44, 1),
]);
/// 0 Unknown, 1 F32 to F32, 2 F16 to F16 from the low half, 3 F16 to F16
/// from the high half (Sh41, which an F32 source leaves clear), 4 F32 to
/// F64 (sizes 0xb). Widening is exact, so Ryujinx applies no rounding
/// there (InstEmitConversion.cs:134-152); its rounding fields stay clear.
/// 5 F64 to F32 (sizes 0xe) in the rounding39..40 nvdisasm prints (RN,
/// RM, RP, RZ); nvdisasm SM52 ignores bit 42 there. 6 F64 to F64 (sizes 0xf), the
/// identity or an integral rounding like F32 to F32. FTZ44 on a binary64
/// source is not modeled and stays clear. 7 and 8 F16 to F32 (sizes 0x6)
/// from the half Sh41 selects: Ryujinx InstDecoders.cs:1591-1608 and
/// InstEmitConversion.cs:13-37,121-162,344-378 unpack it, apply abs/neg
/// and saturate; the widening is exact, so like F32 to F64 its rounding
/// fields stay clear, and FTZ44 stays clear.
const fn f2f_register_rule0(r: u64) -> u8 {
    let rounded = b(r, 42, 1) != 0 || b(r, 39, 2) == 0;
    let kept = b(r, 44, 1) == 0;
    match (b(r, 8, 4), b(r, 41, 1)) {
        (0xa, 0) if rounded => 1,
        (0x5, 0) if rounded => 2,
        (0x5, 1) if rounded => 3,
        (0xb, 0) if b(r, 42, 1) == 0 && b(r, 39, 2) == 0 => 4,
        (0x6, half) if b(r, 42, 1) == 0 && b(r, 39, 2) == 0 && kept => 7 + half as u8,
        // nvdisasm SM52 ignores Sh41 on a binary64 source and prints
        // PASS for integer rounding 1..3: the value as it is
        // (InstEmitConversion.cs:136-143).
        (0xe, _) if kept => 5,
        (0xf, _) if kept => 6,
        _ => 0,
    }
}
const F2F_REGISTER_FORMS0: [u8; F2F_REGISTER_FIELDS0.len()] =
    forms!(F2F_REGISTER_FIELDS0, f2f_register_rule0);
/// CC47 stays clear. A binary64 source (SrcFmt 3): nvdisasm SM52 ignores
/// 12..15, 43, 46, 48 and the register form's 28..38; every other source
/// keeps them clear.
const F2F_REGISTER_FIELDS1: Fields<7> = Fields([
    Field::zero(47, 1),
    Field::bits(10, 2),
    Field::zero(48, 1),
    Field::zero(12, 4),
    Field::zero(43, 1),
    Field::zero(46, 1),
    Field::zero(28, 11),
]);
const fn f2f_register_rule1(r: u64) -> u8 {
    (b(r, 47, 1) == 0
        && (b(r, 10, 2) == 3
            || (b(r, 48, 1) == 0
                && b(r, 12, 4) == 0
                && b(r, 43, 1) == 0
                && b(r, 46, 1) == 0
                && b(r, 28, 11) == 0))) as u8
}
const F2F_REGISTER_FORMS1: [u8; F2F_REGISTER_FIELDS1.len()] =
    forms!(F2F_REGISTER_FIELDS1, f2f_register_rule1);
fn read_f2f_register(x: Word) -> Instruction {
    let paths: [Decode; 9] = [
        unknown,
        f2f,
        f2f_half,
        f2f_half_high,
        f2f_double,
        f2f_narrow,
        f2f_double_round,
        f2f_half_widen,
        f2f_half_high_widen,
    ];
    paths[(F2F_REGISTER_FORMS0[F2F_REGISTER_FIELDS0.index(x.r)]
        * F2F_REGISTER_FORMS1[F2F_REGISTER_FIELDS1.index(x.r)]) as usize](x)
}
// F2F.F16.F16 (sizes 0x5): Ryujinx InstEmitConversion.cs:344-352 widens
// the half Sh41 selects, :121-162 applies abs/neg and the same-size
// integer rounding, then saturation, and :398-409 packs the result
// back into the low half with the high half zero.
fn f2f_half(x: Word) -> Instruction {
    f2f_half_of(x, F2fSource::Float16Low, F2fDestination::Float16Low)
}
fn f2f_half_high(x: Word) -> Instruction {
    f2f_half_of(x, F2fSource::Float16High, F2fDestination::Float16Low)
}
fn f2f_half_widen(x: Word) -> Instruction {
    f2f_half_of(x, F2fSource::Float16Low, F2fDestination::Float32)
}
fn f2f_half_high_widen(x: Word) -> Instruction {
    f2f_half_of(x, F2fSource::Float16High, F2fDestination::Float32)
}
fn f2f_half_of(x: Word, source: F2fSource, destination: F2fDestination) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);

    x.guarded(InstructionKind::Convert(ops::Convert {
        kind: Conversion::F2F {
            source,
            destination,
            ftz: b(r, 44, 1) != 0,
            integer_round: (b(r, 42, 1) == 1).then(|| rounding(b(r, 39, 2))),
            abs: b(r, 49, 1) != 0,
            negate: b(r, 45, 1) != 0,
            saturate: b(r, 50, 1) != 0,
        },
        dst: d,
        src: float_form_src(r),
    }))
}
fn f2f(x: Word) -> Instruction {
    f2f_to(x, F2fDestination::Float32)
}
// F2F.F64.F32 (sizes 0xb): the binary32 source with abs/neg and SAT,
// widened exactly into the pair at D (F2fDestination::Float64).
fn f2f_double(x: Word) -> Instruction {
    f2f_to(x, F2fDestination::Float64)
}
// F2F.F32.F64 and F2F.F64.F64: the binary64 source by form (Ryujinx
// InstEmitConversion.cs:344-396), abs49/neg45, then the narrowing or the
// same-type integral rounding, then SAT50 (:121-162). nvdisasm prints
// rounding39..40 on the narrowing and integer rounding39..42 on F64.
fn f2f_narrow(x: Word) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::DoubleToFloat(ops::DoubleToFloat {
        dst: Register(b(r, 0, 8) as u8),
        src: double_b(r),
        modifier: double_source_modifier(r, 49, 45),
        round: fmul_rounding(r),
        saturate: saturation(r, 50),
    }))
}
fn f2f_double_round(x: Word) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::DoubleRound(ops::DoubleRound {
        dst: pair(r, 0),
        src: double_b(r),
        modifier: double_source_modifier(r, 49, 45),
        integer_round: (b(r, 42, 1) == 1).then(|| rounding(b(r, 39, 2))),
        saturate: saturation(r, 50),
    }))
}
fn f2f_to(x: Word, destination: F2fDestination) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);

    x.guarded(InstructionKind::Convert(ops::Convert {
        kind: Conversion::F2F {
            source: F2fSource::Float32,
            destination,
            ftz: b(r, 44, 1) != 0,
            integer_round: (b(r, 42, 1) == 1).then(|| rounding(b(r, 39, 2))),
            abs: b(r, 49, 1) != 0,
            negate: b(r, 45, 1) != 0,
            saturate: b(r, 50, 1) != 0,
        },
        dst: d,
        src: float_form_src(r),
    }))
}
const F2F_FIELDS0: Fields<5> = Fields([
    Field::bits(8, 4),
    Field::bits(42, 1),
    Field::bits(39, 2),
    Field::bits(41, 1),
    Field::bits(44, 1),
]);
const fn f2f_rule0(r: u64) -> u8 {
    f2f_register_rule0(r)
}
const F2F_FORMS0: [u8; F2F_FIELDS0.len()] = forms!(F2F_FIELDS0, f2f_rule0);
/// As F2F_REGISTER_FIELDS1 without the register form's 28..38.
const F2F_FIELDS1: Fields<6> = Fields([
    Field::zero(47, 1),
    Field::bits(10, 2),
    Field::zero(48, 1),
    Field::zero(12, 4),
    Field::zero(43, 1),
    Field::zero(46, 1),
]);
const fn f2f_rule1(r: u64) -> u8 {
    (b(r, 47, 1) == 0
        && (b(r, 10, 2) == 3
            || (b(r, 48, 1) == 0 && b(r, 12, 4) == 0 && b(r, 43, 1) == 0 && b(r, 46, 1) == 0)))
        as u8
}
const F2F_FORMS1: [u8; F2F_FIELDS1.len()] = forms!(F2F_FIELDS1, f2f_rule1);
fn read_f2f(x: Word) -> Instruction {
    let paths: [Decode; 9] = [
        unknown,
        f2f,
        f2f_half,
        f2f_half_high,
        f2f_double,
        f2f_narrow,
        f2f_double_round,
        f2f_half_widen,
        f2f_half_high_widen,
    ];
    paths[(F2F_FORMS0[F2F_FIELDS0.index(x.r)] * F2F_FORMS1[F2F_FIELDS1.index(x.r)]) as usize](x)
}
// F2F immediate (Ryujinx InstTable.cs, MIT: 0x38a8-0x38af,
// 0x39a8-0x39af; InstDecoders.cs:1610-1627; envytools gm107.c:2117):
// the f20 immediate at 20..38, sign 56, under the cbuf form's field
// rules. F32 to F32 applies abs/neg, integer rounding, then saturation
// (InstEmitConversion.cs:22-29,121-162); a binary64 source is the D20
// high word (:380-396). F16 halves of an f20 immediate and its widening
// to F64 are not lowered and stay Unknown.
fn read_f2f_immediate(x: Word) -> Instruction {
    let paths: [Decode; 9] = [
        unknown,
        f2f,
        unknown,
        unknown,
        unknown,
        f2f_narrow,
        f2f_double_round,
        unknown,
        unknown,
    ];
    paths[(F2F_FORMS0[F2F_FIELDS0.index(x.r)] * F2F_FORMS1[F2F_FIELDS1.index(x.r)]) as usize](x)
}
// I2I: envytools envydis/gm107.c:1993,2067,2113 includes the
// opcode modifier bits, so 0x5ce2 is the same family with abs49.
// Destination size/sign:1208-1216; source size/sign:1218-1226;
// byte41..42:1200-1205; neg45/CC47/abs49/sat50:1993.
// Mesa sm50.rs:1889-1922 confirms all fields and the source forms.
// Ryujinx InstEmitConversion.cs:286-342: extract, abs/neg, clamp,
// copy the full 32-bit result and write Z/N. No destination truncation.
// Reject invalid sizes and selections spanning beyond the source word
// instead of emitting undefined SPIR-V bitfield extraction. A 16-bit
// source selects H0 or H1 only (see I2F).
// Unnamed fields in gm107.c:1993,2067,2113 (14..15, 39..40, 43..44,
// 46, 48, and register 28..38) stay Unknown; Ryujinx InstI2iR/I/C
// (InstDecoders.cs:2731-2781) read none of them.
const I2I_REGISTER_FIELDS0: Fields<3> =
    Fields([Field::bits(8, 2), Field::bits(10, 2), Field::bits(41, 2)]);
const fn i2i_register_rule0(r: u64) -> u8 {
    (b(r, 8, 2) <= 2
        && b(r, 10, 2) <= 2
        && (b(r, 10, 2) == 0
            || b(r, 10, 2) == 1 && b(r, 41, 1) == 0
            || b(r, 10, 2) == 2 && b(r, 41, 2) == 0)) as u8
}
const I2I_REGISTER_FORMS0: [u8; I2I_REGISTER_FIELDS0.len()] =
    forms!(I2I_REGISTER_FIELDS0, i2i_register_rule0);
const I2I_REGISTER_FIELDS1: Fields<6> = Fields([
    Field::zero(14, 2),
    Field::zero(39, 2),
    Field::zero(43, 2),
    Field::zero(46, 1),
    Field::zero(48, 1),
    Field::zero(28, 11),
]);
const fn i2i_register_rule1(r: u64) -> u8 {
    (b(r, 14, 2) == 0
        && b(r, 39, 2) == 0
        && b(r, 43, 2) == 0
        && b(r, 46, 1) == 0
        && b(r, 48, 1) == 0
        && b(r, 28, 11) == 0) as u8
}
const I2I_REGISTER_FORMS1: [u8; I2I_REGISTER_FIELDS1.len()] =
    forms!(I2I_REGISTER_FIELDS1, i2i_register_rule1);
fn read_i2i_register(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, i2i];
    paths[(I2I_REGISTER_FORMS0[I2I_REGISTER_FIELDS0.index(x.r)]
        & I2I_REGISTER_FORMS1[I2I_REGISTER_FIELDS1.index(x.r)]) as usize](x)
}
fn i2i(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);

    x.guarded(InstructionKind::Convert(ops::Convert {
        kind: Conversion::I2I {
            src_signed: b(r, 13, 1) != 0,
            dst_signed: b(r, 12, 1) != 0,
            src_size: b(r, 10, 2) as u8,
            dst_size: b(r, 8, 2) as u8,
            byte: b(r, 41, 2) as u8,
            abs: b(r, 49, 1) != 0,
            negate: b(r, 45, 1) != 0,
            saturate: b(r, 50, 1) != 0,
            cc: b(r, 47, 1) != 0,
        },
        dst: d,
        src: int_form_src(r),
    }))
}
const I2I_FIELDS0: Fields<3> = Fields([Field::bits(8, 2), Field::bits(10, 2), Field::bits(41, 2)]);
const fn i2i_rule0(r: u64) -> u8 {
    (b(r, 8, 2) <= 2
        && b(r, 10, 2) <= 2
        && (b(r, 10, 2) == 0
            || b(r, 10, 2) == 1 && b(r, 41, 1) == 0
            || b(r, 10, 2) == 2 && b(r, 41, 2) == 0)) as u8
}
const I2I_FORMS0: [u8; I2I_FIELDS0.len()] = forms!(I2I_FIELDS0, i2i_rule0);
const I2I_FIELDS1: Fields<5> = Fields([
    Field::zero(14, 2),
    Field::zero(39, 2),
    Field::zero(43, 2),
    Field::zero(46, 1),
    Field::zero(48, 1),
]);
const fn i2i_rule1(r: u64) -> u8 {
    (b(r, 14, 2) == 0
        && b(r, 39, 2) == 0
        && b(r, 43, 2) == 0
        && b(r, 46, 1) == 0
        && b(r, 48, 1) == 0) as u8
}
const I2I_FORMS1: [u8; I2I_FIELDS1.len()] = forms!(I2I_FIELDS1, i2i_rule1);
fn read_i2i(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, i2i];
    paths[(I2I_FORMS0[I2I_FIELDS0.index(x.r)] & I2I_FORMS1[I2I_FIELDS1.index(x.r)]) as usize](x)
}
// Register LOP3: envytools envydis/gm107.c:2020 (mask fff8),
// registers0/8/20/39:203-207, LUT28..36:238, predicate mode36..38:
// 1391-1396, destination48..51:377, X38 and CC47:2020.
// Ryujinx InstDecoders.cs:3735-3750 corroborates every field;
// InstEmitIntegerLogical.cs:118-163 writes the LUT, predicate and
// Z/N, with extended Z propagation in InstEmitAluHelper.cs:95-121.
// Unassigned bits and noncanonical discarded predicate modes stay
// Unknown. X without CC cannot affect an output, so reject it too.
fn lop3_register(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::Lop3(ops::Lop3 {
        dst: d,
        lhs: l,
        rhs: Source::Register(Register(b(r, 20, 8) as u8)),
        third: Register(b(r, 39, 8) as u8),
        lut: b(r, 28, 8) as u8,
        predicate_output: predicate_output(b(r, 36, 2), b(r, 48, 3)),
        extended: b(r, 38, 1) != 0,
        cc: b(r, 47, 1) != 0,
    }))
}
const LOP3_REGISTER_FIELDS: Fields<5> = Fields([
    Field::bits(51, 1),
    Field::bits(48, 3),
    Field::bits(36, 2),
    Field::bits(38, 1),
    Field::bits(47, 1),
]);
const fn lop3_register_rule(r: u64) -> u8 {
    (b(r, 51, 1) == 0
        && (b(r, 48, 3) != 7 || b(r, 36, 2) == 0)
        && (b(r, 38, 1) == 0 || b(r, 47, 1) != 0)) as u8
}
const LOP3_REGISTER_FORMS: [u8; LOP3_REGISTER_FIELDS.len()] =
    forms!(LOP3_REGISTER_FIELDS, lop3_register_rule);
fn read_lop3_register(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, lop3_register];
    paths[(LOP3_REGISTER_FORMS[LOP3_REGISTER_FIELDS.index(x.r)]) as usize](x)
}
// envytools envydis/gm107.c:253,2108: family mask 0xfc00,
// low19 at 20..39 plus sign56, LUT48..56, CC47. The table omits C;
// Ryujinx InstDecoders.cs:3756-3764 and InstEmitIntegerLogical.cs:
// 65-73 establish C39..47, A/register, B/immediate, C/register order.
// Prefixes 0x3f80/0x3f59 set sign56 and X57, with LUT80/59.
// X57 remains Unknown: Ryujinx decodes it (:3763) but passes false
// to EmitLop3 (InstEmitIntegerLogical.cs:73), and envytools :2108
// has no X token. No encoding proof justifies ignoring this field.
fn lop3_immediate(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::Lop3(ops::Lop3 {
        dst: d,
        lhs: l,
        rhs: int_form_src(r),
        third: Register(b(r, 39, 8) as u8),
        lut: b(r, 48, 8) as u8,
        predicate_output: None,
        extended: false,
        cc: b(r, 47, 1) == 1,
    }))
}
const LOP3_IMMEDIATE_FIELDS: Fields<1> = Fields([Field::bits(57, 1)]);
const fn lop3_immediate_rule(r: u64) -> u8 {
    (b(r, 57, 1) == 0) as u8
}
const LOP3_IMMEDIATE_FORMS: [u8; LOP3_IMMEDIATE_FIELDS.len()] =
    forms!(LOP3_IMMEDIATE_FIELDS, lop3_immediate_rule);
fn read_lop3_immediate(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, lop3_immediate];
    paths[(LOP3_IMMEDIATE_FORMS[LOP3_IMMEDIATE_FIELDS.index(x.r)]) as usize](x)
}
// Cbuf LOP3: envytools envydis/gm107.c:2169 (mask fe00),
// registers0/8/39 (:203-218), LUT48..55, X56 and CC47. Cbuf
// offset20..33 in words/bank34..38: Mesa sm50.rs:361-372;
// Ryujinx InstDecoders.cs:3771-3780 independently agrees. NAK
// sm50.rs:3290-3363 has no SM50 LOP3 encoder.
// InstEmitIntegerLogical.cs:76-84,118-142 uses the same three-input
// truth table and Z/N flags.
// Its hard-coded X=false does not establish extended semantics.
fn lop3_constant(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::Lop3(ops::Lop3 {
        extended: false,
        dst: d,
        lhs: l,
        rhs: Source::ConstantBuffer {
            binding: b(r, 34, 5) as u8,
            offset: (b(r, 20, 14) * 4) as u16,
        },
        third: Register(b(r, 39, 8) as u8),
        lut: b(r, 48, 8) as u8,
        cc: b(r, 47, 1) != 0,
        predicate_output: None,
    }))
}
const LOP3_CONSTANT_FIELDS: Fields<1> = Fields([Field::bits(56, 1)]);
const fn lop3_constant_rule(r: u64) -> u8 {
    (b(r, 56, 1) == 0) as u8
}
const LOP3_CONSTANT_FORMS: [u8; LOP3_CONSTANT_FIELDS.len()] =
    forms!(LOP3_CONSTANT_FIELDS, lop3_constant_rule);
fn read_lop3_constant(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, lop3_constant];
    paths[(LOP3_CONSTANT_FORMS[LOP3_CONSTANT_FIELDS.index(x.r)]) as usize](x)
}
// LOP (Ryujinx InstTable.cs:209-211, MIT: 0x5c40-0x5c47,
// 0x4c40-0x4c47, 0x3840-0x3847, 0x3940-0x3947; InstDecoders.cs:
// 3680-3796; envytools gm107.c:2009,2083,2129; sm50.rs OpLop2): op 41..42,
// srcs[0] invert 39, srcs[1] invert 40, .X 43, predicate output mode
// 44..46 and predicate destination 48..51 (PT = no write), .CC 47.
// Predicate modes are 0 F, 1 T, 2 Z and 3 NZ (envytools
// gm107.c:1357-1362; Ryujinx InstEmitIntegerLogical.cs:145-160);
// PRED48 is its destination (gm107.c:2009,2083,2129).
fn lop(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::Lop(ops::Lop {
        dst: d,
        lhs: l,
        rhs: int_form_src(r),
        op: logic_op(b(r, 41, 2)),
        invert: [b(r, 39, 1) == 1, b(r, 40, 1) == 1],
        predicate_output: predicate_output(b(r, 44, 2), b(r, 48, 3)),
        extended: b(r, 43, 1) == 1,
        cc: b(r, 47, 1) == 1,
    }))
}
// EmitLop (InstEmitIntegerLogical.cs:87-115) passes X to SetZnFlags
// (InstEmitAluHelper.cs:95-121), which ANDs Z with the entry Z. Every
// field value has a meaning; nvdisasm and InstLopR/I/C ignore bit 46.
// LOP32I (Ryujinx InstTable.cs:215, MIT: 0x0400-0x07ff;
// InstDecoders.cs:3680-3796; gm107.c:2168, tab0400_0 :1807-1812): 32-bit
// immediate 20..51, .CC 52, op 53..54 (3 is PASS_B), A invert 55,
// B invert 56, .X 57. nvdisasm accepts every value of each field.
fn lop32i(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::Lop(ops::Lop {
        dst: d,
        lhs: l,
        rhs: Source::Immediate(b(r, 20, 32)),
        op: logic_op(b(r, 53, 2)),
        invert: [b(r, 55, 1) == 1, b(r, 56, 1) == 1],
        predicate_output: None,
        extended: b(r, 57, 1) == 1,
        cc: b(r, 52, 1) == 1,
    }))
}
// SHL (gm107.c:2008,2128; sm50.rs OpShl): .W 39, .X 43, .CC 47.
fn shl(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::Shl(ops::Shl {
        dst: d,
        lhs: l,
        rhs: int_form_src(r),
        wrap: b(r, 39, 1) == 1,
    }))
}
// Hybrid v2 bundles Bd7bd733b76367966, B6cb61cd43db49f68 and
// Babc96170ca3c1b20 contain six source-locked SHL words whose low three
// opcode bits are outside the canonical encoder value. Envytools rows
// 2008/2082 use a 0xfff8 opcode mask and row 2128 uses 0xfef8, so these
// bits do not change the register, cbuf or immediate semantics. Keep the
// discovery admission exact: no other noncanonical opcode alias is accepted.
const QUALIFIED_SHL_ALIAS_WORDS: [u64; 6] = [
    0x5c4a_0733_d3f7_4209,
    0x5c4a_07b3_d3f7_4209,
    0x4c4a_037a_0367_2672,
    0x4c4a_03fa_0367_2672,
    0x3849_4406_e117_07a6,
    0x3849_4486_e117_07a6,
];
const SHL_FIELDS: Fields<2> = Fields([Field::zero(43, 1), Field::zero(47, 1)]);
const fn shl_rule(r: u64) -> u8 {
    (b(r, 43, 1) == 0 && b(r, 47, 1) == 0) as u8
}
const SHL_FORMS: [u8; SHL_FIELDS.len()] = forms!(SHL_FIELDS, shl_rule);
fn read_shl(x: Word) -> Instruction {
    if QUALIFIED_SHL_ALIAS_WORDS.contains(&x.r) {
        return shl(x);
    }
    // The opcode map routes the complete architectural 0xfff8/fef8 run
    // here so the qualified aliases are reachable. Preserve the old
    // boundary for every other noncanonical low-three-bit value.
    if b(x.r, 48, 3) != 0 {
        return unknown(x);
    }
    let paths: [Decode; 2] = [unknown, shl];
    paths[(SHL_FORMS[SHL_FIELDS.index(x.r)]) as usize](x)
}
// SHR (Ryujinx InstTable.cs:270-272, MIT: 0x5c28-0x5c2f, 0x4c28-0x4c2f,
// 0x3828-0x382f, 0x3928-0x392f; InstDecoders.cs:4458-4504 InstShrR/I/C):
// signed 48, .W 39, .BREV 40, XMode 43..44, .CC 47; bits 49..50 are not
// read. EmitShr (InstEmitShift.cs:202-247) reverses A before the masked
// or clamped shift. nvdisasm SM52 prints XMode 1 as INVALIDSHRXMODE1 and
// 2/3 as .X/.XHI; XMode and .CC are not modeled and stay Unknown.
fn shr(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded({
        InstructionKind::Shr(ops::Shr {
            dst: d,
            lhs: l,
            rhs: int_form_src(r),
            wrap: b(r, 39, 1) == 1,
            signed: b(r, 48, 1) == 1,
            reverse: b(r, 40, 1) == 1,
        })
    })
}
const SHR_FIELDS: Fields<2> = Fields([Field::zero(43, 2), Field::zero(47, 1)]);
const fn shr_rule(r: u64) -> u8 {
    (b(r, 43, 2) == 0 && b(r, 47, 1) == 0) as u8
}
const SHR_FORMS: [u8; SHR_FIELDS.len()] = forms!(SHR_FIELDS, shr_rule);
fn read_shr(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, shr];
    paths[(SHR_FORMS[SHR_FIELDS.index(x.r)]) as usize](x)
}
// LDG/STG masks: envytools/envydis/gm107.c:1898-1899. Destination
// or store data 0..8, address 8..16 and offset 20..44: Mesa
// src/nouveau/compiler/nak/sm50.rs:2609-2611,2665-2667.
// Signed offset: envytools gm107.c:316,331. E 45 selects A64/A32:
// sm50.rs:2402-2410. Size 48..51: sm50.rs:2386-2397.
// Cache 46..48: sm50.rs:2614,2669,2413-2431. Decode all modes;
// the Vulkan access is identical and the hint survives in OpString.
// See docs/design/global-memory-flat.md for the coherence contract.
// Bit 44 has no defined field in these rows, so reject it.
// UB128 is four ordinary words in Ryujinx InstEmitMemory.cs:502-509.
// STG size 7 is undefined (gm107.c:583-591) and stays Unknown.
fn global_load(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);
    let access = GlobalAccess::Load(ls_size(b(r, 48, 3)));
    x.guarded(InstructionKind::GlobalMemory(ops::GlobalMemory {
        access,
        data: d,
        address: l,
        offset: signed(r, 20, 24),
        extended: b(r, 45, 1) == 1,
        cache: b(r, 46, 2) as u8,
    }))
}
const GLOBAL_LOAD_FIELDS: Fields<1> = Fields([Field::bits(44, 1)]);
const fn global_load_rule(r: u64) -> u8 {
    (b(r, 44, 1) == 0) as u8
}
const GLOBAL_LOAD_FORMS: [u8; GLOBAL_LOAD_FIELDS.len()] =
    forms!(GLOBAL_LOAD_FIELDS, global_load_rule);
fn read_global_load(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, global_load];
    paths[(GLOBAL_LOAD_FORMS[GLOBAL_LOAD_FIELDS.index(x.r)]) as usize](x)
}
fn global_store(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);
    let access = GlobalAccess::Store(LS_SIZES2[b(r, 48, 3) as usize]);
    x.guarded(InstructionKind::GlobalMemory(ops::GlobalMemory {
        access,
        data: d,
        address: l,
        offset: signed(r, 20, 24),
        extended: b(r, 45, 1) == 1,
        cache: b(r, 46, 2) as u8,
    }))
}
const GLOBAL_STORE_FIELDS: Fields<2> = Fields([Field::bits(44, 1), Field::bits(48, 3)]);
const fn global_store_rule(r: u64) -> u8 {
    (b(r, 44, 1) == 0 && b(r, 48, 3) < 7) as u8
}
const GLOBAL_STORE_FORMS: [u8; GLOBAL_STORE_FIELDS.len()] =
    forms!(GLOBAL_STORE_FIELDS, global_store_rule);
fn read_global_store(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, global_store];
    paths[(GLOBAL_STORE_FORMS[GLOBAL_STORE_FIELDS.index(x.r)]) as usize](x)
}
// Global ATOM/RED masks: envytools gm107.c:1903,1905.
// ATOM dst 0..8, data 20..28, type 49..52, op 52..56:
// Mesa sm50.rs:2763-2778. RED data 0..8, type 20..23, op 23..26:
// sm50.rs:2717-2732. Both address 8..16, signed offset 28..48,
// E 48: sm50.rs:2781-2789; gm107.c:321,339,341.
// Ryujinx InstTable.cs (MIT) maps ATOM 0xed00-0xedff and RED
// 0xebf8-0xebff; InstAtom/InstRed (InstDecoders.cs:905-918,4190-4202)
// read no other bits, and nvdisasm ignores RED bits 26..27 and 49..50.
// Add/min/max/bitwise/exchange are implemented, matching Ryujinx
// InstEmitMemory.cs:202-305. INC/DEC, SAFEADD and the float types have
// no implemented mapping here and stay Unknown.
// The 64-bit types are 2 (U64) and 5 (S64) (Ryujinx InstDecoders.cs:
// 13-21 AtomSize). Their data and destination are register pairs whose
// high word is the register with bit 0 set (ir::RegisterPair), so an
// even register or RZ is a pair; an odd one stays Unknown.
/// ATOM (0xed00..0xedff): dst 0..8, data 20..28.
fn atom_of(x: Word, sign: bool, op: u8, width: AtomicWidth) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::GlobalAtomic(ops::GlobalAtomic {
        dst: Some(Register(b(r, 0, 8) as u8)),
        data: Register(b(r, 20, 8) as u8),
        address: Register(b(r, 8, 8) as u8),
        offset: signed(r, 28, 20),
        extended: b(r, 48, 1) != 0,
        signed: sign,
        op,
        width,
    }))
}
/// RED (0xebf8..0xebff): no destination, data 0..8.
fn reduce_of(x: Word, sign: bool, op: u8, width: AtomicWidth) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::GlobalAtomic(ops::GlobalAtomic {
        dst: None,
        data: Register(b(r, 0, 8) as u8),
        address: Register(b(r, 8, 8) as u8),
        offset: signed(r, 28, 20),
        extended: b(r, 48, 1) != 0,
        signed: sign,
        op,
        width,
    }))
}
/// ATOM type 49..52 (1 S32), op 52..56.
fn atom(x: Word) -> Instruction {
    atom_of(x, b(x.r, 49, 3) == 1, b(x.r, 52, 4) as u8, AtomicWidth::B32)
}
/// ATOM type 49..52 (5 S64), op 52..56.
fn atom64(x: Word) -> Instruction {
    atom_of(x, b(x.r, 49, 3) == 5, b(x.r, 52, 4) as u8, AtomicWidth::B64)
}
/// RED type 20..23 (1 S32), op 23..26.
fn reduce(x: Word) -> Instruction {
    reduce_of(x, b(x.r, 20, 3) == 1, b(x.r, 23, 3) as u8, AtomicWidth::B32)
}
/// RED type 20..23 (5 S64), op 23..26.
fn reduce64(x: Word) -> Instruction {
    reduce_of(x, b(x.r, 20, 3) == 5, b(x.r, 23, 3) as u8, AtomicWidth::B64)
}
/// The implemented ops, one bit each: add 0, min 1, max 2, and 5, or 6,
/// xor 7, exchange 8.
const ATOMIC_OPS: u32 = 0b1_1110_0111;
/// The width form of each atomic type: 0 Unknown, 1 32-bit, 2 64-bit.
const ATOMIC_WIDTHS: [u8; 8] = [1, 1, 2, 0, 0, 2, 0, 0];
/// 1 for a register that names a 64-bit pair: even, or RZ.
const PAIRED: [u8; 256] = {
    let mut table = [0; 256];
    let mut register = 0;
    while register < 256 {
        table[register] = (register % 2 == 0 || register == 255) as u8;
        register += 1;
    }
    table
};
const ATOM_FIELDS: Fields<2> = Fields([Field::bits(49, 3), Field::bits(52, 4)]);
const fn atom_rule(r: u64) -> u8 {
    ((ATOMIC_OPS >> b(r, 52, 4)) & 1) as u8 * ATOMIC_WIDTHS[b(r, 49, 3) as usize]
}
const ATOM_FORMS: [u8; ATOM_FIELDS.len()] = forms!(ATOM_FIELDS, atom_rule);
const REDUCE_FIELDS: Fields<2> = Fields([Field::bits(20, 3), Field::bits(23, 3)]);
const fn reduce_rule(r: u64) -> u8 {
    ((ATOMIC_OPS >> b(r, 23, 3)) & 1) as u8 * ATOMIC_WIDTHS[b(r, 20, 3) as usize]
}
const REDUCE_FORMS: [u8; REDUCE_FIELDS.len()] = forms!(REDUCE_FIELDS, reduce_rule);
/// By width form, then by whether the 64-bit registers are pairs.
fn read_atom(x: Word) -> Instruction {
    let paths: [[Decode; 2]; 3] = [[unknown, unknown], [atom, atom], [unknown, atom64]];
    let paired = PAIRED[b(x.r, 0, 8) as usize] & PAIRED[b(x.r, 20, 8) as usize];
    paths[ATOM_FORMS[ATOM_FIELDS.index(x.r)] as usize][paired as usize](x)
}
fn read_reduce(x: Word) -> Instruction {
    let paths: [[Decode; 2]; 3] = [[unknown, unknown], [reduce, reduce], [unknown, reduce64]];
    paths[REDUCE_FORMS[REDUCE_FIELDS.index(x.r)] as usize][PAIRED[b(x.r, 0, 8) as usize] as usize](
        x,
    )
}
// Global packed CAS: envytools gm107.c:1897,667-670; Mesa
// sm50.rs:2733-2761 and older nv50_ir_emit_gm107.cpp:2646-2678.
// The 0xee00 opcode plus subOp 15 is 0xeef0. Type bit 49 is
// 0 U32 / 1 U64; layout 50..52 is 0 packed, 1/2 zero operand.
// Only packed U32 is implemented; other layouts/types stay Unknown.
// Address, E, signed immediate and predicates are as ATOM above.
fn global_compare_exchange(x: Word) -> Instruction {
    let r = x.r;

    x.guarded(InstructionKind::GlobalAtomic(ops::GlobalAtomic {
        dst: Some(Register(b(r, 0, 8) as u8)),
        data: Register(b(r, 20, 8) as u8),
        address: Register(b(r, 8, 8) as u8),
        offset: signed(r, 28, 20),
        extended: b(r, 48, 1) != 0,
        signed: false,
        op: 15,
        width: AtomicWidth::B32,
    }))
}
/// Data R254 and RZ (bits 21..28 all set) stay Unknown.
const GLOBAL_COMPARE_EXCHANGE_FIELDS: Fields<2> = Fields([Field::zero(49, 3), Field::ones(21, 7)]);
const fn global_compare_exchange_rule(r: u64) -> u8 {
    (b(r, 49, 3) == 0 && b(r, 21, 7) != 0x7f) as u8
}
const GLOBAL_COMPARE_EXCHANGE_FORMS: [u8; GLOBAL_COMPARE_EXCHANGE_FIELDS.len()] =
    forms!(GLOBAL_COMPARE_EXCHANGE_FIELDS, global_compare_exchange_rule);
fn read_global_compare_exchange(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, global_compare_exchange];
    paths[(GLOBAL_COMPARE_EXCHANGE_FORMS[GLOBAL_COMPARE_EXCHANGE_FIELDS.index(x.r)]) as usize](x)
}
// Mesa sm50.rs:2603-2614,2659-2669; envytools gm107.c:316,
// 583-607,1892-1895: register8..16, signed offset20..44,
// width48..51. LDS.U44 and reserved options remain Unknown.
// Local cache hints LU/CI/CV (bits 44..45, envytools gm107.c:
// 602-607,1895) do not change results for per-thread local memory.
fn local_load(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);
    let mem_type = LS_SIZES2[b(r, 48, 3) as usize];
    x.guarded(InstructionKind::LocalLoad(ops::LocalLoad {
        dst: d,
        address: l,
        offset: signed(r, 20, 24),
        mem_type,
    }))
}
const LOCAL_LOAD_FIELDS: Fields<2> = Fields([Field::zero(46, 2), Field::bits(48, 3)]);
const fn local_load_rule(r: u64) -> u8 {
    (b(r, 46, 2) == 0 && b(r, 48, 3) < 7) as u8
}
const LOCAL_LOAD_FORMS: [u8; LOCAL_LOAD_FIELDS.len()] = forms!(LOCAL_LOAD_FIELDS, local_load_rule);
fn read_local_load(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, local_load];
    paths[(LOCAL_LOAD_FORMS[LOCAL_LOAD_FIELDS.index(x.r)]) as usize](x)
}
// LDS.U (bit 44, envytools gm107.c:1894) is a uniform-address
// hint: every lane reads the same address; the value is unchanged.
fn shared_load(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);
    let mem_type = LS_SIZES2[b(r, 48, 3) as usize];
    x.guarded(InstructionKind::SharedLoad(ops::SharedLoad {
        dst: d,
        address: l,
        offset: signed(r, 20, 24),
        mem_type,
    }))
}
const SHARED_LOAD_FIELDS: Fields<2> = Fields([Field::zero(45, 3), Field::bits(48, 3)]);
const fn shared_load_rule(r: u64) -> u8 {
    (b(r, 45, 3) == 0 && b(r, 48, 3) < 7) as u8
}
const SHARED_LOAD_FORMS: [u8; SHARED_LOAD_FIELDS.len()] =
    forms!(SHARED_LOAD_FIELDS, shared_load_rule);
fn read_shared_load(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, shared_load];
    paths[(SHARED_LOAD_FORMS[SHARED_LOAD_FIELDS.index(x.r)]) as usize](x)
}

// STL (Ryujinx InstTable.cs, MIT: 0xef50-0xef57; InstDecoders.cs
// InstStl): data 0..7, address 8..15, signed offset 20..43, cache
// 44..45 (default, CG, CS, WT; envytools gm107.c:594-599), size
// 48..50 (7 is invalid). nvdisasm ignores bits 46..47.
const LOCAL_STORE_CACHES: [LocalStoreCache; 4] = [
    LocalStoreCache::Default,
    LocalStoreCache::Cg,
    LocalStoreCache::Cs,
    LocalStoreCache::Wt,
];
fn local_store(x: Word) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::LocalStore(ops::LocalStore {
        data: Register(b(r, 0, 8) as u8),
        address: Register(b(r, 8, 8) as u8),
        offset: signed(r, 20, 24),
        mem_type: LS_SIZES2[b(r, 48, 3) as usize],
        cache: LOCAL_STORE_CACHES[b(r, 44, 2) as usize],
    }))
}
const LOCAL_STORE_FIELDS: Fields<1> = Fields([Field::bits(48, 3)]);
const fn local_store_rule(r: u64) -> u8 {
    (b(r, 48, 3) < 7) as u8
}
const LOCAL_STORE_FORMS: [u8; LOCAL_STORE_FIELDS.len()] =
    forms!(LOCAL_STORE_FIELDS, local_store_rule);
fn read_local_store(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, local_store];
    paths[LOCAL_STORE_FORMS[LOCAL_STORE_FIELDS.index(x.r)] as usize](x)
}

fn shared_store(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);
    let mem_type = LS_SIZES2[b(r, 48, 3) as usize];
    x.guarded(InstructionKind::SharedStore(ops::SharedStore {
        data: d,
        address: l,
        offset: signed(r, 20, 24),
        mem_type,
    }))
}
const SHARED_STORE_FIELDS: Fields<2> = Fields([Field::zero(44, 4), Field::bits(48, 3)]);
const fn shared_store_rule(r: u64) -> u8 {
    (b(r, 44, 4) == 0 && b(r, 48, 3) < 7) as u8
}
const SHARED_STORE_FORMS: [u8; SHARED_STORE_FIELDS.len()] =
    forms!(SHARED_STORE_FIELDS, shared_store_rule);
fn read_shared_store(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, shared_store];
    paths[(SHARED_STORE_FORMS[SHARED_STORE_FIELDS.index(x.r)]) as usize](x)
}
// ATOMS.CAS (Ryujinx InstTable.cs, MIT: 0xee00-0xee7f, InstAtomsCas
// InstDecoders.cs:948-957; Ryujinx does not emit it): dst 0..7,
// address 8..15, B 20..27, BcRz 28..29, signed word offset 30..51, and
// sub-operation 52..55, 4 for the 32-bit CAS (0xee40-0xee4f; Mesa NAK
// sm50.rs:2695-2708,2792-2805,2827-2830). nvdisasm prints BcRz 0 as the
// comparator B with the replacement B + 1, 1 as B with RZ, 2 as RZ with
// B; 3 is rejected. B + 1 needs B below RZ. The CAST, SPIN and 64-bit
// sub-operations stay Unknown. The emitter's memory guard keeps an
// inactive CAS from touching shared memory.
fn shared_compare_exchange_of(x: Word, compare: Register, replacement: Register) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::SharedCompareExchange(
        ops::SharedCompareExchange {
            dst: Register(b(r, 0, 8) as u8),
            address: Register(b(r, 8, 8) as u8),
            offset: ((b(r, 30, 22) << 10) as i32 >> 10) << 2,
            compare,
            replacement,
        },
    ))
}
/// BcRz 0: comparator B, replacement B + 1.
fn shared_compare_exchange(x: Word) -> Instruction {
    let compare = b(x.r, 20, 8) as u8;
    shared_compare_exchange_of(x, Register(compare), Register(compare + 1))
}
/// BcRz 1: comparator B, replacement RZ.
fn shared_compare_exchange_rz_replacement(x: Word) -> Instruction {
    shared_compare_exchange_of(x, Register(b(x.r, 20, 8) as u8), Register(255))
}
/// BcRz 2: comparator RZ, replacement B.
fn shared_compare_exchange_rz_compare(x: Word) -> Instruction {
    shared_compare_exchange_of(x, Register(255), Register(b(x.r, 20, 8) as u8))
}
const SHARED_CAS_FIELDS: Fields<2> = Fields([Field::bits(28, 2), Field::ones(20, 8)]);
const fn shared_cas_rule(r: u64) -> u8 {
    (b(r, 28, 2) != 3 && (b(r, 28, 2) != 0 || b(r, 20, 8) != 255)) as u8 * (1 + b(r, 28, 2) as u8)
}
const SHARED_CAS_FORMS: [u8; SHARED_CAS_FIELDS.len()] = forms!(SHARED_CAS_FIELDS, shared_cas_rule);
fn read_shared_compare_exchange(x: Word) -> Instruction {
    let paths: [Decode; 4] = [
        unknown,
        shared_compare_exchange,
        shared_compare_exchange_rz_replacement,
        shared_compare_exchange_rz_compare,
    ];
    paths[(SHARED_CAS_FORMS[SHARED_CAS_FIELDS.index(x.r)]) as usize](x)
}
/// ATOMS op 52..56. Ops 3, 4 and 9..15 have no mapping; the form table
/// leaves them Unknown, so their entries are never read.
const SHARED_ATOMIC_OPS: [AtomicOp; 16] = [
    AtomicOp::Add,
    AtomicOp::Min,
    AtomicOp::Max,
    AtomicOp::Add,
    AtomicOp::Add,
    AtomicOp::And,
    AtomicOp::Or,
    AtomicOp::Xor,
    AtomicOp::Exchange,
    AtomicOp::Add,
    AtomicOp::Add,
    AtomicOp::Add,
    AtomicOp::Add,
    AtomicOp::Add,
    AtomicOp::Add,
    AtomicOp::Add,
];
// ATOMS fields: envytools gm107.c:697-717 and 1904; Mesa sm50.rs:2807-2834.
fn shared_atomic(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded({
        InstructionKind::Atomic(ops::Atomic {
            dst: d,
            address: l,
            // envytools gm107.c:322 atommem1_imm: bits 30..51, signed,
            // stored shifted right by 2, so the byte offset is the
            // sign-extended field times 4.
            offset: (((b(r, 30, 22) << 10) as i32 >> 10) << 2) as u32,
            data: Register(b(r, 20, 8) as u8),
            op: SHARED_ATOMIC_OPS[b(r, 52, 4) as usize],
            signed: b(r, 28, 2) == 1,
        })
    })
}
const SHARED_ATOMIC_FIELDS: Fields<2> = Fields([Field::bits(28, 2), Field::bits(52, 4)]);
const fn shared_atomic_rule(r: u64) -> u8 {
    (b(r, 28, 2) <= 1 && (b(r, 52, 4) <= 2 || (b(r, 52, 4) >= 5 && b(r, 52, 4) <= 8))) as u8
}
const SHARED_ATOMIC_FORMS: [u8; SHARED_ATOMIC_FIELDS.len()] =
    forms!(SHARED_ATOMIC_FIELDS, shared_atomic_rule);
fn read_shared_atomic(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, shared_atomic];
    paths[(SHARED_ATOMIC_FORMS[SHARED_ATOMIC_FIELDS.index(x.r)]) as usize](x)
}
// Mesa OpMemBar (sm50.rs:2986-3000) encodes the scope in 8..10;
// envytools gm107.c:1888 identifies 0xef98 as MEMBAR. Ryujinx
// InstEmitBarrier.cs:32-44 emits a group barrier for Cta (0) and a
// memory barrier for Gl, Sys and Vc (InstDecoders.cs:479-485).
fn group_memory_barrier(x: Word) -> Instruction {
    x.guarded(InstructionKind::GroupMemoryBarrier(ops::GroupMemoryBarrier))
}
const GROUP_MEMORY_BARRIER_FIELDS: Fields<2> = Fields([Field::bits(10, 1), Field::bits(8, 2)]);
const fn group_memory_barrier_rule(r: u64) -> u8 {
    (b(r, 10, 1) == 0 && b(r, 8, 2) == 0) as u8
}
const GROUP_MEMORY_BARRIER_FORMS: [u8; GROUP_MEMORY_BARRIER_FIELDS.len()] =
    forms!(GROUP_MEMORY_BARRIER_FIELDS, group_memory_barrier_rule);
fn read_group_memory_barrier(x: Word) -> Instruction {
    let paths: [Decode; 2] = [read_memory_barrier, group_memory_barrier];
    paths[(GROUP_MEMORY_BARRIER_FORMS[GROUP_MEMORY_BARRIER_FIELDS.index(x.r)]) as usize](x)
}

fn memory_barrier(x: Word) -> Instruction {
    x.guarded(InstructionKind::MemoryBarrier(ops::MemoryBarrier))
}
const MEMORY_BARRIER_FIELDS: Fields<1> = Fields([Field::bits(10, 1)]);
const fn memory_barrier_rule(r: u64) -> u8 {
    (b(r, 10, 1) == 0) as u8
}
const MEMORY_BARRIER_FORMS: [u8; MEMORY_BARRIER_FIELDS.len()] =
    forms!(MEMORY_BARRIER_FIELDS, memory_barrier_rule);
fn read_memory_barrier(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, memory_barrier];
    paths[(MEMORY_BARRIER_FORMS[MEMORY_BARRIER_FIELDS.index(x.r)]) as usize](x)
}
// Mesa OpBar (sm50.rs:3114-3124) uses 0xf0a8 for the sync barrier;
// envytools gm107.c:1882 identifies this family as BAR.
// Mesa writes RZ (0xff) as the barrier register, mode SYNC (32..35 = 0),
// reduction 0 (35..37) and PT as the predicate (39..42).
// BAR.SYNC (envytools gm107.c:1815-1831): mode bits 32..40 = sync
// (0x80 << 32 in envytools' table equals Mesa's 0), barrier id 8..16
// and thread count 20.. as registers or (bits 43/44) immediates;
// only barrier 0 with all threads (RZ or immediate 0) is modeled.
fn control_barrier(x: Word) -> Instruction {
    x.guarded(InstructionKind::ControlBarrier(ops::ControlBarrier))
}
const CONTROL_BARRIER_FIELDS: Fields<9> = Fields([
    Field::zero(32, 3),
    Field::zero(35, 2),
    Field::bits(39, 4),
    Field::bits(44, 1),
    Field::bits(43, 1),
    Field::zero(8, 8),
    Field::ones(8, 8),
    Field::zero(20, 12),
    Field::ones(20, 8),
]);
const fn control_barrier_rule(r: u64) -> u8 {
    (b(r, 32, 3) == 0
        && b(r, 35, 2) == 0
        && b(r, 39, 4) == 7
        && (b(r, 44, 1) == 1 && b(r, 8, 8) == 0 || b(r, 44, 1) == 0 && b(r, 8, 8) == 0xff)
        && (b(r, 43, 1) == 1 && b(r, 20, 12) == 0 || b(r, 43, 1) == 0 && b(r, 20, 8) == 0xff))
        as u8
}
const CONTROL_BARRIER_FORMS: [u8; CONTROL_BARRIER_FIELDS.len()] =
    forms!(CONTROL_BARRIER_FIELDS, control_barrier_rule);
fn read_control_barrier(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, control_barrier];
    paths[(CONTROL_BARRIER_FORMS[CONTROL_BARRIER_FIELDS.index(x.r)]) as usize](x)
}
// Mesa sm50.rs:3257-3284: stream register 20..28 or signed i20
// 20..39/56, type 39..41, handle 8..16 and destination 0..8.
// envytools gm107.c:1875-1876. Immediate streams 0..3 and register
// RZ (stream zero) are OUTs on their stream. A register or cbuf stream
// is an OUT whose stream a value proof names. Handle updates stay
// Unknown. Reject unmodeled/reserved bits.
fn geometry_out_of(x: Word, stream: u8) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::GeometryOut(ops::GeometryOut {
        emit: b(r, 39, 1) != 0,
        cut: b(r, 40, 1) != 0,
        stream,
    }))
}
fn geometry_out(x: Word) -> Instruction {
    geometry_out_of(x, 0)
}
fn geometry_out_handle(x: Word) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::GeometryOutHandle(ops::GeometryOutHandle {
        dst: Register(b(r, 0, 8) as u8),
        handle: Register(b(r, 8, 8) as u8),
        emit: b(r, 39, 1) != 0,
        cut: b(r, 40, 1) != 0,
        stream: 0,
    }))
}
fn geometry_out_immediate(x: Word) -> Instruction {
    geometry_out_of(x, b(x.r, 20, 19) as u8)
}
fn geometry_out_streamed(x: Word, stream: Source) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::GeometryOutDynamic(
        ops::GeometryOutDynamic {
            emit: b(r, 39, 1) != 0,
            cut: b(r, 40, 1) != 0,
            stream,
        },
    ))
}
fn geometry_out_register(x: Word) -> Instruction {
    geometry_out_streamed(x, Source::Register(Register(b(x.r, 20, 8) as u8)))
}
// Cbuf stream: offset words 20..34 and binding 34..39, Mesa
// sm50.rs:361-372; envytools gm107.c:1907.
fn geometry_out_constant(x: Word) -> Instruction {
    geometry_out_streamed(
        x,
        Source::ConstantBuffer {
            binding: b(x.r, 34, 5) as u8,
            offset: (b(x.r, 20, 14) * 4) as u16,
        },
    )
}
const GEOMETRY_OUT_FIELDS: Fields<6> = Fields([
    Field::ones(0, 8),
    Field::ones(8, 8),
    Field::zero(41, 7),
    Field::bits(39, 2),
    Field::ones(20, 8),
    Field::zero(28, 11),
]);
const fn geometry_out_rule(r: u64) -> u8 {
    let out = (b(r, 0, 8) == 0xff
        && b(r, 8, 8) == 0xff
        && b(r, 41, 7) == 0
        && b(r, 39, 2) != 0
        && b(r, 28, 11) == 0) as u8;
    out * (1 + (b(r, 20, 8) != 0xff) as u8)
}
const GEOMETRY_OUT_FORMS: [u8; GEOMETRY_OUT_FIELDS.len()] =
    forms!(GEOMETRY_OUT_FIELDS, geometry_out_rule);
fn read_geometry_out(x: Word) -> Instruction {
    if x.r == 0xfbe0_0080_0ff7_0000 {
        return geometry_out_handle(x);
    }
    let paths: [Decode; 3] = [unknown, geometry_out, geometry_out_register];
    paths[(GEOMETRY_OUT_FORMS[GEOMETRY_OUT_FIELDS.index(x.r)]) as usize](x)
}
const GEOMETRY_OUT_IMMEDIATE_FIELDS: Fields<5> = Fields([
    Field::ones(0, 8),
    Field::ones(8, 8),
    Field::zero(41, 7),
    Field::bits(39, 2),
    Field::capped(20, 19, 4),
]);
const fn geometry_out_immediate_rule(r: u64) -> u8 {
    (b(r, 0, 8) == 0xff
        && b(r, 8, 8) == 0xff
        && b(r, 41, 7) == 0
        && b(r, 39, 2) != 0
        && b(r, 20, 19) <= 3) as u8
}
const GEOMETRY_OUT_IMMEDIATE_FORMS: [u8; GEOMETRY_OUT_IMMEDIATE_FIELDS.len()] =
    forms!(GEOMETRY_OUT_IMMEDIATE_FIELDS, geometry_out_immediate_rule);
fn read_geometry_out_immediate(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, geometry_out_immediate];
    paths[(GEOMETRY_OUT_IMMEDIATE_FORMS[GEOMETRY_OUT_IMMEDIATE_FIELDS.index(x.r)]) as usize](x)
}
const GEOMETRY_OUT_CONSTANT_FIELDS: Fields<4> = Fields([
    Field::ones(0, 8),
    Field::ones(8, 8),
    Field::zero(41, 7),
    Field::bits(39, 2),
]);
const fn geometry_out_constant_rule(r: u64) -> u8 {
    (b(r, 0, 8) == 0xff && b(r, 8, 8) == 0xff && b(r, 41, 7) == 0 && b(r, 39, 2) != 0) as u8
}
const GEOMETRY_OUT_CONSTANT_FORMS: [u8; GEOMETRY_OUT_CONSTANT_FIELDS.len()] =
    forms!(GEOMETRY_OUT_CONSTANT_FIELDS, geometry_out_constant_rule);
fn read_geometry_out_constant(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, geometry_out_constant];
    paths[(GEOMETRY_OUT_CONSTANT_FORMS[GEOMETRY_OUT_CONSTANT_FIELDS.index(x.r)]) as usize](x)
}
// AL2P: Mesa sm50.rs:2842-2851: dst 0..8, offset 8..16,
// address 20..31, output 32, predicate dst 44..47, size 47..49.
// envytools gm107.c:1887,251,287 calls the immediate signed; use
// only the common nonnegative range. Output, predicate-result and
// vector forms have no proven lowering and remain Unknown.
fn attribute_address(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::AttributeAddress(ops::AttributeAddress {
        dst: d,
        offset: l,
        address: b(r, 20, 10) as u16,
    }))
}
const ATTRIBUTE_ADDRESS_FIELDS: Fields<3> =
    Fields([Field::zero(30, 14), Field::bits(44, 3), Field::bits(47, 1)]);
const fn attribute_address_rule(r: u64) -> u8 {
    (b(r, 30, 14) == 0 && b(r, 44, 3) == 7 && b(r, 47, 1) == 0) as u8
}
const ATTRIBUTE_ADDRESS_FORMS: [u8; ATTRIBUTE_ADDRESS_FIELDS.len()] =
    forms!(ATTRIBUTE_ADDRESS_FIELDS, attribute_address_rule);
fn read_attribute_address(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, attribute_address];
    paths[(ATTRIBUTE_ADDRESS_FORMS[ATTRIBUTE_ADDRESS_FIELDS.index(x.r)]) as usize](x)
}
// No Maxwell ISBEWR opcode is established by Mesa's SM50 dispatch
// (nak/sm50.rs:3288-3363) or envytools gm107.c:1874-2182. Mesa's
// ISBEWR encoder is SM75+ (nak/sm70_encode.rs:4089-4101). Do not
// reinterpret a Maxwell opcode as that instruction or discard it.
// Default input ISBERD: Mesa nv50_ir_emit_gm107.cpp:2549-2553
// writes only dst 0..8 and source 8..16. envytools gm107.c:1886,
// 493-505 defines output 32, skew 31, base 33..35, size 47..49.
// Accept only that exact default form; all modifiers stay Unknown.
fn isbe_read(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::IsbeRead(ops::IsbeRead {
        dst: d,
        offset: l,
    }))
}
const ISBE_READ_FIELDS: Fields<1> = Fields([Field::zero(20, 28)]);
const fn isbe_read_rule(r: u64) -> u8 {
    (b(r, 20, 28) == 0) as u8
}
const ISBE_READ_FORMS: [u8; ISBE_READ_FIELDS.len()] = forms!(ISBE_READ_FIELDS, isbe_read_rule);
fn read_isbe_read(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, isbe_read];
    paths[(ISBE_READ_FORMS[ISBE_READ_FIELDS.index(x.r)]) as usize](x)
}
// Mesa OpALd encodes offset (8..16), address (20..30), patch (31),
// output (32), vtx (39..47), and component count (47..49),
// sm50.rs:2861-2876. The Maxwell emitter confirms the same fields
// and that vtx is an indirect GPR source, nv50_ir_emit_gm107.cpp:
// 2526-2534. RZ is 0xff; register 0 is therefore a real index.
// The address is ten bits, 20..29: Mesa's range 20..30 is
// exclusive, nv50_ir_emit_gm107.cpp:2533 emits ten bits and
// envytools gm107.c:314 reads amem20_imm as {20, 10}. Ryujinx
// InstDecoders.cs:883-887 reads eleven; Mesa and envytools settle
// the encoding, so a word with bit 30 set has no ALD meaning.
const ATTRIBUTE_LOAD_RESERVED_FIELDS: Fields<1> = Fields([Field::bits(30, 1)]);
const fn attribute_load_reserved_rule(r: u64) -> u8 {
    (b(r, 30, 1) != 0) as u8
}
const ATTRIBUTE_LOAD_RESERVED_FORMS: [u8; ATTRIBUTE_LOAD_RESERVED_FIELDS.len()] =
    forms!(ATTRIBUTE_LOAD_RESERVED_FIELDS, attribute_load_reserved_rule);
fn read_attribute_load_reserved(x: Word) -> Instruction {
    let paths: [Decode; 2] = [read_physical_attribute_load, unknown];
    paths[(ATTRIBUTE_LOAD_RESERVED_FORMS[ATTRIBUTE_LOAD_RESERVED_FIELDS.index(x.r)]) as usize](x)
}
// Ryujinx InstDecoders.cs:883-887 makes the access physical
// exactly when the immediate is zero, P is clear and the offset
// register is not RZ.
// Accept scalar physical access only; vector physical results are
// unproven. Mesa nak_private.h:39 / nak_nir.c:422-423 allocates
// patch generics at 0x20 instead of 0x18; the driver supplies the
// producer layout and linked Vulkan locations.
/// Rb (bits 39..46): 0 when RZ, the current vertex.
const VERTEX_REGISTER: Fields<1> = Fields([Field::ones(39, 8)]);
fn output_vertex(r: u64) -> OutputVertex {
    [
        OutputVertex::Current,
        OutputVertex::Lane(Register(b(r, 39, 8) as u8)),
    ][VERTEX_REGISTER.index(r)]
}
/// O: AttributeMap.cs:323-328 has no primitive vertex; Mesa reads Rb
/// as a lane (nak_nir_lower_vtg_io.c:10-49, 86-90).
fn physical_output_attribute_load(x: Word) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::PhysicalOutputAttributeLoad(
        ops::PhysicalOutputAttributeLoad {
            dst: Register(b(r, 0, 8) as u8),
            vertex: output_vertex(r),
            offset: Register(b(r, 8, 8) as u8),
        },
    ))
}
fn physical_attribute_load(x: Word) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::PhysicalAttributeLoad(
        ops::PhysicalAttributeLoad {
            dst: Register(b(r, 0, 8) as u8),
            vertex: [
                PrimitiveVertex::Current,
                PrimitiveVertex::Primitive(Register(b(r, 39, 8) as u8)),
            ][VERTEX_REGISTER.index(r)],
            offset: Register(b(r, 8, 8) as u8),
        },
    ))
}
/// 0 not physical, 1 vector physical (Unknown), 2 input, 3 output.
const PHYSICAL_ATTRIBUTE_LOAD_FIELDS: Fields<5> = Fields([
    Field::zero(20, 10),
    Field::bits(31, 1),
    Field::ones(8, 8),
    Field::zero(47, 2),
    Field::bits(32, 1),
]);
const fn physical_attribute_load_rule(r: u64) -> u8 {
    (b(r, 20, 10) == 0 && b(r, 31, 1) == 0 && b(r, 8, 8) != 0xff) as u8
        * (1 + (b(r, 47, 2) == 0) as u8 * (1 + b(r, 32, 1) as u8))
}
const PHYSICAL_ATTRIBUTE_LOAD_FORMS: [u8; PHYSICAL_ATTRIBUTE_LOAD_FIELDS.len()] =
    forms!(PHYSICAL_ATTRIBUTE_LOAD_FIELDS, physical_attribute_load_rule);
fn read_physical_attribute_load(x: Word) -> Instruction {
    let paths: [Decode; 4] = [
        read_indexed_patch_load,
        unknown,
        physical_attribute_load,
        physical_output_attribute_load,
    ];
    paths[(PHYSICAL_ATTRIBUTE_LOAD_FORMS[PHYSICAL_ATTRIBUTE_LOAD_FIELDS.index(x.r)]) as usize](x)
}
// P with Ra not RZ is an indirect patch load: Mesa adds Ra to the
// immediate (nak_nir_lower_vtg_io.c:203,208-211, nak/sm50.rs:
// 2864-2868). Ryujinx ignores Ra here; Mesa settles the encoding.
// Without P, Ra not RZ is only the physical form (immediate zero,
// read_physical_attribute_load). With a nonzero immediate the word
// has no ALD meaning: nvdisasm SM52 rejects it ("Unrecognized
// operation for functional unit 'uC'"), O or not, while it prints
// the RZ form as a[imm] and the P form as a[Ra+imm]. Ryujinx would
// read it as a static load ignoring Ra; that is not an encoding.
/// 0 Ra is RZ (static), 1 indexed patch (P), 2 Ra without P (none).
fn indexed_patch_load(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);

    x.guarded({
        let base = b(r, 20, 10) as u16;
        let components = (b(r, 47, 2) + 1) as u8;
        let bank = AttributeBank::decode(b(r, 32, 1) != 0);
        InstructionKind::IndexedPatchLoad(ops::IndexedPatchLoad {
            dst: d,
            base,
            index: Register(b(r, 8, 8) as u8),
            components,
            bank,
            lanes: AttributeLanes::patch(base, components, bank),
        })
    })
}
const INDEXED_PATCH_LOAD_FIELDS: Fields<2> = Fields([Field::bits(31, 1), Field::ones(8, 8)]);
const fn indexed_patch_load_rule(r: u64) -> u8 {
    (b(r, 8, 8) != 0xff) as u8 * (2 - b(r, 31, 1) as u8)
}
const INDEXED_PATCH_LOAD_FORMS: [u8; INDEXED_PATCH_LOAD_FIELDS.len()] =
    forms!(INDEXED_PATCH_LOAD_FIELDS, indexed_patch_load_rule);
fn read_indexed_patch_load(x: Word) -> Instruction {
    let paths: [Decode; 3] = [attribute_load, indexed_patch_load, unknown];
    paths[(INDEXED_PATCH_LOAD_FORMS[INDEXED_PATCH_LOAD_FIELDS.index(x.r)]) as usize](x)
}
// Every other ALD is static: the offset register is not read
// (InstEmitAttribute.cs:53-101 uses only the immediate there). Each lane
// is decoded to its Ryujinx attribute (AttributeMap.cs:63-92). A
// span naming no attribute at all is its own kind. Rb selects a
// primitive vertex only without P and O and outside PrimitiveId
// and TessCoord (InstEmitAttribute.cs:25-32,297-302;
// AttributeMap.cs:323-333).
/// P (bit 31) loads the patch whatever O; else O (bit 32) the output.
const ATTRIBUTE_LOADS: [Decode; 4] = [
    attribute_load_vertex,
    attribute_load_patch,
    attribute_load_output,
    attribute_load_patch,
];
fn attribute_load(x: Word) -> Instruction {
    ATTRIBUTE_LOADS[(b(x.r, 31, 1) | b(x.r, 32, 1) << 1) as usize](x)
}
/// A span naming no attribute at all is its own kind.
fn attribute_load_of(x: Word, lanes: AttributeLanes, mapped: InstructionKind) -> Instruction {
    let r = x.r;
    x.guarded(if lanes.all_unmapped() {
        InstructionKind::UnmappedAttributeLoad(ops::UnmappedAttributeLoad {
            dst: Register(b(r, 0, 8) as u8),
            address: b(r, 20, 10) as u16,
            components: (b(r, 47, 2) + 1) as u8,
        })
    } else {
        mapped
    })
}
fn attribute_load_patch(x: Word) -> Instruction {
    let r = x.r;
    let address = b(r, 20, 10) as u16;
    let components = (b(r, 47, 2) + 1) as u8;
    let lanes = AttributeLanes::patch(address, components, AttributeBank::decode(b(r, 32, 1) != 0));
    attribute_load_of(
        x,
        lanes,
        InstructionKind::PatchAttributeLoad(ops::PatchAttributeLoad {
            dst: Register(b(r, 0, 8) as u8),
            address,
            components,
            lanes,
        }),
    )
}
/// Mesa reads O's Rb as a lane (nak_nir_lower_vtg_io.c:10-49,86-90);
/// Ryujinx ignores it.
fn attribute_load_output(x: Word) -> Instruction {
    let r = x.r;
    let address = b(r, 20, 10) as u16;
    let components = (b(r, 47, 2) + 1) as u8;
    let lanes = AttributeLanes::vertex(address, components, AttributeBank::decode(true));
    attribute_load_of(
        x,
        lanes,
        InstructionKind::OutputAttributeLoad(ops::OutputAttributeLoad {
            dst: Register(b(r, 0, 8) as u8),
            vertex: output_vertex(r),
            address,
            components,
            lanes,
        }),
    )
}
/// 1 for PrimitiveId (0x60) and TessCoord (0x2f0, 0x2f4), which have no
/// primitive vertex.
const fn vertexless(address: u16) -> usize {
    matches!(address, 0x60 | 0x2f0 | 0x2f4) as usize
}
fn attribute_load_vertex(x: Word) -> Instruction {
    let r = x.r;
    let address = b(r, 20, 10) as u16;
    let components = (b(r, 47, 2) + 1) as u8;
    let lanes = AttributeLanes::vertex(address, components, AttributeBank::decode(false));
    let current = (1 - VERTEX_REGISTER.index(r)) | vertexless(address);
    attribute_load_of(
        x,
        lanes,
        InstructionKind::AttributeLoad(ops::AttributeLoad {
            dst: Register(b(r, 0, 8) as u8),
            vertex: [
                PrimitiveVertex::Primitive(Register(b(r, 39, 8) as u8)),
                PrimitiveVertex::Current,
            ][current],
            address,
            components,
            lanes,
        }),
    )
}
// AST always stores outputs: envytools gm107.c:1883 has no O
// selector, unlike ALD at :1885. Mesa nv50_ir_emit_gm107.cpp:
// 2538-2545 emits bit32 clear; nak/sm50.rs:2886-2896 sets it.
// Ryujinx InstDecoders.cs:890-902 has no AST O field and
// InstEmitAttribute.cs:105-141 stores outputs for both encodings.
// Do not decode bit32 as a selector or add runtime dispatch.
// Data0..7, offset8..15, vertex39..46, address20..29, patch31,
// count47..48: those Mesa lines; envytools gm107.c:314,325,
// 475-479,1883. Patch forms
// have the same static byte-slot bounds as ALd above. Bit30 is
// rejected for the same ten/eleven-bit address discrepancy.
const GEOMETRY_CURRENT_HANDLE_AST: [u64; 19] = [
    0xeff0_0000_0707_ff01,
    0xeff0_0000_0747_ff02,
    0xeff0_0000_0787_ff03,
    0xeff0_0000_07c7_ff05,
    0xeff0_0000_0807_ff06,
    0xeff0_0000_0847_ff07,
    0xeff0_0000_0647_ff09,
    0xeff0_0000_0907_ff09,
    0xeff0_0000_0707_ff0a,
    0xeff0_0000_0747_ff01,
    0xeff0_0000_0787_ff02,
    0xeff0_0000_07c7_ff03,
    0xeff0_0000_0807_ff05,
    0xeff0_0000_0847_ff06,
    0xeff0_0000_0647_ff07,
    0xeff0_0000_0907_ff07,
    0xeff0_0000_0707_ff09,
    0xeff0_0000_0647_ff0a,
    0xeff0_0000_0907_ff0a,
];
const LEGACY_VARYING_AST: [u64; 49] = [
    0xeff0_7f80_2807_ff2a,
    0xeff0_7f80_2847_ff2b,
    0xeff0_7f80_2887_ff2c,
    0xeff0_7f80_28c7_ff29,
    0xeff0_7f80_2907_ff26,
    0xeff0_7f80_2947_ff25,
    0xeff0_7f80_2987_ff24,
    0xeff0_7f80_29c7_ff23,
    0xeff0_7f80_2a07_ff2a,
    0xeff0_7f80_2a47_ff2b,
    0xeff0_7f80_2a87_ff2c,
    0xeff0_7f80_2ac7_ff29,
    0xeff0_7f80_2b07_ff26,
    0xeff0_7f80_2b47_ff25,
    0xeff0_7f80_2b87_ff24,
    0xeff0_7f80_2bc7_ff23,
    0xeff0_7f80_2e87_ff06,
    0xeff0_7f80_3007_ff22,
    0xeff0_7f80_3047_ff21,
    0xeff0_7f80_3087_ff20,
    0xeff0_7f80_30c7_ff1f,
    0xeff0_7f80_3107_ff1e,
    0xeff0_7f80_3147_ff1d,
    0xeff0_7f80_3187_ff19,
    0xeff0_7f80_31c7_ff1c,
    0xeff0_7f80_3207_ff1b,
    0xeff0_7f80_3247_ff1a,
    0xeff0_7f80_3287_ff15,
    0xeff0_7f80_32c7_ff18,
    0xeff0_7f80_3307_ff17,
    0xeff0_7f80_3347_ff16,
    0xeff0_7f80_3387_ff11,
    0xeff0_7f80_33c7_ff14,
    0xeff0_7f80_3407_ff13,
    0xeff0_7f80_3447_ff12,
    0xeff0_7f80_3487_ff0d,
    0xeff0_7f80_34c7_ff10,
    0xeff0_7f80_3507_ff0f,
    0xeff0_7f80_3547_ff0e,
    0xeff0_7f80_3587_ff11,
    0xeff0_7f80_35c7_ff0c,
    0xeff0_7f80_3607_ff0b,
    0xeff0_7f80_3647_ff0a,
    0xeff0_7f80_3687_ff08,
    0xeff0_7f80_36c7_ff19,
    0xeff0_7f80_3707_ff03,
    0xeff0_7f80_3747_ff02,
    0xeff0_7f80_3787_ff01,
    0xeff0_7f80_37c7_ff00,
];

fn attribute_store(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);

    if GEOMETRY_CURRENT_HANDLE_AST.contains(&r) {
        return x.guarded(InstructionKind::GeometryAttributeStore(
            ops::GeometryAttributeStore {
                data: d,
                address: b(r, 20, 10) as u16,
                handle: Register(b(r, 39, 8) as u8),
            },
        ));
    }

    if r == 0xeff0_7f80_06c7_ff05 {
        return x.guarded(InstructionKind::PointSizeStore(ops::PointSizeStore {
            data: d,
        }));
    }

    if LEGACY_VARYING_AST.contains(&r) {
        return x.guarded(InstructionKind::LegacyVaryingStore(
            ops::LegacyVaryingStore {
                data: d,
                varying: LegacyVarying::decode(b(r, 20, 10) as u16)
                    .expect("exact legacy color lane"),
            },
        ));
    }

    x.guarded(InstructionKind::AttributeStore(ops::AttributeStore {
        data: d,
        address: b(r, 20, 10) as u16,
        offset: (b(r, 8, 8) != 0xff).then(|| Register(b(r, 8, 8) as u8)),
        components: (b(r, 47, 2) + 1) as u8,
        patch: b(r, 31, 1) != 0,
    }))
}
const ATTRIBUTE_STORE_FIELDS0: Fields<6> = Fields([
    Field::ones(39, 8),
    Field::bits(30, 1),
    Field::ones(8, 8),
    Field::zero(20, 10),
    Field::bits(31, 1),
    Field::bits(47, 2),
]);
const fn attribute_store_rule0(r: u64) -> u8 {
    (b(r, 39, 8) == 0xff
        && b(r, 30, 1) == 0
        && (b(r, 8, 8) == 0xff || (b(r, 20, 10) == 0 && b(r, 31, 1) == 0 && b(r, 47, 2) == 0))
        && (b(r, 31, 1) == 0 || b(r, 8, 8) == 0xff)) as u8
}
const ATTRIBUTE_STORE_FORMS0: [u8; ATTRIBUTE_STORE_FIELDS0.len()] =
    forms!(ATTRIBUTE_STORE_FIELDS0, attribute_store_rule0);
const ATTRIBUTE_STORE_FIELDS1: Fields<4> = Fields([
    Field::bits(31, 1),
    Field::zero(20, 2),
    Field::bits(22, 8),
    Field::bits(47, 2),
]);
const fn attribute_store_rule1(r: u64) -> u8 {
    (b(r, 31, 1) == 0 || (b(r, 20, 2) == 0 && b(r, 22, 8) + b(r, 47, 2) < 0x80)) as u8
}
const ATTRIBUTE_STORE_FORMS1: [u8; ATTRIBUTE_STORE_FIELDS1.len()] =
    forms!(ATTRIBUTE_STORE_FIELDS1, attribute_store_rule1);
fn read_attribute_store(x: Word) -> Instruction {
    if GEOMETRY_CURRENT_HANDLE_AST.contains(&x.r) {
        return attribute_store(x);
    }
    let paths: [Decode; 2] = [unknown, attribute_store];
    paths[(ATTRIBUTE_STORE_FORMS0[ATTRIBUTE_STORE_FIELDS0.index(x.r)]
        & ATTRIBUTE_STORE_FORMS1[ATTRIBUTE_STORE_FIELDS1.index(x.r)]) as usize](x)
}
// Bound TEX: envytools envydis/gm107.c:1970,1026-1033,875-880;
// Mesa nv50_ir_emit_gm107.cpp:2908-2924 (NAK sm50.rs:2171-2175
// uses 0x0380, so the older emitter supplies the full c038 opcode).
// D0..7, A8..15, B20..27, dim28..30, mask31..34, table index36..48,
// NODEP49, DC50, AOFFI54, LOD55..57. NDV35, LC58, non-PT fault
// destination51..53 and reserved LOD4/5 stay Unknown.
// LC/fault destination: Ryujinx Decoders/InstDecoders.cs:4841-4842.
// Ryujinx InstEmitTexture.cs:28-39,207-210: bound B begins with LOD/
// offset/comparison, not a register handle. Preserve table identity.
// Without LOD, offset or comparison, B is unused (Ryujinx
// InstEmitTexture.cs:207-210,252-258). Require canonical RZ, matching
// Mesa sm50.rs:2153-2161,201-205, rather than ignoring a source field.
fn texture(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::Texture(ops::Texture {
        op: TexOp::Sample {
            lod: b(r, 55, 3) as u8,
        },
        dst: d,
        src0: l,
        src1: Register(b(r, 20, 8) as u8),
        dim: b(r, 28, 3) as u8,
        mask: b(r, 31, 4) as u8,
        offset: b(r, 54, 1) != 0,
        scalar: false,
        lod_clamp: false,
        multisample: false,
        depth_compare: b(r, 50, 1) != 0,
        nodep: b(r, 49, 1) != 0,
        packed_f16: false,
        scalar_dst: None,
        bound_handle: Some(b(r, 36, 13) as u16),
    }))
}
const TEXTURE_FIELDS: Fields<7> = Fields([
    Field::zero(31, 4),
    Field::bits(28, 3),
    Field::bits(54, 1),
    Field::bits(35, 1),
    Field::bits(55, 3),
    Field::bits(50, 1),
    Field::ones(20, 8),
]);
const fn texture_rule(r: u64) -> u8 {
    (b(r, 31, 4) != 0
        && !(b(r, 28, 3) == 6 && b(r, 54, 1) != 0)
        && b(r, 35, 1) == 0
        && b(r, 28, 3) != 5
        && (b(r, 55, 3) <= 3 || b(r, 55, 3) >= 6)
        && !(b(r, 28, 3) == 4 && b(r, 50, 1) != 0)
        && (b(r, 55, 3) >= 2 || b(r, 54, 1) != 0 || b(r, 50, 1) != 0 || b(r, 20, 8) == 255))
        as u8
}
const TEXTURE_FORMS: [u8; TEXTURE_FIELDS.len()] = forms!(TEXTURE_FIELDS, texture_rule);
fn read_texture(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, texture];
    paths[(TEXTURE_FORMS[TEXTURE_FIELDS.index(x.r)]) as usize](x)
}
// Bindless texture instructions (sm50.rs OpTex 0xdeb8, OpTld 0xdd38,
// OpTld4 0xdef8, OpTxq 0xdf50). Common: dst 0..8, src0 8..16, src1
// 20..28, dim 28..31, channel mask 31..35, nodep 49 (helper skipping,
// nak_nir_lower_tex.c:454). NDV bit 35 on
// TEX/TLD4 changes derivative behavior (sm50.rs:2135-2141,2195),
// so it stays Unknown, as do unimplemented mode bits and predicates.
// envytools gm107.c:916-923 has a THREE-bit LOD mode; accept only
// 0..3 and LBA6/LLA7. Ryujinx InstEmitTexture.cs:263-269,287-291
// maps LBA to bias and LLA to level; retain the full raw selector.
// Mesa sm50.rs:2181-2197 and envytools gm107.c:1962 name no
// operand in bits41..48. LCB40 has no implemented semantics here.
// Invalid mask/dimension/3D comparison/non-array cube offset remain Unknown,
// as for bound TEX (sm50.rs:2193-2197; InstEmitTexture.cs:272-299).
fn texture_bindless(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded({
        InstructionKind::Texture(ops::Texture {
            op: TexOp::Sample {
                lod: b(r, 37, 3) as u8,
            },
            dst: d,
            src0: l,
            src1: Register(b(r, 20, 8) as u8),
            dim: b(r, 28, 3) as u8,
            mask: b(r, 31, 4) as u8,
            offset: b(r, 36, 1) == 1,
            scalar: false,
            packed_f16: false,
            // Ryujinx Decoders/InstDecoders.cs:4862 preserves LCB.
            // Its lowering is unproven here, so emission stays Unknown.
            lod_clamp: b(r, 40, 1) != 0,
            multisample: false,
            depth_compare: b(r, 50, 1) != 0,
            nodep: b(r, 49, 1) != 0,
            scalar_dst: None,
            bound_handle: None,
        })
    })
}
const TEXTURE_BINDLESS_FIELDS: Fields<7> = Fields([
    Field::bits(37, 3),
    Field::zero(40, 9),
    Field::bits(35, 1),
    Field::zero(31, 4),
    Field::bits(28, 3),
    Field::bits(36, 1),
    Field::bits(50, 1),
]);
const fn texture_bindless_rule(r: u64) -> u8 {
    ((b(r, 37, 3) <= 3 || b(r, 37, 3) >= 6)
        && b(r, 40, 9) == 0
        && b(r, 35, 1) == 0
        && b(r, 31, 4) != 0
        && b(r, 28, 3) != 5
        && !(b(r, 28, 3) == 6 && b(r, 36, 1) != 0)
        && !(b(r, 28, 3) == 4 && b(r, 50, 1) != 0)) as u8
}
const TEXTURE_BINDLESS_FORMS: [u8; TEXTURE_BINDLESS_FIELDS.len()] =
    forms!(TEXTURE_BINDLESS_FIELDS, texture_bindless_rule);
fn read_texture_bindless(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, texture_bindless];
    paths[(TEXTURE_BINDLESS_FORMS[TEXTURE_BINDLESS_FIELDS.index(x.r)]) as usize](x)
}
// Bound TLD 0xdc38 and bindless 0xdd38, LOD 55, CL 54,
// MS 50, NODEP 49: Mesa sm50.rs:2206-2236; envytools
// gm107.c:1965-1966. CL semantics are unimplemented.
// The bindless form has no table index: sm50.rs:2215-2216;
// envytools gm107.c:1965 prints literal 0, unlike U13_36 at :1966.
// Nonzero bits36..48 must not be silently ignored.
fn texture_fetch(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::Texture(ops::Texture {
        op: TexOp::Fetch {
            lod: b(r, 55, 1) == 1,
        },
        dst: d,
        src0: l,
        src1: Register(b(r, 20, 8) as u8),
        dim: b(r, 28, 3) as u8,
        mask: b(r, 31, 4) as u8,
        offset: b(r, 35, 1) == 1,
        scalar: false,
        packed_f16: false,
        lod_clamp: b(r, 54, 1) != 0,
        multisample: b(r, 50, 1) != 0,
        depth_compare: false,
        nodep: b(r, 49, 1) != 0,
        scalar_dst: None,
        bound_handle: (b(r, 56, 1) == 0).then(|| b(r, 36, 13) as u16),
    }))
}
const TEXTURE_FETCH_FIELDS: Fields<9> = Fields([
    Field::bits(54, 1),
    Field::zero(31, 4),
    Field::bits(56, 1),
    Field::zero(36, 13),
    Field::bits(28, 3),
    Field::bits(50, 1),
    Field::bits(55, 1),
    Field::bits(35, 1),
    Field::ones(20, 8),
]);
const fn texture_fetch_rule(r: u64) -> u8 {
    (b(r, 54, 1) == 0
        && b(r, 31, 4) != 0
        && (b(r, 56, 1) == 0 || b(r, 36, 13) == 0)
        && b(r, 28, 3) <= 4
        && (b(r, 50, 1) == 0 || b(r, 28, 3) == 2 || b(r, 28, 3) == 3)
        && (b(r, 56, 1) != 0
            || b(r, 55, 1) != 0
            || b(r, 35, 1) != 0
            || b(r, 50, 1) != 0
            || b(r, 20, 8) == 255)) as u8
}
const TEXTURE_FETCH_FORMS: [u8; TEXTURE_FETCH_FIELDS.len()] =
    forms!(TEXTURE_FETCH_FIELDS, texture_fetch_rule);
fn read_texture_fetch(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, texture_fetch];
    paths[(TEXTURE_FETCH_FORMS[TEXTURE_FETCH_FIELDS.index(x.r)]) as usize](x)
}
// TLD4 (Ryujinx InstTable.cs:303-304, MIT; InstDecoders.cs:4944-4983
// InstTld4/InstTld4B; envytools gm107.c:1961,1969; Mesa sm50.rs:
// 2245-2278): offset mode (bound 54..55, bindless 36..37) is TexOffset
// None, AOFFI or PTP (InstDecoders.cs:698-703); PTP packs four signed
// six-bit 2D offsets in two B words (InstEmitTexture.cs:659-797). Mode
// 3 names no TexOffset and stays Unknown, as do NDV35 and bindless LC40.
const TEXTURE_GATHER_BINDLESS_FIELDS: Fields<3> =
    Fields([Field::bits(40, 1), Field::bits(35, 1), Field::bits(36, 2)]);
const fn texture_gather_bindless_rule(r: u64) -> u8 {
    (b(r, 40, 1) == 0 && b(r, 35, 1) == 0 && b(r, 36, 2) < 3) as u8
}
const TEXTURE_GATHER_BINDLESS_FORMS: [u8; TEXTURE_GATHER_BINDLESS_FIELDS.len()] =
    forms!(TEXTURE_GATHER_BINDLESS_FIELDS, texture_gather_bindless_rule);
fn read_texture_gather_bindless(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, texture_gather_bindless];
    paths[(TEXTURE_GATHER_BINDLESS_FORMS[TEXTURE_GATHER_BINDLESS_FIELDS.index(x.r)]) as usize](x)
}
/// TLD4 offset mode. The form tables send mode 3 to Unknown.
const GATHER_OFFSETS: [GatherOffsets; 3] =
    [GatherOffsets::Zero, GatherOffsets::One, GatherOffsets::Four];
fn texture_gather_of(
    x: Word,
    component: u32,
    offsets: u32,
    bound_handle: Option<u16>,
) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::Texture(ops::Texture {
        op: TexOp::Gather {
            component: component as u8,
            offsets: GATHER_OFFSETS[offsets as usize],
        },
        dst: Register(b(r, 0, 8) as u8),
        src0: Register(b(r, 8, 8) as u8),
        src1: Register(b(r, 20, 8) as u8),
        dim: b(r, 28, 3) as u8,
        mask: b(r, 31, 4) as u8,
        offset: false,
        scalar: false,
        packed_f16: false,
        // Ryujinx Decoders/InstDecoders.cs:4974 preserves LC.
        lod_clamp: false,
        multisample: false,
        depth_compare: b(r, 50, 1) != 0,
        nodep: b(r, 49, 1) != 0,
        scalar_dst: None,
        bound_handle,
    }))
}
/// Bound (top nibble 0xc): offsets 54..56, component 56..58, handle
/// 36..49.
fn texture_gather_bound(x: Word) -> Instruction {
    let r = x.r;
    texture_gather_of(x, b(r, 56, 2), b(r, 54, 2), Some(b(r, 36, 13) as u16))
}
/// Bindless (0xd): offsets 36..38, component 38..40.
fn texture_gather_bindless(x: Word) -> Instruction {
    let r = x.r;
    texture_gather_of(x, b(r, 38, 2), b(r, 36, 2), None)
}
const TEXTURE_GATHER_FIELDS: Fields<4> = Fields([
    Field::bits(35, 1),
    Field::bits(54, 2),
    Field::bits(50, 1),
    Field::ones(20, 8),
]);
const fn texture_gather_rule(r: u64) -> u8 {
    (b(r, 35, 1) == 0
        && b(r, 54, 2) < 3
        && (b(r, 54, 2) != 0 || b(r, 50, 1) != 0 || b(r, 20, 8) == 255)) as u8
}
const TEXTURE_GATHER_FORMS: [u8; TEXTURE_GATHER_FIELDS.len()] =
    forms!(TEXTURE_GATHER_FIELDS, texture_gather_rule);
fn read_texture_gather(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, texture_gather_bound];
    paths[(TEXTURE_GATHER_FORMS[TEXTURE_GATHER_FIELDS.index(x.r)]) as usize](x)
}
// Bound TXD (Ryujinx InstTable.cs:310, MIT; InstDecoders.cs:5047-5063
// InstTxd; envytools gm107.c:1964, de38 mask fffc; Mesa sm50.rs:
// 2313-2342): dst0, A8, B20, dim28..30, mask31..34, AOFFI35, index
// 36..48, NODEP49; fault destination 51..53 is PT and LC50 clear inside
// the opcode. InstEmitTexture.cs:929-1008 consumes A as coordinates then
// one packed word (layer & 0xffff, offsets from bit16) and B as dPdx
// then dPdy. Mesa nak_nir_lower_tex.c:328-346 interleaves ddx/ddy per
// axis, which agrees only for one-axis shapes (1D, ARRAY_1D): other
// dimensions and an empty mask stay Unknown.
fn texture_gradient(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::Texture(ops::Texture {
        op: TexOp::Grad,
        dst: d,
        src0: l,
        src1: Register(b(r, 20, 8) as u8),
        dim: b(r, 28, 3) as u8,
        mask: b(r, 31, 4) as u8,
        offset: b(r, 35, 1) != 0,
        scalar: false,
        packed_f16: false,
        lod_clamp: false,
        multisample: false,
        depth_compare: false,
        nodep: b(r, 49, 1) != 0,
        scalar_dst: None,
        bound_handle: Some(b(r, 36, 13) as u16),
    }))
}
const TEXTURE_GRADIENT_FIELDS: Fields<2> = Fields([Field::bits(28, 3), Field::bits(31, 4)]);
const fn texture_gradient_rule(r: u64) -> u8 {
    (b(r, 28, 3) <= 1 && b(r, 31, 4) != 0) as u8
}
const TEXTURE_GRADIENT_FORMS: [u8; TEXTURE_GRADIENT_FIELDS.len()] =
    forms!(TEXTURE_GRADIENT_FIELDS, texture_gradient_rule);
fn read_texture_gradient(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, texture_gradient];
    paths[(TEXTURE_GRADIENT_FORMS[TEXTURE_GRADIENT_FIELDS.index(x.r)]) as usize](x)
}
// TMML: Mesa sm50.rs:2287-2309. NDV35 changes derivatives and
// stays Unknown. Ryujinx Spirv/Instructions.cs:873-895 constructs
// spatial coordinates for each dimension, including 3D.
// Bound B is unused; zero/RZ are canonical (Mesa sm50.rs:2304).
// Bound index36..48 includes opcode bit48 (sm50.rs:2290-2291).
fn texture_query(x: Word) -> Instruction {
    let r = x.r;
    let o = b(r, 48, 16);
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::Texture(ops::Texture {
        op: TexOp::Lod,
        dst: d,
        src0: l,
        src1: Register(b(r, 20, 8) as u8),
        dim: b(r, 28, 3) as u8,
        mask: b(r, 31, 4) as u8,
        offset: false,
        scalar: false,
        packed_f16: false,
        lod_clamp: false,
        multisample: false,
        depth_compare: false,
        nodep: b(r, 49, 1) != 0,
        scalar_dst: None,
        bound_handle: (o & !0x0003 == 0xdf58).then(|| b(r, 36, 13) as u16),
    }))
}
const TEXTURE_QUERY_FIELDS: Fields<5> = Fields([
    Field::bits(35, 1),
    Field::bits(28, 3),
    Field::zero(33, 2),
    Field::zero(20, 8),
    Field::ones(20, 8),
]);
const fn texture_query_rule(r: u64) -> u8 {
    (b(r, 35, 1) == 0
        && b(r, 28, 3) != 5
        && b(r, 33, 2) == 0
        && (b(r, 20, 8) == 0 || b(r, 20, 8) == 255)) as u8
}
const TEXTURE_QUERY_FORMS: [u8; TEXTURE_QUERY_FIELDS.len()] =
    forms!(TEXTURE_QUERY_FIELDS, texture_query_rule);
fn read_texture_query(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, texture_query];
    paths[(TEXTURE_QUERY_FORMS[TEXTURE_QUERY_FIELDS.index(x.r)]) as usize](x)
}
const TEXTURE_QUERY_BINDLESS_FIELDS: Fields<3> =
    Fields([Field::bits(35, 1), Field::bits(28, 3), Field::zero(33, 2)]);
const fn texture_query_bindless_rule(r: u64) -> u8 {
    (b(r, 35, 1) == 0 && b(r, 28, 3) != 5 && b(r, 33, 2) == 0) as u8
}
const TEXTURE_QUERY_BINDLESS_FORMS: [u8; TEXTURE_QUERY_BINDLESS_FIELDS.len()] =
    forms!(TEXTURE_QUERY_BINDLESS_FIELDS, texture_query_bindless_rule);
fn read_texture_query_bindless(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, texture_query];
    paths[(TEXTURE_QUERY_BINDLESS_FORMS[TEXTURE_QUERY_BINDLESS_FIELDS.index(x.r)]) as usize](x)
}
// Bindless TXQ: Mesa sm50.rs:2345-2381, query22..28, A8..16,
// mask31..35, NODEP49. Ryujinx InstEmitTexture.cs:1105-1114
// chooses 3D when Z is requested, MS2D for texture_type. Bound
// queries need QuerySamplerType driver state (1118), unavailable.
// Query2 only writes Z (1152-1182). X/Y mask bits advance the
// destination while preserving those registers; W remains unmodeled.
// Mesa sm50.rs:2359-2361 and Ryujinx InstDecoders.cs:5096-5106:
// bindless TXQ has no table index. Reject unexplained selector bits.
fn texture_df50<const DIM: u8>(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::Texture(ops::Texture {
        op: TexOp::Query {
            query: b(r, 22, 6) as u8,
        },
        dst: d,
        src0: l,
        src1: Register(255),
        dim: DIM,
        mask: b(r, 31, 4) as u8,
        offset: false,
        scalar: false,
        packed_f16: false,
        lod_clamp: false,
        multisample: b(r, 22, 6) == 2,
        depth_compare: false,
        nodep: b(r, 49, 1) != 0,
        scalar_dst: None,
        bound_handle: None,
    }))
}
const TEXTURE_DF50_FIELDS: Fields<4> = Fields([
    Field::zero(36, 13),
    Field::bits(31, 4),
    Field::bits(22, 6),
    Field::bits(33, 1),
]);
const fn texture_df50_rule(r: u64) -> u8 {
    (b(r, 36, 13) == 0
        && b(r, 31, 4) != 0
        && (b(r, 22, 6) == 1 || (b(r, 22, 6) == 2 && b(r, 31, 4) >= 4 && b(r, 31, 4) <= 7)))
        as u8
        * (1 + (b(r, 22, 6) == 1 && b(r, 33, 1) != 0) as u8)
}
const TEXTURE_DF50_FORMS: [u8; TEXTURE_DF50_FIELDS.len()] =
    forms!(TEXTURE_DF50_FIELDS, texture_df50_rule);
fn read_texture_df50(x: Word) -> Instruction {
    let paths: [Decode; 3] = [unknown, texture_df50::<2>, texture_df50::<4>];
    paths[(TEXTURE_DF50_FORMS[TEXTURE_DF50_FIELDS.index(x.r)]) as usize](x)
}
// Bound TXQ: envytools gm107.c:1958; Mesa sm50.rs:2351-2354,
// 2364-2381. dst0..7, A8..15, query22..27, mask31..34,
// binding36..48, NODEP49. There is no texture dimension field.
// Ryujinx InstEmitTexture.cs:1084-1089,1116-1119 consumes A as
// LOD and requires driver sampler type. Emission resolves shape
// from driver metadata or a typed operation on the same binding,
// or rejects the query.
// Query2 texture_type writes samples at the selected Z destination,
// preserving selected X/Y registers (Ryujinx
// InstEmitTexture.cs:1152-1182). Other query selectors stay Unknown:
// its default logs unsupported (1184-1186); gm107.c:882-891 names them.
// A levels-only dimension query (mask W) still names its LOD in A, as
// game code does, but levels do not depend on it: Ryujinx passes A as the
// source (1071-1091) and answers component 3 with ImageQueryLevels alone
// (CodeGen/Spirv/Instructions.cs:1487-1489).
fn texture_df48(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::Texture(ops::Texture {
        op: TexOp::Query {
            query: b(r, 22, 6) as u8,
        },
        dst: d,
        src0: l,
        src1: Register(255),
        dim: 2,
        mask: b(r, 31, 4) as u8,
        offset: false,
        scalar: false,
        packed_f16: false,
        lod_clamp: false,
        multisample: false,
        depth_compare: false,
        nodep: b(r, 49, 1) != 0,
        scalar_dst: None,
        bound_handle: Some(b(r, 36, 13) as u16),
    }))
}
const TEXTURE_DF48_FIELDS: Fields<3> =
    Fields([Field::bits(31, 4), Field::bits(22, 6), Field::ones(8, 8)]);
const fn texture_df48_rule(r: u64) -> u8 {
    (b(r, 31, 4) != 0
        && (b(r, 22, 6) == 1
            || (b(r, 22, 6) == 2 && b(r, 31, 4) >= 4 && b(r, 31, 4) <= 7 && b(r, 8, 8) == 255)))
        as u8
}
const TEXTURE_DF48_FORMS: [u8; TEXTURE_DF48_FIELDS.len()] =
    forms!(TEXTURE_DF48_FIELDS, texture_df48_rule);
fn read_texture_df48(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, texture_df48];
    paths[(TEXTURE_DF48_FORMS[TEXTURE_DF48_FIELDS.index(x.r)]) as usize](x)
}
// Scalar F32/F16 forms. Bit 59 selects F32 for TEXS/TLDS; bit 55
// selects F16 for TLD4S on SM60 (envytools gm107.c:1960,1967-1968).
// Ryujinx InstEmitTexture.cs:609-655 packs selected pairs. Mesa
// nv50_ir_emit_gm107.cpp:2933-2968 emits
// 0xd800/0xda00/0xdf00, destination pairs R0/R28, sources R8/R20,
// and bound texture index 36..49. Mode/mask tables are decoded below.
/// TEXS/TLDS mode and result tables. Fields and table entries:
/// envytools envydis/gm107.c:932-1008,1967-1968; Mesa
/// nv50_ir_emit_gm107.cpp:2787-2882,2933-2968. Register semantics:
/// Ryujinx Instructions/InstEmitTexture.cs:394-562,613-641.
///
/// Write masks by selector 50..52: one destination pair (the second RZ)
/// or two (selectors 5..7 reserved, Unknown by the form table).
const SCALAR_MASKS: [[u8; 8]; 2] = [
    [0x1, 0x2, 0x4, 0x8, 0x3, 0x9, 0xa, 0xc],
    [0x7, 0xb, 0xd, 0xe, 0xf, 0x0, 0x0, 0x0],
];
/// TLDS modes 53..56: (dim, lod, offset, multisample). Modes 3, 9..11,
/// 13..15 are reserved, Unknown by the form table.
const SCALAR_FETCH_MODES: [(u8, bool, bool, bool); 16] = [
    (0, false, false, false),
    (0, true, false, false),
    (2, false, false, false),
    (0, false, false, false),
    (2, false, true, false),
    (2, true, false, false),
    (2, false, false, true),
    (4, false, false, false),
    (3, false, false, false),
    (0, false, false, false),
    (0, false, false, false),
    (0, false, false, false),
    (2, true, true, false),
    (0, false, false, false),
    (0, false, false, false),
    (0, false, false, false),
];
/// TEXS modes 53..56: (dim, lod, depth compare). Modes 14 and 15 are
/// reserved, Unknown by the form table.
const SCALAR_SAMPLE_MODES: [(u8, u8, bool); 16] = [
    (0, 1, false),
    (2, 0, false),
    (2, 1, false),
    (2, 3, false),
    (2, 0, true),
    (2, 3, true),
    (2, 1, true),
    (3, 0, false),
    (3, 1, false),
    (3, 1, true),
    (4, 0, false),
    (4, 1, false),
    (6, 0, false),
    (6, 3, false),
    (0, 0, false),
    (0, 0, false),
];
fn scalar_texture_of(
    x: Word,
    op: TexOp,
    dim: u8,
    offset: bool,
    multisample: bool,
    depth_compare: bool,
) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::Texture(ops::Texture {
        op,
        dst: Register(b(r, 0, 8) as u8),
        src0: Register(b(r, 8, 8) as u8),
        src1: Register(b(r, 20, 8) as u8),
        dim,
        mask: SCALAR_MASKS[SCALAR_SECOND_DST.index(r)][b(r, 50, 3) as usize],
        offset,
        scalar: true,
        packed_f16: b(r, 59, 1) == 0,
        lod_clamp: false,
        multisample,
        depth_compare,
        nodep: b(r, 49, 1) != 0,
        scalar_dst: Some(Register(b(r, 28, 8) as u8)),
        bound_handle: Some(b(r, 36, 13) as u16),
    }))
}
/// TLDS (bit 57 set).
fn scalar_fetch(x: Word) -> Instruction {
    let (dim, lod, offset, multisample) = SCALAR_FETCH_MODES[b(x.r, 53, 4) as usize];
    scalar_texture_of(x, TexOp::Fetch { lod }, dim, offset, multisample, false)
}
/// TEXS (bit 57 clear).
fn scalar_sample(x: Word) -> Instruction {
    let (dim, lod, depth_compare) = SCALAR_SAMPLE_MODES[b(x.r, 53, 4) as usize];
    scalar_texture_of(x, TexOp::Sample { lod }, dim, false, false, depth_compare)
}
/// The modes each kind implements, one bit per mode: TEXS 0..13, TLDS
/// 0, 1, 2, 4..8 and 12.
const SCALAR_MODES: [u32; 2] = [0x3fff, 0x11f7];
const SCALAR_TEXTURE_FIELDS: Fields<4> = Fields([
    Field::ones(28, 8),
    Field::bits(50, 3),
    Field::bits(53, 4),
    Field::bits(57, 1),
]);
const fn scalar_texture_rule(r: u64) -> u8 {
    ((b(r, 28, 8) == 255 || b(r, 50, 3) <= 4)
        && (SCALAR_MODES[b(r, 57, 1) as usize] >> b(r, 53, 4)) & 1 == 1) as u8
        * (1 + b(r, 57, 1) as u8)
}
const SCALAR_TEXTURE_FORMS: [u8; SCALAR_TEXTURE_FIELDS.len()] =
    forms!(SCALAR_TEXTURE_FIELDS, scalar_texture_rule);
fn scalar_texture_word(x: Word) -> Instruction {
    let paths: [Decode; 3] = [unknown, scalar_sample, scalar_fetch];
    paths[SCALAR_TEXTURE_FORMS[SCALAR_TEXTURE_FIELDS.index(x.r)] as usize](x)
}

/// The second destination (28..36): 0 when RZ.
const SCALAR_SECOND_DST: Fields<1> = Fields([Field::ones(28, 8)]);
fn texture_df00(x: Word) -> Instruction {
    let r = x.r;
    let d = Register(b(r, 0, 8) as u8);
    let l = Register(b(r, 8, 8) as u8);

    x.guarded(InstructionKind::Texture(ops::Texture {
        op: TexOp::Gather {
            // envytools gm107.c:893-898,1960.
            component: b(r, 52, 2) as u8,
            offsets: [GatherOffsets::Zero, GatherOffsets::One][b(r, 51, 1) as usize],
        },
        dst: d,
        src0: l,
        src1: Register(b(r, 20, 8) as u8),
        dim: 2,
        // Gather selects mask code 4 in the same destination-dependent
        // table (Ryujinx InstEmitTexture.cs:107,613; envytools:963,971).
        mask: [0x3, 0xf][SCALAR_SECOND_DST.index(r)],
        offset: false,
        scalar: true,
        packed_f16: b(r, 55, 1) != 0,
        lod_clamp: false,
        multisample: false,
        depth_compare: b(r, 50, 1) != 0,
        nodep: b(r, 49, 1) != 0,
        scalar_dst: Some(Register(b(r, 28, 8) as u8)),
        bound_handle: Some(b(r, 36, 13) as u16),
    }))
}
// SUST/SULD: envytools envydis/gm107.c:1909-1910. Dimension is
// THREE bits 33..36 (Mesa sm50.rs:2435-2445,2481,2529), not four:
// bit 36 belongs to the bound handle (envytools gm107.c:788-791).
// .P/.D at 52, mask 20..24 or size 20..23 and BA at 23:
// Mesa sm50.rs:2471-2478,2506-2513; envytools gm107.c:794-822.
// Cache 24..26: gm107.c:825-838; Mesa sm50.rs:2424-2431.
// CG becomes Coherent (Ryujinx InstEmitSurface.cs:364-367,677-679).
// Store CS is CacheStreaming (Mesa ir.rs:2545-2559), lowered to
// Nontemporal (Mesa spirv.core.grammar.json:11990-11994). WT
// requires system ordering (Mesa ir.rs:2574), still unimplemented.
// Clamp 49..51: gm107.c:781-785. IGN maps to Vulkan bounds behavior;
// unnamed/trap modes are unimplemented and remain Unknown.
// Coordinates 8..16, data 0..8, register handle 39..47:
// Mesa sm50.rs:2491-2494,2517-2519; bound U13_36 selected by 51:
// envytools gm107.c:788-791,1909-1910.
const SURFACE_STORE_FIELDS: Fields<10> = Fields([
    Field::bits(33, 3),
    Field::zero(26, 7),
    Field::bits(51, 1),
    Field::zero(36, 3),
    Field::zero(47, 2),
    Field::zero(49, 2),
    Field::bits(24, 2),
    Field::bits(52, 1),
    Field::bits(20, 3),
    Field::zero(23, 1),
]);
const fn surface_store_rule(r: u64) -> u8 {
    (b(r, 33, 3) <= 5
        && b(r, 26, 7) == 0
        && (b(r, 51, 1) != 0 || (b(r, 36, 3) == 0 && b(r, 47, 2) == 0))
        && b(r, 49, 2) == 0
        && (b(r, 52, 1) != 0 || b(r, 20, 4) != 0)
        && (b(r, 52, 1) == 0 || b(r, 20, 3) != 7)
        && b(r, 24, 2) <= 2) as u8
        * (1 + b(r, 52, 1) as u8)
}
static SURFACE_STORE_FORMS: [u8; SURFACE_STORE_FIELDS.len()] =
    forms!(SURFACE_STORE_FIELDS, surface_store_rule);
fn read_surface_store(x: Word) -> Instruction {
    let paths: [Decode; 3] = [unknown, surface_formatted::<true>, surface_data::<true>];
    paths[(SURFACE_STORE_FORMS[SURFACE_STORE_FIELDS.index(x.r)]) as usize](x)
}
/// .D sizes 20..23: 5 (64-bit) two lanes, 6 (128-bit) four, the rest
/// one. nvdisasm SM52 prints 7 as INVALIDSIZE7; the rules reject it.
const SURFACE_DATA_MASKS: [u8; 8] = [1, 1, 1, 1, 1, 3, 15, 0];
fn surface_of(
    x: Word,
    store: bool,
    format: u8,
    mask: u8,
    size: u8,
    byte_address: bool,
) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::Surface(ops::Surface {
        store,
        atomic: false,
        target: b(r, 33, 3) as u8,
        format,
        mask,
        size,
        byte_address,
        cache: b(r, 24, 2) as u8,
        handle: [
            Source::Register(Register(b(r, 39, 8) as u8)),
            Source::Immediate(b(r, 36, 13)),
        ][b(r, 51, 1) as usize],
        coord: Register(b(r, 8, 8) as u8),
        data: Register(b(r, 0, 8) as u8),
        clamp: b(r, 49, 2) as u8,
    }))
}
/// .P (bit 52 clear): the component mask 20..24.
fn surface_formatted<const STORE: bool>(x: Word) -> Instruction {
    surface_of(x, STORE, 0, b(x.r, 20, 4) as u8, 0, false)
}
/// .D (bit 52 set): size 20..23, BA 23.
fn surface_data<const STORE: bool>(x: Word) -> Instruction {
    let r = x.r;
    surface_of(
        x,
        STORE,
        1,
        SURFACE_DATA_MASKS[b(r, 20, 3) as usize],
        b(r, 20, 3) as u8,
        b(r, 23, 1) != 0,
    )
}
const SURFACE_FIELDS: Fields<10> = Fields([
    Field::bits(33, 3),
    Field::zero(26, 7),
    Field::bits(51, 1),
    Field::zero(36, 3),
    Field::zero(47, 2),
    Field::zero(49, 2),
    Field::bits(24, 2),
    Field::bits(52, 1),
    Field::bits(20, 3),
    Field::zero(23, 1),
]);
const fn surface_rule(r: u64) -> u8 {
    (b(r, 33, 3) <= 5
        && b(r, 26, 7) == 0
        && (b(r, 51, 1) != 0 || (b(r, 36, 3) == 0 && b(r, 47, 2) == 0))
        && b(r, 49, 2) == 0
        && (b(r, 52, 1) != 0 || b(r, 20, 4) != 0)
        && (b(r, 52, 1) == 0 || b(r, 20, 3) != 7)
        && b(r, 24, 2) <= 1) as u8
        * (1 + b(r, 52, 1) as u8)
}
static SURFACE_FORMS: [u8; SURFACE_FIELDS.len()] = forms!(SURFACE_FIELDS, surface_rule);
fn read_surface(x: Word) -> Instruction {
    let paths: [Decode; 3] = [unknown, surface_formatted::<false>, surface_data::<false>];
    paths[(SURFACE_FORMS[SURFACE_FIELDS.index(x.r)]) as usize](x)
}
// Integer SUATOM/SURED. The decoder rejects every unimplemented
// size, format conversion, clamp and predicate-result form.
// Mesa sm50.rs:2558-2591; envytools gm107.c:1908,1911-1914.
//
// Each opcode range is its own form: SUATOM bound 0xea00..0xea3f and
// register 0xea60..0xea7f, SUATOM.CAS bound 0xea80..0xeabf and register
// 0xeac0..0xeadf, SURED 0xeb40..0xeb5f. Size above S32, target above 5
// and clamp 49..51 stay Unknown; the register forms need .D (52) and no
// bound flag (51), as size and handle fields overlap there. .P conversion
// depends on driver format state. CAS predicate output is not
// implemented: only PT (30..33) is accepted (encoding in Mesa's emitter
// nv50_ir_emit_gm107.cpp:3392 and envytools gm107.c:1861-1871; predicate
// interpretation Ryujinx InstDecoders.cs:4642,4659). Ryujinx
// CodeGen/Spirv/Instructions.cs:674-675 increments/decrements (ops 3, 4)
// without using the encoded data operand: canonical RZ is required so an
// unexplained source is never silently dropped.
fn surface_atomic_of(
    x: Word,
    operation: u32,
    size: u32,
    handle: Source,
    data: u32,
    dst: Option<Register>,
    byte_address: u32,
) -> Instruction {
    let r = x.r;
    x.guarded(InstructionKind::SurfaceAtomic(ops::SurfaceAtomic {
        op: operation as u8,
        signed: size == 1,
        target: b(r, 33, 3) as u8,
        handle,
        coord: Register(b(r, 8, 8) as u8),
        data: Register(data as u8),
        dst,
        byte_address: b(r, byte_address, 1) != 0,
    }))
}
fn bound_handle(r: u64) -> Source {
    Source::Immediate(b(r, 36, 13))
}
fn register_handle(r: u64) -> Source {
    Source::Register(Register(b(r, 39, 8) as u8))
}
fn surface_atom_bound(x: Word) -> Instruction {
    let r = x.r;
    let dst = Some(Register(b(r, 0, 8) as u8));
    surface_atomic_of(
        x,
        b(r, 29, 4),
        b(r, 51, 3),
        bound_handle(r),
        b(r, 20, 8),
        dst,
        28,
    )
}
fn surface_atom_register(x: Word) -> Instruction {
    let r = x.r;
    let dst = Some(Register(b(r, 0, 8) as u8));
    surface_atomic_of(
        x,
        b(r, 29, 4),
        b(r, 36, 3),
        register_handle(r),
        b(r, 20, 8),
        dst,
        28,
    )
}
fn surface_cas_bound(x: Word) -> Instruction {
    let r = x.r;
    let dst = Some(Register(b(r, 0, 8) as u8));
    surface_atomic_of(x, 9, b(r, 51, 3), bound_handle(r), b(r, 20, 8), dst, 28)
}
fn surface_cas_register(x: Word) -> Instruction {
    let r = x.r;
    let dst = Some(Register(b(r, 0, 8) as u8));
    surface_atomic_of(x, 9, b(r, 36, 3), register_handle(r), b(r, 20, 8), dst, 28)
}
/// SURED: no destination, data 0..8, BA 23, handle by bit 51.
fn surface_reduce(x: Word) -> Instruction {
    let r = x.r;
    let handle = [register_handle(r), bound_handle(r)][b(r, 51, 1) as usize];
    surface_atomic_of(x, b(r, 24, 3), b(r, 20, 3), handle, b(r, 0, 8), None, 23)
}
/// INC (3) and DEC (4) need data RZ.
const fn surface_atomic_data_ok(operation: u32, data: u32) -> bool {
    (0b11000 >> operation) & 1 == 0 || data == 255
}
const SURFACE_ATOM_BOUND_FIELDS: Fields<5> = Fields([
    Field::capped(51, 3, 2),
    Field::bits(29, 4),
    Field::bits(33, 3),
    Field::zero(49, 2),
    Field::ones(20, 8),
]);
const fn surface_atom_bound_rule(r: u64) -> u8 {
    (b(r, 51, 3) <= 1
        && b(r, 29, 4) <= 8
        && b(r, 33, 3) <= 5
        && b(r, 49, 2) == 0
        && surface_atomic_data_ok(b(r, 29, 4), b(r, 20, 8))) as u8
}
const SURFACE_ATOM_BOUND_FORMS: [u8; SURFACE_ATOM_BOUND_FIELDS.len()] =
    forms!(SURFACE_ATOM_BOUND_FIELDS, surface_atom_bound_rule);
const SURFACE_ATOM_REGISTER_FIELDS: Fields<7> = Fields([
    Field::capped(36, 3, 2),
    Field::bits(29, 4),
    Field::bits(33, 3),
    Field::zero(49, 2),
    Field::bits(52, 1),
    Field::bits(51, 1),
    Field::ones(20, 8),
]);
const fn surface_atom_register_rule(r: u64) -> u8 {
    (b(r, 36, 3) <= 1
        && b(r, 29, 4) <= 8
        && b(r, 33, 3) <= 5
        && b(r, 49, 2) == 0
        && b(r, 52, 1) == 1
        && b(r, 51, 1) == 0
        && surface_atomic_data_ok(b(r, 29, 4), b(r, 20, 8))) as u8
}
const SURFACE_ATOM_REGISTER_FORMS: [u8; SURFACE_ATOM_REGISTER_FIELDS.len()] =
    forms!(SURFACE_ATOM_REGISTER_FIELDS, surface_atom_register_rule);
const SURFACE_CAS_BOUND_FIELDS: Fields<5> = Fields([
    Field::capped(51, 3, 2),
    Field::bits(33, 3),
    Field::zero(49, 2),
    Field::ones(30, 3),
    Field::bits(29, 1),
]);
const fn surface_cas_bound_rule(r: u64) -> u8 {
    (b(r, 51, 3) <= 1
        && b(r, 33, 3) <= 5
        && b(r, 49, 2) == 0
        && b(r, 30, 3) == 7
        && b(r, 29, 1) == 0) as u8
}
const SURFACE_CAS_BOUND_FORMS: [u8; SURFACE_CAS_BOUND_FIELDS.len()] =
    forms!(SURFACE_CAS_BOUND_FIELDS, surface_cas_bound_rule);
const SURFACE_CAS_REGISTER_FIELDS: Fields<7> = Fields([
    Field::capped(36, 3, 2),
    Field::bits(33, 3),
    Field::zero(49, 2),
    Field::bits(52, 1),
    Field::bits(51, 1),
    Field::ones(30, 3),
    Field::bits(29, 1),
]);
const fn surface_cas_register_rule(r: u64) -> u8 {
    (b(r, 36, 3) <= 1
        && b(r, 33, 3) <= 5
        && b(r, 49, 2) == 0
        && b(r, 52, 1) == 1
        && b(r, 51, 1) == 0
        && b(r, 30, 3) == 7
        && b(r, 29, 1) == 0) as u8
}
const SURFACE_CAS_REGISTER_FORMS: [u8; SURFACE_CAS_REGISTER_FIELDS.len()] =
    forms!(SURFACE_CAS_REGISTER_FIELDS, surface_cas_register_rule);
const SURFACE_REDUCE_FIELDS: Fields<6> = Fields([
    Field::capped(20, 3, 2),
    Field::bits(24, 3),
    Field::bits(33, 3),
    Field::zero(49, 2),
    Field::bits(52, 1),
    Field::ones(0, 8),
]);
const fn surface_reduce_rule(r: u64) -> u8 {
    (b(r, 20, 3) <= 1
        && b(r, 33, 3) <= 5
        && b(r, 49, 2) == 0
        && b(r, 52, 1) == 1
        && surface_atomic_data_ok(b(r, 24, 3), b(r, 0, 8))) as u8
}
const SURFACE_REDUCE_FORMS: [u8; SURFACE_REDUCE_FIELDS.len()] =
    forms!(SURFACE_REDUCE_FIELDS, surface_reduce_rule);
fn read_surface_atom_bound(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, surface_atom_bound];
    paths[SURFACE_ATOM_BOUND_FORMS[SURFACE_ATOM_BOUND_FIELDS.index(x.r)] as usize](x)
}
fn read_surface_atom_register(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, surface_atom_register];
    paths[SURFACE_ATOM_REGISTER_FORMS[SURFACE_ATOM_REGISTER_FIELDS.index(x.r)] as usize](x)
}
fn read_surface_cas_bound(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, surface_cas_bound];
    paths[SURFACE_CAS_BOUND_FORMS[SURFACE_CAS_BOUND_FIELDS.index(x.r)] as usize](x)
}
fn read_surface_cas_register(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, surface_cas_register];
    paths[SURFACE_CAS_REGISTER_FORMS[SURFACE_CAS_REGISTER_FIELDS.index(x.r)] as usize](x)
}
fn read_surface_reduce(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, surface_reduce];
    paths[SURFACE_REDUCE_FORMS[SURFACE_REDUCE_FIELDS.index(x.r)] as usize](x)
}
// SUATOM.B2 (0xeb80..0xebbf) is the register-handle form with a result
// predicate. Ryujinx InstTable.cs:281 and InstDecoders.cs:4611-4627 establish
// the layout. InstEmitSurface.cs:53-70 executes the same surface atomic and
// does not write DestPred, so only the encoded PT result (zero in this
// complemented predicate field) is admitted. Clamp modes and unexplained
// bits 47..48 remain closed.
fn surface_atom_pt_result(x: Word) -> Instruction {
    let r = x.r;
    surface_atomic_of(
        x,
        b(r, 29, 4),
        b(r, 36, 3),
        register_handle(r),
        b(r, 20, 8),
        Some(Register(b(r, 0, 8) as u8)),
        28,
    )
}
const SURFACE_ATOM_PT_RESULT_FIELDS: Fields<7> = Fields([
    Field::capped(36, 3, 2),
    Field::bits(29, 4),
    Field::bits(33, 3),
    Field::zero(49, 2),
    Field::zero(51, 3),
    Field::zero(47, 2),
    Field::ones(20, 8),
]);
const fn surface_atom_pt_result_rule(r: u64) -> u8 {
    (b(r, 36, 3) <= 1
        && b(r, 29, 4) <= 8
        && b(r, 33, 3) <= 5
        && b(r, 49, 2) == 0
        && b(r, 51, 3) == 0
        && b(r, 47, 2) == 0
        && surface_atomic_data_ok(b(r, 29, 4), b(r, 20, 8))) as u8
}
const SURFACE_ATOM_PT_RESULT_FORMS: [u8; SURFACE_ATOM_PT_RESULT_FIELDS.len()] =
    forms!(SURFACE_ATOM_PT_RESULT_FIELDS, surface_atom_pt_result_rule);
fn read_surface_atom_pt_result(x: Word) -> Instruction {
    let paths: [Decode; 2] = [unknown, surface_atom_pt_result];
    paths[SURFACE_ATOM_PT_RESULT_FORMS[SURFACE_ATOM_PT_RESULT_FIELDS.index(x.r)] as usize](x)
}
// Mesa SM50's immediate/cbuf encoders add modifier bits to these
// opcode families: sm50.rs:568-577, 1462-1476, 1510-1520.
// Mesa OpSync is 0xf0f8 (sm50.rs:3044-3050). DEPBAR
// (0xf0f0..=0xf0f7, envytools gm107.c:1878) waits on the
// scoreboard; it has no data effect and SPIR-V makes dependencies
// explicit. SYNC shares the family (0xf0f8, gm107.c:1877) and is
// control flow, decoded above. Mesa sm50.rs:3044-3052 encodes OpSync
// at 0xf0f8 inside the allowed 0xf0f0..0xf0ff scheduling family.
// Arithmetic opcodes must never be
// mapped to Nop: an undecoded instruction stays Unknown.
fn dependency_barrier(x: Word) -> Instruction {
    x.guarded(InstructionKind::Nop(ops::Nop))
}
/// The family of each opcode chain, in opcode-map order.
pub(super) const FAMILY: [Decode; 168] = [
    unknown,
    read_brx_register,
    read_fadd_register,
    fcmp_immediate,
    icmp_immediate,
    read_half_fma_register,
    read_half_mul_register,
    half_add_register,
    half_add_long,
    read_half_long,
    read_half_add_short,
    half_add_constant,
    read_half_mul_short,
    read_half_fma_short,
    read_half_set_register,
    read_video_multiply_add,
    read_video_min_max,
    read_fmul,
    mov_register,
    mov_constant,
    read_ldc,
    mov32i,
    prmt,
    read_iadd,
    read_xmad_register,
    read_xmad_immediate,
    read_xmad_constant,
    read_xmad_constant_addend,
    read_shift_add,
    read_iadd3_register,
    read_iadd3,
    read_lea_register,
    read_lea,
    read_lea_hi,
    read_lea_hi_immediate,
    read_funnel_shift_right,
    read_funnel_shift,
    read_bfi_register,
    read_bfi_constant,
    read_bfi_immediate,
    read_bfi_constant_insert,
    read_bfe,
    read_bfe_immediate,
    read_fadd_constant,
    read_fadd_immediate,
    read_fadd32i,
    read_fmul32i,
    read_ffma_immediate,
    read_ffma_constant_addend,
    read_ffma_constant,
    read_fmnmx,
    shuffle,
    read_swizzle_add,
    rro_register,
    read_iscadd32i,
    read_imul,
    read_imul_hi,
    read_mufu,
    read_flo,
    popc,
    read_exit,
    read_ipa,
    read_nop,
    read_vote_vtg,
    read_vote,
    read_branch,
    read_schedule,
    read_push_target,
    read_pop_jump,
    read_kill,
    read_s2r,
    read_iset,
    read_p2r_register,
    read_p2r,
    read_video_setp,
    r2p,
    read_isetp,
    read_dfma,
    read_dmul,
    read_dadd,
    read_dmnmx,
    read_dset,
    read_dsetp,
    read_fsetp,
    read_csetp,
    read_psetp,
    read_pset,
    read_fset,
    sel,
    read_sel_immediate,
    read_i2f_register,
    read_i2f,
    read_f2i,
    read_i2i_register,
    read_i2i,
    read_lop3_register,
    read_lop3_immediate,
    read_lop3_constant,
    read_shl,
    read_shr,
    read_global_load,
    read_global_store,
    read_reduce,
    read_atom,
    read_global_compare_exchange,
    read_local_load,
    read_shared_load,
    read_local_store,
    read_shared_store,
    read_shared_compare_exchange,
    read_shared_atomic,
    read_group_memory_barrier,
    read_control_barrier,
    read_geometry_out,
    read_geometry_out_immediate,
    read_attribute_address,
    read_isbe_read,
    read_attribute_load_reserved,
    read_attribute_store,
    read_texture,
    read_texture_bindless,
    read_texture_fetch,
    read_texture_gather_bindless,
    read_texture_gather,
    read_texture_query,
    read_texture_query_bindless,
    read_texture_df50,
    read_texture_df48,
    scalar_texture_word,
    texture_df00,
    read_surface_store,
    read_surface,
    read_surface_atom_bound,
    dependency_barrier,
    read_half_set_short,
    fcmp_constant,
    fcmp_constant_c,
    fcmp_register,
    icmp_constant,
    icmp_constant_c,
    icmp_register,
    read_surface_atom_register,
    read_surface_cas_bound,
    read_surface_cas_register,
    read_surface_reduce,
    read_geometry_out_constant,
    read_padding,
    read_call,
    shift_add32i,
    read_ffma32i,
    rro_operand,
    read_imnmx,
    read_sync,
    read_cset,
    read_f2f,
    read_f2f_immediate,
    read_f2f_register,
    lop32i,
    lop,
    read_texture_gradient,
    read_imad32i,
    read_imul32i,
    read_imad_constant_addend,
    read_imadsp,
    read_imadsp_constant_addend,
    read_surface_atom_pt_result,
    mov_immediate20,
    read_pcnt_alias,
];
/// Opcode runs: each opcode from a run's start up to the next run's start
/// belongs to the run's family.
const OPCODE_RUNS: [(u32, u8); 488] = [
    (0x0000, 146),
    (0x0001, 0),
    (0x0100, 21),
    (0x0200, 97),
    (0x0400, 157),
    (0x0800, 45),
    (0x0c00, 149),
    (0x1000, 160),
    (0x1400, 148),
    (0x1800, 34),
    (0x1c00, 54),
    (0x1d80, 0),
    (0x1e00, 46),
    (0x1f00, 161),
    (0x2000, 0),
    (0x2800, 9),
    (0x2c00, 8),
    (0x2e00, 0),
    (0x3000, 87),
    (0x3200, 81),
    (0x3280, 47),
    (0x3300, 81),
    (0x3380, 47),
    (0x3400, 56),
    (0x3480, 163),
    (0x3500, 56),
    (0x3580, 163),
    (0x3600, 25),
    (0x3640, 4),
    (0x3650, 71),
    (0x3660, 76),
    (0x3670, 77),
    (0x3680, 82),
    (0x3690, 0),
    (0x36a0, 3),
    (0x36b0, 83),
    (0x36c0, 22),
    (0x36c1, 0),
    (0x36d0, 32),
    (0x36d8, 0),
    (0x36f0, 39),
    (0x36f8, 36),
    (0x3700, 25),
    (0x3740, 4),
    (0x3750, 71),
    (0x3760, 76),
    (0x3770, 77),
    (0x3780, 82),
    (0x3790, 0),
    (0x37a0, 3),
    (0x37b0, 83),
    (0x37c0, 22),
    (0x37c1, 0),
    (0x37d0, 32),
    (0x37d8, 0),
    (0x37f0, 39),
    (0x37f8, 36),
    (0x3800, 42),
    (0x3808, 59),
    (0x3810, 23),
    (0x3818, 28),
    (0x3820, 151),
    (0x3828, 99),
    (0x3830, 58),
    (0x3838, 55),
    (0x3839, 0),
    (0x3840, 158),
    (0x3848, 98),
    (0x3850, 80),
    (0x3858, 44),
    (0x385a, 0),
    (0x385c, 44),
    (0x385e, 0),
    (0x3860, 50),
    (0x3868, 17),
    (0x3870, 79),
    (0x3878, 0),
    (0x3880, 78),
    (0x3888, 0),
    (0x3890, 150),
    (0x3898, 166),
    (0x38a0, 88),
    (0x38a1, 89),
    (0x38a8, 155),
    (0x38b0, 92),
    (0x38b8, 91),
    (0x38c0, 30),
    (0x38d0, 0),
    (0x38e0, 94),
    (0x38e8, 73),
    (0x38e9, 0),
    (0x38f0, 75),
    (0x38f8, 35),
    (0x3900, 42),
    (0x3908, 59),
    (0x3910, 23),
    (0x3918, 28),
    (0x3920, 151),
    (0x3928, 99),
    (0x3930, 58),
    (0x3938, 0),
    (0x3940, 158),
    (0x3948, 98),
    (0x3950, 80),
    (0x3958, 44),
    (0x395a, 0),
    (0x395c, 44),
    (0x395e, 0),
    (0x3960, 50),
    (0x3968, 17),
    (0x3970, 79),
    (0x3978, 0),
    (0x3980, 78),
    (0x3988, 0),
    (0x3990, 150),
    (0x3998, 166),
    (0x39a0, 89),
    (0x39a8, 155),
    (0x39b0, 92),
    (0x39b8, 91),
    (0x39c0, 30),
    (0x39d0, 0),
    (0x39e0, 94),
    (0x39e8, 73),
    (0x39e9, 0),
    (0x39f0, 75),
    (0x39f8, 35),
    (0x3a00, 16),
    (0x3c00, 96),
    (0x4000, 0),
    (0x4800, 87),
    (0x4900, 81),
    (0x4980, 49),
    (0x4a00, 56),
    (0x4a80, 163),
    (0x4b00, 0),
    (0x4b40, 138),
    (0x4b50, 71),
    (0x4b60, 76),
    (0x4b70, 77),
    (0x4b80, 82),
    (0x4b90, 0),
    (0x4ba0, 135),
    (0x4bb0, 83),
    (0x4bc0, 22),
    (0x4bc1, 0),
    (0x4bd0, 32),
    (0x4bd8, 0),
    (0x4bf0, 38),
    (0x4bf8, 0),
    (0x4c00, 41),
    (0x4c08, 59),
    (0x4c10, 23),
    (0x4c18, 28),
    (0x4c20, 151),
    (0x4c28, 99),
    (0x4c30, 58),
    (0x4c38, 55),
    (0x4c39, 0),
    (0x4c40, 158),
    (0x4c48, 98),
    (0x4c50, 80),
    (0x4c58, 43),
    (0x4c60, 50),
    (0x4c68, 17),
    (0x4c70, 79),
    (0x4c78, 0),
    (0x4c80, 78),
    (0x4c88, 0),
    (0x4c90, 150),
    (0x4c98, 19),
    (0x4c99, 0),
    (0x4ca0, 88),
    (0x4ca1, 0),
    (0x4ca8, 154),
    (0x4cb0, 92),
    (0x4cb8, 91),
    (0x4cc0, 30),
    (0x4cd0, 0),
    (0x4ce0, 94),
    (0x4ce8, 73),
    (0x4ce9, 0),
    (0x4cf0, 75),
    (0x4cf8, 0),
    (0x4e00, 26),
    (0x5000, 0),
    (0x5080, 57),
    (0x5088, 86),
    (0x5090, 85),
    (0x5091, 0),
    (0x5098, 153),
    (0x50a0, 84),
    (0x50a1, 0),
    (0x50b0, 62),
    (0x50b8, 0),
    (0x50d8, 64),
    (0x50dc, 0),
    (0x50e0, 63),
    (0x50e8, 0),
    (0x50f0, 74),
    (0x50f8, 52),
    (0x50f9, 0),
    (0x5100, 27),
    (0x5180, 48),
    (0x5200, 162),
    (0x5280, 164),
    (0x5300, 0),
    (0x5340, 139),
    (0x5350, 0),
    (0x5370, 77),
    (0x5380, 0),
    (0x53a0, 136),
    (0x53b0, 0),
    (0x53f0, 40),
    (0x53f8, 0),
    (0x5800, 87),
    (0x5900, 81),
    (0x5980, 49),
    (0x5a00, 56),
    (0x5a80, 163),
    (0x5b00, 24),
    (0x5b40, 140),
    (0x5b50, 71),
    (0x5b60, 76),
    (0x5b70, 77),
    (0x5b80, 82),
    (0x5b90, 0),
    (0x5ba0, 137),
    (0x5bb0, 83),
    (0x5bc0, 22),
    (0x5bc1, 0),
    (0x5bd0, 31),
    (0x5bd8, 33),
    (0x5be0, 95),
    (0x5be8, 0),
    (0x5bf0, 37),
    (0x5bf8, 36),
    (0x5c00, 41),
    (0x5c08, 59),
    (0x5c10, 23),
    (0x5c18, 28),
    (0x5c20, 151),
    (0x5c28, 99),
    (0x5c30, 58),
    (0x5c38, 55),
    (0x5c39, 0),
    (0x5c40, 158),
    (0x5c48, 98),
    (0x5c50, 80),
    (0x5c58, 2),
    (0x5c60, 50),
    (0x5c68, 17),
    (0x5c70, 79),
    (0x5c78, 0),
    (0x5c80, 78),
    (0x5c88, 0),
    (0x5c90, 53),
    (0x5c98, 18),
    (0x5c99, 0),
    (0x5ca0, 88),
    (0x5ca1, 0),
    (0x5ca8, 156),
    (0x5cb0, 92),
    (0x5cb8, 90),
    (0x5cc0, 29),
    (0x5cd0, 0),
    (0x5ce0, 93),
    (0x5ce8, 72),
    (0x5ce9, 0),
    (0x5cf0, 75),
    (0x5cf8, 35),
    (0x5d00, 5),
    (0x5d08, 6),
    (0x5d10, 7),
    (0x5d18, 14),
    (0x5d28, 0),
    (0x5f00, 15),
    (0x6000, 0),
    (0x6080, 13),
    (0x6100, 0),
    (0x6180, 13),
    (0x6200, 0),
    (0x6280, 13),
    (0x6300, 0),
    (0x6380, 13),
    (0x6400, 0),
    (0x6480, 13),
    (0x6500, 0),
    (0x6580, 13),
    (0x6600, 0),
    (0x6680, 13),
    (0x6700, 0),
    (0x6780, 13),
    (0x6800, 0),
    (0x7000, 13),
    (0x7800, 12),
    (0x7a00, 10),
    (0x7a80, 11),
    (0x7b00, 10),
    (0x7b80, 11),
    (0x7c00, 134),
    (0x8000, 0),
    (0xc038, 119),
    (0xc040, 0),
    (0xc078, 119),
    (0xc080, 0),
    (0xc0b8, 119),
    (0xc0c0, 0),
    (0xc0f8, 119),
    (0xc100, 0),
    (0xc138, 119),
    (0xc140, 0),
    (0xc178, 119),
    (0xc180, 0),
    (0xc1b8, 119),
    (0xc1c0, 0),
    (0xc1f8, 119),
    (0xc200, 0),
    (0xc238, 119),
    (0xc240, 0),
    (0xc278, 119),
    (0xc280, 0),
    (0xc2b8, 119),
    (0xc2c0, 0),
    (0xc2f8, 119),
    (0xc300, 0),
    (0xc338, 119),
    (0xc340, 0),
    (0xc378, 119),
    (0xc380, 0),
    (0xc3b8, 119),
    (0xc3c0, 0),
    (0xc3f8, 119),
    (0xc400, 0),
    (0xc838, 123),
    (0xc840, 0),
    (0xc878, 123),
    (0xc880, 0),
    (0xc8b8, 123),
    (0xc8c0, 0),
    (0xc8f8, 123),
    (0xc900, 0),
    (0xc938, 123),
    (0xc940, 0),
    (0xc978, 123),
    (0xc980, 0),
    (0xc9b8, 123),
    (0xc9c0, 0),
    (0xc9f8, 123),
    (0xca00, 0),
    (0xca38, 123),
    (0xca40, 0),
    (0xca78, 123),
    (0xca80, 0),
    (0xcab8, 123),
    (0xcac0, 0),
    (0xcaf8, 123),
    (0xcb00, 0),
    (0xcb38, 123),
    (0xcb40, 0),
    (0xcb78, 123),
    (0xcb80, 0),
    (0xcbb8, 123),
    (0xcbc0, 0),
    (0xcbf8, 123),
    (0xcc00, 0),
    (0xd000, 128),
    (0xd400, 0),
    (0xd800, 128),
    (0xdc00, 0),
    (0xdc38, 121),
    (0xdc40, 0),
    (0xdc78, 121),
    (0xdc80, 0),
    (0xdcb8, 121),
    (0xdcc0, 0),
    (0xdcf8, 121),
    (0xdd00, 0),
    (0xdd38, 121),
    (0xdd40, 0),
    (0xdd78, 121),
    (0xdd80, 0),
    (0xddb8, 121),
    (0xddc0, 0),
    (0xddf8, 121),
    (0xde00, 0),
    (0xde38, 159),
    (0xde3c, 0),
    (0xdeb8, 120),
    (0xdec0, 0),
    (0xdef8, 122),
    (0xdef9, 0),
    (0xdefa, 122),
    (0xdefb, 0),
    (0xdefc, 122),
    (0xdefd, 0),
    (0xdefe, 122),
    (0xdeff, 0),
    (0xdf00, 129),
    (0xdf40, 0),
    (0xdf48, 127),
    (0xdf4c, 0),
    (0xdf50, 126),
    (0xdf51, 0),
    (0xdf52, 126),
    (0xdf53, 0),
    (0xdf58, 124),
    (0xdf5c, 0),
    (0xdf60, 125),
    (0xdf61, 0),
    (0xdf62, 125),
    (0xdf63, 0),
    (0xdf80, 129),
    (0xdfc0, 0),
    (0xe000, 61),
    (0xe100, 0),
    (0xe240, 65),
    (0xe241, 0),
    (0xe250, 1),
    (0xe260, 147),
    (0xe270, 0),
    (0xe290, 66),
    (0xe2a0, 67),
    (0xe2b0, 67),
    (0xe2b1, 167),
    (0xe2c0, 0),
    (0xe300, 60),
    (0xe301, 0),
    (0xe330, 69),
    (0xe331, 0),
    (0xe340, 68),
    (0xe341, 0),
    (0xe350, 68),
    (0xe351, 0),
    (0xea00, 132),
    (0xea40, 0),
    (0xea60, 141),
    (0xea80, 142),
    (0xeac0, 143),
    (0xeae0, 0),
    (0xeb00, 131),
    (0xeb20, 130),
    (0xeb40, 144),
    (0xeb60, 0),
    (0xeb80, 165),
    (0xebc0, 0),
    (0xebe0, 145),
    (0xebe1, 0),
    (0xebf8, 102),
    (0xec00, 110),
    (0xed00, 103),
    (0xee00, 0),
    (0xee40, 109),
    (0xee50, 0),
    (0xeed0, 100),
    (0xeed8, 101),
    (0xeee0, 0),
    (0xeef0, 104),
    (0xef00, 0),
    (0xef10, 51),
    (0xef18, 0),
    (0xef40, 105),
    (0xef48, 106),
    (0xef50, 107),
    (0xef58, 108),
    (0xef60, 0),
    (0xef90, 20),
    (0xef98, 111),
    (0xef99, 0),
    (0xefa0, 115),
    (0xefa1, 0),
    (0xefd0, 116),
    (0xefd1, 0),
    (0xefd8, 117),
    (0xefda, 0),
    (0xeff0, 118),
    (0xeff2, 0),
    (0xf0a8, 112),
    (0xf0a9, 0),
    (0xf0c8, 70),
    (0xf0c9, 0),
    (0xf0f0, 133),
    (0xf0f8, 152),
    (0xf100, 0),
    (0xf6e0, 114),
    (0xf6e1, 0),
    (0xfbe0, 113),
    (0xfbe1, 0),
];
/// The opcode map: bits 48..63 select a family.
pub(super) static OPCODE: [u8; 65536] = {
    let mut t = [0u8; 65536];
    let mut k = 0;
    while k < OPCODE_RUNS.len() {
        let (start, family) = OPCODE_RUNS[k];
        let end = [65536, OPCODE_RUNS[(k + 1) % OPCODE_RUNS.len()].0]
            [(k + 1 < OPCODE_RUNS.len()) as usize];
        let mut o = start;
        while o < end {
            t[o as usize] = family;
            o += 1;
        }
        k += 1;
    }
    t
};
