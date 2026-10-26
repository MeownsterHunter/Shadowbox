//! Surface (storage image) semantics for SULD, SUST, SUATOM and SURED.
//! Ported from Ryujinx Instructions/InstEmitSurface.cs:180-799 and
//! CodeGen/Spirv/Instructions.cs:607-684; the MIT notice at the emitter's
//! helpers covers this adapted mapping. Maxwell has no SUQ: image size
//! queries are TXQ. Bindless (.B) forms differ only in where the handle
//! comes from, which the emitter's handle tracing resolves before this.

use crate::spirv::Op;

/// How one surface access treats the cache, parsed from the raw 24..26
/// field. Ryujinx gives only CG a meaning (Coherent,
/// InstEmitSurface.cs:364-367,677-679); every other policy is a plain
/// access there and here.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum CachePolicy {
    Plain,
    /// CG: the image variable is decorated Coherent
    /// (Ryujinx CodeGen/Spirv/Declarations.cs:265-267).
    Coherent,
    /// Store CS (CacheOpSt 2): Mesa sm50.rs:2424-2431,2521-2527 and
    /// ir.rs:2545-2559 call it CacheStreaming. Kept as a Nontemporal write
    /// rather than dropped as Ryujinx does (spirv.core.grammar.json:
    /// 11990-11994).
    Streaming,
}

impl CachePolicy {
    /// CacheOpLd CA, CG, CI, CV (Ryujinx Decoders/InstDecoders.cs:410-416)
    /// or CacheOpSt WB, CG, CS, WT (:418-424).
    pub(super) fn of(store: bool, cache: u8) -> Self {
        const LOAD: [CachePolicy; 4] = [
            CachePolicy::Plain,
            CachePolicy::Coherent,
            CachePolicy::Plain,
            CachePolicy::Plain,
        ];
        const STORE: [CachePolicy; 4] = [
            CachePolicy::Plain,
            CachePolicy::Coherent,
            CachePolicy::Streaming,
            CachePolicy::Plain,
        ];
        [LOAD, STORE][usize::from(store)][usize::from(cache & 3)]
    }
}

/// Coordinate registers read for a surface target: SuDim's dimensions plus
/// one array layer (Ryujinx InstEmitSurface.cs:229-249,606-618). 1D, buffer
/// one; 1D array, 2D two; 2D array, 3D three.
pub(super) fn coordinate_count(target: u8) -> Option<u8> {
    match target {
        0 | 1 => Some(1),
        2 | 3 => Some(2),
        4 | 5 => Some(3),
        _ => None,
    }
}

/// log2 of the texel bytes a byte-addressed SULD.D/SUST.D X is divided by
/// (Ryujinx InstEmitSurface.cs:574-588, by SuSize).
pub(super) fn texel_bytes_log2(size: u8) -> u32 {
    [0, 0, 1, 1, 2, 3, 4, 4][usize::from(size & 7)]
}

/// The SPIR-V atomic for SUATOM/SURED op 0..8 or CAS (9). Signed sizes pick
/// the signed min/max (Ryujinx CodeGen/Spirv/Instructions.cs:663-684).
/// Opcodes: Mesa src/compiler/spirv/spirv.h:1946-1959.
pub(super) fn atomic_opcode(op: u8, signed: bool) -> Option<Op> {
    match op {
        0 => Some(Op::AtomicIAdd),
        1 => Some([Op::AtomicUMin, Op::AtomicSMin][usize::from(signed)]),
        2 => Some([Op::AtomicUMax, Op::AtomicSMax][usize::from(signed)]),
        3 => Some(Op::AtomicIIncrement),
        4 => Some(Op::AtomicIDecrement),
        5 => Some(Op::AtomicAnd),
        6 => Some(Op::AtomicOr),
        7 => Some(Op::AtomicXor),
        8 => Some(Op::AtomicExchange),
        9 => Some(Op::AtomicCompareExchange),
        _ => None,
    }
}

/// Raw image formats: Ryujinx InstEmitSurface.cs:590-604 and
/// CodeGen/Spirv/Declarations.cs:294-300,302,314,326; enum values:
/// Mesa src/compiler/spirv/spirv.h:320-331. Formatted access uses the
/// existing float formatless binding; raw access requires a matching view.
pub(super) fn surface_format(format: u8, size: u8) -> Option<u32> {
    match (format, size) {
        (0, 0) => Some(0),
        (1, 0) => Some(39),
        (1, 1) => Some(29),
        (1, 2) => Some(38),
        (1, 3) => Some(28),
        (1, 4) => Some(33),
        (1, 5) => Some(35),
        (1, 6 | 7) => Some(30),
        _ => None,
    }
}

/// Return the SPIR-V image dimension and arrayed bit for a Maxwell surface
/// target. The decoder reads the field from bit 33, so it holds the
/// envytools/nouveau values (t1d=0, b1d=2, a1d=4, t2d=6, a2d=8, t3d=10 at
/// bit 32; envytools gm107.c:759-766, nv50_ir_emit_gm107.cpp:3261-3282)
/// halved: 0 1D, 1 buffer, 2 1D array, 3 2D, 4 2D array, 5 3D, matching
/// Ryujinx's SuDim (Decoders/InstDecoders.cs:584-592,4589).
pub(super) fn surface_image_shape(target: u8) -> Option<(u32, u32)> {
    match target {
        0 => Some((0, 0)),
        1 => Some((5, 0)),
        2 => Some((0, 1)),
        3 => Some((1, 0)),
        4 => Some((1, 1)),
        5 => Some((2, 0)),
        _ => None,
    }
}
