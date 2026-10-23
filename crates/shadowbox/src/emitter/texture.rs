//! Texture operand registers. Each encoding has its own layout: vector
//! operands pack in order, scalar encodings follow their mode table, TXD
//! packs its layer and offsets after the coordinates.
use super::{alu, late_constant};
use crate::ir::{ops, GatherOffsets, Register, TexOp};
use crate::reaching::ReachingDefinitions;
use crate::spirv::Op;

/// Register positions after applying the scalar mode table. Mesa's vector
/// packing is nak_nir_lower_tex.c:350-375. The SM50 scalar layouts are in
/// Ryujinx Instructions/InstEmitTexture.cs:394-601; current NAK only rebalances
/// scalar pairs on SM70+ (nak_nir_lower_tex.c:147-186). Registers past RZ
/// read zero, following Ryujinx InstEmitTexture.cs:344-361.
pub(super) struct TextureSources {
    pub coords: [Register; 3],
    pub coord_count: usize,
    pub layer: Option<Register>,
    pub lod: Option<Register>,
    pub offset: Option<Register>,
    pub offset2: Option<Register>,
    pub dref: Option<Register>,
    pub sample: Option<Register>,
    /// TXD array layer in bits 0..15 of the packed A word.
    pub packed_layer: Option<Register>,
    /// TXD dPdx, followed by dPdy.
    pub gradients: Option<Register>,
}

impl TextureSources {
    /// These coordinates and no other operand.
    fn at(coords: &[Register]) -> Self {
        let mut all = [Register(255); 3];
        all[..coords.len()].copy_from_slice(coords);
        Self {
            coords: all,
            coord_count: coords.len(),
            layer: None,
            lod: None,
            offset: None,
            offset2: None,
            dref: None,
            sample: None,
            packed_layer: None,
            gradients: None,
        }
    }
}

/// The A and B register runs of one texture instruction.
#[derive(Clone, Copy)]
struct Runs {
    src0: Register,
    src1: Register,
}

impl Runs {
    fn a(self, i: u8) -> Register {
        Register(self.src0.0.saturating_add(i))
    }
    fn b(self, i: u8) -> Register {
        Register(self.src1.0.saturating_add(i))
    }
}

/// One Mesa texture dimension's coordinates.
#[derive(Clone, Copy)]
struct Dimension {
    count: u8,
    /// Coordinates past the first two: one for 3D and cubes.
    third: u8,
    /// An array layer leads the vector A run.
    array: bool,
}

const D1: Dimension = Dimension {
    count: 1,
    third: 0,
    array: false,
};
const D1_ARRAY: Dimension = Dimension {
    count: 1,
    third: 0,
    array: true,
};
const D2: Dimension = Dimension {
    count: 2,
    third: 0,
    array: false,
};
const D2_ARRAY: Dimension = Dimension {
    count: 2,
    third: 0,
    array: true,
};
const D3: Dimension = Dimension {
    count: 3,
    third: 1,
    array: false,
};
const CUBE_ARRAY: Dimension = Dimension {
    count: 3,
    third: 1,
    array: true,
};

type Layout = fn(&ops::Texture) -> Option<TextureSources>;

/// By Mesa set_tex_dim (1D, 1D array, 2D, 2D array, 3D, unassigned, cube,
/// cube array). Dimension 5 names no texture and has no operands.
const BY_DIMENSION: [Layout; 8] = [
    |t| sources(t, D1, SCALAR_1D),
    |t| sources(t, D1_ARRAY, SCALAR_GENERAL),
    |t| sources(t, D2, SCALAR_2D),
    |t| sources(t, D2_ARRAY, SCALAR_2D_ARRAY),
    |t| sources(t, D3, SCALAR_GENERAL),
    |_| None,
    |t| sources(t, D3, SCALAR_GENERAL),
    |t| sources(t, CUBE_ARRAY, SCALAR_GENERAL),
];

pub(super) fn texture_sources(t: &ops::Texture) -> Option<TextureSources> {
    BY_DIMENSION[usize::from(t.dim)](t)
}

/// One encoding's layout for each operation.
struct Encoding {
    sample: fn(&ops::Texture, Runs, Dimension, &Scalar, u8) -> Option<TextureSources>,
    fetch: fn(&ops::Texture, Runs, Dimension, &Scalar, bool) -> Option<TextureSources>,
    gather: fn(&ops::Texture, Runs, Dimension, GatherOffsets) -> Option<TextureSources>,
    /// TXQ and TMML.
    query: fn(&ops::Texture, Runs, Dimension, VectorOperands) -> Option<TextureSources>,
}

/// By scalar: the vector encoding, or the register-pair encoding.
const ENCODING: [Encoding; 2] = [
    Encoding {
        sample: |t, runs, dimension, _, lod| {
            Some(vector(
                t,
                runs,
                dimension,
                VectorOperands {
                    layer: true,
                    b_run: true,
                    lod: SAMPLE_LOD_READS[usize::from(lod)],
                    offsets: AOFFI[usize::from(t.offset)],
                },
            ))
        },
        fetch: |t, runs, dimension, _, lod| {
            Some(vector(
                t,
                runs,
                dimension,
                VectorOperands {
                    layer: true,
                    b_run: true,
                    lod,
                    offsets: AOFFI[usize::from(t.offset)],
                },
            ))
        },
        gather: |t, runs, dimension, offsets| {
            Some(vector(
                t,
                runs,
                dimension,
                VectorOperands {
                    layer: true,
                    b_run: true,
                    lod: false,
                    offsets,
                },
            ))
        },
        query: |t, runs, dimension, operands| Some(vector(t, runs, dimension, operands)),
    },
    Encoding {
        sample: |t, runs, dimension, scalar, lod| {
            Some((scalar.sample)(runs, dimension, t.depth_compare, lod))
        },
        fetch: |t, runs, dimension, scalar, lod| Some((scalar.fetch)(t, runs, dimension, lod)),
        gather: |t, runs, _, offsets| Some(scalar_gather(t, runs, offsets)),
        query: |_, _, _, _| None,
    },
];

fn sources(t: &ops::Texture, dimension: Dimension, scalar: Scalar) -> Option<TextureSources> {
    let runs = Runs {
        src0: t.src0,
        src1: t.src1,
    };
    let encoding = &ENCODING[usize::from(t.scalar)];
    match t.op {
        TexOp::Grad => Some(gradient(t, runs, dimension)),
        TexOp::Sample { lod } => (encoding.sample)(t, runs, dimension, &scalar, lod),
        TexOp::Fetch { lod } => (encoding.fetch)(t, runs, dimension, &scalar, lod),
        TexOp::Gather { offsets, .. } => (encoding.gather)(t, runs, dimension, offsets),
        // TXQ operands are LOD, not B operands.
        TexOp::Query { .. } => (encoding.query)(
            t,
            runs,
            dimension,
            VectorOperands {
                layer: true,
                b_run: false,
                lod: false,
                offsets: GatherOffsets::Zero,
            },
        ),
        // LOD query coordinates omit the layer. Ryujinx Spirv/Instructions.cs:
        // 873-895 consumes only dimension coordinates, although Maxwell
        // InstEmitTexture.cs:858-875 puts the array index first in A.
        TexOp::Lod => (encoding.query)(
            t,
            runs,
            dimension,
            VectorOperands {
                layer: false,
                b_run: true,
                lod: false,
                offsets: AOFFI[usize::from(t.offset)],
            },
        ),
    }
}

/// Ryujinx InstEmitTexture.cs:956-1008: coordinates, then one packed A
/// word for the layer (array) and offsets (AOFFI); B holds dPdx then dPdy.
/// Bound TXD consumes no handle register.
fn gradient(t: &ops::Texture, runs: Runs, dimension: Dimension) -> TextureSources {
    let count = dimension.count;
    TextureSources {
        packed_layer: dimension.array.then_some(runs.a(count)),
        offset: t.offset.then_some(runs.a(count)),
        gradients: Some(runs.b(0)),
        ..TextureSources::at(&[runs.a(0), runs.a(1), runs.a(2)][..usize::from(count)])
    }
}

/// One operation's operands in the vector encoding.
struct VectorOperands {
    layer: bool,
    /// TXQ takes no B operands.
    b_run: bool,
    lod: bool,
    offsets: GatherOffsets,
}

/// By TEX lod mode: whether B carries a bias or level.
const SAMPLE_LOD_READS: [bool; 8] = [false, false, true, true, false, false, true, true];
/// By AOFFI: the offset words an ordinary operation reads.
const AOFFI: [GatherOffsets; 2] = [GatherOffsets::Zero, GatherOffsets::One];
/// By gather offsets: the offset words read, first and second.
const OFFSET_WORDS: [[bool; 2]; 3] = [[false, false], [true, false], [true, true]];

/// The array layer leads A, then the coordinates. Ryujinx
/// InstEmitTexture.cs:207-210,252-258: only bindless consumes a handle from
/// B before LOD/offset/depth parameters, so bound vector TEX starts its
/// auxiliary operands at B0. No extra SPIR-V operations are needed;
/// register positions fold here.
fn vector(
    t: &ops::Texture,
    runs: Runs,
    dimension: Dimension,
    operands: VectorOperands,
) -> TextureSources {
    let first = u8::from(dimension.array);
    let [offset, offset2] =
        OFFSET_WORDS[operands.offsets as usize].map(|word| word & operands.b_run);
    let lod = operands.lod & operands.b_run;
    let dref = t.depth_compare & operands.b_run;
    let sample = t.multisample & operands.b_run;
    let lod_at = 1 - t.bound_handle.into_iter().count() as u8;
    let offset_at = lod_at + u8::from(lod);
    let offset2_at = offset_at + u8::from(offset);
    let dref_at = offset2_at + u8::from(offset2);
    let sample_at = dref_at + u8::from(dref);
    TextureSources {
        layer: (dimension.array & operands.layer).then_some(runs.a(0)),
        lod: lod.then_some(runs.b(lod_at)),
        offset: offset.then_some(runs.b(offset_at)),
        offset2: offset2.then_some(runs.b(offset2_at)),
        dref: dref.then_some(runs.b(dref_at)),
        sample: sample.then_some(runs.b(sample_at)),
        ..TextureSources::at(
            &[runs.a(first), runs.a(first + 1), runs.a(first + 2)][..usize::from(dimension.count)],
        )
    }
}

/// One dimension's scalar register-pair layouts.
struct Scalar {
    sample: fn(Runs, Dimension, bool, u8) -> TextureSources,
    fetch: fn(&ops::Texture, Runs, Dimension, bool) -> TextureSources,
}

/// By TEX lod mode: explicit level, read from B.
const LEVEL_IN_B: [bool; 8] = [false, false, false, true, false, false, false, false];

/// A0, A1 then B0 for a third coordinate.
fn scalar_coordinates(runs: Runs, dimension: Dimension) -> TextureSources {
    TextureSources::at(&[runs.a(0), runs.a(1), runs.b(0)][..usize::from(2 + dimension.third)])
}

fn sample_general(
    runs: Runs,
    dimension: Dimension,
    depth_compare: bool,
    lod: u8,
) -> TextureSources {
    TextureSources {
        dref: depth_compare.then_some(runs.b(0)),
        lod: LEVEL_IN_B[usize::from(lod)]
            .then_some(runs.b(dimension.third | u8::from(depth_compare))),
        ..scalar_coordinates(runs, dimension)
    }
}

fn fetch_general(t: &ops::Texture, runs: Runs, dimension: Dimension, lod: bool) -> TextureSources {
    TextureSources {
        lod: lod.then_some(runs.b(0)),
        sample: t.multisample.then_some(runs.b(0)),
        offset: t.offset.then_some(runs.b(u8::from(lod))),
        ..scalar_coordinates(runs, dimension)
    }
}

/// Places a TEXS sample's sources.
type SampleSources = fn(Runs, Dimension, bool, u8) -> TextureSources;
/// Places a TLDS fetch's sources.
type FetchSources = fn(&ops::Texture, Runs, Dimension, bool) -> TextureSources;

/// By `!depth_compare && lod != 3`: TEXS 2D packs its second coordinate in B.
const SAMPLE_2D: [SampleSources; 2] = [sample_general, |runs, _, _, _| {
    TextureSources::at(&[runs.a(0), runs.b(0)])
}];
/// By lod mode: whether a TEXS 2D pair can hold no level.
const PAIRED_2D: [bool; 8] = [true, true, true, false, true, true, true, true];
/// By `!lod && !offset && !multisample`: TLDS 2D packs its second
/// coordinate in B.
const FETCH_2D: [FetchSources; 2] = [fetch_general, |_, runs, _, _| {
    TextureSources::at(&[runs.a(0), runs.b(0)])
}];

const SCALAR_GENERAL: Scalar = Scalar {
    sample: sample_general,
    fetch: fetch_general,
};
const SCALAR_1D: Scalar = Scalar {
    sample: |runs, _, _, _| TextureSources::at(&[runs.a(0)]),
    fetch: |_, runs, _, lod| TextureSources {
        lod: lod.then_some(runs.b(0)),
        ..TextureSources::at(&[runs.a(0)])
    },
};
const SCALAR_2D: Scalar = Scalar {
    sample: |runs, dimension, depth_compare, lod| {
        SAMPLE_2D[usize::from(!depth_compare & PAIRED_2D[usize::from(lod)])](
            runs,
            dimension,
            depth_compare,
            lod,
        )
    },
    fetch: |t, runs, dimension, lod| {
        FETCH_2D[usize::from(!(lod | t.offset | t.multisample))](t, runs, dimension, lod)
    },
};
const SCALAR_2D_ARRAY: Scalar = Scalar {
    sample: |runs, _, depth_compare, _| TextureSources {
        layer: Some(runs.a(0)),
        dref: depth_compare.then_some(runs.b(1)),
        ..TextureSources::at(&[runs.a(1), runs.b(0)])
    },
    fetch: |_, runs, _, _| TextureSources {
        layer: Some(runs.a(0)),
        ..TextureSources::at(&[runs.b(0), runs.b(1)])
    },
};

/// TLD4S: coordinates in A0 and B0 unless offsets or a reference take B.
fn scalar_gather(t: &ops::Texture, runs: Runs, offsets: GatherOffsets) -> TextureSources {
    let [offset, offset2] = OFFSET_WORDS[offsets as usize];
    TextureSources {
        offset: offset.then_some(runs.b(0)),
        offset2: offset2.then_some(runs.b(1)),
        dref: t.depth_compare.then_some(runs.b(offsets as u8)),
        ..TextureSources::at(
            &[[runs.a(0), runs.b(0)], [runs.a(0), runs.a(1)]]
                [usize::from(offset | t.depth_compare)],
        )
    }
}

/// What coordinate assembly reads: the reaching facts at one instruction,
/// the register variables and the coordinate types.
#[derive(Clone, Copy)]
pub(super) struct CoordinateTypes<'a> {
    pub reaching: &'a ReachingDefinitions,
    pub index: usize,
    pub regs: &'a [u32],
    pub uint: u32,
    pub float: u32,
    /// Float scalar, then vectors of two, three and four.
    pub sampled: [u32; 4],
    /// Unsigned scalar, then vectors of two, three and four.
    pub fetched: [u32; 4],
}

/// Texture coordinates from src0 per Mesa's packing: the array index first
/// (an integer, converted to float when sampling), then the coordinate
/// registers; the layer goes last in the SPIR-V vector. Float coordinates
/// for sampling and gathers, unsigned integers for fetches.
pub(super) struct Coordinates<'w, 'a> {
    pub w: &'w mut Vec<u32>,
    pub globals: &'w mut Vec<u32>,
    pub id: &'w mut u32,
    pub at: CoordinateTypes<'a>,
}

impl Coordinates<'_, '_> {
    /// A register's value as the reaching analysis knows it: a constant
    /// handed to `constant`, or the register variable handed to `dynamic`.
    fn read<T>(
        &mut self,
        register: Register,
        constant: impl FnOnce(&mut Self, u32) -> T,
        dynamic: impl FnOnce(&mut Self, u32) -> T,
    ) -> T {
        match self.at.reaching.constant(self.at.index, register.0) {
            Some(bits) => constant(self, bits),
            None => dynamic(self, self.at.regs[usize::from(register.0)]),
        }
    }

    fn load(&mut self, pointer: u32) -> u32 {
        alu(self.w, self.id, Op::Load, self.at.uint, &[pointer])
    }

    fn constant(&mut self, ty: u32, bits: u32) -> u32 {
        late_constant(self.globals, self.id, ty, bits)
    }

    /// Float coordinates. Array layers are signed, matching Ryujinx
    /// CodeGen/Spirv/Instructions.cs:1218-1225.
    pub fn sampled(&mut self, coords: &[Register], layer: Option<Register>) -> u32 {
        let mut parts = [0; 4];
        let float = self.at.float;
        coords.iter().zip(&mut parts).for_each(|(&register, part)| {
            *part = self.read(
                register,
                |c, bits| c.constant(float, bits),
                |c, pointer| {
                    let value = c.load(pointer);
                    alu(c.w, c.id, Op::Bitcast, float, &[value])
                },
            );
        });
        layer
            .into_iter()
            .zip(&mut parts[coords.len()..])
            .for_each(|(register, part)| {
                *part = self.read(
                    register,
                    |c, bits| c.constant(float, (bits as i32 as f32).to_bits()),
                    |c, pointer| {
                        let layer = c.load(pointer);
                        alu(c.w, c.id, Op::ConvertSToF, float, &[layer])
                    },
                );
            });
        let count = coords.len() + layer.into_iter().count();
        COMPOSE[count - 1](self.w, self.id, self.at.sampled[count - 1], &parts[..count])
    }

    /// Unsigned coordinates, moved by the AOFFI word when there is one:
    /// folded when it is known, extracted at run time otherwise.
    pub fn fetched_by(
        &mut self,
        coords: &[Register],
        layer: Option<Register>,
        offset: Option<Register>,
    ) -> u32 {
        match offset {
            None => self.fetched(coords, layer, NoOffset),
            Some(register) => self.read(
                register,
                |c, packed| c.fetched(coords, layer, ConstantOffset(packed)),
                |c, pointer| {
                    let packed = c.load(pointer);
                    c.fetched(coords, layer, DynamicOffset(packed))
                },
            ),
        }
    }

    /// Unsigned coordinates moved by the fetch offset.
    fn fetched<O: FetchOffset>(
        &mut self,
        coords: &[Register],
        layer: Option<Register>,
        offset: O,
    ) -> u32 {
        let mut parts = [0; 4];
        let uint = self.at.uint;
        coords
            .iter()
            .zip(&mut parts)
            .enumerate()
            .for_each(|(axis, (&register, part))| {
                *part = offset.coordinate(self, axis as u32, register);
            });
        layer
            .into_iter()
            .zip(&mut parts[coords.len()..])
            .for_each(|(register, part)| {
                *part = self.read(register, |c, bits| c.constant(uint, bits), Self::load);
            });
        let count = coords.len() + layer.into_iter().count();
        COMPOSE[count - 1](self.w, self.id, self.at.fetched[count - 1], &parts[..count])
    }
}

type Compose = fn(&mut Vec<u32>, &mut u32, u32, &[u32]) -> u32;
/// By component count less one: one component is itself; more construct
/// a vector, written directly into the preallocated word buffer.
const COMPOSE: [Compose; 4] = [|_, _, _, parts| parts[0], construct, construct, construct];

fn construct(w: &mut Vec<u32>, id: &mut u32, ty: u32, parts: &[u32]) -> u32 {
    alu(w, id, Op::CompositeConstruct, ty, parts)
}

/// How a fetch offset moves each coordinate. For integer fetches an offset
/// is signed coordinate addition. Ryujinx InstEmitTexture.cs:272-276,
/// 559-561. Adding it before OpImageFetch also supports dynamic offsets and
/// MS fetches without illegal Vulkan Offset operands.
trait FetchOffset {
    fn coordinate(&self, c: &mut Coordinates, axis: u32, register: Register) -> u32;
}

/// No AOFFI: coordinates as they are.
struct NoOffset;

impl FetchOffset for NoOffset {
    fn coordinate(&self, c: &mut Coordinates, _: u32, register: Register) -> u32 {
        let uint = c.at.uint;
        c.read(
            register,
            |c, bits| c.constant(uint, bits),
            Coordinates::load,
        )
    }
}

/// A known packed offset word: signed four-bit offsets at strides of four.
/// Constant coordinates fold with their offset.
struct ConstantOffset(u32);

impl FetchOffset for ConstantOffset {
    fn coordinate(&self, c: &mut Coordinates, axis: u32, register: Register) -> u32 {
        let uint = c.at.uint;
        let offset = ((self.0 << (28 - axis * 4)) as i32 >> 28) as u32;
        c.read(
            register,
            |c, bits| c.constant(uint, bits.wrapping_add(offset)),
            |c, pointer| {
                let value = c.load(pointer);
                let offset = c.constant(uint, offset);
                alu(c.w, c.id, Op::IAdd, uint, &[value, offset])
            },
        )
    }
}

/// A packed offset word loaded at run time.
struct DynamicOffset(u32);

impl FetchOffset for DynamicOffset {
    fn coordinate(&self, c: &mut Coordinates, axis: u32, register: Register) -> u32 {
        let uint = c.at.uint;
        let value = c.read(
            register,
            |c, bits| c.constant(uint, bits),
            Coordinates::load,
        );
        let start = c.constant(uint, axis * 4);
        let width = c.constant(uint, 4);
        let offset = alu(
            c.w,
            c.id,
            Op::BitFieldSExtract,
            uint,
            &[self.0, start, width],
        );
        alu(c.w, c.id, Op::IAdd, uint, &[value, offset])
    }
}

/// What a texture result becomes in helper invocations. Mesa
/// nak_nir_lower_tex.c:454 maps NODEP to skip_helpers;
/// nir_opt_load_skip_helpers.c:264-268 only marks results unneeded by
/// helpers. Select undefined helper results while keeping the image
/// operation uniform for implicit derivatives. This avoids a divergent
/// branch around each sample.
#[derive(Clone, Copy)]
pub(super) struct TextureResults {
    /// The helper state, once per instruction.
    helper: fn(&mut Vec<u32>, &mut u32, u32, u32) -> u32,
    /// Each result word, given the helper state.
    result: fn(&mut Vec<u32>, &mut u32, u32, [u32; 2], u32) -> u32,
    /// The HelperInvocation variable and the undefined result.
    variable: u32,
    undefined: u32,
}

/// Every invocation keeps its result.
pub(super) const EVERY_RESULT: TextureResults = TextureResults {
    helper: |_, _, _, _| 0,
    result: |_, _, _, _, value| value,
    variable: 0,
    undefined: 0,
};

/// Reads whether the invocation is a helper, as `boolt`, from `variable`.
type HelperState = fn(&mut Vec<u32>, &mut u32, u32, u32) -> u32;
/// By demotion: HelperInvocation is undefined after demotion, so query the
/// current state (Mesa spirv.h:2312) instead of loading the variable.
const HELPER_STATE: [HelperState; 2] = [
    |w, id, boolt, variable| alu(w, id, Op::Load, boolt, &[variable]),
    |w, id, boolt, _| alu(w, id, Op::IsHelperInvocationEXT, boolt, &[]),
];

impl TextureResults {
    /// Helper invocations take `undefined`. HelperInvocation exists only
    /// in fragment shaders.
    pub fn skip_helpers(variable: u32, undefined: u32, demotes: bool) -> Self {
        Self {
            helper: HELPER_STATE[usize::from(demotes)],
            result: |w, id, uint, [helper, undefined], value| {
                alu(w, id, Op::Select, uint, &[helper, undefined, value])
            },
            variable,
            undefined,
        }
    }

    /// Reads the helper state; the returned writer places each result.
    pub fn begin(self, w: &mut Vec<u32>, id: &mut u32, boolt: u32) -> ResultWriter {
        ResultWriter {
            helper: (self.helper)(w, id, boolt, self.variable),
            results: self,
        }
    }
}

/// One instruction's results under its helper state.
#[derive(Clone, Copy)]
pub(super) struct ResultWriter {
    helper: u32,
    results: TextureResults,
}

impl ResultWriter {
    pub fn result(self, w: &mut Vec<u32>, id: &mut u32, uint: u32, value: u32) -> u32 {
        (self.results.result)(w, id, uint, [self.helper, self.results.undefined], value)
    }
}

/// A decoded texture instruction's operand registers.
#[cfg(test)]
pub(super) fn decoded_sources(x: crate::ir::Instruction) -> TextureSources {
    match x.kind {
        crate::ir::InstructionKind::Texture(t) => texture_sources(&t),
        _ => None,
    }
    .unwrap()
}
