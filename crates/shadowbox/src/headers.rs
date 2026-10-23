//! Shader header and constant-buffer interpretation. Header parsing is separate
//! from instruction decoding because novena supplies two memory locations.
//!
//! The header is an edge: its stage field decides which stage's header the
//! 32 words are, and each stage's header holds only the fields that stage
//! reads (Mesa nak/sph.rs; Ryujinx Translation/ShaderHeader.cs).

/// A parsed SM 5.x shader-program header: the header of the stage its stage
/// field names.
#[derive(Debug, PartialEq, Eq)]
pub enum ShaderHeader {
    Vertex(VertexHeader),
    Fragment(FragmentHeader),
    Compute(ComputeHeader),
    Geometry(GeometryHeader),
    TessellationControl(TessellationControlHeader),
    TessellationEvaluation(TessellationEvaluationHeader),
}

/// Vertex/tessellation/geometry generic input and output components read
/// or written, one bit per component of each location (Mesa sph.rs
/// imap_g_vtg 192..320 and omap_g 432..560).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VertexMaps {
    inputs: u128,
    outputs: u128,
}

#[derive(Debug, PartialEq, Eq)]
pub struct VertexHeader {
    maps: VertexMaps,
}

#[derive(Debug, PartialEq, Eq)]
pub struct FragmentHeader {
    /// The input map: one interpolation per component (Mesa sph.rs
    /// imap_g_ps; Ryujinx Translation/ShaderHeader.cs:30-48).
    inputs: [Interpolation; 128],
    /// Written render-target components, 4 bits per target (Mesa
    /// sph.rs:185-188; Ryujinx Translation/ShaderHeader.cs:162-167).
    targets: u32,
    /// Writes the sample mask (bit 608) and depth (bit 609) (Mesa
    /// cla097sph.h:628-633; Ryujinx ShaderHeader.cs:162-167).
    writes_sample_mask: bool,
    writes_depth: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub struct ComputeHeader(());

#[derive(Debug, PartialEq, Eq)]
pub struct GeometryHeader {
    maps: VertexMaps,
    /// Mesa sph.rs:305-312; cla097sph.h:66 (bits 88..96).
    threads_per_input_primitive: u8,
    /// Mesa sph.rs:326-350; cla097sph.h:68-73 (bits 120..124, 128..140).
    output_topology: u8,
    max_output_vertices: u16,
    /// Mesa nak/sph.rs:240-243; Ryujinx Translation/ShaderHeader.cs:125:
    /// bit24 enables geometry passthrough, lowered at shader return.
    passthrough: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub struct TessellationControlHeader {
    maps: VertexMaps,
    /// The output patch's control points (bits 88..96).
    threads_per_input_primitive: u8,
}

#[derive(Debug, PartialEq, Eq)]
pub struct TessellationEvaluationHeader {
    maps: VertexMaps,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Interpolation {
    Unused,
    Constant,
    Perspective,
    ScreenLinear,
}

/// A header shape the edge leaves outside: someone else's words, never
/// a Shadowbox header.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutsideHeader {
    /// Fewer words than the header's 32.
    Short { words: usize },
    /// A stage field value (6..=15) that names no Maxwell stage.
    ReservedStage(u8),
}

/// Shader stage known to Shadowbox: the name of a header's stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Vertex,
    Fragment,
    Compute,
    Geometry,
    TessellationControl,
    TessellationEvaluation,
}

/// Each 2-bit input-map field value as its interpolation (Mesa sph.rs
/// imap_g_vtg; Ryujinx Translation/ShaderHeader.cs:30-48).
const INTERPOLATION: [Interpolation; 4] = [
    Interpolation::Unused,
    Interpolation::Constant,
    Interpolation::Perspective,
    Interpolation::ScreenLinear,
];

/// `FIRST_USED[earlier][later]`: the first used mode of two components in
/// order, `later` when `earlier` is unused.
const FIRST_USED: [[Interpolation; 4]; 4] = [
    INTERPOLATION,
    [Interpolation::Constant; 4],
    [Interpolation::Perspective; 4],
    [Interpolation::ScreenLinear; 4],
];

/// The interpolation a declared input takes, in `Interpolation` order: a
/// location the map leaves unused is declared perspective.
const DECLARED: [Interpolation; 4] = [
    Interpolation::Perspective,
    Interpolation::Constant,
    Interpolation::Perspective,
    Interpolation::ScreenLinear,
];

impl Interpolation {
    /// The interpolation the emitter declares for an input with this mode:
    /// itself, or perspective for an unused one.
    pub(crate) const fn declared(self) -> Self {
        DECLARED[self as usize]
    }
}

impl ShaderHeader {
    /// The stage this header is.
    pub const fn stage(&self) -> Stage {
        match self {
            Self::Vertex(_) => Stage::Vertex,
            Self::Fragment(_) => Stage::Fragment,
            Self::Compute(_) => Stage::Compute,
            Self::Geometry(_) => Stage::Geometry,
            Self::TessellationControl(_) => Stage::TessellationControl,
            Self::TessellationEvaluation(_) => Stage::TessellationEvaluation,
        }
    }
}

impl VertexMaps {
    fn decode(words: &[u32; 32]) -> Self {
        Self {
            inputs: bits128(words, 192),
            outputs: bits128(words, 432),
        }
    }

    /// Generic input components the header marks.
    pub(crate) const fn inputs(self) -> u128 {
        self.inputs
    }

    /// Generic output components the header marks.
    pub(crate) const fn outputs(self) -> u128 {
        self.outputs
    }

    pub(crate) fn output_locations(self) -> u32 {
        (0..32).fold(0, |mask, location| {
            mask | (((self.outputs >> (location * 4)) & 0xf != 0) as u32) << location
        })
    }
}

impl VertexHeader {
    pub(crate) const fn maps(&self) -> VertexMaps {
        self.maps
    }
}

impl FragmentHeader {
    /// The first used interpolation mode of a generic input location
    /// (Ryujinx Translation/ShaderHeader.cs:30-48), `Unused` when none of
    /// its components is used.
    pub fn input_mode(&self, location: usize) -> Interpolation {
        self.inputs[location * 4..location * 4 + 4]
            .iter()
            .rev()
            .fold(Interpolation::Unused, |later, &earlier| {
                FIRST_USED[earlier as usize][later as usize]
            })
    }

    pub(crate) fn input_locations(&self) -> u32 {
        (0..32).fold(0, |mask, location| {
            mask | ((self.input_mode(location) != Interpolation::Unused) as u32) << location
        })
    }

    /// Written render-target components, 4 bits per target.
    pub const fn targets(&self) -> u32 {
        self.targets
    }

    pub const fn writes_sample_mask(&self) -> bool {
        self.writes_sample_mask
    }

    pub const fn writes_depth(&self) -> bool {
        self.writes_depth
    }
}

impl GeometryHeader {
    pub(crate) const fn maps(&self) -> VertexMaps {
        self.maps
    }

    pub(crate) const fn threads_per_input_primitive(&self) -> u8 {
        self.threads_per_input_primitive
    }

    pub(crate) const fn output_topology(&self) -> u8 {
        self.output_topology
    }

    pub(crate) const fn max_output_vertices(&self) -> u16 {
        self.max_output_vertices
    }

    pub(crate) const fn passthrough(&self) -> bool {
        self.passthrough
    }
}

impl TessellationControlHeader {
    pub(crate) const fn maps(&self) -> VertexMaps {
        self.maps
    }

    pub(crate) const fn threads_per_input_primitive(&self) -> u8 {
        self.threads_per_input_primitive
    }
}

impl TessellationEvaluationHeader {
    pub(crate) const fn maps(&self) -> VertexMaps {
        self.maps
    }
}

fn bits(words: &[u32; 32], start: usize, width: usize) -> u32 {
    (0..width).fold(0, |value, bit| {
        value | (((words[(start + bit) / 32] >> ((start + bit) % 32)) & 1) << bit)
    })
}

fn bits128(words: &[u32; 32], start: usize) -> u128 {
    (0..4).fold(0, |value, i| {
        value | (u128::from(bits(words, start + i * 32, 32)) << (i * 32))
    })
}

type Decode = fn(&[u32; 32]) -> Result<ShaderHeader, OutsideHeader>;

/// Each 4-bit header stage field value (bits 10..14) as the header of the
/// stage it names (Mesa sph.rs:204-215 and cla097sph.h:37-42).
const STAGE_FIELD: [Decode; 16] = [
    |_| Ok(ShaderHeader::Compute(ComputeHeader(()))),
    |words| {
        Ok(ShaderHeader::Vertex(VertexHeader {
            maps: VertexMaps::decode(words),
        }))
    },
    |words| {
        Ok(ShaderHeader::TessellationControl(
            TessellationControlHeader {
                maps: VertexMaps::decode(words),
                threads_per_input_primitive: bits(words, 88, 8) as u8,
            },
        ))
    },
    |words| {
        Ok(ShaderHeader::TessellationEvaluation(
            TessellationEvaluationHeader {
                maps: VertexMaps::decode(words),
            },
        ))
    },
    |words| {
        Ok(ShaderHeader::Geometry(GeometryHeader {
            maps: VertexMaps::decode(words),
            threads_per_input_primitive: bits(words, 88, 8) as u8,
            output_topology: bits(words, 120, 4) as u8,
            max_output_vertices: bits(words, 128, 12) as u16,
            passthrough: bits(words, 24, 1) == 1,
        }))
    },
    |words| {
        Ok(ShaderHeader::Fragment(FragmentHeader {
            inputs: std::array::from_fn(|index| {
                INTERPOLATION[bits(words, 192 + index * 2, 2) as usize]
            }),
            targets: bits(words, 576, 32),
            writes_sample_mask: bits(words, 608, 1) == 1,
            writes_depth: bits(words, 609, 1) == 1,
        }))
    },
    |_| Err(OutsideHeader::ReservedStage(6)),
    |_| Err(OutsideHeader::ReservedStage(7)),
    |_| Err(OutsideHeader::ReservedStage(8)),
    |_| Err(OutsideHeader::ReservedStage(9)),
    |_| Err(OutsideHeader::ReservedStage(10)),
    |_| Err(OutsideHeader::ReservedStage(11)),
    |_| Err(OutsideHeader::ReservedStage(12)),
    |_| Err(OutsideHeader::ReservedStage(13)),
    |_| Err(OutsideHeader::ReservedStage(14)),
    |_| Err(OutsideHeader::ReservedStage(15)),
];

/// Parse the 128-byte SM 5.x type-1 shader-program header: its first 32
/// words, read once as the header of the stage their stage field names.
pub fn parse(words: &[u32]) -> Result<ShaderHeader, OutsideHeader> {
    <&[u32; 32]>::try_from(&words[..words.len().min(32)])
        .map_err(|_| OutsideHeader::Short { words: words.len() })
        .and_then(|words| STAGE_FIELD[bits(words, 10, 4) as usize](words))
}

/// The program interface the emitter reads: a parsed header, or a program
/// record without one. A headerless program is its own kind: a fragment
/// program with no input map that writes all four components of render
/// target 0 (the emitter's established headerless contract).
#[derive(Clone, Copy, Debug)]
pub enum Interface<'a> {
    Header(&'a ShaderHeader),
    Headerless,
}

impl Interface<'_> {
    pub const fn stage(self) -> Stage {
        match self {
            Self::Header(header) => header.stage(),
            Self::Headerless => Stage::Fragment,
        }
    }
}
