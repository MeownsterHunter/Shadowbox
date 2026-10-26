//! Where a translation stops, named by what it met. Each state is its own
//! shape: a word no decoder row names goes to discovery, and every other
//! state is an unwanted shape stopped at the edge that found it. The text
//! each state prints is the text the CLI and novena have always shown.
use crate::{headers::OutsideHeader, ir::Instruction};
use std::fmt;

/// The named states a translation can stop in.
#[derive(Debug)]
pub enum Stopped {
    /// A word no decoder row names. It goes to discovery.
    Unknown(Box<Instruction>),
    /// A decoded form whose lowering is not built.
    Unbuilt(Box<Instruction>),
    /// A decoded form a proof found to be one Shadowbox does not take.
    Unwanted(Box<Instruction>),
    /// Padding that control reaches.
    LivePadding(Box<Instruction>),
    /// A stack push or pop whose destination the decoder did not resolve.
    UnresolvedStack(Box<Instruction>),
    /// An OUT whose stream no value proof named.
    UnprovenStream(Box<Instruction>),
    /// Code bytes that end inside an instruction word.
    PartialWord,
    /// A program record, header or cb1 snapshot Shadowbox does not read.
    Malformed(Malformed),
    /// A surface target with no Vulkan image shape.
    SurfaceTarget(u8),
    /// Thread or CTA system registers outside a compute stage.
    ComputeRegisters,
    /// Workgroup storage outside a compute stage.
    SharedOutsideCompute,
    /// InvocationInfo without the driver's geometry input topology.
    InvocationTopology,
    /// Global memory whose descriptor lowering has no proof.
    DescriptorMemory(DescriptorMemory),
    /// Stage I/O the module cannot lay out.
    NoLayout(NoLayout),
    /// A branch to an address outside the decoded code.
    MissingTarget(MissingTarget),
    /// A stack pop at this byte address that no push proves.
    UnprovenPop(u32),
    /// A conditional BRA at this address with no reducible merge.
    NoMerge(u32),
    /// A final conditional BRA at this address with no fallthrough.
    NoFallthrough(u32),
    /// Warp reconvergence the token regions cannot hold.
    Reconvergence(u32, Unreconverged),
    /// A cache that cannot take the translation.
    Cache(CacheStop),
}

impl fmt::Display for Stopped {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unknown(x)
            | Self::Unbuilt(x)
            | Self::Unwanted(x)
            | Self::LivePadding(x)
            | Self::UnresolvedStack(x)
            | Self::UnprovenStream(x) => {
                write!(f, "unsupported instruction: {x}")
            }
            Self::PartialWord => f.write_str("malformed code: incomplete instruction word"),
            Self::Malformed(m) => write!(f, "invalid SPIR-V module: {m}"),
            Self::SurfaceTarget(target) => {
                write!(f, "unsupported instruction: unsupported surface target {target}")
            }
            Self::ComputeRegisters => f.write_str(
                "unsupported instruction: thread and CTA system registers require a compute stage",
            ),
            Self::SharedOutsideCompute => {
                f.write_str("unsupported control flow: shared memory outside a compute shader")
            }
            Self::InvocationTopology => f.write_str(
                "invalid SPIR-V module: input topology must be supplied for InvocationInfo",
            ),
            Self::DescriptorMemory(why) => {
                write!(f, "unsupported instruction: descriptor global memory: {why}")
            }
            Self::NoLayout(io) => write!(f, "invalid SPIR-V module: {io}"),
            Self::MissingTarget(missing) => write!(f, "unsupported control flow: {missing}"),
            Self::UnprovenPop(address) => write!(
                f,
                "missing cb1 data or control-flow proof: unresolved stack jump at byte 0x{address:x}; every executed history must have a matching push and one destination"
            ),
            Self::NoMerge(address) => write!(
                f,
                "unsupported control flow: conditional BRA at 0x{address:x} has no reducible merge"
            ),
            Self::NoFallthrough(address) => write!(
                f,
                "unsupported control flow: conditional BRA at 0x{address:x} has no fallthrough"
            ),
            Self::Reconvergence(address, why) => write!(
                f,
                "unsupported control flow: warp reconvergence at 0x{address:x}: {why}"
            ),
            Self::Cache(stop) => write!(f, "invalid SPIR-V module: {stop}"),
        }
    }
}

impl std::error::Error for Stopped {}

/// A program record, header or cb1 snapshot Shadowbox does not read.
#[derive(Debug)]
pub enum Malformed {
    /// A tail that names a cb1 trailer but does not form one.
    Trailer,
    /// A cb1 snapshot that ends inside a word.
    PartialCb1Word,
    /// A driver cb1 snapshot that differs from the record's.
    Cb1Disagrees,
    /// A header-prefixed input shorter than its generation's header.
    ShortHeader,
    /// A header that is neither 80 nor at least 128 bytes.
    HeaderLength,
    /// A header whose fields do not parse.
    Header(OutsideHeader),
    /// A record that already carries a cb1 trailer.
    TrailerExists,
    /// A trailer whose header size is not the target generation's.
    HeaderSize,
}

impl fmt::Display for Malformed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Trailer => "invalid cb1 record trailer",
            Self::PartialCb1Word => "cb1 snapshot contains a partial word",
            Self::Cb1Disagrees => "driver cb1 snapshot disagrees with record snapshot",
            Self::ShortHeader => "header-prefixed shader is shorter than its header",
            Self::HeaderLength => "header must contain 80 or at least 128 bytes",
            Self::Header(e) => return write!(f, "header: {e:?}"),
            Self::TrailerExists => "record already has a cb1 trailer",
            Self::HeaderSize => "cb1 record header size disagrees with target generation",
        })
    }
}

impl std::error::Error for Malformed {}

impl From<Malformed> for Stopped {
    fn from(malformed: Malformed) -> Self {
        Self::Malformed(malformed)
    }
}

/// Why global memory has no descriptor lowering.
#[derive(Debug)]
pub enum DescriptorMemory {
    MisalignedBase,
    DuplicateBinding,
    TextureSetOverlap,
    NotWord,
    UnalignedAddress,
    NoLowSource,
    NoHighSource,
    NoBinding,
}

impl fmt::Display for DescriptorMemory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::MisalignedBase => "cbuf base must be an aligned adjacent word pair",
            Self::DuplicateBinding => "duplicate cbuf base or storage binding",
            Self::TextureSetOverlap => "scalar texture set overlaps descriptor storage set",
            Self::NotWord => "only LDG/STG.E.B32 has a descriptor lowering",
            Self::UnalignedAddress => {
                "requires valid register pairs and nonnegative aligned offset"
            }
            Self::NoLowSource => "low address word has no unique cbuf source",
            Self::NoHighSource => "high address word is not the adjacent cbuf source",
            Self::NoBinding => "proven cbuf base has no host storage binding",
        })
    }
}

/// Stage I/O the module cannot lay out.
#[derive(Debug)]
pub enum NoLayout {
    StreamRouting,
    UnlinkedStreams,
    NonPointStreams,
    GeometryTopology,
    PassthroughOut,
    PassthroughUnlinked,
    OutputTopology,
    ZeroCounts,
    ControlPoints,
    TessellationState,
    PatchOutsideTessellation,
    UnlinkedPatch,
    PatchBelowBase,
    UnmappedPatchSlot,
    PatchAddress,
    Attribute,
    LegacyVaryingLocations,
    LegacyColorShading,
    OverlappingLocations,
    StreamLayout,
    AliasedBuiltins,
    LocationOverflow,
    StreamBuiltin,
    UnaliasedBuiltin,
    StreamLocations,
    Undeclared,
    OutOutsideGeometry,
}

impl fmt::Display for NoLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::StreamRouting => "invalid geometry stream routing",
            Self::UnlinkedStreams => "multiple emitted streams need linked output routing",
            Self::NonPointStreams => "nonzero geometry streams require point output",
            Self::GeometryTopology => "geometry input topology must be supplied by the driver",
            Self::PassthroughOut => "passthrough with explicit OUT is unsupported",
            Self::PassthroughUnlinked => "geometry passthrough needs linked next-stage inputs",
            Self::OutputTopology => "unsupported geometry output topology",
            Self::ZeroCounts => "geometry vertex and invocation counts must be nonzero",
            Self::ControlPoints => "tessellation output control point count must be 1..32",
            Self::TessellationState => {
                "tessellation domain, spacing and vertex order must be supplied by the driver"
            }
            Self::PatchOutsideTessellation => "patch I/O is only available in tessellation",
            Self::UnlinkedPatch => "generic patch I/O needs a linked driver layout",
            Self::PatchBelowBase => "patch address precedes the producer's generic base",
            Self::UnmappedPatchSlot => "generic patch slot has no linked Vulkan location",
            Self::PatchAddress => "unsupported patch address",
            Self::Attribute => "unsupported geometry/tessellation attribute",
            Self::LegacyVaryingLocations => "legacy varying interface has no free user location",
            Self::LegacyColorShading => {
                "legacy color shade-model state must be supplied by the driver"
            }
            Self::OverlappingLocations => "attribute locations overlap",
            Self::StreamLayout => "emitted geometry stream needs an output layout",
            Self::AliasedBuiltins => "native geometry builtins cannot have location aliases",
            Self::LocationOverflow => "geometry stream location overflow",
            Self::StreamBuiltin => "unsupported geometry stream builtin",
            Self::UnaliasedBuiltin => "geometry stream builtin needs a location alias",
            Self::StreamLocations => "geometry stream output locations overlap or overflow",
            Self::Undeclared => "attribute not declared",
            Self::OutOutsideGeometry => "OUT requires geometry stage",
        })
    }
}

impl From<NoLayout> for Stopped {
    fn from(io: NoLayout) -> Self {
        Self::NoLayout(io)
    }
}

/// A BRA or BRX at `at` whose target `target` is not decoded code.
#[derive(Debug)]
pub struct MissingTarget {
    pub at: u32,
    pub target: i64,
    pub gap: TargetGap,
}

/// Where a missing branch target lies.
#[derive(Debug)]
pub enum TargetGap {
    /// Outside the code address space.
    OutsideCode,
    /// Outside the decoded range `first..=last`.
    OutsideDecoded { first: u32, last: u32 },
    /// Inside the decoded range, but not an instruction slot.
    NotASlot,
}

impl fmt::Display for MissingTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BRA/BRX at 0x{:x} targets missing address 0x{:x} (",
            self.at, self.target
        )?;
        match self.gap {
            TargetGap::OutsideCode => f.write_str("outside the code address space")?,
            TargetGap::OutsideDecoded { first, last } => write!(
                f,
                "outside decoded instruction range 0x{first:x}..=0x{last:x}; supply all reachable code"
            )?,
            TargetGap::NotASlot => f.write_str("not a decoded instruction slot")?,
        }
        f.write_str(")")
    }
}

/// Why warp reconvergence does not fit the token regions.
#[derive(Debug)]
pub enum Unreconverged {
    Unordered,
    EmptyPush,
    ConditionalPush,
    CrossingRegions,
    EntersBody {
        to: u32,
        start: u32,
        end: u32,
        push: u32,
    },
    BackwardEscape {
        to: u32,
    },
    MultipleLatches {
        head: u32,
    },
    CrossingIntervals,
    BypassesHeader {
        to: u32,
        head: u32,
    },
    IndirectBranch,
    PopOutsideRegion,
}

impl fmt::Display for Unreconverged {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Unordered => f.write_str("instruction addresses must be strictly ordered"),
            Self::EmptyPush => f.write_str("push has no body"),
            Self::ConditionalPush => {
                f.write_str("requires an unconditional push and a forward decoded target")
            }
            Self::CrossingRegions => f.write_str("crossing token regions"),
            Self::EntersBody { to, start, end, push } => write!(
                f,
                "edge to 0x{to:x} enters token body 0x{start:x}..0x{end:x} without push 0x{push:x}"
            ),
            Self::BackwardEscape { to } => write!(
                f,
                "backward escape to 0x{to:x} crosses a token merge; iteration ordering is unprovable"
            ),
            Self::MultipleLatches { head } => write!(
                f,
                "loop at 0x{head:x} has multiple latches; iteration ordering is unprovable"
            ),
            Self::CrossingIntervals => f.write_str("crossing backward-edge intervals"),
            Self::BypassesHeader { to, head } => {
                write!(f, "edge to 0x{to:x} bypasses loop header 0x{head:x}")
            }
            Self::IndirectBranch => f.write_str("indirect branches need structured target regions"),
            Self::PopOutsideRegion => {
                f.write_str("stack pop does not leave an enclosing token region")
            }
        }
    }
}

/// A cache that cannot take the translation.
#[derive(Debug)]
pub enum CacheStop {
    /// The caller's stage is not the one the input header names.
    StageDisagrees,
    /// The program awaits graphics state, so its module has no cache key.
    AwaitingState,
    /// Writing the entry failed.
    Publish(std::io::Error),
}

impl fmt::Display for CacheStop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StageDisagrees => f.write_str("cache stage disagrees with input header"),
            Self::AwaitingState => f.write_str("cannot cache a module with missing graphics state"),
            Self::Publish(e) => write!(f, "cache: {e}"),
        }
    }
}

/// What a translation gives: the module, or the state it stopped in.
#[derive(Debug)]
pub enum Translation<M> {
    Module(M),
    Stopped(Stopped),
}

/// The borrowed shape, for a host that wants `?` on its side.
impl<M> From<Translation<M>> for Result<M, Stopped> {
    fn from(translation: Translation<M>) -> Self {
        match translation {
            Translation::Module(module) => Ok(module),
            Translation::Stopped(stopped) => Err(stopped),
        }
    }
}

impl<M> From<Result<M, Stopped>> for Translation<M> {
    fn from(result: Result<M, Stopped>) -> Self {
        match result {
            Ok(module) => Self::Module(module),
            Err(stopped) => Self::Stopped(stopped),
        }
    }
}
