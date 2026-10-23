//! Narrow novena-facing interface. The host resolves the two locations observed
//! in novena's shader record and passes bounded bytes plus stage/context metadata.

/// Input borrowed for one translation.
pub struct ShaderInput<'a> {
    /// Machine-code bytes from the GPU-memory location.
    pub code: &'a [u8],
    /// Program-memory record with metadata. `header_prefixed_input` retains
    /// the full extended envelope here when it carries an immutable cb1
    /// snapshot; translation and cache identity extract its original header.
    pub program_record: &'a [u8],
}

/// Output suitable for a Vulkan shader module plus binding metadata.
#[derive(Clone, Debug)]
pub struct ShaderOutput {
    /// Serialized SPIR-V words.
    pub spirv: Vec<u32>,
    /// Bind each typed storage view to its source handle before execution.
    pub surface_descriptors: Vec<crate::emitter::SurfaceDescriptor>,
    /// Maxwell VOTE uses a 32-lane warp; the emitted subgroup operations require subgroup size 32.
    pub requires_subgroup_size_32: bool,
    /// Token-region modules declare SPV_KHR_maximal_reconvergence. The host
    /// must enable VK_KHR_shader_maximal_reconvergence and its feature.
    /// Mesa spirv.h:210; vulkan_core.h:13103-13111; SPV_KHR_maximal_reconvergence,
    /// Maximal Reconvergence defines the required nested merge behavior.
    pub requires_maximal_reconvergence: bool,
    /// KIL modules require VK_EXT_shader_demote_to_helper_invocation or its
    /// Vulkan 1.3 core feature shaderDemoteToHelperInvocation.
    /// Mesa from_nir.rs:3035-3050; vulkan_core.h:7747-7751,19635-19636.
    pub requires_demote_to_helper_invocation: bool,
}

/// Where a translation stops, and what it gives, crossing the novena boundary.
pub use crate::stopped::{
    CacheStop, DescriptorMemory, Malformed, MissingTarget, NoLayout, Stopped, TargetGap,
    Translation, Unreconverged,
};

/// Driver state absent from the SPH. Ryujinx Translation/ShaderDefinitions.cs:
/// 18-21,158-160 and CodeGen/Spirv/SpirvGenerator.cs:194-261.
#[derive(Clone, Copy, Debug, Default)]
pub struct GraphicsState<'a> {
    /// Immutable cbuf1 snapshot used to specialize BRX targets. Bind these
    /// same words at execution and retranslate if they change (Ryujinx
    /// Decoders/Decoder.cs:26-34,518-552). None uses general BRX dispatch.
    pub constant_buffer1: Option<&'a [u32]>,
    /// Input primitive topology, also required for vertex InvocationInfo
    /// (Ryujinx Instructions/InstEmitMove.cs:94-109).
    pub geometry_input: Option<GeometryInput>,
    /// Linked next-stage generic input locations, one bit per vec4. Required
    /// for passthrough; Some(0) means no generic inputs. Ryujinx
    /// Translation/AttributeUsage.cs:91-97, EmitterContext.cs:386-426.
    pub geometry_next_inputs: Option<u32>,
    /// Required when OUT emits to multiple streams. The driver must use this
    /// same routing for rasterization, stream capture and downstream linkage.
    pub geometry_streams: Option<GeometryStreams>,
    pub tessellation: Option<TessellationState>,
    /// Linked TCS/TES patch layout and locations. Required for generic patch I/O.
    pub patch_io: Option<PatchIo>,
    /// Bound sampler types for TXQ, which has no dimension field. Ryujinx
    /// Instructions/InstEmitTexture.cs:1116-1119 queries this driver metadata.
    /// Bindings use Mesa's 13-bit table index (nak/sm50.rs:2351-2354).
    pub texture_bindings: &'a [BoundTexture],
    /// Descriptor arrays indexed by handles loaded indirectly from a constant
    /// buffer. The array length is driver state and therefore part of the
    /// translation identity (Ryujinx BindlessToArray.cs:198-235).
    pub texture_arrays: &'a [TextureArray],
    /// Descriptor mode for global memory. Each entry binds the allocation
    /// whose guest base is held in two adjacent cbuf words. Bind at byte zero
    /// of that allocation, with no host alignment padding. The allocation
    /// base must be four-byte aligned. Empty retains the
    /// flat-address contract. Nonempty requires every global access to have
    /// a proven descriptor source. Ryujinx GlobalToStorage.cs:305-357,409-445.
    pub storage_buffers: &'a [StorageBufferBinding],
}

#[derive(Clone, Copy, Debug)]
pub enum GeometryInput {
    Points,
    Lines,
    LinesAdjacency,
    Triangles,
    TrianglesAdjacency,
}

impl GeometryInput {
    /// SPIR-V execution mode InputPoints..InputTrianglesAdjacency (Mesa
    /// spirv.core.grammar.json:12872-12908).
    pub(crate) fn execution_mode(self) -> u32 {
        [19, 20, 21, 22, 23][self as usize]
    }
    /// Vertices per input primitive (Ryujinx InputTopology.cs:26-34).
    pub(crate) fn vertices(self) -> u32 {
        [1, 2, 4, 3, 6][self as usize]
    }
    /// The non-adjacency vertices a passthrough shader re-emits (Ryujinx
    /// EmitterContext.cs:386-426).
    pub(crate) fn passthrough_vertices(self) -> &'static [u32] {
        const VERTICES: [&[u32]; 5] = [&[0], &[0, 1], &[1, 2], &[0, 1, 2], &[0, 2, 4]];
        VERTICES[self as usize]
    }
}

/// Vulkan allows each output BuiltIn only once (Vulkan-Docs
/// appendices/spirvenv.adoc:377-381, commit
/// e4e53e4b31e13eeaee1ad99fb940aa72b2ec1b14). Preserve other streams' builtin
/// values as explicitly linked user outputs, rather than discarding them.
/// Stream operands come from Mesa nak/sm50.rs:3257-3284; Ryujinx
/// Instructions/InstEmitAttribute.cs:232-250 does not retain those operands.
#[derive(Clone, Copy, Debug)]
pub struct GeometryStreams {
    /// Stream with native Position/PrimitiveId. Configure Vulkan's
    /// rasterization stream to match when this stream is rasterized.
    pub builtin_stream: u8,
    /// A layout for every emitted stream, including the builtin stream.
    pub locations: [Option<GeometryStreamLocations>; 4],
}

#[derive(Clone, Copy, Debug)]
pub struct GeometryStreamLocations {
    /// Maxwell generic vec4 slot i uses Vulkan location generic_base + i.
    pub generic_base: u32,
    /// Required on other emitted streams, forbidden on the builtin stream.
    pub position: Option<u32>,
    /// Required on other streams when the shader uses output PrimitiveId,
    /// forbidden on the builtin stream. Its bits retain their integer type.
    pub primitive_id: Option<u32>,
}
#[derive(Clone, Copy, Debug)]
pub enum TessellationDomain {
    Isolines,
    Triangles,
    Quads,
}
#[derive(Clone, Copy, Debug)]
pub enum TessellationSpacing {
    Equal,
    FractionalEven,
    FractionalOdd,
}
#[derive(Clone, Copy, Debug)]
pub struct TessellationState {
    pub domain: TessellationDomain,
    pub spacing: TessellationSpacing,
    /// Vulkan vertex order after the driver's front-face convention adjustment
    /// (Ryujinx SpirvGenerator.cs:222-236).
    pub clockwise: bool,
}

impl TessellationState {
    /// SPIR-V execution modes for the domain (Isolines 25, Triangles 22,
    /// Quads 24), the spacing (SpacingEqual 1, FractionalEven 2,
    /// FractionalOdd 3) and the vertex order (VertexOrderCw 4, Ccw 5).
    /// Mesa spirv.core.grammar.json:12992-13058.
    pub(crate) fn execution_modes(self) -> [u32; 3] {
        [
            [25, 22, 24][self.domain as usize],
            [1, 2, 3][self.spacing as usize],
            [5, 4][usize::from(self.clockwise)],
        ]
    }
}

/// The producer's allocation base for generic patch values. These are compiler
/// layouts, not interchangeable component origins: Mesa nak_private.h:39 and
/// nak_nir.c:422-423 use 0x20; Ryujinx Instructions/AttributeMap.cs:92 uses 0x18.
#[derive(Clone, Copy, Debug)]
pub enum PatchLayout {
    Ryujinx,
    Mesa,
}

/// Shared by the linked TCS and TES. Ryujinx Translation/AttributeUsage.cs:65-87
/// assigns patch slots unused regular locations and shares the map across stages.
#[derive(Clone, Copy, Debug)]
pub struct PatchIo {
    pub layout: PatchLayout,
    /// One Vulkan location per vec4 slot relative to the layout's base.
    /// None rejects that slot. Locations must not overlap per-vertex I/O.
    /// Ryujinx AttributeMap.cs:92,109-121 bounds patch bytes below 0x200.
    pub locations: [Option<u32>; 31],
}

impl PatchLayout {
    /// Generic patch allocation base of each layout, in declaration order.
    const BASE: [u16; 2] = [0x18, 0x20];

    fn base(self) -> u16 {
        Self::BASE[self as usize]
    }
}

impl PatchIo {
    pub(crate) fn base(self) -> u16 {
        self.layout.base()
    }
}

/// Driver declaration for a bound sampler. Include this state in custom cache
/// keys, as required by translate_with_options_and_graphics_state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoundTexture {
    pub binding: u16,
    pub dimension: TextureDimension,
    /// Shadow image type, following Ryujinx CodeGen/Spirv/Declarations.cs:176-183.
    pub depth_compare: bool,
    pub multisample: bool,
}

/// Driver declaration for one constant-buffer-backed texture array.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TextureArray {
    /// Constant-buffer bank holding packed texture/sampler handles.
    pub binding: u8,
    /// Packed handle offset used by the descriptor declaration, in words.
    pub handle: u16,
    /// Number of descriptor entries available from this source.
    pub length: u32,
}

/// The Maxwell texture dimensions, Mesa nak/sm50.rs:2105-2119.
/// Buffer textures require a separate image path and are not supported here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TextureDimension {
    D1 = 0,
    Array1D = 1,
    D2 = 2,
    Array2D = 3,
    D3 = 4,
    Cube = 6,
    ArrayCube = 7,
}

/// Driver-selected SSBO for a cbuf-held 64-bit guest allocation base.
/// Ryujinx Translation/ResourceManager.cs:162-185 maps cbuf slot/offset
/// to host storage bindings. Offsets here are bytes, as in Source::ConstantBuffer
/// (Mesa nak/sm50.rs:361-368); the high word is at offset + 4.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StorageBufferBinding {
    pub bank: u8,
    pub offset: u16,
    /// Set 1 is reserved here for SSBOs; scalar textures in this set cannot
    /// share a module with descriptor global memory.
    pub binding: u32,
}
