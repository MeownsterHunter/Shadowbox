//! Geometry and tessellation interfaces. Encoding sources are Mesa; mappings
//! follow Ryujinx Instructions/AttributeMap.cs:63-92,149-182,210-252 and
//! CodeGen/Spirv/Declarations.cs:393-432,540-551. Independently implemented.
use super::{alu, fresh, inst};
use crate::headers::{Stage, VertexMaps};
use crate::interface::{GeometryStreams, GraphicsState, NoLayout, PatchIo, Stopped};
use crate::ir::{
    AttributeBank, AttributeIo, AttributeLanes, Instruction, InstructionKind, ProvenAddresses,
    Register, StreamMarks,
};
use crate::spirv::Op;
use std::collections::BTreeMap;

/// Resolve only physical addresses whose AL2P definition is unconditional and
/// in the same basic block. Leave the AL2P itself and every GPR result intact.
pub(super) fn resolve_physical_addresses(
    xs: &[Instruction],
    cfg: &crate::structure::ControlFlowGraph,
) -> Vec<Instruction> {
    // Ryujinx InstEmitAttribute.cs:12-16,43-51,118-127 gives AL2P's byte
    // sum and physical input/output addressing. Decoder.cs:501-517 divides
    // control flow at branch destinations. A block-local proof cannot cross
    // a branch entry or keep a value after any intervening register write.
    // This avoids declaring all 32 generic locations when only one is used,
    // which can exceed a device's maxGeometryInputComponents.
    // By whether the instruction is guarded: a guarded AL2P proves nothing.
    const PROVE: [fn(InstructionKind, &mut ProvenAddresses); 2] =
        [|kind, proven| kind.prove_address(proven), |_, _| {}];
    let mut resolved = xs.to_vec();
    for block in &cfg.blocks {
        let mut proven = ProvenAddresses::EMPTY;
        for index in block.instructions.clone() {
            let x = xs[index];
            resolved[index].kind = x.kind.at_proven_address(&proven);
            proven.forget(x.kind);
            PROVE[x.predicate.iter().count()](x.kind, &mut proven);
        }
    }
    resolved
}

fn load(w: &mut Vec<u32>, id: &mut u32, ty: u32, pointer: u32) -> u32 {
    let result = fresh(id);
    inst(w, Op::Load, &[ty, result, pointer]);
    result
}

fn invalid(io: NoLayout) -> Stopped {
    Stopped::NoLayout(io)
}

struct Binding {
    variable: u32,
    // All stream outputs share one persistent Maxwell bank. AST/ALD.O use
    // shadow; OUT publishes only to the selected stream's variable.
    // Ryujinx InstEmitAttribute.cs:268-282 retains outputs after emission;
    // Mesa nak/from_nir.rs:3940-3959 preserves each OUT's stream identity.
    stream_variables: [Option<u32>; 4],
    stream_locations: [Option<u32>; 4],
    shadow: Option<u32>,
    base: u16,
    width: u16,
    patch: bool,
    output: bool,
    builtin: Option<u32>,
    location: Option<u32>,
    vertices: u32,
    array: bool,
    generic_array: bool,
    ty: u32,
}

/// The header of a stage with a vertex-pipeline interface: geometry and
/// tessellation. A field the stage's header does not have reads as zero, as
/// the header parser masked it before.
pub(super) trait VtgHeader {
    const STAGE: Stage;
    fn maps(&self) -> VertexMaps;
    fn threads_per_input_primitive(&self) -> u8 {
        0
    }
    fn output_topology(&self) -> u8 {
        0
    }
    fn max_output_vertices(&self) -> u16 {
        0
    }
    fn passthrough(&self) -> bool {
        false
    }
}

impl VtgHeader for crate::headers::GeometryHeader {
    const STAGE: Stage = Stage::Geometry;
    fn maps(&self) -> VertexMaps {
        self.maps()
    }
    fn threads_per_input_primitive(&self) -> u8 {
        self.threads_per_input_primitive()
    }
    fn output_topology(&self) -> u8 {
        self.output_topology()
    }
    fn max_output_vertices(&self) -> u16 {
        self.max_output_vertices()
    }
    fn passthrough(&self) -> bool {
        self.passthrough()
    }
}

impl VtgHeader for crate::headers::TessellationControlHeader {
    const STAGE: Stage = Stage::TessellationControl;
    fn maps(&self) -> VertexMaps {
        self.maps()
    }
    fn threads_per_input_primitive(&self) -> u8 {
        self.threads_per_input_primitive()
    }
}

impl VtgHeader for crate::headers::TessellationEvaluationHeader {
    const STAGE: Stage = Stage::TessellationEvaluation;
    fn maps(&self) -> VertexMaps {
        self.maps()
    }
}

pub(super) struct StageIo {
    stage: Stage,
    /// The output patch's control points, in tessellation control.
    control_points: u32,
    patch_io: Option<PatchIo>,
    modes: Vec<Vec<u32>>,
    bindings: Vec<Binding>,
    invocation: Option<u32>,
    float_ptrs: [u32; 3], // Input, Output, Private
    uint: u32,
    float: u32,
    dynamic: [bool; 2],
    /// The patch banks an indexed patch load reads in this stage, each
    /// declared whole; a bank absent here reads its static lanes.
    indexed_patch: Vec<AttributeBank>,
    output_stream: u8,
    streams: u8,
    emitted_streams: u8,
    stream_layout: Option<GeometryStreams>,
    passthrough_vertices: Vec<u32>,
    passthrough_attributes: u32,
    input_vec_ptr: u32,
}

/// One instruction's attribute declarations into a stage interface.
struct Declare<'a, N> {
    io: &'a mut StageIo,
    x: &'a Instruction,
    input_vertices: u32,
    n: &'a mut N,
}

impl<N: FnMut() -> u32> AttributeIo for Declare<'_, N> {
    type Error = Stopped;

    fn reject(&mut self) -> Result<(), Stopped> {
        Err(super::bad(self.x))
    }

    fn lanes(
        &mut self,
        dst: Register,
        address: u16,
        lanes: AttributeLanes,
        patch: bool,
    ) -> Result<(), Stopped> {
        self.io
            .add_lanes(dst, address, lanes, patch, self.input_vertices, self.n)
    }

    fn physical(&mut self, output: bool) -> Result<(), Stopped> {
        self.io
            .add(0x80, false, output, self.input_vertices, self.n)
    }

    fn patch_bank(&mut self, bank: AttributeBank) -> Result<(), Stopped> {
        self.io.add_patch_bank(bank, self.input_vertices, self.n)
    }

    fn store(&mut self, address: u16, patch: bool) -> Result<(), Stopped> {
        self.io
            .add(address, patch, true, self.input_vertices, self.n)
    }
}

impl StageIo {
    pub fn plan<H: VtgHeader>(
        h: &H,
        state: &GraphicsState,
        xs: &[Instruction],
        n: &mut impl FnMut() -> u32,
    ) -> Result<Self, Stopped> {
        let stage = H::STAGE;
        // SPIR-V mode values: Mesa src/compiler/spirv/spirv.core.grammar.json:
        // 12872-12908,12992-13058; stage choices follow Ryujinx
        // CodeGen/Spirv/SpirvGenerator.cs:190-261.
        let mut modes = Vec::new();
        // By whether the OUT has a stream encoding in a geometry stage.
        type Admit = fn(&Instruction) -> Result<(), Stopped>;
        const OUT: [Admit; 2] = [|_| Ok(()), |x| Err(super::bad(x))];
        let geometry = stage == Stage::Geometry;
        let mut marks = StreamMarks::NONE;
        xs.iter().try_for_each(|x| {
            let mark = x.kind.stream_marks();
            marks = marks.with(mark);
            OUT[usize::from(mark.outs & (mark.unencodable | !geometry))](x)
        })?;
        let StreamMarks {
            streams,
            emitted: emitted_streams,
            ..
        } = marks;
        let stream_layout = (stage == Stage::Geometry)
            .then_some(state.geometry_streams)
            .flatten();
        let output_stream = if let Some(layout) = stream_layout {
            if layout.builtin_stream > 3 || h.passthrough() {
                return Err(invalid(NoLayout::StreamRouting));
            }
            layout.builtin_stream
        } else if emitted_streams.count_ones() > 1 {
            return Err(invalid(NoLayout::UnlinkedStreams));
        } else {
            // CUT has no output publication. It can select a different stream
            // without requiring another builtin interface. Mesa ir.rs:7967-7980;
            // Ryujinx InstEmitAttribute.cs:260-293 separates emission from cut.
            if emitted_streams == 0 {
                0
            } else {
                emitted_streams.trailing_zeros() as u8
            }
        };
        // Ryujinx InstEmitAttribute.cs:232-250 does not preserve the stream
        // operand. Preserve it through native SPIR-V: Mesa grammar
        // spirv.core.grammar.json:2522-2538,14915-14921,17226-17229.
        // Restrict nonzero streams to points, the portable Vulkan default
        // (Vulkan-Docs appendices/spirvenv.adoc:1551-1554 at the commit above).
        if (streams & !1 != 0 || output_stream != 0) && h.output_topology() != 1 {
            return Err(invalid(NoLayout::NonPointStreams));
        }
        let mut passthrough_vertices = Vec::new();
        let mut passthrough_attributes = 0;
        let input_vertices = if stage == Stage::Geometry {
            // Ryujinx InputTopology.cs:26-34 gives the vertex counts.
            let input = state
                .geometry_input
                .ok_or_else(|| invalid(NoLayout::GeometryTopology))?;
            let (mode, count) = (input.execution_mode(), input.vertices());
            if h.passthrough() {
                // Header bit24: Mesa nak/sph.rs:240-243; Ryujinx ShaderHeader.cs:125.
                // Portable fallback follows Ryujinx EmitterContext.cs:386-426.
                // Explicit OUT combined with automatic return emission has no
                // established Maxwell result, so keep that combination unsupported.
                const EXPLICIT_OUT: [fn() -> Result<(), Stopped>; 2] =
                    [|| Ok(()), || Err(invalid(NoLayout::PassthroughOut))];
                EXPLICIT_OUT[usize::from(marks.outs)]()?;
                passthrough_vertices = input.passthrough_vertices().to_vec();
                // Ryujinx AttributeUsage.cs:91-97 excludes output-used locations.
                // Decoder.cs:349-365 includes AST and ALD.O; :374-385 marks
                // indexed outputs as all used; :410-417 records static spans.
                let used_outputs = xs.iter().fold(0, |used, x| used | x.kind.used_outputs());
                passthrough_attributes = state
                    .geometry_next_inputs
                    .ok_or_else(|| invalid(NoLayout::PassthroughUnlinked))?
                    & !used_outputs;
            }
            let topology = match h.output_topology() {
                1 => 27,
                6 => 28,
                7 => 29,
                _ => return Err(invalid(NoLayout::OutputTopology)),
            };
            // Ryujinx ShaderDefinitions.cs:160 replaces the SPH maximum with
            // the non-adjacency vertex count for passthrough shaders.
            let max_vertices = if h.passthrough() {
                passthrough_vertices.len() as u32
            } else {
                u32::from(h.max_output_vertices())
            };
            if max_vertices == 0 || h.threads_per_input_primitive() == 0 {
                return Err(invalid(NoLayout::ZeroCounts));
            }
            modes.extend([
                vec![mode],
                vec![topology],
                vec![26, max_vertices],
                vec![0, u32::from(h.threads_per_input_primitive())],
            ]);
            count
        } else {
            if stage == Stage::TessellationControl {
                if !(1..=32).contains(&h.threads_per_input_primitive()) {
                    return Err(invalid(NoLayout::ControlPoints));
                }
                modes.push(vec![26, u32::from(h.threads_per_input_primitive())]);
            } else {
                let t = state
                    .tessellation
                    .ok_or_else(|| invalid(NoLayout::TessellationState))?;
                modes.extend(t.execution_modes().map(|mode| vec![mode]));
            }
            // Ryujinx Declarations.cs:393,540: the input patch array has the
            // maximum 32 entries; the pipeline supplies the actual patch size.
            32
        };
        let mut io = Self {
            stage,
            control_points: u32::from(h.threads_per_input_primitive()),
            patch_io: state.patch_io,
            modes,
            bindings: Vec::new(),
            // envytools gm107.c:135 and Ryujinx InstEmitMove.cs:82-84:
            // SR_INVOCATION_ID is scalar BuiltIn InvocationId.
            invocation: (stage == Stage::TessellationControl
                || (stage == Stage::Geometry
                    && xs
                        .iter()
                        .fold(false, |reads, x| reads | x.kind.reads_invocation_id())))
            .then(&mut *n),
            float_ptrs: [0; 3],
            uint: 0,
            float: 0,
            output_stream,
            streams,
            emitted_streams,
            stream_layout,
            passthrough_vertices,
            passthrough_attributes,
            input_vec_ptr: 0,
            // Physical addressing: Ryujinx InstDecoders.cs:887,902 and
            // InstEmitAttribute.cs:43-51,118-127. Each bank has all generic
            // locations (Mesa sph.rs:130-135,166-170, 32 generic vec4 slots).
            // Ryujinx ShaderDefinitions.cs:299-303 and Spirv/IoMap.cs:11,62
            // uses the same array shape but exposes only 16 locations. Keep
            // all 32 Maxwell slots; the driver must check interface limits.
            indexed_patch: Vec::new(),
            dynamic: xs.iter().fold([false; 2], |[input, output], x| {
                let [i, o] = x.kind.dynamic_banks();
                [input | i, output | o]
            }),
        };
        // Declare output Position and the SPH generic maps for every VTG stage
        // (Mesa sph.rs:130-135,166-170). TCS output is per invocation.
        io.add(0x70, false, true, input_vertices, n)?;
        for location in 0..32 {
            if h.maps().inputs() >> (location * 4) & 15 != 0 {
                io.add(0x80 + location * 16, false, false, input_vertices, n)?;
            }
            if h.maps().outputs() >> (location * 4) & 15 != 0 {
                io.add(0x80 + location * 16, false, true, input_vertices, n)?;
            }
        }
        for x in xs {
            x.kind.declare_attributes(&mut Declare {
                io: &mut io,
                x,
                input_vertices,
                n: &mut *n,
            })?;
        }
        if h.passthrough() {
            io.add(0x70, false, false, input_vertices, n)?;
            for location in 0..32 {
                if passthrough_attributes >> location & 1 != 0 {
                    let address = 0x80 + location * 16;
                    io.add(address, false, false, input_vertices, n)?;
                    io.add(address, false, true, input_vertices, n)?;
                }
            }
        }
        io.route_streams(n)?;
        Ok(io)
    }

    #[allow(clippy::too_many_arguments)]
    fn add(
        &mut self,
        address: u16,
        patch: bool,
        output: bool,
        input_vertices: u32,
        n: &mut impl FnMut() -> u32,
    ) -> Result<(), Stopped> {
        // BuiltIn/Decoration values: Mesa spirv.core.grammar.json:
        // 14835-14838,14925-14929,15951-16025; addresses and shapes follow
        // Mesa nak_private.h:29-56,73-75 and Ryujinx AttributeMap.cs:63-91.
        let (base, width, builtin, location, array) = if patch {
            if (output && self.stage != Stage::TessellationControl)
                || (!output && self.stage != Stage::TessellationEvaluation)
            {
                return Err(invalid(NoLayout::PatchOutsideTessellation));
            }
            // Mesa nak_private.h:29-36 and Ryujinx AttributeMap.cs:90-92.
            // The explicit producer layout resolves the different generic
            // origins without shifting either producer's components.
            match address {
                0..=0xc => (0, 4, Some(11), None, true),
                0x10..=0x14 => (0x10, 2, Some(12), None, true),
                0x18..=0x1fc => {
                    let layout = self
                        .patch_io
                        .ok_or_else(|| invalid(NoLayout::UnlinkedPatch))?;
                    let relative = address
                        .checked_sub(layout.base())
                        .ok_or_else(|| invalid(NoLayout::PatchBelowBase))?;
                    let slot = usize::from(relative / 16);
                    let location = layout.locations[slot]
                        .ok_or_else(|| invalid(NoLayout::UnmappedPatchSlot))?;
                    // Ryujinx AttributeUsage.cs:65-87 shares this map in TCS
                    // and TES. The driver links once, before either translation.
                    (
                        layout.base() + (slot as u16) * 16,
                        4,
                        None,
                        Some(location),
                        false,
                    )
                }
                _ => return Err(invalid(NoLayout::PatchAddress)),
            }
        } else {
            match address {
                // Ryujinx AttributeMap.cs:63: PrimitiveId is an input in
                // tessellation/geometry and an output in geometry only.
                0x60 if !output || self.stage == Stage::Geometry => (0x60, 1, Some(7), None, false),
                0x70..=0x7c => (0x70, 4, Some(0), None, false),
                0x80..=0x27c => {
                    let l = (address - 0x80) / 16;
                    if self.dynamic[usize::from(output)] {
                        (0x80, 128, None, Some(0), false)
                    } else {
                        (0x80 + l * 16, 4, None, Some(u32::from(l)), false)
                    }
                }
                0x2f0..=0x2f4 if !output && self.stage == Stage::TessellationEvaluation => {
                    (0x2f0, 3, Some(13), None, false)
                }
                _ => return Err(invalid(NoLayout::Attribute)),
            }
        };
        if self
            .bindings
            .iter()
            .any(|b| b.base == base && b.patch == patch && b.output == output)
        {
            return Ok(());
        }
        // Ryujinx AttributeUsage.cs:65-87 requires disjoint patch and regular
        // locations. Check actual declarations too, including ALD/AST slots
        // absent from the SPH maps and duplicate driver patch locations.
        for location in location.into_iter() {
            // An indexed generic bank occupies 32 consecutive locations, not
            // just its base. Widen driver locations before adding the extent.
            let start = u64::from(location);
            let end = start + if !patch && width == 128 { 32 } else { 1 };
            if self.bindings.iter().any(|b| {
                b.output == output
                    && b.location.is_some_and(|existing| {
                        let existing = u64::from(existing);
                        let existing_end = existing + if b.generic_array { 32 } else { 1 };
                        start < existing_end && existing < end
                    })
            }) {
                return Err(invalid(NoLayout::OverlappingLocations));
            }
        }
        let per_vertex = !patch && matches!(base, 0x70..=0x27c);
        let vertices = if per_vertex {
            if output {
                if self.stage == Stage::TessellationControl {
                    self.control_points
                } else {
                    0
                }
            } else {
                input_vertices
            }
        } else {
            0
        };
        self.bindings.push(Binding {
            variable: n(),
            stream_variables: [None; 4],
            stream_locations: [None; 4],
            shadow: (output && self.stage == Stage::Geometry).then(&mut *n),
            base,
            width,
            patch,
            output,
            builtin,
            location,
            vertices,
            array,
            generic_array: !patch && width == 128,
            ty: 0,
        });
        Ok(())
    }

    /// The InvocationId input, declared for tessellation control (and
    /// geometry when S2R reads it).
    pub fn invocation_id(&self) -> Option<u32> {
        self.invocation
    }

    /// Decide once whether this stage has the patch `bank` an indexed
    /// patch load reads (Ryujinx AttributeMap.cs:90-92). If it does, the
    /// run-time address can name any word of the bank, so declare all of
    /// it: the outer and inner tessellation levels and each linked generic
    /// patch slot. If not, the load reads its static lanes and nothing is
    /// declared.
    fn add_patch_bank(
        &mut self,
        bank: AttributeBank,
        input_vertices: u32,
        n: &mut impl FnMut() -> u32,
    ) -> Result<(), Stopped> {
        if !crate::ir::patch_bank_in(self.stage, bank) || self.indexed_patch.contains(&bank) {
            return Ok(());
        }
        self.indexed_patch.push(bank);
        let output = [false, true][bank as usize];
        self.add(0, true, output, input_vertices, n)?;
        self.add(0x10, true, output, input_vertices, n)?;
        let layout = self
            .patch_io
            .ok_or_else(|| invalid(NoLayout::UnlinkedPatch))?;
        for (slot, location) in layout.locations.iter().enumerate() {
            if location.is_some() {
                self.add(
                    layout.base() + slot as u16 * 16,
                    true,
                    output,
                    input_vertices,
                    n,
                )?;
            }
        }
        Ok(())
    }

    /// The byte address of every declared word of the patch `bank`, in
    /// declaration order, when an indexed patch load reads that bank in
    /// this stage (decided by `add_patch_bank`).
    pub fn patch_words(&self, bank: AttributeBank) -> Option<Vec<u16>> {
        let output = [false, true][bank as usize];
        self.indexed_patch.contains(&bank).then(|| {
            self.bindings
                .iter()
                .filter(|b| b.patch && b.output == output)
                .flat_map(|b| (0..b.width).map(move |word| b.base + word * 4))
                .collect()
        })
    }

    fn route_streams(&mut self, n: &mut impl FnMut() -> u32) -> Result<(), Stopped> {
        if self.stage != Stage::Geometry {
            return Ok(());
        }
        // Vulkan-Docs appendices/spirvenv.adoc:377-381,442-447 (commit
        // e4e53e4b31e13eeaee1ad99fb940aa72b2ec1b14) forbids duplicate
        // output builtins and overlapping output locations. The explicit
        // driver map preserves every builtin value on non-native streams.
        // Mesa from_nir.rs:3940-3959 requires the stream to remain distinct;
        // Ryujinx InstEmitAttribute.cs:232-250 drops that operand, so its
        // single interface cannot implement multiple streams correctly here.
        let mut ranges = Vec::new();
        for stream in 0..4 {
            if stream != self.output_stream && self.emitted_streams >> stream & 1 == 0 {
                continue;
            }
            let layout = self
                .stream_layout
                .map(|routing| {
                    routing.locations[usize::from(stream)]
                        .ok_or_else(|| invalid(NoLayout::StreamLayout))
                })
                .transpose()?;
            if stream == self.output_stream
                && layout.is_some_and(|l| l.position.is_some() || l.primitive_id.is_some())
            {
                return Err(invalid(NoLayout::AliasedBuiltins));
            }
            for b in self.bindings.iter_mut().filter(|b| b.output) {
                let location = if let Some(l) = layout {
                    if let Some(location) = b.location {
                        Some(
                            l.generic_base
                                .checked_add(location)
                                .ok_or_else(|| invalid(NoLayout::LocationOverflow))?,
                        )
                    } else if stream == self.output_stream {
                        None
                    } else {
                        // SPIR-V Position=0 and PrimitiveId=7: Mesa
                        // spirv.core.grammar.json:15951-15953,15987-15989.
                        Some(
                            match b.builtin {
                                Some(0) => l.position,
                                Some(7) => l.primitive_id,
                                _ => return Err(invalid(NoLayout::StreamBuiltin)),
                            }
                            .ok_or_else(|| invalid(NoLayout::UnaliasedBuiltin))?,
                        )
                    }
                } else {
                    b.location
                };
                for location in location.into_iter() {
                    let start = u64::from(location);
                    let end = start + if b.generic_array { 32 } else { 1 };
                    if end > u64::from(u32::MAX) + 1
                        || ranges.iter().any(|&(a, z)| start < z && a < end)
                    {
                        return Err(invalid(NoLayout::StreamLocations));
                    }
                    ranges.push((start, end));
                }
                b.stream_locations[usize::from(stream)] = location;
                b.stream_variables[usize::from(stream)] = Some(if stream == self.output_stream {
                    b.variable
                } else {
                    n()
                });
            }
        }
        Ok(())
    }

    /// Declare each lane of a static ALD that reads an attribute here.
    /// Ryujinx InstEmitAttribute.cs:34-41 stops at RZ, and AttributeMap.cs:
    /// 129-141 loads 0 for a lane with no attribute in this stage, so
    /// neither declares anything.
    #[allow(clippy::too_many_arguments)]
    fn add_lanes(
        &mut self,
        dst: Register,
        address: u16,
        lanes: AttributeLanes,
        patch: bool,
        input_vertices: u32,
        n: &mut impl FnMut() -> u32,
    ) -> Result<(), Stopped> {
        for (c, (address, lane)) in lanes.iter(address).enumerate() {
            if usize::from(dst.0) + c >= 255 {
                break;
            }
            if lane.reads_in(self.stage) {
                self.add(
                    address,
                    patch,
                    lane.bank == AttributeBank::Output,
                    input_vertices,
                    n,
                )?;
            }
        }
        Ok(())
    }

    pub fn invocation(&self) -> Option<u32> {
        self.invocation
    }

    pub fn interfaces(&self) -> Vec<u32> {
        self.bindings
            .iter()
            .flat_map(|b| {
                std::iter::once(b.variable).chain(
                    b.stream_variables
                        .into_iter()
                        .flatten()
                        .filter(|&v| v != b.variable),
                )
            })
            .chain(self.invocation)
            .collect()
    }
    pub fn modes(&self, w: &mut Vec<u32>, fun: u32) {
        for mode in &self.modes {
            let mut operands = vec![fun];
            operands.extend(mode);
            inst(w, Op::ExecutionMode, &operands);
        }
    }

    pub fn uses_streams(&self) -> bool {
        self.streams & !1 != 0 || self.output_stream != 0
    }

    #[allow(clippy::too_many_arguments)]
    pub fn declare(
        &mut self,
        w: &mut Vec<u32>,
        annotations: &mut Vec<u32>,
        n: &mut impl FnMut() -> u32,
        uint: u32,
        float: u32,
        vec4: u32,
        fvec3: u32,
        input_vec_ptr: u32,
        output_vec_ptr: u32,
        output_float_ptr: u32,
    ) {
        self.input_vec_ptr = input_vec_ptr;
        self.uint = uint;
        self.float = float;
        self.float_ptrs = [n(), output_float_ptr, n()];
        for (storage, pointer) in [1, 3, 6].into_iter().zip(self.float_ptrs) {
            inst(w, Op::TypePointer, &[pointer, storage, float]);
        }
        let mut pointers = BTreeMap::from([
            ((1, vec4), input_vec_ptr),
            ((3, vec4), output_vec_ptr),
            ((1, float), self.float_ptrs[0]),
            ((3, float), self.float_ptrs[1]),
            ((6, float), self.float_ptrs[2]),
        ]);
        let mut arrays = BTreeMap::new();
        for b in &mut self.bindings {
            let mut ty = match b.width {
                1 => uint,
                3 => fvec3,
                _ => vec4,
            };
            for length in [
                b.array.then_some(u32::from(b.width)),
                b.generic_array.then_some(32),
                (b.vertices != 0).then_some(b.vertices),
            ]
            .into_iter()
            .flatten()
            {
                if b.array {
                    ty = float;
                }
                ty = *arrays.entry((ty, length)).or_insert_with(|| {
                    let count = n();
                    let array = n();
                    inst(w, Op::Constant, &[uint, count, length]);
                    inst(w, Op::TypeArray, &[array, ty, count]);
                    array
                });
            }
            b.ty = ty;
            let storage = if b.output { 3 } else { 1 };
            let ptr = *pointers.entry((storage, ty)).or_insert_with(|| {
                let p = n();
                inst(w, Op::TypePointer, &[p, storage, ty]);
                p
            });
            inst(w, Op::Variable, &[ptr, b.variable, storage]);
            for builtin in b.builtin.into_iter() {
                inst(annotations, Op::Decorate, &[b.variable, 11, builtin]);
            }
            let location = if b.output && self.stage == Stage::Geometry {
                b.stream_locations[usize::from(self.output_stream)]
            } else {
                b.location
            };
            for location in location.into_iter() {
                inst(annotations, Op::Decorate, &[b.variable, 30, location]);
            }
            if b.output && self.stage == Stage::Geometry {
                if self.streams & !1 != 0 || self.output_stream != 0 {
                    inst(
                        annotations,
                        Op::Decorate,
                        &[b.variable, 29, u32::from(self.output_stream)],
                    );
                }
                for stream in 0..4 {
                    if stream == usize::from(self.output_stream) {
                        continue;
                    }
                    if let Some(variable) = b.stream_variables[stream] {
                        inst(w, Op::Variable, &[ptr, variable, 3]);
                        inst(annotations, Op::Decorate, &[variable, 29, stream as u32]);
                        inst(
                            annotations,
                            Op::Decorate,
                            &[
                                variable,
                                30,
                                b.stream_locations[stream].expect("routed stream output"),
                            ],
                        );
                    }
                }
            }
            if b.patch {
                inst(annotations, Op::Decorate, &[b.variable, 15]);
            }
            for shadow in b.shadow.into_iter() {
                let ptr = *pointers.entry((6, ty)).or_insert_with(|| {
                    let p = n();
                    inst(w, Op::TypePointer, &[p, 6, ty]);
                    p
                });
                inst(w, Op::Variable, &[ptr, shadow, 6]);
            }
        }
        for invocation in self.invocation.into_iter() {
            let ptr = *pointers.entry((1, uint)).or_insert_with(|| {
                let p = n();
                inst(w, Op::TypePointer, &[p, 1, uint]);
                p
            });
            inst(w, Op::Variable, &[ptr, invocation, 1]);
            inst(annotations, Op::Decorate, &[invocation, 11, 8]);
        }
    }

    /// Scalar access avoids loading and extracting a whole vec4 for each
    /// component as in the reference. The selected bits are identical.
    #[allow(clippy::too_many_arguments)]
    pub fn pointer(
        &self,
        w: &mut Vec<u32>,
        id: &mut u32,
        address: u16,
        patch: bool,
        bank: AttributeBank,
        vertex: Option<u32>,
        offset: Option<Register>,
        regs: &[u32],
        globals: &mut Vec<u32>,
    ) -> Result<(u32, u32), Stopped> {
        let output = [false, true][bank as usize];
        let b = self
            .bindings
            .iter()
            .find(|b| {
                b.patch == patch
                    && b.output == output
                    && (if offset.is_some() {
                        b.generic_array
                    } else {
                        address >= b.base && address < b.base + b.width * 4
                    })
            })
            .ok_or_else(|| invalid(NoLayout::Undeclared))?;
        if b.width == 1 {
            return Ok((b.shadow.unwrap_or(b.variable), self.uint));
        }
        let constant = |globals: &mut Vec<u32>, id: &mut u32, value| {
            let c = fresh(id);
            inst(globals, Op::Constant, &[self.uint, c, value]);
            c
        };
        let mut indices = Vec::new();
        if b.vertices != 0 {
            let index = if let Some(index) = vertex {
                index
            } else if output {
                load(w, id, self.uint, self.invocation.expect("TCS invocation"))
            } else {
                constant(globals, id, 0)
            };
            indices.push(index);
        }
        if let Some(offset) = offset {
            // Ryujinx InstEmitAttribute.cs:45-47,120-122: byte address minus
            // 0x80, location = offset >> 4, component = (offset >> 2) & 3.
            // A scalar access chain avoids loading/extracting a full vec4.
            let address = load(w, id, self.uint, regs[offset.0 as usize]);
            let base = constant(globals, id, 0x80);
            let address = alu(w, id, Op::ISub, self.uint, &[address, base]);
            let four = constant(globals, id, 4);
            indices.push(alu(
                w,
                id,
                Op::ShiftRightLogical,
                self.uint,
                &[address, four],
            ));
            let two = constant(globals, id, 2);
            let component = alu(w, id, Op::ShiftRightLogical, self.uint, &[address, two]);
            let three = constant(globals, id, 3);
            indices.push(alu(w, id, Op::BitwiseAnd, self.uint, &[component, three]));
        } else {
            if b.generic_array {
                indices.push(constant(globals, id, u32::from((address - b.base) / 16)));
            }
            let component = (address - b.base) / 4;
            indices.push(constant(
                globals,
                id,
                u32::from(if b.generic_array {
                    component % 4
                } else {
                    component
                }),
            ));
        }
        let pointer = fresh(id);
        let ty = self.float_ptrs[if b.shadow.is_some() {
            2
        } else {
            usize::from(output)
        }];
        let mut args = vec![ty, pointer, b.shadow.unwrap_or(b.variable)];
        args.extend(indices);
        inst(w, Op::AccessChain, &args);
        Ok((pointer, self.float))
    }

    pub fn emit_passthrough(&self, w: &mut Vec<u32>, id: &mut u32, globals: &mut Vec<u32>) {
        // Ryujinx EmitterContext.cs:360-426 copies Position and forwarded
        // generic vec4s, emits the non-adjacency vertices, then ends the primitive.
        // Unroll the known 1..3 indices and copy whole vectors directly to Output.
        // This avoids runtime index arithmetic and four scalar loads/stores per
        // vector, plus the redundant Private round trip used by ordinary OUT.
        for &vertex in &self.passthrough_vertices {
            let index = fresh(id);
            inst(globals, Op::Constant, &[self.uint, index, vertex]);
            for b in self.bindings.iter().filter(|b| b.output) {
                let forwarded = b.base == 0x70
                    || (b
                        .location
                        .is_some_and(|l| l < 32 && self.passthrough_attributes >> l & 1 != 0)
                        && !b.generic_array);
                let value = if forwarded {
                    let input = self
                        .bindings
                        .iter()
                        .find(|a| !a.output && a.base <= b.base && b.base < a.base + a.width * 4)
                        .expect("planned passthrough input");
                    let pointer = fresh(id);
                    let mut args = vec![self.input_vec_ptr, pointer, input.variable, index];
                    if input.generic_array {
                        let location = fresh(id);
                        inst(
                            globals,
                            Op::Constant,
                            &[self.uint, location, u32::from((b.base - input.base) / 16)],
                        );
                        args.push(location);
                    }
                    inst(w, Op::AccessChain, &args);
                    load(w, id, b.ty, pointer)
                } else {
                    load(w, id, b.ty, b.shadow.expect("geometry output bank"))
                };
                inst(w, Op::Store, &[b.variable, value]);
            }
            // Ryujinx CodeGen/Spirv/Instructions.cs:517-528, Mesa grammar
            // spirv.core.grammar.json:2508-2519: EmitVertex and EndPrimitive.
            inst(w, Op::EmitVertex, &[]);
        }
        if !self.passthrough_vertices.is_empty() {
            inst(w, Op::EndPrimitive, &[]);
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn emit_vertex(
        &self,
        w: &mut Vec<u32>,
        id: &mut u32,
        emit: bool,
        cut: bool,
        stream: u8,
        predicate: Option<(u32, bool)>,
        boolt: u32,
        globals: &mut Vec<u32>,
    ) -> Result<(), Stopped> {
        if self.stage != Stage::Geometry {
            return Err(invalid(NoLayout::OutOutsideGeometry));
        }
        let merge = predicate.map(|(p, inverted)| {
            let mut cond = load(w, id, boolt, p);
            if inverted {
                cond = alu(w, id, Op::LogicalNot, boolt, &[cond]);
            }
            let body = fresh(id);
            let merge = fresh(id);
            inst(w, Op::SelectionMerge, &[merge, 0]);
            inst(w, Op::BranchConditional, &[cond, body, merge]);
            inst(w, Op::Label, &[body]);
            merge
        });
        // Mesa spirv.core.grammar.json:2508-2519 defines opcodes 218/219.
        if emit {
            // SPIR-V OpEmitVertex makes Output undefined. Maxwell keeps its
            // output register bank (Ryujinx InstEmitAttribute.cs:268-282 also
            // restores transformed position). Keep a Private bank and publish
            // it before every emission, including repeated EMIT without AST.
            for b in &self.bindings {
                for shadow in b.shadow.into_iter() {
                    let value = load(w, id, b.ty, shadow);
                    let variable =
                        b.stream_variables[usize::from(stream)].expect("planned emitted stream");
                    inst(w, Op::Store, &[variable, value]);
                }
            }
            if self.uses_streams() {
                let s = fresh(id);
                inst(globals, Op::Constant, &[self.uint, s, u32::from(stream)]);
                inst(w, Op::EmitStreamVertex, &[s]);
            } else {
                inst(w, Op::EmitVertex, &[]);
            }
        }
        if cut {
            if self.uses_streams() {
                let s = fresh(id);
                inst(globals, Op::Constant, &[self.uint, s, u32::from(stream)]);
                inst(w, Op::EndStreamPrimitive, &[s]);
            } else {
                inst(w, Op::EndPrimitive, &[]);
            }
        }
        for merge in merge.into_iter() {
            inst(w, Op::Branch, &[merge]);
            inst(w, Op::Label, &[merge]);
        }
        Ok(())
    }
}

#[cfg(test)]
mod address_tests {
    use super::*;
    use crate::ir::{ops, PrimitiveVertex};
    use crate::ir::{Predicate, Source};
    use crate::structure::ControlFlowGraph;

    fn resolve(kinds: &[InstructionKind], predicated: Option<usize>) -> Vec<Instruction> {
        let xs = kinds
            .iter()
            .enumerate()
            .map(|(i, &kind)| Instruction {
                address: (i * 8) as u32,
                kind,
                predicate: (predicated == Some(i)).then_some(Predicate {
                    register: 0,
                    inverted: false,
                }),
            })
            .collect::<Vec<_>>();
        let cfg = ControlFlowGraph::build(&xs).unwrap();
        resolve_physical_addresses(&xs, &cfg)
    }
    fn address(slot: u16) -> InstructionKind {
        InstructionKind::AttributeAddress(ops::AttributeAddress {
            dst: Register(4),
            offset: Register(255),
            address: slot,
        })
    }
    fn read(dst: u8) -> InstructionKind {
        InstructionKind::PhysicalAttributeLoad(ops::PhysicalAttributeLoad {
            dst: Register(dst),
            vertex: PrimitiveVertex::Primitive(Register(2)),
            offset: Register(4),
        })
    }
    #[test]
    fn physical_address_proof_stops_at_branch_entries_and_clobbers() {
        let jump = InstructionKind::Branch(ops::Branch {
            target: crate::ir::BranchTarget::reaching(8, 16),
            cc_test: 15,
        });
        assert_eq!(
            resolve(&[address(0x80), jump, read(0)], None)[2].kind,
            read(0)
        );
        assert_eq!(resolve(&[address(0x80), read(0)], Some(0))[1].kind, read(0));
        // A vector write overlapping R4 must invalidate an earlier AL2P.
        let overwrite = InstructionKind::AttributeLoad(ops::AttributeLoad {
            dst: Register(3),
            vertex: PrimitiveVertex::Current,
            address: 0x80,
            components: 2,
            lanes: AttributeLanes::vertex(0x80, 2, AttributeBank::Input),
        });
        assert_eq!(
            resolve(&[address(0x80), overwrite, read(0)], None)[2].kind,
            read(0)
        );
        let overwrite = InstructionKind::Mov(ops::Mov {
            dst: Register(4),
            quad_lanes: 15,
            src: Source::Immediate(0x90),
        });
        assert_eq!(
            resolve(&[address(0x80), overwrite, read(0)], Some(1))[2].kind,
            read(0)
        );
    }
    #[test]
    fn physical_address_proof_keeps_latest_definition_and_read_before_write() {
        let xs = resolve(&[address(0x80), address(0x94), read(4), read(0)], None);
        assert_eq!(xs[0].kind, address(0x80));
        assert_eq!(xs[1].kind, address(0x94));
        let InstructionKind::AttributeLoad(ops::AttributeLoad {
            address,
            dst,
            vertex,
            ..
        }) = xs[2].kind
        else {
            panic!()
        };
        assert_eq!(
            (address, dst, vertex),
            (0x94, Register(4), PrimitiveVertex::Primitive(Register(2)))
        );
        assert_eq!(xs[3].kind, read(0));
        for slot in [0x70, 0x81, 0x280] {
            assert_eq!(
                resolve(&[self::address(slot), read(0)], None)[1].kind,
                read(0)
            );
        }
    }
}
