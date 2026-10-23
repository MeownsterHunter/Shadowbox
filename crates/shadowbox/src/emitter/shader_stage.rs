//! The shader stage as a type. The header is its stage's header before any
//! instruction is decoded; [`Interface::route`] is the station where that
//! header becomes a track, and emission runs from there as
//! `S: ShaderStage`. Each stage type is built only by the route, from the
//! stage's own header.
use super::stages::StageIo;
use super::{alu, bad, inst, Op, Required};
use crate::fast_hash::HashMap;
use crate::headers::{
    FragmentHeader, GeometryHeader, Interface, Interpolation, ShaderHeader, Stage,
    TessellationControlHeader, TessellationEvaluationHeader, VertexHeader,
};
use crate::interface::{GeometryInput, GraphicsState, Stopped};
use crate::ir::{ops, Instruction, InstructionKind};
use crate::structure::ControlFlowGraph;

/// What emission does once the header's stage is known.
pub(super) trait StagePath {
    type Output;
    fn follow<S: ShaderStage>(self, stage: S) -> Self::Output;
}

impl Interface<'_> {
    /// The station: the program's header sends it onto its stage's own
    /// track, once. A headerless program is a fragment program with no
    /// input map.
    pub(super) fn route<P: StagePath>(self, path: P) -> P::Output {
        match self {
            Self::Header(ShaderHeader::Vertex(h)) => path.follow(Vertex(h)),
            Self::Header(ShaderHeader::Fragment(h)) => path.follow(Fragment(h)),
            Self::Header(ShaderHeader::Compute(_)) => path.follow(Compute(())),
            Self::Header(ShaderHeader::Geometry(h)) => path.follow(Geometry(h)),
            Self::Header(ShaderHeader::TessellationControl(h)) => {
                path.follow(TessellationControl(h))
            }
            Self::Header(ShaderHeader::TessellationEvaluation(h)) => {
                path.follow(TessellationEvaluation(h))
            }
            Self::Headerless => path.follow(Fragment(Headerless(()))),
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Vertex<'h>(&'h VertexHeader);
#[derive(Clone, Copy)]
pub(super) struct Fragment<F>(F);
#[derive(Clone, Copy)]
pub(super) struct Compute(());
#[derive(Clone, Copy)]
pub(super) struct Geometry<'h>(&'h GeometryHeader);
#[derive(Clone, Copy)]
pub(super) struct TessellationControl<'h>(&'h TessellationControlHeader);
#[derive(Clone, Copy)]
pub(super) struct TessellationEvaluation<'h>(&'h TessellationEvaluationHeader);

/// A fragment program without a header: no input map, and all four
/// components of render target 0 written.
#[derive(Clone, Copy)]
pub(super) struct Headerless(());

/// What a fragment program's interface is: its input map and outputs.
pub(super) trait FragmentInterface: Copy {
    /// Generic input locations that carry an input map.
    const INPUT_MAP_LOCATIONS: usize;
    /// The first used interpolation mode of a generic input location.
    fn input_mode(self, location: usize) -> Interpolation;
    /// Written render-target components, 4 bits per target.
    fn targets(self) -> u32;
    fn writes_sample_mask(self) -> bool;
    fn writes_depth(self) -> bool;
}

impl FragmentInterface for &FragmentHeader {
    const INPUT_MAP_LOCATIONS: usize = 32;
    fn input_mode(self, location: usize) -> Interpolation {
        FragmentHeader::input_mode(self, location)
    }
    fn targets(self) -> u32 {
        FragmentHeader::targets(self)
    }
    fn writes_sample_mask(self) -> bool {
        FragmentHeader::writes_sample_mask(self)
    }
    fn writes_depth(self) -> bool {
        FragmentHeader::writes_depth(self)
    }
}

impl FragmentInterface for Headerless {
    const INPUT_MAP_LOCATIONS: usize = 0;
    fn input_mode(self, _: usize) -> Interpolation {
        Interpolation::Unused
    }
    fn targets(self) -> u32 {
        0xf
    }
    fn writes_sample_mask(self) -> bool {
        false
    }
    fn writes_depth(self) -> bool {
        false
    }
}

/// The generic and system inputs a stage declares from its header and the
/// attribute loads it executes.
pub(super) struct Inputs<'a> {
    pub xs: &'a [Instruction],
    pub has_dynamic_input: bool,
    pub decorations: &'a mut Vec<(u32, u32, Interpolation)>,
    pub by_location: &'a mut HashMap<u16, u32>,
    pub vertex_outputs: &'a mut Vec<(u32, u32)>,
    pub system: &'a mut HashMap<u16, u32>,
    pub system_decorations: &'a mut Vec<(u32, u32)>,
}

/// The ids a stage's output variables are declared with.
pub(super) struct OutputTypes<'a> {
    pub po: u32,
    pub pfloat_out: u32,
    pub float: u32,
    pub uint: u32,
    pub constants: &'a HashMap<u32, u32>,
    pub vertex_outputs: &'a [(u32, u32)],
}

/// The ids a stage's terminal output stores read.
pub(super) struct OutputStores<'a, L> {
    pub uint: u32,
    pub float: u32,
    pub vec4: u32,
    pub regs: &'a [u32; 256],
    pub load: &'a L,
}

/// A stage's own output variables: what it declares and what it stores
/// when the program returns.
pub(super) trait Outputs {
    fn entry(&self, _entry: &mut Vec<u32>) {}
    fn decorate(&self, _annotations: &mut Vec<u32>) {}
    fn declare(&self, _w: &mut Vec<u32>, _types: OutputTypes<'_>) {}
    /// The ClipDistance output an AST in the clip lanes writes.
    fn clip_distance(&self) -> Option<u32> {
        None
    }
    fn store<L: Fn(&mut Vec<u32>, &mut u32, u32, u32) -> u32>(
        &self,
        _w: &mut Vec<u32>,
        _id: &mut u32,
        _stores: OutputStores<'_, L>,
    ) {
    }
}

/// A stage whose outputs are its stage interface's, or none.
pub(super) struct InterfaceOutputs;
impl Outputs for InterfaceOutputs {}

/// Vertex output Position (BuiltIn 0), the header's generic outputs and
/// the clip distances the program writes.
pub(super) struct VertexOutputs {
    position: u32,
    clip: Option<ClipDistances>,
}

/// The ClipDistance output (BuiltIn 3): a float array over lane 0 up to
/// the highest lane an AST writes.
struct ClipDistances {
    variable: u32,
    array: u32,
    pointer: u32,
    length: u32,
}

impl Outputs for VertexOutputs {
    fn entry(&self, entry: &mut Vec<u32>) {
        entry.push(self.position);
        entry.extend(self.clip.iter().map(|clip| clip.variable));
    }
    fn decorate(&self, annotations: &mut Vec<u32>) {
        inst(annotations, Op::Decorate, &[self.position, 11, 0]);
        if let Some(clip) = &self.clip {
            inst(annotations, Op::Decorate, &[clip.variable, 11, 3]);
        }
    }
    fn declare(&self, w: &mut Vec<u32>, t: OutputTypes<'_>) {
        inst(w, Op::Variable, &[t.po, self.position, 3]);
        for (_, variable) in t.vertex_outputs {
            inst(w, Op::Variable, &[t.po, *variable, 3]);
        }
        if let Some(clip) = &self.clip {
            let length = t.constants[&clip.length];
            inst(w, Op::TypeArray, &[clip.array, t.float, length]);
            inst(w, Op::TypePointer, &[clip.pointer, 3, clip.array]);
            inst(w, Op::Variable, &[clip.pointer, clip.variable, 3]);
        }
        inst(w, Op::TypePointer, &[t.pfloat_out, 3, t.float]);
    }
    fn clip_distance(&self) -> Option<u32> {
        self.clip.as_ref().map(|clip| clip.variable)
    }
}

/// The ClipDistance lanes 0x2c0..0x2dc (Ryujinx AttributeConsts.cs:12-19)
/// a program's direct AST writes, as Ryujinx Decoders/Decoder.cs:452-462
/// records them: lane 0 up to the highest written lane. A component past
/// RZ is not stored (InstEmitAttribute.cs:109-114).
pub(super) fn clip_distances(xs: &[Instruction]) -> u32 {
    xs.iter()
        .filter_map(|x| match x.kind {
            InstructionKind::AttributeStore(ops::AttributeStore {
                data,
                address,
                components,
                patch: false,
                offset: None,
            }) => (0..u16::from(components))
                .filter(|c| usize::from(data.0) + usize::from(*c) <= 255)
                .map(|c| address + c * 4)
                .filter(|a| (0x2c0..0x2e0).contains(a))
                .map(|a| u32::from((a - 0x2c0) / 4 + 1))
                .max(),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

/// Fragment outputs (Mesa from_nir.rs:2990-3031, OpRegOut): each render
/// target with any component written in the header's target mask takes
/// four consecutive registers from R0, in target order; then one register
/// for the sample mask when the mask or depth is written, then depth.
pub(super) struct FragmentOutputs {
    targets: Vec<(u32, u32)>,
    /// The sample-mask variable, its array type and pointer type.
    sample_mask: Option<(u32, (u32, u32))>,
    depth: Option<u32>,
}

impl Outputs for FragmentOutputs {
    fn entry(&self, entry: &mut Vec<u32>) {
        entry.extend(self.targets.iter().map(|(_, v)| *v));
        entry.extend(self.sample_mask.map(|(mask, _)| mask));
        entry.extend(self.depth);
    }
    fn decorate(&self, annotations: &mut Vec<u32>) {
        for (target, variable) in &self.targets {
            inst(annotations, Op::Decorate, &[*variable, 30, *target]);
        }
        for (mask, _) in self.sample_mask.into_iter() {
            // BuiltIn SampleMask=20 (Mesa src/compiler/spirv/spirv.h:726).
            inst(annotations, Op::Decorate, &[mask, 11, 20]);
        }
        for depth in self.depth.into_iter() {
            // BuiltIn FragDepth (22).
            inst(annotations, Op::Decorate, &[depth, 11, 22]);
        }
    }
    fn declare(&self, w: &mut Vec<u32>, t: OutputTypes<'_>) {
        for (_, variable) in &self.targets {
            inst(w, Op::Variable, &[t.po, *variable, 3]);
        }
        for (mask, (array, pointer)) in self.sample_mask.into_iter() {
            // Vulkan SampleMask VUID-04359: array of 32-bit integers.
            // https://docs.vulkan.org/refpages/latest/refpages/source/SampleMask.html
            // Mesa from_nir.rs:3021-3030 exposes one 32-bit mask register.
            inst(w, Op::TypeArray, &[array, t.uint, t.constants[&1]]);
            inst(w, Op::TypePointer, &[pointer, 3, array]);
            inst(w, Op::Variable, &[pointer, mask, 3]);
        }
        for depth in self.depth.into_iter() {
            inst(w, Op::TypePointer, &[t.pfloat_out, 3, t.float]);
            inst(w, Op::Variable, &[t.pfloat_out, depth, 3]);
        }
    }
    fn store<L: Fn(&mut Vec<u32>, &mut u32, u32, u32) -> u32>(
        &self,
        w: &mut Vec<u32>,
        id: &mut u32,
        s: OutputStores<'_, L>,
    ) {
        let load = s.load;
        let mut register = 0usize;
        for (_, variable) in &self.targets {
            let mut parts = Vec::new();
            for component in 0..4 {
                let u = load(w, id, s.uint, s.regs[register + component]);
                let f = *id;
                *id += 1;
                inst(w, Op::Bitcast, &[s.float, f, u]);
                parts.push(f);
            }
            let value = *id;
            *id += 1;
            inst(
                w,
                Op::CompositeConstruct,
                &[s.vec4, value, parts[0], parts[1], parts[2], parts[3]],
            );
            inst(w, Op::Store, &[*variable, value]);
            register += 4;
        }
        for (mask, (array, _)) in self.sample_mask.into_iter() {
            // Mesa from_nir.rs:3007-3030: mask follows packed color vec4s.
            // Store the complete array once, avoiding an element access chain.
            // Demotion suppresses framebuffer writes while preserving private
            // output stores (SPV_EXT_demote_to_helper_invocation, instructions).
            let bits = load(w, id, s.uint, s.regs[register]);
            let value = alu(w, id, Op::CompositeConstruct, array, &[bits]);
            inst(w, Op::Store, &[mask, value]);
        }
        for depth in self.depth.into_iter() {
            // Mesa from_nir.rs:3021-3030; Ryujinx TranslatorContext.cs:160-164.
            // The sample-mask slot precedes depth even when only depth is written.
            register += 1;
            let u = load(w, id, s.uint, s.regs[register]);
            let f = *id;
            *id += 1;
            inst(w, Op::Bitcast, &[s.float, f, u]);
            inst(w, Op::Store, &[depth, f]);
        }
    }
}

/// What a shader stage is, for emission.
pub(super) trait ShaderStage: Copy {
    /// The stage's name.
    const STAGE: Stage;
    /// SPIR-V ExecutionModel (Ryujinx CodeGen/Spirv/EnumConversion.cs:14-16).
    const EXECUTION_MODEL: u32;
    /// SPIR-V capabilities the stage itself requires (Ryujinx
    /// SpirvGenerator.cs:82-95): Geometry is 2, Tessellation is 3.
    const CAPABILITIES: &'static [u32];
    /// A group barrier's memory scope and semantics. Only compute has
    /// workgroups; elsewhere it widens to Device and drops WorkgroupMemory.
    const GROUP_MEMORY_BARRIER: (u32, u32) = (1, 0x848);
    /// A memory barrier's semantics: AcquireRelease | UniformMemory |
    /// ImageMemory, plus WorkgroupMemory in compute.
    const MEMORY_BARRIER_SEMANTICS: u32 = 0x848;

    type Outputs: Outputs;

    /// The program with AL2P addresses resolved, where the stage has a
    /// physical attribute space.
    fn resolve_addresses<'a>(
        xs: &'a [Instruction],
        _cfg: &ControlFlowGraph,
        _resolved: &'a mut Vec<Instruction>,
    ) -> &'a [Instruction] {
        xs
    }

    /// Whether an ALD.O reads another lane's control point (Mesa
    /// nak_nir_lower_vtg_io.c:10-49).
    fn reads_output_lanes(_required: &Required) -> bool {
        false
    }

    /// The HelperInvocation input, in a stage with helpers.
    fn helper_invocation(_required: &Required, _n: &mut impl FnMut() -> u32) -> Option<u32> {
        None
    }

    /// The undefined value a skipped helper texture result takes.
    fn helper_undefined(_required: &Required, _n: &mut impl FnMut() -> u32) -> Option<u32> {
        None
    }

    /// The FragCoord input, in a stage with fragments.
    fn frag_coord(_reads: bool, _n: &mut impl FnMut() -> u32) -> Option<u32> {
        None
    }

    /// The FrontFacing input, in a stage with fragments that reads a[0x3fc].
    fn front_facing(_xs: &[Instruction], _n: &mut impl FnMut() -> u32) -> Option<u32> {
        None
    }

    /// Declare the stage's generic and system inputs.
    fn declare_inputs(self, _inputs: Inputs<'_>, _n: &mut impl FnMut() -> u32) {}

    /// The first used interpolation mode of a generic input location, in
    /// a stage with an input map.
    fn input_mode(self, _location: usize) -> Interpolation {
        Interpolation::Unused
    }

    /// The geometry or tessellation interface, in those stages.
    fn stage_io(
        self,
        _state: &GraphicsState,
        _xs: &[Instruction],
        _n: &mut impl FnMut() -> u32,
    ) -> Result<Option<StageIo>, Stopped> {
        Ok(None)
    }

    /// Ryujinx CodeGen/Spirv/IoMap.cs:20,44: thread and CTA builtins are
    /// compute inputs. Other stages reject them.
    fn compute_registers(local: bool, workgroup: bool) -> Result<(), Stopped> {
        if local || workgroup {
            return Err(Stopped::ComputeRegisters);
        }
        Ok(())
    }

    /// Ryujinx InstEmitMove.cs:94-114: the dynamic patch vertex count, in
    /// tessellation.
    fn patch_vertices(_info: bool, _n: &mut impl FnMut() -> u32) -> Option<u32> {
        None
    }

    /// The driver's input primitive vertex count (Ryujinx
    /// InputTopology.cs:26-34), in vertex and geometry stages.
    fn primitive_vertices(_info: bool, _state: &GraphicsState) -> Result<Option<u32>, Stopped> {
        Ok(None)
    }

    /// The stage's own output variables.
    fn outputs(self, xs: &[Instruction], output: u32, n: &mut impl FnMut() -> u32)
        -> Self::Outputs;

    /// Whether the program writes ClipDistance outputs the stage declares.
    fn writes_clip_distances(_xs: &[Instruction]) -> bool {
        false
    }

    /// The stage's own execution modes.
    fn execution_modes(_outputs: &Self::Outputs, _w: &mut Vec<u32>, _fun: u32) {}

    /// Workgroup storage is compute-only in Vulkan.
    fn shared_memory(uses: bool) -> Result<(), Stopped> {
        if uses {
            return Err(Stopped::SharedOutsideCompute);
        }
        Ok(())
    }

    /// Instructions only a fragment program executes (IPA, KIL): the
    /// instruction is unsupported elsewhere.
    fn fragment_only(x: &Instruction) -> Result<(), Stopped> {
        Err(bad(x))
    }

    /// Ryujinx AttributeMap.cs:67-68: AST outputs outside the stage
    /// interface exist only in vertex programs.
    fn vertex_only(x: &Instruction) -> Result<(), Stopped> {
        Err(bad(x))
    }

    /// BAR.SYNC's execution and memory scopes and semantics, in the stages
    /// with a control barrier (Ryujinx Spirv/Instructions.cs:252-255).
    fn control_barrier(x: &Instruction) -> Result<(u32, u32), Stopped> {
        Err(bad(x))
    }
}

/// Ryujinx InstEmitAttribute.cs:34-55 advances the byte address per lane,
/// so a static vector ALD can reference two locations.
fn loads_generic(xs: &[Instruction], location: u16) -> bool {
    xs.iter().any(|x| {
        matches!(x.kind, InstructionKind::AttributeLoad(ops::AttributeLoad { address, lanes, .. })
        if lanes.iter(address).any(|(a, lane)| {
            lane.attribute == crate::ir::Attribute::Generic
                && (a - 0x80) / 16 == location
        }))
    })
}

fn vertex_pipeline_addresses<'a>(
    xs: &'a [Instruction],
    cfg: &ControlFlowGraph,
    resolved: &'a mut Vec<Instruction>,
) -> &'a [Instruction] {
    *resolved = super::stages::resolve_physical_addresses(xs, cfg);
    resolved.as_slice()
}

fn topology_vertices(info: bool, state: &GraphicsState) -> Result<Option<u32>, Stopped> {
    if !info {
        return Ok(None);
    }
    Ok(Some(
        match state.geometry_input.ok_or(Stopped::InvocationTopology)? {
            GeometryInput::Points => 1,
            GeometryInput::Lines => 2,
            GeometryInput::LinesAdjacency => 4,
            GeometryInput::Triangles => 3,
            GeometryInput::TrianglesAdjacency => 6,
        },
    ))
}

impl ShaderStage for Vertex<'_> {
    const STAGE: Stage = Stage::Vertex;
    const EXECUTION_MODEL: u32 = 0;
    const CAPABILITIES: &'static [u32] = &[];
    type Outputs = VertexOutputs;

    fn resolve_addresses<'a>(
        xs: &'a [Instruction],
        cfg: &ControlFlowGraph,
        resolved: &'a mut Vec<Instruction>,
    ) -> &'a [Instruction] {
        vertex_pipeline_addresses(xs, cfg, resolved)
    }

    /// Vertex shaders: one vec4 Input and Output per generic location the
    /// header marks (vertex input and output maps).
    fn declare_inputs(self, i: Inputs<'_>, n: &mut impl FnMut() -> u32) {
        let (inputs, outputs) = (self.0.maps().inputs(), self.0.maps().outputs());
        for location in 0..32u32 {
            let referenced = loads_generic(i.xs, location as u16);
            if !i.has_dynamic_input && ((inputs >> (location * 4)) & 0xf != 0 || referenced) {
                let input = n();
                i.decorations
                    .push((input, location, Interpolation::Perspective));
                i.by_location.insert(location as u16, input);
            }
            if (outputs >> (location * 4)) & 0xf != 0 {
                i.vertex_outputs.push((location, n()));
            }
        }
        for (address, builtin) in [(0x2f8u16, 43u32), (0x2fcu16, 42u32)] {
            if i.xs.iter().any(|x| {
                matches!(x.kind, InstructionKind::AttributeLoad(ops::AttributeLoad { address: base, lanes, .. })
                    if lanes.iter(base).any(|(a, _)| a == address))
            }) {
                i.system.insert(address, n());
                let variable = i.system[&address];
                i.system_decorations.push((variable, builtin));
            }
        }
    }

    fn primitive_vertices(info: bool, state: &GraphicsState) -> Result<Option<u32>, Stopped> {
        topology_vertices(info, state)
    }

    fn outputs(
        self,
        xs: &[Instruction],
        output: u32,
        n: &mut impl FnMut() -> u32,
    ) -> VertexOutputs {
        let length = clip_distances(xs);
        VertexOutputs {
            position: output,
            clip: (length > 0).then(|| ClipDistances {
                variable: n(),
                array: n(),
                pointer: n(),
                length,
            }),
        }
    }

    fn writes_clip_distances(xs: &[Instruction]) -> bool {
        clip_distances(xs) > 0
    }

    fn vertex_only(_: &Instruction) -> Result<(), Stopped> {
        Ok(())
    }
}

impl<F: FragmentInterface> ShaderStage for Fragment<F> {
    const STAGE: Stage = Stage::Fragment;
    const EXECUTION_MODEL: u32 = 4;
    const CAPABILITIES: &'static [u32] = &[];
    type Outputs = FragmentOutputs;

    // Mesa nak_nir_lower_tex.c:454 maps NODEP to skip_helpers;
    // nir.h:2704-2708 permits skipped helper results. Ryujinx
    // InstEmitTexture.cs:28-39 emits the same sample for active lanes, and
    // CodeGen/Spirv/IoMap.cs:45 maps helper state to HelperInvocation.
    fn helper_invocation(required: &Required, n: &mut impl FnMut() -> u32) -> Option<u32> {
        (required.helper || required.nodep).then(n)
    }

    fn helper_undefined(required: &Required, n: &mut impl FnMut() -> u32) -> Option<u32> {
        required.nodep.then(n)
    }

    fn frag_coord(reads: bool, n: &mut impl FnMut() -> u32) -> Option<u32> {
        reads.then(n)
    }

    // Ryujinx AttributeMap.cs:81: a[0x3fc] is FrontFacing, read by IPA.
    fn front_facing(xs: &[Instruction], n: &mut impl FnMut() -> u32) -> Option<u32> {
        xs.iter()
            .any(|x| {
                matches!(
                    x.kind,
                    InstructionKind::IpaPass(ops::IpaPass { addr: 0x3fc, .. })
                )
            })
            .then(n)
    }

    /// The fragment input map has one entry per component, index
    /// (addr - 0x80) / 4 (Mesa ir.rs:9526-9531), so location = index / 4;
    /// each location takes the mode of its first used component.
    fn declare_inputs(self, i: Inputs<'_>, n: &mut impl FnMut() -> u32) {
        for location in 0..F::INPUT_MAP_LOCATIONS {
            let declared = self.0.input_mode(location);
            let referenced = loads_generic(i.xs, location as u16);
            // Ryujinx InstEmitAttribute.cs:167-175 references the generic
            // input even if the header's map omitted its declaration.
            let referenced = referenced
                || i.xs.iter().any(|x| {
                    matches!(x.kind, InstructionKind::IpaSaturate(ops::IpaSaturate { addr, .. }) | InstructionKind::IpaPass(ops::IpaPass { addr, .. }) | InstructionKind::IpaPassMulW(ops::IpaPassMulW { addr, .. })
                    if (0x80..0x280).contains(&addr) && (addr - 0x80) / 16 == location as u16)
                });
            if declared == Interpolation::Unused && !referenced {
                continue;
            }
            let interpolation = declared.declared();
            let input = n();
            i.decorations.push((input, location as u32, interpolation));
            i.by_location.insert(location as u16, input);
        }
    }

    fn input_mode(self, location: usize) -> Interpolation {
        self.0.input_mode(location)
    }

    fn outputs(
        self,
        _: &[Instruction],
        output: u32,
        n: &mut impl FnMut() -> u32,
    ) -> FragmentOutputs {
        let fragment_targets = self.0.targets();
        let mut targets = Vec::new();
        for target in 0..8u32 {
            if fragment_targets & (0xf << (target * 4)) != 0 {
                let variable = if targets.is_empty() { output } else { n() };
                targets.push((target, variable));
            }
        }
        let sample_mask = self.0.writes_sample_mask().then(|| {
            let mask = n();
            (mask, (n(), n()))
        });
        FragmentOutputs {
            targets,
            sample_mask,
            depth: self.0.writes_depth().then(n),
        }
    }

    fn execution_modes(outputs: &FragmentOutputs, w: &mut Vec<u32>, fun: u32) {
        inst(w, Op::ExecutionMode, &[fun, 7]);
        for _ in outputs.depth.into_iter() {
            // ExecutionMode DepthReplacing (12).
            inst(w, Op::ExecutionMode, &[fun, 12]);
        }
    }

    fn fragment_only(_: &Instruction) -> Result<(), Stopped> {
        Ok(())
    }
}

impl ShaderStage for Compute {
    const STAGE: Stage = Stage::Compute;
    const EXECUTION_MODEL: u32 = 5;
    const CAPABILITIES: &'static [u32] = &[];
    const GROUP_MEMORY_BARRIER: (u32, u32) = (2, 0x948);
    const MEMORY_BARRIER_SEMANTICS: u32 = 0x948;
    type Outputs = InterfaceOutputs;

    fn compute_registers(_: bool, _: bool) -> Result<(), Stopped> {
        Ok(())
    }

    fn outputs(self, _: &[Instruction], _: u32, _: &mut impl FnMut() -> u32) -> InterfaceOutputs {
        InterfaceOutputs
    }

    fn execution_modes(_: &InterfaceOutputs, w: &mut Vec<u32>, fun: u32) {
        inst(w, Op::ExecutionMode, &[fun, 17, 1, 1, 1]);
    }

    fn shared_memory(_: bool) -> Result<(), Stopped> {
        Ok(())
    }

    fn control_barrier(_: &Instruction) -> Result<(u32, u32), Stopped> {
        Ok((2, 0x108))
    }
}

impl ShaderStage for Geometry<'_> {
    const STAGE: Stage = Stage::Geometry;
    const EXECUTION_MODEL: u32 = 3;
    const CAPABILITIES: &'static [u32] = &[2];
    type Outputs = InterfaceOutputs;

    fn stage_io(
        self,
        state: &GraphicsState,
        xs: &[Instruction],
        n: &mut impl FnMut() -> u32,
    ) -> Result<Option<StageIo>, Stopped> {
        StageIo::plan(self.0, state, xs, n).map(Some)
    }

    fn resolve_addresses<'a>(
        xs: &'a [Instruction],
        cfg: &ControlFlowGraph,
        resolved: &'a mut Vec<Instruction>,
    ) -> &'a [Instruction] {
        vertex_pipeline_addresses(xs, cfg, resolved)
    }

    fn primitive_vertices(info: bool, state: &GraphicsState) -> Result<Option<u32>, Stopped> {
        topology_vertices(info, state)
    }

    fn outputs(self, _: &[Instruction], _: u32, _: &mut impl FnMut() -> u32) -> InterfaceOutputs {
        InterfaceOutputs
    }
}

impl ShaderStage for TessellationControl<'_> {
    const STAGE: Stage = Stage::TessellationControl;
    const EXECUTION_MODEL: u32 = 1;
    const CAPABILITIES: &'static [u32] = &[3];
    type Outputs = InterfaceOutputs;

    fn stage_io(
        self,
        state: &GraphicsState,
        xs: &[Instruction],
        n: &mut impl FnMut() -> u32,
    ) -> Result<Option<StageIo>, Stopped> {
        StageIo::plan(self.0, state, xs, n).map(Some)
    }

    fn resolve_addresses<'a>(
        xs: &'a [Instruction],
        cfg: &ControlFlowGraph,
        resolved: &'a mut Vec<Instruction>,
    ) -> &'a [Instruction] {
        vertex_pipeline_addresses(xs, cfg, resolved)
    }

    fn reads_output_lanes(required: &Required) -> bool {
        required.output_lanes
    }

    fn patch_vertices(info: bool, n: &mut impl FnMut() -> u32) -> Option<u32> {
        info.then(n)
    }

    fn outputs(self, _: &[Instruction], _: u32, _: &mut impl FnMut() -> u32) -> InterfaceOutputs {
        InterfaceOutputs
    }

    /// TCS under GLSL450 implicitly orders Output through the execution
    /// barrier (SPIR-V 1.6, OpControlBarrier). Vulkan
    /// VUID-StandaloneSpirv-ExecutionModel-07320 forbids Workgroup memory
    /// scope for TCS; use Invocation and no memory semantics.
    fn control_barrier(_: &Instruction) -> Result<(u32, u32), Stopped> {
        Ok((4, 0))
    }
}

impl ShaderStage for TessellationEvaluation<'_> {
    const STAGE: Stage = Stage::TessellationEvaluation;
    const EXECUTION_MODEL: u32 = 2;
    const CAPABILITIES: &'static [u32] = &[3];
    type Outputs = InterfaceOutputs;

    fn stage_io(
        self,
        state: &GraphicsState,
        xs: &[Instruction],
        n: &mut impl FnMut() -> u32,
    ) -> Result<Option<StageIo>, Stopped> {
        StageIo::plan(self.0, state, xs, n).map(Some)
    }

    fn resolve_addresses<'a>(
        xs: &'a [Instruction],
        cfg: &ControlFlowGraph,
        resolved: &'a mut Vec<Instruction>,
    ) -> &'a [Instruction] {
        vertex_pipeline_addresses(xs, cfg, resolved)
    }

    fn patch_vertices(info: bool, n: &mut impl FnMut() -> u32) -> Option<u32> {
        info.then(n)
    }

    fn outputs(self, _: &[Instruction], _: u32, _: &mut impl FnMut() -> u32) -> InterfaceOutputs {
        InterfaceOutputs
    }
}
