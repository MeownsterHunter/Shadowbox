//! Original cache-state adapter. Field sources: docs/reference-sources.md
use super::cache::{
    decompress, guests, hash, read, toc, u32_at, u64_at, HEADER_SIZE, SHARED_VERSION,
};
use shadowbox::{
    headers::Stage,
    interface::{
        GeometryInput, ShaderInput, ShaderOutput, Stopped, TessellationDomain, TessellationSpacing,
        TessellationState, Translation,
    },
    CacheContext, CompleteProgram, Follow, Program, ProgramAwaitingState, ProgramPath,
};
use std::{
    collections::{BTreeMap, HashSet},
    fmt,
    path::Path,
};

/// The state recorded for one cached stage.
#[derive(Clone, Debug)]
pub enum RecordedState {
    /// Graphics state the translator classifies; boxed because it is far
    /// larger than the foreign variant.
    Recorded(Box<CacheContext<'static>>),
    /// State this importer cannot supply; the stage is never translated.
    Foreign(ForeignState),
}

/// State a cached stage needs that the graphics-only runtime context cannot
/// carry.
#[derive(Clone, Copy, Debug)]
pub enum ForeignState {
    /// GpuChannelComputeState.cs:13-38.
    ComputeState,
    /// ParallelDiskCacheLoader.cs:623-629,668-676 fuses Vertex A/B.
    VertexAFusion,
}

/// The keys a report lists for the state a program still needs.
pub trait NeededState {
    fn needed_state(&self) -> &'static [&'static str];
}

impl NeededState for ForeignState {
    fn needed_state(&self) -> &'static [&'static str] {
        match self {
            Self::ComputeState => &["compute_state"],
            Self::VertexAFusion => &["vertex_a_fusion"],
        }
    }
}

/// What one recorded state makes of a program, decided once. The program
/// variants are boxed because a parsed program is far larger than the
/// foreign variant.
pub enum StateVariant<'a, 's> {
    /// Every state the stage needs is present in the recorded context the
    /// program is translated under.
    Complete(Box<CompleteProgram<'a>>, &'s CacheContext<'static>),
    /// The stage awaits graphics state the cache did not record.
    AwaitingState(Box<ProgramAwaitingState<'a>>),
    /// The stage needs state this importer cannot supply.
    Foreign(ForeignState),
}

impl NeededState for StateVariant<'_, '_> {
    /// None when complete.
    fn needed_state(&self) -> &'static [&'static str] {
        match self {
            Self::Complete(..) => &[],
            Self::AwaitingState(program) => program.needed_state(),
            Self::Foreign(state) => state.needed_state(),
        }
    }
}

/// Why a state variant gave no module: its translation stopped, or its
/// stage awaits state this importer cannot supply.
pub enum Untranslated {
    Stopped(Stopped),
    AwaitingState {
        stage: Stage,
        needed: &'static [&'static str],
    },
}

impl fmt::Display for Untranslated {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stopped(stopped) => stopped.fmt(f),
            Self::AwaitingState { stage, needed } => write!(
                f,
                "invalid SPIR-V module: {} stage requires state: {}",
                super::aot_scan::stage_name(*stage),
                needed.join(", ")
            ),
        }
    }
}

/// What the AOT scan and cache-import do with each state variant.
pub trait StateWork<'s> {
    /// Translate a complete program with its recorded state; any other
    /// variant fails with the state its `stage` still needs.
    fn translate<R: shadowbox::Run>(
        self,
        run: R,
        stage: Stage,
    ) -> Result<(ShaderOutput, &'s CacheContext<'static>), Untranslated>;
    /// Translate a complete program into the cache at `output`: its import
    /// status and first blocker.
    fn import<R: shadowbox::Run>(
        self,
        run: R,
        input: &ShaderInput<'_>,
        output: &Path,
    ) -> Result<(&'static str, Option<String>), String>;
}

impl<'s> StateWork<'s> for StateVariant<'_, 's> {
    fn translate<R: shadowbox::Run>(
        self,
        run: R,
        stage: Stage,
    ) -> Result<(ShaderOutput, &'s CacheContext<'static>), Untranslated> {
        let needed = self.needed_state();
        match self {
            Self::Complete(program, context) => {
                match shadowbox::translate_parsed(run, program.parsed(), context.options) {
                    Translation::Module(output) => Ok((output, context)),
                    Translation::Stopped(stopped) => Err(Untranslated::Stopped(stopped)),
                }
            }
            Self::AwaitingState(_) | Self::Foreign(_) => {
                Err(Untranslated::AwaitingState { stage, needed })
            }
        }
    }

    fn import<R: shadowbox::Run>(
        self,
        run: R,
        input: &ShaderInput<'_>,
        output: &Path,
    ) -> Result<(&'static str, Option<String>), String> {
        match self {
            Self::AwaitingState(_) | Self::Foreign(_) => Ok(("state_missing", None)),
            Self::Complete(program, context) => {
                match shadowbox::translate_parsed(run, program.parsed(), context.options) {
                    Translation::Stopped(stopped) => Ok(("failed", Some(stopped.to_string()))),
                    Translation::Module(shader) => {
                        shadowbox::emit_translation_cache(output, input, *context, &shader)
                            .map(|_| ("complete", None))
                            .map_err(|e| e.to_string())
                    }
                }
            }
        }
    }
}

/// Where a parsed program goes under one recorded context.
struct UnderContext<'s> {
    context: &'s CacheContext<'static>,
}

impl<'a, 's> ProgramPath<'a> for UnderContext<'s> {
    type Output = StateVariant<'a, 's>;
    fn complete(self, program: CompleteProgram<'a>) -> StateVariant<'a, 's> {
        StateVariant::Complete(Box::new(program), self.context)
    }
    fn awaiting_state(self, program: ProgramAwaitingState<'a>) -> StateVariant<'a, 's> {
        StateVariant::AwaitingState(Box::new(program))
    }
}

/// The program's identity under `context`, complete or not.
pub fn identity(input: &ShaderInput<'_>, context: CacheContext) -> Result<String, Stopped> {
    Ok(
        Program::parse(input, context.graphics_state)?
            .identity(context.generation, context.options),
    )
}

impl RecordedState {
    pub fn absent() -> Self {
        Self::Recorded(Box::default())
    }
}

/// Parse a program once under a recorded state.
pub trait Classify {
    /// The program's identity, and whether it is complete, awaits recorded
    /// state, or needs state this importer cannot supply.
    fn classify<'a, 's>(
        &'s self,
        input: &ShaderInput<'a>,
    ) -> Result<(String, StateVariant<'a, 's>), Stopped>;
}

impl Classify for RecordedState {
    fn classify<'a, 's>(
        &'s self,
        input: &ShaderInput<'a>,
    ) -> Result<(String, StateVariant<'a, 's>), Stopped> {
        Ok(match self {
            Self::Recorded(context) => {
                let program = Program::parse(input, context.graphics_state)?;
                (
                    program.identity(context.generation, context.options),
                    program.follow(UnderContext { context }),
                )
            }
            // Foreign state is named by the program's default identity.
            Self::Foreign(state) => (
                identity(input, CacheContext::default())?,
                StateVariant::Foreign(*state),
            ),
        })
    }
}
pub struct CachedStage {
    pub stage: Stage,
    pub record: Vec<u8>,
    pub state: RecordedState,
}
pub type StateIndex = BTreeMap<(u8, String), Vec<RecordedState>>;

fn graphics_state(bytes: &[u8], stage: Stage) -> Result<RecordedState, String> {
    let mut state = CacheContext::default();
    // Raw LE unmanaged struct: BinarySerializer.cs:99-102. Natural u32
    // alignment puts Topology at +4 and TessMode.Packed at +8 after bool +0:
    // GpuChannelGraphicsState.cs:18-28; ThreedClassState.cs:27-30.
    if stage == Stage::Geometry {
        if bytes.len() < 8 {
            return Ok(RecordedState::Recorded(Box::new(state)));
        }
        let topology = u32_at(bytes, 4)?;
        // GAL/PrimitiveTopology.cs:5-19; GpuChannelGraphicsState.cs:168-187.
        state.graphics_state.geometry_input = Some(match topology {
            0 | 7..=9 => GeometryInput::Points,
            1..=3 => GeometryInput::Lines,
            4..=6 => GeometryInput::Triangles,
            10 | 11 => GeometryInput::LinesAdjacency,
            12 | 13 => GeometryInput::TrianglesAdjacency,
            14 => {
                if bytes.len() < 12 {
                    return Ok(RecordedState::Recorded(Box::new(state)));
                }
                match u32_at(bytes, 8)? & 3 {
                    0 => GeometryInput::Lines,
                    1 | 2 => GeometryInput::Triangles,
                    _ => return Err("invalid tessellation patch type".into()),
                }
            }
            _ => return Err(format!("invalid primitive topology {topology}")),
        });
    }
    if stage == Stage::TessellationEvaluation {
        if bytes.len() < 12 {
            return Ok(RecordedState::Recorded(Box::new(state)));
        }
        let mode = u32_at(bytes, 8)?;
        // ThreedClassState.cs:37-57; Shader/TessPatchType.cs:5-7,
        // TessSpacing.cs:5-7. Other packed bits are not state inputs to this
        // translator, exactly as in the three upstream unpack routines.
        state.graphics_state.tessellation = Some(TessellationState {
            domain: match mode & 3 {
                0 => TessellationDomain::Isolines,
                1 => TessellationDomain::Triangles,
                2 => TessellationDomain::Quads,
                _ => return Err("invalid tessellation patch type".into()),
            },
            spacing: match (mode >> 4) & 3 {
                0 => TessellationSpacing::Equal,
                1 => TessellationSpacing::FractionalEven,
                2 => TessellationSpacing::FractionalOdd,
                _ => return Err("invalid tessellation spacing".into()),
            },
            // Final Vulkan order. Ryujinx SpirvGenerator.cs:222-236 flips
            // guest winding for its Vulkan front-face convention. Fold once
            // here rather than adding a shader operation.
            clockwise: mode & (1 << 8) == 0,
        });
    }
    Ok(RecordedState::Recorded(Box::new(state)))
}

pub fn load(input: &Path) -> Result<Vec<CachedStage>, String> {
    let guests = guests(
        &read(&input.join("guest.toc"))?,
        &read(&input.join("guest.data"))?,
    )?;
    let shared_toc = read(&input.join("shared.toc"))?;
    let data = read(&input.join("shared.data"))?;
    let entries = toc(&shared_toc, b"TOCS", SHARED_VERSION, 8)?;
    // DiskCacheHostStorage.cs:331-343,592-618: offsets, stage mask, indices,
    // then specialization. A zero mask has one compute index.
    let offsets = entries
        .as_chunks::<8>()
        .0
        .iter()
        .map(|e| usize::try_from(u64_at(e, 0)?).map_err(|_| "shared offset too large".into()))
        .collect::<Result<Vec<_>, String>>()?;
    let mut boundaries = offsets.clone();
    boundaries.sort_unstable();
    if boundaries.windows(2).any(|p| p[0] == p[1]) {
        return Err("duplicate shared data offset".into());
    }
    let mut result = Vec::new();
    for start in offsets {
        let end = boundaries
            .get(boundaries.binary_search(&start).unwrap() + 1)
            .copied()
            .unwrap_or(data.len());
        let shared = decompress(
            data.get(start..end).ok_or("shared offset out of range")?,
            None,
        )?;
        let mask = u32_at(&shared, 0)?;
        if mask & !0x3f != 0 {
            return Err("invalid stage mask".into());
        }
        let compute = mask == 0;
        let slots = if compute { 1 } else { mask };
        let mut cursor = 4;
        let mut stages = Vec::new();
        for slot in 0..6 {
            if slots & (1 << slot) == 0 {
                continue;
            }
            let index = u32_at(&shared, cursor)? as i32;
            cursor += 4;
            let guest = usize::try_from(index)
                .ok()
                .and_then(|i| guests.get(i))
                .ok_or("invalid guest index")?;
            let mut record = if compute {
                vec![0; HEADER_SIZE]
            } else {
                Vec::new()
            };
            record.extend_from_slice(&guest.code);
            if record.len() < HEADER_SIZE || !(record.len() - HEADER_SIZE).is_multiple_of(8) {
                return Err("invalid guest shader length".into());
            }
            // ShaderHeader.cs:109; Mesa nak/sph.rs:204-215.
            let expected = if compute { 0 } else { slot.max(1) };
            if (u32_at(&record, 0)? >> 10) & 0xf != expected {
                return Err("SPH stage disagrees with shared slot".into());
            }
            let stage = [
                Stage::Compute,
                Stage::Vertex,
                Stage::TessellationControl,
                Stage::TessellationEvaluation,
                Stage::Geometry,
                Stage::Fragment,
            ][expected as usize];
            stages.push((slot, stage, record));
        }
        // ShaderSpecializationState.cs:39-40,974-983: u32 queried flags,
        // one-byte compute bool, then magic/size-prefixed native state.
        // BinarySerializer.cs:111-116; magic values at specialization:20-21.
        let spec = shared.get(cursor..).ok_or("missing specialization")?;
        if spec.get(4).copied() != Some(u8::from(compute)) {
            return Err("specialization compute flag disagrees with stage mask".into());
        }
        if spec.get(5..9) != Some(if compute { b"COMS" } else { b"GFXS" }) {
            return Err("invalid GPU state magic".into());
        }
        let size = u32_at(spec, 9)? as usize;
        let raw = spec
            .get(13..13usize.checked_add(size).ok_or("GPU state size overflow")?)
            .ok_or("truncated GPU state")?;
        for (slot, stage, record) in stages {
            let state = if compute {
                // GpuChannelComputeState.cs:13-38 cannot be supplied by the
                // current graphics-only runtime context. Never prewarm guesses.
                RecordedState::Foreign(ForeignState::ComputeState)
            } else if mask & 1 != 0 && slot <= 1 {
                // ParallelDiskCacheLoader.cs:623-629,668-676 fuses Vertex A/B.
                // Shadowbox has no fusion input, so independent entries are unsafe.
                RecordedState::Foreign(ForeignState::VertexAFusion)
            } else {
                graphics_state(raw, stage)?
            };
            result.push(CachedStage {
                stage,
                record,
                state,
            });
        }
    }
    Ok(result)
}

pub fn index(input: &Path) -> Result<StateIndex, String> {
    let mut index = StateIndex::new();
    let mut seen = HashSet::new();
    for program in load(input)? {
        let input =
            shadowbox::header_prefixed_input(&program.record, shadowbox::TargetGeneration::Sm5)
                .map_err(|e| e.to_string())?;
        let (key, variant) = program.state.classify(&input).map_err(|e| e.to_string())?;
        if seen.insert((key, variant.needed_state())) {
            index
                .entry((program.stage as u8, hash(input.code)))
                .or_default()
                .push(program.state);
        }
    }
    Ok(index)
}

// Structured results are project-owned reporting fields, not new GPU fields.
// Both cache-import and prewarm-pack use this path and the same recorded state.
pub(super) fn import_report<R: shadowbox::Run>(
    run: R,
    input: &Path,
    output: &Path,
) -> Result<Vec<serde_json::Value>, String> {
    let programs = load(input)?;
    let mut seen = HashSet::new();
    let mut results = Vec::new();
    for program in programs {
        let input =
            shadowbox::header_prefixed_input(&program.record, shadowbox::TargetGeneration::Sm5)
                .map_err(|e| e.to_string())?;
        let (key, variant) = program.state.classify(&input).map_err(|e| e.to_string())?;
        let program_key = identity(&input, CacheContext::default()).map_err(|e| e.to_string())?;
        let missing = variant.needed_state();
        if !seen.insert((key.clone(), missing)) {
            continue;
        }
        let (status, blocker) = variant.import(run, &input, output)?;
        results.push(serde_json::json!({
            "program_key": program_key, "stage": super::aot_scan::stage_name(program.stage),
            "translation_key": key, "status": status, "first_blocker": blocker,
            "missing_state_keys": missing,
        }));
    }
    results.sort_by_key(|v| v.to_string());
    Ok(results)
}

pub fn import<R: shadowbox::Run>(run: R, input: &Path, output: &Path) -> Result<(), String> {
    let results = import_report(run, input, output)?;
    let count = |status: &str| results.iter().filter(|r| r["status"] == status).count();
    for result in &results {
        if let Some(error) = result["first_blocker"].as_str() {
            eprintln!("translation failed: {error}");
        }
    }
    println!(
        "cache-import: complete={} state_missing={} translation_failed={}",
        count("complete"),
        count("state_missing"),
        count("failed")
    );
    Ok(())
}
