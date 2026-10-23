//! Novena integration for Shadowbox's Maxwell/Pascal shader translator.

use novena::{ShaderStage, ShaderTranslator, SHADER_STAGE_UNKNOWN};
use shadowbox::headers::{self, OutsideHeader};
use std::collections::HashSet;
use std::fmt;
use std::fs;
use std::io;
use std::iter::once;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use novena::startup_cache::{TranslatedShader, TranslationContext};
use novena::surfaces::{
    BindlessHandle, HandleOperation, HandleWord, RawFormat, SurfaceBinding, SurfaceDimension,
    SurfaceFormat, SurfaceSource,
};
use shadowbox::emitter::{HandleLeaf, HandleSource, SurfaceDescriptor};
use shadowbox::ir::LogicOp;
use shadowbox::{
    cache::{decode_record, encode_record, CacheEntryStatus},
    interface::{
        GeometryInput, GraphicsState, Malformed, ShaderOutput, Stopped, TessellationDomain,
        TessellationSpacing, TessellationState,
    },
    CacheContext, CacheProgram, Follow, ProgramErrorPath, TargetGeneration, TranslationOptions,
};

/// The runs a host chooses from at its entry.
pub use shadowbox::{QuietRun, Run, TracedRun};

/// The raw stage token that asks the adapter to read the stage from the
/// SM 5.x header.
pub const STAGE_FROM_HEADER: ShaderStage = 255;

/// The stage comes from the shader's own header.
///
/// The SM 5.x header in front of the code always decides the stage: novena
/// passes [`SHADER_STAGE_UNKNOWN`] (0) until a stage is established, and
/// [`STAGE_FROM_HEADER`] asks for the header's stage explicitly; both mean
/// [`FromHeader`]. Tokens 4 and 5 name the fragment and compute stages
/// (their SPIR-V execution models) as a [`headers::Stage`], which must agree
/// with the header. Every other token is rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FromHeader;

/// The stage a caller asks for: the header's stage, after checking a named
/// stage against it.
pub trait Reconcile: Copy {
    fn reconcile(self, header: headers::Stage) -> Result<headers::Stage, AdapterError>;
}

impl Reconcile for FromHeader {
    fn reconcile(self, header: headers::Stage) -> Result<headers::Stage, AdapterError> {
        Ok(header)
    }
}

/// A named stage agrees with the header's, or names another stage.
impl Reconcile for headers::Stage {
    fn reconcile(self, header: headers::Stage) -> Result<headers::Stage, AdapterError> {
        [other_stage, same_stage][usize::from(self == header)](self, header)
    }
}

fn other_stage(
    given: headers::Stage,
    header: headers::Stage,
) -> Result<headers::Stage, AdapterError> {
    Err(AdapterError::TokenNamesOtherStage { given, header })
}

fn same_stage(_: headers::Stage, header: headers::Stage) -> Result<headers::Stage, AdapterError> {
    Ok(header)
}

type Token<R, F> = fn(&ShadowboxTranslator<R, F>, ShaderStage, &[u8]) -> Words;

impl<R: Run, F: Failures> ShadowboxTranslator<R, F> {
    /// What each raw token below 256 translates as; index 256 stands for
    /// every larger token.
    const TOKENS: [Token<R, F>; 257] = {
        let mut tokens: [Token<R, F>; 257] = [Self::unknown_token; 257];
        tokens[SHADER_STAGE_UNKNOWN as usize] = Self::header_stage;
        tokens[STAGE_FROM_HEADER as usize] = Self::header_stage;
        tokens[4] = Self::fragment;
        tokens[5] = Self::compute;
        tokens
    };

    fn unknown_token(&self, token: ShaderStage, _: &[u8]) -> Words {
        Err(AdapterError::UnknownStageToken(token))
    }
    fn header_stage(&self, _: ShaderStage, code: &[u8]) -> Words {
        self.translate_with_stage(FromHeader, code)
    }
    fn fragment(&self, _: ShaderStage, code: &[u8]) -> Words {
        self.translate_with_stage(headers::Stage::Fragment, code)
    }
    fn compute(&self, _: ShaderStage, code: &[u8]) -> Words {
        self.translate_with_stage(headers::Stage::Compute, code)
    }
}

/// Adapter failures. `Display` gives the exact text novena receives.
#[derive(Debug)]
pub enum AdapterError {
    /// Novena passed a stage token the adapter does not define.
    UnknownStageToken(ShaderStage),
    /// The token names a stage other than the one in the SM 5.x header.
    TokenNamesOtherStage {
        given: headers::Stage,
        header: headers::Stage,
    },
    InvalidGeneration,
    InvalidGeometryInput,
    InvalidTessellationWinding,
    InvalidTessellationDomain,
    InvalidTessellationSpacing,
    /// The header-prefixed shader is shorter than its SM 5.x header.
    ShortShaderHeader,
    /// The SM 5.x header in front of the shader code is invalid.
    ShaderHeader(OutsideHeader),
    /// The header of the parsed program record is invalid.
    ProgramHeader(OutsideHeader),
    Translation(Stopped),
    /// Surface aliases need `translate_with_bindings`.
    SurfaceAliases,
    /// Indexed constant-buffer handles require a host-side descriptor array.
    DynamicSurfaceHandle,
    /// A surface target outside Maxwell's SuDim values.
    SurfaceTarget(u8),
    /// A surface image format the translator does not emit.
    SurfaceFormat(u32),
    EncodeRecord(io::Error),
    CacheRecord(CacheEntryStatus),
}

impl fmt::Display for AdapterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownStageToken(token) => write!(f, "unknown shader stage token {token}"),
            Self::TokenNamesOtherStage { given, header } => write!(
                f,
                "shader stage token names {given:?} but the SM 5.x header is {header:?}"
            ),
            Self::InvalidGeneration => f.write_str("invalid target generation"),
            Self::InvalidGeometryInput => f.write_str("invalid geometry input"),
            Self::InvalidTessellationWinding => f.write_str("invalid tessellation winding"),
            Self::InvalidTessellationDomain => f.write_str("invalid tessellation domain"),
            Self::InvalidTessellationSpacing => f.write_str("invalid tessellation spacing"),
            Self::ShortShaderHeader => {
                f.write_str("header-prefixed shader is shorter than its 80-byte SM 5.x header")
            }
            Self::ShaderHeader(error) => write!(f, "invalid SM 5.x shader header: {error:?}"),
            Self::ProgramHeader(error) => write!(f, "{error:?}"),
            Self::Translation(error) => write!(f, "{error}"),
            Self::SurfaceAliases => f.write_str(
                "surface aliases require the binding metadata from translate_with_bindings",
            ),
            Self::DynamicSurfaceHandle => {
                f.write_str("indexed surface handles require a host descriptor array")
            }
            Self::SurfaceTarget(target) => write!(f, "unknown surface target {target}"),
            Self::SurfaceFormat(format) => write!(f, "unknown surface image format {format}"),
            Self::EncodeRecord(error) => write!(f, "{error}"),
            Self::CacheRecord(status) => {
                write!(f, "invalid translation cache record: {}", status.as_str())
            }
        }
    }
}

impl std::error::Error for AdapterError {}

impl From<AdapterError> for String {
    fn from(error: AdapterError) -> Self {
        error.to_string()
    }
}

/// Builds the error for an invalid raw byte.
type Invalid = fn() -> AdapterError;

/// A table of 256 entries for one raw novena byte: `valid` at the front,
/// `invalid` everywhere else.
const fn byte_table<T: Copy, const N: usize>(
    valid: [T; N],
    invalid: Result<T, Invalid>,
) -> [Result<T, Invalid>; 256] {
    let mut table = [invalid; 256];
    let mut index = 0;
    while index < N {
        table[index] = Ok(valid[index]);
        index += 1;
    }
    table
}

/// Target generation of each raw byte.
const GENERATIONS: [Result<TargetGeneration, Invalid>; 256] = byte_table(
    [
        TargetGeneration::Sm5,
        TargetGeneration::Sm6,
        TargetGeneration::Sm7_5Plus,
    ],
    Err(|| AdapterError::InvalidGeneration),
);

/// Geometry input of each raw byte; 0 names none and is invalid.
const GEOMETRY_INPUTS: [Result<GeometryInput, Invalid>; 256] = {
    let mut table: [Result<GeometryInput, Invalid>; 256] =
        [Err(|| AdapterError::InvalidGeometryInput); 256];
    let inputs = [
        GeometryInput::Points,
        GeometryInput::Lines,
        GeometryInput::LinesAdjacency,
        GeometryInput::Triangles,
        GeometryInput::TrianglesAdjacency,
    ];
    let mut index = 0;
    while index < inputs.len() {
        table[index + 1] = Ok(inputs[index]);
        index += 1;
    }
    table
};

/// Winding of each raw byte.
const WINDINGS: [Result<bool, Invalid>; 256] = byte_table(
    [false, true],
    Err(|| AdapterError::InvalidTessellationWinding),
);

/// Tessellation domain of each raw byte.
const DOMAINS: [Result<TessellationDomain, Invalid>; 256] = byte_table(
    [
        TessellationDomain::Isolines,
        TessellationDomain::Triangles,
        TessellationDomain::Quads,
    ],
    Err(|| AdapterError::InvalidTessellationDomain),
);

/// Tessellation spacing of each raw byte.
const SPACINGS: [Result<TessellationSpacing, Invalid>; 256] = byte_table(
    [
        TessellationSpacing::Equal,
        TessellationSpacing::FractionalEven,
        TessellationSpacing::FractionalOdd,
    ],
    Err(|| AdapterError::InvalidTessellationSpacing),
);

fn context(context: &TranslationContext) -> Result<CacheContext<'_>, AdapterError> {
    // Validation order is the original's: generation, geometry input, then
    // winding, domain and spacing.
    let invalid = |invalid: Invalid| invalid();
    GENERATIONS[usize::from(context.generation)]
        .map_err(invalid)
        .and_then(|generation| {
            context
                .geometry_input
                .map(|input| GEOMETRY_INPUTS[usize::from(input)].map_err(invalid))
                .transpose()
                .and_then(|geometry_input| {
                    context
                        .tessellation
                        .map(|[domain, spacing, clockwise]| {
                            WINDINGS[usize::from(clockwise)].map_err(invalid).and_then(
                                |clockwise| {
                                    DOMAINS[usize::from(domain)].map_err(invalid).and_then(
                                        |domain| {
                                            SPACINGS[usize::from(spacing)].map_err(invalid).map(
                                                |spacing| TessellationState {
                                                    clockwise,
                                                    domain,
                                                    spacing,
                                                },
                                            )
                                        },
                                    )
                                },
                            )
                        })
                        .transpose()
                        .map(|tessellation| CacheContext {
                            generation,
                            options: TranslationOptions {
                                // novena's flag is its own shape; the
                                // edge names the delta here once.
                                global_delta: match context.global_delta_zero {
                                    true => shadowbox::GlobalDelta::Zero,
                                    false => shadowbox::GlobalDelta::Pushed,
                                },
                                float64: shadowbox::HostFloat64::Unsupported,
                            },
                            graphics_state: GraphicsState {
                                geometry_input,
                                tessellation,
                                ..Default::default()
                            },
                        })
                })
        })
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    bytes.iter().fold(0xcbf29ce484222325, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3)
    })
}

/// What a translator does with an input that did not translate.
pub trait Failures: Send + Sync {
    fn stopped(&self, code: &[u8], error: &str);
}

/// A translator that keeps nothing of a failed input.
#[derive(Debug, Default, Copy, Clone)]
pub struct Forgets;

impl Failures for Forgets {
    fn stopped(&self, _: &[u8], _: &str) {}
}

/// A translator that dumps each failing input once into its directory: the
/// code as `<fnv1a64>.bin` and the reason as `<fnv1a64>.err`. The host
/// chooses it at its entry (the novena host from `SHADOWBOX_FAIL_DIR`).
#[derive(Debug)]
pub struct Dumps {
    dir: PathBuf,
    /// The inputs already dumped.
    dumped: Mutex<HashSet<u64>>,
}

impl Dumps {
    pub fn to(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            dumped: Mutex::new(HashSet::new()),
        }
    }
}

impl Failures for Dumps {
    fn stopped(&self, code: &[u8], error: &str) {
        let hash = fnv1a64(code);
        // The set only ever gains hashes, so a panic elsewhere leaves it whole.
        let (Ok(mut seen) | Err(mut seen)) = self.dumped.lock().map_err(PoisonError::into_inner);
        once(())
            .take(usize::from(seen.insert(hash)))
            .for_each(|()| {
                let stem = format!("{hash:016x}");
                let dir: &Path = &self.dir;
                let _ = fs::create_dir_all(dir).map(|()| {
                    let _ = fs::write(dir.join(format!("{stem}.bin")), code);
                    let _ = fs::write(dir.join(format!("{stem}.err")), error);
                });
            });
    }
}

/// A translator backed by Shadowbox, on the run and with the failure
/// handling its host chose once at its entry.
#[derive(Debug, Copy, Clone)]
pub struct ShadowboxTranslator<R = QuietRun, F = Forgets> {
    run: R,
    failures: F,
}

impl Default for ShadowboxTranslator {
    fn default() -> Self {
        Self::new()
    }
}

impl ShadowboxTranslator {
    /// A quiet translator that keeps nothing of failed inputs.
    pub const fn new() -> Self {
        Self::on(QuietRun, Forgets)
    }
}

impl<R: Run, F: Failures> ShadowboxTranslator<R, F> {
    /// A translator on `run`, handing failed inputs to `failures`.
    pub const fn on(run: R, failures: F) -> Self {
        Self { run, failures }
    }

    pub fn translate_header_prefixed(&self, data: &[u8]) -> Result<Vec<u32>, AdapterError> {
        self.translate_with_stage(FromHeader, data)
            .map(|output| output.0)
    }

    /// Translate and return the stage recorded in the SM 5.x header.
    ///
    /// The Novena trait only returns SPIR-V, so callers that need the derived
    /// stage can use this query alongside the trait method. The header mapping
    /// follows Mesa NAK's `ShaderType` values and stage construction in
    /// `src/nouveau/compiler/nak/sph.rs:24-52,105-120,204-215`.
    pub fn translate_with_stage(
        &self,
        stage: impl Reconcile,
        code: &[u8],
    ) -> Result<(Vec<u32>, headers::Stage), AdapterError> {
        self.translate_with_stage_and_graphics_state(
            stage,
            code,
            &shadowbox::interface::GraphicsState::default(),
        )
    }

    /// Supply the draw state absent from the SPH for geometry/tessellation.
    /// Ryujinx ShaderDefinitions.cs:18-21,158-160; see the translator's
    /// docs/geometry-tessellation.md for the pipeline contract.
    pub fn translate_with_stage_and_graphics_state(
        &self,
        stage: impl Reconcile,
        code: &[u8],
        state: &shadowbox::interface::GraphicsState,
    ) -> Result<(Vec<u32>, headers::Stage), AdapterError> {
        self.translate_with_bindings(stage, code, state)
            .and_then(|(output, selected)| {
                let descriptors = &output.surface_descriptors;
                // Pairs of descriptors that view one surface source.
                let aliases = descriptors
                    .iter()
                    .enumerate()
                    .flat_map(|(i, descriptor)| {
                        descriptors[..i]
                            .iter()
                            .map(move |other| usize::from(other.source == descriptor.source))
                    })
                    .sum::<usize>();
                [unaliased, aliased][usize::from(aliases != 0)](output.spirv, selected)
            })
    }

    /// Return typed surface views along with SPIR-V. The host must bind every
    /// view before execution; the word-only Novena trait cannot carry aliases.
    pub fn translate_with_bindings(
        &self,
        stage: impl Reconcile,
        code: &[u8],
        state: &shadowbox::interface::GraphicsState,
    ) -> Result<(shadowbox::interface::ShaderOutput, headers::Stage), AdapterError> {
        // Parsed once, header first: the stage, the cb1 words and the
        // translation all come from this one program.
        shadowbox::parse_header_prefixed_header_first(code, TargetGeneration::Sm5, *state)
            .map_err(|error| error.follow(self.fault(code)))
            .and_then(|program| {
                stage.reconcile(program.stage()).and_then(|selected| {
                    Result::from(shadowbox::translate_parsed(
                        self.run,
                        &program,
                        TranslationOptions::default(),
                    ))
                    .map_err(|stopped| self.fault(code).stopped(stopped))
                    .map(|output| (output, selected))
                })
            })
    }

    fn fault<'c>(&'c self, code: &'c [u8]) -> TranslateFault<'c, F> {
        TranslateFault {
            code,
            failures: &self.failures,
        }
    }
}

type Words = Result<(Vec<u32>, headers::Stage), AdapterError>;

fn unaliased(words: Vec<u32>, stage: headers::Stage) -> Words {
    Ok((words, stage))
}
fn aliased(_: Vec<u32>, _: headers::Stage) -> Words {
    Err(AdapterError::SurfaceAliases)
}

/// The startup shader with its surface views, each parsed once here into
/// novena's types.
fn startup_shader(output: ShaderOutput) -> Result<TranslatedShader, AdapterError> {
    output
        .surface_descriptors
        .iter()
        .map(surface_binding)
        .collect::<Result<Vec<_>, _>>()
        .map(|surfaces| TranslatedShader {
            key: String::new(),
            words: output.spirv,
            requires_subgroup_size_32: output.requires_subgroup_size_32,
            requires_demote_to_helper_invocation: output.requires_demote_to_helper_invocation,
            requires_maximal_reconvergence: output.requires_maximal_reconvergence,
            surfaces,
        })
}

fn surface_binding(descriptor: &SurfaceDescriptor) -> Result<SurfaceBinding, AdapterError> {
    Ok(SurfaceBinding {
        set: descriptor.set,
        binding: descriptor.binding,
        source: surface_source(descriptor.source)?,
        dimension: surface_dimension(descriptor.target)?,
        format: surface_format(descriptor.image_format)?,
    })
}

fn surface_source(source: HandleSource) -> Result<SurfaceSource, AdapterError> {
    Ok(match source {
        HandleSource::Bound(slot) => SurfaceSource::Bound { slot },
        HandleSource::Leaf(leaf) => {
            SurfaceSource::Bindless(BindlessHandle::Word(handle_word(leaf)))
        }
        HandleSource::Combined { op, a, b } => SurfaceSource::Bindless(BindlessHandle::Combined {
            operation: match op {
                LogicOp::And => HandleOperation::And,
                LogicOp::Or => HandleOperation::Or,
                LogicOp::Xor => HandleOperation::Xor,
                LogicOp::PassB => HandleOperation::PassInvertedB,
            },
            a: handle_word(a),
            b: handle_word(b),
        }),
        HandleSource::ConstantBufferArray { .. } => return Err(AdapterError::DynamicSurfaceHandle),
    })
}

fn handle_word(leaf: HandleLeaf) -> HandleWord {
    match leaf {
        HandleLeaf::Constant(word) => HandleWord::Constant(word),
        HandleLeaf::ConstantBuffer { bank, offset } => HandleWord::ConstantBuffer { bank, offset },
    }
}

/// Maxwell SuDim (Ryujinx Decoders/InstDecoders.cs, `SuDim`).
fn surface_dimension(target: u8) -> Result<SurfaceDimension, AdapterError> {
    match target {
        0 => Ok(SurfaceDimension::D1),
        1 => Ok(SurfaceDimension::Buffer),
        2 => Ok(SurfaceDimension::D1Array),
        3 => Ok(SurfaceDimension::D2),
        4 => Ok(SurfaceDimension::D2Array),
        5 => Ok(SurfaceDimension::D3),
        _ => Err(AdapterError::SurfaceTarget(target)),
    }
}

/// SPIR-V `ImageFormat` values the translator emits for surfaces: Unknown
/// for formatted access, and the raw and atomic formats of Ryujinx
/// Instructions/InstEmitSurface.cs.
fn surface_format(format: u32) -> Result<SurfaceFormat, AdapterError> {
    match format {
        0 => Ok(SurfaceFormat::Formatted),
        39 => Ok(SurfaceFormat::Raw(RawFormat::R8Uint)),
        29 => Ok(SurfaceFormat::Raw(RawFormat::R8Sint)),
        38 => Ok(SurfaceFormat::Raw(RawFormat::R16Uint)),
        28 => Ok(SurfaceFormat::Raw(RawFormat::R16Sint)),
        33 => Ok(SurfaceFormat::Raw(RawFormat::R32Uint)),
        24 => Ok(SurfaceFormat::Raw(RawFormat::R32Sint)),
        35 => Ok(SurfaceFormat::Raw(RawFormat::Rg32Uint)),
        30 => Ok(SurfaceFormat::Raw(RawFormat::Rgba32Uint)),
        _ => Err(AdapterError::SurfaceFormat(format)),
    }
}

type Decoded = Result<Option<TranslatedShader>, String>;

/// What novena receives for each record status: stale and mismatched
/// records are misses; anything else is an invalid record.
const DECODED: [fn(CacheEntryStatus) -> Decoded; 4] =
    [invalid_record, missed_record, missed_record, invalid_record];

fn missed_record(_: CacheEntryStatus) -> Decoded {
    Ok(None)
}
fn invalid_record(status: CacheEntryStatus) -> Decoded {
    Err(AdapterError::CacheRecord(status).into())
}

/// A program that failed on the translation path: header faults keep their
/// adapter errors, and a translation failure dumps its code first.
struct TranslateFault<'c, F> {
    code: &'c [u8],
    failures: &'c F,
}

impl<F: Failures> ProgramErrorPath for TranslateFault<'_, F> {
    type Output = AdapterError;
    fn short_header(self) -> AdapterError {
        AdapterError::ShortShaderHeader
    }
    fn record(self, error: Malformed) -> AdapterError {
        self.stopped(error.into())
    }
    fn header(self, outside: OutsideHeader) -> AdapterError {
        AdapterError::ShaderHeader(outside)
    }
}

impl<F: Failures> TranslateFault<'_, F> {
    /// A translation that stopped: its code is handed to the failures first.
    fn stopped(self, stopped: Stopped) -> AdapterError {
        self.failures.stopped(self.code, &stopped.to_string());
        AdapterError::Translation(stopped)
    }
}

/// A program that failed on the cache-key path: header faults are program
/// header errors, everything else the translator's error.
struct CacheKeyFault;

impl ProgramErrorPath for CacheKeyFault {
    type Output = AdapterError;
    fn short_header(self) -> AdapterError {
        AdapterError::Translation(Malformed::ShortHeader.into())
    }
    fn record(self, error: Malformed) -> AdapterError {
        AdapterError::Translation(error.into())
    }
    fn header(self, outside: OutsideHeader) -> AdapterError {
        AdapterError::ProgramHeader(outside)
    }
}

impl<R: Run, F: Failures> ShaderTranslator for ShadowboxTranslator<R, F> {
    fn translate(&self, stage: ShaderStage, code: &[u8]) -> Result<Vec<u32>, String> {
        Self::TOKENS[stage.min(256) as usize](self, stage, code)
            .map(|(words, _)| words)
            .map_err(String::from)
    }
    fn translation_cache_key(
        &self,
        _: u32,
        program: &[u8],
        state: &TranslationContext,
    ) -> Result<Option<String>, String> {
        context(state)
            .and_then(|context| {
                shadowbox::parse_header_prefixed_for_cache(program, context)
                    .map_err(|error| error.follow(CacheKeyFault))
                    .map(|program| match program {
                        CacheProgram::Complete(program) => {
                            Some(program.cache_key(context.generation, context.options))
                        }
                        CacheProgram::Incomplete { .. } => None,
                    })
            })
            .map_err(String::from)
    }
    fn translate_for_context(
        &self,
        _: u32,
        program: &[u8],
        state: &TranslationContext,
    ) -> Result<TranslatedShader, String> {
        context(state)
            .and_then(|context| {
                shadowbox::parse_header_prefixed(
                    program,
                    context.generation,
                    context.graphics_state,
                )
                .map_err(Stopped::from)
                .and_then(|program| {
                    shadowbox::translate_parsed(self.run, &program, context.options).into()
                })
                .map_err(AdapterError::Translation)
            })
            .and_then(startup_shader)
            .map_err(String::from)
    }
    fn encode_cache_record(
        &self,
        shader: &TranslatedShader,
        state: &TranslationContext,
    ) -> Result<Option<Vec<u8>>, String> {
        // A record has no form for surface views (Shadowbox's own cache keeps
        // such outputs in memory only), so a shader with surfaces has none.
        match shader.surfaces.as_slice() {
            [] => {
                let output = ShaderOutput {
                    spirv: shader.words.clone(),
                    surface_descriptors: Vec::new(),
                    requires_subgroup_size_32: shader.requires_subgroup_size_32,
                    requires_maximal_reconvergence: shader.requires_maximal_reconvergence,
                    requires_demote_to_helper_invocation: shader
                        .requires_demote_to_helper_invocation,
                };
                context(state)
                    .and_then(|context| {
                        encode_record(&shader.key, context, &output)
                            .map_err(AdapterError::EncodeRecord)
                    })
                    .map(Some)
                    .map_err(String::from)
            }
            _ => Ok(None),
        }
    }

    fn decode_cache_record(
        &self,
        key: &str,
        bytes: &[u8],
    ) -> Result<Option<TranslatedShader>, String> {
        decode_record(bytes, key)
            .map(|output| {
                Some(TranslatedShader {
                    key: key.into(),
                    words: output.spirv,
                    requires_subgroup_size_32: output.requires_subgroup_size_32,
                    requires_maximal_reconvergence: output.requires_maximal_reconvergence,
                    requires_demote_to_helper_invocation: output
                        .requires_demote_to_helper_invocation,
                    // Records exist only for shaders without surfaces.
                    surfaces: Vec::new(),
                })
            })
            .or_else(|status| DECODED[status as usize](status))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn corpus_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..//corpus")
    }

    /// The stem of every `.bin` fixture in the corpus, once the directory
    /// reads whole.
    fn fixture_names() -> Vec<String> {
        let names = fs::read_dir(corpus_root())
            .and_then(|dir| {
                dir.map(|entry| entry.map(|entry| entry.file_name().to_string_lossy().into_owned()))
                    .collect::<io::Result<Vec<_>>>()
            })
            .map_err(|error| error.to_string());
        assert_eq!(names.as_ref().map(drop), Ok(()));
        names
            .iter()
            .flatten()
            .flat_map(|name| name.strip_suffix(".bin"))
            .map(str::to_owned)
            .collect()
    }

    /// A corpus file's bytes, once it reads.
    fn read(path: &Path) -> Vec<u8> {
        let bytes = fs::read(path).map_err(|error| error.to_string());
        assert_eq!(bytes.as_ref().map(drop), Ok(()), "{}", path.display());
        bytes.into_iter().flatten().collect()
    }

    #[test]
    fn indexed_surface_handle_requires_host_descriptor_array() {
        let result = surface_source(HandleSource::ConstantBufferArray {
            bank: 2,
            handle: 16,
        })
        .map_err(|error| error.to_string());
        assert_eq!(result, Err(AdapterError::DynamicSurfaceHandle.to_string()));
    }

    #[test]
    fn surface_alias_adapter_returns_metadata_and_word_only_api_rejects_it() {
        // SUST: envytools gm107.c:1910; raw U32 vs formatted P view.
        // Bundle/header: Mesa sm50.rs:3418-3428, sph.rs:134,169.
        let mut data = vec![0; shadowbox::TargetGeneration::Sm5.shader_header_size()];
        data[1] = 5 << 2;
        let surface =
            (0xeb20u64 << 48) | (1 << 51) | (9 << 36) | (3 << 33) | (7 << 16) | (2 << 8) | 8;
        data.extend(
            [
                0,
                surface | (1 << 20),
                surface | (1 << 52) | (4 << 20),
                (0xe300u64 << 48) | (7 << 16) | 0xf,
            ]
            .into_iter()
            .flat_map(u64::to_le_bytes),
        );
        let translator = ShadowboxTranslator::new();
        let state = shadowbox::interface::GraphicsState::default();
        let bound = translator
            .translate_with_bindings(FromHeader, &data, &state)
            .map(|(output, stage)| {
                let descriptors = &output.surface_descriptors;
                (
                    stage,
                    descriptors.len(),
                    descriptors[0].binding,
                    descriptors[1].binding,
                    descriptors[1].image_format,
                )
            })
            .map_err(|error| error.to_string());
        assert_eq!(bound, Ok((headers::Stage::Fragment, 2, 9, 8193, 33)));
        assert_eq!(
            translator
                .translate_with_stage(FromHeader, &data)
                .map_err(|error| error.to_string()),
            Err(AdapterError::SurfaceAliases.to_string())
        );
    }

    #[test]
    fn state_aware_adapter_forwards_geometry_stream_routing() {
        use shadowbox::interface::{
            GeometryInput, GeometryStreamLocations, GeometryStreams, GraphicsState,
        };
        // Mesa cla097sph.h:37-42,66-73: geometry, one invocation,
        // point output, two maximum emitted vertices.
        let mut data = vec![0; shadowbox::TargetGeneration::Sm5.shader_header_size()];
        data[1] = 4 << 2;
        data[11] = 1;
        data[15] = 1;
        data[16] = 2;
        // Mesa sm50.rs:3263-3284: immediate streams with RZ handle/result.
        let out = |stream: u64| {
            (0xf6e0u64 << 48) | 255 | (255 << 8) | (7 << 16) | (stream << 20) | (3 << 39)
        };
        // Bundle and EXIT: Mesa sm50.rs:3418-3428,3099-3111.
        data.extend(
            [0, out(0), out(1), (0xe300u64 << 48) | (7 << 16) | 15]
                .into_iter()
                .flat_map(u64::to_le_bytes),
        );
        let state = GraphicsState {
            geometry_input: Some(GeometryInput::Points),
            geometry_streams: Some(GeometryStreams {
                builtin_stream: 0,
                locations: [
                    Some(GeometryStreamLocations {
                        generic_base: 0,
                        position: None,
                        primitive_id: None,
                    }),
                    Some(GeometryStreamLocations {
                        generic_base: 32,
                        position: Some(64),
                        primitive_id: Some(65),
                    }),
                    None,
                    None,
                ],
            }),
            ..Default::default()
        };
        let translator = ShadowboxTranslator::new();
        let mut missing = state;
        missing.geometry_streams = None;
        assert_eq!(
            translator
                .translate_with_stage_and_graphics_state(FromHeader, &data, &missing)
                .map(drop)
                .map_err(drop),
            Err(())
        );
        let direct = Result::from(shadowbox::translate_header_prefixed_with_graphics_state(
            shadowbox::QuietRun,
            &data,
            shadowbox::TargetGeneration::Sm5,
            &state,
        ))
        .map(|output| (output.spirv, headers::Stage::Geometry))
        .map_err(|error| error.to_string());
        assert_eq!(direct.as_ref().map(drop), Ok(()));
        assert_eq!(
            translator
                .translate_with_stage_and_graphics_state(FromHeader, &data, &state)
                .map_err(|error| error.to_string()),
            direct
        );
    }

    #[test]
    fn state_aware_adapter_forwards_the_linked_patch_layout() {
        use shadowbox::interface::{GraphicsState, PatchIo, PatchLayout};
        // Mesa cla097sph.h:37-42,66: TCS and three output control points.
        let mut data = vec![0; shadowbox::TargetGeneration::Sm5.shader_header_size()];
        data[1] = 2 << 2;
        data[11] = 3;
        // AST.P byte 0x18: Mesa sm50.rs:2886-2896; RZ source/offset/vertex.
        // Ryujinx AttributeMap.cs:92 places generic patch slot 0 at 0x18.
        let ast = (0xeff0_u64 << 48)
            | 255
            | (255 << 8)
            | (7 << 16)
            | (0x18 << 20)
            | (1 << 31)
            | (1 << 32)
            | (255 << 39);
        // Bundle, EXIT and NOP: Mesa sm50.rs:3418-3428,3099-3111,3178-3184.
        data.extend(
            [
                0,
                ast,
                (0xe300_u64 << 48) | (7 << 16),
                (0x50b0_u64 << 48) | (7 << 16),
            ]
            .into_iter()
            .flat_map(u64::to_le_bytes),
        );
        let mut locations = [None; 31];
        locations[0] = Some(1);
        let state = GraphicsState {
            patch_io: Some(PatchIo {
                layout: PatchLayout::Ryujinx,
                locations,
            }),
            ..Default::default()
        };
        let translator = ShadowboxTranslator::new();
        assert_eq!(
            translator
                .translate_with_stage(FromHeader, &data)
                .map(drop)
                .map_err(drop),
            Err(())
        );
        let direct = Result::from(shadowbox::translate_header_prefixed_with_graphics_state(
            shadowbox::QuietRun,
            &data,
            shadowbox::TargetGeneration::Sm5,
            &state,
        ))
        .map(|output| (output.spirv, headers::Stage::TessellationControl))
        .map_err(|error| error.to_string());
        assert_eq!(direct.as_ref().map(drop), Ok(()));
        assert_eq!(
            translator
                .translate_with_stage_and_graphics_state(FromHeader, &data, &state)
                .map_err(|error| error.to_string()),
            direct
        );
    }

    #[test]
    fn successful_corpus_fixtures_match_library() {
        let root = corpus_root();
        let translator: &dyn ShaderTranslator = &ShadowboxTranslator::new();
        for name in fixture_names() {
            let header = read(&root.join(format!("{name}.hdr")));
            let mut code = header.clone();
            code.truncate(shadowbox::TargetGeneration::Sm5.shader_header_size());
            code.extend(read(&root.join(format!("{name}.bin"))));
            let result = translator.translate(STAGE_FROM_HEADER, &code);
            let (_, shader_code) =
                code.split_at(shadowbox::TargetGeneration::Sm5.shader_header_size());
            let direct = Result::from(shadowbox::translate(
                shadowbox::QuietRun,
                &shadowbox::interface::ShaderInput {
                    code: shader_code,
                    program_record: &header,
                },
            ));
            assert_eq!(
                result,
                direct
                    .map(|library| library.spirv)
                    .map_err(|library| library.to_string()),
                "{name}"
            );
        }
    }

    #[test]
    fn unknown_stage_is_derived_from_every_corpus_header() {
        use headers::Stage::{Compute, Fragment, Vertex};
        let root = corpus_root();
        for name in fixture_names() {
            let mut code = read(&root.join(format!("{name}.hdr")));
            code.truncate(shadowbox::TargetGeneration::Sm5.shader_header_size());
            code.extend(read(&root.join(format!("{name}.bin"))));
            let derived = shadowbox::parse_header_prefixed_header_first(
                &code,
                TargetGeneration::Sm5,
                GraphicsState::default(),
            )
            .map(|program| program.stage())
            .map_err(|error| format!("{error:?}"));
            let vertex = name.starts_with("vertex_")
                || name == "gamelike_skinning"
                || name == "gamelike_varyings";
            let compute = name.starts_with("compute_");
            let expected =
                [[Fragment, Compute], [Vertex, Vertex]][usize::from(vertex)][usize::from(compute)];
            assert_eq!(derived, Ok(expected), "{name}");
        }
    }

    #[test]
    fn unknown_stage_rejects_invalid_header_type() {
        let mut code = vec![0; shadowbox::TargetGeneration::Sm5.shader_header_size()];
        code[1] = 0x18;
        assert_eq!(
            ShadowboxTranslator::new()
                .translate_with_stage(FromHeader, &code)
                .map_err(|error| error.to_string()),
            Err("invalid SM 5.x shader header: ReservedStage(6)".into())
        );
    }

    #[test]
    fn unsupported_instruction_names_first_instruction() {
        let mut code = vec![0; shadowbox::TargetGeneration::Sm5.shader_header_size()];
        code.extend(words_bytes(&[0, 0, 0xdead_beef, 0x1234_5678]));
        let error = ShadowboxTranslator::new().translate(0, &code);
        assert_eq!(
            error
                .as_ref()
                .map(drop)
                .map_err(|error| error.contains("UNKNOWN 0x12345678deadbeef")),
            Err(true),
            "{error:?}"
        );
    }

    #[test]
    fn raw_stage_tokens_translate_as_their_stage() {
        use headers::Stage;
        let translator = ShadowboxTranslator::new();
        let mut code = vec![0; shadowbox::TargetGeneration::Sm5.shader_header_size()];
        code.extend(words_bytes(&[0, 0]));
        let words = |result: Words| result.map(|(words, _)| words).map_err(String::from);
        let header = words(translator.translate_with_stage(FromHeader, &code));
        assert_eq!(translator.translate(SHADER_STAGE_UNKNOWN, &code), header);
        assert_eq!(translator.translate(STAGE_FROM_HEADER, &code), header);
        assert_eq!(
            translator.translate(4, &code),
            words(translator.translate_with_stage(Stage::Fragment, &code))
        );
        assert_eq!(
            translator.translate(5, &code),
            words(translator.translate_with_stage(Stage::Compute, &code))
        );
        for token in [1, 2, 3, 6, 254, 256, u32::MAX] {
            let unknown = format!("unknown shader stage token {token}");
            assert_eq!(translator.translate(token, &code), Err(unknown));
        }
    }

    fn words_bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    #[test]
    fn cache_key_is_none_for_incomplete_program_before_its_cb1_is_read() {
        let mut legacy = vec![0; 80];
        // Stage field 10..14 = 4 is geometry (Mesa cla097sph.h:37-42).
        legacy[1] = 4 << 2;
        legacy.extend(0_u64.to_le_bytes());
        legacy.extend(0xe300_0000_0007_000f_u64.to_le_bytes());
        let record = shadowbox::encode_header_prefixed_with_constant_buffer1(
            &legacy,
            TargetGeneration::Sm5,
            &[1, 2, 3],
        )
        .map_err(|error| error.to_string());
        let translator = ShadowboxTranslator::new();
        let mut state = TranslationContext::default();
        let key = |state: &TranslationContext| {
            record
                .as_ref()
                .map(|record| translator.translation_cache_key(0, record, state))
        };
        assert_eq!(key(&state), Ok(Ok(None)));
        state.geometry_input = Some(4);
        assert_eq!(
            key(&state),
            Ok(Err(
                "invalid SPIR-V module: cb1 snapshot contains a partial word".into()
            ))
        );
    }
}
