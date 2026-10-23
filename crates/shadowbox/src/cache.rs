//! Project-owned cache schema and keys: docs/reference-sources.md
use crate::program::Code;
use crate::{
    headers::Stage,
    interface::{CacheStop, GraphicsState, ShaderInput, ShaderOutput, Stopped, Translation},
    program::{CacheProgram, CompleteProgram, HeaderFirst, ParsedProgram, Program},
    TargetGeneration, TranslationOptions,
};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, HashMap},
    fs::{self, OpenOptions},
    io::{self, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

/// Automatically fingerprints the complete translator source tree and dependencies.
pub const TRANSLATOR_VERSION: &str = env!("SHADOWBOX_TRANSLATOR_VERSION");
pub const CACHE_FORMAT_VERSION: u32 = 8;
const MAGIC: &[u8; 8] = b"SBOXC008";

// Project schema tags, not GPU fields: docs/reference-sources.md.
fn identity_options(generation: TargetGeneration, options: TranslationOptions) -> [u8; 3] {
    // Exhaustive destructuring makes adding an output option require a schema decision.
    let TranslationOptions {
        global_delta,
        float64,
    } = options;
    [
        // Schema tags are the declaration order: Sm5 0, Sm6 1, Sm7_5Plus 2.
        generation as u8,
        global_delta as u8,
        float64 as u8,
    ]
}

fn identity_for_options(options: [u8; 3]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"shadowbox-translator-identity-v1");
    hash.update(TRANSLATOR_VERSION.as_bytes());
    hash.update(CACHE_FORMAT_VERSION.to_le_bytes());
    hash.update(options);
    format!("{:x}", hash.finalize())
}

/// Build semantics, cache format, target generation and every output option.
/// Graphics state and shader inputs additionally participate in translation_cache_key.
pub fn translator_identity(generation: TargetGeneration, options: TranslationOptions) -> String {
    identity_for_options(identity_options(generation, options))
}

pub struct CacheConfig {
    pub memory: Keep,
    pub disk: Disk,
}

/// What a cache keeps in memory: every output, or nothing. Each is
/// constructed with its own remembering path.
#[derive(Clone, Copy)]
pub struct Keep(Remember);

impl Keep {
    /// Keep every translated output in memory.
    pub const fn outputs() -> Self {
        Self(keep_output)
    }
    /// Keep nothing in memory; only the disk, if any, holds outputs.
    pub const fn nothing() -> Self {
        Self(drop_output)
    }
}

/// Where a cache keeps outputs past memory: in a directory, or nowhere.
/// Each is constructed with its own read and publish paths.
pub struct Disk {
    dir: PathBuf,
    read: Read,
    publish: Publish,
}

type Read = fn(&Path, &str, &str) -> Result<ShaderOutput, Miss>;

impl Disk {
    /// Outputs are kept in `dir` as well as memory.
    pub fn at(dir: impl Into<PathBuf>) -> Self {
        Self {
            dir: dir.into(),
            read: read_stored,
            publish: publish_output,
        }
    }

    /// Outputs are kept in memory only.
    pub fn nowhere() -> Self {
        Self {
            dir: PathBuf::new(),
            read: unstored,
            publish: unpublished,
        }
    }

    fn read(&self, key: &str, identity: &str) -> Result<ShaderOutput, Miss> {
        (self.read)(&self.dir, key, identity)
    }

    fn publish(
        &self,
        key: &str,
        context: CacheContext<'_>,
        output: &ShaderOutput,
    ) -> io::Result<()> {
        (self.publish)(&self.dir, key, context, output)
    }
}

impl Default for Disk {
    fn default() -> Self {
        Self::nowhere()
    }
}

fn read_stored(dir: &Path, key: &str, identity: &str) -> Result<ShaderOutput, Miss> {
    read_current(dir, key, identity).map_err(|_| Miss)
}
fn unstored(_: &Path, _: &str, _: &str) -> Result<ShaderOutput, Miss> {
    Err(Miss)
}
/// An output with surface binding metadata has no disk form: it is only
/// remembered in memory.
fn publish_output(
    dir: &Path,
    key: &str,
    context: CacheContext<'_>,
    output: &ShaderOutput,
) -> io::Result<()> {
    PUBLISH[usize::from(output.surface_descriptors.is_empty())](dir, key, context, output)
}
#[derive(Default, Debug, Clone, Copy)]
pub struct CacheStats {
    pub memory_hits: usize,
    pub disk_hits: usize,
    pub misses: usize,
}

/// All external inputs to module generation. Stage is derived from the header.
#[derive(Clone, Copy, Debug)]
pub struct CacheContext<'a> {
    pub generation: TargetGeneration,
    pub options: TranslationOptions,
    pub graphics_state: GraphicsState<'a>,
}
impl Default for CacheContext<'_> {
    fn default() -> Self {
        Self {
            generation: TargetGeneration::Sm5,
            options: TranslationOptions::default(),
            graphics_state: GraphicsState::default(),
        }
    }
}

type Needed = &'static [&'static str];

/// The graphics state keys each stage needs, in `Stage` order.
const NEEDED: [fn(&GraphicsState) -> Needed; 6] = [
    needs_nothing,
    needs_nothing,
    needs_nothing,
    geometry_needs,
    needs_nothing,
    evaluation_needs,
];
/// Indexed by how many geometry inputs the driver gave: none or one.
const GEOMETRY_NEEDS: [Needed; 2] = [&["geometry_input"], &[]];
/// Indexed by how many tessellation states the driver gave: none or one.
const EVALUATION_NEEDS: [Needed; 2] = [
    &[
        "tessellation.domain",
        "tessellation.spacing",
        "tessellation.clockwise",
    ],
    &[],
];

fn needs_nothing(_: &GraphicsState) -> Needed {
    &[]
}
fn geometry_needs(state: &GraphicsState) -> Needed {
    GEOMETRY_NEEDS[state.geometry_input.iter().len()]
}
fn evaluation_needs(state: &GraphicsState) -> Needed {
    EVALUATION_NEEDS[state.tessellation.iter().len()]
}

/// These keys are required by the existing emitter, never guessed during scans.
/// Ryujinx CodeGen/Spirv/SpirvGenerator.cs:194-248; Shadowbox emitter/stages.rs:68-116.
pub(crate) fn missing_state_keys(stage: Stage, state: &GraphicsState) -> Needed {
    NEEDED[stage as usize](state)
}

/// Parse for a stage-keyed entry point: the header and the caller's stage
/// are checked before the cb1 snapshot. Default driver state.
pub(crate) fn parse_for_stage<'a>(
    input: &ShaderInput<'a>,
    stage: Stage,
) -> Result<Program<'a>, Stopped> {
    let program = HeaderFirst::parse(input)?;
    STAGE_AGREEMENT[usize::from(program.stage() == stage)](program)
}
type StageAgreement = for<'a> fn(HeaderFirst<'a>) -> Result<Program<'a>, Stopped>;
/// Indexed by whether the header names the caller's stage.
const STAGE_AGREEMENT: [StageAgreement; 2] = [disagreeing_stage, agreeing_stage];

fn disagreeing_stage<'a>(_: HeaderFirst<'a>) -> Result<Program<'a>, Stopped> {
    Err(Stopped::Cache(CacheStop::StageDisagrees))
}
fn agreeing_stage<'a>(program: HeaderFirst<'a>) -> Result<Program<'a>, Stopped> {
    Ok(program.finish(GraphicsState::default())?.classify())
}

/// SHA-256 of a length-delimited, versioned translation identity.
/// The numeric tags below are cache schema choices, not Maxwell field encodings.
/// [`crate::Program::identity`] names any program, complete or not, with it
/// for reports; [`crate::CompleteProgram::cache_key`] is the only key the
/// cache stores and looks up.
pub(crate) fn program_cache_key(
    program: &ParsedProgram<'_>,
    generation: TargetGeneration,
    options: TranslationOptions,
) -> String {
    let state = program.graphics_state();
    let stage = program.stage();
    let mut hasher = Sha256::new();
    hasher.update(translator_identity(generation, options).as_bytes());
    hasher.update(Sha256::digest(program.code.bytes));
    let header = program.canonical_header();
    hasher.update((header.len() as u64).to_le_bytes());
    hasher.update(header);
    hasher.update([stage as u8]);
    STAGE_STATE[stage as usize](&program.code, &state, &mut hasher);
    // Bound TXQ has no dimension field: Ryujinx InstEmitTexture.cs:1116-1119
    // and Spirv/Declarations.cs:176-183 need driver type/depth/MS metadata.
    // Include the complete borrowed declaration list in every stage. Order
    // is conservatively part of identity; this avoids sorting on warm lookups.
    hasher.update((state.texture_bindings.len() as u64).to_le_bytes());
    for texture in state.texture_bindings {
        hasher.update(texture.binding.to_le_bytes());
        hasher.update([
            texture.dimension as u8,
            u8::from(texture.depth_compare),
            u8::from(texture.multisample),
        ]);
    }
    hasher.update((state.texture_arrays.len() as u64).to_le_bytes());
    for array in state.texture_arrays {
        hasher.update([array.binding]);
        hasher.update(array.handle.to_le_bytes());
        hasher.update(array.length.to_le_bytes());
    }
    // Ryujinx GlobalToStorage.cs:409-445 selects storage bindings by cbuf base.
    hasher.update((state.storage_buffers.len() as u64).to_le_bytes());
    for buffer in state.storage_buffers {
        hasher.update([buffer.bank]);
        hasher.update(buffer.offset.to_le_bytes());
        hasher.update(buffer.binding.to_le_bytes());
    }
    // BRX edges depend on the immutable snapshot (Ryujinx Decoder.cs:543-544).
    present(
        &mut hasher,
        state.constant_buffer1.iter(),
        |hasher, words| {
            hasher.update((words.len() as u64).to_le_bytes());
            words
                .iter()
                .for_each(|word| hasher.update(word.to_le_bytes()));
        },
    );
    format!("{:x}", hasher.finalize())
}

/// Identity of a zero-or-one value: its count, then the value's own identity.
fn present<'v, T: 'v>(
    hasher: &mut Sha256,
    values: impl ExactSizeIterator<Item = &'v T>,
    each: impl Fn(&mut Sha256, &'v T),
) {
    hasher.update([values.len() as u8]);
    values.for_each(|value| each(hasher, value));
}

type StageState = fn(&Code, &GraphicsState, &mut Sha256);

/// The graphics state each stage's identity consumes, in `Stage` order.
const STAGE_STATE: [StageState; 6] = [
    vertex_state,
    fragment_state,
    no_state,
    geometry_state,
    control_state,
    evaluation_state,
];

fn no_state(_: &Code, _: &GraphicsState, _: &mut Sha256) {}

fn fragment_state(code: &Code, state: &GraphicsState, hasher: &mut Sha256) {
    if code.instruction_slots().any(|(address, word)| {
        matches!(
            crate::decoder::decode_word(word, address).kind,
            crate::ir::InstructionKind::LegacyVaryingLoad(crate::ir::ops::LegacyVaryingLoad {
                interpolation: crate::ir::LegacyInterpolation::State,
                ..
            })
        )
    }) {
        present(
            hasher,
            state.legacy_color_flat_shading.iter(),
            |hasher, flat| hasher.update([u8::from(*flat)]),
        );
    }
}

fn vertex_state(code: &Code, state: &GraphicsState, hasher: &mut Sha256) {
    // Ryujinx InstEmitMove.cs:94-114 also consumes topology for vertex
    // InvocationInfo. Scan instruction slots without allocating another IR.
    // Mesa sm50.rs:3220-3224 encodes S2R/index20..27; :3418-3428 bundles.
    // Every slot counts, reached or not, so the identity is conservative;
    // the decoder names what each slot is.
    if code.instruction_slots().any(|(address, word)| {
        crate::decoder::decode_word(word, address)
            .kind
            .reads_invocation_info()
    }) {
        input_state(code, state, hasher);
    }
}

fn input_state(_: &Code, state: &GraphicsState, hasher: &mut Sha256) {
    // Absent 0; Points 1 through TrianglesAdjacency 5, in declaration order.
    hasher.update([state
        .geometry_input
        .iter()
        .fold(0, |_, input| 1 + *input as u8)]);
}

fn geometry_state(code: &Code, state: &GraphicsState, hasher: &mut Sha256) {
    input_state(code, state, hasher);
    // Passthrough consumes the next-stage mask (Ryujinx
    // Translation/AttributeUsage.cs:91-97, EmitterContext.cs:360-426).
    present(hasher, state.geometry_next_inputs.iter(), |hasher, mask| {
        hasher.update(mask.to_le_bytes())
    });
    // Stream layouts change output declarations and each OUT publication.
    // Encoding: Mesa nak/sm50.rs:3257-3284. Routing constraints and Vulkan
    // citations: emitter/stages.rs::route_streams.
    present(hasher, state.geometry_streams.iter(), |hasher, streams| {
        hasher.update([streams.builtin_stream]);
        streams.locations.iter().for_each(|layout| {
            present(hasher, layout.iter(), |hasher, layout| {
                hasher.update(layout.generic_base.to_le_bytes());
                [layout.position, layout.primitive_id]
                    .iter()
                    .for_each(|location| {
                        present(hasher, location.iter(), |hasher, location| {
                            hasher.update(location.to_le_bytes())
                        })
                    });
            })
        });
    });
}

// Linked patch locations affect both TCS and TES declarations. Ryujinx
// AttributeUsage.cs:65-87 and AttributeMap.cs:92,109-121; Mesa
// nak_private.h:39 / nak_nir.c:422-423 supplies the alternative base.
fn control_state(_: &Code, state: &GraphicsState, hasher: &mut Sha256) {
    present(hasher, state.patch_io.iter(), |hasher, patch| {
        // Ryujinx 0, Mesa 1.
        hasher.update([patch.layout as u8]);
        patch.locations.iter().for_each(|location| {
            present(hasher, location.iter(), |hasher, location| {
                hasher.update(location.to_le_bytes())
            })
        });
    });
}

fn evaluation_state(code: &Code, state: &GraphicsState, hasher: &mut Sha256) {
    present(hasher, state.tessellation.iter(), |hasher, t| {
        hasher.update([
            // Isolines 0, Triangles 1, Quads 2.
            t.domain as u8,
            // Equal 0, FractionalEven 1, FractionalOdd 2.
            t.spacing as u8,
            u8::from(t.clockwise),
        ])
    });
    control_state(code, state, hasher);
}

type Memory = HashMap<String, ShaderOutput>;
type Remember = fn(&mut Memory, &str, &ShaderOutput);

pub struct Cache {
    config: CacheConfig,
    /// Keyed by cache key, which hashes the translator identity first.
    memory: Memory,
    /// Chosen once from the configuration: keep outputs in memory, or not.
    remember: Remember,
    pub stats: CacheStats,
}

/// A key one tier of the cache does not hold.
struct Miss;

fn keep_output(memory: &mut Memory, key: &str, output: &ShaderOutput) {
    memory.insert(key.to_owned(), output.clone());
}
fn drop_output(_: &mut Memory, _: &str, _: &ShaderOutput) {}

type Publish = fn(&Path, &str, CacheContext<'_>, &ShaderOutput) -> io::Result<()>;
/// Indexed by whether the output has no surface binding metadata, which has
/// no disk form: such an output is only remembered in memory.
const PUBLISH: [Publish; 2] = [unpublished, write_disk];

fn unpublished(_: &Path, _: &str, _: CacheContext<'_>, _: &ShaderOutput) -> io::Result<()> {
    Ok(())
}

impl Cache {
    pub fn new(config: CacheConfig) -> Self {
        Self {
            remember: config.memory.0,
            config,
            memory: HashMap::new(),
            stats: CacheStats::default(),
        }
    }

    /// Compatibility entry point for default options and SM5. The callback must
    /// implement the same semantics as translate(input). Overrides need a context.
    pub fn translate<F>(
        &mut self,
        input: &ShaderInput<'_>,
        stage: Stage,
        translate: F,
    ) -> Translation<ShaderOutput>
    where
        F: FnOnce() -> Translation<ShaderOutput>,
    {
        Translation::from(parse_for_stage(input, stage).and_then(|program| {
            self.translate_program(&program, CacheContext::default(), || translate().into())
        }))
    }

    /// A hit invokes no callback, decoder, emitter, or optimizer.
    pub fn translate_with_context<F>(
        &mut self,
        input: &ShaderInput<'_>,
        context: CacheContext,
        translate: F,
    ) -> Translation<ShaderOutput>
    where
        F: FnOnce() -> Translation<ShaderOutput>,
    {
        Translation::from(
            ParsedProgram::parse(input, context.graphics_state)
                .map_err(Stopped::from)
                .and_then(|program| {
                    self.translate_program(&program.classify(), context, || translate().into())
                }),
        )
    }

    /// A program missing graphics state is translated, so its translation
    /// errors come first, and then refused: it has no cache key, so it is
    /// never looked up or stored.
    pub(crate) fn translate_program<F>(
        &mut self,
        program: &Program<'_>,
        context: CacheContext,
        translate: F,
    ) -> Result<ShaderOutput, Stopped>
    where
        F: FnOnce() -> Result<ShaderOutput, Stopped>,
    {
        match program {
            Program::Complete(program) => self.complete(program, context, translate),
            Program::Incomplete(_) => self.awaiting_state(translate),
        }
    }

    fn awaiting_state<F>(&mut self, translate: F) -> Result<ShaderOutput, Stopped>
    where
        F: FnOnce() -> Result<ShaderOutput, Stopped>,
    {
        self.stats.misses += 1;
        translate().and(Err(Stopped::Cache(CacheStop::AwaitingState)))
    }

    /// Memory, then disk, then the callback, which publishes to both.
    fn complete<F>(
        &mut self,
        program: &CompleteProgram<'_>,
        context: CacheContext,
        translate: F,
    ) -> Result<ShaderOutput, Stopped>
    where
        F: FnOnce() -> Result<ShaderOutput, Stopped>,
    {
        let key = program.cache_key(context.generation, context.options);
        let identity = translator_identity(context.generation, context.options);
        self.remembered(&key)
            .or_else(|Miss| self.stored(&key, &identity))
            .or_else(|Miss| self.translated(&key, context, translate))
    }

    fn remembered(&mut self, key: &str) -> Result<ShaderOutput, Miss> {
        let remembered = self.memory.get(key).cloned().ok_or(Miss);
        remembered.inspect(|_| self.stats.memory_hits += 1)
    }

    fn stored(&mut self, key: &str, identity: &str) -> Result<ShaderOutput, Miss> {
        self.config.disk.read(key, identity).inspect(|output| {
            self.stats.disk_hits += 1;
            (self.remember)(&mut self.memory, key, output);
        })
    }

    fn translated<F>(
        &mut self,
        key: &str,
        context: CacheContext,
        translate: F,
    ) -> Result<ShaderOutput, Stopped>
    where
        F: FnOnce() -> Result<ShaderOutput, Stopped>,
    {
        self.stats.misses += 1;
        translate().and_then(|output| {
            self.config
                .disk
                .publish(key, context, &output)
                .map_err(|e| Stopped::Cache(CacheStop::Publish(e)))
                .map(|()| {
                    (self.remember)(&mut self.memory, key, &output);
                    output
                })
        })
    }
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}
fn status_error(status: CacheEntryStatus) -> io::Error {
    invalid(status.as_str())
}
/// Holds, or refuses with `message`.
fn insist(holds: bool, message: &'static str) -> io::Result<()> {
    [refuse, accept][usize::from(holds)](message)
}
fn refuse(message: &'static str) -> io::Result<()> {
    Err(invalid(message))
}
fn accept(_: &'static str) -> io::Result<()> {
    Ok(())
}

/// Lowercase hexadecimal digits, by byte.
const HEX: [bool; 256] = {
    let mut hex = [false; 256];
    let mut digit = 0;
    while digit < 16 {
        hex[b"0123456789abcdef"[digit] as usize] = true;
        digit += 1;
    }
    hex
};
fn valid_key(key: &str) -> bool {
    key.len() == 64
        && key
            .bytes()
            .fold(true, |hex, byte| hex & HEX[usize::from(byte)])
}

type Encode = fn(&str, CacheContext<'_>, &ShaderOutput) -> io::Result<Vec<u8>>;
/// Indexed by whether the output has no surface binding metadata.
const ENCODE: [Encode; 2] = [unencodable, encode_retained];

// Envelope offsets are project choices: docs/reference-sources.md.
/// Encode a v8 record with the current translator identity and exact requirements.
/// Reject outputs whose binding metadata the current format cannot retain.
/// Provenance: docs/reference-sources.md
pub fn encode_record(
    key: &str,
    context: CacheContext,
    output: &ShaderOutput,
) -> io::Result<Vec<u8>> {
    ENCODE[usize::from(output.surface_descriptors.is_empty())](key, context, output)
}

fn unencodable(_: &str, _: CacheContext<'_>, _: &ShaderOutput) -> io::Result<Vec<u8>> {
    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        "disk cache cannot retain typed surface binding metadata",
    ))
}

fn encode_retained(
    key: &str,
    context: CacheContext<'_>,
    output: &ShaderOutput,
) -> io::Result<Vec<u8>> {
    let mut payload = vec![
        u8::from(output.requires_subgroup_size_32)
            | (u8::from(output.requires_maximal_reconvergence) << 1)
            | (u8::from(output.requires_demote_to_helper_invocation) << 2),
    ];
    payload.extend(translator_identity(context.generation, context.options).as_bytes());
    payload.extend(identity_options(context.generation, context.options));
    payload.extend(output.spirv.iter().flat_map(|w| w.to_le_bytes()));
    let mut bytes = MAGIC.to_vec();
    bytes.extend(key.as_bytes());
    bytes.extend(Sha256::digest(&payload));
    bytes.extend(payload);
    decode_record(&bytes, key)
        .map_err(status_error)
        .map(|_| bytes)
}

/// Verification categories: docs/reference-sources.md.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CacheEntryStatus {
    Valid,
    Stale,
    Mismatched,
    Corrupt,
}
impl CacheEntryStatus {
    pub fn as_str(self) -> &'static str {
        ["valid", "stale", "mismatched", "corrupt"][self as usize]
    }
}
use CacheEntryStatus::{Corrupt, Mismatched, Stale};

/// Holds, or is `status`.
fn require(holds: bool, status: CacheEntryStatus) -> Result<(), CacheEntryStatus> {
    [Err(status), Ok(())][usize::from(holds)]
}

/// Reads one cache entry's bytes.
type ReadEntry = fn(&Path) -> io::Result<Vec<u8>>;
/// Indexed by whether the entry is a regular file.
const ENTRY: [ReadEntry; 2] = [irregular_entry, regular_entry];

fn irregular_entry(_: &Path) -> io::Result<Vec<u8>> {
    Err(invalid("cache entry is not a regular file"))
}
fn regular_entry(path: &Path) -> io::Result<Vec<u8>> {
    fs::read(path)
}

fn read_entry_bytes(dir: &Path, key: &str) -> io::Result<Vec<u8>> {
    let path = dir.join(format!("{key}.sbc"));
    fs::symlink_metadata(&path)
        .and_then(|metadata| ENTRY[usize::from(metadata.file_type().is_file())](&path))
}

/// A decoded record and the translator identity it recorded.
type Decoded<'b> = Result<(&'b str, ShaderOutput), CacheEntryStatus>;

/// Record layouts, indexed by whether the magic names the historical v7 one.
const LAYOUT: [for<'b> fn(&'b [u8]) -> Decoded<'b>; 2] = [current_record, legacy_record];

fn decode<'b>(bytes: &'b [u8], key: &str) -> Decoded<'b> {
    require(valid_key(key) && bytes.len() >= 125, Corrupt)
        .and_then(|()| require(bytes[8..72] == *key.as_bytes(), Corrupt))
        .and_then(|()| LAYOUT[usize::from(bytes[..8] == *b"SBOXC007")](bytes))
}

/// Historical v7 layout: docs/reference-sources.md. A
/// well-formed one is stale.
fn legacy_record(bytes: &[u8]) -> Decoded<'_> {
    words(bytes, 105).and(Err(Stale))
}

fn current_record(bytes: &[u8]) -> Decoded<'_> {
    require(bytes[..8] == *MAGIC, Corrupt)
        .and_then(|()| words(bytes, 172))
        .and_then(|spirv| recorded(bytes, spirv))
}

/// The SPIR-V words from `offset`, behind the flags byte and payload digest.
fn words(bytes: &[u8], offset: usize) -> Result<Vec<u32>, CacheEntryStatus> {
    require(
        bytes.len() >= offset + 20
            && bytes[104] <= 7
            && (bytes.len() - offset).is_multiple_of(4)
            && Sha256::digest(&bytes[104..])[..] == bytes[72..104],
        Corrupt,
    )
    .map(|()| {
        bytes[offset..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect::<Vec<_>>()
    })
    .and_then(|words| require(words[0] == crate::spirv::MAGIC, Corrupt).map(|()| words))
}

/// The recorded identity and options, checked against each other.
fn recorded(bytes: &[u8], spirv: Vec<u32>) -> Decoded<'_> {
    let options = [bytes[169], bytes[170], bytes[171]];
    std::str::from_utf8(&bytes[105..169])
        .map_err(|_| Corrupt)
        .and_then(|identity| {
            require(
                valid_key(identity) && options[0] <= 2 && options[1] <= 1 && options[2] <= 1,
                Corrupt,
            )
            .map(|()| identity)
        })
        .and_then(|identity| {
            require(identity == identity_for_options(options), Mismatched).map(|()| identity)
        })
        .map(|identity| {
            (
                identity,
                ShaderOutput {
                    spirv,
                    surface_descriptors: Vec::new(),
                    requires_subgroup_size_32: bytes[104] & 1 != 0,
                    requires_maximal_reconvergence: bytes[104] & 2 != 0,
                    requires_demote_to_helper_invocation: bytes[104] & 4 != 0,
                },
            )
        })
}

/// Decode a record, validating its recorded options and automatic build identity.
/// Stale and mismatched records are clean misses for runtime callers.
/// Provenance: docs/reference-sources.md
pub fn decode_record(bytes: &[u8], key: &str) -> Result<ShaderOutput, CacheEntryStatus> {
    decode(bytes, key).map(|(_, output)| output)
}

/// Decode a record the runtime looks up: it must also record `identity`.
fn decode_current(
    bytes: &[u8],
    key: &str,
    identity: &str,
) -> Result<ShaderOutput, CacheEntryStatus> {
    decode(bytes, key)
        .and_then(|(recorded, output)| require(recorded == identity, Mismatched).map(|()| output))
}

fn read_disk(dir: &Path, key: &str) -> io::Result<ShaderOutput> {
    read_entry_bytes(dir, key).and_then(|bytes| decode_record(&bytes, key).map_err(status_error))
}

fn read_current(dir: &Path, key: &str, identity: &str) -> io::Result<ShaderOutput> {
    read_entry_bytes(dir, key)
        .and_then(|bytes| decode_current(&bytes, key, identity).map_err(status_error))
}

/// The bytes of an entry that decodes.
fn read_valid(dir: &Path, key: &str) -> io::Result<Vec<u8>> {
    read_entry_bytes(dir, key).and_then(|bytes| {
        decode_record(&bytes, key)
            .map_err(status_error)
            .map(|_| bytes)
    })
}

/// Publish an already translated module using exactly the runtime identity.
/// Callers must supply the context used to produce the output.
pub fn emit_translation_cache(
    dir: &Path,
    input: &ShaderInput<'_>,
    context: CacheContext,
    output: &ShaderOutput,
) -> io::Result<String> {
    HeaderFirst::parse(input)
        .and_then(|program| program.classify(context.graphics_state))
        .map_err(io::Error::other)
        .and_then(|program| match program {
            CacheProgram::Complete(program) => {
                Ok(program.cache_key(context.generation, context.options))
            }
            CacheProgram::Incomplete { .. } => Err(invalid(
                "cannot emit a cache entry with missing graphics state",
            )),
        })
        .and_then(|key| write_disk(dir, &key, context, output).map(|()| key))
}

fn write_disk(
    dir: &Path,
    key: &str,
    context: CacheContext,
    output: &ShaderOutput,
) -> io::Result<()> {
    encode_record(key, context, output).and_then(|bytes| {
        fs::create_dir_all(dir).and_then(|()| atomic_write(&dir.join(format!("{key}.sbc")), &bytes))
    })
}
// One envelope publishes SPIR-V and metadata together. create_new + a process
// counter permits concurrent writes of the same key without shared temp files.
fn atomic_write(destination: &Path, bytes: &[u8]) -> io::Result<()> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let temporary = destination.with_extension(format!(
        "tmp-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .and_then(|mut file| {
            file.write_all(bytes)
                .and_then(|()| fs::rename(&temporary, destination))
                .inspect_err(|_| {
                    let _ = fs::remove_file(&temporary);
                })
        })
}

#[derive(Default, Debug)]
pub struct DiskCacheStats {
    pub entries: usize,
    pub bytes: u64,
    pub subgroup_size_32: usize,
    pub invalid: usize,
}
fn keys(dir: &Path) -> io::Result<Vec<String>> {
    fs::read_dir(dir)
        .and_then(|mut entries| {
            entries.try_fold(Vec::new(), |mut keys, entry| {
                entry.map(|entry| {
                    keys.extend(sbc_key(&entry.path()));
                    keys
                })
            })
        })
        .map(|mut keys| {
            keys.sort();
            keys
        })
}
/// The key a path names when its extension is `sbc`: zero or one.
fn sbc_key(path: &Path) -> impl Iterator<Item = String> + '_ {
    let sbc = path
        .extension()
        .into_iter()
        .fold(false, |_, extension| extension == "sbc");
    path.file_stem()
        .into_iter()
        .take(usize::from(sbc))
        .map(|stem| stem.to_string_lossy().into_owned())
}
pub fn disk_cache_stats(dir: &Path) -> io::Result<DiskCacheStats> {
    keys(dir).and_then(|keys| {
        keys.iter()
            .try_fold(DiskCacheStats::default(), |stats, key| {
                tally(dir, key, stats)
            })
    })
}
/// An entry that decodes counts with its file size; any other is invalid.
fn tally(dir: &Path, key: &str, stats: DiskCacheStats) -> io::Result<DiskCacheStats> {
    let (Ok(tallied) | Err(tallied)) = read_disk(dir, key)
        .map(|output| {
            fs::metadata(dir.join(format!("{key}.sbc"))).map(|metadata| DiskCacheStats {
                entries: stats.entries + 1,
                bytes: stats.bytes + metadata.len(),
                subgroup_size_32: stats.subgroup_size_32
                    + usize::from(output.requires_subgroup_size_32),
                invalid: stats.invalid,
            })
        })
        .map_err(|_| {
            Ok(DiskCacheStats {
                invalid: stats.invalid + 1,
                ..stats
            })
        });
    tallied
}

/// Each gathered key's source directory and record digest.
type Gathered = BTreeMap<String, (PathBuf, [u8; 32])>;

/// Validate sources and conflicts before publishing any entries. Old cache
/// versions, corrupt files and different outputs for one key fail the merge.
pub fn merge_disk_caches(destination: &Path, sources: &[PathBuf]) -> io::Result<DiskCacheStats> {
    // Retain identities and source locations, not all SPIR-V payloads in RAM.
    sources
        .iter()
        .try_fold(Gathered::new(), |gathered, source| {
            gather(destination, source, gathered)
        })
        .and_then(|gathered| fs::create_dir_all(destination).map(|()| gathered))
        .and_then(|gathered| {
            gathered
                .into_iter()
                .try_for_each(|(key, (source, digest))| {
                    publish_gathered(destination, &key, &source, digest)
                })
        })
        .and_then(|()| disk_cache_stats(destination))
}

fn gather(destination: &Path, source: &Path, gathered: Gathered) -> io::Result<Gathered> {
    keys(source).and_then(|keys| {
        keys.into_iter().try_fold(gathered, |gathered, key| {
            admit(destination, source, gathered, key)
        })
    })
}

fn admit(
    destination: &Path,
    source: &Path,
    mut gathered: Gathered,
    key: String,
) -> io::Result<Gathered> {
    read_valid(source, &key).and_then(|bytes| {
        let digest: [u8; 32] = Sha256::digest(&bytes).into();
        let kept = gathered
            .entry(key.clone())
            .or_insert_with(|| (source.to_path_buf(), digest))
            .1;
        let present = destination.join(format!("{key}.sbc")).exists();
        insist(kept == digest, "conflicting cache entries")
            .and_then(|()| DESTINATION[usize::from(present)](destination, &key, &bytes))
            .map(|()| gathered)
    })
}

type Destination = fn(&Path, &str, &[u8]) -> io::Result<()>;
/// Indexed by whether the destination holds the key, which must then hold
/// the same bytes.
const DESTINATION: [Destination; 2] = [absent_destination, matching_destination];

fn absent_destination(_: &Path, _: &str, _: &[u8]) -> io::Result<()> {
    Ok(())
}
fn matching_destination(destination: &Path, key: &str, bytes: &[u8]) -> io::Result<()> {
    read_valid(destination, key)
        .and_then(|old| insist(old == bytes, "conflicting destination entry"))
}

fn publish_gathered(
    destination: &Path,
    key: &str,
    source: &Path,
    digest: [u8; 32],
) -> io::Result<()> {
    read_valid(source, key)
        .and_then(|bytes| {
            insist(
                Sha256::digest(&bytes)[..] == digest,
                "source cache changed during merge",
            )
            .map(|()| bytes)
        })
        .and_then(|bytes| atomic_write(&destination.join(format!("{key}.sbc")), &bytes))
}

/// Inspect every .sbc entry without changing the directory. Sorted by filename.
/// Current entries are checked against their recorded options, so valid variants
/// do not become mismatches merely because they use nondefault options.
pub fn verify_disk_cache(dir: &Path) -> io::Result<Vec<(String, CacheEntryStatus)>> {
    keys(dir).map(|keys| {
        keys.into_iter()
            .map(|key| {
                let (Ok(status) | Err(status)) = read_entry_bytes(dir, &key)
                    .map_err(|_| Corrupt)
                    .and_then(|bytes| decode_record(&bytes, &key).map(|_| CacheEntryStatus::Valid));
                (format!("{key}.sbc"), status)
            })
            .collect()
    })
}
