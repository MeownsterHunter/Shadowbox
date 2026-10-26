//! Shadowbox translates public-generation Maxwell and Pascal shader machine code to SPIR-V.
//!
//! This crate is intentionally a project skeleton. The module boundaries document the
//! implementation contract before the translator is written.

use crate::ir::ops;
pub mod decoder;
pub mod emitter;
pub mod headers;
pub mod interface;
pub mod ir;
pub mod recorded_state;
pub mod structure;

mod brx;
pub mod cache;
mod fast_hash;
mod flow;
mod global_memory;
mod guard;
mod optimizer;
mod program;
mod reaching;
mod reconvergence;
mod record;
mod run;
pub mod spirv;
mod stack;
mod stopped;
mod storage_memory;

pub use run::{QuietRun, Run, TracedRun};

pub use record::{
    encode_header_prefixed_with_constant_buffer1, snapshot as constant_buffer1_snapshot,
};

pub use program::{
    CacheProgram, CompleteProgram, Follow, ParsedProgram, Program, ProgramAwaitingState,
    ProgramError, ProgramErrorPath, ProgramPath,
};

pub use cache::{
    disk_cache_stats, emit_translation_cache, merge_disk_caches, translator_identity,
    verify_disk_cache, Cache, CacheConfig, CacheContext, CacheEntryStatus, CacheStats,
    DiskCacheStats, CACHE_FORMAT_VERSION, TRANSLATOR_VERSION,
};

/// SPIR-V specialization constant id for Maxwell S2R SR_YDIRECTION.
pub const SPEC_ID_Y_DIRECTION: u32 = 0x5d00;

/// Byte offset of the flat guest-to-host GPU address delta in push constants.
/// One little-endian u64, present only with global memory and nonzero-delta mode.
pub const PUSH_GLOBAL_DELTA_OFFSET: u32 = 0;
pub const PUSH_GLOBAL_DELTA_SIZE: u32 = 8;

/// Translation contract selected by novena.
#[derive(Debug, Clone, Copy, Default)]
pub struct TranslationOptions {
    pub global_delta: GlobalDelta,
    pub float64: HostFloat64,
}

/// Where global memory's guest-to-host GPU address delta comes from. The
/// discriminant is the cache identity's schema tag.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum GlobalDelta {
    /// A push constant the shader adds to every global address.
    #[default]
    Pushed,
    /// No delta push constant and no add: host and guest GPU addresses are
    /// identical for every allocation used by this shader.
    Zero,
}

/// Whether the host has shaderFloat64. Ryujinx CodeGen/Spirv/
/// SpirvGenerator.cs:58-60 gates Float64 on host support. Without it, FMUL
/// uses the round-to-nearest fallback from Ryujinx Instructions/
/// InstEmitFloatArithmetic.cs:435-480; with it, the encoded directed rounding
/// is kept (Mesa sm50.rs:695-700). The discriminant is the cache identity's
/// schema tag.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum HostFloat64 {
    #[default]
    Unsupported,
    Supported,
}

/// Header size selected by GPU generation. Mesa selects the 20-word GF100
/// header before Turing and the 32-word TU102 header from Turing onward
/// (`src/nouveau/vulkan/nvk_shader.c:605-606`, `nvk_shader.h:29-30`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetGeneration {
    Sm5,
    Sm6,
    Sm7_5Plus,
}

impl TargetGeneration {
    pub const fn shader_header_size(self) -> usize {
        [20 * 4, 20 * 4, 32 * 4][self as usize]
    }
}

/// The future translation entry point exposed to novena.
pub fn translate<R: Run>(
    run: R,
    input: &interface::ShaderInput<'_>,
) -> interface::Translation<interface::ShaderOutput> {
    translate_with_options(run, input, TranslationOptions::default())
}

/// Translate with an explicit GPU-address contract. Options are not cached by
/// translate_cached; use translate_cached_with_context to include them in keys.
pub fn translate_with_options<R: Run>(
    run: R,
    input: &interface::ShaderInput<'_>,
    options: TranslationOptions,
) -> interface::Translation<interface::ShaderOutput> {
    translate_with_options_and_graphics_state(
        run,
        input,
        options,
        &interface::GraphicsState::default(),
    )
}

/// Translate geometry/tessellation with the state absent from the SPH.
pub fn translate_with_graphics_state<R: Run>(
    run: R,
    input: &interface::ShaderInput<'_>,
    state: &interface::GraphicsState,
) -> interface::Translation<interface::ShaderOutput> {
    translate_with_options_and_graphics_state(run, input, TranslationOptions::default(), state)
}

/// Translate with both the GPU-address contract and driver stage state.
/// Use translate_cached_with_context to include both in the runtime cache key.
pub fn translate_with_options_and_graphics_state<R: Run>(
    run: R,
    input: &interface::ShaderInput<'_>,
    options: TranslationOptions,
    state: &interface::GraphicsState,
) -> interface::Translation<interface::ShaderOutput> {
    match ParsedProgram::parse(input, *state) {
        Ok(program) => translate_parsed(run, &program, options),
        Err(malformed) => interface::Translation::Stopped(malformed.into()),
    }
}

/// Translate a program parsed once, for example by [`parse_header_prefixed`].
pub fn translate_parsed<R: Run>(
    run: R,
    program: &ParsedProgram<'_>,
    options: TranslationOptions,
) -> interface::Translation<interface::ShaderOutput> {
    DEFAULT_TRANSLATOR
        .with_borrow_mut(|translator| translator.translate_program(run, program, options))
        .into()
}

thread_local! {
    static DEFAULT_TRANSLATOR: std::cell::RefCell<Translator> =
        std::cell::RefCell::new(Translator::default());
}

/// Scratch storage for consecutive translations. Buffers retain capacity,
/// while every analysis resets its input-dependent facts before use. One
/// instance can process mixed stages and failed translations in any order.
#[derive(Default)]
pub struct Translator {
    words: Vec<u32>,
    decoded: Vec<ir::Instruction>,
    stack: decoder::StackScratch,
    graph: structure::GraphStore,
    emission: emitter::Scratch,
    optimization: optimizer::Scratch,
}

impl Translator {
    pub fn translate<R: Run>(
        &mut self,
        run: R,
        input: &interface::ShaderInput<'_>,
    ) -> interface::Translation<interface::ShaderOutput> {
        self.translate_with_options_and_graphics_state(
            run,
            input,
            TranslationOptions::default(),
            &interface::GraphicsState::default(),
        )
    }

    /// Translate inputs across Rayon workers, returning results in input
    /// order. Each worker reuses its thread-local scratch across programs and
    /// batches. Errors do not stop other inputs. A one-thread pool runs here.
    /// Use Rayon ThreadPool::install to choose a pool or a worker count.
    pub fn translate_batch<R: Run>(
        &mut self,
        run: R,
        inputs: &[interface::ShaderInput<'_>],
    ) -> Vec<interface::Translation<interface::ShaderOutput>> {
        self.translate_batch_with_options_and_graphics_state(
            run,
            inputs,
            TranslationOptions::default(),
            &interface::GraphicsState::default(),
        )
    }

    /// Parallel batch translation with shared options and driver state.
    /// Every output owns its words independently of worker scratch and later batches.
    pub fn translate_batch_with_options_and_graphics_state<R: Run>(
        &mut self,
        run: R,
        inputs: &[interface::ShaderInput<'_>],
        options: TranslationOptions,
        state: &interface::GraphicsState,
    ) -> Vec<interface::Translation<interface::ShaderOutput>> {
        use rayon::prelude::*;
        if inputs.len() < 2 || rayon::current_num_threads() == 1 {
            return inputs
                .iter()
                .map(|input| {
                    self.translate_with_options_and_graphics_state(run, input, options, state)
                })
                .collect();
        }
        // Indexed parallel collect preserves positions without sorting. Worker
        // scratch survives jobs and batches; no shared translator mutex is needed.
        inputs
            .par_iter()
            .map(|input| translate_with_options_and_graphics_state(run, input, options, state))
            .collect()
    }

    pub fn translate_with_options_and_graphics_state<R: Run>(
        &mut self,
        run: R,
        input: &interface::ShaderInput<'_>,
        options: TranslationOptions,
        state: &interface::GraphicsState,
    ) -> interface::Translation<interface::ShaderOutput> {
        match ParsedProgram::parse(input, *state) {
            Ok(program) => self.translate_program(run, &program, options).into(),
            Err(malformed) => interface::Translation::Stopped(malformed.into()),
        }
    }

    /// The translator entry, on the run its host chose.
    fn translate_program<R: Run>(
        &mut self,
        run: R,
        program: &ParsedProgram<'_>,
        options: TranslationOptions,
    ) -> Result<interface::ShaderOutput, interface::Stopped> {
        let state = program.graphics_state();
        self.words.clear();
        self.words.extend(program.code.halves());
        let graph = decoder::decode_stream_into(
            run,
            &self.words,
            state.constant_buffer1,
            &mut self.decoded,
            &mut self.stack,
            &mut self.graph,
        );
        program
            .code
            .tail
            .reached(self.stack.program(), self.stack.slots())?;
        let module = emitter::emit_with_scratch(
            &self.decoded,
            graph,
            program.interface(),
            options,
            &state,
            &mut self.emission,
            &mut self.optimization,
        )?;
        Ok(interface::ShaderOutput {
            spirv: module.words,
            surface_descriptors: module.surface_descriptors,
            requires_maximal_reconvergence: module.requires_maximal_reconvergence,
            requires_demote_to_helper_invocation: module.requires_demote_to_helper_invocation,
            requires_subgroup_size_32: self
                .decoded
                .iter()
                .any(|x| matches!(x.kind, ir::InstructionKind::Vote(ops::Vote { .. }))),
        })
    }
}

pub fn translate_header_prefixed<R: Run>(
    run: R,
    input: &[u8],
) -> interface::Translation<interface::ShaderOutput> {
    translate_header_prefixed_for_generation(run, input, TargetGeneration::Sm5)
}

pub fn translate_header_prefixed_for_generation<R: Run>(
    run: R,
    input: &[u8],
    generation: TargetGeneration,
) -> interface::Translation<interface::ShaderOutput> {
    translate_header_prefixed_with_graphics_state(
        run,
        input,
        generation,
        &interface::GraphicsState::default(),
    )
}

/// Header-prefixed translation with driver-supplied stage state.
pub fn translate_header_prefixed_with_graphics_state<R: Run>(
    run: R,
    input: &[u8],
    generation: TargetGeneration,
    state: &interface::GraphicsState,
) -> interface::Translation<interface::ShaderOutput> {
    match parse_header_prefixed(input, generation, *state) {
        Ok(program) => translate_parsed(run, &program, TranslationOptions::default()),
        Err(malformed) => interface::Translation::Stopped(malformed.into()),
    }
}

/// Translate with an optional deterministic in-memory and on-disk cache.
pub fn translate_cached<R: Run>(
    run: R,
    input: &interface::ShaderInput<'_>,
    stage: headers::Stage,
    cache: &mut Cache,
) -> interface::Translation<interface::ShaderOutput> {
    interface::Translation::from(cache::parse_for_stage(input, stage).and_then(|program| {
        cache.translate_program(&program, CacheContext::default(), || {
            translate_parsed(run, program.as_ref(), TranslationOptions::default()).into()
        })
    }))
}

/// Cache-aware runtime translation including every supplied translation input.
pub fn translate_cached_with_context<R: Run>(
    run: R,
    input: &interface::ShaderInput<'_>,
    context: CacheContext,
    cache: &mut Cache,
) -> interface::Translation<interface::ShaderOutput> {
    match ParsedProgram::parse(input, context.graphics_state) {
        Ok(program) => translate_program_cached(run, &program.classify(), context, cache),
        Err(malformed) => interface::Translation::Stopped(malformed.into()),
    }
}

fn translate_program_cached<R: Run>(
    run: R,
    program: &program::Program<'_>,
    context: CacheContext,
    cache: &mut Cache,
) -> interface::Translation<interface::ShaderOutput> {
    cache
        .translate_program(program, context, || {
            translate_parsed(run, program.as_ref(), context.options).into()
        })
        .into()
}

/// Borrow the code and header without decoding instructions. Extended records
/// retain the complete envelope in `program_record` so translation and cache
/// identity automatically consume their cb1 snapshot. Legacy records retain
/// the generation-sized header. See docs/reference-sources.md
pub fn header_prefixed_input(
    input: &[u8],
    generation: TargetGeneration,
) -> Result<interface::ShaderInput<'_>, interface::Malformed> {
    record::open(input, Generation::new(input, generation, AsInput))
}

/// What a caller does with each shape of a header-prefixed input: an
/// extended record with its cb1 trailer already parsed, a legacy record split
/// at the generation's header size, or an input that forms neither.
trait Prefixed<'a> {
    type Output;
    fn legacy(self, program_record: &'a [u8], code: &'a [u8]) -> Self::Output;
    /// `input` is the whole record the trailer ends.
    fn extended(self, input: &'a [u8], record: record::Record<'a>) -> Self::Output;
    fn fault(self, error: ProgramError) -> Self::Output;
}

/// A header-prefixed input read under its target generation.
struct Generation<'a, P> {
    input: &'a [u8],
    header_size: usize,
    path: P,
}

impl<'a, P: Prefixed<'a>> Generation<'a, P> {
    fn new(input: &'a [u8], generation: TargetGeneration, path: P) -> Self {
        Self {
            input,
            header_size: generation.shader_header_size(),
            path,
        }
    }
}

/// Split a legacy record at the generation's header size.
fn split_legacy<'a, P: Prefixed<'a>>(input: &'a [u8], header_size: usize, path: P) -> P::Output {
    if input.len() < header_size {
        return path.fault(ProgramError::ShortHeader);
    }
    let (program_record, code) = input.split_at(header_size);
    path.legacy(program_record, code)
}

impl<'a, P: Prefixed<'a>> record::Envelope<'a> for Generation<'a, P> {
    type Output = P::Output;
    fn legacy(self, input: &'a [u8]) -> P::Output {
        split_legacy(input, self.header_size, self.path)
    }
    fn extended(self, record: record::Record<'a>) -> P::Output {
        if record.header.len() != self.header_size {
            return self
                .path
                .fault(ProgramError::Record(interface::Malformed::HeaderSize));
        }
        self.path.extended(self.input, record)
    }
    fn malformed(self, error: interface::Malformed) -> P::Output {
        self.path.fault(ProgramError::Record(error))
    }
}

/// A header-prefixed input borrowed as a shader input.
struct AsInput;

impl<'a> Prefixed<'a> for AsInput {
    type Output = Result<interface::ShaderInput<'a>, interface::Malformed>;
    fn legacy(self, program_record: &'a [u8], code: &'a [u8]) -> Self::Output {
        Ok(interface::ShaderInput {
            code,
            program_record,
        })
    }
    fn extended(self, input: &'a [u8], record: record::Record<'a>) -> Self::Output {
        Ok(interface::ShaderInput {
            code: record.code,
            program_record: input,
        })
    }
    fn fault(self, error: ProgramError) -> Self::Output {
        Err(error.into())
    }
}

/// A legacy header-prefixed input borrowed as a shader input, for a caller
/// that already read its tail.
fn legacy_input(
    input: &[u8],
    generation: TargetGeneration,
) -> Result<interface::ShaderInput<'_>, interface::Malformed> {
    split_legacy(input, generation.shader_header_size(), AsInput)
}

/// A header-prefixed input as its code and record parts, reusing the parsed
/// cb1 trailer of an extended record.
struct AsParts;

impl<'a> Prefixed<'a> for AsParts {
    type Output = Result<(&'a [u8], program::RecordParts<'a>), ProgramError>;
    fn legacy(self, program_record: &'a [u8], code: &'a [u8]) -> Self::Output {
        Ok((code, program::RecordParts::split(program_record)?))
    }
    fn extended(self, _: &'a [u8], record: record::Record<'a>) -> Self::Output {
        Ok((record.code, program::RecordParts::extended(record)))
    }
    fn fault(self, error: ProgramError) -> Self::Output {
        Err(error)
    }
}

/// The code and record parts of a header-prefixed input.
fn header_prefixed_parts(
    input: &[u8],
    generation: TargetGeneration,
) -> Result<(&[u8], program::RecordParts<'_>), ProgramError> {
    record::open(input, Generation::new(input, generation, AsParts))
}

/// Parse a header-prefixed input once into the program that translation and
/// cache identity take. Order: split, envelope, cb1 words, driver agreement,
/// header.
pub fn parse_header_prefixed<'a>(
    input: &'a [u8],
    generation: TargetGeneration,
    state: interface::GraphicsState<'a>,
) -> Result<ParsedProgram<'a>, interface::Malformed> {
    let (code, parts) = header_prefixed_parts(input, generation)?;
    ParsedProgram::from_parts(code, parts, state)
}

/// Parse a header-prefixed input once, checking its header before the cb1
/// words, for a caller that reports header faults on its own (the novena
/// adapter). Order: split, envelope, header, cb1 words, driver agreement.
pub fn parse_header_prefixed_header_first<'a>(
    input: &'a [u8],
    generation: TargetGeneration,
    state: interface::GraphicsState<'a>,
) -> Result<ParsedProgram<'a>, ProgramError> {
    let (code, parts) = header_prefixed_parts(input, generation)?;
    Ok(program::HeaderFirst::from_parts(code, parts)?.finish(state)?)
}

/// Parse a header-prefixed input once for the runtime cache. Order: split,
/// envelope, header, the stage's graphics state, then (complete programs
/// only) cb1 words and driver agreement.
pub fn parse_header_prefixed_for_cache<'a>(
    input: &'a [u8],
    context: CacheContext<'a>,
) -> Result<CacheProgram<'a>, ProgramError> {
    let (code, parts) = header_prefixed_parts(input, context.generation)?;
    Ok(program::HeaderFirst::from_parts(code, parts)?.classify(context.graphics_state)?)
}

pub fn translate_header_prefixed_cached<R: Run>(
    run: R,
    input: &[u8],
    context: CacheContext,
    cache: &mut Cache,
) -> interface::Translation<interface::ShaderOutput> {
    match parse_header_prefixed(input, context.generation, context.graphics_state) {
        Ok(program) => translate_program_cached(run, &program.classify(), context, cache),
        Err(malformed) => interface::Translation::Stopped(malformed.into()),
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        decoder::decode_stream, emitter::emit_with_header, interface::Stopped, spirv::Op,
        TargetGeneration,
    };
    use std::{fs, path::Path, process::Command};

    #[test]
    fn header_prefixed_matches_separate_translation() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        for entry in fs::read_dir(root.join("corpus")).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|x| x.to_str()) != Some("bin") {
                continue;
            }
            let name = path.file_stem().unwrap();
            let code = fs::read(&path).unwrap();
            let header =
                fs::read(path.with_file_name(format!("{}.hdr", name.to_string_lossy()))).unwrap();
            let mut prefixed = header[..TargetGeneration::Sm5.shader_header_size()].to_vec();
            prefixed.extend_from_slice(&code);
            let separate = Result::from(crate::translate(
                crate::QuietRun,
                &crate::interface::ShaderInput {
                    code: &code,
                    program_record: &header,
                },
            ));
            let combined = Result::from(crate::translate_header_prefixed_for_generation(
                crate::QuietRun,
                &prefixed,
                TargetGeneration::Sm5,
            ));
            // Both paths must agree, whether the fixture translates or not.
            match (combined, separate) {
                (Ok(c), Ok(s)) => assert_eq!(c.spirv, s.spirv, "{name:?}"),
                (Err(c), Err(s)) => assert_eq!(c.to_string(), s.to_string(), "{name:?}"),
                (c, s) => panic!(
                    "{name:?}: prefixed {:?} vs separate {:?}",
                    c.is_ok(),
                    s.is_ok()
                ),
            }
        }
    }

    struct TestOutputDir(std::path::PathBuf);

    impl TestOutputDir {
        fn new(name: &str) -> Self {
            let base = std::env::temp_dir();
            let stamp = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = base.join(format!("shadowbox-{name}-{}-{stamp}", std::process::id()));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestOutputDir {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    #[test]
    #[ignore = "manual throughput benchmark"]
    fn corpus_throughput() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let iterations = std::env::var("SHADOWBOX_BENCH_ITERATIONS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(100);
        let mut instruction_count = 0usize;
        let mut fixtures = Vec::new();
        for entry in fs::read_dir(root.join("corpus")).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|x| x.to_str()) != Some("bin") {
                continue;
            }
            let name = path.file_stem().unwrap().to_string_lossy().into_owned();
            let code = fs::read(&path).unwrap();
            let header = fs::read(root.join("corpus").join(format!("{name}.hdr"))).unwrap();
            let decoded = decode_stream(
                crate::QuietRun,
                &code
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| u32::from_le_bytes(*b))
                    .collect::<Vec<_>>(),
            );
            instruction_count += decoded.len();
            fixtures.push((code, header));
        }
        let start = std::time::Instant::now();
        for _ in 0..iterations {
            for (code, header) in &fixtures {
                let _ = Result::from(crate::translate(
                    crate::QuietRun,
                    &crate::interface::ShaderInput {
                        code,
                        program_record: header,
                    },
                ));
            }
        }
        let elapsed = start.elapsed();
        let shaders = fixtures.len() * iterations;
        println!(
            "fixtures={shaders} shaders_per_second={:.2} microseconds_per_instruction={:.3}",
            shaders as f64 / elapsed.as_secs_f64(),
            elapsed.as_secs_f64() * 1_000_000.0 / (instruction_count * iterations) as f64
        );
    }
    fn words(path: &Path) -> Vec<u32> {
        fs::read(path)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect()
    }
    #[test]
    fn corpus_translates_and_validates() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let expected = fs::read_to_string(root.join("corpus/translates.txt"))
            .unwrap()
            .lines()
            .filter(|line| !line.is_empty())
            .map(str::to_owned)
            .collect::<std::collections::BTreeSet<_>>();
        let output_dir = TestOutputDir::new("corpus");
        let mut table = Vec::new();
        for entry in fs::read_dir(root.join("corpus")).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|x| x.to_str()) != Some("bin") {
                continue;
            }
            let fixture = path.file_stem().unwrap().to_string_lossy().into_owned();
            if !expected.contains(&fixture) {
                continue;
            }
            let decoded = decode_stream(crate::QuietRun, &words(&path));
            let header = words(&root.join("corpus").join(format!("{fixture}.hdr")));
            let header = crate::headers::parse(&header).unwrap();
            match emit_with_header(&decoded, Some(&header)) {
                Ok(module) => {
                    let first = module.words.clone();
                    assert_eq!(
                        first,
                        emit_with_header(&decoded, Some(&header)).unwrap().words,
                        "{fixture} is nondeterministic"
                    );
                    let output = output_dir.0.join(format!("{fixture}.spv"));
                    fs::write(
                        &output,
                        first
                            .iter()
                            .flat_map(|w| w.to_le_bytes())
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
                    let result = Command::new("spirv-val")
                        .arg("--target-env")
                        .arg("vulkan1.2")
                        .arg(&output)
                        .output()
                        .unwrap();
                    assert!(
                        result.status.success(),
                        "{fixture}: {}",
                        String::from_utf8_lossy(&result.stderr)
                    );
                    if fixture == "constant_color" {
                        assert_eq!(constant_output(&first), [1.0, 0.25, 0.5, 1.0]);
                    }
                    table.push((
                        fixture,
                        "translated".to_string(),
                        "all instructions supported".to_string(),
                    ));
                }
                Err(
                    Stopped::Unbuilt(x)
                    | Stopped::Unwanted(x)
                    | Stopped::Unknown(x)
                    | Stopped::LivePadding(x)
                    | Stopped::UnresolvedStack(x)
                    | Stopped::UnprovenStream(x),
                ) => {
                    // A standalone texture lacks the handle definitions of
                    // its shader, so testing isolated instructions can name
                    // an earlier, otherwise supported query. The full-shader
                    // stop must name an actual decoded instruction exactly.
                    assert!(decoded.contains(&*x), "{fixture}: {x}");
                    table.push((fixture, "not translated".to_string(), x.to_string()));
                }
                Err(error) => panic!("{fixture}: unexpected translation error: {error}"),
            }
        }
        table.sort_by(|a, b| a.0.cmp(&b.0));
        eprintln!("fixture | translated | reason");
        let translated = table
            .iter()
            .map(|row| row.0.clone())
            .collect::<std::collections::BTreeSet<_>>();
        for (fixture, translated, reason) in table {
            eprintln!("{fixture} | {translated} | {reason}");
        }
        assert_eq!(translated, expected);
    }

    #[test]
    fn vertex_passthrough_translates_and_validates() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let output_dir = TestOutputDir::new("vertex-passthrough");
        let code = words(&root.join("corpus/vertex_passthrough.bin"));
        let header = words(&root.join("corpus/vertex_passthrough.hdr"));
        let header = crate::headers::parse(&header).unwrap();
        let module = crate::emitter::emit_with_header(
            &crate::decoder::decode_stream(crate::QuietRun, &code),
            Some(&header),
        )
        .unwrap();
        let output = output_dir.0.join("vertex_passthrough.spv");
        fs::write(
            &output,
            module
                .words
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        assert!(Command::new("spirv-val")
            .arg("--target-env")
            .arg("vulkan1.2")
            .arg(&output)
            .status()
            .unwrap()
            .success());
    }
    fn constant_output(words: &[u32]) -> [f32; 4] {
        let mut values = std::collections::HashMap::new();
        let mut pointers = std::collections::HashMap::new();
        let mut output = None;
        let mut i = 5;
        while i < words.len() {
            let count = (words[i] >> 16) as usize;
            match Op::of(words[i]) {
                Op::Constant => {
                    values.insert(words[i + 2], f32::from_bits(words[i + 3]));
                }
                Op::Variable => {
                    pointers.insert(words[i + 2], None);
                }
                Op::Load => {
                    let value = pointers
                        .get(&words[i + 3])
                        .copied()
                        .flatten()
                        .and_then(|id| values.get(&id).copied());
                    for value in value.into_iter() {
                        values.insert(words[i + 2], value);
                    }
                }
                Op::Store => {
                    if let Some(pointer) = pointers.get_mut(&words[i + 1]) {
                        *pointer = Some(words[i + 2]);
                    }
                }
                Op::Bitcast => {
                    if let Some(value) = values.get(&words[i + 3]).copied() {
                        values.insert(words[i + 2], value);
                    }
                }
                _ => {}
            }
            if Op::of(words[i]) == Op::CompositeConstruct {
                output = Some([words[i + 3], words[i + 4], words[i + 5], words[i + 6]]);
            }
            i += count;
        }
        output.unwrap().map(|id| values[&id])
    }

    #[test]
    fn add_inputs_uses_program_record_header() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus");
        let code = Vec::new();
        let header = fs::read(root.join("add_inputs.hdr")).unwrap();
        let input = crate::interface::ShaderInput {
            code: &code,
            program_record: &header,
        };
        let output = Result::from(crate::translate(crate::QuietRun, &input)).unwrap();
        let locations = output
            .spirv
            .windows(4)
            .filter_map(|words| {
                (Op::of(words[0]) == Op::Decorate && words[2] == 30).then_some(words[3])
            })
            .collect::<Vec<_>>();
        assert!(locations.contains(&0));
        assert!(locations.contains(&1));
    }
}
