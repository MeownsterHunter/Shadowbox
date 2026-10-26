//! The program record, parsed once where it enters the translator. Its parts
//! are the optional cb1 envelope (docs/reference-sources.md), the
//! canonical shader-program header and the immutable cb1 words. Translation
//! and cache identity consume a `ParsedProgram` and never read raw bytes.
use crate::{
    cache::{missing_state_keys, program_cache_key},
    headers::{self, Interface, OutsideHeader, ShaderHeader, Stage},
    interface::{GraphicsState, Malformed, ShaderInput},
    record::{self, Envelope, Record},
    TargetGeneration, TranslationOptions,
};

/// A program record split into header bytes and, for an extended record, cb1
/// snapshot bytes. Neither part is interpreted yet.
#[derive(Clone, Copy)]
pub(crate) struct RecordParts<'a> {
    header: &'a [u8],
    cb1: Option<&'a [u8]>,
}

/// Splitting a program record: a legacy record is all header.
struct Split;

impl<'a> Envelope<'a> for Split {
    type Output = Result<RecordParts<'a>, Malformed>;
    fn legacy(self, input: &'a [u8]) -> Self::Output {
        Ok(RecordParts {
            header: input,
            cb1: None,
        })
    }
    fn extended(self, record: Record<'a>) -> Self::Output {
        Ok(RecordParts::extended(record))
    }
    fn malformed(self, error: Malformed) -> Self::Output {
        Err(error)
    }
}

impl<'a> RecordParts<'a> {
    pub(crate) fn split(bytes: &'a [u8]) -> Result<Self, Malformed> {
        record::open(bytes, Split)
    }

    pub(crate) fn extended(record: Record<'a>) -> Self {
        Self {
            header: record.header,
            cb1: Some(record.cb1),
        }
    }

    /// Immutable little-endian cb1 words. Saved uint reads follow Ryujinx
    /// DiskCache/DiskCacheGpuAccessor.cs:62-69.
    pub(crate) fn cb1(self) -> Result<Option<Vec<u32>>, Malformed> {
        self.cb1.map(record::words).transpose()
    }

    fn header(self) -> Result<ProgramHeader, HeaderFault> {
        ProgramHeader::parse(self.header)
    }
}

/// Why a program header did not parse. Callers that report header faults on
/// their own (the novena adapter) keep the parser's typed error.
pub(crate) enum HeaderFault {
    Length,
    Header(OutsideHeader),
}

impl From<HeaderFault> for Malformed {
    fn from(fault: HeaderFault) -> Self {
        match fault {
            HeaderFault::Length => Malformed::HeaderLength,
            HeaderFault::Header(e) => Malformed::Header(e),
        }
    }
}

/// The program's header kind: an SM 5.x header zero-extended to the 128
/// bytes the parser consumes, with its parsed fields, or no header at all.
#[expect(
    clippy::large_enum_variant,
    reason = "one per parsed program; boxing would allocate per translation"
)]
pub(crate) enum ProgramHeader {
    Parsed {
        canonical: [u8; 128],
        parsed: ShaderHeader,
    },
    Headerless,
}

impl ProgramHeader {
    // The parser consumes 32 words; SM5/6 prefixed headers contain only 20.
    // Zero-extension is the existing header-prefixed contract in lib.rs.
    // Mesa nvk_shader.c:605-606 and nvk_shader.h:29-30 specify generation sizes.
    fn parse(header: &[u8]) -> Result<Self, HeaderFault> {
        if header.is_empty() {
            return Ok(Self::Headerless);
        }
        if header.len() != 80 && header.len() < 128 {
            return Err(HeaderFault::Length);
        }
        let mut canonical = [0; 128];
        let used = header.len().min(128);
        canonical[..used].copy_from_slice(&header[..used]);
        let (chunks, _) = canonical.as_chunks::<4>();
        let words: [u32; 32] = std::array::from_fn(|i| u32::from_le_bytes(chunks[i]));
        let parsed = headers::parse(&words).map_err(HeaderFault::Header)?;
        Ok(Self::Parsed { canonical, parsed })
    }

    fn interface(&self) -> Interface<'_> {
        match self {
            Self::Parsed { parsed, .. } => Interface::Header(parsed),
            Self::Headerless => Interface::Headerless,
        }
    }

    fn stage(&self) -> Stage {
        self.interface().stage()
    }

    /// The canonical header bytes: 128 with a header, none without one.
    fn canonical(&self) -> &[u8] {
        match self {
            Self::Parsed { canonical, .. } => canonical,
            Self::Headerless => &[],
        }
    }
}

/// Where the program's cb1 words come from, decided once at parse: the
/// record's snapshot (which the driver's, if supplied, agreed with), or the
/// driver's, when the record carries none.
enum Cb1<'a> {
    Record(Vec<u32>),
    Driver(Option<&'a [u32]>),
}

/// Shader code plus its parsed program record and the driver state checked
/// against the record's cb1 snapshot. Built once where a program enters the
/// translator and taken by translation and cache identity alike.
/// The code bytes, parsed once at the edge: the whole 64-bit instruction
/// words, and where the bytes end.
#[derive(Clone, Copy)]
pub(crate) struct Code<'a> {
    /// The bytes as given, which the cache identity hashes.
    pub(crate) bytes: &'a [u8],
    words: &'a [[u8; 8]],
    pub(crate) tail: Tail,
}

/// Where the code bytes end: on a word boundary, or inside the word at
/// this byte address.
#[derive(Clone, Copy)]
pub(crate) enum Tail {
    Whole,
    Partial(u32),
}

impl<'a> Code<'a> {
    fn parse(bytes: &'a [u8]) -> Self {
        let (words, rest) = bytes.as_chunks::<8>();
        let tail = match rest {
            [] => Tail::Whole,
            [_, ..] => Tail::Partial((bytes.len() - rest.len()) as u32),
        };
        Self { bytes, words, tail }
    }

    /// The whole words as the decoder reads them: low half, then high half.
    pub(crate) fn halves(&self) -> impl Iterator<Item = u32> + '_ {
        self.words.iter().flat_map(|word| {
            let word = u64::from_le_bytes(*word);
            [word as u32, (word >> 32) as u32]
        })
    }

    /// Each instruction slot's address and halves, without the control
    /// word that begins every group of four (Mesa sm50.rs:3418-3428).
    pub(crate) fn instruction_slots(&self) -> impl Iterator<Item = (u32, [u32; 2])> + '_ {
        self.words
            .iter()
            .enumerate()
            .filter(|(slot, _)| slot % 4 != 0)
            .map(|(slot, word)| {
                let word = u64::from_le_bytes(*word);
                (slot as u32 * 8, [word as u32, (word >> 32) as u32])
            })
    }
}

impl Tail {
    /// Whether control reaches the word the bytes end inside.
    pub(crate) fn reached(
        self,
        program: &crate::flow::Program,
        slots: &crate::decoder::Slots,
    ) -> Result<(), crate::interface::Stopped> {
        match self {
            Self::Whole => Ok(()),
            Self::Partial(at) => crate::decoder::partial_word_reached(program, slots, at),
        }
    }
}

pub struct ParsedProgram<'a> {
    pub(crate) code: Code<'a>,
    header: ProgramHeader,
    cb1: Cb1<'a>,
    state: GraphicsState<'a>,
}

impl<'a> ParsedProgram<'a> {
    /// Translation and cache-key order: envelope, cb1 words, driver
    /// agreement, header.
    pub(crate) fn parse(
        input: &ShaderInput<'a>,
        state: GraphicsState<'a>,
    ) -> Result<Self, Malformed> {
        Self::from_parts(input.code, RecordParts::split(input.program_record)?, state)
    }

    pub(crate) fn from_parts(
        code: &'a [u8],
        parts: RecordParts<'a>,
        state: GraphicsState<'a>,
    ) -> Result<Self, Malformed> {
        let cb1 = agreeing_cb1(parts, &state)?;
        Ok(Self {
            code: Code::parse(code),
            header: parts.header()?,
            cb1,
            state,
        })
    }

    pub(crate) fn interface(&self) -> Interface<'_> {
        self.header.interface()
    }

    /// The canonical header bytes: empty without a header, otherwise 128.
    pub(crate) fn canonical_header(&self) -> &[u8] {
        self.header.canonical()
    }

    /// The stage named by the program's header (fragment without one).
    pub fn stage(&self) -> Stage {
        self.header.stage()
    }

    /// The driver state with the program's cb1 words.
    pub(crate) fn graphics_state(&self) -> GraphicsState<'_> {
        GraphicsState {
            constant_buffer1: match &self.cb1 {
                Cb1::Record(words) => Some(words),
                Cb1::Driver(words) => *words,
            },
            ..self.state
        }
    }

    /// Whether the program carries every graphics state its stage needs,
    /// decided once here.
    pub(crate) fn classify(self) -> Program<'a> {
        match missing_state_keys(self.stage(), &self.graphics_state()) {
            [] => Program::Complete(CompleteProgram(self)),
            needed => Program::Incomplete(ProgramAwaitingState {
                program: self,
                needed,
            }),
        }
    }
}

/// A parsed program, by whether its stage has all the graphics state it
/// needs. Only a complete program has a cache key; one awaiting state
/// carries the state its stage needs.
pub enum Program<'a> {
    Complete(CompleteProgram<'a>),
    Incomplete(ProgramAwaitingState<'a>),
}

impl<'a> Program<'a> {
    /// Parse a program once under the driver's graphics state and decide
    /// once whether its stage has every state it needs. Order: envelope, cb1
    /// words, driver agreement, header.
    pub fn parse(input: &ShaderInput<'a>, state: GraphicsState<'a>) -> Result<Self, Malformed> {
        Ok(ParsedProgram::parse(input, state)?.classify())
    }

    /// The program's identity under `generation` and `options`, complete or
    /// not, for reports and deduplication. It is computed like the cache key,
    /// but only [`CompleteProgram::cache_key`] keys the runtime cache.
    pub fn identity(&self, generation: TargetGeneration, options: TranslationOptions) -> String {
        program_cache_key(self.as_ref(), generation, options)
    }
}

impl<'a> AsRef<ParsedProgram<'a>> for Program<'a> {
    fn as_ref(&self) -> &ParsedProgram<'a> {
        match self {
            Self::Complete(program) => &program.0,
            Self::Incomplete(program) => &program.program,
        }
    }
}

/// What a caller does with a program on each of its paths.
pub trait ProgramPath<'a> {
    type Output;
    fn complete(self, program: CompleteProgram<'a>) -> Self::Output;
    fn awaiting_state(self, program: ProgramAwaitingState<'a>) -> Self::Output;
}

/// What a caller does with a program it could not parse, on each path.
pub trait ProgramErrorPath {
    type Output;
    /// The input is shorter than its generation's header.
    fn short_header(self) -> Self::Output;
    /// The record, cb1 snapshot or header length is invalid.
    fn record(self, error: Malformed) -> Self::Output;
    /// The header is a shape the edge leaves outside.
    fn header(self, outside: OutsideHeader) -> Self::Output;
}

/// Take the path `path` names for this value.
pub trait Follow<P> {
    type Output;
    fn follow(self, path: P) -> Self::Output;
}

impl<'a, P: ProgramPath<'a>> Follow<P> for Program<'a> {
    type Output = P::Output;
    fn follow(self, path: P) -> P::Output {
        match self {
            Self::Complete(program) => path.complete(program),
            Self::Incomplete(program) => path.awaiting_state(program),
        }
    }
}

impl<P: ProgramErrorPath> Follow<P> for ProgramError {
    type Output = P::Output;
    fn follow(self, path: P) -> P::Output {
        match self {
            Self::ShortHeader => path.short_header(),
            Self::Record(error) => path.record(error),
            Self::Header(error) => path.header(error),
        }
    }
}

/// A program whose stage has every graphics state it needs.
pub struct CompleteProgram<'a>(ParsedProgram<'a>);

impl<'a> CompleteProgram<'a> {
    /// The runtime cache key of this program under `generation` and
    /// `options`.
    pub fn cache_key(&self, generation: TargetGeneration, options: TranslationOptions) -> String {
        program_cache_key(&self.0, generation, options)
    }

    /// The parsed program, for [`crate::translate_parsed`].
    pub fn parsed(&self) -> &ParsedProgram<'a> {
        &self.0
    }
}

/// A program whose stage awaits graphics state the driver has not given.
/// It is translated but never cached. Built only by classification, with the
/// keys of the state its stage still needs.
pub struct ProgramAwaitingState<'a> {
    program: ParsedProgram<'a>,
    needed: &'static [&'static str],
}

impl ProgramAwaitingState<'_> {
    /// The graphics state keys the program's stage still needs.
    pub fn needed_state(&self) -> &'static [&'static str] {
        self.needed
    }
}

/// A header-prefixed program parsed for the cache: complete, or missing the
/// listed graphics state, in which case its cb1 snapshot is left unread.
#[expect(
    clippy::large_enum_variant,
    reason = "returned once per parse; boxing would allocate per lookup"
)]
pub enum CacheProgram<'a> {
    Complete(CompleteProgram<'a>),
    Incomplete { missing: &'static [&'static str] },
}

/// Why a program could not be parsed with its header checked first: the
/// input is shorter than its generation's header, its record, cb1 snapshot
/// or header length is invalid, or the header's own fields are.
#[derive(Debug)]
pub enum ProgramError {
    ShortHeader,
    Record(Malformed),
    Header(OutsideHeader),
}

/// A program error read as the translator's error, with the translator's
/// text.
struct AsTranslation;

impl ProgramErrorPath for AsTranslation {
    type Output = Malformed;
    fn short_header(self) -> Malformed {
        Malformed::ShortHeader
    }
    fn record(self, error: Malformed) -> Malformed {
        error
    }
    fn header(self, outside: OutsideHeader) -> Malformed {
        Malformed::Header(outside)
    }
}

impl From<ProgramError> for Malformed {
    fn from(error: ProgramError) -> Self {
        error.follow(AsTranslation)
    }
}

impl From<Malformed> for ProgramError {
    fn from(error: Malformed) -> Self {
        Self::Record(error)
    }
}

impl From<HeaderFault> for ProgramError {
    fn from(fault: HeaderFault) -> Self {
        match fault {
            HeaderFault::Length => Self::Record(fault.into()),
            HeaderFault::Header(error) => Self::Header(error),
        }
    }
}

/// Order for entry points keyed by stage: envelope, header, then the caller's
/// stage check or the graphics-state check, then cb1 words and driver
/// agreement.
pub(crate) struct HeaderFirst<'a> {
    code: &'a [u8],
    parts: RecordParts<'a>,
    header: ProgramHeader,
}

impl<'a> HeaderFirst<'a> {
    pub(crate) fn parse(input: &ShaderInput<'a>) -> Result<Self, Malformed> {
        Ok(Self::from_parts(
            input.code,
            RecordParts::split(input.program_record)?,
        )?)
    }

    pub(crate) fn from_parts(code: &'a [u8], parts: RecordParts<'a>) -> Result<Self, HeaderFault> {
        Ok(Self {
            code,
            parts,
            header: parts.header()?,
        })
    }

    pub(crate) fn stage(&self) -> Stage {
        self.header.stage()
    }

    pub(crate) fn finish(self, state: GraphicsState<'a>) -> Result<ParsedProgram<'a>, Malformed> {
        let cb1 = agreeing_cb1(self.parts, &state)?;
        Ok(ParsedProgram {
            code: Code::parse(self.code),
            cb1,
            header: self.header,
            state,
        })
    }

    /// Check the stage's graphics state before the cb1 snapshot is read.
    pub(crate) fn classify(self, state: GraphicsState<'a>) -> Result<CacheProgram<'a>, Malformed> {
        Ok(match missing_state_keys(self.stage(), &state) {
            [] => CacheProgram::Complete(CompleteProgram(self.finish(state)?)),
            missing => CacheProgram::Incomplete { missing },
        })
    }
}

fn agreeing_cb1<'a>(
    parts: RecordParts<'_>,
    state: &GraphicsState<'a>,
) -> Result<Cb1<'a>, Malformed> {
    Ok(match parts.cb1()? {
        Some(words) => {
            if state
                .constant_buffer1
                .is_some_and(|supplied| supplied != words)
            {
                return Err(Malformed::Cb1Disagrees);
            }
            Cb1::Record(words)
        }
        None => Cb1::Driver(state.constant_buffer1),
    })
}

#[cfg(test)]
mod tests {
    use crate::{
        encode_header_prefixed_with_constant_buffer1, header_prefixed_input,
        headers::{OutsideHeader, Stage},
        interface::{GeometryInput, GraphicsState, Malformed},
        parse_header_prefixed_for_cache, translate_cached,
        translate_header_prefixed_with_graphics_state, Cache, CacheConfig, CacheContext,
        CacheProgram, ProgramError, TargetGeneration,
    };

    /// A program's identity under `context`, through [`crate::Program`].
    fn identity(
        input: &crate::interface::ShaderInput<'_>,
        context: CacheContext,
    ) -> Result<String, crate::interface::Stopped> {
        Ok(crate::Program::parse(input, context.graphics_state)?
            .identity(context.generation, context.options))
    }

    // Stage field 10..14 = 15 is unsupported (Mesa cla097sph.h:37-42).
    fn record_with_bad_header(cb1: &[u8]) -> Vec<u8> {
        let mut legacy = vec![0; 80];
        legacy[1] = 0x3c;
        // EXIT bundle: Mesa sm50.rs:3418-3428, 3101-3110.
        legacy.extend(0_u64.to_le_bytes());
        legacy.extend(0xe300_0000_0007_000f_u64.to_le_bytes());
        encode_header_prefixed_with_constant_buffer1(&legacy, TargetGeneration::Sm5, cb1).unwrap()
    }

    #[test]
    fn translation_reports_cb1_before_header_and_stage_entries_the_reverse() {
        let partial = record_with_bad_header(&[1, 2, 3]);
        let message = |e: crate::interface::Stopped| e.to_string();
        let cb1_error = "invalid SPIR-V module: cb1 snapshot contains a partial word";
        let header_error = "invalid SPIR-V module: header: ReservedStage(15)";
        let translated = Result::from(translate_header_prefixed_with_graphics_state(
            crate::QuietRun,
            &partial,
            TargetGeneration::Sm5,
            &GraphicsState::default(),
        ));
        assert_eq!(translated.map_err(message).unwrap_err(), cb1_error);
        let input = header_prefixed_input(&partial, TargetGeneration::Sm5).unwrap();
        let key = identity(&input, CacheContext::default());
        assert_eq!(key.map_err(message).unwrap_err(), cb1_error);
        let mut cache = Cache::new(CacheConfig {
            memory: crate::cache::Keep::nothing(),
            disk: crate::cache::Disk::nowhere(),
        });
        let cached = Result::from(translate_cached(
            crate::QuietRun,
            &input,
            Stage::Fragment,
            &mut cache,
        ));
        assert_eq!(cached.map_err(message).unwrap_err(), header_error);

        let whole = record_with_bad_header(&[1, 0, 0, 0]);
        let disagreeing = GraphicsState {
            constant_buffer1: Some(&[2]),
            ..Default::default()
        };
        let translated = Result::from(translate_header_prefixed_with_graphics_state(
            crate::QuietRun,
            &whole,
            TargetGeneration::Sm5,
            &disagreeing,
        ));
        assert_eq!(
            translated.map_err(message).unwrap_err(),
            "invalid SPIR-V module: driver cb1 snapshot disagrees with record snapshot"
        );
    }

    fn geometry_record(cb1: &[u8]) -> Vec<u8> {
        let mut legacy = vec![0; 80];
        // Stage field 10..14 = 4 is geometry (Mesa cla097sph.h:37-42).
        legacy[1] = 4 << 2;
        legacy.extend(0_u64.to_le_bytes());
        legacy.extend(0xe300_0000_0007_000f_u64.to_le_bytes());
        encode_header_prefixed_with_constant_buffer1(&legacy, TargetGeneration::Sm5, cb1).unwrap()
    }

    #[test]
    fn cache_parse_reads_cb1_only_for_complete_programs() {
        let partial = geometry_record(&[1, 2, 3]);
        let missing = parse_header_prefixed_for_cache(&partial, CacheContext::default());
        assert!(matches!(
            missing,
            Ok(CacheProgram::Incomplete {
                missing: ["geometry_input"]
            })
        ));
        let complete = CacheContext {
            graphics_state: GraphicsState {
                geometry_input: Some(GeometryInput::Triangles),
                ..Default::default()
            },
            ..Default::default()
        };
        match parse_header_prefixed_for_cache(&partial, complete) {
            Err(ProgramError::Record(Malformed::PartialCb1Word)) => {}
            _ => panic!("a complete program reads its cb1 snapshot"),
        }

        let whole = geometry_record(&[1, 0, 0, 0]);
        let Ok(CacheProgram::Complete(program)) = parse_header_prefixed_for_cache(&whole, complete)
        else {
            panic!("a geometry program with its input is complete");
        };
        let input = header_prefixed_input(&whole, TargetGeneration::Sm5).unwrap();
        assert_eq!(
            program.cache_key(complete.generation, complete.options),
            identity(&input, complete).unwrap()
        );
    }

    #[test]
    fn cache_parse_keeps_the_header_error_typed() {
        let record = record_with_bad_header(&[1, 0, 0, 0]);
        assert!(matches!(
            parse_header_prefixed_for_cache(&record, CacheContext::default()),
            Err(ProgramError::Header(OutsideHeader::ReservedStage(15)))
        ));
    }
}
