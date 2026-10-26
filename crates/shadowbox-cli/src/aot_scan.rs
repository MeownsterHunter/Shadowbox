//! Streaming discovery policy and encoding sources: docs/reference-sources.md
mod iostore_containers;
mod nw_containers;
mod ue_libraries;
use super::cache_state::{Classify, NeededState, StateWork};
use super::pak::Integrity;
use nw_containers::NwBudget;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use shadowbox::ir::ops;
use shadowbox::{
    decoder, headers,
    ir::{BranchTarget, InstructionKind},
};
use std::{
    collections::{HashMap, HashSet},
    fs::{self, File, OpenOptions},
    io::{self, BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, AtomicUsize, Ordering},
    time::{Instant, SystemTime, UNIX_EPOCH},
};
use ue_libraries::UeStage;

const CHUNK: usize = 64 * 1024;
const OVERLAP: usize = 7;
// Project policy, not a hardware limit. Reject rather than truncate larger programs.
const MAX_CODE: usize = 1024 * 1024;
const CONTEXT: u64 = 4096;
const HEADER_SIZE: usize = 80; // shadowbox/src/lib.rs:36-40,85-112.

#[derive(Debug)]
enum ProbeError {
    Reject(String),
    Io(io::Error),
}

impl From<io::Error> for ProbeError {
    fn from(error: io::Error) -> Self {
        if error.kind() == io::ErrorKind::UnexpectedEof {
            Self::Reject("truncated record or bundle".into())
        } else {
            Self::Io(error)
        }
    }
}

fn reject<T>(reason: &str) -> Result<T, ProbeError> {
    Err(ProbeError::Reject(reason.into()))
}

struct Program {
    bytes: Vec<u8>,
    stage: headers::Stage,
    code_offset: usize,
}

fn scheduling_valid(raw: u64) -> bool {
    // Mesa nak/sm50.rs:238-246,3406-3407: three 21-bit dependency fields.
    // envytools envydis/gm107.c:2174-2176: bit 63 must be zero.
    // Mesa nak/ir.rs:8716-8723: barriers 0..5; sm50.rs:243-244: 7 = absent.
    raw >> 63 == 0
        && (0..3).all(|slot| {
            let control = raw >> (slot * 21);
            (control >> 5) & 7 != 6 && (control >> 8) & 7 != 6
        })
}

fn raw_word(bytes: &[u8]) -> [u32; 2] {
    [
        u32::from_le_bytes(bytes[..4].try_into().unwrap()),
        u32::from_le_bytes(bytes[4..8].try_into().unwrap()),
    ]
}

fn validate_instruction(
    bytes: &[u8],
    address: u32,
) -> Result<shadowbox::ir::Instruction, ProbeError> {
    let raw = raw_word(bytes);
    let instruction = decoder::decode_word(raw, address);
    if matches!(
        instruction.kind.legacy_form(),
        InstructionKind::Unknown(ops::Unknown { .. })
    ) {
        return Err(ProbeError::Reject(format!(
            "unknown instruction at code +0x{address:x}: 0x{:08x}{:08x}",
            raw[1], raw[0]
        )));
    }
    if matches!(instruction.kind, InstructionKind::Exit(ops::Exit { .. })) {
        // Mesa nak/sm50.rs:3106-3110 encodes CC.T in bits 0..4;
        // :220-235 encodes predicate 16..19. Discovery keeps a proven
        // unconditional end boundary; conditional EXIT remains rejected here.
        let word = u64::from(raw[0]) | (u64::from(raw[1]) << 32);
        if word & !(0xf << 16) != 0xe300_0000_0000_000f {
            return reject("unsupported EXIT fields");
        }
    }
    Ok(instruction)
}

fn instruction_address(address: i64) -> Result<u32, ProbeError> {
    if address < 0 || address >= MAX_CODE as i64 || address % 8 != 0 {
        return reject("control-flow target outside bounded code or unaligned");
    }
    // Bundle-start targets run slot 0 (shadowbox::ir::run_slot); Mesa
    // nak/sm50.rs:3418-3428 places slot 0 at bundle start + 8.
    Ok(shadowbox::ir::run_slot(address as u32))
}

/// A decoded target already runs slot 0 for a bundle start. MAX_CODE is a
/// multiple of 32, so checking that slot equals checking the named address.
fn target(target: BranchTarget) -> Result<u32, ProbeError> {
    instruction_address(i64::from(target.slot()))
}

fn outside<T>() -> Result<T, ProbeError> {
    reject("control-flow target outside bounded code or unaligned")
}

fn validate(file: &mut File, offset: u64, compute: bool) -> Result<Program, ProbeError> {
    // novena docs/reference-sources.md supplies both magic
    // layouts: graphics SPH +0x30, code +0x80; compute code +0x100, zero SPH.
    let code_offset = if compute { 0x100 } else { 0x80 };
    let mut prefix = vec![0; code_offset];
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(&mut prefix)?;
    // novena docs/reference-sources.md: words 1..11
    // are zero. Unknown prefix layouts are rejected, never interpreted loosely.
    if prefix[4..0x30].iter().any(|&byte| byte != 0) {
        return reject("nonzero layout prefix");
    }
    let header = if compute {
        // novena docs/reference-sources.md: +0x30..+0x100 zero.
        if prefix[0x30..].iter().any(|&byte| byte != 0) {
            return reject("nonzero compute header area");
        }
        vec![0; HEADER_SIZE]
    } else {
        prefix[0x30..0x80].to_vec()
    };
    let mut words = [0; 32];
    for (word, chunk) in words.iter_mut().zip(header.as_chunks::<4>().0) {
        *word = u32::from_le_bytes(*chunk);
    }
    // headers.rs:64-69 defines supported stages. Other parsed fields are
    // bitmaps or full-range 2-bit interpolation values (:71-99).
    let parsed = headers::parse(&words)
        .map_err(|_| ProbeError::Reject("unsupported header stage".into()))?;
    if !compute {
        // Ryujinx Translation/ShaderHeader.cs:106-109: SPH type 0..5,
        // version 5..10, stage 10..14. Mesa nak/sph.rs:112-120: type 2 for
        // fragment, 1 for VTG; Maxwell version 3. Each test changes acceptance.
        let expected_type = match parsed.stage() {
            headers::Stage::Fragment => 2,
            // Type 1 applies to all VTG stages, including tessellation.
            // Mesa nak/sph.rs:112-120,204-215; Ryujinx ShaderHeader.cs:109.
            headers::Stage::Vertex
            | headers::Stage::Geometry
            | headers::Stage::TessellationControl
            | headers::Stage::TessellationEvaluation => 1,
            headers::Stage::Compute => return reject("graphics record has compute stage"),
        };
        if words[0] & 31 != expected_type || (words[0] >> 5) & 31 != 3 {
            return reject("invalid Maxwell SPH type or version");
        }
    }

    let mut code = Vec::new();
    let mut pending = vec![8u32];
    let mut visited = vec![false; MAX_CODE / 8];
    let mut saw_exit = false;
    // Ryujinx Decoders/Decoder.cs:129-160,501-515: explore branch targets
    // and conditional fallthrough, not just the first EXIT. A worklist of
    // instruction addresses suffices here because no SSA is constructed.
    while let Some(address) = pending.pop() {
        if visited[address as usize / 8] {
            continue;
        }
        visited[address as usize / 8] = true;
        let needed = (address as usize / 32 + 1) * 32;
        while code.len() < needed {
            let mut bundle = [0; 32];
            file.read_exact(&mut bundle)?;
            let control = u64::from_le_bytes(bundle[..8].try_into().unwrap());
            if !scheduling_valid(control) {
                return reject("invalid scheduling control word");
            }
            // Mesa nak/sm50.rs:3468-3471: control + three 8-byte instructions.
            // Check every slot, including retained unreachable slots. Never
            // drop a real instruction to manufacture a valid candidate.
            for slot in [8, 16, 24] {
                validate_instruction(&bundle[slot..slot + 8], (code.len() + slot) as u32)?;
            }
            code.extend_from_slice(&bundle);
        }
        let instruction = validate_instruction(&code[address as usize..], address)?;
        let mut fallthrough = true;
        match instruction.kind {
            InstructionKind::Exit(ops::Exit { cc_test: 15 }) => {
                // envytools envydis/gm107.c:363-364,408-411: !PT never runs.
                saw_exit |= instruction.predicate
                    != Some(shadowbox::ir::Predicate {
                        register: 7,
                        inverted: true,
                    });
                fallthrough = instruction.predicate.is_some();
            }
            InstructionKind::Branch(ops::Branch {
                target: to,
                cc_test,
            }) => {
                pending.push(target(to)?);
                fallthrough = instruction.predicate.is_some() || cc_test != 0xf;
            }
            InstructionKind::Schedule(ops::Schedule { target: to, .. })
            | InstructionKind::PushTarget(ops::PushTarget { target: to, .. }) => {
                // Ryujinx Decoder.cs:133-138 also visits push targets.
                pending.push(target(to)?);
            }
            InstructionKind::OutsideBranch(ops::OutsideBranch { .. })
            | InstructionKind::OutsideSchedule(ops::OutsideSchedule { .. })
            | InstructionKind::OutsidePushTarget(ops::OutsidePushTarget { .. })
            | InstructionKind::MisalignedBranch(ops::MisalignedBranch { .. })
            | InstructionKind::MisalignedSchedule(ops::MisalignedSchedule { .. })
            | InstructionKind::MisalignedPushTarget(ops::MisalignedPushTarget { .. }) => {
                return outside()
            }
            InstructionKind::PopJump(ops::PopJump { .. })
            | InstructionKind::Brx(ops::Brx { .. })
            | InstructionKind::OutsideBrx(ops::OutsideBrx { .. }) => {
                return reject("unproven stack or indirect control flow");
            }
            _ => {}
        }
        if fallthrough {
            pending.push(instruction_address(i64::from(address) + 8)?);
        }
    }
    if !saw_exit {
        return reject("no reachable EXIT");
    }
    // All retained direct targets must belong to the output, even in slots
    // not reached by the entry point. Their raw instructions are preserved.
    for address in (0..code.len()).step_by(8).filter(|at| at % 32 != 0) {
        let instruction = validate_instruction(&code[address..], address as u32)?;
        match instruction.kind {
            InstructionKind::Branch(ops::Branch { target: to, .. })
            | InstructionKind::Schedule(ops::Schedule { target: to, .. })
            | InstructionKind::PushTarget(ops::PushTarget { target: to, .. }) => {
                if target(to)? as usize >= code.len() {
                    return reject("retained control-flow target outside program");
                }
            }
            InstructionKind::OutsideBranch(ops::OutsideBranch { .. })
            | InstructionKind::OutsideSchedule(ops::OutsideSchedule { .. })
            | InstructionKind::OutsidePushTarget(ops::OutsidePushTarget { .. })
            | InstructionKind::MisalignedBranch(ops::MisalignedBranch { .. })
            | InstructionKind::MisalignedSchedule(ops::MisalignedSchedule { .. })
            | InstructionKind::MisalignedPushTarget(ops::MisalignedPushTarget { .. }) => {
                return outside()
            }
            InstructionKind::PopJump(ops::PopJump { .. })
            | InstructionKind::Brx(ops::Brx { .. })
            | InstructionKind::OutsideBrx(ops::OutsideBrx { .. }) => {
                return reject("unproven retained stack or indirect control flow");
            }
            _ => {}
        }
    }
    let mut bytes = header;
    bytes.extend_from_slice(&code);
    Ok(Program {
        bytes,
        stage: parsed.stage(),
        code_offset,
    })
}

// Container-declared blocks have an exact end. Preserve and validate every
// trailing bundle, including unreachable real instructions, rather than using
// EXIT to discard the rest of an allocation. Opaque control sections stay in
// separate artifacts; they are not Maxwell instruction streams.
fn validate_nw(
    file: &mut File,
    offset: u64,
    compute: bool,
    stage: headers::Stage,
) -> Result<Program, ProbeError> {
    if offset != 0 {
        return reject("unsupported bytes before container program record");
    }
    let mut program = validate(file, offset, compute)?;
    if program.stage != stage {
        return reject("container stage disagrees with record stage");
    }
    let end = file.metadata()?.len();
    let code_start = offset + program.code_offset as u64;
    let size = end
        .checked_sub(code_start)
        .ok_or_else(|| ProbeError::Reject("truncated stage block".into()))?;
    if size > MAX_CODE as u64 || size % 32 != 0 {
        return reject("unsupported container code extent");
    }
    let retained = program.bytes.len() - HEADER_SIZE;
    file.seek(SeekFrom::Start(code_start + retained as u64))?;
    // Mesa src/nouveau/compiler/nak/sm50.rs:3468-3471: scheduling word
    // followed by three 8-byte instruction slots. Validation preserves all four.
    for at in (retained..size as usize).step_by(32) {
        let mut bundle = [0; 32];
        file.read_exact(&mut bundle)?;
        if !scheduling_valid(u64::from_le_bytes(bundle[..8].try_into().unwrap())) {
            return reject("invalid trailing scheduling word");
        }
        for slot in [8, 16, 24] {
            validate_instruction(&bundle[slot..slot + 8], (at + slot) as u32)?;
        }
        program.bytes.extend_from_slice(&bundle);
    }
    for at in (0..size as usize).step_by(8).filter(|at| at % 32 != 0) {
        let instruction = validate_instruction(&program.bytes[HEADER_SIZE + at..], at as u32)?;
        match instruction.kind {
            InstructionKind::Branch(ops::Branch { target: to, .. })
            | InstructionKind::Schedule(ops::Schedule { target: to, .. })
            | InstructionKind::PushTarget(ops::PushTarget { target: to, .. }) => {
                if target(to)? as u64 >= size {
                    return reject("trailing control-flow target outside stage");
                }
            }
            InstructionKind::OutsideBranch(ops::OutsideBranch { .. })
            | InstructionKind::OutsideSchedule(ops::OutsideSchedule { .. })
            | InstructionKind::OutsidePushTarget(ops::OutsidePushTarget { .. })
            | InstructionKind::MisalignedBranch(ops::MisalignedBranch { .. })
            | InstructionKind::MisalignedSchedule(ops::MisalignedSchedule { .. })
            | InstructionKind::MisalignedPushTarget(ops::MisalignedPushTarget { .. }) => {
                return outside()
            }
            InstructionKind::PopJump(ops::PopJump { .. })
            | InstructionKind::Brx(ops::Brx { .. })
            | InstructionKind::OutsideBrx(ops::OutsideBrx { .. }) => {
                return reject("unproven trailing control flow")
            }
            _ => {}
        }
    }
    Ok(program)
}

// Diagnostic byte signatures only, not claims that a container is valid.
// Project experiment citations are in docs/reference-sources.md
const SIGNATURES: [&str; 9] = [
    "zstd",
    "lz4",
    "lz4-legacy",
    "zstd-or-lz4-skippable",
    "zlib",
    "Yaz0",
    "zip-local",
    "zip-directory",
    "zip-end",
];

fn signature(bytes: &[u8]) -> Option<usize> {
    match bytes.first().copied()? {
        // Synthetic zstd frame prefix.
        0x28 if bytes.starts_with(&[0x28, 0xb5, 0x2f, 0xfd]) => Some(0),
        // Synthetic LZ4 frame prefixes.
        4 if bytes.starts_with(&[4, 0x22, 0x4d, 0x18]) => Some(1),
        2 if bytes.starts_with(&[2, 0x21, 0x4c, 0x18]) => Some(2),
        // Both tools accept these
        // synthetic skippable frames. Do not infer which codec surrounds it.
        0x50..=0x5f if bytes.get(1..4) == Some(&[0x2a, 0x4d, 0x18]) => Some(3),
        // Exhaustive synthetic prefixes
        // establish CM=8, CINFO<=7, BE header divisible by 31 as a hint.
        first
            if first & 15 == 8
                && first >> 4 <= 7
                && bytes.len() >= 2
                && u16::from_be_bytes([first, bytes[1]]).is_multiple_of(31) =>
        {
            Some(4)
        }
        // Fixed ASCII query: docs/reference-sources.md.
        // Not a decoded header field or a claim of valid compressed content.
        b'Y' if bytes.starts_with(b"Yaz0") => Some(5),
        // Synthetic ZIP markers.
        b'P' if bytes.starts_with(b"PK\x03\x04") => Some(6),
        b'P' if bytes.starts_with(b"PK\x01\x02") => Some(7),
        b'P' if bytes.starts_with(b"PK\x05\x06") => Some(8),
        _ => None,
    }
}

#[derive(Default)]
struct SignatureCounts {
    counts: [u64; SIGNATURES.len()],
    examples: [Vec<u64>; SIGNATURES.len()],
}

impl SignatureCounts {
    fn add(&mut self, kind: usize, offset: u64) {
        self.counts[kind] += 1;
        if self.examples[kind].len() < 8 {
            self.examples[kind].push(offset);
        }
    }

    fn json(&self) -> Vec<Value> {
        self.counts
            .iter()
            .enumerate()
            .filter(|(_, count)| **count != 0)
            .map(|(kind, count)| {
                json!({
                    "signature": SIGNATURES[kind], "count": count,
                    "example_offsets": self.examples[kind],
                })
            })
            .collect()
    }
}

fn nearby(file: &mut File, offset: u64) -> io::Result<Vec<Value>> {
    let start = offset.saturating_sub(CONTEXT);
    file.seek(SeekFrom::Start(start))?;
    let mut bytes = Vec::new();
    file.take(offset - start + CONTEXT + OVERLAP as u64)
        .read_to_end(&mut bytes)?;
    Ok(
        (0..bytes.len().min((offset - start + CONTEXT + 1) as usize))
            .filter_map(|at| {
                signature(&bytes[at..]).map(|kind| {
                    json!({
                        "signature": SIGNATURES[kind], "offset": start + at as u64,
                    })
                })
            })
            .collect(),
    )
}

#[derive(Default)]
struct Counts {
    pak_entries_verified: u64,
    pak_entries_unverified: u64,
    files: u64,
    candidates: u64,
    accepted: u64,
    rejected: u64,
    unique: u64,
    translated: u64,
    translation_failed: u64,
    cache_emitted: u64,
    missing_state: u64,
    translation_seconds: f64,
    translation_workers: usize,
    skipped_symlinks: u64,
    nw_containers: u64,
    nw_variations: u64,
    nw_stage_failed: u64,
    nw_skipped: u64,
    nw_variation_skipped: u64,
}

// Replace output entries by rename, so a pre-existing output symlink cannot
// redirect a write into the input tree. A failed write leaves no partial asset.
struct PendingFile {
    temporary: PathBuf,
    destination: PathBuf,
    committed: bool,
}

impl PendingFile {
    fn new(destination: &Path) -> io::Result<(Self, File)> {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(io::Error::other)?
            .as_nanos();
        let temporary = destination.with_extension(format!(
            "{}-{stamp}-{}.partial",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        Ok((
            Self {
                temporary,
                destination: destination.to_path_buf(),
                committed: false,
            },
            file,
        ))
    }

    fn commit(mut self) -> io::Result<()> {
        fs::rename(&self.temporary, &self.destination)?;
        self.committed = true;
        Ok(())
    }
}

impl Drop for PendingFile {
    fn drop(&mut self) {
        if !self.committed {
            let _ = fs::remove_file(&self.temporary);
        }
    }
}

fn write_asset(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let (pending, mut file) = PendingFile::new(path)?;
    file.write_all(bytes)?;
    drop(file);
    pending.commit()
}

// Content hint: the existing scanner's record magic, at any offset, including
// across decompressor writes. This does not claim a shader container layout.
struct ShaderSpool<W: Write> {
    writer: W,
    window: u32,
    has_magic: bool,
}
impl<W: Write> Write for ShaderSpool<W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.writer.write_all(bytes)?;
        for &byte in bytes {
            self.window = (self.window << 8) | u32::from(byte);
            self.has_magic |= matches!(self.window, 0x7856_3412 | 0x7956_3412);
        }
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

struct NwStage {
    meta: Value,
    stage: Option<headers::Stage>,
}

enum RecordSource<'a> {
    Raw,
    Pak(&'a str, Integrity),
    Unity(&'a Value),
    Nw(&'a NwStage),
    IoStore(&'a Value),
    Unreal(&'a UeStage),
    UnrealRaw(&'a Value),
}

struct Scanner<'a> {
    root: PathBuf,
    output: PathBuf,
    translate: bool,
    cache_dir: Option<PathBuf>,
    states: super::cache_state::StateIndex,
    state_dirs: Vec<PathBuf>,
    translation_tasks: Vec<(String, headers::Stage)>,
    pak_all: bool,
    pak_key: Option<super::pak::AesKey>,
    seen: HashSet<String>,
    dictionaries: super::nw::Dictionaries,
    totals: Counts,
    manifest: BufWriter<File>,
    report: &'a mut dyn Write,
}

pub(super) fn stage_name(stage: headers::Stage) -> &'static str {
    match stage {
        headers::Stage::Vertex => "vertex",
        headers::Stage::Fragment => "fragment",
        headers::Stage::Compute => "compute",
        headers::Stage::Geometry => "geometry",
        headers::Stage::TessellationControl => "tessellation-control",
        headers::Stage::TessellationEvaluation => "tessellation-evaluation",
    }
}

impl Scanner<'_> {
    fn event(&mut self, value: Value) -> io::Result<()> {
        serde_json::to_writer(&mut self.manifest, &value)?;
        writeln!(self.manifest)
    }

    fn file(&mut self, path: &Path) -> io::Result<()> {
        let source = path.strip_prefix(&self.root).unwrap();
        // Keep non-UTF-8 Unix paths recoverable alongside the human-readable name.
        #[cfg(unix)]
        let source_bytes = {
            use std::os::unix::ffi::OsStrExt;
            source
                .as_os_str()
                .as_bytes()
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>()
        };
        #[cfg(not(unix))]
        let source_bytes = Value::Null;
        let source = source.to_string_lossy();
        if path
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("ucas"))
        {
            self.totals.files += 1;
            return self.event(json!({"kind": "iostore_data", "source_file": source,
                "source_path_hex": source_bytes, "status": "handled_by_utoc"}));
        }
        let mut archive = File::open(path)?;
        // AssetStudio/BundleFile.cs:72-75,93-99: UnityFS signature.
        // AssetStudio/SerializedFile.cs:43-65: 20/48-byte serialized headers.
        // References are pinned in docs/reference-sources.md
        let mut prefix = [0; 48];
        let prefix_len = archive.read(&mut prefix)?;
        archive.seek(SeekFrom::Start(0))?;
        if prefix[..prefix_len].starts_with(super::iostore::MAGIC)
            || path
                .extension()
                .is_some_and(|e| e.eq_ignore_ascii_case("utoc"))
        {
            self.totals.files += 1;
            return match super::iostore::Toc::open(path, self.pak_key.as_ref()) {
                Ok(toc) => self.iostore(path, &source, json!(source_bytes), toc),
                Err(error) => {
                    writeln!(self.report, "{source}: IoStore skipped: {error}")?;
                    self.event(json!({"kind": "iostore_skipped", "source_file": source,
                        "source_path_hex": source_bytes, "reason": error.to_string()}))
                }
            };
        }
        if super::ue_shaderlib::named(&source) {
            self.totals.files += 1;
            return self.ue_library(
                &source,
                json!(source_bytes),
                json!({"pak_entry": null, "integrity": null}),
                path,
            );
        }
        if super::nw::kind(&prefix[..prefix_len]).is_some()
            || super::nw::dictionary_hint(&source, &prefix[..prefix_len])
        {
            self.totals.files += 1;
            if archive.metadata()?.len() > super::nw::MAX_BYTES as u64 {
                return self.nw_skipped(&source, &json!([]), "container exceeds resource limit");
            }
            let mut bytes = Vec::new();
            archive
                .take(super::nw::MAX_BYTES as u64 + 1)
                .read_to_end(&mut bytes)?;
            if bytes.len() > super::nw::MAX_BYTES {
                return self.nw_skipped(&source, &json!([]), "container exceeds resource limit");
            }
            let mut budget = NwBudget::default();
            return self.nw(&source, &bytes, &[], &mut budget);
        }
        let unity_bundle = prefix[..prefix_len].starts_with(b"UnityFS\0");
        let unity_serialized = super::unity::serialized_name(&source)
            || super::unity::serialized_header_hint(
                &prefix[..prefix_len],
                archive.metadata()?.len(),
            );
        if unity_bundle || unity_serialized {
            self.totals.files += 1;
            if !unity_bundle {
                return self.unity_serialized_stream(&source, None, &mut archive);
            }
            return self.unity_bundle(&source, &mut archive);
        }
        match super::pak::Pak::open_with_key(&mut archive, self.pak_key.as_ref()) {
            Ok(Some(pak)) => return self.pak(path, &mut archive, pak),
            Err(error) => {
                self.totals.files += 1;
                writeln!(self.report, "{source}: pak skipped: {error}")?;
                return self.event(json!({"kind": "pak_skipped", "source_file": source,
                    "source_path_hex": source_bytes, "reason": error.to_string()}));
            }
            Ok(None) => {
                if path
                    .extension()
                    .is_some_and(|e| e.eq_ignore_ascii_case("pak"))
                {
                    self.totals.files += 1;
                    writeln!(
                        self.report,
                        "{source}: pak skipped: unsupported or missing footer"
                    )?;
                    return self.event(json!({"kind": "pak_skipped", "source_file": source,
                        "reason": "unsupported or missing footer"}));
                }
            }
        }
        self.records(path, &source, json!(source_bytes), RecordSource::Raw)
    }

    fn records(
        &mut self,
        path: &Path,
        source: &str,
        source_bytes: Value,
        member: RecordSource<'_>,
    ) -> io::Result<()> {
        let (pak_entry, integrity) = match member {
            RecordSource::Pak(name, integrity) => (Some(name), Some(integrity.as_str())),
            _ => (None, None),
        };
        let unity = if let RecordSource::Unity(meta) = member {
            Some(meta)
        } else {
            None
        };
        let nw = if let RecordSource::Nw(stage) = member {
            Some(stage)
        } else {
            None
        };
        let iostore = if let RecordSource::IoStore(meta) = member {
            Some(meta)
        } else {
            None
        };
        let unreal = if let RecordSource::Unreal(context) = member {
            Some(context)
        } else {
            None
        };
        let raw_library = if let RecordSource::UnrealRaw(meta) = member {
            Some(meta)
        } else {
            None
        };
        let unreal_meta = unreal.map(|s| &s.meta).or(raw_library);
        let pak_entry = pak_entry.or_else(|| unreal_meta.and_then(|s| s["pak_entry"].as_str()));
        // Project reporting policy: docs/reference-sources.md.
        // Parsed shaders and raw fallbacks inherit the entry's checksum evidence.
        let integrity = integrity.or_else(|| unreal_meta.and_then(|s| s["integrity"].as_str()));
        let iostore = iostore.or_else(|| unreal_meta.and_then(|s| s.get("iostore")));
        let label = if let Some(meta) = unreal_meta {
            ue_libraries::label(source, meta)
        } else if let Some(meta) = iostore {
            format!("{source}::chunk[{}]", meta["chunk_id"].as_str().unwrap())
        } else if let Some(stage) = nw {
            format!(
                "{source}::{}::variation[{}]::{}",
                stage.meta["container_path"],
                stage.meta["variation"],
                stage.stage.map(stage_name).unwrap_or("records")
            )
        } else if let Some(context) = unity {
            let inner = context["inner_file"]
                .as_str()
                .map(|s| format!("::{s}"))
                .unwrap_or_default();
            format!(
                "{source}{inner}::shader[{}]::platform[{}]::segment[{}]",
                context["path_id"], context["platform"], context["segment"]
            )
        } else {
            format!(
                "{source}{}",
                pak_entry.map(|s| format!("::{s}")).unwrap_or_default()
            )
        };
        let mut reader = BufReader::with_capacity(CHUNK, File::open(path)?);
        // Independent handles keep seeking probes from disturbing the stream.
        let mut probe = File::open(path)?;
        let mut counts = Counts::default();
        let mut signatures = SignatureCounts::default();
        let mut buffer = vec![0; CHUNK + OVERLAP];
        let mut carry = 0;
        let mut base = 0u64;
        let mut record_end = 0u64;
        loop {
            let read = reader.read(&mut buffer[carry..carry + CHUNK])?;
            let len = carry + read;
            let end = if read == 0 {
                len
            } else {
                len.saturating_sub(OVERLAP)
            };
            for at in 0..end {
                let bytes = &buffer[at..len];
                let offset = base + at as u64;
                // A proven record's instruction immediates can contain magic.
                // Skip its already validated extent only for bounded UE blobs.
                if unreal.is_some() && offset < record_end {
                    continue;
                }
                if let Some(kind) = signature(bytes) {
                    signatures.add(kind, offset);
                }
                // A declared BNSH stage block has one program record. Interior
                // magic can be a MOV immediate or constant data, not another
                // stage. Unsupported leading layouts fail instead of discarding
                // bytes before an apparent record (BnshLoader.cs:99-102 bounds
                // the entire bytecode block).
                if nw.is_some_and(|s| s.stage.is_some()) && offset != 0 {
                    continue;
                }
                // novena docs/reference-sources.md, LE magic.
                if !bytes.starts_with(&[0x78, 0x56, 0x34, 0x12])
                    && !bytes.starts_with(&[0x79, 0x56, 0x34, 0x12])
                {
                    continue;
                }
                counts.candidates += 1;
                let context = nearby(&mut probe, offset)?;
                let validation = if let Some(stage) = nw.and_then(|s| s.stage) {
                    validate_nw(&mut probe, offset, bytes[0] == 0x79, stage)
                } else {
                    validate(&mut probe, offset, bytes[0] == 0x79)
                };
                let validation = validation.and_then(|program| {
                    if unreal.is_some_and(|s| s.stage != program.stage) {
                        reject("Unreal library frequency disagrees with record stage")
                    } else {
                        Ok(program)
                    }
                });
                match validation {
                    Err(ProbeError::Io(error)) => return Err(error),
                    Err(ProbeError::Reject(reason)) => {
                        counts.rejected += 1;
                        self.event(json!({"kind": "rejected", "source_file": source,
                            "source_path_hex": source_bytes, "pak_entry": pak_entry, "integrity": integrity, "unreal": unreal_meta, "iostore": iostore, "unity": unity, "nw": nw.map(|s| &s.meta), "offset": offset,
                            "reason": reason, "nearby_signatures": context}))?;
                    }
                    Ok(program) => {
                        if unreal.is_some() {
                            record_end = offset
                                + (program.code_offset + program.bytes.len() - HEADER_SIZE) as u64;
                        }
                        counts.accepted += 1;
                        let hash = format!("{:x}", Sha256::digest(&program.bytes));
                        let name = format!("{hash}.bin");
                        let unique = self.seen.insert(hash.clone());
                        if unique {
                            self.totals.unique += 1;
                            write_asset(&self.output.join(&name), &program.bytes)?;
                            if self.translate {
                                self.translation_tasks.push((hash.clone(), program.stage));
                            }
                        }
                        self.event(json!({"kind": "program", "source_file": source,
                            "source_path_hex": source_bytes, "pak_entry": pak_entry, "integrity": integrity, "unreal": unreal_meta, "iostore": iostore, "unity": unity, "nw": nw.map(|s| &s.meta), "offset": offset,
                            "stage": stage_name(program.stage), "code_offset": program.code_offset,
                            "code_size": program.bytes.len() - HEADER_SIZE,
                            "record_size": program.code_offset + program.bytes.len() - HEADER_SIZE,
                            "size": program.bytes.len(), "sha256": hash, "output": name,
                            "unique": unique, "nearby_signatures": context,
                            "translation": "not_requested", "translation_error": null}))?;
                    }
                }
            }
            base += end as u64;
            if read == 0 {
                break;
            }
            buffer.copy_within(end..len, 0);
            carry = len - end;
        }
        if matches!(member, RecordSource::Raw) {
            self.totals.files += 1;
        }
        self.totals.candidates += counts.candidates;
        self.totals.accepted += counts.accepted;
        self.totals.rejected += counts.rejected;
        writeln!(
            self.report,
            "{label}: candidates={} accepted={} rejected={} signatures={}",
            counts.candidates,
            counts.accepted,
            counts.rejected,
            serde_json::to_string(&signatures.json())?
        )?;
        self.event(json!({"kind": if unreal.is_some() || raw_library.is_some_and(|m| !m["shader"].is_null()) { "ue_shader_scan" } else if iostore.is_some() { "iostore_chunk_scan" } else if nw.is_some_and(|s| s.stage.is_some()) { "nw_stage_scan" } else if nw.is_some() { "nw_member_scan" } else if unity.is_some() { "unity_blob_scan" } else if pak_entry.is_some() { "pak_entry_scan" } else { "file" }, "unreal": unreal_meta, "iostore": iostore, "unity": unity, "nw": nw.map(|s| &s.meta), "source_file": source, "pak_entry": pak_entry,
            "source_path_hex": source_bytes, "integrity": integrity, "candidates": counts.candidates,
            "accepted": counts.accepted, "rejected": counts.rejected,
            "signatures": signatures.json()}))
    }

    fn unity_skipped(
        &mut self,
        source: &str,
        inner: Option<&str>,
        reason: impl ToString,
    ) -> io::Result<()> {
        let reason = reason.to_string();
        writeln!(
            self.report,
            "{source}{}: Unity skipped: {reason}",
            inner.map(|s| format!("::{s}")).unwrap_or_default()
        )?;
        self.event(json!({"kind": "unity_skipped", "source_file": source, "inner_file": inner, "reason": reason}))
    }

    fn unity_bundle(&mut self, source: &str, input: &mut (impl Read + Seek)) -> io::Result<()> {
        let (pending, mut spool) = PendingFile::new(&self.output.join("unity-bundle"))?;
        let bundle = match super::unity::Bundle::read_stream(input, &mut spool) {
            Ok(bundle) => bundle,
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::InvalidData | io::ErrorKind::UnexpectedEof
                ) =>
            {
                return self.unity_skipped(source, None, error);
            }
            Err(error) => return Err(error),
        };
        drop(spool);
        let mut spool = File::open(&pending.temporary)?;
        self.event(json!({"kind": "unity_bundle", "source_file": source, "version": bundle.version,
            "unity_version": bundle.unity_version, "revision": bundle.revision, "flags": bundle.flags,
            "data_hash": bundle.data_hash, "files": bundle.files.len()}))?;
        for file in &bundle.files {
            let mut inner = super::unity::SubFile::new(&mut spool, file.range.clone());
            let mut prefix = [0; 48];
            let prefix_len = inner.read(&mut prefix)?;
            inner.seek(SeekFrom::Start(0))?;
            let serialized = super::unity::serialized_name(&file.name)
                || super::unity::serialized_header_hint(
                    &prefix[..prefix_len],
                    file.range.len() as u64,
                );
            self.event(json!({"kind": "unity_bundle_file", "source_file": source, "inner_file": file.name,
                "offset": file.range.start, "size": file.range.len(), "flags": file.flags, "serialized": serialized}))?;
            if serialized {
                self.unity_serialized_stream(source, Some(&file.name), &mut inner)?;
            }
        }
        drop(spool);
        drop(pending);
        Ok(())
    }

    fn unity_serialized_stream(
        &mut self,
        source: &str,
        inner: Option<&str>,
        input: &mut (impl Read + Seek),
    ) -> io::Result<()> {
        let serialized = match super::unity::SerializedFile::read_stream(input) {
            Ok(file) => file,
            Err(error) => return self.unity_skipped(source, inner, error),
        };
        self.unity_shaders(source, inner, &serialized, input)
    }

    fn unity_shaders(
        &mut self,
        source: &str,
        inner: Option<&str>,
        serialized: &super::unity::SerializedFile<'_>,
        input: &mut (impl Read + Seek),
    ) -> io::Result<()> {
        self.event(json!({"kind": "unity_serialized", "source_file": source, "inner_file": inner,
            "version": serialized.version, "unity_version": serialized.unity_version, "target": serialized.target}))?;
        for object in serialized.shaders() {
            let shader = match object.extract_stream(input) {
                Ok(shader) => shader,
                Err(error) => {
                    writeln!(
                        self.report,
                        "{source}{}::shader[{}]: skipped: {error}",
                        inner.map(|s| format!("::{s}")).unwrap_or_default(),
                        object.path_id
                    )?;
                    self.event(json!({"kind": "unity_shader", "source_file": source, "inner_file": inner,
                        "path_id": object.path_id, "status": "skipped", "reason": error.to_string()}))?;
                    continue;
                }
            };
            let before = (
                self.totals.candidates,
                self.totals.accepted,
                self.totals.rejected,
            );
            for blob in &shader.blobs {
                let context = json!({"inner_file": inner, "path_id": object.path_id, "shader_name": shader.name,
                    "platform": blob.platform, "segment": blob.segment, "compressed_blob_offset": blob.offset,
                    "compressed_length": blob.compressed_length, "decompressed_length": blob.decompressed_length});
                // Archive names and shader names are metadata only. Reuse the
                // seekable spool and strict raw-record validation unchanged.
                let (pending, mut file) = PendingFile::new(&self.output.join("unity-blob"))?;
                file.write_all(&blob.bytes)?;
                drop(file);
                self.records(
                    &pending.temporary,
                    source,
                    Value::Null,
                    RecordSource::Unity(&context),
                )?;
            }
            let candidates = self.totals.candidates - before.0;
            let accepted = self.totals.accepted - before.1;
            let rejected = self.totals.rejected - before.2;
            writeln!(self.report, "{source}{}::shader[{}] {:?}: blobs={} candidates={candidates} accepted={accepted} rejected={rejected}",
                inner.map(|s| format!("::{s}")).unwrap_or_default(), object.path_id, shader.name, shader.blobs.len())?;
            self.event(json!({"kind": "unity_shader", "source_file": source, "inner_file": inner,
                "path_id": object.path_id, "shader_name": shader.name, "status": "scanned", "blobs": shader.blobs.len(),
                "candidates": candidates, "accepted": accepted, "rejected": rejected}))?;
        }
        Ok(())
    }

    fn pak(&mut self, path: &Path, archive: &mut File, pak: super::pak::Pak) -> io::Result<()> {
        let source = path.strip_prefix(&self.root).unwrap().to_string_lossy();
        self.totals.files += 1;
        self.event(json!({"kind": "pak", "source_file": source, "version": pak.version,
            "mount_point": pak.mount, "entries": pak.entries.len(), "path_hash_seed": pak.path_hash_seed,
            "encryption_guid": pak.encryption_guid}))?;
        for (name, entry) in &pak.entries {
            let mut integrity = Integrity::Unverified;
            let reason = pak.skip_reason(entry);
            let meta = json!({"kind": "pak_entry", "source_file": source, "pak_entry": name,
                "archive_offset": entry.offset, "compressed_size": entry.compressed,
                "uncompressed_size": entry.uncompressed, "compression": pak.method(entry)});
            if let Some(reason) = reason {
                self.pak_entry_event(meta, "skipped", Some(reason), integrity)?;
                continue;
            }
            // One seekable spool at a time. PendingFile removes it on success,
            // rejection and unwinding; archive names never become output paths.
            let (pending, file) = PendingFile::new(&self.output.join("pak-entry"))?;
            let mut spool = ShaderSpool {
                writer: BufWriter::new(file),
                window: 0,
                has_magic: false,
            };
            if let Err(error) = pak.extract(archive, entry, &mut spool, &mut integrity) {
                self.pak_entry_event(meta, "failed", Some(error.to_string()), integrity)?;
                continue;
            }
            spool.writer.flush()?;
            let named = name.to_ascii_lowercase().ends_with(".ushaderbytecode");
            let selected = self.pak_all || named || spool.has_magic;
            drop(spool);
            if named {
                self.pak_entry_event(meta, "library_selected", None, integrity)?;
                self.ue_library(
                    &source,
                    Value::Null,
                    json!({"pak_entry": name, "integrity": integrity.as_str()}),
                    &pending.temporary,
                )?;
            } else if selected {
                self.pak_entry_event(meta, "scanned", None, integrity)?;
                self.records(
                    &pending.temporary,
                    &source,
                    Value::Null,
                    RecordSource::Pak(name, integrity),
                )?;
            } else {
                self.pak_entry_event(meta, "no_shader_hint", None, integrity)?;
            }
        }
        Ok(())
    }

    fn pak_entry_event(
        &mut self,
        mut meta: Value,
        status: &str,
        reason: Option<String>,
        integrity: Integrity,
    ) -> io::Result<()> {
        // docs/reference-sources.md: count each pak_entry once,
        // including skipped and failed entries.
        // Related scan/program events carry the same evidence but do not count again.
        match integrity {
            Integrity::Verified => self.totals.pak_entries_verified += 1,
            Integrity::Unverified => self.totals.pak_entries_unverified += 1,
        }
        meta["status"] = json!(status);
        meta["reason"] = json!(reason);
        meta["integrity"] = json!(integrity.as_str());
        writeln!(
            self.report,
            "{}::{}: {status} integrity={}{}",
            meta["source_file"].as_str().unwrap(),
            meta["pak_entry"].as_str().unwrap(),
            integrity.as_str(),
            reason.map(|r| format!(": {r}")).unwrap_or_default()
        )?;
        self.event(meta)
    }

    fn walk(&mut self) -> io::Result<()> {
        // Sort one directory at a time so discovery and manifest order are stable.
        // Do not follow symlinks or read generated output on subsequent runs.
        let mut directories = vec![sorted_directory(&self.root)?];
        while let Some(directory) = directories.last_mut() {
            let Some(entry) = directory.next() else {
                directories.pop();
                continue;
            };
            let path = entry;
            if path == self.output
                || self.cache_dir.as_ref() == Some(&path)
                || self.state_dirs.contains(&path)
            {
                continue;
            }
            let kind = fs::symlink_metadata(&path)?.file_type();
            if kind.is_symlink() {
                self.totals.skipped_symlinks += 1;
            } else if kind.is_dir() {
                directories.push(sorted_directory(&path)?);
            } else if kind.is_file() {
                self.file(&path)?;
            }
        }
        Ok(())
    }
}

fn sorted_directory(path: &Path) -> io::Result<std::vec::IntoIter<PathBuf>> {
    let mut paths = fs::read_dir(path)?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<io::Result<Vec<_>>>()?;
    paths.sort();
    Ok(paths.into_iter())
}

struct TranslationResult {
    identity: String,
    artifact: Option<String>,
    error: Option<String>,
    missing: &'static [&'static str],
    cache_key: Option<String>,
}

impl Scanner<'_> {
    fn translate_all<R: shadowbox::Run>(
        &mut self,
        run: R,
        manifest: PendingFile,
    ) -> io::Result<PendingFile> {
        if !self.translate {
            return Ok(manifest);
        }
        let start = Instant::now();
        // A repeated scan may have a different state set. Remove our previous
        // modules for the current programs in one directory pass, so a missing
        // or failed variant cannot leave a stale successful artifact behind.
        for entry in fs::read_dir(&self.output)? {
            let path = entry?.path();
            if path.extension().is_none_or(|extension| extension != "spv") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) else {
                continue;
            };
            let (hash, suffix) = stem
                .split_once('.')
                .map_or((stem, None), |(hash, key)| (hash, Some(key)));
            let valid_suffix = suffix.is_none_or(|key| {
                key.len() == 64
                    && key
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
            });
            if valid_suffix && self.seen.contains(hash) {
                fs::remove_file(path)?;
            }
        }
        let next = AtomicUsize::new(0);
        let workers = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(self.translation_tasks.len());
        let tasks = &self.translation_tasks;
        let output_dir = &self.output;
        let cache_dir = &self.cache_dir;
        let states = &self.states;
        // The queue stores hashes, not shader bytes. Each worker holds just one
        // program, and results are placed in discovery order after joining.
        let results = std::thread::scope(|scope| {
            let handles = (0..workers)
                .map(|_| {
                    scope.spawn(|| {
                        let mut results = Vec::new();
                        loop {
                            let index = next.fetch_add(1, Ordering::Relaxed);
                            let Some((hash, stage)) = tasks.get(index) else {
                                break;
                            };
                            let result = (|| -> io::Result<Vec<TranslationResult>> {
                                let bytes = fs::read(output_dir.join(format!("{hash}.bin")))?;
                                let input = shadowbox::header_prefixed_input(
                                    &bytes,
                                    shadowbox::TargetGeneration::Sm5,
                                )
                                .map_err(io::Error::other)?;
                                // Compare instruction bytes, excluding the AOT/cache envelopes.
                                // Keep AOT header bytes in the runtime identity.
                                let code_hash = format!("{:x}", Sha256::digest(input.code));
                                let absent = [super::cache_state::RecordedState::absent()];
                                let variants = states
                                    .get(&(*stage as u8, code_hash))
                                    .map_or(absent.as_slice(), Vec::as_slice);
                                let mut results = Vec::new();
                                let mut seen = HashSet::new();
                                for state in variants {
                                    let (key, variant) =
                                        state.classify(&input).map_err(io::Error::other)?;
                                    let missing = variant.needed_state();
                                    if !seen.insert((key.clone(), missing)) {
                                        continue;
                                    }
                                    let translated = variant.translate(run, *stage);
                                    let artifact = if variants.len() == 1 {
                                        output_dir.join(format!("{hash}.spv"))
                                    } else {
                                        output_dir.join(format!("{hash}.{key}.spv"))
                                    };
                                    let result = match translated {
                                        Ok((output, context)) => {
                                            let spirv = output
                                                .spirv
                                                .iter()
                                                .flat_map(|w| w.to_le_bytes())
                                                .collect::<Vec<_>>();
                                            write_asset(&artifact, &spirv)?;
                                            TranslationResult {
                                                identity: key,
                                                artifact: Some(
                                                    artifact
                                                        .file_name()
                                                        .unwrap()
                                                        .to_string_lossy()
                                                        .into_owned(),
                                                ),
                                                error: None,
                                                missing,
                                                cache_key: cache_dir
                                                    .as_ref()
                                                    .map(|dir| {
                                                        shadowbox::emit_translation_cache(
                                                            dir, &input, *context, &output,
                                                        )
                                                    })
                                                    .transpose()?,
                                            }
                                        }
                                        Err(error) => {
                                            match fs::remove_file(&artifact) {
                                                Ok(()) => {}
                                                Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                                                Err(e) => return Err(e),
                                            }
                                            TranslationResult {
                                                identity: key,
                                                artifact: None,
                                                error: Some(error.to_string()),
                                                missing,
                                                cache_key: None,
                                            }
                                        }
                                    };
                                    results.push(result);
                                }
                                Ok(results)
                            })();
                            results.push((index, result));
                        }
                        results
                    })
                })
                .collect::<Vec<_>>();
            let mut results = Vec::new();
            for handle in handles {
                results.extend(
                    handle
                        .join()
                        .map_err(|_| io::Error::other("translation worker panicked"))?,
                );
            }
            Ok::<_, io::Error>(results)
        })?;
        self.totals.translation_seconds = start.elapsed().as_secs_f64();
        self.totals.translation_workers = workers;
        let mut ordered = results;
        ordered.sort_by_key(|(index, _)| *index);
        let mut results = HashMap::new();
        for (index, result) in ordered {
            let result = result?;
            for variant in &result {
                if variant.error.is_some() {
                    self.totals.translation_failed += 1;
                } else {
                    self.totals.translated += 1;
                }
                self.totals.cache_emitted += u64::from(variant.cache_key.is_some());
                self.totals.missing_state += u64::from(!variant.missing.is_empty());
            }
            results.insert(tasks[index].0.clone(), result);
        }
        // Stream the provisional manifest back through the result map, preserving
        // discovery order and duplicate occurrences without buffering events.
        self.manifest.flush()?;
        let (published, file) = PendingFile::new(&self.output.join("manifest.jsonl"))?;
        let mut writer = BufWriter::new(file);
        for line in BufReader::new(File::open(&manifest.temporary)?).lines() {
            let mut event: Value = serde_json::from_str(&line?)?;
            if event["kind"] == "program" {
                let variants = &results[event["sha256"].as_str().unwrap()];
                event["state_variants"] = json!(variants
                    .iter()
                    .map(|v| json!({
                        "translation_key": v.identity, "spirv_output": v.artifact,
                        "cache_key": v.cache_key, "missing_state_keys": v.missing,
                        "translation_error": v.error,
                    }))
                    .collect::<Vec<_>>());
                event["cache_keys"] = json!(variants
                    .iter()
                    .filter_map(|v| v.cache_key.as_ref())
                    .collect::<Vec<_>>());
                let result = &variants[0];
                event["translation"] = json!(if event["unique"] != true {
                    "duplicate_see_first_occurrence"
                } else if result.error.is_some() {
                    "failed"
                } else {
                    "translated"
                });
                event["spirv_output"] = json!(result.artifact);
                event["translation_error"] = json!(result.error);
                event["missing_state_keys"] = json!(result.missing);
                event["cache_key"] = json!(result.cache_key);
                event["cache_status"] = json!(if variants.iter().any(|v| v.error.is_none())
                    && variants.iter().any(|v| v.error.is_some())
                {
                    "partial"
                } else if !result.missing.is_empty() {
                    "missing_state"
                } else if result.error.is_some() {
                    "translation_failed"
                } else if result.cache_key.is_some() {
                    "emitted"
                } else {
                    "not_requested"
                });
                if event["unique"] == true {
                    for variant in variants {
                        if let Some(error) = &variant.error {
                            writeln!(
                                self.report,
                                "{}@{}: translation failed: {error}",
                                event["source_file"], event["offset"]
                            )?;
                        }
                    }
                }
            }
            serde_json::to_writer(&mut writer, &event)?;
            writeln!(writer)?;
        }
        self.manifest = writer;
        drop(manifest);
        Ok(published)
    }
}

#[cfg(test)]
fn scan(root: &Path, output: &Path, translate: bool, report: &mut dyn Write) -> io::Result<Counts> {
    scan_options(root, output, translate, false, None, report)
}

#[cfg(test)]
fn scan_options(
    root: &Path,
    output: &Path,
    translate: bool,
    pak_all: bool,
    pak_key: Option<super::pak::AesKey>,
    report: &mut dyn Write,
) -> io::Result<Counts> {
    scan_with_state(
        shadowbox::QuietRun,
        root,
        output,
        ScanOptions {
            translate,
            pak_all,
            pak_key,
            ..Default::default()
        },
        report,
    )
}

#[derive(Default)]
struct ScanOptions {
    translate: bool,
    pak_all: bool,
    cache_dir: Option<PathBuf>,
    state_from: Vec<PathBuf>,
    pak_key: Option<super::pak::AesKey>,
}

fn scan_with_state<R: shadowbox::Run>(
    run: R,
    root: &Path,
    output: &Path,
    options: ScanOptions,
    report: &mut dyn Write,
) -> io::Result<Counts> {
    let ScanOptions {
        translate,
        pak_all,
        cache_dir,
        state_from,
        pak_key,
    } = options;
    let mut states = super::cache_state::StateIndex::new();
    for source in &state_from {
        for (key, variants) in super::cache_state::index(source).map_err(io::Error::other)? {
            states.entry(key).or_default().extend(variants);
        }
    }
    let translate = translate || cache_dir.is_some() || !state_from.is_empty();
    let root = root.canonicalize()?;
    if !root.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "input must be a directory",
        ));
    }
    fs::create_dir_all(output)?;
    let output = output.canonicalize()?;
    if root.starts_with(&output) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "output must not contain the input directory",
        ));
    }
    let cache_dir = cache_dir
        .map(|dir| {
            fs::create_dir_all(&dir)?;
            let dir = dir.canonicalize()?;
            if root.starts_with(&dir) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "cache must not contain the input directory",
                ));
            }
            Ok(dir)
        })
        .transpose()?;
    let (manifest_output, manifest_file) = PendingFile::new(&output.join("manifest.jsonl"))?;
    let mut scanner = Scanner {
        manifest: BufWriter::new(manifest_file),
        root,
        output,
        translate,
        cache_dir,
        translation_tasks: Vec::new(),
        states,
        state_dirs: state_from
            .iter()
            .map(|p| p.canonicalize())
            .collect::<io::Result<_>>()?,
        pak_all,
        pak_key,
        seen: HashSet::new(),
        dictionaries: super::nw::Dictionaries::default(),
        totals: Counts::default(),
        report,
    };
    scanner.event(json!({"kind": "format", "version": 1,
        "hash": "sha256", "shader_format": "sm5-header-prefixed",
        "max_code_bytes": MAX_CODE, "context_bytes": CONTEXT,
        "translate": translate, "emit_cache": scanner.cache_dir.is_some(), "pak_all": pak_all}))?;
    scanner.load_dictionaries()?;
    scanner.walk()?;
    let manifest_output = scanner.translate_all(run, manifest_output)?;
    let totals = &scanner.totals;
    let summary = json!({"kind": "summary", "files": totals.files,
        "pak_entries_verified": totals.pak_entries_verified,
        "pak_entries_unverified": totals.pak_entries_unverified,
        "candidates": totals.candidates, "accepted": totals.accepted,
        "rejected": totals.rejected, "unique": totals.unique,
        "duplicates": totals.accepted - totals.unique, "translated": totals.translated,
        "translation_failed": totals.translation_failed,
        "cache_emitted": totals.cache_emitted, "complete": totals.translated, "missing_state": totals.missing_state,
        "translation_workers": totals.translation_workers,
        "translation_seconds": totals.translation_seconds,
        "programs_per_second": (totals.translated + totals.translation_failed - totals.missing_state) as f64 / totals.translation_seconds.max(f64::MIN_POSITIVE),
        "skipped_symlinks": totals.skipped_symlinks,
        "nw_containers": totals.nw_containers, "nw_variations": totals.nw_variations,
        "nw_stage_failed": totals.nw_stage_failed,
        "nw_skipped": totals.nw_skipped, "nw_variation_skipped": totals.nw_variation_skipped});
    scanner.event(summary)?;
    scanner.manifest.flush()?;
    drop(scanner.manifest);
    manifest_output.commit()?;
    let totals = scanner.totals;
    writeln!(scanner.report, "summary: files={} candidates={} accepted={} rejected={} unique={} duplicates={} translated={} translation_failed={} skipped_symlinks={} pak_entries_verified={} pak_entries_unverified={}",
        totals.files, totals.candidates, totals.accepted, totals.rejected, totals.unique,
        totals.accepted - totals.unique, totals.translated, totals.translation_failed, totals.skipped_symlinks,
        totals.pak_entries_verified, totals.pak_entries_unverified)?;
    if translate {
        writeln!(
            scanner.report,
            "translation: workers={} programs/s={:.2} cache_emitted={} complete={} missing_state={}",
            totals.translation_workers,
            (totals.translated + totals.translation_failed - totals.missing_state) as f64
                / totals.translation_seconds.max(f64::MIN_POSITIVE),
            totals.cache_emitted,
            totals.translated,
            totals.missing_state
        )?;
    }
    Ok(totals)
}

/// Where `--pak-aes-key-file` or `--pak-aes-key-env` says the key lives.
enum KeySource<'a> {
    File(&'a str),
    Env(&'a str),
}

// Never include source names, OS values, contents or parsing details in
// errors. Bound file reads and erase the temporary text on drop.
fn read_key_file(source: &Path) -> Result<super::pak::AesKey, String> {
    let mut data = zeroize::Zeroizing::new(Vec::new());
    File::open(source)
        .and_then(|f| f.take(1025).read_to_end(&mut data))
        .map_err(|_| "cannot read pak AES key file".to_owned())?;
    if data.len() > 1024 {
        return Err("pak AES key file exceeds size limit".to_owned());
    }
    super::pak::AesKey::parse(&data).map_err(|e| e.to_string())
}

fn read_key_env(source: &str) -> Result<super::pak::AesKey, String> {
    let value = zeroize::Zeroizing::new(
        std::env::var(source)
            .map_err(|_| "cannot read pak AES key environment variable".to_owned())?,
    );
    let data = zeroize::Zeroizing::new(value.as_bytes().to_vec());
    super::pak::AesKey::parse(&data).map_err(|e| e.to_string())
}

fn parse_flags(flags: &[String]) -> Result<ScanOptions, String> {
    let mut translate = false;
    let mut pak_all = false;
    let mut key_source = None;
    let mut cache_dir = None;
    let mut state_from = Vec::new();
    let mut flags = flags.iter();
    while let Some(flag) = flags.next() {
        match flag.as_str() {
            "--translate" if !translate => translate = true,
            "--pak-all" if !pak_all => pak_all = true,
            "--emit-cache" if cache_dir.is_none() => {
                cache_dir = Some(PathBuf::from(
                    flags.next().ok_or("--emit-cache needs a directory")?,
                ));
            }
            "--state-from" => {
                state_from.push(PathBuf::from(
                    flags.next().ok_or("--state-from needs a directory")?,
                ));
            }
            "--pak-aes-key-file" if key_source.is_none() => {
                let source = flags.next().ok_or_else(|| super::usage().to_owned())?;
                key_source = Some(KeySource::File(source));
            }
            "--pak-aes-key-env" if key_source.is_none() => {
                let source = flags.next().ok_or_else(|| super::usage().to_owned())?;
                key_source = Some(KeySource::Env(source));
            }
            _ => return Err(super::usage().into()),
        }
    }
    let key = key_source
        .map(|source| match source {
            KeySource::File(path) => read_key_file(Path::new(path)),
            KeySource::Env(name) => read_key_env(name),
        })
        .transpose()?;
    Ok(ScanOptions {
        translate,
        pak_all,
        cache_dir,
        state_from,
        pak_key: key,
    })
}

/// `shadowbox aot-scan`.
pub(super) struct AotScan {
    root: PathBuf,
    output: PathBuf,
    options: ScanOptions,
}

impl super::Subcommand for AotScan {
    fn parse(args: &[String]) -> Result<Self, String> {
        let [root, output, flags @ ..] = args else {
            return Err(super::usage().into());
        };
        Ok(Self {
            root: PathBuf::from(root),
            output: PathBuf::from(output),
            options: parse_flags(flags)?,
        })
    }

    fn run<R: shadowbox::Run>(self, run: R) -> Result<(), String> {
        scan_with_state(
            run,
            &self.root,
            &self.output,
            self.options,
            &mut io::stdout().lock(),
        )
        .map_err(|error| format!("aot-scan: {error}"))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests;

// Package orchestration uses the same scanner and emit-cache path, with all
// recorded state sources. Package manifests omit run-dependent timings.
pub(super) fn pack<R: shadowbox::Run>(
    run: R,
    root: &Path,
    output: &Path,
    sources: &[PathBuf],
    key_file: Option<&Path>,
) -> Result<(), String> {
    let options = ScanOptions {
        translate: false,
        pak_all: false,
        cache_dir: Some(output.join("cache")),
        state_from: sources.to_vec(),
        pak_key: key_file.map(read_key_file).transpose()?,
    };
    scan_with_state(run, root, output, options, &mut io::sink()).map_err(|e| e.to_string())?;
    let path = output.join("manifest.jsonl");
    let (pending, file) = PendingFile::new(&path).map_err(|e| e.to_string())?;
    let mut normalized = BufWriter::new(file);
    for line in BufReader::new(File::open(&path).map_err(|e| e.to_string())?).lines() {
        let mut event: Value =
            serde_json::from_str(&line.map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        if event["kind"] == "summary" {
            for field in [
                "translation_workers",
                "translation_seconds",
                "programs_per_second",
            ] {
                event.as_object_mut().unwrap().remove(field);
            }
        }
        serde_json::to_writer(&mut normalized, &event).map_err(|e| e.to_string())?;
        writeln!(normalized).map_err(|e| e.to_string())?;
    }
    normalized.flush().map_err(|e| e.to_string())?;
    drop(normalized);
    pending.commit().map_err(|e| e.to_string())
}
