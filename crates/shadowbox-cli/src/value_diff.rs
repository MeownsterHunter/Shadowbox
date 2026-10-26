// Project-owned differential harness. This does not decode Maxwell fields.
mod gpu;
mod inputs;
use inputs::Inputs;
use shadowbox::ir::ops;
mod module;
mod source;

use serde_json::{json, Value};
use std::{
    fs,
    io::{Read, Write},
    os::unix::process::CommandExt,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    sync::atomic::{AtomicUsize, Ordering},
    thread,
    time::{Duration, Instant},
};

pub(super) const MAX_PROGRAM: u64 = 1024 * 1024 + 128 + 65536 + 40;
const MAX_MODULE: u64 = 4 * 1024 * 1024;
const MAX_LOG: usize = 65536;

/// `shadowbox value-diff`; also the comparison settings its workers share.
pub(super) struct ValueDiff {
    old: Side,
    new: Side,
    directory: PathBuf,
    jobs: usize,
    settings: Settings,
}

/// What every comparison shares: the input seed and each process's timeout.
pub(super) struct Settings {
    seed: u64,
    timeout: Duration,
}

/// What `--old` or `--new` names: a built translator, or a crate this run
/// builds under the CARGO_TARGET_DIR the environment named at the start.
enum Side {
    Built(PathBuf),
    Crate {
        manifest: PathBuf,
        target_base: PathBuf,
    },
}

impl Side {
    fn parse(path: &Path, cargo_target: Option<&Path>) -> Result<Self, String> {
        let path = fs::canonicalize(path).map_err(|e| format!("{}: {e}", path.display()))?;
        if path.is_file() && path.file_name().and_then(|x| x.to_str()) != Some("Cargo.toml") {
            return Ok(Self::Built(path));
        }
        let manifest = if path.is_dir() {
            path.join("Cargo.toml")
        } else {
            path
        };
        if !manifest.is_file() {
            return Err(format!("missing manifest {}", manifest.display()));
        }
        let target_base =
            cargo_target.ok_or("crate builds require CARGO_TARGET_DIR outside the source tree")?;
        if !target_base.is_absolute() {
            return Err("CARGO_TARGET_DIR must be absolute".into());
        }
        if target_base.starts_with(manifest.parent().unwrap()) {
            return Err("CARGO_TARGET_DIR must be outside the source tree".into());
        }
        Ok(Self::Crate {
            manifest,
            target_base: target_base.to_owned(),
        })
    }

    /// The translator binary, built into `scratch`'s directory under the
    /// target base for a crate. A crate's build directory is guarded in
    /// `guards` until every translator process has exited.
    fn binary(
        self,
        scratch: &Path,
        side: &str,
        guards: &mut Vec<Scratch>,
    ) -> Result<PathBuf, String> {
        let (manifest, target_base) = match self {
            Self::Built(path) => return Ok(path),
            Self::Crate {
                manifest,
                target_base,
            } => (manifest, target_base),
        };
        let build_dir = target_base.join(scratch.file_name().unwrap());
        let target = build_dir.join(side);
        guards.push(Scratch(build_dir));
        fs::create_dir_all(&target).map_err(|e| format!("create build directory: {e}"))?;
        let result = bounded(
            Command::new("cargo")
                .args(["build", "--bin", "shadowbox", "--manifest-path"])
                .arg(&manifest)
                .env("CARGO_TARGET_DIR", &target)
                .current_dir(manifest.parent().unwrap()),
            Duration::from_secs(120),
        );
        match result {
            Ok(result) if result.success => Ok(target.join("debug/shadowbox")),
            Ok(result) => Err(format!("{side} crate build failed: {}", result.log)),
            Err(e) => Err(format!("{side} crate build: {e}")),
        }
    }
}

impl crate::Subcommand for ValueDiff {
    fn parse(args: &[String]) -> Result<Self, String> {
        let mut old = None;
        let mut new = None;
        let mut directory = None;
        let mut seed = 0;
        let mut jobs = 1;
        let mut timeout = 5000;
        let mut i = 0;
        while i < args.len() {
            let arg = &args[i];
            if arg.starts_with("--") {
                i += 1;
                let value = args.get(i).ok_or_else(|| format!("{arg} needs a value"))?;
                match arg.as_str() {
                    "--old" => old = Some(PathBuf::from(value)),
                    "--new" => new = Some(PathBuf::from(value)),
                    "--seed" => seed = value.parse().map_err(|_| "invalid seed")?,
                    "--jobs" => jobs = value.parse().map_err(|_| "invalid jobs")?,
                    "--timeout-ms" => timeout = value.parse().map_err(|_| "invalid timeout")?,
                    _ => return Err(format!("unknown value-diff option {arg}")),
                }
            } else if directory.replace(PathBuf::from(arg)).is_some() {
                return Err("value-diff requires exactly one program directory".into());
            }
            i += 1;
        }
        if !(1..=64).contains(&jobs) || !(1..=60000).contains(&timeout) {
            return Err("jobs must be 1..64 and timeout-ms must be 1..60000".into());
        }
        // The environment names where crate builds go, read once here.
        let cargo_target = std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from);
        let old: PathBuf = old.ok_or("value-diff requires --old")?;
        let new: PathBuf = new.ok_or("value-diff requires --new")?;
        Ok(Self {
            old: Side::parse(&old, cargo_target.as_deref())?,
            new: Side::parse(&new, cargo_target.as_deref())?,
            directory: directory.ok_or("value-diff requires a program directory")?,
            jobs,
            settings: Settings {
                seed,
                timeout: Duration::from_millis(timeout),
            },
        })
    }

    fn run<R: shadowbox::Run>(self, run: R) -> Result<(), String> {
        let ValueDiff {
            old,
            new,
            directory,
            jobs,
            settings,
        } = self;
        let mut paths = Vec::new();
        for entry in fs::read_dir(&directory).map_err(|e| format!("program directory: {e}"))? {
            let entry = entry.map_err(|e| e.to_string())?;
            if entry.file_type().map_err(|e| e.to_string())?.is_file()
                && entry.path().extension().and_then(|e| e.to_str()) == Some("bin")
            {
                paths.push(fs::canonicalize(entry.path()).map_err(|e| e.to_string())?);
            }
        }
        paths.sort();
        if paths.is_empty() {
            return Err("no .bin programs found".into());
        }
        let scratch = Scratch::new()?;
        // Guard the build directories too, including failed builds.
        let mut build_dirs = Vec::new();
        let old = old.binary(&scratch.0, "old", &mut build_dirs)?;
        let new = new.binary(&scratch.0, "new", &mut build_dirs)?;
        let next = AtomicUsize::new(0);
        let rows = thread::scope(|scope| {
            let handles = (0..jobs.min(paths.len()))
                .map(|_| {
                    scope.spawn(|| {
                        let mut rows = Vec::new();
                        loop {
                            let index = next.fetch_add(1, Ordering::Relaxed);
                            let Some(path) = paths.get(index) else { break };
                            rows.push((index, compare(run, path, &old, &new, &settings)));
                        }
                        rows
                    })
                })
                .collect::<Vec<_>>();
            handles
                .into_iter()
                .flat_map(|h| h.join().unwrap())
                .collect::<Vec<_>>()
        });
        drop(build_dirs);
        let mut rows = rows;
        rows.sort_by_key(|r| r.0);
        let mut tally = Tally::default();
        for (_, (outcome, row)) in rows {
            tally.add(&outcome);
            println!("{row}");
        }
        let Tally {
            equal,
            different,
            uncheckable,
            translation_failed,
        } = tally;
        println!(
            "{}",
            json!({"summary": {
                "equal": equal, "different": different, "uncheckable": uncheckable,
                "translation_failed": translation_failed, "seed": settings.seed
            }})
        );
        if different + uncheckable + translation_failed != 0 {
            return Err("value-diff did not establish equality for every program".into());
        }
        Ok(())
    }
}

/// TMPDIR is explicitly honored because Rust's temp_dir uses TMPDIR on Unix.
pub(super) struct Scratch(pub(super) PathBuf);

impl Scratch {
    pub(super) fn new() -> Result<Self, String> {
        static SERIAL: AtomicUsize = AtomicUsize::new(0);
        let base = std::env::temp_dir();
        for _ in 0..100 {
            let path = base.join(format!(
                "shadowbox-value-diff-{}-{}",
                std::process::id(),
                SERIAL.fetch_add(1, Ordering::Relaxed)
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Ok(Self(path)),
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(e) => return Err(format!("create scratch {}: {e}", path.display())),
            }
        }
        Err("cannot allocate a unique scratch directory".into())
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[derive(Debug)]
pub(super) struct ProcessOutput {
    pub(super) success: bool,
    pub(super) log: String,
}

fn drain(mut reader: impl Read) -> String {
    let mut saved = Vec::new();
    let mut buffer = [0; 4096];
    while let Ok(count) = reader.read(&mut buffer) {
        if count == 0 {
            break;
        }
        let keep = count.min(MAX_LOG - saved.len());
        saved.extend_from_slice(&buffer[..keep]);
    }
    String::from_utf8_lossy(&saved).trim().to_string()
}

/// Each phase owns a process group. Killing it also closes inherited pipes
/// held by compiler or shell descendants, so joining the readers is bounded.
pub(super) fn bounded(command: &mut Command, timeout: Duration) -> Result<ProcessOutput, String> {
    let mut child = command
        .process_group(0)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env("MESA_SHADER_CACHE_DISABLE", "true")
        .spawn()
        .map_err(|e| format!("start {:?}: {e}", command.get_program()))?;
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let out = thread::spawn(move || drain(stdout));
    let err = thread::spawn(move || drain(stderr));
    let start = Instant::now();
    let result = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status.success()),
            Ok(None) if start.elapsed() < timeout => thread::sleep(Duration::from_millis(5)),
            Ok(None) => break Err(format!("timeout after {} ms", timeout.as_millis())),
            Err(e) => break Err(format!("wait: {e}")),
        }
    };
    // Safety: the child creates its own group with PGID equal to its PID.
    // kill(2) with a negative PID addresses that group, never our own group.
    unsafe { libc::kill(-(child.id() as i32), libc::SIGKILL) };
    let _ = child.wait();
    let log = [out.join().unwrap(), err.join().unwrap()]
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    result.map(|success| ProcessOutput { success, log })
}

pub(super) fn read_limited(path: &Path, limit: u64) -> Result<Vec<u8>, String> {
    let mut bytes = Vec::new();
    fs::File::open(path)
        .map_err(|e| format!("read {}: {e}", path.display()))?
        .take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    if bytes.len() as u64 > limit {
        return Err(format!("{} exceeds {limit} bytes", path.display()));
    }
    Ok(bytes)
}

pub(super) fn translate<R: shadowbox::Run>(
    run: R,
    binary: &Path,
    program: &Path,
    output: &Path,
    timeout: Duration,
    cb1: Option<&Path>,
) -> Result<(), Rejected> {
    let mut command = Command::new(binary);
    command.arg("translate").arg(program);
    if let Some(cb1) = cb1 {
        command.arg("--cb1").arg(cb1);
    }
    let result = bounded(
        command
            .args(["--header-prefixed", "--no-memory-cache", "-o"])
            .arg(output),
        timeout,
    )?;
    if !result.success {
        return Err(Rejected::Failed(result.log));
    }
    let words = super::words(&read_limited(output, MAX_MODULE)?)?;
    if source_requires_reconvergence(run, &read_limited(program, MAX_PROGRAM)?)? {
        module::require_maximal_reconvergence(&words)?;
    }
    let result = bounded(
        Command::new("spirv-val")
            .args(["--target-env", "vulkan1.2"])
            .arg(output),
        timeout,
    )?;
    if !result.success {
        return Err(Rejected::Failed(format!("invalid SPIR-V: {}", result.log)));
    }
    Ok(())
}

/// Why a translated module cannot be value-checked.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Rejected {
    /// Translation, module reading or validation failed with this text.
    Failed(String),
    /// A token-region source whose module lacks the maximal-reconvergence
    /// contract: the comparison cannot be trusted, so it is uncheckable.
    TokenRegionUncheckable,
}

impl std::fmt::Display for Rejected {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Failed(text) => f.write_str(text),
            Self::TokenRegionUncheckable => f.write_str(
                "uncheckable: token-region source requires MaximallyReconvergesKHR on main and SPV_KHR_maximal_reconvergence",
            ),
        }
    }
}

impl From<String> for Rejected {
    fn from(text: String) -> Self {
        Self::Failed(text)
    }
}

impl From<Rejected> for String {
    fn from(rejected: Rejected) -> Self {
        rejected.to_string()
    }
}

/// The result of comparing one program, typed where it is decided. `json`
/// gives the row exactly as it is printed or handed to the gate.
#[derive(Debug)]
enum Outcome {
    Equal {
        compared: usize,
        excluded: usize,
    },
    Different {
        first_difference: Value,
    },
    Uncheckable(Uncheckable),
    TranslationFailed {
        old: Option<String>,
        new: Option<String>,
    },
}

#[derive(Debug)]
enum Uncheckable {
    Reason(String),
    /// One or both GPU captures failed.
    Captures {
        old: Option<String>,
        new: Option<String>,
    },
}

impl Outcome {
    fn json(&self) -> Value {
        match self {
            Self::Equal { compared, excluded } => {
                json!({"status": "equal", "compared_bytes": compared, "excluded_bytes": excluded})
            }
            Self::Different { first_difference } => {
                json!({"status": "different", "first_difference": first_difference})
            }
            Self::Uncheckable(Uncheckable::Reason(reason)) => {
                json!({"status": "uncheckable", "reason": reason})
            }
            Self::Uncheckable(Uncheckable::Captures { old, new }) => {
                json!({"status": "uncheckable", "old": old, "new": new})
            }
            Self::TranslationFailed { old, new } => {
                json!({"status": "translation-failed", "old": old, "new": new})
            }
        }
    }
}

/// Summary counts, one per outcome kind.
#[derive(Default)]
struct Tally {
    equal: usize,
    different: usize,
    uncheckable: usize,
    translation_failed: usize,
}

impl Tally {
    fn add(&mut self, outcome: &Outcome) {
        let count = match outcome {
            Outcome::Equal { .. } => &mut self.equal,
            Outcome::Different { .. } => &mut self.different,
            Outcome::Uncheckable(_) => &mut self.uncheckable,
            Outcome::TranslationFailed { .. } => &mut self.translation_failed,
        };
        *count += 1;
    }
}

/// The token-region rejection of one side, if that side has one.
fn token_region(side: &Result<(), Rejected>) -> Option<&Rejected> {
    match side {
        Err(rejected @ Rejected::TokenRegionUncheckable) => Some(rejected),
        Ok(()) | Err(Rejected::Failed(_)) => None,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Capture {
    set: u32,
    binding: u32,
    bytes: Vec<u8>,
    defined: Vec<bool>,
    output: Option<String>,
}

fn first_difference(old: &[Capture], new: &[Capture]) -> Option<Value> {
    if old
        .iter()
        .map(|c| (c.set, c.binding, c.bytes.len(), &c.output))
        .collect::<Vec<_>>()
        != new
            .iter()
            .map(|c| (c.set, c.binding, c.bytes.len(), &c.output))
            .collect::<Vec<_>>()
    {
        return Some(
            json!({"location": if old.iter().chain(new).any(|c| c.output.is_some()) { "stage output interface" } else { "storage interface" }, "reason": "set, binding, extent or output identity changed"}),
        );
    }
    for (old, new) in old.iter().zip(new) {
        if let Some(offset) = old
            .defined
            .iter()
            .zip(&new.defined)
            .position(|(a, b)| a != b)
        {
            return Some(json!({
                "location": "definedness", "set": old.set, "binding": old.binding,
                "output": old.output, "byte": offset, "old": old.defined[offset], "new": new.defined[offset],
            }));
        }
        for (offset, (&a, &b)) in old.bytes.iter().zip(&new.bytes).enumerate() {
            if old.defined[offset] && new.defined[offset] && a != b {
                return Some(json!({
                    "location": if old.output.is_some() { "stage output" } else { "storage" }, "output": old.output, "set": old.set, "binding": old.binding,
                    "byte": offset, "old": a, "new": b,
                }));
            }
        }
    }
    None
}

fn load_capture(path: &Path) -> Result<Vec<Capture>, String> {
    let value: Value = serde_json::from_slice(&read_limited(path, 16 * 1024 * 1024)?)
        .map_err(|e| format!("capture: {e}"))?;
    let buffers = value.as_array().ok_or("capture must be an array")?;
    let mut result = Vec::new();
    for buffer in buffers {
        let field = |key| {
            buffer[key]
                .as_u64()
                .and_then(|x| u32::try_from(x).ok())
                .ok_or_else(|| format!("invalid capture {key}"))
        };
        let bytes = buffer["bytes"]
            .as_array()
            .ok_or("invalid capture bytes")?
            .iter()
            .map(|x| {
                x.as_u64()
                    .and_then(|n| u8::try_from(n).ok())
                    .ok_or("invalid capture byte")
            })
            .collect::<Result<Vec<_>, _>>()?;
        let defined = buffer["defined"]
            .as_array()
            .ok_or("invalid capture mask")?
            .iter()
            .map(|x| x.as_bool().ok_or("invalid defined bit"))
            .collect::<Result<Vec<_>, _>>()?;
        if bytes.len() != defined.len() {
            return Err("capture mask length differs from byte length".into());
        }
        result.push(Capture {
            output: buffer["output"].as_str().map(str::to_owned),
            set: field("set")?,
            binding: field("binding")?,
            bytes,
            defined,
        });
    }
    Ok(result)
}

fn execute(
    module: &Path,
    output: &Path,
    options: &Settings,
    inputs: &Inputs,
) -> Result<Vec<Capture>, String> {
    let mut command = Command::new(std::env::current_exe().map_err(|e| e.to_string())?);
    command
        .arg("__value-diff-worker")
        .arg(module)
        .arg(output)
        .arg(options.seed.to_string());
    let path = output.with_extension("inputs.json");
    fs::write(
        &path,
        serde_json::to_vec(inputs).map_err(|e| e.to_string())?,
    )
    .map_err(|e| format!("snapshot capture inputs: {e}"))?;
    command.arg("--inputs").arg(path);
    let result = bounded(&mut command, options.timeout)?;
    if !result.success {
        return Err(result.log);
    }
    load_capture(output)
}

// Missing source definitions must not become an equality claim merely
// because an emitter initializes registers to zero. Ryujinx
// Translation/Ssa.cs:250-282 makes those definitions undefined.
// Graphics retain the straight-line subset. Compute control flow additionally
// uses the must-defined CFG proof in source.rs.
// It rejects other sources rather than trying to infer source taint from
// optimized SPIR-V where that information has already been erased.
fn prove_source<R: shadowbox::Run>(run: R, bytes: &[u8]) -> Result<u32, String> {
    prove_source_with_constant_buffer1(run, bytes, None)
}

fn prove_source_with_constant_buffer1<R: shadowbox::Run>(
    run: R,
    bytes: &[u8],
    cb1: Option<&[u8]>,
) -> Result<u32, String> {
    use shadowbox::ir::{InstructionKind as K, Source, TexOp};
    let input = shadowbox::header_prefixed_input(bytes, shadowbox::TargetGeneration::Sm5)
        .map_err(|e| e.to_string())?;
    let mut header = bytes[..80].to_vec();
    header.resize(128, 0);
    let parsed = shadowbox::headers::parse(&super::words(&header)?)
        .map_err(|e| format!("source header: {e:?}"))?;
    if input.code.is_empty() || !input.code.len().is_multiple_of(32) {
        return Err("source proof requires complete Maxwell bundles".into());
    }
    let instructions = shadowbox::decoder::decode_stream(run, &super::words(input.code)?);
    if parsed.stage() == shadowbox::headers::Stage::Compute
        && instructions.iter().any(|x| {
            x.predicate.is_some()
                || matches!(
                    x.kind,
                    K::Branch(ops::Branch { .. })
                        | K::OutsideBranch(ops::OutsideBranch { .. })
                        | K::Brx(ops::Brx { .. })
                        | K::OutsideBrx(ops::OutsideBrx { .. })
                        | K::BrxRegister(ops::BrxRegister { .. })
                        | K::S2R(ops::S2R { .. })
                        | K::Vote(ops::Vote { .. })
                        | K::Shuffle(ops::Shuffle { .. })
                        | K::LocalStore(ops::LocalStore { .. })
                        | K::SharedStore(ops::SharedStore { .. })
                        | K::GlobalMemory(ops::GlobalMemory { .. })
                        | K::Reconverge(ops::Reconverge { .. })
                        | K::Schedule(ops::Schedule { .. })
                        | K::ResolvedSchedule(ops::ResolvedSchedule { .. })
                        | K::OutsideSchedule(ops::OutsideSchedule { .. })
                        | K::PushTarget(ops::PushTarget { .. })
                        | K::ResolvedPushTarget(ops::ResolvedPushTarget { .. })
                        | K::OutsidePushTarget(ops::OutsidePushTarget { .. })
                )
        })
    {
        source::prove(&instructions)?;
        return Ok(5);
    }
    let mut defined = [false; 256];
    defined[255] = true;
    let mut exited = false;
    for instruction in instructions {
        if instruction.predicate.is_some() {
            return Err("source proof does not support predication".into());
        }
        // Static user-attribute reads are defined host inputs. ALD/AST lane
        // counts and addressing: Mesa nak/sm50.rs:2863-2876,2883-2897;
        // semantics: Ryujinx Instructions/InstEmitAttribute.cs:81-100,130-141;
        // user attribute byte range: Translation/AttributeConsts.cs:31-33.
        // Output reads require source output taint, which is not implemented.
        if !exited && parsed.stage() != shadowbox::headers::Stage::Compute {
            if let K::AttributeLoad(ops::AttributeLoad {
                dst,
                vertex: shadowbox::ir::PrimitiveVertex::Current,
                address,
                components,
                lanes: _,
            }) = instruction.kind
            {
                if address.is_multiple_of(4)
                    && (0x80..0x280).contains(&address)
                    && address as usize + components as usize * 4 <= 0x280
                    && (1..=4).contains(&components)
                    && dst.0 as usize + components as usize <= 255
                {
                    defined[dst.0 as usize..dst.0 as usize + components as usize].fill(true);
                    continue;
                }
            }
        }
        // Bound vector TEX.LZ uses A for coordinates with an integer layer
        // first for arrays, and B for optional Dref. Ryujinx
        // Instructions/InstEmitTexture.cs:207-210,229-258; vector packing:
        // Mesa src/nouveau/compiler/nak_nir_lower_tex.c:350-375.
        // Mode/dimension/mask/source fields: Mesa nak/sm50.rs:2105-2119,2121-2131,
        // 2169-2197. Scalar pairs, bindless and other operands need separate
        // source proofs. All implicit coordinate registers are checked.
        if !exited {
            if let K::Texture(ops::Texture {
                op: TexOp::Sample { lod: 1 },
                src0,
                src1,
                dim,
                mask,
                scalar: false,
                packed_f16: false,
                offset: false,
                lod_clamp: false,
                multisample: false,
                depth_compare,
                nodep: false,
                scalar_dst: None,
                bound_handle: Some(_),
                ..
            }) = instruction.kind
            {
                let count = match dim {
                    0 => 1,
                    1 | 2 => 2,
                    3 | 4 | 6 => 3,
                    7 => 4,
                    _ => return Err("unsupported texture source dimension".into()),
                };
                if src0.0 as usize + count > 256 || mask == 0 {
                    return Err("texture register extent unsupported".into());
                }
                let reads = (src0.0 as usize..src0.0 as usize + count)
                    .chain(depth_compare.then_some(src1.0 as usize));
                if let Some(register) = reads.into_iter().find(|&r| !defined[r]) {
                    return Err(format!("uninitialized texture source R{register}"));
                }
                for register in 0..255 {
                    if instruction.kind.writes_register(register) {
                        defined[register as usize] = true;
                    }
                }
                continue;
            }
        }
        let mut reads = instruction.kind.sources();
        let dst = match instruction.kind.legacy_form() {
            K::Nop(ops::Nop) => None,
            K::Exit(ops::Exit { cc_test: 0xf }) => {
                // Fragment colors leave through packed register vec4s rather
                // than AST: Ryujinx Translation/EmitterContext.cs:482-524.
                // Mask/depth slots: TranslatorContext.cs:150-164. Check the
                // complete vec4, since Shadowbox exports every component.
                if let shadowbox::headers::ShaderHeader::Fragment(fragment) = &parsed {
                    let colors = (0..8)
                        .filter(|target| fragment.targets() & (15 << (target * 4)) != 0)
                        .count()
                        * 4;
                    let reads = (0..colors)
                        .chain(fragment.writes_sample_mask().then_some(colors))
                        .chain(fragment.writes_depth().then_some(colors + 1));
                    if let Some(register) = reads.into_iter().find(|&r| !defined[r]) {
                        return Err(format!("uninitialized fragment output R{register}"));
                    }
                }
                exited = true;
                continue;
            }
            _ if exited => {
                return Err("source proof does not support instructions after EXIT".into())
            }
            // OUT emit/cut order: Ryujinx InstEmitAttribute.cs:253-295;
            // stream/handle fields: Mesa nak/sm50.rs:3257-3284. The module
            // proof observes every event instead of dropping it at return.
            K::GeometryOut(ops::GeometryOut { .. })
                if parsed.stage() == shadowbox::headers::Stage::Geometry =>
            {
                None
            }
            // Patch AST uses the same register reads as vertex AST, following
            // Ryujinx InstEmitAttribute.cs:106-141. Patch identity is retained
            // by the stage-interface proof and in every output capture key.
            K::AttributeStore(ops::AttributeStore {
                data,
                offset: None,
                components,
                patch,
                ..
            }) if (!patch || parsed.stage() == shadowbox::headers::Stage::TessellationControl)
                && parsed.stage() != shadowbox::headers::Stage::Compute
                && (1..=4).contains(&components)
                && data.0 as usize + components as usize <= 256 =>
            {
                for i in 0..components {
                    reads.push(Source::Register(shadowbox::ir::Register(data.0 + i)));
                }
                None
            }
            K::Mov(ops::Mov {
                dst,
                quad_lanes: 0xf,
                ..
            }) => Some(dst),
            K::IAdd(ops::IAdd {
                dst, lhs, x: false, ..
            }) => {
                reads.push(Source::Register(lhs));
                Some(dst)
            }
            K::IntegerMul(ops::IntegerMul { dst, lhs, .. }) => {
                reads.push(Source::Register(lhs));
                Some(dst)
            }
            _ => {
                return Err(format!(
                    "source definedness proof unsupported at {:#x}: {instruction}",
                    instruction.address
                ))
            }
        };
        // Saved CB1 reads must fit the snapshot, never invented padding.
        // Ryujinx DiskCache/DiskCacheGpuAccessor.cs:62-69. MOV cbuf word
        // addressing: Mesa nak/sm50.rs:361-368,1943-1946.
        if let Some(cb1) = cb1 {
            for source in &reads {
                if let Source::ConstantBuffer { binding: 1, offset } = source {
                    if usize::from(*offset) + 4 > cb1.len() {
                        return Err("CB1 read exceeds supplied snapshot".into());
                    }
                }
            }
        }
        if let Some(register) = reads.iter().find_map(|source| match source {
            Source::Register(register) if !defined[register.0 as usize] => Some(register),
            _ => None,
        }) {
            return Err(format!(
                "uninitialized source R{} at {:#x}; source-to-output taint is not supported yet",
                register.0, instruction.address
            ));
        }
        if let Some(dst) = dst {
            defined[dst.0 as usize] = true;
        }
    }
    if !exited {
        return Err("source proof requires an unconditional EXIT".into());
    }
    Ok(match parsed.stage() {
        shadowbox::headers::Stage::Vertex => 0,
        shadowbox::headers::Stage::Fragment => 4,
        shadowbox::headers::Stage::Compute => 5,
        shadowbox::headers::Stage::Geometry => 3,
        shadowbox::headers::Stage::TessellationControl => 1,
        shadowbox::headers::Stage::TessellationEvaluation => 2,
    })
}

fn constant_buffer1(bytes: &[u8], sidecar: &Path) -> Result<Option<Vec<u8>>, String> {
    let input = shadowbox::header_prefixed_input(bytes, shadowbox::TargetGeneration::Sm5)
        .map_err(|e| e.to_string())?;
    let embedded = shadowbox::constant_buffer1_snapshot(input.program_record)
        .map_err(|e| e.to_string())?
        .map(|words| {
            words
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect::<Vec<_>>()
        });
    let supplied = if sidecar.is_file() {
        let bytes = read_limited(sidecar, 65536)?;
        if !bytes.len().is_multiple_of(4) {
            return Err("CB1 sidecar byte length must be a multiple of four".into());
        }
        Some(bytes)
    } else {
        None
    };
    if embedded
        .as_ref()
        .is_some_and(|a| supplied.as_ref().is_some_and(|b| a != b))
    {
        return Err("CB1 sidecar disagrees with record snapshot".into());
    }
    let result = embedded.or(supplied);
    if result.as_ref().is_some_and(|bytes| bytes.len() > 65536) {
        return Err("CB1 snapshot exceeds 65536 bytes".into());
    }
    Ok(result)
}

fn compare<R: shadowbox::Run>(
    run: R,
    path: &Path,
    old: &Path,
    new: &Path,
    options: &Settings,
) -> (Outcome, Value) {
    let result = (|| -> Result<Outcome, String> {
        let temp = Scratch::new()?;
        let (program, sidecar) = super::gate::snapshot(path, &temp.0)?;
        let bytes = read_limited(&program, MAX_PROGRAM)?;
        let inputs = Inputs::load(&bytes, &program)?;
        let old_spv = temp.0.join("old.spv");
        let new_spv = temp.0.join("new.spv");
        let a = translate(
            run,
            old,
            &program,
            &old_spv,
            options.timeout,
            sidecar.as_deref(),
        );
        let b = translate(
            run,
            new,
            &program,
            &new_spv,
            options.timeout,
            sidecar.as_deref(),
        );
        match (a, b) {
            (Ok(()), Ok(())) => compare_modules(
                run,
                &bytes,
                Modules {
                    old: &old_spv,
                    new: &new_spv,
                },
                &temp.0,
                options,
                execute,
                &inputs,
            ),
            (a, b) => match token_region(&a).or(token_region(&b)) {
                Some(rejected) => Err(rejected.to_string()),
                None => Ok(Outcome::TranslationFailed {
                    old: a.err().map(String::from),
                    new: b.err().map(String::from),
                }),
            },
        }
    })();
    let outcome = result.unwrap_or_else(|e| Outcome::Uncheckable(Uncheckable::Reason(e)));
    let mut row = outcome.json();
    row["program"] = json!(path);
    (outcome, row)
}

// Resolved consumers encode token-region participation. Ryujinx Instructions/
// InstEmitFlowControl.cs:239-275; project reconvergence.rs:49-75 only activates
// regions for consumed push targets. Ordinary branch shaders need no mode.
fn source_requires_reconvergence<R: shadowbox::Run>(run: R, bytes: &[u8]) -> Result<bool, String> {
    let input = shadowbox::header_prefixed_input(bytes, shadowbox::TargetGeneration::Sm5)
        .map_err(|e| e.to_string())?;
    Ok(
        shadowbox::decoder::decode_stream(run, &super::words(input.code)?)
            .iter()
            .any(|x| {
                matches!(
                    x.kind,
                    shadowbox::ir::InstructionKind::Reconverge(ops::Reconverge { .. })
                )
            }),
    )
}

type Execute = fn(&Path, &Path, &Settings, &Inputs) -> Result<Vec<Capture>, String>;

/// The two SPIR-V modules a comparison runs.
#[derive(Clone, Copy)]
struct Modules<'p> {
    old: &'p Path,
    new: &'p Path,
}

// Reuse the same proof and captures for gate without translating a second time.
fn compare_modules<R: shadowbox::Run>(
    run: R,
    bytes: &[u8],
    Modules {
        old: old_spv,
        new: new_spv,
    }: Modules<'_>,
    scratch: &Path,
    options: &Settings,
    execute: Execute,
    inputs: &Inputs,
) -> Result<Outcome, String> {
    let stage = if inputs.cb1.is_some() {
        prove_source_with_constant_buffer1(run, bytes, inputs.cb1.as_deref())?
    } else {
        prove_source(run, bytes)?
    };
    let input = shadowbox::header_prefixed_input(bytes, shadowbox::TargetGeneration::Sm5)
        .map_err(|e| e.to_string())?;
    let xs = shadowbox::decoder::decode_stream(run, &super::words(input.code)?);
    let uses_global = xs.iter().any(|x| {
        matches!(
            x.kind,
            shadowbox::ir::InstructionKind::GlobalMemory(ops::GlobalMemory { .. })
        )
    });
    let uses_reconvergence = xs.iter().any(|x| {
        matches!(
            x.kind,
            shadowbox::ir::InstructionKind::Reconverge(ops::Reconverge { .. })
        )
    });
    for path in [old_spv, new_spv] {
        let words = super::words(&read_limited(path, MAX_MODULE)?)?;
        if uses_reconvergence {
            module::require_maximal_reconvergence(&words)?;
        }
        let plan = module::Plan::parse(&words, options.seed, inputs)?;
        if uses_global
            && inputs.storage.is_empty()
            && !plan
                .buffers
                .iter()
                .any(|b| b.kind == module::BufferKind::Physical)
        {
            return Err("descriptor global memory requires a storage capture sidecar".into());
        }
        if plan.stage != stage {
            return Err("translated stage differs from source header".into());
        }
    }
    let a = execute(old_spv, &scratch.join("old.json"), options, inputs);
    let b = execute(new_spv, &scratch.join("new.json"), options, inputs);
    let (a, b) = match (a, b) {
        (Ok(a), Ok(b)) => (a, b),
        (a, b) => {
            return Ok(Outcome::Uncheckable(Uncheckable::Captures {
                old: a.err(),
                new: b.err(),
            }))
        }
    };
    let compared = a
        .iter()
        .zip(&b)
        .map(|(a, b)| {
            a.defined
                .iter()
                .zip(&b.defined)
                .filter(|(x, y)| **x && **y)
                .count()
        })
        .sum::<usize>();
    if let Some(difference) = first_difference(&a, &b) {
        return Ok(Outcome::Different {
            first_difference: difference,
        });
    }
    if compared == 0 {
        return Ok(Outcome::Uncheckable(Uncheckable::Reason(
            "no defined output bytes".into(),
        )));
    }
    let excluded = a.iter().map(|c| c.bytes.len()).sum::<usize>() - compared;
    Ok(Outcome::Equal { compared, excluded })
}

// Gate runs this entire proof in a bounded subprocess, including source decoding
// and module analysis, which must not hold up the corpus worker pool.
/// `shadowbox __value-diff-compare`, spawned by the gate.
pub(super) struct CompareWorker {
    program: PathBuf,
    old: PathBuf,
    new: PathBuf,
    output: PathBuf,
    scratch: PathBuf,
    seed: u64,
    timeout: Duration,
}

impl crate::Subcommand for CompareWorker {
    fn parse(args: &[String]) -> Result<Self, String> {
        let [program, old, new, output, seed, timeout] = args else {
            return Err("invalid value comparison worker arguments".into());
        };
        let scratch = Path::new(output)
            .parent()
            .ok_or("missing comparison directory")?;
        Ok(Self {
            program: PathBuf::from(program),
            old: PathBuf::from(old),
            new: PathBuf::from(new),
            output: PathBuf::from(output),
            scratch: scratch.to_path_buf(),
            seed: seed.parse().map_err(|_| "invalid worker seed")?,
            timeout: Duration::from_millis(timeout.parse().map_err(|_| "invalid worker timeout")?),
        })
    }

    fn run<R: shadowbox::Run>(self, run: R) -> Result<(), String> {
        let options = Settings {
            seed: self.seed,
            timeout: self.timeout,
        };
        let bytes = read_limited(&self.program, MAX_PROGRAM)?;
        let inputs = Inputs::load(&bytes, &self.program)?;
        let row = compare_modules(
            run,
            &bytes,
            Modules {
                old: &self.old,
                new: &self.new,
            },
            &self.scratch,
            &options,
            execute_in_group,
            &inputs,
        )
        .unwrap_or_else(|reason| Outcome::Uncheckable(Uncheckable::Reason(reason)));
        fs::write(&self.output, row.json().to_string())
            .map_err(|e| format!("write comparison: {e}"))
    }
}

/// `shadowbox __value-diff-worker`, spawned for one GPU capture.
pub(super) struct CaptureWorker {
    input: String,
    output: String,
    inputs: Inputs,
    seed: u64,
}

impl crate::Subcommand for CaptureWorker {
    fn parse(args: &[String]) -> Result<Self, String> {
        let [input, output, seed, rest @ ..] = args else {
            return Err("invalid value-diff worker arguments".into());
        };
        let inputs = match rest {
            [] => Inputs::default(),
            [path] => Inputs {
                cb1: Some(read_limited(Path::new(path), 65536)?),
                ..Default::default()
            },
            [flag, path] if flag == "--inputs" => {
                serde_json::from_slice(&read_limited(Path::new(path), 1024 * 1024)?)
                    .map_err(|e| format!("capture inputs: {e}"))?
            }
            _ => return Err("invalid capture worker arguments".into()),
        };
        Ok(Self {
            input: input.clone(),
            output: output.clone(),
            inputs,
            seed: seed.parse().map_err(|_| "invalid worker seed")?,
        })
    }

    fn run<R: shadowbox::Run>(self, _: R) -> Result<(), String> {
        let captures = capture(
            &self.input,
            &self.output,
            self.seed,
            validate_worker,
            &self.inputs,
        )?;
        let result = captures
            .iter()
            .map(|c| {
                json!({
                    "set": c.set, "binding": c.binding, "bytes": c.bytes, "defined": c.defined, "output": c.output
                })
            })
            .collect::<Vec<_>>();
        fs::File::create(&self.output)
            .map_err(|e| e.to_string())?
            .write_all(serde_json::to_string(&result).unwrap().as_bytes())
            .map_err(|e| e.to_string())
    }
}

// A gate comparison owns one process group. Run both GPU captures here and
// let validator children inherit that group, so a timeout kills the entire
// comparison instead of leaving a separately grouped GPU worker behind.
fn execute_in_group(
    module: &Path,
    output: &Path,
    options: &Settings,
    inputs: &Inputs,
) -> Result<Vec<Capture>, String> {
    capture(
        module.to_str().ok_or("non-UTF-8 module path")?,
        output.to_str().ok_or("non-UTF-8 capture path")?,
        options.seed,
        validate_in_group,
        inputs,
    )
}

fn validate_in_group(path: &Path) -> Result<(), String> {
    let result = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.2"])
        .arg(path)
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("start spirv-val: {e}"))?;
    if !result.status.success() {
        return Err(format!(
            "invalid worker SPIR-V: {}",
            String::from_utf8_lossy(&result.stderr)
        ));
    }
    Ok(())
}

fn validate_worker(path: &Path) -> Result<(), String> {
    let result = bounded(
        Command::new("spirv-val")
            .args(["--target-env", "vulkan1.2"])
            .arg(path),
        Duration::from_secs(2),
    )?;
    if !result.success {
        return Err(format!("invalid worker SPIR-V: {}", result.log));
    }
    Ok(())
}

fn capture(
    input: &str,
    output: &str,
    seed: u64,
    validate: fn(&Path) -> Result<(), String>,
    inputs: &Inputs,
) -> Result<Vec<Capture>, String> {
    validate(Path::new(input))?;
    let bytes = read_limited(Path::new(input), MAX_MODULE)?;
    let words = super::words(&bytes)?;
    let plan = module::Plan::parse(&words, seed, inputs)?;
    let mut observed = Vec::new();
    if let Some(graphics) = &plan.graphics {
        observed.extend([
            ("observed", &graphics.words),
            ("partner", &graphics.partner),
        ]);
        if let Some(partner) = &graphics.tess_partner {
            observed.push(("tess-partner", partner));
        }
    }
    if let Some(compute) = &plan.compute_words {
        observed.push(("observed-compute", compute));
    }
    for (name, module) in observed {
        let path = Path::new(output).with_extension(format!("{name}.spv"));
        fs::write(
            &path,
            module
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .map_err(|e| e.to_string())?;
        let validation = validate(&path);
        let _ = fs::remove_file(&path);
        validation?;
    }
    gpu::execute(&words, &plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Register MOV: Mesa nak/sm50.rs:1936-1941,1954,217-235,321-333;
    // EXIT/NOP: same file:3101-3112,3178-3187; bundle:3418-3428.
    // Header envelope: shadowbox/src/lib.rs:53-58,178-192.
    fn source_program(register: u8) -> Vec<u8> {
        let mut bytes = vec![0; 80];
        let mov = (0x5c98_u64 << 48) | (u64::from(register) << 20) | (15 << 39) | (7 << 16);
        let exit = (0xe300_u64 << 48) | (7 << 16) | 15;
        let nop = (0x50b0_u64 << 48) | (7 << 16) | (15 << 8);
        for word in [0, mov, exit, nop] {
            bytes.extend_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    #[test]
    fn cb1_source_reads_require_saved_words_and_ignore_the_record_trailer() {
        // MOV cbuf binding34..39 and word offset20..34, full lanes39..43:
        // Mesa nak/sm50.rs:361-368,1943-1946. Saved-word bounds follow
        // Ryujinx DiskCacheGpuAccessor.cs:62-69.
        let mut source = source_program(255);
        let mov = (0x4c98_u64 << 48) | (1 << 34) | (1 << 20) | (15 << 39) | (7 << 16);
        source[88..96].copy_from_slice(&mov.to_le_bytes());
        let snapshot = [0xff; 8];
        assert!(
            prove_source_with_constant_buffer1(shadowbox::QuietRun, &source, Some(&snapshot))
                .is_ok()
        );
        for bytes in [&snapshot[..4], &snapshot[..0]] {
            assert!(
                prove_source_with_constant_buffer1(shadowbox::QuietRun, &source, Some(bytes))
                    .unwrap_err()
                    .contains("exceeds supplied snapshot")
            );
        }
        let embedded = shadowbox::encode_header_prefixed_with_constant_buffer1(
            &source,
            shadowbox::TargetGeneration::Sm5,
            &snapshot,
        )
        .unwrap();
        assert_eq!(
            prove_source_with_constant_buffer1(shadowbox::QuietRun, &embedded, Some(&snapshot))
                .unwrap(),
            5
        );
    }

    #[test]
    fn source_proof_does_not_accept_emitter_zero_initialization_as_defined() {
        assert!(prove_source(shadowbox::QuietRun, &source_program(255)).is_ok());
        assert!(prove_source(shadowbox::QuietRun, &source_program(1))
            .unwrap_err()
            .contains("uninitialized source R1"));
    }

    // ALD/AST fields: Mesa nak/sm50.rs:2863-2876,2885-2897;
    // predicate/destination/register: same file:217-235,321-333.
    // Vertex header: Mesa nak/sph.rs:204-215; headers.rs:73-81.
    #[test]
    fn attribute_proof_checks_every_source_and_destination_component() {
        let ald =
            (0xefd8_u64 << 48) | (3 << 47) | (255 << 39) | (0x80 << 20) | (7 << 16) | (255 << 8);
        let ast = (0xeff0_u64 << 48)
            | (3 << 47)
            | (255 << 39)
            | (1 << 32)
            | (0x70 << 20)
            | (7 << 16)
            | (255 << 8);
        let exit = (0xe300_u64 << 48) | (7 << 16) | 15;
        let source = |first: u64| {
            let mut bytes = vec![0; 80];
            bytes[..4].copy_from_slice(&(1_u32 << 10).to_le_bytes());
            for word in [0, first, ast, exit] {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
            bytes
        };
        assert_eq!(prove_source(shadowbox::QuietRun, &source(ald)).unwrap(), 0);
        let mov = (0x0100_u64 << 48) | (7 << 16) | (15 << 12);
        assert!(prove_source(shadowbox::QuietRun, &source(mov))
            .unwrap_err()
            .contains("uninitialized source R1"));
    }

    #[test]
    fn fragment_exit_reads_are_not_defined_by_emitter_initialization() {
        // Fragment target mask: Mesa nak/sph.rs:185-188. Output register
        // packing: Ryujinx Translation/EmitterContext.cs:482-524.
        let mut source = source_program(255);
        source[..4].copy_from_slice(&(5_u32 << 10).to_le_bytes());
        source[72..76].copy_from_slice(&15_u32.to_le_bytes());
        assert!(prove_source(shadowbox::QuietRun, &source)
            .unwrap_err()
            .contains("uninitialized fragment output R1"));
    }

    #[test]
    fn source_proof_requires_complete_bundles() {
        let mut source = source_program(255);
        source.pop();
        assert!(prove_source(shadowbox::QuietRun, &source)
            .unwrap_err()
            .contains("complete Maxwell bundles"));
    }

    #[test]
    fn comparison_keeps_defined_bytes_of_partially_undefined_output() {
        let capture = Capture {
            output: None,
            set: 1,
            binding: 3,
            bytes: vec![9, 8, 7, 6],
            defined: vec![false, true, true, true],
        };
        let mut changed = capture.clone();
        changed.bytes[0] = 0;
        assert_eq!(
            first_difference(
                std::slice::from_ref(&capture),
                std::slice::from_ref(&changed)
            ),
            None
        );
        changed.bytes[2] = 0;
        assert_eq!(first_difference(&[capture], &[changed]).unwrap()["byte"], 2);
    }

    #[test]
    fn changing_definedness_is_a_difference() {
        let a = Capture {
            output: None,
            set: 0,
            binding: 0,
            bytes: vec![1],
            defined: vec![true],
        };
        let mut b = a.clone();
        b.defined[0] = false;
        assert_eq!(
            first_difference(&[a], &[b]).unwrap()["location"],
            "definedness"
        );
    }

    #[test]
    fn subprocess_timeout_kills_descendants_holding_pipes() {
        let start = Instant::now();
        let result = bounded(
            Command::new("sh").args(["-c", "sleep 60 & wait"]),
            Duration::from_millis(30),
        );
        assert!(result.unwrap_err().contains("timeout"));
        assert!(start.elapsed() < Duration::from_secs(3));
    }

    #[test]
    fn outcomes_keep_their_rows_and_count_by_kind() {
        let outcomes = [
            Outcome::Equal {
                compared: 4,
                excluded: 2,
            },
            Outcome::Different {
                first_difference: json!({"byte": 0}),
            },
            Outcome::Uncheckable(Uncheckable::Reason("no defined output bytes".into())),
            Outcome::Uncheckable(Uncheckable::Captures {
                old: Some("x".into()),
                new: None,
            }),
            Outcome::TranslationFailed {
                old: None,
                new: Some("y".into()),
            },
        ];
        let rows = outcomes
            .iter()
            .map(|outcome| outcome.json().to_string())
            .collect::<Vec<_>>();
        assert_eq!(
            rows,
            [
                r#"{"compared_bytes":4,"excluded_bytes":2,"status":"equal"}"#,
                r#"{"first_difference":{"byte":0},"status":"different"}"#,
                r#"{"reason":"no defined output bytes","status":"uncheckable"}"#,
                r#"{"new":null,"old":"x","status":"uncheckable"}"#,
                r#"{"new":"y","old":null,"status":"translation-failed"}"#,
            ]
        );
        let mut tally = Tally::default();
        outcomes.iter().for_each(|outcome| tally.add(outcome));
        assert_eq!(
            (
                tally.equal,
                tally.different,
                tally.uncheckable,
                tally.translation_failed
            ),
            (1, 1, 2, 1)
        );
    }

    #[test]
    fn only_a_token_region_rejection_makes_a_failed_side_uncheckable() {
        let token = Err(Rejected::TokenRegionUncheckable);
        let failed = Err(Rejected::Failed(
            "uncheckable: token-region source requires".into(),
        ));
        assert_eq!(
            token_region(&token),
            Some(&Rejected::TokenRegionUncheckable)
        );
        assert_eq!(token_region(&failed), None);
        assert_eq!(token_region(&Ok(())), None);
        assert_eq!(
            Rejected::TokenRegionUncheckable.to_string(),
            "uncheckable: token-region source requires MaximallyReconvergesKHR on main and SPV_KHR_maximal_reconvergence"
        );
    }
}
