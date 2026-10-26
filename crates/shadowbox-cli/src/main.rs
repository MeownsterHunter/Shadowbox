mod aot_scan;
mod cache;
mod cache_state;
mod gate;
mod iostore;
mod nw;
mod pak;
mod prewarm_pack;
mod report;
mod ue_shaderlib;
mod unity;
mod value_diff;

use shadowbox::ir::ops;
use shadowbox::{headers, interface::ShaderInput, ir::InstructionKind};
use std::{collections::HashMap, env, fs, path::PathBuf, process::ExitCode};

fn usage() -> &'static str {
    "usage: shadowbox translate <code.bin> [--header <file> | --header-prefixed] [--target-generation sm5|sm6|sm7.5] [--stage vertex|fragment|compute] [--legacy-color-shading smooth|flat] [--cb1 <file>] [--shared <file>] [--cache-dir <dir>] [--no-memory-cache] -o <out.spv>\n       shadowbox gate --old <binary> --new <binary> <corpus-dir>... [--jobs N] [--seed N] [--timeout-ms N]\n       shadowbox value-diff --old <binary-or-crate> --new <binary-or-crate> <program-dir> [--seed N] [--jobs N] [--timeout-ms N]\n       shadowbox disasm <code.bin>\n       shadowbox coverage <dir>\n       shadowbox cache-extract <cache-dir> <out-dir>\n       shadowbox cache-import <cache-dir> --emit-cache <dir>\n       shadowbox cache-report <out-dir>\n       shadowbox aot-scan <dir> <out-dir> [--translate] [--emit-cache <dir>] [--state-from <cache-dir>]... [--pak-all] [--pak-aes-key-file <file> | --pak-aes-key-env <VAR>]\n       shadowbox prewarm-pack <out-dir> --romfs <dir> [--emulator-cache <dir>]... [--pak-aes-key-file <file>]\n       shadowbox cache-merge <out-dir> <cache-dir>...\n       shadowbox cache-stats <cache-dir>\n       shadowbox cache-verify <cache-dir>"
}

fn words(bytes: &[u8]) -> Result<Vec<u32>, String> {
    if !bytes.len().is_multiple_of(4) {
        return Err("malformed code: byte length is not a multiple of four".into());
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|chunk| u32::from_le_bytes(*chunk))
        .collect())
}

fn stage(value: &str) -> Result<headers::Stage, String> {
    match value {
        "vertex" => Ok(headers::Stage::Vertex),
        "fragment" => Ok(headers::Stage::Fragment),
        "compute" => Ok(headers::Stage::Compute),
        _ => Err(format!(
            "invalid stage {value:?}, expected vertex, fragment, or compute"
        )),
    }
}

fn legacy_color_shading(value: &str) -> Result<bool, String> {
    match value {
        "smooth" => Ok(false),
        "flat" => Ok(true),
        value => Err(format!(
            "invalid legacy color shading {value:?}, expected smooth or flat"
        )),
    }
}

/// Where the program record comes from: decided once from the flags.
enum Header {
    /// No header flag: the record is empty (or the `--stage` envelope).
    Absent,
    /// `--header <file>`: the record is read from a separate file.
    File(PathBuf),
    /// `--header-prefixed`: the record precedes the code in the same file.
    Prefixed,
}

/// The translation input, built once from the flags and the files they name.
/// Every way of supplying a program record ends here as its bytes plus the
/// code, so translation takes it without asking where it came from.
struct Program {
    code: Vec<u8>,
    record: Vec<u8>,
}

impl Program {
    /// Build the input for the header source the flags chose. A given stage
    /// is checked against a supplied header, or, with no header bytes,
    /// becomes the project-owned explicit-stage envelope.
    fn build(
        code: Vec<u8>,
        header: Header,
        generation: shadowbox::TargetGeneration,
        stage: Option<headers::Stage>,
    ) -> Result<Self, String> {
        match header {
            Header::Prefixed => {
                let input = shadowbox::header_prefixed_input(&code, generation)
                    .map_err(|e| e.to_string())?;
                let program = Self {
                    record: input.program_record.to_vec(),
                    code: input.code.to_vec(),
                };
                if let Some(stage) = stage {
                    check_stage(&program.record[..generation.shader_header_size()], stage)?;
                }
                Ok(program)
            }
            Header::File(path) => {
                let record =
                    fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
                match (record.is_empty(), stage) {
                    (true, Some(stage)) => Ok(Self::envelope(code, stage)),
                    (false, Some(stage)) => {
                        check_stage(&record, stage)?;
                        Ok(Self { code, record })
                    }
                    (true | false, None) => Ok(Self { code, record }),
                }
            }
            Header::Absent => Ok(match stage {
                Some(stage) => Self::envelope(code, stage),
                None => Self {
                    code,
                    record: Vec::new(),
                },
            }),
        }
    }

    /// Project-owned explicit-stage envelope. SPH stage10..14 and fragment
    /// target576..608: Mesa sph.rs:185-188,204-215,431-433; Ryujinx
    /// Translation/ShaderHeader.cs:109,162-167.
    fn envelope(code: Vec<u8>, stage: headers::Stage) -> Self {
        let mut record = vec![0; 128];
        let tag: u32 = match stage {
            headers::Stage::Compute => 0,
            headers::Stage::Vertex => 1,
            headers::Stage::TessellationControl => 2,
            headers::Stage::TessellationEvaluation => 3,
            headers::Stage::Geometry => 4,
            headers::Stage::Fragment => 5,
        };
        record[..4].copy_from_slice(&(tag << 10).to_le_bytes());
        if stage == headers::Stage::Fragment {
            record[72] = 0xf;
        }
        Self { code, record }
    }

    fn input(&self) -> ShaderInput<'_> {
        ShaderInput {
            code: &self.code,
            program_record: &self.record,
        }
    }
}

/// Reject a `--stage` that disagrees with the supplied header.
fn check_stage(header: &[u8], stage: headers::Stage) -> Result<(), String> {
    let mut padded = header.to_vec();
    padded.resize(padded.len().max(128), 0);
    let parsed = headers::parse(&words(&padded)?).map_err(|e| format!("parse header: {e:?}"))?;
    if parsed.stage() != stage {
        return Err("--stage disagrees with header stage".into());
    }
    Ok(())
}

/// `shadowbox translate`.
struct Translate {
    program: Program,
    generation: shadowbox::TargetGeneration,
    cb1: Option<Vec<u32>>,
    legacy_color_flat_shading: Option<bool>,
    textures: Vec<shadowbox::interface::BoundTexture>,
    texture_arrays: Vec<shadowbox::interface::TextureArray>,
    cache_dir: Option<PathBuf>,
    memory: shadowbox::cache::Keep,
    output: PathBuf,
}

impl Subcommand for Translate {
    fn parse(args: &[String]) -> Result<Self, String> {
        let code_path = args.first().ok_or_else(|| usage().to_string())?;
        let mut header_path = None;
        let mut cb1 = None;
        let mut shared = None;
        let mut header_prefixed = false;
        let mut generation = shadowbox::TargetGeneration::Sm5;
        let mut stage_flag = None;
        let mut legacy_color_flat_shading = None;
        let mut output = None;
        let mut cache_dir = None;
        let mut memory = shadowbox::cache::Keep::outputs();
        let mut i = 1;
        while i < args.len() {
            match args[i].as_str() {
                "--header" => {
                    i += 1;
                    header_path = Some(PathBuf::from(
                        args.get(i)
                            .ok_or_else(|| "--header needs a file".to_string())?,
                    ));
                }
                "--cb1" => {
                    i += 1;
                    cb1 = Some(PathBuf::from(args.get(i).ok_or("--cb1 needs a file")?));
                }
                "--shared" => {
                    i += 1;
                    shared = Some(PathBuf::from(args.get(i).ok_or("--shared needs a file")?));
                }
                "--header-prefixed" => header_prefixed = true,
                "--target-generation" => {
                    i += 1;
                    generation = match args
                        .get(i)
                        .ok_or("--target-generation needs a value")?
                        .as_str()
                    {
                        "sm5" => shadowbox::TargetGeneration::Sm5,
                        "sm6" => shadowbox::TargetGeneration::Sm6,
                        "sm7.5" => shadowbox::TargetGeneration::Sm7_5Plus,
                        value => return Err(format!("unsupported target generation: {value}")),
                    };
                }
                "--stage" => {
                    i += 1;
                    stage_flag = Some(stage(
                        args.get(i)
                            .ok_or_else(|| "--stage needs a value".to_string())?,
                    )?);
                }
                "--legacy-color-shading" => {
                    i += 1;
                    legacy_color_flat_shading = Some(legacy_color_shading(
                        args.get(i)
                            .ok_or("--legacy-color-shading needs smooth or flat")?,
                    )?);
                }
                "-o" => {
                    i += 1;
                    output = Some(PathBuf::from(
                        args.get(i).ok_or_else(|| "-o needs a file".to_string())?,
                    ));
                }
                "--cache-dir" => {
                    i += 1;
                    cache_dir = Some(PathBuf::from(
                        args.get(i).ok_or("--cache-dir needs a directory")?,
                    ));
                }
                "--no-memory-cache" => memory = shadowbox::cache::Keep::nothing(),
                value => return Err(format!("unknown translate argument {value:?}\n{}", usage())),
            }
            i += 1;
        }
        let output = output.ok_or_else(|| "translate requires -o <out.spv>".to_string())?;
        // The code is read before the header flags are reconciled, so an
        // unreadable code file is reported first, as it always has been.
        let code = fs::read(code_path).map_err(|e| format!("read {code_path}: {e}"))?;
        let header = match (header_path, header_prefixed) {
            (Some(_), true) => {
                return Err("--header-prefixed cannot be combined with --header".into())
            }
            (Some(path), false) => Header::File(path),
            (None, true) => Header::Prefixed,
            (None, false) => Header::Absent,
        };
        let program = Program::build(code, header, generation, stage_flag)?;
        // Saved uint words: Ryujinx DiskCache/DiskCacheGpuAccessor.cs:62-69.
        // The library checks embedded/explicit agreement and hashes the snapshot.
        let cb1 = cb1
            .map(|path: PathBuf| {
                let bytes = value_diff::read_limited(&path, 65536)?;
                words(&bytes).map_err(|e| format!("CB1 snapshot: {e}"))
            })
            .transpose()?;
        // A decompressed Ryujinx shared.data entry (cache-extract .shared).
        // Its stage index is the SPH type less one; compute and Vertex A are 0.
        let (textures, texture_arrays) = shared
            .map(|path: PathBuf| {
                let bytes = fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
                let tag = (u32::from_le_bytes(program.record[..4].try_into().unwrap()) >> 10) & 0xf;
                let stage = tag.max(1) - 1;
                Ok::<_, String>((
                    shadowbox::recorded_state::queried_textures(&bytes, stage)?,
                    shadowbox::recorded_state::texture_arrays(&bytes, stage)?,
                ))
            })
            .transpose()?
            .unwrap_or_default();
        Ok(Self {
            program,
            generation,
            cb1,
            legacy_color_flat_shading,
            textures,
            texture_arrays,
            cache_dir,
            memory,
            output,
        })
    }

    fn run<R: shadowbox::Run>(self, run: R) -> Result<(), String> {
        let Self {
            program,
            generation,
            cb1,
            legacy_color_flat_shading,
            textures,
            texture_arrays,
            cache_dir,
            memory,
            output,
        } = self;
        let report_cache = cache_dir.is_some();
        let mut cache = shadowbox::Cache::new(shadowbox::CacheConfig {
            memory,
            disk: cache_dir
                .into_iter()
                .fold(shadowbox::cache::Disk::nowhere(), |_, dir| {
                    shadowbox::cache::Disk::at(dir)
                }),
        });
        let spirv = match shadowbox::translate_cached_with_context(
            run,
            &program.input(),
            shadowbox::CacheContext {
                generation,
                graphics_state: shadowbox::interface::GraphicsState {
                    legacy_color_flat_shading,
                    constant_buffer1: cb1.as_deref(),
                    texture_bindings: &textures,
                    texture_arrays: &texture_arrays,
                    ..Default::default()
                },
                ..Default::default()
            },
            &mut cache,
        ) {
            shadowbox::interface::Translation::Module(output) => output.spirv,
            shadowbox::interface::Translation::Stopped(stopped) => return Err(stopped.to_string()),
        };
        if report_cache {
            eprintln!(
                "cache: memory_hits={} disk_hits={} misses={}",
                cache.stats.memory_hits, cache.stats.disk_hits, cache.stats.misses
            );
        }
        let bytes = spirv
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<_>>();
        fs::write(&output, bytes).map_err(|e| format!("write {}: {e}", output.display()))
    }
}

/// `shadowbox disasm`.
struct Disasm {
    path: PathBuf,
}

impl Subcommand for Disasm {
    fn parse(args: &[String]) -> Result<Self, String> {
        let [path] = args else {
            return Err(usage().into());
        };
        Ok(Self {
            path: PathBuf::from(path),
        })
    }

    fn run<R: shadowbox::Run>(self, run: R) -> Result<(), String> {
        let path = self.path.display();
        let data = fs::read(&self.path).map_err(|e| format!("read {path}: {e}"))?;
        let decoded = shadowbox::decoder::decode_stream(run, &words(&data)?);
        for instruction in decoded {
            println!("{instruction}");
        }
        Ok(())
    }
}

/// `shadowbox coverage`.
struct Coverage {
    path: PathBuf,
}

impl Subcommand for Coverage {
    fn parse(args: &[String]) -> Result<Self, String> {
        let [path] = args else {
            return Err(usage().into());
        };
        Ok(Self {
            path: PathBuf::from(path),
        })
    }

    fn run<R: shadowbox::Run>(self, run: R) -> Result<(), String> {
        let mut counts = HashMap::<u16, usize>::new();
        let mut translated = 0;
        let mut stopped = 0;
        let mut files = fs::read_dir(&self.path)
            .map_err(|e| format!("read directory {}: {e}", self.path.display()))?
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|entry| entry.extension().and_then(|x| x.to_str()) == Some("bin"))
            .collect::<Vec<_>>();
        files.sort();
        for file in files {
            let data = fs::read(&file).map_err(|e| format!("read {}: {e}", file.display()))?;
            let decoded = shadowbox::decoder::decode_stream(run, &words(&data)?);
            for instruction in &decoded {
                if let InstructionKind::Unknown(ops::Unknown { raw }) =
                    instruction.kind.legacy_form()
                {
                    *counts.entry((raw[1] >> 16) as u16).or_default() += 1;
                }
            }
            // The sibling `.hdr` is the program record, as `translate --header` reads it.
            let header_path = file.with_extension("hdr");
            let header = match header_path.exists() {
                true => Header::File(header_path),
                false => Header::Absent,
            };
            let program = Program::build(data, header, shadowbox::TargetGeneration::Sm5, None)?;
            match shadowbox::translate(run, &program.input()) {
                shadowbox::interface::Translation::Module(_) => {
                    println!("{}: translated", file.display());
                    translated += 1;
                }
                shadowbox::interface::Translation::Stopped(state) => {
                    println!("{}: stopped: {state}", file.display());
                    stopped += 1;
                }
            }
        }
        println!("summary: translated={translated} stopped={stopped}");
        let mut entries = counts.into_iter().collect::<Vec<_>>();
        entries.sort_by(|(a_opcode, a_count), (b_opcode, b_count)| {
            b_count.cmp(a_count).then(a_opcode.cmp(b_opcode))
        });
        for (opcode, count) in entries {
            println!("unknown opcode 0x{opcode:04x}: {count}");
        }
        Ok(())
    }
}

fn print_cache_stats(stats: &shadowbox::DiskCacheStats) {
    println!(
        "entries={} bytes={} subgroup_size_32={} invalid={}",
        stats.entries, stats.bytes, stats.subgroup_size_32, stats.invalid
    );
}

/// `shadowbox cache-merge`.
struct CacheMerge {
    output: PathBuf,
    sources: Vec<PathBuf>,
}

impl Subcommand for CacheMerge {
    fn parse(args: &[String]) -> Result<Self, String> {
        let [output, sources @ ..] = args else {
            return Err(usage().into());
        };
        if sources.is_empty() {
            return Err(usage().into());
        }
        Ok(Self {
            output: PathBuf::from(output),
            sources: sources.iter().map(PathBuf::from).collect(),
        })
    }

    fn run<R: shadowbox::Run>(self, _: R) -> Result<(), String> {
        let stats = shadowbox::merge_disk_caches(&self.output, &self.sources)
            .map_err(|e| format!("cache-merge: {e}"))?;
        print_cache_stats(&stats);
        Ok(())
    }
}

/// `shadowbox cache-verify`.
struct CacheVerify {
    dir: PathBuf,
}

impl Subcommand for CacheVerify {
    fn parse(args: &[String]) -> Result<Self, String> {
        let [dir] = args else {
            return Err(usage().into());
        };
        Ok(Self {
            dir: PathBuf::from(dir),
        })
    }

    fn run<R: shadowbox::Run>(self, _: R) -> Result<(), String> {
        let entries =
            shadowbox::verify_disk_cache(&self.dir).map_err(|e| format!("cache-verify: {e}"))?;
        let (mut valid, mut stale, mut mismatched, mut corrupt) = (0, 0, 0, 0);
        for (file, status) in entries {
            println!("{}\t{file}", status.as_str());
            match status {
                shadowbox::CacheEntryStatus::Valid => valid += 1,
                shadowbox::CacheEntryStatus::Stale => stale += 1,
                shadowbox::CacheEntryStatus::Mismatched => mismatched += 1,
                shadowbox::CacheEntryStatus::Corrupt => corrupt += 1,
            }
        }
        println!(
            "cache-verify: valid={valid} stale={stale} mismatched={mismatched} corrupt={corrupt}"
        );
        if stale + mismatched + corrupt != 0 {
            return Err("cache-verify: incompatible or corrupt entries found".into());
        }
        Ok(())
    }
}

/// `shadowbox cache-stats`.
struct CacheStats {
    dir: PathBuf,
}

impl Subcommand for CacheStats {
    fn parse(args: &[String]) -> Result<Self, String> {
        let [dir] = args else {
            return Err(usage().into());
        };
        Ok(Self {
            dir: PathBuf::from(dir),
        })
    }

    fn run<R: shadowbox::Run>(self, _: R) -> Result<(), String> {
        let stats =
            shadowbox::disk_cache_stats(&self.dir).map_err(|e| format!("cache-stats: {e}"))?;
        print_cache_stats(&stats);
        Ok(())
    }
}

/// `shadowbox cache-import`.
struct CacheImport {
    input: PathBuf,
    output: PathBuf,
}

impl Subcommand for CacheImport {
    fn parse(args: &[String]) -> Result<Self, String> {
        let [input, flag, output] = args else {
            return Err(usage().into());
        };
        if flag != "--emit-cache" {
            return Err(usage().into());
        }
        Ok(Self {
            input: PathBuf::from(input),
            output: PathBuf::from(output),
        })
    }

    fn run<R: shadowbox::Run>(self, run: R) -> Result<(), String> {
        cache_state::import(run, &self.input, &self.output)
    }
}

/// `shadowbox cache-extract`.
struct CacheExtract {
    input: PathBuf,
    output: PathBuf,
}

impl Subcommand for CacheExtract {
    fn parse(args: &[String]) -> Result<Self, String> {
        let [input, output] = args else {
            return Err(usage().into());
        };
        Ok(Self {
            input: PathBuf::from(input),
            output: PathBuf::from(output),
        })
    }

    fn run<R: shadowbox::Run>(self, _: R) -> Result<(), String> {
        cache::extract(&self.input, &self.output)
    }
}

/// `shadowbox cache-report`.
struct CacheReport {
    path: PathBuf,
}

impl Subcommand for CacheReport {
    fn parse(args: &[String]) -> Result<Self, String> {
        let [path] = args else {
            return Err(usage().into());
        };
        Ok(Self {
            path: PathBuf::from(path),
        })
    }

    fn run<R: shadowbox::Run>(self, _: R) -> Result<(), String> {
        report::report(&self.path)
    }
}

/// A subcommand: its arguments are decoded once into `Self`, which then runs.
trait Subcommand: Sized {
    fn parse(args: &[String]) -> Result<Self, String>;
    fn run<R: shadowbox::Run>(self, run: R) -> Result<(), String>;
}

fn parse_and_run<C: Subcommand, R: shadowbox::Run>(run: R, args: &[String]) -> Result<(), String> {
    C::parse(args)?.run(run)
}

/// The single argv-to-type step: the command name picks the type that owns
/// the rest of the arguments, on the run the program's start chose.
fn cli<R: shadowbox::Run>(run: R) -> Result<(), String> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let Some((command, rest)) = args.split_first() else {
        return Err(usage().into());
    };
    match command.as_str() {
        "translate" => parse_and_run::<Translate, _>(run, rest),
        "gate" => parse_and_run::<gate::Gate, _>(run, rest),
        "value-diff" => parse_and_run::<value_diff::ValueDiff, _>(run, rest),
        "__value-diff-compare" => parse_and_run::<value_diff::CompareWorker, _>(run, rest),
        "__value-diff-worker" => parse_and_run::<value_diff::CaptureWorker, _>(run, rest),
        "disasm" => parse_and_run::<Disasm, _>(run, rest),
        "coverage" => parse_and_run::<Coverage, _>(run, rest),
        "cache-extract" => parse_and_run::<CacheExtract, _>(run, rest),
        "cache-import" => parse_and_run::<CacheImport, _>(run, rest),
        "cache-report" => parse_and_run::<CacheReport, _>(run, rest),
        "aot-scan" => parse_and_run::<aot_scan::AotScan, _>(run, rest),
        "prewarm-pack" => parse_and_run::<prewarm_pack::PrewarmPack, _>(run, rest),
        "cache-merge" => parse_and_run::<CacheMerge, _>(run, rest),
        "cache-stats" => parse_and_run::<CacheStats, _>(run, rest),
        "cache-verify" => parse_and_run::<CacheVerify, _>(run, rest),
        _ => Err(usage().into()),
    }
}

fn main() -> ExitCode {
    // The program's start is its edge: the environment says once what the
    // run is, and the run travels into the library as its own type.
    let outcome = match env::var("SHADOWBOX_TRACE_STACK").as_deref() {
        Ok("1") => cli(shadowbox::TracedRun),
        _ => cli(shadowbox::QuietRun),
    };
    if let Err(error) = outcome {
        eprintln!("shadowbox: {error}");
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build(
        header: Header,
        stage: Option<headers::Stage>,
        code: Vec<u8>,
    ) -> Result<Program, String> {
        Program::build(code, header, shadowbox::TargetGeneration::Sm5, stage)
    }

    #[test]
    fn translate_builds_each_header_source_once() {
        let code = vec![1, 2, 3, 4];
        let headerless = build(Header::Absent, None, code.clone()).unwrap();
        assert!(headerless.record.is_empty());
        assert_eq!(headerless.code, code);

        let envelope = build(Header::Absent, Some(headers::Stage::Fragment), code.clone()).unwrap();
        assert_eq!(envelope.record.len(), 128);
        assert_eq!(envelope.record[..4], (5u32 << 10).to_le_bytes());
        assert_eq!(envelope.record[72], 0xf);

        let mut prefixed = envelope.record[..80].to_vec();
        prefixed.extend(&code);
        let program = build(
            Header::Prefixed,
            Some(headers::Stage::Fragment),
            prefixed.clone(),
        )
        .unwrap();
        assert_eq!(program.record, envelope.record[..80]);
        assert_eq!(program.code, code);
        assert_eq!(
            build(Header::Prefixed, Some(headers::Stage::Vertex), prefixed)
                .err()
                .unwrap(),
            "--stage disagrees with header stage"
        );
    }

    #[test]
    fn translate_header_file_without_bytes_takes_the_stage_envelope() {
        let dir = std::env::temp_dir().join(format!("shadowbox-cli-header-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let empty = dir.join("empty.hdr");
        fs::write(&empty, []).unwrap();
        let program = build(
            Header::File(empty.clone()),
            Some(headers::Stage::Compute),
            vec![0; 4],
        )
        .unwrap();
        assert_eq!(program.record.len(), 128);
        assert!(build(Header::File(empty), None, vec![0; 4])
            .unwrap()
            .record
            .is_empty());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn legacy_color_shading_is_closed_and_typed() {
        assert!(!legacy_color_shading("smooth").unwrap());
        assert!(legacy_color_shading("flat").unwrap());
        assert_eq!(
            legacy_color_shading("default").unwrap_err(),
            "invalid legacy color shading \"default\", expected smooth or flat"
        );
    }
}
