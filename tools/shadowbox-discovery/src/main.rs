use clap::{Parser, Subcommand};
use shadowbox_discovery::{
    conveyor::{advance_campaign, CampaignSpec},
    hybrid::{
        build_bundles as build_hybrid_bundles, build_packets as build_hybrid_packets,
        build_plan as build_hybrid_plan, render_words as render_hybrid_words, validate_preflight,
        validate_qualification_space, HybridBundleSet, HybridPackets, HybridPlan,
    },
    sudoku::{build_catalog, SourceLock},
    Inputs, ReplayVerification,
};
use std::{
    fs,
    path::{Path, PathBuf},
    process::ExitCode,
};

#[derive(Parser)]
#[command(about = "Build and verify read-only Shadowbox discovery plans")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Reconcile a source-locked hybrid-v3 campaign into immutable work units.
    HybridConveyor {
        #[arg(long)]
        source_root: PathBuf,
        #[arg(long)]
        expected_head: String,
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        conflicts: PathBuf,
        #[arg(long)]
        state_dir: PathBuf,
        #[arg(long)]
        answers: Option<PathBuf>,
        #[arg(long, default_value_t = 10)]
        max_bits: u8,
        #[arg(long, default_value_t = 1_000_000)]
        max_words: u64,
        #[arg(long, default_value_t = 2_000_000)]
        max_domain_words: u64,
        #[arg(long, default_value_t = 100_000)]
        max_packets: usize,
        #[arg(long, default_value_t = 64)]
        max_regions: usize,
        #[arg(long, default_value_t = 4000)]
        max_bundle_words: usize,
        #[arg(long, default_value_t = 8 * 1024 * 1024 * 1024_u64)]
        min_free_bytes: u64,
    },
    Plan {
        #[arg(long)]
        rows: PathBuf,
        #[arg(long)]
        samples: PathBuf,
        #[arg(long)]
        results: PathBuf,
        #[arg(long)]
        semantic_source: PathBuf,
        #[arg(long)]
        claims: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    Verify {
        #[arg(long)]
        rows: PathBuf,
        #[arg(long)]
        samples: PathBuf,
        #[arg(long)]
        results: PathBuf,
        #[arg(long)]
        semantic_source: PathBuf,
        #[arg(long)]
        claims: PathBuf,
        #[arg(long)]
        plan: PathBuf,
    },
    Classify {
        #[arg(long)]
        rows: PathBuf,
        #[arg(long)]
        samples: PathBuf,
        #[arg(long)]
        results: PathBuf,
        #[arg(long)]
        semantic_source: PathBuf,
        #[arg(long)]
        claims: PathBuf,
    },
    SudokuCatalog {
        #[arg(long)]
        source_root: PathBuf,
        #[arg(long)]
        expected_head: String,
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    HybridPlan {
        #[arg(long)]
        source_root: PathBuf,
        #[arg(long)]
        expected_head: String,
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        conflicts: PathBuf,
        #[arg(long, default_value_t = 10)]
        max_bits: u8,
        #[arg(long, default_value_t = 1_000_000)]
        max_words: u64,
        #[arg(long, default_value_t = 2_000_000)]
        max_domain_words: u64,
        #[arg(long)]
        plan: PathBuf,
    },
    HybridWords {
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        out: PathBuf,
    },
    HybridReplay {
        #[arg(long)]
        source_root: PathBuf,
        #[arg(long)]
        expected_head: String,
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        conflicts: PathBuf,
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        words: PathBuf,
    },
    HybridPacket {
        #[arg(long)]
        source_root: PathBuf,
        #[arg(long)]
        expected_head: String,
        #[arg(long)]
        model: PathBuf,
        #[arg(long)]
        input: PathBuf,
        #[arg(long)]
        conflicts: PathBuf,
        #[arg(long)]
        plan: PathBuf,
        #[arg(long)]
        answers: PathBuf,
        #[arg(long, default_value_t = 100_000)]
        max_packets: usize,
        #[arg(long)]
        out: PathBuf,
    },
    HybridBundle {
        #[arg(long)]
        packets: PathBuf,
        #[arg(long, default_value_t = 64)]
        max_regions: usize,
        #[arg(long, default_value_t = 4000)]
        max_words: usize,
        #[arg(long)]
        out: PathBuf,
    },
    HybridPreflight {
        #[arg(long)]
        source_root: PathBuf,
        #[arg(long)]
        expected_head: String,
        #[arg(long)]
        packets: PathBuf,
        #[arg(long)]
        bundles: PathBuf,
        #[arg(long)]
        work_path: PathBuf,
        #[arg(long, default_value_t = 8 * 1024 * 1024 * 1024_u64)]
        min_free_bytes: u64,
    },
}

fn load(
    rows: &Path,
    samples: &Path,
    results: &Path,
    semantic_source: &Path,
    claims: &Path,
) -> Result<shadowbox_discovery::Ledger, String> {
    Inputs::read(rows, samples, results, semantic_source, claims)?.classify()
}

fn run() -> Result<ExitCode, String> {
    match Cli::parse().command {
        Command::HybridConveyor {
            source_root,
            expected_head,
            model,
            input,
            conflicts,
            state_dir,
            answers,
            max_bits,
            max_words,
            max_domain_words,
            max_packets,
            max_regions,
            max_bundle_words,
            min_free_bytes,
        } => {
            let receipt = advance_campaign(CampaignSpec {
                source_root,
                expected_head,
                model,
                input,
                conflicts,
                state_dir,
                answers,
                max_bits,
                max_words,
                max_domain_words,
                max_packets,
                max_regions,
                max_bundle_words,
                min_free_bytes,
            })?;
            println!("{:?}\t{}", receipt.phase, receipt.receipt_digest);
            Ok(ExitCode::SUCCESS)
        }
        Command::Plan {
            rows,
            samples,
            results,
            semantic_source,
            claims,
            out,
        } => {
            let bytes = load(&rows, &samples, &results, &semantic_source, &claims)?.render_plan();
            fs::write(&out, bytes).map_err(|error| format!("write {}: {error}", out.display()))?;
            Ok(ExitCode::SUCCESS)
        }
        Command::Verify {
            rows,
            samples,
            results,
            semantic_source,
            claims,
            plan,
        } => {
            let ledger = load(&rows, &samples, &results, &semantic_source, &claims)?;
            let actual = fs::read_to_string(&plan)
                .map_err(|error| format!("read {}: {error}", plan.display()))?;
            match ledger.verify_plan(&actual) {
                ReplayVerification::Verified => {
                    println!("Verified\tbyte-identical");
                    Ok(ExitCode::SUCCESS)
                }
                ReplayVerification::Stale => {
                    println!("Stale\tplan bytes differ from source-locked replay");
                    Ok(ExitCode::from(3))
                }
            }
        }
        Command::Classify {
            rows,
            samples,
            results,
            semantic_source,
            claims,
        } => {
            print!(
                "{}",
                load(&rows, &samples, &results, &semantic_source, &claims)?.render_classification()
            );
            Ok(ExitCode::SUCCESS)
        }
        Command::SudokuCatalog {
            source_root,
            expected_head,
            model,
            input,
            out,
        } => {
            // Establish current source before reading either discovery artifact.
            let lock = SourceLock::acquire(&source_root, &expected_head)?;
            let bytes = build_catalog(lock, &model, &input)?;
            fs::write(&out, bytes).map_err(|error| format!("write {}: {error}", out.display()))?;
            Ok(ExitCode::SUCCESS)
        }
        Command::HybridPlan {
            source_root,
            expected_head,
            model,
            input,
            conflicts,
            max_bits,
            max_words,
            max_domain_words,
            plan,
        } => {
            let lock = SourceLock::acquire(&source_root, &expected_head)?;
            let hybrid = build_hybrid_plan(
                lock,
                &model,
                &input,
                &conflicts,
                max_bits,
                max_words,
                max_domain_words,
            )?;
            write_new(
                &plan,
                &serde_json::to_vec_pretty(&hybrid).map_err(|error| {
                    format!("serialize hybrid plan {}: {error}", plan.display())
                })?,
            )?;
            println!(
                "cubes={} words={} digest={}",
                hybrid.cubes.len(),
                hybrid.emitted_words,
                hybrid.words_digest
            );
            Ok(ExitCode::SUCCESS)
        }
        Command::HybridWords { plan, out } => {
            let hybrid: HybridPlan = serde_json::from_slice(
                &fs::read(&plan).map_err(|error| format!("read {}: {error}", plan.display()))?,
            )
            .map_err(|error| format!("parse {}: {error}", plan.display()))?;
            let words = render_hybrid_words(&hybrid)?;
            write_new(&out, words.as_bytes())?;
            println!("{}", hybrid.words_digest);
            Ok(ExitCode::SUCCESS)
        }
        Command::HybridReplay {
            source_root,
            expected_head,
            model,
            input,
            conflicts,
            plan,
            words,
        } => {
            let hybrid: HybridPlan = serde_json::from_slice(
                &fs::read(&plan).map_err(|error| format!("read {}: {error}", plan.display()))?,
            )
            .map_err(|error| format!("parse {}: {error}", plan.display()))?;
            let lock = SourceLock::acquire(&source_root, &expected_head)?;
            let rebuilt = build_hybrid_plan(
                lock,
                &model,
                &input,
                &conflicts,
                hybrid.max_bits,
                hybrid.max_words,
                hybrid.max_domain_words,
            )?;
            if rebuilt != hybrid {
                return Err("hybrid plan is not a byte-identical source replay".into());
            }
            let expected = render_hybrid_words(&rebuilt)?;
            let actual =
                fs::read(&words).map_err(|error| format!("read {}: {error}", words.display()))?;
            if expected.as_bytes() != actual {
                return Err("hybrid words are not a byte-identical plan replay".into());
            }
            println!("Verified\tbyte-identical\t{}", hybrid.words_digest);
            Ok(ExitCode::SUCCESS)
        }
        Command::HybridPacket {
            source_root,
            expected_head,
            model,
            input,
            conflicts,
            plan,
            answers,
            max_packets,
            out,
        } => {
            let hybrid: HybridPlan = serde_json::from_slice(
                &fs::read(&plan).map_err(|error| format!("read {}: {error}", plan.display()))?,
            )
            .map_err(|error| format!("parse {}: {error}", plan.display()))?;
            let lock = SourceLock::acquire(&source_root, &expected_head)?;
            let rebuilt = build_hybrid_plan(
                lock,
                &model,
                &input,
                &conflicts,
                hybrid.max_bits,
                hybrid.max_words,
                hybrid.max_domain_words,
            )?;
            if rebuilt != hybrid {
                return Err("hybrid packet plan is not a byte-identical source replay".into());
            }
            let packets = build_hybrid_packets(&hybrid, &answers, max_packets)?;
            write_new(
                &out,
                &serde_json::to_vec_pretty(&packets).map_err(|error| {
                    format!("serialize hybrid packets {}: {error}", out.display())
                })?,
            )?;
            println!(
                "packets={} accepted={} rejected={} digest={}",
                packets.packets.len(),
                packets.oracle_accepted_words,
                packets.oracle_rejected_words,
                packets.packets_digest
            );
            Ok(ExitCode::SUCCESS)
        }
        Command::HybridBundle {
            packets,
            max_regions,
            max_words,
            out,
        } => {
            let packets: HybridPackets = serde_json::from_slice(
                &fs::read(&packets)
                    .map_err(|error| format!("read {}: {error}", packets.display()))?,
            )
            .map_err(|error| format!("parse {}: {error}", packets.display()))?;
            let bundles = build_hybrid_bundles(&packets, max_regions, max_words)?;
            write_new(
                &out,
                &serde_json::to_vec_pretty(&bundles).map_err(|error| {
                    format!("serialize hybrid bundles {}: {error}", out.display())
                })?,
            )?;
            println!(
                "bundles={} regions={} words={} digest={}",
                bundles.bundles.len(),
                bundles.region_count,
                bundles.word_count,
                bundles.bundles_digest
            );
            Ok(ExitCode::SUCCESS)
        }
        Command::HybridPreflight {
            source_root,
            expected_head,
            packets,
            bundles,
            work_path,
            min_free_bytes,
        } => {
            let _lock = SourceLock::acquire(&source_root, &expected_head)?;
            let packets: HybridPackets = serde_json::from_slice(
                &fs::read(&packets)
                    .map_err(|error| format!("read {}: {error}", packets.display()))?,
            )
            .map_err(|error| format!("parse {}: {error}", packets.display()))?;
            let bundles: HybridBundleSet = serde_json::from_slice(
                &fs::read(&bundles)
                    .map_err(|error| format!("read {}: {error}", bundles.display()))?,
            )
            .map_err(|error| format!("parse {}: {error}", bundles.display()))?;
            validate_preflight(&packets, &bundles, &expected_head)?;
            let free = available_bytes(&work_path)?;
            validate_qualification_space(free, min_free_bytes)
                .map_err(|error| format!("{error} at {}", work_path.display()))?;
            // Recheck the source lock after reading every external artifact.
            let _lock = SourceLock::acquire(&source_root, &expected_head)?;
            println!(
                "Verified\thybrid-v3\t{}\t{}\tfree={free}",
                packets.receipt_digest, bundles.bundles_digest
            );
            Ok(ExitCode::SUCCESS)
        }
    }
}

fn available_bytes(path: &Path) -> Result<u64, String> {
    use std::{ffi::CString, mem::MaybeUninit, os::unix::ffi::OsStrExt};

    let path = path
        .canonicalize()
        .map_err(|error| format!("canonicalize {}: {error}", path.display()))?;
    let path_bytes = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| format!("{} contains a NUL byte", path.display()))?;
    let mut stats = MaybeUninit::<libc::statvfs>::uninit();
    // SAFETY: path_bytes is NUL terminated and stats points to writable storage.
    if unsafe { libc::statvfs(path_bytes.as_ptr(), stats.as_mut_ptr()) } != 0 {
        return Err(format!(
            "statvfs {}: {}",
            path.display(),
            std::io::Error::last_os_error()
        ));
    }
    // SAFETY: statvfs returned success and initialized stats.
    let stats = unsafe { stats.assume_init() };
    Ok(stats.f_bavail.saturating_mul(stats.f_frsize))
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), String> {
    use std::io::Write as _;
    let mut options = fs::OpenOptions::new();
    options.write(true).create_new(true);
    let mut file = options
        .open(path)
        .map_err(|error| format!("create {}: {error}", path.display()))?;
    file.write_all(bytes)
        .map_err(|error| format!("write {}: {error}", path.display()))?;
    file.sync_all()
        .map_err(|error| format!("sync {}: {error}", path.display()))
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("shadowbox-discovery: {error}");
            ExitCode::FAILURE
        }
    }
}
