//! Original package orchestration.
//! No GPU fields are decoded here; the existing scanner/importer record sources.
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicUsize, Ordering},
};

const FORMAT: &str = "shadowbox-prewarm-pack-1\n";
const CHECKPOINT: &str = "checkpoint.json";

// Default SM5/options are the scanner/importer's current translation contract.
// Project manifest/checkpoint field: docs/reference-sources.md.
fn translator_identity() -> String {
    shadowbox::translator_identity(
        shadowbox::TargetGeneration::Sm5,
        shadowbox::TranslationOptions::default(),
    )
}

type Result<T> = std::result::Result<T, String>;

fn file_hash(path: &Path) -> Result<String> {
    let mut reader = File::open(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let mut hash = Sha256::new();
    let mut buffer = [0; 64 * 1024];
    loop {
        let count = reader.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn identity(parts: &[&str]) -> String {
    let mut hash = Sha256::new();
    for part in parts {
        hash.update((part.len() as u64).to_le_bytes());
        hash.update(part.as_bytes());
    }
    format!("{:x}", hash.finalize())
}

// Hash names, types, empty directories and file contents in sorted order.
// Reject links and special files so reads/writes cannot escape a package tree.
fn tree_hash(root: &Path, checkpoint: bool) -> Result<String> {
    fn visit(root: &Path, dir: &Path, checkpoint: bool, hash: &mut Sha256) -> Result<()> {
        let mut entries = fs::read_dir(dir)
            .map_err(|e| e.to_string())?
            .map(|e| e.map(|e| e.path()))
            .collect::<std::io::Result<Vec<_>>>()
            .map_err(|e| e.to_string())?;
        entries.sort();
        for path in entries {
            if checkpoint && path == root.join(CHECKPOINT) {
                continue;
            }
            let kind = fs::symlink_metadata(&path)
                .map_err(|e| e.to_string())?
                .file_type();
            if !kind.is_dir() && !kind.is_file() {
                return Err(format!(
                    "links and special files are unsupported: {}",
                    path.display()
                ));
            }
            let relative = path
                .strip_prefix(root)
                .unwrap()
                .as_os_str()
                .as_encoded_bytes();
            hash.update((relative.len() as u64).to_le_bytes());
            hash.update(relative);
            hash.update([u8::from(kind.is_dir())]);
            if kind.is_dir() {
                visit(root, &path, false, hash)?;
            } else {
                hash.update(file_hash(&path)?);
            }
        }
        Ok(())
    }
    let mut hash = Sha256::new();
    visit(root, root, checkpoint, &mut hash)?;
    Ok(format!("{:x}", hash.finalize()))
}

fn validate_output(root: &Path) -> Result<()> {
    let mut pending = vec![root.to_owned()];
    while let Some(dir) = pending.pop() {
        for entry in fs::read_dir(dir).map_err(|e| e.to_string())? {
            let path = entry.map_err(|e| e.to_string())?.path();
            let kind = fs::symlink_metadata(&path)
                .map_err(|e| e.to_string())?
                .file_type();
            if kind.is_dir() {
                pending.push(path);
            } else if !kind.is_file() {
                return Err(format!(
                    "links and special files are unsupported: {}",
                    path.display()
                ));
            }
        }
    }
    Ok(())
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    if fs::read(path).ok().as_deref() == Some(bytes) {
        return Ok(());
    }
    let temporary = path.with_extension("pending");
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(|e| format!("create {}: {e}", temporary.display()))?;
    let result = file
        .write_all(bytes)
        .and_then(|()| file.sync_all())
        .and_then(|()| fs::rename(&temporary, path));
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result.map_err(|e| e.to_string())
}

fn write_json(path: &Path, value: &Value) -> Result<()> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?;
    bytes.push(b'\n');
    atomic_write(path, &bytes)
}

struct Source {
    path: PathBuf,
    hash: String,
}
impl Source {
    fn new(path: &Path) -> Result<Self> {
        let path = path
            .canonicalize()
            .map_err(|e| format!("input {}: {e}", path.display()))?;
        if !path.is_dir() {
            return Err(format!("input must be a directory: {}", path.display()));
        }
        let hash = tree_hash(&path, false)?;
        Ok(Self { path, hash })
    }
    fn verify(&self) -> Result<()> {
        if tree_hash(&self.path, false)? != self.hash {
            return Err(format!(
                "input changed during prewarm-pack: {}",
                self.path.display()
            ));
        }
        Ok(())
    }
}

fn key_hash(path: Option<&Path>) -> Result<String> {
    let Some(path) = path else {
        return Ok("no-key".into());
    };
    let mut bytes = zeroize::Zeroizing::new(Vec::new());
    File::open(path)
        .and_then(|f| f.take(1025).read_to_end(&mut bytes))
        .map_err(|_| "cannot read pak AES key file".to_owned())?;
    if bytes.len() > 1024 {
        return Err("pak AES key file exceeds size limit".into());
    }
    super::pak::AesKey::parse(&bytes).map_err(|e| e.to_string())?;
    Ok(super::cache::hash(&bytes))
}

#[derive(Clone)]
enum Job {
    Scan,
    Import(usize),
}
struct Task {
    job: Job,
    hash: String,
    output: PathBuf,
}

fn scan_results(output: &Path) -> Result<Value> {
    let reader =
        BufReader::new(File::open(output.join("manifest.jsonl")).map_err(|e| e.to_string())?);
    let mut programs = BTreeMap::new();
    let mut diagnostics = Vec::new();
    let mut summary = Value::Null;
    for line in reader.lines() {
        let event: Value =
            serde_json::from_str(&line.map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        if event["kind"] == "summary" {
            summary = event.clone();
        }
        if event["kind"] == "rejected"
            || event["kind"]
                .as_str()
                .is_some_and(|k| k.ends_with("skipped"))
        {
            diagnostics.push(event.clone());
        }
        if event["kind"] != "program" || event["unique"] != true {
            continue;
        }
        let file = event["output"]
            .as_str()
            .ok_or("scan program lacks output")?;
        let record = fs::read(output.join(file)).map_err(|e| e.to_string())?;
        let input = shadowbox::header_prefixed_input(&record, shadowbox::TargetGeneration::Sm5)
            .map_err(|e| e.to_string())?;
        let key = super::cache_state::identity(&input, shadowbox::CacheContext::default())
            .map_err(|e| e.to_string())?;
        for variant in event["state_variants"]
            .as_array()
            .ok_or("scan lacks state variants")?
        {
            let missing = &variant["missing_state_keys"];
            let status = if missing.as_array().is_some_and(|m| !m.is_empty()) {
                "state_missing"
            } else if variant["translation_error"].is_null() {
                "complete"
            } else {
                "failed"
            };
            let result = json!({"program_key": key, "stage": event["stage"],
                "translation_key": variant["translation_key"], "status": status,
                "missing_state_keys": missing,
                "first_blocker": if status == "failed" { variant["translation_error"].clone() } else { Value::Null }});
            programs.insert(result.to_string(), result);
        }
    }
    Ok(
        json!({"programs": programs.into_values().collect::<Vec<_>>(),
        "diagnostics": diagnostics, "summary": summary}),
    )
}

fn execute<R: shadowbox::Run>(
    run: R,
    task: &Task,
    romfs: &Source,
    caches: &[Source],
    key: Option<&Path>,
    expected_key_hash: &str,
) -> Result<Value> {
    let checkpoint = task.output.join(CHECKPOINT);
    if checkpoint.is_file() {
        let old = fs::read(&checkpoint)
            .ok()
            .and_then(|b| serde_json::from_slice::<Value>(&b).ok());
        if let Some(old) = old {
            if old["translator_identity"] == translator_identity()
                && old["input_sha256"] == task.hash
                && old["output_sha256"] == tree_hash(&task.output, true)?
                && old["result_sha256"] == super::cache::hash(old["result"].to_string().as_bytes())
            {
                return Ok(old["result"].clone());
            }
        }
    }
    if task.output.exists() {
        fs::remove_dir_all(&task.output).map_err(|e| e.to_string())?;
    }
    fs::create_dir_all(task.output.join("cache")).map_err(|e| e.to_string())?;
    let result = match task.job {
        Job::Scan => {
            let paths = caches.iter().map(|c| c.path.clone()).collect::<Vec<_>>();
            super::aot_scan::pack(run, &romfs.path, &task.output, &paths, key)?;
            if key_hash(key)? != expected_key_hash {
                return Err("pak AES key changed during prewarm-pack".into());
            }
            romfs.verify()?;
            for source in caches {
                source.verify()?;
            }
            scan_results(&task.output)?
        }
        Job::Import(index) => {
            let programs = super::cache_state::import_report(
                run,
                &caches[index].path,
                &task.output.join("cache"),
            )?;
            caches[index].verify()?;
            json!({"programs": programs})
        }
    };
    write_json(
        &checkpoint,
        &json!({"translator_identity": translator_identity(), "input_sha256": task.hash,
        "output_sha256": tree_hash(&task.output, true)?,
        "result_sha256": super::cache::hash(result.to_string().as_bytes()), "result": result}),
    )?;
    Ok(result)
}

fn merge(output: &Path, sources: &[PathBuf]) -> Result<PathBuf> {
    let hashes = sources
        .iter()
        .map(|s| tree_hash(s, false))
        .collect::<Result<Vec<_>>>()?;
    let translator = translator_identity();
    let mut parts = vec![translator.as_str()];
    parts.extend(hashes.iter().map(String::as_str));
    let hash = identity(&parts);
    let stage = output.join(".work").join(format!("merge-{hash}"));
    let checkpoint = stage.join(CHECKPOINT);
    if let Ok(bytes) = fs::read(&checkpoint) {
        if let Ok(old) = serde_json::from_slice::<Value>(&bytes) {
            if old["translator_identity"] == translator
                && old["input_sha256"] == hash
                && old["output_sha256"] == tree_hash(&stage, true)?
            {
                return Ok(stage.join("cache"));
            }
        }
    }
    if stage.exists() {
        fs::remove_dir_all(&stage).map_err(|e| e.to_string())?;
    }
    fs::create_dir_all(&stage).map_err(|e| e.to_string())?;
    shadowbox::merge_disk_caches(&stage.join("cache"), sources)
        .map_err(|e| format!("cache-merge: {e}"))?;
    write_json(
        &checkpoint,
        &json!({"translator_identity": translator, "input_sha256": hash, "output_sha256": tree_hash(&stage, true)?}),
    )?;
    Ok(stage.join("cache"))
}

// Publish a fresh set instead of merging into an old set, so removed inputs
// cannot leave obsolete runtime entries. The previous manifest is kept on errors.
fn publish_cache(output: &Path, source: &Path) -> Result<()> {
    let cache = output.join("cache");
    let pending = output.join("cache.pending");
    let previous = output.join("cache.previous");
    // Recover directory swaps interrupted before or after the second rename.
    if previous.exists() {
        if !cache.exists() {
            fs::rename(&previous, &cache).map_err(|e| e.to_string())?;
        } else {
            fs::remove_dir_all(&previous).map_err(|e| e.to_string())?;
        }
    }
    if pending.exists() {
        fs::remove_dir_all(&pending).map_err(|e| e.to_string())?;
    }
    if cache.exists() && tree_hash(&cache, false)? == tree_hash(source, false)? {
        return Ok(());
    }
    fs::create_dir(&pending).map_err(|e| e.to_string())?;
    let mut files = fs::read_dir(source)
        .map_err(|e| e.to_string())?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()
        .map_err(|e| e.to_string())?;
    files.sort();
    for file in files {
        fs::copy(&file, pending.join(file.file_name().unwrap())).map_err(|e| e.to_string())?;
    }
    if cache.exists() {
        fs::rename(&cache, &previous).map_err(|e| e.to_string())?;
    }
    fs::rename(&pending, &cache).map_err(|e| e.to_string())?;
    if previous.exists() {
        fs::remove_dir_all(previous).map_err(|e| e.to_string())?;
    }
    Ok(())
}

fn manifest(
    results: &[Value],
    tasks: &[Task],
    romfs: &Source,
    caches: &[Source],
    builder: &str,
) -> Result<Value> {
    let mut programs =
        BTreeMap::<String, (String, BTreeSet<String>, BTreeMap<String, Value>)>::new();
    let mut sources = Vec::new();
    let mut diagnostics = Value::Null;
    let mut scan_summary = Value::Null;
    for (result, task) in results.iter().zip(tasks) {
        let source = match task.job {
            Job::Scan => {
                diagnostics = result["diagnostics"].clone();
                scan_summary = result["summary"].clone();
                "scan".to_owned()
            }
            Job::Import(i) => format!("cache:{}", caches[i].hash),
        };
        sources.push(json!({"source": source, "input_sha256": task.hash}));
        for program in result["programs"]
            .as_array()
            .ok_or("checkpoint lacks programs")?
        {
            let id = program["program_key"]
                .as_str()
                .ok_or("checkpoint lacks program key")?;
            let entry = programs.entry(id.to_owned()).or_insert_with(|| {
                (
                    program["stage"].as_str().unwrap_or("unknown").to_owned(),
                    BTreeSet::new(),
                    BTreeMap::new(),
                )
            });
            entry.1.insert(source.clone());
            let mut variant = program.clone();
            variant.as_object_mut().unwrap().remove("program_key");
            variant.as_object_mut().unwrap().remove("stage");
            entry.2.insert(variant.to_string(), variant);
        }
    }
    let (mut scan, mut cache, mut both, mut complete, mut missing, mut failed) = (0, 0, 0, 0, 0, 0);
    let mut failures = Vec::new();
    let mut entries = Vec::new();
    for (id, (stage, sources, variants)) in programs {
        let scanned = sources.contains("scan");
        let cached = sources.iter().any(|s| s.starts_with("cache:"));
        scan += usize::from(scanned);
        cache += usize::from(cached);
        both += usize::from(scanned && cached);
        let variants = variants.into_values().collect::<Vec<_>>();
        let status = if variants.iter().any(|v| v["status"] == "complete") {
            complete += 1;
            "complete"
        } else if variants.iter().any(|v| v["status"] == "failed") {
            failed += 1;
            "failed"
        } else {
            missing += 1;
            "state_missing"
        };
        for variant in &variants {
            if variant["status"] == "failed" {
                failures.push(
                    json!({"program_key": id, "translation_key": variant["translation_key"],
                    "first_blocker": variant["first_blocker"]}),
                );
            }
        }
        entries.push(
            json!({"program_key": id, "stage": stage, "sources": sources,
            "status": status, "variants": variants}),
        );
    }
    Ok(json!({"format": "shadowbox-prewarm-pack", "version": 1,
        "translator_version": shadowbox::TRANSLATOR_VERSION,
        "translator_identity": translator_identity(),
        "cache_format_version": shadowbox::CACHE_FORMAT_VERSION, "builder_sha256": builder,
        "romfs_sha256": romfs.hash, "sources": sources,
        "counts": {"programs_found_by_scan": scan, "programs_found_by_cache": cache,
            "programs_found_by_both": both, "translated_complete": complete,
            "state_missing": missing, "failed": failed, "total_programs": entries.len()},
        "programs": entries, "failures": failures, "scan_summary": scan_summary,
        "scan_diagnostics": diagnostics, "runtime_cache": "cache"}))
}

/// `shadowbox prewarm-pack`.
pub(super) struct PrewarmPack {
    output: PathBuf,
    romfs: PathBuf,
    caches: Vec<PathBuf>,
    key: Option<PathBuf>,
}

impl super::Subcommand for PrewarmPack {
    fn parse(args: &[String]) -> Result<Self> {
        let Some(output) = args.first() else {
            return Err(super::usage().into());
        };
        let (mut romfs, mut caches, mut key) = (None, Vec::new(), None);
        let mut flags = args[1..].iter();
        while let Some(flag) = flags.next() {
            match flag.as_str() {
                "--romfs" if romfs.is_none() => {
                    romfs = Some(PathBuf::from(
                        flags.next().ok_or("--romfs needs a directory")?,
                    ))
                }
                "--emulator-cache" => caches.push(PathBuf::from(
                    flags.next().ok_or("--emulator-cache needs a directory")?,
                )),
                "--pak-aes-key-file" if key.is_none() => {
                    key = Some(PathBuf::from(
                        flags.next().ok_or("--pak-aes-key-file needs a file")?,
                    ))
                }
                _ => {
                    return Err(format!(
                        "unknown or repeated prewarm-pack argument {flag:?}"
                    ))
                }
            }
        }
        Ok(Self {
            output: PathBuf::from(output),
            romfs: romfs.ok_or("prewarm-pack requires --romfs <dir>")?,
            caches,
            key,
        })
    }

    fn run<R: shadowbox::Run>(self, run: R) -> Result<()> {
        let Self {
            output,
            romfs,
            caches,
            key,
        } = self;
        let romfs = Source::new(&romfs)?;
        let mut caches = caches
            .iter()
            .map(|p| Source::new(p))
            .collect::<Result<Vec<_>>>()?;
        caches.sort_by(|a, b| a.hash.cmp(&b.hash).then(a.path.cmp(&b.path)));
        caches.dedup_by(|a, b| a.hash == b.hash);
        let secret_hash = key_hash(key.as_deref())?;
        let builder = file_hash(&std::env::current_exe().map_err(|e| e.to_string())?)?;
        fs::create_dir_all(&output).map_err(|e| e.to_string())?;
        let output = output.canonicalize().map_err(|e| e.to_string())?;
        for source in std::iter::once(&romfs).chain(&caches) {
            if output.starts_with(&source.path) || source.path.starts_with(&output) {
                return Err("package output and input directories must not overlap".into());
            }
        }
        if key
            .as_deref()
            .is_some_and(|p| p.canonicalize().is_ok_and(|p| p.starts_with(&output)))
        {
            return Err("pak AES key file must be outside the package output".into());
        }
        // Validate the lock itself before opening it; other paths are checked once
        // we hold the lock, avoiding races with another package builder.
        let lock_path = output.join(".prewarm-pack.lock");
        if let Ok(metadata) = fs::symlink_metadata(&lock_path) {
            if !metadata.is_file() || metadata.file_type().is_symlink() {
                return Err("package lock must be a regular file".into());
            }
        }
        // OS locks release on process death, so interrupted builds resume without
        // PID polling or stale-lock deletion. Keep the inode for waiting builders.
        let mut lock = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(&lock_path)
            .map_err(|e| format!("open package lock: {e}"))?;
        lock.try_lock()
            .map_err(|e| format!("another prewarm-pack build holds the package lock: {e}"))?;
        validate_output(&output)?;
        lock.set_len(0).map_err(|e| e.to_string())?;
        writeln!(lock, "{}", std::process::id()).map_err(|e| e.to_string())?;
        let marker = output.join(".prewarm-pack-format");
        if marker.exists() {
            if fs::read(&marker).map_err(|e| e.to_string())? != FORMAT.as_bytes() {
                return Err("unsupported package format".into());
            }
        } else {
            for entry in fs::read_dir(&output).map_err(|e| e.to_string())? {
                let path = entry.map_err(|e| e.to_string())?.path();
                if path != lock_path && path != marker.with_extension("pending") {
                    return Err("output must be empty or an existing prewarm package".into());
                }
            }
            let pending = marker.with_extension("pending");
            if pending.exists() {
                fs::remove_file(pending).map_err(|e| e.to_string())?;
            }
            atomic_write(&marker, FORMAT.as_bytes())?;
        }
        // A killed process can leave atomic-file temporaries. They never count as checkpoints.
        let pending = output.join("manifest.pending");
        if pending.exists() {
            fs::remove_file(pending).map_err(|e| e.to_string())?;
        }
        let work = output.join(".work");
        fs::create_dir_all(&work).map_err(|e| e.to_string())?;
        let translator = translator_identity();
        let mut scan_parts = vec![
            translator.as_str(),
            FORMAT.trim(),
            builder.as_str(),
            romfs.hash.as_str(),
            secret_hash.as_str(),
        ];
        scan_parts.extend(caches.iter().map(|c| c.hash.as_str()));
        let hash = identity(&scan_parts);
        let mut tasks = vec![Task {
            job: Job::Scan,
            output: work.join(format!("scan-{hash}")),
            hash,
        }];
        for (index, source) in caches.iter().enumerate() {
            let hash = identity(&[FORMAT.trim(), &translator, &builder, &source.hash]);
            tasks.push(Task {
                job: Job::Import(index),
                output: work.join(format!("import-{hash}")),
                hash,
            });
        }
        let workers = std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(tasks.len());
        let next = AtomicUsize::new(0);
        let results = std::thread::scope(|scope| {
            let handles = (0..workers)
                .map(|_| {
                    scope.spawn(|| {
                        let mut results = Vec::new();
                        loop {
                            let index = next.fetch_add(1, Ordering::Relaxed);
                            let Some(task) = tasks.get(index) else { break };
                            results.push((
                                index,
                                execute(run, task, &romfs, &caches, key.as_deref(), &secret_hash),
                            ));
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
                        .map_err(|_| "prewarm worker panicked".to_owned())?,
                );
            }
            results.sort_by_key(|(index, _)| *index);
            results
                .into_iter()
                .map(|(_, result)| result)
                .collect::<Result<Vec<_>>>()
        })?;
        romfs.verify()?;
        for source in &caches {
            source.verify()?;
        }
        if key_hash(key.as_deref())? != secret_hash {
            return Err("pak AES key changed during prewarm-pack".into());
        }
        let sources = tasks
            .iter()
            .map(|t| t.output.join("cache"))
            .collect::<Vec<_>>();
        let merged = merge(&output, &sources)?;
        let mut manifest = manifest(&results, &tasks, &romfs, &caches, &builder)?;
        let stats = shadowbox::disk_cache_stats(&merged).map_err(|e| e.to_string())?;
        manifest["runtime_cache_entries"] = json!(stats.entries);
        publish_cache(&output, &merged)?;
        write_json(&output.join("manifest.json"), &manifest)?;
        println!(
        "prewarm-pack: scan={} cache={} both={} complete={} state_missing={} failed={} entries={}",
        manifest["counts"]["programs_found_by_scan"],
        manifest["counts"]["programs_found_by_cache"],
        manifest["counts"]["programs_found_by_both"],
        manifest["counts"]["translated_complete"],
        manifest["counts"]["state_missing"],
        manifest["counts"]["failed"],
        stats.entries
    );
        Ok(())
    }
}
