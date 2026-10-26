// Project-owned corpus gate. No instruction fields or shader mappings are changed.
use crate::value_diff::{self, bounded, read_limited, Scratch};
use serde_json::{json, Value};
use std::{
    collections::BTreeMap,
    fs,
    os::unix::fs::PermissionsExt,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
    thread,
    time::Duration,
};

/// `shadowbox gate`.
pub(super) struct Gate {
    old: PathBuf,
    new: PathBuf,
    corpora: Vec<PathBuf>,
    jobs: usize,
    seed: u64,
    timeout: Duration,
}

impl crate::Subcommand for Gate {
    fn parse(args: &[String]) -> Result<Self, String> {
        let (mut old, mut new) = (None, None);
        let mut corpora = Vec::new();
        let (mut jobs, mut seed, mut timeout) = (1, 0, 5000);
        let mut i = 0;
        while i < args.len() {
            let arg = &args[i];
            if arg.starts_with("--") {
                i += 1;
                let value = args.get(i).ok_or_else(|| format!("{arg} needs a value"))?;
                match arg.as_str() {
                    "--old" => old = Some(executable(value)?),
                    "--new" => new = Some(executable(value)?),
                    "--jobs" => jobs = value.parse().map_err(|_| "invalid jobs")?,
                    "--seed" => seed = value.parse().map_err(|_| "invalid seed")?,
                    "--timeout-ms" => timeout = value.parse().map_err(|_| "invalid timeout")?,
                    _ => return Err(format!("unknown gate option {arg}")),
                }
            } else {
                corpora.push(PathBuf::from(arg));
            }
            i += 1;
        }
        if corpora.is_empty() {
            return Err("gate requires at least one corpus directory".into());
        }
        if !(1..=64).contains(&jobs) || !(1..=60000).contains(&timeout) {
            return Err("jobs must be 1..64 and timeout-ms must be 1..60000".into());
        }
        Ok(Self {
            old: old.ok_or("gate requires --old")?,
            new: new.ok_or("gate requires --new")?,
            corpora,
            jobs,
            seed,
            timeout: Duration::from_millis(timeout),
        })
    }

    fn run<R: shadowbox::Run>(self, run: R) -> Result<(), String> {
        let options = self;
        // Discover every corpus first to avoid misleading reports from partial setup.
        let corpora = options
            .corpora
            .iter()
            .map(|directory| {
                let directory = fs::canonicalize(directory)
                    .map_err(|e| format!("corpus {}: {e}", directory.display()))?;
                let paths = discover(&directory)?;
                Ok((directory, paths))
            })
            .collect::<Result<Vec<_>, String>>()?;
        let validator = bounded(Command::new("spirv-val").arg("--version"), options.timeout)?;
        if !validator.success {
            return Err(format!(
                "spirv-val setup failed: {}",
                first_blocker(&validator.log)
            ));
        }
        let mut failed = false;
        for (corpus, paths) in corpora {
            let next = AtomicUsize::new(0);
            let mut rows = thread::scope(|scope| {
                let handles = (0..options.jobs.min(paths.len()))
                    .map(|_| {
                        scope.spawn(|| {
                            let mut rows = Vec::new();
                            loop {
                                let index = next.fetch_add(1, Ordering::Relaxed);
                                let Some(path) = paths.get(index) else { break };
                                rows.push((index, check(run, path, &options)));
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
            rows.sort_by_key(|r| r.0);
            let rows = rows
                .into_iter()
                .map(|(_, mut row)| {
                    row["corpus"] = json!(corpus);
                    println!("{row}");
                    row
                })
                .collect::<Vec<_>>();
            let summary = summary(&rows, options.seed);
            failed |= [
                "regressions",
                "value_changes",
                "uncheckable",
                "input_errors",
            ]
            .iter()
            .any(|key| summary[key].as_u64().unwrap() != 0);
            println!("{}", json!({"corpus": corpus, "summary": summary}));
        }
        if failed {
            Err("gate found regressions, value changes, or incomplete comparisons".into())
        } else {
            Ok(())
        }
    }
}

fn executable(path: &str) -> Result<PathBuf, String> {
    let path = fs::canonicalize(path).map_err(|e| format!("translator {path}: {e}"))?;
    let metadata = fs::metadata(&path).map_err(|e| e.to_string())?;
    if !metadata.is_file() || metadata.permissions().mode() & 0o111 == 0 {
        return Err(format!(
            "translator {} must be an executable file",
            path.display()
        ));
    }
    Ok(path)
}

fn discover(directory: &Path) -> Result<Vec<PathBuf>, String> {
    let mut pending = vec![directory.to_path_buf()];
    let mut programs = Vec::new();
    while let Some(directory) = pending.pop() {
        for entry in
            fs::read_dir(&directory).map_err(|e| format!("corpus {}: {e}", directory.display()))?
        {
            let entry = entry.map_err(|e| format!("corpus entry: {e}"))?;
            let kind = entry.file_type().map_err(|e| e.to_string())?;
            if kind.is_dir() {
                pending.push(entry.path());
            } else if kind.is_file()
                && entry.path().extension().and_then(|e| e.to_str()) == Some("bin")
            {
                programs.push(entry.path());
            }
        }
    }
    programs.sort();
    if programs.is_empty() {
        return Err(format!("no .bin programs found in {}", directory.display()));
    }
    Ok(programs)
}

fn first_blocker(reason: &str) -> String {
    reason
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("process failed without a diagnostic")
        .chars()
        .take(1024)
        .collect()
}

fn outcome(result: Result<(), String>) -> Value {
    match result {
        Ok(()) => json!({"pass": true}),
        Err(reason) => json!({"pass": false, "first_blocker": first_blocker(&reason)}),
    }
}

// Snapshot both inputs. Raw little-endian CB1 words follow the project decoder
// contract at decoder.rs:3584-3585,3623-3630. No CB1 fields are decoded here.
pub(super) fn snapshot(
    program: &Path,
    scratch: &Path,
) -> Result<(PathBuf, Option<PathBuf>), String> {
    let bytes = read_limited(program, value_diff::MAX_PROGRAM)?;
    shadowbox::header_prefixed_input(&bytes, shadowbox::TargetGeneration::Sm5)
        .map_err(|e| format!("not a header-prefixed program: {e}"))?;
    let input = scratch.join(program.file_name().ok_or("missing program filename")?);
    fs::write(&input, bytes).map_err(|e| format!("snapshot program: {e}"))?;
    let sidecar = program.with_extension("cb1");
    let sidecar = match fs::symlink_metadata(&sidecar) {
        Ok(metadata) if metadata.is_file() => {
            let bytes = read_limited(&sidecar, 65536)?;
            if !bytes.len().is_multiple_of(4) {
                return Err("CB1 sidecar byte length must be a multiple of four".into());
            }
            let path = input.with_extension("cb1");
            fs::write(&path, bytes).map_err(|e| format!("snapshot CB1: {e}"))?;
            Some(path)
        }
        Ok(_) => {
            return Err("CB1 sidecar must be a regular file, not a symlink or directory".into())
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(format!("CB1 sidecar: {e}")),
    };
    let storage = program.with_extension("storage.json");
    match fs::symlink_metadata(&storage) {
        Ok(metadata) if metadata.is_file() => {
            fs::write(
                input.with_extension("storage.json"),
                read_limited(&storage, 65536)?,
            )
            .map_err(|e| format!("snapshot storage state: {e}"))?;
        }
        Ok(_) => return Err("storage sidecar must be a regular file".into()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(format!("storage sidecar: {e}")),
    }
    Ok((input, sidecar))
}

fn compare(
    input: &Path,
    old: &Path,
    new: &Path,
    scratch: &Path,
    options: &Gate,
) -> Result<Value, String> {
    let output = scratch.join("comparison.json");
    let result = bounded(
        Command::new(std::env::current_exe().map_err(|e| e.to_string())?)
            .arg("__value-diff-compare")
            .arg(input)
            .arg(old)
            .arg(new)
            .arg(&output)
            .arg(options.seed.to_string())
            .arg(options.timeout.as_millis().to_string()),
        // Two GPU workers plus bounded-size source and module analysis.
        options.timeout * 3,
    )?;
    if !result.success {
        return Err(first_blocker(&result.log));
    }
    let row: Value = serde_json::from_slice(&read_limited(&output, 65536)?)
        .map_err(|e| format!("comparison report: {e}"))?;
    if !matches!(
        row["status"].as_str(),
        Some("equal" | "different" | "uncheckable")
    ) {
        return Err("invalid value checker status".into());
    }
    Ok(row)
}

fn check<R: shadowbox::Run>(run: R, program: &Path, options: &Gate) -> Value {
    let result = (|| -> Result<Value, String> {
        let scratch = Scratch::new()?;
        let (input, sidecar) = snapshot(program, &scratch.0)?;
        let old_spv = scratch.0.join("old.spv");
        let new_spv = scratch.0.join("new.spv");
        let old = outcome(
            value_diff::translate(
                run,
                &options.old,
                &input,
                &old_spv,
                options.timeout,
                sidecar.as_deref(),
            )
            .map_err(String::from),
        );
        let new = outcome(
            value_diff::translate(
                run,
                &options.new,
                &input,
                &new_spv,
                options.timeout,
                sidecar.as_deref(),
            )
            .map_err(String::from),
        );
        let comparison = if old["pass"] == true && new["pass"] == true {
            compare(&input, &old_spv, &new_spv, &scratch.0, options)
                .unwrap_or_else(|reason| json!({"status": "uncheckable", "reason": reason}))
        } else {
            Value::Null
        };
        Ok(json!({"old": old, "new": new, "comparison": comparison}))
    })();
    let mut row = result.unwrap_or_else(|reason| {
        json!({
            "old": outcome(Err(reason.clone())), "new": outcome(Err(reason.clone())),
            "input_error": reason, "comparison": null
        })
    });
    row["program"] = json!(program);
    row["regression"] = json!(row["old"]["pass"] == true && row["new"]["pass"] == false);
    row["value_change"] = json!(row["comparison"]["status"] == "different");
    row
}

fn summary(rows: &[Value], seed: u64) -> Value {
    let count = |predicate: fn(&Value) -> bool| rows.iter().filter(|r| predicate(r)).count();
    let mut blockers = BTreeMap::<String, usize>::new();
    for row in rows {
        if let Some(reason) = row["new"]["first_blocker"].as_str() {
            *blockers.entry(reason.to_string()).or_default() += 1;
        }
    }
    let mut blockers = blockers.into_iter().collect::<Vec<_>>();
    blockers.sort_by(|(a, ac), (b, bc)| bc.cmp(ac).then(a.cmp(b)));
    let blockers = blockers
        .into_iter()
        .take(10)
        .map(|(first_blocker, count)| json!({"first_blocker": first_blocker, "count": count}))
        .collect::<Vec<_>>();
    json!({
        "programs": rows.len(), "seed": seed,
        "old_pass": count(|r| r["old"]["pass"] == true),
        "new_pass": count(|r| r["new"]["pass"] == true),
        "regressions": count(|r| r["regression"] == true),
        "value_changes": count(|r| r["value_change"] == true),
        "equal": count(|r| r["comparison"]["status"] == "equal"),
        "uncheckable": count(|r| r["comparison"]["status"] == "uncheckable"),
        "input_errors": count(|r| r.get("input_error").is_some()),
        "top_first_blockers": blockers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn summary_keeps_translation_passes_separate_from_value_changes() {
        let rows = vec![
            json!({"old": {"pass": true}, "new": {"pass": false, "first_blocker": "Unknown 0x1234"}, "regression": true}),
            json!({"old": {"pass": false}, "new": {"pass": true}}),
            json!({"old": {"pass": true}, "new": {"pass": true}, "value_change": true, "comparison": {"status": "different"}}),
            json!({"old": {"pass": false}, "new": {"pass": false, "first_blocker": "Unknown 0x1234"}}),
        ];
        let report = summary(&rows, 42);
        assert_eq!(report["old_pass"], 2);
        assert_eq!(report["new_pass"], 2);
        assert_eq!(report["regressions"], 1);
        assert_eq!(report["value_changes"], 1);
        assert_eq!(report["top_first_blockers"][0]["count"], 2);
    }
}
