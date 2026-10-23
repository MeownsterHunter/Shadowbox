use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use shadowbox::{decoder::decode_word, ir::InstructionKind, TRANSLATOR_VERSION};
use std::{collections::BTreeMap, fs, path::Path, process::Command};

const FORMAT: &str = "shadowbox-sudoku-integration-catalog-v1";
const HEADER: &str = "#row\tword\tbase\texact\tcanonical\tpredicted\tnvdisasm";

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct Candidate {
    pub(crate) row: u32,
    pub(crate) word: u64,
    pub(crate) base: u64,
    pub(crate) predicted: String,
}

#[derive(Debug)]
pub(crate) struct ValidatedInput {
    pub(crate) candidates: Vec<Candidate>,
    pub(crate) excluded: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceLock {
    head: String,
    root: Option<std::path::PathBuf>,
}

impl SourceLock {
    pub fn acquire(root: &Path, expected_head: &str) -> Result<Self, String> {
        let root = root
            .canonicalize()
            .map_err(|error| format!("canonicalize {}: {error}", root.display()))?;
        let head = git(&root, &["rev-parse", "HEAD"])?;
        if head != expected_head {
            return Err(format!(
                "stale source checkout: expected {expected_head}, found {head}"
            ));
        }
        let dirty = git(&root, &["status", "--porcelain=v1"])?;
        if !dirty.is_empty() {
            return Err("source checkout has changes".into());
        }
        let actual = translator_source_digest(&root)?;
        if actual != TRANSLATOR_VERSION {
            return Err(format!(
                "running decoder source digest {TRANSLATOR_VERSION} does not match checkout {actual}"
            ));
        }
        Ok(Self {
            head,
            root: Some(root),
        })
    }

    pub(crate) fn verify_unchanged(&self) -> Result<(), String> {
        let Some(root) = &self.root else {
            return Ok(());
        };
        let current = Self::acquire(root, &self.head)?;
        debug_assert_eq!(current.head, self.head);
        Ok(())
    }

    pub(crate) fn head(&self) -> &str {
        &self.head
    }

    pub(crate) fn root(&self) -> Option<&Path> {
        self.root.as_deref()
    }
}

pub fn build_catalog(lock: SourceLock, model: &Path, input: &Path) -> Result<Vec<u8>, String> {
    let model_bytes = read(model)?;
    let input_bytes = read(input)?;
    let model_rows = model_rows(&model_bytes)?;
    let validated = candidates(&input_bytes, &model_rows)?;

    let input_forms = validated.candidates.len();
    let input_rows_excluded = validated.excluded;
    let mut unknown = BTreeMap::new();
    let mut decoded = 0_u64;
    for candidate in validated.candidates {
        let words = [candidate.word as u32, (candidate.word >> 32) as u32];
        if matches!(decode_word(words, 0).kind, InstructionKind::Unknown { .. }) {
            unknown.insert((candidate.row, candidate.word), candidate.predicted);
        } else {
            decoded += 1;
        }
    }
    lock.verify_unchanged()?;

    let mut body = String::new();
    for ((row, word), predicted) in &unknown {
        body.push_str(&format!("{row}\t{word:016x}\t{predicted}\n"));
    }
    let output = format!(
        "# {FORMAT}\nsource_head\t{}\nsource_sha256\t{}\nmodel_sha256\t{}\ninput_sha256\t{}\ncatalog_sha256\t{}\ninput_rows_excluded\t{}\ninput_forms\t{}\ndecoded_subtracted\t{}\nunknown_forms\t{}\n# row\tword\tpredicted\n{}",
        lock.head,
        TRANSLATOR_VERSION,
        digest(&model_bytes),
        digest(&input_bytes),
        digest(body.as_bytes()),
        input_rows_excluded,
        input_forms,
        decoded,
        unknown.len(),
        body
    );
    Ok(output.into_bytes())
}

#[derive(Debug, Deserialize)]
pub(crate) struct ModelRow {
    pub(crate) fmask: u64,
    pub(crate) groups: BTreeMap<String, ModelGroup>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ModelGroup {
    pub(crate) fields: Vec<ModelField>,
    pub(crate) bases: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ModelField {
    pub(crate) parts: Vec<(u8, u8)>,
    pub(crate) role: String,
    #[serde(default)]
    float: bool,
    #[serde(default)]
    weights: Option<Value>,
    #[serde(default)]
    fenc: Option<Value>,
}

pub(crate) fn model_rows(bytes: &[u8]) -> Result<BTreeMap<u32, ModelRow>, String> {
    let object: BTreeMap<String, ModelRow> = serde_json::from_slice(bytes)
        .map_err(|error| format!("Sudoku model is not valid JSON: {error}"))?;
    object
        .into_iter()
        .map(|(key, model)| {
            key.parse::<u32>()
                .map(|row| (row, model))
                .map_err(|_| format!("Sudoku model row {key:?} is not an unsigned integer"))
        })
        .collect()
}

pub(crate) fn candidates(
    bytes: &[u8],
    model_rows: &BTreeMap<u32, ModelRow>,
) -> Result<ValidatedInput, String> {
    let input = std::str::from_utf8(bytes).map_err(|_| "Sudoku input is not UTF-8")?;
    let mut lines = input.lines();
    if lines.next() != Some(HEADER) {
        return Err(format!("Sudoku input must start with {HEADER:?}"));
    }
    let mut candidates = Vec::new();
    let mut excluded = 0;
    for (index, line) in lines.enumerate() {
        if line.is_empty() {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 7 {
            return Err(format!(
                "Sudoku input line {} must have seven tab-separated fields",
                index + 2
            ));
        }
        let row = fields[0]
            .parse::<u32>()
            .map_err(|_| format!("invalid Sudoku row on line {}", index + 2))?;
        let model = model_rows.get(&row).ok_or_else(|| {
            format!(
                "Sudoku input line {} names row {row}, absent from the pinned model",
                index + 2
            )
        })?;
        let word = u64::from_str_radix(fields[1], 16)
            .map_err(|_| format!("invalid Sudoku word on line {}", index + 2))?;
        let base = u64::from_str_radix(fields[2], 16)
            .map_err(|_| format!("invalid Sudoku base on line {}", index + 2))?;
        if !matches!(fields[3], "0" | "1") || !matches!(fields[4], "0" | "1") {
            return Err(format!(
                "Sudoku input line {} has a non-boolean exact or canonical field",
                index + 2
            ));
        }
        if fields[3] != "1" || fields[4] != "1" || fields[5] == "!REJECT" {
            excluded += 1;
            continue;
        }
        if fields[5] != fields[6] {
            return Err(format!(
                "Sudoku input line {} has conflicting exact prediction and oracle text",
                index + 2
            ));
        }
        require_model_coverage(model, base, word, index + 2)?;
        candidates.push(Candidate {
            row,
            word,
            base,
            predicted: fields[5].to_owned(),
        });
    }
    candidates.sort();
    for pair in candidates.windows(2) {
        if (pair[0].row, pair[0].word) == (pair[1].row, pair[1].word)
            && (pair[0].base != pair[1].base || pair[0].predicted != pair[1].predicted)
        {
            return Err(format!(
                "Sudoku input gives conflicting predictions for row {} word {:016x}",
                pair[0].row, pair[0].word
            ));
        }
    }
    candidates.dedup();
    Ok(ValidatedInput {
        candidates,
        excluded,
    })
}

fn require_model_coverage(
    model: &ModelRow,
    base: u64,
    word: u64,
    line: usize,
) -> Result<(), String> {
    let group_name = if model.fmask == 0 {
        "-".to_owned()
    } else {
        format!("{:x}", base & model.fmask)
    };
    let group = model.groups.get(&group_name).ok_or_else(|| {
        format!("Sudoku input line {line} base is absent from the pinned model group")
    })?;
    if !group
        .bases
        .iter()
        .any(|value| u64::from_str_radix(value, 16) == Ok(base))
    {
        return Err(format!(
            "Sudoku input line {line} base is absent from the pinned model"
        ));
    }
    let free = group
        .fields
        .iter()
        .filter(|field| field.composed())
        .fold(0_u64, |mask, field| {
            field.parts.iter().fold(mask, |mask, &(lo, hi)| {
                let width = u32::from(hi - lo + 1);
                let bits = if width == 64 {
                    u64::MAX
                } else {
                    (1_u64 << width) - 1
                };
                mask | (bits << lo)
            })
        });
    if (word ^ base) & !free != 0 {
        return Err(format!(
            "Sudoku input line {line} word is not composed from its pinned model base"
        ));
    }
    Ok(())
}

impl ModelField {
    fn composed(&self) -> bool {
        matches!(self.role.as_str(), "reg" | "mod" | "flag" | "none")
            || self.role == "imm"
                && ((!self.float && self.weights.as_ref().is_some_and(nonempty_json))
                    || self.fenc.as_ref().is_some_and(nonempty_json))
    }
}

fn nonempty_json(value: &Value) -> bool {
    match value {
        Value::Null | Value::Bool(false) => false,
        Value::Array(values) => !values.is_empty(),
        Value::Object(values) => !values.is_empty(),
        Value::String(value) => !value.is_empty(),
        Value::Number(_) | Value::Bool(true) => true,
    }
}

fn translator_source_digest(root: &Path) -> Result<String, String> {
    let crate_root = root.join("crates/shadowbox");
    let mut source_paths = Vec::new();
    rust_sources(&crate_root.join("src"), &mut source_paths)?;
    let mut files = source_paths
        .into_iter()
        .map(|path| {
            let name = path.strip_prefix(&crate_root).unwrap().to_path_buf();
            (name, path)
        })
        .collect::<Vec<_>>();
    files.extend([
        (
            Path::new("build.rs").to_path_buf(),
            crate_root.join("build.rs"),
        ),
        (
            Path::new("Cargo.toml").to_path_buf(),
            crate_root.join("Cargo.toml"),
        ),
        (
            Path::new("../../Cargo.toml").to_path_buf(),
            root.join("Cargo.toml"),
        ),
        (
            Path::new("../../Cargo.lock").to_path_buf(),
            root.join("Cargo.lock"),
        ),
    ]);
    files.sort_by(|a, b| a.0.cmp(&b.0));
    let mut hash = Sha256::new();
    hash.update(b"shadowbox-translator-sources-v1");
    for (name, path) in files {
        let name = name
            .to_str()
            .ok_or_else(|| format!("translator source path is not UTF-8: {}", path.display()))?
            .as_bytes();
        let bytes = read(&path)?;
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name);
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn rust_sources(dir: &Path, files: &mut Vec<std::path::PathBuf>) -> Result<(), String> {
    for entry in
        fs::read_dir(dir).map_err(|error| format!("read directory {}: {error}", dir.display()))?
    {
        let path = entry
            .map_err(|error| format!("read entry in {}: {error}", dir.display()))?
            .path();
        if path.is_dir() {
            rust_sources(&path, files)?;
        } else if path.extension().is_some_and(|extension| extension == "rs") {
            files.push(path);
        }
    }
    Ok(())
}

fn git(root: &Path, args: &[&str]) -> Result<String, String> {
    let safe = format!("safe.directory={}", root.display());
    let output = Command::new("git")
        .arg("-c")
        .arg(safe)
        .arg("-C")
        .arg(root)
        .args(args)
        .output()
        .map_err(|error| format!("run git in {}: {error}", root.display()))?;
    if !output.status.success() {
        return Err(format!(
            "git in {} failed: {}",
            root.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout)
        .map(|value| value.trim().to_owned())
        .map_err(|_| "git output is not UTF-8".into())
}

fn read(path: &Path) -> Result<Vec<u8>, String> {
    fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))
}

pub(crate) fn digest(bytes: impl AsRef<[u8]>) -> String {
    format!("{:x}", Sha256::digest(bytes.as_ref()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    const EXIT: &str = "e30000000007000f";
    const UNKNOWN: &str = "0000000000070000";

    fn model() -> Vec<u8> {
        format!(
            r#"{{"7":{{"fmask":0,"groups":{{"-":{{"fields":[{{"parts":[[0,63]],"role":"none"}}],"bases":["{UNKNOWN}","{EXIT}"]}}}}}}}}"#
        )
        .into_bytes()
    }

    fn input(rows: &[(u32, &str, &str, &str, &str)]) -> Vec<u8> {
        let mut value = format!("{HEADER}\n");
        for (row, word, exact, canonical, text) in rows {
            value.push_str(&format!(
                "{row}\t{word}\t{word}\t{exact}\t{canonical}\t{text}\t{text}\n"
            ));
        }
        value.into_bytes()
    }

    fn unknown_words(model: &[u8], input: &[u8]) -> Vec<u64> {
        let rows = model_rows(model).unwrap();
        candidates(input, &rows)
            .unwrap()
            .candidates
            .into_iter()
            .filter_map(|candidate| {
                let words = [candidate.word as u32, (candidate.word >> 32) as u32];
                matches!(decode_word(words, 0).kind, InstructionKind::Unknown { .. })
                    .then_some(candidate.word)
            })
            .collect()
    }

    #[test]
    fn complete_catalog_replays_byte_identically_with_all_digests() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "shadowbox-sudoku-catalog-{}-{suffix}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        let model_path = root.join("model.json");
        let input_path = root.join("validate-words.tsv");
        fs::write(&model_path, model()).unwrap();
        fs::write(&input_path, input(&[(7, UNKNOWN, "1", "1", "UNKNOWN")])).unwrap();
        let lock = || SourceLock {
            head: "a".repeat(40),
            root: None,
        };
        let first = build_catalog(lock(), &model_path, &input_path).unwrap();
        let second = build_catalog(lock(), &model_path, &input_path).unwrap();
        assert_eq!(first, second);
        let text = String::from_utf8(first).unwrap();
        for label in [
            "source_sha256",
            "model_sha256",
            "input_sha256",
            "catalog_sha256",
        ] {
            assert!(text.contains(&format!("{label}\t")));
        }
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn decoded_rediscovery_is_subtracted() {
        let words = unknown_words(
            &model(),
            &input(&[
                (7, UNKNOWN, "1", "1", "UNKNOWN"),
                (7, EXIT, "1", "1", "EXIT"),
            ]),
        );
        assert_eq!(words, vec![0x0000_0000_0007_0000]);
    }

    #[test]
    fn input_order_does_not_change_the_unknown_catalog() {
        let a = input(&[
            (7, UNKNOWN, "1", "1", "UNKNOWN"),
            (7, EXIT, "1", "1", "EXIT"),
        ]);
        let b = input(&[
            (7, EXIT, "1", "1", "EXIT"),
            (7, UNKNOWN, "1", "1", "UNKNOWN"),
        ]);
        assert_eq!(unknown_words(&model(), &a), unknown_words(&model(), &b));
    }

    #[test]
    fn unmodeled_and_inexact_inputs_are_rejected() {
        let rows = model_rows(&model()).unwrap();
        let absent = input(&[(8, UNKNOWN, "1", "1", "UNKNOWN")]);
        assert!(candidates(&absent, &rows)
            .unwrap_err()
            .contains("absent from the pinned model"));
        let malformed = input(&[(7, UNKNOWN, "maybe", "1", "UNKNOWN")]);
        assert!(candidates(&malformed, &rows)
            .unwrap_err()
            .contains("non-boolean"));
        let conflicting = String::from_utf8(input(&[(7, UNKNOWN, "1", "1", "UNKNOWN")]))
            .unwrap()
            .replace("\tUNKNOWN\tUNKNOWN", "\tUNKNOWN\tOTHER");
        assert!(candidates(conflicting.as_bytes(), &rows)
            .unwrap_err()
            .contains("conflicting exact prediction"));
        let not_from_model = input(&[(7, "deadbeefdeadbeef", "1", "1", "fabricated")]);
        assert!(candidates(&not_from_model, &rows)
            .unwrap_err()
            .contains("base is absent from the pinned model"));
    }

    #[test]
    fn native_nonexact_rows_are_counted_and_skipped() {
        let rows = model_rows(&model()).unwrap();
        let native = input(&[
            (7, UNKNOWN, "0", "0", "predicted rejection"),
            (7, EXIT, "1", "0", "canonical mismatch"),
            (7, EXIT, "1", "1", "!REJECT"),
            (7, UNKNOWN, "1", "1", "UNKNOWN"),
        ]);
        let validated = candidates(&native, &rows).unwrap();
        assert_eq!(validated.excluded, 3);
        assert_eq!(validated.candidates.len(), 1);
    }

    #[test]
    fn stale_head_is_rejected_before_artifacts_are_read() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let error = SourceLock::acquire(&root, &"0".repeat(40)).unwrap_err();
        assert!(error.starts_with("stale source checkout:"));
    }

    #[test]
    fn compiled_decoder_is_bound_to_checkout_sources() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        assert_eq!(translator_source_digest(&root).unwrap(), TRANSLATOR_VERSION);
    }

    #[test]
    fn untracked_file_makes_checkout_unclean() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "shadowbox-sudoku-source-lock-{}-{suffix}",
            std::process::id()
        ));
        fs::create_dir(&root).unwrap();
        let run = |args: &[&str]| {
            let status = Command::new("git")
                .arg("-C")
                .arg(&root)
                .args(args)
                .env("GIT_AUTHOR_NAME", "test")
                .env("GIT_AUTHOR_EMAIL", "test@example.invalid")
                .env("GIT_COMMITTER_NAME", "test")
                .env("GIT_COMMITTER_EMAIL", "test@example.invalid")
                .status()
                .unwrap();
            assert!(status.success());
        };
        run(&["init", "--quiet"]);
        fs::write(root.join("tracked"), b"tracked").unwrap();
        run(&["add", "tracked"]);
        run(&["commit", "--quiet", "-m", "fixture"]);
        let head = git(&root, &["rev-parse", "HEAD"]).unwrap();
        fs::write(root.join("untracked"), b"untracked").unwrap();
        assert_eq!(
            SourceLock::acquire(&root, &head).unwrap_err(),
            "source checkout has changes"
        );
        fs::remove_dir_all(root).unwrap();
    }
}
