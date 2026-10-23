use sha2::{Digest, Sha256};
use shadowbox::{decoder::decode_word, ir::InstructionKind};
use std::{collections::BTreeMap, fmt::Write as _, fs, path::Path, process::Command};

pub mod conveyor;
pub mod hybrid;
pub mod sudoku;

const FORMAT: &str = "shadowbox-discovery-plan-v1";
const SOURCE_PREFIX: &str = "# Source: ";
const ENVYTOOLS_COMMIT: &str = "f102b82381f3f11cee113d16374c87091db039d9";
const ENVYTOOLS_LICENSE: &str = "MIT";
const ENVYTOOLS_REMOTE: &str = "https://github.com/envytools/envytools.git";
const ENVYTOOLS_LICENSE_SHA256: &str =
    "51ab82ea0c4d850ab5326da7fa02daa5d5f6f02b677831a8e37be93cc7a89493";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Inputs {
    rows: Vec<u8>,
    samples: Vec<u8>,
    results: Vec<u8>,
    semantic_source: SemanticSource,
    claims: Vec<u8>,
}

impl Inputs {
    pub fn read(
        rows: &Path,
        samples: &Path,
        results: &Path,
        semantic_source: &Path,
        claims: &Path,
    ) -> Result<Self, String> {
        Ok(Self {
            rows: read(rows)?,
            samples: read(samples)?,
            results: read(results)?,
            semantic_source: SemanticSource::read(semantic_source)?,
            claims: read(claims)?,
        })
    }

    pub fn classify(self) -> Result<Ledger, String> {
        let rows = text("decoder rows", self.rows)?;
        let samples = text("decoder samples", self.samples)?;
        let results = text("decoder results", self.results)?;
        let claims = text("discovery claims", self.claims)?;
        let row_pin = SourcePin::from_manifest(&rows)?;
        let sample_pin = SourcePin::from_manifest(&samples)?;
        if row_pin != sample_pin {
            return Ok(Ledger::Stale(StaleEvidence {
                rows: row_pin,
                samples: sample_pin,
            }));
        }

        if self.semantic_source.sha256 != row_pin.sha256 {
            return Ok(Ledger::Stale(StaleEvidence {
                rows: row_pin.clone(),
                samples: SourcePin {
                    path: "actual semantic source bytes".into(),
                    sha256: self.semantic_source.sha256,
                },
            }));
        }

        let row_shapes = parse_rows(&rows)?;
        let raw_probes = parse_samples(&samples, &row_shapes)?;
        let outcomes = records(&results)
            .map(|fields| fields.join("\t"))
            .collect::<Vec<_>>();
        if raw_probes.len() != outcomes.len() {
            return Err(format!(
                "decoder result count {} does not match sample count {}",
                outcomes.len(),
                raw_probes.len()
            ));
        }
        let mut probes = Vec::with_capacity(raw_probes.len());
        for (raw_probe, outcome) in raw_probes.into_iter().zip(outcomes) {
            let computed = operation_name(
                &decode_word([raw_probe.raw as u32, (raw_probe.raw >> 32) as u32], 0).kind,
            );
            if outcome != computed {
                return Err(format!(
                    "decoder result drift at row {} field {}: recorded {outcome}, computed {computed}",
                    raw_probe.row, raw_probe.field
                ));
            }
            let evidence = match (outcome.as_str(), raw_probe.reservation) {
                ("Unknown", Reservation::Available) => ProbeEvidence::NeedsEvidence,
                ("Unknown", Reservation::SourceReserved) => ProbeEvidence::SourceReserved,
                (kind, _) => ProbeEvidence::Decoded(kind.to_owned()),
            };
            probes.push(Probe {
                row: raw_probe.row,
                field: raw_probe.field,
                citation: raw_probe.citation,
                word: raw_probe.word,
                evidence,
            });
        }
        probes.sort_by(|a, b| {
            (a.row, &a.field, &a.word, &a.citation).cmp(&(b.row, &b.field, &b.word, &b.citation))
        });
        let conflicts = parse_claims(&claims, &row_shapes)?;
        let mut canonical_evidence = String::new();
        for probe in &probes {
            writeln!(
                canonical_evidence,
                "{}\t{}\t{}\t{}\t{:?}",
                probe.row, probe.field, probe.citation, probe.word, probe.evidence
            )
            .unwrap();
        }
        let evidence_sha256 = digest(canonical_evidence.as_bytes());
        Ok(Ledger::SourceLocked(SourceLockedEvidence {
            source: row_pin,
            evidence_sha256,
            probes,
            conflicts,
        }))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Ledger {
    Stale(StaleEvidence),
    SourceLocked(SourceLockedEvidence),
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StaleEvidence {
    rows: SourcePin,
    samples: SourcePin,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceLockedEvidence {
    source: SourcePin,
    evidence_sha256: String,
    probes: Vec<Probe>,
    conflicts: Vec<Conflict>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SemanticSource {
    sha256: String,
}

impl SemanticSource {
    fn read(path: &Path) -> Result<Self, String> {
        let path = path
            .canonicalize()
            .map_err(|error| format!("canonicalize {}: {error}", path.display()))?;
        let root = path
            .ancestors()
            .find(|ancestor| ancestor.join(".git").exists())
            .ok_or_else(|| {
                format!(
                    "{} is outside the pinned envytools Git repository",
                    path.display()
                )
            })?;
        let relative = path
            .strip_prefix(root)
            .map_err(|_| "semantic source is outside repository root")?;
        if relative != Path::new("envydis/gm107.c") {
            return Err(format!(
                "semantic source must be envydis/gm107.c, got {}",
                relative.display()
            ));
        }
        let safe = format!("safe.directory={}", root.display());
        let output = |args: &[&str]| -> Result<String, String> {
            let output = Command::new("git")
                .arg("-c")
                .arg(&safe)
                .arg("-C")
                .arg(root)
                .args(args)
                .output()
                .map_err(|error| format!("run git in {}: {error}", root.display()))?;
            if !output.status.success() {
                return Err(format!(
                    "git provenance check failed: {}",
                    String::from_utf8_lossy(&output.stderr).trim()
                ));
            }
            String::from_utf8(output.stdout)
                .map(|value| value.trim().to_owned())
                .map_err(|_| "git provenance output is not UTF-8".into())
        };
        let head = output(&["rev-parse", "HEAD"])?;
        if head != ENVYTOOLS_COMMIT {
            return Err(format!(
                "envytools revision is {head}, expected {ENVYTOOLS_COMMIT}"
            ));
        }
        let source_commit = output(&["log", "-1", "--format=%H", "--", "envydis/gm107.c"])?;
        if source_commit != ENVYTOOLS_COMMIT {
            return Err(format!(
                "gm107.c comes from {source_commit}, expected {ENVYTOOLS_COMMIT}"
            ));
        }
        let remote = output(&["remote", "get-url", "origin"])?;
        if remote != ENVYTOOLS_REMOTE {
            return Err(format!(
                "envytools origin is {remote:?}, expected {ENVYTOOLS_REMOTE:?}"
            ));
        }
        output(&["ls-files", "--error-unmatch", "envydis/gm107.c", "COPYING"])?;
        let dirty = output(&[
            "status",
            "--porcelain=v1",
            "--",
            "envydis/gm107.c",
            "COPYING",
        ])?;
        if !dirty.is_empty() {
            return Err("pinned semantic source or license anchor has local changes".into());
        }
        let license = read(&root.join("COPYING"))?;
        if digest(&license) != ENVYTOOLS_LICENSE_SHA256 {
            return Err("envytools COPYING does not match the pinned MIT license anchor".into());
        }
        let bytes = read(&path)?;
        Ok(Self {
            sha256: digest(&bytes),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SourcePin {
    path: String,
    sha256: String,
}

impl SourcePin {
    fn from_manifest(input: &str) -> Result<Self, String> {
        let line = input
            .lines()
            .find(|line| line.starts_with(SOURCE_PREFIX))
            .ok_or("manifest has no source pin")?;
        let (path, sha256) = line[SOURCE_PREFIX.len()..]
            .split_once(" SHA256 ")
            .ok_or("malformed source pin")?;
        if path != "envytools/envydis/gm107.c" {
            return Err(format!(
                "{path:?} is not a permitted semantic SourcePin; methodology references do not establish instruction semantics"
            ));
        }
        if sha256.len() != 64 || !sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err("source pin SHA256 must contain 64 hexadecimal digits".into());
        }
        Ok(Self {
            path: path.to_owned(),
            sha256: sha256.to_ascii_lowercase(),
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Probe {
    row: u32,
    field: String,
    citation: String,
    word: String,
    evidence: ProbeEvidence,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RawProbe {
    row: u32,
    field: String,
    citation: String,
    word: String,
    raw: u64,
    reservation: Reservation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Reservation {
    Available,
    SourceReserved,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RowShape {
    mnemonic: String,
    value: u64,
    mask: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Conflict {
    claim: String,
    claimed: String,
    opcode: String,
    source_row: u32,
    actual: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum ProbeEvidence {
    NeedsEvidence,
    SourceReserved,
    Decoded(String),
}

impl Ledger {
    pub fn verify_plan(&self, actual: &str) -> ReplayVerification {
        if actual == self.render_plan() {
            ReplayVerification::Verified
        } else {
            ReplayVerification::Stale
        }
    }

    pub fn render_plan(&self) -> String {
        match self {
            Self::Stale(stale) => format!(
                "# {FORMAT}\nstate\tStale\nreason\tsource pin drift\nrows_source\t{}\t{}\nsamples_source\t{}\t{}\n",
                stale.rows.path,
                stale.rows.sha256,
                stale.samples.path,
                stale.samples.sha256
            ),
            Self::SourceLocked(ledger) => {
                let mut body = String::new();
                for probe in ledger
                    .probes
                    .iter()
                    .filter(|probe| matches!(probe.evidence, ProbeEvidence::NeedsEvidence))
                {
                    writeln!(
                        body,
                        "{}\t{}\t{}\t{}",
                        probe.row, probe.field, probe.word, probe.citation
                    )
                    .unwrap();
                }
                let mut conflicts = String::new();
                for conflict in &ledger.conflicts {
                    writeln!(conflicts, "conflict\t{}\t{}\t{}\t{}\t{}", conflict.claim, conflict.claimed, conflict.opcode, conflict.source_row, conflict.actual).unwrap();
                }
                let plan_sha256 = digest([conflicts.as_bytes(), body.as_bytes()].concat());
                let state = if ledger.conflicts.is_empty() { "NeedsEvidence" } else { "Conflict" };
                format!(
                    "# {FORMAT}\nstate\t{state}\nsource\t{}\t{}\nsource_commit\t{}\nsource_remote\t{}\nsource_license\t{}\nlicense_anchor\tCOPYING\t{}\nevidence_sha256\t{}\nplan_sha256\t{}\nconflict_count\t{}\nprobe_count\t{}\n# conflict\tclaim\tclaimed\topcode\tsource_row\tactual\n{}# row\tfield\tword\tcitation\n{}",
                    ledger.source.path,
                    ledger.source.sha256,
                    ENVYTOOLS_COMMIT,
                    ENVYTOOLS_REMOTE,
                    ENVYTOOLS_LICENSE,
                    ENVYTOOLS_LICENSE_SHA256,
                    ledger.evidence_sha256,
                    plan_sha256,
                    ledger.conflicts.len(),
                    body.lines().count(),
                    conflicts,
                    body
                )
            }
        }
    }

    pub fn render_classification(&self) -> String {
        match self {
            Self::Stale(_) => self.render_plan(),
            Self::SourceLocked(ledger) => {
                let mut decoded = 0;
                let mut needs_evidence = 0;
                let mut source_reserved = 0;
                for probe in &ledger.probes {
                    match &probe.evidence {
                        ProbeEvidence::Decoded(_) => decoded += 1,
                        ProbeEvidence::NeedsEvidence => needs_evidence += 1,
                        ProbeEvidence::SourceReserved => source_reserved += 1,
                    }
                }
                format!(
                    "state\t{}\nsource\t{}\t{}\nsource_commit\t{}\nsource_remote\t{}\nsource_license\t{}\nlicense_anchor\tCOPYING\t{}\nevidence_sha256\t{}\ndecoded\t{decoded}\nneeds_evidence\t{needs_evidence}\nsource_reserved\t{source_reserved}\nconflicts\t{}\n",
                    if ledger.conflicts.is_empty() { "NeedsEvidence" } else { "Conflict" },
                    ledger.source.path, ledger.source.sha256, ENVYTOOLS_COMMIT, ENVYTOOLS_REMOTE, ENVYTOOLS_LICENSE, ENVYTOOLS_LICENSE_SHA256, ledger.evidence_sha256, ledger.conflicts.len()
                )
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ReplayVerification {
    Verified,
    Stale,
}

fn parse_rows(input: &str) -> Result<BTreeMap<u32, RowShape>, String> {
    records(input)
        .map(|fields| {
            if fields.len() != 7 {
                return Err("decoder row must have seven tab-separated fields".into());
            }
            let row = fields[0]
                .parse()
                .map_err(|_| "invalid decoder row number")?;
            Ok((
                row,
                RowShape {
                    mnemonic: fields[2].to_owned(),
                    value: u64::from_str_radix(fields[3], 16).map_err(|_| "invalid row value")?,
                    mask: u64::from_str_radix(fields[4], 16).map_err(|_| "invalid row mask")?,
                },
            ))
        })
        .collect()
}

fn parse_samples(input: &str, rows: &BTreeMap<u32, RowShape>) -> Result<Vec<RawProbe>, String> {
    records(input)
        .map(|fields| {
            if fields.len() != 5 {
                return Err("decoder sample must have five tab-separated fields".into());
            }
            let row = fields[0]
                .parse()
                .map_err(|_| format!("invalid source row {:?}", fields[0]))?;
            let raw = u64::from_str_radix(fields[3], 16)
                .map_err(|_| format!("invalid probe word {:?}", fields[3]))?;
            let shape = rows
                .get(&row)
                .ok_or_else(|| format!("sample refers to absent row {row}"))?;
            if raw & shape.mask != shape.value {
                return Err(format!("sample {raw:016x} escapes row {row} mask"));
            }
            Ok(RawProbe {
                row,
                field: fields[1].to_owned(),
                citation: fields[2].to_owned(),
                word: fields[3].to_owned(),
                reservation: match fields[4] {
                    "0" => Reservation::Available,
                    "1" => Reservation::SourceReserved,
                    value => return Err(format!("invalid reserved marker {value:?}")),
                },
                raw,
            })
        })
        .collect()
}

fn parse_claims(input: &str, rows: &BTreeMap<u32, RowShape>) -> Result<Vec<Conflict>, String> {
    let mut conflicts = Vec::new();
    for fields in records(input) {
        if fields.len() != 3 {
            return Err("discovery claim must have three tab-separated fields".into());
        }
        let opcode = u16::from_str_radix(fields[2], 16).map_err(|_| "invalid claim opcode")?;
        let word = u64::from(opcode) << 48;
        let (source_row, actual) = rows
            .iter()
            .find(|(_, shape)| word & shape.mask == shape.value)
            .ok_or_else(|| format!("claim opcode {} has no decoder row", fields[2]))?;
        if !actual.mnemonic.eq_ignore_ascii_case(fields[1]) {
            conflicts.push(Conflict {
                claim: fields[0].to_owned(),
                claimed: fields[1].to_owned(),
                opcode: fields[2].to_owned(),
                source_row: *source_row,
                actual: actual.mnemonic.clone(),
            });
        }
    }
    conflicts.sort_by(|a, b| (&a.claim, &a.opcode).cmp(&(&b.claim, &b.opcode)));
    Ok(conflicts)
}

fn operation_name(kind: &InstructionKind) -> String {
    match kind {
        InstructionKind::OutsideBranch { .. } => "Branch".to_owned(),
        InstructionKind::OutsideSchedule { .. } => "Schedule".to_owned(),
        InstructionKind::OutsidePushTarget { .. } => "PushTarget".to_owned(),
        InstructionKind::Unknown { .. } => "Unknown".to_owned(),
        kind => format!("{kind:?}")
            .split([' ', '{'])
            .next()
            .unwrap()
            .to_owned(),
    }
}

fn records(input: &str) -> impl Iterator<Item = Vec<&str>> {
    input
        .lines()
        .filter(|line| !line.starts_with('#') && !line.is_empty())
        .map(|line| line.split('\t').collect())
}

fn read(path: &Path) -> Result<Vec<u8>, String> {
    fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))
}

fn text(label: &str, bytes: Vec<u8>) -> Result<String, String> {
    String::from_utf8(bytes).map_err(|_| format!("{label} is not UTF-8"))
}

fn digest(bytes: impl AsRef<[u8]>) -> String {
    format!("{:x}", Sha256::digest(bytes.as_ref()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            let suffix = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "shadowbox-discovery-test-{}-{suffix}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            fs::remove_dir_all(&self.0).unwrap();
        }
    }

    fn inputs(samples: &str, results: &str) -> Inputs {
        let source = b"permitted semantic source".to_vec();
        let hash = digest(&source);
        let header = format!("# Source: envytools/envydis/gm107.c SHA256 {hash}\n");
        Inputs {
            rows: format!("{header}1\troot\tunknown\t0000000000000000\t0000000000000000\t0000000000000000\tsource:1\n2\troot\tunknown\t0000000000000000\t0000000000000000\t0000000000000000\tsource:2\n").into_bytes(),
            samples: format!("{header}{samples}").into_bytes(),
            results: results.as_bytes().to_vec(),
            semantic_source: SemanticSource { sha256: hash },
            claims: Vec::new(),
        }
    }

    #[test]
    fn plan_is_identical_across_sample_order() {
        let a = inputs(
            "2\tb\tsource:2\t0002\t0\n1\ta\tsource:1\t0001\t0\n",
            "Unknown\nUnknown\n",
        )
        .classify()
        .unwrap()
        .render_plan();
        let b = inputs(
            "1\ta\tsource:1\t0001\t0\n2\tb\tsource:2\t0002\t0\n",
            "Unknown\nUnknown\n",
        )
        .classify()
        .unwrap()
        .render_plan();
        assert_eq!(a, b);
    }

    #[test]
    fn source_drift_constructs_stale_state() {
        let source = b"permitted semantic source".to_vec();
        let hash = digest(&source);
        let header = format!("# Source: envytools/envydis/gm107.c SHA256 {hash}\n");
        let drift = format!(
            "# Source: envytools/envydis/gm107.c SHA256 {}\n",
            "0".repeat(64)
        );
        let value = Inputs {
            rows: header.into_bytes(),
            samples: format!("{drift}1\ta\tsource:1\t0001\t0\n").into_bytes(),
            results: b"Unknown\n".to_vec(),
            semantic_source: SemanticSource { sha256: hash },
            claims: Vec::new(),
        };
        assert!(matches!(value.classify().unwrap(), Ledger::Stale(_)));
    }

    #[test]
    fn methodology_reference_cannot_be_a_semantic_source_pin() {
        let hash = digest(b"methodology");
        let value = Inputs {
            rows: format!("# Source: AnyPS5/methodology SHA256 {hash}\n").into_bytes(),
            samples: format!("# Source: AnyPS5/methodology SHA256 {hash}\n").into_bytes(),
            results: Vec::new(),
            semantic_source: SemanticSource { sha256: hash },
            claims: Vec::new(),
        };
        assert!(value.classify().unwrap_err().contains("methodology"));
    }

    #[test]
    fn plan_has_no_worker_or_mutable_campaign_file() {
        let plan = inputs("1\ta\tsource:1\t0001\t0\n", "Unknown\n")
            .classify()
            .unwrap()
            .render_plan();
        assert!(!plan.contains("worker"));
        assert!(!plan.contains("status_file"));
        assert!(!plan.contains("lease"));
    }

    #[test]
    fn copied_header_does_not_lock_mutated_source_bytes() {
        let mut value = inputs("1\ta\tsource:1\t0001\t0\n", "Unknown\n");
        value.semantic_source.sha256 = digest(b"mutated semantic source");
        assert!(matches!(value.classify().unwrap(), Ledger::Stale(_)));
    }

    #[test]
    fn copied_source_outside_pinned_repository_is_rejected() {
        let dir = TestDir::new();
        let source = dir.0.join("gm107.c");
        // Repository identity is checked before content, so even a byte-for-byte
        // copy of the pinned file cannot inherit its commit or license anchor.
        fs::write(&source, b"copied source").unwrap();
        assert!(SemanticSource::read(&source)
            .unwrap_err()
            .contains("outside the pinned envytools Git repository"));
    }

    #[test]
    fn unknown_sample_row_is_rejected() {
        let error = inputs("3\ta\tsource:3\t0001\t0\n", "Unknown\n")
            .classify()
            .unwrap_err();
        assert!(error.contains("absent row 3"));
    }

    #[test]
    fn sample_escaping_row_mask_is_rejected() {
        let mut value = inputs("1\ta\tsource:1\t0001\t0\n", "Unknown\n");
        let header = String::from_utf8(
            value.rows[..value.rows.iter().position(|byte| *byte == b'\n').unwrap() + 1].to_vec(),
        )
        .unwrap();
        value.rows = format!(
            "{header}1\troot\tunknown\t0000000000000000\tffffffffffffffff\t0000000000000000\tsource:1\n"
        )
        .into_bytes();
        assert!(value.classify().unwrap_err().contains("escapes row 1 mask"));
    }

    #[test]
    fn tampered_plan_is_stale() {
        let ledger = inputs("1\ta\tsource:1\t0001\t0\n", "Unknown\n")
            .classify()
            .unwrap();
        let mut plan = ledger.render_plan();
        plan.push_str("tampered\n");
        assert_eq!(ledger.verify_plan(&plan), ReplayVerification::Stale);
    }

    #[test]
    fn tracked_corpus_count_matches_the_translation_manifest_scope() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
        let output = Command::new("git")
            .arg("-C")
            .arg(&root)
            .args(["ls-files", "corpus/*.bin"])
            .output()
            .unwrap();
        assert!(output.status.success());
        let tracked = String::from_utf8(output.stdout).unwrap();
        assert_eq!(tracked.lines().count(), 33);
        // The translation gate covers every tracked fixture (518c9454).
        let translates = fs::read_to_string(root.join("corpus/translates.txt")).unwrap();
        assert_eq!(
            translates.lines().filter(|line| !line.is_empty()).count(),
            tracked.lines().count()
        );
    }

    #[test]
    fn mislabeled_imnmx_claim_is_a_conflict() {
        let mut value = inputs("1\ta\tsource:1\t5c28000000000000\t0\n", "Shr\n");
        let header = String::from_utf8(
            value.rows[..value.rows.iter().position(|b| *b == b'\n').unwrap() + 1].to_vec(),
        )
        .unwrap();
        value.rows = format!("{header}1\troot\tshr\t5c28000000000000\tfff8000000000000\t5c28000000000000\tsource:1\n").into_bytes();
        value.claims = b"claim:1\tIMNMX\t5c28\n".to_vec();
        let plan = value.classify().unwrap().render_plan();
        assert!(plan.contains("state\tConflict"));
        assert!(plan.contains("conflict\tclaim:1\tIMNMX\t5c28\t1\tshr"));
    }
}
