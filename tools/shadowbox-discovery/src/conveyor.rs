use crate::{
    hybrid::{
        build_bundles, build_packets, build_plan, render_words, validate_preflight,
        validate_qualification_space, HybridBundle, HybridBundleSet, HybridPackets,
    },
    sudoku::SourceLock,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

const FORMAT: &str = "shadowbox-hybrid-conveyor-v2";
const MIN_QUALIFICATION_WORDS: usize = 1_000;
const MAX_QUALIFICATION_WORDS: usize = 4_000;
const STANDING_ORDERS: &str = "Stay within the exact bundle. Preserve typed decode-to-IR-to-emitter state. Do not merge or claim semantics without the required evidence and independent review.";

#[derive(Clone, Debug)]
pub struct CampaignSpec {
    pub source_root: PathBuf,
    pub expected_head: String,
    pub model: PathBuf,
    pub input: PathBuf,
    pub conflicts: PathBuf,
    pub state_dir: PathBuf,
    pub answers: Option<PathBuf>,
    pub max_bits: u8,
    pub max_words: u64,
    pub max_domain_words: u64,
    pub max_packets: usize,
    pub max_regions: usize,
    pub max_bundle_words: usize,
    pub min_free_bytes: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum CampaignPhase {
    OraclePending,
    ReadyForSemanticWork,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct CampaignReceipt {
    pub format: String,
    pub receipt_digest: String,
    pub phase: CampaignPhase,
    pub source_head: String,
    pub plan_digest: String,
    pub words_digest: String,
    pub packets_digest: Option<String>,
    pub bundles_digest: Option<String>,
    pub units_digest: Option<String>,
    pub unit_count: usize,
    pub deferred_bundle_count: usize,
    pub deferred_word_count: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct SemanticUnit {
    format: String,
    id: String,
    source_head: String,
    packet_receipt_digest: String,
    bundles_digest: String,
    bundle: HybridBundle,
    requirements: UnitRequirements,
    standing_orders: String,
    standing_orders_digest: String,
    brief: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
struct UnitRequirements {
    permitted_semantic_source: bool,
    cpu_oracle: bool,
    spirv_raw: bool,
    spirv_optimized: bool,
    radv_exact: bool,
    independent_review: bool,
    signed_clean_commit: bool,
}

pub fn advance_campaign(spec: CampaignSpec) -> Result<CampaignReceipt, String> {
    fs::create_dir_all(&spec.state_dir).map_err(|e| format!("create campaign state: {e}"))?;
    let _lock = CampaignLock::acquire(&spec.state_dir)?;
    validate_qualification_space(available_bytes(&spec.state_dir)?, spec.min_free_bytes)?;
    if !(MIN_QUALIFICATION_WORDS..=MAX_QUALIFICATION_WORDS).contains(&spec.max_bundle_words) {
        return Err(format!(
            "hybrid qualification units require max-bundle-words in {MIN_QUALIFICATION_WORDS}..={MAX_QUALIFICATION_WORDS}"
        ));
    }
    let lock = SourceLock::acquire(&spec.source_root, &spec.expected_head)?;
    fs::create_dir_all(spec.state_dir.join("units"))
        .map_err(|e| format!("create campaign state: {e}"))?;
    let plan = build_plan(
        lock,
        &spec.model,
        &spec.input,
        &spec.conflicts,
        spec.max_bits,
        spec.max_words,
        spec.max_domain_words,
    )?;
    publish_json(&spec.state_dir.join("plan.json"), &plan)?;
    let words = render_words(&plan)?;
    publish(&spec.state_dir.join("words.tsv"), words.as_bytes())?;
    let mut receipt = CampaignReceipt {
        format: FORMAT.into(),
        receipt_digest: String::new(),
        phase: CampaignPhase::OraclePending,
        source_head: plan.source_head.clone(),
        plan_digest: plan.plan_digest.clone(),
        words_digest: plan.words_digest.clone(),
        packets_digest: None,
        bundles_digest: None,
        units_digest: None,
        unit_count: 0,
        deferred_bundle_count: 0,
        deferred_word_count: 0,
    };
    if let Some(answers) = &spec.answers {
        let packets = build_packets(&plan, answers, spec.max_packets)?;
        publish_json(&spec.state_dir.join("packets.json"), &packets)?;
        let bundles = build_bundles(&packets, spec.max_regions, spec.max_bundle_words)?;
        validate_preflight(&packets, &bundles, &spec.expected_head)?;
        publish_json(&spec.state_dir.join("bundles.json"), &bundles)?;
        let publication = publish_units(&spec.state_dir, &packets, &bundles)?;
        receipt.phase = CampaignPhase::ReadyForSemanticWork;
        receipt.packets_digest = Some(packets.packets_digest);
        receipt.bundles_digest = Some(bundles.bundles_digest);
        receipt.units_digest = Some(publication.digest);
        receipt.unit_count = publication.count;
        receipt.deferred_bundle_count = publication.deferred_count;
        receipt.deferred_word_count = publication.deferred_words;
    }
    SourceLock::acquire(&spec.source_root, &spec.expected_head)?;
    receipt.receipt_digest = receipt_digest(&receipt)?;
    let phase = match receipt.phase {
        CampaignPhase::OraclePending => "oracle-pending",
        CampaignPhase::ReadyForSemanticWork => "ready-for-semantic-work",
    };
    publish_json(
        &spec
            .state_dir
            .join("manifests")
            .join(format!("{phase}.json")),
        &receipt,
    )?;
    Ok(receipt)
}

#[derive(Debug)]
struct CampaignLock {
    _file: fs::File,
}

impl CampaignLock {
    fn acquire(state: &Path) -> Result<Self, String> {
        let path = state.join(".campaign.lock");
        let file = OpenOptions::new()
            .read(true)
            .append(true)
            .create(true)
            .open(&path)
            .map_err(|error| format!("open campaign lock {}: {error}", path.display()))?;
        if unsafe {
            libc::flock(
                std::os::fd::AsRawFd::as_raw_fd(&file),
                libc::LOCK_EX | libc::LOCK_NB,
            )
        } != 0
        {
            return Err(format!(
                "campaign state is already locked at {}",
                path.display()
            ));
        }
        Ok(Self { _file: file })
    }
}

fn publish_units(
    state: &Path,
    packets: &HybridPackets,
    bundles: &HybridBundleSet,
) -> Result<UnitPublication, String> {
    let mut covered = BTreeSet::new();
    let mut published = Vec::new();
    for bundle in &bundles.bundles {
        if !is_qualification_bundle(bundle) {
            continue;
        }
        for region in &bundle.region_ids {
            if !covered.insert(region.clone()) {
                return Err(format!("bundle region {region} is multiply owned"));
            }
        }
        let unit = SemanticUnit {
            format: FORMAT.into(),
            id: bundle.id.clone(),
            source_head: packets.source_head.clone(),
            packet_receipt_digest: packets.receipt_digest.clone(),
            bundles_digest: bundles.bundles_digest.clone(),
            bundle: bundle.clone(),
            requirements: UnitRequirements {
                permitted_semantic_source: true,
                cpu_oracle: true,
                spirv_raw: true,
                spirv_optimized: true,
                radv_exact: true,
                independent_review: true,
                signed_clean_commit: true,
            },
            standing_orders: STANDING_ORDERS.into(),
            standing_orders_digest: digest(STANDING_ORDERS.as_bytes()),
            brief: format!(
                "Implement only bundle {} (row {}, family {}, mnemonic {}, semantic mask {}, semantic value {}) covering packet IDs [{}]. The packet receipt, bundle digest, typed requirements, and standing orders in this unit define the review boundary. Report evidence receipts and remaining risks.",
                bundle.id,
                bundle.row,
                bundle.family,
                bundle.mnemonic,
                bundle.semantic_mask,
                bundle.semantic_value,
                bundle.region_ids.join(",")
            ),
        };
        publish_json(
            &state.join("units").join(format!("{}.json", bundle.id)),
            &unit,
        )?;
        published.extend_from_slice(
            &fs::read(state.join("units").join(format!("{}.json", bundle.id)))
                .map_err(|e| format!("read published unit: {e}"))?,
        );
    }
    let expected_unit_files = expected_unit_files(bundles);
    for entry in
        fs::read_dir(state.join("units")).map_err(|e| format!("read unit directory: {e}"))?
    {
        let path = entry.map_err(|e| format!("read unit entry: {e}"))?.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            return Err("unit file name is not UTF-8".into());
        };
        if !expected_unit_files.contains(name) {
            return Err(format!(
                "stale semantic unit {} is not in this bundle manifest",
                path.display()
            ));
        }
    }
    let deferred = bundles
        .bundles
        .iter()
        .filter(|bundle| !is_qualification_bundle(bundle))
        .collect::<Vec<_>>();
    Ok(UnitPublication {
        digest: digest(&published),
        count: bundles.bundles.len() - deferred.len(),
        deferred_count: deferred.len(),
        deferred_words: deferred.iter().map(|bundle| bundle.word_count).sum(),
    })
}

#[derive(Debug)]
struct UnitPublication {
    digest: String,
    count: usize,
    deferred_count: usize,
    deferred_words: usize,
}

fn is_qualification_bundle(bundle: &HybridBundle) -> bool {
    (MIN_QUALIFICATION_WORDS..=MAX_QUALIFICATION_WORDS).contains(&bundle.word_count)
}

fn expected_unit_files(bundles: &HybridBundleSet) -> BTreeSet<String> {
    bundles
        .bundles
        .iter()
        .filter(|bundle| is_qualification_bundle(bundle))
        .map(|bundle| format!("{}.json", bundle.id))
        .collect()
}

fn publish_json(path: &Path, value: &impl Serialize) -> Result<(), String> {
    let mut bytes = serde_json::to_vec_pretty(value)
        .map_err(|e| format!("serialize {}: {e}", path.display()))?;
    bytes.push(b'\n');
    publish(path, &bytes)
}

fn publish(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| format!("{} has no parent", path.display()))?;
    fs::create_dir_all(parent).map_err(|e| format!("create {}: {e}", parent.display()))?;
    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
    let temporary = parent.join(format!(
        ".{}.{}.{}.tmp",
        path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("artifact"),
        std::process::id(),
        NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
    ));
    match OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
    {
        Ok(mut file) => {
            file.write_all(bytes)
                .and_then(|_| file.sync_all())
                .map_err(|e| format!("publish {}: {e}", temporary.display()))?;
            drop(file);
            match fs::hard_link(&temporary, path) {
                Ok(()) => {
                    fs::remove_file(&temporary)
                        .map_err(|e| format!("remove {}: {e}", temporary.display()))?;
                    fs::File::open(parent)
                        .and_then(|dir| dir.sync_all())
                        .map_err(|e| format!("sync {}: {e}", parent.display()))?;
                    Ok(())
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    fs::remove_file(&temporary)
                        .map_err(|e| format!("remove {}: {e}", temporary.display()))?;
                    identical(path, bytes)
                }
                Err(error) => Err(format!("atomically publish {}: {error}", path.display())),
            }
        }
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => identical(path, bytes),
        Err(error) => Err(format!("create temporary {}: {error}", temporary.display())),
    }
}

fn identical(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let existing = fs::read(path).map_err(|e| format!("read existing {}: {e}", path.display()))?;
    if existing == bytes {
        Ok(())
    } else {
        Err(format!(
            "immutable artifact {} already has different bytes",
            path.display()
        ))
    }
}

fn receipt_digest(receipt: &CampaignReceipt) -> Result<String, String> {
    let mut canonical = receipt.clone();
    canonical.receipt_digest.clear();
    Ok(digest(
        &serde_json::to_vec(&canonical).map_err(|e| e.to_string())?,
    ))
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn available_bytes(path: &Path) -> Result<u64, String> {
    use std::{ffi::CString, mem::MaybeUninit, os::unix::ffi::OsStrExt};
    let path = path
        .canonicalize()
        .map_err(|e| format!("canonicalize {}: {e}", path.display()))?;
    let c = CString::new(path.as_os_str().as_bytes()).map_err(|_| "state path contains NUL")?;
    let mut stats = MaybeUninit::<libc::statvfs>::uninit();
    if unsafe { libc::statvfs(c.as_ptr(), stats.as_mut_ptr()) } != 0 {
        return Err(format!(
            "statvfs {}: {}",
            path.display(),
            std::io::Error::last_os_error()
        ));
    }
    let stats = unsafe { stats.assume_init() };
    Ok(stats.f_bavail.saturating_mul(stats.f_frsize))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(name: &str) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("shadowbox-conveyor-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn immutable_publish_replays_or_rejects_drift() {
        let dir = scratch("publish");
        let artifact = dir.join("artifact");
        publish(&artifact, b"same").unwrap();
        publish(&artifact, b"same").unwrap();
        assert!(publish(&artifact, b"different")
            .unwrap_err()
            .contains("different bytes"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn campaign_lock_rejects_a_concurrent_writer() {
        let dir = scratch("lock");
        let _first = CampaignLock::acquire(&dir).unwrap();
        assert!(CampaignLock::acquire(&dir)
            .unwrap_err()
            .contains("already locked"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn qualification_units_enforce_the_production_floor() {
        let bundle = |word_count| HybridBundle {
            id: format!("B{word_count}"),
            row: 1,
            family: "X".into(),
            mnemonic: "X".into(),
            semantic_mask: "0".into(),
            semantic_value: "0".into(),
            region_count: 1,
            word_count,
            region_ids: vec!["P1".into()],
        };
        assert!(!is_qualification_bundle(&bundle(999)));
        assert!(is_qualification_bundle(&bundle(1_000)));
        assert!(is_qualification_bundle(&bundle(4_000)));
        assert!(!is_qualification_bundle(&bundle(4_001)));

        let bundles = HybridBundleSet {
            format: "test".into(),
            source_packet_receipt_digest: "receipt".into(),
            max_regions: 2,
            max_words: 4_000,
            region_count: 2,
            word_count: 1_999,
            bundles_digest: "digest".into(),
            bundles: vec![bundle(999), bundle(1_000)],
        };
        assert_eq!(
            expected_unit_files(&bundles),
            BTreeSet::from(["B1000.json".into()])
        );
    }

    #[test]
    fn deferred_v1_tail_unit_is_rejected_as_stale() {
        let dir = scratch("stale-tail");
        fs::create_dir_all(dir.join("units")).unwrap();
        fs::write(dir.join("units/B999.json"), b"legacy tail").unwrap();
        let bundle = |id: &str, word_count| HybridBundle {
            id: id.into(),
            row: 1,
            family: "X".into(),
            mnemonic: "X".into(),
            semantic_mask: "0".into(),
            semantic_value: "0".into(),
            region_count: 1,
            word_count,
            region_ids: vec![format!("P{id}")],
        };
        let bundles = HybridBundleSet {
            format: "test".into(),
            source_packet_receipt_digest: "receipt".into(),
            max_regions: 2,
            max_words: 4_000,
            region_count: 2,
            word_count: 1_999,
            bundles_digest: "bundles".into(),
            bundles: vec![bundle("B999", 999), bundle("B1000", 1_000)],
        };
        let packets = HybridPackets {
            format: "test".into(),
            receipt_digest: "receipt".into(),
            source_head: "head".into(),
            source_sha256: "source".into(),
            plan_digest: "plan".into(),
            plan_words_digest: "words".into(),
            answers_raw_digest: "raw".into(),
            answers_digest: "answers".into(),
            packets_digest: "packets".into(),
            planned_unknown_words: 1_999,
            oracle_accepted_words: 1_999,
            oracle_rejected_words: 0,
            packet_words: 1_999,
            family_drifted_words: 0,
            cubes_family_drifted: Vec::new(),
            packets: Vec::new(),
        };
        assert!(publish_units(&dir, &packets, &bundles)
            .unwrap_err()
            .contains("stale semantic unit"));
        fs::remove_dir_all(dir).unwrap();
    }
}
