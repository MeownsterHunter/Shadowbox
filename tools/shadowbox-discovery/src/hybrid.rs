use crate::sudoku::{candidates, model_rows, ModelField, SourceLock};
use serde::{Deserialize, Serialize};
use shadowbox::{decoder::decode_word, ir::InstructionKind, TRANSLATOR_VERSION};
use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};

const PLAN_FORMAT: &str = "shadowbox-hybrid-plan-v3";
const PACKET_FORMAT: &str = "shadowbox-hybrid-packets-v3";
const BUNDLE_FORMAT: &str = "shadowbox-hybrid-bundles-v4";
const CONFLICTS_HEADER: &str = "#row\tform\tbit\tkind\tcounts\texample_base";

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct HybridPlan {
    pub format: String,
    pub plan_digest: String,
    pub source_head: String,
    pub source_sha256: String,
    pub model_digest: String,
    pub input_digest: String,
    pub conflicts_digest: String,
    pub max_bits: u8,
    pub max_words: u64,
    pub max_domain_words: u64,
    pub input_rows_excluded: usize,
    pub decoded_candidates_subtracted: usize,
    pub groups_considered: usize,
    pub groups_conflicted: usize,
    pub groups_family_ambiguous: usize,
    pub groups_empty: usize,
    pub groups_too_wide: usize,
    pub groups_budget_deferred: usize,
    pub groups_domain_deferred: usize,
    pub overlap_words_subtracted: usize,
    pub emitted_words: usize,
    pub cubes: Vec<HybridCube>,
    pub words_digest: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct HybridCube {
    pub id: String,
    pub row: u32,
    pub form: String,
    pub base: String,
    pub semantic_mask: String,
    pub semantic_bits: u8,
    pub family: String,
    pub domain_words: u64,
    pub decoded_subtracted: usize,
    pub unknown_words: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct HybridPackets {
    pub format: String,
    pub receipt_digest: String,
    pub source_head: String,
    pub source_sha256: String,
    pub plan_digest: String,
    pub plan_words_digest: String,
    pub answers_raw_digest: String,
    pub answers_digest: String,
    pub packets_digest: String,
    pub planned_unknown_words: usize,
    pub oracle_accepted_words: usize,
    pub oracle_rejected_words: usize,
    pub packet_words: usize,
    pub family_drifted_words: usize,
    pub cubes_family_drifted: Vec<String>,
    pub packets: Vec<HybridPacket>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct HybridBundleSet {
    pub format: String,
    pub source_packet_receipt_digest: String,
    pub max_regions: usize,
    pub max_words: usize,
    pub region_count: usize,
    pub word_count: usize,
    pub bundles_digest: String,
    pub bundles: Vec<HybridBundle>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct HybridBundle {
    pub id: String,
    pub row: u32,
    pub family: String,
    pub mnemonic: String,
    pub semantic_mask: String,
    pub semantic_value: String,
    pub region_count: usize,
    pub word_count: usize,
    pub region_ids: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct HybridPacket {
    pub id: String,
    pub cube: String,
    pub row: u32,
    pub family: String,
    pub mnemonic: String,
    pub semantic_mask: String,
    pub semantic_value: String,
    pub word_count: usize,
    pub words: Vec<String>,
}

#[derive(Default)]
struct Skips {
    considered: usize,
    conflicted: usize,
    family_ambiguous: usize,
    empty: usize,
    too_wide: usize,
    budget_deferred: usize,
    domain_deferred: usize,
}

#[derive(Clone)]
struct CubeDraft {
    row: u32,
    form: u64,
    base: u64,
    mask: u64,
    family: String,
    domain_words: u64,
    decoded_subtracted: usize,
    unknown_words: Vec<u64>,
}

pub fn build_plan(
    lock: SourceLock,
    model: &Path,
    input: &Path,
    conflicts: &Path,
    max_bits: u8,
    max_words: u64,
    max_domain_words: u64,
) -> Result<HybridPlan, String> {
    if max_bits == 0 || max_bits > 20 {
        return Err("hybrid max-bits must be in 1..=20".into());
    }
    if max_words == 0 || max_domain_words == 0 {
        return Err("hybrid word limits must be nonzero".into());
    }

    let model_bytes = read(model)?;
    let input_bytes = read(input)?;
    let conflict_bytes = read(conflicts)?;
    let models = model_rows(&model_bytes)?;
    let validated = candidates(&input_bytes, &models)?;
    let conflicted = parse_conflicts(&conflict_bytes)?;
    let source_opcode_masks = match lock.root() {
        Some(root) => source_opcode_masks(&read(
            &root.join("crates/shadowbox/tests/data/decoder-rows.tsv"),
        )?)?,
        None => BTreeMap::new(),
    };

    let mut decoded_candidates_subtracted = 0;
    let mut selected = BTreeMap::<(u32, u64, u64), BTreeSet<String>>::new();
    for candidate in validated.candidates {
        let words = [candidate.word as u32, (candidate.word >> 32) as u32];
        if !matches!(decode_word(words, 0).kind, InstructionKind::Unknown { .. }) {
            decoded_candidates_subtracted += 1;
            continue;
        }
        let model = models
            .get(&candidate.row)
            .ok_or_else(|| format!("row {} disappeared from parsed model", candidate.row))?;
        let form = candidate.base & model.fmask;
        let group_name = if model.fmask == 0 {
            "-".to_owned()
        } else {
            format!("{form:x}")
        };
        let group = model.groups.get(&group_name).ok_or_else(|| {
            format!(
                "row {} base {:016x} has no model group {group_name}",
                candidate.row, candidate.base
            )
        })?;
        let mask = semantic_mask(&group.fields)?
            | source_opcode_masks
                .get(&candidate.row)
                .copied()
                .unwrap_or(0);
        let base = candidate.base & !mask;
        selected
            .entry((candidate.row, form, base))
            .or_default()
            .insert(family(&candidate.predicted)?);
    }

    let mut skips = Skips::default();
    let mut drafts = Vec::new();
    let mut domain_words_used = 0_u64;
    for ((row, form, base), families) in selected {
        skips.considered += 1;
        if conflicted.contains(&(row, form)) {
            skips.conflicted += 1;
            continue;
        }
        if families.len() != 1 {
            skips.family_ambiguous += 1;
            continue;
        }
        let model = &models[&row];
        let group_name = if model.fmask == 0 {
            "-".to_owned()
        } else {
            format!("{form:x}")
        };
        let mask = semantic_mask(&model.groups[&group_name].fields)?
            | source_opcode_masks.get(&row).copied().unwrap_or(0);
        let bits = mask.count_ones() as u8;
        if bits == 0 {
            skips.empty += 1;
            continue;
        }
        if bits > max_bits {
            skips.too_wide += 1;
            continue;
        }
        let positions = bit_positions(mask);
        let domain_words = 1_u64 << bits;
        if !reserve_domain(&mut domain_words_used, domain_words, max_domain_words) {
            skips.domain_deferred += 1;
            continue;
        }
        let mut unknown_words = Vec::new();
        let mut decoded_subtracted = 0;
        for value in 0..domain_words {
            let word = base | scatter(value, &positions);
            let words = [word as u32, (word >> 32) as u32];
            if matches!(decode_word(words, 0).kind, InstructionKind::Unknown { .. }) {
                unknown_words.push(word);
            } else {
                decoded_subtracted += 1;
            }
        }
        if unknown_words.is_empty() {
            skips.empty += 1;
            continue;
        }
        drafts.push(CubeDraft {
            row,
            form,
            base,
            mask,
            family: families.into_iter().next().unwrap(),
            domain_words,
            decoded_subtracted,
            unknown_words,
        });
    }

    drafts.sort_by(rank);
    let (accepted, overlap_words_subtracted) = select_drafts(drafts, max_words, &mut skips);

    let cubes = accepted
        .into_iter()
        .enumerate()
        .map(|(index, cube)| HybridCube {
            id: format!("H{:06}", index + 1),
            row: cube.row,
            form: format!("{:x}", cube.form),
            base: format!("{:016x}", cube.base),
            semantic_mask: format!("{:016x}", cube.mask),
            semantic_bits: cube.mask.count_ones() as u8,
            family: cube.family,
            domain_words: cube.domain_words,
            decoded_subtracted: cube.decoded_subtracted,
            unknown_words: cube
                .unknown_words
                .into_iter()
                .map(|word| format!("{word:016x}"))
                .collect(),
        })
        .collect::<Vec<_>>();
    let words = render_words_from_cubes(&cubes);
    lock.verify_unchanged()?;

    let mut plan = HybridPlan {
        format: PLAN_FORMAT.into(),
        plan_digest: String::new(),
        source_head: lock.head().into(),
        source_sha256: TRANSLATOR_VERSION.into(),
        model_digest: digest(&model_bytes),
        input_digest: digest(&input_bytes),
        conflicts_digest: digest(&conflict_bytes),
        max_bits,
        max_words,
        max_domain_words,
        input_rows_excluded: validated.excluded,
        decoded_candidates_subtracted,
        groups_considered: skips.considered,
        groups_conflicted: skips.conflicted,
        groups_family_ambiguous: skips.family_ambiguous,
        groups_empty: skips.empty,
        groups_too_wide: skips.too_wide,
        groups_budget_deferred: skips.budget_deferred,
        groups_domain_deferred: skips.domain_deferred,
        overlap_words_subtracted,
        emitted_words: cubes.iter().map(|cube| cube.unknown_words.len()).sum(),
        cubes,
        words_digest: digest(words.as_bytes()),
    };
    plan.plan_digest = plan_digest(&plan)?;
    Ok(plan)
}

fn reserve_domain(used: &mut u64, next: u64, cap: u64) -> bool {
    let Some(total) = used.checked_add(next) else {
        return false;
    };
    if total > cap {
        return false;
    }
    *used = total;
    true
}

// The source row's high-16 mask defines opcode aliases. Include every free
// opcode bit in the cube so a seed cannot qualify the middle of a source row
// while silently omitting either endpoint (for example PSET 0x508f).
fn source_opcode_masks(bytes: &[u8]) -> Result<BTreeMap<u32, u64>, String> {
    const OPCODE: u64 = 0xffff_0000_0000_0000;
    let text = std::str::from_utf8(bytes).map_err(|_| "decoder rows are not UTF-8")?;
    let mut masks = BTreeMap::new();
    for (index, line) in text.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 7 {
            return Err(format!(
                "decoder rows line {} must have seven tab-separated fields",
                index + 1
            ));
        }
        let row = fields[0]
            .parse::<u32>()
            .map_err(|_| format!("invalid decoder row on line {}", index + 1))?;
        let source_mask = u64::from_str_radix(fields[4], 16)
            .map_err(|_| format!("invalid decoder mask on line {}", index + 1))?;
        let alias_mask = !source_mask & OPCODE;
        if masks.insert(row, alias_mask).is_some() {
            return Err(format!("duplicate decoder row {row}"));
        }
    }
    Ok(masks)
}

fn select_drafts(
    drafts: Vec<CubeDraft>,
    max_words: u64,
    skips: &mut Skips,
) -> (Vec<CubeDraft>, usize) {
    let mut accepted = Vec::new();
    let mut owned = BTreeSet::new();
    let mut overlap_words_subtracted = 0;
    let mut used = 0_u64;
    for mut draft in drafts {
        let before = draft.unknown_words.len();
        draft.unknown_words.retain(|word| !owned.contains(word));
        overlap_words_subtracted += before - draft.unknown_words.len();
        if draft.unknown_words.is_empty() {
            skips.empty += 1;
            continue;
        }
        let words = draft.unknown_words.len() as u64;
        if used + words > max_words {
            skips.budget_deferred += 1;
            continue;
        }
        used += words;
        owned.extend(draft.unknown_words.iter().copied());
        accepted.push(draft);
    }
    (accepted, overlap_words_subtracted)
}

pub fn render_words(plan: &HybridPlan) -> Result<String, String> {
    validate_plan(plan)?;
    let words = render_words_from_cubes(&plan.cubes);
    Ok(words)
}

pub fn build_packets(
    plan: &HybridPlan,
    answers: &Path,
    max_packets: usize,
) -> Result<HybridPackets, String> {
    if max_packets == 0 {
        return Err("hybrid max-packets must be nonzero".into());
    }
    let planned_words = render_words(plan)?;
    let answer_bytes = read(answers)?;
    let answers = parse_answers(&answer_bytes)?;
    let expected = plan
        .cubes
        .iter()
        .flat_map(|cube| cube.unknown_words.iter())
        .map(|word| parse_hex(word, "planned word"))
        .collect::<Result<BTreeSet<_>, _>>()?;
    let actual = answers.keys().copied().collect::<BTreeSet<_>>();
    if actual != expected {
        let missing = expected.difference(&actual).count();
        let extra = actual.difference(&expected).count();
        return Err(format!(
            "hybrid answers do not exactly cover planned words: {missing} missing, {extra} extra"
        ));
    }

    let mut packets = Vec::new();
    let mut oracle_accepted_words = 0;
    let mut oracle_rejected_words = 0;
    let mut family_drifted_words = 0;
    let mut cubes_family_drifted = Vec::new();
    for cube in &plan.cubes {
        let base = parse_hex(&cube.base, "cube base")?;
        let mask = parse_hex(&cube.semantic_mask, "cube semantic mask")?;
        let positions = bit_positions(mask);
        let unknown = cube
            .unknown_words
            .iter()
            .map(|word| parse_hex(word, "cube unknown word"))
            .collect::<Result<BTreeSet<_>, _>>()?;
        let mut assignments = Vec::with_capacity(cube.domain_words as usize);
        let mut mnemonic_ids = BTreeMap::<String, usize>::new();
        let mut mnemonics = Vec::<String>::new();
        let mut family_drifted = false;
        let mut cube_accepted = 0;
        for value in 0..cube.domain_words {
            let word = base | scatter(value, &positions);
            let label = if !unknown.contains(&word) {
                Label::Existing
            } else {
                let text = &answers[&word];
                if rejected_oracle_label(text) {
                    oracle_rejected_words += 1;
                    Label::Rejected
                } else {
                    let mnemonic = mnemonic(text)?;
                    if family(&mnemonic)? != cube.family {
                        family_drifted = true;
                    }
                    let mnemonic_id = if let Some(id) = mnemonic_ids.get(&mnemonic) {
                        *id
                    } else {
                        let id = mnemonics.len();
                        mnemonics.push(mnemonic.clone());
                        mnemonic_ids.insert(mnemonic, id);
                        id
                    };
                    oracle_accepted_words += 1;
                    cube_accepted += 1;
                    Label::New(mnemonic_id)
                }
            };
            assignments.push((word, label));
        }
        if family_drifted {
            family_drifted_words += cube_accepted;
            cubes_family_drifted.push(cube.id.clone());
            continue;
        }
        let mut regions = Vec::new();
        partition(&assignments, &positions, 0, 0, &mut regions);
        let new_regions = regions
            .iter()
            .filter(|region| matches!(region.label, Label::New(_)))
            .count();
        if packets
            .len()
            .checked_add(new_regions)
            .is_none_or(|total| total > max_packets)
        {
            return Err(format!(
                "hybrid packet cap {max_packets} exceeded while adding cube {}",
                cube.id
            ));
        }
        for region in regions {
            let Label::New(mnemonic_id) = region.label else {
                continue;
            };
            let words = region
                .words
                .into_iter()
                .map(|word| format!("{word:016x}"))
                .collect::<Vec<_>>();
            packets.push(HybridPacket {
                id: String::new(),
                cube: cube.id.clone(),
                row: cube.row,
                family: cube.family.clone(),
                mnemonic: mnemonics[mnemonic_id].clone(),
                semantic_mask: format!("{:016x}", region.mask),
                semantic_value: format!("{:016x}", region.value),
                word_count: words.len(),
                words,
            });
        }
    }
    packets.sort_by(|a, b| {
        a.row
            .cmp(&b.row)
            .then_with(|| a.family.cmp(&b.family))
            .then_with(|| a.mnemonic.cmp(&b.mnemonic))
            .then_with(|| a.semantic_mask.cmp(&b.semantic_mask))
            .then_with(|| a.semantic_value.cmp(&b.semantic_value))
    });
    for (index, packet) in packets.iter_mut().enumerate() {
        packet.id = format!("P{:06}", index + 1);
    }
    let normalized_answers = answers
        .iter()
        .map(|(word, text)| format!("{word:016x}\t{text}\n"))
        .collect::<String>();
    let packets_digest = digest(
        serde_json::to_vec(&packets)
            .map_err(|error| format!("serialize hybrid packets: {error}"))?,
    );
    let packet_words = packets.iter().map(|packet| packet.word_count).sum();
    if oracle_accepted_words + oracle_rejected_words != expected.len() {
        return Err("hybrid oracle counts do not cover every planned word".into());
    }
    if packet_words + family_drifted_words != oracle_accepted_words {
        return Err("hybrid packet counts do not cover every accepted word".into());
    }
    let mut receipt = HybridPackets {
        format: PACKET_FORMAT.into(),
        receipt_digest: String::new(),
        source_head: plan.source_head.clone(),
        source_sha256: plan.source_sha256.clone(),
        plan_digest: plan.plan_digest.clone(),
        plan_words_digest: digest(planned_words.as_bytes()),
        answers_raw_digest: digest(&answer_bytes),
        answers_digest: digest(normalized_answers.as_bytes()),
        packets_digest,
        planned_unknown_words: expected.len(),
        oracle_accepted_words,
        oracle_rejected_words,
        packet_words,
        family_drifted_words,
        cubes_family_drifted,
        packets,
    };
    receipt.receipt_digest = packet_receipt_digest(&receipt)?;
    Ok(receipt)
}

pub fn build_bundles(
    packets: &HybridPackets,
    max_regions: usize,
    max_words: usize,
) -> Result<HybridBundleSet, String> {
    validate_packets(packets)?;
    if max_regions == 0 || max_words == 0 {
        return Err("hybrid bundle limits must be nonzero".into());
    }
    let mut regions = packets.packets.iter().collect::<Vec<_>>();
    regions.sort_by(|a, b| {
        a.row
            .cmp(&b.row)
            .then_with(|| a.family.cmp(&b.family))
            .then_with(|| a.mnemonic.cmp(&b.mnemonic))
            .then_with(|| a.semantic_mask.cmp(&b.semantic_mask))
            .then_with(|| a.semantic_value.cmp(&b.semantic_value))
            .then_with(|| a.cube.cmp(&b.cube))
            .then_with(|| a.id.cmp(&b.id))
    });

    let mut seen_words = BTreeSet::new();
    let mut bundles = Vec::new();
    let mut current: Option<BundleDraft> = None;
    for region in regions {
        if region.word_count > max_words {
            return Err(format!(
                "hybrid region {} has {} words, above bundle max {max_words}",
                region.id, region.word_count
            ));
        }
        for word in &region.words {
            if !seen_words.insert(word) {
                return Err(format!(
                    "hybrid word {word} appears in more than one region"
                ));
            }
        }
        let can_append = current.as_ref().is_some_and(|bundle| {
            bundle.row == region.row
                && bundle.family == region.family
                && bundle.mnemonic == region.mnemonic
                && bundle.semantic_mask == region.semantic_mask
                && bundle.semantic_value == region.semantic_value
                && bundle.region_ids.len() < max_regions
                && bundle.word_count + region.word_count <= max_words
        });
        if !can_append {
            if let Some(bundle) = current.take() {
                bundles.push(bundle.finish(&packets.receipt_digest, max_regions, max_words));
            }
            current = Some(BundleDraft {
                row: region.row,
                family: region.family.clone(),
                mnemonic: region.mnemonic.clone(),
                semantic_mask: region.semantic_mask.clone(),
                semantic_value: region.semantic_value.clone(),
                word_count: 0,
                region_ids: Vec::new(),
            });
        }
        let bundle = current.as_mut().expect("bundle was just initialized");
        bundle.word_count += region.word_count;
        bundle.region_ids.push(region.id.clone());
    }
    if let Some(bundle) = current {
        bundles.push(bundle.finish(&packets.receipt_digest, max_regions, max_words));
    }
    bundles.sort_by(|a, b| {
        b.word_count
            .cmp(&a.word_count)
            .then_with(|| a.row.cmp(&b.row))
            .then_with(|| a.family.cmp(&b.family))
            .then_with(|| a.mnemonic.cmp(&b.mnemonic))
            .then_with(|| a.semantic_mask.cmp(&b.semantic_mask))
            .then_with(|| a.semantic_value.cmp(&b.semantic_value))
            .then_with(|| a.id.cmp(&b.id))
    });

    let bundles_digest = digest(
        serde_json::to_vec(&bundles)
            .map_err(|error| format!("serialize hybrid bundles: {error}"))?,
    );
    Ok(HybridBundleSet {
        format: BUNDLE_FORMAT.into(),
        source_packet_receipt_digest: packets.receipt_digest.clone(),
        max_regions,
        max_words,
        region_count: packets.packets.len(),
        word_count: seen_words.len(),
        bundles_digest,
        bundles,
    })
}

pub fn validate_preflight(
    packets: &HybridPackets,
    bundles: &HybridBundleSet,
    source_head: &str,
) -> Result<(), String> {
    validate_packets(packets)?;
    if packets.source_head != source_head {
        return Err(format!(
            "hybrid packets target source {}, not current source {source_head}",
            packets.source_head
        ));
    }
    let rebuilt = build_bundles(packets, bundles.max_regions, bundles.max_words)?;
    if &rebuilt != bundles {
        return Err("hybrid bundles are not a byte-identical packet replay".into());
    }
    Ok(())
}

pub fn validate_qualification_space(available: u64, required: u64) -> Result<(), String> {
    if required == 0 {
        return Err("qualification minimum free bytes must be nonzero".into());
    }
    if available < required {
        return Err(format!(
            "qualification requires {required} free bytes, found {available}"
        ));
    }
    Ok(())
}

struct BundleDraft {
    row: u32,
    family: String,
    mnemonic: String,
    semantic_mask: String,
    semantic_value: String,
    word_count: usize,
    region_ids: Vec<String>,
}

impl BundleDraft {
    fn finish(self, source: &str, max_regions: usize, max_words: usize) -> HybridBundle {
        let identity = format!(
            "{source}\t{max_regions}\t{max_words}\t{}\t{}\t{}\t{}\t{}\t{}",
            self.row,
            self.family,
            self.mnemonic,
            self.semantic_mask,
            self.semantic_value,
            self.region_ids.join(",")
        );
        HybridBundle {
            id: format!("B{}", &digest(identity.as_bytes())[..16]),
            row: self.row,
            family: self.family,
            mnemonic: self.mnemonic,
            semantic_mask: self.semantic_mask,
            semantic_value: self.semantic_value,
            region_count: self.region_ids.len(),
            word_count: self.word_count,
            region_ids: self.region_ids,
        }
    }
}

fn render_words_from_cubes(cubes: &[HybridCube]) -> String {
    // The established shared answer store consumes column three as the word.
    // Keep this headerless so the same file can flow into it without a shim.
    let mut out = String::new();
    for cube in cubes {
        for word in &cube.unknown_words {
            out.push_str(&format!(
                "{}\t{}\t{}\t{}\t{}\n",
                cube.id, cube.row, word, cube.base, cube.semantic_mask
            ));
        }
    }
    out
}

fn plan_digest(plan: &HybridPlan) -> Result<String, String> {
    let mut canonical = plan.clone();
    canonical.plan_digest.clear();
    Ok(digest(serde_json::to_vec(&canonical).map_err(|error| {
        format!("serialize hybrid plan: {error}")
    })?))
}

fn validate_plan(plan: &HybridPlan) -> Result<(), String> {
    if plan.format != PLAN_FORMAT {
        return Err(format!("unsupported hybrid plan format {:?}", plan.format));
    }
    if plan.source_sha256 != TRANSLATOR_VERSION {
        return Err("hybrid plan was built by a different translator source".into());
    }
    if plan.max_bits == 0 || plan.max_bits > 20 || plan.max_words == 0 || plan.max_domain_words == 0
    {
        return Err("hybrid plan has invalid limits".into());
    }
    if plan.plan_digest != plan_digest(plan)? {
        return Err("hybrid plan digest is stale".into());
    }
    let mut seen = BTreeSet::new();
    let mut emitted = 0_usize;
    let mut domain_words = 0_u64;
    for (index, cube) in plan.cubes.iter().enumerate() {
        if cube.id != format!("H{:06}", index + 1) {
            return Err(format!("hybrid cube {} has a noncanonical id", cube.id));
        }
        let base = parse_hex(&cube.base, "cube base")?;
        let mask = parse_hex(&cube.semantic_mask, "cube semantic mask")?;
        let bits = mask.count_ones() as u8;
        if bits != cube.semantic_bits || bits == 0 || bits > plan.max_bits {
            return Err(format!(
                "hybrid cube {} has inconsistent semantic bits",
                cube.id
            ));
        }
        if cube.domain_words != 1_u64 << bits {
            return Err(format!(
                "hybrid cube {} has inconsistent domain size",
                cube.id
            ));
        }
        domain_words = domain_words
            .checked_add(cube.domain_words)
            .ok_or("hybrid plan domain-word count overflow")?;
        for text in &cube.unknown_words {
            let word = parse_hex(text, "cube unknown word")?;
            if word & !mask != base {
                return Err(format!(
                    "hybrid cube {} contains a word outside its domain",
                    cube.id
                ));
            }
            if !seen.insert(word) {
                return Err(format!("hybrid word {word:016x} has multiple cube owners"));
            }
            let raw = [word as u32, (word >> 32) as u32];
            if !matches!(decode_word(raw, 0).kind, InstructionKind::Unknown { .. }) {
                return Err(format!("hybrid word {word:016x} is no longer Unknown"));
            }
        }
        emitted += cube.unknown_words.len();
    }
    if emitted != plan.emitted_words || emitted as u64 > plan.max_words {
        return Err("hybrid plan emitted-word count is inconsistent".into());
    }
    if domain_words > plan.max_domain_words {
        return Err("hybrid plan exceeds its domain-work cap".into());
    }
    let words = render_words_from_cubes(&plan.cubes);
    if digest(words.as_bytes()) != plan.words_digest {
        return Err("hybrid plan words digest is stale".into());
    }
    Ok(())
}

fn validate_packets(packets: &HybridPackets) -> Result<(), String> {
    if packets.format != PACKET_FORMAT {
        return Err(format!(
            "unsupported hybrid packet format {:?}",
            packets.format
        ));
    }
    if packets.source_sha256 != TRANSLATOR_VERSION {
        return Err("hybrid packets were built by a different translator source".into());
    }
    if packets.receipt_digest != packet_receipt_digest(packets)? {
        return Err("hybrid packet receipt digest is stale".into());
    }
    let actual_packets_digest = digest(
        serde_json::to_vec(&packets.packets)
            .map_err(|error| format!("serialize hybrid packets: {error}"))?,
    );
    if actual_packets_digest != packets.packets_digest {
        return Err("hybrid packet digest is stale".into());
    }
    let mut seen = BTreeSet::new();
    let mut word_count = 0;
    for (index, packet) in packets.packets.iter().enumerate() {
        if packet.id != format!("P{:06}", index + 1) {
            return Err(format!("hybrid packet {} has a noncanonical id", packet.id));
        }
        if packet.word_count != packet.words.len() || packet.word_count == 0 {
            return Err(format!(
                "hybrid packet {} has an inconsistent word count",
                packet.id
            ));
        }
        let mask = parse_hex(&packet.semantic_mask, "packet semantic mask")?;
        let value = parse_hex(&packet.semantic_value, "packet semantic value")?;
        if value & !mask != 0 {
            return Err(format!(
                "hybrid packet {} has a value outside its mask",
                packet.id
            ));
        }
        for text in &packet.words {
            let word = parse_hex(text, "packet word")?;
            if word & mask != value {
                return Err(format!(
                    "hybrid packet {} contains a word outside its region",
                    packet.id
                ));
            }
            if !seen.insert(word) {
                return Err(format!(
                    "hybrid word {word:016x} appears in more than one region"
                ));
            }
        }
        word_count += packet.word_count;
    }
    if word_count != packets.packet_words
        || word_count + packets.family_drifted_words != packets.oracle_accepted_words
        || packets.oracle_accepted_words + packets.oracle_rejected_words
            != packets.planned_unknown_words
    {
        return Err("hybrid packet summary counts are inconsistent".into());
    }
    Ok(())
}

fn packet_receipt_digest(packets: &HybridPackets) -> Result<String, String> {
    let mut canonical = packets.clone();
    canonical.receipt_digest.clear();
    Ok(digest(serde_json::to_vec(&canonical).map_err(|error| {
        format!("serialize hybrid packet receipt: {error}")
    })?))
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
enum Label {
    Existing,
    Rejected,
    New(usize),
}

struct Region {
    mask: u64,
    value: u64,
    label: Label,
    words: Vec<u64>,
}

fn partition(
    assignments: &[(u64, Label)],
    positions: &[u8],
    mask: u64,
    value: u64,
    out: &mut Vec<Region>,
) {
    let first = assignments[0].1;
    if assignments.iter().all(|(_, label)| *label == first) {
        out.push(Region {
            mask,
            value,
            label: first,
            words: assignments.iter().map(|(word, _)| *word).collect(),
        });
        return;
    }
    let bit = *positions
        .iter()
        .min_by_key(|&&bit| split_score(assignments, bit))
        .expect("mixed labels must retain a distinguishing semantic bit");
    let bit_mask = 1_u64 << bit;
    let (zero, one): (Vec<_>, Vec<_>) = assignments
        .iter()
        .copied()
        .partition(|(word, _)| word & bit_mask == 0);
    let remaining = positions
        .iter()
        .copied()
        .filter(|candidate| *candidate != bit)
        .collect::<Vec<_>>();
    partition(&zero, &remaining, mask | bit_mask, value, out);
    partition(&one, &remaining, mask | bit_mask, value | bit_mask, out);
}

fn split_score(assignments: &[(u64, Label)], bit: u8) -> (usize, usize, u8) {
    let mask = 1_u64 << bit;
    let mut zero = BTreeSet::new();
    let mut one = BTreeSet::new();
    for (word, label) in assignments {
        if word & mask == 0 {
            zero.insert(*label);
        } else {
            one.insert(*label);
        }
    }
    (zero.len() + one.len(), zero.len().max(one.len()), bit)
}

fn parse_answers(bytes: &[u8]) -> Result<BTreeMap<u64, String>, String> {
    let input = std::str::from_utf8(bytes).map_err(|_| "hybrid answers are not UTF-8")?;
    let mut answers = BTreeMap::new();
    for (index, line) in input.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 3 {
            return Err(format!(
                "answer line {} must have word, nvdisasm text, and address",
                index + 1
            ));
        }
        let word = parse_hex(fields[0], "answer word")?;
        let text = fields[1];
        if text.is_empty() {
            return Err(format!("answer line {} has empty nvdisasm text", index + 1));
        }
        let address = fields[2]
            .strip_prefix('@')
            .ok_or_else(|| format!("answer line {} has invalid address", index + 1))?;
        if address != "-1"
            && (address.is_empty() || !address.bytes().all(|byte| byte.is_ascii_hexdigit()))
        {
            return Err(format!("answer line {} has invalid address", index + 1));
        }
        if answers.insert(word, text.to_owned()).is_some() {
            return Err(format!("duplicate answer for {word:016x}"));
        }
    }
    Ok(answers)
}

fn mnemonic(text: &str) -> Result<String, String> {
    let mut tokens = text.split_ascii_whitespace();
    let first = tokens.next().ok_or("empty nvdisasm answer")?;
    Ok(if first.starts_with('@') {
        tokens
            .next()
            .ok_or("predicate-only nvdisasm answer")?
            .to_owned()
    } else {
        first.to_owned()
    })
}

fn rejected_oracle_label(text: &str) -> bool {
    let label = text.trim().to_ascii_uppercase();
    label.starts_with('!') || label.contains("INVALID") || label.contains("???")
}

fn parse_hex(value: &str, what: &str) -> Result<u64, String> {
    u64::from_str_radix(value, 16).map_err(|_| format!("invalid {what} {value:?}"))
}

fn parse_conflicts(bytes: &[u8]) -> Result<BTreeSet<(u32, u64)>, String> {
    let input = std::str::from_utf8(bytes).map_err(|_| "conflicts input is not UTF-8")?;
    let mut lines = input.lines();
    if lines.next() != Some(CONFLICTS_HEADER) {
        return Err(format!(
            "conflicts input must start with {CONFLICTS_HEADER:?}"
        ));
    }
    let mut conflicts = BTreeSet::new();
    for (index, line) in lines.enumerate() {
        if line.is_empty() {
            continue;
        }
        let fields = line.split('\t').collect::<Vec<_>>();
        if fields.len() != 6 {
            return Err(format!(
                "conflicts line {} must have six tab-separated fields",
                index + 2
            ));
        }
        let row = fields[0]
            .parse::<u32>()
            .map_err(|_| format!("invalid conflict row on line {}", index + 2))?;
        let form = if fields[1] == "-" {
            0
        } else {
            u64::from_str_radix(fields[1], 16)
                .map_err(|_| format!("invalid conflict form on line {}", index + 2))?
        };
        fields[2]
            .parse::<u8>()
            .map_err(|_| format!("invalid conflict bit on line {}", index + 2))?;
        conflicts.insert((row, form));
    }
    Ok(conflicts)
}

fn semantic_mask(fields: &[ModelField]) -> Result<u64, String> {
    let mut mask = 0_u64;
    for field in fields {
        if !matches!(field.role.as_str(), "enum" | "flag" | "mod") {
            continue;
        }
        for &(lo, hi) in &field.parts {
            if lo > hi || hi >= 64 {
                return Err(format!("invalid model field range {hi}:{lo}"));
            }
            let width = u32::from(hi - lo + 1);
            let bits = if width == 64 {
                u64::MAX
            } else {
                ((1_u64 << width) - 1) << lo
            };
            mask |= bits;
        }
    }
    Ok(mask)
}

fn family(predicted: &str) -> Result<String, String> {
    let mut tokens = predicted.split_ascii_whitespace();
    let first = tokens.next().ok_or("empty Sudoku prediction")?;
    let mnemonic = if first.starts_with('@') {
        tokens.next().ok_or("predicate-only Sudoku prediction")?
    } else {
        first
    };
    Ok(mnemonic.split('.').next().unwrap().to_owned())
}

fn bit_positions(mask: u64) -> Vec<u8> {
    (0..64).filter(|bit| mask & (1_u64 << bit) != 0).collect()
}

fn scatter(value: u64, positions: &[u8]) -> u64 {
    positions
        .iter()
        .enumerate()
        .fold(0_u64, |word, (index, bit)| {
            word | (((value >> index) & 1) << bit)
        })
}

fn rank(a: &CubeDraft, b: &CubeDraft) -> Ordering {
    let a_density = a.unknown_words.len() as u128 * b.domain_words as u128;
    let b_density = b.unknown_words.len() as u128 * a.domain_words as u128;
    b_density
        .cmp(&a_density)
        .then_with(|| b.unknown_words.len().cmp(&a.unknown_words.len()))
        .then_with(|| a.mask.count_ones().cmp(&b.mask.count_ones()))
        .then_with(|| a.row.cmp(&b.row))
        .then_with(|| a.base.cmp(&b.base))
}

fn digest(bytes: impl AsRef<[u8]>) -> String {
    blake3::hash(bytes.as_ref()).to_hex().to_string()
}

fn read(path: &Path) -> Result<Vec<u8>, String> {
    fs::read(path).map_err(|error| format!("read {}: {error}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn packet(id: &str, row: u32, family: &str, words: &[u64]) -> HybridPacket {
        HybridPacket {
            id: id.into(),
            cube: "H000001".into(),
            row,
            family: family.into(),
            mnemonic: format!("{family}.X"),
            semantic_mask: "0000000000000000".into(),
            semantic_value: "0000000000000000".into(),
            word_count: words.len(),
            words: words.iter().map(|word| format!("{word:016x}")).collect(),
        }
    }

    fn packet_set(packets: Vec<HybridPacket>) -> HybridPackets {
        let packets_digest = digest(serde_json::to_vec(&packets).unwrap());
        let packet_words = packets.iter().map(|packet| packet.word_count).sum();
        let mut receipt = HybridPackets {
            format: PACKET_FORMAT.into(),
            receipt_digest: String::new(),
            source_head: "a".repeat(40),
            source_sha256: TRANSLATOR_VERSION.into(),
            plan_digest: "plan-full".into(),
            plan_words_digest: "plan".into(),
            answers_raw_digest: "answers-raw".into(),
            answers_digest: "answers".into(),
            packets_digest,
            planned_unknown_words: packet_words,
            oracle_accepted_words: packet_words,
            oracle_rejected_words: 0,
            packet_words,
            family_drifted_words: 0,
            cubes_family_drifted: Vec::new(),
            packets,
        };
        receipt.receipt_digest = packet_receipt_digest(&receipt).unwrap();
        receipt
    }

    fn test_plan(labels: &[u64]) -> HybridPlan {
        const BASE: u64 = 0x0000_0000_0007_0000;
        let cube = HybridCube {
            id: "H000001".into(),
            row: 7,
            form: "0".into(),
            base: format!("{BASE:016x}"),
            semantic_mask: "0000000000000003".into(),
            semantic_bits: 2,
            family: "X".into(),
            domain_words: 4,
            decoded_subtracted: 4 - labels.len(),
            unknown_words: labels
                .iter()
                .map(|word| format!("{:016x}", BASE | word))
                .collect(),
        };
        let words = render_words_from_cubes(std::slice::from_ref(&cube));
        let mut plan = HybridPlan {
            format: PLAN_FORMAT.into(),
            plan_digest: String::new(),
            source_head: "a".repeat(40),
            source_sha256: TRANSLATOR_VERSION.into(),
            model_digest: "model".into(),
            input_digest: "input".into(),
            conflicts_digest: "conflicts".into(),
            max_bits: 2,
            max_words: 4,
            max_domain_words: 4,
            input_rows_excluded: 0,
            decoded_candidates_subtracted: 0,
            groups_considered: 1,
            groups_conflicted: 0,
            groups_family_ambiguous: 0,
            groups_empty: 0,
            groups_too_wide: 0,
            groups_budget_deferred: 0,
            groups_domain_deferred: 0,
            overlap_words_subtracted: 0,
            emitted_words: labels.len(),
            cubes: vec![cube],
            words_digest: digest(words.as_bytes()),
        };
        plan.plan_digest = plan_digest(&plan).unwrap();
        plan
    }

    fn answer_file(contents: &[u8]) -> std::path::PathBuf {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "shadowbox-hybrid-answers-{}-{suffix}",
            std::process::id()
        ));
        fs::write(&path, contents).unwrap();
        path
    }

    #[test]
    fn scatter_uses_only_requested_bits() {
        assert_eq!(scatter(0b101, &[2, 7, 9]), (1 << 2) | (1 << 9));
    }

    #[test]
    fn source_opcode_alias_bits_are_always_part_of_the_cube() {
        let rows = source_opcode_masks(
            b"# line\ttable\tname\tvalue\tmask\tbaseline\tcitation\n\
2061\troot\tpset\t5088000000000000\tfff8000000000000\t5088000000070000\tsource:2061\n",
        )
        .unwrap();
        assert_eq!(rows[&2061], 0x0007_0000_0000_0000);
    }

    #[test]
    fn rank_prefers_density_then_count_then_width() {
        let cube = |mask, domain_words, words: &[u64]| CubeDraft {
            row: 1,
            form: 0,
            base: 0,
            mask,
            family: "X".into(),
            domain_words,
            decoded_subtracted: 0,
            unknown_words: words.to_vec(),
        };
        let dense = cube(0b11, 4, &[0, 1, 2]);
        let sparse = cube(0b1, 2, &[0]);
        assert_eq!(rank(&dense, &sparse), Ordering::Less);
    }

    #[test]
    fn family_ignores_predicate_and_modifiers() {
        assert_eq!(family("@!P0 DMUL.RP R0, R2, R4").unwrap(), "DMUL");
        assert_eq!(family("DADD R0, R2, R4").unwrap(), "DADD");
    }

    #[test]
    fn conflict_parser_is_strict_and_normalizes_dash() {
        let bytes =
            b"#row\tform\tbit\tkind\tcounts\texample_base\n7\t-\t3\tk\tc\t-\n8\t10\t4\tk\tc\t0\n";
        let conflicts = parse_conflicts(bytes).unwrap();
        assert!(conflicts.contains(&(7, 0)));
        assert!(conflicts.contains(&(8, 0x10)));
    }

    #[test]
    fn partition_finds_the_small_exact_semantic_mask() {
        let assignments = vec![
            (0, Label::New(0)),
            (1, Label::New(0)),
            (2, Label::Rejected),
            (3, Label::Rejected),
        ];
        let mut regions = Vec::new();
        partition(&assignments, &[0, 1], 0, 0, &mut regions);
        let accepted = regions
            .iter()
            .find(|region| matches!(region.label, Label::New(0)))
            .unwrap();
        assert_eq!(accepted.mask, 0b10);
        assert_eq!(accepted.value, 0);
        assert_eq!(accepted.words, vec![0, 1]);
    }

    #[test]
    fn answers_are_normalized_without_addresses() {
        let answers = parse_answers(b"0000000000000001\tX R0\t@28\n").unwrap();
        assert_eq!(answers[&1], "X R0");
        assert!(parse_answers(b"1\tX\n").is_err());
        assert!(parse_answers(b"1\tX\t@0\textra\n").is_err());
        assert!(parse_answers(b"1\tX\tnot-an-address\n").is_err());
        assert!(parse_answers(b"1\tX\t@0\n1\tX\t@4\n").is_err());
    }

    #[test]
    fn overlapping_cubes_assign_each_unknown_word_once() {
        let draft = |row, words: &[u64]| CubeDraft {
            row,
            form: 0,
            base: 0,
            mask: 3,
            family: "X".into(),
            domain_words: 4,
            decoded_subtracted: 0,
            unknown_words: words.to_vec(),
        };
        let mut skips = Skips::default();
        let (selected, overlap) = select_drafts(
            vec![draft(7, &[0, 1, 2]), draft(8, &[1, 2, 3])],
            4,
            &mut skips,
        );
        assert_eq!(overlap, 2);
        assert_eq!(selected[0].unknown_words, vec![0, 1, 2]);
        assert_eq!(selected[1].unknown_words, vec![3]);
        let unique = selected
            .iter()
            .flat_map(|cube| cube.unknown_words.iter())
            .copied()
            .collect::<BTreeSet<_>>();
        assert_eq!(unique, BTreeSet::from([0, 1, 2, 3]));
    }

    #[test]
    fn bundles_are_largest_first_and_do_not_cross_semantic_roads() {
        let mut distinct_semantics = packet("P000004", 7, "X", &[6]);
        distinct_semantics.mnemonic = "X.Y".into();
        let mut distinct_region = packet("P000005", 7, "X", &[7]);
        distinct_region.semantic_mask = "0000000000000001".into();
        distinct_region.semantic_value = "0000000000000001".into();
        let packets = packet_set(vec![
            packet("P000001", 8, "Y", &[5]),
            packet("P000002", 7, "X", &[3, 4]),
            packet("P000003", 7, "X", &[1, 2]),
            distinct_semantics,
            distinct_region,
        ]);
        let first = build_bundles(&packets, 2, 4).unwrap();
        let second = build_bundles(&packets, 2, 4).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.region_count, 5);
        assert_eq!(first.word_count, 7);
        assert_eq!(first.bundles.len(), 4);
        assert_eq!(first.bundles[0].family, "X");
        assert_eq!(first.bundles[0].mnemonic, "X.X");
        assert_eq!(first.bundles[0].region_count, 2);
        assert_eq!(first.bundles[0].word_count, 4);
        assert_eq!(first.bundles[1].semantic_mask, "0000000000000001");
        assert_eq!(first.bundles[2].mnemonic, "X.Y");
        assert_eq!(first.bundles[3].family, "Y");
    }

    #[test]
    fn bundles_reject_tamper_duplicates_and_oversized_regions() {
        let mut metadata = packet_set(vec![packet("P000001", 7, "X", &[1])]);
        metadata.plan_digest = "forged".into();
        assert!(build_bundles(&metadata, 1, 1)
            .unwrap_err()
            .contains("receipt digest is stale"));

        let mut stale = packet_set(vec![packet("P000001", 7, "X", &[1])]);
        stale.packets[0].family = "Y".into();
        assert!(build_bundles(&stale, 1, 1)
            .unwrap_err()
            .contains("digest is stale"));

        let duplicate = packet_set(vec![
            packet("P000001", 7, "X", &[1]),
            packet("P000002", 7, "X", &[1]),
        ]);
        assert!(build_bundles(&duplicate, 2, 2)
            .unwrap_err()
            .contains("more than one region"));

        let oversized = packet_set(vec![packet("P000001", 7, "X", &[1, 2])]);
        assert!(build_bundles(&oversized, 2, 1)
            .unwrap_err()
            .contains("above bundle max"));
    }

    #[test]
    fn preflight_rejects_stale_formats_heads_and_bundle_receipts() {
        let packets = packet_set(vec![packet("P000001", 7, "X", &[1])]);
        let bundles = build_bundles(&packets, 1, 1).unwrap();
        assert!(validate_preflight(&packets, &bundles, &"a".repeat(40)).is_ok());
        assert!(validate_preflight(&packets, &bundles, &"b".repeat(40))
            .unwrap_err()
            .contains("not current source"));

        let mut stale_packets = packets.clone();
        stale_packets.format = "shadowbox-hybrid-packets-v2".into();
        assert!(
            validate_preflight(&stale_packets, &bundles, &"a".repeat(40))
                .unwrap_err()
                .contains("unsupported hybrid packet format")
        );

        let mut stale_bundles = bundles.clone();
        stale_bundles.source_packet_receipt_digest = "obsolete".into();
        assert!(
            validate_preflight(&packets, &stale_bundles, &"a".repeat(40))
                .unwrap_err()
                .contains("not a byte-identical packet replay")
        );
    }

    #[test]
    fn qualification_space_guard_cannot_be_disabled() {
        assert!(validate_qualification_space(u64::MAX, 0)
            .unwrap_err()
            .contains("must be nonzero"));
        assert!(validate_qualification_space(7, 8)
            .unwrap_err()
            .contains("requires 8"));
        assert!(validate_qualification_space(8, 8).is_ok());
    }

    #[test]
    fn plan_digest_and_structure_reject_tampering() {
        let mut plan = test_plan(&[0, 1]);
        assert!(validate_plan(&plan).is_ok());
        plan.groups_considered += 1;
        assert!(validate_plan(&plan)
            .unwrap_err()
            .contains("plan digest is stale"));

        let mut malformed = test_plan(&[0, 1]);
        malformed.cubes[0].domain_words = 8;
        malformed.plan_digest = plan_digest(&malformed).unwrap();
        assert!(validate_plan(&malformed)
            .unwrap_err()
            .contains("inconsistent domain size"));
    }

    #[test]
    fn packet_cap_fails_closed_on_alternating_oracle_labels() {
        let plan = test_plan(&[0, 1, 2, 3]);
        let answers = answer_file(
            b"0000000000070000\tX.A R0\t@0\n0000000000070001\tX.B R0\t@4\n0000000000070002\tX.A R0\t@8\n0000000000070003\tX.B R0\t@c\n",
        );
        let error = build_packets(&plan, &answers, 1).unwrap_err();
        fs::remove_file(answers).unwrap();
        assert!(error.contains("packet cap 1 exceeded"));
    }

    #[test]
    fn invalid_oracle_labels_are_rejected_before_packetization() {
        let plan = test_plan(&[0, 1]);
        let answers = answer_file(b"0000000000070000\tSTS.INVALIDSIZE7 R0\t@0\n0000000000070001\tVOTE.VTG.INVALIDVTGMODE3 R0\t@4\n");
        let packets = build_packets(&plan, &answers, 10).unwrap();
        fs::remove_file(answers).unwrap();
        assert_eq!(packets.oracle_rejected_words, 2);
        assert!(packets.packets.is_empty());
    }

    #[test]
    fn packet_receipt_binds_raw_oracle_addresses() {
        let plan = test_plan(&[0, 1]);
        let first = answer_file(b"0000000000070000\tX.A R0\t@0\n0000000000070001\tX.A R0\t@4\n");
        let second = answer_file(b"0000000000070000\tX.A R0\t@8\n0000000000070001\tX.A R0\t@c\n");
        let a = build_packets(&plan, &first, 10).unwrap();
        let b = build_packets(&plan, &second, 10).unwrap();
        fs::remove_file(first).unwrap();
        fs::remove_file(second).unwrap();
        assert_eq!(a.plan_digest, plan.plan_digest);
        assert_eq!(a.answers_digest, b.answers_digest);
        assert_ne!(a.answers_raw_digest, b.answers_raw_digest);
        assert_eq!(a.packets_digest, b.packets_digest);
    }

    #[test]
    fn domain_reservation_never_truncates_or_overflows() {
        let mut used = 0;
        assert!(reserve_domain(&mut used, 1024, 1024));
        assert_eq!(used, 1024);
        assert!(!reserve_domain(&mut used, 1, 1024));
        assert_eq!(used, 1024);
        used = u64::MAX - 1;
        assert!(!reserve_domain(&mut used, 2, u64::MAX));
        assert_eq!(used, u64::MAX - 1);
    }
}
