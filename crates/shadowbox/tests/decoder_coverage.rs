//! Row sweep generated from MIT envytools/envydis/gm107.c:1874-2182.
//! Regenerate inputs with tools/generate-decoder-coverage.py. This measures
//! decoding, not emission or semantic correctness. Unknown is a valid result.

use shadowbox::ir::ops;
use std::{collections::BTreeMap, fmt::Write, fs, path::Path};

use shadowbox::{decoder::decode_word, ir::InstructionKind};

const ROWS: &str = include_str!("data/decoder-rows.tsv");
const SAMPLES: &str = include_str!("data/decoder-samples.tsv");
const RESULTS: &str = include_str!("data/decoder-results.tsv");
const REPORT: &str = include_str!("../../../docs/decoder-coverage.md");
const MAIN_ALL: &str = include_str!("data/decoder-main-d3f01ae.tsv");

const MAIN_FSET: &str = include_str!("data/fset-main-252cf73.tsv");
const FORMATS: &str = include_str!("data/decoder-formats.tsv");
const FORMAT_SAMPLES: &str = include_str!("data/decoder-format-samples.tsv");
const FORMAT_TABLE: &str = include_str!("../src/decoder/formats/table.rs");

#[derive(Default)]
struct Counts {
    kinds: BTreeMap<String, usize>,
    unknown: usize,
    reserved_unknown: usize,
    reserved_decoded: usize,
    baseline: String,
}

/// The recorded name of a decoded operation. Main d3f01ae had one kind per
/// BRA, SSY and PBK/PCNT operation; a target outside the code address space
/// now decodes into its own lane of that operation (Ryujinx
/// Decoders/InstOp.cs:22-25 does not wrap the target either).
fn operation_name(kind: &InstructionKind) -> String {
    match kind {
        InstructionKind::OutsideBranch(ops::OutsideBranch { .. }) => "Branch".to_owned(),
        InstructionKind::OutsideSchedule(ops::OutsideSchedule { .. }) => "Schedule".to_owned(),
        InstructionKind::OutsidePushTarget(ops::OutsidePushTarget { .. }) => {
            "PushTarget".to_owned()
        }
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

#[test]
fn main_252cf73_fset_acceptance_is_preserved() {
    // Captured by running main's actual decoder against SAMPLES, including
    // PT variants so the field probes reach main's FSET arm. Accumulator:
    // Mesa sm50.rs:858; FTZ:861; envytools gm107.c:2036,2106,2154.
    let samples = records(SAMPLES).collect::<Vec<_>>();
    let accepted = records(MAIN_FSET).collect::<Vec<_>>();
    assert_eq!(accepted.len(), 500);
    let mut original = 0;
    let mut reserved = 0;
    for record in accepted {
        let sample = &samples[record[0].parse::<usize>().unwrap()];
        let raw = u64::from_str_radix(sample[3], 16).unwrap();
        let expected = match record[1] {
            "sample" => {
                original += 1;
                raw
            }
            "PT" => (raw & !(15 << 39)) | (7 << 39),
            "PT.FTZ" => (raw & !(15 << 39)) | (7 << 39) | (1 << 55),
            variant => panic!("unknown baseline variant {variant}"),
        };
        let raw = u64::from_str_radix(record[2], 16).unwrap();
        assert_eq!(raw, expected, "baseline no longer matches its sample");
        let kind = decode_word([raw as u32, (raw >> 32) as u32], 0).kind;
        if (raw >> 45) & 3 == 3 {
            // The only intentional acceptance exception: Boolean3 is OOPS
            // in envytools gm107.c:1430-1434. Do not invent an operation.
            reserved += 1;
            assert!(matches!(
                kind,
                InstructionKind::Unknown(ops::Unknown { .. })
            ));
        } else {
            assert!(
                matches!(kind, InstructionKind::FSet(ops::FSet { .. })),
                "main accepted {raw:016x}: row {} {} ({}) became {kind:?}",
                sample[0],
                sample[1],
                sample[2]
            );
        }
    }
    assert_eq!(
        original, 3,
        "all original main FSET samples must be retained"
    );
    assert_eq!(reserved, 6);
}

#[test]
fn dset_samples_are_never_substituted_with_fset() {
    // DSET register/cbuf/immediate precede FSET in envytools gm107.c:
    // 2035-2036,2105-2106,2153-2154. Ryujinx InstTable.cs:78-80,117-119
    // distinguishes these double comparisons from float32 FSET. In
    // particular, cbuf bit56 is an opcode bit even though the later FSET
    // table row's fe00 mask overlaps it. A sample nvdisasm rejects, such as
    // bop 3 (DSET.F.INVALIDBOP3), stays Unknown.
    for sample in records(SAMPLES) {
        if ![2035, 2105, 2153].contains(&sample[0].parse::<usize>().unwrap()) {
            continue;
        }
        let raw = u64::from_str_radix(sample[3], 16).unwrap();
        for raw in [raw, (raw & !(15 << 39)) | (7 << 39)] {
            let kind = decode_word([raw as u32, (raw >> 32) as u32], 0).kind;
            assert!(
                matches!(
                    kind,
                    InstructionKind::DoubleSet(ops::DoubleSet { .. })
                        | InstructionKind::Unknown(ops::Unknown { .. })
                ),
                "DSET {raw:016x} became {kind:?}"
            );
        }
    }
}

#[test]
fn every_op8b_row_and_significant_field_value_is_recorded() {
    let row_words = records(ROWS)
        .map(|r| {
            (
                r[0].parse::<usize>().unwrap(),
                (
                    u64::from_str_radix(r[3], 16).unwrap(),
                    u64::from_str_radix(r[4], 16).unwrap(),
                    u64::from_str_radix(r[5], 16).unwrap(),
                ),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let mut counts = BTreeMap::<usize, Counts>::new();
    let mut results =
        String::from("# Decoder kind per decoder-samples.tsv record, in the same order.\n");
    for sample in records(SAMPLES) {
        assert_eq!(sample.len(), 5);
        let row: usize = sample[0].parse().unwrap();
        let raw = u64::from_str_radix(sample[3], 16).unwrap();
        let (value, mask, baseline) = row_words[&row];
        assert_eq!(
            raw & mask,
            value,
            "sample escaped its OP8B row: {} {}",
            row,
            sample[1]
        );
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 0);
        // All sampled S2R, conversion, predicate and texture fields must
        // change complete IR or guard, or be rejected. Scalar tables:
        // envytools gm107.c:893-898,932-972,1960,1967.
        if [
            1879, 1925, 1956, 1957, 1958, 1960, 1967, 1968, 1970, 1973, 1974, 1975, 1976, 1977,
            1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1995, 1997,
            1999, 1992, 2016, 2020, 2022, 2025, 2030, 2036, 2041, 2044, 2046, 2052, 2053, 2054,
            2058, 2060, 2061, 2066, 2069, 2071, 2073, 2074, 2090, 2096, 2101, 2106, 2112, 2115,
            2136, 2139, 2143, 2148, 2154, 2155, 2156, 2157, 2169, 2170,
        ]
        .contains(&row)
            && raw != baseline
        {
            let plain = decode_word([baseline as u32, (baseline >> 32) as u32], 0);
            assert_ne!(
                instruction, plain,
                "ignored field: {} {} ({})",
                row, sample[1], sample[2]
            );
        }
        let kind = match instruction.kind {
            InstructionKind::Unknown(ops::Unknown { raw: words }) => {
                assert_eq!(words, [raw as u32, (raw >> 32) as u32]);
                "Unknown".to_owned()
            }
            ref kind => operation_name(kind),
        };
        if matches!(instruction.kind, InstructionKind::Nop(ops::Nop)) {
            // Mesa sm50.rs:3178-3184 OpNop; envytools gm107.c:1877-1878
            // SYNC/DEPBAR. No other row may be silently dropped.
            assert!(raw >> 48 == 0x50b0 || (0xf0f0..=0xf0ff).contains(&(raw >> 48)));
        }
        let count = counts.entry(row).or_default();
        if sample[1] == "baseline" {
            assert!(count.baseline.is_empty());
            count.baseline.clone_from(&kind);
        }
        if kind == "Unknown" {
            count.unknown += 1;
            count.reserved_unknown += usize::from(sample[4] == "1");
        } else {
            *count.kinds.entry(kind.clone()).or_default() += 1;
            count.reserved_decoded += usize::from(sample[4] == "1");
        }
        writeln!(results, "{kind}").unwrap();
    }

    // Exactly two actual Nop arms stay visible; all other gaps use Unknown.
    let decoder = include_str!("../src/decoder/families.rs");
    assert_eq!(
        decoder
            .matches("x.guarded(InstructionKind::Nop(ops::Nop))")
            .count(),
        2
    );

    let mut report = String::from(
        "# Decoder coverage\n\n\
         Generated by `cargo test --test decoder_coverage` from the checked-in\n\
         [row manifest](../crates/shadowbox/tests/data/decoder-rows.tsv) and\n\
         [field samples](../crates/shadowbox/tests/data/decoder-samples.tsv).\n\
         [Results](../crates/shadowbox/tests/data/decoder-results.tsv) record one\n\
         InstructionKind or Unknown for every sample, in manifest order.\n\n\
         Encoding reference: MIT `envytools/envydis/gm107.c:1874-2182`.\n\
         Its SHA256 is recorded in both manifests. Every field sample cites its\n\
         OP8B row, table, macro and operand definition. The generator is\n\
         [tools/generate-decoder-coverage.py](../tools/generate-decoder-coverage.py).\n\n\
         Each selector value (including unassigned values), flag value,\n\
         predicate, small immediate/control field (up to 8 bits), and named or\n\
         unassigned system register is sampled. Larger data operands use zero,\n\
         one, maximum, sign boundary and each single bit; registers use bit\n\
         boundaries and RZ. Nested operands select their containing table arm.\n\
         Wide selectors that also constrain data operands use each table arm\n\
         and each constrained-bit flip. These are one-field-at-a-time probes,\n\
         not every combination or every possible operand value.\n\n\
         Decoded means a real IR kind was returned; it does not certify correct\n\
         semantics, emitter support, or Vulkan representability. Nop is listed\n\
         explicitly and permitted only for Mesa OpNop (`sm50.rs:3178-3184`) or\n\
         the scheduling family (`gm107.c:1877-1878`). Reserved means envytools\n\
         selects OOPS, not a guess about hardware behaviour. Accepted reserved\n\
         probes are reported separately for follow-up. Scheduling control\n\
         records and catch-all/assembly dispatch records are included because\n\
         they contain OP8B; they are not ordinary instruction rows.\n\n\
         Unknown reasons: **not implemented yet** means this operand, mode or\n\
         instruction lacks a decoder representation. **Reserved** means OOPS\n\
         in the source table. No row is classified **not representable in\n\
         Vulkan** without proof; absent implementation alone is insufficient.\n\
         GPU semcheck and Vulkan comparisons were unavailable on the build\n\
         server and were skipped. CPU spirv-val validation is separate.\n\n",
    );
    let total: usize = counts
        .values()
        .map(|c| c.kinds.values().sum::<usize>() + c.unknown)
        .sum();
    writeln!(
        report,
        "{} OP8B rows, {total} representative words.\n",
        counts.len()
    )
    .unwrap();
    report.push_str("| Source row | Table / mnemonic | Match / mask | Baseline | Decoded kinds (samples) | Unknown | Accepted reserved | Unknown reason |\n\
                     | --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for row in records(ROWS) {
        assert_eq!(row.len(), 7);
        let line: usize = row[0].parse().unwrap();
        let count = counts.remove(&line).unwrap();
        let kinds = count
            .kinds
            .iter()
            .map(|(k, n)| format!("{k} ({n})"))
            .collect::<Vec<_>>()
            .join(", ");
        let reason = match (
            count.unknown - count.reserved_unknown,
            count.reserved_unknown,
        ) {
            (0, 0) => "",
            (0, _) => "reserved: source table OOPS",
            (_, 0) => "not implemented yet: instruction / operand / mode",
            (_, _) => {
                "not implemented yet: instruction / operand / mode; reserved: source table OOPS"
            }
        };
        writeln!(
            report,
            "| `gm107.c:{}` | {} / {} | `{}` / `{}` | {} | {} | {} ({} reserved) | {} | {} |",
            row[0],
            row[1],
            row[2],
            row[3],
            row[4],
            count.baseline,
            kinds,
            count.unknown,
            count.reserved_unknown,
            count.reserved_decoded,
            reason
        )
        .unwrap();
    }
    assert!(counts.is_empty(), "sample refers to an absent OP8B row");
    if std::env::var_os("SHADOWBOX_UPDATE_COVERAGE").is_some() {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        fs::write(manifest.join("tests/data/decoder-results.tsv"), &results).unwrap();
        fs::write(manifest.join("../../docs/decoder-coverage.md"), &report).unwrap();
    } else {
        assert_eq!(
            results.lines().count(),
            RESULTS.lines().count(),
            "coverage size changed"
        );
        for ((actual, expected), sample) in records(&results)
            .zip(records(RESULTS))
            .zip(records(SAMPLES))
        {
            assert_eq!(
                actual, expected,
                "row {} {} word {} ({})",
                sample[0], sample[1], sample[3], sample[2]
            );
        }
        assert!(
            report == REPORT,
            "coverage documentation is stale; regenerate the reviewed snapshot"
        );
    }
}

#[test]
fn every_main_d3f01ae_opcode_and_field_sample_stays_decoded() {
    // Actual main decoder results, captured by tools/bisect-batch12-decoder.py
    // --capture. Inputs exhaust all 65536 high16 values with four operand
    // profiles, plus every cited sample from envytools gm107.c:1874-2182.
    // Check each word and its operation, not only the total acceptance count.
    let mut failures = Vec::new();
    let mut count = 0;
    let mut swizzle_expansions = 0;
    let mut unmapped_reads = 0;
    let mut i2i_unnamed = 0;
    let mut surface_size7 = 0;
    let mut misaligned = 0;
    for row in records(MAIN_ALL) {
        let raw = u64::from_str_radix(row[0], 16).unwrap();
        let kind = decode_word([raw as u32, (raw >> 32) as u32], 0).kind;
        let name = operation_name(&kind);
        if row[1] == "Derivative" {
            // These two old baseline words were classified using the mask
            // alone. FSWZADD reads A8..15 and B20..27, plus mask28..35 and
            // FTZ44: Mesa sm50.rs:913-943. Ryujinx InstEmitWarp.cs:11-21
            // and CodeGen/Spirv/Instructions.cs:1140-1168 use both sources.
            // Require the full encoded operation; do not restore the lossy
            // one-source derivative substitution just to match an old label.
            assert!([0x50f8_0009_9007_0000, 0x50f8_000a_5007_0000].contains(&raw));
            assert_eq!(
                kind,
                InstructionKind::SwizzleAdd(ops::SwizzleAdd {
                    dst: shadowbox::ir::Register(raw as u8),
                    src: [
                        shadowbox::ir::Register((raw >> 8) as u8),
                        shadowbox::ir::Register((raw >> 20) as u8),
                    ],
                    mask: (raw >> 28) as u8,
                    ftz: raw & (1 << 44) != 0,
                })
            );
            swizzle_expansions += 1;
        } else if name == "UnmappedAttributeLoad" && row[1] == "AttributeLoad" {
            // ALD R0 at an offset with no attribute: Ryujinx
            // AttributeMap.cs:63-92,129-133 maps no offset below 0x60 and
            // no unaligned offset, and loads 0, so these are their own kind.
            assert!([
                0xefd8_0000_0007_ff00,
                0xefd8_7f80_0007_ff00,
                0xefd8_7f80_0ff7_ff00,
                0xefd8_7f81_0007_ff00,
                0xefd9_7f80_0007_ff00,
                0xefd9_7f80_0ff7_ff00,
                0xefd9_7f81_0007_ff00,
            ]
            .contains(&raw));
            assert_eq!(
                kind,
                InstructionKind::UnmappedAttributeLoad(ops::UnmappedAttributeLoad {
                    dst: shadowbox::ir::Register(0),
                    // Imm11 is bits 20..30 (InstDecoders.cs:883).
                    address: ((raw >> 20) & 0x7ff) as u16,
                    // Count bits 47..48; efd9 sets bit 48.
                    components: ((raw >> 47) & 3) as u8 + 1,
                })
            );
            unmapped_reads += 1;
        } else if row[1] == "AttributeLoad"
            && matches!(
                name.as_str(),
                "OutputAttributeLoad"
                    | "PatchAttributeLoad"
                    | "PhysicalAttributeLoad"
                    | "PhysicalOutputAttributeLoad"
            )
        {
            // Main's one ALD kind is split by P, O and physical addressing
            // (Ryujinx InstDecoders.cs:883-887, InstEmitAttribute.cs:43-101).
        } else if name == "GroupMemoryBarrier" && row[1] == "MemoryBarrier" {
            // MEMBAR.CTA (scope bits 8..9 = 0) is its own kind: Ryujinx
            // InstEmitBarrier.cs:32-44 emits a group barrier for it.
            assert_eq!(raw >> 51, 0xef98 >> 3);
            assert_eq!((raw >> 8) & 3, 0);
        } else if matches!(
            (name.as_str(), row[1]),
            ("LocalLoad" | "SharedLoad", "Load") | ("LocalStore" | "SharedStore", "Store")
        ) {
            // LD/ST of each memory space is its own kind (Ryujinx
            // InstEmitMemory.cs:110-167 passes LocalMemory or SharedMemory).
            assert!(matches!(raw >> 51, 0x1de8..=0x1deb));
        } else if (name.as_str(), row[1]) == ("Unknown", "Convert")
            && (raw >> 48 & 0xfef9 == 0x38e1 || matches!(raw >> 48 & 0xfff9, 0x4ce1 | 0x5ce1))
        {
            // I2I bit 48 is unnamed in envytools gm107.c:1993,2067,2113
            // and unread by Ryujinx InstI2iR/I/C (InstDecoders.cs:
            // 2731-2781); main admitted it, admission is now exact.
            i2i_unnamed += 1;
        } else if (name.as_str(), row[1]) == ("Unknown", "Surface")
            && raw >> 52 & 1 == 1
            && raw >> 20 & 7 == 7
        {
            // .D size 7 is OOPS in envytools gm107.c:822 and INVALIDSIZE7
            // in nvdisasm SM52; main admitted it as Ryujinx UB128.
            surface_size7 += 1;
        } else if name.strip_prefix("Misaligned") == Some(row[1]) {
            // A target off the 8-byte word grid is its own kind at the
            // word edge; main decoded it and stopped later at the CFG.
            assert_ne!((raw >> 20 & 0xff_ffff) % 8, 0);
            misaligned += 1;
        } else if name != row[1] {
            failures.push(format!("{raw:016x}: {} -> {name}", row[1]));
        }
        count += 1;
    }
    assert_eq!(swizzle_expansions, 2);
    assert_eq!(unmapped_reads, 7);
    assert_eq!(i2i_unnamed, 16);
    assert_eq!(surface_size7, 4);
    assert_eq!(misaligned, 20);
    assert_eq!(count, 51_518, "baseline must be regenerated from main");
    assert!(
        failures.is_empty(),
        "{} main encodings regressed: {}",
        failures.len(),
        failures.join(", ")
    );
}

/// The proof gate of the formats learned from the disassembler
/// (tools/generate-decoder-formats): the generated table is the checked-in
/// format table's, and every checked-in word of an admitted variant prints
/// exactly as the disassembler printed it. A word no hand-written family
/// decodes stays Unknown to translation and its `text()` is the same text.
#[test]
fn every_generated_format_prints_its_samples_exactly() {
    use sha2::{Digest, Sha256};
    use shadowbox::decoder::formats::TABLE;
    let digest = format!("{:x}", Sha256::digest(FORMATS.as_bytes()));
    assert!(
        FORMAT_TABLE.contains(&digest) && FORMAT_SAMPLES.contains(&digest),
        "formats/table.rs or its samples were not generated from decoder-formats.tsv"
    );
    let mut words = 0;
    for sample in records(FORMAT_SAMPLES) {
        let word = u64::from_str_radix(sample[0], 16).unwrap();
        let id = TABLE
            .recognise(word)
            .unwrap_or_else(|| panic!("no admitted variant prints {word:016x}"));
        assert_eq!(
            TABLE.text(id, word).as_deref(),
            Some(sample[1]),
            "variant {id} misprints {word:016x}"
        );
        if let InstructionKind::Unknown(unknown) =
            decode_word([word as u32, (word >> 32) as u32], 0).kind
        {
            assert_eq!(unknown.text().as_deref(), Some(sample[1]), "{word:016x}");
        }
        words += 1;
    }
    assert!(words >= TABLE.variants.len());
}
