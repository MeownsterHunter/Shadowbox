use super::*;
use std::{
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

struct TestDir(PathBuf);

impl TestDir {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        // Each parallel test keeps synthetic inputs and outputs under TMPDIR.
        let path = std::env::temp_dir()
            .join("shadowbox-aot-tests")
            .join(format!(
                "{}-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}

impl Drop for TestDir {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

// Project-authored instructions using Mesa nak/sm50.rs:213-235,3106-3110,
// 3183-3187,1938-1941,1951: opcode, PT guard, CC.T and MOV32I fields.
const EXIT: u64 = 0xe300_0000_0007_000f;
const NOP: u64 = 0x50b0_0000_0007_0f00;
const MOV: u64 = 0x0100_0000_02a7_f000;
// Mesa sm50.rs:243-244: absent read/write barriers are 7.
const CONTROL: u64 = 0x7e0 | (0x7e0 << 21) | (0x7e0 << 42);

fn bundle(instructions: [u64; 3]) -> Vec<u8> {
    [CONTROL]
        .into_iter()
        .chain(instructions)
        .flat_map(u64::to_le_bytes)
        .collect()
}

fn record(stage: headers::Stage, code: &[u8]) -> Vec<u8> {
    let compute = stage == headers::Stage::Compute;
    let mut bytes = vec![0; if compute { 0x100 } else { 0x80 }];
    bytes[..4].copy_from_slice(&(if compute { 0x1234_5679u32 } else { 0x1234_5678 }).to_le_bytes());
    if !compute {
        // headers.rs:64-69 and Mesa nak/sph.rs:112-120: stage/type/version.
        let (stage, ty) = match stage {
            headers::Stage::Vertex => (1u32, 1),
            headers::Stage::Fragment => (5, 2),
            headers::Stage::Geometry => (4, 1),
            // Mesa nak/sph.rs:204-215; headers.rs:74-81: tessellation stages.
            headers::Stage::TessellationControl => (2, 1),
            headers::Stage::TessellationEvaluation => (3, 1),
            headers::Stage::Compute => unreachable!(),
        };
        bytes[0x30..0x34].copy_from_slice(&((stage << 10) | (3 << 5) | ty).to_le_bytes());
    }
    bytes.extend_from_slice(code);
    bytes
}

fn noise(len: usize) -> Vec<u8> {
    let mut seed = 0xdead_beefu32;
    (0..len)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as u8
        })
        .collect()
}

fn insert(bytes: &mut [u8], at: usize, program: &[u8]) {
    bytes[at..at + program.len()].copy_from_slice(program);
}

fn manifest(output: &Path) -> Vec<Value> {
    fs::read_to_string(output.join("manifest.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

fn probe_record(bytes: &[u8]) -> Result<Program, ProbeError> {
    let dir = TestDir::new();
    let path = dir.0.join("synthetic.dat");
    fs::write(&path, bytes).unwrap();
    validate(&mut File::open(path).unwrap(), 0, bytes[0] == 0x79)
}

#[test]
fn recursive_odd_offsets_false_positives_dedup_and_translation() {
    let dir = TestDir::new();
    let input = dir.0.join("input");
    let output = input.join("generated");
    fs::create_dir_all(input.join("nested/deeper")).unwrap();
    let graphics = record(headers::Stage::Fragment, &bundle([MOV, EXIT, NOP]));
    let compute = record(headers::Stage::Compute, &bundle([MOV, EXIT, NOP]));
    let mut first = noise(CHUNK * 4);
    insert(&mut first, CHUNK - 2, &graphics);
    insert(&mut first, CHUNK * 3 + 17, &graphics);
    insert(&mut first, CHUNK - 11, &[0x28, 0xb5, 0x2f, 0xfd]);
    insert(&mut first, CHUNK * 3 + 5, &[4, 0x22, 0x4d, 0x18]);
    first.extend_from_slice(&[0x78, 0x56, 0x34, 0x12]); // Truncated near true matches.
    fs::write(input.join("odd\"name\n.dat"), first).unwrap();
    let mut second = noise(4096);
    insert(&mut second, 1, b"Yaz0"); // Diagnostic hint, not a top-level Yaz0 container.
    insert(&mut second, 19, &compute);
    let mut unknown = graphics.clone();
    unknown[0x88..0x90].fill(0xff);
    insert(&mut second, 513, &unknown);
    let mut bad_control = graphics.clone();
    bad_control[0x87] |= 0x80;
    insert(&mut second, 1025, &bad_control);
    let mut bad_stage = graphics.clone();
    bad_stage[0x30..0x34].copy_from_slice(&((6u32 << 10) | (3 << 5) | 1).to_le_bytes());
    insert(&mut second, 1537, &bad_stage);
    fs::write(input.join("nested/deeper/programs.data"), second).unwrap();
    fs::write(input.join("empty"), []).unwrap();
    let mut report = Vec::new();
    let totals = scan(&input, &output, true, &mut report).unwrap();
    assert_eq!(
        (
            totals.files,
            totals.candidates,
            totals.accepted,
            totals.rejected
        ),
        (3, 7, 3, 4)
    );
    assert_eq!(
        (totals.unique, totals.translated, totals.translation_failed),
        (2, 2, 0)
    );
    let entries = manifest(&output);
    let programs: Vec<_> = entries.iter().filter(|e| e["kind"] == "program").collect();
    assert_eq!(programs.len(), 3);
    for entry in programs {
        let bytes = fs::read(output.join(entry["output"].as_str().unwrap())).unwrap();
        let expected = if entry["stage"] == "compute" {
            [vec![0; HEADER_SIZE], compute[0x100..].to_vec()].concat()
        } else {
            graphics[0x30..].to_vec()
        };
        assert_eq!(bytes, expected); // Header and every instruction byte preserved.
        assert_eq!(entry["sha256"], format!("{:x}", Sha256::digest(&bytes)));
        assert_eq!(entry["size"], bytes.len());
        assert!(!entry["nearby_signatures"].as_array().unwrap().is_empty());
        let spv =
            fs::read(output.join(format!("{}.spv", entry["sha256"].as_str().unwrap()))).unwrap();
        assert_eq!(
            spv,
            Result::from(shadowbox::translate_header_prefixed(
                shadowbox::QuietRun,
                &bytes
            ))
            .unwrap()
            .spirv
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>()
        );
    }
    assert!(String::from_utf8(report)
        .unwrap()
        .contains("accepted=2 rejected=1"));
    // Repeated scans must exclude the output subtree, including the old manifest.
    let again = scan(&input, &output, false, &mut Vec::new()).unwrap();
    assert_eq!((again.files, again.candidates, again.unique), (3, 7, 2));
}

// Mesa nak/sm50.rs:3025-3028: BRA relative 20..44 and condition 0..5.
fn branch(address: u32, destination: u32, predicate: u8) -> u64 {
    let relative = (i64::from(destination) - i64::from(address) - 8) as u64 & 0xff_ffff;
    (0xe240u64 << 48) | (relative << 20) | (u64::from(predicate) << 16) | 0xf
}

#[test]
fn conditional_branch_past_first_exit_preserves_later_block_and_padding() {
    let code = [
        bundle([branch(8, 64, 0), EXIT, NOP]),
        bundle([MOV, NOP, NOP]),
        bundle([MOV, EXIT, MOV]),
    ]
    .concat();
    let program = record(headers::Stage::Vertex, &code);
    let accepted = probe_record(&program).unwrap();
    assert_eq!(accepted.bytes, program[0x30..]);
    assert_eq!(accepted.bytes.len(), HEADER_SIZE + 96);
    // An unknown slot between the first EXIT and the target must not be dropped.
    let mut unknown = program.clone();
    unknown[0x80 + 40..0x80 + 48].fill(0xff);
    assert!(matches!(probe_record(&unknown), Err(ProbeError::Reject(_))));
}

#[test]
fn predicated_exit_does_not_end_program() {
    let predicated_exit = EXIT & !(7 << 16);
    let program = record(
        headers::Stage::Fragment,
        &[
            bundle([predicated_exit, MOV, NOP]),
            bundle([MOV, EXIT, NOP]),
        ]
        .concat(),
    );
    assert_eq!(probe_record(&program).unwrap().bytes, program[0x30..]);
}

#[test]
fn invalid_header_scheduling_and_exit_fields_are_rejected() {
    let valid = record(headers::Stage::Vertex, &bundle([MOV, EXIT, NOP]));
    for (byte, mask) in [(4, 1), (0x30, 1), (0x30, 0x20), (0x87, 0x80), (0x90, 0x10)] {
        let mut invalid = valid.clone();
        invalid[byte] ^= mask;
        assert!(
            matches!(probe_record(&invalid), Err(ProbeError::Reject(_))),
            "byte {byte:x}"
        );
    }
    for slot in 0..3 {
        for shift in [5, 8] {
            let mut invalid = valid.clone();
            let control = (CONTROL & !(7 << (slot * 21 + shift))) | (6 << (slot * 21 + shift));
            invalid[0x80..0x88].copy_from_slice(&control.to_le_bytes());
            assert!(matches!(probe_record(&invalid), Err(ProbeError::Reject(_))));
        }
    }
    let mut compute = record(headers::Stage::Compute, &bundle([MOV, EXIT, NOP]));
    compute[0xff] = 1;
    assert!(matches!(probe_record(&compute), Err(ProbeError::Reject(_))));
}

#[test]
fn unreachable_real_slots_are_preserved_and_unknown_slots_rejected() {
    let program = record(headers::Stage::Vertex, &bundle([EXIT, MOV, MOV]));
    assert_eq!(probe_record(&program).unwrap().bytes, program[0x30..]);
    let mut unknown = program.clone();
    unknown[0x90..0x98].fill(0xff);
    assert!(matches!(probe_record(&unknown), Err(ProbeError::Reject(_))));
}

#[test]
fn truncated_unterminated_and_unprovable_control_flow_are_rejected() {
    let valid = record(headers::Stage::Vertex, &bundle([MOV, EXIT, NOP]));
    for length in [4, 0x30, 0x7f, 0x87, 0x8f, 0x9f] {
        assert!(matches!(
            probe_record(&valid[..length]),
            Err(ProbeError::Reject(_))
        ));
    }
    for instructions in [
        [MOV, MOV, MOV],                           // No terminator before EOF.
        [branch(8, 8, 7), NOP, NOP],               // Closed infinite loop without EXIT.
        [EXIT | (8 << 16), branch(16, 8, 7), NOP], // Never-guarded EXIT does not terminate.
        [branch(8, (MAX_CODE + 8) as u32, 7), EXIT, NOP],
        [branch(8, 9, 7), EXIT, NOP],       // Unaligned target.
        [EXIT, branch(16, 72, 7), NOP],     // Retained branch outside output.
        [0xf0f8_0000_0007_000f, EXIT, NOP], // Unproven stack pop, Mesa sm50.rs:3049-3051.
    ] {
        assert!(matches!(
            probe_record(&record(headers::Stage::Vertex, &bundle(instructions))),
            Err(ProbeError::Reject(_))
        ));
    }
    // Exercise the bounded-size stop with valid, unterminated instruction bundles.
    let code = bundle([MOV, MOV, MOV]).repeat(MAX_CODE / 32);
    assert!(matches!(
        probe_record(&record(headers::Stage::Vertex, &code)),
        Err(ProbeError::Reject(_))
    ));
}

#[test]
fn every_magic_split_across_scan_boundary_is_found_once() {
    let dir = TestDir::new();
    let input = dir.0.join("input");
    fs::create_dir(&input).unwrap();
    let program = record(headers::Stage::Vertex, &bundle([MOV, EXIT, NOP]));
    for split in 0..8 {
        let mut bytes = noise(CHUNK + 1024);
        insert(&mut bytes, CHUNK - split, &program);
        fs::write(input.join(split.to_string()), bytes).unwrap();
    }
    let totals = scan(&input, &dir.0.join("out"), false, &mut Vec::new()).unwrap();
    assert_eq!(
        (totals.candidates, totals.accepted, totals.unique),
        (8, 8, 1)
    );
}

#[test]
fn embedded_signatures_reported_without_programs_do_not_trigger_unpacking() {
    let dir = TestDir::new();
    let input = dir.0.join("input");
    let output = dir.0.join("out");
    fs::create_dir(&input).unwrap();
    let bytes = b" \x28\xb5\x2f\xfd \x04\x22\x4d\x18 \x02\x21\x4c\x18 \x50\x2a\x4d\x18 \x78\x9c Yaz0 PK\x03\x04 PK\x01\x02 PK\x05\x06";
    fs::write(input.join("wrappers"), bytes).unwrap();
    let totals = scan(&input, &output, true, &mut Vec::new()).unwrap();
    assert_eq!(
        (totals.candidates, totals.unique, totals.translated),
        (0, 0, 0)
    );
    let entries = manifest(&output);
    let file = entries.iter().find(|e| e["kind"] == "file").unwrap();
    assert_eq!(
        file["signatures"].as_array().unwrap().len(),
        SIGNATURES.len()
    );
    assert!(file["signatures"]
        .as_array()
        .unwrap()
        .iter()
        .all(|s| s["count"] == 1));
    assert_eq!(fs::read_dir(&output).unwrap().count(), 1);
}

#[test]
fn accepted_but_unsupported_translation_is_counted_without_hiding_failure() {
    let dir = TestDir::new();
    let input = dir.0.join("input");
    let output = dir.0.join("out");
    fs::create_dir(&input).unwrap();
    // Geometry cannot be prewarmed without driver-supplied input topology.
    let program = record(headers::Stage::Geometry, &bundle([MOV, EXIT, NOP]));
    fs::write(input.join("geometry"), program).unwrap();
    let totals = scan(&input, &output, true, &mut Vec::new()).unwrap();
    assert_eq!(
        (
            totals.accepted,
            totals.translated,
            totals.translation_failed
        ),
        (1, 0, 1)
    );
    let entries = manifest(&output);
    let program = entries.iter().find(|e| e["kind"] == "program").unwrap();
    assert_eq!(program["translation"], "failed");
    assert!(program["translation_error"]
        .as_str()
        .unwrap()
        .contains("geometry"));
}

#[test]
fn sparse_large_offsets_and_large_streamed_file() {
    let dir = TestDir::new();
    let program = record(headers::Stage::Compute, &bundle([MOV, EXIT, NOP]));
    let sparse = dir.0.join("sparse");
    let mut writer = File::create(&sparse).unwrap();
    let offset = u64::from(u32::MAX) + 17;
    writer.seek(SeekFrom::Start(offset)).unwrap();
    writer.write_all(&program).unwrap();
    drop(writer);
    let found = validate(&mut File::open(sparse).unwrap(), offset, true).unwrap();
    assert_eq!(&found.bytes[HEADER_SIZE..], &program[0x100..]);
    let input = dir.0.join("input");
    fs::create_dir(&input).unwrap();
    let mut file = File::create(input.join("large")).unwrap();
    let padding = [0x55; CHUNK];
    for _ in 0..128 {
        file.write_all(&padding).unwrap();
    }
    file.write_all(&[0x55; 13]).unwrap();
    file.write_all(&program).unwrap();
    drop(file);
    let totals = scan(&input, &dir.0.join("out"), false, &mut Vec::new()).unwrap();
    assert_eq!((totals.candidates, totals.accepted), (1, 1));
}

#[cfg(unix)]
#[test]
fn symlinks_are_skipped_and_non_utf8_paths_are_lossless() {
    use std::os::unix::{ffi::OsStringExt, fs::symlink};
    let dir = TestDir::new();
    let input = dir.0.join("input");
    let output = dir.0.join("out");
    fs::create_dir(&input).unwrap();
    let name = std::ffi::OsString::from_vec(b"bad-\xff-name".to_vec());
    fs::write(
        input.join(name),
        record(headers::Stage::Vertex, &bundle([MOV, EXIT, NOP])),
    )
    .unwrap();
    symlink(&input, input.join("cycle")).unwrap();
    let totals = scan(&input, &output, false, &mut Vec::new()).unwrap();
    assert_eq!(
        (totals.files, totals.accepted, totals.skipped_symlinks),
        (1, 1, 1)
    );
    let entries = manifest(&output);
    assert_eq!(
        entries.iter().find(|e| e["kind"] == "program").unwrap()["source_path_hex"],
        "6261642dff2d6e616d65"
    );
}

#[test]
fn invalid_cli_arguments_and_output_containing_input_are_errors() {
    assert!(crate::parse_and_run::<AotScan, _>(shadowbox::QuietRun, &[]).is_err());
    assert!(crate::parse_and_run::<AotScan, _>(
        shadowbox::QuietRun,
        &["in".into(), "out".into(), "--typo".into()]
    )
    .is_err());
    let dir = TestDir::new();
    assert!(scan(&dir.0, &dir.0, false, &mut Vec::new()).is_err());
    let input = dir.0.join("input");
    fs::create_dir(&input).unwrap();
    assert!(scan(&input, &dir.0, false, &mut Vec::new()).is_err());
}

#[cfg(unix)]
#[test]
fn output_symlinks_do_not_redirect_writes_and_failed_translation_removes_stale_spirv() {
    use std::os::unix::fs::symlink;
    let dir = TestDir::new();
    let input = dir.0.join("input");
    let output = dir.0.join("out");
    fs::create_dir(&input).unwrap();
    fs::create_dir(&output).unwrap();
    let program = record(headers::Stage::Geometry, &bundle([MOV, EXIT, NOP]));
    let hash = format!("{:x}", Sha256::digest(&program[0x30..]));
    let sentinel = dir.0.join("untouched");
    fs::write(&sentinel, b"must stay unchanged").unwrap();
    for name in [
        "manifest.jsonl".into(),
        format!("{hash}.bin"),
        format!("{hash}.spv"),
    ] {
        symlink(&sentinel, output.join(name)).unwrap();
    }
    fs::write(input.join("program"), &program).unwrap();
    let totals = scan(&input, &output, true, &mut Vec::new()).unwrap();
    assert_eq!((totals.accepted, totals.translation_failed), (1, 1));
    assert_eq!(fs::read(sentinel).unwrap(), b"must stay unchanged");
    assert_eq!(
        fs::read(output.join(format!("{hash}.bin"))).unwrap(),
        &program[0x30..]
    );
    assert!(!output.join(format!("{hash}.spv")).exists());
    assert_eq!(manifest(&output).last().unwrap()["kind"], "summary");
    assert_eq!(fs::read_dir(output).unwrap().count(), 2);
}

#[test]
fn nearby_signature_window_includes_boundaries_but_excludes_lookahead() {
    let dir = TestDir::new();
    let offset = 8192u64;
    let mut bytes = vec![0x55; 16384];
    insert(
        &mut bytes,
        (offset - CONTEXT) as usize,
        &[0x28, 0xb5, 0x2f, 0xfd],
    );
    insert(&mut bytes, (offset + CONTEXT) as usize, &[0x78, 0x9c]);
    insert(
        &mut bytes,
        (offset + CONTEXT + 2) as usize,
        &[4, 0x22, 0x4d, 0x18],
    );
    let path = dir.0.join("context");
    fs::write(&path, bytes).unwrap();
    let found = nearby(&mut File::open(path).unwrap(), offset).unwrap();
    assert_eq!(
        found,
        vec![
            json!({"signature": "zstd", "offset": offset - CONTEXT}),
            json!({"signature": "zlib", "offset": offset + CONTEXT}),
        ]
    );
}

#[test]
fn pak_nested_records_names_content_and_entry_reports() {
    use crate::pak::tests::{build, Codec, Options};
    let dir = TestDir::new();
    let input = dir.0.join("input");
    let output = dir.0.join("output");
    fs::create_dir_all(&input).unwrap();
    let graphics = record(headers::Stage::Fragment, &bundle([MOV, EXIT, NOP]));
    let compute = record(headers::Stage::Compute, &bundle([MOV, EXIT, NOP]));
    let content = [noise(96), graphics.clone(), noise(CHUNK), compute.clone()].concat();
    for (version, codec) in [
        (3, Codec::Zlib),
        (5, Codec::Zlib),
        (8, Codec::Zstd),
        (9, Codec::None),
        (10, Codec::Zlib),
        (11, Codec::Zstd),
    ] {
        fs::write(
            input.join(format!("synthetic-{version}.pak")),
            build(
                version,
                codec,
                &content,
                Options {
                    raw_shader_name: true,
                    ..Options::default()
                },
            ),
        )
        .unwrap();
    }
    let mut report = Vec::new();
    let totals = scan(&input, &output, true, &mut report).unwrap();
    assert_eq!(
        (
            totals.files,
            totals.accepted,
            totals.rejected,
            totals.unique,
            totals.translated
        ),
        (6, 24, 0, 2, 2)
    );
    let events = manifest(&output);
    let programs: Vec<_> = events.iter().filter(|e| e["kind"] == "program").collect();
    for e in programs {
        assert!(e["source_file"].as_str().unwrap().ends_with(".pak"));
        assert!(e["pak_entry"].as_str().is_some());
        assert_eq!(e["integrity"], "verified");
        assert!(e["offset"] == 96 || e["offset"] == 96 + graphics.len() + CHUNK);
    }
    assert_eq!(
        events
            .iter()
            .filter(|e| e["kind"] == "pak_entry_scan")
            .count(),
        12
    );
    assert!(!fs::read_dir(&output).unwrap().any(|p| p
        .unwrap()
        .path()
        .extension()
        .is_some_and(|e| e == "partial")));
}

#[test]
fn pak_all_encryption_and_codec_skips_are_reported() {
    use crate::pak::tests::{build, Codec, Options};
    let dir = TestDir::new();
    let input = dir.0.join("input");
    let output = dir.0.join("output");
    fs::create_dir_all(&input).unwrap();
    for (name, codec, options) in [
        ("plain", Codec::None, Options::default()),
        ("oodle", Codec::Oodle, Options::default()),
        (
            "encrypted-entry",
            Codec::Zstd,
            Options {
                encrypted_entry: true,
                ..Options::default()
            },
        ),
        (
            "encrypted-index",
            Codec::None,
            Options {
                encrypted_index: true,
                ..Options::default()
            },
        ),
    ] {
        fs::write(
            input.join(format!("{name}.pak")),
            build(11, codec, b"no shader records", options),
        )
        .unwrap();
    }
    let counts = scan(&input, &output, false, &mut Vec::new()).unwrap();
    assert_eq!(counts.files, 4);
    let events = manifest(&output);
    assert_eq!(
        events
            .iter()
            .filter(|e| e["kind"] == "pak_entry_scan")
            .count(),
        1
    ); // The invalid named library falls back even without --pak-all.
    assert_eq!(counts.accepted, 0); // Synthetic payload contains no records.
    assert!(events
        .iter()
        .any(|e| e["kind"] == "ue_library" && e["status"] == "raw-scanned"));
    assert_eq!(
        events.iter().filter(|e| e["status"] == "skipped").count(),
        4
    );
    assert_eq!(
        events.iter().filter(|e| e["kind"] == "pak_skipped").count(),
        1
    );
    scan_options(&input, &output, false, true, None, &mut Vec::new()).unwrap();
    assert_eq!(
        manifest(&output)
            .iter()
            .filter(|e| e["kind"] == "pak_entry_scan")
            .count(),
        2
    );
}

#[test]
fn unity_shaders_in_serialized_files_and_bundles_are_scanned_and_reported() {
    use crate::unity::tests::{bundle as unity_bundle, serialized};
    let dir = TestDir::new();
    let input = dir.0.join("input");
    let output = input.join("output");
    fs::create_dir_all(&input).unwrap();
    let graphics = record(headers::Stage::Fragment, &bundle([MOV, EXIT, NOP]));
    let compute = record(headers::Stage::Compute, &bundle([MOV, EXIT, NOP]));
    let mut unknown = graphics.clone();
    unknown[0x88..0x90].fill(0xff);
    let payloads = vec![[noise(13), graphics, compute, unknown].concat(), noise(39)];
    // Standalone name discovery, header discovery, and bundle signature
    // discovery all lead to exactly the same strict blob record scanner.
    for name in [
        "sharedassets0.assets",
        "globalgamemanagers",
        "level0",
        "arbitrary.dat",
    ] {
        fs::write(input.join(name), serialized(17, true, false, &payloads)).unwrap();
    }
    fs::write(
        input.join("synthetic.bundle"),
        unity_bundle(
            7,
            true,
            true,
            1,
            3,
            &[
                (
                    "../../CAB-synthetic",
                    serialized(22, false, true, &payloads),
                ),
                ("resource.resS", vec![0; 27]),
            ],
        ),
    )
    .unwrap();
    let mut report = Vec::new();
    let counts = scan(&input, &output, true, &mut report).unwrap();
    assert_eq!(
        (
            counts.files,
            counts.accepted,
            counts.rejected,
            counts.unique,
            counts.translated
        ),
        (5, 10, 5, 2, 2)
    );
    let events = manifest(&output);
    let shaders: Vec<_> = events
        .iter()
        .filter(|e| e["kind"] == "unity_shader")
        .collect();
    assert_eq!(shaders.len(), 10);
    assert_eq!(
        shaders
            .iter()
            .filter(|e| e["accepted"] == 2 && e["rejected"] == 1)
            .count(),
        5
    );
    assert_eq!(shaders.iter().filter(|e| e["candidates"] == 0).count(), 5);
    let programs: Vec<_> = events.iter().filter(|e| e["kind"] == "program").collect();
    assert_eq!(programs.len(), 10);
    for program in programs {
        assert_eq!(program["unity"]["path_id"], 100);
        assert_eq!(program["unity"]["shader_name"], "../synthetic/shader-0");
        assert_eq!(program["unity"]["platform"], 7);
        assert_eq!(program["unity"]["segment"], 0);
        assert!(program["offset"].as_u64().unwrap() >= 13);
    }
    assert_eq!(
        events
            .iter()
            .filter(|e| e["kind"] == "unity_blob_scan")
            .count(),
        22
    );
    assert!(events
        .iter()
        .filter(|e| e["kind"] == "rejected")
        .all(|e| e["reason"]
            .as_str()
            .unwrap()
            .contains("unknown instruction")));
    let text = String::from_utf8(report).unwrap();
    assert!(text.contains("shader[100]"));
    assert!(text.contains("shader[101]"));
    let mut second_report = Vec::new();
    let second = scan(&input, &output, false, &mut second_report).unwrap();
    assert_eq!((second.files, second.accepted, second.rejected), (5, 10, 5));
    assert!(!fs::read_dir(&output).unwrap().any(|p| p
        .unwrap()
        .path()
        .extension()
        .is_some_and(|e| e == "partial")));
}

#[test]
fn unity_unknown_versions_and_unsupported_stripped_layouts_are_reported_without_raw_fallback() {
    use crate::unity::tests::{bundle as unity_bundle, serialized, stripped};
    let dir = TestDir::new();
    let input = dir.0.join("input");
    let output = dir.0.join("output");
    fs::create_dir_all(&input).unwrap();
    let payload = record(headers::Stage::Fragment, &bundle([MOV, EXIT, NOP]));
    let original = serialized(17, true, false, std::slice::from_ref(&payload));
    let mut unknown = original.clone();
    unknown[8..12].copy_from_slice(&27u32.to_be_bytes());
    let missing = stripped("6000.0.0f1", true, &[payload]);
    fs::write(input.join("unknown.assets"), &unknown).unwrap();
    fs::write(input.join("level1"), &missing).unwrap();
    fs::write(
        input.join("synthetic.bundle"),
        unity_bundle(
            8,
            false,
            true,
            2,
            1,
            &[("unknown.assets", unknown), ("CAB-missing-tree", missing)],
        ),
    )
    .unwrap();
    let counts = scan(&input, &output, false, &mut Vec::new()).unwrap();
    assert_eq!(
        (counts.files, counts.candidates, counts.accepted),
        (3, 0, 0)
    );
    let events = manifest(&output);
    let skipped: Vec<_> = events
        .iter()
        .filter(|e| e["kind"] == "unity_skipped")
        .collect();
    assert_eq!(skipped.len(), 2);
    assert_eq!(
        skipped
            .iter()
            .filter(|e| e["reason"]
                .as_str()
                .unwrap()
                .contains("unsupported Unity serialized version 27"))
            .count(),
        2
    );
    assert_eq!(
        events
            .iter()
            .filter(|e| e["kind"] == "unity_shader"
                && e["status"] == "skipped"
                && e["reason"]
                    .as_str()
                    .unwrap()
                    .contains("unsupported Unity Shader layout version"))
            .count(),
        2
    );
}

#[test]
fn unity_skips_one_bad_shader_and_detects_unnamed_future_serialized_files() {
    use crate::unity::tests::serialized;
    let dir = TestDir::new();
    let input = dir.0.join("input");
    let output = dir.0.join("output");
    fs::create_dir_all(&input).unwrap();
    let payload = record(headers::Stage::Fragment, &bundle([MOV, EXIT, NOP]));
    let mut bytes = serialized(17, true, false, &[payload.clone(), payload.clone()]);
    let object_start = u32::from_be_bytes(bytes[12..16].try_into().unwrap()) as usize;
    bytes[object_start..object_start + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    fs::write(input.join("shaders.assets"), bytes).unwrap();
    for version in [17, 22] {
        let mut bytes = serialized(version, true, false, std::slice::from_ref(&payload));
        bytes[8..12].copy_from_slice(&27u32.to_be_bytes());
        fs::write(input.join(format!("future-{version}.dat")), bytes).unwrap();
    }
    let counts = scan(&input, &output, false, &mut Vec::new()).unwrap();
    assert_eq!(
        (counts.files, counts.candidates, counts.accepted),
        (3, 1, 1)
    );
    let events = manifest(&output);
    assert_eq!(
        events
            .iter()
            .filter(|e| e["kind"] == "unity_skipped")
            .count(),
        2
    );
    assert!(events
        .iter()
        .any(|e| e["kind"] == "unity_shader" && e["path_id"] == 100 && e["status"] == "skipped"));
    assert!(events
        .iter()
        .any(|e| e["kind"] == "unity_shader" && e["path_id"] == 101 && e["accepted"] == 1));
}

#[test]
fn unity_stripped_players_scan_blobs_and_keep_unknown_instruction_rejections() {
    use crate::unity::tests::{bundle as unity_bundle, stripped};
    let dir = TestDir::new();
    let input = dir.0.join("input");
    let output = dir.0.join("output");
    fs::create_dir_all(&input).unwrap();
    let graphics = record(headers::Stage::Fragment, &bundle([MOV, EXIT, NOP]));
    let mut unknown = graphics.clone();
    unknown[0x88..0x90].fill(0xff);
    let payload = [noise(13), graphics, unknown].concat();
    let mut bytes = stripped("2017.4.9f1", true, &[payload.clone(), payload.clone()]);
    // Last three bytes belong to a non-Shader object. The last Shader's baked
    // boolean is four bytes before that (Shader.cs:1015-1016). Invalid boolean
    // skips only that object and exposes the parser position in both reports.
    let baked = bytes.len() - 7;
    bytes[baked] = 2;
    fs::write(input.join("sharedassets0.assets"), bytes).unwrap();
    fs::write(
        input.join("synthetic.bundle"),
        unity_bundle(
            7,
            false,
            true,
            2,
            3,
            &[("CAB-stripped", stripped("2019.4.1f1", false, &[payload]))],
        ),
    )
    .unwrap();
    let mut report = Vec::new();
    let counts = scan(&input, &output, true, &mut report).unwrap();
    assert_eq!(
        (
            counts.files,
            counts.accepted,
            counts.rejected,
            counts.unique,
            counts.translated
        ),
        (2, 2, 2, 1, 1)
    );
    let events = manifest(&output);
    let shaders: Vec<_> = events
        .iter()
        .filter(|e| e["kind"] == "unity_shader")
        .collect();
    assert_eq!(shaders.len(), 3);
    assert_eq!(
        shaders.iter().filter(|e| e["status"] == "scanned").count(),
        2
    );
    let skipped = shaders.iter().find(|e| e["status"] == "skipped").unwrap();
    assert_eq!(skipped["path_id"], 101);
    assert!(skipped["reason"].as_str().unwrap().contains("object byte"));
    assert!(skipped["reason"]
        .as_str()
        .unwrap()
        .contains("serialized byte"));
    assert!(events
        .iter()
        .filter(|e| e["kind"] == "rejected")
        .all(|e| e["reason"]
            .as_str()
            .unwrap()
            .contains("unknown instruction")));
    assert!(events
        .iter()
        .filter(|e| e["kind"] == "program")
        .all(|e| e["unity"]["platform"] == 19
            && e["unity"]["compressed_blob_offset"] == 3
            && e["offset"] == 13));
    let report = String::from_utf8(report).unwrap();
    assert!(report.contains("object byte"));
    assert!(report.contains("serialized byte"));
    assert!(!events.iter().any(|e| e["kind"] == "unity_skipped"));
}

#[test]
fn nw_nested_compression_dictionary_pack_variations_and_translation() {
    use crate::nw::tests::{
        binary_file, bnsh_file, compress_with_dict, sarc_file, trained_dictionary, yaz0_literals,
    };
    let dir = TestDir::new();
    let input = dir.0.join("input");
    let output = input.join("output");
    fs::create_dir_all(&input).unwrap();
    let vertex = record(
        headers::Stage::Vertex,
        &[bundle([MOV, EXIT, NOP]), bundle([MOV, NOP, MOV])].concat(),
    );
    let fragment = record(headers::Stage::Fragment, &bundle([MOV, EXIT, NOP]));
    let compute = record(headers::Stage::Compute, &bundle([MOV, EXIT, NOP]));
    let shader = bnsh_file(&[
        vec![
            (headers::Stage::Vertex, vertex.clone()),
            (headers::Stage::Fragment, fragment.clone()),
        ],
        vec![(headers::Stage::Compute, compute.clone())],
    ]);
    let archive = binary_file(b"FSHA", 0x0008_0000, &shader);
    let model = binary_file(b"FRES", 0x000a_0000, &archive);
    let dictionary = trained_dictionary(42);
    let packed = sarc_file(
        &[(
            "../../must-not-extract/shaders.bfres.zs",
            compress_with_dict(&model, &dictionary),
        )],
        false,
    );
    let nested = zstd::stream::encode_all(&packed[..], 1).unwrap();
    fs::write(input.join("a-shaders.pack.zs"), nested).unwrap();
    // Dictionary arrives after the dependent file and inside a compressed pack.
    fs::write(
        input.join("ZsDic.pack"),
        yaz0_literals(&sarc_file(&[("shaders.zsdic", dictionary)], true)),
    )
    .unwrap();
    let mut report = Vec::new();
    let totals = scan(&input, &output, true, &mut report).unwrap();
    assert_eq!(
        (
            totals.files,
            totals.accepted,
            totals.rejected,
            totals.unique,
            totals.translated,
            totals.nw_variations,
            totals.nw_stage_failed
        ),
        (2, 3, 0, 3, 3, 2, 0)
    );
    let events = manifest(&output);
    let programs: Vec<_> = events.iter().filter(|e| e["kind"] == "program").collect();
    assert_eq!(programs.len(), 3);
    for p in programs {
        let bytes = fs::read(output.join(p["output"].as_str().unwrap())).unwrap();
        let expected = match p["stage"].as_str().unwrap() {
            "vertex" => vertex[0x30..].to_vec(),
            "fragment" => fragment[0x30..].to_vec(),
            "compute" => [vec![0; HEADER_SIZE], compute[0x100..].to_vec()].concat(),
            _ => panic!("unexpected stage"),
        };
        assert_eq!(bytes, expected); // Real MOVs after EXIT are retained.
        assert!(p["nw"]["container_path"].as_array().unwrap().len() >= 7);
        assert_eq!(
            fs::read(output.join(p["nw"]["control_output"].as_str().unwrap())).unwrap(),
            b"opaque-control"
        );
        assert_eq!(
            fs::read(output.join(p["nw"]["code_output"].as_str().unwrap()))
                .unwrap()
                .len(),
            p["nw"]["code_size"].as_u64().unwrap() as usize
        );
    }
    assert!(events.iter().any(
        |e| e["kind"] == "nw_compression" && !e["dictionaries"].as_array().unwrap().is_empty()
    ));
    assert_eq!(
        events
            .iter()
            .filter(|e| e["kind"] == "nw_variation")
            .count(),
        2
    );
    assert!(String::from_utf8(report).unwrap().contains("variation[1]"));
    assert!(!input.join("must-not-extract").exists());
    // Repeat scans exclude generated dictionary/artifact metadata.
    let again = scan(&input, &output, false, &mut Vec::new()).unwrap();
    assert_eq!(
        (again.accepted, again.unique, again.nw_variations),
        (3, 3, 2)
    );
}

#[test]
fn nw_stage_mismatch_unknown_trailing_code_and_unsupported_versions_stay_failed() {
    use crate::nw::tests::{binary_file, bnsh_file, sarc_file};
    let dir = TestDir::new();
    let input = dir.0.join("input");
    let output = dir.0.join("output");
    fs::create_dir_all(&input).unwrap();
    let valid = record(headers::Stage::Vertex, &bundle([MOV, EXIT, NOP]));
    let mut unknown = [valid.clone(), bundle([MOV, EXIT, NOP])].concat();
    unknown[valid.len() + 8..valid.len() + 16].fill(0xff);
    fs::write(
        input.join("stages.bnsh"),
        bnsh_file(&[
            vec![(headers::Stage::Vertex, unknown)],
            vec![(headers::Stage::Fragment, valid.clone())],
            vec![(headers::Stage::Compute, b"unrecognized bytecode".to_vec())],
            vec![(headers::Stage::Vertex, valid.clone())],
        ]),
    )
    .unwrap();
    let mut future = bnsh_file(&[vec![(headers::Stage::Vertex, valid.clone())]]);
    future[8] = 6;
    fs::write(input.join("future.bnsh"), future).unwrap();
    fs::write(
        input.join("future.bfsha"),
        binary_file(
            b"FSHA",
            0x000b_0000,
            &bnsh_file(&[vec![(headers::Stage::Vertex, valid)]]),
        ),
    )
    .unwrap();
    let mut unsupported = sarc_file(&[("hidden.bnsh", bnsh_file(&[]))], false);
    unsupported[16] = 1;
    unsupported[17] = 2;
    fs::write(input.join("future.sarc"), unsupported).unwrap();
    let totals = scan(&input, &output, true, &mut Vec::new()).unwrap();
    assert_eq!(
        (
            totals.accepted,
            totals.rejected,
            totals.nw_variations,
            totals.nw_stage_failed
        ),
        (1, 2, 4, 3)
    );
    let events = manifest(&output);
    assert!(events.iter().any(|e| e["kind"] == "rejected"
        && e["reason"]
            .as_str()
            .unwrap()
            .contains("unknown instruction")));
    assert!(events
        .iter()
        .any(|e| e["kind"] == "rejected"
            && e["reason"].as_str().unwrap().contains("stage disagrees")));
    assert_eq!(
        events
            .iter()
            .filter(|e| e["kind"] == "nw_stage" && e["status"] == "failed")
            .count(),
        3
    );
    assert_eq!(
        events
            .iter()
            .filter(
                |e| e["kind"] == "nw_skipped" && e["reason"].as_str().unwrap().contains("version")
            )
            .count(),
        3
    );
    // Failed stages keep their exact bytecode, including the unknown encoding.
    for e in events.iter().filter(|e| e["kind"] == "nw_stage") {
        assert!(output
            .join(e["nw"]["code_output"].as_str().unwrap())
            .exists());
    }
}

#[test]
fn nw_nested_work_limit_and_missing_dictionary_are_reported() {
    use crate::nw::tests::{bnsh_file, compress_with_dict, trained_dictionary, yaz0_literals};
    let dir = TestDir::new();
    let input = dir.0.join("input");
    let output = dir.0.join("output");
    fs::create_dir_all(&input).unwrap();
    let mut nested = bnsh_file(&[]);
    for _ in 0..crate::nw::MAX_DEPTH + 1 {
        nested = yaz0_literals(&nested);
    }
    fs::write(input.join("nested.szs"), nested).unwrap();
    fs::write(
        input.join("missing.zs"),
        compress_with_dict(&bnsh_file(&[]), &trained_dictionary(3)),
    )
    .unwrap();
    let totals = scan(&input, &output, false, &mut Vec::new()).unwrap();
    assert_eq!(totals.nw_variations, 0);
    let events = manifest(&output);
    assert!(events.iter().any(|e| e["kind"] == "nw_skipped"
        && e["reason"]
            .as_str()
            .unwrap()
            .contains("recursion/work limit")));
    assert!(events.iter().any(|e| e["kind"] == "nw_skipped"
        && e["reason"]
            .as_str()
            .unwrap()
            .contains("missing zstd dictionary")));
}

#[test]
fn nw_bad_variation_does_not_hide_siblings_and_interior_magic_is_not_a_stage() {
    use crate::nw::tests::bnsh_file;
    let dir = TestDir::new();
    let input = dir.0.join("input");
    let output = dir.0.join("output");
    fs::create_dir_all(&input).unwrap();
    // Mesa sm50.rs:1938-1941,1951: MOV32I immediate at 20..52. It contains
    // the record magic as raw bytes at an odd offset inside a real instruction.
    // Preserve it; a declared stage is one program, not a byte-signature bag.
    let magic_immediate = (MOV & !(0xffff_ffff << 20)) | (0x1234_5678u64 << 20);
    let code = record(
        headers::Stage::Vertex,
        &bundle([magic_immediate, EXIT, NOP]),
    );
    let mut bytes = bnsh_file(&[
        vec![(headers::Stage::Vertex, code.clone())],
        vec![(headers::Stage::Vertex, code.clone())],
    ]);
    bytes[0xc0 + 16..0xc0 + 24].copy_from_slice(&u64::MAX.to_le_bytes());
    fs::write(input.join("bad-variation.bnsh"), bytes).unwrap();
    fs::write(
        input.join("unsupported-leading-layout.bnsh"),
        bnsh_file(&[vec![(
            headers::Stage::Vertex,
            [bundle([MOV, NOP, MOV]), code.clone()].concat(),
        )]]),
    )
    .unwrap();
    let totals = scan(&input, &output, true, &mut Vec::new()).unwrap();
    assert_eq!(
        (
            totals.accepted,
            totals.rejected,
            totals.nw_variations,
            totals.nw_stage_failed,
            totals.nw_variation_skipped
        ),
        (1, 0, 3, 1, 1)
    );
    let entries = manifest(&output);
    let program = entries.iter().find(|e| e["kind"] == "program").unwrap();
    assert_eq!(
        fs::read(output.join(program["output"].as_str().unwrap())).unwrap(),
        code[0x30..]
    );
    assert!(entries
        .iter()
        .any(|e| e["kind"] == "nw_container" && e["skipped_variations"] == 1));
}

#[test]
fn encrypted_pak_scan_validates_key_before_reporting_names() {
    use crate::pak::{
        tests::{build, random_key, Codec, Options},
        AesKey,
    };
    let key = random_key();
    let hex: String = key.iter().map(|b| format!("{b:02x}")).collect();
    let dir = TestDir::new();
    let input = dir.0.join("input");
    fs::create_dir_all(&input).unwrap();
    let program = record(headers::Stage::Fragment, &bundle([MOV, EXIT, NOP]));
    fs::write(
        input.join("synthetic.pak"),
        build(
            11,
            Codec::Zlib,
            &program,
            Options {
                key: Some(key),
                raw_shader_name: true,
                encrypted_index: true,
                encrypted_entry: true,
                ..Options::default()
            },
        ),
    )
    .unwrap();
    let key_file = dir.0.join("key");
    fs::write(&key_file, format!(" 0x{hex}\n")).unwrap();
    let options = parse_flags(&[
        "--translate".into(),
        "--emit-cache".into(),
        dir.0.join("runtime-cache").to_str().unwrap().into(),
        "--pak-aes-key-file".into(),
        key_file.to_str().unwrap().into(),
    ])
    .unwrap();
    let output = dir.0.join("output");
    let mut report = Vec::new();
    let totals =
        scan_with_state(shadowbox::QuietRun, &input, &output, options, &mut report).unwrap();
    assert_eq!(totals.cache_emitted, 1);
    assert_eq!(
        (totals.pak_entries_verified, totals.pak_entries_unverified),
        (0, 2)
    );
    for event in manifest(&output)
        .iter()
        .filter(|e| !e["pak_entry"].is_null())
    {
        assert_eq!(event["integrity"], "unverified");
    }
    assert_eq!(
        (totals.accepted, totals.translated, totals.unique),
        (2, 1, 1)
    );
    assert!(!String::from_utf8(report).unwrap().contains(&hex));
    assert!(!fs::read_to_string(output.join("manifest.jsonl"))
        .unwrap()
        .contains(&hex));
    let wrong = random_key();
    let wrong: String = wrong.iter().map(|b| format!("{b:02x}")).collect();
    let output = dir.0.join("wrong-output");
    scan_options(
        &input,
        &output,
        false,
        false,
        Some(AesKey::parse(wrong.as_bytes()).unwrap()),
        &mut Vec::new(),
    )
    .unwrap();
    let events = manifest(&output);
    assert!(events.iter().any(|e| e["kind"] == "pak_skipped"
        && e["reason"]
            .as_str()
            .is_some_and(|s| s.starts_with("malformed index:"))));
    assert!(!events
        .iter()
        .any(|e| e["kind"] == "pak" || e["kind"] == "pak_entry" || e["kind"] == "program"));
}

#[test]
fn pak_key_flags_reject_missing_conflicting_and_invalid_sources() {
    let dir = TestDir::new();
    let source = dir.0.join("invalid-key");
    fs::write(&source, b"private-invalid-value").unwrap();
    for flags in [
        vec!["--pak-aes-key-file"],
        vec!["--pak-aes-key-env"],
        vec!["--pak-aes-key-env", "SHADOWBOX_AES_MISSING_TEST"],
        vec!["--pak-aes-key-file", "nonexistent"],
        vec!["--pak-aes-key-file", source.to_str().unwrap()],
        vec![
            "--pak-aes-key-file",
            source.to_str().unwrap(),
            "--pak-aes-key-env",
            "VAR",
        ],
        vec!["--pak-aes-key-env", "VAR", "--pak-aes-key-env", "VAR"],
    ] {
        let error = parse_flags(&flags.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>())
            .err()
            .unwrap();
        assert!(!error.contains("private-invalid-value"));
        assert!(!error.contains(source.to_str().unwrap()));
    }
    fs::write(&source, vec![b' '; 1025]).unwrap();
    assert!(parse_flags(&["--pak-aes-key-file".into(), source.to_str().unwrap().into()]).is_err());
}

// Child process keeps environment-key tests isolated from parallel tests.
#[test]
fn pak_key_environment_child() {
    let Ok(value) = std::env::var("SHADOWBOX_AES_CHILD_TEST") else {
        return;
    };
    let parsed = parse_flags(&[
        "--pak-aes-key-env".into(),
        "SHADOWBOX_AES_CHILD_TEST".into(),
    ]);
    if value == "invalid-private-value" {
        let error = parsed.err().unwrap();
        assert!(!error.contains(&value));
    } else {
        assert!(parsed.unwrap().pak_key.is_some());
    }
}

#[test]
fn pak_key_environment_source_and_errors_are_private() {
    let key = crate::pak::tests::random_key();
    let hex: String = key.iter().map(|b| format!("{b:02X}")).collect();
    for value in [format!("0x{hex}"), "invalid-private-value".to_owned()] {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "aot_scan::tests::pak_key_environment_child"])
            .env("SHADOWBOX_AES_CHILD_TEST", &value)
            .output()
            .unwrap();
        assert!(result.status.success());
        assert!(!String::from_utf8_lossy(&result.stdout).contains(&value));
        assert!(!String::from_utf8_lossy(&result.stderr).contains(&value));
    }
}

#[test]
fn prewarm_package_command_forwards_aes_key_to_scanner() {
    use crate::pak::tests::{build, random_key, Codec, Options};
    let key = random_key();
    let hex: String = key.iter().map(|b| format!("{b:02x}")).collect();
    let dir = TestDir::new();
    let input = dir.0.join("romfs");
    fs::create_dir(&input).unwrap();
    let program = record(headers::Stage::Fragment, &bundle([MOV, EXIT, NOP]));
    fs::write(
        input.join("synthetic.pak"),
        build(
            11,
            Codec::Zlib,
            &program,
            Options {
                key: Some(key),
                raw_shader_name: true,
                encrypted_index: true,
                encrypted_entry: true,
                ..Options::default()
            },
        ),
    )
    .unwrap();
    let key_file = dir.0.join("key");
    fs::write(&key_file, &hex).unwrap();
    let output = dir.0.join("package");
    crate::parse_and_run::<crate::prewarm_pack::PrewarmPack, _>(
        shadowbox::QuietRun,
        &[
            output.to_str().unwrap().into(),
            "--romfs".into(),
            input.to_str().unwrap().into(),
            "--pak-aes-key-file".into(),
            key_file.to_str().unwrap().into(),
        ],
    )
    .unwrap();
    let text = fs::read_to_string(output.join("manifest.json")).unwrap();
    let manifest: Value = serde_json::from_str(&text).unwrap();
    assert_eq!(manifest["counts"]["programs_found_by_scan"], 1);
    assert_eq!(manifest["counts"]["translated_complete"], 1);
    assert_eq!(manifest["runtime_cache_entries"], 1);
    assert!(!text.contains(&hex));
}

#[path = "iostore_tests.rs"]
mod iostore_tests;

#[test]
fn nw_admitted_revisions_and_hash_only_archives_extract_exact_stages() {
    use crate::nw::tests::{binary_file, bnsh_file, hash_only_sarc, WRAPPER_REVISIONS};
    let dir = TestDir::new();
    let input = dir.0.join("input");
    let output = dir.0.join("output");
    fs::create_dir_all(&input).unwrap();
    let vertex = record(
        headers::Stage::Vertex,
        &[bundle([MOV, EXIT, NOP]), bundle([MOV, NOP, MOV])].concat(),
    );
    let compute = record(headers::Stage::Compute, &bundle([MOV, EXIT, NOP]));
    let shader = bnsh_file(&[vec![
        (headers::Stage::Vertex, vertex.clone()),
        (headers::Stage::Compute, compute.clone()),
    ]]);
    for micro in 0..=5 {
        let mut bytes = shader.clone();
        bytes[8] = micro;
        fs::write(input.join(format!("revision-2.1.{micro}")), bytes).unwrap();
    }
    for &(format, magic, version) in WRAPPER_REVISIONS {
        fs::write(
            input.join(format!("{format}-{version:08x}")),
            binary_file(magic, version, &shader),
        )
        .unwrap();
    }
    for be in [false, true] {
        // BNTX and BNSH are discovered from bytes, with no archive filename.
        fs::write(
            input.join(format!("hash-only-{be}")),
            hash_only_sarc(&[binary_file(b"BNTX", 0x0004_0100, &shader)], be),
        )
        .unwrap();
    }
    let totals = scan(&input, &output, true, &mut Vec::new()).unwrap();
    let occurrences = 6 + WRAPPER_REVISIONS.len() as u64 + 2;
    assert_eq!(totals.accepted, 2 * occurrences);
    assert_eq!(totals.nw_variations, occurrences);
    assert_eq!((totals.unique, totals.translated), (2, 2));
    assert_eq!(
        (totals.rejected, totals.nw_stage_failed, totals.nw_skipped),
        (0, 0, 0)
    );
    let events = manifest(&output);
    for program in events.iter().filter(|e| e["kind"] == "program") {
        let code = if program["stage"] == "vertex" {
            &vertex
        } else {
            &compute
        };
        assert_eq!(
            fs::read(output.join(program["nw"]["code_output"].as_str().unwrap())).unwrap(),
            *code
        );
        assert_eq!(
            fs::read(output.join(program["nw"]["control_output"].as_str().unwrap())).unwrap(),
            b"opaque-control"
        );
    }
    let anonymous: Vec<_> = events
        .iter()
        .filter(|e| e["kind"] == "nw_sarc_entry")
        .collect();
    assert_eq!(anonymous.len(), 2);
    for entry in anonymous {
        let member = entry["container_path"].as_array().unwrap().last().unwrap();
        assert_eq!(member["entry"], "#12340000");
        assert_eq!(member["hash"], 0x1234_0000);
        assert_eq!(member["name_hex"], Value::Null);
    }
}

#[test]
fn nw_unknown_binary_children_are_boundaries_and_trailing_bytes_are_excluded() {
    use crate::nw::tests::{binary_file, bnsh_file};
    let dir = TestDir::new();
    let input = dir.0.join("input");
    let output = dir.0.join("output");
    fs::create_dir_all(&input).unwrap();
    let valid = record(headers::Stage::Vertex, &bundle([MOV, EXIT, NOP]));
    let shader = bnsh_file(&[vec![(headers::Stage::Vertex, valid.clone())]]);
    // Every recognized child is a boundary. An unknown wrapper cannot be
    // bypassed by the enclosing wrapper's search for a known shader signature.
    let unknowns: [(&[u8; 4], u32); 4] = [
        (b"BNTX", 0x0004_0200),
        (b"FSHA", 0x0009_0100),
        (b"FRES", 0x000b_0000),
        (b"BNSH", 0x0002_0106),
    ];
    for (i, (magic, version)) in unknowns.into_iter().enumerate() {
        let unknown = if magic == b"BNSH" {
            let mut unknown = shader.clone();
            unknown[8..12].copy_from_slice(&version.to_le_bytes());
            unknown
        } else {
            binary_file(magic, version, &shader)
        };
        let payload = [unknown, shader.clone()].concat();
        let mut outer = binary_file(b"FRES", 0x0008_0000, &payload);
        outer.extend_from_slice(&shader); // Outside declared extent.
        fs::write(input.join(format!("nested-{i}")), outer).unwrap();
    }
    // Unknown instructions in every newly admitted BNSH revision still fail.
    for micro in 0..5 {
        let mut code = valid.clone();
        code[0x88..0x90].fill(0xff);
        let mut bytes = bnsh_file(&[vec![(headers::Stage::Vertex, code)]]);
        bytes[8] = micro;
        fs::write(input.join(format!("unknown-instruction-{micro}")), bytes).unwrap();
    }
    let totals = scan(&input, &output, true, &mut Vec::new()).unwrap();
    assert_eq!(
        (totals.accepted, totals.rejected, totals.nw_skipped),
        (4, 5, 4)
    );
    assert_eq!(totals.nw_stage_failed, 5);
    let events = manifest(&output);
    let skips: Vec<_> = events
        .iter()
        .filter(|e| e["kind"] == "nw_skipped")
        .collect();
    assert_eq!(skips.len(), 4);
    for skip in skips {
        assert!(skip["reason"].as_str().unwrap().contains("version"));
    }
    for rejected in events.iter().filter(|e| e["kind"] == "rejected") {
        assert!(rejected["reason"]
            .as_str()
            .unwrap()
            .contains("unknown instruction"));
    }
}

#[test]
fn unity_modern_trees_and_2021_players_preserve_instruction_rejections() {
    use crate::unity::tests::{bundle as unity_bundle, modern_serialized, stripped};
    let dir = TestDir::new();
    let input = dir.0.join("input");
    let output = dir.0.join("output");
    fs::create_dir_all(&input).unwrap();
    let graphics = record(headers::Stage::Fragment, &bundle([MOV, EXIT, NOP]));
    let mut unknown = graphics.clone();
    unknown[0x88..0x90].fill(0xff);
    let payload = [noise(13), graphics, unknown].concat();
    for unity in ["2021.3.2f1", "2022.1.1f1"] {
        fs::write(
            input.join(format!("{unity}.assets")),
            stripped(unity, true, std::slice::from_ref(&payload)),
        )
        .unwrap();
    }
    let files = [
        (
            "CAB-2021",
            modern_serialized(22, true, "2021.3.2f1", std::slice::from_ref(&payload)),
        ),
        (
            "CAB-2022",
            modern_serialized(23, false, "2022.3.0f1", std::slice::from_ref(&payload)),
        ),
        (
            "CAB-2023",
            modern_serialized(25, true, "2023.2.0f1", std::slice::from_ref(&payload)),
        ),
        (
            "CAB-6000",
            modern_serialized(26, false, "6000.0.0f1", &[payload]),
        ),
    ];
    fs::write(
        input.join("modern.bundle"),
        unity_bundle(8, true, true, 1, 3, &files),
    )
    .unwrap();
    let counts = scan(&input, &output, true, &mut Vec::new()).unwrap();
    assert_eq!(
        (
            counts.files,
            counts.accepted,
            counts.rejected,
            counts.unique,
            counts.translated
        ),
        (3, 6, 6, 1, 1)
    );
    let events = manifest(&output);
    assert_eq!(
        events
            .iter()
            .filter(|e| e["kind"] == "unity_shader" && e["status"] == "scanned")
            .count(),
        6
    );
    assert_eq!(
        events
            .iter()
            .filter(|e| e["kind"] == "unity_blob_scan")
            .count(),
        18
    );
    assert_eq!(
        events
            .iter()
            .filter(|e| e["kind"] == "rejected"
                && e["reason"]
                    .as_str()
                    .unwrap()
                    .contains("unknown instruction"))
            .count(),
        6
    );
    assert!(events
        .iter()
        .filter(|e| e["kind"] == "program")
        .all(|e| e["offset"] == 13 && e["unity"]["compressed_blob_offset"] == 3));
    assert!(!events
        .iter()
        .any(|e| e["kind"] == "unity_skipped" || e["kind"] == "file"));
}

#[test]
fn unity_sparse_files_shared_trees_and_patch_layouts_keep_instruction_failures() {
    use crate::unity::tests::{serialized, shared_serialized, sparse_serialized_parts, stripped};
    let dir = TestDir::new();
    let input = dir.0.join("input");
    let output = input.join("output");
    fs::create_dir_all(&input).unwrap();
    let valid = record(headers::Stage::Fragment, &bundle([MOV, EXIT, NOP]));
    let mut unknown = valid.clone();
    unknown[0x88..0x90].fill(0xff);
    let payload = [noise(13), valid, unknown].concat();
    for (index, bytes) in [
        serialized(22, false, true, std::slice::from_ref(&payload)),
        shared_serialized(true, &payload, ""),
        stripped("2022.1.21f1", true, std::slice::from_ref(&payload)),
        stripped("2022.2.0f1", false, std::slice::from_ref(&payload)),
        stripped("2022.3.0f1", true, std::slice::from_ref(&payload)),
        stripped("2023.1.0f1", false, std::slice::from_ref(&payload)),
        stripped("2023.2.0f1", true, std::slice::from_ref(&payload)),
    ]
    .into_iter()
    .enumerate()
    {
        // Project-authored sparse files exceed 4 GiB in file size, data offset
        // and object start. No large backing buffer or game data is used.
        let (prefix, at, data) = sparse_serialized_parts(&bytes, (1u64 << 32) + 16);
        let mut file = File::create(input.join(format!("sparse-{index}.dat"))).unwrap();
        file.write_all(&prefix).unwrap();
        file.seek(SeekFrom::Start(at)).unwrap();
        file.write_all(&data).unwrap();
        assert!(file.metadata().unwrap().len() > 1u64 << 33);
    }
    let mut report = Vec::new();
    let counts = scan(&input, &output, true, &mut report).unwrap();
    assert_eq!(
        (
            counts.files,
            counts.candidates,
            counts.accepted,
            counts.rejected
        ),
        (7, 14, 7, 7)
    );
    let events = manifest(&output);
    assert_eq!(
        events
            .iter()
            .filter(|e| e["kind"] == "unity_shader" && e["status"] == "scanned")
            .count(),
        7
    );
    assert!(events
        .iter()
        .filter(|e| e["kind"] == "rejected")
        .all(|e| e["reason"]
            .as_str()
            .unwrap()
            .contains("unknown instruction")));
    assert!(!events.iter().any(|e| e["kind"] == "unity_skipped"
        || (e["kind"] == "unity_shader" && e["status"] == "skipped")));
    assert!(!fs::read_dir(&output).unwrap().any(|entry| entry
        .unwrap()
        .file_name()
        .to_string_lossy()
        .contains(".tmp-")));
}

#[test]
fn unity_stripped_6000_layouts_fail_without_scanning_embedded_records() {
    use crate::unity::tests::stripped;
    let dir = TestDir::new();
    let input = dir.0.join("input");
    let output = dir.0.join("output");
    fs::create_dir_all(&input).unwrap();
    let payload = record(headers::Stage::Fragment, &bundle([MOV, EXIT, NOP]));
    for version in ["6000.0.0f1", "6000.1.0f1", "6000.2.0f1", "6000.3.0f1"] {
        fs::write(
            input.join(format!("{version}.assets")),
            stripped(version, true, std::slice::from_ref(&payload)),
        )
        .unwrap();
    }
    let counts = scan(&input, &output, true, &mut Vec::new()).unwrap();
    assert_eq!(
        (counts.files, counts.candidates, counts.accepted),
        (4, 0, 0)
    );
    let events = manifest(&output);
    assert_eq!(
        events
            .iter()
            .filter(|e| e["kind"] == "unity_shader"
                && e["status"] == "skipped"
                && e["reason"]
                    .as_str()
                    .unwrap()
                    .contains("newer layout requires a type tree"))
            .count(),
        4
    );
    assert!(!events
        .iter()
        .any(|e| e["kind"] == "unity_blob_scan" || e["kind"] == "file"));
}

#[path = "ue_shaderlib_tests.rs"]
mod ue_shaderlib_tests;

#[test]
fn pak_entry_integrity_and_summary_follow_actual_hash_checks() {
    let dir = TestDir::new();
    let input = dir.0.join("input");
    fs::create_dir_all(&input).unwrap();
    // Synthetic pak fixtures, made with the public synthetic key.
    for (name, bytes) in [
        (
            "mixed",
            include_bytes!("../pak/fixtures/synthetic-integrity.pak").as_slice(),
        ),
        (
            "plain",
            include_bytes!("../pak/fixtures/repak-ascii-plain.pak").as_slice(),
        ),
        (
            "encrypted-index",
            include_bytes!("../pak/fixtures/repak-ascii-encrypted.pak").as_slice(),
        ),
        (
            "bad-plain",
            include_bytes!("../pak/fixtures/repak-plain-entry-bad-hash.pak").as_slice(),
        ),
        (
            "bad-compressed",
            include_bytes!("../pak/fixtures/repak-compressed-entry-bad-hash.pak").as_slice(),
        ),
        (
            "bad-legacy",
            include_bytes!("../pak/fixtures/repak-legacy-entry-hash-disagreement.pak").as_slice(),
        ),
    ] {
        fs::write(input.join(format!("{name}.pak")), bytes).unwrap();
    }
    let key_hex: String = (0u8..32).map(|byte| format!("{byte:02x}")).collect();
    for pak_all in [false, true] {
        let output = dir.0.join(format!("output-{pak_all}"));
        let mut report = Vec::new();
        let totals = scan_options(
            &input,
            &output,
            false,
            pak_all,
            Some(crate::pak::AesKey::parse(key_hex.as_bytes()).unwrap()),
            &mut report,
        )
        .unwrap();
        assert_eq!(
            (totals.pak_entries_verified, totals.pak_entries_unverified),
            (8, 5)
        );
        let events = manifest(&output);
        let entries: Vec<_> = events.iter().filter(|e| e["kind"] == "pak_entry").collect();
        assert_eq!(entries.len(), 13);
        assert_eq!(
            entries
                .iter()
                .filter(|e| e["integrity"] == "verified")
                .count(),
            8
        );
        assert_eq!(
            entries
                .iter()
                .filter(|e| e["integrity"] == "unverified")
                .count(),
            5
        );
        for event in &entries {
            if event["status"] == "failed" {
                assert_eq!(event["integrity"], "unverified");
                assert!(event["reason"].as_str().unwrap().starts_with("bad hash:"));
            } else if event["status"] == "skipped" {
                assert_eq!(event["integrity"], "unverified");
                assert!(event["reason"]
                    .as_str()
                    .unwrap()
                    .contains("unsupported compression method"));
            } else {
                assert_eq!(
                    event["status"],
                    if pak_all { "scanned" } else { "no_shader_hint" }
                );
                assert!(event["reason"].is_null());
                assert_eq!(
                    event["integrity"],
                    if event["pak_entry"] == "encrypted/payload.txt" {
                        "unverified"
                    } else {
                        "verified"
                    }
                );
            }
        }
        assert_eq!(
            entries.iter().filter(|e| e["status"] == "failed").count(),
            3
        );
        for event in events.iter().filter(|e| e["kind"] == "pak_entry_scan") {
            let entry = entries
                .iter()
                .find(|entry| {
                    entry["source_file"] == event["source_file"]
                        && entry["pak_entry"] == event["pak_entry"]
                })
                .unwrap();
            assert_eq!(event["integrity"], entry["integrity"]);
        }
        assert_eq!(
            events
                .iter()
                .filter(|e| e["kind"] == "pak_entry_scan")
                .count(),
            if pak_all { 9 } else { 0 }
        );
        let summary = events.last().unwrap();
        assert_eq!(summary["pak_entries_verified"], 8);
        assert_eq!(summary["pak_entries_unverified"], 5);
        let report = String::from_utf8(report).unwrap();
        assert!(report.contains("pak_entries_verified=8 pak_entries_unverified=5"));
        assert!(!report.contains(&key_hex));
        assert!(!fs::read_to_string(output.join("manifest.jsonl"))
            .unwrap()
            .contains(&key_hex));
    }
}
