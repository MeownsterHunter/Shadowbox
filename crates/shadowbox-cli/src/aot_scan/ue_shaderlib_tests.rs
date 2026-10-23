use super::*;
use crate::iostore::tests as synthetic;
use crate::ue_shaderlib::tests::{group_fixture, loose_fixture};

fn shader_blobs() -> Vec<(u8, Vec<u8>)> {
    let first = [
        vec![0xaa; 17],
        record(headers::Stage::Fragment, &bundle([MOV, EXIT, NOP])),
        vec![0xcc; 11],
    ]
    .concat();
    let second = [
        vec![0xbb; 9],
        record(headers::Stage::Compute, &bundle([MOV, EXIT, NOP])),
    ]
    .concat();
    vec![(3, first), (5, second)]
}

#[test]
fn loose_files_and_paks_report_each_decompressed_shader() {
    use crate::pak::tests::{build, Codec, Options};
    for codec in ["None", "Zlib", "Zstd", "LZ4"] {
        let dir = TestDir::new();
        let input = dir.0.join("input");
        let output = dir.0.join("output");
        fs::create_dir(&input).unwrap();
        let library = loose_fixture(&shader_blobs(), codec);
        fs::write(input.join("ShaderCode-Synthetic.ushaderbytecode"), &library).unwrap();
        fs::write(
            input.join("ShaderArchive-Synthetic.ushaderbytecode"),
            &library,
        )
        .unwrap();
        fs::write(
            input.join("synthetic.pak"),
            build(
                11,
                Codec::Zlib,
                &library,
                Options {
                    shader_only: true,
                    ..Options::default()
                },
            ),
        )
        .unwrap();
        let totals = scan(&input, &output, true, &mut Vec::new()).unwrap();
        assert_eq!(
            (totals.unique, totals.translated, totals.rejected),
            (2, 2, 0),
            "codec {codec}"
        );
        let events = manifest(&output);
        let shaders: Vec<_> = events.iter().filter(|e| e["kind"] == "ue_shader").collect();
        assert_eq!(shaders.len(), 6);
        for pair in shaders.chunks(2) {
            assert_eq!(pair[0]["unreal"]["shader"]["shader_index"], 0);
            assert_eq!(pair[0]["unreal"]["shader"]["shader_hash"], "01".repeat(20));
            assert_eq!(pair[0]["unreal"]["shader"]["shader_maps"], json!([0]));
            assert_eq!(pair[0]["unreal"]["stage"], "fragment");
            assert_eq!(pair[1]["unreal"]["stage"], "compute");
            assert_eq!(pair[0]["unreal"]["compression"], codec);
            assert_eq!(pair[0]["status"], "scanned");
            for (event, (_, expected)) in pair.iter().zip(shader_blobs()) {
                assert_eq!(
                    fs::read(output.join(event["unreal"]["blob_output"].as_str().unwrap()))
                        .unwrap(),
                    expected
                );
            }
        }
        for program in events
            .iter()
            .filter(|e| e["kind"] == "program" && !e["unreal"].is_null())
        {
            let stage = program["stage"].as_str().unwrap();
            let expected = record(
                if stage == "fragment" {
                    headers::Stage::Fragment
                } else {
                    headers::Stage::Compute
                },
                &bundle([MOV, EXIT, NOP]),
            );
            let normalized = if stage == "fragment" {
                expected[0x30..].to_vec()
            } else {
                [vec![0; 80], expected[0x100..].to_vec()].concat()
            };
            assert_eq!(
                fs::read(output.join(program["output"].as_str().unwrap())).unwrap(),
                normalized
            );
            assert_eq!(program["offset"], if stage == "fragment" { 17 } else { 9 });
        }
        assert!(fs::read_dir(&output).unwrap().all(|e| !e
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains("partial")));
    }
}

#[test]
fn iostore_groups_split_shaders_before_program_discovery() {
    for version in [4, 8] {
        for codec in ["None", "Zlib", "Zstd", "LZ4"] {
            let dir = TestDir::new();
            let input = dir.0.join("input");
            let output = dir.0.join("output");
            fs::create_dir(&input).unwrap();
            let entries = shader_blobs();
            let bytes: Vec<_> = entries
                .iter()
                .flat_map(|(_, b)| b.iter().copied())
                .collect();
            let encoded = synthetic::compressed(codec, &bytes);
            let id = synthetic::code_id(version, 1);
            let mut lib_id = id;
            lib_id[0] = 2;
            lib_id[11] = if version > 4 { 8 } else { 11 };
            let header = group_fixture(id, &entries, encoded.len() as u32);
            synthetic::container(
                version,
                &[(id, None, encoded), (lib_id, None, header)],
                "Zstd",
                false,
                u64::MAX,
            )
            .write(&input.join("synthetic.utoc"));
            let totals = scan(&input, &output, true, &mut Vec::new()).unwrap();
            assert_eq!(
                (totals.accepted, totals.rejected, totals.translated),
                (2, 0, 2)
            );
            let events = manifest(&output);
            let programs: Vec<_> = events.iter().filter(|e| e["kind"] == "program").collect();
            assert_eq!(programs[0]["offset"], 17);
            assert_eq!(programs[1]["offset"], 9);
            assert_eq!(programs[0]["iostore"]["shader_group"]["compression"], codec);
            assert_eq!(
                programs[1]["unreal"]["shader"]["offset"],
                entries[0].1.len()
            );
            assert_eq!(programs[1]["unreal"]["shader"]["frequency"], 5);
        }
    }
}

#[test]
fn parsed_libraries_keep_stage_and_instruction_validation_during_fallback() {
    let dir = TestDir::new();
    let input = dir.0.join("input");
    let output = dir.0.join("output");
    fs::create_dir(&input).unwrap();
    let good = record(headers::Stage::Fragment, &bundle([MOV, EXIT, NOP]));
    let unknown = record(headers::Stage::Fragment, &bundle([u64::MAX, EXIT, NOP]));
    let library = loose_fixture(
        &[
            (3, unknown),
            (5, good.clone()),
            (1, good.clone()),
            (6, good.clone()),
            (3, vec![0; 100]),
            (3, good.clone()),
        ],
        "Zlib",
    );
    fs::write(
        input.join("ShaderArchive-Negative.ushaderbytecode"),
        library,
    )
    .unwrap();
    for version in [0u32, 1, 3, u32::MAX] {
        let mut bytes = loose_fixture(&[(3, good.clone())], "None");
        bytes[..4].copy_from_slice(&version.to_le_bytes());
        fs::write(
            input.join(format!("ShaderCode-Version{version}.ushaderbytecode")),
            bytes,
        )
        .unwrap();
    }
    let id = synthetic::code_id(8, 1);
    let mut lib_id = id;
    lib_id[0] = 2;
    lib_id[11] = 8;
    let mut header = group_fixture(id, &[(3, good.clone())], good.len() as u32);
    header[0] = 2;
    synthetic::container(
        8,
        &[(id, None, good), (lib_id, None, header)],
        "None",
        false,
        u64::MAX,
    )
    .write(&input.join("unknown.utoc"));
    let totals = scan_options(&input, &output, false, true, None, &mut Vec::new()).unwrap();
    assert_eq!((totals.accepted, totals.rejected), (7, 3));
    let events = manifest(&output);
    assert_eq!(
        events
            .iter()
            .filter(|e| e["kind"] == "ue_library" && e["status"] == "raw-scanned")
            .count(),
        4
    );
    assert!(events.iter().any(|e| e["reason"]
        .as_str()
        .is_some_and(|r| r.contains("frequency disagrees"))));
    assert!(events.iter().any(|e| e["reason"]
        .as_str()
        .is_some_and(|r| r.contains("unknown instruction"))));
    assert!(events.iter().any(|e| e["reason"]
        .as_str()
        .is_some_and(|r| r.contains("no known Maxwell"))));
    assert!(events
        .iter()
        .any(|e| e["kind"] == "program" && e["source_file"] == "unknown.utoc"));
}

#[test]
fn program_extent_cannot_cross_a_shader_boundary_and_interior_magic_is_data() {
    let dir = TestDir::new();
    let input = dir.0.join("input");
    let output = dir.0.join("output");
    fs::create_dir(&input).unwrap();
    let program = record(headers::Stage::Fragment, &bundle([MOV, EXIT, NOP]));
    // Mesa nak/sm50.rs:213-235,1938-1941: inverted P0 at 16..20
    // plus literal 0x01234567 at 20..52 puts exact magic at bytes 2..6.
    let magic_mov = 0x0100_0000_0008_f000 | (0x0123_4567u64 << 20);
    let interior = record(headers::Stage::Fragment, &bundle([magic_mov, EXIT, NOP]));
    assert_eq!(
        interior
            .windows(4)
            .filter(|b| *b == [0x78, 0x56, 0x34, 0x12])
            .count(),
        2
    );
    let library = loose_fixture(
        &[
            (3, program[..0x80].to_vec()),
            (3, program[0x80..].to_vec()),
            (3, interior),
        ],
        "None",
    );
    fs::write(
        input.join("ShaderArchive-Boundaries.ushaderbytecode"),
        library,
    )
    .unwrap();
    let totals = scan(&input, &output, false, &mut Vec::new()).unwrap();
    assert_eq!(
        (totals.candidates, totals.accepted, totals.rejected),
        (2, 1, 1)
    );
}

// Compare full normalized program bytes, stages and original record offsets.
// Main 537276a1 uses this same raw validator; fallback must not bound records by
// an unparsed header or skip magic inside a raw record's proven extent.
fn program_set(output: &Path) -> Vec<(String, u64, Vec<u8>)> {
    let mut programs: Vec<_> = manifest(output)
        .iter()
        .filter(|e| e["kind"] == "program")
        .map(|e| {
            (
                e["stage"].as_str().unwrap().to_owned(),
                e["offset"].as_u64().unwrap(),
                fs::read(output.join(e["output"].as_str().unwrap())).unwrap(),
            )
        })
        .collect();
    programs.sort();
    programs
}

#[test]
fn unparseable_loose_and_pak_libraries_match_raw_program_sets() {
    use crate::pak::tests::{build, Codec, Options};
    let entries = shader_blobs();
    let library = loose_fixture(&entries, "None");
    let mut unsupported = library.clone();
    unsupported[..4].copy_from_slice(&99u32.to_le_bytes());
    let mut malformed = library;
    // First code offset in the synthetic two-entry layout, cited in the writer.
    malformed[96..104].copy_from_slice(&u64::MAX.to_le_bytes());
    // A complete hash array of opaque, project-authored bytes, followed by a
    // missing shader-hash count. Raw scanning still sees both records.
    let mut hashes: Vec<_> = entries.iter().flat_map(|(_, b)| b.clone()).collect();
    hashes.resize(hashes.len().div_ceil(20) * 20, 0);
    let truncated = [
        2u32.to_le_bytes().to_vec(),
        (hashes.len() as u32 / 20).to_le_bytes().to_vec(),
        hashes,
    ]
    .concat();
    for (name, bytes, reason) in [
        (
            "unsupported",
            unsupported,
            "unsupported loose shader library version 99 (only 2 supported)",
        ),
        (
            "malformed",
            malformed,
            "invalid or oversized shader code range",
        ),
        ("truncated", truncated, "truncated shader library table"),
    ] {
        let dir = TestDir::new();
        let raw_input = dir.0.join("raw-input");
        fs::create_dir(&raw_input).unwrap();
        fs::write(raw_input.join("raw.dat"), &bytes).unwrap();
        let raw_output = dir.0.join("raw-output");
        let raw = scan(&raw_input, &raw_output, false, &mut Vec::new()).unwrap();
        assert_eq!((raw.accepted, raw.unique), (2, 2), "{name}");
        let expected = program_set(&raw_output);
        for pak in [false, true] {
            let input = dir.0.join(format!("input-{pak}"));
            let output = dir.0.join(format!("output-{pak}"));
            fs::create_dir(&input).unwrap();
            if pak {
                fs::write(
                    input.join("synthetic.pak"),
                    build(
                        11,
                        Codec::Zlib,
                        &bytes,
                        Options {
                            shader_only: true,
                            ..Options::default()
                        },
                    ),
                )
                .unwrap();
            } else {
                fs::write(input.join("synthetic.ushaderbytecode"), &bytes).unwrap();
            }
            let mut report = Vec::new();
            let totals = scan(&input, &output, false, &mut report).unwrap();
            assert_eq!(
                (totals.accepted, totals.unique, totals.files),
                (2, 2, 1),
                "{name} pak={pak}"
            );
            assert_eq!(program_set(&output), expected, "{name} pak={pak}");
            let events = manifest(&output);
            let event = events.iter().find(|e| e["kind"] == "ue_library").unwrap();
            assert_eq!(event["status"], "raw-scanned");
            assert_eq!(event["reason"], format!("library not parsed: {reason}"));
            assert!(String::from_utf8(report)
                .unwrap()
                .contains(&format!("raw-scanned (library not parsed: {reason})")));
        }
    }
}

#[test]
fn descriptorless_iostore_pak_all_matches_raw_program_set() {
    for version in [4, 8] {
        let dir = TestDir::new();
        let input = dir.0.join("input");
        let output = dir.0.join("output");
        fs::create_dir(&input).unwrap();
        let bytes: Vec<_> = shader_blobs().into_iter().flat_map(|(_, b)| b).collect();
        let id = synthetic::code_id(version, 1);
        synthetic::container(
            version,
            &[(id, None, bytes.clone())],
            "Zstd",
            false,
            u64::MAX,
        )
        .write(&input.join("descriptorless.utoc"));
        let totals = scan(&input, &output, false, &mut Vec::new()).unwrap();
        assert_eq!(totals.accepted, 0);
        assert!(program_set(&output).is_empty());
        let mut report = Vec::new();
        let totals = scan_options(&input, &output, false, true, None, &mut report).unwrap();
        assert_eq!((totals.accepted, totals.unique), (2, 2));
        let raw_input = dir.0.join("raw-input");
        let raw_output = dir.0.join("raw-output");
        fs::create_dir(&raw_input).unwrap();
        fs::write(raw_input.join("raw.dat"), bytes).unwrap();
        scan(&raw_input, &raw_output, false, &mut Vec::new()).unwrap();
        assert_eq!(program_set(&output), program_set(&raw_output));
        let events = manifest(&output);
        let event = events
            .iter()
            .find(|e| e["kind"] == "iostore_chunk")
            .unwrap();
        let reason = "shader code has no supported same-container library descriptor; per-shader boundaries and frequency are unknown";
        assert_eq!(event["status"], "raw-scanned");
        assert_eq!(event["reason"], reason);
        assert!(String::from_utf8(report)
            .unwrap()
            .contains(&format!("raw-scanned: {reason}")));
    }
}

#[test]
fn tessellation_libraries_retain_validated_programs_and_stages() {
    let hull = record(
        headers::Stage::TessellationControl,
        &bundle([MOV, EXIT, NOP]),
    );
    let domain = record(
        headers::Stage::TessellationEvaluation,
        &bundle([MOV, EXIT, NOP]),
    );
    let entries = [(1, hull.clone()), (2, domain.clone())];
    for iostore in [false, true] {
        let dir = TestDir::new();
        let input = dir.0.join("input");
        let output = dir.0.join("output");
        fs::create_dir(&input).unwrap();
        if iostore {
            let bytes = [hull.clone(), domain.clone()].concat();
            let id = synthetic::code_id(8, 1);
            let mut lib_id = id;
            lib_id[0] = 2;
            lib_id[11] = 8;
            let header = group_fixture(id, &entries, bytes.len() as u32);
            synthetic::container(
                8,
                &[(id, None, bytes), (lib_id, None, header)],
                "None",
                false,
                u64::MAX,
            )
            .write(&input.join("tessellation.utoc"));
        } else {
            fs::write(
                input.join("tessellation.ushaderbytecode"),
                loose_fixture(&entries, "Zlib"),
            )
            .unwrap();
        }
        for translate in [false, true] {
            let mut report = Vec::new();
            let totals = scan(&input, &output, translate, &mut report).unwrap();
            assert_eq!((totals.accepted, totals.unique, totals.rejected), (2, 2, 0));
            let mut expected = vec![
                ("tessellation-control".to_owned(), 0, hull[0x30..].to_vec()),
                (
                    "tessellation-evaluation".to_owned(),
                    0,
                    domain[0x30..].to_vec(),
                ),
            ];
            expected.sort();
            assert_eq!(program_set(&output), expected);
            let events = manifest(&output);
            for (index, stage) in [(0, "tessellation-control"), (1, "tessellation-evaluation")] {
                let event = events
                    .iter()
                    .find(|e| {
                        e["kind"] == "ue_shader" && e["unreal"]["shader"]["shader_index"] == index
                    })
                    .unwrap();
                assert_eq!(event["status"], "scanned");
                assert_eq!(event["unreal"]["stage"], stage);
                assert!(
                    String::from_utf8_lossy(&report).contains(&format!("stage={stage}: scanned"))
                );
                let program = events
                    .iter()
                    .find(|e| e["kind"] == "program" && e["stage"] == stage)
                    .unwrap();
                assert_eq!(
                    program["translation"],
                    if translate { "failed" } else { "not_requested" }
                );
                if translate {
                    // Stages map to translator types. This fixture has no hull
                    // output-count metadata or domain driver state; both remain
                    // found even when their translations fail for those reasons.
                    let error = program["translation_error"].as_str().unwrap();
                    assert!(
                        error.contains(if stage == "tessellation-control" {
                            "tessellation output control point count must be 1..32"
                        } else {
                            "stage requires state"
                        }),
                        "{stage}: {error}"
                    );
                }
            }
        }
    }
}

fn write_loose_transport(input: &Path, bytes: &[u8], transport: &str) {
    use crate::pak::tests::{build, Codec, Options};
    match transport {
        "loose" => fs::write(input.join("synthetic.ushaderbytecode"), bytes).unwrap(),
        "pak" => fs::write(
            input.join("synthetic.pak"),
            build(
                11,
                Codec::Zlib,
                bytes,
                Options {
                    shader_only: true,
                    ..Options::default()
                },
            ),
        )
        .unwrap(),
        "iostore" => {
            let mut id = synthetic::code_id(8, 1);
            id[11] = 1; // Ordinary named chunk, not a ShaderCodeLibrary header.
            synthetic::container(
                8,
                &[(id, Some("synthetic.ushaderbytecode"), bytes.to_vec())],
                "Zstd",
                false,
                u64::MAX,
            )
            .write(&input.join("synthetic.utoc"));
        }
        _ => panic!("unknown synthetic transport"),
    }
}

fn raw_blob_programs(dir: &Path, blobs: &[Vec<u8>]) -> Vec<(String, u64, Vec<u8>)> {
    let input = dir.join("raw-input");
    let output = dir.join("raw-output");
    fs::create_dir(&input).unwrap();
    for (index, blob) in blobs.iter().enumerate() {
        fs::write(input.join(format!("{index}.dat")), blob).unwrap();
    }
    scan(&input, &output, false, &mut Vec::new()).unwrap();
    program_set(&output)
}

#[test]
fn unknown_shader_frequencies_match_raw_blob_program_sets() {
    for frequency in [6, 15, 255] {
        for codec in ["None", "Zlib"] {
            for transport in ["loose", "pak", "iostore", "group4", "group8"] {
                // IoStore's packed frequency is four bits (retoc
                // shader_library.rs:54-70); loose entries have a full byte.
                if frequency > 15 && transport.starts_with("group") {
                    continue;
                }
                let dir = TestDir::new();
                let input = dir.0.join("input");
                let output = dir.0.join("output");
                fs::create_dir(&input).unwrap();
                let mut entries = shader_blobs();
                entries[0].0 = frequency;
                let expected = raw_blob_programs(
                    &dir.0,
                    &entries.iter().map(|(_, b)| b.clone()).collect::<Vec<_>>(),
                );
                assert_eq!(expected.len(), 2);
                assert_eq!(
                    expected[0],
                    (
                        "compute".into(),
                        9,
                        [vec![0; 80], entries[1].1[9 + 0x100..].to_vec()].concat()
                    )
                );
                assert_eq!(
                    expected[1],
                    (
                        "fragment".into(),
                        17,
                        entries[0].1[17 + 0x30..entries[0].1.len() - 11].to_vec()
                    )
                );
                if transport.starts_with("group") {
                    let version = if transport == "group4" { 4 } else { 8 };
                    let bytes: Vec<_> = entries.iter().flat_map(|(_, b)| b.clone()).collect();
                    let encoded = synthetic::compressed(codec, &bytes);
                    let id = synthetic::code_id(version, 1);
                    let mut lib_id = id;
                    lib_id[0] = 2;
                    lib_id[11] = if version == 4 { 11 } else { 8 };
                    let header = group_fixture(id, &entries, encoded.len() as u32);
                    synthetic::container(
                        version,
                        &[(id, None, encoded), (lib_id, None, header)],
                        "Zstd",
                        false,
                        u64::MAX,
                    )
                    .write(&input.join("synthetic.utoc"));
                } else {
                    write_loose_transport(&input, &loose_fixture(&entries, codec), transport);
                }
                let mut report = Vec::new();
                let totals = scan(&input, &output, false, &mut report).unwrap();
                assert_eq!(
                    (totals.accepted, totals.unique, totals.rejected),
                    (2, 2, 0),
                    "{frequency} {codec} {transport}"
                );
                assert_eq!(
                    program_set(&output),
                    expected,
                    "{frequency} {codec} {transport}"
                );
                let events = manifest(&output);
                let event = events
                    .iter()
                    .find(|e| {
                        e["kind"] == "ue_shader" && e["unreal"]["shader"]["shader_index"] == 0
                    })
                    .unwrap();
                let reason =
                    format!("unsupported or ambiguous Unreal shader frequency {frequency}");
                assert_eq!(event["status"], "raw-scanned");
                assert_eq!(event["reason"], reason);
                assert_eq!(event["unreal"]["accepted"], 1);
                if transport.starts_with("group") {
                    assert_eq!(
                        event["unreal"]["iostore"]["shader_group"]["compression"],
                        codec
                    );
                } else {
                    assert_eq!(event["unreal"]["compression"], codec);
                }
                assert!(String::from_utf8(report)
                    .unwrap()
                    .contains(&format!("raw-scanned: {reason}")));
            }
        }
    }
}

#[test]
fn failed_shader_decompression_matches_raw_stored_program_sets() {
    for unusable in [false, true] {
        for transport in ["loose", "pak", "iostore"] {
            let dir = TestDir::new();
            let input = dir.0.join("input");
            let output = dir.0.join("output");
            fs::create_dir(&input).unwrap();
            let mut entries = shader_blobs();
            if unusable {
                entries[0].1 = vec![0xff; 128];
            }
            let mut library = loose_fixture(&entries, "None");
            // retoc shader_library.rs:361-375: expanded size is 12 bytes
            // into the first 17-byte entry. The writer's two-entry layout
            // starts entries at 96 (same layout as the malformed-extent test).
            library[108..112].copy_from_slice(&(entries[0].1.len() as u32 + 37).to_le_bytes());
            let parsed = crate::ue_shaderlib::loose(&library).unwrap();
            let shader = &parsed.shaders[0];
            assert!(
                crate::ue_shaderlib::expand(&entries[0].1, shader.size, shader.uncompressed)
                    .is_err()
            );
            let expected = raw_blob_programs(
                &dir.0,
                &entries.iter().map(|(_, b)| b.clone()).collect::<Vec<_>>(),
            );
            let count = if unusable { 1 } else { 2 };
            let fragment = record(headers::Stage::Fragment, &bundle([MOV, EXIT, NOP]));
            let compute = record(headers::Stage::Compute, &bundle([MOV, EXIT, NOP]));
            let mut exact = vec![(
                "compute".into(),
                9,
                [vec![0; 80], compute[0x100..].to_vec()].concat(),
            )];
            if !unusable {
                exact.push(("fragment".into(), 17, fragment[0x30..].to_vec()));
            }
            exact.sort();
            assert_eq!(expected, exact);
            write_loose_transport(&input, &library, transport);
            let mut report = Vec::new();
            let totals = scan(&input, &output, false, &mut report).unwrap();
            assert_eq!(
                (totals.accepted, totals.unique, totals.rejected),
                (count as u64, count as u64, 0)
            );
            assert_eq!(program_set(&output), expected);
            let events = manifest(&output);
            let event = events
                .iter()
                .find(|e| e["kind"] == "ue_shader" && e["unreal"]["shader"]["shader_index"] == 0)
                .unwrap();
            assert_eq!(event["status"], "raw-scanned");
            let reason = event["reason"].as_str().unwrap();
            assert!(reason.starts_with("shader decompression failed: unsupported or invalid shader compression; Oodle is unsupported"));
            assert_eq!(reason.contains("no known Maxwell program record"), unusable);
            assert_eq!(event["unreal"]["accepted"], if unusable { 0 } else { 1 });
            assert!(String::from_utf8(report).unwrap().contains(reason));
            assert!(fs::read_dir(&output).unwrap().all(|e| !e
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains("partial")));
        }
    }
}

#[test]
fn failed_group_decompression_matches_raw_stored_group_program_sets() {
    for version in [4, 8] {
        for unusable in [false, true] {
            let dir = TestDir::new();
            let input = dir.0.join("input");
            let output = dir.0.join("output");
            fs::create_dir(&input).unwrap();
            let entries = shader_blobs();
            let bytes: Vec<_> = entries.iter().flat_map(|(_, b)| b.clone()).collect();
            let stored = if unusable {
                vec![0xff; bytes.len()]
            } else {
                bytes.clone()
            };
            let id = synthetic::code_id(version, 1);
            let mut lib_id = id;
            lib_id[0] = 2;
            lib_id[11] = if version == 4 { 11 } else { 8 };
            let mut header = group_fixture(id, &entries, stored.len() as u32);
            // retoc shader_library.rs:90-105: group expanded size at +8.
            // The synthetic writer places the sole group before the final
            // shader-index count and two indices.
            let at = header.len() - 12 - 16 + 8;
            header[at..at + 4].copy_from_slice(&(bytes.len() as u32 + 37).to_le_bytes());
            let parsed = crate::iostore::shaders::library(&header).unwrap();
            assert!(crate::iostore::shaders::expand(&stored, &parsed.groups[0]).is_err());
            let expected = raw_blob_programs(&dir.0, std::slice::from_ref(&stored));
            let count = if unusable { 0 } else { 2 };
            let fragment = record(headers::Stage::Fragment, &bundle([MOV, EXIT, NOP]));
            let compute = record(headers::Stage::Compute, &bundle([MOV, EXIT, NOP]));
            let mut exact = if unusable {
                Vec::new()
            } else {
                vec![
                    ("fragment".into(), 17, fragment[0x30..].to_vec()),
                    (
                        "compute".into(),
                        entries[0].1.len() as u64 + 9,
                        [vec![0; 80], compute[0x100..].to_vec()].concat(),
                    ),
                ]
            };
            exact.sort();
            assert_eq!(expected, exact);
            synthetic::container(
                version,
                &[(id, None, stored), (lib_id, None, header)],
                "Zstd",
                false,
                u64::MAX,
            )
            .write(&input.join("synthetic.utoc"));
            let mut report = Vec::new();
            let totals = scan(&input, &output, false, &mut report).unwrap();
            assert_eq!(
                (totals.accepted, totals.unique, totals.rejected),
                (count as u64, count as u64, 0)
            );
            assert_eq!(program_set(&output), expected);
            let events = manifest(&output);
            let shaders: Vec<_> = events.iter().filter(|e| e["kind"] == "ue_shader").collect();
            assert_eq!(shaders.len(), 2);
            for event in shaders {
                assert_eq!(event["status"], "skipped");
                assert!(event["reason"]
                    .as_str()
                    .unwrap()
                    .contains("shader bytes unusable: shader group decompression failed"));
                assert!(event["reason"]
                    .as_str()
                    .unwrap()
                    .contains("expanded shader boundaries unavailable; raw-scanning stored group"));
            }
            let event = events.iter().find(|e| e["kind"] == "ue_group").unwrap();
            assert_eq!(event["status"], "raw-scanned");
            assert_eq!(event["unreal"]["accepted"], count);
            assert_eq!(
                event["reason"]
                    .as_str()
                    .unwrap()
                    .contains("no known Maxwell program record"),
                unusable
            );
            assert!(String::from_utf8(report)
                .unwrap()
                .contains("raw-scanning stored group"));
        }
    }
}

#[test]
fn unknown_frequency_fallback_keeps_instruction_and_shader_boundary_validation() {
    for grouped in [false, true] {
        let dir = TestDir::new();
        let input = dir.0.join("input");
        let output = dir.0.join("output");
        fs::create_dir(&input).unwrap();
        let good = record(headers::Stage::Fragment, &bundle([MOV, EXIT, NOP]));
        let sibling = record(headers::Stage::Compute, &bundle([MOV, EXIT, NOP]));
        let entries = [
            (
                6,
                record(headers::Stage::Fragment, &bundle([u64::MAX, EXIT, NOP])),
            ),
            (15, good[..0x80].to_vec()),
            (7, vec![0xff; 128]),
            (5, sibling.clone()),
        ];
        let expected = raw_blob_programs(
            &dir.0,
            &entries.iter().map(|(_, b)| b.clone()).collect::<Vec<_>>(),
        );
        assert_eq!(
            expected,
            vec![(
                "compute".into(),
                0,
                [vec![0; 80], sibling[0x100..].to_vec()].concat()
            )]
        );
        if grouped {
            let bytes: Vec<_> = entries.iter().flat_map(|(_, b)| b.clone()).collect();
            let encoded = synthetic::compressed("Zlib", &bytes);
            let id = synthetic::code_id(8, 1);
            let mut lib_id = id;
            lib_id[0] = 2;
            lib_id[11] = 8;
            let header = group_fixture(id, &entries, encoded.len() as u32);
            synthetic::container(
                8,
                &[(id, None, encoded), (lib_id, None, header)],
                "None",
                false,
                u64::MAX,
            )
            .write(&input.join("synthetic.utoc"));
        } else {
            write_loose_transport(&input, &loose_fixture(&entries, "Zlib"), "loose");
        }
        let totals = scan(&input, &output, false, &mut Vec::new()).unwrap();
        assert_eq!(
            (
                totals.candidates,
                totals.accepted,
                totals.unique,
                totals.rejected
            ),
            (3, 1, 1, 2)
        );
        assert_eq!(program_set(&output), expected);
        let events = manifest(&output);
        assert!(events.iter().any(|e| e["kind"] == "rejected"
            && e["reason"]
                .as_str()
                .unwrap()
                .contains("unknown instruction")));
        for index in 0..3 {
            let event = events
                .iter()
                .find(|e| {
                    e["kind"] == "ue_shader" && e["unreal"]["shader"]["shader_index"] == index
                })
                .unwrap();
            assert_eq!(event["status"], "raw-scanned");
            assert_eq!(event["unreal"]["accepted"], 0);
            assert_eq!(
                event["reason"]
                    .as_str()
                    .unwrap()
                    .contains("no known Maxwell program record"),
                index == 2
            );
        }
    }
}

#[test]
fn synthetic_integrity_pak_libraries_inherit_actual_hash_evidence() {
    use crate::pak::tests::integrity_library_fixture;
    let good = record(headers::Stage::Fragment, &bundle([MOV, EXIT, NOP]));
    let unknown = record(headers::Stage::Fragment, &bundle([u64::MAX, EXIT, NOP]));
    // Existing project writer: ue_shaderlib.rs:289-328. An unsupported frequency
    // exercises per-shader fallback, and an unknown instruction stays rejected.
    let library = loose_fixture(&[(3, good.clone()), (6, good), (3, unknown)], "None");
    let key_hex: String = (0u8..32).map(|byte| format!("{byte:02x}")).collect();
    for encrypted in [false, true] {
        for raw_library in [false, true] {
            let mut bytes = library.clone();
            if raw_library {
                bytes[..4].copy_from_slice(&99u32.to_le_bytes());
            }
            let dir = TestDir::new();
            let input = dir.0.join("input");
            fs::create_dir(&input).unwrap();
            fs::write(
                input.join("synthetic-integrity.pak"),
                integrity_library_fixture(&bytes, encrypted),
            )
            .unwrap();
            for pak_all in [false, true] {
                let output = dir.0.join(format!("output-{pak_all}"));
                let totals = scan_options(
                    &input,
                    &output,
                    false,
                    pak_all,
                    Some(crate::pak::AesKey::parse(key_hex.as_bytes()).unwrap()),
                    &mut Vec::new(),
                )
                .unwrap();
                let expected = if encrypted { "unverified" } else { "verified" };
                let expected_totals = if encrypted { (1, 3) } else { (2, 2) };
                assert_eq!(
                    (totals.pak_entries_verified, totals.pak_entries_unverified),
                    expected_totals
                );
                assert_eq!((totals.accepted, totals.rejected), (2, 1));
                let events = manifest(&output);
                let entries: Vec<_> = events.iter().filter(|e| e["kind"] == "pak_entry").collect();
                assert_eq!(entries.len(), 4);
                for event in events.iter().filter(|e| !e["pak_entry"].is_null()) {
                    let entry = entries
                        .iter()
                        .find(|e| e["pak_entry"] == event["pak_entry"])
                        .unwrap();
                    assert_eq!(event["integrity"], entry["integrity"]);
                    if !event["unreal"].is_null() {
                        assert_eq!(event["unreal"]["pak_entry"], event["pak_entry"]);
                        assert_eq!(event["unreal"]["integrity"], expected);
                    }
                }
                let selected = entries
                    .iter()
                    .find(|e| e["status"] == "library_selected")
                    .unwrap();
                assert_eq!(selected["integrity"], expected);
                let library_event = events.iter().find(|e| e["kind"] == "ue_library").unwrap();
                assert_eq!(library_event["integrity"], expected);
                assert_eq!(
                    library_event["status"],
                    if raw_library { "raw-scanned" } else { "parsed" }
                );
                for kind in [
                    "program",
                    "rejected",
                    if raw_library {
                        "pak_entry_scan"
                    } else {
                        "ue_shader_scan"
                    },
                ] {
                    let related: Vec<_> = events
                        .iter()
                        .filter(|e| e["kind"] == kind && e["pak_entry"] == selected["pak_entry"])
                        .collect();
                    assert!(!related.is_empty(), "missing {kind}");
                    for event in related {
                        assert_eq!(event["integrity"], expected);
                        assert!(!event["unreal"].is_null());
                    }
                }
                let verified = entries
                    .iter()
                    .filter(|e| e["integrity"] == "verified")
                    .count();
                assert_eq!(
                    (verified as u64, (entries.len() - verified) as u64),
                    expected_totals
                );
                let summary = events.last().unwrap();
                assert_eq!(summary["pak_entries_verified"], expected_totals.0);
                assert_eq!(summary["pak_entries_unverified"], expected_totals.1);
            }
        }
    }
}
