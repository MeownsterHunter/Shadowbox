use super::*;
use crate::iostore::{tests as synthetic, Toc};

// Keep transport fixtures valid shader archives now that unnamed ShaderCode
// chunks require descriptors. Frequencies come from our own fixture choices.
fn described_container(
    version: u8,
    chunks: &[([u8; 12], Option<&str>, Vec<u8>)],
    codec: &str,
    encrypted: bool,
    partition: u64,
) -> synthetic::Fixture {
    let mut entries = chunks.to_vec();
    for (ordinal, (id, _, data)) in chunks.iter().enumerate() {
        if id[11] != if version > 4 { 9 } else { 12 } {
            continue;
        }
        let frequency = if data.starts_with(&0x1234_5679u32.to_le_bytes()) {
            5
        } else {
            3
        };
        let header = crate::ue_shaderlib::tests::group_fixture(
            *id,
            &[(frequency, data.clone())],
            data.len() as u32,
        );
        let mut lib = *id;
        lib[0] = 100 + ordinal as u8;
        lib[11] = if version > 4 { 8 } else { 11 };
        entries.push((lib, None, header));
    }
    synthetic::container(version, &entries, codec, encrypted, partition)
}

#[test]
fn iostore_records_across_blocks_versions_and_global_container() {
    for version in 2..=8 {
        let dir = TestDir::new();
        let input = dir.0.join("input");
        let output = dir.0.join("output");
        fs::create_dir(&input).unwrap();
        let graphics = record(headers::Stage::Fragment, &bundle([MOV, EXIT, NOP]));
        let compute = record(headers::Stage::Compute, &bundle([MOV, EXIT, NOP]));
        let mut blob = vec![0xaa; 17];
        blob.extend(&graphics);
        blob.extend(&compute);
        let id = synthetic::code_id(version, 1);
        let mut lib = id;
        lib[0] = 100;
        lib[11] = if version > 4 { 8 } else { 11 };
        let header = crate::ue_shaderlib::tests::group_fixture(
            id,
            &[(3, [vec![0xaa; 17], graphics].concat()), (5, compute)],
            blob.len() as u32,
        );
        let fixture = synthetic::container(
            version,
            &[
                (id, Some("Éclairage.ushaderbytecode"), blob),
                (lib, None, header),
            ],
            "mixed",
            false,
            u64::MAX,
        );
        fixture.write(&input.join("global.utoc"));
        let mut report = Vec::new();
        let totals = scan(&input, &output, true, &mut report).unwrap();
        assert_eq!(
            (
                totals.files,
                totals.accepted,
                totals.unique,
                totals.translated
            ),
            (2, 2, 2, 2)
        );
        let events = manifest(&output);
        let programs: Vec<_> = events.iter().filter(|e| e["kind"] == "program").collect();
        assert_eq!(programs[0]["offset"], 17);
        assert_eq!(programs[1]["offset"], 0);
        assert_eq!(programs[1]["unreal"]["shader"]["offset"], 177);
        assert_eq!(programs[0]["source_file"], "global.utoc");
        assert_eq!(
            programs[0]["iostore"]["entry_name"],
            "Shaders/Éclairage.ushaderbytecode"
        );
        assert_eq!(programs[0]["iostore"]["logical_offset"], 9);
        assert!(events.iter().any(|e| e["kind"] == "iostore_data"));
        assert!(fs::read_dir(&output).unwrap().all(|e| !e
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains("partial")));
    }
}

#[test]
fn iostore_partitioned_partial_blocks_and_aes() {
    for encrypted in [false, true] {
        let dir = TestDir::new();
        let input = dir.0.join("input");
        let output = dir.0.join("output");
        fs::create_dir(&input).unwrap();
        let first = record(headers::Stage::Fragment, &bundle([MOV, EXIT, NOP]));
        let second = record(headers::Stage::Compute, &bundle([MOV, EXIT, NOP]));
        let fixture = described_container(
            8,
            &[
                (
                    synthetic::code_id(8, 1),
                    Some("first.ushaderbytecode"),
                    first.clone(),
                ),
                (
                    synthetic::code_id(8, 2),
                    Some("second.ushaderbytecode"),
                    second.clone(),
                ),
            ],
            "mixed",
            encrypted,
            160,
        );
        let path = input.join("chunk.utoc");
        fixture.write(&path);
        assert!(fixture.data.len() > 1);
        let key = synthetic::key();
        let toc = Toc::open(&path, Some(&key)).unwrap();
        for (chunk, expected) in toc.chunks.iter().zip([first, second]) {
            let mut actual = Vec::new();
            toc.extract(&path, chunk, Some(&key), &mut actual).unwrap();
            assert_eq!(actual, expected);
        }
        let totals = scan_options(
            &input,
            &output,
            false,
            false,
            Some(synthetic::key()),
            &mut Vec::new(),
        )
        .unwrap();
        assert_eq!((totals.accepted, totals.unique), (2, 2));
        if encrypted {
            scan(&input, &output, false, &mut Vec::new()).unwrap();
            let events = manifest(&output);
            assert!(events.iter().any(|e| e["kind"] == "iostore_skipped"
                && e["reason"].as_str().unwrap().contains("encrypted")));
            assert!(!events.iter().any(|e| e["kind"] == "program"));
            let wrong =
                super::super::super::pak::AesKey::parse("18".repeat(32).as_bytes()).unwrap();
            scan_options(&input, &output, false, false, Some(wrong), &mut Vec::new()).unwrap();
            let events = manifest(&output);
            assert!(events.iter().any(|e| e["kind"] == "iostore_skipped"));
            assert!(!fs::read_to_string(output.join("manifest.jsonl"))
                .unwrap()
                .contains("first.ushaderbytecode"));
        }
    }
}

#[test]
fn iostore_shader_library_groups_have_secondary_compression() {
    for codec in ["None", "Zlib", "Zstd", "LZ4"] {
        let dir = TestDir::new();
        let input = dir.0.join("input");
        let output = dir.0.join("output");
        fs::create_dir(&input).unwrap();
        let program = record(headers::Stage::Fragment, &bundle([MOV, EXIT, NOP]));
        let id = synthetic::code_id(8, 1);
        let mut library_id = synthetic::code_id(8, 2);
        library_id[11] = 8;
        let data = synthetic::compressed(codec, &program);
        let library = synthetic::library(id, data.len() as u32, program.len() as u32);
        // Code before library proves discovery does not depend on TOC order.
        synthetic::container(
            8,
            &[
                (id, None, data),
                (
                    library_id,
                    Some("ShaderArchive-Global.ushaderbytecode"),
                    library,
                ),
            ],
            "Zstd",
            false,
            u64::MAX,
        )
        .write(&input.join("archive.utoc"));
        let totals = scan(&input, &output, false, &mut Vec::new()).unwrap();
        assert_eq!(totals.accepted, 1);
        let events = manifest(&output);
        let program = events.iter().find(|e| e["kind"] == "program").unwrap();
        assert_eq!(program["iostore"]["shader_group"]["compression"], codec);
        assert_eq!(program["iostore"]["shader_group"]["uncompressed_size"], 160);
    }
}

#[test]
fn iostore_selection_unknown_instructions_and_oodle_are_reported() {
    let dir = TestDir::new();
    let input = dir.0.join("input");
    let output = dir.0.join("output");
    fs::create_dir(&input).unwrap();
    let program = record(headers::Stage::Fragment, &bundle([MOV, EXIT, NOP]));
    let unknown = record(headers::Stage::Fragment, &bundle([u64::MAX, EXIT, NOP]));
    let code = synthetic::code_id(8, 1);
    let mut external = code;
    external[0] = 2;
    external[11] = 7;
    described_container(
        8,
        &[
            (code, None, unknown),
            (external, Some("regular.asset"), program.clone()),
        ],
        "None",
        false,
        u64::MAX,
    )
    .write(&input.join("selected.utoc"));
    let mut named = code;
    named[0] = 3;
    named[11] = 7;
    described_container(
        8,
        &[(
            named,
            Some("ShaderArchive-Named.ushaderbytecode"),
            crate::ue_shaderlib::tests::loose_fixture(&[(3, program.clone())], "Zlib"),
        )],
        "LZ4",
        false,
        u64::MAX,
    )
    .write(&input.join("named.utoc"));
    described_container(8, &[(code, None, program)], "Oodle", false, u64::MAX)
        .write(&input.join("oodle.utoc"));
    let totals = scan(&input, &output, false, &mut Vec::new()).unwrap();
    assert_eq!((totals.accepted, totals.rejected), (1, 1));
    let events = manifest(&output);
    assert!(
        events
            .iter()
            .any(|e| e["kind"] == "rejected"
                && e["iostore"]["chunk_id"] == crate::iostore::hex(&code))
    );
    assert!(events.iter().any(|e| e["status"] == "not_selected"));
    assert!(events.iter().any(|e| e["reason"]
        .as_str()
        .is_some_and(|s| s.contains("unsupported IoStore compression Oodle"))));
    let totals = scan_options(&input, &output, false, true, None, &mut Vec::new()).unwrap();
    assert_eq!((totals.accepted, totals.rejected), (2, 1));
}

#[test]
fn iostore_missing_truncated_and_symlinked_data_fail_without_partial_scan() {
    let dir = TestDir::new();
    let input = dir.0.join("input");
    let output = dir.0.join("output");
    fs::create_dir(&input).unwrap();
    let program = record(headers::Stage::Fragment, &bundle([MOV, EXIT, NOP]));
    let fixture = synthetic::container(
        8,
        &[(synthetic::code_id(8, 1), Some("a.ushaderbytecode"), program)],
        "Zlib",
        false,
        u64::MAX,
    );
    let path = input.join("bad.utoc");
    fixture.write(&path);
    for mode in 0..3 {
        fixture.write(&path);
        if mode == 0 {
            fs::remove_file(path.with_extension("ucas")).unwrap();
        } else if mode == 1 {
            fs::write(path.with_extension("ucas"), [0; 2]).unwrap();
        } else {
            #[cfg(unix)]
            {
                fs::remove_file(path.with_extension("ucas")).unwrap();
                std::os::unix::fs::symlink(&path, path.with_extension("ucas")).unwrap();
            }
            #[cfg(not(unix))]
            {
                continue;
            }
        }
        let totals = scan(&input, &output, false, &mut Vec::new()).unwrap();
        assert_eq!(totals.accepted, 0);
        assert!(manifest(&output)
            .iter()
            .any(|e| e["kind"] == "iostore_chunk" && e["status"] == "skipped"));
        assert!(fs::read_dir(&output).unwrap().all(|e| !e
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains("partial")));
        if mode == 2 {
            fs::remove_file(path.with_extension("ucas")).unwrap();
        }
    }
}

#[test]
fn iostore_signed_table_layout_and_missing_shader_groups() {
    let dir = TestDir::new();
    let input = dir.0.join("input");
    let output = dir.0.join("output");
    fs::create_dir(&input).unwrap();
    let code = synthetic::code_id(8, 1);
    let mut lib = code;
    lib[0] = 2;
    lib[11] = 8;
    let mut fixture = synthetic::container(
        8,
        &[(
            lib,
            Some("library.ushaderbytecode"),
            synthetic::library(code, 32, 160),
        )],
        "None",
        false,
        u64::MAX,
    );
    let blocks = u32::from_le_bytes(fixture.toc[28..32].try_into().unwrap()) as usize;
    let at = 144 + 22 + 8 + 4 + blocks * 12 + 128;
    let mut signatures = Vec::new();
    synthetic::p32(&mut signatures, 3);
    signatures.extend([0; 6]);
    signatures.extend(vec![0; blocks * 20]);
    fixture.toc.splice(at..at, signatures);
    fixture.toc[80] |= 4;
    fixture.write(&input.join("signed.utoc"));
    scan(&input, &output, false, &mut Vec::new()).unwrap();
    let events = manifest(&output);
    assert!(events
        .iter()
        .any(|e| e["kind"] == "iostore" && e["signature_verification"] == "not_performed"));
    assert!(events.iter().any(|e| e["reason"]
        .as_str()
        .is_some_and(|s| s.contains("cross-container"))));
}

#[test]
fn iostore_key_file_cli_flags_use_existing_pak_key() {
    let dir = TestDir::new();
    let input = dir.0.join("input");
    let output = dir.0.join("output");
    fs::create_dir(&input).unwrap();
    let program = record(headers::Stage::Fragment, &bundle([MOV, EXIT, NOP]));
    described_container(
        8,
        &[(
            synthetic::code_id(8, 1),
            Some("private.ushaderbytecode"),
            program,
        )],
        "Zlib",
        true,
        u64::MAX,
    )
    .write(&input.join("encrypted.utoc"));
    let key_path = dir.0.join("synthetic-key");
    fs::write(&key_path, "37".repeat(32)).unwrap();
    crate::parse_and_run::<AotScan, _>(
        shadowbox::QuietRun,
        &[
            input.to_string_lossy().into_owned(),
            output.to_string_lossy().into_owned(),
            "--pak-aes-key-file".into(),
            key_path.to_string_lossy().into_owned(),
        ],
    )
    .unwrap();
    assert!(manifest(&output).iter().any(|e| e["kind"] == "program"));
    assert!(!fs::read_to_string(output.join("manifest.jsonl"))
        .unwrap()
        .contains(&"37".repeat(32)));
}

#[test]
fn iostore_physical_offsets_above_four_gib_are_not_truncated() {
    let dir = TestDir::new();
    let input = dir.0.join("input");
    let output = dir.0.join("output");
    fs::create_dir(&input).unwrap();
    let program = record(headers::Stage::Fragment, &bundle([MOV, EXIT, NOP]));
    let mut fixture = described_container(
        8,
        &[(synthetic::code_id(8, 1), None, program)],
        "Zstd",
        false,
        u64::MAX,
    );
    let chunks = u32::from_le_bytes(fixture.toc[24..28].try_into().unwrap()) as usize;
    let first_block = 144 + chunks * 22 + 8 + 4;
    let count = u32::from_le_bytes(fixture.toc[28..32].try_into().unwrap()) as usize;
    let base = (1u64 << 32) + 32;
    for i in 0..count {
        let at = first_block + i * 12;
        let mut b = [0; 8];
        b[..5].copy_from_slice(&fixture.toc[at..at + 5]);
        let offset = u64::from_le_bytes(b) + base;
        fixture.toc[at..at + 5].copy_from_slice(&offset.to_le_bytes()[..5]);
    }
    let path = input.join("sparse.utoc");
    fs::write(&path, &fixture.toc).unwrap();
    let mut data = File::create(path.with_extension("ucas")).unwrap();
    data.seek(SeekFrom::Start(base)).unwrap();
    data.write_all(&fixture.data[0]).unwrap();
    drop(data);
    let totals = scan(&input, &output, false, &mut Vec::new()).unwrap();
    assert_eq!(totals.accepted, 1);
}
