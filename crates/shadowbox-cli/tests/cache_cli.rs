//! Synthetic caches only. See docs/reference-sources.md
use sha2::{Digest, Sha256};

#[path = "../../shadowbox/tests/support/spirv_validation.rs"]
mod spirv_validation;
#[path = "../../shadowbox/tests/support/stack_execution.rs"]
mod stack_execution;

fn with_cb1(record: &[u8], cb1: &[u8]) -> Vec<u8> {
    shadowbox::encode_header_prefixed_with_constant_buffer1(
        record,
        shadowbox::TargetGeneration::Sm5,
        cb1,
    )
    .unwrap()
}
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicUsize, Ordering},
};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        // Keep generated inputs and subprocess outputs under TMPDIR.
        let path = std::env::temp_dir()
            .join("shadowbox-cache-cli-tests")
            .join(format!(
                "{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
        fs::create_dir_all(&path).unwrap();
        Self(path)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn cli(args: &[&Path]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_shadowbox"))
        .args(args)
        .output()
        .unwrap()
}
fn pack(bytes: &[u8], algorithm: u8) -> Vec<u8> {
    // BinarySerializer.cs:227-245; CompressionAlgorithm.cs:6-21.
    let mut packed = vec![algorithm];
    match algorithm {
        0 => packed.extend_from_slice(bytes),
        1 => {
            let mut encoder =
                flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::fast());
            encoder.write_all(bytes).unwrap();
            packed.extend(encoder.finish().unwrap());
        }
        2 => {
            let mut encoder = brotli::CompressorWriter::new(Vec::new(), 4096, 1, 22);
            encoder.write_all(bytes).unwrap();
            packed.extend(encoder.into_inner());
        }
        _ => unreachable!(),
    }
    packed
}
fn code(unknown: bool) -> Vec<u8> {
    // Mesa sm50.rs:3418-3428: scheduling word, then instruction slots.
    // EXIT opcode and CC.T: sm50.rs:3106-3110; PT: sm50.rs:208-235.
    [
        0_u32,
        0,
        if unknown { 0x70000 } else { 0x7000f },
        if unknown { 0x12340000 } else { 0xe3000000 },
    ]
    .into_iter()
    .flat_map(u32::to_le_bytes)
    .collect()
}
fn graphics(stage: u32, unknown: bool) -> Vec<u8> {
    // Mesa sph.rs:18 and cla097sph.h:37-42,407-412.
    let mut bytes = vec![0; 80];
    bytes[..4].copy_from_slice(&(stage << 10).to_le_bytes());
    bytes.extend(code(unknown));
    bytes
}
fn specialization(compute: bool) -> Vec<u8> {
    // ShaderSpecializationState.cs:31-41,974-1021. Zero queried flags,
    // zero-sized extensible state, no CB uses, pipeline, or textures.
    // BinarySerializer.cs:81-87 permits smaller structures.
    let mut data = vec![0; 4];
    data.push(u8::from(compute));
    data.extend_from_slice(if compute { b"COMS" } else { b"GFXS" });
    data.extend(0_i32.to_le_bytes());
    data.extend([0, 0, 0, 0]); // CB mask byte, pipeline bool, texture count u16.
    data
}
fn cache(root: &Path, guests: &[(Vec<u8>, Vec<u8>, u8)], programs: &[(u32, Vec<i32>, u8)]) {
    fs::create_dir_all(root).unwrap();
    let mut toc = vec![0; 32];
    // DiskCacheGuestStorage.cs:14-18,28-85,391-413.
    toc[..4].copy_from_slice(b"TOCG");
    toc[4..8].copy_from_slice(&0x10001_u32.to_le_bytes());
    toc[12..16].copy_from_slice(&(guests.len() as u32).to_le_bytes());
    let mut data = vec![];
    for (code, cb, algorithm) in guests {
        for field in [data.len() as u32, code.len() as u32, cb.len() as u32, 0] {
            toc.extend(field.to_le_bytes());
        }
        // Hash is informational to this reader. Synthetic entries use zero.
        data.extend(cb);
        data.extend(pack(code, *algorithm));
    }
    fs::write(root.join("guest.toc"), toc).unwrap();
    fs::write(root.join("guest.data"), data).unwrap();
    // DiskCacheHostStorage.cs:16-28,37-68,592-619.
    let mut toc = vec![0; 32];
    toc[..4].copy_from_slice(b"TOCS");
    toc[4..8].copy_from_slice(&0x10002_u32.to_le_bytes());
    // Deliberately old codegen version: it gates host binaries, not guest data
    // (DiskCacheHostStorage.cs:325,379-424).
    toc[8..12].copy_from_slice(&42_u32.to_le_bytes());
    let mut data = vec![];
    for (mask, indices, algorithm) in programs {
        toc.extend((data.len() as u64).to_le_bytes());
        let mut entry = mask.to_le_bytes().to_vec();
        for index in indices {
            entry.extend(index.to_le_bytes());
        }
        entry.extend(specialization(*mask == 0));
        data.extend(pack(&entry, *algorithm));
    }
    fs::write(root.join("shared.toc"), toc).unwrap();
    fs::write(root.join("shared.data"), data).unwrap();
}
fn extract(root: &Path, out: &Path) -> Output {
    cli(&[Path::new("cache-extract"), root, out])
}
fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
fn simple(root: &Path) {
    cache(
        root,
        &[(graphics(5, false), vec![9, 8, 7, 6], 0)],
        &[(32, vec![0], 0)],
    );
}

#[test]
fn all_compressions_graphics_compute_and_duplicate_code() {
    for guest_algorithm in 0..=2 {
        for shared_algorithm in 0..=2 {
            let temp = Temp::new();
            let root = temp.0.join("cache");
            let out = temp.0.join("out");
            let fragment = graphics(5, false);
            let compute = code(false);
            cache(
                &root,
                &[
                    (fragment.clone(), vec![1, 2, 3], guest_algorithm),
                    (compute.clone(), vec![], guest_algorithm),
                    (fragment.clone(), vec![4, 5], guest_algorithm),
                ],
                &[
                    (32, vec![0], shared_algorithm),
                    (0, vec![1], shared_algorithm),
                    (32, vec![2], shared_algorithm),
                ],
            );
            assert_success(&extract(&root, &out));
            let first = with_cb1(&fragment, &[1, 2, 3]);
            let fragment_name = format!("{:x}.bin", Sha256::digest(&first));
            assert_eq!(fs::read(out.join(&fragment_name)).unwrap(), first);
            let mut record = vec![0; 80];
            record.extend(compute);
            let record = with_cb1(&record, &[]);
            assert_eq!(
                fs::read(out.join(format!("{:x}.bin", Sha256::digest(&record)))).unwrap(),
                record
            );
            let manifest = fs::read_to_string(out.join("manifest.tsv")).unwrap();
            assert_eq!(manifest.lines().count(), 4);
            assert!(manifest.contains("\tcompute\t"));
            assert_eq!(manifest.matches(&fragment_name).count(), 1);
            let variant_name = format!("{:x}.bin", Sha256::digest(with_cb1(&fragment, &[4, 5])));
            assert_ne!(fragment_name, variant_name);
            assert_eq!(manifest.matches(&variant_name).count(), 1);
            for line in manifest.lines().skip(1) {
                let fields: Vec<_> = line.split('\t').collect();
                assert_eq!(
                    fs::read(out.join(fields[3])).unwrap().len(),
                    fields[6].parse::<usize>().unwrap()
                );
                let bytes = fs::read(out.join(fields[3])).unwrap();
                let shader =
                    shadowbox::header_prefixed_input(&bytes, shadowbox::TargetGeneration::Sm5)
                        .unwrap();
                assert_eq!(shader.code.len(), fields[5].parse::<usize>().unwrap());
                let cb = fs::read(out.join(fields[7])).unwrap();
                let header_and_code = &bytes[..80 + shader.code.len()];
                assert_eq!(bytes, with_cb1(header_and_code, &cb));
                assert_eq!(
                    cb,
                    [vec![1, 2, 3], vec![], vec![4, 5]][fields[9].parse::<usize>().unwrap()]
                );
                let shared = fs::read(out.join(fields[11])).unwrap();
                assert_eq!(shared.len(), fields[12].parse::<usize>().unwrap());
                assert_eq!(&shared[8..], specialization(fields[2] == "compute"));
            }
            assert_eq!(
                fs::read(out.join("shared.toc")).unwrap(),
                fs::read(root.join("shared.toc")).unwrap()
            );
        }
    }
}
#[test]
fn all_graphics_slots_remain_distinct_and_tessellation_is_kept() {
    let temp = Temp::new();
    let root = temp.0.join("cache");
    let out = temp.0.join("out");
    let guests = (0..6)
        .map(|slot| (graphics(if slot == 0 { 1 } else { slot }, false), vec![], 2))
        .collect::<Vec<_>>();
    cache(&root, &guests, &[(0x3f, (0..6).collect(), 1)]);
    assert_success(&extract(&root, &out));
    let manifest = fs::read_to_string(out.join("manifest.tsv")).unwrap();
    assert_eq!(manifest.lines().count(), 7);
    for stage in [
        "vertex-a",
        "vertex",
        "tessellation-control",
        "tessellation-evaluation",
        "geometry",
        "fragment",
    ] {
        assert!(manifest.contains(&format!("\t{stage}\t")));
    }
    let report = cli(&[Path::new("cache-report"), &out]);
    assert_success(&report);
    // Identical vertex-a/vertex bytes share a binary while retaining both manifest rows.
    let text = String::from_utf8_lossy(&report.stdout);
    assert!(
        text.contains("total=5 translated=2 translate_failed=3 validated=2 validate_failed=0"),
        "{text}"
    );
    for missing_state in [
        "tessellation output control point count must be 1..32",
        "tessellation domain, spacing and vertex order must be supplied by the driver",
        "geometry input topology must be supplied by the driver",
    ] {
        assert!(text.contains(missing_state), "{text}");
    }
}
#[test]
fn report_runs_real_translation_and_validator_and_counts_first_failures() {
    let temp = Temp::new();
    let root = temp.0.join("cache");
    let out = temp.0.join("out");
    cache(
        &root,
        &[
            (graphics(5, false), vec![], 2),
            (graphics(5, true), vec![], 1),
            (code(false), vec![], 0),
        ],
        &[(32, vec![0], 2), (32, vec![1], 1), (0, vec![2], 0)],
    );
    assert_success(&extract(&root, &out));
    let report = cli(&[Path::new("cache-report"), &out]);
    assert_success(&report);
    let text = String::from_utf8_lossy(&report.stdout);
    assert!(
        text.contains("total=3 translated=2 translate_failed=1 validated=2 validate_failed=0"),
        "{text}"
    );
    assert!(
        text.contains("1\ttranslate: unsupported instruction:"),
        "{text}"
    );
    assert!(text.contains("UNKNOWN 0x1234"), "{text}");
    assert!(!fs::read_dir(&out).unwrap().any(|e| e
        .unwrap()
        .file_name()
        .to_string_lossy()
        .starts_with(".cache-report")));
}
#[test]
fn malformed_caches_fail_before_output_is_created() {
    for case in 0..12 {
        let temp = Temp::new();
        let root = temp.0.join("cache");
        let out = temp.0.join("out");
        simple(&root);
        let (file, offset, bytes): (&str, usize, Vec<u8>) = match case {
            0 => ("guest.toc", 0, b"BAD!".to_vec()),
            1 => ("guest.toc", 4, 0x10002_u32.to_le_bytes().to_vec()),
            2 => ("shared.toc", 4, 0x10003_u32.to_le_bytes().to_vec()),
            3 => ("guest.toc", 32, u32::MAX.to_le_bytes().to_vec()),
            4 => ("guest.toc", 36, 8_u32.to_le_bytes().to_vec()),
            5 => ("guest.toc", 40, u32::MAX.to_le_bytes().to_vec()),
            6 => ("guest.data", 4, vec![9]),
            7 => ("shared.data", 0, vec![9]),
            8 => ("shared.data", 1, 64_u32.to_le_bytes().to_vec()),
            9 => ("shared.data", 5, (-1_i32).to_le_bytes().to_vec()),
            10 => ("shared.data", 5, 1_i32.to_le_bytes().to_vec()),
            _ => ("guest.data", 5, (1_u32 << 10).to_le_bytes().to_vec()),
        };
        let path = root.join(file);
        let mut data = fs::read(&path).unwrap();
        data[offset..offset + bytes.len()].copy_from_slice(&bytes);
        fs::write(path, data).unwrap();
        let result = extract(&root, &out);
        assert!(!result.status.success(), "case {case}");
        assert!(!out.exists());
    }
}
#[test]
fn truncation_and_short_graphics_code_are_rejected() {
    for algorithm in 0..=2 {
        for file in ["guest.toc", "shared.toc", "guest.data", "shared.data"] {
            let temp = Temp::new();
            let root = temp.0.join("cache");
            let out = temp.0.join("out");
            cache(
                &root,
                &[(graphics(5, false), vec![], algorithm)],
                &[(32, vec![0], algorithm)],
            );
            let path = root.join(file);
            let mut bytes = fs::read(&path).unwrap();
            bytes.truncate(if file.ends_with("toc") {
                bytes.len() - 1
            } else {
                2
            });
            fs::write(path, bytes).unwrap();
            assert!(
                !extract(&root, &out).status.success(),
                "{file} algorithm {algorithm}"
            );
            assert!(!out.exists());
        }
    }
    let temp = Temp::new();
    let root = temp.0.join("cache");
    let out = temp.0.join("out");
    cache(&root, &[(code(false), vec![], 0)], &[(32, vec![0], 0)]);
    assert!(!extract(&root, &out).status.success());
}
#[test]
fn empty_cache_and_nonempty_destination() {
    let temp = Temp::new();
    let root = temp.0.join("cache");
    let out = temp.0.join("out");
    cache(&root, &[], &[]);
    assert_success(&extract(&root, &out));
    let report = cli(&[Path::new("cache-report"), &out]);
    assert_success(&report);
    assert!(String::from_utf8_lossy(&report.stdout).contains("total=0 translated=0"));
    let before = fs::read(out.join("manifest.tsv")).unwrap();
    assert!(!extract(&root, &out).status.success());
    assert_eq!(fs::read(out.join("manifest.tsv")).unwrap(), before);
}
#[cfg(unix)]
#[test]
fn report_validation_failure_and_missing_validator() {
    use std::os::unix::fs::PermissionsExt;
    let temp = Temp::new();
    let root = temp.0.join("cache");
    let out = temp.0.join("out");
    simple(&root);
    assert_success(&extract(&root, &out));
    let bin = temp.0.join("bin");
    fs::create_dir(&bin).unwrap();
    let validator = bin.join("spirv-val");
    fs::write(&validator, "#!/bin/sh\nif [ \"$1\" = --version ]; then exit 0; fi\nif [ \"$1\" != --target-env ] || [ \"$2\" != vulkan1.2 ]; then exit 2; fi\necho 'synthetic validation blocker' >&2\nexit 1\n").unwrap();
    fs::set_permissions(&validator, fs::Permissions::from_mode(0o700)).unwrap();
    let invoke = || {
        Command::new(env!("CARGO_BIN_EXE_shadowbox"))
            .arg("cache-report")
            .arg(&out)
            .env("PATH", &bin)
            .output()
            .unwrap()
    };
    let result = invoke();
    assert_success(&result);
    let text = String::from_utf8_lossy(&result.stdout);
    assert!(
        text.contains("translated=1 translate_failed=0 validated=0 validate_failed=1"),
        "{text}"
    );
    assert!(text.contains("1\tvalidate: synthetic validation blocker"));
    fs::remove_file(validator).unwrap();
    let result = invoke();
    assert!(!result.status.success());
    assert!(String::from_utf8_lossy(&result.stderr).contains("run spirv-val"));
}
#[test]
fn compressed_end_markers_cannot_be_truncated() {
    for algorithm in 1..=2 {
        for file in ["guest.data", "shared.data"] {
            let temp = Temp::new();
            let root = temp.0.join("cache");
            let out = temp.0.join("out");
            cache(
                &root,
                &[(graphics(5, false), vec![], algorithm)],
                &[(32, vec![0], algorithm)],
            );
            let path = root.join(file);
            let mut bytes = fs::read(&path).unwrap();
            bytes.pop();
            fs::write(path, bytes).unwrap();
            let result = extract(&root, &out);
            assert!(
                !result.status.success(),
                "accepted missing end marker: {file}, algorithm {algorithm}"
            );
            assert!(!out.exists());
        }
    }
}
#[test]
fn report_groups_only_the_first_blocker() {
    let temp = Temp::new();
    let root = temp.0.join("cache");
    let out = temp.0.join("out");
    let mut fragment = graphics(5, true);
    // A second invented instruction must not create a second histogram entry.
    fragment.extend(
        [0x70000_u32, 0x56780000]
            .into_iter()
            .flat_map(u32::to_le_bytes),
    );
    let mut vertex = fragment.clone();
    vertex[..4].copy_from_slice(&(1_u32 << 10).to_le_bytes());
    cache(
        &root,
        &[(fragment, vec![], 1), (vertex, vec![], 2)],
        &[(32, vec![0], 1), (2, vec![1], 2)],
    );
    assert_success(&extract(&root, &out));
    let result = cli(&[Path::new("cache-report"), &out]);
    assert_success(&result);
    let text = String::from_utf8_lossy(&result.stdout);
    assert!(
        text.contains("total=2 translated=0 translate_failed=2 validated=0 validate_failed=0"),
        "{text}"
    );
    assert!(
        text.contains("2\ttranslate: unsupported instruction:"),
        "{text}"
    );
    assert!(!text.contains("5678"), "{text}");
}
#[test]
fn partial_instruction_words_are_rejected_without_losing_bytes() {
    for algorithm in 0..=2 {
        let temp = Temp::new();
        let root = temp.0.join("cache");
        let out = temp.0.join("out");
        let mut partial = code(false);
        partial.push(0xab);
        cache(
            &root,
            &[(partial, vec![], algorithm)],
            &[(0, vec![0], algorithm)],
        );
        let result = extract(&root, &out);
        assert!(!result.status.success());
        assert!(String::from_utf8_lossy(&result.stderr).contains("partial instruction word"));
        assert!(!out.exists());
    }
}

#[test]
fn extracted_cb1_tables_reach_cli_library_and_cache_without_decoding_data_islands() {
    // Synthetic switch, including data islands that remain Unknown without a
    // table proof. Encodings: Mesa nak/sm50.rs:1462-1490 (IMNMX),1702-1721
    // (SHL),1931-1951 (MOV),2633-2648 (LDC),2652-2671 (STS),3032-3052
    // (SSY/SYNC),3106-3110 (EXIT); envytools gm107.c:1943 (BRX).
    let address = |slot: usize| (slot / 3 * 32 + slot % 3 * 8 + 8) as u32;
    let op = |opcode: u64| opcode << 48 | 7 << 16;
    let ssy_offset = address(13) - address(0) - 8;
    let slots = [
        op(0xe290) | u64::from(ssy_offset) << 20 | 15,
        op(0xef94) | 1 | 0xff << 8,
        op(0x3820) | 2 | 1 << 8 | 1 << 20 | 7 << 39,
        op(0x3848) | 2 | 2 << 8 | 2 << 20,
        op(0xef94) | 3 | 2 << 8 | 8 << 20 | 1 << 36,
        op(0xe250) | 3 << 8 | 15,
        op(0x1234),
        op(0x0100) | 15 << 12 | 10 << 20,
        op(0xf0f8) | 15,
        op(0x1234),
        op(0x0100) | 15 << 12 | 20 << 20,
        op(0xf0f8) | 15,
        op(0x1234),
        op(0xef5c) | 0xff << 8,
        op(0xe300) | 15,
    ];
    let mut code = Vec::new();
    for bundle in slots.chunks(3) {
        code.extend(0_u64.to_le_bytes());
        code.extend(bundle.iter().flat_map(|slot| slot.to_le_bytes()));
    }
    let base = address(5) + 8;
    let first = [0xfeed, 0xbeef, address(7) - base, address(10) - base];
    let second = [0xfeed, 0xbeef, first[3], first[2]];
    let bytes = |words: &[u32]| words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let temp = Temp::new();
    let root = temp.0.join("cache");
    let out = temp.0.join("out");
    cache(
        &root,
        &[
            (code.clone(), bytes(&first), 1),
            (code.clone(), bytes(&second), 2),
        ],
        &[(0, vec![0], 2), (0, vec![1], 1)],
    );
    assert_success(&extract(&root, &out));
    let manifest = fs::read_to_string(out.join("manifest.tsv")).unwrap();
    // Identical guest code with different immutable tables stays associated
    // with two records, even when translation later yields identical modules.
    let rows: Vec<_> = manifest
        .lines()
        .skip(1)
        .map(|line| line.split('\t').collect::<Vec<_>>())
        .collect();
    assert_eq!(rows.len(), 2);
    assert_ne!(rows[0][3], rows[1][3]);
    assert_ne!(rows[0][7], rows[1][7]);
    let mut translation_cache = shadowbox::Cache::new(shadowbox::CacheConfig {
        memory: shadowbox::cache::Keep::outputs(),
        disk: shadowbox::cache::Disk::nowhere(),
    });
    let mut outputs = Vec::new();
    for (line, cb1) in manifest.lines().skip(1).zip([first, second]) {
        let fields: Vec<_> = line.split('\t').collect();
        let path = out.join(fields[3]);
        let record = fs::read(&path).unwrap();
        let input =
            shadowbox::header_prefixed_input(&record, shadowbox::TargetGeneration::Sm5).unwrap();
        assert_eq!(input.code, code);
        let direct = Result::from(shadowbox::translate_header_prefixed(
            shadowbox::QuietRun,
            &record,
        ))
        .unwrap();
        let borrowed = Result::from(shadowbox::translate(shadowbox::QuietRun, &input)).unwrap();
        assert_eq!(direct.spirv, borrowed.spirv);
        let cached = Result::from(shadowbox::translate_header_prefixed_cached(
            shadowbox::QuietRun,
            &record,
            shadowbox::CacheContext::default(),
            &mut translation_cache,
        ))
        .unwrap();
        assert_eq!(cached.spirv, direct.spirv);
        let spv_path = temp.0.join("result.spv");
        let disk_cache = temp.0.join("translated-cache");
        assert_success(&cli(&[
            Path::new("translate"),
            &path,
            Path::new("--header-prefixed"),
            Path::new("--stage"),
            Path::new("compute"),
            Path::new("--cache-dir"),
            &disk_cache,
            Path::new("-o"),
            &spv_path,
        ]));
        let cli_words: Vec<_> = fs::read(&spv_path)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| u32::from_le_bytes(*word))
            .collect();
        assert_eq!(cli_words, direct.spirv);
        let mut legacy = vec![0; 80];
        legacy.extend(&code);
        let legacy_path = temp.0.join("legacy.bin");
        let cb1_path = temp.0.join("explicit.cb1");
        fs::write(&legacy_path, &legacy).unwrap();
        fs::write(&cb1_path, bytes(&cb1)).unwrap();
        for source in [&legacy_path, &path] {
            assert_success(&cli(&[
                Path::new("translate"),
                source,
                Path::new("--header-prefixed"),
                Path::new("--cb1"),
                &cb1_path,
                Path::new("-o"),
                &spv_path,
            ]));
            assert_eq!(
                fs::read(&spv_path).unwrap(),
                direct
                    .spirv
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect::<Vec<_>>()
            );
        }
        let mut conflicting = cb1;
        conflicting[0] ^= 1;
        fs::write(&cb1_path, bytes(&conflicting)).unwrap();
        let failed = cli(&[
            Path::new("translate"),
            &path,
            Path::new("--header-prefixed"),
            Path::new("--cb1"),
            &cb1_path,
            Path::new("-o"),
            &spv_path,
        ]);
        assert!(!failed.status.success());
        assert!(String::from_utf8_lossy(&failed.stderr).contains("disagrees"));
        assert_success(&spirv_validation::validate(&cli_words));
        for index in [0, 1, 2, u32::MAX] {
            let target = cb1[2 + index.min(1) as usize] + base;
            let expected = if target == address(7) { 10 } else { 20 };
            assert_eq!(
                stack_execution::run_with_constants(&cli_words, &[(0, &[index]), (1, &cb1)]),
                (expected, 0),
            );
        }
        outputs.push(direct.spirv);
    }
    // Both tables have the same target set, so their modules may be shared
    // byte for byte. The snapshot still changes cache identity and execution.
    assert_eq!(outputs[0], outputs[1]);
    assert_eq!(translation_cache.stats.misses, 2);
    let report = cli(&[Path::new("cache-report"), &out]);
    assert_success(&report);
    assert!(String::from_utf8_lossy(&report.stdout)
        .contains("total=2 translated=2 translate_failed=0 validated=2 validate_failed=0"));
    // Removing the snapshot must retain Unknown data, never replace it with NOP.
    let mut legacy = vec![0; 80];
    legacy.extend(&code);
    let error = Result::from(shadowbox::translate_header_prefixed(
        shadowbox::QuietRun,
        &legacy,
    ))
    .unwrap_err()
    .to_string();
    assert!(error.contains("unsupported instruction"), "{error}");
    // A guarded BRX can execute its fallthrough data word, so even a
    // snapshot must not hide it. Ryujinx Decoder.cs:157-162,501-515.
    let mut guarded = legacy.clone();
    let at = 80 + address(5) as usize;
    let raw = u64::from_le_bytes(guarded[at..at + 8].try_into().unwrap());
    guarded[at..at + 8].copy_from_slice(&(raw & !(15 << 16)).to_le_bytes());
    assert!(Result::from(shadowbox::translate_header_prefixed(
        shadowbox::QuietRun,
        &with_cb1(&guarded, &bytes(&first))
    ))
    .is_err());
    assert!(Result::from(shadowbox::translate_header_prefixed(
        shadowbox::QuietRun,
        &with_cb1(&legacy, &bytes(&first[..3]))
    ))
    .is_err());
    // Constant-result baseline for the first switch case, with the same
    // MOV/STS/EXIT encodings cited above, requires no bound buffers.
    let mut baseline = vec![0; 80];
    for raw in [
        0,
        op(0x0100) | 15 << 12 | 10 << 20,
        op(0xef5c) | 0xff << 8,
        op(0xe300) | 15,
    ] {
        baseline.extend(raw.to_le_bytes());
    }
    assert_eq!(
        stack_execution::run(
            &Result::from(shadowbox::translate_header_prefixed(
                shadowbox::QuietRun,
                &baseline
            ))
            .unwrap()
            .spirv
        ),
        (10, 0)
    );
}
