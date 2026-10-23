//! Synthetic programs only. Provenance: docs/reference-sources.md
use serde_json::Value;
use shadowbox::{headers::Stage, interface::ShaderInput, Cache, CacheConfig};
use std::{
    fs,
    path::{Path, PathBuf},
    process::{Command, Output},
    sync::atomic::{AtomicUsize, Ordering},
};

struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir()
            .join("shadowbox-aot-prewarm-tests")
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
fn success(output: Output) -> Output {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    output
}
fn record(stage: u32, immediate: u32) -> Vec<u8> {
    // Mesa sm50.rs:243-244,3418-3428: bundle controls and slots;
    // :1938-1951 MOV32I immediate20..52, lanes12..16; :321-328 dst0..8;
    // :3106-3110 EXIT CC.T;
    // :3183-3187 OpNop; :220-235 PT. SPH: sph.rs:112-120,204-215.
    let mut bytes = vec![0; if stage == 0 { 0x100 } else { 0x80 }];
    bytes[..4].copy_from_slice(
        &(if stage == 0 {
            0x1234_5679u32
        } else {
            0x1234_5678
        })
        .to_le_bytes(),
    );
    if stage != 0 {
        bytes[0x30..0x34].copy_from_slice(
            &((stage << 10) | (3 << 5) | if stage == 5 { 2 } else { 1 }).to_le_bytes(),
        );
    }
    if stage == 2 {
        // TCS output count in SPH88..96: Mesa sph.rs:305-312;
        // Ryujinx CodeGen/Spirv/SpirvGenerator.cs:190-192.
        bytes[0x3b] = 3;
    }
    let control = 0x7e0u64 | (0x7e0 << 21) | (0x7e0 << 42);
    bytes.extend(
        [
            control,
            0x0100_0000_0007_f000 | (u64::from(immediate) << 20),
            0xe300_0000_0007_000f,
            0x50b0_0000_0007_0f00,
        ]
        .into_iter()
        .flat_map(u64::to_le_bytes),
    );
    bytes
}
fn manifest(out: &Path) -> Vec<Value> {
    fs::read_to_string(out.join("manifest.jsonl"))
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect()
}

#[test]
fn scans_prewarm_cli_and_library_with_identical_cold_output() {
    let temp = Temp::new();
    let input = temp.0.join("input");
    let out = temp.0.join("out");
    // Exercise exclusion of a generated cache subtree from future scans.
    let cache_dir = input.join("prewarm");
    fs::create_dir_all(&input).unwrap();
    for (name, stage, immediate) in [
        ("c", 0, 1),
        ("v", 1, 2),
        ("f", 5, 3),
        ("duplicate", 5, 3),
        ("g", 4, 4),
        ("tcs", 2, 5),
        ("tes", 3, 6),
    ] {
        fs::write(input.join(name), record(stage, immediate)).unwrap();
    }
    let scan = success(cli(&[
        Path::new("aot-scan"),
        &input,
        &out,
        Path::new("--emit-cache"),
        &cache_dir,
    ]));
    assert!(String::from_utf8_lossy(&scan.stdout).contains("programs/s="));
    let events = manifest(&out);
    let summary = events.last().unwrap();
    assert_eq!(summary["cache_emitted"], 4);
    assert_eq!(summary["missing_state"], 2);
    assert_eq!(summary["unique"], 6);
    let mut order = Vec::new();
    for event in events.iter().filter(|e| e["kind"] == "program") {
        order.push(event["source_file"].clone());
        if event["stage"] == "geometry" || event["stage"] == "tessellation-evaluation" {
            let expected = if event["stage"] == "geometry" {
                serde_json::json!(["geometry_input"])
            } else {
                serde_json::json!([
                    "tessellation.domain",
                    "tessellation.spacing",
                    "tessellation.clockwise"
                ])
            };
            assert_eq!(event["missing_state_keys"], expected);
            assert_eq!(event["cache_status"], "missing_state");
            assert!(event["cache_key"].is_null());
            continue;
        }
        assert_eq!(event["missing_state_keys"], serde_json::json!([]));
        let file = out.join(event["output"].as_str().unwrap());
        let cold = temp.0.join("cold.spv");
        let warm = temp.0.join("warm.spv");
        success(cli(&[
            Path::new("translate"),
            &file,
            Path::new("--header-prefixed"),
            Path::new("-o"),
            &cold,
        ]));
        let hit = success(cli(&[
            Path::new("translate"),
            &file,
            Path::new("--header-prefixed"),
            Path::new("--cache-dir"),
            &cache_dir,
            Path::new("--no-memory-cache"),
            Path::new("-o"),
            &warm,
        ]));
        assert!(String::from_utf8_lossy(&hit.stderr).contains("disk_hits=1 misses=0"));
        assert_eq!(fs::read(&cold).unwrap(), fs::read(&warm).unwrap());
        assert_eq!(
            fs::read(file.with_extension("spv")).unwrap(),
            fs::read(&cold).unwrap()
        );
        let bytes = fs::read(&file).unwrap();
        let (header, code) = bytes.split_at(80);
        let mut padded = header.to_vec();
        padded.resize(128, 0);
        let input = ShaderInput {
            code,
            program_record: &padded,
        };
        let stage = match event["stage"].as_str().unwrap() {
            "compute" => Stage::Compute,
            "vertex" => Stage::Vertex,
            "fragment" => Stage::Fragment,
            "tessellation-control" => Stage::TessellationControl,
            _ => unreachable!(),
        };
        let mut cache = Cache::new(CacheConfig {
            memory: shadowbox::cache::Keep::nothing(),
            disk: shadowbox::cache::Disk::at(cache_dir.clone()),
        });
        let output = Result::from(cache.translate(&input, stage, || {
            panic!("cache hit must do zero translation work")
        }))
        .unwrap();
        assert_eq!(cache.stats.disk_hits, 1);
        assert_eq!(
            output
                .spirv
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>(),
            fs::read(&cold).unwrap()
        );
        // Separate header CLI path must share the same key too.
        let hdr = temp.0.join("header");
        let bin = temp.0.join("code");
        fs::write(&hdr, &padded).unwrap();
        fs::write(&bin, code).unwrap();
        let hit = success(cli(&[
            Path::new("translate"),
            &bin,
            Path::new("--header"),
            &hdr,
            Path::new("--cache-dir"),
            &cache_dir,
            Path::new("-o"),
            &warm,
        ]));
        assert!(String::from_utf8_lossy(&hit.stderr).contains("disk_hits=1 misses=0"));
        let miss = success(cli(&[
            Path::new("translate"),
            &bin,
            Path::new("--header"),
            &hdr,
            Path::new("--target-generation"),
            Path::new("sm6"),
            Path::new("--cache-dir"),
            &cache_dir,
            Path::new("-o"),
            &warm,
        ]));
        // The duplicate has already written its SM6 variant.
        if event["unique"] == true {
            assert!(String::from_utf8_lossy(&miss.stderr).contains("disk_hits=0 misses=1"));
        }
    }
    let next = temp.0.join("next");
    success(cli(&[
        Path::new("aot-scan"),
        &input,
        &next,
        Path::new("--emit-cache"),
        &cache_dir,
    ]));
    let next = manifest(&next);
    assert_eq!(next.last().unwrap()["files"], 7);
    assert_eq!(
        order,
        next.iter()
            .filter(|e| e["kind"] == "program")
            .map(|e| e["source_file"].clone())
            .collect::<Vec<_>>()
    );
    let normalize = |events: &[Value]| {
        events
            .iter()
            .cloned()
            .map(|mut event| {
                if event["kind"] == "summary" {
                    event.as_object_mut().unwrap().remove("translation_seconds");
                    event.as_object_mut().unwrap().remove("programs_per_second");
                }
                event
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(normalize(&events), normalize(&next));
    assert_eq!(
        summary["translation_workers"].as_u64().unwrap() as usize,
        std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(6)
    );
}

#[test]
fn merging_and_stats_preserve_warm_entries_and_reject_corruption() {
    let temp = Temp::new();
    let mut sources = Vec::new();
    for (index, immediate) in [1, 2, 1].into_iter().enumerate() {
        let input = temp.0.join(format!("input{index}"));
        let out = temp.0.join(format!("out{index}"));
        let cache = temp.0.join(format!("cache{index}"));
        fs::create_dir(&input).unwrap();
        fs::write(input.join("program"), record(5, immediate)).unwrap();
        success(cli(&[
            Path::new("aot-scan"),
            &input,
            &out,
            Path::new("--emit-cache"),
            &cache,
        ]));
        sources.push((cache, out));
    }
    let merged = temp.0.join("merged");
    let output = success(cli(&[
        Path::new("cache-merge"),
        &merged,
        &sources[0].0,
        &sources[1].0,
        &sources[2].0,
    ]));
    assert!(String::from_utf8_lossy(&output.stdout).contains("entries=2"));
    let output = success(cli(&[Path::new("cache-stats"), &merged]));
    assert!(String::from_utf8_lossy(&output.stdout).contains("invalid=0"));
    for (_, out) in &sources {
        let event = manifest(out)
            .into_iter()
            .find(|e| e["kind"] == "program")
            .unwrap();
        let file = out.join(event["output"].as_str().unwrap());
        let warm = temp.0.join("warm.spv");
        let hit = success(cli(&[
            Path::new("translate"),
            &file,
            Path::new("--header-prefixed"),
            Path::new("--cache-dir"),
            &merged,
            Path::new("-o"),
            &warm,
        ]));
        assert!(String::from_utf8_lossy(&hit.stderr).contains("disk_hits=1 misses=0"));
        assert_eq!(
            fs::read(file.with_extension("spv")).unwrap(),
            fs::read(warm).unwrap()
        );
    }
    let bad = sources[0].0.join(format!("{}.sbc", "a".repeat(64)));
    fs::write(bad, b"truncated").unwrap();
    let stats = success(cli(&[Path::new("cache-stats"), &sources[0].0]));
    assert!(String::from_utf8_lossy(&stats.stdout).contains("invalid=1"));
    let rejected = temp.0.join("rejected");
    assert!(!cli(&[
        Path::new("cache-merge"),
        &rejected,
        &sources[1].0,
        &sources[0].0
    ])
    .status
    .success());
    assert!(!rejected.exists());
}

#[test]
fn cache_verify_reports_each_category_and_does_not_modify_entries() {
    use sha2::{Digest, Sha256};
    let temp = Temp::new();
    let input = temp.0.join("input");
    let out = temp.0.join("out");
    let cache = temp.0.join("cache");
    fs::create_dir(&input).unwrap();
    fs::write(input.join("program"), record(5, 1)).unwrap();
    success(cli(&[
        Path::new("aot-scan"),
        &input,
        &out,
        Path::new("--emit-cache"),
        &cache,
    ]));
    let clean = success(cli(&[Path::new("cache-verify"), &cache]));
    assert!(
        String::from_utf8_lossy(&clean.stdout).contains("valid=1 stale=0 mismatched=0 corrupt=0")
    );
    let path = fs::read_dir(&cache)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    let current = fs::read(&path).unwrap();
    // Project envelope fields: docs/reference-sources.md.
    let mut expected = Vec::new();
    for (tag, status) in [('a', "stale"), ('b', "mismatched"), ('c', "corrupt")] {
        let key = tag.to_string().repeat(64);
        let file = format!("{key}.sbc");
        let mut bytes = current.clone();
        bytes[8..72].copy_from_slice(key.as_bytes());
        match status {
            "stale" => {
                bytes[..8].copy_from_slice(b"SBOXC007");
                bytes.drain(105..172);
            }
            "mismatched" => bytes[105..169].fill(b'0'),
            // Bits 0 and 1 are requirement flags; any higher bit is corrupt.
            _ => bytes[104] = 0x80,
        }
        let checksum = Sha256::digest(&bytes[104..]);
        bytes[72..104].copy_from_slice(&checksum);
        fs::write(cache.join(&file), &bytes).unwrap();
        expected.push((file, status, bytes));
    }
    let report = cli(&[Path::new("cache-verify"), &cache]);
    assert!(!report.status.success());
    let text = String::from_utf8_lossy(&report.stdout);
    assert!(
        text.contains("valid=1 stale=1 mismatched=1 corrupt=1"),
        "{text}"
    );
    for (file, status, bytes) in expected {
        assert!(text.contains(&format!("{status}\t{file}")), "{text}");
        assert_eq!(fs::read(cache.join(file)).unwrap(), bytes);
    }
    assert_eq!(fs::read(path).unwrap(), current);
    assert!(!cli(&[Path::new("cache-verify"), &temp.0.join("missing")])
        .status
        .success());
}
