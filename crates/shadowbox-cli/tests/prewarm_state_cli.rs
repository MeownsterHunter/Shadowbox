//! Project-owned synthetic cache records.
use serde_json::Value;
use shadowbox::{
    interface::{GeometryInput, TessellationDomain, TessellationSpacing, TessellationState},
    Cache, CacheConfig, CacheContext,
};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    sync::atomic::{AtomicUsize, Ordering},
};

/// A program's identity under `context`, through [`shadowbox::Program`].
fn identity(
    input: &shadowbox::interface::ShaderInput<'_>,
    context: shadowbox::CacheContext,
) -> Result<String, shadowbox::interface::Stopped> {
    Ok(shadowbox::Program::parse(input, context.graphics_state)?
        .identity(context.generation, context.options))
}
struct Temp(PathBuf);
impl Temp {
    fn new() -> Self {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir()
            .join("shadowbox-prewarm-state-tests")
            .join(format!(
                "{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, Ordering::Relaxed)
            ));
        fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}
fn cli(args: &[&Path]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_shadowbox"))
        .args(args)
        .output()
        .unwrap()
}
fn success(args: &[&Path]) -> String {
    let output = cli(args);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}
fn record(stage: u32) -> Vec<u8> {
    // SPH type/version/stage/count/output: Mesa nak/sph.rs:112-120,204-215,
    // 305-312,326-350. EXIT/PT/NOP/bundles: sm50.rs:220-235,3106-3110,
    // 3183-3187,3418-3428. The same code in all stages tests stage separation.
    let mut bytes = vec![0; 80];
    if stage != 0 {
        bytes[..4].copy_from_slice(
            &((stage << 10) | (3 << 5) | if stage == 5 { 2 } else { 1 }).to_le_bytes(),
        );
    }
    if stage == 2 || stage == 4 {
        bytes[11] = 1;
    }
    if stage == 4 {
        bytes[15] = 1;
        bytes[16] = 1;
    }
    bytes.extend(
        [
            0x7e0u64 | (0x7e0 << 21) | (0x7e0 << 42),
            0xe300_0000_0007_000f,
            0x50b0_0000_0007_0f00,
            0x50b0_0000_0007_0f00,
        ]
        .into_iter()
        .flat_map(u64::to_le_bytes),
    );
    bytes
}
fn pack(bytes: &[u8], tag: u8) -> Vec<u8> {
    let mut out = vec![tag];
    match tag {
        0 => out.extend(bytes),
        1 => {
            let mut w = flate2::write::DeflateEncoder::new(Vec::new(), flate2::Compression::fast());
            w.write_all(bytes).unwrap();
            out.extend(w.finish().unwrap());
        }
        2 => {
            let mut w = brotli::CompressorWriter::new(Vec::new(), 4096, 1, 22);
            w.write_all(bytes).unwrap();
            out.extend(w.into_inner());
        }
        _ => unreachable!(),
    }
    out
}
// (stage slot, topology, tessellation mode, native graphics-state length).
fn fixture(dir: &Path, states: &[(u32, u32, u32, usize)]) {
    // Headers/entries: DiskCacheGuestStorage.cs:28-85,395-413;
    // DiskCacheHostStorage.cs:37-68,592-618. Specialization envelope/tail:
    // ShaderSpecializationState.cs:974-1021; BinarySerializer.cs:111-116.
    fs::create_dir_all(dir).unwrap();
    let mut guest_toc = vec![0; 32];
    guest_toc[..4].copy_from_slice(b"TOCG");
    guest_toc[4..8].copy_from_slice(&0x10001u32.to_le_bytes());
    guest_toc[12..16].copy_from_slice(&(states.len() as u32).to_le_bytes());
    let mut shared_toc = vec![0; 32];
    shared_toc[..4].copy_from_slice(b"TOCS");
    shared_toc[4..8].copy_from_slice(&0x10002u32.to_le_bytes());
    let (mut guest_data, mut shared_data) = (Vec::new(), Vec::new());
    for (index, &(slot, topology, tess, size)) in states.iter().enumerate() {
        let record = record(slot.max(1));
        let code = if slot == 6 {
            &record[80..]
        } else {
            record.as_slice()
        };
        for value in [guest_data.len() as u32, code.len() as u32, 0, 0] {
            guest_toc.extend(value.to_le_bytes());
        }
        guest_data.extend(pack(code, (index % 3) as u8));
        shared_toc.extend((shared_data.len() as u64).to_le_bytes());
        let compute = slot == 6;
        let mut shared = (if compute { 0u32 } else { 1 << slot })
            .to_le_bytes()
            .to_vec();
        shared.extend((index as u32).to_le_bytes());
        shared.extend(0u32.to_le_bytes()); // queried flags
        shared.push(u8::from(compute));
        shared.extend(if compute { b"COMS" } else { b"GFXS" });
        shared.extend((size as u32).to_le_bytes());
        let mut raw = vec![0; size];
        if size >= 8 {
            raw[4..8].copy_from_slice(&topology.to_le_bytes());
        }
        if size >= 12 {
            raw[8..12].copy_from_slice(&tess.to_le_bytes());
        }
        shared.extend(raw);
        shared.extend([0, 0]); // no CB-use records, no pipeline
        shared.extend(0u32.to_le_bytes()); // texture count, no queried arrays
        shared_data.extend(pack(&shared, ((index + 1) % 3) as u8));
    }
    for (name, bytes) in [
        ("guest.toc", guest_toc),
        ("guest.data", guest_data),
        ("shared.toc", shared_toc),
        ("shared.data", shared_data),
    ] {
        fs::write(dir.join(name), bytes).unwrap();
    }
}
fn context(stage: u32, topology: u32, mode: u32) -> CacheContext<'static> {
    let mut context = CacheContext::default();
    if stage == 4 {
        context.graphics_state.geometry_input = Some(match topology {
            0 | 7..=9 => GeometryInput::Points,
            1..=3 => GeometryInput::Lines,
            4..=6 => GeometryInput::Triangles,
            10 | 11 => GeometryInput::LinesAdjacency,
            12 | 13 => GeometryInput::TrianglesAdjacency,
            14 if mode & 3 == 0 => GeometryInput::Lines,
            14 => GeometryInput::Triangles,
            _ => unreachable!(),
        });
    }
    if stage == 3 {
        context.graphics_state.tessellation = Some(TessellationState {
            domain: [
                TessellationDomain::Isolines,
                TessellationDomain::Triangles,
                TessellationDomain::Quads,
            ][(mode & 3) as usize],
            spacing: [
                TessellationSpacing::Equal,
                TessellationSpacing::FractionalEven,
                TessellationSpacing::FractionalOdd,
            ][((mode >> 4) & 3) as usize],
            clockwise: mode & 256 == 0,
        });
    }
    context
}
fn assert_hit(dir: &Path, stage: u32, topology: u32, mode: u32) -> String {
    let bytes = record(stage);
    let input = shadowbox::header_prefixed_input(&bytes, shadowbox::TargetGeneration::Sm5).unwrap();
    let context = context(stage, topology, mode);
    let cold = Result::from(shadowbox::translate_with_graphics_state(
        shadowbox::QuietRun,
        &input,
        &context.graphics_state,
    ))
    .unwrap();
    let mut cache = Cache::new(CacheConfig {
        memory: shadowbox::cache::Keep::nothing(),
        disk: shadowbox::cache::Disk::at(dir),
    });
    let warm = Result::from(cache.translate_with_context(&input, context, || {
        panic!("warm cache must skip translation")
    }))
    .unwrap();
    assert_eq!(cache.stats.disk_hits, 1);
    assert_eq!(warm.spirv, cold.spirv);
    assert_eq!(
        warm.requires_subgroup_size_32,
        cold.requires_subgroup_size_32
    );
    identity(&input, context).unwrap()
}
#[test]
fn imports_and_pairs_all_graphics_variants_by_code_hash() {
    let temp = Temp::new();
    let guest = temp.0.join("guest");
    let imported = temp.0.join("imported");
    let mut states = vec![(1, 0, 0, 204), (2, 0, 0, 204), (5, 0, 0, 204)];
    for topology in 0..=14 {
        states.push((4, topology, 1, 204));
    }
    states.push((4, 14, 0, 204)); // isoline patches become lines
    for domain in 0..3 {
        for spacing in 0..3 {
            for cw in [0, 256] {
                states.push((3, 0, domain | spacing << 4 | cw, 204));
            }
        }
    }
    states.push(states[0]); // deduplicate same identity
    fixture(&guest, &states);
    let report = success(&[
        Path::new("cache-import"),
        &guest,
        Path::new("--emit-cache"),
        &imported,
    ]);
    assert!(
        report.contains("complete=26 state_missing=0 translation_failed=0"),
        "{report}"
    );
    for &(stage, topology, mode, _) in &states {
        assert_hit(&imported, stage, topology, mode);
    }
    let input = temp.0.join("input");
    fs::create_dir(&input).unwrap();
    for stage in 1..=5 {
        let mut aot = vec![0; 48];
        aot[..4].copy_from_slice(&0x1234_5678u32.to_le_bytes());
        aot.extend(record(stage));
        fs::write(input.join(format!("stage{stage}")), aot).unwrap();
    }
    // An unmatched geometry program retains its missing state rather than guessing.
    let mut unmatched = fs::read(input.join("stage4")).unwrap();
    // Replace an unreachable OpNop with MOV32I, keeping a valid instruction.
    // MOV32I encoding: Mesa nak/sm50.rs:1938-1951.
    unmatched[144..152].copy_from_slice(&0x0100_0000_0017_f000u64.to_le_bytes());
    fs::write(input.join("unmatched"), unmatched).unwrap();
    let out = temp.0.join("out");
    let paired = temp.0.join("paired");
    let report = success(&[
        Path::new("aot-scan"),
        &input,
        &out,
        Path::new("--state-from"),
        &guest,
        Path::new("--emit-cache"),
        &paired,
    ]);
    assert!(report.contains("complete=26 missing_state=1"), "{report}");
    let events: Vec<Value> = fs::read_to_string(out.join("manifest.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(events.last().unwrap()["cache_emitted"], 26);
    assert_eq!(events.last().unwrap()["missing_state"], 1);
    let geometry = events
        .iter()
        .find(|e| {
            e["kind"] == "program"
                && e["stage"] == "geometry"
                && e["cache_status"] != "missing_state"
        })
        .unwrap();
    assert_eq!(geometry["cache_keys"].as_array().unwrap().len(), 5);
    let tes = events
        .iter()
        .find(|e| e["stage"] == "tessellation-evaluation")
        .unwrap();
    assert_eq!(tes["cache_keys"].as_array().unwrap().len(), 18);
    for &(stage, topology, mode, _) in &states {
        assert_hit(&paired, stage, topology, mode);
    }
    assert_eq!(shadowbox::disk_cache_stats(&paired).unwrap().entries, 26);
    // Repeating the same scan without recorded state must remove the old
    // geometry/TES artifacts, while keeping valid state-independent output.
    success(&[
        Path::new("aot-scan"),
        &input,
        &out,
        Path::new("--translate"),
    ]);
    let modules = fs::read_dir(&out)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "spv"))
        .count();
    assert_eq!(modules, 3);
}
#[test]
fn missing_unsupported_and_malformed_state_never_emit_guesses() {
    let temp = Temp::new();
    let guest = temp.0.join("guest");
    let out = temp.0.join("out");
    fixture(
        &guest,
        &[(4, 0, 0, 4), (3, 0, 0, 8), (6, 0, 0, 24), (0, 0, 0, 204)],
    );
    let report = success(&[
        Path::new("cache-import"),
        &guest,
        Path::new("--emit-cache"),
        &out,
    ]);
    assert!(report.contains("complete=0 state_missing=4"), "{report}");
    assert!(!out.exists());
    for state in [(4, 99, 0, 204), (3, 0, 3, 204), (3, 0, 48, 204)] {
        fixture(&guest, &[state]);
        assert!(!cli(&[
            Path::new("cache-import"),
            &guest,
            Path::new("--emit-cache"),
            &out
        ])
        .status
        .success());
        assert!(!out.exists());
    }
    fixture(&guest, &[(4, 0, 0, 204)]);
    let mut data = fs::read(guest.join("shared.data")).unwrap();
    data.truncate(15);
    fs::write(guest.join("shared.data"), data).unwrap();
    assert!(!cli(&[
        Path::new("cache-import"),
        &guest,
        Path::new("--emit-cache"),
        &out
    ])
    .status
    .success());
    assert!(!out.exists());
}

#[test]
fn shared_program_state_applies_after_all_stage_indices_and_keeps_aot_header() {
    let temp = Temp::new();
    let guest = temp.0.join("guest");
    let imported = temp.0.join("imported");
    let mode = 2 | 2 << 4;
    fixture(
        &guest,
        &[
            (1, 11, mode, 204),
            (3, 11, mode, 204),
            (4, 11, mode, 204),
            (5, 11, mode, 204),
        ],
    );
    // One shared program references four guest shaders in ascending slot order.
    // DiskCacheHostStorage.cs:595-618; specialization fields as in fixture().
    let mut shared = ((1u32 << 1) | (1 << 3) | (1 << 4) | (1 << 5))
        .to_le_bytes()
        .to_vec();
    for index in 0u32..4 {
        shared.extend(index.to_le_bytes());
    }
    shared.extend(0u32.to_le_bytes());
    shared.push(0);
    shared.extend(b"GFXS");
    shared.extend(204u32.to_le_bytes());
    let mut raw = vec![0; 204];
    raw[4..8].copy_from_slice(&11u32.to_le_bytes());
    raw[8..12].copy_from_slice(&mode.to_le_bytes());
    shared.extend(raw);
    shared.extend([0, 0]);
    shared.extend(0u32.to_le_bytes());
    let mut toc = fs::read(guest.join("shared.toc")).unwrap();
    toc.truncate(40);
    fs::write(guest.join("shared.toc"), toc).unwrap();
    fs::write(guest.join("shared.data"), pack(&shared, 2)).unwrap();
    let report = success(&[
        Path::new("cache-import"),
        &guest,
        Path::new("--emit-cache"),
        &imported,
    ]);
    assert!(report.contains("complete=4 state_missing=0"), "{report}");
    for stage in [1, 3, 4, 5] {
        assert_hit(&imported, stage, 11, mode);
    }

    let input_dir = temp.0.join("input");
    fs::create_dir(&input_dir).unwrap();
    let mut changed = record(4);
    // Generic output bitmap at SPH432..560: Mesa nak/sph.rs:166-170.
    changed[54] = 1;
    let mut aot = vec![0; 48];
    aot[..4].copy_from_slice(&0x1234_5678u32.to_le_bytes());
    aot.extend(&changed);
    fs::write(input_dir.join("program"), aot).unwrap();
    let out = temp.0.join("out");
    let paired = temp.0.join("paired");
    success(&[
        Path::new("aot-scan"),
        &input_dir,
        &out,
        Path::new("--state-from"),
        &guest,
        Path::new("--emit-cache"),
        &paired,
    ]);
    let input =
        shadowbox::header_prefixed_input(&changed, shadowbox::TargetGeneration::Sm5).unwrap();
    let context = context(4, 11, mode);
    let mut cache = Cache::new(CacheConfig {
        memory: shadowbox::cache::Keep::nothing(),
        disk: shadowbox::cache::Disk::at(paired),
    });
    let warm = Result::from(
        cache.translate_with_context(&input, context, || panic!("must retain AOT header in key")),
    )
    .unwrap();
    assert_eq!(
        warm.spirv,
        Result::from(shadowbox::translate_with_graphics_state(
            shadowbox::QuietRun,
            &input,
            &context.graphics_state
        ))
        .unwrap()
        .spirv
    );
    assert_eq!(cache.stats.disk_hits, 1);

    // The state-only option implies translation and reports the artifact path.
    let standalone = temp.0.join("standalone");
    success(&[
        Path::new("aot-scan"),
        &input_dir,
        &standalone,
        Path::new("--state-from"),
        &guest,
    ]);
    let events: Vec<Value> = fs::read_to_string(standalone.join("manifest.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    let event = events.iter().find(|e| e["kind"] == "program").unwrap();
    assert!(standalone
        .join(event["spirv_output"].as_str().unwrap())
        .exists());
    assert_eq!(events.last().unwrap()["complete"], 1);
    assert_eq!(events.last().unwrap()["cache_emitted"], 0);
}

#[test]
fn unknown_instruction_keeps_failing_on_import() {
    let temp = Temp::new();
    let guest = temp.0.join("guest");
    let out = temp.0.join("out");
    fixture(&guest, &[(1, 0, 0, 204)]);
    let mut code = record(1);
    code[88..96].fill(0); // Unknown, never a fabricated NOP.
    fs::write(guest.join("guest.data"), pack(&code, 0)).unwrap();
    let report = success(&[
        Path::new("cache-import"),
        &guest,
        Path::new("--emit-cache"),
        &out,
    ]);
    assert!(
        report.contains("complete=0 state_missing=0 translation_failed=1"),
        "{report}"
    );
    assert!(!out.exists());
}
