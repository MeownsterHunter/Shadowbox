//! Offline measurement inputs. spirv-opt is used only by this ignored test.
use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_stream,
    emitter::{emit_unoptimized_with_header, emit_with_header},
    headers::{self, ShaderHeader},
    ir::{Instruction, InstructionKind, InterpFreq, InterpLoc, Register, Source},
};
use std::{fs, path::Path, process::Command, time::Instant};

fn words(path: &Path) -> Vec<u32> {
    fs::read(path)
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| u32::from_le_bytes(*bytes))
        .collect()
}

fn write(path: &Path, words: &[u32]) {
    fs::write(
        path,
        words
            .iter()
            .flat_map(|w| w.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
}

fn count(words: &[u32]) -> usize {
    let mut at = 5;
    let mut count = 0;
    while at < words.len() {
        at += (words[at] >> 16) as usize;
        count += 1;
    }
    assert_eq!(at, words.len());
    count
}

fn measure(name: &str, xs: &[Instruction], header: &ShaderHeader, dir: &Path, rows: &mut String) {
    let raw = emit_unoptimized_with_header(xs, Some(header))
        .unwrap()
        .words;
    let ours = emit_with_header(xs, Some(header)).unwrap().words;
    assert!(count(&ours) <= count(&raw), "{name}");
    assert_eq!(ours, emit_with_header(xs, Some(header)).unwrap().words);
    let raw_path = dir.join(format!("{name}.raw.spv"));
    let ours_path = dir.join(format!("{name}.ours.spv"));
    let offline_path = dir.join(format!("{name}.offline.spv"));
    write(&raw_path, &raw);
    write(&ours_path, &ours);
    assert!(Command::new("spirv-opt")
        .arg("-O")
        .arg(&raw_path)
        .arg("-o")
        .arg(&offline_path)
        .status()
        .unwrap()
        .success());
    for path in [&raw_path, &ours_path, &offline_path] {
        let result = Command::new("spirv-val")
            .args(["--target-env", "vulkan1.2"])
            .arg(path)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
    let iterations = 16;
    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(emit_unoptimized_with_header(xs, Some(header)).unwrap());
    }
    let raw_us = start.elapsed().as_secs_f64() * 1e6 / f64::from(iterations);
    let start = Instant::now();
    for _ in 0..iterations {
        std::hint::black_box(emit_with_header(xs, Some(header)).unwrap());
    }
    let ours_us = start.elapsed().as_secs_f64() * 1e6 / f64::from(iterations);
    rows.push_str(&format!(
        "{name}\t{}\t{}\t{}\t{raw_us:.3}\t{ours_us:.3}\n",
        count(&raw),
        count(&ours),
        count(&words(&offline_path))
    ));
}

fn synthetic(kind: &str, length: u32) -> Vec<Instruction> {
    let mut xs = Vec::new();
    let mut add = |kind| {
        xs.push(Instruction {
            address: xs.len() as u32 * 8,
            predicate: None,
            kind,
        });
    };
    add(if kind == "dynamic" {
        // FragCoordX mapping already implemented and cited in emitter.rs.
        InstructionKind::IpaPass(ops::IpaPass {
            dst: Register(0),
            addr: 0x70,
            inv_w: Source::Register(Register(255)),
            offset: Source::Register(Register(255)),
            loc: InterpLoc::Default,
            freq: InterpFreq::Pass,
            saturate: false,
        })
    } else {
        InstructionKind::Mov(ops::Mov {
            quad_lanes: 0xf,
            dst: Register(0),
            src: Source::Immediate(0x3f000000),
        })
    });
    for index in 0..length {
        add(if kind == "dead" {
            InstructionKind::Mov(ops::Mov {
                quad_lanes: 0xf,
                dst: Register((index % 230 + 8) as u8),
                src: Source::Immediate(index),
            })
        } else {
            InstructionKind::IAdd(ops::IAdd {
                dst: Register(0),
                lhs: Register(0),
                rhs: Source::Immediate(1),
                negate: [false; 2],
                cc: false,
                x: false,
            })
        });
    }
    for reg in 1..=3 {
        add(InstructionKind::Mov(ops::Mov {
            quad_lanes: 0xf,
            dst: Register(reg),
            src: Source::Immediate(0x3f800000),
        }));
    }
    add(InstructionKind::Exit(ops::Exit { cc_test: 0xf }));
    xs
}

#[test]
#[ignore = "offline corpus and synthetic SPIR-V benchmark"]
fn export_optimization_measurements() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = std::env::temp_dir().join("shadowbox-spv-opt-measurements");
    fs::create_dir_all(&dir).unwrap();
    let mut rows = "name\traw_instructions\tours_instructions\toffline_instructions\traw_emit_us\tours_emit_us\n".to_owned();
    let mut paths = fs::read_dir(root.join("corpus"))
        .unwrap()
        .map(|x| x.unwrap().path())
        .filter(|p| p.extension().is_some_and(|x| x == "bin"))
        .collect::<Vec<_>>();
    paths.sort();
    for path in paths {
        let name = path.file_stem().unwrap().to_str().unwrap();
        let xs = decode_stream(shadowbox::QuietRun, &words(&path));
        // The corpus marks a program's stage beside its header; rewrite the
        // SPH stage field (bits 10..14: compute 0, vertex 1, fragment 5;
        // Mesa sph.rs:204-215) before parsing.
        let mut hdr = words(&path.with_extension("hdr"));
        let field = if path.with_extension("comp").exists() {
            0
        } else if path.with_extension("vert").exists() {
            1
        } else {
            5
        };
        hdr[0] = hdr[0] & !(0xf << 10) | field << 10;
        let header = headers::parse(&hdr).unwrap();
        match emit_unoptimized_with_header(&xs, Some(&header)) {
            Ok(_) => measure(name, &xs, &header, &dir, &mut rows),
            Err(error) => println!("unsupported {name}: {error}"),
        }
    }
    // A fragment program writing RGBA to render target 0 (SPH stage 5,
    // target mask bits 576..608).
    let mut sph = [0u32; 32];
    sph[0] = 5 << 10;
    sph[18] = 0xf;
    let header = headers::parse(&sph).unwrap();
    for kind in ["constant", "dynamic", "dead"] {
        for length in [64, 512, 2048] {
            measure(
                &format!("synthetic_{kind}_{length}"),
                &synthetic(kind, length),
                &header,
                &dir,
                &mut rows,
            );
        }
    }
    fs::write(dir.join("instructions.tsv"), &rows).unwrap();
    print!("{rows}");
}
