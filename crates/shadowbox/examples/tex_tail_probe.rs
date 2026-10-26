//! One-field decoder probe on project-owned words. No game inputs.
//! Run with cargo run -p shadowbox --example tex_tail_probe.
use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{Instruction, InstructionKind, Register, Source},
};
use std::collections::BTreeMap;

fn sweep(family: &str, base: u64, fields: &[(&str, u32, u32)]) {
    let emit = std::env::args().any(|arg| arg == "--emit");
    for &(name, start, width) in fields {
        let mask = ((1u64 << width) - 1) << start;
        let mut groups: BTreeMap<&str, Vec<u64>> = BTreeMap::new();
        for value in 0..1u64 << width {
            let raw = (base & !mask) | value << start;
            let status = match decode_word([raw as u32, (raw >> 32) as u32], 0).kind {
                InstructionKind::Unknown(ops::Unknown { .. }) => "Unknown",
                InstructionKind::Texture(ops::Texture { .. }) => "Texture",
                InstructionKind::Surface(ops::Surface { .. }) => "Surface",
                InstructionKind::S2R(ops::S2R { .. }) => "S2R",
                InstructionKind::Lea(ops::Lea { .. }) => "Lea",
                InstructionKind::BitFieldInsert(ops::BitFieldInsert { .. }) => "BitFieldInsert",
                _ => "other",
            };
            let status = if emit && status != "Unknown" {
                if translates(raw) {
                    "emits"
                } else {
                    "rejected"
                }
            } else {
                status
            };
            groups.entry(status).or_default().push(value);
        }
        for (status, values) in groups {
            let mut ranges = Vec::new();
            let mut i = 0;
            while i < values.len() {
                let first = values[i];
                let mut last = first;
                i += 1;
                while i < values.len() && values[i] == last + 1 {
                    last = values[i];
                    i += 1;
                }
                ranges.push(if first == last {
                    first.to_string()
                } else {
                    format!("{first}-{last}")
                });
            }
            println!(
                "{family}\t{name}\t{start}\t{width}\t{status}\t{}",
                ranges.join(",")
            );
        }
    }
}

// The emission probe defines bindless handles and offsets, then uses compute.
// It exposes translation-only stops separately from decoder Unknowns.
// The setup does not constitute a semantic oracle or texture-type metadata.
fn translates(raw: u64) -> bool {
    let mut xs: Vec<_> = (0..8)
        .chain(20..28)
        .chain([40])
        .enumerate()
        .map(|(i, r)| Instruction {
            address: i as u32 * 8,
            predicate: None,
            kind: InstructionKind::Mov(ops::Mov {
                dst: Register(r),
                src: Source::Immediate(0),
                quad_lanes: 15,
            }),
        })
        .collect();
    xs.push(decode_word(
        [raw as u32, (raw >> 32) as u32],
        xs.len() as u32 * 8,
    ));
    xs.push(decode_word([0x0007_000f, 0xe300_0000], xs.len() as u32 * 8));
    emit_unoptimized_with_header(&xs, Some(&headers::parse(&[0; 32]).unwrap())).is_ok()
}

fn main() {
    let pt = 7 << 16;
    println!("family\tfield\tstart\twidth\tkind\tvalues");
    // Mesa nak/sm50.rs:2171-2197; envytools gm107.c:1026-1033,1970.
    let tex = 0xc038u64 << 48 | 9 << 36 | 15 << 31 | 2 << 28 | 255 << 20 | pt | 2 << 8 | 8;
    sweep(
        "TEX",
        tex,
        &[
            ("dim", 28, 3),
            ("mask", 31, 4),
            ("ndv", 35, 1),
            ("binding", 36, 13),
            ("nodep", 49, 1),
            ("dc", 50, 1),
            ("fault_pred", 51, 3),
            ("offset", 54, 1),
            ("lod", 55, 3),
            ("lc", 58, 1),
            ("src_b", 20, 8),
        ],
    );
    // Mesa nak/sm50.rs:2181-2197; envytools gm107.c:916-923,1962.
    let texb = 0xdeb8u64 << 48 | 15 << 31 | 2 << 28 | 20 << 20 | pt | 2 << 8 | 8;
    sweep(
        "TEX.B",
        texb,
        &[
            ("dim", 28, 3),
            ("mask", 31, 4),
            ("ndv", 35, 1),
            ("offset", 36, 1),
            ("lod", 37, 3),
            ("lcb", 40, 1),
            ("unnamed", 41, 8),
            ("nodep", 49, 1),
            ("dc", 50, 1),
            ("fault_pred", 51, 3),
        ],
    );
    // Mesa nak/sm50.rs:2206-2236; envytools gm107.c:1965-1966.
    let tld = 0xdc38u64 << 48 | 9 << 36 | 15 << 31 | 2 << 28 | 255 << 20 | pt | 2 << 8 | 8;
    let fields = [
        ("dim", 28, 3),
        ("mask", 31, 4),
        ("offset", 35, 1),
        ("binding", 36, 13),
        ("nodep", 49, 1),
        ("ms", 50, 1),
        ("fault_pred", 51, 3),
        ("cl", 54, 1),
        ("lod", 55, 1),
        ("src_b", 20, 8),
    ];
    sweep("TLD", tld, &fields);
    sweep("TLD.B", (tld & !(0x1fff << 36)) | 1 << 56, &fields);
    // envytools gm107.c:753-838,1909. SuDim is three bits, not four.
    let sust = 0xeb20u64 << 48 | 1 << 52 | 1 << 51 | 9 << 36 | 3 << 33 | 4 << 20 | pt | 2 << 8 | 8;
    let fields = [
        ("dim", 33, 3),
        ("size_or_mask", 20, 4),
        ("raw", 52, 1),
        ("cache", 24, 2),
        ("clamp", 49, 2),
        ("bound", 51, 1),
        ("unnamed_low", 26, 7),
        ("handle_high", 47, 2),
    ];
    sweep("SUST.D", sust, &fields);
    sweep("SUST.P", (sust & !(1 << 52 | 15 << 20)) | 15 << 20, &fields);
    // Bindless handles are REG39, not a bound table index.
    // envytools gm107.c:788-791; Mesa sm50.rs:2491-2494.
    sweep(
        "SUST.DB",
        (sust & !(1 << 51 | 0x1fff << 36)) | 40 << 39,
        &fields,
    );
    // Mesa nak/sm50.rs:3220-3224; envytools gm107.c:120-192,1879.
    sweep(
        "S2R",
        0xf0c8u64 << 48 | 0x21 << 20 | pt | 8,
        &[
            ("sreg", 20, 8),
            ("unnamed_src", 8, 8),
            ("unnamed_low", 28, 12),
            ("unnamed_high", 40, 8),
            ("opcode_low", 48, 3),
        ],
    );
    // TXQ: Mesa src/nouveau/compiler/nak/sm50.rs:2351-2381;
    // envytools envydis/gm107.c:1958-1959,882-891.
    for (family, opcode, bound) in [("TXQ", 0xdf48u64, true), ("TXQ.B", 0xdf50, false)] {
        sweep(
            family,
            opcode << 48 | 1 << 22 | 3 << 31 | pt | 2 << 8 | 8 | if bound { 9 << 36 } else { 0 },
            &[
                ("query", 22, 6),
                ("mask", 31, 4),
                ("nodep", 49, 1),
                ("binding_or_reserved", 36, 13),
                ("src_a", 8, 8),
            ],
        );
    }
    // Query2 selects sample count Z, Mesa sm50.rs:2371;
    // envytools gm107.c:883-884. Ryujinx InstEmitTexture.cs:1161-1178
    // preserves skipped X/Y destinations when those mask bits are set.
    for (family, opcode, bound) in [
        ("TXQ.SAMPLES", 0xdf48u64, true),
        ("TXQ.B.SAMPLES", 0xdf50, false),
    ] {
        sweep(
            family,
            opcode << 48
                | 2 << 22
                | 4 << 31
                | pt
                | (if bound { 255 } else { 2 }) << 8
                | 8
                | if bound { 9 << 36 } else { 0 },
            &[("mask", 31, 4)],
        );
    }
    // BFI: Mesa src/gallium/drivers/nouveau/codegen/nv50_ir_emit_gm107.cpp:
    // 2293-2326; envytools envydis/gm107.c:203-207,2019,2092,2139,2041.
    sweep(
        "BFI.R",
        0x5bf0u64 << 48 | 4 << 39 | 3 << 20 | pt | 2 << 8 | 8,
        &[
            ("dst", 0, 8),
            ("insert", 8, 8),
            ("range", 20, 8),
            ("base", 39, 8),
            ("unnamed", 28, 11),
            ("cc", 47, 1),
            ("opcode_low", 48, 3),
        ],
    );
    // envytools gm107.c:2022; Ryujinx InstDecoders.cs:3581-3594.
    sweep(
        "LEA",
        0x5bd7u64 << 48 | 20 << 20 | pt | 2 << 8 | 8,
        &[
            ("shift", 39, 5),
            ("neg", 45, 1),
            ("x", 46, 1),
            ("cc", 47, 1),
            ("dest_pred", 48, 3),
            ("unnamed", 28, 11),
            ("unnamed_44", 44, 1),
        ],
    );
}
