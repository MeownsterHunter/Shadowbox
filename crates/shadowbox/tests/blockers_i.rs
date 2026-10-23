//! Project-owned field probes, no captured shader inputs.
//! Field and semantic sources: docs/reference-sources.md
use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_graphics_state,
    headers,
    interface::{BoundTexture, GraphicsState, TextureDimension},
    ir::{Instruction, InstructionKind as K, Register as R, Source as S},
};
use std::fmt::Write;

const PT: u64 = 7 << 16;
fn decode(raw: u64) -> Instruction {
    decode_word([raw as u32, (raw >> 32) as u32], 8)
}
fn mov(dst: u8, value: u32) -> K {
    K::Mov(ops::Mov {
        quad_lanes: 15,
        dst: R(dst),
        src: S::Immediate(value),
    })
}
fn emit(kind: K, metadata: bool) -> Result<(), String> {
    let mut kinds = (0..32).map(|r| mov(r, 0)).collect::<Vec<_>>();
    // Packed BFI offset/count, and a traced cbuf handle. Ryujinx
    // Translation/Optimizations/BindlessElimination.cs:193-204 tracks cbuf
    // handles. MOV encoding: Mesa sm50.rs:1943-1949.
    kinds.push(mov(3, 0x0804));
    kinds.push(K::Mov(ops::Mov {
        quad_lanes: 15,
        dst: R(20),
        src: S::ConstantBuffer {
            binding: 0,
            offset: 0,
        },
    }));
    kinds.push(kind);
    kinds.push(K::Exit(ops::Exit { cc_test: 15 }));
    let xs = kinds
        .into_iter()
        .enumerate()
        .map(|(i, kind)| Instruction {
            address: i as u32 * 8,
            predicate: None,
            kind,
        })
        .collect::<Vec<_>>();
    let header = headers::parse(&[0; 32]).unwrap();
    let binding = [BoundTexture {
        binding: 9,
        dimension: TextureDimension::D2,
        depth_compare: false,
        multisample: false,
    }];
    emit_unoptimized_with_graphics_state(
        &xs,
        Some(&header),
        &GraphicsState {
            texture_bindings: if metadata { &binding } else { &[] },
            ..Default::default()
        },
    )
    .map(|_| ())
    .map_err(|e| e.to_string())
}

fn cases() -> Vec<(String, String, u64, u64)> {
    let mut rows = Vec::new();
    let mut sweep = |name: &str, raw: u64, field: &str, start: u8, width: u8| {
        for value in 0..1u64 << width {
            rows.push((
                name.to_owned(),
                field.to_owned(),
                value,
                raw & !(((1u64 << width) - 1) << start) | value << start,
            ));
        }
    };
    // Registers/predication: gm107.c:203-207,368-374,404-419.
    // SEL: gm107.c:2118; Ryujinx InstDecoders.cs:4292-4302.
    let sel = 0x38a0u64 << 48 | PT | 7 << 39 | 2 << 8 | 1;
    for (f, s, w) in [
        ("sign", 56, 1),
        ("selector", 39, 3),
        ("selector_not", 42, 1),
        ("prefix_low", 48, 3),
    ] {
        sweep("SEL.I", sel, f, s, w);
    }
    // LEA: gm107.c:2022,2093,2140; Ryujinx InstDecoders.cs:3581-3627.
    for (name, prefix) in [("LEA.R", 0x5bd7), ("LEA.I", 0x36d7), ("LEA.C", 0x4bd7)] {
        let raw = prefix << 48 | PT | 3 << 20 | 2 << 8 | 1;
        for (f, s, w) in [
            ("shift", 39, 5),
            ("negate", 45, 1),
            ("X", 46, 1),
            ("CC", 47, 1),
            ("carry_pred", 48, 3),
        ] {
            sweep(name, raw, f, s, w);
        }
        if name == "LEA.I" {
            sweep(name, raw, "sign", 56, 1);
        }
    }
    // BFI: gm107.c:2019,2041,2092,2139; Ryujinx InstDecoders.cs:1030-1081.
    for (name, prefix) in [
        ("BFI.R", 0x5bf0),
        ("BFI.C", 0x4bf0),
        ("BFI.RC", 0x53f0),
        ("BFI.I", 0x36f0),
    ] {
        let raw = prefix << 48
            | PT
            | 4 << 39
            | (if name == "BFI.I" { 0x0804 } else { 3 }) << 20
            | 2 << 8
            | 1;
        sweep(name, raw, "CC", 47, 1);
        sweep(name, raw, "prefix_low", 48, 3);
    }
    // P2R field inventory: envytools gm107.c:1992,2066,2112,1194-1204;
    // Ryujinx InstDecoders.cs:3916-3953. Mesa's emitter uses PSET for
    // predicate copies (nv50_ir_emit_gm107.cpp:748-753).
    for (name, prefix) in [("P2R.R", 0x5ce8u64), ("P2R.I", 0x38e8), ("P2R.C", 0x4ce8)] {
        let raw = prefix << 48 | PT | 2 << 8 | 1;
        for (field, start, width) in [("byte", 41, 2), ("CC", 40, 1), ("prefix_low", 48, 3)] {
            sweep(name, raw, field, start, width);
        }
        for bit in (43..48).chain([39]) {
            sweep(name, raw, &format!("unnamed{bit}"), bit, 1);
        }
        if name == "P2R.I" {
            sweep(name, raw, "sign", 56, 1);
        }
        if name == "P2R.R" {
            for bit in 28..39 {
                sweep(name, raw, &format!("unnamed{bit}"), bit, 1);
            }
        }
    }
    // VSETP and RRO: envytools gm107.c:2052,1588-1628,2074,1320-1323;
    // RRO also Mesa sm50.rs:725-739. The VSETP base has unsigned U8
    // operands, an immediate B and PT accumulator.
    let vsetp = 0x50f0u64 << 48 | PT | 7 << 39 | 2 << 8 | 1;
    for (field, start, width) in [
        ("A_select", 36, 3),
        ("B_select", 28, 3),
        ("A_signed", 48, 1),
        ("B_signed", 49, 1),
        ("B_register", 50, 1),
        ("cmp_low", 43, 2),
        ("cmp_high", 47, 1),
        ("boolean", 45, 2),
        ("accumulator", 39, 3),
        ("accumulator_not", 42, 1),
    ] {
        sweep("VSETP", vsetp, field, start, width);
    }
    for (field, start, width) in [
        ("mode", 39, 1),
        ("abs", 49, 1),
        ("negate", 45, 1),
        ("prefix_low", 48, 3),
    ] {
        sweep("RRO.C", 0x4c90u64 << 48 | PT | 1, field, start, width);
    }
    // S2R: sm50.rs:3220-3224; Ryujinx InstDecoders.cs:4265-4268.
    sweep("S2R", 0xf0c8u64 << 48 | PT | 1, "index", 20, 8);
    sweep("S2R", 0xf0c8u64 << 48 | PT | 1, "prefix_low", 48, 3);
    for bit in (8..16).chain(28..48) {
        sweep(
            "S2R",
            0xf0c8u64 << 48 | PT | 1,
            &format!("unnamed{bit}"),
            bit,
            1,
        );
    }
    // BRK: gm107.c:1921,426-442; Ryujinx InstDecoders.cs:1105-1113.
    sweep("BRK", 0xe340u64 << 48 | PT | 15, "condition", 0, 5);
    sweep("BRK", 0xe340u64 << 48 | PT | 15, "prefix_low", 48, 4);
    // TEX: gm107.c:1970,875-880,916-923,1026-1033; sm50.rs:2171-2197.
    for bound in [true, false] {
        let raw = (if bound { 0xc038 } else { 0xdeb8 }) << 48
            | (if bound { 3 << 55 } else { 3 << 37 })
            | PT
            | 2 << 28
            | 15 << 31
            | 20 << 20
            | 2 << 8
            | 1
            | if bound { 9 << 36 } else { 0 };
        let name = if bound { "TEX.bound" } else { "TEX.bindless" };
        for (f, s, w) in [
            ("lod", if bound { 55 } else { 37 }, 3),
            ("dim", 28, 3),
            ("mask", 31, 4),
            ("NDV", 35, 1),
            ("offset", if bound { 54 } else { 36 }, 1),
            ("clamp", if bound { 58 } else { 40 }, 1),
            ("fault_pred", 51, 3),
            ("depth", 50, 1),
            ("NODEP", 49, 1),
        ] {
            sweep(name, raw, f, s, w);
        }
    }
    // TLD: sm50.rs:2206-2236; Ryujinx InstDecoders.cs:4889-4925.
    for bound in [true, false] {
        let raw = (if bound { 0xdc38 } else { 0xdd38 }) << 48
            | 1 << 55
            | PT
            | 2 << 28
            | 15 << 31
            | 20 << 20
            | 2 << 8
            | 1
            | if bound { 9 << 36 } else { 0 };
        let name = if bound { "TLD.bound" } else { "TLD.bindless" };
        for (f, s, w) in [
            ("lod", 55, 1),
            ("dim", 28, 3),
            ("mask", 31, 4),
            ("offset", 35, 1),
            ("clamp", 54, 1),
            ("fault_pred", 51, 3),
            ("MS", 50, 1),
            ("NODEP", 49, 1),
        ] {
            sweep(name, raw, f, s, w);
        }
    }
    // SUST: gm107.c:1909,753-838; sm50.rs:2506-2529.
    for bound in [true, false] {
        let raw = 0xeb20u64 << 48
            | PT
            | 3 << 33
            | 15 << 20
            | 2 << 8
            | 8
            | if bound { 1 << 51 | 9 << 36 } else { 20 << 39 };
        let name = if bound { "SUST.bound" } else { "SUST.bindless" };
        for (f, s, w) in [
            ("target", 33, 3),
            ("raw", 52, 1),
            ("size_or_mask", 20, 4),
            ("cache", 24, 2),
            ("clamp", 49, 2),
        ] {
            sweep(name, raw, f, s, w);
        }
    }
    // TXQ: sm50.rs:2345-2381; Ryujinx InstDecoders.cs:5082-5106.
    for bound in [true, false] {
        let raw = (if bound { 0xdf48 } else { 0xdf50 }) << 48
            | PT
            | 1 << 22
            | 3 << 31
            | (if bound { 2 } else { 20 }) << 8
            | 1
            | if bound { 9 << 36 } else { 0 };
        let name = if bound { "TXQ.bound" } else { "TXQ.bindless" };
        for (f, s, w) in [("query", 22, 6), ("mask", 31, 4), ("NODEP", 49, 1)] {
            sweep(name, raw, f, s, w);
        }
    }
    // Coupled fields need a Cartesian probe. RZ is used only where B
    // is actually unused (Ryujinx InstEmitTexture.cs:207-210,252-258).
    for bound in [true, false] {
        for dim in 0..8u64 {
            for lod in 0..8u64 {
                for offset in 0..2u64 {
                    let raw = (if bound { 0xc038 } else { 0xdeb8 }) << 48
                        | lod << if bound { 55 } else { 37 }
                        | offset << if bound { 54 } else { 36 }
                        | if bound { 9 << 36 } else { 0 }
                        | (if bound && lod < 2 && offset == 0 {
                            255
                        } else {
                            20
                        }) << 20
                        | dim << 28
                        | 15 << 31
                        | PT
                        | 2 << 8
                        | 1;
                    rows.push((
                        if bound { "TEX.bound" } else { "TEX.bindless" }.into(),
                        "dim_lod_offset".into(),
                        (dim << 4) | (lod << 1) | offset,
                        raw,
                    ));
                }
            }
            for lod in 0..2u64 {
                for offset in 0..2u64 {
                    for ms in 0..2u64 {
                        let raw = (if bound { 0xdc38 } else { 0xdd38 }) << 48
                            | lod << 55
                            | offset << 35
                            | ms << 50
                            | if bound { 9 << 36 } else { 0 }
                            | (if bound && lod == 0 && offset == 0 && ms == 0 {
                                255
                            } else {
                                20
                            }) << 20
                            | dim << 28
                            | 15 << 31
                            | PT
                            | 2 << 8
                            | 1;
                        rows.push((
                            if bound { "TLD.bound" } else { "TLD.bindless" }.into(),
                            "dim_lod_offset_ms".into(),
                            (dim << 3) | (lod << 2) | (offset << 1) | ms,
                            raw,
                        ));
                    }
                }
            }
        }
    }
    rows
}
fn status(raw: u64) -> (bool, bool, bool) {
    let kind = decode(raw).kind;
    if matches!(kind, K::Unknown(ops::Unknown { .. })) {
        return (false, false, false);
    }
    (true, emit(kind, false).is_ok(), emit(kind, true).is_ok())
}

#[test]
#[ignore = "writes field evidence to BLOCKERS_I_PROBE, run before and after changes"]
fn field_probe() {
    let mut output =
        String::from("family\tfield\tvalue\traw\tdecoded\temitted\temitted_with_metadata\n");
    for (name, field, value, raw) in cases() {
        let (decoded, emitted, metadata) = status(raw);
        writeln!(
            output,
            "{name}\t{field}\t{value}\t{raw:016x}\t{decoded}\t{emitted}\t{metadata}"
        )
        .unwrap();
    }
    std::fs::write(std::env::var("BLOCKERS_I_PROBE").unwrap(), output).unwrap();
}

#[test]
fn baseline_accepted_forms_still_decode_and_emit() {
    for row in include_str!("data/blockers-i-main.tsv").lines().skip(1) {
        let columns = row.split('\t').collect::<Vec<_>>();
        let raw = u64::from_str_radix(columns[3], 16).unwrap();
        let now = status(raw);
        for (column, accepted) in [now.0, now.1, now.2].into_iter().enumerate() {
            assert!(
                columns[column + 4] != "true" || accepted,
                "regression: {row}"
            );
        }
    }
    assert_eq!(
        include_str!("../src/decoder/families.rs")
            .matches("x.guarded(InstructionKind::Nop(ops::Nop))")
            .count(),
        2
    );
}

#[test]
fn exact_ir_and_unimplemented_fields() {
    // Citations for each field are in the probe above and decoder arms.
    let raw = 0x39a0u64 << 48 | PT | 0x7ffefu64 << 20 | 2 << 8 | 1 | 6 << 39 | 1 << 42;
    assert_eq!(
        decode(raw).kind,
        K::Sel(ops::Sel {
            dst: R(1),
            when_true: R(2),
            when_false: S::Immediate((-17i32) as u32),
            condition: 6,
            condition_negate: true
        })
    );
    for bit in 43..51 {
        assert!(matches!(
            decode(raw | 1 << bit).kind,
            K::Unknown(ops::Unknown { .. })
        ));
    }
    let lea = 0x37d7u64 << 48 | PT | 0x7ffff << 20 | 31 << 39 | 1 << 45 | 2 << 8 | 1;
    assert_eq!(
        decode(lea).kind,
        K::Lea(ops::Lea {
            dst: R(1),
            lhs: R(2),
            rhs: S::Immediate(u32::MAX),
            shift: 31,
            negate: true
        })
    );
    for prefix in [0x36d7u64, 0x4bd7] {
        let raw = prefix << 48 | PT;
        for bit in [44, 46, 47] {
            assert!(matches!(
                decode(raw | 1 << bit).kind,
                K::Unknown(ops::Unknown { .. })
            ));
        }
        for pred in 0..7u64 {
            assert!(matches!(
                decode(raw & !(7 << 48) | pred << 48).kind,
                K::Unknown(ops::Unknown { .. })
            ));
        }
    }
    let bfi = 0x5bf0u64 << 48 | PT | 4 << 39 | 3 << 20 | 2 << 8 | 1;
    assert_eq!(
        decode(bfi).kind,
        K::BitFieldInsert(ops::BitFieldInsert {
            dst: R(1),
            insert: R(2),
            base: S::Register(R(4)),
            range: S::Register(R(3))
        })
    );
    assert!(emit(decode(bfi).kind, false).is_ok());
    for bit in (28..39).chain(47..51) {
        assert!(matches!(
            decode(bfi | 1 << bit).kind,
            K::Unknown(ops::Unknown { .. })
        ));
    }
    for control in [0x010020u32, 0x0120, 0xff00, 0x0021] {
        // Proven invalid control remains unsupported, never clamped or dropped.
        let kinds = [mov(3, control), decode(bfi).kind];
        let xs = kinds
            .into_iter()
            .enumerate()
            .map(|(i, kind)| Instruction {
                address: i as u32 * 8,
                predicate: None,
                kind,
            })
            .collect::<Vec<_>>();
        let header = headers::parse(&[0; 32]).unwrap();
        if control == 0x010020 {
            // high bits are outside packed offset/count.
            assert!(emit_unoptimized_with_graphics_state(
                &xs,
                Some(&header),
                &GraphicsState::default()
            )
            .is_ok());
        } else {
            assert!(emit_unoptimized_with_graphics_state(
                &xs,
                Some(&header),
                &GraphicsState::default()
            )
            .is_err());
        }
    }
    assert!(
        emit_unoptimized_with_graphics_state(&[decode(bfi)], None, &GraphicsState::default())
            .is_ok()
    );
}

#[test]
fn added_lod_modes_and_bound_samples_keep_their_encoded_fields() {
    for bound in [false, true] {
        for lod in 0..8u64 {
            let raw = if bound {
                0xc038u64 << 48 | 9 << 36 | lod << 55
            } else {
                0xdeb8u64 << 48 | lod << 37
            };
            let raw = raw
                | PT
                | 15 << 31
                | 2 << 28
                | (if bound && lod < 2 { 255 } else { 20 }) << 20
                | 2 << 8
                | 1;
            let x = decode(raw);
            if matches!(lod, 4 | 5) {
                assert!(matches!(x.kind, K::Unknown(ops::Unknown { .. })));
            } else {
                assert!(
                    matches!(x.kind,K::Texture(ops::Texture {op:shadowbox::ir::TexOp::Sample {lod:mode},..}) if u64::from(mode)==lod)
                );
            }
        }
    }
    // Query2 samples consumes no A operand. Only the Z result is defined.
    // Ryujinx InstEmitTexture.cs:1152-1182; fields Mesa sm50.rs:2364-2381.
    let bindings = [BoundTexture {
        binding: 9,
        dimension: TextureDimension::D2,
        depth_compare: false,
        multisample: true,
    }];
    let state = GraphicsState {
        texture_bindings: &bindings,
        ..Default::default()
    };
    let header = headers::parse(&[0; 32]).unwrap();
    let raw = 0xdf48u64 << 48 | PT | 9 << 36 | 2 << 22 | 4 << 31 | 255 << 8 | 1;
    let mut x = decode(raw);
    assert!(matches!(
        x.kind,
        K::Texture(ops::Texture {
            op: shadowbox::ir::TexOp::Query { query: 2 },
            ..
        })
    ));
    x.address = 0;
    let module = emit_unoptimized_with_graphics_state(&[x], Some(&header), &state).unwrap();
    // Query needs the typed driver declaration, never a guessed sampler.
    assert!(
        emit_unoptimized_with_graphics_state(&[x], Some(&header), &GraphicsState::default())
            .is_err()
    );
    assert!(module.words.windows(1).any(|w| w[0] == (4 << 16 | 107)));
    for mask in (0..16).filter(|&mask| !(4..=7).contains(&mask)) {
        let modified = raw & !(15 << 31) | mask << 31;
        assert!(matches!(
            decode(modified).kind,
            K::Unknown(ops::Unknown { .. })
        ));
    }
    for a in [0, 1, 128, 254] {
        assert!(matches!(
            decode(raw & !(255 << 8) | a << 8).kind,
            K::Unknown(ops::Unknown { .. })
        ));
    }
}

#[test]
fn bfi_cbuf_control_fields_and_runtime_emission() {
    // Mesa nv50_ir_emit_gm107.cpp:2300-2302,2312,2324-2326;
    // envytools gm107.c:2092,203-207,302-309.
    for bank in 0..32u8 {
        for offset in [0u16, 4, 0x7ffc, 0x8000, 0xfffc] {
            let raw = 0x4bf0u64 << 48
                | u64::from(bank) << 34
                | u64::from(offset / 4) << 20
                | 4 << 39
                | PT
                | 2 << 8
                | 1;
            assert_eq!(
                decode(raw).kind,
                K::BitFieldInsert(ops::BitFieldInsert {
                    dst: R(1),
                    insert: R(2),
                    base: S::Register(R(4)),
                    range: S::ConstantBuffer {
                        binding: bank,
                        offset
                    },
                })
            );
            assert!(emit(decode(raw).kind, false).is_ok());
            for bit in 47..51 {
                assert!(matches!(
                    decode(raw | 1 << bit).kind,
                    K::Unknown(ops::Unknown { .. })
                ));
            }
        }
    }
}
