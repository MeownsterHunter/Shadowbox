//! Project-owned Maxwell texture snippets. Encodings: envytools
//! gm107.c:932-1008,1960-1968, Mesa nv50_ir_emit_gm107.cpp:2787-2968.
//! Source and destination semantics: Ryujinx InstEmitTexture.cs:394-641.
use shadowbox::ir::ops;
use shadowbox::spirv::Op;
use shadowbox::{
    decoder::decode_word,
    interface::ShaderInput,
    ir::{InstructionKind, Register, TexOp},
    translate,
};
use std::{collections::HashMap, fs, process::Command};

// Unconditional PT: envytools gm107.c:409; predicates are bits 16..18
// with inversion at 19 (gm107.c:371,411).
const PT: u64 = 7 << 16;

#[path = "support/spirv_validation.rs"]
mod spirv_validation;

fn decode(word: u64) -> InstructionKind {
    decode_word([word as u32, (word >> 32) as u32], 8).kind
}

#[test]
fn tlds_mode_and_result_tables_preserve_every_field() {
    // Independent table oracle: envytools gm107.c:932-972,1967;
    // Mesa nv50_ir_emit_gm107.cpp:2787-2807,2856-2882,2940-2968.
    // (dimension, explicit LOD, offset, multisample), in selector order.
    let modes = [
        Some((0, false, false, false)),
        Some((0, true, false, false)),
        Some((2, false, false, false)),
        None,
        Some((2, false, true, false)),
        Some((2, true, false, false)),
        Some((2, false, false, true)),
        Some((4, false, false, false)),
        Some((3, false, false, false)),
        None,
        None,
        None,
        Some((2, true, true, false)),
        None,
        None,
        None,
    ];
    for (mode, fields) in modes.into_iter().enumerate() {
        for dst2 in [12, 255] {
            for selector in 0..8 {
                for nodep in [false, true] {
                    let word = scalar(0xda00, mode as u64, selector, dst2) | u64::from(nodep) << 49;
                    let masks: &[u8] = if dst2 == 255 {
                        &[1, 2, 4, 8, 3, 9, 10, 12]
                    } else {
                        &[7, 11, 13, 14, 15]
                    };
                    let expected = match (fields, masks.get(selector as usize)) {
                        (Some((dim, lod, offset, multisample)), Some(&mask)) => {
                            InstructionKind::Texture(ops::Texture {
                                op: TexOp::Fetch { lod },
                                dst: Register(0),
                                src0: Register(8),
                                src1: Register(if mode == 0 { 255 } else { 20 }),
                                dim,
                                mask,
                                offset,
                                scalar: true,
                                packed_f16: false,
                                lod_clamp: false,
                                multisample,
                                depth_compare: false,
                                nodep,
                                scalar_dst: Some(Register(dst2 as u8)),
                                bound_handle: Some(9),
                            })
                        }
                        _ => InstructionKind::Unknown(ops::Unknown {
                            raw: [word as u32, (word >> 32) as u32],
                        }),
                    };
                    assert_eq!(decode(word), expected, "{word:016x}");
                    // Bit 59=0 selects packed F16 (envytools gm107.c:1967).
                    // Exact packed values are exercised below.
                    if fields.is_some() && masks.get(selector as usize).is_some() {
                        assert!(matches!(
                            decode(word & !(1 << 59)),
                            InstructionKind::Texture(ops::Texture {
                                packed_f16: true,
                                ..
                            })
                        ));
                    }
                }
            }
        }
    }
}

#[test]
fn named_prefixes_emit_real_texture_operations() {
    // Fields: envytools gm107.c:932-972,1960,1967. These are synthetic
    // instructions with both destination pairs present and bound index 9.
    for (prefix, mask, op) in [
        (0xda42, 7, TexOp::Fetch { lod: false }),
        (0xda52, 15, TexOp::Fetch { lod: false }),
        (
            0xdf02,
            15,
            TexOp::Gather {
                component: 0,
                offsets: shadowbox::ir::GatherOffsets::Zero,
            },
        ),
    ] {
        let word = prefix << 48 | 9 << 36 | 12 << 28 | 20 << 20 | PT | 8 << 8;
        assert_eq!(
            decode(word),
            InstructionKind::Texture(ops::Texture {
                op,
                dst: Register(0),
                src0: Register(8),
                src1: Register(20),
                dim: 2,
                mask,
                offset: false,
                scalar: true,
                packed_f16: false,
                lod_clamp: false,
                multisample: false,
                depth_compare: false,
                nodep: true,
                scalar_dst: Some(Register(12)),
                bound_handle: Some(9),
            }),
            "{word:016x}"
        );
        let words = translation(&[word]).unwrap();
        let ops = opcodes(&words);
        assert_eq!(
            ops.iter()
                .filter(|op| matches!(
                    op,
                    Op::ImageSampleImplicitLod
                        | Op::ImageSampleExplicitLod
                        | Op::ImageSampleDrefImplicitLod
                        | Op::ImageSampleDrefExplicitLod
                        | Op::ImageSampleProjImplicitLod
                        | Op::ImageSampleProjExplicitLod
                        | Op::ImageSampleProjDrefImplicitLod
                        | Op::ImageSampleProjDrefExplicitLod
                        | Op::ImageFetch
                        | Op::ImageGather
                        | Op::ImageDrefGather
                ))
                .count(),
            1
        );
        assert!(ops.contains(&if prefix == 0xdf02 {
            Op::ImageGather
        } else {
            Op::ImageFetch
        }));
        let check = spirv_validation::validate(&words);
        assert!(
            check.status.success(),
            "{}",
            String::from_utf8_lossy(&check.stderr)
        );
        // NODEP must affect helper results, not erase the image instruction.
        assert_ne!(words, translation(&[word & !(1 << 49)]).unwrap());
    }
}

#[test]
fn tld4s_component_and_modifier_fields_survive_decode_and_emission() {
    // Component bits 52..53, AOFFI 51, DC 50, NODEP 49, operands and bound
    // index: envytools gm107.c:893-898,1960; Mesa
    // nv50_ir_emit_gm107.cpp:2945-2968. Ryujinx InstEmitTexture.cs:564-601
    // defines the distinct coordinate/offset/compare register layouts.
    for component in 0..4 {
        for offset in [false, true] {
            for depth_compare in [false, true] {
                for nodep in [false, true] {
                    let word = scalar(0xdf00, 0, u64::from(depth_compare), 12)
                        | component << 52
                        | u64::from(offset) << 51
                        | u64::from(nodep) << 49;
                    assert_eq!(
                        decode(word),
                        InstructionKind::Texture(ops::Texture {
                            op: TexOp::Gather {
                                component: component as u8,
                                offsets: if offset {
                                    shadowbox::ir::GatherOffsets::One
                                } else {
                                    shadowbox::ir::GatherOffsets::Zero
                                },
                            },
                            dst: Register(0),
                            src0: Register(8),
                            src1: Register(20),
                            dim: 2,
                            mask: 15,
                            offset: false,
                            scalar: true,
                            packed_f16: false,
                            lod_clamp: false,
                            multisample: false,
                            depth_compare,
                            nodep,
                            scalar_dst: Some(Register(12)),
                            bound_handle: Some(9),
                        }),
                        "{word:016x}"
                    );
                    // TLD4S uses the opposite half-result flag at bit 55
                    // (envytools gm107.c:1960). Preserve it in the IR.
                    assert!(matches!(
                        decode(word | 1 << 55),
                        InstructionKind::Texture(ops::Texture {
                            packed_f16: true,
                            ..
                        })
                    ));
                    let result = translation(&[word]);
                    if depth_compare && component != 0 {
                        // The component field cannot be silently ignored
                        // when Dref gather has no component operand.
                        assert!(result.is_err(), "{word:016x}");
                        continue;
                    }
                    let words = result.unwrap();
                    let check = spirv_validation::validate(&words);
                    assert!(
                        check.status.success(),
                        "{word:016x}: {}",
                        String::from_utf8_lossy(&check.stderr)
                    );
                    assert_eq!(
                        opcodes(&words)
                            .iter()
                            .filter(|op| matches!(
                                op,
                                Op::ImageSampleImplicitLod
                                    | Op::ImageSampleExplicitLod
                                    | Op::ImageSampleDrefImplicitLod
                                    | Op::ImageSampleDrefExplicitLod
                                    | Op::ImageSampleProjImplicitLod
                                    | Op::ImageSampleProjExplicitLod
                                    | Op::ImageSampleProjDrefImplicitLod
                                    | Op::ImageSampleProjDrefExplicitLod
                                    | Op::ImageFetch
                                    | Op::ImageGather
                                    | Op::ImageDrefGather
                            ))
                            .count(),
                        1
                    );
                }
            }
        }
    }
}

fn scalar(opcode: u64, mode: u64, selector: u64, dst2: u64) -> u64 {
    // R0/R8/R20/R28 fields: envytools gm107.c:203-206,1960,1967;
    // bound index 36..48: Mesa nv50_ir_emit_gm107.cpp:2958;
    // mode 53..56 and selector 50..52: same file:2942-2943.
    let src_b = if matches!(opcode, 0xd000 | 0xd200 | 0xd800 | 0xda00) && mode == 0 {
        255
    } else {
        20
    };
    opcode << 48 | mode << 53 | selector << 50 | 9 << 36 | dst2 << 28 | src_b << 20 | PT | 8 << 8
}

fn program(instructions: &[u64]) -> Vec<u8> {
    let nop = 0x50b0 << 48 | PT;
    // Mesa sm50.rs:3107-3110: unconditional EXIT is CC.T, selector15.
    let exit = 0xe300 << 48 | PT | 15;
    let mut slots = instructions.to_vec();
    slots.push(exit);
    while !slots.len().is_multiple_of(3) {
        slots.push(nop);
    }
    slots
        .chunks(3)
        .flat_map(|s| [0, s[0], s[1], s[2]])
        .flat_map(u64::to_le_bytes)
        .collect()
}

fn translation(instructions: &[u64]) -> Result<Vec<u32>, shadowbox::interface::Stopped> {
    let mut header = vec![0; 128];
    header[1] = 5 << 2;
    header[72] = 15;
    Result::from(translate(
        shadowbox::QuietRun,
        &ShaderInput {
            code: &program(instructions),
            program_record: &header,
        },
    ))
    .map(|o| o.spirv)
}

fn opcodes(words: &[u32]) -> Vec<Op> {
    let mut at = 5;
    let mut out = Vec::new();
    while at < words.len() {
        out.push(Op::of(words[at]));
        at += (words[at] >> 16) as usize;
    }
    out
}

#[test]
fn qualified_tld4_ptp_forms_decode_emit_four_gathers_and_validate() {
    // PTP selector and packed register order: Mesa sm50.rs:2245-2278;
    // Ryujinx InstEmitTexture.cs:700-779. SPIR-V's dynamic Offset form is
    // independently validated for each gather by spirv-val.
    let forms = [
        0xdef8_0027_a107_0820,
        0xdef8_0067_b107_0820,
        0xcab8_0097_a107_0820,
        0xcbb8_0097_b107_0820,
        0xdef8_00e7_a107_0820,
    ];
    for raw in forms {
        assert!(matches!(
            decode(raw),
            InstructionKind::Texture(ops::Texture {
                op: TexOp::Gather {
                    offsets: shadowbox::ir::GatherOffsets::Four,
                    ..
                },
                offset: false,
                ..
            })
        ));
        let mov = |dst: u64, value: u64| 0x0100_u64 << 48 | value << 20 | 0xf << 12 | PT | dst;
        let program_words = [
            mov(16, 1),
            mov(17, 0x01_3f_02_3e),
            mov(18, 0x03_3d_04_3c),
            raw,
        ];
        let words = translation(&program_words).unwrap();
        assert_eq!(
            opcodes(&words)
                .into_iter()
                .filter(|op| *op == Op::ImageGather)
                .count(),
            4,
            "{raw:016x}"
        );
        let validation = spirv_validation::validate(&words);
        assert!(
            validation.status.success(),
            "{raw:016x}: {}",
            String::from_utf8_lossy(&validation.stderr)
        );
        let decoded = program_words
            .into_iter()
            .enumerate()
            .map(|(index, word)| decode_word([word as u32, (word >> 32) as u32], index as u32 * 8))
            .collect::<Vec<_>>();
        let raw_words = shadowbox::emitter::emit_unoptimized_with_header(&decoded, None)
            .unwrap()
            .words;
        let raw_validation = spirv_validation::validate(&raw_words);
        assert!(
            raw_validation.status.success(),
            "raw {raw:016x}: {}",
            String::from_utf8_lossy(&raw_validation.stderr)
        );
    }
    let four = decode_word([forms[0] as u32, (forms[0] >> 32) as u32], 8).to_string();
    assert!(four.contains("offsets: Four"));
    let one = decode_word([0x0007_ff00, 0xc878_7f80], 8).to_string();
    assert!(one.contains("Gather { component: 0 }"));
    assert!(!one.contains("offsets:"));
    assert!(one.ends_with(", aoffi"));
}

#[test]
fn all_scalar_modes_emit_one_image_operation_and_validate() {
    assert!(Command::new("spirv-val")
        .arg("--version")
        .output()
        .unwrap()
        .status
        .success());
    let dir = std::env::temp_dir().join(format!("texture-validation-{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let mut forms = Vec::new();
    for mode in 0..14 {
        forms.push((
            scalar(0xd800, mode, 4, 12),
            if matches!(mode, 4..=6 | 9) {
                if mode == 4 {
                    Op::ImageSampleDrefImplicitLod
                } else {
                    Op::ImageSampleDrefExplicitLod
                }
            } else if matches!(mode, 1 | 7 | 10 | 12) {
                Op::ImageSampleImplicitLod
            } else {
                Op::ImageSampleExplicitLod
            },
        ));
    }
    for mode in [0, 1, 2, 4, 5, 6, 7, 8, 12] {
        forms.push((scalar(0xda00, mode, 4, 12), Op::ImageFetch));
    }
    for dc in 0..2 {
        for offset in 0..2 {
            forms.push((
                scalar(0xdf00, 0, dc, 12) | offset << 51,
                if dc == 0 {
                    Op::ImageGather
                } else {
                    Op::ImageDrefGather
                },
            ));
        }
    }
    // Half results use the same modes and image operations, with packed
    // destinations (envytools gm107.c:1960,1967-1968; Ryujinx
    // InstEmitTexture.cs:609-655). Exercise the optimized translator too.
    let half_forms = forms
        .iter()
        .map(|&(word, op)| {
            let half = if word >> 48 & 0xff40 == 0xdf00 {
                word | 1 << 55
            } else {
                word & !(1 << 59)
            };
            (half, op)
        })
        .collect::<Vec<_>>();
    forms.extend(half_forms);
    // Mesa sm50.rs:2226-2236 encodes TLD.MS in both 2D dimensions.
    for dim in [2, 3] {
        forms.push((
            0xdd38 << 48 | 1 << 50 | 15 << 31 | dim << 28 | 20 << 20 | PT | 8 << 8,
            Op::ImageFetch,
        ));
    }
    // Validate both forms: the NODEP field changes helper results but
    // retains exactly the same image operation for active lanes.
    // Bit59 clear packs F16 results while retaining the same image op.
    // envytools gm107.c:1967-1968; Ryujinx InstEmitTexture.cs:609-655.
    let packed = forms
        .iter()
        .filter(|(word, _)| matches!(word >> 57, 0x6c | 0x6d))
        .map(|&(word, op)| (word & !(1 << 59), op))
        .collect::<Vec<_>>();
    forms.extend(packed);
    let with_nodep = forms
        .iter()
        .map(|&(word, op)| (word | 1 << 49, op))
        .collect::<Vec<_>>();
    forms.extend(with_nodep);
    for (i, (word, opcode)) in forms.into_iter().enumerate() {
        // Offsets are constant so the fetch ConstOffset contract is met.
        // Mesa sm50.rs:1938-1941: setup MOV32I enables all quad positions.
        let mov = 0x0100 << 48 | 15 << 12 | PT | 20;
        let words = translation(&[mov, mov + 1, word]).unwrap();
        let ops = opcodes(&words);
        assert_eq!(
            ops.iter()
                .filter(|op| matches!(
                    op,
                    Op::ImageSampleImplicitLod
                        | Op::ImageSampleExplicitLod
                        | Op::ImageSampleDrefImplicitLod
                        | Op::ImageSampleDrefExplicitLod
                        | Op::ImageSampleProjImplicitLod
                        | Op::ImageSampleProjExplicitLod
                        | Op::ImageSampleProjDrefImplicitLod
                        | Op::ImageSampleProjDrefExplicitLod
                        | Op::ImageFetch
                        | Op::ImageGather
                        | Op::ImageDrefGather
                ))
                .count(),
            1
        );
        assert!(ops.contains(&opcode), "{word:016x}");
        let path = dir.join(format!("{i}.spv"));
        fs::write(
            &path,
            words
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let check = Command::new("spirv-val")
            .args(["--target-env", "vulkan1.2"])
            .arg(&path)
            .output()
            .unwrap();
        assert!(
            check.status.success(),
            "{word:016x}: {}",
            String::from_utf8_lossy(&check.stderr)
        );
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn masks_and_disjoint_register_writes_follow_the_tables() {
    for (dst2, masks) in [
        (255, &[1, 2, 4, 8, 3, 9, 10, 12][..]),
        (12, &[7, 11, 13, 14, 15][..]),
    ] {
        for (selector, mask) in masks.iter().enumerate() {
            let word = scalar(0xd800, 2, selector as u64, dst2);
            let kind = decode_word([word as u32, (word >> 32) as u32], 8).kind;
            assert!(
                matches!(kind, InstructionKind::Texture(ops::Texture { mask: m, .. }) if m == *mask)
            );
            assert_eq!(kind.writes_register(12), mask.count_ones() > 2);
            assert_eq!(kind.writes_register(13), mask.count_ones() == 4);
            assert!(!kind.writes_register(2));
        }
    }
}

// Evaluate the data path after a synthetic image response. The four exact
// binary16 values are -0, 1, -2 and the smallest half subnormal. This isolates
// packing/channel order from image sampling, which requires a GPU.
fn packed_stores(words: &[u32]) -> Vec<(u8, u32)> {
    let mut values: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut packed = HashMap::new();
    let mut registers = Vec::new();
    let mut stores = Vec::new();
    let mut uint = 0;
    let mut pointer = 0;
    let mut at = 5;
    while at < words.len() {
        let len = (words[at] >> 16) as usize;
        let op = Op::of(words[at]);
        let a = &words[at + 1..at + len];
        match op {
            Op::TypeInt if a[1..] == [32, 0] => uint = a[0],
            Op::TypePointer if a[1..] == [7, uint] => pointer = a[0],
            Op::Variable if a[0] == pointer => registers.push(a[1]),
            Op::Constant => {
                values.insert(a[1], a[2..].to_vec());
            }
            Op::ImageSampleImplicitLod | Op::ImageSampleExplicitLod | Op::ImageFetch => {
                values.insert(
                    a[1],
                    vec![0x8000_0000, 0x3f80_0000, 0xc000_0000, 0x3380_0000],
                );
            }
            Op::CompositeConstruct if a[2..].iter().all(|id| values.contains_key(id)) => {
                values.insert(
                    a[1],
                    a[2..].iter().flat_map(|id| values[id].clone()).collect(),
                );
            }
            Op::CompositeExtract if values.contains_key(&a[2]) => {
                values.insert(a[1], vec![values[&a[2]][a[3] as usize]]);
            }
            Op::ExtInst if a[3] == 58 && values.contains_key(&a[4]) => {
                let half = |bits| match bits {
                    0 => 0,
                    0x8000_0000 => 0x8000,
                    0x3f80_0000 => 0x3c00,
                    0xc000_0000 => 0xc000,
                    0x3380_0000 => 1,
                    other => panic!("unexpected sample bits {other:08x}"),
                };
                let pair = &values[&a[4]];
                assert_eq!(pair.len(), 2);
                packed.insert(a[1], half(pair[0]) | half(pair[1]) << 16);
            }
            Op::Store if packed.contains_key(&a[1]) => {
                let register = registers.iter().position(|id| *id == a[0]).unwrap();
                stores.push((register as u8, packed[&a[1]]));
            }
            _ => {}
        }
        at += len;
    }
    stores
}

#[test]
fn packed_scalar_results_use_masks_zero_fill_and_single_register_destinations() {
    // Encoding: envytools gm107.c:958-1008,1967-1968. Semantics:
    // Ryujinx InstEmitTexture.cs:609-655. Every selector and both destination
    // layouts, including a discarded first destination and source aliasing.
    for opcode in [0xd000, 0xd200] {
        for (dst2, masks) in [
            (255, &[1u8, 2, 4, 8, 3, 9, 10, 12][..]),
            (12, &[7, 11, 13, 14, 15][..]),
        ] {
            for (selector, &mask) in masks.iter().enumerate() {
                for dst in [0, 8, 255] {
                    let raw = scalar(opcode, 2, selector as u64, dst2) | dst;
                    let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
                    for register in 0..255u8 {
                        assert_eq!(
                            instruction.kind.writes_register(register),
                            u64::from(register) == dst || u64::from(register) == dst2
                        );
                    }
                    let header = shadowbox::headers::parse(&[0; 32]).unwrap();
                    let exit = decode_word([(PT | 15) as u32, 0xe300_0000], 16);
                    let words = shadowbox::emitter::emit_unoptimized_with_header(
                        &[instruction, exit],
                        Some(&header),
                    )
                    .unwrap()
                    .words;
                    let halves = [0x8000u32, 0x3c00, 0xc000, 1];
                    let selected = (0..4)
                        .filter(|c| mask & (1 << c) != 0)
                        .map(|c| halves[c])
                        .collect::<Vec<_>>();
                    let mut expected = Vec::new();
                    if dst != 255 {
                        expected.push((
                            dst as u8,
                            selected[0] | selected.get(1).copied().unwrap_or(0) << 16,
                        ));
                    }
                    if dst2 != 255 {
                        expected.push((
                            dst2 as u8,
                            selected[2] | selected.get(3).copied().unwrap_or(0) << 16,
                        ));
                    }
                    assert_eq!(packed_stores(&words), expected, "{raw:016x}");
                }
            }
        }
    }
    for prefix in [0xd032u64, 0xd02e] {
        let raw = prefix << 48 | 9 << 36 | 12 << 28 | 20 << 20 | PT | 8 << 8;
        assert!(matches!(
            decode_word([raw as u32, (raw >> 32) as u32], 8).kind,
            InstructionKind::Texture(ops::Texture {
                packed_f16: true,
                dim: 2,
                nodep: true,
                ..
            })
        ));
        assert!(translation(&[raw]).is_ok());
    }
}

#[test]
fn unproven_blocker_forms_remain_unknown() {
    // VSET: envytools gm107.c:2107; LOP3:2108. These forms still lack
    // independently established semantics. BRX has separate exact tests.
    for prefix in [0x4196u64, 0x3f80, 0x3f59] {
        for operands in [PT, PT | 1 << 47, PT | 255 << 8 | 255 << 39] {
            let word = prefix << 48 | operands;
            assert!(
                matches!(
                    decode_word([word as u32, (word >> 32) as u32], 8).kind,
                    InstructionKind::Unknown(ops::Unknown { .. })
                ),
                "{word:016x}"
            );
            assert!(translation(&[word]).is_err());
        }
    }
}

#[test]
fn unsupported_fields_and_image_type_conflicts_stay_unknown() {
    for word in [
        scalar(0xd800, 14, 4, 12),
        scalar(0xda00, 3, 4, 12),
        scalar(0xd800, 2, 5, 12),
    ] {
        assert!(translation(&[word]).is_err(), "{word:016x}");
    }
    // NODEP is now implemented for fragment shaders. Mesa sm50.rs:2196
    // encodes bit49; nak_nir_lower_tex.c:454 and nir.h:2704-2708 define
    // the permission to discard helper results.
    let nodep = scalar(0xd800, 2, 4, 12) | 1 << 49;
    let words = translation(&[nodep]).unwrap();
    assert!(opcodes(&words).contains(&Op::Undef)); // OpUndef helper result.
    assert!(words
        .windows(4)
        .any(|w| w[0] == (4 << 16 | 71) && w[2..] == [11, 23]));
    assert_ne!(words, translation(&[nodep & !(1 << 49)]).unwrap());
    for opcode in [0xdeb8, 0xdef8] {
        let word = opcode << 48 | PT | 1 << 40 | 2 << 28;
        assert!(translation(&[word]).is_err());
    }
    let ordinary = scalar(0xd800, 2, 4, 12);
    let depth = scalar(0xd800, 6, 4, 12);
    assert!(translation(&[ordinary, depth]).is_err());
    let fetch = scalar(0xda00, 2, 4, 12);
    let ms = scalar(0xda00, 6, 4, 12);
    assert!(translation(&[fetch, ms]).is_err());
    // Bound table identity affects Vulkan binding numbers, even with one op.
    assert_ne!(
        translation(&[ordinary]).unwrap(),
        translation(&[ordinary ^ 1 << 36]).unwrap()
    );
}

#[test]
fn second_pair_invalidates_constant_and_handle_tracking() {
    // The gather overwrites R12/R13. R12 must not retain the earlier known
    // bindless handle or known constant offset after the gather.
    // Mesa sm50.rs:1938-1941: all-lane handle setup.
    let initial = 0x0100 << 48 | 1 << 20 | 15 << 12 | PT | 12;
    let bindless = 0xdeb8 << 48 | PT | 12 << 20 | 8 << 8 | 2 << 28 | 15 << 31;
    for opcode in [0xdf00, 0xdf80] {
        let gather = scalar(opcode, 0, 0, 12);
        assert!(translation(&[initial, gather, bindless]).is_err());
    }
    // F16 packs into R12 only, preserving R13's known handle. F32 writes
    // both registers. InstEmitTexture.cs:634-641,652-655.
    let initial = initial + 1;
    let bindless = bindless + (1 << 20);
    assert!(translation(&[initial, scalar(0xdf80, 0, 0, 12), bindless]).is_ok());
    assert!(translation(&[initial, scalar(0xdf00, 0, 0, 12), bindless]).is_err());

    let packed = scalar(0xd000, 2, 4, 12);
    let original = 0x0100 << 48 | 1 << 20 | 15 << 12 | PT | 12;
    let original_use = 0xdeb8 << 48 | PT | 12 << 20 | 8 << 8 | 2 << 28 | 15 << 31;
    assert!(translation(&[original, packed, original_use]).is_err());
    assert!(translation(&[original + 1, packed, original_use | 1 << 20]).is_ok());
}

#[test]
fn array_layers_use_signed_conversion() {
    let word = scalar(0xd800, 8, 4, 12);
    // Mesa sm50.rs:1938-1941: all-lane array-coordinate setup.
    let mov_negative_layer = 0x0100 << 48 | u64::from(u32::MAX) << 20 | 15 << 12 | PT | 8;
    let words = translation(&[mov_negative_layer, word]).unwrap();
    assert!(words
        .windows(4)
        .any(|w| w[0] == (4 << 16 | 43) && w[3] == (-1.0f32).to_bits()));
    let words = translation(&[word]).unwrap();
    assert!(opcodes(&words).contains(&Op::ConvertSToF));
}

#[test]
fn a_real_branch_to_the_next_slot_stays_a_branch() {
    // Mesa sm50.rs:3020-3030 encodes BRA, not OpNop. Even a branch to the
    // following slot must retain its instruction kind and condition code.
    let bra = 0xe240 << 48 | PT | 0xf;
    let words = program(&[bra])
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| u32::from_le_bytes(*bytes))
        .collect::<Vec<_>>();
    let decoded = shadowbox::decoder::decode_stream(shadowbox::QuietRun, &words);
    assert!(matches!(
        decoded[0].kind,
        InstructionKind::Branch(ops::Branch {
            target,
            cc_test: 0xf
        }) if target.offset() == 0
    ));
    assert!(translation(&[bra]).is_ok());
}

#[test]
fn only_qualified_nop_and_scheduling_prefixes_decode_as_nop() {
    // Mesa sm50.rs:3178-3184 encodes OpNop as 0x50b0. Exact qualified
    // opcode aliases occupy 0x50b1..0x50b7 and have their own boundary
    // suites. Envytools gm107.c:1877-1878 identifies SYNC/DEPBAR in the
    // 0xf0f0 family.
    for prefix in 0..=0xffffu64 {
        let word = prefix << 48 | PT;
        if matches!(
            decode_word([word as u32, (word >> 32) as u32], 8).kind,
            InstructionKind::Nop(ops::Nop)
        ) {
            assert!((0x50b0..=0x50b7).contains(&prefix) || (0xf0f0..=0xf0ff).contains(&prefix));
        }
    }
    let source = include_str!("../src/decoder/families.rs");
    assert_eq!(
        source
            .matches("x.guarded(InstructionKind::Nop(ops::Nop))")
            .count(),
        2
    );
}

#[test]
fn stack_pushes_keep_targets_and_require_proven_consumers() {
    // SSY is a stack push, distinct from OpSync scheduling metadata.
    // Mesa sm50.rs:3032-3041; PBK/PCNT: envytools gm107.c:1930-1933.
    // Unused pushes have no consumer tag writes in Ryujinx
    // Instructions/InstEmitFlowControl.cs:239-251. Preserve their IR kind
    // and target while accepting the proven absence of consumers.
    for opcode in [0xe290u64, 0xe2a0, 0xe2b0] {
        let push = opcode << 48 | PT | 0xf;
        let code = program(&[push]);
        let words = code
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| u32::from_le_bytes(*bytes))
            .collect::<Vec<_>>();
        let decoded = shadowbox::decoder::decode_stream(shadowbox::QuietRun, &words);
        assert!(matches!(
            decoded[0].kind,
            InstructionKind::ResolvedSchedule(ops::ResolvedSchedule { push })
                | InstructionKind::ResolvedPushTarget(ops::ResolvedPushTarget { push, .. })
                if push.target().offset() == 0
        ));
        assert!(translation(&[push]).is_ok());
        // Missing push destinations still reject translation.
        assert!(translation(&[push | (0x20 << 20)]).is_err());
    }
    // Mesa sm50.rs:3050-3051,3061-3062,3084-3085: SYNC/BRK/CONT
    // encodings with CC.T. A pop without a reaching push remains an error.
    for opcode in [0xf0f8u64, 0xe340, 0xe350] {
        assert!(translation(&[opcode << 48 | PT | 0xf]).is_err());
    }
}

// TMML encodings: Mesa sm50.rs:2287-2309; envytools gm107.c:1956.
fn tmml(mask: u64, index: u64) -> u64 {
    0xdf58 << 48 | index << 36 | mask << 31 | 2 << 28 | 255 << 20 | PT | 8 << 8 | 5
}

#[test]
fn bound_tmml_queries_once_and_preserves_index_and_modes() {
    use shadowbox::ir::{Register, TexOp};
    for index in [0, 9, 0x1fff] {
        for mask in 1..=3 {
            let raw = tmml(mask, index);
            assert!(
                matches!(decode_word([raw as u32, (raw >> 32) as u32], 8).kind,
                InstructionKind::Texture(ops::Texture { op: TexOp::Lod, src0: Register(8), mask: m,
                    bound_handle: Some(i), .. }) if u64::from(m) == mask && u64::from(i) == index)
            );
            for nodep in [0, 1 << 49] {
                let words = translation(&[raw | nodep]).unwrap();
                assert_eq!(
                    opcodes(&words)
                        .into_iter()
                        .filter(|op| *op == Op::ImageQueryLod)
                        .count(),
                    1
                );
                // Descriptor identity must change with the encoded index,
                // not merely survive in the IR (set 1, image 2*t, sampler 2*t+1).
                let mut sets = std::collections::HashMap::new();
                let mut bindings = std::collections::HashMap::new();
                let mut at = 5;
                while at < words.len() {
                    let count = (words[at] >> 16) as usize;
                    let a = &words[at + 1..at + count];
                    if Op::of(words[at]) == Op::Decorate && a.len() == 3 {
                        match a[1] {
                            34 => {
                                sets.insert(a[0], a[2]);
                            }
                            33 => {
                                bindings.insert(a[0], a[2]);
                            }
                            _ => {}
                        }
                    }
                    at += count;
                }
                let mut table_bindings = bindings
                    .into_iter()
                    .filter_map(|(id, binding)| (sets.get(&id) == Some(&1)).then_some(binding))
                    .collect::<Vec<_>>();
                table_bindings.sort_unstable();
                assert_eq!(table_bindings, [2 * index as u32, 2 * index as u32 + 1]);
                let path = std::env::temp_dir().join(format!(
                    "tmml-{}-{index}-{mask}-{nodep}.spv",
                    std::process::id()
                ));
                fs::write(
                    &path,
                    words
                        .iter()
                        .flat_map(|word| word.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
                let result = Command::new("spirv-val")
                    .args(["--target-env", "vulkan1.2"])
                    .arg(&path)
                    .output()
                    .unwrap();
                fs::remove_file(path).unwrap();
                assert!(
                    result.status.success(),
                    "{}",
                    String::from_utf8_lossy(&result.stderr)
                );
            }
        }
    }
    let canonical = tmml(3, 9);
    for raw in [canonical | 1 << 35, canonical | 1 << 50, tmml(4, 9)] {
        assert!(matches!(
            decode_word([raw as u32, (raw >> 32) as u32], 8).kind,
            InstructionKind::Unknown(ops::Unknown { .. })
        ));
    }
    // SPIR-V ImageQueryLod is fragment-only. Keep the instruction unsupported
    // elsewhere instead of substituting a constant LOD.
    let raw = canonical;
    let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
    for stage in [0, 1, 2, 3, 4] {
        let mut header = [0u32; 32];
        header[0] = stage << 10;
        let header = shadowbox::headers::parse(&header).unwrap();
        assert!(
            shadowbox::emitter::emit_unoptimized_with_header(&[instruction], Some(&header))
                .is_err()
        );
    }
}

#[test]
fn tmml_fixed_point_outputs_reverse_xy_and_truncate_toward_zero() {
    use shadowbox::emitter::emit_unoptimized_with_header;
    use std::collections::HashMap;
    // Ryujinx InstEmitTexture.cs:915-923 reverses query components then
    // multiplies by 256 and converts FP32 to S32. Inject known image-query
    // values, then execute the actual emitted numeric dependency chain.
    for mask in 1..=3 {
        let raw = tmml(mask, 9);
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        let words = emit_unoptimized_with_header(&[instruction], None)
            .unwrap()
            .words;
        let mut values: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut stored = HashMap::new();
        let mut registers = Vec::new();
        let mut at = 5;
        while at < words.len() {
            let count = (words[at] >> 16) as usize;
            let op = Op::of(words[at]);
            let a = &words[at + 1..at + count];
            let value = match op {
                Op::Constant => Some(vec![a[2]]),
                Op::Variable if a[2] == 7 => {
                    registers.push(a[1]);
                    None
                }
                Op::ImageQueryLod => Some(vec![
                    (-1.501_953_1_f32).to_bits(),
                    3.001_953_1_f32.to_bits(),
                ]),
                Op::CompositeExtract if values.contains_key(&a[2]) => {
                    Some(vec![values[&a[2]][a[3] as usize]])
                }
                Op::FMul if values.contains_key(&a[2]) && values.contains_key(&a[3]) => {
                    Some(vec![(f32::from_bits(values[&a[2]][0])
                        * f32::from_bits(values[&a[3]][0]))
                    .to_bits()])
                }
                Op::ConvertFToS if values.contains_key(&a[2]) => {
                    Some(vec![(f32::from_bits(values[&a[2]][0]) as i32) as u32])
                }
                Op::Bitcast if values.contains_key(&a[2]) => Some(values[&a[2]].clone()),
                Op::Store if values.contains_key(&a[1]) => {
                    stored.insert(a[0], values[&a[1]][0]);
                    None
                }
                _ => None,
            };
            if let Some(value) = value {
                values.insert(a[1], value);
            }
            at += count;
        }
        let expected = [768u32, (-384i32) as u32]
            .into_iter()
            .enumerate()
            .filter_map(|(i, value)| (mask & (1 << i) != 0).then_some(value))
            .collect::<Vec<_>>();
        for (i, value) in expected.into_iter().enumerate() {
            assert_eq!(stored[&registers[5 + i]], value);
        }
    }
}

// Bound fields: envytools gm107.c:1958,1970 and Mesa sm50.rs:2170-2197,
// 2351-2381. Query type requirements: Ryujinx InstEmitTexture.cs:1116-1119.
#[test]
fn bound_query_requires_a_consistent_typed_binding() {
    let tex = |dim: u64, binding: u64| {
        0xc038_u64 << 48
            | PT
            | 1 << 55
            | 15 << 31
            | dim << 28
            | binding << 36
            | 255 << 20
            | 20 << 8
            | 16
    };
    let query = 0xdf48_u64 << 48 | PT | 1 << 22 | 11 << 31 | 7 << 36 | 8 << 8;
    assert!(translation(&[query]).is_err());
    assert!(translation(&[tex(2, 7), query]).is_ok());
    assert!(translation(&[tex(2, 8), query]).is_err());
    assert!(translation(&[tex(2, 7), tex(4, 7), query]).is_err());
    assert!(translation(&[tex(4, 7), query]).is_ok());
}

#[test]
fn bound_texture_rejects_unimplemented_derivatives_and_reserved_lod_modes() {
    for lod in 0..8_u64 {
        for ndv in 0..2_u64 {
            let raw = 0xc038_u64 << 48 | PT | lod << 55 | ndv << 35 | 255 << 20 | 1 << 31;
            let kind = decode_word([raw as u32, (raw >> 32) as u32], 8).kind;
            assert_eq!(
                matches!(kind, InstructionKind::Texture(ops::Texture { .. })),
                matches!(lod, 0..=3 | 6..=7) && ndv == 0
            );
        }
    }
}

#[test]
fn bound_array_and_cube_forms_validate_and_invalid_dimension_stays_unknown() {
    // Dimension encodings: Mesa sm50.rs:2107-2116. Both merge inputs
    // share this table; batch 6 already validates array/cube sampling.
    for dim in [1, 3, 6, 7_u64] {
        let raw = 0xc038_u64 << 48 | PT | dim << 28 | 1 << 55 | 255 << 20 | 15 << 31;
        assert!(matches!(
            decode_word([raw as u32, (raw >> 32) as u32], 8).kind,
            InstructionKind::Texture(ops::Texture { .. })
        ));
        let words = translation(&[raw]).unwrap();
        let check = spirv_validation::validate(&words);
        assert!(
            check.status.success(),
            "{}",
            String::from_utf8_lossy(&check.stderr)
        );
    }
    let raw = 0xc038_u64 << 48 | PT | 5 << 28 | 255 << 20;
    assert!(matches!(
        decode_word([raw as u32, (raw >> 32) as u32], 8).kind,
        InstructionKind::Unknown(ops::Unknown { .. })
    ));
}

#[test]
fn unimplemented_bound_query_selectors_stay_unknown() {
    // Query selectors: envytools gm107.c:882-891. Only dimension is implemented.
    for query in 0..64_u64 {
        let raw = 0xdf48_u64 << 48 | PT | query << 22 | 1 << 31;
        let kind = decode_word([raw as u32, (raw >> 32) as u32], 8).kind;
        assert_eq!(
            matches!(kind, InstructionKind::Texture(ops::Texture { .. })),
            query == 1
        );
    }
}

#[test]
fn bound_three_dimensional_depth_sampling_stays_unknown() {
    // Dimension/DC fields: Mesa sm50.rs:2193,2197. The existing image
    // path rejects this form before SPIR-V emission.
    let raw = 0xc038_u64 << 48 | PT | 4 << 28 | 1 << 50;
    assert!(matches!(
        decode_word([raw as u32, (raw >> 32) as u32], 8).kind,
        InstructionKind::Unknown(ops::Unknown { .. })
    ));
}

#[test]
fn bound_dimensions_follow_mesa_and_non_array_cube_offsets_stay_unknown() {
    // Dimension28..30: Mesa sm50.rs:2105-2119,2193. AOFFI54: :2196.
    for dim in 0..8_u64 {
        for offset in 0..2_u64 {
            let raw = 0xc038_u64 << 48 | PT | dim << 28 | offset << 54 | 255 << 20 | 1 << 31;
            assert_eq!(
                matches!(
                    decode_word([raw as u32, (raw >> 32) as u32], 8).kind,
                    InstructionKind::Texture(ops::Texture { .. })
                ),
                dim != 5 && !(dim == 6 && offset != 0)
            );
        }
    }
}

#[test]
fn requested_texture_forms_without_a_vulkan_lowering_stay_unknown() {
    // LOD4/5 are absent: envytools gm107.c:916-923,1026-1033.
    // NDV35: Mesa sm50.rs:2135-2141,2195. Ryujinx decodes it at
    // InstDecoders.cs:4847,4867 but does not lower it in InstEmitTexture.cs:28-39.
    // 3D DC50: Mesa sm50.rs:2193,2197; Khronos Vulkan-Docs
    // appendices/spirvenv.adoc:379-381, commit
    // e4e53e4b31e13eeaee1ad99fb940aa72b2ec1b14 forbids 3D Dref.
    // TLD CL54: Ryujinx InstDecoders.cs:4922 / InstEmitTexture.cs:65-72.
    for bound in [false, true] {
        let sample = (if bound { 0xc038u64 } else { 0xdeb8 }) << 48
            | if bound { 9 << 36 } else { 0 }
            | 15 << 31
            | 2 << 28
            | 20 << 20
            | PT
            | 8 << 8;
        for mode in [4, 5] {
            let raw = sample | mode << if bound { 55 } else { 37 };
            assert!(matches!(
                decode(raw),
                InstructionKind::Unknown(ops::Unknown { .. })
            ));
        }
        for lod in [0, 1, 2, 3, 6, 7] {
            let sample = sample | lod << if bound { 55 } else { 37 };
            for raw in [
                sample | 1 << 35,
                (sample & !(7 << 28)) | 4 << 28 | 1 << 50,
                (sample & !(7 << 28)) | 6 << 28 | 1 << if bound { 54 } else { 36 },
            ] {
                assert!(
                    matches!(decode(raw), InstructionKind::Unknown(ops::Unknown { .. })),
                    "{raw:016x}"
                );
            }
        }
        let fetch = (if bound { 0xdc38u64 } else { 0xdd38 }) << 48
            | if bound { 9 << 36 } else { 0 }
            | 15 << 31
            | 20 << 20
            | PT
            | 8 << 8;
        for dim in [6, 7] {
            for lod in [0, 1] {
                for offset in [0, 1] {
                    let raw = fetch | dim << 28 | lod << 55 | offset << 35;
                    assert!(matches!(
                        decode(raw),
                        InstructionKind::Unknown(ops::Unknown { .. })
                    ));
                }
            }
        }
        for lod in [0, 1] {
            for offset in [0, 1] {
                let raw = fetch | 2 << 28 | lod << 55 | offset << 35 | 1 << 54;
                assert!(matches!(
                    decode(raw),
                    InstructionKind::Unknown(ops::Unknown { .. })
                ));
            }
        }
    }
}
