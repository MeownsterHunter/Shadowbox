//! Project-owned instructions, encoded from the permitted Maxwell tables.
//! Encoding and semantic citations are in docs/reference-sources.md
use shadowbox::ir::ops;
use shadowbox::spirv::Op;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{Instruction, InstructionKind, Register, Source, TexOp},
};

#[path = "support/spirv_validation.rs"]
mod spirv_validation;

const PT: u64 = 7 << 16;

fn decode(raw: u64) -> Instruction {
    decode_word([raw as u32, (raw >> 32) as u32], 0)
}

fn module(raw: u64, stage: u32) -> Result<Vec<u32>, shadowbox::interface::Stopped> {
    let mut header = [0; 32];
    header[0] = stage << 10;
    let header = headers::parse(&header).unwrap();
    let mut xs = (1..16)
        .map(|r| Instruction {
            address: r * 8,
            predicate: None,
            kind: InstructionKind::Mov(ops::Mov {
                quad_lanes: 15,
                dst: Register(r as u8),
                src: Source::Immediate(0),
            }),
        })
        .collect::<Vec<_>>();
    xs.push(decode(raw | PT));
    xs.push(decode((0xe300 << 48) | PT | 15));
    emit_unoptimized_with_header(&xs, Some(&header)).map(|s| s.words)
}

fn instructions(words: &[u32]) -> Vec<(Op, &[u32])> {
    let mut at = 5;
    let mut result = Vec::new();
    while at < words.len() {
        let end = at + (words[at] >> 16) as usize;
        result.push((Op::of(words[at]), &words[at + 1..end]));
        at = end;
    }
    result
}

#[test]
fn bound_tex_prefixes_preserve_full_table_index_and_nodep() {
    for prefix in [0xc138u64, 0xc13a] {
        for table in [9u64, 0x1009] {
            let raw = (prefix << 48)
                | (table << 36)
                | (15 << 31)
                | (2 << 28)
                | (10 << 20)
                | (8 << 8)
                | PT;
            assert!(
                matches!(decode(raw).kind, InstructionKind::Texture(ops::Texture {
            op: TexOp::Sample { lod: 2 }, bound_handle: Some(index),
            src1: Register(10), nodep, ..
        }) if nodep == (prefix == 0xc13a) && index == table as u16)
            );
            // Compute and fragment, including NODEP's fragment helper selection.
            for stage in [0, 5] {
                let words = module(raw, stage).unwrap();
                let check = spirv_validation::validate(&words);
                assert!(
                    check.status.success(),
                    "{}",
                    String::from_utf8_lossy(&check.stderr)
                );
                let ops = instructions(&words);
                assert_eq!(
                    ops.iter()
                        .filter(|(op, _)| matches!(
                            op,
                            Op::ImageSampleImplicitLod
                                | Op::ImageSampleExplicitLod
                                | Op::ImageSampleDrefImplicitLod
                                | Op::ImageSampleDrefExplicitLod
                        ))
                        .count(),
                    1
                );
                assert!(ops
                    .iter()
                    .any(|(op, a)| *op == Op::Decorate && a[1..] == [33, 2 * table as u32]));
                assert!(ops
                    .iter()
                    .any(|(op, a)| *op == Op::Decorate && a[1..] == [33, 2 * table as u32 + 1]));
                assert!(ops
                    .iter()
                    .any(|(op, a)| *op == Op::Decorate && a[1..] == [34, 1]));
                assert_eq!(
                    ops.iter()
                        .any(|(op, a)| *op == Op::Decorate && a[1..] == [11, 23]),
                    prefix == 0xc13a && stage == 5
                );
            }
        }
    }
}

#[test]
fn bound_tex_all_lods_offset_and_comparison_validate() {
    for lod in 0..4 {
        for offset in 0..2 {
            for depth in 0..2 {
                let raw = (0xc038 << 48)
                    | (lod << 55)
                    | (offset << 54)
                    | (depth << 50)
                    | (3 << 36)
                    | (15 << 31)
                    | (2 << 28)
                    | ((if lod < 2 && offset == 0 && depth == 0 {
                        255
                    } else {
                        10
                    }) << 20)
                    | (8 << 8);
                let words = module(raw, 5).unwrap();
                let check = spirv_validation::validate(&words);
                assert!(
                    check.status.success(),
                    "{raw:016x}: {}",
                    String::from_utf8_lossy(&check.stderr)
                );
                let ops = instructions(&words);
                assert_eq!(
                    ops.iter()
                        .filter(|(op, _)| matches!(
                            op,
                            Op::ImageSampleImplicitLod
                                | Op::ImageSampleExplicitLod
                                | Op::ImageSampleDrefImplicitLod
                                | Op::ImageSampleDrefExplicitLod
                        ))
                        .count(),
                    1
                );
            }
        }
    }
}

#[test]
fn immediate_half_forms_validate_and_preserve_every_field() {
    for prefix in [0x7a02u64, 0x7a05] {
        for low in [0u64, 1, 0x0e0, 0x100, 0x1f0, 0x1ff, 0x200, 0x3ff] {
            for high in [0u64, 1, 0x0e0, 0x1f0, 0x1ff, 0x200, 0x3ff] {
                let raw = (prefix << 48)
                    | (low << 20)
                    | ((high & 511) << 30)
                    | ((high >> 9) << 56)
                    | (8 << 8)
                    | PT;
                assert!(
                    matches!(decode(raw).kind, InstructionKind::HalfArithmetic(ops::HalfArithmetic {
                    src: [_, Source::Immediate(bits), _], ..
                }) if bits == ((low << 6) | (high << 22)) as u32)
                );
                let words = module(raw, 0).unwrap();
                let check = spirv_validation::validate(&words);
                assert!(
                    check.status.success(),
                    "{}",
                    String::from_utf8_lossy(&check.stderr)
                );
            }
        }
    }
    let half = (0x7a00 << 48) | (0xe0 << 20) | (0xf0 << 30) | (8 << 8) | PT;
    // FTZ39 is implemented; unnamed fields still reject (envytools gm107.c:1977).
    assert!(matches!(
        decode(half | (1 << 39)).kind,
        InstructionKind::HalfArithmetic(ops::HalfArithmetic {
            flush: shadowbox::ir::HalfFlush::Ftz,
            ..
        })
    ));
    for bit in [40, 41, 42, 45, 46, 51, 53, 54] {
        assert!(
            matches!(
                decode(half | (1 << bit)).kind,
                InstructionKind::Unknown(ops::Unknown { .. })
            ),
            "bit{bit}"
        );
    }
    let tex = (0xc038 << 48) | (1 << 31) | (2 << 28) | (255 << 20) | (8 << 8) | PT;
    assert!(matches!(
        decode(tex).kind,
        InstructionKind::Texture(ops::Texture { .. })
    ));
    // LOD0/1 without other auxiliaries consumes no B words; unexplained B
    // values must be Unknown instead of silently dropping a decoded source.
    for lod in 0..2 {
        let raw = (tex & !(255 << 20)) | (10 << 20) | (lod << 55);
        assert!(matches!(
            decode(raw).kind,
            InstructionKind::Unknown(ops::Unknown { .. })
        ));
    }
    for raw in [tex | (1 << 35), tex | (1 << 58), tex & !(1 << 51)] {
        assert!(matches!(
            decode(raw).kind,
            InstructionKind::Unknown(ops::Unknown { .. })
        ));
    }
    for lod in 4..6 {
        assert!(matches!(
            decode(tex | (lod << 55)).kind,
            InstructionKind::Unknown(ops::Unknown { .. })
        ));
    }
}

#[test]
fn unresolved_blockers_remain_unknown_in_every_stage() {
    // Reserved VTG mode3 (envytools gm107.c:1637), unidentified rows, generic ST.WT.B128, reserved HMUL2_32I
    // flush mode3, and immediate LOP3.X. Never substitute an unrelated op.
    for prefix in [0x50e3u64, 0x4b09, 0xbfc9, 0x38d1, 0x2b8c, 0x3f7f] {
        let raw = (prefix << 48) | PT;
        assert!(
            matches!(decode(raw).kind, InstructionKind::Unknown(ops::Unknown { raw: w })
            if w == [raw as u32, (raw >> 32) as u32])
        );
        for stage in 0..=5 {
            assert!(module(raw, stage).is_err(), "{prefix:04x}, stage{stage}");
        }
    }
    assert_eq!(
        include_str!("../src/decoder/families.rs")
            .matches("x.guarded(InstructionKind::Nop(ops::Nop))")
            .count(),
        2
    );
}
