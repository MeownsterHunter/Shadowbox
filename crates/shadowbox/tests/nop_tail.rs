#[path = "support/spirv_validation.rs"]
mod spirv_validation;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{Instruction, InstructionKind},
};

const REGIONS: &[(u64, u64, &[u32])] = &[
    (
        0xfffb_ffff_fff7_e0ff,
        0x50b2_0000_0007_0000,
        &[8, 9, 10, 11, 12, 19, 50],
    ),
    (
        0xfffb_ffff_fff7_e0ff,
        0x50b2_28de_18a7_00f2,
        &[8, 9, 10, 11, 12, 19, 50],
    ),
    (
        0xfffb_ffff_fff7_e0ff,
        0x50b2_6f7a_8a07_00b2,
        &[8, 9, 10, 11, 12, 19, 50],
    ),
    (
        0xfffb_ffff_fff7_e0ff,
        0x50b2_953c_3f97_8072,
        &[8, 9, 10, 11, 12, 19, 50],
    ),
    (
        0xfffb_ffff_fff7_e0ff,
        0x50b2_9607_1787_0070,
        &[8, 9, 10, 11, 12, 19, 50],
    ),
    (
        0xfffb_ffff_fff7_e2ff,
        0x50b2_87f3_87f7_8061,
        &[8, 10, 11, 12, 19, 50],
    ),
    (
        0xfffb_ffff_fff7_e8ff,
        0x50b2_f38f_1097_40df,
        &[8, 9, 10, 12, 19, 50],
    ),
    (
        0xffff_ffff_fff7_e0ff,
        0x50b4_0000_0007_0000,
        &[8, 9, 10, 11, 12, 19],
    ),
    (
        0xffff_ffff_fff7_e0ff,
        0x50b4_28de_18a7_00f2,
        &[8, 9, 10, 11, 12, 19],
    ),
    (
        0xffff_ffff_fff7_e0ff,
        0x50b4_6f7a_8a07_00b2,
        &[8, 9, 10, 11, 12, 19],
    ),
    (
        0xffff_ffff_fff7_e0ff,
        0x50b4_953c_3f97_8072,
        &[8, 9, 10, 11, 12, 19],
    ),
    (
        0xffff_ffff_fff7_e0ff,
        0x50b4_9607_1787_0070,
        &[8, 9, 10, 11, 12, 19],
    ),
    (
        0xffff_ffff_fff7_e2ff,
        0x50b4_87f3_87f7_8061,
        &[8, 10, 11, 12, 19],
    ),
    (
        0xffff_ffff_fff7_e8ff,
        0x50b4_f38f_1097_40df,
        &[8, 9, 10, 12, 19],
    ),
];

fn words() -> impl Iterator<Item = u64> {
    REGIONS.iter().flat_map(|&(_, base, bits)| {
        (0..(1_u16 << bits.len())).map(move |selector| {
            bits.iter().enumerate().fold(base, |word, (index, bit)| {
                word | u64::from((selector >> index) & 1) << bit
            })
        })
    })
}

fn contains(raw: u64) -> bool {
    REGIONS.iter().any(|&(mask, value, _)| raw & mask == value)
}

fn kind(raw: u64) -> InstructionKind {
    decode_word([raw as u32, (raw >> 32) as u32], 0).kind
}

fn parent_admitted(raw: u64) -> bool {
    const PARENT_REGIONS: [(u64, u64); 7] = [
        (0xfff9_ffff_fff7_e0ff, 0x50b1_0000_0007_0000),
        (0xfff9_ffff_fff7_e0ff, 0x50b1_28de_18a7_00f2),
        (0xfff9_ffff_fff7_e0ff, 0x50b1_6f7a_8a07_00b2),
        (0xfff9_ffff_fff7_e0ff, 0x50b1_953c_3f97_8072),
        (0xfff9_ffff_fff7_e0ff, 0x50b1_9607_1787_0070),
        (0xfff9_ffff_fff7_e2ff, 0x50b1_87f3_87f7_8061),
        (0xfff9_ffff_fff7_e8ff, 0x50b1_f38f_1097_40df),
    ];
    (raw >> 48) & 7 == 0
        || PARENT_REGIONS
            .iter()
            .any(|&(mask, value)| raw & mask == value)
}

#[test]
fn all_1152_tail_words_decode_to_typed_nop() {
    assert_eq!(
        words()
            .inspect(|&raw| assert!(matches!(kind(raw), InstructionKind::Nop(_)), "{raw:016x}"))
            .count(),
        1152
    );
}

#[test]
fn exact_tail_cube_boundaries_hold() {
    for raw in words() {
        for bit in 0..64 {
            let neighbor = raw ^ (1 << bit);
            assert_eq!(
                matches!(kind(neighbor), InstructionKind::Nop(_)),
                contains(neighbor) || parent_admitted(neighbor),
                "{neighbor:016x} from {raw:016x} bit {bit}"
            );
        }
    }
}

#[test]
fn all_1152_tail_words_validate_raw_and_optimized_spirv() {
    let header = headers::parse(&[0; 32]).unwrap();
    for raw in words() {
        let decoded = decode_word([raw as u32, (raw >> 32) as u32], 0);
        let mut module = emit_unoptimized_with_header(
            &[Instruction {
                address: 0,
                predicate: decoded.predicate,
                kind: decoded.kind,
            }],
            Some(&header),
        )
        .unwrap();
        for variant in ["raw", "optimized"] {
            let result = spirv_validation::validate(&module.words);
            assert!(
                result.status.success(),
                "{raw:016x}/{variant}: {}",
                String::from_utf8_lossy(&result.stderr)
            );
            module.optimize();
        }
    }
}
