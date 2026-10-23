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
        0xfff9_ffff_fff7_e0ff,
        0x50b1_0000_0007_0000,
        &[8, 9, 10, 11, 12, 19, 49, 50],
    ),
    (
        0xfff9_ffff_fff7_e0ff,
        0x50b1_28de_18a7_00f2,
        &[8, 9, 10, 11, 12, 19, 49, 50],
    ),
    (
        0xfff9_ffff_fff7_e0ff,
        0x50b1_6f7a_8a07_00b2,
        &[8, 9, 10, 11, 12, 19, 49, 50],
    ),
    (
        0xfff9_ffff_fff7_e0ff,
        0x50b1_953c_3f97_8072,
        &[8, 9, 10, 11, 12, 19, 49, 50],
    ),
    (
        0xfff9_ffff_fff7_e0ff,
        0x50b1_9607_1787_0070,
        &[8, 9, 10, 11, 12, 19, 49, 50],
    ),
    (
        0xfff9_ffff_fff7_e2ff,
        0x50b1_87f3_87f7_8061,
        &[8, 10, 11, 12, 19, 49, 50],
    ),
    (
        0xfff9_ffff_fff7_e8ff,
        0x50b1_f38f_1097_40df,
        &[8, 9, 10, 12, 19, 49, 50],
    ),
];
fn words() -> impl Iterator<Item = u64> {
    REGIONS.iter().flat_map(|&(_, base, bits)| {
        (0..(1_u16 << bits.len())).map(move |s| {
            bits.iter()
                .enumerate()
                .fold(base, |r, (i, b)| r | u64::from((s >> i) & 1) << b)
        })
    })
}
fn contains(raw: u64) -> bool {
    REGIONS.iter().any(|&(m, v, _)| raw & m == v)
}
fn tail_contains(raw: u64) -> bool {
    const TAIL: [(u64, u64); 14] = [
        (0xfffb_ffff_fff7_e0ff, 0x50b2_0000_0007_0000),
        (0xfffb_ffff_fff7_e0ff, 0x50b2_28de_18a7_00f2),
        (0xfffb_ffff_fff7_e0ff, 0x50b2_6f7a_8a07_00b2),
        (0xfffb_ffff_fff7_e0ff, 0x50b2_953c_3f97_8072),
        (0xfffb_ffff_fff7_e0ff, 0x50b2_9607_1787_0070),
        (0xfffb_ffff_fff7_e2ff, 0x50b2_87f3_87f7_8061),
        (0xfffb_ffff_fff7_e8ff, 0x50b2_f38f_1097_40df),
        (0xffff_ffff_fff7_e0ff, 0x50b4_0000_0007_0000),
        (0xffff_ffff_fff7_e0ff, 0x50b4_28de_18a7_00f2),
        (0xffff_ffff_fff7_e0ff, 0x50b4_6f7a_8a07_00b2),
        (0xffff_ffff_fff7_e0ff, 0x50b4_953c_3f97_8072),
        (0xffff_ffff_fff7_e0ff, 0x50b4_9607_1787_0070),
        (0xffff_ffff_fff7_e2ff, 0x50b4_87f3_87f7_8061),
        (0xffff_ffff_fff7_e8ff, 0x50b4_f38f_1097_40df),
    ];
    TAIL.iter().any(|&(mask, value)| raw & mask == value)
}
fn kind(raw: u64) -> InstructionKind {
    decode_word([raw as u32, (raw >> 32) as u32], 0).kind
}

#[test]
fn all_1536_words_decode_to_typed_nop() {
    assert_eq!(
        words()
            .inspect(|&r| assert!(matches!(kind(r), InstructionKind::Nop(_)), "{r:016x}"))
            .count(),
        1536
    );
}
#[test]
fn exact_cube_boundaries_hold() {
    for raw in words() {
        for bit in 0..64 {
            let n = raw ^ (1 << bit);
            let expected = contains(n) || tail_contains(n) || ((n >> 48) as u16 == 0x50b0);
            assert_eq!(
                matches!(kind(n), InstructionKind::Nop(_)),
                expected,
                "{n:016x} from {raw:016x} bit {bit}"
            );
        }
    }
}
#[test]
fn all_1536_words_validate_raw_and_optimized_spirv() {
    let h = headers::parse(&[0; 32]).unwrap();
    for raw in words() {
        let d = decode_word([raw as u32, (raw >> 32) as u32], 0);
        let mut m = emit_unoptimized_with_header(
            &[Instruction {
                address: 0,
                predicate: d.predicate,
                kind: d.kind,
            }],
            Some(&h),
        )
        .unwrap();
        for v in ["raw", "optimized"] {
            let x = spirv_validation::validate(&m.words);
            assert!(
                x.status.success(),
                "{raw:016x}/{v}: {}",
                String::from_utf8_lossy(&x.stderr)
            );
            m.optimize();
        }
    }
}
