//! Exact CC.F slice of the register BRX cubes.

#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{ops, BrxBase, Instruction, InstructionKind, Predicate, Register},
};

// Two register BRX cubes, fixing CC0..4 to F and LMT clear.
const CUBES: [(u64, u64); 2] = [
    (0xe250_0000_0007_0000, 0x000f_0000_0008_0000),
    (0xe250_5bb1_b767_1200, 0x000f_0000_0008_0000),
];

fn words() -> impl Iterator<Item = u64> {
    CUBES.into_iter().flat_map(|(base, variable)| {
        let bits: Vec<u32> = (0..64).filter(|bit| variable & (1 << bit) != 0).collect();
        (0..(1_u16 << bits.len())).map(move |selector| {
            bits.iter().enumerate().fold(base, |word, (index, bit)| {
                word | u64::from((selector >> index) & 1) << bit
            })
        })
    })
}

fn contains(raw: u64) -> bool {
    CUBES
        .iter()
        .any(|&(base, variable)| raw & !variable == base)
}

fn parent_admitted(raw: u64) -> bool {
    const CC_LT_CUBES: [(u64, u64); 2] = [
        (0xe250_0000_0007_0001, 0x000f_0000_0008_0000),
        (0xe250_5bb1_b767_1201, 0x000f_0000_0008_0000),
    ];
    CC_LT_CUBES
        .iter()
        .any(|&(base, variable)| raw & !variable == base)
        || raw >> 48 == 0xe250 && raw & 0xff == 0x0f && raw >> 44 & 0xf == 0
}

fn field(raw: u64, at: u32, width: u32) -> u32 {
    ((raw >> at) & ((1_u64 << width) - 1)) as u32
}

#[test]
fn all_64_words_decode_to_complete_typed_state() {
    assert_eq!(
        words()
            .inspect(|&raw| {
                let decoded = decode_word([raw as u32, (raw >> 32) as u32], 0);
                assert_eq!(
                    decoded.kind,
                    InstructionKind::BrxRegister(ops::BrxRegister {
                        displacement: Register(field(raw, 8, 8) as u8),
                        base: BrxBase::decode(0, field(raw, 20, 24)),
                        cc_test: 0,
                    }),
                    "{raw:016x}"
                );
                assert_eq!(
                    decoded.predicate,
                    if field(raw, 19, 1) == 0 {
                        None
                    } else {
                        Some(Predicate {
                            register: 7,
                            inverted: true,
                        })
                    },
                    "{raw:016x}"
                );
            })
            .count(),
        64
    );
}

#[test]
fn exact_slice_boundaries_hold() {
    for raw in words() {
        for bit in 0..64 {
            let neighbor = raw ^ (1_u64 << bit);
            let admitted = matches!(
                decode_word([neighbor as u32, (neighbor >> 32) as u32], 0).kind,
                InstructionKind::BrxRegister(_)
            );
            assert_eq!(
                admitted,
                contains(neighbor) || parent_admitted(neighbor),
                "{neighbor:016x} from {raw:016x} bit {bit}"
            );
        }
    }
}

#[test]
fn all_64_words_validate_as_raw_and_optimized_spirv() {
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
            let validation = spirv_validation::validate(&module.words);
            assert!(
                validation.status.success(),
                "{raw:016x}/{variant}: {}",
                String::from_utf8_lossy(&validation.stderr)
            );
            module.optimize();
        }
    }
}
