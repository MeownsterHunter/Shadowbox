//! Exact CC.LT slice of the register BRX cubes.

#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::{
    decoder::{decode_stream, decode_word},
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{ops, BrxBase, Instruction, InstructionKind, Predicate, Register},
};

const CUBES: [(u64, u64); 2] = [
    (0xe250_0000_0007_0001, 0x000f_0000_0008_0000),
    (0xe250_5bb1_b767_1201, 0x000f_0000_0008_0000),
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

fn contains(raw: u64, cubes: &[(u64, u64)]) -> bool {
    cubes
        .iter()
        .any(|&(base, variable)| raw & !variable == base)
}

fn parent_admitted(raw: u64) -> bool {
    const CC_F_CUBES: [(u64, u64); 2] = [
        (0xe250_0000_0007_0000, 0x000f_0000_0008_0000),
        (0xe250_5bb1_b767_1200, 0x000f_0000_0008_0000),
    ];
    contains(raw, &CC_F_CUBES)
        || (raw >> 48 == 0xe250 && raw & 0xff == 0x0f && raw >> 44 & 0xf == 0)
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
                        cc_test: 1,
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
                contains(neighbor, &CUBES) || parent_admitted(neighbor),
                "{neighbor:016x} from {raw:016x} bit {bit}"
            );
        }
    }
}

#[test]
fn conditional_register_brx_discovers_runtime_target_slots() {
    // CC.LT can take an unresolved register target. Discovery must retain
    // the target slot instead of emitting the dynamic edge to EXIT.
    let raw = 0xe250_u64 << 48 | 7 << 16 | 1;
    let mov = 0x0100_u64 << 48 | 2 << 20 | 15 << 12 | 7 << 16 | 240;
    let exit = 0xe300_u64 << 48 | 7 << 16 | 15;
    let nop = 0x50b0_u64 << 48 | 7 << 16;
    let skip = 0xe240_u64 << 48 | 16 << 20 | 7 << 16 | 15;
    let code = [0, raw, nop, skip, 0, mov, exit, nop]
        .into_iter()
        .flat_map(|word| [word as u32, (word >> 32) as u32])
        .collect::<Vec<_>>();
    let xs = decode_stream(shadowbox::QuietRun, &code);
    assert!(xs.iter().any(|x| x.address == 40));
    assert!(matches!(
        xs.iter().find(|x| x.address == 8).map(|x| x.kind),
        Some(InstructionKind::BrxRegister(ops::BrxRegister {
            cc_test: 1,
            ..
        }))
    ));
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
