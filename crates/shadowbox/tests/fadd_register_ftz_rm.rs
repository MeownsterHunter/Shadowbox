//! Exhaustive FADD.FTZ.RM register words.

#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{
        ops, Denormals, FAddModifiers, FloatFlags, Instruction, InstructionKind, Register,
        RoundingMode, Saturation, Source, SourceAbs, SourceNegation,
    },
};

const CUBE_MASK: u64 = 0xfffc_1fff_fff7_ffff;
const CUBE_BASES: [u64; 11] = [
    0x5c58_1080_0000_0000,
    0x5c58_1080_0007_0000,
    0x5c58_10c1_e777_1d6e,
    0x5c58_1286_3767_da24,
    0x5c58_14bd_d8e7_972e,
    0x5c58_14f1_e8b7_f5ce,
    0x5c58_18d7_7b67_7d3d,
    0x5c58_1ad8_99f7_42aa,
    0x5c58_1ccf_af87_30a7,
    0x5c58_1e8d_1517_72c3,
    0x5c58_1e90_6e97_1225,
];
const VARIABLE_BITS: [u32; 6] = [19, 45, 46, 47, 48, 49];

fn cube_words() -> impl Iterator<Item = u64> {
    CUBE_BASES.into_iter().flat_map(|base| {
        (0_u8..64).map(move |selector| {
            VARIABLE_BITS
                .iter()
                .enumerate()
                .fold(base, |raw, (index, bit)| {
                    raw | u64::from(selector >> index & 1) << bit
                })
        })
    })
}

fn cube_contains(raw: u64) -> bool {
    CUBE_BASES.contains(&(raw & CUBE_MASK))
}

fn kind(raw: u64) -> InstructionKind {
    decode_word([raw as u32, (raw >> 32) as u32], 0).kind
}

fn field(raw: u64, at: u32, width: u32) -> u32 {
    ((raw >> at) & ((1_u64 << width) - 1)) as u32
}

#[test]
fn all_704_words_decode_to_complete_typed_state() {
    let mut count = 0;
    for raw in cube_words() {
        assert_eq!(raw & CUBE_MASK, CUBE_BASES[count / 64]);
        assert_eq!(
            kind(raw),
            InstructionKind::FAdd(ops::FAdd {
                dst: Register(field(raw, 0, 8) as u8),
                lhs: Register(field(raw, 8, 8) as u8),
                rhs: Source::Register(Register(field(raw, 20, 8) as u8)),
                modifiers: FAddModifiers::Operand {
                    lhs_abs: [SourceAbs::Plain, SourceAbs::Absolute][field(raw, 46, 1) as usize],
                    rhs_abs: [SourceAbs::Plain, SourceAbs::Absolute][field(raw, 49, 1) as usize],
                    lhs_negation: [SourceNegation::Plain, SourceNegation::Negated]
                        [field(raw, 48, 1) as usize],
                    rhs_negation: [SourceNegation::Plain, SourceNegation::Negated]
                        [field(raw, 45, 1) as usize],
                    saturation: Saturation::Unclamped,
                    denormals: Denormals::FlushedToZero,
                },
                round: RoundingMode::NegativeInfinity,
                flags: [FloatFlags::Kept, FloatFlags::Written][field(raw, 47, 1) as usize],
            }),
            "{raw:016x}"
        );
        count += 1;
    }
    assert_eq!(count, 704);
}

#[test]
fn boundary_does_not_admit_other_rm_neighbors() {
    for raw in cube_words() {
        for bit in 0..64 {
            if VARIABLE_BITS.contains(&bit) || matches!(bit, 39 | 40) {
                continue;
            }
            let neighbor = raw ^ (1_u64 << bit);
            let admitted = matches!(
                kind(neighbor),
                InstructionKind::FAdd(ops::FAdd {
                    round: RoundingMode::NegativeInfinity,
                    ..
                })
            );
            assert_eq!(
                admitted,
                cube_contains(neighbor),
                "{neighbor:016x} from {raw:016x} bit {bit}"
            );
        }

        let rn = raw ^ (1_u64 << 39);
        assert!(
            matches!(
                kind(rn),
                InstructionKind::FAdd(ops::FAdd {
                    round: RoundingMode::NearestEven,
                    ..
                })
            ),
            "admitted RN sibling {rn:016x}"
        );
        for other in [raw ^ (1_u64 << 40), raw ^ (3_u64 << 39)] {
            assert!(
                matches!(kind(other), InstructionKind::Unknown(_)),
                "unlisted directed sibling {other:016x}"
            );
        }
    }
}

#[test]
fn all_704_words_validate_as_raw_and_optimized_spirv() {
    let header = headers::parse(&[0; 32]).unwrap();
    for raw in cube_words() {
        let decoded = decode_word([raw as u32, (raw >> 32) as u32], 0);
        let instruction = Instruction {
            address: 0,
            predicate: decoded.predicate,
            kind: decoded.kind,
        };
        let mut module = emit_unoptimized_with_header(&[instruction], Some(&header)).unwrap();
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
