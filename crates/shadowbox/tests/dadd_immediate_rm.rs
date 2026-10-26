//! Exhaustive DADD.RM immediate words.

#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{
        ops, DoubleSource, DoubleSourceModifier, FloatFlags, Instruction, InstructionKind,
        Register, RegisterPair, RoundingMode,
    },
};

const CUBE_MASK: u64 = 0xfef8_1fff_fff7_ffff;
const CUBE_BASES: [u64; 8] = [
    0x3870_0080_0007_0000,
    0x3870_02cd_7ec7_7914,
    0x3870_0ca7_1987_c17e,
    0x3870_0ec3_ba37_5b2b,
    0x3870_18e6_f617_7807,
    0x3870_1e80_0ff7_ffff,
    0x3870_1e84_6767_b14c,
    0x3870_1efa_5ea7_cf2a,
];
const VARIABLE_BITS: [u32; 8] = [19, 45, 46, 47, 48, 49, 50, 56];

fn cube_words() -> impl Iterator<Item = u64> {
    CUBE_BASES.into_iter().flat_map(|base| {
        (0_u16..256).map(move |selector| {
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

fn modifier(raw: u64, absolute: u32, negate: u32) -> DoubleSourceModifier {
    match (field(raw, absolute, 1), field(raw, negate, 1)) {
        (0, 0) => DoubleSourceModifier::Plain,
        (1, 0) => DoubleSourceModifier::Absolute,
        (0, 1) => DoubleSourceModifier::Negated,
        (1, 1) => DoubleSourceModifier::NegatedAbsolute,
        _ => unreachable!(),
    }
}

#[test]
fn all_2048_words_decode_to_complete_typed_state() {
    let mut count = 0;
    for raw in cube_words() {
        assert_eq!(raw & CUBE_MASK, CUBE_BASES[count / 256]);
        assert_eq!(
            kind(raw),
            InstructionKind::DoubleAdd(ops::DoubleAdd {
                dst: RegisterPair(Register(field(raw, 0, 8) as u8)),
                lhs: RegisterPair(Register(field(raw, 8, 8) as u8)),
                rhs: DoubleSource::Immediate(field(raw, 20, 19) << 12 | field(raw, 56, 1) << 31),
                modifiers: [modifier(raw, 46, 48), modifier(raw, 49, 45)],
                round: RoundingMode::NegativeInfinity,
                flags: [FloatFlags::Kept, FloatFlags::Written][field(raw, 47, 1) as usize],
            }),
            "{raw:016x}"
        );
        count += 1;
    }
    assert_eq!(count, 2_048);
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
                InstructionKind::DoubleAdd(ops::DoubleAdd {
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

        let rp = raw ^ (1_u64 << 39) ^ (1_u64 << 40);
        let rz = raw ^ (1_u64 << 40);
        assert!(
            matches!(
                kind(rp),
                InstructionKind::DoubleAdd(ops::DoubleAdd {
                    round: RoundingMode::PositiveInfinity,
                    ..
                })
            ),
            "admitted RP sibling {rp:016x}"
        );
        assert!(
            matches!(
                kind(rz),
                InstructionKind::DoubleAdd(ops::DoubleAdd {
                    round: RoundingMode::Zero,
                    ..
                })
            ),
            "admitted RZ sibling {rz:016x}"
        );
    }
}

#[test]
fn all_2048_words_validate_as_raw_and_optimized_spirv() {
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
