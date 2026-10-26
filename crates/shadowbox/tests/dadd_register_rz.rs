//! Exhaustive DADD.RZ register words.

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

const CUBE_MASK: u64 = 0xfff8_1fff_fff7_ffff;
const CUBE_BASES: [u64; 8] = [
    0x5c70_0180_0007_0000,
    0x5c70_0180_a427_440b,
    0x5c70_03e1_8077_a0d1,
    0x5c70_07dc_cc47_3d36,
    0x5c70_0db2_f917_817b,
    0x5c70_0db4_1db7_aa3a,
    0x5c70_1da4_8b47_e19b,
    0x5c70_1daa_6b67_2c35,
];
const VARIABLE_BITS: [u32; 7] = [19, 45, 46, 47, 48, 49, 50];
const CBUF_RZ_BASES: [u64; 9] = [
    0x4c70_0180_0000_0000,
    0x4c70_0180_0007_0000,
    0x4c70_0bde_fac7_4812,
    0x4c70_0f83_90c7_921f,
    0x4c70_17d1_9b07_39ac,
    0x4c70_17ec_8347_824a,
    0x4c70_19fd_a827_4194,
    0x4c70_1de6_b537_0757,
    0x4c70_1f80_0ff7_ffff,
];

fn cube_words() -> impl Iterator<Item = u64> {
    CUBE_BASES.into_iter().flat_map(|base| {
        (0_u8..128).map(move |selector| {
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

fn cbuf_rz_contains(raw: u64) -> bool {
    CBUF_RZ_BASES.contains(&(raw & CUBE_MASK))
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
fn all_1024_words_decode_to_complete_typed_state() {
    let mut count = 0;
    for raw in cube_words() {
        assert_eq!(raw & CUBE_MASK, CUBE_BASES[count / 128]);
        assert_eq!(
            kind(raw),
            InstructionKind::DoubleAdd(ops::DoubleAdd {
                dst: RegisterPair(Register(field(raw, 0, 8) as u8)),
                lhs: RegisterPair(Register(field(raw, 8, 8) as u8)),
                rhs: DoubleSource::RegisterPair(RegisterPair(Register(field(raw, 20, 8) as u8))),
                modifiers: [modifier(raw, 46, 48), modifier(raw, 49, 45)],
                round: RoundingMode::Zero,
                flags: [FloatFlags::Kept, FloatFlags::Written][field(raw, 47, 1) as usize],
            }),
            "{raw:016x}"
        );
        count += 1;
    }
    assert_eq!(count, 1_024);
}

#[test]
fn boundary_preserves_only_decoded_rounding_neighbors() {
    for raw in cube_words() {
        for bit in 0..64 {
            if VARIABLE_BITS.contains(&bit) || matches!(bit, 39 | 40) {
                continue;
            }
            let neighbor = raw ^ (1_u64 << bit);
            let admitted = matches!(
                kind(neighbor),
                InstructionKind::DoubleAdd(ops::DoubleAdd {
                    round: RoundingMode::Zero,
                    ..
                })
            );
            assert_eq!(
                admitted,
                cube_contains(neighbor) || cbuf_rz_contains(neighbor),
                "{neighbor:016x} from {raw:016x} bit {bit}"
            );
        }

        let rn = raw ^ (1_u64 << 39) ^ (1_u64 << 40);
        if field(raw, 47, 1) == 0 {
            assert!(
                matches!(
                    kind(rn),
                    InstructionKind::DoubleAdd(ops::DoubleAdd {
                        round: RoundingMode::NearestEven,
                        flags: FloatFlags::Kept,
                        ..
                    })
                ),
                "admitted RN sibling {rn:016x}"
            );
        } else {
            assert!(
                matches!(kind(rn), InstructionKind::Unknown(_)),
                "unlisted RN.CC sibling {rn:016x}"
            );
        }

        let rm = raw ^ (1_u64 << 40);
        let rp = raw ^ (1_u64 << 39);
        assert!(
            matches!(
                kind(rm),
                InstructionKind::DoubleAdd(ops::DoubleAdd {
                    round: RoundingMode::NegativeInfinity,
                    ..
                })
            ),
            "admitted RM sibling {rm:016x}"
        );
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
    }
}

#[test]
fn all_1024_words_validate_as_raw_and_optimized_spirv() {
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
