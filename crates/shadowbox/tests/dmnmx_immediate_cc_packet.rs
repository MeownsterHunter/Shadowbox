//! Exact hybrid DMNMX immediate condition-code bundle coverage.

#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{
        ops, DoubleMinMaxOperation, DoubleSource, DoubleSourceModifier, FloatFlags, Instruction,
        InstructionKind, Predicate, Register, RegisterPair,
    },
};

const PACKET_MASK: u64 = 0xfef8_9bff_fff7_ffff;
const PACKET_BASES: [u64; 7] = [
    0x3850_8000_0007_0000,
    0x3850_805d_3db7_5ce2,
    0x3850_822e_ade7_8146,
    0x3850_82e3_d857_c6ee,
    0x3850_83b0_d857_31fe,
    0x3850_885c_2af7_8254,
    0x3850_8a5a_3f57_c9ab,
];
const VARIABLE_BITS: [u32; 8] = [19, 42, 45, 46, 48, 49, 50, 56];
const LEGACY_EXACT: [u64; 15] = [
    0x5c50_8000_0007_0000,
    0x5c51_e26d_1df0_3060,
    0x5c53_abe9_f630_8c0c,
    0x5c54_8037_73ec_86fe,
    0x5c57_d1ca_73fe_9cd5,
    0x4c50_8000_0007_0000,
    0x4c50_f2c6_4fc6_7370,
    0x4c54_8129_8a24_86bd,
    0x4c54_df5c_97d5_b24a,
    0x4c57_9e13_2e27_584f,
    0x3853_858d_815e_e051,
    0x3855_b18f_0e65_7ed7,
    0x3952_81b4_453c_4204,
    0x3952_c50d_ef49_2290,
    0x3956_f5db_a153_81c0,
];

fn packet_words() -> impl Iterator<Item = u64> {
    PACKET_BASES.into_iter().flat_map(|base| {
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

fn packet_contains(raw: u64) -> bool {
    PACKET_BASES.contains(&(raw & PACKET_MASK))
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

fn operation(raw: u64) -> DoubleMinMaxOperation {
    match (field(raw, 39, 3), field(raw, 42, 1)) {
        (7, 0) => DoubleMinMaxOperation::Minimum,
        (7, 1) => DoubleMinMaxOperation::Maximum,
        (register, inverted) => DoubleMinMaxOperation::Selected(Predicate {
            register: register as u8,
            inverted: inverted != 0,
        }),
    }
}

#[test]
fn all_1792_words_decode_to_complete_typed_state() {
    let mut count = 0;
    for raw in packet_words() {
        assert_eq!(raw & PACKET_MASK, PACKET_BASES[count / 256]);
        assert_eq!(
            kind(raw),
            InstructionKind::DoubleMinMax(ops::DoubleMinMax {
                dst: RegisterPair(Register(field(raw, 0, 8) as u8)),
                lhs: RegisterPair(Register(field(raw, 8, 8) as u8)),
                rhs: DoubleSource::Immediate(field(raw, 20, 19) << 12 | field(raw, 56, 1) << 31),
                operation: operation(raw),
                modifiers: [modifier(raw, 46, 48), modifier(raw, 49, 45)],
                flags: FloatFlags::Written,
            }),
            "{raw:016x}"
        );
        count += 1;
    }
    assert_eq!(count, 1_792);
}

#[test]
fn packet_boundary_does_not_admit_unqualified_neighbors() {
    for raw in packet_words() {
        for bit in 0..64 {
            if VARIABLE_BITS.contains(&bit) || bit == 47 {
                continue;
            }
            let neighbor = raw ^ (1_u64 << bit);
            let admitted = matches!(
                kind(neighbor),
                InstructionKind::DoubleMinMax(ops::DoubleMinMax {
                    flags: FloatFlags::Written,
                    ..
                })
            );
            assert_eq!(
                admitted,
                packet_contains(neighbor) || LEGACY_EXACT.contains(&neighbor),
                "neighbor {neighbor:016x} from {raw:016x} bit {bit}"
            );
        }
    }
}

#[test]
fn all_1792_words_validate_as_raw_and_optimized_spirv() {
    let header = headers::parse(&[0; 32]).unwrap();
    for raw in packet_words() {
        let instruction = Instruction {
            address: 0,
            predicate: decode_word([raw as u32, (raw >> 32) as u32], 0).predicate,
            kind: kind(raw),
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
