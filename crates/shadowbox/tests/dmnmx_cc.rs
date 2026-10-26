#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{ops, FloatFlags, Instruction, InstructionKind},
};

const FORMS: [u64; 15] = [
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

fn kind(raw: u64) -> InstructionKind {
    decode_word([raw as u32, (raw >> 32) as u32], 0).kind
}

#[test]
fn exact_fifteen_construct_written_flags_and_validate() {
    let header = headers::parse(&[0; 32]).unwrap();
    for raw in FORMS {
        assert!(matches!(
            kind(raw),
            InstructionKind::DoubleMinMax(ops::DoubleMinMax {
                flags: FloatFlags::Written,
                ..
            })
        ));
        assert!(matches!(
            kind(raw & !(1 << 47)),
            InstructionKind::DoubleMinMax(ops::DoubleMinMax {
                flags: FloatFlags::Kept,
                ..
            })
        ));
        let instruction = Instruction {
            address: 0,
            predicate: None,
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

#[test]
fn condition_code_admission_is_exact() {
    for raw in FORMS {
        for bit in 0..64 {
            if bit == 47 {
                continue;
            }
            let neighbor = raw ^ (1_u64 << bit);
            if FORMS.contains(&neighbor) {
                continue;
            }
            assert!(
                !matches!(
                    kind(neighbor),
                    InstructionKind::DoubleMinMax(ops::DoubleMinMax {
                        flags: FloatFlags::Written,
                        ..
                    })
                ),
                "unlisted neighbor {neighbor:016x} admitted from {raw:016x} bit {bit}"
            );
        }
    }
}
