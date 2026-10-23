#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{Instruction, InstructionKind, LdcAddressMode, LdcWidth, Register},
};

const FORMS: [(u64, LdcWidth, u8, u8, i32, u8); 5] = [
    (0xef90_7b7b_c947_9d94, LdcWidth::U8, 148, 157, -17260, 23),
    (0xef90_b3f4_86f7_3aae, LdcWidth::U8, 174, 58, 18543, 31),
    (0xef91_7b7b_c947_9d94, LdcWidth::S8, 148, 157, -17260, 23),
    (0xef94_7b7b_c947_9d94, LdcWidth::B32, 148, 157, -17260, 23),
    (0xef95_b3f4_86f7_3aae, LdcWidth::B64, 174, 58, 18543, 31),
];

fn decode(raw: u64) -> Instruction {
    decode_word([raw as u32, (raw >> 32) as u32], 8)
}

#[test]
fn five_ldc_isl_operand_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for (raw, mem_type, dst, offset, immediate, binding) in FORMS {
        let instruction = decode(raw);
        assert_eq!(
            instruction.kind,
            InstructionKind::Ldc(ops::Ldc {
                dst: Register(dst),
                binding,
                offset: Register(offset),
                immediate,
                mem_type,
                address_mode: LdcAddressMode::IndexedSegmentedLocal,
            })
        );
        assert!(instruction.to_string().starts_with("LDC.ISL."));

        let mut module = emit_unoptimized_with_header(
            &[
                instruction,
                Instruction {
                    address: 16,
                    predicate: None,
                    kind: InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
                },
            ],
            Some(&header),
        )
        .unwrap();
        for phase in ["raw", "optimized"] {
            let validation = spirv_validation::validate(&module.words);
            assert!(
                validation.status.success(),
                "{raw:016x} {phase}: {}",
                String::from_utf8_lossy(&validation.stderr)
            );
            module.optimize();
        }
    }
}

#[test]
fn ldc_isl_operand_admission_is_exact() {
    let words = FORMS.map(|(raw, ..)| raw);
    for raw in words {
        for bit in 0..64 {
            let neighbor = raw ^ (1_u64 << bit);
            let same_family = (neighbor >> 48) as u16 & 0xfff8 == 0xef90;
            let still_isl = neighbor >> 44 & 3 == 3;
            if same_family && still_isl && !words.contains(&neighbor) {
                assert!(
                    matches!(decode(neighbor).kind, InstructionKind::Unknown(_)),
                    "unqualified ISL neighbor decoded: {neighbor:016x}"
                );
            }
        }
    }
}
