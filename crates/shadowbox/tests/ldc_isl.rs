#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{Instruction, InstructionKind, LdcAddressMode, LdcWidth, Register},
};

const FORMS: [(u64, LdcWidth); 6] = [
    (0xef90_3000_0007_0000, LdcWidth::U8),
    (0xef91_3000_0007_0000, LdcWidth::S8),
    (0xef92_3000_0007_0000, LdcWidth::U16),
    (0xef93_3000_0007_0000, LdcWidth::S16),
    (0xef94_3000_0007_0000, LdcWidth::B32),
    (0xef95_3000_0007_0000, LdcWidth::B64),
];

fn decode(raw: u64) -> Instruction {
    decode_word([raw as u32, (raw >> 32) as u32], 8)
}

#[test]
fn six_ldc_isl_states_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for (raw, mem_type) in FORMS {
        let instruction = decode(raw);
        assert_eq!(
            instruction.kind,
            InstructionKind::Ldc(ops::Ldc {
                dst: Register(0),
                binding: 0,
                offset: Register(0),
                immediate: 0,
                mem_type,
                address_mode: LdcAddressMode::IndexedSegmentedLocal,
            })
        );
        assert!(instruction.to_string().starts_with("LDC.ISL."));
        assert_eq!(
            mem_type.registers(),
            if mem_type == LdcWidth::B64 { 2 } else { 1 }
        );

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
fn ldc_isl_admission_is_exact() {
    let words = FORMS.map(|(raw, _)| raw);
    for raw in words {
        for bit in 0..64 {
            let neighbor = raw ^ (1_u64 << bit);
            let same_family = (neighbor >> 48) as u16 & 0xfff8 == 0xef90;
            let still_isl = neighbor >> 44 & 3 == 3;
            if same_family && still_isl && !words.contains(&neighbor) {
                assert!(
                    matches!(decode(neighbor).kind, InstructionKind::Unknown(_)),
                    "unlisted ISL neighbor decoded: {neighbor:016x}"
                );
            }
        }
    }
}
