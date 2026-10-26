#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{Instruction, InstructionKind, LdcAddressMode, LdcWidth, Register},
};

fn forms() -> [(u64, LdcWidth); 6] {
    [
        (0xef90_1000_0007_0000, LdcWidth::U8),
        (0xef91_1000_0007_0000, LdcWidth::S8),
        (0xef92_1000_0007_0000, LdcWidth::U16),
        (0xef93_1000_0007_0000, LdcWidth::S16),
        (0xef94_1000_0007_0000, LdcWidth::B32),
        (0xef95_1000_0007_0000, LdcWidth::B64),
    ]
}

#[test]
fn six_ldc_il_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for (raw, expected_size) in forms() {
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        assert!(matches!(
            instruction.kind,
            InstructionKind::Ldc(ops::Ldc {
                dst: Register(0),
                binding: 0,
                offset: Register(0),
                immediate: 0,
                mem_type,
                address_mode: LdcAddressMode::Il,
            }) if mem_type == expected_size
        ));
        assert!(instruction.to_string().starts_with("LDC.IL."));

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
        let raw_check = spirv_validation::validate(&module.words);
        assert!(
            raw_check.status.success(),
            "{raw:016x} raw: {}",
            String::from_utf8_lossy(&raw_check.stderr)
        );
        module.optimize();
        let optimized_check = spirv_validation::validate(&module.words);
        assert!(
            optimized_check.status.success(),
            "{raw:016x} optimized: {}",
            String::from_utf8_lossy(&optimized_check.stderr)
        );
    }
}
