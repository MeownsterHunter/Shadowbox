#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{Instruction, InstructionKind, LdcAddressMode, LdcWidth, Register},
};

fn forms() -> [(u64, Register, Register, i32, LdcWidth); 5] {
    [
        (
            0xef90_1000_0047_070c,
            Register(12),
            Register(7),
            4,
            LdcWidth::U8,
        ),
        (
            0xef91_1000_0057_080e,
            Register(14),
            Register(8),
            5,
            LdcWidth::S8,
        ),
        (
            0xef92_1000_0067_0910,
            Register(16),
            Register(9),
            6,
            LdcWidth::U16,
        ),
        (
            0xef93_1000_0077_0a12,
            Register(18),
            Register(10),
            7,
            LdcWidth::S16,
        ),
        (
            0xef95_1000_0087_0b14,
            Register(20),
            Register(11),
            8,
            LdcWidth::B64,
        ),
    ]
}

#[test]
fn next_five_ldc_il_operand_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for (raw, dst, offset, immediate, mem_type) in forms() {
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        assert_eq!(
            instruction.kind,
            InstructionKind::Ldc(ops::Ldc {
                dst,
                binding: 0,
                offset,
                immediate,
                mem_type,
                address_mode: LdcAddressMode::Il,
            })
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
