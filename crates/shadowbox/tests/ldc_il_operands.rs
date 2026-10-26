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
            0xef90_1000_0037_0201,
            Register(1),
            Register(2),
            3,
            LdcWidth::U8,
        ),
        (
            0xef91_1000_0017_0304,
            Register(4),
            Register(3),
            1,
            LdcWidth::S8,
        ),
        (
            0xef92_1000_0027_0406,
            Register(6),
            Register(4),
            2,
            LdcWidth::U16,
        ),
        (
            0xef93_1000_0067_0508,
            Register(8),
            Register(5),
            6,
            LdcWidth::S16,
        ),
        (
            0xef94_1000_0087_060a,
            Register(10),
            Register(6),
            8,
            LdcWidth::B32,
        ),
    ]
}

#[test]
fn five_ldc_il_operand_forms_decode_and_validate_raw_and_optimized() {
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
