#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{Instruction, InstructionKind, Register, Source},
};

struct Form {
    name: &'static str,
    raw: u64,
    immediate: u32,
    shift: u8,
    cc: bool,
}

const FORMS: [Form; 6] = [
    Form {
        name: "shift 0 immediate 0x10",
        raw: 0x1400_0000_0107_0201,
        immediate: 0x10,
        shift: 0,
        cc: false,
    },
    Form {
        name: "shift 1 immediate 0x3",
        raw: 0x1420_0000_0037_0201,
        immediate: 0x3,
        shift: 1,
        cc: false,
    },
    Form {
        name: "shift 4 immediate -0x10",
        raw: 0x148f_ffff_ff07_0201,
        immediate: 0xffff_fff0,
        shift: 4,
        cc: false,
    },
    Form {
        name: "shift 31 immediate 0x12345678",
        raw: 0x17e1_2345_6787_0201,
        immediate: 0x1234_5678,
        shift: 31,
        cc: false,
    },
    Form {
        name: "shift 16 immediate -0x80000000",
        raw: 0x1608_0000_0007_0201,
        immediate: 0x8000_0000,
        shift: 16,
        cc: false,
    },
    Form {
        name: "shift 2 immediate 0x7fffffff CC",
        raw: 0x1457_ffff_fff7_0201,
        immediate: 0x7fff_ffff,
        shift: 2,
        cc: true,
    },
];

#[test]
fn six_iscadd32i_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for form in FORMS {
        let name = form.name;
        let raw = form.raw;
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        assert_eq!(
            instruction.kind,
            InstructionKind::ShiftAdd(ops::ShiftAdd {
                dst: Register(1),
                lhs: Register(2),
                rhs: Source::Immediate(form.immediate),
                shift: form.shift,
                negate: [false; 2],
                cc: form.cc,
            }),
            "{name}"
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
            "{name} raw: {}",
            String::from_utf8_lossy(&raw_check.stderr)
        );
        module.optimize();
        let optimized_check = spirv_validation::validate(&module.words);
        assert!(
            optimized_check.status.success(),
            "{name} optimized: {}",
            String::from_utf8_lossy(&optimized_check.stderr)
        );
    }
}
