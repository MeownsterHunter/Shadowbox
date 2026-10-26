#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{Instruction, InstructionKind, Predicate, Register, Source},
};

struct Form {
    name: &'static str,
    raw: u64,
    rhs: Source,
    min_when: Predicate,
    signed: bool,
}

const B4: Source = Source::Register(Register(4));

const fn p(register: u8, inverted: bool) -> Predicate {
    Predicate { register, inverted }
}

fn forms() -> [Form; 6] {
    [
        Form {
            name: "register unsigned P0",
            raw: 0x5c20_0000_0047_0201,
            rhs: B4,
            min_when: p(0, false),
            signed: false,
        },
        Form {
            name: "register signed !P1",
            raw: 0x5c21_0480_0047_0201,
            rhs: B4,
            min_when: p(1, true),
            signed: true,
        },
        Form {
            name: "register signed P3",
            raw: 0x5c21_0180_0047_0201,
            rhs: B4,
            min_when: p(3, false),
            signed: true,
        },
        Form {
            name: "register unsigned !P6",
            raw: 0x5c20_0700_0047_0201,
            rhs: B4,
            min_when: p(6, true),
            signed: false,
        },
        Form {
            name: "cbuf unsigned !P0",
            raw: 0x4c20_0400_0047_0201,
            rhs: Source::ConstantBuffer {
                binding: 0,
                offset: 16,
            },
            min_when: p(0, true),
            signed: false,
        },
        Form {
            name: "negative immediate signed P4",
            raw: 0x3921_0200_0047_0201,
            rhs: Source::Immediate(0xfff8_0004),
            min_when: p(4, false),
            signed: true,
        },
    ]
}

#[test]
fn six_imnmx_select_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for form in forms() {
        let name = form.name;
        let raw = form.raw;
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        assert_eq!(
            instruction.kind,
            InstructionKind::IMinMaxSelect(ops::IMinMaxSelect {
                dst: Register(1),
                lhs: Register(2),
                rhs: form.rhs,
                min_when: form.min_when,
                signed: form.signed,
                cc: false,
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
