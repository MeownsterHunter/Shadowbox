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
    wrap: bool,
    signed: bool,
}

const B4: Source = Source::Register(Register(4));
const C10: Source = Source::ConstantBuffer {
    binding: 0,
    offset: 16,
};

fn forms() -> [Form; 6] {
    [
        Form {
            name: "register U32 clamp",
            raw: 0x5c28_0100_0047_0203,
            rhs: B4,
            wrap: false,
            signed: false,
        },
        Form {
            name: "register S32 .W",
            raw: 0x5c29_0180_0047_0203,
            rhs: B4,
            wrap: true,
            signed: true,
        },
        Form {
            name: "cbuf U32 .W",
            raw: 0x4c28_0180_0047_0203,
            rhs: C10,
            wrap: true,
            signed: false,
        },
        Form {
            name: "cbuf S32 clamp",
            raw: 0x4c29_0100_0047_0203,
            rhs: C10,
            wrap: false,
            signed: true,
        },
        Form {
            name: "immediate U32 7 clamp",
            raw: 0x3828_0100_0077_0203,
            rhs: Source::Immediate(7),
            wrap: false,
            signed: false,
        },
        Form {
            name: "immediate S32 -1 clamp",
            raw: 0x3929_017f_fff7_0203,
            rhs: Source::Immediate(u32::MAX),
            wrap: false,
            signed: true,
        },
    ]
}

#[test]
fn six_shr_brev_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for form in forms() {
        let name = form.name;
        let raw = form.raw;
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        assert_eq!(
            instruction.kind,
            InstructionKind::Shr(ops::Shr {
                dst: Register(3),
                lhs: Register(2),
                rhs: form.rhs,
                wrap: form.wrap,
                signed: form.signed,
                reverse: true,
            }),
            "{name}"
        );
        for predicate in [
            None,
            Some(Predicate {
                register: 0,
                inverted: true,
            }),
        ] {
            let mut module = emit_unoptimized_with_header(
                &[
                    Instruction {
                        predicate,
                        ..instruction
                    },
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
}
