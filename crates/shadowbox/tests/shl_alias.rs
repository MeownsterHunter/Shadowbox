#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{Instruction, InstructionKind, Register, Source},
};

#[derive(Clone, Copy)]
struct Form {
    name: &'static str,
    raw: u64,
    dst: u8,
    lhs: u8,
    rhs: Source,
    wrap: bool,
}

const FORMS: [Form; 6] = [
    Form {
        name: "register clamp",
        raw: 0x5c4a_0733_d3f7_4209,
        dst: 9,
        lhs: 66,
        rhs: Source::Register(Register(63)),
        wrap: false,
    },
    Form {
        name: "register wrap",
        raw: 0x5c4a_07b3_d3f7_4209,
        dst: 9,
        lhs: 66,
        rhs: Source::Register(Register(63)),
        wrap: true,
    },
    Form {
        name: "constant-buffer clamp",
        raw: 0x4c4a_037a_0367_2672,
        dst: 114,
        lhs: 38,
        rhs: Source::ConstantBuffer {
            binding: 30,
            offset: 32984,
        },
        wrap: false,
    },
    Form {
        name: "constant-buffer wrap",
        raw: 0x4c4a_03fa_0367_2672,
        dst: 114,
        lhs: 38,
        rhs: Source::ConstantBuffer {
            binding: 30,
            offset: 32984,
        },
        wrap: true,
    },
    Form {
        name: "immediate clamp",
        raw: 0x3849_4406_e117_07a6,
        dst: 166,
        lhs: 7,
        rhs: Source::Immediate(0x6e11),
        wrap: false,
    },
    Form {
        name: "immediate wrap",
        raw: 0x3849_4486_e117_07a6,
        dst: 166,
        lhs: 7,
        rhs: Source::Immediate(0x6e11),
        wrap: true,
    },
];

fn decode(raw: u64) -> Instruction {
    decode_word([raw as u32, (raw >> 32) as u32], 8)
}

#[test]
fn six_hybrid_shl_aliases_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for form in FORMS {
        let instruction = decode(form.raw);
        assert_eq!(
            instruction.kind,
            InstructionKind::Shl(ops::Shl {
                dst: Register(form.dst),
                lhs: Register(form.lhs),
                rhs: form.rhs,
                wrap: form.wrap,
            }),
            "{}",
            form.name
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
                "{} {phase}: {}",
                form.name,
                String::from_utf8_lossy(&validation.stderr)
            );
            module.optimize();
        }
    }
}

#[test]
fn hybrid_shl_alias_admission_is_exact() {
    let words = FORMS.map(|form| form.raw);
    for raw in words {
        for bit in 0..64 {
            let neighbor = raw ^ (1_u64 << bit);
            let opcode = (neighbor >> 48) as u16;
            let same_family =
                matches!(opcode & 0xfff8, 0x5c48 | 0x4c48) || opcode & 0xfef8 == 0x3848;
            let noncanonical_alias = neighbor >> 48 & 7 != 0;
            if same_family && noncanonical_alias && !words.contains(&neighbor) {
                assert!(
                    matches!(decode(neighbor).kind, InstructionKind::Unknown(_)),
                    "unqualified SHL alias decoded: {neighbor:016x}"
                );
            }
        }
    }
}
