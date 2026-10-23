#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{
        DoubleProductSign, DoubleSource, FloatFlags, Instruction, InstructionKind, Register,
        RegisterPair,
    },
};

const FORMS: [(&str, u64, Register, Register, Register, DoubleProductSign); 5] = [
    (
        "plain zero registers",
        0x5c80_8000_0007_0000,
        Register(0),
        Register(0),
        Register(0),
        DoubleProductSign::Positive,
    ),
    (
        "negated zero registers",
        0x5c81_8000_0007_0000,
        Register(0),
        Register(0),
        Register(0),
        DoubleProductSign::Negated,
    ),
    (
        "plain varied registers a",
        0x5c80_c82a_d4c7_25a4,
        Register(164),
        Register(37),
        Register(76),
        DoubleProductSign::Positive,
    ),
    (
        "negated varied registers",
        0x5c81_c82a_d4c7_25a4,
        Register(164),
        Register(37),
        Register(76),
        DoubleProductSign::Negated,
    ),
    (
        "plain varied registers b",
        0x5c80_dc03_b1f7_0950,
        Register(80),
        Register(9),
        Register(31),
        DoubleProductSign::Positive,
    ),
];

#[test]
fn hybrid_packet_dmul_cc_forms_have_closed_state_and_exact_boundary() {
    let qualified = FORMS.map(|(_, raw, _, _, _, _)| raw);
    let header = headers::parse(&[0; 32]).unwrap();

    for (name, raw, expected_dst, expected_lhs, expected_rhs, expected_sign) in FORMS {
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        let InstructionKind::DoubleMultiply(ops::DoubleMultiply {
            dst,
            lhs,
            rhs,
            sign,
            flags,
        }) = instruction.kind
        else {
            panic!("{name} did not decode as DMUL.CC: {:?}", instruction.kind);
        };
        assert_eq!(dst, RegisterPair(expected_dst), "{name}");
        assert_eq!(lhs, RegisterPair(expected_lhs), "{name}");
        assert_eq!(
            rhs,
            DoubleSource::RegisterPair(RegisterPair(expected_rhs)),
            "{name}"
        );
        assert_eq!(sign, expected_sign, "{name}");
        assert_eq!(flags, FloatFlags::Written, "{name}");

        for bit in 0..64 {
            let changed = raw ^ (1_u64 << bit);
            if bit == 47 || qualified.contains(&changed) {
                continue;
            }
            assert!(
                !matches!(
                    decode_word([changed as u32, (changed >> 32) as u32], 8).kind,
                    InstructionKind::DoubleMultiply(ops::DoubleMultiply {
                        flags: FloatFlags::Written,
                        ..
                    })
                ),
                "{name} one-bit neighbor {changed:016x} escaped the exact boundary"
            );
        }

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
        for variant in ["raw", "optimized"] {
            let check = spirv_validation::validate(&module.words);
            assert!(
                check.status.success(),
                "{name} {variant}: {}",
                String::from_utf8_lossy(&check.stderr)
            );
            module.optimize();
        }
    }
}
