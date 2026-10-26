#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{
        DoubleSource, FloatCompare, Instruction, InstructionKind, Register, RegisterPair, Relation,
    },
};

fn forms() -> [(&'static str, u64); 5] {
    let common = 7_u64 << 39 | 7 << 16 | 2 << 8 | 4 << 20 | 1;
    [
        ("register EQ", 0x5b80_u64 << 48 | 2 << 48 | common),
        ("cbuf LT", 0x4b80_u64 << 48 | 1 << 48 | common),
        (
            "immediate LE",
            0x3680_u64 << 48 | 3 << 48 | 0x3ff00 << 20 | common & !(0xff << 20),
        ),
        (
            "register EQ lhs ABS",
            0x5b80_u64 << 48 | 2 << 48 | common | 1 << 7,
        ),
        (
            "register EQ rhs NEG",
            0x5b80_u64 << 48 | 2 << 48 | common | 1 << 6,
        ),
    ]
}

#[test]
fn five_dsetp_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for (name, raw) in forms() {
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        let InstructionKind::DoubleSetP(ops::DoubleSetP {
            dst,
            lhs,
            rhs,
            compare,
            abs,
            negate,
            ..
        }) = instruction.kind
        else {
            panic!("{name} did not decode as DSETP: {:?}", instruction.kind);
        };
        assert_eq!(dst, [0, 1], "{name}");
        assert_eq!(lhs, RegisterPair(Register(2)), "{name}");
        assert_eq!(
            compare,
            if name.contains("LT") {
                FloatCompare::Ordered(Relation::Lt)
            } else if name.contains("LE") {
                FloatCompare::Ordered(Relation::Le)
            } else {
                FloatCompare::Ordered(Relation::Eq)
            },
            "{name}"
        );
        assert_eq!(abs, [name.contains("ABS"), false], "{name}");
        assert_eq!(negate, [false, name.contains("NEG")], "{name}");
        match name {
            "register EQ" | "register EQ lhs ABS" | "register EQ rhs NEG" => {
                assert_eq!(rhs, DoubleSource::RegisterPair(RegisterPair(Register(4))))
            }
            "cbuf LT" => assert_eq!(
                rhs,
                DoubleSource::ConstantBuffer {
                    binding: 0,
                    offset: 16
                }
            ),
            "immediate LE" => assert_eq!(rhs, DoubleSource::Immediate(0x3ff0_0000)),
            _ => unreachable!(),
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
