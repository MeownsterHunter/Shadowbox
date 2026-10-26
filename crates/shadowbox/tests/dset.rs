#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{
        DoubleSetResult, DoubleSource, FloatCompare, FloatFlags, Instruction, InstructionKind,
        Register, RegisterPair, Relation,
    },
};

fn forms() -> [(&'static str, u64); 6] {
    [
        ("register EQ", 0x5902_0380_0047_0201),
        ("cbuf LT", 0x4901_0380_0047_0201),
        ("immediate LE", 0x3203_03bf_f007_0201),
        ("register NUM", 0x5907_0380_0047_0201),
        ("register NAN", 0x5908_0380_0047_0201),
        ("register unordered NE", 0x590d_0380_0047_0201),
    ]
}

#[test]
fn six_dset_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for (name, raw) in forms() {
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        let InstructionKind::DoubleSet(ops::DoubleSet {
            dst,
            lhs,
            rhs,
            compare,
            result,
            ..
        }) = instruction.kind
        else {
            panic!("{name} did not decode as DSET: {:?}", instruction.kind);
        };
        assert_eq!(dst, Register(1), "{name}");
        assert_eq!(lhs, RegisterPair(Register(2)), "{name}");
        assert_eq!(result, DoubleSetResult::Mask(FloatFlags::Kept), "{name}");
        assert_eq!(
            compare,
            match name {
                "register EQ" => FloatCompare::Ordered(Relation::Eq),
                "cbuf LT" => FloatCompare::Ordered(Relation::Lt),
                "immediate LE" => FloatCompare::Ordered(Relation::Le),
                "register NUM" => FloatCompare::Num,
                "register NAN" => FloatCompare::Nan,
                "register unordered NE" => FloatCompare::Unordered(Relation::Ne),
                _ => unreachable!(),
            },
            "{name}"
        );
        match name {
            "cbuf LT" => assert_eq!(
                rhs,
                DoubleSource::ConstantBuffer {
                    binding: 0,
                    offset: 16,
                }
            ),
            "immediate LE" => assert_eq!(rhs, DoubleSource::Immediate(0x3ff0_0000)),
            _ => assert_eq!(rhs, DoubleSource::RegisterPair(RegisterPair(Register(4)))),
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
