#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{ops, DoubleSetResult, FloatFlags, Instruction, InstructionKind},
};

const DSET_CC_WORDS: [u64; 7] = [
    0x5940_bf2f_80e1_2c1a,
    0x5961_cbc8_e1e6_78bd,
    0x3248_b199_07bb_bb0d,
    0x334a_d6b1_3595_21a1,
    0x593f_94c6_cf78_22b3,
    0x3255_cb90_b738_8bba,
    0x3313_b2ae_d9bc_64e7,
];

#[test]
fn seven_exact_dset_cc_words_are_admitted() {
    let header = headers::parse(&[0; 32]).unwrap();
    for (index, raw) in DSET_CC_WORDS.into_iter().enumerate() {
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        let InstructionKind::DoubleSet(ops::DoubleSet { result, .. }) = instruction.kind else {
            panic!("DSET.CC {raw:016x} stayed {:?}", instruction.kind)
        };
        assert_eq!(
            result,
            if index < 4 {
                DoubleSetResult::Mask(FloatFlags::Written)
            } else {
                DoubleSetResult::FloatOne(FloatFlags::Written)
            },
            "{raw:016x}"
        );

        // Admission is exact: a one-bit destination neighbor remains outside
        // the qualified .CC boundary. Clearing CC retains the established
        // broad non-.CC DSET road with the coupled kept state.
        assert!(matches!(
            decode_word([(raw ^ 1) as u32, ((raw ^ 1) >> 32) as u32], 8).kind,
            InstructionKind::Unknown(_)
        ));
        let without_cc = decode_word(
            [(raw & !(1 << 47)) as u32, ((raw & !(1 << 47)) >> 32) as u32],
            8,
        );
        let InstructionKind::DoubleSet(ops::DoubleSet { result, .. }) = without_cc.kind else {
            panic!("non-.CC DSET neighbor was rejected")
        };
        assert_eq!(
            result,
            if index < 4 {
                DoubleSetResult::Mask(FloatFlags::Kept)
            } else {
                DoubleSetResult::FloatOne(FloatFlags::Kept)
            }
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
        for variant in ["raw", "optimized"] {
            let validation = spirv_validation::validate(&module.words);
            assert!(
                validation.status.success(),
                "{raw:016x}/{variant}: {}",
                String::from_utf8_lossy(&validation.stderr)
            );
            module.optimize();
        }
    }
}
