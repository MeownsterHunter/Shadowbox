//! Six more CC-tested SYNC words (LE, GT, GE, NUM, NAN, HI). Fields:
//! envytools envydis/gm107.c:1877,408-411,426-458. Semantics: Ryujinx
//! Instructions/InstEmitFlowControl.cs:232-237,254-262,279-313 and
//! InstEmitConditionCode.cs:52-83.
#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::{decode_stream, decode_word},
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{BranchTarget, InstructionKind, Predicate, TargetStack},
};

const fn guard(register: u8, inverted: bool) -> Option<Predicate> {
    Some(Predicate { register, inverted })
}

const FORMS: [(&str, u64, u8, Option<Predicate>); 6] = [
    (
        "@P1 SYNC CC.LE",
        0xf0f8_0000_0001_0003,
        0x03,
        guard(1, false),
    ),
    ("SYNC CC.GT", 0xf0f8_0000_0007_0004, 0x04, None),
    (
        "@!P0 SYNC CC.GE",
        0xf0f8_0000_0008_0006,
        0x06,
        guard(0, true),
    ),
    ("SYNC CC.NUM", 0xf0f8_0000_0007_0007, 0x07, None),
    (
        "@P2 SYNC CC.NAN",
        0xf0f8_0000_0002_0008,
        0x08,
        guard(2, false),
    ),
    ("SYNC CC.HI", 0xf0f8_0000_0007_0014, 0x14, None),
];

// Three instruction slots after each control word.
fn address(slot: usize) -> u32 {
    (slot / 3 * 32 + slot % 3 * 8 + 8) as u32
}

fn program(sync: u64) -> Vec<u32> {
    // SSY to slot4, SYNC, MOV32I R240 = 1, EXIT, MOV32I R240 = 2, EXIT.
    // SSY gm107.c:1935,297-299; MOV32I gm107.c:2170; EXIT gm107.c:1925.
    let mov = |value: u64| 0x0100_u64 << 48 | value << 20 | 15 << 12 | 7 << 16 | 240;
    let exit = 0xe300_u64 << 48 | 7 << 16 | 15;
    let ssy = 0xe290_u64 << 48 | 0x20 << 20 | 7 << 16 | 15;
    let slots = [ssy, sync, mov(1), exit, mov(2), exit];
    let mut words = Vec::new();
    for bundle in slots.chunks(3) {
        words.extend([0, 0]);
        for &raw in bundle {
            words.extend([raw as u32, (raw >> 32) as u32]);
        }
    }
    words
}

#[test]
fn six_more_sync_cc_forms_decode_resolve_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for (name, raw, cc_test, predicate) in FORMS {
        let word = decode_word([raw as u32, (raw >> 32) as u32], address(1));
        assert_eq!(
            word.kind,
            InstructionKind::PopJump(ops::PopJump {
                stack: TargetStack::Sync,
                cc_test,
            }),
            "{name}"
        );
        assert_eq!(word.predicate, predicate, "{name}");
        let xs = decode_stream(shadowbox::QuietRun, &program(raw));
        let sync = xs.iter().find(|x| x.address == address(1)).unwrap();
        assert_eq!(sync.predicate, predicate, "{name}");
        assert_eq!(
            sync.kind,
            InstructionKind::Reconverge(ops::Reconverge {
                target: BranchTarget::reaching(address(1), address(4)),
                cc_test,
                stack: TargetStack::Sync,
            }),
            "{name}"
        );
        let mut module = emit_unoptimized_with_header(&xs, Some(&header)).unwrap();
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
