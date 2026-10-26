//! Six CC-tested SYNC words. Fields: envytools envydis/gm107.c:1877,
//! 408-411,426-458. Semantics: Ryujinx Instructions/InstEmitFlowControl.cs:
//! 232-237,254-262,279-313 and InstEmitConditionCode.cs:52-83.
#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::{decode_stream, decode_word},
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{BranchTarget, InstructionKind, Predicate, TargetStack},
};

const FORMS: [(&str, u64, u8, Option<Predicate>); 6] = [
    ("SYNC CC.LT", 0xf0f8_0000_0007_0001, 0x01, None),
    (
        "@P0 SYNC CC.EQ",
        0xf0f8_0000_0000_0002,
        0x02,
        Some(Predicate {
            register: 0,
            inverted: false,
        }),
    ),
    (
        "@!P2 SYNC CC.NE",
        0xf0f8_0000_000a_0005,
        0x05,
        Some(Predicate {
            register: 2,
            inverted: true,
        }),
    ),
    ("SYNC CC.LO", 0xf0f8_0000_0007_0011, 0x11, None),
    (
        "@P3 SYNC CC.HS",
        0xf0f8_0000_0003_0016,
        0x16,
        Some(Predicate {
            register: 3,
            inverted: false,
        }),
    ),
    (
        "@!P1 SYNC CC.RLE",
        0xf0f8_0000_0009_001e,
        0x1e,
        Some(Predicate {
            register: 1,
            inverted: true,
        }),
    ),
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
fn six_sync_cc_forms_decode_resolve_and_validate_raw_and_optimized() {
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
