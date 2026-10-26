//! Project-owned blocker words. Field citations are in decoder.rs and
//! docs/reference-sources.md No captured shaders are used.
use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{Instruction, InstructionKind as K, Register, Source},
};

#[path = "support/spirv_validation.rs"]
mod spirv_validation;

const PT: u64 = 7 << 16;
fn decode(raw: u64) -> Instruction {
    decode_word([raw as u32, (raw >> 32) as u32], 0)
}

fn validate(raw: u64) {
    let mut xs = (1..4)
        .map(|r| Instruction {
            address: (r - 1) * 8,
            predicate: None,
            kind: K::Mov(ops::Mov {
                quad_lanes: 15,
                dst: Register(r as u8),
                src: Source::Immediate(0x807f_ff01),
            }),
        })
        .collect::<Vec<_>>();
    xs.push(Instruction {
        address: 24,
        ..decode(raw)
    });
    xs.push(Instruction {
        address: 32,
        predicate: None,
        kind: K::Exit(ops::Exit { cc_test: 15 }),
    });
    let header = headers::parse(&[0; 32]).unwrap();
    let mut module = emit_unoptimized_with_header(&xs, Some(&header)).unwrap();
    for optimized in [false, true] {
        if optimized {
            module.optimize();
        }
        let check = spirv_validation::validate(&module.words);
        assert!(
            check.status.success(),
            "{raw:016x}: {}",
            String::from_utf8_lossy(&check.stderr)
        );
    }
}

#[test]
fn p2r_variants_validate_and_unnamed_fields_remain_unknown() {
    // Registers/predication: envytools gm107.c:203-205,408-411;
    // variants and PR/CC/byte: :1992,2066,2112,1194-1204.
    for prefix in [0x5ce8u64, 0x4ce8, 0x38e8, 0x39e8] {
        for byte in 0..4 {
            for cc in 0..2 {
                let raw = prefix << 48 | byte << 41 | cc << 40 | PT | 2 << 8 | 1;
                assert!(matches!(decode(raw).kind, K::P2R(ops::P2R { .. })));
                validate(raw);
                for bit in (43..51).chain([39]) {
                    assert!(
                        matches!(decode(raw | 1 << bit).kind, K::Unknown(ops::Unknown { .. })),
                        "{raw:016x}, {bit}"
                    );
                }
                if prefix == 0x5ce8 {
                    for bit in 28..39 {
                        assert!(matches!(
                            decode(raw | 1 << bit).kind,
                            K::Unknown(ops::Unknown { .. })
                        ));
                    }
                }
            }
        }
    }
}

#[test]
fn vsetp_variants_validate_and_reserved_selectors_stay_unknown() {
    // VSETP: envytools gm107.c:2052,1588-1628; U32/S32 selector
    // cross-check: Ryujinx InstDecoders.cs:717-732,5261-5262.
    for mode in [0x50f0u64, 0x50f4] {
        for signed in [false, true] {
            for select in 0..7 {
                let raw = mode << 48
                    | u64::from(signed) << 48
                    | u64::from(signed) << 49
                    | select << 36
                    | select << 28
                    | 1 << 43
                    | 7 << 39
                    | 3 << 20
                    | PT
                    | 2 << 8
                    | 1;
                assert!(matches!(
                    decode(raw).kind,
                    K::VideoSetP(ops::VideoSetP { .. })
                ));
                validate(raw);
            }
        }
    }
    let raw = 0x50f4u64 << 48 | 7 << 39 | 3 << 20 | PT | 2 << 8 | 1;
    for modified in [
        raw | 7 << 36,
        raw | 7 << 28,
        raw | 3 << 45,
        raw | 1 << 6,
        raw | 1 << 7,
        raw | 1 << 31,
        raw | 1 << 35,
        raw | 1 << 48 | 6 << 36,
        raw | 1 << 49 | 6 << 28,
    ] {
        assert!(
            matches!(decode(modified).kind, K::Unknown(ops::Unknown { .. })),
            "{modified:016x}"
        );
    }
}

#[test]
fn unsupported_system_registers_and_rro_ex2_are_never_nops() {
    // S2R index20..27: Mesa sm50.rs:3220-3224. Inventory:
    // envytools gm107.c:121-200. CS2R is a separate opcode (:2056).
    // W-scale1e/1f is reference-matched zero in Ryujinx
    // Instructions/InstEmitMove.cs:165-167; other Unknown indices stay so.
    for idx in 0..256u64 {
        let raw = 0xf0c8u64 << 48 | idx << 20 | PT | 1;
        let supported = matches!(idx, 0 | 0x11..=0x13 | 0x1d..=0x23 | 0x25..=0x27 | 0x38..=0x3c);
        assert_eq!(
            matches!(decode(raw).kind, K::S2R(ops::S2R { .. })),
            supported
        );
        assert!(!matches!(decode(raw).kind, K::Nop(ops::Nop)));
    }
    // Exp2 mode39: Mesa sm50.rs:733-738; envytools gm107.c:1322. The
    // cbuf form keeps the mode like the register form (Ryujinx
    // InstEmitMultifunction.cs:25-29,86-94).
    for modifiers in [0, 1 << 45, 1 << 49, 1 << 45 | 1 << 49] {
        let raw = 0x4c90u64 << 48 | 1 << 39 | modifiers | PT | 1;
        assert!(matches!(
            decode(raw).kind,
            K::Rro(ops::Rro { ex2: true, .. })
        ));
    }
    assert_eq!(
        include_str!("../src/decoder/families.rs")
            .matches("x.guarded(InstructionKind::Nop(ops::Nop))")
            .count(),
        2
    );
}
