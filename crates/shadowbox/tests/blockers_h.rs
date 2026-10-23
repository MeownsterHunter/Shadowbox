//! Project-owned probes. Encoding and semantic citations are in
//! docs/reference-sources.md No excluded binaries are read.
#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::{emit_unoptimized_with_header, emit_with_header},
    headers,
    ir::{Instruction, InstructionKind as K, Register, Source, TargetStack},
};

const PT: u64 = 7 << 16;

#[test]
fn newly_supported_families_validate_before_and_after_optimization() {
    // LEA: gm107.c:2022; cbuf LOP3:2169; FSET cbuf:2106.
    for raw in [
        (0x5bd7u64 << 48) | (1 << 45) | (31 << 39) | (3 << 20),
        (0x02f8u64 << 48) | (1 << 47) | (4 << 39) | (31 << 34) | (0x3fff << 20),
        (0x4876u64 << 48) | (2 << 45) | (1 << 42) | (1 << 39),
        // Mesa sm50.rs:861: FSET FTZ, register/cbuf/immediate forms.
        (0x5891u64 << 48) | (7 << 39) | (3 << 20),
        (0x4894u64 << 48) | (7 << 39),
        (0x3091u64 << 48) | (7 << 39) | (0x3f800 << 20),
    ] {
        let mut xs = (1..5)
            .map(|r| Instruction {
                address: r as u32 * 8,
                predicate: None,
                kind: K::Mov(ops::Mov {
                    quad_lanes: 0xf, // Mesa sm50.rs:1936,1941,1946.
                    dst: Register(r),
                    src: Source::Immediate(0x3f800000),
                }),
            })
            .collect::<Vec<_>>();
        xs.push(decode_word(
            [(raw | PT) as u32, ((raw | PT) >> 32) as u32],
            40,
        ));
        assert!(!matches!(
            xs.last().unwrap().kind,
            K::Unknown(ops::Unknown { .. })
        ));
        xs.push(Instruction {
            address: 48,
            predicate: None,
            kind: K::Exit(ops::Exit { cc_test: 0xf }),
        });
        for emit in [emit_unoptimized_with_header, emit_with_header] {
            let module = emit(&xs, Some(&headers::parse(&[0; 32]).unwrap())).unwrap();
            let result = spirv_validation::validate(&module.words);
            assert!(
                result.status.success(),
                "{raw:016x}: {}",
                String::from_utf8_lossy(&result.stderr)
            );
        }
    }
}

#[test]
fn unidentified_prefixes_stay_unknown() {
    // Neither gm107.c:1874-2171 nor Ryujinx InstTable.cs:36-325 has an
    // instruction row for 4202/43d6. They do not match VSET's fe00/4000
    // (gm107.c:2107). The scheduling assembly fallback (:2173-2181) is
    // not evidence for treating them as an instruction-space no-op.
    for prefix in [0x4202u64, 0x43d6] {
        let raw = prefix << 48 | (7 << 39) | PT;
        let x = decode_word([raw as u32, (raw >> 32) as u32], 0);
        assert_eq!(
            x.kind,
            K::Unknown(ops::Unknown {
                raw: [raw as u32, (raw >> 32) as u32]
            })
        );
        assert!(emit_with_header(&[x], None).is_err());
    }
    // BRK remains a real stack pop, never a no-op: Mesa sm50.rs:3055-3063,
    // envytools gm107.c:1921,426-442; Ryujinx InstTable.cs:53.
    let raw = (0xe340u64 << 48) | PT | 0xf;
    assert_eq!(
        decode_word([raw as u32, (raw >> 32) as u32], 0).kind,
        K::PopJump(ops::PopJump {
            cc_test: 0xf,
            stack: TargetStack::Break
        })
    );
    for cc in 0..31u64 {
        if (24..30).contains(&cc) {
            let raw = (0xe340u64 << 48) | PT | cc;
            assert!(matches!(
                decode_word([raw as u32, (raw >> 32) as u32], 0).kind,
                K::Unknown(ops::Unknown { .. })
            ));
        }
    }
    assert_eq!(
        include_str!("../src/decoder/families.rs")
            .matches("x.guarded(InstructionKind::Nop(ops::Nop))")
            .count(),
        2
    );
}
