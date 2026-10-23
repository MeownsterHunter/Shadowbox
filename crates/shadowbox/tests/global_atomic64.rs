//! Eighteen source-locked 64-bit global ATOM/RED words (SHADOWBOX 0080,
//! 0083, 0086). Provenance: docs/reference-sources.md,
//! docs/reference-sources.md and
//! docs/reference-sources.md
#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::spirv::Op;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{AtomicWidth, Instruction, InstructionKind, Predicate, Register},
};

/// raw, guard, dst, data, address, offset, extended, signed, op, SPIR-V op.
type Form = (
    u64,
    Option<Predicate>,
    Option<Register>,
    Register,
    Register,
    i32,
    bool,
    bool,
    u8,
    Op,
);

fn forms() -> [Form; 18] {
    let p0 = Some(Predicate {
        register: 0,
        inverted: false,
    });
    let p1 = Some(Predicate {
        register: 1,
        inverted: false,
    });
    [
        (
            0xed05_0001_0067_0204,
            None,
            Some(Register(4)),
            Register(6),
            Register(2),
            0x10,
            true,
            false,
            0,
            Op::AtomicIAdd,
        ),
        (
            0xed1b_ffff_80a7_0208,
            None,
            Some(Register(8)),
            Register(10),
            Register(2),
            -8,
            true,
            true,
            1,
            Op::AtomicSMin,
        ),
        (
            0xed25_0001_80e7_020c,
            None,
            Some(Register(12)),
            Register(14),
            Register(2),
            0x18,
            true,
            false,
            2,
            Op::AtomicUMax,
        ),
        (
            0xed65_0000_8060_0206,
            p0,
            Some(Register(6)),
            Register(6),
            Register(2),
            0x8,
            true,
            false,
            6,
            Op::AtomicOr,
        ),
        (
            0xed84_0002_0127_0310,
            None,
            Some(Register(16)),
            Register(18),
            Register(3),
            0x20,
            false,
            false,
            8,
            Op::AtomicExchange,
        ),
        (
            0xebf9_0001_0027_0206,
            None,
            None,
            Register(6),
            Register(2),
            0x10,
            true,
            false,
            0,
            Op::AtomicIAdd,
        ),
        (
            0xed15_0001_0067_0204,
            None,
            Some(Register(4)),
            Register(6),
            Register(2),
            0x10,
            true,
            false,
            1,
            Op::AtomicUMin,
        ),
        (
            0xed2b_ffff_80a7_0208,
            None,
            Some(Register(8)),
            Register(10),
            Register(2),
            -8,
            true,
            true,
            2,
            Op::AtomicSMax,
        ),
        (
            0xed55_0001_80e7_020c,
            None,
            Some(Register(12)),
            Register(14),
            Register(2),
            0x18,
            true,
            false,
            5,
            Op::AtomicAnd,
        ),
        (
            0xed74_0002_0127_0310,
            None,
            Some(Register(16)),
            Register(18),
            Register(3),
            0x20,
            false,
            false,
            7,
            Op::AtomicXor,
        ),
        (
            0xebf9_0001_00d7_0206,
            None,
            None,
            Register(6),
            Register(2),
            0x10,
            true,
            true,
            1,
            Op::AtomicSMin,
        ),
        (
            0xebf9_ffff_83a1_0208,
            p1,
            None,
            Register(8),
            Register(2),
            -8,
            true,
            false,
            7,
            Op::AtomicXor,
        ),
        (
            0xebf9_0001_00a7_0206,
            None,
            None,
            Register(6),
            Register(2),
            0x10,
            true,
            false,
            1,
            Op::AtomicUMin,
        ),
        (
            0xebf9_ffff_8127_020a,
            None,
            None,
            Register(10),
            Register(2),
            -8,
            true,
            false,
            2,
            Op::AtomicUMax,
        ),
        (
            0xebf9_0001_8157_020e,
            None,
            None,
            Register(14),
            Register(2),
            0x18,
            true,
            true,
            2,
            Op::AtomicSMax,
        ),
        (
            0xebf9_0001_02a7_0208,
            None,
            None,
            Register(8),
            Register(2),
            0x10,
            true,
            false,
            5,
            Op::AtomicAnd,
        ),
        (
            0xebf9_0000_8320_0206,
            p0,
            None,
            Register(6),
            Register(2),
            0x8,
            true,
            false,
            6,
            Op::AtomicOr,
        ),
        (
            0xebf8_0002_0027_0312,
            None,
            None,
            Register(18),
            Register(3),
            0x20,
            false,
            false,
            0,
            Op::AtomicIAdd,
        ),
    ]
}

fn instructions(words: &[u32]) -> Vec<(Op, &[u32])> {
    let mut at = 5;
    let mut out = Vec::new();
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        out.push((Op::of(words[at]), &words[at + 1..at + count]));
        at += count;
    }
    out
}

/// The atomic's result type is the 64-bit integer type.
fn has_wide_atomic(words: &[u32], op: Op) -> bool {
    let xs = instructions(words);
    xs.iter().any(|(o, a)| {
        *o == op
            && xs
                .iter()
                .any(|(t, b)| *t == Op::TypeInt && b[0] == a[0] && b[1] == 64)
    })
}

#[test]
fn six_global_atomic64_forms_decode_and_validate() {
    let header = headers::parse(&[0; 32]).unwrap();
    for (raw, predicate, dst, data, address, offset, extended, signed, op, spirv) in forms() {
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        assert_eq!(instruction.predicate, predicate, "{raw:016x}");
        assert_eq!(
            instruction.kind,
            InstructionKind::GlobalAtomic(ops::GlobalAtomic {
                dst,
                data,
                address,
                offset,
                extended,
                signed,
                op,
                width: AtomicWidth::B64,
            }),
            "{raw:016x}"
        );
        let text = instruction.to_string();
        assert!(
            text.contains(if signed { " S64 " } else { " U64 " }),
            "{text}"
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
        assert!(has_wide_atomic(&module.words, spirv), "{raw:016x}");
        let raw_check = spirv_validation::validate(&module.words);
        assert!(
            raw_check.status.success(),
            "{raw:016x} raw: {}",
            String::from_utf8_lossy(&raw_check.stderr)
        );
        module.optimize();
        // A standalone module starts with every predicate false, so the
        // optimizer may remove the guarded @P0/@P1 access.
        if predicate.is_none() {
            assert!(has_wide_atomic(&module.words, spirv), "{raw:016x}");
        }
        let optimized_check = spirv_validation::validate(&module.words);
        assert!(
            optimized_check.status.success(),
            "{raw:016x} optimized: {}",
            String::from_utf8_lossy(&optimized_check.stderr)
        );
    }
}
