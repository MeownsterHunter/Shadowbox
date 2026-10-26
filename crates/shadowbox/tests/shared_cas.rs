#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::spirv::Op;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{Instruction, InstructionKind, Predicate, Register},
};

fn forms() -> [(u64, Register, Register, i32, Register, Register); 10] {
    [
        (
            0xee40_0000_0047_0201,
            Register(1),
            Register(2),
            0,
            Register(4),
            Register(5),
        ),
        (
            0xee40_0000_4087_0307,
            Register(7),
            Register(3),
            4,
            Register(8),
            Register(9),
        ),
        (
            0xee40_0000_c107_050d,
            Register(13),
            Register(5),
            12,
            Register(16),
            Register(17),
        ),
        (
            0xee40_0001_c187_0715,
            Register(21),
            Register(7),
            28,
            Register(24),
            Register(25),
        ),
        (
            0xee40_0003_c207_091f,
            Register(31),
            Register(9),
            60,
            Register(32),
            Register(33),
        ),
        (
            0xee40_0000_0047_0204,
            Register(4),
            Register(2),
            0,
            Register(4),
            Register(5),
        ),
        (
            0xee40_0000_4087_0303,
            Register(3),
            Register(3),
            4,
            Register(8),
            Register(9),
        ),
        (
            0xee40_0000_c067_060c,
            Register(12),
            Register(6),
            12,
            Register(6),
            Register(7),
        ),
        (
            0xee40_0001_cfd7_0814,
            Register(20),
            Register(8),
            28,
            Register(253),
            Register(254),
        ),
        (
            0xee4f_ffff_c287_0a1e,
            Register(30),
            Register(10),
            -4,
            Register(40),
            Register(41),
        ),
    ]
}

type Guarded = (u64, Option<Predicate>, Register, Register, i32, Register);

/// Guard, RZ-destination and offset-extreme words.
/// Fields: raw, guard, destination, address, byte offset, comparator.
fn guarded_forms() -> [Guarded; 6] {
    let guard = |register, inverted| Some(Predicate { register, inverted });
    [
        (
            0xee40_0000_80a0_0306,
            guard(0, false),
            Register(6),
            Register(3),
            8,
            Register(10),
        ),
        (
            0xee40_0001_00ea_050c,
            guard(2, true),
            Register(12),
            Register(5),
            16,
            Register(14),
        ),
        (
            0xee40_0000_4081_0209,
            guard(1, false),
            Register(9),
            Register(2),
            4,
            Register(8),
        ),
        (
            0xee40_0001_4087_04ff,
            None,
            Register(255),
            Register(4),
            20,
            Register(8),
        ),
        (
            0xee47_ffff_c0c7_0602,
            None,
            Register(2),
            Register(6),
            0x7f_fffc,
            Register(12),
        ),
        (
            0xee48_0000_0107_0703,
            None,
            Register(3),
            Register(7),
            -0x80_0000,
            Register(16),
        ),
    ]
}

fn emit_and_validate(raw: u64, instruction: Instruction, live_in_standalone: bool) {
    let header = headers::parse(&[0; 32]).unwrap();
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
    assert!(opcodes(&module.words).any(|op| op == Op::AtomicCompareExchange));
    let raw_check = spirv_validation::validate(&module.words);
    assert!(
        raw_check.status.success(),
        "{raw:016x} raw: {}",
        String::from_utf8_lossy(&raw_check.stderr)
    );
    module.optimize();
    // A standalone module starts with every predicate false, so the optimizer
    // may remove a CAS behind a true-sense guard. The GPU case observes
    // runtime predicates before optimizing and checks both variants.
    if live_in_standalone {
        assert!(opcodes(&module.words).any(|op| op == Op::AtomicCompareExchange));
    }
    let optimized_check = spirv_validation::validate(&module.words);
    assert!(
        optimized_check.status.success(),
        "{raw:016x} optimized: {}",
        String::from_utf8_lossy(&optimized_check.stderr)
    );
}

#[test]
fn six_guarded_rz_edge_shared_cas_forms_decode_and_validate() {
    for (raw, predicate, dst, address, offset, compare) in guarded_forms() {
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        assert_eq!(instruction.predicate, predicate, "{raw:016x}");
        assert_eq!(
            instruction.kind,
            InstructionKind::SharedCompareExchange(ops::SharedCompareExchange {
                dst,
                address,
                offset,
                compare,
                replacement: Register(compare.0 + 1),
            })
        );
        assert!(instruction.to_string().contains("ATOMS.CAS.B32"));
        emit_and_validate(raw, instruction, predicate.is_none_or(|p| p.inverted));
    }
}

fn opcodes(words: &[u32]) -> impl Iterator<Item = Op> + '_ {
    words[5..].iter().scan(0usize, |at, _| {
        if *at >= words.len() - 5 {
            return None;
        }
        let word = words[5 + *at];
        *at += (word >> 16) as usize;
        Some(Op::of(word))
    })
}

#[test]
fn ten_shared_cas_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for (raw, dst, address, offset, compare, replacement) in forms() {
        let instruction = decode_word([raw as u32, (raw >> 32) as u32], 8);
        assert_eq!(
            instruction.kind,
            InstructionKind::SharedCompareExchange(ops::SharedCompareExchange {
                dst,
                address,
                offset,
                compare,
                replacement,
            })
        );
        assert!(instruction.to_string().starts_with("ATOMS.CAS.B32"));
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
        assert!(opcodes(&module.words).any(|op| op == Op::AtomicCompareExchange));
        let raw_check = spirv_validation::validate(&module.words);
        assert!(
            raw_check.status.success(),
            "{raw:016x} raw: {}",
            String::from_utf8_lossy(&raw_check.stderr)
        );
        module.optimize();
        assert!(opcodes(&module.words).any(|op| op == Op::AtomicCompareExchange));
        let optimized_check = spirv_validation::validate(&module.words);
        assert!(
            optimized_check.status.success(),
            "{raw:016x} optimized: {}",
            String::from_utf8_lossy(&optimized_check.stderr)
        );
    }
}
