#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::{decode_stream, decode_word},
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{BranchTarget, Instruction, InstructionKind, Register, Source},
};

const WORDS: [u64; 16] = [
    0xe290_0000_0000_0020,
    0xe291_0000_0000_0020,
    0xe292_0000_0000_0020,
    0xe293_0000_0000_0020,
    0xe294_0000_0000_0020,
    0xe295_0000_0000_0020,
    0xe296_0000_0000_0020,
    0xe297_0000_0000_0020,
    0xe298_0000_0000_0020,
    0xe299_0000_0000_0020,
    0xe29a_0000_0000_0020,
    0xe29b_0000_0000_0020,
    0xe29c_0000_0000_0020,
    0xe29d_0000_0000_0020,
    0xe29e_0000_0000_0020,
    0xe29f_0000_0000_0020,
];

fn decode(raw: u64) -> Instruction {
    decode_word([raw as u32, (raw >> 32) as u32], 8)
}

fn program(raw: u64) -> Vec<u32> {
    let mov = 0x0100_u64 << 48 | 1 << 20 | 15 << 12 | 7 << 16 | 240;
    let exit = 0xe300_u64 << 48 | 7 << 16 | 15;
    [0, raw, mov, exit]
        .into_iter()
        .flat_map(|word| [word as u32, (word >> 32) as u32])
        .collect()
}

fn cpu_oracle(raw: u64) -> (u32, u32) {
    assert!((0xe290..=0xe29f).contains(&(raw >> 48)));
    assert_eq!(raw & !(0xffff_u64 << 48), 0x20);
    let offset = (((raw >> 20) & 0x00ff_ffff) as u32 as i32) << 8 >> 8;
    let target = 16_i32.wrapping_add(offset) as u32;
    (target, 1)
}

#[test]
fn sixteen_ssy_aliases_decode_execute_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for raw in WORDS {
        let (target, result) = cpu_oracle(raw);
        assert_eq!(target, 16, "{raw:016x}");
        assert_eq!(result, 1, "{raw:016x}");
        assert_eq!(
            decode(raw),
            Instruction {
                address: 8,
                predicate: None,
                kind: InstructionKind::Schedule(ops::Schedule {
                    target: BranchTarget::reaching(8, target),
                }),
            },
            "{raw:016x}"
        );

        let decoded = decode_stream(shadowbox::QuietRun, &program(raw));
        let InstructionKind::ResolvedSchedule(ops::ResolvedSchedule { push }) = decoded[0].kind
        else {
            panic!("{raw:016x} did not preserve typed SSY state");
        };
        assert_eq!(push.target(), BranchTarget::reaching(8, target));
        assert_eq!(
            decoded[1].kind,
            InstructionKind::Mov(ops::Mov {
                quad_lanes: 15,
                dst: Register(240),
                src: Source::Immediate(result),
            }),
            "{raw:016x}"
        );
        let mut module = emit_unoptimized_with_header(&decoded, Some(&header)).unwrap();
        for phase in ["raw", "optimized"] {
            let validation = spirv_validation::validate(&module.words);
            assert!(
                validation.status.success(),
                "{raw:016x} {phase}: {}",
                String::from_utf8_lossy(&validation.stderr)
            );
            module.optimize();
        }
    }
}

#[test]
fn ssy_alias_admission_is_whole_word_exact() {
    for raw in WORDS {
        for bit in 0..64 {
            let neighbor = raw ^ (1_u64 << bit);
            let opcode = neighbor >> 48;
            let canonical = opcode == 0xe290 && neighbor & 0x20 == 0;
            if (0xe290..=0xe29f).contains(&opcode) && !canonical && !WORDS.contains(&neighbor) {
                assert!(
                    matches!(decode(neighbor).kind, InstructionKind::Unknown(_)),
                    "unqualified SSY alias decoded: {neighbor:016x}"
                );
            }
        }
    }
    for raw in [0xe28f_0000_0000_0020, 0xe2a0_0000_0000_0020] {
        assert!(
            !matches!(
                decode(raw).kind,
                InstructionKind::Schedule(ops::Schedule { .. })
            ),
            "adjacent opcode decoded as SSY: {raw:016x}"
        );
    }
}
