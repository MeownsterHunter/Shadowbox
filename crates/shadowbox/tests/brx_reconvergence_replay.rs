//! Finite-target reconvergence for Novena strict-cache case 03.

#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{ops, BrxBase, Instruction, InstructionKind, Register},
};

#[derive(Clone, Copy)]
struct Site {
    address: u32,
    raw: u64,
    register: u8,
    loaded: [u32; 3],
    targets: [u32; 3],
}

const SITES: [Site; 3] = [
    Site {
        address: 0x1c90,
        raw: 0xe250_0ffe_3687_3f0f,
        register: 63,
        loaded: [0x1c98, 0x1da0, 0x1ea0],
        targets: [0x1c98, 0x1da8, 0x1ea8],
    },
    Site {
        address: 0x4a18,
        raw: 0xe250_0ffb_5e07_170f,
        register: 23,
        loaded: [0x4a20, 0x4d40, 0x4f50],
        targets: [0x4a28, 0x4d48, 0x4f50],
    },
    Site {
        address: 0x4f78,
        raw: 0xe250_0ffb_0807_160f,
        register: 22,
        loaded: [0x4f80, 0x5070, 0x51b8],
        targets: [0x4f88, 0x5070, 0x51b8],
    },
];

fn immediate(raw: u64) -> u32 {
    ((raw >> 20) & 0xff_ffff) as u32
}

fn normalize_bundle_target(address: u32) -> u32 {
    if address.is_multiple_of(32) {
        address + 8
    } else {
        address
    }
}

fn move_immediate(address: u32, value: u32) -> Instruction {
    let raw = 0x0100_u64 << 48 | u64::from(value) << 20 | 15 << 12 | 7 << 16 | 240;
    decode_word([raw as u32, (raw >> 32) as u32], address)
}

fn exit(address: u32) -> Instruction {
    let raw = 0xe300_u64 << 48 | 7 << 16 | 15;
    decode_word([raw as u32, (raw >> 32) as u32], address)
}

fn resolved_program(
    site: u32,
    raw: u64,
    register: u8,
    loaded: [u32; 3],
    targets: [u32; 3],
) -> Vec<Instruction> {
    let base = BrxBase::decode(site, immediate(raw));
    let mut instructions = vec![Instruction {
        address: site,
        predicate: None,
        kind: InstructionKind::brx(Register(register), base, loaded[0], &loaded[1..]),
    }];
    for (index, target) in targets.into_iter().enumerate() {
        instructions.push(move_immediate(target, index as u32 + 1));
        instructions.push(exit(target + 8));
    }
    instructions.sort_by_key(|instruction| instruction.address);
    instructions
}

#[test]
fn exact_real_sites_decode_to_the_closed_typed_brx_state() {
    for site in SITES {
        assert_eq!(
            decode_word([site.raw as u32, (site.raw >> 32) as u32], site.address,).kind,
            InstructionKind::BrxRegister(ops::BrxRegister {
                displacement: Register(site.register),
                base: BrxBase::decode(site.address, immediate(site.raw)),
                cc_test: 15,
            })
        );
        assert_eq!(
            BrxBase::decode(site.address, immediate(site.raw)).runtime(),
            0
        );
    }
}

#[test]
fn independent_target_oracle_covers_every_hardware_reachable_edge() {
    let mut evaluations = 0;
    for site in SITES {
        for (displacement, target) in site.loaded.into_iter().zip(site.targets) {
            assert_eq!(normalize_bundle_target(displacement), target);
            evaluations += 1;
        }
    }
    assert_eq!(evaluations, 9);
}

#[test]
fn each_resolved_site_validates_as_raw_and_optimized_spirv() {
    let header = headers::parse(&[0; 32]).unwrap();
    let mut raw_count = 0;
    let mut optimized_count = 0;
    for site in SITES {
        let program = resolved_program(
            site.address,
            site.raw,
            site.register,
            site.loaded,
            site.targets,
        );
        let mut module = emit_unoptimized_with_header(&program, Some(&header))
            .unwrap_or_else(|error| panic!("{:#x}: {error}", site.address));
        let validation = spirv_validation::validate(&module.words);
        assert!(
            validation.status.success(),
            "{:#x}/raw: {}",
            site.address,
            String::from_utf8_lossy(&validation.stderr)
        );
        raw_count += 1;
        module.optimize();
        let validation = spirv_validation::validate(&module.words);
        assert!(
            validation.status.success(),
            "{:#x}/optimized: {}",
            site.address,
            String::from_utf8_lossy(&validation.stderr)
        );
        optimized_count += 1;
    }
    assert_eq!((raw_count, optimized_count), (3, 3));
}

#[test]
fn unresolved_real_word_still_fails_closed_for_subgroup_control_flow() {
    let site = SITES[0];
    let vote_raw = 0x50d8_u64 << 48 | 2 << 45 | 1 << 39 | 7 << 16 | 50;
    let exit_raw = 0xe300_u64 << 48 | 7 << 16 | 15;
    let instructions = [
        decode_word([site.raw as u32, (site.raw >> 32) as u32], site.address),
        decode_word([vote_raw as u32, (vote_raw >> 32) as u32], site.address + 8),
        decode_word(
            [exit_raw as u32, (exit_raw >> 32) as u32],
            site.address + 16,
        ),
    ];
    let header = headers::parse(&[0; 32]).unwrap();
    let error = emit_unoptimized_with_header(&instructions, Some(&header))
        .err()
        .unwrap();
    assert!(
        error
            .to_string()
            .contains("indirect branches need structured target regions"),
        "{error}"
    );
}
