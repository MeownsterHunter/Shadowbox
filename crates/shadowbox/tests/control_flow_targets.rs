//! Project-owned SM50 programs, with no shader captures.
//! Mesa src/nouveau/compiler/nak/sm50.rs:3006-3028: BRA opcode, signed
//! displacement in bits 20..44, next-instruction base and CC.T in bits 0..5.
//! sm50.rs:225-235: guard predicate 16..19 and inversion 19.
//! sm50.rs:3106-3110,3178-3184: EXIT and NOP opcodes.
//! sm50.rs:3418-3428: 32-byte bundles, control word and three 8-byte slots.

#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_stream,
    interface::{ShaderInput, Stopped},
    ir::InstructionKind,
    structure::analyze,
    translate,
};

const PT: u64 = 7 << 16;
const EXIT: u64 = 0xe300 << 48 | PT | 0xf;
const NOP: u64 = 0x50b0 << 48 | PT;
const UNKNOWN: u64 = 0x1234_5678_dead_beef;

fn bra(address: u32, destination: u32, conditional: bool) -> u64 {
    let offset = i64::from(destination) - i64::from(address) - 8;
    assert!((-0x80_0000..=0x7f_ffff).contains(&offset));
    0xe240 << 48 | ((offset as u64 & 0xff_ffff) << 20) | if conditional { 0xf } else { PT | 0xf }
}

fn words(bundles: &[u64]) -> Vec<u32> {
    bundles
        .iter()
        .flat_map(|raw| [*raw as u32, (*raw >> 32) as u32])
        .collect()
}

fn translation(bundles: &[u64]) -> Result<Vec<u32>, Stopped> {
    let code = bundles
        .iter()
        .flat_map(|raw| raw.to_le_bytes())
        .collect::<Vec<_>>();
    translation_bytes(&code)
}

fn translation_bytes(code: &[u8]) -> Result<Vec<u32>, Stopped> {
    Result::from(translate(
        shadowbox::QuietRun,
        &ShaderInput {
            code,
            program_record: &[],
        },
    ))
    .map(|output| output.spirv)
}

fn validates(bundles: &[u64]) {
    let spirv = translation(bundles).unwrap();
    let output = spirv_validation::validate(&spirv);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn branch_reaches_code_after_first_exit_and_self_branch() {
    // P0 BRA -> 0x48; the other path exits at 0x10. A dead self-branch
    // at 0x28 must not truncate the live target in the third bundle.
    let program = [
        0,
        bra(8, 0x48, true),
        EXIT,
        NOP,
        0,
        bra(0x28, 0x28, false),
        NOP,
        NOP,
        0,
        EXIT,
        NOP,
        NOP,
    ];
    let decoded = decode_stream(shadowbox::QuietRun, &words(&program));
    assert_eq!(decoded.len(), 3);
    assert_eq!(decoded[2].address, 0x48);
    assert_eq!(
        decoded[2].kind,
        InstructionKind::Exit(ops::Exit { cc_test: 15 })
    );
    assert!(decoded.iter().all(|x| x.address != 0x28));
    analyze(&decoded).unwrap();
    validates(&program);

    // Simulate a caller cutting the supplied code at EXIT or self-BRA.
    for end in [3, 6] {
        let error = translation(&program[..end]).unwrap_err();
        assert!(matches!(error, Stopped::MissingTarget(_)));
        assert!(
            error.to_string().contains("targets missing address 0x48"),
            "{error}"
        );
    }
}

#[test]
fn forward_backward_and_bundle_start_targets_keep_real_branches() {
    // Bundle control address 0x20 means slot 0 at 0x28. Slots 1 (0x30)
    // and 2 (0x38) are legal targets too. Ryujinx Decoder.cs:265-270 skips
    // a control word when decoding a target; it does not round slot 1/2.
    for destination in [0x20, 0x28, 0x30, 0x38] {
        let program = [0, bra(8, destination, true), EXIT, NOP, 0, EXIT, EXIT, EXIT];
        let decoded = decode_stream(shadowbox::QuietRun, &words(&program));
        let actual = if destination == 0x20 {
            0x28
        } else {
            destination
        };
        assert_eq!(
            decoded[0].kind,
            InstructionKind::Branch(ops::Branch {
                target: shadowbox::ir::BranchTarget::reaching(decoded[0].address, actual),
                cc_test: 0xf
            })
        );
        validates(&program);
    }
    for destination in [0, 8, 16, 24] {
        let program = [0, NOP, NOP, NOP, 0, bra(0x28, destination, true), EXIT, NOP];
        let decoded = decode_stream(shadowbox::QuietRun, &words(&program));
        assert!(matches!(
            decoded[3].kind,
            InstructionKind::Branch(ops::Branch { .. })
        ));
        validates(&program);
    }
}

#[test]
fn unknown_target_is_present_and_stays_unknown() {
    let program = [0, bra(8, 0x28, true), EXIT, NOP, 0, UNKNOWN, EXIT, NOP];
    let decoded = decode_stream(shadowbox::QuietRun, &words(&program));
    assert_eq!(decoded[2].address, 0x28);
    assert_eq!(
        decoded[2].kind,
        InstructionKind::Unknown(ops::Unknown {
            raw: [UNKNOWN as u32, (UNKNOWN >> 32) as u32]
        })
    );
    analyze(&decoded).unwrap();
    let error = translation(&program).unwrap_err();
    assert!(matches!(error, Stopped::Unknown(_)), "{error}");
    assert!(
        error.to_string().contains("UNKNOWN 0x12345678deadbeef"),
        "{error}"
    );

    // A different encoding at an unvisited word does not change the module.
    let program = [0, EXIT, UNKNOWN, NOP];
    assert_eq!(
        decode_stream(shadowbox::QuietRun, &words(&program)).len(),
        1
    );
    assert_eq!(
        translation(&program).unwrap(),
        translation(&[0, EXIT]).unwrap()
    );
    validates(&program);
}

#[test]
fn truly_invalid_and_incomplete_targets_are_errors() {
    for destination in [4, 12, 28, 36, 44, 0x80] {
        let program = [0, bra(8, destination, true), EXIT, NOP, 0, EXIT, NOP, NOP];
        let error = translation(&program).unwrap_err();
        assert!(matches!(error, Stopped::MissingTarget(_)), "{error}");
        assert!(
            error.to_string().contains("targets missing address"),
            "{error}"
        );
    }
    // A final control word without slot 0 supplies no executable target.
    let program = [0, bra(8, 0x20, true), EXIT, NOP, 0];
    assert!(matches!(
        translation(&program),
        Err(Stopped::MissingTarget(_))
    ));
    // An unreachable branch cannot introduce a target or an error.
    let program = [0, EXIT, bra(16, 0x80, false), NOP];
    validates(&program);
    // Half of the destination instruction must not be silently discarded.
    let program = [0, bra(8, 0x28, true), EXIT, NOP, 0, EXIT];
    let raw = words(&program);
    let code = raw[..raw.len() - 1]
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect::<Vec<_>>();
    assert!(matches!(
        Result::from(translate(
            shadowbox::QuietRun,
            &ShaderInput {
                code: &code,
                program_record: &[]
            }
        )),
        Err(Stopped::PartialWord)
    ));
}

#[test]
fn trailing_half_bundle_after_exit_translates() {
    // The second bundle is a control word plus one slot, only half of a
    // 32-byte bundle. Also exercise every partial-word byte count within it.
    // Ryujinx Decoder.cs:157-162,332-338 stops at the reachable EXIT.
    let program = [0, EXIT, NOP, NOP, 0, NOP];
    let code = program
        .iter()
        .flat_map(|raw| raw.to_le_bytes())
        .collect::<Vec<_>>();
    for end in 33..=48 {
        let spirv = translation_bytes(&code[..end]).unwrap();
        let output = spirv_validation::validate(&spirv);
        assert!(
            output.status.success(),
            "length {end}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn branch_into_partial_word_is_malformed() {
    // Conditional BRA must discover 0x28 even though the fallthrough EXIT
    // terminates earlier. A bundle-control destination also means 0x28.
    // Ryujinx Decoder.cs:150-162,265-270 follows both successors.
    for destination in [0x20, 0x28] {
        let program = [0, bra(8, destination, true), EXIT, NOP, 0, EXIT];
        let code = program
            .iter()
            .flat_map(|raw| raw.to_le_bytes())
            .collect::<Vec<_>>();
        for tail in 1..8 {
            assert!(matches!(
                translation_bytes(&code[..40 + tail]),
                Err(Stopped::PartialWord)
            ));
        }
    }
}

#[test]
fn fallthrough_into_partial_word_is_malformed() {
    for program in [
        // Straight-line fallthrough skips the complete second control word.
        [0, NOP, NOP, NOP, 0, EXIT],
        // A predicated EXIT must retain its non-exiting path.
        [0, EXIT & !PT, NOP, NOP, 0, EXIT],
    ] {
        let code = program
            .iter()
            .flat_map(|raw| raw.to_le_bytes())
            .collect::<Vec<_>>();
        for tail in 1..8 {
            assert!(matches!(
                translation_bytes(&code[..40 + tail]),
                Err(Stopped::PartialWord)
            ));
        }
    }
}

#[test]
fn unconditional_branch_does_not_reach_partial_tail() {
    // A self-loop has no fallthrough. A forward BRA can reach EXIT without
    // executing the later tail. Neither transfer is rewritten or removed.
    for destination in [8, 24] {
        let program = [0, bra(8, destination, false), NOP, EXIT, 0, NOP];
        let code = program
            .iter()
            .flat_map(|raw| raw.to_le_bytes())
            .collect::<Vec<_>>();
        for tail in 1..8 {
            let spirv = translation_bytes(&code[..40 + tail]).unwrap();
            assert!(spirv_validation::validate(&spirv).status.success());
        }
    }
}

#[test]
fn partial_tail_preserves_reachable_unknown_and_invalid_branch_errors() {
    // Both instructions execute from entry. The missing BRA target is beyond
    // the partial tail, so its stop remains MissingTarget.
    for instruction in [UNKNOWN, bra(8, 0x80, false)] {
        let program = [0, instruction, EXIT, NOP, 0, NOP];
        let code = program
            .iter()
            .flat_map(|raw| raw.to_le_bytes())
            .collect::<Vec<_>>();
        let error = translation_bytes(&code[..44]).unwrap_err();
        if instruction == UNKNOWN {
            assert!(matches!(error, Stopped::Unknown(_)));
        } else {
            assert!(matches!(error, Stopped::MissingTarget(_)));
        }
    }
}

// Project-owned constants and unsupported encodings, including every reported
// data prefix. These are arbitrary input bytes, not instruction field claims.
const DATA_WORDS: [u64; 12] = [
    0x3f80_0000_3de1_47ae,
    0x3f7f_ffff_1234_5678,
    0x3f59_999a_8765_4321,
    0x4b09_8000_3f80_0000,
    0x38d1_b717_3f80_0000,
    0,
    UNKNOWN,
    0xffff_ffff_ffff_ffff,
    0x8123_4567_89ab_cdef,
    0x8ace_1357_2468_bdf0,
    // EXIT with keep-reference-count bit 5 stays Unknown (envytools
    // gm107.c:1925); CSM selectors themselves now decode.
    0xe300_0000_0007_0038,
    // envytools gm107.c:1944-1945: unsupported cbuf BRA (bit 5).
    0xe240_0000_0007_002f,
];

#[test]
fn data_after_exit_and_self_branch_is_never_decoded() {
    let baseline = translation(&[0, EXIT]).unwrap();
    for data in DATA_WORDS {
        assert!(
            matches!(
                shadowbox::decoder::decode_word([data as u32, (data >> 32) as u32], 8).kind,
                InstructionKind::Unknown(ops::Unknown { .. })
                    | InstructionKind::Padding(ops::Padding)
            ),
            "test data {data:016x} must be an unsupported reachable word"
        );
        // Try each instruction slot and a control-word position in the tail.
        for padding in 0..8 {
            let mut program = vec![0, EXIT, bra(16, 16, false)];
            program.extend(std::iter::repeat_n(NOP, padding));
            program.extend([data, data, data, data]);
            let decoded = decode_stream(shadowbox::QuietRun, &words(&program));
            assert_eq!(decoded.len(), 1);
            assert_eq!(
                decoded[0].kind,
                InstructionKind::Exit(ops::Exit { cc_test: 15 })
            );
            assert_eq!(translation(&program).unwrap(), baseline);
            validates(&program);
        }
    }
}

#[test]
fn the_same_data_words_reject_when_reachable() {
    for data in DATA_WORDS {
        for conditional_exit in [false, true] {
            let program = if conditional_exit {
                vec![0, EXIT & !PT, data, EXIT]
            } else {
                vec![0, data, EXIT, bra(24, 24, false)]
            };
            assert!(
                matches!(
                    translation(&program),
                    Err(Stopped::Unknown(_) | Stopped::LivePadding(_))
                ),
                "reachable {data:016x} must reject"
            );
        }
        // A branch into the otherwise unreachable tail must expose the data.
        let program = [0, bra(8, 0x28, true), EXIT, bra(24, 24, false), 0, data];
        assert!(matches!(
            translation(&program),
            Err(Stopped::Unknown(_) | Stopped::LivePadding(_))
        ));
    }
}

#[test]
fn branch_skips_an_embedded_data_island() {
    let baseline = [0, bra(8, 0x48, false), NOP, NOP, 0, NOP, NOP, NOP, 0, EXIT];
    let expected = translation(&baseline).unwrap();
    for data in DATA_WORDS {
        let program = [
            0,
            bra(8, 0x40, false),
            data,
            data,
            data,
            data,
            data,
            data,
            data,
            EXIT,
        ];
        let decoded = decode_stream(shadowbox::QuietRun, &words(&program));
        assert_eq!(
            decoded.iter().map(|x| x.address).collect::<Vec<_>>(),
            [8, 0x48]
        );
        assert_eq!(translation(&program).unwrap(), expected);
        validates(&program);
    }
}

#[test]
fn a_reachable_self_branch_keeps_its_loop_and_ignores_tail_data() {
    for data in DATA_WORDS {
        let program = [0, EXIT & !PT, bra(16, 16, false), data, data, data];
        let decoded = decode_stream(shadowbox::QuietRun, &words(&program));
        assert_eq!(decoded.len(), 2);
        assert!(matches!(
            decoded[1].kind,
            InstructionKind::Branch(ops::Branch { target, .. }) if target.offset() == -8
        ));
        validates(&program);
    }
}

#[test]
fn all_push_targets_are_discovered_even_without_a_pop() {
    // Mesa sm50.rs:3038-3040,3072-3074,3095-3097: SSY/PBK/PCNT
    // opcodes and relative displacement at bits 20..44.
    for opcode in [0xe290_u64, 0xe2a0, 0xe2b0] {
        for destination in [0x20_u32, 0x28] {
            let push = opcode << 48 | (u64::from(destination - 16) << 20) | PT | 0xf;
            let program = [0, push, EXIT, UNKNOWN, 0, UNKNOWN];
            let decoded = decode_stream(shadowbox::QuietRun, &words(&program));
            assert_eq!(
                decoded.iter().map(|x| x.address).collect::<Vec<_>>(),
                [8, 16, 0x28]
            );
            assert!(matches!(
                decoded[2].kind,
                InstructionKind::Unknown(ops::Unknown { .. })
            ));
            assert!(translation(&program).is_err());
            let code = program
                .iter()
                .flat_map(|raw| raw.to_le_bytes())
                .collect::<Vec<_>>();
            assert!(matches!(
                translation_bytes(&code[..44]),
                Err(Stopped::PartialWord)
            ));
        }
    }
}

#[test]
fn call_target_and_continuation_are_discovered_but_call_is_unbuilt() {
    // envytools gm107.c:1941,297-299: CAL relative target. Ryujinx
    // Decoder.cs:146-148,157-162: callee discovery and continuation.
    let call = 0xe260_u64 << 48 | 16 << 20;
    let program = [0, call, EXIT, UNKNOWN, 0, EXIT];
    let decoded = decode_stream(shadowbox::QuietRun, &words(&program));
    assert_eq!(
        decoded.iter().map(|x| x.address).collect::<Vec<_>>(),
        [8, 16, 0x28]
    );
    assert!(matches!(
        decoded[0].kind,
        InstructionKind::Call(ops::Call { .. })
    ));
    assert!(matches!(translation(&program), Err(Stopped::Unbuilt(_))));
}

#[test]
fn condition_code_branch_retains_both_paths() {
    // envytools gm107.c:1945,426-442: CC.F (0) is not CC.T (0xf).
    // Ryujinx Decoder.cs:510-515 keeps the fallthrough for any CC test.
    let conditional = bra(8, 0x28, false) & !0x1f;
    let program = [0, conditional, UNKNOWN, UNKNOWN, 0, EXIT];
    let decoded = decode_stream(shadowbox::QuietRun, &words(&program));
    assert_eq!(
        decoded.iter().map(|x| x.address).collect::<Vec<_>>(),
        [8, 16, 0x28]
    );
    assert!(matches!(translation(&program), Err(Stopped::Unknown(_))));
}

#[test]
fn arbitrary_random_tail_words_cannot_change_translation() {
    let expected = translation(&[0, EXIT]).unwrap();
    let mut state = 0x5eed_dada_cafe_beef_u64;
    let mut unsupported = 0;
    for _ in 0..256 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let data = state;
        let program = [0, EXIT, bra(16, 16, false), data, data, data, data, data];
        assert_eq!(
            decode_stream(shadowbox::QuietRun, &words(&program)).len(),
            1
        );
        assert_eq!(translation(&program).unwrap(), expected);
        // Arbitrary bits can also encode a supported real instruction. Only
        // unsupported encodings must reject when moved to an executable slot.
        if matches!(
            shadowbox::decoder::decode_word([data as u32, (data >> 32) as u32], 8).kind,
            InstructionKind::Unknown(ops::Unknown { .. })
        ) {
            unsupported += 1;
            assert!(matches!(
                translation(&[0, data, EXIT]),
                Err(Stopped::Unknown(_))
            ));
        }
    }
    assert!(unsupported > 200);
}

// Instruction indices exclude control words. Mesa sm50.rs:3418-3428 places
// three 8-byte instructions after each 8-byte bundle control word.
fn slot_address(index: usize) -> u32 {
    (index / 3 * 32 + index % 3 * 8 + 8) as u32
}

fn slots(instructions: &[u64]) -> Vec<u64> {
    let mut program = Vec::new();
    for chunk in instructions.chunks(3) {
        program.push(0);
        program.extend_from_slice(chunk);
    }
    program
}

// Mesa sm50.rs:225-235: predicate bits 16..19 and inversion bit 19.
fn guarded(raw: u64, register: u8, inverted: bool) -> u64 {
    (raw & !(0xf << 16)) | (u64::from(register) << 16) | (u64::from(inverted) << 19)
}

// Mesa sm50.rs:3014-3016,3038-3040: SSY opcode, signed 24-bit relative
// displacement at bits 20..44, measured from the next 8-byte instruction.
fn ssy(index: usize, target: usize) -> u64 {
    let offset = i64::from(slot_address(target)) - i64::from(slot_address(index)) - 8;
    0xe290 << 48 | ((offset as u64 & 0xff_ffff) << 20) | PT | 0xf
}

#[test]
fn repeated_unconsumed_ssy_regions_reconverge_by_real_branches() {
    // Project-owned reconstruction of the reported 180-instruction shape.
    // Plain branches reconverge without SYNC; the -0x338 backedge targets
    // a control word before slot 66, then repeats the pushes at 67 and 103.
    let mut instructions = vec![NOP; 180];
    instructions[14] = guarded(EXIT, 0, false);
    for (index, target, predicate) in [
        (20, 63, Some((1, true))),
        (25, 59, Some((1, true))),
        (57, 59, None),
        (63, 103, Some((1, false))),
        (70, 104, Some((1, true))),
        (102, 104, None),
        (104, 138, Some((2, true))),
        (137, 138, None),
        (142, 66, Some((1, true))),
        (179, 179, None),
    ] {
        let mut raw = bra(slot_address(index), slot_address(target), false);
        if index == 137 {
            // BRA +0 addresses a control word; discovery skips it.
            raw = bra(slot_address(index), slot_address(index) + 8, false);
        } else if index == 142 {
            raw = bra(slot_address(index), slot_address(index) + 8 - 0x338, false);
        }
        instructions[index] = if let Some((register, inverted)) = predicate {
            guarded(raw, register, inverted)
        } else {
            raw
        };
    }
    for (index, target) in [(22, 59), (67, 104), (103, 138)] {
        instructions[index] = ssy(index, target);
    }
    for index in [144, 146] {
        instructions[index] = guarded(EXIT, 0, true);
    }
    instructions[178] = EXIT;
    let program = slots(&instructions);
    let decoded = decode_stream(shadowbox::QuietRun, &words(&program));
    assert!(!decoded
        .iter()
        .any(|x| matches!(x.kind, InstructionKind::PopJump(ops::PopJump { .. }))));
    for index in [22, 67, 103] {
        let x = decoded
            .iter()
            .find(|x| x.address == slot_address(index))
            .unwrap();
        assert!(matches!(
            x.kind,
            InstructionKind::ResolvedSchedule(ops::ResolvedSchedule { .. })
        ));
    }
    for index in [20, 25, 57, 63, 70, 102, 104, 137, 142] {
        let x = decoded
            .iter()
            .find(|x| x.address == slot_address(index))
            .unwrap();
        assert!(
            matches!(x.kind, InstructionKind::Branch(ops::Branch { .. })),
            "slot {index}: {x:?}"
        );
    }
    // The final unconditional EXIT makes the trailing self-branch unreachable.
    assert!(!decoded.iter().any(|x| x.address == slot_address(179)));
    analyze(&decoded).unwrap();
    validates(&program);
}

#[test]
fn ssy_loops_with_no_sync_with_sync_inside_and_with_sync_afterward_validate() {
    // Mesa sm50.rs:3050-3051: SYNC opcode and CC.T in bits 0..5.
    const SYNC: u64 = 0xf0f8 << 48 | PT | 0xf;
    for sync_inside in [false, true] {
        for sync_afterward in [false, true] {
            // Loop: SSY -> optional SYNC -> P0 BRA back -> optional SYNC.
            // Without SYNC inside, each iteration pushes another identical
            // static target. A later pop still has a unique destination.
            let instructions = [
                ssy(0, if sync_inside { 2 } else { 4 }),
                if sync_inside { SYNC } else { NOP },
                bra(slot_address(2), slot_address(0), true),
                if sync_afterward { SYNC } else { NOP },
                EXIT,
            ];
            // SYNC after a balanced loop needs its own producer.
            let mut instructions = instructions.to_vec();
            if sync_inside && sync_afterward {
                instructions.insert(3, ssy(3, 5));
                instructions[0] = ssy(0, 2);
            }
            let program = slots(&instructions);
            let decoded = decode_stream(shadowbox::QuietRun, &words(&program));
            assert!(
                decoded.iter().all(|x| !matches!(
                    x.kind,
                    InstructionKind::PopJump(ops::PopJump { .. })
                        | InstructionKind::Schedule(ops::Schedule { .. })
                )),
                "inside={sync_inside}, afterward={sync_afterward}: {decoded:?}"
            );
            assert!(decoded.iter().any(
                |x| matches!(x.kind, InstructionKind::Branch(ops::Branch { target, .. }) if target.offset() < 0)
            ));
            validates(&program);
        }
    }
}

#[test]
fn unconsumed_pushes_of_each_kind_remain_pushes_and_validate() {
    // Mesa sm50.rs:3038-3040,3072-3074,3095-3097: push opcodes and
    // relative displacement. Each unused target must still be discovered.
    for opcode in [0xe290_u64, 0xe2a0, 0xe2b0] {
        let program = [0, opcode << 48 | 8 << 20 | PT | 0xf, EXIT, EXIT];
        let decoded = decode_stream(shadowbox::QuietRun, &words(&program));
        assert!(matches!(
            decoded[0].kind,
            InstructionKind::ResolvedSchedule(ops::ResolvedSchedule { push })
                | InstructionKind::ResolvedPushTarget(ops::ResolvedPushTarget { push, .. })
                if push.target().offset() == 8
        ));
        assert_eq!(decoded[2].address, 24);
        validates(&program);
    }
}

#[test]
fn loop_accumulates_unused_ssy_while_consuming_an_inner_ssy() {
    // Mesa sm50.rs:3038-3040,3050-3051: SSY target and SYNC CC.T.
    // Each iteration leaves an outer entry and consumes a fresh inner one.
    // A full-stack traversal grows forever even though the SYNC destination
    // is always the inner target (Ryujinx Decoder.cs:824-869).
    let program = slots(&[
        ssy(0, 6),
        ssy(1, 3),
        0xf0f8 << 48 | PT | 0xf,
        bra(slot_address(3), slot_address(0), true),
        EXIT,
        NOP,
        EXIT,
    ]);
    let decoded = decode_stream(shadowbox::QuietRun, &words(&program));
    for index in [0, 1] {
        assert!(matches!(
            decoded[index].kind,
            InstructionKind::ResolvedSchedule(ops::ResolvedSchedule { .. })
        ));
    }
    assert_eq!(
        decoded[2].kind,
        InstructionKind::Reconverge(ops::Reconverge {
            target: shadowbox::ir::BranchTarget::reaching(slot_address(2), slot_address(3)),
            cc_test: 0xf,
            stack: shadowbox::ir::TargetStack::Sync,
        })
    );
    assert_eq!(
        decoded[3].kind,
        InstructionKind::Branch(ops::Branch {
            target: shadowbox::ir::BranchTarget::reaching(slot_address(3), slot_address(0)),
            cc_test: 0xf,
        })
    );
    validates(&program);
}
