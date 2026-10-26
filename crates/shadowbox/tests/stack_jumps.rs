//! Project-owned Maxwell control-flow regressions. Encoding authority:
//! Mesa src/nouveau/compiler/nak/sm50.rs:3032-3085 and envytools
//! envydis/gm107.c:1877,1920-1921,1930-1935,1942-1945.
use shadowbox::ir::ops;
use shadowbox::ir::{NoSlotEntry, TableEntry};
use shadowbox::spirv::Op;
use shadowbox::{
    decoder::{decode_stream, decode_stream_with_constant_buffer},
    emitter::emit,
    ir::{Instruction, InstructionKind, TargetStack},
};

/// A program's identity under `context`, through [`shadowbox::Program`].
fn identity(
    input: &shadowbox::interface::ShaderInput<'_>,
    context: shadowbox::CacheContext,
) -> Result<String, shadowbox::interface::Stopped> {
    Ok(shadowbox::Program::parse(input, context.graphics_state)?
        .identity(context.generation, context.options))
}

#[path = "support/spirv_validation.rs"]
mod spirv_validation;

#[path = "support/stack_execution.rs"]
mod stack_execution;

// Mesa sm50.rs:3418-3428: three instruction slots after each control word.
fn address(slot: usize) -> u32 {
    (slot / 3 * 32 + slot % 3 * 8 + 8) as u32
}

fn op(opcode: u64) -> u64 {
    // Predicate PT, not inverted: Mesa sm50.rs:217-235.
    opcode << 48 | 7 << 16
}

fn target(opcode: u64, from: usize, to: usize) -> u64 {
    // BTARG signed 24-bit relative offset in bits 20..44, PC + 8:
    // envytools gm107.c:297-299; Mesa sm50.rs:3006-3016,3038-3040.
    let offset = address(to).wrapping_sub(address(from) + 8) & 0xff_ffff;
    op(opcode) | u64::from(offset) << 20 | 0xf
}

fn conditional(raw: u64, predicate: u64) -> u64 {
    (raw & !(0xf << 16)) | predicate << 16
}

fn compare() -> u64 {
    // ISETP.LT.U32 P0, PT, R0, R1, PT. Register sources 8..16,20..28,
    // outputs 0..3,3..6, compare 49..52, accumulator 39..42:
    // envytools gm107.c:2028; Mesa sm50.rs:1505-1541.
    op(0x5b62) | 7 | 1 << 20 | 7 << 39
}

fn decode(slots: &[u64], cb1: Option<&[u32]>) -> Vec<Instruction> {
    let mut words = Vec::new();
    for bundle in slots.chunks(3) {
        words.extend([0, 0]);
        for &raw in bundle {
            words.extend([raw as u32, (raw >> 32) as u32]);
        }
    }
    if let Some(cb1) = cb1 {
        decode_stream_with_constant_buffer(shadowbox::QuietRun, &words, cb1)
    } else {
        decode_stream(shadowbox::QuietRun, &words)
    }
}

fn pop_kind(raw: u64) -> TargetStack {
    // Opcode families: Mesa sm50.rs:3050-3051,3061-3062,3084-3085.
    match raw >> 48 {
        0xe340 => TargetStack::Break,
        0xe350 => TargetStack::Continue,
        0xf0f8..=0xf0ff => TargetStack::Sync,
        opcode => panic!("expected a stack pop, got {opcode:x}"),
    }
}

fn check(slots: &[u64], jumps: &[(usize, usize)], cb1: Option<&[u32]>) {
    let xs = decode(slots, cb1);
    for &(from, to) in jumps {
        assert_eq!(
            xs.iter().find(|x| x.address == address(from)).unwrap().kind,
            InstructionKind::Reconverge(ops::Reconverge {
                target: shadowbox::ir::BranchTarget::reaching(address(from), address(to)),
                cc_test: 0xf,
                stack: pop_kind(slots[from]),
            }),
            "stack jump at slot {from}"
        );
    }
    let module = emit(&xs).unwrap();
    let validation = spirv_validation::validate(&module.words);
    assert!(
        validation.status.success(),
        "{}",
        String::from_utf8_lossy(&validation.stderr)
    );
}

#[test]
fn nested_ssy_inside_a_pbk_loop() {
    // Each backedge can leave additional SSY entries. BRK must discard all
    // intervening SSY entries and consume PBK, rather than enumerate stacks.
    // Ryujinx Decoders/Decoder.cs:824-850, Instructions/InstEmitFlowControl.cs:258-262.
    check(
        &[
            target(0xe2a0, 0, 10),
            target(0xe290, 1, 9),
            target(0xe290, 2, 7),
            compare(),
            conditional(target(0xe240, 4, 1), 0),
            conditional(op(0xe340) | 0xf, 1),
            op(0xf0f8) | 0xf,
            op(0xf0f8) | 0xf,
            op(0x50b0),
            op(0xe340) | 0xf,
            op(0xe300) | 0xf,
        ],
        &[(5, 10), (6, 7), (7, 9), (9, 10)],
        None,
    );
}

#[test]
fn sync_join_with_different_stacks_and_the_same_target() {
    // One path pushes once and the other pushes twice. SYNC has the same
    // target on both paths; the next iteration can retain the extra entry.
    check(
        &[
            target(0xe2a0, 0, 10),
            conditional(target(0xe240, 1, 4), 0),
            target(0xe290, 2, 8),
            target(0xe240, 3, 6),
            target(0xe290, 4, 8),
            target(0xe290, 5, 8),
            compare(),
            op(0xf0f8) | 0xf,
            conditional(target(0xe240, 8, 1), 0),
            op(0xe340) | 0xf,
            op(0xe300) | 0xf,
        ],
        &[(7, 8), (9, 10)],
        None,
    );
}

#[test]
fn pcnt_cont_loop() {
    check(
        &[
            target(0xe2a0, 0, 9),
            target(0xe2b0, 1, 6),
            target(0xe290, 2, 5),
            conditional(op(0xe340) | 0xf, 0),
            op(0xe350) | 0xf,
            op(0x50b0),
            compare(),
            conditional(target(0xe240, 7, 1), 0),
            op(0xe340) | 0xf,
            op(0xe300) | 0xf,
        ],
        &[(3, 9), (4, 6), (8, 9)],
        None,
    );
}

#[test]
fn brk_discards_nested_ssy() {
    check(
        &[
            target(0xe2a0, 0, 5),
            target(0xe290, 1, 4),
            target(0xe290, 2, 4),
            op(0xe340) | 0xf,
            op(0x50b0),
            op(0xe300) | 0xf,
        ],
        &[(3, 5)],
        None,
    );
}

#[test]
fn stack_jump_inside_a_brx_target() {
    // Compiler BRX chain: Ryujinx Decoder.cs:527-616. IMNMX immediate
    // bound and PT (min): Mesa sm50.rs:1462-1490; SHL imm2: :1702-1721;
    // LDC binding 36..41, address reg 8..16: :2633-2648;
    // BRX register 8..16, base 20..44: envytools gm107.c:1942-1943.
    let brx_base = address(4) + 8;
    let cb1 = [address(5) - brx_base, address(6) - brx_base];
    check(
        &[
            target(0xe290, 0, 7),
            op(0x3820) | 2 | 1 << 8 | 1 << 20 | 7 << 39,
            op(0x3848) | 2 | 2 << 8 | 2 << 20,
            op(0xef94) | 2 | 2 << 8 | 1 << 36,
            op(0xe250) | 2 << 8 | 0xf,
            op(0xf0f8) | 0xf,
            op(0xf0f8) | 0xf,
            op(0xe300) | 0xf,
        ],
        &[(5, 7), (6, 7)],
        Some(&cb1),
    );
}

fn brx_slots(base: i32, byte_base: i32) -> Vec<u64> {
    // All field encodings are cited in stack_jump_inside_a_brx_target.
    // LDC B32 is type 4 (Mesa sm50.rs:2386-2398,2648).
    vec![
        target(0xe290, 0, 7),
        op(0x3820) | 2 | 1 << 8 | 1 << 20 | 7 << 39,
        op(0x3848) | 2 | 2 << 8 | 2 << 20,
        op(0xef94) | 2 | 2 << 8 | 1 << 36 | u64::from(byte_base as u16) << 20,
        op(0xe250) | 2 << 8 | 0xf | u64::from(base as u32 & 0xff_ffff) << 20,
        op(0xf0f8) | 0xf,
        op(0xf0f8) | 0xf,
        op(0xe300) | 0xf,
    ]
}

#[test]
fn brx_preserves_negative_displacements_byte_bases_and_predication() {
    let base = 32;
    let mut slots = brx_slots(base, 8);
    slots[4] = conditional(slots[4], 0);
    let pc = address(4) + 8 + base as u32;
    let offsets = [address(5).wrapping_sub(pc), address(6).wrapping_sub(pc)];
    let cb1 = [0xfeed, 0xbeef, offsets[0], offsets[1]];
    check(&slots, &[(5, 7), (6, 7)], Some(&cb1));
    let xs = decode(&slots, Some(&cb1));
    assert!(
        matches!(xs[4].kind, InstructionKind::Brx(ops::Brx { base, table, .. })
        if base.offset() == 32 && table.count() == 2 && table.targets().iter().map(|t| t.loaded()).eq([offsets[0], offsets[1]]))
    );
    // A structural semantics assertion as well as spirv-val: the comparison
    // literals must be the raw loaded offsets, never table ordinals 0 and 1.
    let words = shadowbox::emitter::emit_unoptimized_with_header(&xs, None)
        .unwrap()
        .words;
    let mut constants = std::collections::HashMap::new();
    let mut compared = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        let args = &words[at + 1..at + count];
        match Op::of(words[at]) {
            Op::Constant if args.len() == 3 => {
                constants.insert(args[1], args[2]);
            }
            Op::IEqual => {
                compared.push(constants.get(&args[3]).copied());
            }
            _ => {}
        }
        at += count;
    }
    for offset in offsets {
        assert!(
            compared.contains(&Some(offset)),
            "missing BRX offset {offset:x}"
        );
    }
}

#[test]
fn brx_control_word_targets_keep_the_original_selector_value() {
    let slots = brx_slots(0, 0);
    let pc = address(4) + 8;
    let cb1 = [address(5) - pc, address(6) - 8 - pc];
    check(&slots, &[(5, 7), (6, 7)], Some(&cb1));
    assert!(matches!(decode(&slots, Some(&cb1))[4].kind,
        InstructionKind::Brx(ops::Brx { table, .. }) if table.targets()[1].loaded() == cb1[1]));
}

#[test]
fn unproven_brx_tables_remain_general_and_unsupported_fields_stay_unknown() {
    let slots = brx_slots(0, 0);
    let cb1 = [0, address(6) - address(4) - 8];
    let mut variants = Vec::new();
    for (slot, xor) in [
        (1, 1 << 42), // MAX is a lower bound.
        (1, 1 << 48), // Signed minimum does not bound negative inputs.
        (2, 1 << 39), // Unsupported wrap form of this compiler pattern.
        (3, 4 << 48), // Byte load, not a 32-bit table offset.
        (4, 1 << 5),  // Cbuf branch form.
        (4, 1 << 6),  // LMT needs separate semantics.
        (4, 1),       // CC is no longer T.
        (4, 1 << 19), // Inverted PT cannot become an unguarded branch.
    ] {
        let mut variant = slots.clone();
        variant[slot] ^= xor;
        variants.push((variant, slot != 4 || xor == 1 << 19));
    }
    let mut predicated_load = slots.clone();
    predicated_load[3] = conditional(predicated_load[3], 0);
    variants.push((predicated_load, true));
    let mut middle_entry = slots.clone();
    middle_entry[7] = target(0xe240, 7, 2);
    variants.push((middle_entry, true));
    for (variant, general) in variants {
        let xs = decode(&variant, Some(&cb1));
        if let Some(brx) = xs.iter().find(|x| x.address == address(4)) {
            if variant[3] == conditional(slots[3], 0) && general {
                // A skipped LDC retains the SHL result, so the recovered
                // set includes 0/4 as well as the loaded 0/16. Offset 4 is
                // off the word grid: the table names no slot there.
                assert!(
                    matches!(brx.kind, InstructionKind::OutsideBrx(ops::OutsideBrx { table, .. })
                    if table.entries().map(|t| t.loaded()).eq([0, 4, 16])
                        && matches!(table.first_outside(), NoSlotEntry::Misaligned(_)))
                );
                assert!(emit(&xs).is_err());
            } else if general {
                assert!(
                    matches!(
                        brx.kind,
                        InstructionKind::BrxRegister(ops::BrxRegister { .. })
                    ),
                    "{xs:?}"
                );
            } else {
                assert!(matches!(
                    brx.kind,
                    InstructionKind::Unknown(ops::Unknown { .. })
                ));
            }
        } else {
            // Unknown earlier in the chain already blocks execution.
            assert!(xs
                .iter()
                .any(|x| matches!(x.kind, InstructionKind::Unknown(ops::Unknown { .. }))));
        }
        // General BRX now preserves the stack on its possible edges. A
        // unique pop can therefore translate even without a table proof.
        if xs
            .iter()
            .any(|x| matches!(x.kind, InstructionKind::Unknown(ops::Unknown { .. })))
        {
            assert!(emit(&xs).is_err());
        } else if let Ok(module) = emit(&xs) {
            assert!(spirv_validation::validate(&module.words).status.success());
        }
    }
    for invalid_base in [2, -4] {
        assert!(matches!(
            decode(&brx_slots(0, invalid_base), Some(&cb1))[4].kind,
            InstructionKind::BrxRegister(ops::BrxRegister { .. })
        ));
    }
    assert!(matches!(
        decode(&slots, Some(&[]))[4].kind,
        InstructionKind::BrxRegister(ops::BrxRegister { .. })
    ));
    // Ryujinx Decoder.cs:577-616 needs reaching writes. An arbitrary second
    // BRX could enter the middle of the first chain, so neither gets narrowed.
    let mut arbitrary_entry = slots.clone();
    arbitrary_entry.push(op(0xe250) | 3 << 8 | 15);
    // A dead second BRX cannot enter the chain. Conditional EXIT makes it
    // reachable (Ryujinx Decoder.cs:501-515), exercising proof invalidation.
    assert!(matches!(
        decode(&arbitrary_entry, Some(&cb1))[4].kind,
        InstructionKind::Brx(ops::Brx { .. })
    ));
    arbitrary_entry[7] = conditional(op(0xe300) | 15, 0);
    assert!(matches!(
        decode(&arbitrary_entry, Some(&cb1))[4].kind,
        InstructionKind::BrxRegister(ops::BrxRegister { .. })
    ));
}

// Mesa sm50.rs:1264-1286: IADD R0, R0, 1, no .X/.CC or negation.
// Addition: Mesa ir.rs:3802-3821; Ryujinx InstEmitIntegerArithmetic.cs:317-340.
fn add_one() -> u64 {
    op(0x3810) | 1 << 20
}

fn barrier() -> u64 {
    // Mesa sm50.rs:3119-3136: BAR.SYNC, RZ barrier id, PT reduction.
    // RZ = 255: sm50.rs:204-205,255-265.
    // Thread count RZ: envytools gm107.c:1825-1829,1882.
    op(0xf0a8) | 0xff << 8 | 0xff << 20 | 7 << 39
}

fn long_ssy_regions(
    regions: usize,
    barriers: bool,
    inverted: bool,
    before: usize,
) -> (Vec<u64>, Vec<(usize, usize)>) {
    // NOP opcode: Mesa sm50.rs:3178-3189. SSY/SYNC and target fields
    // are cited in the module comment and the target helper above.
    let mut slots = Vec::new();
    let mut jumps = Vec::new();
    for _ in 0..regions {
        slots.push(op(0x50b0));
        if barriers {
            slots.push(barrier());
        }
        let push = slots.len();
        slots.push(0);
        slots.extend(std::iter::repeat_n(add_one(), before));
        let early = slots.len();
        slots.push(conditional(op(0xf0f8) | 0xf, if inverted { 8 } else { 0 }));
        slots.extend(std::iter::repeat_n(add_one(), 40));
        let late = slots.len();
        slots.push(op(0xf0f8) | 0xf);
        let join = slots.len();
        slots[push] = target(0xe290, push, join);
        jumps.extend([(early, join), (late, join)]);
    }
    slots.push(op(0x50b0));
    if barriers {
        slots.push(barrier());
    }
    // EXIT CC.T: Mesa sm50.rs:3101-3111, predicate: :217-235.
    slots.push(conditional(op(0xe300) | 0xf, 0));
    slots.push(add_one());
    slots.push(op(0xe300) | 0xf);
    let end = slots.len();
    slots.push(target(0xe240, end, end));
    (slots, jumps)
}

#[test]
fn long_predicated_sync_compute_regions() {
    // A skipped SYNC keeps SSY for the later unconditional SYNC:
    // Ryujinx Decoder.cs:815-818,824-869, Translator.cs:359-393.
    let header = shadowbox::headers::parse(&[0; 32]).unwrap();
    for (regions, before) in [(1, 100), (3, 100), (36, 100), (1, 115), (3, 115), (36, 115)] {
        for barriers in [false, true] {
            for inverted in [false, true] {
                let (slots, jumps) = long_ssy_regions(regions, barriers, inverted, before);
                let xs = decode(&slots, None);
                if before == 115 {
                    let push = if barriers { 2 } else { 1 };
                    assert_eq!(address(jumps[0].1) - address(push), 0x698);
                }
                assert_eq!(
                    xs.last().unwrap().kind,
                    InstructionKind::Exit(ops::Exit { cc_test: 0xf })
                );
                assert_eq!(
                    xs.iter()
                        .filter(|x| matches!(
                            x.kind,
                            InstructionKind::ResolvedSchedule(ops::ResolvedSchedule { .. })
                        ))
                        .count(),
                    regions
                );
                for (index, (from, to)) in jumps.into_iter().enumerate() {
                    assert_eq!(
                        xs.iter()
                            .find(|x| x.address == address(from))
                            .unwrap()
                            .predicate,
                        (index % 2 == 0).then_some(shadowbox::ir::Predicate {
                            register: 0,
                            inverted
                        })
                    );
                    assert_eq!(
                        xs.iter().find(|x| x.address == address(from)).unwrap().kind,
                        InstructionKind::Reconverge(ops::Reconverge {
                            target: shadowbox::ir::BranchTarget::reaching(
                                address(from),
                                address(to)
                            ),
                            cc_test: 0xf,
                            stack: TargetStack::Sync,
                        }),
                        "regions={regions}, barriers={barriers}, inverted={inverted}, slot={from}"
                    );
                }
                let module = shadowbox::emitter::emit_with_header(&xs, Some(&header)).unwrap();
                let validation = spirv_validation::validate(&module.words);
                assert!(
                    validation.status.success(),
                    "{}",
                    String::from_utf8_lossy(&validation.stderr)
                );
            }
        }
    }
}

#[test]
fn unknown_opcode_reports_the_instruction_that_blocks_stack_resolution() {
    let (mut slots, _) = long_ssy_regions(1, true, true, 100);
    // PEXIT is listed by envytools gm107.c:1946, but its semantics are
    // unimplemented in Ryujinx Instructions/InstEmit.cs:190-195.
    slots[4] = target(0xe230, 4, 6);
    let xs = decode(&slots, None);
    assert!(matches!(
        xs[4].kind,
        InstructionKind::Unknown(ops::Unknown { .. })
    ));
    // Unknown's control effects are unproven. Later pops are not discovered.
    assert!(xs
        .iter()
        .all(|x| !matches!(x.kind, InstructionKind::PopJump(ops::PopJump { .. }))));
    let error = emit(&xs).err().expect("unknown instruction must fail");
    assert!(
        matches!(&error, shadowbox::interface::Stopped::Unknown(x) if **x == xs[4]),
        "{error}"
    );
}

#[test]
fn predicated_sync_executes_the_correct_lane_path_and_exit() {
    let header = shadowbox::headers::parse(&[0; 32]).unwrap();
    for (regions, before) in [(1, 100), (3, 100), (36, 100), (1, 115), (3, 115), (36, 115)] {
        for barriers in [false, true] {
            for inverted in [false, true] {
                for predicate in [false, true] {
                    let (slots, _) = long_ssy_regions(regions, barriers, inverted, before);
                    // Observe the accumulator through shared memory so SSA/DCE
                    // cannot discard a dead Function register. STS.B32 [RZ], R0:
                    // Mesa sm50.rs:2652-2671,2386-2398; RZ:204-205.
                    // Ryujinx InstEmitMemory.cs:156-166,405-429 stores one word.
                    let store = op(0xef5c) | 0xff << 8;
                    let slots: Vec<_> = slots
                        .into_iter()
                        .flat_map(|raw| {
                            if raw >> 48 == 0xe300 {
                                vec![store, raw]
                            } else {
                                vec![raw]
                            }
                        })
                        .collect();
                    let mut xs = decode(&slots, None);
                    // ISETP.LT/EQ.U32 P0, PT, RZ, RZ, PT sets P0 explicitly.
                    // Mesa sm50.rs:1505-1541, int compare codes :810-825,
                    // RZ index :204-205, predicate encoding :217-235.
                    let raw = op(if predicate { 0x5b64 } else { 0x5b62 })
                        | 7
                        | 0xff << 8
                        | 0xff << 20
                        | 7 << 39;
                    xs.insert(
                        0,
                        shadowbox::decoder::decode_word([raw as u32, (raw >> 32) as u32], 0),
                    );
                    let takes_sync = predicate != inverted;
                    let expected = regions as u32
                        * if takes_sync {
                            before as u32
                        } else {
                            before as u32 + 40
                        }
                        + u32::from(!predicate);
                    for words in [
                        shadowbox::emitter::emit_unoptimized_with_header(&xs, Some(&header))
                            .unwrap()
                            .words,
                        shadowbox::emitter::emit_with_header(&xs, Some(&header))
                            .unwrap()
                            .words,
                    ] {
                        let validation = spirv_validation::validate(&words);
                        assert!(
                            validation.status.success(),
                            "{}",
                            String::from_utf8_lossy(&validation.stderr)
                        );
                        assert_eq!(stack_execution::run(&words), (expected, if barriers { regions + 1 } else { 0 }),
                            "regions={regions}, barriers={barriers}, inverted={inverted}, P0={predicate}");
                    }
                }
            }
        }
    }
}

#[path = "support/brk_cases.rs"]
mod brk_cases;

#[test]
fn conditional_brk_keeps_every_condition_and_guard() {
    for cc in brk_cases::CONDITIONS {
        for guard in [0, 8, 7, 15] {
            let xs = decode_stream(shadowbox::QuietRun, &brk_cases::program(cc, guard));
            let jump = xs
                .iter()
                .find(|x| x.address == brk_cases::address(1))
                .unwrap();
            let decoded = shadowbox::decoder::decode_word(
                [
                    brk_cases::brk(cc, guard) as u32,
                    (brk_cases::brk(cc, guard) >> 32) as u32,
                ],
                jump.address,
            );
            assert_eq!(jump.predicate, decoded.predicate);
            // PT is immutable true (Ryujinx OperandHelper.cs:49-51).
            // !PT has no executed pop history, so the local proof has no
            // destination. Keep the pop and reject it instead of inventing
            // a target or dropping the instruction (Decoder.cs:824-869).
            if guard == 15 {
                assert_eq!(jump.kind, decoded.kind);
                assert!(matches!(
                    emit(&xs),
                    Err(shadowbox::interface::Stopped::UnprovenPop(_))
                ));
                continue;
            }
            assert_eq!(
                jump.kind,
                InstructionKind::Reconverge(ops::Reconverge {
                    target: shadowbox::ir::BranchTarget::reaching(
                        brk_cases::address(1),
                        brk_cases::address(4)
                    ),
                    cc_test: cc,
                    stack: TargetStack::Break,
                })
            );
            let module = emit(&xs).unwrap();
            let validation = spirv_validation::validate(&module.words);
            assert!(
                validation.status.success(),
                "{}",
                String::from_utf8_lossy(&validation.stderr)
            );
        }
    }
    for cc in 24..30 {
        let raw = brk_cases::brk(cc, 7);
        assert!(matches!(
            shadowbox::decoder::decode_word([raw as u32, (raw >> 32) as u32], 0).kind,
            InstructionKind::Unknown(ops::Unknown { .. })
        ));
    }
}

#[test]
fn failed_cc_pop_preserves_outer_pbk_across_a_nested_pbk() {
    let slots = [
        brk_cases::push(0, 6),
        brk_cases::brk(2, 7),
        brk_cases::mov(8, 1),
        brk_cases::push(3, 8),
        brk_cases::brk(15, 7),
        brk_cases::exit(),
        brk_cases::mov(8, 2),
        brk_cases::exit(),
        brk_cases::brk(15, 7),
    ];
    let xs = decode_stream(shadowbox::QuietRun, &brk_cases::words(&slots));
    for (from, to, cc_test) in [(1, 6, 2), (4, 8, 15), (8, 6, 15)] {
        assert_eq!(
            xs.iter()
                .find(|x| x.address == brk_cases::address(from))
                .unwrap()
                .kind,
            InstructionKind::Reconverge(ops::Reconverge {
                target: shadowbox::ir::BranchTarget::reaching(
                    brk_cases::address(from),
                    brk_cases::address(to),
                ),
                cc_test,
                stack: TargetStack::Break,
            })
        );
    }
}

// Immediate MOV: Mesa sm50.rs:1939-1941 (quad mask12..16, imm20..52).
fn mov_immediate(register: u64, value: u32) -> u64 {
    op(0x0100) | register | 0xf << 12 | u64::from(value) << 20
}

// These switch programs are generated here. Inputs are cbuf0 words, leaving
// the index unknown to the cbuf1-only target recovery. Each case observes
// both paths of a predicated inner SYNC, then consumes the outer SSY.
fn switch_program(computed: bool) -> (Vec<u64>, Vec<u32>, usize) {
    let mut slots = vec![
        0, // outer SSY
        0, // inner SSY
        // Indexed B32 LDC R1, c0[RZ]: Mesa sm50.rs:2633-2648.
        op(0xef94) | 1 | 0xff << 8,
        mov_immediate(4, 1),
        // IADD R4, R4, 1: Mesa sm50.rs:1264-1286.
        op(0x3810) | 4 | 4 << 8 | 1 << 20,
        // IMNMX.U32 R2, R1, R4, PT: Mesa sm50.rs:1462-1490.
        op(0x5c20) | 2 | 1 << 8 | 4 << 20 | 7 << 39,
        // LDC R6, c0[RZ+4], feeds predicated SYNC.
        op(0xef94) | 6 | 0xff << 8 | 4 << 20,
        // ISETP.EQ P0, PT, R6, RZ, PT: Mesa sm50.rs:1505-1541.
        op(0x5b64) | 7 | 6 << 8 | 0xff << 20 | 7 << 39,
    ];
    // Cross a block boundary and a long interval with no relevant writers.
    // Ryujinx Decoder.cs:652-675 searches both block interiors and predecessors.
    let crossing = slots.len();
    slots.push(target(0xe240, crossing, crossing + 1));
    slots.extend(std::iter::repeat_n(op(0x50b0), 140));
    let table_instruction;
    if computed {
        // SHL R3, R2, 6: Mesa sm50.rs:1702-1721. Six physical
        // instruction slots span two bundles = 64 bytes (:3418-3428).
        slots.push(op(0x3848) | 3 | 2 << 8 | 6 << 20);
        table_instruction = slots.len();
        slots.push(0); // add first case's BRX-relative displacement
    } else {
        slots.push(op(0x3848) | 2 | 2 << 8 | 2 << 20);
        slots.push(mov_immediate(5, 16));
        // IADD R2, R2, R5 computes the table byte base.
        slots.push(op(0x5c10) | 2 | 2 << 8 | 5 << 20);
        table_instruction = slots.len();
        slots.push(op(0xef94) | 3 | 2 << 8 | 1 << 36);
    }
    let brx = slots.len();
    slots.push(op(0xe250) | 3 << 8 | 15);
    let mut cases = Vec::new();
    for value in [10, 20, 30] {
        let case = slots.len();
        cases.push(case);
        slots.extend([
            mov_immediate(0, value),
            conditional(op(0xf0f8) | 15, 0),
            add_one(),
            op(0xf0f8) | 15,
            op(0x50b0),
            op(0x50b0),
        ]);
    }
    let inner_join = slots.len();
    slots.push(op(0x3810) | 100 << 20);
    slots.push(op(0xf0f8) | 15);
    let outer_join = slots.len();
    // STS.B32 [RZ], R0: Mesa sm50.rs:2652-2671,2386-2398.
    slots.push(op(0xef5c) | 0xff << 8);
    slots.push(op(0xe300) | 15);
    slots[0] = target(0xe290, 0, outer_join);
    slots[1] = target(0xe290, 1, inner_join);
    let mut cb1 = vec![0; 4];
    cb1.extend(
        cases
            .iter()
            .map(|&case| address(case).wrapping_sub(address(brx) + 8)),
    );
    if computed {
        slots[table_instruction] = op(0x3810) | 3 | 3 << 8 | u64::from(cb1[4]) << 20;
    }
    (slots, cb1, brx)
}

#[test]
fn bounded_switches_inside_nested_ssy_execute_predicated_sync_exactly() {
    let header = shadowbox::headers::parse(&[0; 32]).unwrap();
    for (computed, lea) in [(false, false), (true, false), (true, true)] {
        let (mut slots, cb1, brx) = switch_program(computed);
        if lea {
            // LEA.I, shift0, carry PT: envytools gm107.c:2140,253,302-309.
            // Same wrapping sum: Ryujinx InstEmitIntegerArithmetic.cs:481-490.
            slots[brx - 1] = (slots[brx - 1] & 0x0000_ffff_ffff_ffff) | 0x36d7_u64 << 48;
        }
        let xs = decode(&slots, (!computed).then_some(cb1.as_slice()));
        assert!(
            matches!(
                xs.iter().find(|x| x.address == address(brx)).unwrap().kind,
                InstructionKind::Brx(ops::Brx { table, .. }) if table.count() == 3
            ),
            "{xs:?}"
        );
        assert!(xs
            .iter()
            .all(|x| !matches!(x.kind, InstructionKind::PopJump(ops::PopJump { .. }))));
        for words in [
            shadowbox::emitter::emit_unoptimized_with_header(&xs, Some(&header))
                .unwrap()
                .words,
            shadowbox::emitter::emit_with_header(&xs, Some(&header))
                .unwrap()
                .words,
        ] {
            let validation = spirv_validation::validate(&words);
            assert!(
                validation.status.success(),
                "{}",
                String::from_utf8_lossy(&validation.stderr)
            );
            for index in [0, 1, 2, 3, u32::MAX] {
                for skip_sync in [0, 1] {
                    let inputs = [index, skip_sync];
                    let expected = (index.min(2) + 1) * 10 + 100 + skip_sync;
                    assert_eq!(
                        stack_execution::run_with_constants(&words, &[(0, &inputs), (1, &cb1)]),
                        (expected, 0),
                        "computed={computed}, lea={lea}, index={index}, skip_sync={skip_sync}"
                    );
                }
            }
        }
    }
}

#[test]
fn translator_receives_cbuf_table_and_cache_identity_tracks_every_word() {
    use shadowbox::{
        interface::{GraphicsState, ShaderInput},
        CacheContext,
    };
    let (slots, cb1, _) = switch_program(false);
    let mut code = Vec::new();
    for bundle in slots.chunks(3) {
        code.extend([0; 8]);
        for raw in bundle {
            code.extend(raw.to_le_bytes());
        }
    }
    let header = [0_u8; 128];
    let input = ShaderInput {
        code: &code,
        program_record: &header,
    };
    let state = GraphicsState {
        constant_buffer1: Some(&cb1),
        ..Default::default()
    };
    let output = Result::from(shadowbox::translate_with_graphics_state(
        shadowbox::QuietRun,
        &input,
        &state,
    ))
    .unwrap();
    assert!(spirv_validation::validate(&output.spirv).status.success());
    assert_eq!(
        stack_execution::run_with_constants(&output.spirv, &[(0, &[1, 1]), (1, &cb1)]),
        (121, 0)
    );
    let original = identity(
        &input,
        CacheContext {
            graphics_state: state,
            ..Default::default()
        },
    )
    .unwrap();
    for index in 0..cb1.len() {
        let mut changed = cb1.clone();
        changed[index] ^= 1;
        let state = GraphicsState {
            constant_buffer1: Some(&changed),
            ..Default::default()
        };
        let key = identity(
            &input,
            CacheContext {
                graphics_state: state,
                ..Default::default()
            },
        )
        .unwrap();
        assert_ne!(key, original);
    }
}

#[test]
fn unresolved_brx_dispatches_to_case_starts_and_interiors_with_sync() {
    // All aligned slots remain cases when cbuf0 supplies an unbounded target.
    // A single SSY target is still provable on every possible BRX edge.
    let slots = [
        target(0xe290, 0, 12),
        op(0xef94) | 3 | 0xff << 8,
        op(0xe250) | 3 << 8 | 15,
        mov_immediate(0, 10),
        conditional(op(0xf0f8) | 15, 0),
        add_one(),
        op(0xf0f8) | 15,
        mov_immediate(0, 20),
        conditional(op(0xf0f8) | 15, 0),
        add_one(),
        op(0xf0f8) | 15,
        op(0x50b0),
        op(0xef5c) | 0xff << 8,
        op(0xe300) | 15,
    ];
    let xs = decode(&slots, None);
    assert!(matches!(
        xs[2].kind,
        InstructionKind::BrxRegister(ops::BrxRegister { .. })
    ));
    assert!(xs
        .iter()
        .all(|x| !matches!(x.kind, InstructionKind::PopJump(ops::PopJump { .. }))));
    let header = shadowbox::headers::parse(&[0; 32]).unwrap();
    for words in [
        shadowbox::emitter::emit_unoptimized_with_header(&xs, Some(&header))
            .unwrap()
            .words,
        shadowbox::emitter::emit_with_header(&xs, Some(&header))
            .unwrap()
            .words,
    ] {
        assert!(spirv_validation::validate(&words).status.success());
        for (to, expected) in [(3, 11), (7, 21), (5, 1), (9, 1)] {
            let displacement = address(to).wrapping_sub(address(2) + 8);
            assert_eq!(
                stack_execution::run_with_constants(&words, &[(0, &[displacement])]),
                (expected, 0)
            );
        }
    }
}

#[test]
fn dispatcher_before_ssy_requires_cb1_or_a_local_proof() {
    // A runtime cbuf0 value in a test is no translation-time proof. BRX
    // can enter SYNC with an empty stack: InstEmitFlowControl.cs:38-40.
    // Keep this shader rejected until an immutable table or local proof exists.
    let slots = [
        op(0xef94) | 3 | 0xff << 8,
        op(0xef94) | 1 | 0xff << 8 | 4 << 20,
        // ISETP.EQ.U32 P0, PT, R1, RZ, PT: Mesa sm50.rs:1505-1541.
        op(0x5b64) | 7 | 1 << 8 | 0xff << 20 | 7 << 39,
        op(0xe250) | 3 << 8 | 15,
        target(0xe290, 4, 10),
        mov_immediate(0, 10),
        conditional(op(0xf0f8) | 15, 0),
        mov_immediate(0, 20),
        op(0xf0f8) | 15,
        op(0x50b0),
        op(0xef5c) | 0xff << 8,
        op(0xe300) | 15,
    ];
    let xs = decode(&slots, None);
    assert!(matches!(
        xs[3].kind,
        InstructionKind::BrxRegister(ops::BrxRegister { .. })
    ));
    assert!(xs
        .iter()
        .any(|x| matches!(x.kind, InstructionKind::PopJump(ops::PopJump { .. }))));
    for error in [
        shadowbox::emitter::emit_unoptimized_with_header(&xs, None)
            .err()
            .unwrap(),
        shadowbox::emitter::emit_with_header(&xs, None)
            .err()
            .unwrap(),
    ] {
        assert!(matches!(
            error,
            shadowbox::interface::Stopped::UnprovenPop(_)
        ));
        assert!(error
            .to_string()
            .contains("missing cb1 data or control-flow proof"));
    }
}

#[test]
fn cb1_proof_and_rejected_snapshot_have_distinct_cache_identity() {
    use shadowbox::{interface::Stopped, Cache, CacheConfig, CacheContext, TargetGeneration};
    // Indexed LDC.B32 R3, c1[RZ]: Mesa sm50.rs:2633-2648,2386-2398;
    // Ryujinx InstEmitMemory.cs:73-95. BRX R3: envytools gm107.c:1943.
    // SSY/SYNC: Mesa sm50.rs:3038-3051. MOV immediate and STS fields:
    // Mesa sm50.rs:1931-1951,2652-2671. All bytes are synthetic.
    let slots = [
        op(0xef94) | 3 | 0xff << 8 | 1 << 36,
        op(0xe250) | 3 << 8 | 15,
        target(0xe290, 2, 5),
        mov_immediate(0, 42),
        op(0xf0f8) | 15,
        op(0xef5c) | 0xff << 8,
        op(0xe300) | 15,
    ];
    let mut legacy = vec![0; 80];
    for bundle in slots.chunks(3) {
        legacy.extend(0_u64.to_le_bytes());
        legacy.extend(bundle.iter().flat_map(|word| word.to_le_bytes()));
    }
    let base = address(1) + 8;
    let valid_cb1 = [address(2) - base];
    let invalid_cb1 = [address(4) - base];
    let record = |words: &[u32]| {
        let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        shadowbox::encode_header_prefixed_with_constant_buffer1(
            &legacy,
            TargetGeneration::Sm5,
            &bytes,
        )
        .unwrap()
    };
    let valid = record(&valid_cb1);
    let invalid = record(&invalid_cb1);
    let incomplete = record(&[]);
    let context = CacheContext::default();
    let keys: Vec<_> = [&legacy, &valid, &invalid, &incomplete]
        .into_iter()
        .map(|record| {
            let input = shadowbox::header_prefixed_input(record, TargetGeneration::Sm5).unwrap();
            identity(&input, context).unwrap()
        })
        .collect();
    for (i, key) in keys.iter().enumerate() {
        assert!(!keys[..i].contains(key));
    }
    let mut cache = Cache::new(CacheConfig {
        memory: shadowbox::cache::Keep::outputs(),
        disk: shadowbox::cache::Disk::nowhere(),
    });
    for record in [&valid, &valid] {
        let output = Result::from(shadowbox::translate_header_prefixed_cached(
            shadowbox::QuietRun,
            record,
            context,
            &mut cache,
        ))
        .unwrap();
        assert!(spirv_validation::validate(&output.spirv).status.success());
        assert_eq!(
            stack_execution::run_with_constants(&output.spirv, &[(1, &valid_cb1)]),
            (42, 0)
        );
    }
    // An invalid or absent snapshot must miss the valid entry and remain
    // rejected on repeated requests. Failure must not populate the cache.
    for record in [&invalid, &incomplete, &legacy, &invalid] {
        let error = Result::from(shadowbox::translate_header_prefixed_cached(
            shadowbox::QuietRun,
            record,
            context,
            &mut cache,
        ))
        .unwrap_err();
        assert!(matches!(error, Stopped::UnprovenPop(_)), "{error}");
        assert!(error
            .to_string()
            .contains("missing cb1 data or control-flow proof"));
    }
    assert_eq!(cache.stats.misses, 5);
    assert_eq!(cache.stats.memory_hits, 1);
}
