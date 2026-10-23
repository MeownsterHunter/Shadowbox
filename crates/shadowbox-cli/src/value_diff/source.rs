// Project-owned must-defined dataflow. Missing Maxwell definitions must stay
// uncheckable even if optimization removes the undefined use. Ryujinx
// Translation/Ssa.cs:250-282; guarded instructions: Translator.cs:359-393.
// BRA/BRX destinations: InstEmitFlowControl.cs:26-42,93-119;
// Decoders/Decoder.cs:265-270,518-688. No upstream code is copied.
use shadowbox::ir::ops;
use shadowbox::ir::{
    AtomicWidth, GlobalAccess, Instruction, InstructionKind as K, LsSize, LsSize2, Source,
};
use std::collections::{BTreeMap, VecDeque};

#[derive(Clone, PartialEq, Eq)]
struct Facts {
    registers: [bool; 256],
    predicates: [bool; 8],
}

fn operands(x: &Instruction) -> Result<(Vec<u8>, Vec<u8>), String> {
    let mut registers = x
        .kind
        .sources()
        .into_iter()
        .filter_map(|s| match s {
            Source::Register(r) => Some(r.0),
            _ => None,
        })
        .collect::<Vec<_>>();
    let mut predicates = x
        .predicate
        .into_iter()
        .map(|p| p.register)
        .collect::<Vec<_>>();
    match x.kind.legacy_form() {
        K::Nop(ops::Nop)
        | K::Exit(ops::Exit { cc_test: 0xf })
        | K::Branch(ops::Branch { cc_test: 0xf, .. })
        // prove() rejects the outside target itself.
        | K::OutsideBranch(ops::OutsideBranch { cc_test: 0xf, .. })
        // Resolved token pops transfer this lane to their proven target.
        // Ryujinx InstEmitFlowControl.cs:239-275; Decoder.cs:824-869.
        | K::Reconverge(ops::Reconverge { cc_test: 0xf, .. })
        | K::ResolvedSchedule(ops::ResolvedSchedule { .. })
        | K::ResolvedPushTarget(ops::ResolvedPushTarget { .. })
        | K::S2R(ops::S2R {
            idx: 0 | 0x20..=0x23 | 0x25..=0x27,
            ..
        })
        | K::Mov(ops::Mov { quad_lanes: 15, .. }) => {}
        K::IAdd(ops::IAdd {
            lhs,
            x: false,
            cc: false,
            ..
        })
        | K::IntegerMul(ops::IntegerMul { lhs, .. })
        | K::Shl(ops::Shl { lhs, .. })
        | K::Shr(ops::Shr { lhs, .. })
        | K::IMinMax(ops::IMinMax { lhs, .. }) => registers.push(lhs.0),
        // Lane ID and collectives: Ryujinx InstEmitMove.cs:78-80,
        // InstEmitWarp.cs:23-67. Predicated collectives need an emitter proof
        // that the operation itself executes under the source guard.
        K::Vote(ops::Vote { pred, .. }) if x.predicate.is_none() => predicates.push(pred.register),
        K::Shuffle(ops::Shuffle { src, .. }) if x.predicate.is_none() => registers.push(src.0),
        K::ISetP(ops::ISetP { lhs, accum, .. }) => {
            registers.push(lhs.0);
            predicates.push(accum);
        }
        // A table holds 1..=32 entries by construction.
        K::Brx(ops::Brx { index, .. }) => registers.push(index.0),
        K::OutsideBrx(ops::OutsideBrx { index, .. }) => registers.push(index.0),
        // Only B32 stores. Workgroup reads would need a separate source memory
        // definedness proof. Addresses and races are checked again in SPIR-V.
        // Encoding: Mesa nak/sm50.rs:2386-2398,2652-2671. Semantics:
        // Ryujinx Instructions/InstEmitMemory.cs:156-167,502-509.
        K::SharedStore(ops::SharedStore {
            data,
            address,
            mem_type: LsSize2::B32,
            ..
        }) => {
            registers.extend([data.0, address.0]);
        }
        // Global address and vector operands: Ryujinx InstEmitMemory.cs:
        // 359-386,457-486,512-532. Encodings: Mesa sm50.rs:2386-2410,2603-2616,2652-2671.
        // Narrow accesses retain their existing rejection until byte proofs.
        K::GlobalMemory(ops::GlobalMemory {
            access:
                access @ (GlobalAccess::Load(LsSize::B32 | LsSize::B64 | LsSize::B128 | LsSize::UB128)
                | GlobalAccess::Store(LsSize2::B32 | LsSize2::B64 | LsSize2::B128)),
            data,
            address,
            extended,
            ..
        }) => {
            registers.push(address.0);
            if extended && address.0 != 255 {
                registers.push(
                    address
                        .0
                        .checked_add(1)
                        .ok_or("global address register overflow")?,
                );
            }
            if let (GlobalAccess::Store(size), false) = (access, data.0 == 255) {
                for i in 0..size.registers() {
                    registers.push(
                        data.0
                            .checked_add(i)
                            .ok_or("global data register overflow")?,
                    );
                }
            }
        }
        // ATOM.Add snapshots the address/data before its destination. Ryujinx
        // InstEmitMemory.cs:12-24; Mesa sm50.rs:2763-2789.
        K::GlobalAtomic(ops::GlobalAtomic {
            address,
            data,
            extended,
            op: 0,
            width: AtomicWidth::B32,
            ..
        }) => {
            registers.extend([address.0, data.0]);
            if extended && address.0 != 255 {
                registers.push(
                    address
                        .0
                        .checked_add(1)
                        .ok_or("global address register overflow")?,
                );
            }
        }
        // ATOMS reads address and data before returning the old word.
        // Ryujinx InstEmitMemory.cs:27-56; Mesa sm50.rs:2807-2834.
        // Predicated ATOMS needs a separate proof that the access is guarded.
        K::Atomic(ops::Atomic { address, data, .. }) if x.predicate.is_none() => {
            registers.extend([address.0, data.0]);
        }
        _ => {
            return Err(format!(
                "source CFG proof unsupported at {:#x}: {x}",
                x.address
            ))
        }
    }
    Ok((registers, predicates))
}

pub(super) fn prove(xs: &[Instruction]) -> Result<(), String> {
    if xs.is_empty() || xs.len() > 4096 {
        return Err("source CFG exceeds 4096 instructions".into());
    }
    let addresses = xs
        .iter()
        .enumerate()
        .map(|(i, x)| (x.address, i))
        .collect::<BTreeMap<_, _>>();
    // Decoded targets already run slot 0 for a bundle control word.
    let target = |slot: u32| -> Result<usize, String> {
        addresses
            .get(&slot)
            .copied()
            .ok_or_else(|| format!("missing source branch target {slot:#x}"))
    };
    let invalid = || "invalid source branch address".to_string();
    let mut edges = Vec::new();
    let mut reads = Vec::new();
    for (i, x) in xs.iter().enumerate() {
        reads.push(operands(x)?);
        let mut successors = Vec::new();
        let falls = match x.kind {
            K::Branch(ops::Branch { target: to, .. })
            | K::Reconverge(ops::Reconverge { target: to, .. }) => {
                successors.push(target(to.slot())?);
                x.predicate.is_some()
            }
            K::OutsideBranch(ops::OutsideBranch { .. }) => return Err(invalid()),
            K::Brx(ops::Brx { table, .. }) => {
                for entry in table.targets() {
                    successors.push(target(entry.slot())?);
                }
                x.predicate.is_some()
            }
            K::OutsideBrx(ops::OutsideBrx { table, .. }) => {
                // A leading missing slot names the error first, as for Brx.
                for entry in table.leading() {
                    target(entry.slot())?;
                }
                return Err(invalid());
            }
            K::Exit(ops::Exit { .. }) => x.predicate.is_some(),
            _ => true,
        };
        if falls && i + 1 < xs.len() {
            successors.push(i + 1);
        }
        edges.push(successors);
    }
    let mut entry = Facts {
        registers: [false; 256],
        predicates: [false; 8],
    };
    entry.registers[255] = true;
    entry.predicates[7] = true;
    let mut states = vec![None; xs.len()];
    states[0] = Some(entry);
    let mut queue = VecDeque::from([0]);
    let mut iterations = 0;
    while let Some(i) = queue.pop_front() {
        iterations += 1;
        if iterations > 1048576 {
            return Err("source CFG exceeds dataflow budget".into());
        }
        let mut after = states[i].as_ref().unwrap().clone();
        if xs[i].predicate.is_none() {
            {
                let shadowbox::ir::Written { base: dst, count } = xs[i].kind.written_registers();
                for r in usize::from(dst.0)..usize::from(dst.0) + usize::from(count) {
                    if r > 255 {
                        return Err("source register write exceeds RZ".into());
                    }
                    after.registers[r] = true;
                }
            }
            let predicates = match xs[i].kind {
                K::ISetP(ops::ISetP { dst, .. }) => dst.map(Some),
                K::Vote(ops::Vote { vote, .. }) => [Some(vote), None],
                K::Shuffle(ops::Shuffle { in_bounds, .. }) => [Some(in_bounds), None],
                _ => [None; 2],
            };
            for predicate in predicates.into_iter().flatten() {
                after.predicates[usize::from(predicate)] = true;
            }
        }
        for &next in &edges[i] {
            let joined = match &states[next] {
                None => after.clone(),
                Some(before) => Facts {
                    registers: std::array::from_fn(|r| before.registers[r] && after.registers[r]),
                    predicates: std::array::from_fn(|p| {
                        before.predicates[p] && after.predicates[p]
                    }),
                },
            };
            if states[next].as_ref() != Some(&joined) {
                states[next] = Some(joined);
                queue.push_back(next);
            }
        }
    }
    let mut exit = false;
    for (i, state) in states.iter().enumerate() {
        if let Some(state) = state {
            if edges[i].is_empty()
                && (!matches!(xs[i].kind, K::Exit(ops::Exit { cc_test: 0xf }))
                    || xs[i].predicate.is_some())
            {
                return Err("source path falls off the program".into());
            }
            for &r in &reads[i].0 {
                if !state.registers[usize::from(r)] {
                    return Err(format!("uninitialized source R{r} at {:#x}", xs[i].address));
                }
            }
            for &p in &reads[i].1 {
                if !state.predicates[usize::from(p)] {
                    return Err(format!("uninitialized source P{p} at {:#x}", xs[i].address));
                }
            }
            exit |= matches!(xs[i].kind, K::Exit(ops::Exit { .. }));
        }
    }
    if !exit {
        return Err("source CFG lacks reachable EXIT".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use shadowbox::ir::{
        BranchTarget, IntCompare, IntegerCompareMode, Predicate, Register, Relation, SetOp,
        Signedness,
    };

    fn address(i: usize) -> u32 {
        (i / 3 * 32 + i % 3 * 8 + 8) as u32
    }
    fn program() -> Vec<Instruction> {
        let kinds = [
            K::S2R(ops::S2R {
                dst: Register(1),
                idx: 0x21,
            }),
            K::Mov(ops::Mov {
                dst: Register(0),
                src: Source::Immediate(1),
                quad_lanes: 15,
            }),
            K::ISetP(ops::ISetP {
                dst: [0, 7],
                lhs: Register(1),
                rhs: Source::Immediate(32),
                compare: IntCompare::Relation(Relation::Lt),
                signedness: Signedness::Unsigned,
                mode: IntegerCompareMode::Ordinary,
                set_op: SetOp::And,
                accum: 7,
                accum_negate: false,
                abs: [false; 2],
                negate: [false; 2],
            }),
            K::Branch(ops::Branch {
                target: BranchTarget::reaching(address(3), address(6)),
                cc_test: 15,
            }),
            K::Mov(ops::Mov {
                dst: Register(2),
                src: Source::Immediate(3),
                quad_lanes: 15,
            }),
            K::Branch(ops::Branch {
                target: BranchTarget::reaching(address(5), address(7)),
                cc_test: 15,
            }),
            K::Mov(ops::Mov {
                dst: Register(2),
                src: Source::Immediate(4),
                quad_lanes: 15,
            }),
            K::IAdd(ops::IAdd {
                dst: Register(0),
                lhs: Register(2),
                rhs: Source::Register(Register(0)),
                x: false,
                cc: false,
                negate: [false; 2],
            }),
            K::Exit(ops::Exit { cc_test: 15 }),
        ];
        kinds
            .into_iter()
            .enumerate()
            .map(|(i, kind)| Instruction {
                address: address(i),
                kind,
                predicate: (i == 3).then_some(Predicate {
                    register: 0,
                    inverted: false,
                }),
            })
            .collect()
    }

    #[test]
    fn source_join_requires_definition_on_both_paths() {
        let mut xs = program();
        prove(&xs).unwrap();
        xs[6].kind = K::Nop(ops::Nop);
        assert!(prove(&xs).unwrap_err().contains("uninitialized source R2"));
    }

    #[test]
    fn guarded_definition_and_undefined_guard_do_not_prove_values() {
        let mut xs = program();
        xs[6].predicate = Some(Predicate {
            register: 0,
            inverted: false,
        });
        assert!(prove(&xs).unwrap_err().contains("uninitialized source R2"));
        let mut xs = program();
        xs[2].kind = K::Nop(ops::Nop);
        assert!(prove(&xs).unwrap_err().contains("uninitialized source P0"));
    }

    #[test]
    fn token_arrivals_require_definitions_on_both_edges() {
        let mut xs = program();
        let K::Branch(ops::Branch { target, cc_test }) = xs[3].kind else {
            unreachable!()
        };
        xs[3].kind = K::Reconverge(ops::Reconverge {
            target,
            cc_test,
            stack: shadowbox::ir::TargetStack::Sync,
        });
        prove(&xs).unwrap();
        xs[6].predicate = Some(Predicate {
            register: 0,
            inverted: false,
        });
        assert!(prove(&xs).unwrap_err().contains("uninitialized source R2"));
    }

    #[test]
    fn arbitrary_brx_stays_uncheckable() {
        let mut xs = program();
        xs[3].kind = K::BrxRegister(ops::BrxRegister {
            displacement: Register(0),
            base: shadowbox::ir::BrxBase::decode(address(3), 0),
            cc_test: 0xf,
        });
        assert!(prove(&xs)
            .unwrap_err()
            .contains("source CFG proof unsupported"));
    }
}
