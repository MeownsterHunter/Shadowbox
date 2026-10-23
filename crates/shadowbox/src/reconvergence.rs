//! Token regions around the PC dispatcher. Early arrivals wait at the region
//! loop merge before executing the target instruction.
//! SPIR-V numeric fields: Mesa src/compiler/spirv/spirv.h:1791-1792
//! (load/store),1892,1899,1901 (logical/range tests),1961-1966 (flow).
//! Independent implementation of Ryujinx's token-target and structured-merge
//! semantics, not a port of its goto elimination implementation.
use crate::emitter::inst;
use crate::fast_hash::HashMap;
use crate::interface::{Stopped, Unreconverged};
use crate::ir::ops;
use crate::ir::{Instruction, InstructionKind, TargetStack};
use crate::spirv::Op;
use crate::structure::ControlFlowGraph;

struct Region {
    push: u32,
    start: u32,
    end: u32,
    stack: TargetStack,
}

#[derive(Default)]
pub(crate) struct Regions {
    tokens: Vec<Region>,
    ordered: bool,
}

impl Regions {
    pub fn analyze(xs: &[Instruction], cfg: &ControlFlowGraph) -> Result<Self, Stopped> {
        let error = Stopped::Reconvergence;
        if let Some(pair) = xs
            .windows(2)
            .find(|pair| pair[0].address >= pair[1].address)
        {
            return Err(error(pair[1].address, Unreconverged::Unordered));
        }
        let mut regions = Vec::<Region>::new();
        for (index, x) in xs.iter().enumerate() {
            let (stack, target) = match x.kind {
                InstructionKind::ResolvedSchedule(ops::ResolvedSchedule { push }) => {
                    (TargetStack::Sync, push.target())
                }
                InstructionKind::ResolvedPushTarget(ops::ResolvedPushTarget { push, stack }) => {
                    (stack, push.target())
                }
                _ => continue,
            };
            // Target fields: Mesa nak/sm50.rs:3038-3040,3072-3074,3095-3097.
            // Ryujinx InstEmitFlowControl.cs:239-262 associates each consumer
            // with a push target. Unused pushes have no consumer tag writes.
            // An outside push is never resolved, so its target is a slot.
            let end = target.slot();
            if !xs.iter().any(|consumer| {
                matches!(consumer.kind,
                InstructionKind::Reconverge(ops::Reconverge { target, stack: kind, .. })
                    if stack == kind && consumer.address > x.address && consumer.address < end && target.slot() == end)
            }) {
                continue;
            }
            let Some(next) = xs.get(index + 1) else {
                return Err(error(x.address, Unreconverged::EmptyPush));
            };
            if x.predicate.is_some() || end <= next.address || !xs.iter().any(|x| x.address == end)
            {
                return Err(error(x.address, Unreconverged::ConditionalPush));
            }
            if regions
                .iter()
                .any(|r| r.push < x.address && x.address < r.end && end > r.end)
            {
                return Err(error(x.address, Unreconverged::CrossingRegions));
            }
            regions.push(Region {
                push: x.address,
                start: next.address,
                end,
                stack,
            });
        }
        // A nested SPIR-V loop must have one entry. A jump into a token body
        // bypassing its push cannot inherit that body's participation set.
        // Ryujinx StructuredIr/GotoElimination.cs:65-98 encloses forward
        // transfers in selections and backward transfers in do/while loops;
        // CodeGen/Spirv/SpirvGenerator.cs:334-344,356-369 emits their merges.
        for r in &regions {
            for block in &cfg.blocks {
                let from = xs[block.instructions.end - 1].address;
                for successor in cfg.successors(block) {
                    let to = xs[cfg.blocks[successor].instructions.start].address;
                    if (r.start..r.end).contains(&to) && !(r.push..r.end).contains(&from) {
                        return Err(error(
                            from,
                            Unreconverged::EntersBody {
                                to,
                                start: r.start,
                                end: r.end,
                                push: r.push,
                            },
                        ));
                    }
                }
            }
        }
        // A forward escape keeps its PC and waits at the loop merge. EXIT
        // stores the exit sentinel. KIL demotes and retains its PC and stack
        // (Mesa from_nir.rs:3035-3050), unlike Ryujinx's ambiguous OpKill
        // (Spirv/Instructions.cs:506-509). CONT/BRK may consume intervening
        // kinds (Ryujinx Decoder.cs:824-869).
        // The parent orders surviving PCs, including BRA to a shared tail.
        let mut backedges = Vec::new();
        for block in &cfg.blocks {
            let from = xs[block.instructions.end - 1].address;
            for successor in cfg.successors(block) {
                let to = xs[cfg.blocks[successor].instructions.start].address;
                if to <= from {
                    if regions.iter().any(|r| {
                        (r.start..r.end).contains(&from)
                            && !(r.start..r.end).contains(&to)
                            && !regions.iter().any(|outer| {
                                outer.push < r.push && outer.end >= r.end && outer.end == to
                            })
                    }) {
                        return Err(error(from, Unreconverged::BackwardEscape { to }));
                    }
                    backedges.push((to, from));
                }
            }
        }
        backedges.sort_unstable();
        backedges.dedup();
        // Address order is an execution order only for single-entry loops
        // with one latch. An early backedge could otherwise start iteration
        // N+1 before another path finishes iteration N. Ryujinx encloses
        // backward transfers in do/while loops (GotoElimination.cs:79-85).
        for &(head, latch) in &backedges {
            if backedges.iter().any(|&(h, l)| h == head && l != latch) {
                return Err(error(latch, Unreconverged::MultipleLatches { head }));
            }
            for &(other_head, other_latch) in &backedges {
                if head < other_head && other_head <= latch && latch < other_latch {
                    return Err(error(other_latch, Unreconverged::CrossingIntervals));
                }
            }
            for block in &cfg.blocks {
                let from = xs[block.instructions.end - 1].address;
                for successor in cfg.successors(block) {
                    let to = xs[cfg.blocks[successor].instructions.start].address;
                    if head < to && to <= latch && !(head..=latch).contains(&from) {
                        return Err(error(from, Unreconverged::BypassesHeader { to, head }));
                    }
                }
            }
        }
        for x in xs {
            match x.kind {
                InstructionKind::Brx(ops::Brx { .. })
                | InstructionKind::OutsideBrx(ops::OutsideBrx { .. })
                | InstructionKind::BrxRegister(ops::BrxRegister { .. }) => {
                    return Err(error(x.address, Unreconverged::IndirectBranch));
                }
                InstructionKind::Reconverge(ops::Reconverge { target, stack, .. }) => {
                    let target = target.slot();
                    if !regions.iter().any(|r| {
                        (r.start..r.end).contains(&x.address) && target == r.end && stack == r.stack
                    }) {
                        return Err(error(x.address, Unreconverged::PopOutsideRegion));
                    }
                }
                _ => {}
            }
        }
        let ordered = !regions.is_empty()
            || xs.iter().any(|x| {
                matches!(
                    x.kind,
                    InstructionKind::Branch(ops::Branch { .. })
                        | InstructionKind::OutsideBranch(ops::OutsideBranch { .. })
                )
            });
        Ok(Self {
            tokens: regions,
            ordered,
        })
    }

    pub fn boundaries(&self) -> impl Iterator<Item = u32> + '_ {
        self.tokens.iter().flat_map(|r| [r.push, r.start, r.end])
    }

    pub fn is_enabled(&self) -> bool {
        self.ordered
    }

    /// Rebuild only dispatcher structure. Original instruction bodies,
    /// including predication and collectives, are preserved verbatim.
    pub fn emit(
        &self,
        w: &mut Vec<u32>,
        id: &mut u32,
        cases: &HashMap<u32, Vec<u32>>,
        state: &Dispatch,
    ) {
        self.emit_scope(w, id, cases, state, Root);
    }

    fn emit_scope<S: Scope>(
        &self,
        w: &mut Vec<u32>,
        id: &mut u32,
        cases: &HashMap<u32, Vec<u32>>,
        state: &Dispatch,
        scope: S,
    ) {
        let fresh = |id: &mut u32| {
            let result = *id;
            *id += 1;
            result
        };
        let header = fresh(id);
        let body = fresh(id);
        let merge = fresh(id);
        let continuing = fresh(id);
        let switch_merge = fresh(id);
        let default = fresh(id);
        let selected = fresh(id);
        let turn_merge = fresh(id);
        let members = cases.keys().copied().filter(|address| {
            scope.holds(&self.tokens, *address)
                && !self.tokens.iter().enumerate().any(|(index, r)| {
                    scope.nests(&self.tokens, index) && (r.start..r.end).contains(address)
                })
        });
        let mut members = members.collect::<Vec<_>>();
        members.sort_unstable();
        inst(w, Op::Branch, &[header]);
        inst(w, Op::Label, &[header]);
        inst(w, Op::LoopMerge, &[merge, continuing, 0]);
        inst(w, Op::Branch, &[body]);
        inst(w, Op::Label, &[body]);
        let selector = fresh(id);
        inst(w, Op::Load, &[state.uint, selector, state.pc]);
        // Choose the first pending block uniformly. Unlike independent PC
        // dispatch, a forward BRA cannot execute a shared tail ahead of the
        // fallthrough path. One reduction replaces a runtime stack or a chain
        // of per-target ballots. Ryujinx GotoElimination.cs:79-85 likewise
        // orders forward paths before their shared label.
        // Mesa src/compiler/spirv/spirv.h:1034,1043,2064: Subgroup, Reduce, UMin;
        // :1895,1961-1966: equality and structured branch/merge opcodes.
        let first = fresh(id);
        let ready = fresh(id);
        inst(
            w,
            Op::GroupNonUniformUMin,
            &[state.uint, first, state.constants[&3], 0, selector],
        );
        inst(w, Op::IEqual, &[state.boolt, ready, selector, first]);
        inst(w, Op::SelectionMerge, &[turn_merge, 0]);
        inst(w, Op::BranchConditional, &[ready, selected, turn_merge]);
        inst(w, Op::Label, &[selected]);
        inst(w, Op::SelectionMerge, &[switch_merge, 0]);
        let mut switch = vec![selector, default];
        for address in &members {
            switch.extend([state.case_of[address], state.labels[address]]);
        }
        inst(w, Op::Switch, &switch);
        for address in members {
            let code = &cases[&address];
            let mut at = 0;
            while at < code.len() {
                let count = (code[at] >> 16) as usize;
                if Op::of(code[at]) == Op::Branch && code[at + 1] == state.old_merge {
                    // The region this member pushes, if any, nests here.
                    for (child, _) in self
                        .tokens
                        .iter()
                        .enumerate()
                        .filter(|(_, r)| r.push == address)
                    {
                        self.emit_scope(w, id, cases, state, Child(child));
                    }
                    inst(w, Op::Branch, &[switch_merge]);
                } else {
                    w.extend_from_slice(&code[at..at + count]);
                }
                at += count;
            }
        }
        inst(w, Op::Label, &[default]);
        inst(w, Op::Store, &[state.pc, state.exit]);
        inst(w, Op::Branch, &[switch_merge]);
        inst(w, Op::Label, &[switch_merge]);
        inst(w, Op::Branch, &[turn_merge]);
        inst(w, Op::Label, &[turn_merge]);
        inst(w, Op::Branch, &[continuing]);
        inst(w, Op::Label, &[continuing]);
        let pc = fresh(id);
        inst(w, Op::Load, &[state.uint, pc, state.pc]);
        let running = fresh(id);
        scope.running(&self.tokens, w, id, state, pc, running);
        inst(w, Op::BranchConditional, &[running, header, merge]);
        inst(w, Op::Label, &[merge]);
    }
}

/// Where a dispatcher loop runs: the whole program, or one token region's
/// body nested in its parent's loop.
trait Scope: Copy {
    /// Whether `address` lies in this scope.
    fn holds(self, tokens: &[Region], address: u32) -> bool;
    /// Whether region `index` is a region nested inside this scope.
    fn nests(self, tokens: &[Region], index: usize) -> bool;
    /// Writes `running`: whether the loaded `pc` keeps this loop iterating.
    fn running(
        self,
        tokens: &[Region],
        w: &mut Vec<u32>,
        id: &mut u32,
        state: &Dispatch,
        pc: u32,
        running: u32,
    );
}

/// The program's outermost dispatcher loop.
#[derive(Clone, Copy)]
struct Root;

/// The dispatcher loop of the token region at this index.
#[derive(Clone, Copy)]
struct Child(usize);

impl Scope for Root {
    fn holds(self, _: &[Region], _: u32) -> bool {
        true
    }

    fn nests(self, _: &[Region], _: usize) -> bool {
        true
    }

    /// The program runs until its PC holds the exit sentinel.
    fn running(
        self,
        _: &[Region],
        w: &mut Vec<u32>,
        _: &mut u32,
        state: &Dispatch,
        pc: u32,
        running: u32,
    ) {
        inst(w, Op::INotEqual, &[state.boolt, running, pc, state.exit]);
    }
}

impl Scope for Child {
    fn holds(self, tokens: &[Region], address: u32) -> bool {
        (tokens[self.0].start..tokens[self.0].end).contains(&address)
    }

    fn nests(self, tokens: &[Region], index: usize) -> bool {
        let parent = &tokens[self.0];
        let r = &tokens[index];
        index != self.0 && r.push >= parent.start && r.end <= parent.end
    }

    /// A region runs while its PC stays inside its body.
    fn running(
        self,
        tokens: &[Region],
        w: &mut Vec<u32>,
        id: &mut u32,
        state: &Dispatch,
        pc: u32,
        running: u32,
    ) {
        let r = &tokens[self.0];
        let lower = *id;
        let upper = *id + 1;
        *id += 2;
        inst(
            w,
            Op::UGreaterThanEqual,
            &[
                state.boolt,
                lower,
                pc,
                state.constants[&state.case_of[&r.start]],
            ],
        );
        inst(
            w,
            Op::ULessThan,
            &[
                state.boolt,
                upper,
                pc,
                state.constants[&state.case_of[&r.end]],
            ],
        );
        inst(w, Op::LogicalAnd, &[state.boolt, running, lower, upper]);
    }
}

pub(crate) struct Dispatch<'a> {
    pub uint: u32,
    pub boolt: u32,
    pub pc: u32,
    pub old_merge: u32,
    pub exit: u32,
    pub case_of: &'a HashMap<u32, u32>,
    pub labels: &'a HashMap<u32, u32>,
    pub constants: &'a HashMap<u32, u32>,
}
