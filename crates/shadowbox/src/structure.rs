//! Control-flow analysis and structurisation: blocks, dominance, reconvergence,
//! and structured selections and loops before emission.

/// A structured control-flow result.
#[derive(Debug)]
pub struct StructuredProgram {
    /// Number of regions discovered by the structuriser.
    pub region_count: usize,
}

use crate::decoder::Slots;
use crate::flow;
use crate::interface::{MissingTarget, Stopped, TargetGap};
use crate::ir::{Instruction, NoSlotEntry, Predicate};
use std::ops::Range;

/// The program's blocks, built once from the decoded instructions. Each block
/// holds its instruction positions and the blocks its terminator reaches:
/// first the edges taken under a guard predicate, then the edges taken always,
/// in the order the terminator names them. SSY/SYNC and PBK/BRK/PCNT/CONT are
/// resolved by decode_stream before this graph is built by
/// decoder::resolve_stack_jumps. Ryujinx likewise branches to the push target:
/// Instructions/InstEmitFlowControl.cs:254-275.
#[derive(Default)]
pub(crate) struct ControlFlowGraph {
    pub blocks: Vec<BasicBlock>,
    guarded: Vec<Guarded>,
    always: Vec<usize>,
    /// Each guarded BRA or resolved pop with an instruction after it, in
    /// program order.
    conditionals: Vec<Conditional>,
    /// The guarded BRA or resolved pop that is the last instruction.
    finals: Vec<Final>,
}

pub(crate) struct BasicBlock {
    pub instructions: Range<usize>,
    guarded: Range<usize>,
    always: Range<usize>,
}

/// An edge taken only when `guard` holds.
#[derive(Clone, Copy)]
pub(crate) struct Guarded {
    pub block: usize,
    pub guard: Predicate,
}

/// A guarded BRA or resolved pop at position `at` to slot `target`, with
/// the instruction at `next` after it.
#[derive(Clone, Copy)]
struct Conditional {
    at: usize,
    target: u32,
    next: usize,
}

/// A guarded BRA or resolved pop at position `at` to slot `target` that ends
/// the program: nothing follows it.
#[derive(Clone, Copy)]
struct Final {
    at: usize,
    target: u32,
}

impl ControlFlowGraph {
    /// Edges `block` takes under a guard.
    pub fn guarded(&self, block: &BasicBlock) -> &[Guarded] {
        &self.guarded[block.guarded.clone()]
    }

    /// Edges `block` takes always.
    pub fn always(&self, block: &BasicBlock) -> &[usize] {
        &self.always[block.always.clone()]
    }

    /// Every block `block` reaches, guarded edges first.
    pub fn successors<'a>(&'a self, block: &'a BasicBlock) -> impl Iterator<Item = usize> + 'a {
        self.guarded(block)
            .iter()
            .map(|edge| edge.block)
            .chain(self.always(block).iter().copied())
    }
}

#[derive(Default)]
pub(crate) struct Scratch {
    starts: Vec<bool>,
    guarded: Vec<(usize, Predicate)>,
    always: Vec<usize>,
    /// Each instruction's guarded and always edges.
    leaves: Vec<(Range<usize>, Range<usize>)>,
    owner: Vec<usize>,
}

/// One pass over the instructions: block starts and each instruction's edges.
pub(crate) struct Build<'a> {
    xs: &'a [Instruction],
    slots: &'a Slots,
    first_address: u32,
    last_address: u32,
    starts: &'a mut [bool],
    guarded: &'a mut Vec<(usize, Predicate)>,
    always: &'a mut Vec<usize>,
    conditionals: &'a mut Vec<Conditional>,
    finals: &'a mut Vec<Final>,
}

// Mesa src/nouveau/compiler/nak/sm50.rs:3025-3028 encodes BRA relative to
// the next slot; sm50.rs:3006-3016 defines target_ip - ip - 8. BRX adds its
// table offset to that base (envytools/envydis/gm107.c:1942-1945).
fn missing(x: &Instruction, target: i64, gap: TargetGap) -> Stopped {
    Stopped::MissingTarget(MissingTarget {
        at: x.address,
        target,
        gap,
    })
}

fn outside(x: &Instruction, target: i64) -> Stopped {
    missing(x, target, TargetGap::OutsideCode)
}

impl Build<'_> {
    /// The instruction running `slot`, named by the branch at `at`.
    fn destination(&self, at: usize, slot: u32) -> Result<usize, Stopped> {
        self.slots.get(slot).ok_or_else(|| {
            let (first, last) = (self.first_address, self.last_address);
            let gap = if slot < first || slot > last {
                TargetGap::OutsideDecoded { first, last }
            } else {
                TargetGap::NotASlot
            };
            missing(&self.xs[at], i64::from(slot), gap)
        })
    }

    /// A branch from `at` to `slot` taken under `guard`; its target starts a block.
    fn jump(&mut self, at: usize, slot: u32, guard: impl EdgeGuard) -> Result<(), Stopped> {
        self.destination(at, slot).map(|to| {
            guard.taken(to, self);
            self.starts[to] = true;
        })
    }
}

/// When an edge is taken: always, or under the guard field's predicate.
pub(crate) trait EdgeGuard: Copy {
    /// An edge to `to` taken when this guard holds.
    fn taken(self, to: usize, build: &mut Build);
    /// The fallthrough to `to` of a terminator, taken when this guard fails.
    fn skipped(self, to: usize, build: &mut Build);
    /// A BRA or resolved pop at `at` to slot `target` under this guard.
    fn branch(self, at: usize, target: u32, next: impl Next, build: &mut Build);
}

/// No guard: the instruction always executes.
impl EdgeGuard for crate::guard::Unguarded {
    fn taken(self, to: usize, build: &mut Build) {
        build.always.push(to);
    }
    fn skipped(self, _: usize, _: &mut Build) {}
    fn branch(self, _: usize, _: u32, _: impl Next, _: &mut Build) {}
}

/// A guard field, PT included: its edges are taken under its predicate.
impl<G: crate::guard::Field> EdgeGuard for G {
    fn taken(self, to: usize, build: &mut Build) {
        build.guarded.push((to, self.predicate()));
    }
    fn skipped(self, to: usize, build: &mut Build) {
        let p = self.predicate();
        build.guarded.push((
            to,
            Predicate {
                inverted: !p.inverted,
                ..p
            },
        ));
    }
    fn branch(self, at: usize, target: u32, next: impl Next, build: &mut Build) {
        next.branch(at, target, build);
    }
}

/// What follows an instruction: the next one, or the program's end.
pub(crate) trait Next: Copy {
    /// Execution continues to the next instruction.
    fn always(self, build: &mut Build);
    /// Execution continues to the next instruction when `guard` fails.
    fn skipped(self, guard: impl EdgeGuard, build: &mut Build);
    /// The instruction ends its block.
    fn ends(self, build: &mut Build);
    /// A guarded BRA or resolved pop at `at` to slot `target`.
    fn branch(self, at: usize, target: u32, build: &mut Build);
}

/// The instruction at the position it holds follows.
#[derive(Clone, Copy)]
pub(crate) struct Follows(pub(crate) usize);

/// The last instruction: its block already ends with the program.
#[derive(Clone, Copy)]
pub(crate) struct End;

impl Next for Follows {
    fn always(self, build: &mut Build) {
        build.always.push(self.0);
    }
    fn skipped(self, guard: impl EdgeGuard, build: &mut Build) {
        guard.skipped(self.0, build);
    }
    fn ends(self, build: &mut Build) {
        build.starts[self.0] = true;
    }
    fn branch(self, at: usize, target: u32, build: &mut Build) {
        build.conditionals.push(Conditional {
            at,
            target,
            next: self.0,
        });
    }
}

impl Next for End {
    fn always(self, _: &mut Build) {}
    fn skipped(self, _: impl EdgeGuard, _: &mut Build) {}
    fn ends(self, _: &mut Build) {}
    fn branch(self, at: usize, target: u32, build: &mut Build) {
        build.finals.push(Final { at, target });
    }
}

/// How an instruction leaves its block: each transfer type's behavior here.
pub(crate) trait Leave {
    fn leave(
        self,
        at: usize,
        guard: impl EdgeGuard,
        next: impl Next,
        build: &mut Build,
    ) -> Result<(), Stopped>;
}

/// What a condition-code test does to a terminator's fallthrough.
pub(crate) trait CcEdge: Copy {
    /// The fallthrough of a terminator under `guard` and this test.
    fn fallthrough(self, guard: impl EdgeGuard, next: impl Next, build: &mut Build);
}

/// CC.T: the fallthrough is taken exactly when the guard fails.
impl CcEdge for flow::CcT {
    fn fallthrough(self, guard: impl EdgeGuard, next: impl Next, build: &mut Build) {
        next.skipped(guard, build);
    }
}

/// Any other test can fail with its guard true, so the fallthrough is
/// taken always.
impl CcEdge for flow::CcTest {
    fn fallthrough(self, _: impl EdgeGuard, next: impl Next, build: &mut Build) {
        next.always(build);
    }
}

impl<C: flow::Cc> Leave for flow::Jump<C> {
    fn leave(
        self,
        at: usize,
        guard: impl EdgeGuard,
        next: impl Next,
        build: &mut Build,
    ) -> Result<(), Stopped> {
        build.jump(at, self.slot(), guard)?;
        guard.branch(at, self.slot(), next, build);
        self.cc().fallthrough(guard, next, build);
        next.ends(build);
        Ok(())
    }
}

/// A proven pop branches to its destination like BRA.
impl<C: flow::Cc> Leave for flow::Reconverge<C> {
    fn leave(
        self,
        at: usize,
        guard: impl EdgeGuard,
        next: impl Next,
        build: &mut Build,
    ) -> Result<(), Stopped> {
        build.jump(at, self.slot(), guard)?;
        guard.branch(at, self.slot(), next, build);
        self.cc().fallthrough(guard, next, build);
        next.ends(build);
        Ok(())
    }
}

impl<C, Y: crate::flow::Gap> Leave for flow::NoSlotJump<C, Y> {
    fn leave(
        self,
        at: usize,
        _: impl EdgeGuard,
        _: impl Next,
        build: &mut Build,
    ) -> Result<(), Stopped> {
        Err(missing(&build.xs[at], self.address(), self.gap()))
    }
}

impl<C: flow::Cc> Leave for flow::AnyJump<C> {
    fn leave(
        self,
        _: usize,
        guard: impl EdgeGuard,
        next: impl Next,
        build: &mut Build,
    ) -> Result<(), Stopped> {
        // Ryujinx InstEmitFlowControl.cs:38-42: runtime target address.
        // Every provided slot is a possible successor; no reaching-definition
        // fact may cross a guessed edge.
        for target in 0..build.xs.len() {
            guard.taken(target, build);
            build.starts[target] = true;
        }
        self.cc().fallthrough(guard, next, build);
        next.ends(build);
        Ok(())
    }
}

impl Leave for flow::TableJump {
    fn leave(
        self,
        at: usize,
        guard: impl EdgeGuard,
        next: impl Next,
        build: &mut Build,
    ) -> Result<(), Stopped> {
        for target in self.targets() {
            build.jump(at, target.slot(), guard)?;
        }
        next.skipped(guard, build);
        next.ends(build);
        Ok(())
    }
}

impl Leave for flow::OutsideTable {
    fn leave(
        self,
        at: usize,
        _: impl EdgeGuard,
        _: impl Next,
        build: &mut Build,
    ) -> Result<(), Stopped> {
        // The first missing entry names the error, as for Brx: a leading
        // slot entry, else the first outside entry.
        for target in self.table().leading() {
            build.destination(at, target.slot())?;
        }
        Err(match self.table().first_outside() {
            NoSlotEntry::Outside(entry) => outside(&build.xs[at], entry.address(self.base())),
            NoSlotEntry::Misaligned(entry) => missing(
                &build.xs[at],
                i64::from(entry.address()),
                TargetGap::NotASlot,
            ),
        })
    }
}

impl<C: flow::Cc> Leave for flow::Exit<C> {
    fn leave(
        self,
        _: usize,
        guard: impl EdgeGuard,
        next: impl Next,
        build: &mut Build,
    ) -> Result<(), Stopped> {
        self.cc().fallthrough(guard, next, build);
        next.ends(build);
        Ok(())
    }
}

impl<C> Leave for flow::Pop<C> {
    fn leave(
        self,
        at: usize,
        _: impl EdgeGuard,
        _: impl Next,
        build: &mut Build,
    ) -> Result<(), Stopped> {
        Err(Stopped::UnprovenPop(build.xs[at].address))
    }
}

/// A resolved push is not a jump. Its target starts a block, with an
/// incoming edge only from a resolved pop, never from the push.
impl<P: flow::PushOp> Leave for flow::ResolvedPush<P> {
    fn leave(
        self,
        _: usize,
        _: impl EdgeGuard,
        next: impl Next,
        build: &mut Build,
    ) -> Result<(), Stopped> {
        next.always(build);
        if let Some(target) = build.slots.get(self.slot()) {
            build.starts[target] = true;
        }
        Ok(())
    }
}

/// Every other instruction continues to the next. Mesa from_nir.rs:3035-3050:
/// KIL demotes and continues.
macro_rules! steps {
    ($($t:ty),*) => {$(
        impl Leave for $t {
            fn leave(
                self,
                _: usize,
                _: impl EdgeGuard,
                next: impl Next,
                build: &mut Build,
            ) -> Result<(), Stopped> {
                next.always(build);
                Ok(())
            }
        }
    )*};
}
steps!(
    flow::NoSlotPush,
    flow::Call,
    flow::NoSlotCall,
    flow::Unknown,
    flow::Step,
    flow::PredicateWrite
);

/// An unresolved push continues to the next instruction.
impl<P: flow::PushOp> Leave for flow::Push<P> {
    fn leave(
        self,
        _: usize,
        _: impl EdgeGuard,
        next: impl Next,
        build: &mut Build,
    ) -> Result<(), Stopped> {
        next.always(build);
        Ok(())
    }
}

impl ControlFlowGraph {
    #[cfg(test)]
    pub fn build(xs: &[Instruction]) -> Result<Self, Stopped> {
        let mut cfg = Self::default();
        cfg.build_into(
            xs,
            &flow::Program::of(xs),
            &Slots::of(xs),
            &mut Scratch::default(),
        )?;
        Ok(cfg)
    }

    fn build_into(
        &mut self,
        xs: &[Instruction],
        program: &flow::Program,
        slots: &Slots,
        scratch: &mut Scratch,
    ) -> Result<(), Stopped> {
        self.blocks.clear();
        self.guarded.clear();
        self.always.clear();
        self.conditionals.clear();
        self.finals.clear();
        let line = program.line();
        let Some((end, things)) = line.split_last() else {
            return Ok(());
        };
        let last = &xs[things.len()];
        let Scratch {
            starts,
            guarded,
            always,
            leaves,
            owner,
        } = scratch;
        starts.clear();
        starts.resize(xs.len() + 1, false);
        starts[0] = true;
        starts[xs.len()] = true;
        guarded.clear();
        always.clear();
        leaves.clear();
        let mut build = Build {
            xs,
            slots,
            first_address: xs.iter().fold(last.address, |m, x| m.min(x.address)),
            last_address: xs.iter().fold(last.address, |m, x| m.max(x.address)),
            starts,
            guarded,
            always,
            conditionals: &mut self.conditionals,
            finals: &mut self.finals,
        };
        for thing in things {
            record(&mut build, leaves, |build| thing.leave(build))?;
        }
        record(&mut build, leaves, |build| end.leave_last(build))?;
        owner.clear();
        owner.resize(xs.len(), 0);
        let mut begin = 0;
        for (end, &is_start) in starts.iter().enumerate().skip(1) {
            if is_start {
                let instructions = begin..end;
                owner[instructions.clone()].fill(self.blocks.len());
                self.blocks.push(BasicBlock {
                    instructions,
                    guarded: 0..0,
                    always: 0..0,
                });
                begin = end;
            }
        }
        for block in &mut self.blocks {
            let (taken, falls) = leaves[block.instructions.end - 1].clone();
            let begin = self.guarded.len();
            self.guarded
                .extend(guarded[taken].iter().map(|&(i, guard)| Guarded {
                    block: owner[i],
                    guard,
                }));
            block.guarded = begin..self.guarded.len();
            let begin = self.always.len();
            self.always.extend(always[falls].iter().map(|&i| owner[i]));
            block.always = begin..self.always.len();
        }
        Ok(())
    }
}

/// The edges one instruction `leave`s by, recorded in `leaves`.
fn record(
    build: &mut Build,
    leaves: &mut Vec<(Range<usize>, Range<usize>)>,
    leave: impl FnOnce(&mut Build) -> Result<(), Stopped>,
) -> Result<(), Stopped> {
    let (taken, falls) = (build.guarded.len(), build.always.len());
    leave(build)?;
    leaves.push((taken..build.guarded.len(), falls..build.always.len()));
    Ok(())
}

/// Validate reducible conditional diamonds and natural back-edge loops.
/// A branch target is an SM50 byte offset relative to the instruction following BRA.
pub fn analyze(xs: &[Instruction]) -> Result<StructuredProgram, Stopped> {
    analyze_into(
        xs,
        &flow::Program::of(xs),
        &Slots::of(xs),
        &mut ControlFlowGraph::default(),
        &mut Scratch::default(),
    )
}

/// The graph and its build storage, owned by one translator. The decoder
/// builds it once; every later pass borrows it.
#[derive(Default)]
pub(crate) struct GraphStore {
    cfg: ControlFlowGraph,
    scratch: Scratch,
}

impl GraphStore {
    /// The structured graph of `xs`, or why it has none.
    pub fn build(&mut self, xs: &[Instruction]) -> Result<&ControlFlowGraph, Stopped> {
        self.build_in(xs, &flow::Program::of(xs), &Slots::of(xs))
    }

    /// [`Self::build`] over the program `xs` arrived as and the slots
    /// discovery established for it.
    pub fn build_in(
        &mut self,
        xs: &[Instruction],
        program: &flow::Program,
        slots: &Slots,
    ) -> Result<&ControlFlowGraph, Stopped> {
        analyze_into(xs, program, slots, &mut self.cfg, &mut self.scratch).map(|_| &self.cfg)
    }
}

fn analyze_into(
    xs: &[Instruction],
    program: &flow::Program,
    slots: &Slots,
    cfg: &mut ControlFlowGraph,
    scratch: &mut Scratch,
) -> Result<StructuredProgram, Stopped> {
    cfg.build_into(xs, program, slots, scratch)?;
    let max_address = xs.iter().map(|x| x.address).max().unwrap_or(0);
    let mut regions = 0;
    for &Conditional { at, target, next } in &cfg.conditionals {
        let x = xs[at];
        if target < x.address || (max_address > xs[next].address && max_address >= target) {
            regions += 1;
        } else {
            return Err(Stopped::NoMerge(x.address));
        }
    }
    for &Final { at, target } in &cfg.finals {
        let x = xs[at];
        if target < x.address {
            regions += 1;
        } else {
            return Err(Stopped::NoFallthrough(x.address));
        }
    }
    Ok(StructuredProgram {
        region_count: regions,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::ops;
    use crate::ir::{BranchTarget, BrxBase, Instruction, InstructionKind, Named, Predicate};

    fn branch(address: u32, target: i32, conditional: bool) -> Instruction {
        Instruction {
            address,
            predicate: conditional.then_some(Predicate {
                register: 0,
                inverted: false,
            }),
            kind: match BranchTarget::named(address, target) {
                Named::Outside(target) => InstructionKind::OutsideBranch(ops::OutsideBranch {
                    target,
                    cc_test: 0xf,
                }),
                Named::Misaligned(target) => {
                    InstructionKind::MisalignedBranch(ops::MisalignedBranch {
                        target,
                        cc_test: 0xf,
                    })
                }
                // Synthetic slots are dense: reach the named address itself.
                Named::Slot(_) => InstructionKind::Branch(ops::Branch {
                    target: BranchTarget::reaching(
                        address,
                        (i64::from(address) + 8 + i64::from(target)) as u32,
                    ),
                    cc_test: 0xf,
                }),
            },
        }
    }
    fn nop(address: u32) -> Instruction {
        Instruction {
            address,
            predicate: None,
            kind: InstructionKind::Nop(ops::Nop),
        }
    }

    #[test]
    fn accepts_if_else_diamond() {
        let xs = [branch(0, 16, true), nop(8), nop(16), nop(24)];
        assert_eq!(analyze(&xs).unwrap().region_count, 1);
    }

    #[test]
    fn accepts_simple_loop_back_edge() {
        let xs = [nop(0), branch(8, -16, true)];
        assert_eq!(analyze(&xs).unwrap().region_count, 1);
    }

    #[test]
    fn rejects_missing_target() {
        let error = analyze(&[branch(0, 100, true), nop(8)]).unwrap_err();
        assert!(error.to_string().contains("missing address"));
    }

    #[test]
    fn rejects_targets_that_wrap_into_existing_addresses() {
        // Relative displacement base: Mesa sm50.rs:3006-3028. These IR
        // address-space extremes must not wrap a nonexistent target onto
        // the other instruction, even though that address is in the map.
        for xs in [
            [branch(8, -24, false), nop(u32::MAX - 7)],
            [branch(u32::MAX - 7, 0, false), nop(0)],
        ] {
            let error = analyze(&xs).unwrap_err();
            assert!(error.to_string().contains("outside the code address space"));
        }
    }

    #[test]
    fn missing_brx_table_targets_remain_errors() {
        // BRX table offsets: envytools gm107.c:1942-1943; Ryujinx
        // Decoder.cs:543-552. Validate every entry, including later ones.
        for missing in [12, 64] {
            let mut targets = [0; 32];
            targets[..2].copy_from_slice(&[0, missing]);
            let xs = [
                Instruction {
                    address: 8,
                    predicate: None,
                    kind: InstructionKind::brx(
                        crate::ir::Register(2),
                        BrxBase::decode(8, 0),
                        targets[0],
                        &targets[1..2],
                    ),
                },
                nop(16),
                nop(24),
            ];
            let error = analyze(&xs).unwrap_err();
            assert!(error.to_string().contains("targets missing address"));
        }
    }

    #[test]
    fn accepts_branch_to_previous_bundle_slot() {
        let xs = [nop(8), nop(16), nop(24), branch(40, -32, true)];
        assert_eq!(analyze(&xs).unwrap().region_count, 1);
    }

    #[test]
    fn accepts_one_instruction_loop() {
        let xs = [nop(8), branch(16, -16, true)];
        assert_eq!(analyze(&xs).unwrap().region_count, 1);
    }

    #[test]
    fn accepts_forward_branch_to_next_bundle_slot() {
        let xs = [branch(8, 24, true), nop(16), nop(24), nop(32), nop(40)];
        assert_eq!(analyze(&xs).unwrap().region_count, 1);
    }

    #[test]
    fn rejects_branch_to_middle_of_instruction_pair() {
        // The target 0x10 is not a decoded slot address in this synthetic
        // stream, so a branch into the middle of a bundle is rejected.
        let error = analyze(&[branch(0, 8, true), nop(8), nop(24)]).unwrap_err();
        assert!(error.to_string().contains("missing address 0x10"));
    }
}
