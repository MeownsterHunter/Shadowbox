//! Bounded block dataflow for Maxwell's ordered control stack.
//!
//! Adapted semantics: Ryujinx Decoders/Decoder.cs:753-883. Unlike its
//! per-producer DFS with one visited set, union all reaching stack heads and
//! parent relationships. This conservatively represents differing joins and
//! growing loop stacks without cloning or enumerating complete stacks.
//
// Copyright (c) Ryujinx Team and Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

use crate::decoder::Slots;
use crate::fast_hash::{HashMap, HashSet};
use crate::flow::{self, CcT, CcTest, Consumed, Program, PushOp, Reached};
use crate::guard::Guard;
use crate::ir::{BranchTarget, Instruction, TargetStack};
use crate::run::Run;
use std::collections::VecDeque;
use std::hash::Hash;
use std::iter::{once, repeat};

/// The empty stack's head. No push has it, so searching it is a possible
/// underflow.
const EMPTY: usize = usize::MAX;

/// A proof that does not complete: a target outside the program, an
/// instruction that might hide control flow, a possible underflow, or
/// exhausted work. It publishes nothing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Unproven;

/// Continues exactly when `fact` holds.
fn holds(fact: bool) -> Result<(), Unproven> {
    [Err(Unproven), Ok(())][usize::from(fact)]
}

/// The one value a standard lookup carries; a lookup that carries none leaves
/// the proof unfinished.
fn carried<T>(value: impl IntoIterator<Item = T>) -> Result<T, Unproven> {
    value.into_iter().last().ok_or(Unproven)
}

/// Every item when `keep` is true, none when it is false.
fn masked<I: IntoIterator>(items: I, keep: bool) -> std::iter::Take<I::IntoIter> {
    items.into_iter().take([0, usize::MAX][usize::from(keep)])
}

pub(crate) struct Resolution {
    /// Each proven pop and the block start it reaches.
    reached: Reached,
    consumed: Consumed,
    /// Whether every pop of the program was proven.
    covers_pops: bool,
}

impl Resolution {
    pub(crate) fn covers_pops(&self) -> bool {
        self.covers_pops
    }

    pub(crate) fn apply(self, program: &mut Program, xs: &[Instruction]) {
        program.lower(&self.reached, xs);
        program.settle(&self.consumed);
    }
}

/// A push: the stack it uses and the block its target starts.
#[derive(Clone, Copy)]
struct Push {
    stack: TargetStack,
    block: usize,
}

/// Block starts and pushes, gathered as each thing marks its instruction.
pub(crate) struct Marks<'s> {
    slots: &'s Slots,
    /// Tells of each guard proven PT or !PT.
    pruned: fn(&Instruction),
    starts: Vec<bool>,
    /// Each push: where it is, its stack, and the instruction it targets.
    pushed: Vec<(usize, TargetStack, usize)>,
}

impl Marks<'_> {
    /// The instruction running `slot`, now a block start.
    fn jump(&mut self, slot: u32) -> Result<usize, Unproven> {
        carried(self.slots.get(slot)).inspect(|&to| self.starts[to] = true)
    }

    fn push(
        &mut self,
        at: usize,
        stack: TargetStack,
        target: BranchTarget,
    ) -> Result<(), Unproven> {
        // Ryujinx Decoder.cs:133-138 creates push-target blocks.
        self.jump(target.slot()).map(|to| {
            self.pushed.push((at, stack, to));
            self.starts[at] = true;
            self.starts[at + 1] = true;
        })
    }

    /// The instruction at `at` ends its block.
    fn ends(&mut self, at: usize) {
        self.starts[at + 1] = true;
    }

    /// The block graph, each block ending as its last instruction's
    /// `endings` says.
    fn graph(self, xs: &[Instruction], endings: Vec<Ending>, pops: usize) -> Graph {
        let starts = self
            .starts
            .iter()
            .enumerate()
            .flat_map(|(i, &start)| masked(once(i), start))
            .collect::<Vec<_>>();
        let mut owner = vec![0; xs.len()];
        starts
            .windows(2)
            .enumerate()
            .for_each(|(block, pair)| owner[pair[0]..pair[1]].fill(block));
        let pushes = self
            .pushed
            .into_iter()
            .map(|(at, stack, to)| {
                (
                    at,
                    Push {
                        stack,
                        block: owner[to],
                    },
                )
            })
            .collect::<HashMap<_, _>>();
        // Each block closes at its last instruction: the one before a start.
        let lasts = endings
            .into_iter()
            .zip(&self.starts[1..])
            .flat_map(|(ending, &end)| masked(once(ending), end));
        let blocks = starts
            .windows(2)
            .zip(lasts)
            .map(|(pair, ending)| ending.close(pair[0], pair[1] - 1, &xs[pair[1] - 1], &owner))
            .collect();
        Graph {
            blocks,
            pushes,
            pops,
        }
    }
}

/// Whether execution continues past an instruction that runs, as what
/// that instruction is.
pub(crate) trait Fall: Copy {
    /// Where execution continues on the arriving stack: `next` or nowhere.
    fn falls(self, next: usize) -> Option<usize>;
    /// Where an unguarded instruction continues: on the arriving stack,
    /// and with its push on it.
    fn unguarded(self, next: usize) -> (Option<usize>, Option<usize>) {
        (self.falls(next), None)
    }
    /// Where a skipped push continues with the push on the stack.
    fn skipped(self, _: usize) -> Option<usize> {
        None
    }
    /// The push it executes, as the block's last instruction `last`.
    fn executes(self, _: usize) -> Option<usize> {
        None
    }
}

/// An instruction that continues to the next.
#[derive(Clone, Copy)]
struct Falls(());

/// A branch that never continues to the next instruction.
#[derive(Clone, Copy)]
struct Stops(());

/// A push: it continues to the next instruction with its target on the
/// stack.
#[derive(Clone, Copy)]
struct Pushes(());

impl Fall for Falls {
    fn falls(self, next: usize) -> Option<usize> {
        Some(next)
    }
}

impl Fall for Stops {
    fn falls(self, _: usize) -> Option<usize> {
        None
    }
}

impl Fall for Pushes {
    fn falls(self, next: usize) -> Option<usize> {
        Some(next)
    }
    /// A push changes the stack on the fallthrough it always executes.
    fn unguarded(self, next: usize) -> (Option<usize>, Option<usize>) {
        (None, Some(next))
    }
    /// The synthetic push's skipped edge: the next block, with the push on
    /// the stack. Maxwell's SSY/PBK/PCNT have no predicate field
    /// (envytools gm107.c:1930-1935).
    fn skipped(self, next: usize) -> Option<usize> {
        Some(next)
    }
    fn executes(self, last: usize) -> Option<usize> {
        Some(last)
    }
}

/// Conditional control retains its fallthrough, Ryujinx
/// InstEmitFlowControl.cs:153-186.
impl Fall for CcTest {
    fn falls(self, next: usize) -> Option<usize> {
        Some(next)
    }
}

impl Fall for CcT {
    fn falls(self, _: usize) -> Option<usize> {
        None
    }
}

/// How a thing marks its instruction for the block proof, and how the
/// block it would end ends.
pub(crate) trait Mark: Copy {
    fn mark<G: Guard>(self, at: usize, guard: G, marks: &mut Marks) -> Result<Ending, Unproven>;
}

/// What leaves the instruction at `at`: where it falls, the instructions
/// its execution reaches, the stack its pop searches, and its waiting pop.
pub(crate) struct Ends<F> {
    at: usize,
    pruned: fn(&Instruction),
    fall: F,
    successors: Vec<usize>,
    searches: Vec<Stack>,
    lowers: Vec<usize>,
}

impl<F: Fall> Ends<F> {
    /// The instruction after it.
    fn next(&self) -> usize {
        self.at + 1
    }
    /// The ending of an executed instruction: `carries` on the arriving
    /// stack, `stacks` with its push on it.
    fn executes(self, carries: Vec<usize>, stacks: Vec<usize>, report: fn(&Instruction)) -> Ending {
        Ending {
            carries,
            stacks,
            pushes: self.fall.executes(self.at),
            searches: self.searches,
            lowers: self.lowers,
            report,
        }
    }
    /// Instructions the executed instruction reaches on the arriving stack.
    fn executed(&self) -> impl Iterator<Item = usize> + '_ {
        self.successors.iter().copied()
    }
}

/// The instruction at `at` under `guard`, with `fall`, reaching
/// `successors` when it executes, searching `searches` and with the pop
/// `lowers`.
fn ending<G: Guard, F: Fall>(
    at: usize,
    guard: G,
    marks: &Marks,
    fall: F,
    successors: Vec<usize>,
    searches: Vec<Stack>,
    lowers: Vec<usize>,
) -> Ending {
    guard.block(Ends {
        at,
        pruned: marks.pruned,
        fall,
        successors,
        searches,
        lowers,
    })
}

impl<C: flow::Cc> Mark for flow::Jump<C> {
    fn mark<G: Guard>(self, at: usize, guard: G, marks: &mut Marks) -> Result<Ending, Unproven> {
        marks.jump(self.slot()).map(|to| {
            marks.ends(at);
            ending(
                at,
                guard,
                marks,
                self.cc(),
                vec![to],
                Vec::new(),
                Vec::new(),
            )
        })
    }
}

impl<C: flow::Cc> Mark for flow::Reconverge<C> {
    fn mark<G: Guard>(self, at: usize, guard: G, marks: &mut Marks) -> Result<Ending, Unproven> {
        marks.ends(at);
        let searches = vec![Stack(self.stack())];
        Ok(ending(
            at,
            guard,
            marks,
            self.cc(),
            Vec::new(),
            searches,
            Vec::new(),
        ))
    }
}

impl<C: flow::Cc> Mark for flow::Pop<C> {
    fn mark<G: Guard>(self, at: usize, guard: G, marks: &mut Marks) -> Result<Ending, Unproven> {
        marks.ends(at);
        let searches = vec![Stack(self.stack())];
        Ok(ending(
            at,
            guard,
            marks,
            self.cc(),
            Vec::new(),
            searches,
            vec![at],
        ))
    }
}

impl<C: flow::Cc> Mark for flow::Exit<C> {
    fn mark<G: Guard>(self, at: usize, guard: G, marks: &mut Marks) -> Result<Ending, Unproven> {
        marks.ends(at);
        Ok(ending(
            at,
            guard,
            marks,
            self.cc(),
            Vec::new(),
            Vec::new(),
            Vec::new(),
        ))
    }
}

impl Mark for flow::TableJump {
    fn mark<G: Guard>(self, at: usize, guard: G, marks: &mut Marks) -> Result<Ending, Unproven> {
        self.targets()
            .iter()
            .map(|entry| marks.jump(entry.slot()))
            .collect::<Result<Vec<_>, _>>()
            .map(|successors| {
                marks.ends(at);
                ending(
                    at,
                    guard,
                    marks,
                    Stops(()),
                    successors,
                    Vec::new(),
                    Vec::new(),
                )
            })
    }
}

impl<C: flow::Cc> Mark for flow::AnyJump<C> {
    fn mark<G: Guard>(self, at: usize, guard: G, marks: &mut Marks) -> Result<Ending, Unproven> {
        // Its only effect is a branch (Ryujinx InstEmitFlowControl.cs:
        // 38-40). Preserve the stack on every supplied slot, matching
        // the address dispatcher. This overapproximation cannot
        // prove a pop when an arbitrary jump may bypass its push.
        let len = marks.starts.len() - 1;
        marks.starts[..len].fill(true);
        marks.ends(at);
        Ok(ending(
            at,
            guard,
            marks,
            self.cc(),
            (0..len).collect(),
            Vec::new(),
            Vec::new(),
        ))
    }
}

impl<P: PushOp> Mark for flow::Push<P> {
    fn mark<G: Guard>(self, at: usize, guard: G, marks: &mut Marks) -> Result<Ending, Unproven> {
        marks.push(at, self.stack(), self.target()).map(|()| {
            ending(
                at,
                guard,
                marks,
                Pushes(()),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
        })
    }
}

impl<P: PushOp> Mark for flow::ResolvedPush<P> {
    fn mark<G: Guard>(self, at: usize, guard: G, marks: &mut Marks) -> Result<Ending, Unproven> {
        marks.push(at, self.stack(), self.target()).map(|()| {
            ending(
                at,
                guard,
                marks,
                Pushes(()),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
        })
    }
}

macro_rules! falls {
    ($($t:ty),*) => {$(
        impl Mark for $t {
            fn mark<G: Guard>(self, at: usize, guard: G, marks: &mut Marks) -> Result<Ending, Unproven> {
                Ok(ending(at, guard, marks, Falls(()), Vec::new(), Vec::new(), Vec::new()))
            }
        }
    )*};
}
falls!(flow::Step, flow::PredicateWrite);

/// An outside target names no block; Unknown might hide a branch, push or
/// predicate write.
macro_rules! unproven {
    ($($t:ty),*) => {$(
        impl Mark for $t {
            fn mark<G: Guard>(self, _: usize, _: G, _: &mut Marks) -> Result<Ending, Unproven> {
                Err(Unproven)
            }
        }
    )*};
}
unproven!(
    flow::OutsideTable,
    flow::NoSlotPush,
    flow::Call,
    flow::NoSlotCall,
    flow::Unknown
);

impl<C: flow::Cc, Y: crate::flow::Gap> Mark for flow::NoSlotJump<C, Y> {
    fn mark<G: Guard>(self, _: usize, _: G, _: &mut Marks) -> Result<Ending, Unproven> {
        Err(Unproven)
    }
}

/// How the block an instruction ends would end: by position, the
/// instructions it reaches on the arriving stack and with its push on it,
/// the push it executes, the stack its pop searches, its waiting pop, and
/// what tells of its guard.
pub(crate) struct Ending {
    carries: Vec<usize>,
    stacks: Vec<usize>,
    pushes: Option<usize>,
    searches: Vec<Stack>,
    lowers: Vec<usize>,
    report: fn(&Instruction),
}

impl Ending {
    /// The block `first..=last` this ending ends; `owner` names the block
    /// of each instruction, and the program's end names none.
    fn close(self, first: usize, last: usize, x: &Instruction, owner: &[usize]) -> Block {
        (self.report)(x);
        let blocks = |at: Vec<usize>| {
            at.into_iter()
                .flat_map(|at| owner.get(at).copied())
                .collect()
        };
        Block {
            first,
            last,
            carries: blocks(self.carries),
            stacks: blocks(self.stacks),
            pushes: self.pushes,
            searches: self.searches,
            lowers: self.lowers,
        }
    }
}

/// A guard that tells nothing of itself.
fn untold(_: &Instruction) {}

/// How a guard ends its block. Predication: Ryujinx Translator.cs:339-385;
/// Mesa sm50.rs:217-235. Both possibilities of a non-PT predicate join;
/// never assume two guarded edges carry the same value after a register
/// write. Only PT is constant in the block overapproximation; that local
/// edge exclusion is logged even when every pop is resolved.
pub(crate) trait BlockGuard {
    fn block<F: Fall>(self, ends: Ends<F>) -> Ending;
}

impl BlockGuard for crate::guard::Unguarded {
    fn block<F: Fall>(self, ends: Ends<F>) -> Ending {
        let (falls, pushed) = ends.fall.unguarded(ends.next());
        let carries = falls.into_iter().chain(ends.executed()).collect();
        ends.executes(carries, pushed.into_iter().collect(), untold)
    }
}

impl BlockGuard for crate::guard::When {
    /// A failed guard skips to the next block on the arriving stack.
    fn block<F: Fall>(self, ends: Ends<F>) -> Ending {
        let next = ends.next();
        let carries = once(next).chain(ends.executed()).collect();
        let stacks = ends.fall.skipped(next).into_iter().collect();
        ends.executes(carries, stacks, untold)
    }
}

impl BlockGuard for crate::guard::Pt {
    /// PT never fails: the next block follows only where the instruction falls.
    fn block<F: Fall>(self, ends: Ends<F>) -> Ending {
        let next = ends.next();
        let carries = ends
            .fall
            .falls(next)
            .into_iter()
            .chain(ends.executed())
            .collect();
        let stacks = ends.fall.skipped(next).into_iter().collect();
        let pruned = ends.pruned;
        ends.executes(carries, stacks, pruned)
    }
}

impl BlockGuard for crate::guard::NotPt {
    /// !PT never executes: only the next block follows, on the arriving stack.
    fn block<F: Fall>(self, ends: Ends<F>) -> Ending {
        Ending {
            carries: vec![ends.next()],
            stacks: Vec::new(),
            pushes: None,
            searches: Vec::new(),
            lowers: ends.lowers,
            report: ends.pruned,
        }
    }
}

pub(crate) struct Block {
    first: usize,
    last: usize,
    /// Blocks it reaches on the stack it was reached with.
    carries: Vec<usize>,
    /// Blocks it reaches with its own push on that stack.
    stacks: Vec<usize>,
    /// The push it executes on the stack it was reached with.
    pushes: Option<usize>,
    /// The stack its executed pop searches.
    searches: Vec<Stack>,
    /// Its unresolved pop.
    lowers: Vec<usize>,
}

struct Graph {
    blocks: Vec<Block>,
    /// Each push instruction, by position.
    pushes: HashMap<usize, Push>,
    /// How many unresolved pops the program has.
    pops: usize,
}

impl Graph {
    fn build<R: Run>(
        program: &Program,
        xs: &[Instruction],
        slots: &Slots,
    ) -> Result<Self, Unproven> {
        holds(!xs.is_empty()).and_then(|()| {
            // Targets and BRX entries already run slot 0 for a control word.
            // Ryujinx Decoder.cs:265-271; Mesa sm50.rs:3418-3428.
            let mut marks = Marks {
                slots,
                pruned: R::pruned,
                starts: vec![false; xs.len() + 1],
                pushed: Vec::new(),
            };
            marks.starts[0] = true;
            marks.starts[xs.len()] = true;
            program
                .line()
                .things()
                .map(|thing| thing.mark(&mut marks))
                .collect::<Result<Vec<_>, _>>()
                .map(|endings| marks.graph(xs, endings, program.pops()))
        })
    }

    /// The pops a finished proof reached one destination with, and the
    /// pushes they consumed.
    fn resolution(self, targets: &[HashSet<usize>], visited: &[HashSet<usize>]) -> Resolution {
        let firsts = self.blocks.iter().map(|b| b.first).collect::<Vec<_>>();
        let mut reached = HashMap::default();
        let mut consumed = HashSet::default();
        for ((b, destinations), visited) in self.blocks.into_iter().zip(targets).zip(visited) {
            // A single destination proves this block's pop.
            for &target in masked(destinations, destinations.len() == 1) {
                // Use block start, not its terminator, for the actual byte target.
                reached.extend(b.lowers.iter().map(|&pop| (pop, firsts[target])));
                consumed.extend(visited);
            }
        }
        Resolution {
            covers_pops: reached.len() == self.pops,
            reached: Reached(reached),
            consumed: Consumed(consumed),
        }
    }
}

/// The stack a pop reads, carried by its facts.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Stack(TargetStack);

impl std::hash::Hash for Stack {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(&self.0).hash(state);
    }
}

impl Push {
    /// A pop searching `stack` through this push, whose next head is `head`,
    /// either arrives at this push's target or keeps searching.
    fn transfer(
        self,
        work: &mut Work,
        block: usize,
        head: usize,
        stack: Stack,
    ) -> Result<(), Unproven> {
        [Self::passes, Self::arrives][usize::from(self.stack == stack.0)](
            self, work, block, head, stack,
        )
    }

    fn arrives(self, work: &mut Work, _: usize, head: usize, _: Stack) -> Result<(), Unproven> {
        work.add(Reach {
            block: self.block,
            head,
        })
    }

    fn passes(
        self,
        work: &mut Work,
        block: usize,
        head: usize,
        stack: Stack,
    ) -> Result<(), Unproven> {
        work.add(Pop { block, head, stack })
    }
}

/// A block reached with a stack whose head is `head`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Reach {
    block: usize,
    head: usize,
}

/// A push executed on a stack whose head is `head`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Parent {
    producer: usize,
    head: usize,
}

/// A pop in `block` searching `stack` from the head `head`.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Pop {
    block: usize,
    head: usize,
    stack: Stack,
}

struct Queue<F> {
    seen: HashSet<F>,
    pending: VecDeque<F>,
}

impl<F: Copy + Eq + Hash> Queue<F> {
    fn new() -> Self {
        Self {
            seen: HashSet::default(),
            pending: VecDeque::new(),
        }
    }

    fn add(&mut self, fact: F) {
        self.pending
            .extend(masked(once(fact), self.seen.insert(fact)));
    }
}

/// A fact of the proof: where it waits and what it adds.
trait Fact: Copy + Eq + Hash {
    fn queue(work: &mut Work) -> &mut Queue<Self>;
    fn follow(self, proof: &mut Proof<'_>) -> Result<(), Unproven>;
}

impl Fact for Reach {
    fn queue(work: &mut Work) -> &mut Queue<Self> {
        &mut work.reaches
    }

    fn follow(self, proof: &mut Proof<'_>) -> Result<(), Unproven> {
        let Reach { block, head } = self;
        let b = &proof.graph.blocks[block];
        let work = &mut proof.work;
        b.carries
            .iter()
            .try_for_each(|&to| work.add(Reach { block: to, head }))
            .and_then(|()| {
                b.stacks.iter().try_for_each(|&to| {
                    work.add(Reach {
                        block: to,
                        head: b.last,
                    })
                })
            })
            .and_then(|()| {
                b.pushes
                    .iter()
                    .try_for_each(|&producer| work.add(Parent { producer, head }))
            })
            .and_then(|()| {
                b.searches
                    .iter()
                    .try_for_each(|&stack| work.add(Pop { block, head, stack }))
            })
    }
}

impl Fact for Parent {
    fn queue(work: &mut Work) -> &mut Queue<Self> {
        &mut work.parents
    }

    fn follow(self, proof: &mut Proof<'_>) -> Result<(), Unproven> {
        let Parent { producer, head } = self;
        proof.parents[producer].insert(head);
        let push = proof.graph.pushes[&producer];
        let work = &mut proof.work;
        proof.listeners[producer]
            .iter()
            .try_for_each(|&(block, stack)| push.transfer(work, block, head, stack))
    }
}

impl Fact for Pop {
    fn queue(work: &mut Work) -> &mut Queue<Self> {
        &mut work.pops
    }

    fn follow(self, proof: &mut Proof<'_>) -> Result<(), Unproven> {
        let Pop { block, head, stack } = self;
        // Empty-stack searches have no successor (Ryujinx
        // Decoder.cs:832-835), but that is not a proof that the
        // history cannot execute on hardware. This overapproximation
        // must retain every possible underflow.
        carried(proof.graph.pushes.get(&head).copied()).and_then(|push| {
            proof.visited[block].insert(head);
            proof.listeners[head].push((block, stack));
            proof.targets[block].extend(masked(once(push.block), push.stack == stack.0));
            let work = &mut proof.work;
            proof.parents[head]
                .iter()
                .try_for_each(|&parent| push.transfer(work, block, parent, stack))
        })
    }
}

struct Work {
    remaining: usize,
    reaches: Queue<Reach>,
    parents: Queue<Parent>,
    pops: Queue<Pop>,
}

impl Work {
    fn add<F: Fact>(&mut self, fact: F) -> Result<(), Unproven> {
        // Charge attempted additions too, bounding repeated scans of parent
        // sets and BRX edges. Exhaustion publishes no incomplete proof.
        carried(self.remaining.checked_sub(1)).map(|remaining| {
            self.remaining = remaining;
            F::queue(self).add(fact);
        })
    }
}

/// Why the fixed point stopped.
#[derive(Clone, Copy)]
enum Stop {
    /// Every fact was followed.
    Drained,
    Unproven,
}

impl From<Unproven> for Stop {
    fn from(Unproven: Unproven) -> Self {
        Self::Unproven
    }
}

impl Stop {
    fn settle(self) -> Result<(), Unproven> {
        [Ok(()), Err(Unproven)][self as usize]
    }
}

struct Proof<'g> {
    graph: &'g Graph,
    work: Work,
    /// The heads each push was executed on.
    parents: Vec<HashSet<usize>>,
    /// The pops searching through each push.
    listeners: Vec<Vec<(usize, Stack)>>,
    /// The pushes each block's pop searched through.
    visited: Vec<HashSet<usize>>,
    /// The targets each block's pop reaches.
    targets: Vec<HashSet<usize>>,
}

impl<'g> Proof<'g> {
    fn new(graph: &'g Graph, len: usize, work_per_block: usize) -> Self {
        Self {
            graph,
            work: Work {
                remaining: graph.blocks.len().saturating_mul(work_per_block),
                reaches: Queue::new(),
                parents: Queue::new(),
                pops: Queue::new(),
            },
            parents: vec![HashSet::default(); len],
            listeners: vec![Vec::new(); len],
            visited: vec![HashSet::default(); graph.blocks.len()],
            targets: vec![HashSet::default(); graph.blocks.len()],
        }
    }

    /// Follows every fact from the entry block on the empty stack.
    fn run(mut self) -> Result<Self, Unproven> {
        self.work
            .add(Reach {
                block: 0,
                head: EMPTY,
            })
            .and_then(|()| {
                repeat(())
                    .try_for_each(|()| self.step())
                    .or_else(Stop::settle)
            })
            .map(|()| self)
    }

    /// Follows the next waiting fact of each kind.
    fn step(&mut self) -> Result<(), Stop> {
        let work = &self.work;
        let waiting =
            work.reaches.pending.len() + work.parents.pending.len() + work.pops.pending.len();
        [Err(Stop::Drained), Ok(())][waiting.min(1)]
            .and_then(|()| self.next::<Reach>())
            .and_then(|()| self.next::<Parent>())
            .and_then(|()| self.next::<Pop>())
    }

    fn next<F: Fact>(&mut self) -> Result<(), Stop> {
        let fact = F::queue(&mut self.work).pending.pop_front();
        fact.into_iter()
            .try_for_each(|fact| fact.follow(self))
            .map_err(Stop::from)
    }
}

/// Heads identify push instructions; each push owns the union of its possible
/// parents. This is an overapproximation of concrete stacks: sharing a parent
/// can add paths, but cannot remove a real path or an underflow. A pop searches
/// through intervening kinds, then propagates the matching push's parents to
/// its target (Ryujinx Decoder.cs:824-869). Subscriptions replay that transfer
/// whenever a loop or a joining path supplies a new parent.
///
/// Fixed-point work and memory are capped at 256 times the block count. This
/// intentionally leaves difficult programs unresolved rather than accepting
/// the first stack seen at a join or enumerating exponentially many paths.
#[cfg(test)]
pub(crate) fn resolve(xs: &[Instruction]) -> Option<Resolution> {
    resolve_in(crate::run::QuietRun, &Program::of(xs), xs, &Slots::of(xs))
}

/// [`resolve`] over the program `xs` arrived as and the slots that place it,
/// on `run`'s track.
pub(crate) fn resolve_in<R: Run>(
    _: R,
    program: &Program,
    xs: &[Instruction],
    slots: &Slots,
) -> Option<Resolution> {
    prove::<R>(program, xs, slots, 256).ok()
}

#[cfg(test)]
fn resolve_with_limit(xs: &[Instruction], work_per_block: usize) -> Result<Resolution, Unproven> {
    prove::<crate::run::QuietRun>(&Program::of(xs), xs, &Slots::of(xs), work_per_block)
}

fn prove<R: Run>(
    program: &Program,
    xs: &[Instruction],
    slots: &Slots,
    work_per_block: usize,
) -> Result<Resolution, Unproven> {
    let graph = Graph::build::<R>(program, xs, slots)?;
    let proof = Proof::new(&graph, xs.len(), work_per_block).run()?;
    let (targets, visited) = (proof.targets, proof.visited);
    Ok(graph.resolution(&targets, &visited))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ops, InstructionKind, Predicate, ResolvedPush, Translation};

    fn instruction(at: usize, kind: InstructionKind) -> Instruction {
        Instruction {
            address: at as u32 * 8,
            predicate: Default::default(),
            kind,
        }
    }

    fn push(kind: TargetStack, at: usize, to: usize) -> InstructionKind {
        let target = BranchTarget::reaching(at as u32 * 8, to as u32 * 8);
        [
            InstructionKind::PushTarget(ops::PushTarget {
                target,
                stack: kind,
            }),
            InstructionKind::Schedule(ops::Schedule { target }),
        ][usize::from(kind == TargetStack::Sync)]
    }

    fn pop(kind: TargetStack, _: usize, _: usize) -> InstructionKind {
        InstructionKind::PopJump(ops::PopJump {
            cc_test: 0xf,
            stack: kind,
        })
    }

    fn branch(at: usize, to: usize) -> InstructionKind {
        InstructionKind::Branch(ops::Branch {
            target: BranchTarget::reaching(at as u32 * 8, to as u32 * 8),
            cc_test: 0xf,
        })
    }

    fn jump(_: TargetStack, at: usize, to: usize) -> InstructionKind {
        branch(at, to)
    }

    fn nop(_: TargetStack, _: usize, _: usize) -> InstructionKind {
        InstructionKind::Nop(ops::Nop)
    }

    /// One for an unresolved stack instruction.
    struct Unresolved;
    impl Translation for Unresolved {
        type Out = usize;
        fn unknown(self) -> usize {
            0
        }
        fn padding(self) -> usize {
            0
        }
        fn unbuilt(self) -> usize {
            0
        }
        fn unwanted(self) -> usize {
            0
        }
        fn unresolved_stack(self) -> usize {
            1
        }
        fn unproven_stream(self) -> usize {
            0
        }
        fn known(self) -> usize {
            0
        }
    }

    /// Whether resolution completed, without its contents.
    fn proven(xs: &[Instruction], work_per_block: usize) -> Result<(), Unproven> {
        resolve_with_limit(xs, work_per_block).map(|_| ())
    }

    #[test]
    fn resolved_pop_consumes_the_stack_before_an_unresolved_pop() {
        // Ryujinx Decoder.cs:824-850: a resolved SYNC still pops its SSY.
        // Re-running a proof must not let a second SYNC reuse that token.
        let xs = [
            instruction(
                0,
                InstructionKind::ResolvedSchedule(ops::ResolvedSchedule {
                    push: ResolvedPush::checked(BranchTarget::reaching(0, 16)),
                }),
            ),
            instruction(
                1,
                InstructionKind::Reconverge(ops::Reconverge {
                    target: BranchTarget::reaching(8, 16),
                    cc_test: 15,
                    stack: TargetStack::Sync,
                }),
            ),
            instruction(
                2,
                InstructionKind::PopJump(ops::PopJump {
                    cc_test: 15,
                    stack: TargetStack::Sync,
                }),
            ),
            instruction(3, InstructionKind::Exit(ops::Exit { cc_test: 15 })),
        ];
        assert_eq!(proven(&xs, 256), Err(Unproven));
    }

    #[test]
    fn exponentially_many_joining_stacks_share_parent_facts() {
        // 512 diamonds admit 2^512 producer histories. The abstract proof
        // needs only two heads per join and two parents per push.
        let levels = 512;
        let pop = 1 + levels * 5;
        let exit = pop + 1;
        let mut xs = vec![instruction(0, push(TargetStack::Break, 0, exit))];
        for level in 0..levels {
            let at = 1 + level * 5;
            let mut test = instruction(at, branch(at, at + 3));
            test.predicate = Predicate {
                register: 0,
                inverted: false,
            }
            .into();
            xs.push(test);
            xs.push(instruction(at + 1, push(TargetStack::Sync, at + 1, exit)));
            xs.push(instruction(at + 2, branch(at + 2, at + 5)));
            xs.push(instruction(
                at + 3,
                push(TargetStack::Continue, at + 3, exit),
            ));
            xs.push(instruction(at + 4, branch(at + 4, at + 5)));
        }
        xs.push(instruction(
            pop,
            InstructionKind::PopJump(ops::PopJump {
                cc_test: 0xf,
                stack: TargetStack::Break,
            }),
        ));
        xs.push(instruction(
            exit,
            InstructionKind::Exit(ops::Exit { cc_test: 15 }),
        ));
        assert_eq!(
            proven(&xs, 1),
            Err(Unproven),
            "an exhausted proof must not publish results"
        );
        let lowered = resolve_with_limit(&xs, 256).map(|resolution| {
            let mut program = Program::of(&xs);
            resolution.apply(&mut program, &xs);
            program.write(&mut xs);
            let unresolved = xs
                .iter()
                .map(|x| x.kind.translation(Unresolved))
                .sum::<usize>();
            (xs[pop].kind, unresolved)
        });
        assert_eq!(
            lowered.map_err(|Unproven| "bounded shared-parent proof"),
            Ok((
                InstructionKind::Reconverge(ops::Reconverge {
                    target: BranchTarget::reaching(pop as u32 * 8, exit as u32 * 8),
                    cc_test: 15,
                    stack: TargetStack::Break,
                }),
                0
            ))
        );
    }

    /// What the generator placed at a slot, as the concrete oracle runs it.
    #[derive(Clone, Copy)]
    struct Slot {
        /// Push, pop, branch, no-op or exit.
        class: usize,
        stack: TargetStack,
        target: usize,
        predicated: bool,
    }

    /// A concrete execution point: the slot and the producers on its stack.
    type State = (usize, Vec<usize>);

    /// Independent ordered-stack execution of the generated program, checked
    /// against the lowered one.
    struct Oracle<'a> {
        slots: &'a [Slot],
        original: &'a [Instruction],
        lowered: &'a [Instruction],
    }

    type Step<'a> = fn(&Oracle<'a>, usize, Vec<usize>, &mut usize) -> Vec<State>;

    impl<'a> Oracle<'a> {
        const STEP: [Step<'a>; 5] = [
            Self::pushes,
            Self::pops,
            Self::branches,
            Self::continues,
            Self::exits,
        ];

        /// Every state one instruction later. A predicated instruction may
        /// also be skipped on every visit.
        fn step(&self, (at, stack): State, checked: &mut usize) -> Vec<State> {
            let slot = self.slots[at];
            masked(once((at + 1, stack.clone())), slot.predicated)
                .chain(Self::STEP[slot.class](self, at, stack, checked))
                .collect()
        }

        fn pushes(&self, at: usize, mut stack: Vec<usize>, _: &mut usize) -> Vec<State> {
            stack.push(at);
            vec![(at + 1, stack)]
        }

        fn pops(&self, at: usize, mut stack: Vec<usize>, checked: &mut usize) -> Vec<State> {
            let kind = self.slots[at].stack;
            let matching = stack
                .iter()
                .enumerate()
                .flat_map(|(position, &producer)| {
                    masked(once(position), self.slots[producer].stack == kind)
                })
                .collect::<Vec<_>>();
            // An empty search must leave the pop unproven.
            masked(once(()), matching.is_empty())
                .for_each(|()| assert_eq!(self.lowered[at].kind, self.original[at].kind));
            matching
                .last()
                .into_iter()
                .map(|&position| {
                    let to = self.slots[stack[position]].target;
                    let mut rest = std::mem::take(&mut stack);
                    rest.truncate(position);
                    // An observed pop challenges any target the proof accepted.
                    let reached = InstructionKind::Reconverge(ops::Reconverge {
                        target: BranchTarget::reaching(at as u32 * 8, to as u32 * 8),
                        cc_test: 0xf,
                        stack: kind,
                    });
                    let observed = self.lowered[at].kind;
                    *checked += usize::from(observed == reached);
                    assert_eq!(
                        usize::from(observed == reached)
                            + usize::from(observed == self.original[at].kind),
                        1,
                        "slot {at} lowered to {observed:?}, but the stack reached slot {to}"
                    );
                    (to, rest)
                })
                .collect()
        }

        fn branches(&self, at: usize, stack: Vec<usize>, _: &mut usize) -> Vec<State> {
            vec![(self.slots[at].target, stack)]
        }

        fn continues(&self, at: usize, stack: Vec<usize>, _: &mut usize) -> Vec<State> {
            vec![(at + 1, stack)]
        }

        fn exits(&self, _: usize, _: Vec<usize>, _: &mut usize) -> Vec<State> {
            Vec::new()
        }
    }

    #[test]
    fn abstract_proofs_agree_with_concrete_ordered_stack_executions() {
        // Independent bounded execution of project-generated CFGs. Branch
        // predicates may independently choose either edge on every visit.
        // No conclusion is inferred from the oracle's depth cutoff: each
        // observed pop only challenges a target the abstract proof accepted.
        let mut seed = 0x04d4_dbc2_u64;
        let mut next = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed as usize
        };
        let kinds = [TargetStack::Sync, TargetStack::Break, TargetStack::Continue];
        // Choices 0-2 push, 3-5 pop, 6-7 branch (7 predicated), 8 is a no-op.
        let classes = [0, 0, 0, 1, 1, 1, 2, 2, 3];
        let make: [fn(TargetStack, usize, usize) -> InstructionKind; 4] = [push, pop, jump, nop];
        let mut checked = 0;
        for _ in 0..512 {
            let mut slots = Vec::new();
            let mut xs = Vec::new();
            for at in 0..9 {
                let choice = next() % 9;
                let target = next() % 10;
                let slot = Slot {
                    class: classes[choice],
                    stack: kinds[choice % 3],
                    target,
                    predicated: choice == 7,
                };
                let mut x = instruction(at, make[slot.class](slot.stack, at, target));
                x.predicate = slot.predicated.then_some(Predicate {
                    register: 0,
                    inverted: false,
                });
                slots.push(slot);
                xs.push(x);
            }
            slots.push(Slot {
                class: 4,
                stack: TargetStack::Sync,
                target: 9,
                predicated: false,
            });
            xs.push(instruction(
                9,
                InstructionKind::Exit(ops::Exit { cc_test: 15 }),
            ));
            let original = xs.clone();
            resolve_with_limit(&xs, 256).into_iter().for_each(|proof| {
                let mut program = Program::of(&xs);
                proof.apply(&mut program, &xs);
                program.write(&mut xs);
                let oracle = Oracle {
                    slots: &slots,
                    original: &original,
                    lowered: &xs,
                };
                (0..12).fold(vec![(0, Vec::new())], |frontier, _| {
                    frontier
                        .into_iter()
                        .flat_map(|state| oracle.step(state, &mut checked))
                        .collect()
                });
            });
        }
        assert_ne!(
            checked, 0,
            "the concrete oracle must exercise a lowered pop"
        );
    }
}
