//! Equal-source reaching definitions for bindless handles and constants.
//!
//! Ryujinx Translation/Ssa.cs:191-200,297-318 joins definitions at dominance
//! frontiers and gives phi nodes one source per predecessor. Its
//! Translation/Optimizations/Utils.cs:115-150 uses matching branch conditions
//! to select a phi source; BindlessElimination.cs:181-218 traces the handle
//! and its OR operands. Here equal-source joins compute the needed result
//! directly, without building SSA operations or emitting extra SPIR-V.
//!
//! This is an independent implementation, not a port of those routines.
use crate::emitter::{HandleLeaf, HandleSource};
use crate::fast_hash::{HashMap, HashSet};
use crate::ir::ops;
use crate::ir::{
    BranchTarget, DetachedBounds, Instruction, InstructionKind, LogicOp, Predicate, Source, Written,
};
use crate::structure::{ControlFlowGraph, Guarded};

/// A traced leaf: an immediate word, or a cbuf bank and byte offset
/// packed as `bank << 16 | offset`.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Leaf {
    buffer: bool,
    word: u32,
}

impl Leaf {
    const fn word(word: u32) -> Self {
        Self {
            buffer: false,
            word,
        }
    }

    // Mesa src/nouveau/compiler/nak/sm50.rs:361-368 defines the cbuf bank
    // and four-byte word offset. The decoder has already converted to bytes.
    fn buffer(bank: u8, offset: u16) -> Self {
        Self {
            buffer: true,
            word: u32::from(bank) << 16 | u32::from(offset),
        }
    }

    /// The emitter's name for this leaf.
    fn handle(self) -> HandleLeaf {
        [
            HandleLeaf::Constant(self.word),
            HandleLeaf::ConstantBuffer {
                bank: (self.word >> 16) as u8,
                offset: self.word as u16,
            },
        ][usize::from(self.buffer)]
    }
}

/// How much of a register's definition is known.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum Shape {
    /// Undefined, conflicting or unsupported.
    Unknown,
    Leaf,
    Combined,
    /// A packed texture handle loaded indirectly from a constant buffer.
    Array,
}

/// What reaches a register: nothing known, one leaf, or two leaves
/// combined by a logic op. Unused fields stay zero, so equal facts are
/// equal values.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Fact {
    shape: Shape,
    op: LogicOp,
    a: Leaf,
    b: Leaf,
}

impl Fact {
    const UNKNOWN: Self = Self {
        shape: Shape::Unknown,
        op: LogicOp::And,
        a: Leaf::word(0),
        b: Leaf::word(0),
    };

    const fn leaf(a: Leaf) -> Self {
        Self {
            shape: Shape::Leaf,
            a,
            ..Self::UNKNOWN
        }
    }

    const fn constant(value: u32) -> Self {
        Self::leaf(Leaf::word(value))
    }

    // Ryujinx ConstantFolding.cs:27-40. Folding here avoids retaining a
    // host-side expression for literal offsets.
    fn combine(op: LogicOp, a: Leaf, b: Leaf) -> Self {
        let folded = Self::constant(
            [a.word & b.word, a.word | b.word, a.word ^ b.word, b.word][op as usize],
        );
        let combined = Self {
            shape: Shape::Combined,
            op,
            a,
            b,
        };
        pick(a.buffer | b.buffer, folded, combined)
    }

    fn known(self) -> bool {
        self.shape != Shape::Unknown
    }

    fn is_leaf(self) -> bool {
        self.shape == Shape::Leaf
    }

    fn is_constant(self) -> bool {
        self.is_leaf() & !self.a.buffer
    }

    /// The literal complement where inverted; an inverted non-literal is
    /// not traced.
    fn inverted(self, invert: bool) -> Self {
        let complement = pick(
            self.is_constant(),
            Self::UNKNOWN,
            Self::constant(!self.a.word),
        );
        pick(invert, self, complement)
    }

    /// The emitter's name for this fact, at the outgoing edge.
    fn handle(self) -> Option<HandleSource> {
        HANDLE[self.shape as usize](self)
    }

    fn array(binding: u8) -> Self {
        Self {
            shape: Shape::Array,
            a: Leaf::word(u32::from(binding)),
            ..Self::UNKNOWN
        }
    }

    fn copied(self) -> Self {
        pick(self.shape == Shape::Array, self, Self::UNKNOWN)
    }
}

const HANDLE: [fn(Fact) -> Option<HandleSource>; 4] = [
    |_| None,
    |f| Some(HandleSource::Leaf(f.a.handle())),
    |f| {
        Some(HandleSource::Combined {
            op: f.op,
            a: f.a.handle(),
            b: f.b.handle(),
        })
    },
    |f| {
        Some(HandleSource::ConstantBufferArray {
            bank: f.a.word as u8,
            handle: 0,
        })
    },
];

type Registers = [Fact; 256];

/// RZ reads zero; every other register starts undefined.
const UNKNOWN: Registers = {
    let mut registers = [Fact::UNKNOWN; 256];
    registers[255] = Fact::constant(0);
    registers
};

/// One guard slice: the registers that reach it, or no path at all. An
/// unreached slice's registers mean nothing and are never read.
#[derive(Clone, Copy)]
struct Slice {
    reached: bool,
    registers: Registers,
}

const ENTRY: Slice = Slice {
    reached: true,
    registers: UNKNOWN,
};

const UNREACHED: Slice = Slice {
    reached: false,
    registers: UNKNOWN,
};

// Slice zero has no condition. Other slices assume one predicate value.
type State = Vec<Slice>;

/// Guard keys: 0 is no condition, `1 + 2 * register + inverted` one
/// predicate value. `KEYS` itself names the slice past every slice.
const KEYS: usize = 513;

/// Each key's opposite predicate value. No condition has none.
const OPPOSITE: [usize; KEYS] = {
    let mut table = [KEYS; KEYS];
    let mut key = 1;
    while key < KEYS {
        table[key] = ((key - 1) ^ 1) + 1;
        key += 1;
    }
    table
};

fn key(guard: impl IntoIterator<Item = Predicate>) -> usize {
    guard
        .into_iter()
        .map(|p| 1 + usize::from(p.register) * 2 + usize::from(p.inverted))
        .sum()
}

fn pick<T: Copy>(c: bool, no: T, yes: T) -> T {
    [no, yes][usize::from(c)]
}

// Both values of every guarding predicate have a slice, in register order.
// Slice order changes storage only; predicate facts still follow Ryujinx
// Utils.cs:115-150.
struct Guards {
    keys: Vec<usize>,
    slices: [usize; KEYS + 1],
}
impl Default for Guards {
    fn default() -> Self {
        Self {
            keys: vec![0],
            slices: [0; KEYS + 1],
        }
    }
}
impl Guards {
    fn build(&mut self, xs: &[Instruction]) {
        let mut guarding = [false; 256];
        for p in xs.iter().flat_map(|x| x.predicate) {
            guarding[usize::from(p.register)] = true;
        }
        self.keys.clear();
        self.keys.push(0);
        self.slices.fill(0);
        for (register, guarding) in guarding.into_iter().enumerate() {
            let guarding = usize::from(guarding);
            let first = 1 + register * 2;
            let slice = self.keys.len();
            self.slices[first] = slice * guarding;
            self.slices[first + 1] = (slice + 1) * guarding;
            self.keys
                .extend([first, first + 1].into_iter().take(2 * guarding));
        }
        self.slices[KEYS] = self.keys.len();
    }
    fn slice(&self, guard: impl IntoIterator<Item = Predicate>) -> usize {
        self.slices[key(guard)]
    }
    /// The slice of the opposite value, or one past every slice.
    fn opposite(&self, guard: impl IntoIterator<Item = Predicate>) -> usize {
        self.slices[OPPOSITE[key(guard)]]
    }
}

#[derive(Default)]
pub(crate) struct ReachingDefinitions {
    // Each instruction's snapshot in `facts`; entry 0 knows nothing.
    uses: Vec<usize>,
    // One arena holds all use snapshots, including detached-code proofs.
    facts: Vec<Registers>,
}

#[derive(Default)]
pub(crate) struct Scratch {
    guards: Guards,
    incoming: State,
    pending: Vec<usize>,
    round: Vec<usize>,
    queued: Vec<bool>,
    state: State,
    edge_state: State,
    detached: Vec<usize>,
}

fn reset<T: Clone>(buffer: &mut Vec<T>, len: usize, value: T) {
    buffer.clear();
    buffer.resize(len, value);
}

type Flow = fn(&mut ReachingDefinitions, &[Instruction], &ControlFlowGraph, &mut Scratch, bool);

/// A shader without handle uses has nothing to trace.
const FLOW: [Flow; 2] = [|_, _, _, _, _| {}, flow];

impl ReachingDefinitions {
    pub fn analyze(xs: &[Instruction], cfg: &ControlFlowGraph) -> Self {
        let mut definitions = Self::default();
        definitions.analyze_into(xs, cfg, &mut Scratch::default(), false);
        definitions
    }

    /// Flat-address translation does not need cbuf address tracing, so
    /// global memory is a use only with `include_global`.
    pub fn analyze_into(
        &mut self,
        xs: &[Instruction],
        cfg: &ControlFlowGraph,
        scratch: &mut Scratch,
        include_global: bool,
    ) {
        reset(&mut self.uses, xs.len(), 0);
        self.facts.clear();
        self.facts.push(UNKNOWN);
        let uses = xs
            .iter()
            .map(|x| usize::from(x.kind.handle_use(include_global)))
            .sum::<usize>();
        FLOW[uses.min(1)](self, xs, cfg, scratch, include_global);
    }

    pub fn source(&self, index: usize, source: Source) -> Option<HandleSource> {
        read(source, &self.facts[self.uses[index]]).handle()
    }

    pub fn constant(&self, index: usize, register: u8) -> Option<u32> {
        let fact = self.facts[self.uses[index]][usize::from(register)];
        [None, Some(fact.a.word)][usize::from(fact.is_constant())]
    }
}

fn flow(
    definitions: &mut ReachingDefinitions,
    xs: &[Instruction],
    cfg: &ControlFlowGraph,
    scratch: &mut Scratch,
    include_global: bool,
) {
    let ReachingDefinitions { uses, facts } = definitions;
    let Scratch {
        guards,
        incoming,
        pending,
        round,
        queued,
        state,
        edge_state,
        detached,
    } = scratch;
    guards.build(xs);
    let slices = guards.keys.len();
    // One dense block/slice arena replaces one allocation per reached block.
    // Slice order and equal-source joins stay as in Ryujinx Ssa.cs:191-200,
    // :297-318 and Optimizations/Utils.cs:115-150. A block no path has
    // reached holds no slice, so its first incoming state joins as a copy.
    reset(incoming, cfg.blocks.len() * slices, UNREACHED);
    incoming[..slices].fill(ENTRY);
    pending.clear();
    pending.push(0);
    reset(queued, cfg.blocks.len(), false);
    queued[0] = true;
    reset(state, slices, UNREACHED);
    reset(edge_state, slices, UNREACHED);
    // Rounds visit blocks in the order a first-in first-out queue would.
    while !pending.is_empty() {
        std::mem::swap(pending, round);
        for block in round.drain(..) {
            queued[block] = false;
            state.copy_from_slice(&incoming[block * slices..(block + 1) * slices]);
            for x in &xs[cfg.blocks[block].instructions.clone()] {
                transfer(state, guards, *x);
            }
            let block = &cfg.blocks[block];
            // An edge taken under its guard assumes it; an edge taken
            // always carries the state as it is.
            for &Guarded { block: next, guard } in cfg.guarded(block) {
                edge_state.copy_from_slice(state);
                restrict(edge_state, guards, guard);
                arrive(next, edge_state, incoming, slices, pending, queued);
            }
            for &next in cfg.always(block) {
                arrive(next, state, incoming, slices, pending, queued);
            }
        }
    }
    // Replay each block once after convergence. Keep input facts only
    // at texture/surface uses, including texture offset registers. A use
    // no path reaches takes its facts from detached code below.
    detached.clear();
    for (index, block) in cfg.blocks.iter().enumerate() {
        state.copy_from_slice(&incoming[index * slices..(index + 1) * slices]);
        for i in block.instructions.clone() {
            let x = xs[i];
            let snapshot = &state[guards.slice(x.predicate)];
            RECORD[usize::from(x.kind.handle_use(include_global))](
                i, snapshot, uses, facts, detached,
            );
            transfer(state, guards, x);
        }
    }
    DETACH[detached.len().min(1)](xs, guards, detached, uses, facts, state);
}

/// `edge` arrives at block `next`: join it, and queue `next` when it changed.
fn arrive(
    next: usize,
    edge: &[Slice],
    incoming: &mut [Slice],
    slices: usize,
    pending: &mut Vec<usize>,
    queued: &mut [bool],
) {
    // A block whose input changed waits once for the next round; a block
    // already waiting reads the joined input when its round comes.
    let changed = join(&mut incoming[next * slices..(next + 1) * slices], edge);
    if changed && !queued[next] {
        queued[next] = true;
        pending.push(next);
    }
}

type Record = fn(usize, &Slice, &mut [usize], &mut Vec<Registers>, &mut Vec<usize>);

/// A use keeps the facts that reach it; one no path reaches is detached.
const RECORD: [Record; 2] = [
    |_, _, _, _, _| {},
    |i, snapshot, uses, facts, detached| {
        let reached = usize::from(snapshot.reached);
        uses[i] = facts.len() * reached;
        facts.extend(std::iter::repeat_n(snapshot.registers, reached));
        detached.extend(std::iter::repeat_n(i, 1 - reached));
    },
];

type Detach = fn(&[Instruction], &Guards, &[usize], &mut [usize], &mut Vec<Registers>, &mut State);

const DETACH: [Detach; 2] = [
    |_, _, _, _, _, _| {},
    |xs, guards, detached, uses, facts, state| {
        DetachedDefinitions::new(xs, guards).record_uses(xs, guards, detached, uses, facts, state)
    },
];

type Restrict = fn(&mut Slice, &Slice);

/// A slice the restricted slice does not reach is unreached; a reached
/// one takes each register the restriction knows.
const RESTRICT: [Restrict; 2] = [
    |slice, _| slice.reached = false,
    |slice, restricted| {
        for (value, restricted) in slice.registers.iter_mut().zip(restricted.registers) {
            *value = pick(restricted.known(), *value, restricted);
        }
    },
];

// An edge taken only under `guard`: every slice assumes it, the slice
// without a condition becomes the guard's own, and the opposite value of
// the same predicate cannot reach the use. Keep that absence distinct
// from an unknown register. Ryujinx Utils.cs:115-150 selects phi sources
// by matching conditions rather than joining the mutually exclusive
// sources. An unreached guard slice carries nothing along the edge.
fn restrict(state: &mut [Slice], guards: &Guards, guard: Predicate) {
    let restricted = state[guards.slice([guard])];
    for slice in state.iter_mut() {
        RESTRICT[usize::from(slice.reached & restricted.reached)](slice, &restricted);
    }
    state[guards.opposite([guard])].reached = false;
    state[0] = restricted;
}

// The emitter visits disconnected code too. For those uses only, retain main's
// block-local and immutable-entry proofs (3251694 emitter.rs:4732-4839).
// These facts never flow into a live block. No guessed definition can affect
// an executed shader instruction.
struct DetachedDefinitions {
    begins: Vec<usize>,
    entry: State,
}

/// What one instruction tells detached code: where the entry run ends and
/// which blocks start.
struct Bounds<'a> {
    at: usize,
    next: &'a [Instruction],
    addresses: &'a HashMap<u32, usize>,
    end: usize,
    starts: HashSet<u32>,
}

impl DetachedBounds for Bounds<'_> {
    fn end_entry(&mut self) {
        self.end = self.end.min(self.at);
    }
    fn start(&mut self, slot: u32) {
        self.starts.insert(slot);
    }
    fn fall_through(&mut self) {
        self.starts
            .extend(self.next.first().map(|next| next.address));
    }
    fn end_at(&mut self, target: BranchTarget) {
        for &target in self.addresses.get(&target.slot()).into_iter() {
            self.end = self.end.min(target);
        }
    }
}

impl DetachedDefinitions {
    fn new(xs: &[Instruction], guards: &Guards) -> Self {
        let mut addresses = HashMap::default();
        for (i, x) in xs.iter().enumerate() {
            // Retain position()'s first occurrence for duplicate addresses.
            addresses.entry(x.address).or_insert(i);
        }
        let mut bounds = Bounds {
            at: 0,
            next: xs,
            addresses: &addresses,
            end: xs.len(),
            starts: HashSet::default(),
        };
        for (i, x) in xs.iter().enumerate() {
            bounds.at = i;
            bounds.next = &xs[i + 1..];
            x.kind.bound_detached(&mut bounds);
        }
        let Bounds { end, starts, .. } = bounds;
        let mut entry = vec![ENTRY; guards.keys.len()];
        for x in &xs[..end] {
            transfer(&mut entry, guards, *x);
        }
        // Entry facts are valid everywhere only for registers never written
        // outside entry. Predicated entry facts require the same predicate
        // version, so keep only unconditional facts across detached regions.
        let unconditional = entry[0];
        entry.fill(unconditional);
        for x in &xs[end..] {
            for Written { base: first, count } in x.kind.destinations() {
                for slice in entry.iter_mut() {
                    let start = usize::from(first.0);
                    let end = (start + usize::from(count)).min(255);
                    slice.registers[start..end].fill(Fact::UNKNOWN);
                }
            }
        }
        let mut begin = 0;
        let begins = xs
            .iter()
            .enumerate()
            .map(|(i, x)| {
                for _ in starts.get(&x.address).into_iter() {
                    begin = i;
                }
                begin
            })
            .collect();
        Self { begins, entry }
    }

    // Each detached use replays its own block from entry facts.
    fn record_uses(
        &self,
        xs: &[Instruction],
        guards: &Guards,
        detached: &[usize],
        uses: &mut [usize],
        facts: &mut Vec<Registers>,
        state: &mut State,
    ) {
        for &i in detached {
            state.copy_from_slice(&self.entry);
            for x in &xs[self.begins[i]..i] {
                transfer(state, guards, *x);
            }
            let snapshot = &state[guards.slice(xs[i].predicate)];
            let reached = usize::from(snapshot.reached);
            uses[i] = facts.len() * reached;
            facts.extend(std::iter::repeat_n(snapshot.registers, reached));
        }
    }
}

type Join = fn(&mut Slice, &Slice) -> bool;

/// Rows: the previous slice unreached or reached; columns: the incoming
/// one. An unreached incoming slice changes nothing; a first reaching
/// one is copied whole; two reached slices keep each register only where
/// both agree.
const JOIN: [[Join; 2]; 2] = [
    [
        |_, _| false,
        |previous, incoming| {
            *previous = *incoming;
            true
        },
    ],
    [
        |_, _| false,
        |previous, incoming| {
            let mut changed = false;
            for (previous, incoming) in previous.registers.iter_mut().zip(&incoming.registers) {
                let kept = pick(*previous == *incoming, Fact::UNKNOWN, *previous);
                changed |= kept != *previous;
                *previous = kept;
            }
            changed
        },
    ],
];

fn join(previous: &mut [Slice], incoming: &[Slice]) -> bool {
    let mut changed = false;
    for (previous, incoming) in previous.iter_mut().zip(incoming) {
        changed |=
            JOIN[usize::from(previous.reached)][usize::from(incoming.reached)](previous, incoming);
    }
    changed
}

fn transfer(state: &mut [Slice], guards: &Guards, x: Instruction) {
    // Each slice reads only itself, except uncertain writes read the
    // executed slice. Snapshot that one before mutation to avoid a Vec
    // allocation per instruction while retaining simultaneous reads.
    let executed = &state[guards.slice(x.predicate)];
    let executed_value = pick(
        executed.reached,
        Fact::UNKNOWN,
        value(x.kind, &executed.registers),
    );
    // A predicated write reads its inputs with its own guard true.
    // In another slice it may or may not execute, so both definitions
    // reach the next instruction. Opposite guards exclude the write.
    // Encoding: Mesa src/nouveau/compiler/nak/sm50.rs:225-235.
    let guard = key(x.predicate);
    let opposite = guards.opposite(x.predicate);
    // Partial MOV enables may preserve the previous definition.
    // Mesa sm50.rs:1936,1941,1946. Keep the allocation-free transfer.
    let whole = x.kind.whole_write();
    let writes = 2 * usize::from(x.kind.written_registers().count.min(1));
    let destinations = x.kind.destinations();
    let (before, after) = state.split_at_mut(opposite);
    let slices = before.iter_mut().enumerate().chain(
        after
            .iter_mut()
            .enumerate()
            .skip(1)
            .map(|(i, r)| (opposite + i, r)),
    );
    // An unreached slice's registers are never read, so writing them too
    // changes nothing.
    for (slice, Slice { registers, .. }) in slices {
        let certain = ((guards.keys[slice] == guard) | (guard == 0)) & whole;
        let value = pick(certain, executed_value, value(x.kind, registers));
        for Written { base: first, count } in destinations.into_iter().take(writes) {
            let start = usize::from(first.0);
            let end = (start + usize::from(count)).min(255);
            for (offset, register) in registers[start..end].iter_mut().enumerate() {
                let written = pick(offset == 0, Fact::UNKNOWN, value);
                *register = pick(
                    certain,
                    pick(*register == written, Fact::UNKNOWN, written),
                    written,
                );
            }
        }
    }
    // The legacy helper conservatively marks control flow as a potential
    // predicate writer. BRA/SSY/SYNC do not change predicate registers
    // (Mesa sm50.rs:3020-3052; Ryujinx InstEmitFlowControl.cs:239-275).
    // A predicate write ends the guard proofs on the registers it writes:
    // their slices restart from the unconditional state. A slice on any
    // other predicate still holds, as that predicate keeps its value.
    let unconditional = state[0];
    let written = x.kind.written_predicates();
    for (slice, &key) in state.iter_mut().zip(&guards.keys).skip(1) {
        let register = ((key - 1) / 2) as u8;
        *slice = [*slice, unconditional][usize::from(written.contains(register))];
    }
}

/// An operand's fact under `registers`; RZ reads zero from them.
fn read(source: Source, registers: &Registers) -> Fact {
    match source {
        Source::Register(r) => registers[usize::from(r.0)],
        Source::Immediate(value) => Fact::constant(value),
        Source::ConstantBuffer { binding, offset } => Fact::leaf(Leaf::buffer(binding, offset)),
    }
}

fn value(kind: InstructionKind, registers: &Registers) -> Fact {
    match kind {
        // Ryujinx Instructions/InstEmitMove.cs:12-37 copies the source.
        // Mesa sm50.rs:1931-1951 encodes register, literal and cbuf MOV.
        InstructionKind::Mov(ops::Mov { src, .. }) => read(src, registers).copied(),
        InstructionKind::Ldc(ops::Ldc {
            binding,
            mem_type: crate::ir::LdcWidth::B32,
            address_mode: crate::ir::LdcAddressMode::Default | crate::ir::LdcAddressMode::Il,
            ..
        }) => Fact::array(binding),
        InstructionKind::Lea(ops::Lea {
            lhs,
            rhs,
            shift,
            negate,
            ..
        }) => {
            // Ryujinx InstEmitIntegerArithmetic.cs:481-488. Fold literal
            // arithmetic while tracing texture/surface handles and offsets.
            // Cbuf identities are not values, so they cannot be added here.
            let a = registers[usize::from(lhs.0)];
            let b = read(rhs, registers);
            let shifted = a.a.word.wrapping_shl(u32::from(shift));
            let sum = pick(
                negate,
                b.a.word.wrapping_add(shifted),
                b.a.word.wrapping_sub(shifted),
            );
            pick(
                a.is_constant() & b.is_constant(),
                Fact::UNKNOWN,
                Fact::constant(sum),
            )
        }
        InstructionKind::Lop(ops::Lop {
            lhs,
            rhs,
            op,
            invert: modifiers,
            ..
        }) => {
            // Ryujinx Instructions/InstEmitIntegerLogical.cs:100-115 applies
            // inversion before AND/OR/XOR/PassB; predicate and CC outputs do
            // not alter the GPR result. Mesa sm50.rs:1561-1611 encodes these
            // operands, operation and inversion fields.
            let b = read(rhs, registers).inverted(modifiers[1]);
            let a = registers[usize::from(lhs.0)].inverted(modifiers[0]);
            let combined = pick(
                a.is_leaf() & b.is_leaf(),
                Fact::UNKNOWN,
                Fact::combine(op, a.a, b.a),
            );
            pick(op == LogicOp::PassB, combined, b)
        }
        _ => Fact::UNKNOWN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emitter::emit;
    use crate::interface::Stopped;
    use crate::ir::{
        BranchTarget, BrxBase, IntCompare, IntegerCompareMode, Register, Relation, SetOp,
        Signedness, TargetStack, TexOp,
    };
    use std::mem::{discriminant, Discriminant};

    fn instruction(kind: InstructionKind) -> Instruction {
        Instruction {
            address: 0,
            predicate: None,
            kind,
        }
    }

    /// `x` executed only where predicate `register` reads `!inverted`.
    fn when(x: Instruction, register: u8, inverted: bool) -> Instruction {
        Instruction {
            predicate: Some(Predicate { register, inverted }),
            ..x
        }
    }

    fn always(x: Instruction) -> Instruction {
        x
    }

    fn when_p0(x: Instruction) -> Instruction {
        when(x, 0, false)
    }

    fn when_not_p0(x: Instruction) -> Instruction {
        when(x, 0, true)
    }

    fn when_p1(x: Instruction) -> Instruction {
        when(x, 1, false)
    }

    /// No guard, P0 and !P0.
    const GUARDS: [fn(Instruction) -> Instruction; 3] = [always, when_p0, when_not_p0];

    fn mov_lanes(register: u8, source: Source, quad_lanes: u8) -> Instruction {
        instruction(InstructionKind::Mov(ops::Mov {
            quad_lanes,
            dst: Register(register),
            src: source,
        }))
    }

    fn mov(register: u8, source: Source) -> Instruction {
        mov_lanes(register, source, 0xf)
    }

    fn cb(offset: u16) -> Source {
        Source::ConstantBuffer { binding: 2, offset }
    }

    const RZ: Source = Source::Register(Register(255));

    fn branch_cc(at: usize, to: usize, cc_test: u8) -> Instruction {
        instruction(InstructionKind::Branch(ops::Branch {
            target: BranchTarget::reaching(at as u32 * 8, to as u32 * 8),
            cc_test,
        }))
    }

    fn branch(at: usize, to: usize) -> Instruction {
        branch_cc(at, to, 0xf)
    }

    fn exit(cc_test: u8) -> Instruction {
        instruction(InstructionKind::Exit(ops::Exit { cc_test }))
    }

    fn sample(op: TexOp, src0: u8, register: u8, offset: bool) -> Instruction {
        instruction(InstructionKind::Texture(ops::Texture {
            op,
            dst: Register(30),
            src0: Register(src0),
            src1: Register(register),
            dim: 2,
            mask: 1,
            offset,
            scalar: false,
            packed_f16: false,
            lod_clamp: false,
            multisample: false,
            depth_compare: false,
            nodep: false,
            scalar_dst: None,
            bound_handle: None,
        }))
    }

    fn texture(register: u8) -> Instruction {
        sample(TexOp::Sample { lod: 1 }, 4, register, false)
    }

    fn logic(dst: u8, lhs: u8, rhs: Source, op: LogicOp, invert: [bool; 2]) -> Instruction {
        instruction(InstructionKind::Lop(ops::Lop {
            dst: Register(dst),
            lhs: Register(lhs),
            rhs,
            op,
            invert,
            predicate_output: None,
            extended: false,
            cc: false,
        }))
    }

    fn lop(dst: u8, lhs: u8, rhs: Source) -> Instruction {
        logic(dst, lhs, rhs, LogicOp::Or, [false; 2])
    }

    fn surface(handle: Source, data: u8, mask: u8) -> Instruction {
        instruction(InstructionKind::Surface(ops::Surface {
            store: false,
            atomic: false,
            target: 3,
            format: 0,
            mask,
            handle,
            coord: Register(4),
            data: Register(data),
            size: 0,
            byte_address: false,
            cache: 0,
            clamp: 0,
        }))
    }

    fn program(mut xs: Vec<Instruction>) -> Vec<Instruction> {
        xs.push(exit(0xf));
        for (i, x) in xs.iter_mut().enumerate() {
            x.address = i as u32 * 8;
        }
        xs
    }

    type Resolved<T> = Result<Vec<T>, Discriminant<Stopped>>;

    /// What the analysis proves for `source` at instruction `index`.
    fn resolved_source(xs: &[Instruction], index: usize, source: Source) -> Resolved<HandleSource> {
        ControlFlowGraph::build(xs)
            .map(|cfg| {
                ReachingDefinitions::analyze(xs, &cfg)
                    .source(index, source)
                    .into_iter()
                    .collect()
            })
            .map_err(|error| discriminant(&error))
    }

    fn resolved(xs: &[Instruction], index: usize, register: u8) -> Resolved<HandleSource> {
        resolved_source(xs, index, Source::Register(Register(register)))
    }

    fn resolves(xs: &[Instruction], index: usize, register: u8, expected: HandleSource) {
        assert_eq!(resolved(xs, index, register), Ok(vec![expected]));
    }

    fn unresolved(xs: &[Instruction], index: usize, register: u8) {
        assert_eq!(resolved(xs, index, register), Ok(vec![]));
    }

    fn constant(value: u32) -> HandleSource {
        HandleSource::Leaf(HandleLeaf::Constant(value))
    }

    fn leaf_cb(offset: u16) -> HandleSource {
        HandleSource::Leaf(cb_leaf(offset))
    }

    fn cb_leaf(offset: u16) -> HandleLeaf {
        HandleLeaf::ConstantBuffer { bank: 2, offset }
    }

    fn validates(xs: &[Instruction], name: &str) {
        let words = emit(xs).into_iter().flat_map(|module| module.words);
        let path = std::env::temp_dir().join(format!(
            "shadowbox-reaching-{}-{name}.spv",
            std::process::id()
        ));
        let checked = std::fs::write(
            &path,
            words.flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>(),
        )
        .and_then(|()| {
            std::process::Command::new("spirv-val")
                .args(["--target-env", "vulkan1.2"])
                .arg(&path)
                .output()
        })
        .and_then(|result| {
            std::fs::remove_file(&path).map(|()| {
                (
                    result.status.success(),
                    String::from_utf8_lossy(&result.stderr).into_owned(),
                )
            })
        })
        .map_err(|error| error.to_string());
        assert_eq!(
            checked.as_ref().map(|(valid, _)| *valid),
            Ok(true),
            "{checked:?}"
        );
    }

    fn unknown(xs: &[Instruction], index: usize, register: u8) {
        unresolved(xs, index, register);
        let emitted = emit(xs).map(|_| ());
        assert!(matches!(emitted, Err(Stopped::Unbuilt(_))), "{emitted:?}");
    }

    #[test]
    fn main_accepted_leaf_guard_and_copy_patterns_remain_resolved() {
        // main 3251694 emitter.rs:4812-4866 (last_writer/definition/guards),
        // :4909-4930 (leaf/LOP), :4940-4959 (copies), :4976-4987 (leaves).
        // Encoding/semantics are the MOV/LOP references above, not game data.
        for (source, expected) in [
            (cb(0x568), leaf_cb(0x568)),
            (Source::Immediate(12), constant(12)),
            (RZ, constant(0)),
        ] {
            for (g, guard) in GUARDS.into_iter().enumerate() {
                for copies in 0..=3 {
                    let mut xs = vec![guard(mov(2, source))];
                    let mut register = 2;
                    for _ in 0..copies {
                        xs.push(guard(mov(
                            register + 1,
                            Source::Register(Register(register)),
                        )));
                        register += 1;
                    }
                    // Main follows each copy at its write, so subsequent
                    // source reuse cannot change the copied value.
                    xs.extend(std::iter::repeat_n(
                        mov(2, Source::Register(Register(20))),
                        copies.min(1),
                    ));
                    let at = xs.len();
                    xs.push(guard(texture(register)));
                    let xs = program(xs);
                    assert_eq!(
                        resolved(&xs, at, register),
                        Ok(vec![expected]),
                        "source={source:?}, guard={g}, copies={copies}"
                    );
                    validates(&xs, "main-leaf-copy");
                }
            }
        }
        let xs = program(vec![texture(255)]);
        resolves(&xs, 0, 255, constant(0));
        validates(&xs, "main-rz");
    }

    fn lop_pattern(
        a: Source,
        b: Source,
        op: LogicOp,
        guard: fn(Instruction) -> Instruction,
    ) -> Vec<Instruction> {
        program(vec![
            guard(mov(2, a)),
            guard(logic(19, 2, b, op, [false; 2])),
            guard(mov(20, Source::Register(Register(19)))),
            guard(texture(20)),
        ])
    }

    #[test]
    fn main_accepted_lop_and_copied_lop_patterns_remain_resolved() {
        // main 3251694 emitter.rs:4910-4930 permits AND/OR/XOR of a
        // MOV-defined left leaf and a literal, cbuf or RZ right leaf.
        // Ryujinx ConstantFolding.cs:27-40 folds two literals.
        let literal_a = [(Source::Immediate(12), 12), (RZ, 0)];
        let literal_b = [(Source::Immediate(3), 3), (RZ, 0)];
        for op in [LogicOp::And, LogicOp::Or, LogicOp::Xor] {
            for guard in GUARDS {
                for (a, x) in literal_a {
                    for (b, y) in literal_b {
                        let xs = lop_pattern(a, b, op, guard);
                        resolves(&xs, 3, 20, constant([x & y, x | y, x ^ y][op as usize]));
                        validates(&xs, "main-lop-copy");
                    }
                }
                let leaves = [
                    (cb(0x568), cb_leaf(0x568)),
                    (Source::Immediate(12), HandleLeaf::Constant(12)),
                    (RZ, HandleLeaf::Constant(0)),
                ];
                let with_cbuf = leaves
                    .iter()
                    .map(|&a| (a, (cb(0x178), cb_leaf(0x178))))
                    .chain(
                        leaves[1..]
                            .iter()
                            .map(|&b| ((cb(0x568), cb_leaf(0x568)), b)),
                    );
                for ((a, x), (b, y)) in with_cbuf {
                    let xs = lop_pattern(a, b, op, guard);
                    resolves(&xs, 3, 20, HandleSource::Combined { op, a: x, b: y });
                    validates(&xs, "main-lop-copy");
                }
            }
        }
    }

    #[test]
    fn main_immutable_entry_rule_resolves_disconnected_sample_and_fetch() {
        // main 3251694 emitter.rs:4812-4817 uses the entry writer when
        // no instruction outside entry writes the register, even when no
        // executable CFG edge reaches the texture block.
        for op in [TexOp::Sample { lod: 1 }, TexOp::Fetch { lod: false }] {
            let xs = program(vec![
                mov(2, cb(0x568)),
                branch(1, 5),
                when_p0(lop(19, 2, cb(0x178))),
                when_p0(mov(20, Source::Register(Register(19)))),
                when_p0(sample(op, 4, 20, false)),
            ]);
            resolves(
                &xs,
                4,
                20,
                HandleSource::Combined {
                    op: LogicOp::Or,
                    a: cb_leaf(0x568),
                    b: cb_leaf(0x178),
                },
            );
            validates(&xs, "main-detached-entry");
        }
    }

    #[test]
    fn main_local_definition_crosses_detached_push_target_boundary() {
        // main 3251694 emitter.rs:4742-4759 starts local tracing at BRA
        // destinations/fallthrough, not SSY or PBK target addresses.
        let xs = program(vec![
            branch(0, 5),
            instruction(InstructionKind::ResolvedSchedule(ops::ResolvedSchedule {
                push: crate::ir::ResolvedPush::checked(BranchTarget::reaching(8, 24)),
            })),
            mov(2, cb(0x568)),
            mov(19, Source::Register(Register(2))),
            texture(19),
        ]);
        resolves(&xs, 4, 19, leaf_cb(0x568));
        validates(&xs, "main-detached-push-target");
    }

    #[test]
    fn unreachable_overwrite_never_flows_into_live_handle_use() {
        let xs = program(vec![
            mov(2, cb(0x568)),
            branch(1, 4),
            mov(2, cb(0x178)),
            texture(2),
            texture(2),
        ]);
        resolves(&xs, 3, 2, leaf_cb(0x178));
        resolves(&xs, 4, 2, leaf_cb(0x568));
        validates(&xs, "detached-isolation");
    }

    #[test]
    fn main_direct_surface_and_query_handle_paths_remain_resolved() {
        // main 3251694 emitter.rs:4886-4905 selects TXQ src0 and accepts
        // immediate/cbuf surface handles without any register definition.
        for (handle, expected) in [
            (cb(0x568), leaf_cb(0x568)),
            (Source::Immediate(0), constant(0)),
            (RZ, constant(0)),
        ] {
            let xs = program(vec![surface(handle, 30, 1)]);
            assert_eq!(resolved_source(&xs, 0, handle), Ok(vec![expected]));
            validates(&xs, "main-surface-leaves");
        }
        let query = sample(TexOp::Query { query: 1 }, 2, 19, false);
        let xs = program(vec![mov(2, cb(0x568)), branch(1, 3), query]);
        resolves(&xs, 2, 2, leaf_cb(0x568));
        // src1 remains undefined: successful emission proves TXQ used src0.
        validates(&xs, "main-disconnected-query");
    }

    #[test]
    fn detached_code_still_requires_supported_and_guarded_definitions() {
        let xs = program(vec![
            branch(0, 3),
            when_p0(mov(2, cb(0x568))),
            when_not_p0(texture(2)),
        ]);
        unknown(&xs, 2, 2);
        let xs = program(vec![
            mov(2, cb(0x568)),
            branch(1, 4),
            texture(2),
            mov(2, Source::Register(Register(4))),
        ]);
        // Main's entry rule excludes registers written outside entry.
        unknown(&xs, 2, 2);
    }

    #[test]
    fn mutually_exclusive_paths_do_not_poison_same_guard_join() {
        let xs = program(vec![
            when_p0(branch(0, 3)),
            mov(2, cb(0x178)),
            branch(2, 4),
            mov(2, cb(0x568)),
            when_p0(texture(2)),
            when_not_p0(texture(2)),
        ]);
        resolves(&xs, 4, 2, leaf_cb(0x568));
        resolves(&xs, 5, 2, leaf_cb(0x178));
        validates(&xs, "exclusive-guards");
    }

    #[test]
    fn entry_definition_crosses_blocks_and_survives_later_reuse() {
        // Project-owned IR reproducing the reported R2/R19 pattern. No game
        // binary or capture is used. The overwrite is after the texture.
        let xs = program(vec![
            mov(2, cb(0x568)),
            when_p1(branch(1, 4)),
            mov(7, Source::Immediate(9)),
            branch(3, 4),
            when_p0(lop(19, 2, cb(0x178))),
            when_p0(texture(19)),
            instruction(InstructionKind::IAdd(ops::IAdd {
                dst: Register(2),
                lhs: Register(4),
                rhs: Source::Register(Register(5)),
                cc: false,
                x: false,
                negate: [false; 2],
            })),
        ]);
        resolves(
            &xs,
            5,
            19,
            HandleSource::Combined {
                op: LogicOp::Or,
                a: cb_leaf(0x568),
                b: cb_leaf(0x178),
            },
        );
        validates(&xs, "entry");
    }

    #[test]
    fn equal_definitions_at_join_resolve_through_copies() {
        let xs = program(vec![
            when_p0(branch(0, 3)),
            mov(2, cb(0x568)),
            branch(2, 4),
            mov(2, cb(0x568)),
            mov(19, Source::Register(Register(2))),
            texture(19),
        ]);
        resolves(&xs, 5, 19, leaf_cb(0x568));
        validates(&xs, "equal");
    }

    #[test]
    fn two_different_definitions_reaching_use_are_unknown() {
        let xs = program(vec![
            when_p0(branch(0, 3)),
            mov(2, cb(0x568)),
            branch(2, 4),
            mov(2, cb(0x178)),
            texture(2),
        ]);
        unknown(&xs, 4, 2);
    }

    #[test]
    fn definition_overwritten_on_one_path_is_unknown() {
        let xs = program(vec![
            mov(2, cb(0x568)),
            when_p0(branch(1, 3)),
            mov(2, Source::Register(Register(4))),
            texture(2),
        ]);
        unknown(&xs, 3, 2);
    }

    #[test]
    fn undefined_path_is_unknown() {
        let xs = program(vec![when_p0(branch(0, 2)), mov(2, cb(0x568)), texture(2)]);
        unknown(&xs, 2, 2);
    }

    #[test]
    fn loop_invariant_and_copy_cycle_converge() {
        let xs = program(vec![
            mov(2, cb(0x568)),
            mov(19, Source::Register(Register(2))),
            texture(19),
            mov(2, Source::Register(Register(19))),
            when_p0(branch(4, 1)),
        ]);
        resolves(&xs, 2, 19, leaf_cb(0x568));
        validates(&xs, "loop");
    }

    #[test]
    fn loop_carried_overwrite_is_unknown() {
        let xs = program(vec![
            mov(2, cb(0x568)),
            texture(2),
            mov(2, cb(0x178)),
            when_p0(branch(3, 1)),
        ]);
        unknown(&xs, 1, 2);
    }

    #[test]
    fn uninitialized_copy_cycle_is_unknown_without_recursion() {
        let xs = program(vec![
            mov(2, Source::Register(Register(19))),
            mov(19, Source::Register(Register(2))),
            texture(19),
            when_p0(branch(3, 0)),
        ]);
        unknown(&xs, 2, 19);
    }

    #[test]
    fn same_guard_definition_is_accepted() {
        let xs = program(vec![
            when_p0(mov(2, cb(0x568))),
            mov(3, Source::Immediate(5)),
            branch(2, 3),
            when_p0(texture(2)),
        ]);
        resolves(&xs, 3, 2, leaf_cb(0x568));
        validates(&xs, "guard");
    }

    #[test]
    fn different_or_rewritten_guard_is_rejected() {
        let writer = when_p0(mov(2, cb(0x568)));
        for guard in [always, when_not_p0, when_p1] {
            let xs = program(vec![writer, guard(texture(2))]);
            unknown(&xs, 1, 2);
        }
        let xs = program(vec![
            writer,
            instruction(InstructionKind::ISetP(ops::ISetP {
                dst: [0, 7],
                lhs: Register(4),
                rhs: Source::Immediate(0),
                compare: IntCompare::Relation(Relation::Eq),
                signedness: Signedness::Unsigned,
                mode: IntegerCompareMode::Ordinary,
                set_op: SetOp::And,
                accum: 7,
                accum_negate: false,
                abs: [false; 2],
                negate: [false; 2],
            })),
            when_p0(texture(2)),
        ]);
        unknown(&xs, 2, 2);
    }

    #[test]
    fn opposite_predicated_overwrite_cannot_reach_guarded_use() {
        let xs = program(vec![
            mov(2, cb(0x568)),
            when_not_p0(mov(2, Source::Register(Register(4)))),
            when_p0(texture(2)),
        ]);
        resolves(&xs, 2, 2, leaf_cb(0x568));
        validates(&xs, "opposite");
    }

    #[test]
    fn conditional_branch_proves_write_guard() {
        let xs = program(vec![
            when_p0(mov(2, cb(0x568))),
            when_p0(branch(1, 3)),
            exit(0xf),
            texture(2),
        ]);
        resolves(&xs, 3, 2, leaf_cb(0x568));
        validates(&xs, "branch-guard");
    }

    #[test]
    fn cc_branch_retains_fallthrough_with_true_predicate() {
        let xs = program(vec![
            mov(2, cb(0x568)),
            when_p0(mov(2, Source::Register(Register(4)))),
            when_p0(branch_cc(2, 4, 0)),
            texture(2),
            exit(0xf),
        ]);
        unknown(&xs, 3, 2);
    }

    #[test]
    fn quad_masked_move_does_not_prove_a_uniform_handle() {
        // Mesa sm50.rs:1936: partial masks retain the old destination on
        // disabled quad positions. Different cbuf definitions cannot fold.
        let xs = program(vec![
            mov(2, cb(0x568)),
            mov_lanes(2, cb(0x56c), 5),
            texture(2),
        ]);
        unknown(&xs, 2, 2);
        let same = program(vec![
            mov(2, cb(0x568)),
            mov_lanes(2, cb(0x568), 5),
            texture(2),
        ]);
        resolves(&same, 2, 2, leaf_cb(0x568));
    }

    #[test]
    fn cc_exit_keeps_fallthrough_with_true_predicate() {
        // Ryujinx InstEmitFlowControl.cs:170-185: EXIT.F cannot terminate
        // the block or imply the opposite predicate on its fallthrough.
        let xs = program(vec![
            mov(2, cb(0x568)),
            when_p0(mov(2, Source::Register(Register(4)))),
            when_p0(exit(0)),
            texture(2),
        ]);
        unknown(&xs, 3, 2);
    }

    #[test]
    fn lea_constant_handles_and_offsets_keep_definition_proofs() {
        for (negate, expected, name) in
            [(false, 25, "lea"), (true, 9u32.wrapping_sub(16), "lea-neg")]
        {
            let lea = instruction(InstructionKind::Lea(ops::Lea {
                dst: Register(20),
                lhs: Register(2),
                rhs: Source::Immediate(9),
                shift: 3,
                negate,
            }));
            let xs = program(vec![mov(2, Source::Immediate(2)), lea, texture(20)]);
            resolves(&xs, 2, 20, constant(expected));
            validates(&xs, name);
            // An unknown input, cbuf identity, or conditional definition
            // cannot become a guessed constant handle.
            for unknown in [
                [mov(2, cb(0)), lea],
                [mov(2, Source::Register(Register(6))), lea],
                [mov(2, Source::Immediate(2)), when_p0(lea)],
            ] {
                let xs = program(vec![unknown[0], unknown[1], texture(20)]);
                unresolved(&xs, 2, 20);
            }
        }
    }

    #[test]
    fn constants_cross_blocks_and_fold_lop_inputs() {
        let xs = program(vec![
            mov(2, cb(0x568)),
            mov(6, Source::Immediate(0x10)),
            branch(2, 3),
            mov(7, Source::Register(Register(6))),
            lop(3, 7, Source::Immediate(2)),
            sample(TexOp::Sample { lod: 1 }, 4, 2, true),
        ]);
        let constants = |register| {
            ControlFlowGraph::build(&xs)
                .map(|cfg| {
                    ReachingDefinitions::analyze(&xs, &cfg)
                        .constant(5, register)
                        .into_iter()
                        .collect::<Vec<_>>()
                })
                .map_err(|error| discriminant(&error))
        };
        assert_eq!(constants(3), Ok(vec![0x12]));
        assert_eq!(constants(255), Ok(vec![0]));
        validates(&xs, "constant");
    }

    #[test]
    fn multi_register_and_previously_missing_writers_kill_definitions() {
        let writers = [
            // Ryujinx InstEmitConversion.cs:410-415 writes both double words;
            // InstEmitBitfield.cs:182 writes FLO's destination.
            InstructionKind::Convert(ops::Convert {
                dst: Register(1),
                src: Source::Immediate(1),
                kind: crate::ir::Conversion::I2F {
                    signed: false,
                    src_size: 2,
                    dst_size: 3,
                    round: crate::ir::RoundingMode::NearestEven,
                    byte: 0,
                    abs: false,
                    negate: false,
                },
            }),
            InstructionKind::Flo(ops::Flo {
                dst: Register(2),
                src: Source::Immediate(1),
                signed: false,
                invert: false,
                shift: false,
            }),
            InstructionKind::Lop3(ops::Lop3 {
                dst: Register(2),
                lhs: Register(4),
                rhs: Source::Immediate(6),
                third: Register(5),
                lut: 0x96,
                predicate_output: None,
                extended: false,
                cc: false,
            }),
            InstructionKind::ISet(ops::ISet {
                dst: Register(2),
                lhs: Register(4),
                rhs: Source::Immediate(0),
                compare: IntCompare::Relation(Relation::Eq),
                signedness: Signedness::Unsigned,
                bool_float: false,
                cc: false,
                x: false,
                accum: 7,
                accum_negate: false,
                set_op: crate::ir::SetOp::And,
            }),
            InstructionKind::Ldc(ops::Ldc {
                dst: Register(1),
                binding: 2,
                offset: Register(4),
                immediate: 0,
                mem_type: crate::ir::LdcWidth::B64,
                address_mode: crate::ir::LdcAddressMode::Default,
            }),
            surface(cb(0x178), 1, 3).kind,
            InstructionKind::AttributeLoad(ops::AttributeLoad {
                vertex: crate::ir::PrimitiveVertex::Current,
                dst: Register(1),
                address: 0x80,
                components: 2,
                lanes: crate::ir::AttributeLanes::vertex(0x80, 2, crate::ir::AttributeBank::Input),
            }),
            InstructionKind::Atomic(ops::Atomic {
                dst: Register(2),
                address: Register(4),
                offset: 0,
                data: Register(5),
                op: crate::ir::AtomicOp::Add,
                signed: false,
            }),
            InstructionKind::Texture(ops::Texture {
                op: TexOp::Sample { lod: 1 },
                dst: Register(30),
                src0: Register(4),
                src1: Register(5),
                dim: 2,
                mask: 15,
                offset: false,
                scalar: true,
                packed_f16: false,
                lod_clamp: false,
                multisample: false,
                depth_compare: false,
                nodep: false,
                scalar_dst: Some(Register(1)),
                bound_handle: Some(0),
            }),
        ];
        for writer in writers {
            let xs = program(vec![mov(2, cb(0x568)), instruction(writer), texture(2)]);
            unknown(&xs, 2, 2);
        }
    }

    #[test]
    fn surface_handle_crosses_blocks() {
        let xs = program(vec![
            mov(2, cb(0x568)),
            branch(1, 2),
            surface(Source::Register(Register(2)), 30, 1),
        ]);
        resolves(&xs, 2, 2, leaf_cb(0x568));
        validates(&xs, "surface");
    }

    #[test]
    fn brx_joins_every_table_target() {
        let xs = program(vec![
            mov(2, cb(0x568)),
            instruction(InstructionKind::brx(
                Register(4),
                BrxBase::decode(8, 0),
                0,
                &[],
            )),
            texture(2),
        ]);
        resolves(&xs, 2, 2, leaf_cb(0x568));
        let mut targets = [0; 32];
        targets[1] = 16;
        let xs = program(vec![
            mov(2, cb(0x568)),
            instruction(InstructionKind::brx(
                Register(4),
                BrxBase::decode(8, 0),
                targets[0],
                &targets[1..2],
            )),
            texture(2),
            mov(2, cb(0x178)),
            branch(4, 2),
        ]);
        unknown(&xs, 2, 2);
    }

    fn words(bundles: &[u64]) -> Vec<u32> {
        bundles
            .iter()
            .flat_map(|&word| [word as u32, (word >> 32) as u32])
            .collect()
    }

    #[test]
    fn encoded_ssy_sync_reaches_predicated_lop_and_tex() {
        // Synthetic SM50 bundles. Encoding references:
        // MOV: Mesa sm50.rs:1931-1951; cbuf bank/word offset :361-368;
        // guard :225-235; SSY/SYNC :3032-3052; NOP :3178-3184;
        // LOP :1591-1611; TEX :2181-2196; EXIT :3101-3110.
        let mov =
            (0x4c98_u64 << 48) | (0xf << 39) | (2 << 34) | ((0x568 / 4) << 20) | (7 << 16) | 2;
        let ssy = (0xe290_u64 << 48) | (0x30 << 20) | 0xf;
        let sync = (0xf0f8_u64 << 48) | (7 << 16) | 0xf;
        let nop = (0x50b0_u64 << 48) | (7 << 16);
        let unrelated = (0x0100_u64 << 48) | (5 << 20) | (7 << 16) | (0xf << 12) | 2;
        let lop = (0x4c40_u64 << 48)
            | (7 << 48)
            | (1 << 41)
            | (2 << 34)
            | ((0x178 / 4) << 20)
            | (2 << 8)
            | 19;
        let tex =
            (0xdeb8_u64 << 48) | (1 << 37) | (1 << 31) | (2 << 28) | (19 << 20) | (4 << 8) | 30;
        let exit = (0xe300_u64 << 48) | (7 << 16) | 0xf;
        let xs = crate::decoder::decode_stream(
            crate::QuietRun,
            &words(&[
                0, mov, ssy, sync, 0, unrelated, nop, nop, 0, lop, tex, unrelated, 0, exit, nop,
                nop,
            ]),
        );
        assert_eq!(
            xs[2].kind,
            InstructionKind::Reconverge(ops::Reconverge {
                target: BranchTarget::reaching(xs[2].address, xs[2].address + 48),
                cc_test: 0xf,
                stack: TargetStack::Sync,
            })
        );
        let at = xs.partition_point(|x| x.address < 80);
        assert_eq!(xs[at].address, 80);
        resolves(
            &xs,
            at,
            19,
            HandleSource::Combined {
                op: LogicOp::Or,
                a: cb_leaf(0x568),
                b: cb_leaf(0x178),
            },
        );
        validates(&xs, "ssy-sync");
    }

    #[test]
    fn encoded_stack_targets_skip_disconnected_code_and_preserve_live_handles() {
        // MOV/cbuf/guards: Mesa sm50.rs:1931-1951,361-368,225-235;
        // LOP: :1591-1611; TEX: :2181-2196; EXIT/NOP: :3101-3110,3178-3184.
        // SSY/SYNC: :3032-3052; PBK/PCNT/BRK/CONT: envytools
        // gm107.c:1930-1935,1920-1921. Relative target: Mesa :3006-3016.
        let mov =
            (0x4c98_u64 << 48) | (0xf << 39) | (2 << 34) | ((0x568 / 4) << 20) | (7 << 16) | 2;
        let lop = (0x4c47_u64 << 48) | (1 << 41) | (2 << 34) | ((0x178 / 4) << 20) | (2 << 8) | 19;
        let tex =
            (0xdeb8_u64 << 48) | (1 << 37) | (1 << 31) | (2 << 28) | (19 << 20) | (4 << 8) | 30;
        let copy = (0x5c98_u64 << 48) | (0xf << 39) | (2 << 20) | (7 << 16) | 20;
        let live_tex = (tex & !(0xff << 20)) | (20 << 20) | (7 << 16);
        let nop = (0x50b0_u64 << 48) | (7 << 16);
        let exit = (0xe300_u64 << 48) | (7 << 16) | 0xf;
        for (push, pop, stack) in [
            (0xe290_u64, 0xf0f8_u64, TargetStack::Sync),
            (0xe2a0, 0xe340, TargetStack::Break),
            (0xe2b0, 0xe350, TargetStack::Continue),
        ] {
            let xs = crate::decoder::decode_stream(
                crate::QuietRun,
                &words(&[
                    0,
                    mov,
                    (push << 48) | (0x28 << 20) | 0xf,
                    (pop << 48) | (7 << 16) | 0xf,
                    0,
                    lop,
                    tex,
                    nop,
                    0,
                    copy,
                    live_tex,
                    exit,
                ]),
            );
            assert_eq!(
                xs[2].kind,
                InstructionKind::Reconverge(ops::Reconverge {
                    target: BranchTarget::reaching(xs[2].address, xs[2].address + 48),
                    cc_test: 0xf,
                    stack,
                })
            );
            // Ryujinx Decoder.cs:133-162: the resolved pop bypasses this
            // disconnected block. Its handle writes never enter translation.
            assert_eq!(
                xs.iter().map(|x| x.address).collect::<Vec<_>>(),
                [8, 16, 24, 72, 80, 88]
            );
            resolves(&xs, 4, 20, leaf_cb(0x568));
            validates(&xs, "encoded-stack-handles");
        }
    }

    #[test]
    fn unsupported_lop_expression_stays_unknown() {
        let xs = program(vec![
            mov(2, cb(0x568)),
            when_p0(logic(19, 2, cb(0x178), LogicOp::Or, [true, false])),
            when_p0(texture(19)),
        ]);
        unknown(&xs, 2, 19);
    }
}
