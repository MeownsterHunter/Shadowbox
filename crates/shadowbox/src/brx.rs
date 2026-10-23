//! Register BRX targets from what the displacement register is where the
//! BRX reads it. Independently follows Ryujinx Decoders/Decoder.cs:518-688:
//! a displacement loaded from cbuf1 by an unsigned-bounded index is a table
//! read that knows its entries. What reaches a register is a [`Value`]: the
//! values learned there, or the places whose values were not learned. Every
//! edge of the [`Graph`] is forward or backward from the moment the graph
//! makes it, so the walk never asks what an edge is. A BRX reads a value
//! that already is what it is. See docs/reference-sources.md
use crate::decoder::{FallGuard, Slots};
use crate::flow::{Cc, Program, PushOp, Seat, Transfer};
use crate::guard::Guard;
use crate::ir::ops;
use crate::ir::{
    BrxBase, BrxTable, Instruction, InstructionKind as K, OutsideBrxTable, Register, ShownSlot,
    SlotTable, Source, TableEntry, TargetStack, Translation,
};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Write;
use std::iter::once;

/// A table holds at most this many entries.
const LIMIT: usize = 32;

/// Values a register may hold, in ascending order.
type Set = BTreeSet<u32>;

/// The places learning stopped, by position.
type Stops = BTreeSet<usize>;

/// Where a value is made: an instruction, or the edge that enters the
/// program there.
#[derive(Clone, Copy, Debug)]
struct Place(usize);

impl Place {
    /// Learning stops here: what this place makes.
    fn stops(self) -> Value {
        Value {
            values: Set::new(),
            stopped: Stops::from([self.0]),
        }
    }

    /// The values and stops that reach a register here. Each falls into
    /// its own tube: a learned value into the values, a place where
    /// learning stopped into the stops. The values tube holds what a table
    /// holds; values that overflow it are a shape not learned, so they
    /// become one stop, here. A stop here is that shape: values meeting it
    /// here fall in with it and do not return when later values join, so
    /// what reaches a place only grows. A stop made elsewhere keeps the
    /// values it meets.
    fn shapes(self, mut values: Set, mut stopped: Stops) -> Value {
        if values.len() > LIMIT || stopped.contains(&self.0) {
            values.clear();
            stopped.insert(self.0);
        }
        Value { values, stopped }
    }
}

/// What may reach a register at a point of the program: the values
/// learned there, and the places where learning stopped for a value that
/// reaches it. Where nothing reaches yet, it holds no value and no place.
/// Every value but that is made by [`Place`].
#[derive(Clone, PartialEq, Eq)]
struct Value {
    values: Set,
    stopped: Stops,
}

impl Value {
    /// Nothing the program made reaches here yet.
    fn nothing() -> Self {
        Self {
            values: Set::new(),
            stopped: Stops::new(),
        }
    }

    /// `self` and `other`, both reaching `at`.
    fn join(self, other: &Self, at: Place) -> Self {
        let Self {
            mut values,
            mut stopped,
        } = self;
        values.extend(&other.values);
        stopped.extend(&other.stopped);
        at.shapes(values, stopped)
    }
}

/// The cb1 word at byte `address`, when the snapshot holds an aligned one:
/// the aligned-table proof boundary.
fn word(cb1: &[u32], address: u32) -> Option<u32> {
    cb1.get(address as usize / 4)
        .copied()
        .filter(|_| address.is_multiple_of(4))
}

/// Where a write reads the values that form it.
trait Read {
    /// The values `register` holds. RZ reads zero: Ryujinx
    /// IntermediateRepresentation/OperandHelper.cs:43-52.
    fn register(&mut self, register: Register) -> Set {
        match register {
            Register(255) => Set::from([0]),
            register => self.tracked(register),
        }
    }

    fn tracked(&mut self, register: Register) -> Set;

    fn cb1(&self) -> &[u32];

    /// The write's value has a shape not learned: learning stops at it.
    fn stop(&mut self) -> Set;

    /// The unsigned minimum of the register `input` and `ceiling`.
    fn clamp(&mut self, ceiling: Source, input: Register) -> Set;

    /// The values `source` holds. Cbuf word offset: Mesa sm50.rs:361-368;
    /// Decoder.cs:543-544.
    fn source(&mut self, source: Source) -> Set {
        match source {
            Source::Register(register) => self.register(register),
            Source::Immediate(value) => Set::from([value]),
            Source::ConstantBuffer { binding: 1, offset } => {
                match word(self.cb1(), u32::from(offset)) {
                    Some(value) => Set::from([value]),
                    None => self.stop(),
                }
            }
            Source::ConstantBuffer { .. } => self.stop(),
        }
    }

    /// `set`, formed here: more values than a table holds is a shape not
    /// learned.
    fn bounded(&mut self, set: Set) -> Set {
        if set.len() <= LIMIT {
            set
        } else {
            self.stop()
        }
    }

    /// `f` of every pair of values.
    fn pairs(&mut self, a: Set, b: Set, f: impl Fn(u32, u32) -> u32) -> Set {
        let set = a
            .iter()
            .flat_map(|&a| b.iter().map(move |&b| (a, b)))
            .map(|(a, b)| f(a, b))
            .collect();
        self.bounded(set)
    }

    /// Both values.
    fn union(&mut self, mut a: Set, b: Set) -> Set {
        a.extend(b);
        self.bounded(a)
    }

    /// The cb1 words at byte addresses `addresses + immediate`.
    fn loaded(&mut self, addresses: Set, immediate: i32) -> Set {
        let cb1 = self.cb1();
        let words = addresses
            .into_iter()
            .map(|address| address.wrapping_add(immediate as u32))
            .try_fold(Set::new(), |mut words, address| {
                words.insert(word(cb1, address)?);
                Some(words)
            });
        match words {
            Some(words) => self.bounded(words),
            None => self.stop(),
        }
    }
}

/// How an instruction forms the value it writes from the values it reads.
trait Forms {
    fn form<R: Read>(&self, read: &mut R) -> Set;
}

impl Forms for ops::Mov {
    fn form<R: Read>(&self, read: &mut R) -> Set {
        read.source(self.src)
    }
}

/// IMNMX min PT: Ryujinx InstEmitIntegerMinMax.cs:45-73. Only the unsigned
/// minimum bounds its input.
impl Forms for ops::IMinMax {
    fn form<R: Read>(&self, read: &mut R) -> Set {
        match self {
            Self {
                min: true,
                signed: false,
                ..
            } => read.clamp(self.rhs, self.lhs),
            _ => {
                drop(read.source(self.rhs));
                drop(read.register(self.lhs));
                read.stop()
            }
        }
    }
}

/// SHL and oversized-count zero: InstEmitShift.cs:180-198.
impl Forms for ops::Shl {
    fn form<R: Read>(&self, read: &mut R) -> Set {
        let a = read.register(self.lhs);
        let b = read.source(self.rhs);
        if self.wrap {
            read.stop()
        } else {
            read.pairs(a, b, |a, b| a.checked_shl(b).unwrap_or(0))
        }
    }
}

/// A source read negated, as the instruction encodes it.
fn negated(value: u32, negate: bool) -> u32 {
    if negate {
        value.wrapping_neg()
    } else {
        value
    }
}

/// IADD modulo32 with source negations: InstEmitIntegerArithmetic.cs:317-340.
impl Forms for ops::IAdd {
    fn form<R: Read>(&self, read: &mut R) -> Set {
        let a = read.register(self.lhs);
        let b = read.source(self.rhs);
        let [first, second] = self.negate;
        if self.x {
            read.stop()
        } else {
            read.pairs(a, b, |a, b| {
                negated(a, first).wrapping_add(negated(b, second))
            })
        }
    }
}

/// LEA/ISCADD: InstEmitIntegerArithmetic.cs:469-490.
impl Forms for ops::ShiftAdd {
    fn form<R: Read>(&self, read: &mut R) -> Set {
        let a = read.register(self.lhs);
        let b = read.source(self.rhs);
        let (shift, [first, second]) = (self.shift, self.negate);
        read.pairs(a, b, |a, b| {
            negated(a << shift, first).wrapping_add(negated(b, second))
        })
    }
}

/// Dedicated low LEA has the same bounded arithmetic. envytools
/// envydis/gm107.c:2022,2093,2140; Ryujinx
/// Instructions/InstEmitIntegerArithmetic.cs:481-490.
impl Forms for ops::Lea {
    fn form<R: Read>(&self, read: &mut R) -> Set {
        let a = read.register(self.lhs);
        let b = read.source(self.rhs);
        let (shift, negate) = (self.shift, self.negate);
        read.pairs(a, b, |a, b| negated(a << shift, negate).wrapping_add(b))
    }
}

/// SEL preserves raw bits: InstEmitMove.cs:219-249,271-275. PT (7) selects
/// `when_true`, !PT `when_false`, and a predicate either.
impl Forms for ops::Sel {
    fn form<R: Read>(&self, read: &mut R) -> Set {
        match (self.condition, self.condition_negate) {
            (7, false) => read.register(self.when_true),
            (7, true) => read.source(self.when_false),
            _ => {
                let a = read.register(self.when_true);
                let b = read.source(self.when_false);
                read.union(a, b)
            }
        }
    }
}

/// Indexed LDC B32 from cb1, address register + signed byte immediate:
/// Mesa sm50.rs:2633-2648; InstEmitMemory.cs:59-95. A bounded index makes
/// a table read that knows its entries.
impl Forms for ops::Ldc {
    fn form<R: Read>(&self, read: &mut R) -> Set {
        let addresses = read.register(self.offset);
        match self {
            Self {
                binding: 1,
                mem_type: crate::ir::LdcWidth::B32,
                ..
            } => read.loaded(addresses, self.immediate),
            _ => read.stop(),
        }
    }
}

/// The value `kind` writes to its first destination register.
fn form<R: Read>(kind: K, read: &mut R) -> Set {
    match kind {
        K::Mov(x) => x.form(read),
        K::IMinMax(x) => x.form(read),
        K::Shl(x) => x.form(read),
        K::IAdd(x) => x.form(read),
        K::ShiftAdd(x) => x.form(read),
        K::Lea(x) => x.form(read),
        K::Sel(x) => x.form(read),
        K::Ldc(x) => x.form(read),
        _ => read.stop(),
    }
}

/// The registers BRX values flow through: the displacements, then every
/// register a write to one of them reads.
struct Tracked(Vec<Register>);

/// Reads that only name the registers a write reads.
struct Naming<'a> {
    named: &'a mut Vec<Register>,
    cb1: &'a [u32],
}

impl Read for Naming<'_> {
    fn tracked(&mut self, register: Register) -> Set {
        self.named.push(register);
        Set::new()
    }

    fn cb1(&self) -> &[u32] {
        self.cb1
    }

    fn stop(&mut self) -> Set {
        Set::new()
    }

    fn clamp(&mut self, ceiling: Source, input: Register) -> Set {
        drop(self.source(ceiling));
        self.register(input)
    }
}

impl Tracked {
    fn new(xs: &[Instruction], sites: &[Site], cb1: &[u32]) -> Self {
        let mut named = Vec::new();
        let mut naming = Naming {
            named: &mut named,
            cb1,
        };
        sites
            .iter()
            .for_each(|site| drop(naming.register(site.displacement)));
        let mut registers = Vec::new();
        while let Some(register) = named.pop() {
            if registers.contains(&register) {
                continue;
            }
            registers.push(register);
            let mut naming = Naming {
                named: &mut named,
                cb1,
            };
            xs.iter()
                .filter(|x| x.kind.written_registers().base == register)
                .for_each(|x| drop(form(x.kind, &mut naming)));
        }
        registers.sort_unstable_by_key(|r| r.0);
        Self(registers)
    }

    /// Where tracked `register`'s value stands among the tracked ones.
    fn slot(&self, register: Register) -> usize {
        self.0.partition_point(|r| r.0 < register.0)
    }
}

/// A register BRX that may run: where it is and what it reads.
struct Site {
    at: usize,
    displacement: Register,
    base: BrxBase,
}

/// A register BRX whose displacement is a shape not learned: the BRX, and
/// the instruction where its value's shape was not learned. Each may jump
/// to any slot, so discovery keeps every remaining slot.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Unlearned {
    pub(crate) brx: u32,
    pub(crate) at: u32,
    pub(crate) shape: K,
}

/// The register BRXs discovery could not turn into tables this round.
#[derive(Debug, Default)]
pub(crate) struct Track(Vec<Unlearned>);

impl Track {
    /// Every register BRX here may jump to any real slot.
    pub(crate) fn unlearned(&self) -> &[Unlearned] {
        &self.0
    }

    fn record(&mut self, xs: &[Instruction], brx: usize, at: usize) {
        self.0.push(Unlearned {
            brx: xs[brx].address,
            at: xs[at].address,
            shape: xs[at].kind,
        });
    }
}

/// A table the proof made from the values reaching a BRX: at least one
/// entry, and no more than a table holds.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Proven {
    table: BrxTable,
    base: BrxBase,
}

impl Proven {
    /// The base the BRX adds each entry to.
    pub(crate) fn base(self) -> BrxBase {
        self.base
    }

    /// `meets` meets the table as the shape it is.
    pub(crate) fn meets<M: crate::ir::TableMeets>(self, meets: M) -> M::Out {
        self.table.meets(meets)
    }
}

/// Where one round's displacements go: proposed tables, in site order, or
/// the track.
struct Proposal<'a> {
    xs: &'a [Instruction],
    tables: &'a mut Vec<Proven>,
    track: &'a mut Track,
}

/// The program's edges under the tables proven so far. Each edge is born
/// forward, into an instruction after the one it leaves, or backward, to
/// one at or before it, and stays what it was born as.
#[derive(Clone)]
pub(crate) struct Graph {
    /// The instructions each instruction is entered from forward.
    forward: Vec<Vec<usize>>,
    /// The instructions each instruction enters forward.
    onward: Vec<Vec<usize>>,
    /// The instructions each instruction returns to backward.
    backward: Vec<Vec<usize>>,
    /// How many edges enter each instruction.
    incoming: Vec<usize>,
}

impl Graph {
    /// The edges `program`'s things make, standing where `xs` places them.
    /// A register BRX has no table yet, so it makes no jump edge.
    pub(crate) fn of(program: &Program, xs: &[Instruction], slots: &Slots) -> Self {
        let len = xs.len();
        let mut pushes = Pushes(Vec::new());
        program.each(&mut pushes);
        let mut graph = Self {
            forward: vec![Vec::new(); len],
            onward: vec![Vec::new(); len],
            backward: vec![Vec::new(); len],
            incoming: vec![0; len],
        };
        program.each(&mut Edging {
            graph: &mut graph,
            xs,
            slots,
            pushes: &pushes.0,
        });
        graph
    }

    /// The edge from `from` to `to`, born forward or backward.
    fn edge(&mut self, from: usize, to: usize) {
        self.incoming[to] += 1;
        if to > from {
            self.forward[to].push(from);
            self.onward[from].push(to);
        } else {
            self.backward[from].push(to);
        }
    }

    /// This graph, with each site's BRX jumping to every entry of the table
    /// this round proposed there.
    fn tabled(&self, sites: &[Site], tables: &[Proven], slots: &Slots) -> Self {
        let mut graph = self.clone();
        for (site, proven) in sites.iter().zip(tables) {
            proven.meets(Tabled {
                at: site.at,
                graph: &mut graph,
                slots,
            });
        }
        graph
    }
}

/// Where a thing leaves: its position and address, the graph its edges
/// join, and the pushes a pop may return to.
pub(crate) struct Leaving<'g> {
    at: usize,
    address: u32,
    graph: &'g mut Graph,
    slots: &'g Slots,
    pushes: &'g [(TargetStack, u32)],
}

impl Leaving<'_> {
    /// An edge to the instruction at `slot`, where one stands there.
    fn to(&mut self, slot: u32) {
        let (at, graph) = (self.at, &mut *self.graph);
        self.slots
            .get(slot)
            .into_iter()
            .for_each(|to| graph.edge(at, to));
    }

    /// The physical next slot, skipping only bundle control words:
    /// Mesa sm50.rs:3418-3428; Ryujinx Decoder.cs:265-270.
    fn next(&mut self) {
        self.to(crate::ir::run_slot(self.address + 8));
    }

    /// Until pops have unique targets, a pop may return to every push of
    /// its stack. This adds histories, never removes one (Ryujinx
    /// Decoder.cs:824-869).
    fn pops(&mut self, stack: TargetStack) {
        let pushes = self.pushes;
        pushes
            .iter()
            .filter(|&&(pushed, _)| pushed == stack)
            .for_each(|&(_, slot)| self.to(slot));
    }

    /// Every instruction in the program.
    fn everywhere(&mut self) {
        let (at, graph) = (self.at, &mut *self.graph);
        (0..graph.incoming.len()).for_each(|to| graph.edge(at, to));
    }
}

/// A BRX at `at` with a proposed table: its entries are its edges.
struct Tabled<'g> {
    at: usize,
    graph: &'g mut Graph,
    slots: &'g Slots,
}

impl Tabled<'_> {
    fn to(&mut self, slot: u32) {
        let (at, graph) = (self.at, &mut *self.graph);
        self.slots
            .get(slot)
            .into_iter()
            .for_each(|to| graph.edge(at, to));
    }
}

impl crate::ir::TableMeets for Tabled<'_> {
    type Out = ();
    fn slots(mut self, table: SlotTable) {
        table
            .targets()
            .iter()
            .for_each(|target| self.to(target.slot()));
    }
    fn outside(mut self, table: OutsideBrxTable) {
        table.slots().for_each(|entry| self.to(entry.slot()));
    }
}

/// The edges a thing makes in the BRX proof's graph where it leaves.
pub(crate) trait Edges: Copy {
    fn edges(self, leaving: Leaving<'_>, guard: impl FallGuard);
    /// The push a pop of its stack may return to, where this thing is one.
    fn pushes(self, _: &mut Vec<(TargetStack, u32)>) {}
}

impl<C: Cc> Edges for crate::flow::Jump<C> {
    fn edges(self, mut leaving: Leaving<'_>, guard: impl FallGuard) {
        leaving.to(self.slot());
        guard.falls_tested(self.cc(), || leaving.next());
    }
}

impl<C: Cc> Edges for crate::flow::Reconverge<C> {
    fn edges(self, mut leaving: Leaving<'_>, guard: impl FallGuard) {
        leaving.to(self.slot());
        guard.falls_tested(self.cc(), || leaving.next());
    }
}

impl<C: Cc, Y: crate::flow::Gap> Edges for crate::flow::NoSlotJump<C, Y> {
    fn edges(self, mut leaving: Leaving<'_>, guard: impl FallGuard) {
        guard.falls_tested(self.cc(), || leaving.next());
    }
}

impl<C: Cc> Edges for crate::flow::Exit<C> {
    fn edges(self, mut leaving: Leaving<'_>, guard: impl FallGuard) {
        guard.falls_tested(self.cc(), || leaving.next());
    }
}

impl<C: Cc> Edges for crate::flow::Pop<C> {
    fn edges(self, mut leaving: Leaving<'_>, guard: impl FallGuard) {
        leaving.pops(self.stack());
        guard.falls_tested(self.cc(), || leaving.next());
    }
}

impl Edges for crate::flow::TableJump {
    fn edges(self, mut leaving: Leaving<'_>, guard: impl FallGuard) {
        self.targets()
            .iter()
            .for_each(|target| leaving.to(target.slot()));
        guard.falls(|| leaving.next());
    }
}

impl Edges for crate::flow::OutsideTable {
    fn edges(self, mut leaving: Leaving<'_>, guard: impl FallGuard) {
        self.table()
            .slots()
            .for_each(|entry| leaving.to(entry.slot()));
        guard.falls(|| leaving.next());
    }
}

/// A register BRX without a proven table names no destination here. Its
/// condition-code test still supplies the fallthrough edge when it can fail.
impl<C: Cc> Edges for crate::flow::AnyJump<C> {
    fn edges(self, mut leaving: Leaving<'_>, guard: impl FallGuard) {
        guard.falls_tested(self.cc(), || leaving.next());
    }
}

impl<P: PushOp> Edges for crate::flow::Push<P> {
    fn edges(self, mut leaving: Leaving<'_>, _: impl FallGuard) {
        leaving.next();
    }
    fn pushes(self, pushes: &mut Vec<(TargetStack, u32)>) {
        pushes.push((self.stack(), self.slot()));
    }
}

/// Things that continue to the next word, guarded or not. A resolved push
/// and an outside push name no slot an unresolved pop returns to.
macro_rules! continues {
    ($($t:ty),*) => {$(
        impl Edges for $t {
            fn edges(self, mut leaving: Leaving<'_>, _: impl FallGuard) {
                leaving.next();
            }
        }
    )*};
}
continues!(
    crate::flow::NoSlotPush,
    crate::flow::Step,
    crate::flow::PredicateWrite
);

impl<P: PushOp> Edges for crate::flow::ResolvedPush<P> {
    fn edges(self, mut leaving: Leaving<'_>, _: impl FallGuard) {
        leaving.next();
    }
}

/// An instruction the decoder could not name can hide arbitrary register
/// and control effects: it may jump anywhere.
impl Edges for crate::flow::Unknown {
    fn edges(self, mut leaving: Leaving<'_>, _: impl FallGuard) {
        leaving.everywhere();
    }
}

/// A call's function can write any register before it returns.
impl Edges for crate::flow::Call {
    fn edges(self, mut leaving: Leaving<'_>, _: impl FallGuard) {
        leaving.everywhere();
    }
}

impl Edges for crate::flow::NoSlotCall {
    fn edges(self, mut leaving: Leaving<'_>, _: impl FallGuard) {
        leaving.everywhere();
    }
}

/// The pushes in the program.
struct Pushes(Vec<(TargetStack, u32)>);

impl<'p> crate::flow::Meet<'p, usize> for Pushes {
    fn meet<G: Guard, T: Transfer>(&mut self, seat: &'p Seat<usize, G, T>) {
        seat.transfer().pushes(&mut self.0);
    }
}

/// Each thing makes its edges.
struct Edging<'g> {
    graph: &'g mut Graph,
    xs: &'g [Instruction],
    slots: &'g Slots,
    pushes: &'g [(TargetStack, u32)],
}

impl<'p> crate::flow::Meet<'p, usize> for Edging<'_> {
    fn meet<G: Guard, T: Transfer>(&mut self, seat: &'p Seat<usize, G, T>) {
        let at = seat.at();
        seat.transfer().edges(
            Leaving {
                at,
                address: self.xs[at].address,
                graph: &mut *self.graph,
                slots: self.slots,
                pushes: self.pushes,
            },
            seat.guard(),
        );
    }
}

/// What one instruction does to the tracked registers when it runs.
struct Step {
    at: Place,
    kind: K,
    /// Registers whose value after the instruction is where learning
    /// stopped: all of them past an opaque instruction, and the extra
    /// destinations.
    stopped: Vec<usize>,
    /// The register the instruction forms from the values it reads.
    formed: Vec<usize>,
    /// Registers a guarded or lane-partial write keeps its older value in.
    kept: Vec<usize>,
}

impl Step {
    fn new(at: usize, x: &Instruction, tracked: &Tracked) -> Self {
        x.kind.translation(Arrival { at, x, tracked })
    }

    /// The step at an instruction the decoder could not name, padding or
    /// not: it may write any register, so no value crosses it.
    fn opaque(at: usize, x: &Instruction, tracked: &Tracked) -> Self {
        Self {
            at: Place(at),
            kind: x.kind,
            stopped: (0..tracked.0.len()).collect(),
            formed: Vec::new(),
            kept: Vec::new(),
        }
    }

    /// The step at an instruction that writes only what it names.
    fn written(at: usize, x: &Instruction, tracked: &Tracked) -> Self {
        let mut step = Self {
            at: Place(at),
            kind: x.kind,
            stopped: Vec::new(),
            formed: Vec::new(),
            kept: Vec::new(),
        };
        let written = x.kind.written_registers().base;
        let partial = x.predicate.is_some() | !x.kind.whole_write();
        for (slot, &register) in tracked.0.iter().enumerate() {
            if !x.kind.writes_register(register.0) {
                continue;
            }
            if partial {
                step.kept.push(slot);
            }
            // Only the first destination register holds the formed value.
            if written == register {
                step.formed.push(slot);
            } else {
                step.stopped.push(slot);
            }
        }
        step
    }
}

/// An instruction arriving at the walk: the step it makes.
struct Arrival<'a> {
    at: usize,
    x: &'a Instruction,
    tracked: &'a Tracked,
}

impl Translation for Arrival<'_> {
    type Out = Step;
    fn unknown(self) -> Step {
        Step::opaque(self.at, self.x, self.tracked)
    }
    fn padding(self) -> Step {
        Step::opaque(self.at, self.x, self.tracked)
    }
    fn unbuilt(self) -> Step {
        Step::opaque(self.at, self.x, self.tracked)
    }
    fn unwanted(self) -> Step {
        Step::opaque(self.at, self.x, self.tracked)
    }
    fn unresolved_stack(self) -> Step {
        Step::written(self.at, self.x, self.tracked)
    }
    fn unproven_stream(self) -> Step {
        Step::written(self.at, self.x, self.tracked)
    }
    fn known(self) -> Step {
        Step::written(self.at, self.x, self.tracked)
    }
}

/// A write forming its value from the things it reads.
struct Making<'a> {
    tracked: &'a Tracked,
    file: &'a [Value],
    cb1: &'a [u32],
    at: Place,
    /// The places learning stopped for the values the reads met.
    stopped: Stops,
}

impl Read for Making<'_> {
    /// The register's learned values; where learning stopped for it goes
    /// into the write.
    fn tracked(&mut self, register: Register) -> Set {
        let thing = &self.file[self.tracked.slot(register)];
        self.stopped.extend(&thing.stopped);
        thing.values.clone()
    }

    fn cb1(&self) -> &[u32] {
        self.cb1
    }

    /// Learning stops at this write.
    fn stop(&mut self) -> Set {
        self.stopped.insert(self.at.0);
        Set::new()
    }

    /// A ceiling below the table size bounds the input to `0..=upper`: an
    /// unsigned minimum of a value not learned is any value up to it.
    fn clamp(&mut self, ceiling: Source, input: Register) -> Set {
        let ceiling = self.source(ceiling);
        match ceiling.last().copied() {
            // No value reaches the ceiling yet.
            None => Set::new(),
            Some(upper) if upper < LIMIT as u32 => match input {
                Register(255) => self.pairs(Set::from([0]), ceiling, u32::min),
                input => {
                    let thing = &self.file[self.tracked.slot(input)];
                    let unlearned = thing.stopped.iter().flat_map(|_| 0..=upper);
                    let learned = self.pairs(thing.values.clone(), ceiling, u32::min);
                    learned.into_iter().chain(unlearned).collect()
                }
            },
            Some(_) => self.stop(),
        }
    }
}

impl<'a> Making<'a> {
    /// A write at `at` reading the things in `file`.
    fn new(tracked: &'a Tracked, file: &'a [Value], cb1: &'a [u32], at: Place) -> Self {
        Self {
            tracked,
            file,
            cb1,
            at,
            stopped: Stops::new(),
        }
    }

    /// The thing `kind` writes to its first destination.
    fn made(mut self, kind: K) -> Value {
        let values = form(kind, &mut self);
        self.formed(values)
    }

    /// The thing formed here from `values` and what the reads met.
    fn formed(self, values: Set) -> Value {
        self.at.shapes(values, self.stopped)
    }
}

/// The value things one round makes, each from the things before it, in
/// program order. A forward edge's thing is made before the instruction it
/// enters; a backward edge carries its thing around to where it returns,
/// and that instruction is made again while what it carries grows.
struct Things<'a> {
    graph: &'a Graph,
    steps: &'a [Step],
    tracked: &'a Tracked,
    cb1: &'a [u32],
    width: usize,
    /// What enters each instruction from outside the program's edges: the
    /// program entry and every instruction no edge enters stop learning
    /// there.
    start: Vec<Value>,
    /// The things arriving at each instruction, as it was last made.
    ins: Vec<Value>,
    /// The things leaving each instruction.
    outs: Vec<Value>,
    /// The things backward edges carry to each instruction.
    carried: Vec<Value>,
}

impl<'a> Things<'a> {
    fn new(graph: &'a Graph, steps: &'a [Step], tracked: &'a Tracked, cb1: &'a [u32]) -> Self {
        let (len, width) = (steps.len(), tracked.0.len());
        let mut start = vec![Value::nothing(); len];
        let entries = (0..len).filter(|&at| graph.incoming[at] == 0);
        for at in once(0).chain(entries) {
            start[at] = Place(at).stops();
        }
        Self {
            graph,
            steps,
            tracked,
            cb1,
            width,
            start,
            ins: vec![Value::nothing(); len * width],
            outs: vec![Value::nothing(); len * width],
            carried: vec![Value::nothing(); len * width],
        }
    }

    /// The thing arriving at `at` in `slot`: from outside, carried around,
    /// and along every forward edge.
    fn arriving(&self, at: usize, slot: usize) -> Value {
        let (width, place) = (self.width, Place(at));
        let carried = self.carried[at * width + slot]
            .clone()
            .join(&self.start[at], place);
        self.graph.forward[at]
            .iter()
            .fold(carried, |joined, &from| {
                joined.join(&self.outs[from * width + slot], place)
            })
    }

    /// The things arriving at `at`.
    fn arrive(&self, at: usize, arriving: &mut Vec<Value>) {
        arriving.clear();
        arriving.extend((0..self.width).map(|slot| self.arriving(at, slot)));
    }

    /// The thing leaving `at` in `slot`, made from what last arrived there.
    fn leaving(&self, at: usize, slot: usize) -> Value {
        let (step, width) = (&self.steps[at], self.width);
        let ins = &self.ins[at * width..(at + 1) * width];
        let written = if step.formed.contains(&slot) {
            Making::new(self.tracked, ins, self.cb1, step.at).made(step.kind)
        } else if step.stopped.contains(&slot) {
            step.at.stops()
        } else {
            return ins[slot].clone();
        };
        if step.kept.contains(&slot) {
            ins[slot].clone().join(&written, step.at)
        } else {
            written
        }
    }

    /// Make every value thing. The least fixed point: an instruction waits,
    /// with the slots whose arriving thing may have changed, and the
    /// earliest waiting one is made first. A slot leaving it that changes
    /// waits on each instruction a forward edge enters, and on each one a
    /// backward edge returns to where what it carries grows. The register
    /// it forms reads every slot, so it is made each time. A thing made
    /// from what it was made from before is the same thing, so the things
    /// are those of making the program again from each return.
    fn make(&mut self) {
        let width = self.width;
        let steps = self.steps;
        let mut waiting = (0..steps.len())
            .map(|at| (at, (0..width).collect::<BTreeSet<_>>()))
            .collect::<BTreeMap<_, _>>();
        while let Some((at, slots)) = waiting.pop_first() {
            let out = at * width;
            for &slot in &slots {
                self.ins[out + slot] = self.arriving(at, slot);
            }
            let step = &steps[at];
            let made = slots.iter().chain(&step.formed).chain(&step.stopped);
            for slot in made.copied().collect::<BTreeSet<_>>() {
                let leaving = self.leaving(at, slot);
                if self.outs[out + slot] == leaving {
                    continue;
                }
                self.outs[out + slot] = leaving;
                for &to in &self.graph.onward[at] {
                    waiting.entry(to).or_default().insert(slot);
                }
                for &to in &self.graph.backward[at] {
                    let carried = &mut self.carried[to * width + slot];
                    let grown = carried.clone().join(&self.outs[out + slot], Place(to));
                    if grown != *carried {
                        *carried = grown;
                        waiting.entry(to).or_default().insert(slot);
                    }
                }
            }
        }
    }

    /// The thing the register BRX at `site` reads as its displacement.
    fn displacement(&self, site: &Site) -> Value {
        let mut arriving = Vec::with_capacity(self.width);
        self.arrive(site.at, &mut arriving);
        let mut making = Making::new(self.tracked, &arriving, self.cb1, Place(site.at));
        let values = making.register(site.displacement);
        making.formed(values)
    }
}

/// The table `values` makes at `site`, holding the entries an earlier
/// round proved there too, or the site's place on the track.
fn propose(mut values: Set, previous: &Set, site: &Site, into: &mut Proposal<'_>) {
    let Some(learned) = values.pop_first() else {
        // No value reaches the BRX from the program's edges.
        return into.track.record(into.xs, site.at, site.at);
    };
    // The table runs in ascending order: its first entry is the least of
    // the learned values and the entries proven before.
    let first = previous
        .iter()
        .fold(learned, |least, &entry| least.min(entry));
    values.insert(learned);
    values.extend(previous);
    values.remove(&first);
    if values.len() < LIMIT {
        into.tables.push(Proven {
            table: BrxTable::decode(site.base, first, values),
            base: site.base,
        });
    } else {
        into.track.record(into.xs, site.at, site.at);
    }
}

/// Propose edges from the displacement values, then prove every proposal
/// under their combined graph. Induction over actual execution establishes
/// that the first branch leaving this graph is impossible: its register
/// value was included. A BRX whose value is not learned may jump anywhere,
/// so it cannot justify narrowing another branch's values.
/// `rewritten` receives each BRX the converged proof rewrote, in order,
/// with its position and the table proven there.
#[cfg(test)]
pub(crate) fn resolve<F: FnMut(usize, &Instruction, Proven)>(
    xs: &mut [Instruction],
    cb1: &[u32],
    rewritten: F,
) {
    let slots = Slots::of(xs);
    let graph = Graph::of(&Program::of(xs), xs, &slots);
    resolve_in(crate::run::QuietRun, xs, &slots, &graph, cb1, rewritten);
}

/// [`resolve`] over the slots that place `xs`: the register BRXs left on
/// the discovery track. The proof explains itself where `run` sends it.
pub(crate) fn resolve_in<R: crate::run::Run, F: FnMut(usize, &Instruction, Proven)>(
    run: R,
    xs: &mut [Instruction],
    slots: &Slots,
    graph: &Graph,
    cb1: &[u32],
    rewritten: F,
) -> Track {
    run.brx(|output| resolve_into(xs, slots, graph, cb1, output, rewritten))
}

/// Each BRX on `track` reads a value whose shape is not learned.
pub(crate) fn write_unlearned(output: &mut impl Write, track: &Track) {
    for unlearned in track.unlearned() {
        let _ = writeln!(output, "SHADOWBOX_TRACE_STACK: BRX at byte 0x{:x} reads a value whose shape is not learned: {:?} at byte 0x{:x}", unlearned.brx, unlearned.shape, unlearned.at);
    }
}

/// The resolution with its trace written to `output`, for the decoder's tests.
#[cfg(test)]
pub(crate) fn resolve_traced(
    xs: &mut [Instruction],
    cb1: &[u32],
    output: Option<&mut dyn Write>,
    rewritten: impl FnMut(usize, &Instruction, Proven),
) {
    let mut sink = std::io::sink();
    let output = output
        .into_iter()
        .fold(&mut sink as &mut dyn Write, |_, output| output);
    let slots = Slots::of(xs);
    let graph = Graph::of(&Program::of(xs), xs, &slots);
    resolve_into(xs, &slots, &graph, cb1, output, rewritten);
}

fn resolve_into<F: FnMut(usize, &Instruction, Proven)>(
    xs: &mut [Instruction],
    slots: &Slots,
    graph: &Graph,
    cb1: &[u32],
    output: &mut dyn Write,
    mut rewritten: F,
) -> Track {
    let mut track = Track::default();
    let mut sites = Vec::new();
    for (at, x) in xs.iter().enumerate() {
        if let K::BrxRegister(ops::BrxRegister {
            displacement,
            base,
            cc_test,
        }) = x.kind
        {
            match (cc_test, x.predicate) {
                // CC.F and !PT cannot take the dynamic edge. Their typed
                // fallthrough needs no target discovery.
                (0, _)
                | (
                    _,
                    Some(crate::ir::Predicate {
                        register: 7,
                        inverted: true,
                    }),
                ) => {}
                // A conditional register BRX cannot be rewritten to Brx:
                // that IR has no condition state. Keep the typed instruction
                // and retain every real target for runtime dispatch instead.
                (1..=14 | 16..=31, _) => track.record(xs, at, at),
                (0xf, _) => sites.push(Site {
                    at,
                    displacement,
                    base,
                }),
                _ => unreachable!(),
            }
        }
    }
    if sites.is_empty() || !track.0.is_empty() {
        return track;
    }
    let tracked = Tracked::new(xs, &sites, cb1);
    let steps = xs
        .iter()
        .enumerate()
        .map(|(at, x)| Step::new(at, x, &tracked))
        .collect::<Vec<_>>();
    let mut tables = Vec::<Proven>::new();
    let mut previous = vec![Set::new(); sites.len()];
    // Each round proves under the previous round's tables; tables only
    // grow, so the rounds reach their fixed point.
    loop {
        let round = graph.tabled(&sites, &tables, slots);
        let mut things = Things::new(&round, &steps, &tracked, cb1);
        things.make();
        let mut next = Vec::with_capacity(sites.len());
        let mut proposal = Proposal {
            xs,
            tables: &mut next,
            track: &mut track,
        };
        for (site, previous) in sites.iter().zip(&previous) {
            // Each stop falls into the track, the values into a table. A
            // round whose track is not empty proposes nothing.
            let Value { values, stopped } = things.displacement(site);
            for &at in &stopped {
                proposal.track.record(xs, site.at, at);
            }
            propose(values, previous, site, &mut proposal);
        }
        if !track.0.is_empty() {
            track.0.sort_by_key(|unlearned| unlearned.brx);
            return track;
        }
        if next == tables {
            break;
        }
        for (previous, proven) in previous.iter_mut().zip(&next) {
            *previous = proven.table.entries().map(|t| t.loaded()).collect();
        }
        tables = next;
    }
    for (site, proven) in sites.iter().zip(tables) {
        let table = proven.table;
        let x = &mut xs[site.at];
        // Only the converged proof removes edges. The first
        // hypothetical execution leaving the recovered graph
        // must have a displacement included by its value.
        // BRX addition: InstEmitFlowControl.cs:38-40; saved cb1
        // targets: Decoder.cs:543-552. Log the entire permitted
        // set; its complement is precisely the excluded histories.
        let destinations = table.entries().map(|&t| ShownSlot(t)).collect::<Vec<_>>();
        let displacements = table.entries().map(|t| t.loaded()).collect::<Vec<_>>();
        let _ = writeln!(output, "SHADOWBOX_TRACE_STACK: prune BRX histories at byte 0x{:x} requiring targets outside {destinations:x?}: finite reaching-value proof under recovered graph, displacements {displacements:x?}, immutable cb1 snapshot {} words", x.address, cb1.len());
        x.kind = table.kind(site.displacement, site.base);
        rewritten(site.at, x, proven);
    }
    track
}
