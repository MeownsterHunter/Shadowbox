//! How an instruction moves control. Each instruction arrives at this
//! station once, where the program's words are decoded: its guard field
//! forms its guard type (see [`crate::guard`]) and its kind forms its
//! transfer type, with the condition-code test as its own type. What leaves
//! is the thing the pair is, and it persists in the [`Program`]. Discovery,
//! the stack proofs and the CFG builder run the behavior the thing carries;
//! no pass reads the guard field or a kind again.
//!
//! A proof that changes an instruction makes the next thing by consuming
//! the one before: a pop proven to one destination lowers into a
//! [`Reconverge`], and a push whose consumers are checked settles into a
//! [`ResolvedPush`]. No proof writes an instruction's kind: the program
//! writes each kind once, from the thing, when it leaves
//! ([`Program::write`]).
//!
//! CC.T (0xf) always holds (envytools envydis/gm107.c:1944-1945); every
//! other test may fail with the guard true (Ryujinx
//! InstEmitFlowControl.cs:153-186,295-312).

use crate::decoder::{Abort, Explore, Explores, Slots, Successor, Walk};
use crate::fast_hash::{HashMap, HashSet};
use crate::guard::{Guard, NotPt, Pt, Unguarded, When};
use crate::interface::Stopped;
use crate::ir::{ops, Kind};
use crate::ir::{
    BranchTarget, BrxBase, BrxEntry, Instruction, InstructionKind, OutsideBrxTable, SlotTable,
    TargetStack,
};
use crate::stack::{Ending, Mark, Marks, Unproven};
use crate::structure::{Build, Leave};

/// CC.T: the condition-code test always holds.
#[derive(Clone, Copy)]
pub(crate) struct CcT(());

/// A condition-code test other than T, by its encoding: it may fail.
#[derive(Clone, Copy)]
pub(crate) struct CcTest(u8);

/// What a condition-code test does in every pass that meets one.
pub(crate) trait Cc:
    Copy
    + Send
    + Sync
    + 'static
    + crate::decoder::CcFall
    + crate::decoder::CcExplore
    + crate::structure::CcEdge
    + crate::stack::Fall
{
    /// The test's encoding, as the emitter reads it.
    fn code(self) -> u8;
}

impl Cc for CcT {
    fn code(self) -> u8 {
        0xf
    }
}

impl Cc for CcTest {
    fn code(self) -> u8 {
        self.0
    }
}

/// BRA to `slot` under test `cc`.
#[derive(Clone, Copy)]
pub(crate) struct Jump<C> {
    slot: u32,
    cc: C,
}

/// A pop of `stack` proven to reach `slot`, under test `cc`.
#[derive(Clone, Copy)]
pub(crate) struct Reconverge<C> {
    slot: u32,
    cc: C,
    stack: TargetStack,
}

/// BRA whose target names no slot, for the reason `G`.
#[derive(Clone, Copy)]
pub(crate) struct NoSlotJump<C, G> {
    address: i64,
    cc: C,
    gap: G,
}

/// Why a target names no slot.
pub(crate) trait Gap: Copy + 'static {
    fn gap(self) -> crate::interface::TargetGap;
}

/// The target lies outside the code address space.
#[derive(Clone, Copy)]
pub(crate) struct OutsideCode(());

/// The target lies between instruction slots.
#[derive(Clone, Copy)]
pub(crate) struct Misaligned(());

impl Gap for OutsideCode {
    fn gap(self) -> crate::interface::TargetGap {
        crate::interface::TargetGap::OutsideCode
    }
}

impl Gap for Misaligned {
    fn gap(self) -> crate::interface::TargetGap {
        crate::interface::TargetGap::NotASlot
    }
}

/// BRX with a proven table: at least one entry, every one in the code
/// address space, and no more than a table holds. Only the BRX proof and
/// the station, for a program handed in from outside, make one.
#[derive(Clone, Copy)]
pub(crate) struct TableJump(SlotTable);

/// BRX with an entry outside the code address space.
#[derive(Clone, Copy)]
pub(crate) struct OutsideTable {
    base: BrxBase,
    table: OutsideBrxTable,
}

/// Register BRX: runtime target address.
#[derive(Clone, Copy)]
pub(crate) struct AnyJump<C>(C);

/// EXIT under test `cc`.
#[derive(Clone, Copy)]
pub(crate) struct Exit<C>(C);

/// An unresolved pop of `stack` under test `cc`.
#[derive(Clone, Copy)]
pub(crate) struct Pop<C> {
    stack: TargetStack,
    cc: C,
}

/// An unresolved SSY/PBK/PCNT push: the decoded op.
#[derive(Clone, Copy)]
pub(crate) struct Push<P>(P);

/// A push whose target names no slot: no pop can reach it.
#[derive(Clone, Copy)]
pub(crate) struct NoSlotPush(TargetStack);

/// A push whose consumers stack resolution checked.
#[derive(Clone, Copy)]
pub(crate) struct ResolvedPush<P>(P);

/// An encoding no modeled instruction has, or padding.
#[derive(Clone, Copy)]
pub(crate) struct Unknown(());

/// A relative CAL into the code: its function starts at `slot`.
#[derive(Clone, Copy)]
pub(crate) struct Call {
    slot: u32,
}

/// A relative CAL whose target names no slot.
#[derive(Clone, Copy)]
pub(crate) struct NoSlotCall(());

/// An instruction that continues to the next and writes no predicate.
#[derive(Clone, Copy)]
pub(crate) struct Step(());

/// An instruction that continues to the next and may write a predicate.
#[derive(Clone, Copy)]
pub(crate) struct PredicateWrite(());

impl<C: Copy> Jump<C> {
    pub(crate) fn slot(&self) -> u32 {
        self.slot
    }
    pub(crate) fn cc(&self) -> C {
        self.cc
    }
}

impl<C: Copy> Reconverge<C> {
    pub(crate) fn slot(&self) -> u32 {
        self.slot
    }
    pub(crate) fn cc(&self) -> C {
        self.cc
    }
    pub(crate) fn stack(&self) -> TargetStack {
        self.stack
    }
}

impl<C: Copy> AnyJump<C> {
    pub(crate) fn cc(&self) -> C {
        self.0
    }
}

impl<C, G: Gap> NoSlotJump<C, G> {
    pub(crate) fn address(&self) -> i64 {
        self.address
    }
    pub(crate) fn gap(&self) -> crate::interface::TargetGap {
        self.gap.gap()
    }
}

impl<C: Copy, G> NoSlotJump<C, G> {
    pub(crate) fn cc(&self) -> C {
        self.cc
    }
}

impl TableJump {
    /// The proven entries, in table order.
    pub(crate) fn targets(&self) -> &[BrxEntry] {
        self.0.targets()
    }
}

impl OutsideTable {
    pub(crate) fn base(&self) -> BrxBase {
        self.base
    }
    pub(crate) fn table(&self) -> &OutsideBrxTable {
        &self.table
    }
}

impl<C: Copy> Exit<C> {
    pub(crate) fn cc(&self) -> C {
        self.0
    }
}

impl<C: Copy> Pop<C> {
    pub(crate) fn cc(&self) -> C {
        self.cc
    }
    pub(crate) fn stack(&self) -> TargetStack {
        self.stack
    }
}

/// A decoded push op: the stack it uses and the target it pushes. Its
/// [`Kind::resolved`] is the kind it is once its consumers are checked.
pub(crate) trait PushOp: Kind + Send + Sync + 'static {
    fn stack(self) -> TargetStack;
    fn target(self) -> BranchTarget;
}

impl PushOp for ops::Schedule {
    fn stack(self) -> TargetStack {
        TargetStack::Sync
    }
    fn target(self) -> BranchTarget {
        self.target
    }
}

impl PushOp for ops::PushTarget {
    fn stack(self) -> TargetStack {
        self.stack
    }
    fn target(self) -> BranchTarget {
        self.target
    }
}

impl<P: PushOp> Push<P> {
    pub(crate) fn stack(&self) -> TargetStack {
        self.0.stack()
    }
    pub(crate) fn target(&self) -> BranchTarget {
        self.0.target()
    }
    pub(crate) fn slot(&self) -> u32 {
        self.0.target().slot()
    }
}

impl NoSlotPush {
    pub(crate) fn stack(&self) -> TargetStack {
        self.0
    }
}

impl<P: PushOp> ResolvedPush<P> {
    pub(crate) fn stack(&self) -> TargetStack {
        self.0.stack()
    }
    pub(crate) fn target(&self) -> BranchTarget {
        self.0.target()
    }
    pub(crate) fn slot(&self) -> u32 {
        self.0.target().slot()
    }
}

impl Call {
    pub(crate) fn slot(&self) -> u32 {
        self.slot
    }
}

/// A pop proven to reach one destination: what it becomes.
pub(crate) trait Lowers: Copy + 'static {
    type Cc: Cc;
    /// The reconvergence this pop becomes, reaching `slot`.
    fn lower(self, slot: u32) -> Reconverge<Self::Cc>;
}

impl<C: Cc> Lowers for Pop<C> {
    type Cc = C;
    fn lower(self, slot: u32) -> Reconverge<C> {
        Reconverge {
            slot,
            cc: self.cc,
            stack: self.stack,
        }
    }
}

/// A reconvergence proven again reaches the destination of that proof.
impl<C: Cc> Lowers for Reconverge<C> {
    type Cc = C;
    fn lower(self, slot: u32) -> Reconverge<C> {
        Reconverge { slot, ..self }
    }
}

impl<C: Cc> Reconverge<C> {
    /// The kind of this reconvergence standing at `address`.
    fn kind(self, address: u32) -> InstructionKind {
        InstructionKind::Reconverge(ops::Reconverge {
            target: BranchTarget::reaching(address, self.slot),
            cc_test: self.cc.code(),
            stack: self.stack,
        })
    }
}

/// Where stack diagnostics see blocks start.
pub(crate) trait Starts: Copy {
    /// The starts this transfer names besides its successors.
    fn starts(self, _: &Slots, _: &mut dyn FnMut(usize)) {}
    /// The instruction after this one, at `at`, starts a block.
    fn ends(self, _: usize, _: &mut dyn FnMut(usize)) {}
}

fn names(slot: u32, slots: &Slots, start: &mut dyn FnMut(usize)) {
    if let Some(target) = slots.get(slot) {
        start(target);
    }
}

macro_rules! terminators {
    ($($t:ty),*) => {$(
        impl Starts for $t {
            fn ends(self, at: usize, start: &mut dyn FnMut(usize)) {
                start(at + 1);
            }
        }
    )*};
}
terminators!(TableJump, OutsideTable, Call, NoSlotCall, Unknown);

impl<C: Cc> Starts for Jump<C> {
    fn starts(self, slots: &Slots, start: &mut dyn FnMut(usize)) {
        names(self.slot, slots, start);
    }
    fn ends(self, at: usize, start: &mut dyn FnMut(usize)) {
        start(at + 1);
    }
}

impl<C: Cc> Starts for Reconverge<C> {
    fn starts(self, slots: &Slots, start: &mut dyn FnMut(usize)) {
        names(self.slot, slots, start);
    }
    fn ends(self, at: usize, start: &mut dyn FnMut(usize)) {
        start(at + 1);
    }
}

impl<C: Cc, Y: Gap> Starts for NoSlotJump<C, Y> {
    fn ends(self, at: usize, start: &mut dyn FnMut(usize)) {
        start(at + 1);
    }
}

impl<C: Cc> Starts for Exit<C> {
    fn ends(self, at: usize, start: &mut dyn FnMut(usize)) {
        start(at + 1);
    }
}

impl<C: Cc> Starts for Pop<C> {
    fn ends(self, at: usize, start: &mut dyn FnMut(usize)) {
        start(at + 1);
    }
}

impl<P: PushOp> Starts for Push<P> {
    fn starts(self, slots: &Slots, start: &mut dyn FnMut(usize)) {
        names(self.slot(), slots, start);
    }
}

impl<P: PushOp> Starts for ResolvedPush<P> {
    fn starts(self, slots: &Slots, start: &mut dyn FnMut(usize)) {
        names(self.slot(), slots, start);
    }
}

impl<C: Cc> Starts for AnyJump<C> {}
impl Starts for NoSlotPush {}
impl Starts for Step {}
impl Starts for PredicateWrite {}

/// What a transfer does in every pass that meets one.
pub(crate) trait Transfer:
    Copy
    + Send
    + Sync
    + 'static
    + Walk
    + Leave
    + Mark
    + Explores
    + Starts
    + Column
    + Proves
    + Writes
    + crate::brx::Edges
{
}
impl<T> Transfer for T where
    T: Copy
        + Send
        + Sync
        + 'static
        + Walk
        + Leave
        + Mark
        + Explores
        + Starts
        + Column
        + Proves
        + Writes
        + crate::brx::Edges
{
}

/// One thing standing in the program: where it stands, as the guard and
/// transfer it arrived as.
#[derive(Clone, Copy)]
pub(crate) struct Seat<A, G, T> {
    at: A,
    guard: G,
    transfer: T,
}

impl<A: Copy, G: Copy, T: Copy> Seat<A, G, T> {
    /// Where the thing stands.
    pub(crate) fn at(&self) -> A {
        self.at
    }

    /// The guard it arrived under.
    pub(crate) fn guard(&self) -> G {
        self.guard
    }

    /// What it does.
    pub(crate) fn transfer(&self) -> T {
        self.transfer
    }
}

/// Where a discovered thing stands before discovery orders the program:
/// the byte address of its instruction.
#[derive(Clone, Copy)]
pub(crate) struct Address(u32);

/// A pass that meets every seated thing as its own type.
pub(crate) trait Meet<'p, A> {
    fn meet<G: Guard, T: Transfer>(&mut self, seat: &'p Seat<A, G, T>);
}

/// Where a transfer's things sit among the things of one guard.
pub(crate) trait Column: Sized {
    fn column<A, G>(things: &mut Things<A, G>) -> &mut Vec<Seat<A, G, Self>>;
}

/// Where a guard's things sit in the program.
pub(crate) trait Stands: Sized {
    fn things<A>(columns: &mut Columns<A>) -> &mut Things<A, Self>;
}

macro_rules! columns {
    ($($field:ident: $t:ty),* $(,)?) => {
        /// The things of one guard: each transfer's things sit together,
        /// in a column of their own type.
        pub(crate) struct Things<A, G> {
            $($field: Vec<Seat<A, G, $t>>,)*
        }

        impl<A, G> Default for Things<A, G> {
            fn default() -> Self {
                Self { $($field: Vec::new(),)* }
            }
        }

        $(impl Column for $t {
            fn column<A, G>(things: &mut Things<A, G>) -> &mut Vec<Seat<A, G, Self>> {
                &mut things.$field
            }
        })*

        impl<A: Copy, G: Guard> Things<A, G> {
            fn each<'p>(&'p self, meet: &mut impl Meet<'p, A>) {
                $(self.$field.iter().for_each(|seat| meet.meet(seat));)*
            }

            fn clear(&mut self) {
                $(self.$field.clear();)*
            }

            /// Every thing seated again in `into`, standing where `place`
            /// puts it.
            fn seat_into<B>(&self, into: &mut Things<B, G>, place: &impl Fn(A) -> B) {
                $(into.$field.extend(self.$field.iter().map(|seat| Seat {
                    at: place(seat.at),
                    guard: seat.guard,
                    transfer: seat.transfer,
                }));)*
            }
        }

        impl<G: Guard> Things<usize, G> {
            /// The thing at `at` leaves: a new instruction stands there.
            fn leave(&mut self, at: usize) {
                $(self.$field.retain(|seat| seat.at != at);)*
            }
        }
    };
}

columns! {
    jump_t: Jump<CcT>,
    jump: Jump<CcTest>,
    reconverge_t: Reconverge<CcT>,
    reconverge: Reconverge<CcTest>,
    outside_jump_t: NoSlotJump<CcT, OutsideCode>,
    outside_jump: NoSlotJump<CcTest, OutsideCode>,
    misaligned_jump_t: NoSlotJump<CcT, Misaligned>,
    misaligned_jump: NoSlotJump<CcTest, Misaligned>,
    table: TableJump,
    outside_table: OutsideTable,
    any_t: AnyJump<CcT>,
    any: AnyJump<CcTest>,
    exit_t: Exit<CcT>,
    exit: Exit<CcTest>,
    pop_t: Pop<CcT>,
    pop: Pop<CcTest>,
    schedule: Push<ops::Schedule>,
    push_target: Push<ops::PushTarget>,
    outside_push: NoSlotPush,
    resolved_schedule: ResolvedPush<ops::Schedule>,
    resolved_push_target: ResolvedPush<ops::PushTarget>,
    call: Call,
    outside_call: NoSlotCall,
    unknown: Unknown,
    step: Step,
    predicate_write: PredicateWrite,
}

/// The things of every guard.
pub(crate) struct Columns<A> {
    unguarded: Things<A, Unguarded>,
    pt: Things<A, Pt>,
    not_pt: Things<A, NotPt>,
    when: Things<A, When>,
}

impl<A> Default for Columns<A> {
    fn default() -> Self {
        Self {
            unguarded: Things::default(),
            pt: Things::default(),
            not_pt: Things::default(),
            when: Things::default(),
        }
    }
}

macro_rules! stands {
    ($($guard:ident: $field:ident),*) => {$(
        impl Stands for $guard {
            fn things<A>(columns: &mut Columns<A>) -> &mut Things<A, Self> {
                &mut columns.$field
            }
        }
    )*};
}
stands!(Unguarded: unguarded, Pt: pt, NotPt: not_pt, When: when);

impl<A: Copy> Columns<A> {
    fn each<'p>(&'p self, meet: &mut impl Meet<'p, A>) {
        self.unguarded.each(meet);
        self.pt.each(meet);
        self.not_pt.each(meet);
        self.when.each(meet);
    }

    fn clear(&mut self) {
        self.unguarded.clear();
        self.pt.clear();
        self.not_pt.clear();
        self.when.clear();
    }

    fn seat<G: Guard, T: Transfer>(&mut self, seat: Seat<A, G, T>) {
        T::column(G::things(self)).push(seat);
    }

    fn seat_into<B>(&self, into: &mut Columns<B>, place: &impl Fn(A) -> B) {
        self.unguarded.seat_into(&mut into.unguarded, place);
        self.pt.seat_into(&mut into.pt, place);
        self.not_pt.seat_into(&mut into.not_pt, place);
        self.when.seat_into(&mut into.when, place);
    }
}

/// Seats the thing an instruction arrives as at `at`; `visit` meets each
/// successor of the thing from `address`.
struct SeatAt<'c, A, V> {
    at: A,
    columns: &'c mut Columns<A>,
    address: u32,
    visit: V,
}

impl<A: Copy, V: FnMut(Successor)> Meets for SeatAt<'_, A, V> {
    type Out = ();
    fn meets<G: Guard, T: Transfer>(mut self, guard: G, transfer: T) {
        transfer.walk(self.address, guard, &mut self.visit);
        self.columns.seat(Seat {
            at: self.at,
            guard,
            transfer,
        });
    }
}

/// A proven table whose guard is forming, and where it is seated.
struct Tabled<'c, V> {
    table: crate::brx::Proven,
    seat: SeatAt<'c, usize, V>,
}

impl<V: FnMut(Successor)> crate::guard::Arrive for Tabled<'_, V> {
    type Out = ();
    fn arrive(self, guard: impl Guard) {
        let base = self.table.base();
        self.table.meets(Shaped {
            guard,
            base,
            seat: self.seat,
        });
    }
}

/// A guarded proven table meets the shape it is.
struct Shaped<'c, G, V> {
    guard: G,
    base: BrxBase,
    seat: SeatAt<'c, usize, V>,
}

/// A proven table holds at least one entry, so one whose every
/// destination is a slot is a [`TableJump`].
impl<G: Guard, V: FnMut(Successor)> crate::ir::TableMeets for Shaped<'_, G, V> {
    type Out = ();
    fn slots(self, table: SlotTable) {
        self.seat.meets(self.guard, TableJump(table));
    }
    fn outside(self, table: OutsideBrxTable) {
        let base = self.base;
        self.seat.meets(self.guard, OutsideTable { base, table });
    }
}

/// The things discovery decoded, each standing at its address until
/// discovery orders the program.
#[derive(Default)]
pub(crate) struct Found(Columns<Address>);

/// Walks the thing `x` arrives as.
#[cfg(test)]
struct Walked<V>(u32, V);

#[cfg(test)]
impl<V: FnMut(Successor)> Meets for Walked<V> {
    type Out = ();
    fn meets<G: Guard, T: Transfer>(mut self, guard: G, transfer: T) {
        transfer.walk(self.0, guard, &mut self.1);
    }
}

/// `visit` meets each successor of the thing `x` arrives as.
#[cfg(test)]
pub(crate) fn walk(x: &Instruction, visit: impl FnMut(Successor)) {
    arrive(x, Walked(x.address, visit));
}

impl Found {
    pub(crate) fn clear(&mut self) {
        self.0.clear();
    }

    /// The decoded `x` arrives at the station once: `visit` meets each
    /// successor of the thing it is, and the thing stands found.
    pub(crate) fn arrive(&mut self, x: &Instruction, visit: impl FnMut(Successor)) {
        arrive(
            x,
            SeatAt {
                at: Address(x.address),
                columns: &mut self.0,
                address: x.address,
                visit,
            },
        );
    }
}

/// Pops proven to reach one destination: each pop's position and the
/// position it reaches.
pub(crate) struct Reached(pub(crate) HashMap<usize, usize>);

/// Pushes whose consumers a proof checked, by position.
pub(crate) struct Consumed(pub(crate) HashSet<usize>);

/// The program's control flow: each instruction as the thing it arrived
/// as, standing at its position. Things alike sit together: a waiting pop
/// is a [`Pop`], a proven one a [`Reconverge`]; an unchecked push is a
/// [`Push`], a checked one a [`ResolvedPush`].
#[derive(Default)]
pub(crate) struct Program {
    seats: Columns<usize>,
    len: usize,
}

impl Program {
    /// The program `xs` that arrives from outside: each instruction meets
    /// the station once.
    pub(crate) fn of(xs: &[Instruction]) -> Self {
        let mut program = Self {
            seats: Columns::default(),
            len: xs.len(),
        };
        for (at, x) in xs.iter().enumerate() {
            arrive(
                x,
                SeatAt {
                    at,
                    columns: &mut program.seats,
                    address: x.address,
                    visit: |_| {},
                },
            );
        }
        program
    }

    /// `meet` meets every thing in the program as its own type.
    pub(crate) fn each<'p>(&'p self, meet: &mut impl Meet<'p, usize>) {
        self.seats.each(meet);
    }

    /// The program of the things `found`, ordered as `xs`, the decoded
    /// instructions sorted by address.
    pub(crate) fn order(&mut self, found: &Found, xs: &[Instruction]) {
        self.seats.clear();
        self.len = xs.len();
        found.0.seat_into(&mut self.seats, &|Address(address)| {
            xs.partition_point(|x| x.address < address)
        });
    }

    /// `x`, a new instruction a proof made, stands at `at` in place of the
    /// one before.
    pub(crate) fn replace(&mut self, at: usize, x: &Instruction) {
        self.leave(at);
        arrive(
            x,
            SeatAt {
                at,
                columns: &mut self.seats,
                address: x.address,
                visit: |_| {},
            },
        );
    }

    /// The thing at `at` leaves: a new one takes its place.
    fn leave(&mut self, at: usize) {
        self.seats.unguarded.leave(at);
        self.seats.pt.leave(at);
        self.seats.not_pt.leave(at);
        self.seats.when.leave(at);
    }

    /// The BRX at `at`, `x` as the proof rewrote it, is the table the
    /// proof made there: `visit` meets each successor of the new thing.
    pub(crate) fn prove(
        &mut self,
        at: usize,
        x: &Instruction,
        table: crate::brx::Proven,
        visit: impl FnMut(Successor),
    ) {
        self.leave(at);
        crate::guard::arrive(
            x.predicate,
            Tabled {
                table,
                seat: SeatAt {
                    at,
                    columns: &mut self.seats,
                    address: x.address,
                    visit,
                },
            },
        );
    }

    /// The number of instructions.
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// The things in program order, for the passes that walk positions.
    pub(crate) fn line(&self) -> Line<'_> {
        let mut lining = Lining(Vec::with_capacity(self.len));
        self.seats.each(&mut lining);
        lining.0.sort_unstable_by_key(|&(at, _)| at);
        Line(lining.0.into_iter().map(|(_, thing)| thing).collect())
    }

    /// The pops no proof has lowered yet.
    pub(crate) fn pops(&self) -> usize {
        let mut count = Count(0);
        self.each_pop(&mut count);
        count.0
    }

    /// The pops no proof has lowered yet, in program order, with the stack
    /// each reads.
    pub(crate) fn waiting(&self) -> Vec<(usize, TargetStack)> {
        let mut waiting = Waiting(Vec::new());
        self.each_pop(&mut waiting);
        waiting.0.sort_unstable_by_key(|&(at, _)| at);
        waiting.0
    }

    fn each_pop<W: Wait>(&self, wait: &mut W) {
        fn of<G, W: Wait>(things: &Things<usize, G>, wait: &mut W) {
            things
                .pop_t
                .iter()
                .for_each(|s| wait.pop(s.at, s.transfer.stack));
            things
                .pop
                .iter()
                .for_each(|s| wait.pop(s.at, s.transfer.stack));
        }
        of(&self.seats.unguarded, wait);
        of(&self.seats.pt, wait);
        of(&self.seats.not_pt, wait);
        of(&self.seats.when, wait);
    }

    /// The pushes whose consumers are not checked yet, outside ones
    /// included.
    pub(crate) fn pushes(&self) -> usize {
        fn of<G>(things: &Things<usize, G>) -> usize {
            things.schedule.len() + things.push_target.len() + things.outside_push.len()
        }
        of(&self.seats.unguarded)
            + of(&self.seats.pt)
            + of(&self.seats.not_pt)
            + of(&self.seats.when)
    }

    /// Every pop or reconvergence `reached` proves lowers into the
    /// reconvergence it is now; the rest still wait. `xs` places each
    /// destination.
    pub(crate) fn lower(&mut self, reached: &Reached, xs: &[Instruction]) {
        fn of<G: Guard>(things: &mut Things<usize, G>, reached: &Reached, xs: &[Instruction]) {
            relower(&mut things.reconverge_t, reached, xs);
            relower(&mut things.reconverge, reached, xs);
            lower(&mut things.pop_t, &mut things.reconverge_t, reached, xs);
            lower(&mut things.pop, &mut things.reconverge, reached, xs);
        }
        of(&mut self.seats.unguarded, reached, xs);
        of(&mut self.seats.pt, reached, xs);
        of(&mut self.seats.not_pt, reached, xs);
        of(&mut self.seats.when, reached, xs);
    }

    /// The pushes `consumed` names, their consumers checked.
    pub(crate) fn settle(&mut self, consumed: &Consumed) {
        self.settle_where(|seat| consumed.0.contains(&seat.at));
    }

    /// Every waiting push, its consumers checked.
    pub(crate) fn settle_pushes(&mut self) {
        self.settle_where(|_| true);
    }

    /// Every waiting push whose target `slots` hold, its consumers checked.
    pub(crate) fn settle_held(&mut self, slots: &Slots) {
        self.settle_where(|seat| slots.get(seat.slot).is_some());
    }

    fn settle_where(&mut self, checked: impl Fn(Waits) -> bool) {
        fn of<G: Guard>(things: &mut Things<usize, G>, checked: &impl Fn(Waits) -> bool) {
            settle(&mut things.schedule, &mut things.resolved_schedule, checked);
            settle(
                &mut things.push_target,
                &mut things.resolved_push_target,
                checked,
            );
        }
        of(&mut self.seats.unguarded, &checked);
        of(&mut self.seats.pt, &checked);
        of(&mut self.seats.not_pt, &checked);
        of(&mut self.seats.when, &checked);
    }

    /// The linear reference's proofs carried at each address both programs
    /// hold: a pop it lowered lowers here too, and a push it settled
    /// settles here. Each pop still waiting here says where it now stands.
    pub(crate) fn carry(
        &mut self,
        xs: &[Instruction],
        linear: &Program,
        linear_xs: &[Instruction],
    ) -> Vec<(usize, Fallback)> {
        let mut proofs = Proofs {
            xs: linear_xs,
            pops: HashMap::default(),
            settled: HashSet::default(),
        };
        linear.seats.each(&mut proofs);
        fn of<G: Guard>(
            things: &mut Things<usize, G>,
            xs: &[Instruction],
            proofs: &Proofs,
            fallbacks: &mut Vec<(usize, Fallback)>,
        ) {
            agree(&things.reconverge_t, xs, proofs);
            agree(&things.reconverge, xs, proofs);
            carry(
                &mut things.pop_t,
                &mut things.reconverge_t,
                xs,
                proofs,
                fallbacks,
            );
            carry(
                &mut things.pop,
                &mut things.reconverge,
                xs,
                proofs,
                fallbacks,
            );
            let checked = &|seat: Waits| proofs.settled.contains(&xs[seat.at].address);
            settle(&mut things.schedule, &mut things.resolved_schedule, checked);
            settle(
                &mut things.push_target,
                &mut things.resolved_push_target,
                checked,
            );
        }
        let mut fallbacks = Vec::new();
        of(&mut self.seats.unguarded, xs, &proofs, &mut fallbacks);
        of(&mut self.seats.pt, xs, &proofs, &mut fallbacks);
        of(&mut self.seats.not_pt, xs, &proofs, &mut fallbacks);
        of(&mut self.seats.when, xs, &proofs, &mut fallbacks);
        fallbacks.sort_unstable_by_key(|&(at, _)| at);
        fallbacks
    }

    /// The program leaves: each instruction of `xs` takes the kind of the
    /// thing it is now.
    pub(crate) fn write(&self, xs: &mut [Instruction]) {
        self.seats.each(&mut Writing(xs));
    }
}

/// A waiting push: where it stands and the slot it pushes.
#[derive(Clone, Copy)]
struct Waits {
    at: usize,
    slot: u32,
}

/// Writes each thing's kind into its instruction.
struct Writing<'x>(&'x mut [Instruction]);

impl<'p> Meet<'p, usize> for Writing<'_> {
    fn meet<G: Guard, T: Transfer>(&mut self, seat: &'p Seat<usize, G, T>) {
        seat.transfer.write(&mut self.0[seat.at]);
    }
}

/// The kind a thing leaves the program as, where a proof made it.
pub(crate) trait Writes: Sized {
    /// Any other thing is the kind it arrived from.
    fn write(self, _: &mut Instruction) {}
}

impl<C: Cc> Writes for Reconverge<C> {
    fn write(self, x: &mut Instruction) {
        x.kind = self.kind(x.address);
    }
}

impl<P: PushOp> Writes for ResolvedPush<P> {
    fn write(self, x: &mut Instruction) {
        x.kind = self.0.resolved();
    }
}

macro_rules! writes_nothing {
    ($($t:ty),*) => {$(impl Writes for $t {})*};
}
writes_nothing!(
    TableJump,
    OutsideTable,
    NoSlotPush,
    Call,
    NoSlotCall,
    Unknown,
    Step,
    PredicateWrite
);
impl<C: Cc> Writes for AnyJump<C> {}
impl<C: Cc> Writes for Jump<C> {}
impl<C: Cc, Y: Gap> Writes for NoSlotJump<C, Y> {}
impl<C: Cc> Writes for Exit<C> {}
impl<C: Cc> Writes for Pop<C> {}
impl<P: PushOp> Writes for Push<P> {}

/// What the linear fallback did with a pop the reachable proof left waiting.
pub(crate) enum Fallback {
    /// It lowered the pop into this kind.
    Resolves(Box<InstructionKind>),
    /// It left the pop waiting too.
    Leaves,
}

/// The linear reference's proofs, by address: what each of its pops is,
/// and each push whose consumers it checked.
pub(crate) struct Proofs<'x> {
    xs: &'x [Instruction],
    pops: HashMap<u32, LinearPop>,
    settled: HashSet<u32>,
}

/// A pop of the linear reference.
#[derive(Clone, Copy)]
enum LinearPop {
    /// Proven to reach this slot.
    Reached(u32),
    /// Still waiting.
    Waits,
}

impl<'p> Meet<'p, usize> for Proofs<'_> {
    fn meet<G: Guard, T: Transfer>(&mut self, seat: &'p Seat<usize, G, T>) {
        seat.transfer.proves(self.xs[seat.at], self);
    }
}

/// What a thing of the linear reference proves at its instruction.
pub(crate) trait Proves: Sized {
    fn proves(self, _: Instruction, _: &mut Proofs<'_>) {}
}

impl<C: Cc> Proves for Reconverge<C> {
    fn proves(self, x: Instruction, proofs: &mut Proofs<'_>) {
        proofs.pops.insert(x.address, LinearPop::Reached(self.slot));
    }
}

impl<P: PushOp> Proves for ResolvedPush<P> {
    fn proves(self, x: Instruction, proofs: &mut Proofs<'_>) {
        proofs.settled.insert(x.address);
    }
}

macro_rules! proves_nothing {
    ($($t:ty),*) => {$(impl Proves for $t {})*};
}
proves_nothing!(
    TableJump,
    OutsideTable,
    NoSlotPush,
    Call,
    NoSlotCall,
    Unknown,
    Step,
    PredicateWrite
);
impl<C: Cc> Proves for AnyJump<C> {}
impl<C: Cc> Proves for Jump<C> {}
impl<C: Cc, Y: Gap> Proves for NoSlotJump<C, Y> {}
impl<C: Cc> Proves for Exit<C> {}
impl<C: Cc> Proves for Pop<C> {
    fn proves(self, x: Instruction, proofs: &mut Proofs<'_>) {
        proofs.pops.insert(x.address, LinearPop::Waits);
    }
}
impl<P: PushOp> Proves for Push<P> {}

/// A proven pop here agrees with the linear reference's proof of it.
fn agree<G, C: Cc>(proven: &[Seat<usize, G, Reconverge<C>>], xs: &[Instruction], proofs: &Proofs) {
    for seat in proven {
        let address = xs[seat.at].address;
        if let Some(&LinearPop::Reached(slot)) = proofs.pops.get(&address) {
            debug_assert_eq!(
                seat.transfer.slot, slot,
                "stack resolvers disagree at byte 0x{:x}",
                address
            );
        }
    }
}

/// Each pop still waiting meets the linear reference's proof at its address.
fn carry<G: Guard, C: Cc>(
    pops: &mut Vec<Seat<usize, G, Pop<C>>>,
    into: &mut Vec<Seat<usize, G, Reconverge<C>>>,
    xs: &[Instruction],
    proofs: &Proofs,
    fallbacks: &mut Vec<(usize, Fallback)>,
) {
    for seat in std::mem::take(pops) {
        let address = xs[seat.at].address;
        match proofs.pops.get(&address) {
            Some(&LinearPop::Reached(slot)) => {
                let lowered = seat.lower(slot);
                let kind = lowered.transfer.kind(address);
                fallbacks.push((seat.at, Fallback::Resolves(Box::new(kind))));
                into.push(lowered);
            }
            Some(LinearPop::Waits) => {
                fallbacks.push((seat.at, Fallback::Leaves));
                pops.push(seat);
            }
            None => pops.push(seat),
        }
    }
}

impl<G: Guard, P: Lowers> Seat<usize, G, P> {
    /// This pop, proven to reach `slot`: the reconvergence it is now.
    fn lower(self, slot: u32) -> Seat<usize, G, Reconverge<P::Cc>> {
        Seat {
            at: self.at,
            guard: self.guard,
            transfer: self.transfer.lower(slot),
        }
    }
}

/// Each waiting pop meets `reached`: proven, it is a reconvergence now.
fn lower<G: Guard, C: Cc>(
    pops: &mut Vec<Seat<usize, G, Pop<C>>>,
    into: &mut Vec<Seat<usize, G, Reconverge<C>>>,
    reached: &Reached,
    xs: &[Instruction],
) {
    for seat in std::mem::take(pops) {
        match reached.0.get(&seat.at) {
            Some(&to) => into.push(seat.lower(xs[to].address)),
            None => pops.push(seat),
        }
    }
}

/// Each reconvergence `reached` proves again reaches that proof's
/// destination.
fn relower<G: Guard, C: Cc>(
    proven: &mut [Seat<usize, G, Reconverge<C>>],
    reached: &Reached,
    xs: &[Instruction],
) {
    for seat in proven {
        if let Some(&to) = reached.0.get(&seat.at) {
            *seat = seat.lower(xs[to].address);
        }
    }
}

/// Each waiting push `checked` names settles into a checked push.
fn settle<G: Guard, P: PushOp>(
    pushes: &mut Vec<Seat<usize, G, Push<P>>>,
    into: &mut Vec<Seat<usize, G, ResolvedPush<P>>>,
    checked: &impl Fn(Waits) -> bool,
) {
    for seat in std::mem::take(pushes) {
        let waits = Waits {
            at: seat.at,
            slot: seat.transfer.slot(),
        };
        match checked(waits) {
            true => into.push(Seat {
                at: seat.at,
                guard: seat.guard,
                transfer: ResolvedPush(seat.transfer.0),
            }),
            false => pushes.push(seat),
        }
    }
}

/// What meets each waiting pop.
trait Wait {
    fn pop(&mut self, at: usize, stack: TargetStack);
}

struct Count(usize);

impl Wait for Count {
    fn pop(&mut self, _: usize, _: TargetStack) {
        self.0 += 1;
    }
}

struct Waiting(Vec<(usize, TargetStack)>);

impl Wait for Waiting {
    fn pop(&mut self, at: usize, stack: TargetStack) {
        self.0.push((at, stack));
    }
}

/// An instruction's control flow, as the passes that walk positions meet
/// it.
pub(crate) trait Flow {
    /// Where control goes from `address` during discovery.
    fn walk(&self, address: u32, visit: &mut dyn FnMut(Successor));
    /// [`Self::walk`] where a runtime target may reach any word.
    fn reach(&self, address: u32, visit: &mut dyn FnMut(Successor)) -> Result<(), Stopped>;
    /// The CFG edges of the instruction, followed by the next one.
    fn leave(&self, build: &mut Build) -> Result<(), Stopped>;
    /// The CFG edges of the program's last instruction.
    fn leave_last(&self, build: &mut Build) -> Result<(), Stopped>;
    /// What the stack block proof marks here, and how its block would end.
    fn mark(&self, marks: &mut Marks) -> Result<Ending, Unproven>;
    /// One path of an ordered stack exploration through this instruction.
    fn explore(&self, explore: &mut dyn Explore) -> Result<(), Abort>;
    /// The block starts stack diagnostics see here.
    fn starts(&self, slots: &Slots, start: &mut dyn FnMut(usize));
}

impl<G: Guard, T: Transfer> Flow for Seat<usize, G, T> {
    fn walk(&self, address: u32, visit: &mut dyn FnMut(Successor)) {
        self.transfer.walk(address, self.guard, visit);
    }
    fn reach(&self, address: u32, visit: &mut dyn FnMut(Successor)) -> Result<(), Stopped> {
        self.transfer.reach(address, self.guard, visit)
    }
    fn leave(&self, build: &mut Build) -> Result<(), Stopped> {
        self.transfer.leave(
            self.at,
            self.guard,
            crate::structure::Follows(self.at + 1),
            build,
        )
    }
    fn leave_last(&self, build: &mut Build) -> Result<(), Stopped> {
        self.transfer
            .leave(self.at, self.guard, crate::structure::End, build)
    }
    fn mark(&self, marks: &mut Marks) -> Result<Ending, Unproven> {
        self.transfer.mark(self.at, self.guard, marks)
    }
    fn explore(&self, explore: &mut dyn Explore) -> Result<(), Abort> {
        self.transfer.explore(self.guard, explore)
    }
    fn starts(&self, slots: &Slots, start: &mut dyn FnMut(usize)) {
        Starts::starts(self.transfer, slots, start);
        Starts::ends(self.transfer, self.at, start);
    }
}

/// The program's things in program order: the track the passes that walk
/// positions follow.
pub(crate) struct Line<'p>(Vec<&'p dyn Flow>);

impl<'p> Line<'p> {
    /// The thing at `at`.
    pub(crate) fn at(&self, at: usize) -> &'p dyn Flow {
        self.0[at]
    }

    /// Every thing, in program order.
    pub(crate) fn things(&self) -> impl ExactSizeIterator<Item = &'p dyn Flow> + '_ {
        self.0.iter().copied()
    }

    /// The last thing, and every thing before it.
    pub(crate) fn split_last(&self) -> Option<(&&'p dyn Flow, &[&'p dyn Flow])> {
        self.0.split_last()
    }
}

struct Lining<'p>(Vec<(usize, &'p dyn Flow)>);

impl<'p> Meet<'p, usize> for Lining<'p> {
    fn meet<G: Guard, T: Transfer>(&mut self, seat: &'p Seat<usize, G, T>) {
        self.0.push((seat.at, seat));
    }
}

/// A pass that meets the thing an instruction arrives as.
pub(crate) trait Meets {
    type Out;
    fn meets<G: Guard, T: Transfer>(self, guard: G, transfer: T) -> Self::Out;
}

/// The station: `x` arrives, and `meets` meets the thing its guard and
/// kind form, as its own type. Only the decode edge, a proof that makes a
/// new instruction, and a program handed in from outside call it.
pub(crate) fn arrive<M: Meets>(x: &Instruction, meets: M) -> M::Out {
    crate::guard::arrive(
        x.predicate,
        Kinded {
            kind: &x.kind,
            meets,
        },
    )
}

/// An instruction whose guard has formed, waiting for its transfer, and
/// the pass that meets the thing it is.
struct Kinded<'x, M> {
    kind: &'x InstructionKind,
    meets: M,
}

impl<M: Meets> crate::guard::Arrive for Kinded<'_, M> {
    type Out = M::Out;
    fn arrive(self, guard: impl Guard) -> M::Out {
        let meets = self.meets;
        match *self.kind {
            InstructionKind::Branch(ops::Branch {
                target,
                cc_test: 0xf,
            }) => meets.meets(
                guard,
                Jump {
                    slot: target.slot(),
                    cc: CcT(()),
                },
            ),
            InstructionKind::Branch(ops::Branch { target, cc_test }) => meets.meets(
                guard,
                Jump {
                    slot: target.slot(),
                    cc: CcTest(cc_test),
                },
            ),
            InstructionKind::Reconverge(ops::Reconverge {
                target,
                cc_test: 0xf,
                stack,
            }) => meets.meets(
                guard,
                Reconverge {
                    slot: target.slot(),
                    cc: CcT(()),
                    stack,
                },
            ),
            InstructionKind::Reconverge(ops::Reconverge {
                target,
                cc_test,
                stack,
            }) => meets.meets(
                guard,
                Reconverge {
                    slot: target.slot(),
                    cc: CcTest(cc_test),
                    stack,
                },
            ),
            InstructionKind::OutsideBranch(ops::OutsideBranch {
                target,
                cc_test: 0xf,
            }) => meets.meets(
                guard,
                NoSlotJump {
                    address: target.address(),
                    cc: CcT(()),
                    gap: OutsideCode(()),
                },
            ),
            InstructionKind::OutsideBranch(ops::OutsideBranch { target, cc_test }) => meets.meets(
                guard,
                NoSlotJump {
                    address: target.address(),
                    cc: CcTest(cc_test),
                    gap: OutsideCode(()),
                },
            ),
            InstructionKind::MisalignedBranch(ops::MisalignedBranch {
                target,
                cc_test: 0xf,
            }) => meets.meets(
                guard,
                NoSlotJump {
                    address: i64::from(target.address()),
                    cc: CcT(()),
                    gap: Misaligned(()),
                },
            ),
            InstructionKind::MisalignedBranch(ops::MisalignedBranch { target, cc_test }) => meets
                .meets(
                    guard,
                    NoSlotJump {
                        address: i64::from(target.address()),
                        cc: CcTest(cc_test),
                        gap: Misaligned(()),
                    },
                ),
            // CC.F is a typed fallthrough, not an arbitrary jump. Keeping it
            // out of AnyJump prevents false target and stack/reconvergence
            // edges for the exact never-taken semantic slice.
            InstructionKind::BrxRegister(ops::BrxRegister { cc_test: 0, .. }) => {
                meets.meets(guard, Step(()))
            }
            InstructionKind::BrxRegister(ops::BrxRegister { cc_test: 0xf, .. }) => {
                meets.meets(guard, AnyJump(CcT(())))
            }
            InstructionKind::BrxRegister(ops::BrxRegister { cc_test, .. }) => {
                meets.meets(guard, AnyJump(CcTest(cc_test)))
            }
            InstructionKind::Brx(ops::Brx { table, .. }) => meets.meets(guard, TableJump(table)),
            InstructionKind::OutsideBrx(ops::OutsideBrx { base, table, .. }) => {
                meets.meets(guard, OutsideTable { base, table })
            }
            InstructionKind::Exit(ops::Exit { cc_test: 0xf }) => meets.meets(guard, Exit(CcT(()))),
            InstructionKind::Exit(ops::Exit { cc_test }) => {
                meets.meets(guard, Exit(CcTest(cc_test)))
            }
            InstructionKind::PopJump(ops::PopJump {
                stack,
                cc_test: 0xf,
            }) => meets.meets(guard, Pop { stack, cc: CcT(()) }),
            InstructionKind::PopJump(ops::PopJump { stack, cc_test }) => meets.meets(
                guard,
                Pop {
                    stack,
                    cc: CcTest(cc_test),
                },
            ),
            InstructionKind::Schedule(op) => meets.meets(guard, Push(op)),
            InstructionKind::PushTarget(op) => meets.meets(guard, Push(op)),
            InstructionKind::CbufPushTarget(ops::CbufPushTarget { stack, .. }) => {
                meets.meets(guard, NoSlotPush(stack))
            }
            InstructionKind::OutsideSchedule(ops::OutsideSchedule { .. }) => {
                meets.meets(guard, NoSlotPush(TargetStack::Sync))
            }
            InstructionKind::OutsidePushTarget(ops::OutsidePushTarget { stack, .. }) => {
                meets.meets(guard, NoSlotPush(stack))
            }
            InstructionKind::MisalignedSchedule(ops::MisalignedSchedule { .. }) => {
                meets.meets(guard, NoSlotPush(TargetStack::Sync))
            }
            InstructionKind::MisalignedPushTarget(ops::MisalignedPushTarget { stack, .. }) => {
                meets.meets(guard, NoSlotPush(stack))
            }
            InstructionKind::ResolvedSchedule(ops::ResolvedSchedule { push }) => meets.meets(
                guard,
                ResolvedPush(ops::Schedule {
                    target: push.target(),
                }),
            ),
            InstructionKind::ResolvedPushTarget(ops::ResolvedPushTarget { push, stack }) => meets
                .meets(
                    guard,
                    ResolvedPush(ops::PushTarget {
                        target: push.target(),
                        stack,
                    }),
                ),
            InstructionKind::Call(ops::Call { target }) => meets.meets(
                guard,
                Call {
                    slot: target.slot(),
                },
            ),
            InstructionKind::OutsideCall(ops::OutsideCall { .. }) => {
                meets.meets(guard, NoSlotCall(()))
            }
            InstructionKind::MisalignedCall(ops::MisalignedCall { .. }) => {
                meets.meets(guard, NoSlotCall(()))
            }
            InstructionKind::Padding(ops::Padding) => meets.meets(guard, Unknown(())),
            InstructionKind::Unknown(ops::Unknown { .. }) => meets.meets(guard, Unknown(())),
            kind if kind.may_write_predicate() => meets.meets(guard, PredicateWrite(())),
            _ => meets.meets(guard, Step(())),
        }
    }
}
