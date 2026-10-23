//! Cross-block facts for the emitter's single-function SPIR-V.
//!
//! Ryujinx Translation/Ssa.cs:122-205,250-318 renames registers and joins
//! predecessor definitions with phis. We use live-in phis rather than a
//! dominance-frontier pass: single-predecessor phis collapse to copies, and
//! the dispatch loop has few joins. This keeps one SPIR-V representation.
//! No shader instruction semantics or Maxwell fields are decoded here.
//!
//! As in the parent module, each opcode reaches its behavior through a table
//! indexed by its number, and a state with work of its own reaches that work
//! through a table indexed by the state.
use super::*;
use crate::fast_hash::HashSet;
use std::collections::VecDeque;
use std::ops::Range;

struct Block {
    code: Range<usize>,
    label: u32,
    predecessors: Vec<usize>,
    successors: Vec<usize>,
}

/// The blocks of the module's function. `labels` maps each label ID to its
/// block and every other ID to `blocks.len()`, the index past every block.
/// `formed` holds for one function whose blocks each end in a terminator
/// naming its own labels; only a formed graph reaches a pass. `whole` holds
/// for a formed graph whose every block the entry reaches.
struct Graph {
    blocks: Vec<Block>,
    labels: Vec<usize>,
    reachable: Vec<bool>,
    formed: bool,
    whole: bool,
}

/// Pushes the labels a terminator names; false for an instruction that ends
/// no block.
type Exit = fn(&Instruction, &mut Vec<u32>) -> bool;

fn open_end(_: &Instruction, _: &mut Vec<u32>) -> bool {
    false
}

fn final_exit(_: &Instruction, _: &mut Vec<u32>) -> bool {
    true
}

fn branch_exit(i: &Instruction, out: &mut Vec<u32>) -> bool {
    out.push(i.word(0));
    true
}

fn conditional_exit(i: &Instruction, out: &mut Vec<u32>) -> bool {
    out.extend([i.word(1), i.word(2)]);
    true
}

fn switch_exit(i: &Instruction, out: &mut Vec<u32>) -> bool {
    out.push(i.word(1));
    out.extend(i.args[2..].as_chunks::<2>().0.iter().map(|pair| pair[1]));
    true
}

// Terminators and label IDs: Khronos spirv.core.grammar.json:2820-2886.
// Switch literals are uint32, as checked by the enclosing optimizer.
static EXITS: [Exit; OPCODES] = {
    let mut e = [open_end as Exit; OPCODES];
    e[at(Op::Branch)] = branch_exit;
    e[at(Op::BranchConditional)] = conditional_exit;
    e[at(Op::Switch)] = switch_exit;
    e[at(Op::Kill)] = final_exit;
    e[at(Op::Return)] = final_exit;
    e[at(Op::ReturnValue)] = final_exit;
    e[at(Op::Unreachable)] = final_exit;
    e
};

impl Graph {
    fn new(xs: &[Instruction], bound: usize) -> Self {
        let functions = xs
            .iter()
            .map(|i| usize::from(i.op == Op::Function))
            .sum::<usize>();
        let n = xs
            .iter()
            .map(|i| usize::from(i.op == Op::Label))
            .sum::<usize>();
        let mut labels = vec![n; bound];
        let mut starts = Vec::with_capacity(n);
        // A label or the function end closes the block before it: `ends[k + 1]`
        // closes block `k`, and `ends[0]` is what closes before any block.
        let mut ends = vec![xs.len(); n + 1];
        let mut open = 0;
        let mut orphan = false;
        for (at, i) in xs.iter().enumerate() {
            let label = i.op == Op::Label;
            let end = i.op == Op::FunctionEnd;
            ends[open] = pick(label | end, ends[open], at);
            orphan |= end & (open == 0);
            let k = slot(i.word(0), bound);
            labels[k] = pick(label, labels[k], open);
            starts.extend(once((at, i.word(0))).take(usize::from(label)));
            open += usize::from(label);
        }
        // Successor lists, one past the blocks for the index no block has.
        let mut successors: Vec<Vec<usize>> = (0..=n).map(|_| Vec::new()).collect();
        let mut named = vec![usize::MAX; n + 1];
        let mut targets = Vec::new();
        let mut terminated = true;
        for (k, next) in successors.iter_mut().take(n).enumerate() {
            let term = &xs[ends[k + 1] - 1];
            targets.clear();
            terminated &= EXITS[code(term.op)](term, &mut targets);
            for &target in &targets {
                let to = pick((target as usize) < bound, n, labels[slot(target, bound)]);
                terminated &= to < n;
                let fresh = (named[to] != k) & (to < n);
                named[to] = k;
                next.extend(once(to).take(usize::from(fresh)));
            }
        }
        let mut predecessors: Vec<Vec<usize>> = (0..n).map(|_| Vec::new()).collect();
        for (from, next) in successors.iter().enumerate() {
            for &to in next {
                predecessors[to].push(from);
            }
        }
        let mut reachable = vec![false; n + 1];
        reachable[0] = true;
        let mut pending = vec![0];
        let mut cursor = 0;
        while cursor < pending.len() {
            let b = pending[cursor];
            cursor += 1;
            for &next in &successors[b] {
                let fresh = !reachable[next];
                reachable[next] = true;
                pending.extend(once(next).take(usize::from(fresh)));
            }
        }
        reachable.truncate(n);
        let blocks = starts
            .iter()
            .zip(&ends[1..])
            .zip(successors)
            .zip(predecessors)
            .map(
                |(((&(start, label), &end), successors), predecessors)| Block {
                    code: start..end,
                    label,
                    predecessors,
                    successors,
                },
            )
            .collect();
        let formed = (functions == 1) & !orphan & terminated;
        let whole = formed & reachable.iter().fold(true, |all, &r| all & r);
        Self {
            blocks,
            labels,
            reachable,
            formed,
            whole,
        }
    }
}

pub(super) fn instruction(op: Op, args: &[u32]) -> Instruction {
    Instruction::new(op, args)
}

// TypeBool, TypeInt width and TypeFloat width: Khronos grammar :300-329.
fn scalar_types(xs: &[Instruction], bound: usize) -> Vec<Scalar> {
    let mut types = vec![Scalar::Other; bound];
    for i in xs {
        let scalar = declared_scalar(i);
        let k = slot(i.word(0), bound);
        types[k] = pick(scalar != Scalar::Other, types[k], scalar);
    }
    types
}

// No Function variables means no scalar can be promoted. This avoids graph
// construction and identity rewrites after local forwarding.
pub(super) fn promote(xs: &mut Vec<Instruction>, bound: &mut usize) -> bool {
    let function_variables = xs.iter().fold(false, |any, i| {
        any | ((i.op == Op::Variable) & (i.word(2) == 7))
    });
    [unpromoted, promote_variables][usize::from(function_variables)](xs, bound)
}

fn unpromoted(_: &mut Vec<Instruction>, _: &mut usize) -> bool {
    false
}

fn promote_variables(xs: &mut Vec<Instruction>, bound: &mut usize) -> bool {
    let graph = Graph::new(xs, *bound);
    [partial_graph, promote_whole][usize::from(graph.whole)](xs, bound, &graph)
}

fn partial_graph(_: &mut Vec<Instruction>, _: &mut usize, _: &Graph) -> bool {
    false
}

fn promote_whole(xs: &mut Vec<Instruction>, bound: &mut usize, graph: &Graph) -> bool {
    let size = *bound;
    let types = scalar_types(xs, size);
    let mut local = scalar_variables(xs, &types, size);
    let mut value_types = vec![0; size];
    let mut decorated = vec![false; size];
    for i in xs.iter() {
        let k = slot(i.word(0), size);
        value_types[k] = pick(i.op == Op::TypePointer, value_types[k], i.word(2));
        decorated[k] |= i.op == Op::Decorate;
    }
    for i in xs.iter() {
        let loaded = (i.op == Op::Load) & decorated[slot(i.word(1), size)];
        local[slot(i.word(2), size)] &= !loaded;
    }
    // Leave uninitialized storage alone. The emitter explicitly initializes
    // registers, predicates, condition codes and pc in its entry block.
    // An entry store must precede every entry load, and unreachable accesses
    // must not survive removal of a promoted variable.
    let mut initialized = vec![false; size];
    for i in &xs[graph.blocks[0].code.clone()] {
        initialized[slot(i.word(0), size)] |= i.op == Op::Store;
        let k = slot(i.word(2), size);
        local[k] &= !((i.op == Op::Load) & !initialized[k]);
    }
    let mut variables = Vec::new();
    for i in xs.iter() {
        let v = slot(i.word(1), size);
        let promoted = (i.op == Op::Variable) & local[v] & initialized[v];
        let ty = value_types[slot(i.word(0), size)];
        variables.extend(once((i.word(1), ty)).take(usize::from(promoted)));
    }
    [no_variables, promote_each][variables.len().min(1)](xs, bound, graph, &variables)
}

fn no_variables(_: &mut Vec<Instruction>, _: &mut usize, _: &Graph, _: &[(u32, u32)]) -> bool {
    false
}

/// Where one variable lives across blocks: the last value each block stores,
/// whether it stores one, whether its value enters from predecessors, and the
/// phi carrying that value in.
struct Flow {
    last_store: Vec<u32>,
    stored: Vec<bool>,
    live_in: Vec<bool>,
    incoming: Vec<u32>,
    entered: Vec<bool>,
}

impl Flow {
    /// The value leaving block `b`.
    fn outgoing(&self, b: usize) -> u32 {
        pick(self.stored[b], self.incoming[b], self.last_store[b])
    }
}

fn promote_each(
    xs: &mut Vec<Instruction>,
    bound: &mut usize,
    graph: &Graph,
    variables: &[(u32, u32)],
) -> bool {
    let n = graph.blocks.len();
    // One phi list per block, and an empty one past them for every
    // instruction that is not a label.
    let mut phis: Vec<Vec<Instruction>> = (0..=n).map(|_| Vec::new()).collect();
    let mut replacements = (0..*bound as u32).collect::<Vec<_>>();
    let mut flow = Flow {
        last_store: vec![0; n],
        stored: vec![false; n],
        live_in: vec![false; n],
        incoming: vec![0; n],
        entered: vec![false; n],
    };
    let mut pending = Vec::new();
    let mut args = Vec::new();
    for &(variable, ty) in variables {
        flow.last_store.fill(0);
        flow.stored.fill(false);
        flow.live_in.fill(false);
        flow.incoming.fill(0);
        flow.entered.fill(false);
        pending.clear();
        for (b, block) in graph.blocks.iter().enumerate() {
            for i in &xs[block.code.clone()] {
                let store = (i.op == Op::Store) & (i.word(0) == variable);
                let load = (i.op == Op::Load) & (i.word(2) == variable);
                flow.last_store[b] = pick(store, flow.last_store[b], i.word(1));
                flow.live_in[b] |= load & !flow.stored[b];
                flow.stored[b] |= store;
            }
            pending.extend(once(b).take(usize::from(flow.live_in[b])));
        }
        let mut cursor = 0;
        while cursor < pending.len() {
            let b = pending[cursor];
            cursor += 1;
            for &p in &graph.blocks[b].predecessors {
                let fresh = !flow.stored[p] & !flow.live_in[p];
                flow.live_in[p] |= fresh;
                pending.extend(once(p).take(usize::from(fresh)));
            }
        }
        for b in 1..n {
            let fresh = flow.live_in[b] & graph.reachable[b];
            let id = *bound as u32;
            flow.incoming[b] = pick(fresh, 0, id);
            flow.entered[b] = fresh;
            replacements.extend(once(id).take(usize::from(fresh)));
            *bound += usize::from(fresh);
        }
        for (b, block) in graph.blocks.iter().enumerate() {
            // One value/parent pair per distinct predecessor. Promotion's
            // entry check makes all of them reachable.
            let phi = || {
                args.clear();
                args.extend([ty, flow.incoming[b]]);
                for &p in &block.predecessors {
                    args.extend([flow.outgoing(p), graph.blocks[p].label]);
                }
                instruction(Op::Phi, &args)
            };
            phis[b].extend(repeat_with(phi).take(usize::from(flow.entered[b])));
            let mut value = flow.incoming[b];
            for i in &mut xs[block.code.clone()] {
                let store = (i.op == Op::Store) & (i.word(0) == variable);
                let load = (i.op == Op::Load) & (i.word(2) == variable);
                value = pick(store, value, i.word(1));
                let k = slot(i.word(1), replacements.len());
                replacements[k] = pick(load, replacements[k], value);
                i.removed |= store | load;
            }
        }
        for i in xs.iter_mut() {
            i.removed |= (i.op == Op::Variable) & (i.word(1) == variable);
        }
    }
    let mut rebuilt = Vec::with_capacity(xs.len());
    let mut opened = 0;
    for mut i in xs.drain(..) {
        i.rewrite(&replacements);
        let kept = !i.removed;
        let label = (i.op == Op::Label) & kept;
        rebuilt.extend(once(i).take(usize::from(kept)));
        let k = pick(label, n, opened);
        opened += usize::from(label);
        for mut phi in phis[k].drain(..) {
            phi.rewrite(&replacements);
            rebuilt.push(phi);
        }
    }
    *xs = rebuilt;
    true
}

// Ryujinx Translation/Optimizations/Optimizer.cs:156-183 propagates a phi
// with identical sources. Self references add no new value; retaining one
// distinct external value is sufficient even for a loop-carried definition.
fn simplify_phis(xs: &mut Vec<Instruction>, bound: usize) -> bool {
    let phis = xs.iter().fold(false, |any, i| any | (i.op == Op::Phi));
    [unsimplified, simplify_present][usize::from(phis)](xs, bound)
}

fn unsimplified(_: &mut Vec<Instruction>, _: usize) -> bool {
    false
}

fn simplify_present(xs: &mut Vec<Instruction>, bound: usize) -> bool {
    let mut decorated = vec![false; bound];
    for i in xs.iter() {
        decorated[slot(i.word(0), bound)] |= i.op == Op::Decorate;
    }
    let mut phis = Vec::new();
    for (at, i) in xs.iter().enumerate() {
        let open = (i.op == Op::Phi) & !decorated[slot(i.word(1), bound)];
        phis.extend(once(at).take(usize::from(open)));
    }
    let mut replacements = (0..bound as u32).collect::<Vec<_>>();
    let mut changed = false;
    let mut progress = true;
    while progress {
        progress = false;
        for &at in &phis {
            let i = &mut xs[at];
            i.rewrite(&replacements);
            let id = i.word(1);
            // The phi's own ID stands for "no external value yet".
            let mut source = id;
            let mut same = true;
            for &value in i.args[2..].iter().step_by(2) {
                let external = value != id;
                let first = source == id;
                same &= !external | first | (source == value);
                source = pick(external & first, source, value);
            }
            let folds = same & (source != id) & !i.removed;
            replacements[id as usize] = pick(folds, replacements[id as usize], source);
            i.removed |= folds;
            progress |= folds;
        }
        changed |= progress;
    }
    for i in xs.iter_mut() {
        i.rewrite(&replacements);
    }
    xs.retain(|i| !i.removed);
    changed
}

pub(super) fn simplify(xs: &mut Vec<Instruction>, bound: usize) -> bool {
    let mut changed = simplify_phis(xs, bound);
    changed |= prune(xs, bound);
    changed |= cse(xs, bound);
    changed
}

/// No executable definition has reached the value yet.
const BOTTOM: usize = 0;
/// The value is one of a known, capped set.
const KNOWN: usize = 1;
/// The value may be anything.
const TOP: usize = 2;

// A capped powerset is needed for pc: it can hold several case numbers.
// Bottom waits for executable incoming edges; unknown operations are Top.
// Widening only loses optimization opportunities, never possible executions.
// Scalar constants are by far the common case, so one value stays inline in
// `one`; a second moves every value, in join order, into `many`.
#[derive(Clone)]
struct Fact {
    level: usize,
    one: [u32; 1],
    many: Vec<u32>,
}

/// The fact joining no value: a path not taken.
static NOTHING: Fact = Fact::BOTTOM;

impl Fact {
    const BOTTOM: Self = Self {
        level: BOTTOM,
        one: [0],
        many: Vec::new(),
    };

    /// The values a known fact holds.
    fn values(&self) -> &[u32] {
        let views: [&[u32]; 2] = [&self.one, &self.many];
        views[self.many.len().min(1)]
    }

    /// The value a fact holds, and whether it holds exactly that one.
    fn singleton(&self) -> (u32, bool) {
        (self.one[0], (self.level == KNOWN) & (self.many.len() < 2))
    }
}

type Join = fn(&mut Fact, &Fact) -> bool;

/// Indexed by the joined fact's level, then the incoming fact's level.
const JOIN: [[Join; 3]; 3] = [
    [keep, adopt, raise],
    [keep, merge, raise],
    [keep, keep, keep],
];

fn keep(_: &mut Fact, _: &Fact) -> bool {
    false
}

fn adopt(a: &mut Fact, b: &Fact) -> bool {
    a.clone_from(b);
    true
}

fn raise(a: &mut Fact, _: &Fact) -> bool {
    a.level = TOP;
    a.many = Vec::new();
    true
}

fn merge(a: &mut Fact, b: &Fact) -> bool {
    let before = a.values().len();
    let seed = a.one[0];
    for &value in b.values() {
        let fresh = a
            .values()
            .iter()
            .fold(true, |absent, &x| absent & (x != value));
        let second = fresh & a.many.is_empty();
        a.many.extend(once(seed).take(usize::from(second)));
        a.many.extend(once(value).take(usize::from(fresh)));
    }
    let after = a.values().len();
    let over = a.many.len() > 64;
    a.many.truncate(a.many.len() * usize::from(!over));
    a.level = pick(over, KNOWN, TOP);
    before != after
}

fn join(a: &mut Fact, b: &Fact) -> bool {
    JOIN[a.level][b.level](a, b)
}

/// The blocks reading a value. Consecutive reads from one block count once;
/// the first reader stays inline.
#[derive(Clone)]
struct Users {
    last: usize,
    first: usize,
    count: usize,
    others: Vec<usize>,
}

impl Users {
    /// Nobody reads the value; `last` names no block.
    const NOBODY: Self = Self {
        last: usize::MAX,
        first: 0,
        count: 0,
        others: Vec::new(),
    };

    fn add(&mut self, block: usize) {
        let fresh = self.last != block;
        let first = fresh & (self.count == 0);
        self.first = pick(first, self.first, block);
        self.others
            .extend(once(block).take(usize::from(fresh & !first)));
        self.count += usize::from(fresh);
        self.last = block;
    }

    fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        once(self.first)
            .take(self.count.min(1))
            .chain(self.others.iter().copied())
    }
}

struct Execution {
    facts: Vec<Fact>,
    edges: HashSet<(usize, usize)>,
    reachable: Vec<bool>,
}

impl Execution {
    /// 1 where the edge from `from` to `to` has executed, 0 elsewhere.
    fn taken(&self, from: usize, to: usize) -> usize {
        usize::from(self.edges.contains(&(from, to)))
    }
}

/// The sparse conditional constant propagation worklist.
struct Sccp<'g> {
    graph: &'g Graph,
    types: Vec<Scalar>,
    result_types: Vec<u32>,
    users: Vec<Users>,
    constants: Vec<Constant>,
    queued: Vec<bool>,
    pending: VecDeque<usize>,
    targets: Vec<u32>,
    block: usize,
    execution: Execution,
}

type Evaluate = fn(&Instruction, &mut Sccp) -> Fact;

fn unknown(_: &Instruction, _: &mut Sccp) -> Fact {
    Fact {
        level: TOP,
        ..Fact::BOTTOM
    }
}

fn phi_fact(i: &Instruction, s: &mut Sccp) -> Fact {
    let mut value = Fact::BOTTOM;
    for pair in i.args[2..].as_chunks::<2>().0 {
        let p = s.graph.labels[pair[1] as usize];
        let taken = s.execution.taken(p, s.block);
        join(
            &mut value,
            [&NOTHING, &s.execution.facts[pair[0] as usize]][taken],
        );
    }
    value
}

fn select_fact(i: &Instruction, s: &mut Sccp) -> Fact {
    let facts = &s.execution.facts;
    let condition = &facts[i.word(2) as usize];
    let reached = condition.level != BOTTOM;
    let top = condition.level == TOP;
    let values = condition.values();
    let nonzero = values.iter().fold(false, |any, &x| any | (x != 0));
    let zero = values.iter().fold(false, |any, &x| any | (x == 0));
    let mut value = Fact::BOTTOM;
    let yes = usize::from(reached & (top | nonzero));
    let no = usize::from(reached & (top | zero));
    join(&mut value, [&NOTHING, &facts[i.word(3) as usize]][yes]);
    join(&mut value, [&NOTHING, &facts[i.word(4) as usize]][no]);
    value
}

fn constant_fact(i: &Instruction, s: &mut Sccp) -> Fact {
    [unknown, declared_fact][usize::from(i.args.len() <= 3)](i, s)
}

fn declared_fact(i: &Instruction, s: &mut Sccp) -> Fact {
    s.execution.facts[i.result() as usize].clone()
}

fn arithmetic_fact(i: &Instruction, s: &mut Sccp) -> Fact {
    let scalar = s.types[slot(i.word(0), s.types.len())] != Scalar::Other;
    [unknown, folded_fact][usize::from(scalar & (i.args.len() >= 3))](i, s)
}

/// Bottom while an operand waits; otherwise the folded constant, or Top.
fn folded_fact(i: &Instruction, s: &mut Sccp) -> Fact {
    let Sccp {
        execution,
        constants,
        result_types,
        types,
        ..
    } = s;
    let mut waiting = false;
    i.inputs(|at| {
        let operand = at >= 2;
        let source = i.args[at] as usize;
        let fact = &execution.facts[source];
        waiting |= operand & (fact.level == BOTTOM);
        let (bits, single) = fact.singleton();
        let constant = Constant {
            ty: result_types[source],
            bits,
            known: true,
        };
        let value = pick(single, UNKNOWN, constant);
        constants[source] = pick(operand, constants[source], value);
    });
    let (bits, folds) = fold(i, constants, types);
    i.inputs(|at| {
        let source = i.args[at] as usize;
        constants[source] = pick(at >= 2, constants[source], UNKNOWN);
    });
    Fact {
        level: pick(waiting, pick(folds, TOP, KNOWN), BOTTOM),
        one: [bits],
        many: Vec::new(),
    }
}

static EVALUATE: [Evaluate; OPCODES] = {
    let mut e = [unknown as Evaluate; OPCODES];
    e[at(Op::Phi)] = phi_fact;
    e[at(Op::Select)] = select_fact;
    e[at(Op::ConstantTrue)] = constant_fact;
    e[at(Op::ConstantFalse)] = constant_fact;
    e[at(Op::Constant)] = constant_fact;
    let folded = [
        Op::Bitcast,
        Op::SNegate,
        Op::IAdd,
        Op::ISub,
        Op::IMul,
        Op::LogicalEqual,
        Op::LogicalNotEqual,
        Op::LogicalOr,
        Op::LogicalAnd,
        Op::LogicalNot,
        Op::IEqual,
        Op::INotEqual,
        Op::UGreaterThan,
        Op::SGreaterThan,
        Op::UGreaterThanEqual,
        Op::SGreaterThanEqual,
        Op::ULessThan,
        Op::SLessThan,
        Op::ULessThanEqual,
        Op::SLessThanEqual,
        Op::ShiftRightLogical,
        Op::ShiftRightArithmetic,
        Op::ShiftLeftLogical,
        Op::BitwiseOr,
        Op::BitwiseXor,
        Op::BitwiseAnd,
        Op::Not,
        Op::BitFieldSExtract,
        Op::BitFieldUExtract,
        Op::BitReverse,
        Op::BitCount,
    ];
    let mut k = 0;
    while k < folded.len() {
        e[at(folded[k])] = arithmetic_fact;
        k += 1;
    }
    e
};

/// An untyped instruction defines no value to propagate.
fn untyped(_: &Instruction, _: &mut Sccp) {}

fn settle(i: &Instruction, s: &mut Sccp) {
    let id = i.result() as usize;
    let value = EVALUATE[code(i.op)](i, s);
    let changed = join(&mut s.execution.facts[id], &value);
    let Sccp {
        users,
        execution,
        queued,
        pending,
        ..
    } = s;
    for user in users[id].iter().take(pick(changed, 0, usize::MAX)) {
        let wake = execution.reachable[user] & !queued[user];
        queued[user] |= wake;
        pending.extend(once(user).take(usize::from(wake)));
    }
}

const SETTLE: [fn(&Instruction, &mut Sccp); 2] = [untyped, settle];

fn analyze(xs: &[Instruction], graph: &Graph, bound: usize) -> Execution {
    let types = scalar_types(xs, bound);
    let mut facts = vec![Fact::BOTTOM; bound];
    let mut result_types = vec![0; bound];
    let mut users = vec![Users::NOBODY; bound];
    let mut in_block = false;
    for i in xs {
        in_block = (in_block & (i.op != Op::FunctionEnd)) | (i.op == Op::Label);
        let typed = i.layout.typed;
        let k = slot(i.word(1), bound);
        result_types[k] = pick(typed, result_types[k], i.word(0));
        // Specialization constants, parameters and other global values
        // are runtime inputs, not pending definitions in this worklist.
        // OpSpecConstant fields: Khronos grammar :606-615.
        facts[k].level = pick(typed & !in_block, facts[k].level, TOP);
        let (bits, constant) = constant_value(i);
        facts[k].one[0] = pick(constant, facts[k].one[0], bits);
        facts[k].level = pick(constant, facts[k].level, KNOWN);
    }
    for (b, block) in graph.blocks.iter().enumerate() {
        for i in &xs[block.code.clone()] {
            i.inputs(|at| users[i.args[at] as usize].add(b));
        }
    }
    let n = graph.blocks.len();
    let mut s = Sccp {
        graph,
        types,
        result_types,
        users,
        constants: vec![UNKNOWN; bound],
        queued: vec![false; n],
        pending: VecDeque::from([0]),
        targets: Vec::new(),
        block: 0,
        execution: Execution {
            facts,
            edges: HashSet::default(),
            reachable: vec![false; n],
        },
    };
    s.execution.reachable[0] = true;
    s.queued[0] = true;
    while !s.pending.is_empty() {
        let b = s.pending[0];
        s.pending.pop_front();
        s.queued[b] = false;
        s.block = b;
        let block = &graph.blocks[b];
        for i in &xs[block.code.clone()] {
            SETTLE[usize::from(i.layout.typed)](i, &mut s);
        }
        let term = &xs[block.code.end - 1];
        let Sccp {
            targets,
            execution,
            queued,
            pending,
            ..
        } = &mut s;
        targets.clear();
        FEASIBLE[code(term.op)](term, &execution.facts, targets);
        for &label in targets.iter() {
            let to = graph.labels[label as usize];
            let new = execution.edges.insert((b, to));
            execution.reachable[to] |= new;
            let wake = new & !queued[to];
            queued[to] |= wake;
            pending.extend(once(to).take(usize::from(wake)));
        }
    }
    s.execution
}

/// Pushes the labels a terminator can reach under the facts.
type Feasible = fn(&Instruction, &[Fact], &mut Vec<u32>);

/// Pushes the labels a terminator can reach under its selector's fact.
type Reach = fn(&Instruction, &Fact, &mut Vec<u32>);

fn no_feasible(_: &Instruction, _: &[Fact], _: &mut Vec<u32>) {}

fn no_targets(_: &Instruction, _: &Fact, _: &mut Vec<u32>) {}

fn branch_feasible(i: &Instruction, _: &[Fact], out: &mut Vec<u32>) {
    out.push(i.word(0));
}

// Operand fields are grammar :2838-2859, not inferred from numeric IDs.
fn conditional_feasible(i: &Instruction, facts: &[Fact], out: &mut Vec<u32>) {
    let fact = &facts[i.word(0) as usize];
    CONDITIONAL[fact.level](i, fact, out);
}

const CONDITIONAL: [Reach; 3] = [no_targets, conditional_known, both_targets];

fn conditional_known(i: &Instruction, fact: &Fact, out: &mut Vec<u32>) {
    let values = fact.values();
    let nonzero = values.iter().fold(false, |any, &v| any | (v != 0));
    let zero = values.iter().fold(false, |any, &v| any | (v == 0));
    out.extend(once(i.word(1)).take(usize::from(nonzero)));
    out.extend(once(i.word(2)).take(usize::from(zero)));
}

fn both_targets(i: &Instruction, _: &Fact, out: &mut Vec<u32>) {
    out.extend([i.word(1), i.word(2)]);
}

fn switch_feasible(i: &Instruction, facts: &[Fact], out: &mut Vec<u32>) {
    let fact = &facts[i.word(0) as usize];
    SWITCH[fact.level](i, fact, out);
}

const SWITCH: [Reach; 3] = [no_targets, switch_known, every_case];

/// Each selector value's first matching case, or the default.
fn switch_known(i: &Instruction, fact: &Fact, out: &mut Vec<u32>) {
    let cases = i.args[2..].as_chunks::<2>().0;
    out.extend(fact.values().iter().map(|&value| {
        let (target, _) = cases.iter().fold((i.word(1), false), |(t, found), pair| {
            let hit = !found & (pair[0] == value);
            (pick(hit, t, pair[1]), found | hit)
        });
        target
    }));
}

fn every_case(i: &Instruction, _: &Fact, out: &mut Vec<u32>) {
    switch_exit(i, out);
}

static FEASIBLE: [Feasible; OPCODES] = {
    let mut f = [no_feasible as Feasible; OPCODES];
    f[at(Op::Branch)] = branch_feasible;
    f[at(Op::BranchConditional)] = conditional_feasible;
    f[at(Op::Switch)] = switch_feasible;
    f
};

/// Opcodes whose presence gives pruning something to do in a whole graph.
static STRUCTURED: [bool; OPCODES] = {
    let mut s = [false; OPCODES];
    s[at(Op::Select)] = true;
    s[at(Op::Phi)] = true;
    s[at(Op::LoopMerge)] = true;
    s[at(Op::BranchConditional)] = true;
    s[at(Op::Switch)] = true;
    s
};

/// What pruning has found and rewritten so far.
struct Pruning {
    changed: bool,
    cfg_changed: bool,
    terminators: Vec<(usize, Instruction)>,
    replacements: Vec<u32>,
    /// Each (type, bits) constant's ID; 0, the ID SPIR-V reserves, names no
    /// constant.
    canonical: HashMap<(u32, u32), u32>,
    decorated: Vec<bool>,
    targets: Vec<u32>,
}

fn prune(xs: &mut Vec<Instruction>, bound: usize) -> bool {
    let graph = Graph::new(xs, bound);
    // A reachable graph with no phi/select/conditional/loop has no fact
    // substitution, edge pruning or loop-marker removal available here.
    let structured = xs.iter().fold(false, |any, i| any | STRUCTURED[code(i.op)]);
    let work = graph.formed & (!graph.whole | structured);
    [unpruned, prune_formed][usize::from(work)](xs, bound, &graph)
}

fn unpruned(_: &mut Vec<Instruction>, _: usize, _: &Graph) -> bool {
    false
}

fn prune_formed(xs: &mut Vec<Instruction>, bound: usize, graph: &Graph) -> bool {
    let execution = analyze(xs, graph, bound);
    let mut canonical = HashMap::default();
    canonical.extend(xs.iter().flat_map(|i| {
        let (bits, constant) = constant_value(i);
        once(((i.word(0), bits), i.word(1))).take(usize::from(constant))
    }));
    let mut decorated = vec![false; bound];
    for i in xs.iter() {
        decorated[slot(i.word(0), bound)] |= i.op == Op::Decorate;
    }
    let mut p = Pruning {
        changed: false,
        cfg_changed: false,
        terminators: Vec::new(),
        replacements: (0..bound as u32).collect(),
        canonical,
        decorated,
        targets: Vec::new(),
    };
    for (b, block) in graph.blocks.iter().enumerate() {
        [unreached_block, prune_block][usize::from(execution.reachable[b])](
            xs, block, &execution, &mut p,
        );
    }
    // Re-derive the CFG and construct ancestry after the proposed edge edits.
    // An acyclic loop cannot keep OpLoopMerge, but deleting that boundary
    // can invalidate a nested break/continue. Preserve its original CFG until
    // we can restructure those exits together. SSA and cbuf CSE still run.
    [settled_cfg, check_cfg][usize::from(p.cfg_changed)](xs, bound, &mut p);
    for i in xs.iter_mut() {
        i.rewrite(&p.replacements);
    }
    xs.retain(|i| !i.removed);
    p.changed | remove_unreachable(xs, bound)
}

fn unreached_block(_: &mut [Instruction], _: &Block, _: &Execution, _: &mut Pruning) {}

fn prune_block(xs: &mut [Instruction], block: &Block, e: &Execution, p: &mut Pruning) {
    for i in &mut xs[block.code.clone()] {
        let result = slot(i.word(1), p.decorated.len());
        let candidate = ((i.op == Op::Phi) | (i.op == Op::Select)) & !p.decorated[result];
        let (bits, single) = e.facts[result].singleton();
        [kept_value, substitute][usize::from(candidate & single)](i, bits, p);
    }
    let at = block.code.end - 1;
    RETARGET[code(xs[at].op)](xs, at, block, e, p);
}

fn kept_value(_: &mut Instruction, _: u32, _: &mut Pruning) {}

fn substitute(i: &mut Instruction, bits: u32, p: &mut Pruning) {
    let value = *p.canonical.entry((i.word(0), bits)).or_insert(0);
    let hit = value != 0;
    let id = i.word(1) as usize;
    p.replacements[id] = pick(hit, p.replacements[id], value);
    i.removed |= hit;
    p.changed |= hit;
}

type Retarget = fn(&mut [Instruction], usize, &Block, &Execution, &mut Pruning);

fn settled_exit(_: &mut [Instruction], _: usize, _: &Block, _: &Execution, _: &mut Pruning) {}

/// Builds a block's one surviving exit, keeping its selection header form.
type Header = fn(&Instruction, u32) -> Instruction;

/// Rewires a terminator: with several feasible targets, or with one.
type Rewire = fn(&mut [Instruction], usize, Header, &Execution, &mut Pruning);

static RETARGET: [Retarget; OPCODES] = {
    let mut r = [settled_exit as Retarget; OPCODES];
    r[at(Op::BranchConditional)] = retarget_conditional;
    r[at(Op::Switch)] = retarget_switch;
    r
};

fn retarget_conditional(
    xs: &mut [Instruction],
    at: usize,
    block: &Block,
    e: &Execution,
    p: &mut Pruning,
) {
    retarget(xs, at, block, e, p, conditional_header, keep_wide);
}

fn retarget_switch(
    xs: &mut [Instruction],
    at: usize,
    block: &Block,
    e: &Execution,
    p: &mut Pruning,
) {
    retarget(xs, at, block, e, p, switch_header, narrow_switch);
}

fn retarget(
    xs: &mut [Instruction],
    at: usize,
    block: &Block,
    e: &Execution,
    p: &mut Pruning,
    header: Header,
    wide: Rewire,
) {
    p.targets.clear();
    FEASIBLE[code(xs[at].op)](&xs[at], &e.facts, &mut p.targets);
    p.targets.sort_unstable();
    p.targets.dedup();
    // A loop header must keep its merge paired with a legal terminator.
    let code = &xs[block.code.clone()];
    let loop_header = code
        .iter()
        .fold(false, |any, i| any | (i.op == Op::LoopMerge));
    let selection_header = code
        .iter()
        .fold(false, |any, i| any | (i.op == Op::SelectionMerge));
    let single = (p.targets.len() == 1) & !loop_header;
    // Ryujinx CodeGen/Spirv/SpirvGenerator.cs:334-335 pairs a selection
    // merge with its conditional terminator. Keep that boundary: a nested
    // selection may exit to this switch's merge. Removing it changes
    // construct membership even without rerouting a single surviving edge.
    // Grammar :2811-2817,2838-2859.
    let exit = [branch_to as Header, header][usize::from(selection_header)];
    [wide, single_exit][usize::from(single)](xs, at, exit, e, p);
}

fn branch_to(_: &Instruction, target: u32) -> Instruction {
    instruction(Op::Branch, &[target])
}

fn conditional_header(i: &Instruction, target: u32) -> Instruction {
    instruction(Op::BranchConditional, &[i.word(0), target, target])
}

fn switch_header(i: &Instruction, target: u32) -> Instruction {
    instruction(Op::Switch, &[i.word(0), target])
}

fn single_exit(xs: &mut [Instruction], at: usize, exit: Header, _: &Execution, p: &mut Pruning) {
    let replacement = exit(&xs[at], p.targets[0]);
    let differs = (xs[at].op != replacement.op) | (*xs[at].args != *replacement.args);
    replace_exit(xs, at, replacement, differs, p);
}

fn keep_wide(_: &mut [Instruction], _: usize, _: Header, _: &Execution, _: &mut Pruning) {}

fn narrow_switch(xs: &mut [Instruction], at: usize, _: Header, e: &Execution, p: &mut Pruning) {
    let fact = &e.facts[xs[at].word(0) as usize];
    NARROW[fact.level](xs, at, fact, p);
}

type Narrow = fn(&mut [Instruction], usize, &Fact, &mut Pruning);

const NARROW: [Narrow; 3] = [unnarrowed, narrow_known, unnarrowed];

fn unnarrowed(_: &mut [Instruction], _: usize, _: &Fact, _: &mut Pruning) {}

// Keep the default edge conservatively, even where no analyzed selector takes
// it. This preserves the enclosing construct.
fn narrow_known(xs: &mut [Instruction], at: usize, fact: &Fact, p: &mut Pruning) {
    let term = &xs[at];
    let values = fact.values();
    let mut args = term.args[..2].to_vec();
    for pair in term.args[2..].as_chunks::<2>().0 {
        let live = values.iter().fold(false, |any, &v| any | (v == pair[0]));
        args.extend_from_slice(&pair[..2 * usize::from(live)]);
    }
    let differs = args.len() != term.args.len();
    replace_exit(xs, at, instruction(Op::Switch, &args), differs, p);
}

/// Installs `replacement`, keeping the original to restore where the edited CFG
/// must stay as it was.
fn replace_exit(
    xs: &mut [Instruction],
    at: usize,
    replacement: Instruction,
    differs: bool,
    p: &mut Pruning,
) {
    let original = std::mem::replace(&mut xs[at], replacement);
    p.terminators.push((at, original));
    p.cfg_changed |= differs;
}

fn settled_cfg(_: &mut [Instruction], _: usize, _: &mut Pruning) {}

fn check_cfg(xs: &mut [Instruction], bound: usize, p: &mut Pruning) {
    let proposed = Graph::new(xs, bound);
    let idom = dominators(&proposed);
    let held = acyclic_loop_with_structured_exits(xs, &proposed, &idom);
    [accept_cfg, restore_cfg][usize::from(held)](xs, p);
}

fn accept_cfg(_: &mut [Instruction], p: &mut Pruning) {
    p.changed = true;
}

fn restore_cfg(xs: &mut [Instruction], p: &mut Pruning) {
    for (at, terminator) in p.terminators.drain(..) {
        xs[at] = terminator;
    }
}

/// The first merge declaration in `code`: its merge and continue words, and
/// whether there is one. `selections` admits OpSelectionMerge too.
fn declaration(code: &[Instruction], selections: bool) -> (u32, u32, bool) {
    code.iter().fold((0, 0, false), |(merge, cont, found), i| {
        let merge_op = (i.op == Op::LoopMerge) | (selections & (i.op == Op::SelectionMerge));
        let hit = !found & merge_op;
        (
            pick(hit, merge, i.word(0)),
            pick(hit, cont, i.word(1)),
            found | hit,
        )
    })
}

/// Whether a reachable predecessor of `b` is a block `b` dominates.
fn backedge(graph: &Graph, idom: &[usize], b: usize) -> bool {
    graph.blocks[b].predecessors.iter().fold(false, |any, &p| {
        any | (graph.reachable[p] & dominates(idom, b, p))
    })
}

/// What the structured-exit scan reads.
struct Scan<'a> {
    xs: &'a [Instruction],
    graph: &'a Graph,
    idom: &'a [usize],
}

// Ryujinx CodeGen/Spirv/SpirvGenerator.cs:339-367 keeps merge and
// continue targets attached to the enclosing loop. An inner construct can
// exit through them, so an acyclic loop with nested exits to its targets
// needs more than deletion of its merge instruction. Use fresh dominance
// and exclude the loop merge's descendants when deriving loop membership.
fn acyclic_loop_with_structured_exits(xs: &[Instruction], graph: &Graph, idom: &[usize]) -> bool {
    let scan = Scan { xs, graph, idom };
    graph
        .blocks
        .iter()
        .enumerate()
        .fold(false, |exits, (b, block)| {
            let (merge, cont, found) = declaration(&xs[block.code.clone()], false);
            exits | [no_loop, loop_header][usize::from(found)](&scan, b, merge, cont)
        })
}

fn no_loop(_: &Scan, _: usize, _: u32, _: u32) -> bool {
    false
}

fn loop_header(scan: &Scan, b: usize, merge: u32, cont: u32) -> bool {
    let open = scan.graph.reachable[b] & !backedge(scan.graph, scan.idom, b);
    [no_loop, acyclic_loop][usize::from(open)](scan, b, merge, cont)
}

fn acyclic_loop(scan: &Scan, b: usize, merge: u32, cont: u32) -> bool {
    let graph = scan.graph;
    let merge = graph.labels[merge as usize];
    let cont = graph.labels[cont as usize];
    (0..graph.blocks.len()).fold(false, |exits, c| {
        let inside = (c != b)
            & graph.reachable[c]
            & dominates(scan.idom, b, c)
            & !dominates(scan.idom, merge, c);
        exits | [outside, nested][usize::from(inside)](scan, c, merge, cont)
    })
}

fn outside(_: &Scan, _: usize, _: usize, _: usize) -> bool {
    false
}

fn nested(scan: &Scan, c: usize, merge: usize, cont: usize) -> bool {
    let code = &scan.xs[scan.graph.blocks[c].code.clone()];
    let (inner, _, declared) = declaration(code, true);
    [undeclared, inner_exits][usize::from(declared)](scan, c, inner, merge, cont)
}

fn undeclared(_: &Scan, _: usize, _: u32, _: usize, _: usize) -> bool {
    false
}

fn inner_exits(scan: &Scan, c: usize, inner: u32, merge: usize, cont: usize) -> bool {
    let graph = scan.graph;
    let inner_merge = graph.labels[inner as usize];
    graph
        .blocks
        .iter()
        .enumerate()
        .fold(false, |exits, (from, source)| {
            let leaves = source.successors.iter().fold(false, |any, &to| {
                any | (((to == merge) | (to == cont)) & (to != inner_merge))
            });
            exits
                | (graph.reachable[from]
                    & dominates(scan.idom, c, from)
                    & !dominates(scan.idom, inner_merge, from)
                    & leaves)
        })
}

// A merge/continue target may be structurally required without an executable
// incoming edge. Keep its label as an OpUnreachable block, then rebuild phi
// pairs from actual retained predecessors. Grammar :2799-2817,2788-2796.
// If pruning removes every reachable backedge, the header is now acyclic;
// remove OpLoopMerge so it does not claim a loop without a backedge.
fn remove_unreachable(xs: &mut Vec<Instruction>, bound: usize) -> bool {
    let graph = Graph::new(xs, bound);
    [unremoved, remove_formed][usize::from(graph.formed)](xs, bound, &graph)
}

fn unremoved(_: &mut Vec<Instruction>, _: usize, _: &Graph) -> bool {
    false
}

fn remove_formed(xs: &mut Vec<Instruction>, bound: usize, graph: &Graph) -> bool {
    let idom = dominators(graph);
    let held = acyclic_loop_with_structured_exits(xs, graph, &idom);
    [remove_dead_blocks, held_blocks][usize::from(held)](xs, bound, graph, &idom)
}

fn held_blocks(_: &mut Vec<Instruction>, _: usize, _: &Graph, _: &[usize]) -> bool {
    false
}

fn never_strips(_: &Graph, _: &[usize], _: usize) -> bool {
    false
}

fn strips(graph: &Graph, idom: &[usize], b: usize) -> bool {
    !backedge(graph, idom, b)
}

/// What the rebuild of a pruned function reads. Index `blocks.len()` of each
/// table stands for every instruction outside a block.
struct Removal<'a> {
    graph: &'a Graph,
    required: Vec<bool>,
    shells: Vec<bool>,
    preds: Vec<&'a [usize]>,
    live: Vec<bool>,
}

/// What the rebuild writes.
struct Rebuild {
    changed: bool,
    dead: Vec<bool>,
    rebuilt: Vec<Instruction>,
}

type Route = fn(Instruction, usize, &Removal, &mut Rebuild);

/// Indexed by an instruction's place: a stripped loop merge, an unreached
/// block, a reached block, or outside every block.
const ROUTE: [Route; 4] = [stripped, unreached, reached, kept];

fn remove_dead_blocks(
    xs: &mut Vec<Instruction>,
    bound: usize,
    graph: &Graph,
    idom: &[usize],
) -> bool {
    let n = graph.blocks.len();
    let mut strip = Vec::with_capacity(n + 1);
    for (b, block) in graph.blocks.iter().enumerate() {
        let (_, _, looped) = declaration(&xs[block.code.clone()], false);
        strip.push([never_strips, strips][usize::from(looped)](graph, idom, b));
    }
    strip.push(false);
    let mut required = vec![false; n + 1];
    for (b, block) in graph.blocks.iter().enumerate() {
        let reach = graph.reachable[b];
        for i in &xs[block.code.clone()] {
            let loops = (i.op == Op::LoopMerge) & !strip[b] & reach;
            let selects = (i.op == Op::SelectionMerge) & reach;
            required[graph.labels[slot(i.word(0), bound)]] |= loops | selects;
            required[graph.labels[slot(i.word(1), bound)]] |= loops;
        }
    }
    let mut owner = vec![n; xs.len()];
    for (b, block) in graph.blocks.iter().enumerate() {
        owner[block.code.clone()].fill(b);
    }
    let mut shells = Vec::with_capacity(n + 1);
    let mut preds: Vec<&[usize]> = Vec::with_capacity(n + 1);
    for block in &graph.blocks {
        let end = xs[block.code.end - 1].op == Op::Unreachable;
        shells.push((block.code.len() == 2) & end);
        preds.push(&block.predecessors);
    }
    shells.push(false);
    preds.push(&[]);
    let mut live = graph.reachable.clone();
    live.push(true);
    let removal = Removal {
        graph,
        required,
        shells,
        preds,
        live,
    };
    let mut r = Rebuild {
        changed: false,
        dead: vec![false; bound],
        rebuilt: Vec::with_capacity(xs.len()),
    };
    for (at, i) in xs.drain(..).enumerate() {
        let b = owner[at];
        let stripped = (i.op == Op::LoopMerge) & strip[b];
        let inner = pick(stripped, pick(removal.live[b], 1, 2), 0);
        ROUTE[pick(b == n, inner, 3)](i, b, &removal, &mut r);
    }
    let Rebuild {
        changed,
        dead,
        mut rebuilt,
    } = r;
    rebuilt.retain(|i| !(i.annotation() & dead[slot(i.word(0), bound)]));
    *xs = rebuilt;
    changed
}

fn stripped(_: Instruction, _: usize, _: &Removal, r: &mut Rebuild) {
    r.changed = true;
}

fn unreached(i: Instruction, b: usize, c: &Removal, r: &mut Rebuild) {
    r.changed |= !c.required[b] | !c.shells[b];
    let keep = (i.op == Op::Label) & c.required[b];
    let id = i.result() as usize;
    r.dead[id] = (r.dead[id] | (i.layout.defines != 0)) & !keep;
    r.rebuilt.extend(once(i).take(usize::from(keep)));
    let unreachable = || instruction(Op::Unreachable, &[]);
    r.rebuilt
        .extend(repeat_with(unreachable).take(usize::from(keep)));
}

fn reached(i: Instruction, b: usize, c: &Removal, r: &mut Rebuild) {
    [kept, trim_phi][usize::from(i.op == Op::Phi)](i, b, c, r);
}

fn kept(i: Instruction, _: usize, _: &Removal, r: &mut Rebuild) {
    r.rebuilt.push(i);
}

fn trim_phi(i: Instruction, b: usize, c: &Removal, r: &mut Rebuild) {
    let preds = c.preds[b];
    let mut args = i.args[..2].to_vec();
    for pair in i.args[2..].as_chunks::<2>().0 {
        let p = c.graph.labels[pair[1] as usize];
        let parent = preds.iter().fold(false, |any, &q| any | (q == p));
        let kept = c.live[p] & parent;
        args.extend_from_slice(&pair[..2 * usize::from(kept)]);
    }
    r.changed |= args.len() != i.args.len();
    r.rebuilt.push(instruction(Op::Phi, &args));
}

// Original non-speculative value numbering. Only an earlier dominating load
// is reused; no access moves into an untaken branch.
// Ryujinx CodeGen/Spirv/Declarations.cs:126-152 uses Block for cbuf and
// BufferBlock for legacy writable buffers, both with Uniform storage.
// Instructions.cs:1658-1707 loads through the selected buffer pointer.
// AccessChain fields: Khronos grammar :756-765; Load :704-713.
// Uniform=2: :13751-13754; Block=2 and BufferBlock=3: :14843-14853.
// CompositeExtract type/composite/index fields: :937-947; Bitcast :1457-1465.
// Ryujinx Translation/Optimizations/Utils.cs:37-45 treats identical cbuf
// operands as the same immutable input. Storage-buffer loads remain distinct.
struct Candidates {
    first: (usize, u32),
    others: Vec<(usize, u32)>,
}

/// How value numbering treats an opcode: not at all, through a uniform
/// pointer operand, or always.
static NUMBERED: [usize; OPCODES] = {
    let mut k = [0; OPCODES];
    k[at(Op::AccessChain)] = 1;
    k[at(Op::Load)] = 1;
    k[at(Op::CompositeExtract)] = 2;
    k[at(Op::Bitcast)] = 2;
    k
};

struct Numbering {
    expressions: HashMap<(Op, u32, Operands), Candidates>,
    replacements: Vec<u32>,
    dominators: Vec<usize>,
    uniform: Vec<bool>,
    decorated: Vec<bool>,
    changed: bool,
}

fn cse(xs: &mut Vec<Instruction>, bound: usize) -> bool {
    let graph = Graph::new(xs, bound);
    [unnumbered, number][usize::from(graph.formed)](xs, bound, &graph)
}

fn unnumbered(_: &mut Vec<Instruction>, _: usize, _: &Graph) -> bool {
    false
}

fn number(xs: &mut Vec<Instruction>, bound: usize, graph: &Graph) -> bool {
    let mut block_struct = vec![false; bound];
    // The pointee of each Uniform pointer type; 0, the ID SPIR-V reserves,
    // for every other ID.
    let mut uniform_pointer = vec![0; bound];
    let mut uniform = vec![false; bound];
    let mut decorated = vec![false; bound];
    for i in xs.iter() {
        let decorate = i.op == Op::Decorate;
        let k = slot(i.word(0), bound);
        decorated[k] |= decorate;
        block_struct[k] |= decorate & (i.word(1) == 2);
        let pointer = (i.op == Op::TypePointer) & (i.word(1) == 2);
        uniform_pointer[k] = pick(pointer, uniform_pointer[k], i.word(2));
    }
    for i in xs.iter() {
        let variable = (i.op == Op::Variable) & (i.word(2) == 2);
        let k = slot(i.word(1), bound);
        let pointee = uniform_pointer[slot(i.word(0), bound)] as usize;
        uniform[k] = pick(variable, uniform[k], block_struct[pointee]);
    }
    let mut s = Numbering {
        expressions: HashMap::default(),
        replacements: (0..bound as u32).collect(),
        dominators: dominators(graph),
        uniform,
        decorated,
        changed: false,
    };
    // Dominance is explicit in each candidate entry, so serialization order
    // cannot accidentally reuse a value from a sibling or later block.
    for (b, block) in graph.blocks.iter().enumerate() {
        [unnumbered_block, number_block][usize::from(graph.reachable[b])](
            &mut xs[block.code.clone()],
            b,
            &mut s,
        );
    }
    for i in xs.iter_mut() {
        i.rewrite(&s.replacements);
    }
    xs.retain(|i| !i.removed);
    s.changed
}

fn unnumbered_block(_: &mut [Instruction], _: usize, _: &mut Numbering) {}

fn number_block(body: &mut [Instruction], b: usize, s: &mut Numbering) {
    for i in body {
        i.rewrite(&s.replacements);
        let len = s.uniform.len();
        let base = slot(i.word(2), len);
        let result = slot(i.word(1), len);
        let chain = i.op == Op::AccessChain;
        s.uniform[result] = pick(chain, s.uniform[result], s.uniform[base]);
        let eligible = [false, s.uniform[base], true][NUMBERED[code(i.op)]];
        let numbered = eligible & !s.decorated[result];
        [unnumbered_value, number_value][usize::from(numbered)](i, b, s);
    }
}

fn unnumbered_value(_: &mut Instruction, _: usize, _: &mut Numbering) {}

// Most expression keys and first definitions fit inline. Preserve exact
// operand order and the first dominating candidate; literals still follow the
// cited grammar and storage loads stay distinct.
fn number_value(i: &mut Instruction, b: usize, s: &mut Numbering) {
    let id = i.word(1);
    let key = (i.op, i.word(0), Operands::new(&i.args[2..]));
    let mut fresh = false;
    let entries = s.expressions.entry(key).or_insert_with(|| {
        fresh = true;
        Candidates {
            first: (b, id),
            others: Vec::new(),
        }
    });
    let dominators = &s.dominators;
    let (value, found) = once(entries.first)
        .chain(entries.others.iter().copied())
        .fold((0, false), |(value, found), (owner, candidate)| {
            let hit = !found & !fresh & dominates(dominators, owner, b);
            (pick(hit, value, candidate), found | hit)
        });
    entries
        .others
        .extend(once((b, id)).take(usize::from(!found & !fresh)));
    s.replacements[id as usize] = pick(found, s.replacements[id as usize], value);
    i.removed |= found;
    s.changed |= found;
}

// Iterative immediate dominators in reverse postorder. The graph is usually
// a shallow dispatch loop; use indices rather than recursive tree walks.
// Index `n` is the block no edge has reached yet: it is its own dominator, so
// every walk up the tree ends at the entry block or at it.
fn dominators(graph: &Graph) -> Vec<usize> {
    let n = graph.blocks.len();
    let mut visited = vec![false; n];
    let mut postorder = Vec::with_capacity(n);
    let mut stack = vec![(0, false)];
    while !stack.is_empty() {
        let top = stack.len() - 1;
        let (b, exit) = stack[top];
        stack.truncate(top);
        postorder.extend(std::iter::repeat_n(b, usize::from(exit)));
        let enter = !exit & !visited[b];
        visited[b] |= enter;
        stack.extend(std::iter::repeat_n((b, true), usize::from(enter)));
        let successors = &graph.blocks[b].successors;
        for &next in successors
            .iter()
            .rev()
            .take(successors.len() * usize::from(enter))
        {
            stack.extend(std::iter::repeat_n(
                (next, false),
                usize::from(!visited[next]),
            ));
        }
    }
    postorder.reverse();
    let mut rank = vec![0; n + 1];
    for (r, &b) in postorder.iter().enumerate() {
        rank[b] = r;
    }
    let mut idom = vec![n; n + 1];
    idom[0] = 0;
    let mut changed = true;
    while changed {
        changed = false;
        for &b in postorder.iter().skip(1) {
            let mut common = n;
            for &p in graph.blocks[b].predecessors.iter() {
                let p = [common, p][usize::from(idom[p] != n)];
                common = [common, p][usize::from(common == n)];
                let mut p = p;
                while common != p {
                    let up = usize::from(rank[common] > rank[p]);
                    let next = [common, idom[common]][up];
                    p = [idom[p], p][up];
                    common = next;
                }
            }
            changed |= idom[b] != common;
            idom[b] = common;
        }
    }
    idom
}

fn dominates(idom: &[usize], ancestor: usize, mut b: usize) -> bool {
    let mut parent = idom[b];
    while (ancestor != b) & (parent != b) {
        b = parent;
        parent = idom[b];
    }
    ancestor == b
}
