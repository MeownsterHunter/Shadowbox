use super::*;

fn inst(words: &mut Vec<u32>, op: Op, args: &[u32]) {
    words.push(((args.len() as u32 + 1) << 16) | u32::from(op));
    words.extend_from_slice(args);
}

fn module(body: &[(Op, &[u32])]) -> Vec<u32> {
    let mut words = vec![0x07230203, 0x00010300, 0, 128, 0];
    for (op, a) in [
        (Op::Capability, vec![1]),
        (Op::Capability, vec![50]),
        (Op::MemoryModel, vec![0, 1]),
        (Op::EntryPoint, vec![4, 10, 0x6e69616d, 0, 9]),
        (Op::ExecutionMode, vec![10, 7]),
        (Op::Decorate, vec![9, 30, 0]),
        (Op::TypeVoid, vec![1]),
        (Op::TypeInt, vec![2, 32, 0]),
        (Op::TypeBool, vec![3]),
        (Op::TypeFloat, vec![4, 32]),
        (Op::TypePointer, vec![5, 3, 2]),
        (Op::TypePointer, vec![6, 7, 2]),
        (Op::TypeFunction, vec![7, 1]),
        (Op::Constant, vec![2, 13, 0]),
        (Op::Constant, vec![2, 14, 7]),
        (Op::Constant, vec![2, 15, 32]),
        (Op::ConstantTrue, vec![3, 16]),
        (Op::Variable, vec![5, 9, 3]),
        (Op::Function, vec![1, 10, 0, 7]),
        (Op::Label, vec![11]),
        (Op::Variable, vec![6, 12, 7]),
    ] {
        inst(&mut words, op, &a);
    }
    for (op, a) in body {
        inst(&mut words, *op, a);
    }
    inst(&mut words, Op::Return, &[]);
    inst(&mut words, Op::FunctionEnd, &[]);
    words
}

fn count(words: &[u32], op: Op) -> usize {
    tally(&parse(words), |i| i.op == op)
}

/// How many instructions of `xs` satisfy `holds`.
fn tally(xs: &[Instruction], holds: impl Fn(&Instruction) -> bool) -> usize {
    xs.iter().map(|i| usize::from(holds(i))).sum()
}

/// The first instruction of `xs` satisfying `holds`, or `xs.len()`.
fn first(xs: &[Instruction], holds: impl Fn(&Instruction) -> bool) -> usize {
    xs.iter()
        .enumerate()
        .rev()
        .fold(xs.len(), |first, (at, i)| {
            [first, at][usize::from(holds(i))]
        })
}

/// The word offset of the first function.
fn function_start(words: &[u32]) -> usize {
    let xs = parse(words);
    xs[..first(&xs, |i| i.op == Op::Function)]
        .iter()
        .map(|i| i.args.len() + 1)
        .sum::<usize>()
        + 5
}

/// Straight-line values and storage of the evaluated fixture.
#[derive(Default)]
struct Values {
    values: HashMap<u32, u32>,
    memory: HashMap<u32, u32>,
}

impl Values {
    fn get(&self, a: &[u32], n: usize) -> u32 {
        self.values[&a[n]]
    }

    fn set(&mut self, a: &[u32], value: u32) {
        self.values.insert(a[1], value);
    }
}

type Eval = fn(&mut Values, &[u32]);

// Independent evaluator for the tested observable integer output. It reads
// bits from the original binary, without calling the optimiser's fold helper.
// Other instructions produce no value.
fn output(words: &[u32]) -> u32 {
    let eval: [(Op, Eval); 12] = [
        (Op::ConstantTrue, |v, a| v.set(a, 1)),
        (Op::ConstantFalse, |v, a| v.set(a, 0)),
        (Op::Constant, |v, a| v.set(a, a[2])),
        (Op::Load, |v, a| v.set(a, v.memory[&a[2]])),
        (Op::Store, |v, a| {
            let value = v.get(a, 1);
            v.memory.insert(a[0], value);
        }),
        (Op::IAdd, |v, a| {
            v.set(a, v.get(a, 2).wrapping_add(v.get(a, 3)))
        }),
        (Op::ISub, |v, a| {
            v.set(a, v.get(a, 2).wrapping_sub(v.get(a, 3)))
        }),
        (Op::IMul, |v, a| {
            v.set(a, v.get(a, 2).wrapping_mul(v.get(a, 3)))
        }),
        (Op::BitwiseOr, |v, a| v.set(a, v.get(a, 2) | v.get(a, 3))),
        (Op::BitwiseXor, |v, a| v.set(a, v.get(a, 2) ^ v.get(a, 3))),
        (Op::BitwiseAnd, |v, a| v.set(a, v.get(a, 2) & v.get(a, 3))),
        (Op::Bitcast, |v, a| v.set(a, v.get(a, 2))),
    ];
    let eval = eval.into_iter().collect::<HashMap<_, _>>();
    let mut values = Values::default();
    for i in parse(words) {
        eval.get(&i.op)
            .into_iter()
            .for_each(|eval| eval(&mut values, &i.args));
    }
    values.memory[&9]
}

#[test]
fn forwards_values_removes_overwritten_and_unread_storage() {
    let mut words = module(&[
        (Op::Store, &[12, 13]),
        (Op::Store, &[12, 14]),
        (Op::Load, &[2, 20, 12]),
        (Op::IAdd, &[2, 21, 20, 14]),
        (Op::Store, &[12, 21]),
        (Op::Load, &[2, 22, 12]),
        (Op::Store, &[9, 22]),
    ]);
    let before = output(&words);
    optimize(&mut words);
    assert_eq!(output(&words), before);
    assert_eq!(before, 14);
    assert_eq!(count(&words, Op::Load), 0);
    assert_eq!(count(&words, Op::Store), 1);
    assert_eq!(count(&words, Op::Variable), 1);
    assert_eq!(count(&words, Op::IAdd), 0);
    assert_eq!(count(&words, Op::TypeFloat), 0);
    assert_eq!(count(&words, Op::Capability), 1);
    let once = words.clone();
    optimize(&mut words);
    assert_eq!(words, once);
}

#[test]
fn promotes_initialized_storage_across_successor_blocks() {
    let mut words = module(&[
        (Op::Store, &[12, 14]),
        (Op::Branch, &[30]),
        (Op::Label, &[30]),
        (Op::Load, &[2, 20, 12]),
        (Op::Store, &[9, 20]),
    ]);
    optimize(&mut words);
    assert_eq!(count(&words, Op::Load), 0);
    assert_eq!(count(&words, Op::Store), 1);
    assert_eq!(count(&words, Op::Variable), 1);
    assert_eq!(output(&words), 7);
}

#[test]
fn escaped_pointers_and_optional_memory_operands_are_conservative() {
    let mut escaped = module(&[
        (Op::Store, &[12, 14]),
        (Op::CopyObject, &[6, 30, 12]),
        (Op::Load, &[2, 20, 30]),
        (Op::Store, &[9, 20]),
    ]);
    optimize(&mut escaped);
    assert_eq!(count(&escaped, Op::Load), 1);
    assert_eq!(count(&escaped, Op::Variable), 2);
    let mut volatile = module(&[
        (Op::Store, &[12, 14]),
        (Op::Load, &[2, 20, 12, 1]),
        (Op::Store, &[9, 20]),
    ]);
    let original = volatile.clone();
    optimize(&mut volatile);
    assert_eq!(volatile, original);
    let mut unknown = module(&[(Op::of(60000), &[20]), (Op::Store, &[9, 14])]);
    let original = unknown.clone();
    optimize(&mut unknown);
    assert_eq!(unknown, original);
}

#[test]
fn literals_equal_to_replaced_ids_are_not_rewritten() {
    let words = module(&[
        (Op::Store, &[12, 14]),
        (Op::Load, &[2, 20, 12]),
        (Op::Branch, &[30]),
        (Op::Label, &[30]),
        (Op::Switch, &[20, 31, 20, 32]),
        (Op::Label, &[31]),
        (Op::Branch, &[32]),
        (Op::Label, &[32]),
        (Op::Store, &[9, 14]),
    ]);
    // Exercise operand rewriting directly, before constant-branch pruning.
    let mut instructions = parse(&words);
    let mut replacements = (0..128).collect::<Vec<_>>();
    replacements[20] = 14;
    for i in &mut instructions {
        i.rewrite(&replacements);
    }
    let switch = &instructions[first(&instructions, |i| i.op == Op::Switch)];
    assert_eq!(switch.args[0], 14);
    assert_eq!(switch.args[2], 20);
}

#[test]
fn wrapping_arithmetic_matches_original_for_generated_programs() {
    let mut seed = 0x12345678u32;
    for _ in 0..256 {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        let mut words = module(&[]);
        let at = function_start(&words);
        let mut constant = Vec::new();
        inst(&mut constant, Op::Constant, &[2, 18, seed]);
        words.splice(at..at, constant);
        words.truncate(words.len() - 2);
        for (op, a) in [
            (Op::Store, vec![12, 18]),
            (Op::Load, vec![2, 20, 12]),
            (Op::IAdd, vec![2, 21, 20, 14]),
            (Op::IMul, vec![2, 22, 21, 18]),
            (Op::BitwiseXor, vec![2, 23, 22, 21]),
            (Op::Store, vec![9, 23]),
            (Op::Return, vec![]),
            (Op::FunctionEnd, vec![]),
        ] {
            inst(&mut words, op, &a);
        }
        let expected = output(&words);
        optimize(&mut words);
        assert_eq!(output(&words), expected);
    }
}

#[test]
fn undefined_shifts_and_float_arithmetic_stay_unfolded() {
    let mut words = module(&[
        (Op::ShiftLeftLogical, &[2, 20, 14, 15]),
        (Op::Store, &[9, 20]),
        (Op::Bitcast, &[4, 21, 14]),
        (Op::FAdd, &[4, 22, 21, 21]),
        (Op::Bitcast, &[2, 23, 22]),
        (Op::Store, &[9, 23]),
    ]);
    optimize(&mut words);
    assert_eq!(count(&words, Op::ShiftLeftLogical), 1);
    assert_eq!(count(&words, Op::FAdd), 1);
}

#[test]
fn bitcasts_preserve_nan_payloads_infinities_and_zero_signs() {
    for bits in [
        0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc12345, 0xffa01234, 1,
    ] {
        let mut words = module(&[
            (Op::Bitcast, &[4, 20, 18]),
            (Op::Bitcast, &[2, 21, 20]),
            (Op::Store, &[9, 21]),
        ]);
        let at = function_start(&words);
        let mut constant = Vec::new();
        inst(&mut constant, Op::Constant, &[2, 18, bits]);
        words.splice(at..at, constant);
        optimize(&mut words);
        assert_eq!(output(&words), bits);
        assert_eq!(count(&words, Op::Bitcast), 0);
    }
}

#[test]
fn unknown_id_decorations_and_execution_modes_leave_module_untouched() {
    for (op, args) in [
        (Op::Decorate, vec![12, 46, 14]),
        (Op::ExecutionMode, vec![10, 38, 14, 14, 14]),
    ] {
        let mut words = module(&[(Op::Store, &[9, 14])]);
        let mut instruction = Vec::new();
        inst(&mut instruction, op, &args);
        words.splice(7..7, instruction);
        let original = words.clone();
        optimize(&mut words);
        assert_eq!(words, original);
    }
}

#[test]
fn broadcast_keeps_ballot_capability() {
    let mut words = module(&[
        (Op::GroupNonUniformBroadcast, &[2, 20, 14, 14, 13]),
        (Op::Store, &[9, 20]),
    ]);
    let mut capability = Vec::new();
    inst(&mut capability, Op::Capability, &[64]);
    words.splice(7..7, capability);
    optimize(&mut words);
    assert_ne!(
        tally(&parse(&words), |i| i.op == Op::Capability
            && *i.args == [64]),
        0
    );
}

fn input_module(body: &[(Op, &[u32])], globals: &[(Op, &[u32])]) -> Vec<u32> {
    let words = module(body);
    let mut xs = parse(&words);
    let at = first(&xs, |i| i.op == Op::EntryPoint);
    let entry = &mut xs[at];
    let mut args = entry.args.to_vec();
    args.push(18);
    entry.args = Operands::new(&args);
    let globals = [
        (Op::Decorate, &[18, 30, 0][..]),
        (Op::Decorate, &[18, 14][..]),
        (Op::TypePointer, &[17, 1, 2][..]),
        (Op::Variable, &[17, 18, 1][..]),
    ]
    .into_iter()
    .chain(globals.iter().copied())
    .map(|(op, args)| cross_block::instruction(op, args))
    .collect::<Vec<_>>();
    let (annotations, declarations): (Vec<_>, Vec<_>) =
        globals.into_iter().partition(|i| i.annotation());
    let at = first(&xs, |i| i.op == Op::TypeVoid);
    xs.splice(at..at, annotations);
    let at = first(&xs, |i| i.op == Op::Function);
    xs.splice(at..at, declarations);
    let mut result = words[..5].to_vec();
    for i in xs {
        inst(&mut result, i.op, &i.args);
    }
    result
}

fn validate(words: &[u32]) {
    use std::{
        io::Write,
        process::{Command, Stdio},
    };
    let bytes = words
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let checked = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.2"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            // The input closes before the validator is awaited.
            child
                .stdin
                .take()
                .into_iter()
                .try_for_each(|mut input| input.write_all(&bytes))
                .and_then(|()| child.wait_with_output())
        })
        .map(|result| {
            (
                result.status.success(),
                String::from_utf8_lossy(&result.stderr).into_owned(),
            )
        })
        .map_err(|error| error.to_string());
    assert_eq!(
        checked.as_ref().map(|(valid, _)| *valid),
        Ok(true),
        "{checked:?}"
    );
}

// Independent execution of the project-owned integer CFG fixtures. Phi
// operands read the previous block's values simultaneously, including loops.
struct Machine<'a> {
    xs: &'a [Instruction],
    values: HashMap<u32, u32>,
    memory: HashMap<u32, u32>,
    labels: HashMap<u32, usize>,
    /// Each label's phis, in order.
    phis: HashMap<u32, Vec<usize>>,
    current: u32,
    previous: u32,
}

/// The next instruction, or the output a return produced.
type Next = Result<usize, u32>;
type Exec<'a> = fn(&mut Machine<'a>, usize) -> Next;
type Setup<'a> = fn(&mut Machine<'a>, usize);

impl<'a> Machine<'a> {
    fn get(&self, at: usize, n: usize) -> u32 {
        self.values[&self.xs[at].args[n]]
    }

    fn set(&mut self, at: usize, value: u32) -> Next {
        self.values.insert(self.xs[at].args[1], value);
        Ok(at + 1)
    }

    fn constant(&mut self, at: usize, value: u32) {
        self.values.insert(self.xs[at].args[1], value);
    }

    /// Constants with one value word are known before execution.
    fn scalar(&mut self, at: usize) {
        let xs = self.xs;
        let a = &xs[at].args;
        self.values
            .extend(std::iter::once((a[1], a[2])).take(usize::from(a.len() == 3)));
    }

    fn mark(&mut self, at: usize) {
        let label = self.xs[at].args[0];
        self.labels.insert(label, at);
        self.phis.insert(label, Vec::new());
        self.current = label;
    }

    fn gather(&mut self, at: usize) {
        self.phis.entry(self.current).or_default().push(at);
    }

    fn enter(&mut self, at: usize) -> Next {
        self.current = self.xs[at].args[0];
        let assigned = self.phis[&self.current]
            .iter()
            .map(|&phi| {
                let phi = &self.xs[phi].args;
                let sources = phi[2..]
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|p| (p[1], p[0]))
                    .collect::<HashMap<_, _>>();
                (phi[1], self.values[&sources[&self.previous]])
            })
            .collect::<Vec<_>>();
        self.values.extend(assigned);
        Ok(at + 1)
    }

    fn pass(&mut self, at: usize) -> Next {
        Ok(at + 1)
    }

    fn jump(&mut self, label: u32) -> Next {
        self.previous = self.current;
        Ok(self.labels[&label])
    }

    fn branch(&mut self, at: usize) -> Next {
        self.jump(self.xs[at].args[0])
    }

    fn conditional(&mut self, at: usize) -> Next {
        let xs = self.xs;
        let a = &xs[at].args;
        self.jump([a[2], a[1]][usize::from(self.get(at, 0) != 0)])
    }

    fn switch(&mut self, at: usize) -> Next {
        let xs = self.xs;
        let a = &xs[at].args;
        let selector = self.get(at, 0);
        let label = a[2..]
            .as_chunks::<2>()
            .0
            .iter()
            .rev()
            .fold(a[1], |label, pair| {
                [label, pair[1]][usize::from(pair[0] == selector)]
            });
        self.jump(label)
    }

    fn load(&mut self, at: usize) -> Next {
        let value = self.memory[&self.xs[at].args[2]];
        self.set(at, value)
    }

    fn store(&mut self, at: usize) -> Next {
        let value = self.get(at, 1);
        self.memory.insert(self.xs[at].args[0], value);
        Ok(at + 1)
    }

    fn add(&mut self, at: usize) -> Next {
        self.set(at, self.get(at, 2).wrapping_add(self.get(at, 3)))
    }

    fn equal(&mut self, at: usize) -> Next {
        self.set(at, u32::from(self.get(at, 2) == self.get(at, 3)))
    }

    fn not_equal(&mut self, at: usize) -> Next {
        self.set(at, u32::from(self.get(at, 2) != self.get(at, 3)))
    }

    fn less(&mut self, at: usize) -> Next {
        self.set(at, u32::from(self.get(at, 2) < self.get(at, 3)))
    }

    fn select(&mut self, at: usize) -> Next {
        let chosen = [self.get(at, 4), self.get(at, 3)][usize::from(self.get(at, 2) != 0)];
        self.set(at, chosen)
    }

    fn returns(&mut self, _: usize) -> Next {
        Err(self.memory[&9])
    }
}

/// Any opcode outside these tables is not a fixture opcode and panics.
fn execute(words: &[u32], input: u32) -> u32 {
    let xs = parse(words);
    let mut machine = Machine {
        xs: &xs,
        values: HashMap::default(),
        memory: HashMap::default(),
        labels: HashMap::default(),
        phis: HashMap::default(),
        current: 0,
        previous: 0,
    };
    machine.memory.insert(18, input);
    let setup: [(Op, Setup<'_>); 5] = [
        (Op::ConstantTrue, |m, at| m.constant(at, 1)),
        (Op::ConstantFalse, |m, at| m.constant(at, 0)),
        (Op::Constant, Machine::scalar),
        (Op::Label, Machine::mark),
        (Op::Phi, Machine::gather),
    ];
    let setup = setup.into_iter().collect::<HashMap<_, _>>();
    for (at, i) in xs.iter().enumerate() {
        setup
            .get(&i.op)
            .into_iter()
            .for_each(|setup| setup(&mut machine, at));
    }
    machine.current = 0;
    let exec: [(Op, Exec<'_>); 16] = [
        (Op::Label, Machine::enter),
        (Op::Phi, Machine::pass),
        (Op::Variable, Machine::pass),
        (Op::LoopMerge, Machine::pass),
        (Op::SelectionMerge, Machine::pass),
        (Op::Load, Machine::load),
        (Op::Store, Machine::store),
        (Op::IAdd, Machine::add),
        (Op::IEqual, Machine::equal),
        (Op::INotEqual, Machine::not_equal),
        (Op::ULessThan, Machine::less),
        (Op::Select, Machine::select),
        (Op::Branch, Machine::branch),
        (Op::BranchConditional, Machine::conditional),
        (Op::Switch, Machine::switch),
        (Op::Return, Machine::returns),
    ];
    let exec = exec.into_iter().collect::<HashMap<_, _>>();
    let start = first(&xs, |i| i.op == Op::Label);
    let finished = (0..10000).try_fold(start, |at, _| exec[&xs[at].op](&mut machine, at));
    let (Ok(output) | Err(output)) =
        finished.map(|_| -> u32 { panic!("fixture did not terminate") });
    output
}

#[test]
fn diamond_phi_preserves_both_dynamic_paths() {
    let mut words = input_module(
        &[
            (Op::Store, &[12, 13]),
            (Op::Load, &[2, 20, 18]),
            (Op::IEqual, &[3, 21, 20, 13]),
            (Op::SelectionMerge, &[30, 0]),
            (Op::BranchConditional, &[21, 31, 32]),
            (Op::Label, &[31]),
            (Op::Store, &[12, 14]),
            (Op::Branch, &[30]),
            (Op::Label, &[32]),
            (Op::Store, &[12, 15]),
            (Op::Branch, &[30]),
            (Op::Label, &[30]),
            (Op::Load, &[2, 22, 12]),
            (Op::Store, &[9, 22]),
        ],
        &[],
    );
    validate(&words);
    let original = words.clone();
    optimize(&mut words);
    validate(&words);
    assert_eq!(count(&words, Op::Phi), 1);
    assert_eq!(count(&words, Op::Load), 1);
    for input in [0, 1, u32::MAX, 0x80000000] {
        assert_eq!(execute(&words, input), execute(&original, input));
    }
    let once = words.clone();
    optimize(&mut words);
    assert_eq!(words, once);
}

#[test]
fn dispatch_loop_removes_unentered_cycle_and_preserves_loop_state() {
    let mut words = input_module(
        &[
            (Op::Variable, &[6, 19, 7]),
            (Op::Store, &[12, 13]),
            (Op::Store, &[19, 13]),
            (Op::Branch, &[30]),
            (Op::Label, &[30]),
            (Op::LoopMerge, &[36, 35, 0]),
            (Op::Branch, &[33]),
            (Op::Label, &[33]),
            (Op::Load, &[2, 20, 19]),
            (Op::SelectionMerge, &[34, 0]),
            (Op::Switch, &[20, 37, 0, 60, 1, 61, 2, 62, 3, 63]),
            (Op::Label, &[60]),
            (Op::Load, &[2, 21, 18]),
            (Op::Store, &[12, 21]),
            (Op::Store, &[19, 40]),
            (Op::Branch, &[34]),
            (Op::Label, &[61]),
            (Op::Load, &[2, 22, 12]),
            (Op::IAdd, &[2, 24, 22, 40]),
            (Op::Store, &[12, 24]),
            (Op::Store, &[19, 42]),
            (Op::Branch, &[34]),
            (Op::Label, &[62]),
            (Op::Store, &[19, 43]),
            (Op::Branch, &[34]),
            (Op::Label, &[63]),
            (Op::Store, &[19, 41]),
            (Op::Branch, &[34]),
            (Op::Label, &[37]),
            (Op::Store, &[19, 42]),
            (Op::Branch, &[34]),
            (Op::Label, &[34]),
            (Op::Branch, &[35]),
            (Op::Label, &[35]),
            (Op::Load, &[2, 25, 19]),
            (Op::INotEqual, &[3, 26, 25, 42]),
            (Op::BranchConditional, &[26, 30, 36]),
            (Op::Label, &[36]),
            (Op::Load, &[2, 27, 12]),
            (Op::Store, &[9, 27]),
        ],
        &[
            (Op::Constant, &[2, 40, 1]),
            (Op::Constant, &[2, 41, 2]),
            (Op::Constant, &[2, 42, u32::MAX]),
            (Op::Constant, &[2, 43, 3]),
        ],
    );
    validate(&words);
    let original = words.clone();
    optimize(&mut words);
    validate(&words);
    let xs = parse(&words);
    assert_ne!(tally(&xs, |i| i.op == Op::Phi), 0);
    assert_eq!(
        tally(&xs, |i| i.op == Op::Label
            && (i.args[0] == 62 || i.args[0] == 63)),
        0
    );
    assert_eq!(tally(&xs, |i| i.op == Op::Variable && i.args[2] == 7), 0);
    let mut seed = 0xf00dcafeu32;
    for _ in 0..256 {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        assert_eq!(execute(&words, seed), execute(&original, seed));
        assert_eq!(execute(&words, seed), seed.wrapping_add(1));
    }
    let once = words.clone();
    optimize(&mut words);
    assert_eq!(words, once);
}

#[test]
fn uninitialized_storage_is_not_promoted() {
    let mut words = module(&[
        (Op::Branch, &[30]),
        (Op::Label, &[30]),
        (Op::Load, &[2, 20, 12]),
        (Op::Store, &[9, 20]),
    ]);
    optimize(&mut words);
    validate(&words);
    assert_eq!(count(&words, Op::Load), 1);
    assert_eq!(count(&words, Op::Variable), 2);
}

#[test]
fn cbuf_load_is_reused_only_when_it_dominates_the_use() {
    let globals: &[(Op, &[u32])] = &[
        (Op::Decorate, &[40, 2]),
        (Op::MemberDecorate, &[40, 0, 35, 0]),
        (Op::Decorate, &[43, 34, 0]),
        (Op::Decorate, &[43, 33, 0]),
        (Op::TypeStruct, &[40, 2]),
        (Op::TypePointer, &[41, 2, 40]),
        (Op::TypePointer, &[42, 2, 2]),
        (Op::Variable, &[41, 43, 2]),
    ];
    let mut dominated = input_module(
        &[
            (Op::AccessChain, &[42, 20, 43, 13]),
            (Op::Load, &[2, 21, 20]),
            (Op::Branch, &[60]),
            (Op::Label, &[60]),
            (Op::AccessChain, &[42, 22, 43, 13]),
            (Op::Load, &[2, 23, 22]),
            (Op::Store, &[9, 23]),
        ],
        globals,
    );
    validate(&dominated);
    optimize(&mut dominated);
    validate(&dominated);
    assert_eq!(count(&dominated, Op::Load), 1);
    assert_eq!(count(&dominated, Op::AccessChain), 1);

    let mut siblings = input_module(
        &[
            (Op::Store, &[12, 13]),
            (Op::Load, &[2, 20, 18]),
            (Op::IEqual, &[3, 21, 20, 13]),
            (Op::SelectionMerge, &[60, 0]),
            (Op::BranchConditional, &[21, 61, 62]),
            (Op::Label, &[61]),
            (Op::AccessChain, &[42, 22, 43, 13]),
            (Op::Load, &[2, 23, 22]),
            (Op::Store, &[12, 23]),
            (Op::Branch, &[60]),
            (Op::Label, &[62]),
            (Op::AccessChain, &[42, 24, 43, 13]),
            (Op::Load, &[2, 25, 24]),
            (Op::Store, &[12, 25]),
            (Op::Branch, &[60]),
            (Op::Label, &[60]),
            (Op::Load, &[2, 26, 12]),
            (Op::Store, &[9, 26]),
        ],
        globals,
    );
    validate(&siblings);
    optimize(&mut siblings);
    validate(&siblings);
    assert_eq!(count(&siblings, Op::Load), 3);
}

#[test]
fn unknown_computation_keeps_conditional_edges_executable() {
    let mut words = input_module(
        &[
            (Op::Bitcast, &[4, 20, 13]),
            (Op::ExtInst, &[4, 21, 44, 13, 20]), // GLSL.std.450 Sin, already accepted by parser.
            (Op::FOrdEqual, &[3, 22, 21, 20]),
            (Op::SelectionMerge, &[30, 0]),
            (Op::BranchConditional, &[22, 31, 32]),
            (Op::Label, &[31]),
            (Op::Store, &[9, 14]),
            (Op::Branch, &[30]),
            (Op::Label, &[32]),
            (Op::Store, &[9, 15]),
            (Op::Branch, &[30]),
            (Op::Label, &[30]),
        ],
        &[],
    );
    let mut ext = Vec::new();
    inst(
        &mut ext,
        Op::ExtInstImport,
        &[44, 0x4c534c47, 0x6474732e, 0x3035342e, 0],
    );
    // Extended-instruction imports precede the memory model.
    words.splice(9..9, ext);
    validate(&words);
    optimize(&mut words);
    validate(&words);
    assert_eq!(count(&words, Op::BranchConditional), 1);
}

#[test]
fn specialization_constants_are_runtime_inputs_to_phis() {
    let mut words = input_module(
        &[
            (Op::Store, &[12, 13]),
            (Op::Load, &[2, 20, 18]),
            (Op::IEqual, &[3, 21, 20, 13]),
            (Op::SelectionMerge, &[30, 0]),
            (Op::BranchConditional, &[21, 31, 32]),
            (Op::Label, &[31]),
            (Op::Store, &[12, 40]),
            (Op::Branch, &[30]),
            (Op::Label, &[32]),
            (Op::Store, &[12, 15]),
            (Op::Branch, &[30]),
            (Op::Label, &[30]),
            (Op::Load, &[2, 22, 12]),
            (Op::Store, &[9, 22]),
        ],
        &[(Op::SpecConstant, &[2, 40, 7]), (Op::Decorate, &[40, 1, 0])],
    );
    validate(&words);
    optimize(&mut words);
    validate(&words);
    assert_eq!(count(&words, Op::SpecConstant), 1);
    assert_eq!(count(&words, Op::Phi), 1);
}

#[test]
fn constant_exit_removes_the_acyclic_dispatch_loop_marker() {
    let mut words = module(&[
        (Op::Store, &[12, 13]),
        (Op::Branch, &[30]),
        (Op::Label, &[30]),
        (Op::LoopMerge, &[36, 35, 0]),
        (Op::Branch, &[33]),
        (Op::Label, &[33]),
        (Op::Load, &[2, 20, 12]),
        (Op::SelectionMerge, &[34, 0]),
        (Op::Switch, &[20, 37, 0, 60]),
        (Op::Label, &[60]),
        (Op::Store, &[12, 14]),
        (Op::Branch, &[34]),
        (Op::Label, &[37]),
        (Op::Store, &[12, 14]),
        (Op::Branch, &[34]),
        (Op::Label, &[34]),
        (Op::Branch, &[35]),
        (Op::Label, &[35]),
        (Op::Load, &[2, 25, 12]),
        (Op::INotEqual, &[3, 26, 25, 14]),
        (Op::BranchConditional, &[26, 30, 36]),
        (Op::Label, &[36]),
        (Op::Load, &[2, 27, 12]),
        (Op::Store, &[9, 27]),
    ]);
    validate(&words);
    let original = words.clone();
    optimize(&mut words);
    validate(&words);
    assert_eq!(count(&words, Op::LoopMerge), 0);
    assert_eq!(count(&words, Op::Switch), 1);
    assert_eq!(execute(&words, 0), execute(&original, 0));
}

#[test]
fn mutable_storage_buffer_loads_are_not_commoned() {
    let mut words = input_module(
        &[
            (Op::AccessChain, &[42, 20, 43, 13]),
            (Op::Load, &[2, 21, 20]),
            (Op::Store, &[20, 14]),
            (Op::Branch, &[60]),
            (Op::Label, &[60]),
            (Op::AccessChain, &[42, 22, 43, 13]),
            (Op::Load, &[2, 23, 22]),
            (Op::IAdd, &[2, 24, 21, 23]),
            (Op::Store, &[9, 24]),
        ],
        &[
            (Op::Decorate, &[40, 2]),
            (Op::MemberDecorate, &[40, 0, 35, 0]),
            (Op::Decorate, &[43, 34, 0]),
            (Op::Decorate, &[43, 33, 0]),
            (Op::TypeStruct, &[40, 2]),
            (Op::TypePointer, &[41, 12, 40]),
            (Op::TypePointer, &[42, 12, 2]),
            (Op::Variable, &[41, 43, 12]),
        ],
    );
    validate(&words);
    optimize(&mut words);
    validate(&words);
    assert_eq!(count(&words, Op::Load), 2);
}

#[test]
fn default_only_switch_keeps_its_target_and_side_effect() {
    let mut words = input_module(
        &[
            (Op::Load, &[2, 20, 18]),
            (Op::SelectionMerge, &[30, 0]),
            (Op::Switch, &[20, 31]),
            (Op::Label, &[31]),
            (Op::Store, &[9, 14]),
            (Op::Branch, &[30]),
            (Op::Label, &[30]),
        ],
        &[],
    );
    validate(&words);
    let original = words.clone();
    optimize(&mut words);
    validate(&words);
    assert_eq!(count(&words, Op::Switch), 1);
    for input in [0, 1, u32::MAX] {
        assert_eq!(execute(&words, input), execute(&original, input));
    }
}

#[test]
fn dead_block_decorations_do_not_reference_deleted_results() {
    let mut words = input_module(
        &[
            (Op::SelectionMerge, &[30, 0]),
            (Op::BranchConditional, &[16, 31, 32]),
            (Op::Label, &[31]),
            (Op::Store, &[9, 14]),
            (Op::Branch, &[30]),
            (Op::Label, &[32]),
            (Op::FMul, &[4, 45, 40, 40]),
            (Op::Bitcast, &[2, 46, 45]),
            (Op::Store, &[9, 46]),
            (Op::Branch, &[30]),
            (Op::Label, &[30]),
        ],
        &[(Op::Constant, &[4, 40, 0]), (Op::Decorate, &[45, 42])],
    );
    validate(&words);
    let original = words.clone();
    optimize(&mut words);
    validate(&words);
    assert_eq!(
        tally(&parse(&words), |i| i.op == Op::Decorate && i.args[0] == 45),
        0
    );
    assert_eq!(execute(&words, 0), execute(&original, 0));
}

#[test]
fn terminating_case_removes_unreachable_loop_continuation() {
    let mut words = module(&[
        (Op::Store, &[12, 13]),
        (Op::Branch, &[30]),
        (Op::Label, &[30]),
        (Op::LoopMerge, &[36, 35, 0]),
        (Op::Branch, &[33]),
        (Op::Label, &[33]),
        (Op::Load, &[2, 20, 12]),
        (Op::SelectionMerge, &[34, 0]),
        (Op::Switch, &[20, 37, 0, 60]),
        (Op::Label, &[60]),
        (Op::Kill, &[]),
        (Op::Label, &[37]),
        (Op::Store, &[12, 14]),
        (Op::Branch, &[34]),
        (Op::Label, &[34]),
        (Op::Branch, &[35]),
        (Op::Label, &[35]),
        (Op::Load, &[2, 25, 12]),
        (Op::INotEqual, &[3, 26, 25, 14]),
        (Op::BranchConditional, &[26, 30, 36]),
        (Op::Label, &[36]),
        (Op::Store, &[9, 14]),
    ]);
    validate(&words);
    optimize(&mut words);
    validate(&words);
    assert_eq!(count(&words, Op::LoopMerge), 0);
    assert_eq!(count(&words, Op::Kill), 1);
    assert_eq!(count(&words, Op::Store), 0);
}

#[test]
fn folded_outer_selection_preserves_nested_early_exit() {
    let mut words = input_module(
        &[
            (Op::SelectionMerge, &[30, 0]),
            (Op::Switch, &[13, 32, 0, 31]),
            (Op::Label, &[31]),
            (Op::Load, &[2, 20, 18]),
            (Op::IEqual, &[3, 21, 20, 13]),
            (Op::SelectionMerge, &[33, 0]),
            (Op::BranchConditional, &[21, 34, 33]),
            (Op::Label, &[34]),
            (Op::Store, &[9, 14]),
            (Op::Branch, &[30]),
            (Op::Label, &[33]),
            (Op::Store, &[9, 15]),
            (Op::Branch, &[30]),
            (Op::Label, &[32]),
            (Op::Store, &[9, 13]),
            (Op::Branch, &[30]),
            (Op::Label, &[30]),
        ],
        &[],
    );
    validate(&words);
    let original = words.clone();
    optimize(&mut words);
    validate(&words);
    for input in [0, 1, u32::MAX] {
        assert_eq!(execute(&words, input), execute(&original, input));
    }
}

#[test]
fn folded_loop_exit_preserves_nested_break_and_continue() {
    for exit in [35, 36] {
        let mut words = input_module(
            &[
                (Op::Branch, &[30]),
                (Op::Label, &[30]),
                (Op::LoopMerge, &[36, 35, 0]),
                (Op::Branch, &[31]),
                (Op::Label, &[31]),
                (Op::Load, &[2, 20, 18]),
                (Op::IEqual, &[3, 21, 20, 13]),
                (Op::SelectionMerge, &[33, 0]),
                (Op::BranchConditional, &[21, 34, 33]),
                (Op::Label, &[34]),
                (Op::Store, &[9, 14]),
                (Op::Branch, &[exit]),
                (Op::Label, &[33]),
                (Op::Store, &[9, 15]),
                (Op::Branch, &[35]),
                (Op::Label, &[35]),
                (Op::BranchConditional, &[16, 36, 30]),
                (Op::Label, &[36]),
            ],
            &[],
        );
        validate(&words);
        let original = words.clone();
        optimize(&mut words);
        validate(&words);
        for input in [0, 1, u32::MAX] {
            assert_eq!(execute(&words, input), execute(&original, input));
        }
        assert_eq!(count(&words, Op::LoopMerge), 1);
        let once = words.clone();
        optimize(&mut words);
        assert_eq!(words, once);
    }
}

#[test]
fn nested_if_else_constant_conditions_preserve_phis_and_outputs() {
    for outer in [16, 40, 21] {
        for inner in [16, 40, 21] {
            let mut words = input_module(
                &[
                    (Op::Store, &[12, 13]),
                    (Op::Load, &[2, 20, 18]),
                    (Op::IEqual, &[3, 21, 20, 13]),
                    (Op::SelectionMerge, &[30, 0]),
                    (Op::BranchConditional, &[outer, 31, 32]),
                    (Op::Label, &[31]),
                    (Op::SelectionMerge, &[35, 0]),
                    (Op::BranchConditional, &[inner, 36, 37]),
                    (Op::Label, &[36]),
                    (Op::Store, &[12, 14]),
                    (Op::Branch, &[35]),
                    (Op::Label, &[37]),
                    (Op::Store, &[12, 15]),
                    (Op::Branch, &[35]),
                    (Op::Label, &[35]),
                    (Op::Branch, &[30]),
                    (Op::Label, &[32]),
                    (Op::Store, &[12, 13]),
                    (Op::Branch, &[30]),
                    (Op::Label, &[30]),
                    (Op::Load, &[2, 22, 12]),
                    (Op::Store, &[9, 22]),
                ],
                &[(Op::ConstantFalse, &[3, 40])],
            );
            validate(&words);
            let original = words.clone();
            optimize(&mut words);
            validate(&words);
            for input in [0, 1, u32::MAX, 0x8000_0000] {
                assert_eq!(execute(&words, input), execute(&original, input));
            }
            let once = words.clone();
            optimize(&mut words);
            assert_eq!(words, once);
        }
    }
}

#[test]
fn terminating_nested_selection_keeps_unreachable_merge_shells() {
    let mut words = input_module(
        &[
            (Op::SelectionMerge, &[30, 0]),
            (Op::Switch, &[13, 32, 0, 31]),
            (Op::Label, &[31]),
            (Op::Load, &[2, 20, 18]),
            (Op::IEqual, &[3, 21, 20, 13]),
            (Op::SelectionMerge, &[33, 0]),
            (Op::BranchConditional, &[21, 34, 35]),
            (Op::Label, &[34]),
            (Op::Return, &[]),
            (Op::Label, &[35]),
            (Op::Return, &[]),
            (Op::Label, &[33]),
            (Op::Branch, &[30]),
            (Op::Label, &[32]),
            (Op::Branch, &[30]),
            (Op::Label, &[30]),
        ],
        &[],
    );
    validate(&words);
    optimize(&mut words);
    validate(&words);
    let xs = parse(&words);
    for label in [30, 33] {
        let at = first(&xs, |i| i.op == Op::Label && i.args[0] == label);
        assert_eq!(xs[at + 1].op, Op::Unreachable);
    }
    let once = words.clone();
    optimize(&mut words);
    assert_eq!(words, once);
}

#[test]
fn switch_id_masks_preserve_literals_beyond_word_64() {
    let mut args = vec![20, 31];
    for _ in 0..70 {
        args.extend_from_slice(&[20, 32]);
    }
    let expected: Vec<_> = (0..args.len())
        .flat_map(|i| std::iter::once(i).take(usize::from(i < 2 || i % 2 == 1)))
        .collect();
    let mut instruction = Instruction::new(Op::Switch, &args);
    let mut inputs = Vec::new();
    instruction.inputs(|at| inputs.push(at));
    assert_eq!(inputs, expected);
    let mut replacements: Vec<_> = (0..128).collect();
    replacements[20] = 99;
    replacements[31] = 100;
    replacements[32] = 101;
    instruction.rewrite(&replacements);
    args = instruction.args.to_vec();
    assert_eq!(&args[..2], &[99, 100]);
    for pair in args[2..].as_chunks::<2>().0 {
        assert_eq!(pair, &[20, 101]);
    }
}
