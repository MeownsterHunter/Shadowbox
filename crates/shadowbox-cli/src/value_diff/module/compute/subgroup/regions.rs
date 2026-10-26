// Structured subgroup execution under maximal reconvergence. Token consumers
// wait at their target: Ryujinx InstEmitFlowControl.cs:239-275, Decoder.cs:824-869.
// Loop/selection merge operands: Mesa src/compiler/spirv/spirv.h:1961-1966;
// maximal mode: :210. SPV_KHR_maximal_reconvergence, Maximal Reconvergence,
// SPIRV-Registry extensions/KHR/SPV_KHR_maximal_reconvergence.asciidoc:
// 252-280 defines selection, loop-merge and continue-target entering sets.
// It defines the entering active set reunited at a loop merge. Nested execution
// below holds departing lanes there until every entering lane departs.
// This is checker-owned bounded analysis, not a translated shader substitute.
use super::*;
use std::collections::BTreeSet;

type Arrivals = BTreeMap<usize, Vec<usize>>;

struct Block {
    merge: Option<(bool, usize, Option<usize>)>,
    branches: Arrivals,
}

struct Machine<'a, 'b> {
    instructions: &'a [(Op, &'a [u32])],
    base: &'b mut Analysis,
    lanes: Vec<Analysis>,
    blocks: [u32; 32],
    predecessors: [Option<u32>; 32],
    accesses: Vec<BTreeMap<u32, Access>>,
    labels: BTreeMap<u32, usize>,
    sites: BTreeMap<usize, usize>,
    steps: usize,
}

fn append(into: &mut Arrivals, from: Arrivals) {
    for (pc, lanes) in from {
        into.entry(pc).or_default().extend(lanes);
    }
}

impl Machine<'_, '_> {
    fn label(&self, id: u32) -> Result<usize, String> {
        self.labels
            .get(&id)
            .copied()
            .ok_or_else(|| "missing region branch label".into())
    }

    fn block(&mut self, active: &[usize], mut pc: usize) -> Result<Block, String> {
        let mut merge = None;
        loop {
            let &(op, args) = self
                .instructions
                .get(pc)
                .ok_or("region falls outside function")?;
            self.steps += active.len();
            if self.steps > MAX_STEPS {
                return Err("subgroup instruction budget exceeded".into());
            }
            if op == Op::Phi {
                let first = pc;
                while self
                    .instructions
                    .get(pc)
                    .is_some_and(|(op, _)| *op == Op::Phi)
                {
                    pc += 1;
                }
                for &lane in active {
                    let a = &mut self.lanes[lane];
                    let mut phis = Vec::new();
                    for &(_, args) in &self.instructions[first..pc] {
                        let source = args[2..]
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .find(|p| Some(p[1]) == self.predecessors[lane])
                            .ok_or("missing region Phi predecessor")?[0];
                        phis.push((
                            args[0],
                            args[1],
                            a.value(source)?.to_vec(),
                            a.literals.get(&source).cloned(),
                        ));
                    }
                    for (ty, id, mask, bits) in phis {
                        a.constants.remove(&id);
                        a.literals.remove(&id);
                        a.result(ty, id, mask)?;
                        if let Some(bits) = bits {
                            a.set_literal(id, bits);
                        }
                    }
                }
                continue;
            }
            if matches!(
                op,
                Op::GroupNonUniformAll
                    | Op::GroupNonUniformAny
                    | Op::GroupNonUniformAllEqual
                    | Op::GroupNonUniformBallot
                    | Op::GroupNonUniformShuffle
                    | Op::GroupNonUniformShuffleXor
                    | Op::GroupNonUniformShuffleUp
                    | Op::GroupNonUniformShuffleDown
                    | Op::GroupNonUniformUMin
            ) {
                let mask = active.iter().fold(0u32, |mask, &lane| mask | (1 << lane));
                for &lane in active {
                    self.base.subgroup_masks[self.sites[&pc]][lane] = mask;
                    self.base.subgroup_traces[self.sites[&pc]][lane].push(mask);
                }
                collective(&mut self.lanes, active, op, args)?;
            } else if matches!(op, Op::LoopMerge | Op::SelectionMerge) {
                if merge
                    .replace((
                        op == Op::LoopMerge,
                        self.label(args[0])?,
                        if op == Op::LoopMerge {
                            Some(self.label(args[1])?)
                        } else {
                            None
                        },
                    ))
                    .is_some()
                {
                    return Err("multiple region merges in one block".into());
                }
            } else if matches!(
                op,
                Op::Branch | Op::BranchConditional | Op::Switch | Op::Return
            ) {
                let mut branches = Arrivals::new();
                for &lane in active {
                    let target = match op {
                        Op::Return => usize::MAX,
                        Op::Branch => self.label(args[0])?,
                        Op::BranchConditional => {
                            self.label(if literal(&self.lanes[lane], args[0])? != 0 {
                                args[1]
                            } else {
                                args[2]
                            })?
                        }
                        _ => {
                            let selector = literal(&self.lanes[lane], args[0])?;
                            let id = args[2..]
                                .as_chunks::<2>()
                                .0
                                .iter()
                                .find(|p| p[0] == selector)
                                .map_or(args[1], |p| p[1]);
                            self.label(id)?
                        }
                    };
                    self.predecessors[lane] = Some(self.blocks[lane]);
                    branches.entry(target).or_default().push(lane);
                }
                return Ok(Block { merge, branches });
            } else {
                for &lane in active {
                    let a = &mut self.lanes[lane];
                    match op {
                        Op::Label => self.blocks[lane] = args[0],
                        Op::FunctionEnd => return Err("region invocation did not return".into()),
                        Op::Load => {
                            if !a.handles.contains_key(&args[2]) {
                                access(a, args[2], false, &mut self.accesses[lane])?;
                            }
                            a.instruction(op, args)
                                .map_err(|e| format!("region instruction {pc}: {e}"))?;
                        }
                        Op::Store => {
                            access(a, args[0], true, &mut self.accesses[lane])?;
                            a.instruction(op, args)
                                .map_err(|e| format!("region instruction {pc}: {e}"))?;
                        }
                        Op::AtomicExchange
                        | Op::AtomicCompareExchange
                        | Op::AtomicIIncrement
                        | Op::AtomicIDecrement
                        | Op::AtomicIAdd
                        | Op::AtomicISub
                        | Op::AtomicSMin
                        | Op::AtomicUMin
                        | Op::AtomicSMax
                        | Op::AtomicUMax
                        | Op::AtomicAnd
                        | Op::AtomicOr
                        | Op::AtomicXor => {
                            access(a, args[2], false, &mut self.accesses[lane])?;
                            access(a, args[2], true, &mut self.accesses[lane])?;
                            a.instruction(op, args)
                                .map_err(|e| format!("region instruction {pc}: {e}"))?;
                        }
                        _ => a
                            .instruction(op, args)
                            .map_err(|e| format!("region instruction {pc}: {e}"))?,
                    }
                }
            }
            pc += 1;
        }
    }

    fn walk(
        &mut self,
        mut active: Vec<usize>,
        mut pc: usize,
        stops: &BTreeSet<usize>,
        depth: usize,
    ) -> Result<Arrivals, String> {
        if depth > 64 {
            return Err("subgroup region nesting exceeds 64".into());
        }
        let mut result = Arrivals::new();
        loop {
            if active.is_empty() {
                return Ok(result);
            }
            if pc == usize::MAX || stops.contains(&pc) {
                result.entry(pc).or_default().extend(active);
                return Ok(result);
            }
            let mut block = self.block(&active, pc)?;
            if let Some((is_loop, merge, continuing)) = block.merge {
                let mut nested = stops.clone();
                nested.insert(merge);
                if is_loop {
                    nested.insert(pc);
                    nested.insert(continuing.unwrap());
                }
                let mut at_merge = Vec::new();
                loop {
                    let mut arrivals = Arrivals::new();
                    for (next, lanes) in block.branches {
                        append(&mut arrivals, self.walk(lanes, next, &nested, depth + 1)?);
                    }
                    if let Some(continuing) = continuing {
                        let at_continue = arrivals.remove(&continuing).unwrap_or_default();
                        if !at_continue.is_empty() {
                            let mut continuation_stops = nested.clone();
                            continuation_stops.remove(&continuing);
                            append(
                                &mut arrivals,
                                self.walk(at_continue, continuing, &continuation_stops, depth + 1)?,
                            );
                        }
                    }
                    at_merge.extend(arrivals.remove(&merge).unwrap_or_default());
                    let back = if is_loop {
                        arrivals.remove(&pc).unwrap_or_default()
                    } else {
                        Vec::new()
                    };
                    append(&mut result, arrivals);
                    if back.is_empty() {
                        break;
                    }
                    // No lane that already left the loop is reactivated on an
                    // iteration. It rejoins only at the maximal loop merge.
                    block = self.block(&back, pc)?;
                    if block.merge != Some((true, merge, continuing)) {
                        return Err("inconsistent subgroup loop header".into());
                    }
                }
                active = at_merge;
                pc = merge;
            } else if block.branches.len() == 1 {
                let (next, lanes) = block.branches.pop_first().unwrap();
                if next <= pc && !stops.contains(&next) {
                    return Err("backward subgroup path lacks a structured loop merge".into());
                }
                active = lanes;
                pc = next;
            } else {
                // A loop condition can branch from a body block directly to
                // its merge or continue target without a selection header.
                // Those departures wait at the enclosing frame's stop.
                let mut running = None;
                for (next, lanes) in block.branches {
                    if next == usize::MAX || stops.contains(&next) {
                        result.entry(next).or_default().extend(lanes);
                    } else if running.replace((next, lanes)).is_some() {
                        return Err(format!(
                            "divergent subgroup branch at {pc} lacks a structured merge"
                        ));
                    }
                }
                let Some((next, lanes)) = running else {
                    return Ok(result);
                };
                if next <= pc {
                    return Err("backward subgroup path lacks a structured loop merge".into());
                }
                active = lanes;
                pc = next;
            }
        }
    }
}

pub(super) fn analyze(
    base: &mut Analysis,
    instructions: &[(Op, &[u32])],
    seed: u64,
    size: [u32; 3],
) -> Result<(), String> {
    let start = instructions
        .iter()
        .position(|(op, _)| *op == Op::Label)
        .ok_or("missing region entry")?;
    let labels = instructions
        .iter()
        .enumerate()
        .filter(|(_, (op, _))| *op == Op::Label)
        .map(|(pc, (_, args))| (args[0], pc))
        .collect();
    let sites: BTreeMap<_, _> = instructions
        .iter()
        .enumerate()
        .filter(|(_, (op, _))| {
            matches!(
                op,
                Op::GroupNonUniformAll
                    | Op::GroupNonUniformAny
                    | Op::GroupNonUniformAllEqual
                    | Op::GroupNonUniformBallot
                    | Op::GroupNonUniformShuffle
                    | Op::GroupNonUniformShuffleXor
                    | Op::GroupNonUniformShuffleUp
                    | Op::GroupNonUniformShuffleDown
                    | Op::GroupNonUniformUMin
            )
        })
        .enumerate()
        .map(|(site, (pc, _))| (pc, site))
        .collect();
    base.subgroup_masks = vec![vec![0; 32]; sites.len()];
    base.subgroup_traces = vec![vec![Vec::new(); 32]; sites.len()];
    let mut template = base.clone();
    initialize_host(&mut template, seed)?;
    let lanes = (0..32)
        .map(|lane| {
            let mut a = template.clone();
            initialize_inputs(&mut a, lane, size)?;
            Ok(a)
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut machine = Machine {
        instructions,
        base,
        lanes,
        blocks: [0; 32],
        predecessors: [None; 32],
        accesses: (0..32).map(|_| BTreeMap::new()).collect(),
        labels,
        sites,
        steps: 0,
    };
    let arrivals = machine.walk((0..32).collect(), start, &BTreeSet::new(), 0)?;
    if arrivals.len() != 1
        || arrivals
            .get(&usize::MAX)
            .is_none_or(|lanes| lanes.len() != 32)
    {
        return Err("subgroup region invocation did not return".into());
    }
    machine.base.stores = 0;
    let mut prior = BTreeMap::new();
    for (a, accesses) in machine.lanes.iter().zip(machine.accesses) {
        machine.base.stores += a.stores;
        finish_lane(machine.base, a, accesses, &mut prior)?;
    }
    Ok(())
}
