// Subgroup operand and result shapes follow Ryujinx CodeGen/Spirv/Instructions.cs:
// 227-237 (ballot), 1080-1117 (shuffle), 1569-1587 (votes), and IoMap.cs:38
// (lane ID). Numeric fields: /usr/include/spirv/unified1/spirv.h:748,753,
// 1044 (Subgroup scope),1143-1148 (capabilities),2082-2096 (opcodes).
// Project-owned collective interpreter. No upstream implementation is copied.
mod regions;
use super::*;

pub(super) fn analyze(
    base: &mut Analysis,
    instructions: &[(Op, &[u32])],
    seed: u64,
    size: [u32; 3],
) -> Result<(), String> {
    if size != [32, 1, 1] {
        return Err("subgroup capture requires one full 32-lane subgroup".into());
    }
    if base.maximal_reconvergence {
        return regions::analyze(base, instructions, seed, size);
    }
    let start = instructions
        .iter()
        .position(|(op, _)| *op == Op::Label)
        .ok_or("missing entry block")?;
    let labels = instructions
        .iter()
        .enumerate()
        .filter(|(_, (op, _))| *op == Op::Label)
        .map(|(i, (_, a))| (a[0], i))
        .collect::<BTreeMap<_, _>>();
    // Process executed blocks in module order. Every taken predecessor has
    // finished before a join. Structural loop declarations in the production
    // EXIT wrapper are allowed; an actual backward edge fails below. This
    // neither guesses convergence nor replays collectives independently.
    let mut template = base.clone();
    initialize_host(&mut template, seed)?;
    let mut lanes = (0..32)
        .map(|lane| {
            let mut a = template.clone();
            initialize_inputs(&mut a, lane, size)?;
            Ok(a)
        })
        .collect::<Result<Vec<_>, String>>()?;
    let mut pcs = [start; 32];
    let mut blocks = [0; 32];
    let mut predecessors = [None; 32];
    let mut accesses = (0..32).map(|_| BTreeMap::new()).collect::<Vec<_>>();
    let sites = instructions
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
            )
        })
        .enumerate()
        .map(|(site, (pc, _))| (pc, site))
        .collect::<BTreeMap<_, _>>();
    base.subgroup_masks = vec![vec![0; 32]; sites.len()];
    let mut steps = 0;
    for (pc, &(op, args)) in instructions.iter().enumerate().skip(start) {
        let active = (0..32).filter(|&lane| pcs[lane] == pc).collect::<Vec<_>>();
        if active.is_empty() {
            continue;
        }
        steps += active.len();
        if steps > MAX_STEPS {
            return Err("subgroup instruction budget exceeded".into());
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
        ) {
            let mask = active.iter().fold(0u32, |mask, &lane| mask | (1 << lane));
            for &lane in &active {
                base.subgroup_masks[sites[&pc]][lane] = mask;
            }
            collective(&mut lanes, &active, op, args)?;
            for lane in active {
                pcs[lane] += 1;
            }
            continue;
        }
        for lane in active {
            let a = &mut lanes[lane];
            match op {
                Op::Label => {
                    blocks[lane] = args[0];
                    let mut phis = Vec::new();
                    while let Some(&(Op::Phi, args)) = instructions.get(pcs[lane] + 1) {
                        let source = args[2..]
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .find(|p| Some(p[1]) == predecessors[lane])
                            .ok_or("missing subgroup Phi predecessor")?[0];
                        phis.push((
                            args[0],
                            args[1],
                            a.value(source)?.to_vec(),
                            a.literals.get(&source).cloned(),
                        ));
                        pcs[lane] += 1;
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
                Op::Branch | Op::BranchConditional | Op::Switch => {
                    let target = match op {
                        Op::Branch => args[0],
                        Op::BranchConditional => {
                            if literal(a, args[0])? != 0 {
                                args[1]
                            } else {
                                args[2]
                            }
                        }
                        _ => {
                            let selector = literal(a, args[0])?;
                            args[2..]
                                .as_chunks::<2>()
                                .0
                                .iter()
                                .find(|p| p[0] == selector)
                                .map_or(args[1], |p| p[1])
                        }
                    };
                    predecessors[lane] = Some(blocks[lane]);
                    let next = *labels.get(&target).ok_or("missing subgroup branch label")?;
                    if next <= pc {
                        return Err(
                            "subgroup loops/backward paths require a convergence proof".into()
                        );
                    }
                    pcs[lane] = next;
                    continue;
                }
                Op::Return => {
                    pcs[lane] = usize::MAX;
                    continue;
                }
                Op::FunctionEnd | Op::Phi => return Err("invalid subgroup function path".into()),
                Op::Load => {
                    if !a.handles.contains_key(&args[2]) {
                        access(a, args[2], false, &mut accesses[lane])?;
                    }
                    a.instruction(op, args)?;
                }
                Op::Store => {
                    access(a, args[0], true, &mut accesses[lane])?;
                    a.instruction(op, args)?;
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
                    access(a, args[2], false, &mut accesses[lane])?;
                    access(a, args[2], true, &mut accesses[lane])?;
                    a.instruction(op, args)?;
                }
                _ => a.instruction(op, args)?,
            }
            pcs[lane] += 1;
        }
    }
    if pcs.iter().any(|&pc| pc != usize::MAX) {
        return Err("subgroup invocation did not return".into());
    }
    let mut prior = BTreeMap::new();
    base.stores = 0;
    for (a, accesses) in lanes.iter().zip(accesses) {
        base.stores += a.stores;
        finish_lane(base, a, accesses, &mut prior)?;
    }
    Ok(())
}

fn collective(
    lanes: &mut [Analysis],
    active: &[usize],
    op: Op,
    args: &[u32],
) -> Result<(), String> {
    let expected = if matches!(
        op,
        Op::GroupNonUniformShuffle
            | Op::GroupNonUniformShuffleXor
            | Op::GroupNonUniformShuffleUp
            | Op::GroupNonUniformShuffleDown
            | Op::GroupNonUniformUMin
    ) {
        5
    } else {
        4
    };
    if args.len() != expected
        || active
            .iter()
            .any(|&lane| lanes[lane].constants.get(&args[2]) != Some(&3))
    {
        return Err("collective requires Subgroup scope and exact operands".into());
    }
    let values = active
        .iter()
        .map(|&lane| {
            lanes[lane]
                .literals
                .get(&args[if op == Op::GroupNonUniformUMin { 4 } else { 3 }])
                .cloned()
                .ok_or("unknown/undefined collective input")
        })
        .collect::<Result<Vec<_>, _>>()?;
    if matches!(
        op,
        Op::GroupNonUniformAll | Op::GroupNonUniformAny | Op::GroupNonUniformBallot
    ) && values.iter().any(|v| v.len() != 1 || v[0] > 1)
    {
        return Err("vote/ballot requires a proved Boolean predicate".into());
    }
    if op == Op::GroupNonUniformAllEqual
        && active.iter().any(|&lane| {
            let ty = lanes[lane].value_types.get(&args[3]);
            let element = ty.map(|ty| match lanes[lane].types.get(ty) {
                Some(Type::Vector(element, _)) => *element,
                _ => *ty,
            });
            element.is_some_and(|ty| {
                lanes[lane].scalars.get(&ty) == Some(&super::super::graphics::Scalar::Float)
            })
        })
    {
        return Err("floating AllEqual requires numerical equality proof".into());
    }
    let results = active
        .iter()
        .map(|&lane| {
            Ok(match op {
                // Ordered token dispatch takes unsigned minimum with Reduce.
                // Mesa spirv.h:1034,1043,2064; project reconvergence.rs:271-285.
                Op::GroupNonUniformUMin
                    if args[3] == 0
                        && values.iter().all(|v| v.len() == 1)
                        && active.iter().all(|&lane| {
                            lanes[lane].scalars.get(&args[0])
                                == Some(&super::super::graphics::Scalar::Uint)
                        }) =>
                {
                    vec![values.iter().map(|v| v[0]).min().unwrap()]
                }
                Op::GroupNonUniformUMin => {
                    return Err("subgroup UMin requires uint32 Reduce".into())
                }
                Op::GroupNonUniformAll => vec![u32::from(values.iter().all(|v| v[0] != 0))],
                Op::GroupNonUniformAny => vec![u32::from(values.iter().any(|v| v[0] != 0))],
                Op::GroupNonUniformAllEqual => {
                    vec![u32::from(values.iter().all(|v| *v == values[0]))]
                }
                Op::GroupNonUniformBallot => vec![
                    active
                        .iter()
                        .zip(&values)
                        .fold(0, |mask, (&lane, v)| mask | (v[0] << lane)),
                    0,
                    0,
                    0,
                ],
                Op::GroupNonUniformShuffle
                | Op::GroupNonUniformShuffleXor
                | Op::GroupNonUniformShuffleUp
                | Op::GroupNonUniformShuffleDown => {
                    let index = literal(&lanes[lane], args[4])?;
                    let target = match op {
                        Op::GroupNonUniformShuffle => Some(index),
                        Op::GroupNonUniformShuffleXor => Some(lane as u32 ^ index),
                        Op::GroupNonUniformShuffleUp => (lane as u32).checked_sub(index),
                        _ => (lane as u32).checked_add(index),
                    }
                    .filter(|&target| target < 32)
                    .ok_or("shuffle target outside subgroup")?
                        as usize;
                    if !active.contains(&target) {
                        return Err("shuffle source lane is inactive".into());
                    }
                    lanes[target]
                        .literals
                        .get(&args[3])
                        .cloned()
                        .ok_or("undefined shuffle source")?
                }
                _ => unreachable!(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    for (&lane, bits) in active.iter().zip(results) {
        let a = &mut lanes[lane];
        a.constants.remove(&args[1]);
        a.literals.remove(&args[1]);
        a.result(args[0], args[1], vec![true; bits.len() * 4])?;
        a.set_literal(args[1], bits);
    }
    Ok(())
}

pub(super) fn instrument(
    words: &[u32],
    a: &Analysis,
    buffers: &mut Vec<super::super::Buffer>,
) -> Result<Vec<u32>, String> {
    use super::super::{
        graphics::{Binary, Scalar},
        Buffer,
    };
    let sites = a.subgroup_masks.len();
    if sites == 0 || sites > 512 {
        return Err("subgroup capture exceeds 64 KiB".into());
    }
    let traced = a.maximal_reconvergence;
    let capacities = (0..sites)
        .map(|site| {
            if traced {
                a.subgroup_traces[site]
                    .iter()
                    .map(Vec::len)
                    .max()
                    .unwrap_or(0)
                    .max(1)
            } else {
                0
            }
        })
        .collect::<Vec<_>>();
    let mut extent = 0usize;
    let bases = capacities
        .iter()
        .map(|capacity| {
            let base = extent;
            extent += (capacity + 1) * 32;
            base
        })
        .collect::<Vec<_>>();
    if extent > 16384 {
        return Err("subgroup dynamic mask trace exceeds 64 KiB".into());
    }
    let mut b = Binary::new(words[3]);
    for (&id, t) in &a.types {
        let signature = match t {
            Type::Scalar => Some(match a.scalars.get(&id) {
                Some(Scalar::Uint) => (Op::TypeInt, vec![32, 0]),
                Some(Scalar::Sint) => (Op::TypeInt, vec![32, 1]),
                Some(Scalar::Float) => (Op::TypeFloat, vec![32]),
                None => (Op::TypeBool, Vec::new()),
            }),
            Type::Vector(element, count) => Some((Op::TypeVector, vec![*element, *count as u32])),
            Type::Pointer(storage, element) => Some((Op::TypePointer, vec![*storage, *element])),
            _ => None,
        };
        if let Some(signature) = signature {
            b.types.insert(signature, id);
        }
    }
    let uint = b.ty(Op::TypeInt, &[32, 0]);
    let boolean = b.ty(Op::TypeBool, &[]);
    let vector = b.ty(Op::TypeVector, &[uint, 4]);
    let needed = [0, 1, 3, 32, extent as u32]
        .into_iter()
        .chain(bases.iter().map(|&base| base as u32))
        .chain(capacities.iter().map(|&capacity| capacity as u32))
        .collect::<std::collections::BTreeSet<_>>();
    let constants = needed
        .into_iter()
        .map(|n| (n, b.result(Op::Constant, &[uint, n])))
        .collect::<BTreeMap<_, _>>();
    let truth = b.result(Op::ConstantTrue, &[boolean]);
    let array = b.ty(Op::TypeArray, &[uint, constants[&(extent as u32)]]);
    let block = b.ty(Op::TypeStruct, &[array]);
    let pblock = b.ty(Op::TypePointer, &[12, block]);
    let pword = b.ty(Op::TypePointer, &[12, uint]);
    let capture = b.result(Op::Variable, &[pblock, 12]);
    let mut decoration = Binary::new(0);
    decoration.op(Op::Decorate, &[array, 6, 4]);
    decoration.op(Op::Decorate, &[block, 2]);
    decoration.op(Op::MemberDecorate, &[block, 0, 35, 0]);
    decoration.op(Op::Decorate, &[capture, 34, 3]);
    decoration.op(Op::Decorate, &[capture, 33, 257]);
    let existing = a.io.iter().find_map(|(&(root, member), io)| {
        (member.is_none()
            && io.compute_builtin().ok() == Some(41)
            && a.interfaces.get(&root) == Some(&1))
        .then_some(root)
    });
    let lane = if let Some(root) = existing {
        root
    } else {
        let pointer = b.ty(Op::TypePointer, &[1, uint]);
        let root = b.result(Op::Variable, &[pointer, 1]);
        decoration.op(Op::Decorate, &[root, 11, 41]);
        root
    };
    // Function-local counters witness every dynamic execution, not only the
    // final mask at a site. An unexpected extra execution writes its count but
    // never writes outside the bounded trace. Function-variable initialization:
    // SPIR-V OpVariable: spirv.h:1827; spirv.core.grammar.json:679-706.
    // Counter values and the bounded trace layout are checker-owned state.
    let mut local_variables = Vec::new();
    let function_pointer = if traced {
        b.ty(Op::TypePointer, &[7, uint])
    } else {
        0
    };
    let cursors = (0..sites)
        .map(|_| {
            if traced {
                let id = b.next;
                b.next += 1;
                local_variables.extend([
                    (5 << 16) | u32::from(Op::Variable),
                    function_pointer,
                    id,
                    7,
                    constants[&0],
                ]);
                id
            } else {
                0
            }
        })
        .collect::<Vec<_>>();
    let mut tails = Vec::new();
    let mut next = b.next;
    for site in 0..sites {
        let mut t = Binary::new(next);
        let id = t.result(Op::Load, &[uint, lane]);
        let ballot = t.result(Op::GroupNonUniformBallot, &[vector, constants[&3], truth]);
        let mask = t.result(Op::CompositeExtract, &[uint, ballot, 0]);
        let base = t.result(Op::IAdd, &[uint, id, constants[&(bases[site] as u32)]]);
        let index = if traced {
            let count = t.result(Op::Load, &[uint, cursors[site]]);
            let increment = t.result(Op::IAdd, &[uint, count, constants[&1]]);
            t.op(Op::Store, &[cursors[site], increment]);
            let count_pointer = t.result(Op::AccessChain, &[pword, capture, constants[&0], base]);
            t.op(Op::Store, &[count_pointer, increment]);
            let fits = t.result(
                Op::ULessThan,
                &[boolean, count, constants[&(capacities[site] as u32)]],
            );
            let body = t.next;
            let merge = t.next + 1;
            t.next += 2;
            t.op(Op::SelectionMerge, &[merge, 0]);
            t.op(Op::BranchConditional, &[fits, body, merge]);
            t.op(Op::Label, &[body]);
            let offset = t.result(Op::IMul, &[uint, increment, constants[&32]]);
            let index = t.result(Op::IAdd, &[uint, base, offset]);
            let pointer = t.result(Op::AccessChain, &[pword, capture, constants[&0], index]);
            t.op(Op::Store, &[pointer, mask]);
            t.op(Op::Branch, &[merge]);
            t.op(Op::Label, &[merge]);
            None
        } else {
            Some(base)
        };
        if let Some(index) = index {
            let pointer = t.result(Op::AccessChain, &[pword, capture, constants[&0], index]);
            t.op(Op::Store, &[pointer, mask]);
        }
        next = t.next;
        tails.push(t.words);
    }
    buffers.push(Buffer {
        set: 3,
        binding: 257,
        kind: BufferKind::Storage,
        initial: vec![0; extent * 4],
        defined: vec![true; extent * 4],
        observation: Some("compute:active masks".into()),
        expected: Some(if traced {
            let mut expected = vec![0u32; extent];
            for (site, traces) in a.subgroup_traces.iter().enumerate() {
                for (lane, trace) in traces.iter().enumerate() {
                    expected[bases[site] + lane] = trace.len() as u32;
                    for (event, &mask) in trace.iter().enumerate() {
                        expected[bases[site] + (event + 1) * 32 + lane] = mask;
                    }
                }
            }
            expected.iter().flat_map(|n| n.to_le_bytes()).collect()
        } else {
            a.subgroup_masks
                .iter()
                .flatten()
                .flat_map(|n| n.to_le_bytes())
                .collect()
        }),
    });
    let mut observed = words[..5].to_vec();
    observed[3] = next;
    // The added witness ballot requires its capability even when the original
    // module only declared vote or shuffle. All original capabilities remain.
    observed.extend_from_slice(&[(2 << 16) | 17, 64]);
    let mut at = 5;
    let mut decorated = false;
    let mut site = 0;
    let mut locals_inserted = false;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        let op = Op::of(words[at]);
        if !decorated
            && matches!(
                op,
                Op::TypeVoid
                    | Op::TypeBool
                    | Op::TypeInt
                    | Op::TypeFloat
                    | Op::TypeVector
                    | Op::TypeMatrix
                    | Op::TypeImage
                    | Op::TypeSampler
                    | Op::TypeSampledImage
                    | Op::TypeArray
                    | Op::TypeRuntimeArray
                    | Op::TypeStruct
                    | Op::TypeOpaque
                    | Op::TypePointer
                    | Op::TypeFunction
                    | Op::TypeEvent
                    | Op::TypeDeviceEvent
                    | Op::TypeReserveId
                    | Op::TypeQueue
                    | Op::TypePipe
                    | Op::TypeForwardPointer
                    | Op::ConstantTrue
                    | Op::ConstantFalse
                    | Op::Constant
                    | Op::ConstantComposite
                    | Op::ConstantSampler
                    | Op::ConstantNull
                    | Op::SpecConstantTrue
                    | Op::SpecConstantFalse
                    | Op::SpecConstant
                    | Op::SpecConstantComposite
                    | Op::SpecConstantOp
                    | Op::Function
                    | Op::Variable
            )
        {
            observed.extend_from_slice(&decoration.words);
            decorated = true;
        }
        if op == Op::Function {
            observed.extend_from_slice(&b.words);
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
            observed.extend_from_slice(&tails[site]);
            site += 1;
        }
        if op == Op::EntryPoint {
            let mut operands = words[at + 1..at + count].to_vec();
            if existing.is_none() {
                operands.push(lane);
            }
            if words[1] >= 0x00010400 {
                operands.push(capture);
            }
            observed.push(((operands.len() as u32 + 1) << 16) | 15);
            observed.extend(operands);
        } else {
            observed.extend_from_slice(&words[at..at + count]);
        }
        if op == Op::Label && !locals_inserted {
            observed.extend_from_slice(&local_variables);
            locals_inserted = true;
        }
        at += count;
    }
    Ok(observed)
}
