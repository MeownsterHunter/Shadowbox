// Project-owned bounded invocation analysis. It proves addresses and byte masks,
// not shader results. The GPU executes the unchanged module. Builtin mappings:
// Ryujinx CodeGen/Spirv/IoMap.cs:20,25,44. Control-flow operand ordering:
// /usr/include/spirv/unified1/spirv.core.grammar.json:2788-2863; opcodes and
// builtin numbers: spirv.h:737-742,1998-2006. No upstream code is copied.
mod subgroup;

pub(super) fn capture_subgroup(
    words: &[u32],
    a: &Analysis,
    buffers: &mut Vec<super::Buffer>,
) -> Result<Vec<u32>, String> {
    subgroup::instrument(words, a, buffers)
}

use super::{pattern, Analysis, BufferKind, Type};
use shadowbox::spirv::Op;
use std::collections::BTreeMap;

const MAX_STEPS: usize = 131072;
const MAX_INVOCATIONS: u32 = 128;

pub(super) fn local_size(size: &[u32]) -> Result<[u32; 3], String> {
    let size: [u32; 3] = size.try_into().map_err(|_| "invalid LocalSize")?;
    let count = size.into_iter().try_fold(1u32, |n, v| n.checked_mul(v));
    if size.contains(&0) || count.is_none_or(|n| n > MAX_INVOCATIONS) {
        return Err("LocalSize must contain 1..128 total invocations".into());
    }
    Ok(size)
}

fn literal(a: &Analysis, id: u32) -> Result<u32, String> {
    a.constants
        .get(&id)
        .copied()
        .ok_or_else(|| format!("control flow or address depends on unknown/undefined value %{id}"))
}

fn numeric_layout(a: &Analysis, ty: u32, depth: usize) -> Result<bool, String> {
    if depth > 16 {
        return Err("host type nesting exceeds 16".into());
    }
    Ok(match a.ty(ty)? {
        Type::Scalar => a.scalars.contains_key(&ty),
        Type::Uint64 => a.physical,
        Type::Vector(element, _) | Type::Array(element, _) => {
            numeric_layout(a, *element, depth + 1)?
        }
        Type::Struct(members) => {
            for &member in members {
                if !numeric_layout(a, member, depth + 1)? {
                    return Ok(false);
                }
            }
            true
        }
        _ => false,
    })
}

fn initialize_host(a: &mut Analysis, seed: u64) -> Result<(), String> {
    // Seeded bytes have a proved representation only for numeric host types.
    // Boolean function values remain valid; raw host Boolean encodings do not
    // establish Boolean literals for path proofs.
    for (&root, &storage) in &a.storage_classes {
        if matches!(storage, 2 | 9 | 12) && !numeric_layout(a, a.pointer(root)?.ty, 0)? {
            return Err("nonnumeric host memory layout unsupported".into());
        }
    }
    // Sampled-image plans retain their existing constant-coordinate proof.
    // Mixing host-derived coordinates with this extension needs independent
    // sampling proofs. Keep those literals unknown rather than widening it.
    if !a.resources.is_empty() {
        return Ok(());
    }
    for &(root, _) in a.roots.values() {
        for (i, bytes) in a.host_initial[&root].as_chunks::<4>().0.iter().enumerate() {
            // Undefined saved-buffer padding must not prove a branch/address.
            // Ryujinx DiskCache/DiskCacheGpuAccessor.cs:62-69 supplies only
            // saved words, and Translation/Ssa.cs:250-282 preserves undefined.
            if a.memory[&root].data[i * 4..i * 4 + 4]
                .iter()
                .all(|defined| *defined)
            {
                a.memory_literals
                    .insert((root, i * 4), u32::from_le_bytes(*bytes));
            }
        }
    }
    for (&root, p) in &a.pointers {
        if matches!(a.types.get(&p.ty), Some(Type::Struct(_)))
            && matches!(a.interfaces.get(&root), Some(1))
        {
            return Err("compute builtin blocks unsupported".into());
        }
    }
    let push = pattern(seed, 0x50555348, 0, 0, a.push);
    for (&root, &storage) in &a.storage_classes {
        if storage == 9 {
            for (i, bytes) in push.as_chunks::<4>().0.iter().enumerate() {
                a.memory_literals
                    .insert((root, i * 4), u32::from_le_bytes(*bytes));
            }
        }
    }
    Ok(())
}

fn initialize_inputs(a: &mut Analysis, lane: u32, size: [u32; 3]) -> Result<(), String> {
    let coordinates = [
        lane % size[0],
        lane / size[0] % size[1],
        lane / size[0] / size[1],
    ];
    for (&root, &storage) in &a.interfaces {
        if storage != 1 {
            return Err("compute output interface unsupported".into());
        }
        let io =
            a.io.get(&(root, None))
                .ok_or("compute input lacks builtin")?;
        let p = a.pointer(root)?;
        let builtin = io.compute_builtin()?;
        if a.subgroup && matches!(builtin, 27..=29) {
            return Err(
                "subgroup proof uses subgroup lane IDs, not an assumed local-ID mapping".into(),
            );
        }
        let bits = match builtin {
            24 => vec![1; 3],
            25 => size.to_vec(),
            26 => vec![0; 3],
            27 | 28 => coordinates.to_vec(),
            29 if !a.subgroup => vec![lane],
            36 if a.subgroup => vec![32],
            41 if a.subgroup => vec![lane],
            _ => return Err("unsupported compute builtin".into()),
        };
        let ty = if bits.len() == 3 {
            match a.ty(p.ty)? {
                Type::Vector(ty, 3) => *ty,
                _ => return Err("compute builtin must be uvec3".into()),
            }
        } else {
            p.ty
        };
        if a.scalars.get(&ty) != Some(&super::graphics::Scalar::Uint)
            || a.size(p.ty)? != bits.len() * 4
        {
            return Err("compute builtin type mismatch".into());
        }
        for (i, bits) in bits.into_iter().enumerate() {
            a.memory_literals.insert((root, i * 4), bits);
        }
    }
    Ok(())
}

#[derive(Default)]
struct Access {
    reads: Vec<bool>,
    writes: Vec<bool>,
}

// Reject cross-invocation read/write and write/write overlaps even when the
// resulting bytes happen to match. Without synchronization they are races.
fn access(
    a: &Analysis,
    pointer: u32,
    write: bool,
    accesses: &mut BTreeMap<u32, Access>,
) -> Result<(), String> {
    let p = a.pointer(pointer)?;
    let shared = a.roots.values().any(|(root, kind)| {
        *root == p.root && matches!(kind, BufferKind::Storage | BufferKind::Physical)
    }) || a.storage_classes.get(&p.root) == Some(&4);
    if shared {
        let len = a.memory[&p.root].data.len();
        let entry = accesses.entry(p.root).or_insert_with(|| Access {
            reads: vec![false; len],
            writes: vec![false; len],
        });
        let range = a.range(p)?;
        if write {
            entry.writes[range].fill(true);
        } else {
            entry.reads[range].fill(true);
        }
    }
    Ok(())
}

pub(super) fn analyze(
    base: &mut Analysis,
    instructions: &[(Op, &[u32])],
    seed: u64,
    size: [u32; 3],
) -> Result<(), String> {
    if base.subgroup {
        return subgroup::analyze(base, instructions, seed, size);
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
    let mut prior: BTreeMap<u32, Access> = BTreeMap::new();
    let mut steps = 0;
    let mut stores = 0;
    let mut template = base.clone();
    // Host bytes are identical for every invocation. Initialize once, then
    // clone their facts instead of recomputing the PRNG for every lane.
    initialize_host(&mut template, seed)?;
    for lane in 0..size.into_iter().product() {
        let mut a = template.clone();
        initialize_inputs(&mut a, lane, size)?;
        let mut accesses = BTreeMap::new();
        let mut pc = start;
        let mut predecessor = None;
        let mut block = 0;
        loop {
            steps += 1;
            if steps > MAX_STEPS {
                return Err("compute analysis exceeded instruction budget".into());
            }
            let &(op, args) = instructions
                .get(pc)
                .ok_or("control flow falls outside function")?;
            match op {
                Op::Label => {
                    block = args[0];
                    // Phi inputs all refer to the predecessor state, including
                    // mutually dependent loop-carried values. Copy simultaneously.
                    let mut phis = Vec::new();
                    while let Some(&(Op::Phi, args)) = instructions.get(pc + 1) {
                        if matches!(a.ty(args[0])?, Type::Uint64) {
                            return Err("64-bit Phi unsupported".into());
                        }
                        let source = args[2..]
                            .as_chunks::<2>()
                            .0
                            .iter()
                            .find(|pair| Some(pair[1]) == predecessor)
                            .ok_or("missing Phi predecessor")?[0];
                        let bits = a.literals.get(&source).cloned();
                        phis.push((args[0], args[1], a.value(source)?.to_vec(), bits));
                        pc += 1;
                    }
                    for (ty, id, bytes, bits) in phis {
                        a.constants.remove(&id);
                        a.literals.remove(&id);
                        a.result(ty, id, bytes)?;
                        if let Some(bits) = bits {
                            a.set_literal(id, bits);
                        }
                    }
                }
                Op::Branch | Op::BranchConditional | Op::Switch => {
                    let target = match op {
                        Op::Branch => args[0],
                        Op::BranchConditional => {
                            if literal(&a, args[0])? != 0 {
                                args[1]
                            } else {
                                args[2]
                            }
                        }
                        Op::Switch => {
                            let selector = literal(&a, args[0])?;
                            args[2..]
                                .as_chunks::<2>()
                                .0
                                .iter()
                                .find(|pair| pair[0] == selector)
                                .map_or(args[1], |pair| pair[1])
                        }
                        _ => unreachable!(),
                    };
                    predecessor = Some(block);
                    pc = *labels.get(&target).ok_or("missing branch label")?;
                    continue;
                }
                Op::Return => break,
                Op::FunctionEnd | Op::Phi => return Err("invalid function path".into()),
                Op::Load => {
                    if !a.handles.contains_key(&args[2]) {
                        access(&a, args[2], false, &mut accesses)?;
                    }
                    a.instruction(op, args)?;
                }
                Op::Store => {
                    access(&a, args[0], true, &mut accesses)?;
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
                    access(&a, args[2], false, &mut accesses)?;
                    access(&a, args[2], true, &mut accesses)?;
                    a.instruction(op, args)?;
                }
                _ => a.instruction(op, args)?,
            }
            pc += 1;
        }
        stores += a.stores;
        finish_lane(base, &a, accesses, &mut prior)?;
    }
    base.stores = stores;
    Ok(())
}

fn finish_lane(
    base: &mut Analysis,
    a: &Analysis,
    accesses: BTreeMap<u32, Access>,
    prior: &mut BTreeMap<u32, Access>,
) -> Result<(), String> {
    for (root, current) in accesses {
        let earlier = prior.entry(root).or_insert_with(|| Access {
            reads: vec![false; current.reads.len()],
            writes: vec![false; current.writes.len()],
        });
        for i in 0..current.reads.len() {
            if current.writes[i] && (earlier.reads[i] || earlier.writes[i])
                || current.reads[i] && earlier.writes[i]
            {
                return Err(format!(
                    "cross-invocation memory race at allocation %{root}, byte {i}"
                ));
            }
            earlier.reads[i] |= current.reads[i];
            earlier.writes[i] |= current.writes[i];
        }
    }
    // Sampler policy belongs to the device resource, so all invocation
    // paths must agree. Keep it when merging invocation-local analyses.
    merge_resources(base, a)?;
    for &(root, kind) in a.roots.values() {
        if matches!(kind, BufferKind::Storage | BufferKind::Physical) {
            let dest = base.memory.get_mut(&root).unwrap();
            for (i, &written) in a.memory[&root].written.iter().enumerate() {
                if written {
                    dest.data[i] = a.memory[&root].data[i];
                    dest.written[i] = true;
                }
            }
        }
    }
    Ok(())
}

// Atomic pointer/scope/semantics/value ordering and CAS replacement/comparator:
// /usr/include/spirv/unified1/spirv.core.grammar.json:2587-2786; opcodes:
// spirv.h:1984-1997; scopes/semantics: :991-1007,1042-1045. Ryujinx
// CodeGen/Spirv/Instructions.cs:1630-1655 returns the previous word.
// Prove only exclusive per-invocation words. The access merger rejects every
// cross-lane overlap, including atomics, so GPU scheduling cannot change results.
pub(super) fn atomic(a: &mut Analysis, op: Op, args: &[u32]) -> Result<(), String> {
    let count = match op {
        Op::AtomicCompareExchange => 8,
        Op::AtomicIIncrement | Op::AtomicIDecrement => 5,
        _ => 6,
    };
    if args.len() != count {
        return Err("invalid atomic operand count".into());
    }
    let p = a.pointer(args[2])?;
    let class = a.storage_classes[&p.root];
    if !matches!(class, 4 | 12 | 5349)
        || p.ty != args[0]
        || !matches!(
            a.scalars.get(&p.ty),
            Some(super::graphics::Scalar::Uint | super::graphics::Scalar::Sint)
        )
        || !p.offset.is_multiple_of(4)
    {
        return Err("atomic requires an aligned 32-bit shared/storage word".into());
    }
    let scope = literal(a, args[3])?;
    // Relaxed or the production shared-memory AcquireRelease form. Other
    // semantics need their own proof, especially availability/visibility bits.
    let semantics = literal(a, args[4])?;
    if !matches!(scope, 1 | 2) || semantics != 0 && !(class == 4 && semantics == 0x108) {
        return Err("unsupported atomic scope or semantics".into());
    }
    if op == Op::AtomicCompareExchange && literal(a, args[5])? != 0 {
        return Err("unsupported atomic unequal semantics".into());
    }
    let range = a.range(p)?;
    if !a.memory[&p.root].writable || !a.memory[&p.root].data[range.clone()].iter().all(|b| *b) {
        return Err("atomic memory must be writable and defined".into());
    }
    let old = *a
        .memory_literals
        .get(&(p.root, p.offset))
        .ok_or("atomic memory value is unknown")?;
    let value = if args.len() > 5 {
        literal(
            a,
            args[if op == Op::AtomicCompareExchange {
                6
            } else {
                5
            }],
        )?
    } else {
        0
    };
    let new = match op {
        Op::AtomicExchange => value,
        Op::AtomicCompareExchange => {
            if old == literal(a, args[7])? {
                value
            } else {
                old
            }
        }
        Op::AtomicIIncrement => old.wrapping_add(1),
        Op::AtomicIDecrement => old.wrapping_sub(1),
        Op::AtomicIAdd => old.wrapping_add(value),
        Op::AtomicISub => old.wrapping_sub(value),
        Op::AtomicSMin => (old as i32).min(value as i32) as u32,
        Op::AtomicUMin => old.min(value),
        Op::AtomicSMax => (old as i32).max(value as i32) as u32,
        Op::AtomicUMax => old.max(value),
        Op::AtomicAnd => old & value,
        Op::AtomicOr => old | value,
        Op::AtomicXor => old ^ value,
        _ => return Err("unsupported atomic operation".into()),
    };
    a.constants.remove(&args[1]);
    a.literals.remove(&args[1]);
    a.result(args[0], args[1], vec![true; 4])?;
    a.set_literal(args[1], vec![old]);
    a.memory_literals.insert((p.root, p.offset), new);
    a.memory.get_mut(&p.root).unwrap().written[range].fill(true);
    if matches!(class, 12 | 5349) {
        a.stores += 1;
    }
    Ok(())
}

fn merge_resources(base: &mut Analysis, lane: &Analysis) -> Result<(), String> {
    use super::images::ResourceKind;
    for (key, (_, kind)) in &lane.resources {
        if let ResourceKind::Sampler {
            compare: Some(policy),
        } = kind
        {
            let (_, ResourceKind::Sampler { compare }) = base.resources.get_mut(key).unwrap()
            else {
                unreachable!()
            };
            if compare.is_some_and(|old| old != *policy) {
                return Err("sampler comparison policy conflict across invocations".into());
            }
            *compare = Some(*policy);
        }
    }
    Ok(())
}

// Integer literal evaluation only supplies path/address proofs. Unknown float
// calculations retain their dependency masks and cannot steer a branch/address.
// Arithmetic and comparisons: spirv.h:1890-1916,1955-1967. Undefined shift
// counts are rejected rather than applying the host's masked shift semantics.
pub(super) fn fold(op: Op, x: u32, y: u32) -> Option<u32> {
    Some(match op {
        Op::SNegate => x.wrapping_neg(),
        Op::IAdd => x.wrapping_add(y),
        Op::ISub => x.wrapping_sub(y),
        Op::IMul => x.wrapping_mul(y),
        Op::LogicalEqual | Op::IEqual => u32::from(x == y),
        Op::LogicalNotEqual | Op::INotEqual => u32::from(x != y),
        Op::LogicalOr => u32::from(x != 0 || y != 0),
        Op::LogicalAnd => u32::from(x != 0 && y != 0),
        Op::LogicalNot => u32::from(x == 0),
        Op::UGreaterThan => u32::from(x > y),
        Op::SGreaterThan => u32::from((x as i32) > (y as i32)),
        Op::UGreaterThanEqual => u32::from(x >= y),
        Op::SGreaterThanEqual => u32::from((x as i32) >= (y as i32)),
        Op::ULessThan => u32::from(x < y),
        Op::SLessThan => u32::from((x as i32) < (y as i32)),
        Op::ULessThanEqual => u32::from(x <= y),
        Op::SLessThanEqual => u32::from((x as i32) <= (y as i32)),
        Op::ShiftRightLogical if y < 32 => x >> y,
        Op::ShiftRightArithmetic if y < 32 => ((x as i32) >> y) as u32,
        Op::ShiftLeftLogical if y < 32 => x << y,
        Op::BitwiseOr => x | y,
        Op::BitwiseXor => x ^ y,
        Op::BitwiseAnd => x & y,
        Op::Not => !x,
        Op::BitReverse => x.reverse_bits(),
        Op::BitCount => x.count_ones(),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mutually_dependent_loop_phis_use_predecessor_state() {
        let mut a = Analysis {
            concrete: true,
            ..Analysis::default()
        };
        a.types.insert(1, Type::Scalar);
        a.scalars.insert(1, super::super::graphics::Scalar::Uint);
        a.types.insert(2, Type::Scalar);
        a.types.insert(3, Type::Array(1, Some(2)));
        a.types.insert(4, Type::Struct(vec![3]));
        a.types.insert(5, Type::Pointer(12, 4));
        a.types.insert(6, Type::Pointer(12, 1));
        a.strides.insert(3, 4);
        a.blocks.insert(4, false);
        a.bindings.insert(10, (Some(0), Some(0)));
        a.variable(5, 10, 12, None).unwrap();
        for (id, value) in [(20, 0), (21, 1), (22, 2)] {
            a.instruction(Op::Constant, &[1, id, value]).unwrap();
        }
        // Enter with x=1,y=2. One backedge swaps x and y. If Phi results
        // are assigned sequentially, the second incorrectly reads new x=2.
        let instructions: &[(Op, &[u32])] = &[
            (Op::Label, &[100]),
            (Op::Branch, &[200]),
            (Op::Label, &[200]),
            (Op::Phi, &[1, 30, 21, 100, 31, 300]),
            (Op::Phi, &[1, 31, 22, 100, 30, 300]),
            (Op::Phi, &[1, 32, 20, 100, 33, 300]),
            (Op::LoopMerge, &[400, 300, 0]),
            (Op::Branch, &[300]),
            (Op::Label, &[300]),
            (Op::IAdd, &[1, 33, 32, 21]),
            (Op::ULessThan, &[2, 34, 33, 22]),
            (Op::BranchConditional, &[34, 200, 400]),
            (Op::Label, &[400]),
            // The result proves an address. Wrong Phi semantics produce
            // index 2, which is outside the two-element allocation.
            (Op::AccessChain, &[6, 40, 10, 20, 31]),
            (Op::Store, &[40, 30]),
            (Op::Return, &[]),
            (Op::FunctionEnd, &[]),
        ];
        super::super::storage::initialize(&mut a, &crate::value_diff::Inputs::default(), 0)
            .unwrap();
        analyze(&mut a, instructions, 0, [1; 3]).unwrap();
        assert_eq!(
            &a.memory[&10].written[..8],
            &[false, false, false, false, true, true, true, true]
        );
    }

    #[test]
    fn raw_host_booleans_cannot_establish_branch_literals() {
        let mut a = Analysis::default();
        a.types.insert(1, Type::Scalar); // OpTypeBool has no numeric scalar entry.
        a.types.insert(2, Type::Struct(vec![1]));
        a.types.insert(3, Type::Pointer(9, 2));
        a.variable(3, 10, 9, None).unwrap();
        assert!(initialize_host(&mut a, 42)
            .unwrap_err()
            .contains("nonnumeric"));
    }

    #[test]
    fn sampler_policy_is_retained_and_conflicting_lanes_are_rejected() {
        use super::super::images::ResourceKind;
        let mut base = Analysis::default();
        base.resources
            .insert((0, 0), (10, ResourceKind::Sampler { compare: None }));
        let mut lane = base.clone();
        lane.resources.insert(
            (0, 0),
            (
                10,
                ResourceKind::Sampler {
                    compare: Some(true),
                },
            ),
        );
        merge_resources(&mut base, &lane).unwrap();
        assert!(matches!(
            base.resources[&(0, 0)].1,
            ResourceKind::Sampler {
                compare: Some(true)
            }
        ));
        lane.resources.insert(
            (0, 0),
            (
                10,
                ResourceKind::Sampler {
                    compare: Some(false),
                },
            ),
        );
        assert!(merge_resources(&mut base, &lane)
            .unwrap_err()
            .contains("conflict"));
    }

    #[test]
    fn rejects_oversized_and_overflowing_launches() {
        assert_eq!(local_size(&[4, 4, 4]).unwrap(), [4; 3]);
        assert!(local_size(&[0, 1, 1]).is_err());
        assert!(local_size(&[129, 1, 1]).is_err());
        assert!(local_size(&[u32::MAX; 3]).is_err());
    }
}
