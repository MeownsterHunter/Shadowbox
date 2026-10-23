// Emission semantics: Ryujinx Instructions/InstEmitAttribute.cs:253-295 and
// CodeGen/Spirv/Instructions.cs:517-529. Execution modes: SpirvGenerator.cs:
// 239-261. Numeric modes, streams and emit/cut opcodes: public
// /usr/include/spirv/unified1/spirv.h:119-147,539,1136,1976-1979.
// Capture design and bounded event sequence are project-owned experiments.
// No original operation is removed or replaced by instrumentation.
use super::*;

#[derive(Clone)]
pub(in crate::value_diff::module) struct Event {
    pub emit: bool,
    pub stream: u32,
    pub masks: BTreeMap<u32, Vec<bool>>,
}

pub(in crate::value_diff::module) fn record(
    a: &mut Analysis,
    op: Op,
    args: &[u32],
) -> Result<(), String> {
    let stream = if matches!(op, Op::EmitStreamVertex | Op::EndStreamPrimitive) {
        *a.constants
            .get(&args[0])
            .ok_or("unproved geometry stream")?
    } else {
        0
    };
    if stream > 3 || a.geometry_events.len() >= 64 {
        return Err("geometry stream/event count exceeds capture bounds".into());
    }
    let emit = matches!(op, Op::EmitVertex | Op::EmitStreamVertex);
    let masks = a
        .interfaces
        .iter()
        .filter(|(_, storage)| **storage == 3)
        .map(|(&root, _)| (root, a.memory[&root].data.clone()))
        .collect();
    a.geometry_events.push(Event {
        emit,
        stream,
        masks,
    });
    if emit {
        // Project-owned conservative proof rule: forget output definitions
        // after emission. Reusing the prior mask would claim an unwritten
        // next vertex is defined. Require a new write before capturing it.
        for (&root, &storage) in &a.interfaces {
            if storage == 3 {
                a.memory.get_mut(&root).unwrap().data.fill(false);
            }
        }
    }
    Ok(())
}

pub(super) fn prepare(
    words: &[u32],
    a: &Analysis,
    buffers: &mut Vec<Buffer>,
    seed: u64,
) -> Result<Graphics, String> {
    if a.stores != 0 || a.interfaces.values().any(|&storage| storage == 1) {
        return Err(
            "geometry capture currently requires output-only IO and no source storage writes"
                .into(),
        );
    }
    let input = a
        .graphics_modes
        .keys()
        .filter(|&&m| matches!(m, 19 | 22))
        .copied()
        .collect::<Vec<_>>();
    let output = a
        .graphics_modes
        .keys()
        .filter(|&&m| matches!(m, 27..=29))
        .copied()
        .collect::<Vec<_>>();
    let max_vertices = *a
        .graphics_modes
        .get(&26)
        .ok_or("geometry lacks OutputVertices")?;
    if input.len() != 1
        || output.len() != 1
        || !(1..=32).contains(&max_vertices)
        || a.graphics_modes.get(&0).is_some_and(|&n| n != 1)
    {
        return Err(
            "geometry requires one invocation, a supported topology and 1..32 vertices".into(),
        );
    }
    if a.geometry_events.is_empty()
        || a.geometry_events.iter().filter(|e| e.emit).count() > max_vertices as usize
    {
        return Err("missing geometry events or OutputVertices exceeded".into());
    }
    let mut outputs = Vec::new();
    for (&root, &storage) in &a.interfaces {
        if storage != 3 {
            continue;
        }
        let ty = a.pointers[&root].ty;
        let io = a.io.get(&(root, None)).cloned().unwrap_or_default();
        if io.builtin.is_some_and(|b| !matches!(b, 0 | 1)) {
            return Err("geometry output builtin unsupported".into());
        }
        leaves(a, root, ty, &io, Vec::new(), 0, &mut outputs)?;
    }
    outputs.sort_by(|x, y| x.key.cmp(&y.key));
    if outputs.is_empty() || outputs.len() > 16 || outputs.windows(2).any(|p| p[0].key == p[1].key)
    {
        return Err("missing, duplicate or too many geometry outputs".into());
    }
    // Blocks with member stream decorations need member-level stream reflection.
    // Until then reject them, rather than assuming the variable stream.
    if outputs
        .iter()
        .any(|o| matches!(a.types.get(&a.pointers[&o.root].ty), Some(Type::Struct(_))))
    {
        return Err("geometry output blocks need member stream reflection".into());
    }
    let stream_of = |o: &Leaf| a.io.get(&(o.root, None)).map_or(0, |io| io.stream);
    if output[0] != 27
        && (outputs.iter().any(|o| stream_of(o) != 0)
            || a.geometry_events.iter().any(|e| e.stream != 0))
    {
        return Err("nonzero geometry streams require point output".into());
    }
    let events = &a.geometry_events;
    let mut b = Binary::new(words[3]);
    // Intern original numeric/pointer types before declaring new ones.
    for (&id, ty) in &a.types {
        let signature = match ty {
            Type::Scalar => a.scalars.get(&id).map(|s| match s {
                Scalar::Uint => (Op::TypeInt, vec![32, 0]),
                Scalar::Sint => (Op::TypeInt, vec![32, 1]),
                Scalar::Float => (Op::TypeFloat, vec![32]),
            }),
            Type::Pointer(storage, element) => Some((Op::TypePointer, vec![*storage, *element])),
            _ => None,
        };
        if let Some(signature) = signature {
            b.types.insert(signature, id);
        }
    }
    let uint = b.ty(Op::TypeInt, &[32, 0]);
    let mut needed = (0..=events.len() as u32).collect::<std::collections::BTreeSet<_>>();
    needed.insert(1);
    for o in &outputs {
        needed.extend(&o.path);
    }
    for e in events {
        needed.insert((if e.emit { 1 } else { 2 }) | e.stream << 8);
    }
    let constants = needed
        .into_iter()
        .map(|v| (v, b.result(Op::Constant, &[uint, v])))
        .collect::<BTreeMap<_, _>>();
    let array = b.ty(Op::TypeArray, &[uint, constants[&(events.len() as u32)]]);
    let block = b.ty(Op::TypeStruct, &[array]);
    let pb = b.ty(Op::TypePointer, &[12, block]);
    let pu = b.ty(Op::TypePointer, &[12, uint]);
    let mut decor = Binary::new(0);
    decor.op(Op::Decorate, &[array, 6, 4]);
    decor.op(Op::Decorate, &[block, 2]);
    decor.op(Op::MemberDecorate, &[block, 0, 35, 0]);
    let mut vars = Vec::new();
    let mut any_defined = false;
    for i in 0..=outputs.len() + 1 {
        let var = b.result(Op::Variable, &[pb, 12]);
        vars.push(var);
        decor.op(Op::Decorate, &[var, 34, 3]);
        decor.op(Op::Decorate, &[var, 33, 256 + i as u32]);
        let mut defined = Vec::new();
        for event in events {
            let mask = if let Some(o) = outputs.get(i) {
                if event.emit && event.stream == stream_of(o) {
                    event.masks[&o.root][o.offset..o.offset + 4].to_vec()
                } else {
                    vec![false; 4]
                }
            } else {
                vec![true; 4]
            };
            if i < outputs.len() && mask.iter().all(|b| *b) {
                any_defined = true;
            }
            defined.extend(mask);
        }
        let observation = if let Some(o) = outputs.get(i) {
            format!("geometry:stream:{}:{}", stream_of(o), o.key)
        } else if i == outputs.len() {
            format!("geometry:events:topology:{}", output[0])
        } else {
            "__completion".into()
        };
        buffers.push(Buffer {
            set: 3,
            binding: 256 + i as u32,
            kind: BufferKind::Storage,
            initial: vec![0; events.len() * 4],
            defined,
            observation: Some(observation),
            expected: None,
        });
    }
    if !any_defined {
        return Err("no defined emitted geometry values".into());
    }
    let mut pointers = BTreeMap::new();
    for o in &outputs {
        pointers
            .entry(o.ty)
            .or_insert_with(|| b.ty(Op::TypePointer, &[3, o.ty]));
    }
    let mut tails = Vec::new();
    let mut next = b.next;
    for (slot, event) in events.iter().enumerate() {
        let mut tail = Binary::new(next);
        for (i, o) in outputs.iter().enumerate() {
            if !event.emit || event.stream != stream_of(o) {
                continue;
            }
            let mask = &event.masks[&o.root][o.offset..o.offset + 4];
            if mask.iter().all(|b| !*b) {
                continue;
            }
            if !mask.iter().all(|b| *b) {
                return Err("partially undefined geometry scalar".into());
            }
            let source = if o.path.is_empty() {
                o.root
            } else {
                let mut args = vec![pointers[&o.ty], o.root];
                args.extend(o.path.iter().map(|i| constants[i]));
                tail.result(Op::AccessChain, &args)
            };
            let loaded = tail.result(Op::Load, &[o.ty, source]);
            let bits = if o.ty == uint {
                loaded
            } else {
                tail.result(Op::Bitcast, &[uint, loaded])
            };
            let target = tail.result(
                Op::AccessChain,
                &[pu, vars[i], constants[&0], constants[&(slot as u32)]],
            );
            tail.op(Op::Store, &[target, bits]);
        }
        let tag = (if event.emit { 1 } else { 2 }) | event.stream << 8;
        for (i, value) in [(outputs.len(), tag), (outputs.len() + 1, 1)] {
            let target = tail.result(
                Op::AccessChain,
                &[pu, vars[i], constants[&0], constants[&(slot as u32)]],
            );
            tail.op(Op::Store, &[target, constants[&value]]);
        }
        next = tail.next;
        tails.push(tail.words);
    }
    let mut instrumented = words[..5].to_vec();
    instrumented[3] = next;
    let mut at = 5;
    let mut decorated = false;
    let mut event = 0;
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
            instrumented.extend_from_slice(&decor.words);
            decorated = true;
        }
        if op == Op::Function {
            instrumented.extend_from_slice(&b.words);
        }
        if matches!(
            op,
            Op::EmitVertex | Op::EndPrimitive | Op::EmitStreamVertex | Op::EndStreamPrimitive
        ) {
            instrumented.extend_from_slice(&tails[event]);
            event += 1;
        }
        if op == Op::EntryPoint && words[1] >= 0x00010400 {
            let mut operands = words[at + 1..at + count].to_vec();
            operands.extend_from_slice(&vars);
            instrumented.push(((operands.len() as u32 + 1) << 16) | 15);
            instrumented.extend(operands);
        } else {
            instrumented.extend_from_slice(&words[at..at + count]);
        }
        at += count;
    }
    Ok(Graphics {
        words: instrumented,
        partner: partner(false, &[], seed),
        vertex: false,
        stage: 3,
        topology: if input[0] == 19 {
            vk::PrimitiveTopology::POINT_LIST
        } else {
            vk::PrimitiveTopology::TRIANGLE_LIST
        },
        vertices: if input[0] == 19 { 1 } else { 3 },
        streams: a.geometry_streams,
        tess_partner: None,
        attributes: Vec::new(),
    })
}
