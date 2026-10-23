// Stage modes and builtin representations follow Ryujinx
// CodeGen/Spirv/SpirvGenerator.cs:190-237 and IoMap.cs:28,30,41-43.
// Numeric fields: public spirv.h:79-85,119-145,525,725-728,1970-2006,1042-1045,990-1007,1087.
// CAS operands: spirv.core.grammar.json:2599-2614. Pure evaluation may be
// invoked more than once for a generated vertex. A single winning capture
// per coordinate avoids duplicate writes; the shader itself is unchanged.
use super::*;

pub(super) fn check(stage: u32, a: &Analysis) -> Result<(), String> {
    if stage == 1 {
        if a.graphics_modes.get(&26) != Some(&1) {
            return Err("control capture requires OutputVertices 1".into());
        }
        // With one control invocation there is one writer for every patch
        // output and tessellation level. Require finite bounded levels before
        // allowing the partner evaluation stage to consume them.
        for builtin in [11, 12] {
            let root =
                a.io.iter()
                    .find_map(|(&(root, member), io)| {
                        (member.is_none()
                            && io.builtin == Some(builtin)
                            && a.interfaces.get(&root) == Some(&3))
                        .then_some(root)
                    })
                    .ok_or("control capture requires standalone tessellation level outputs")?;
            let count = if builtin == 11 { 4 } else { 2 };
            if a.size(a.pointer(root)?.ty)? != count * 4 {
                return Err("tessellation level array extent mismatch".into());
            }
            for i in 0..count {
                let bits = *a
                    .memory_literals
                    .get(&(root, i * 4))
                    .ok_or("unproved tessellation level")?;
                let value = f32::from_bits(bits);
                if !a.memory[&root].data[i * 4..i * 4 + 4].iter().all(|b| *b)
                    || !value.is_finite()
                    || !(1.0..=4.0).contains(&value)
                {
                    return Err("tessellation levels must be defined finite values in 1..4".into());
                }
            }
        }
    } else if !a.graphics_modes.contains_key(&22)
        || !a.graphics_modes.contains_key(&1)
        || (a.graphics_modes.contains_key(&4) == a.graphics_modes.contains_key(&5))
    {
        return Err(
            "evaluation capture requires triangles, equal spacing and one vertex order".into(),
        );
    }
    Ok(())
}

pub(super) struct Evaluation {
    pub coordinate: u32,
    pub injected: bool,
    float: u32,
    vector: u32,
    boolean: u32,
    one: u32,
}

pub(super) fn setup(
    a: &Analysis,
    b: &mut Binary,
    decoration: &mut Binary,
) -> Result<Evaluation, String> {
    let float = a
        .scalars
        .iter()
        .find_map(|(&ty, s)| (*s == Scalar::Float).then_some(ty))
        .unwrap_or_else(|| b.ty(Op::TypeFloat, &[32]));
    let boolean = a
        .types
        .iter()
        .find_map(|(&ty, t)| {
            (matches!(t, Type::Scalar) && !a.scalars.contains_key(&ty)).then_some(ty)
        })
        .unwrap_or_else(|| b.ty(Op::TypeBool, &[]));
    let vector = a
        .types
        .iter()
        .find_map(|(&ty, t)| matches!(t,Type::Vector(element,3) if *element == float).then_some(ty))
        .unwrap_or_else(|| b.ty(Op::TypeVector, &[float, 3]));
    let existing = a.io.iter().find_map(|(&(root, member), io)| {
        (member.is_none() && io.builtin == Some(13) && a.interfaces.get(&root) == Some(&1))
            .then_some(root)
    });
    let coordinate = if let Some(root) = existing {
        if a.pointer(root)?.ty != vector {
            return Err("TessCoord must be vec3".into());
        }
        root
    } else {
        let pointer = a
            .types
            .iter()
            .find_map(|(&ty, t)| {
                matches!(t,Type::Pointer(1,element) if *element == vector).then_some(ty)
            })
            .unwrap_or_else(|| b.ty(Op::TypePointer, &[1, vector]));
        let root = b.result(Op::Variable, &[pointer, 1]);
        decoration.op(Op::Decorate, &[root, 11, 13]);
        root
    };
    let one = b.result(Op::Constant, &[float, 1.0f32.to_bits()]);
    Ok(Evaluation {
        coordinate,
        injected: existing.is_none(),
        float,
        vector,
        boolean,
        one,
    })
}

impl Evaluation {
    pub(super) fn slot(&self, b: &mut Binary, uint: u32, constants: &BTreeMap<u32, u32>) -> u32 {
        let coordinates = b.result(Op::Load, &[self.vector, self.coordinate]);
        let x = b.result(Op::CompositeExtract, &[self.float, coordinates, 0]);
        let y = b.result(Op::CompositeExtract, &[self.float, coordinates, 1]);
        let is_x = b.result(Op::FOrdEqual, &[self.boolean, x, self.one]);
        let is_y = b.result(Op::FOrdEqual, &[self.boolean, y, self.one]);
        let yz = b.result(Op::Select, &[uint, is_y, constants[&1], constants[&2]]);
        b.result(Op::Select, &[uint, is_x, constants[&0], yz])
    }
    pub(super) fn claim(
        &self,
        b: &mut Binary,
        uint: u32,
        pointer: u32,
        constants: &BTreeMap<u32, u32>,
    ) -> u32 {
        // Device scope, relaxed equal/unequal semantics, value one, comparator
        // zero. The completed fence orders host readback after winner stores.
        let previous = b.result(
            Op::AtomicCompareExchange,
            &[
                uint,
                pointer,
                constants[&1],
                constants[&0],
                constants[&0],
                constants[&1],
                constants[&0],
            ],
        );
        let winner = b.result(Op::IEqual, &[self.boolean, previous, constants[&0]]);
        let merge = b.id();
        let capture = b.id();
        b.op(Op::SelectionMerge, &[merge, 0]);
        b.op(Op::BranchConditional, &[winner, capture, merge]);
        b.op(Op::Label, &[capture]);
        merge
    }
}

pub(super) fn partner(control: bool) -> Vec<u32> {
    // Authored fixed partner, a one-vertex patch with unit triangle levels.
    // The level-one triangle has precisely the three barycentric corners.
    let mut b = Binary::new(1);
    b.op(Op::Capability, &[1]);
    b.op(Op::Capability, &[3]);
    b.op(Op::MemoryModel, &[0, 1]);
    let function = b.id();
    let mut d = Binary::new(b.next);
    let void = d.ty(Op::TypeVoid, &[]);
    let float = d.ty(Op::TypeFloat, &[32]);
    let uint = d.ty(Op::TypeInt, &[32, 0]);
    let vec4 = d.ty(Op::TypeVector, &[float, 4]);
    let fn_ty = d.ty(Op::TypeFunction, &[void]);
    let constants = (0..=4)
        .map(|v| (v, d.result(Op::Constant, &[uint, v])))
        .collect::<BTreeMap<_, _>>();
    let zero = d.result(Op::Constant, &[float, 0]);
    let one = d.result(Op::Constant, &[float, 1.0f32.to_bits()]);
    let pos = d.result(Op::ConstantComposite, &[vec4, zero, zero, zero, one]);
    let mut decoration = Binary::new(0);
    let mut vars = Vec::new();
    let mut body = Binary::new(0);
    let mut stores = Vec::new();
    if control {
        for (count, builtin) in [(4, 11), (2, 12)] {
            let array = d.ty(Op::TypeArray, &[float, constants[&count]]);
            let pointer = d.ty(Op::TypePointer, &[3, array]);
            let scalar_pointer = d.ty(Op::TypePointer, &[3, float]);
            let var = d.result(Op::Variable, &[pointer, 3]);
            decoration.op(Op::Decorate, &[var, 11, builtin]);
            decoration.op(Op::Decorate, &[var, 15]);
            vars.push(var);
            stores.push((var, scalar_pointer, count));
        }
    } else {
        let pointer = d.ty(Op::TypePointer, &[3, vec4]);
        let var = d.result(Op::Variable, &[pointer, 3]);
        decoration.op(Op::Decorate, &[var, 11, 0]);
        vars.push(var);
    }
    b.op(
        Op::EntryPoint,
        &[if control { 1 } else { 2 }, function, 0x6e69616d, 0],
    );
    // Append interface operands to the entry declaration before execution modes.
    let count_at = b.words.len() - 5;
    b.words[count_at] += (vars.len() as u32) << 16;
    b.words.extend_from_slice(&vars);
    if control {
        b.op(Op::ExecutionMode, &[function, 26, 1]);
    } else {
        for mode in [22, 1, 5] {
            b.op(Op::ExecutionMode, &[function, mode]);
        }
    }
    b.words.extend(decoration.words);
    b.words.extend(&d.words);
    body.next = d.next;
    body.op(Op::Function, &[void, function, 0, fn_ty]);
    let label = body.id();
    body.op(Op::Label, &[label]);
    if control {
        for (var, pointer, count) in stores {
            for i in 0..count {
                let p = body.result(Op::AccessChain, &[pointer, var, constants[&i]]);
                body.op(Op::Store, &[p, one]);
            }
        }
    } else {
        body.op(Op::Store, &[vars[0], pos]);
    }
    body.op(Op::Return, &[]);
    body.op(Op::FunctionEnd, &[]);
    let mut words = vec![0x07230203, 0x00010300, 0, body.next, 0];
    words.extend(b.words);
    words.extend(body.words);
    words
}
