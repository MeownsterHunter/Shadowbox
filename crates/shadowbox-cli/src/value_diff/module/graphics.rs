// Project-owned draw observation, not Maxwell lowering. Interface shape follows
// Ryujinx CodeGen/Spirv/IoMap.cs:19,22-24,27,32-33,47 and
// Declarations.cs:452-476,554-574,587-589,628-638. No code is copied.
// SampleMask is a one-element signed-int array for this one-sample experiment:
// /usr/include/spirv/unified1/spirv.h:734. Other numeric fields and opcodes:
// same header:79-85,129-135,229-241,511-545,715-755,1783-1845,1860-1861,
// 1889-1896,2001-2006,2065,2078. Capture occurs before fixed-function conversion.
// Operand ordering: /usr/include/spirv/unified1/spirv.core.grammar.json:
// 249-285,309-454,540-562,679-729,756-769,838-859,1457-1469.
pub(super) mod geometry;
mod tessellation;

use super::{pattern, Analysis, Buffer, BufferKind, Type};
use ash::vk;
use shadowbox::spirv::Op;
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Scalar {
    Float,
    Sint,
    Uint,
}

#[derive(Clone, Default)]
pub(super) struct Io {
    builtin: Option<u32>,
    location: Option<u32>,
    component: u32,
    stream: u32,
    qualifiers: Vec<u32>,
}

impl Io {
    pub(super) fn compute_builtin(&self) -> Result<u32, String> {
        if self.location.is_some() || self.component != 0 || !self.qualifiers.is_empty() {
            return Err("unsupported compute IO decorations".into());
        }
        self.builtin
            .ok_or_else(|| "compute input lacks builtin".into())
    }

    pub fn decorate(&mut self, decoration: u32, value: u32) -> Result<(), String> {
        match decoration {
            11 => self.builtin = Some(value),
            30 => self.location = Some(value),
            29 if value < 4 => self.stream = value,
            31 if value < 4 => self.component = value,
            _ => return Err("unsupported IO decoration".into()),
        }
        Ok(())
    }

    pub fn qualify(&mut self, decoration: u32) {
        self.qualifiers.push(decoration);
    }
}

pub(in crate::value_diff) struct Attribute {
    pub location: u32,
    pub format: vk::Format,
    pub buffer: usize,
    pub stride: u32,
}

pub(in crate::value_diff) struct Graphics {
    pub words: Vec<u32>,
    pub partner: Vec<u32>,
    pub vertex: bool,
    pub stage: u32,
    pub topology: vk::PrimitiveTopology,
    pub vertices: u32,
    pub streams: bool,
    pub tess_partner: Option<Vec<u32>>,
    pub attributes: Vec<Attribute>,
}

struct Leaf {
    root: u32,
    ty: u32,
    path: Vec<u32>,
    offset: usize,
    key: String,
}

// This private binary writer emits only observation and project-owned partner
// shaders. IDs start above the translated module bound; no existing ID changes.
pub(super) struct Binary {
    pub(super) next: u32,
    pub(super) words: Vec<u32>,
    pub(super) types: HashMap<(Op, Vec<u32>), u32>,
}

impl Binary {
    pub(super) fn new(next: u32) -> Self {
        Self {
            next,
            words: Vec::new(),
            types: HashMap::new(),
        }
    }
    pub(super) fn id(&mut self) -> u32 {
        let id = self.next;
        self.next += 1;
        id
    }
    pub(super) fn op(&mut self, opcode: Op, operands: &[u32]) {
        self.words
            .push(((operands.len() as u32 + 1) << 16) | u32::from(opcode));
        self.words.extend_from_slice(operands);
    }
    pub(super) fn result(&mut self, opcode: Op, operands: &[u32]) -> u32 {
        let id = self.id();
        let mut args = vec![operands[0], id];
        args.extend_from_slice(&operands[1..]);
        self.op(opcode, &args);
        id
    }
    pub(super) fn ty(&mut self, opcode: Op, args: &[u32]) -> u32 {
        // SPIR-V forbids duplicate non-aggregate type declarations. Intern
        // generated types so repeated varyings also share their vector types.
        if let Some(id) = self.types.get(&(opcode, args.to_vec())) {
            return *id;
        }
        let id = self.id();
        let mut operands = vec![id];
        operands.extend_from_slice(args);
        self.op(opcode, &operands);
        if matches!(
            opcode,
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
        ) {
            self.types.insert((opcode, args.to_vec()), id);
        }
        id
    }
}

fn shape(a: &Analysis, ty: u32) -> Result<(u32, Scalar, usize), String> {
    let (element, count) = match a.ty(ty)? {
        Type::Scalar => (ty, 1),
        Type::Vector(element, count) => (*element, *count),
        _ => return Err("only scalar/vector user inputs supported".into()),
    };
    let scalar = *a
        .scalars
        .get(&element)
        .ok_or("IO must use 32-bit float or integer")?;
    if !(1..=4).contains(&count) {
        return Err("IO vector extent unsupported".into());
    }
    Ok((element, scalar, count))
}

fn leaves(
    a: &Analysis,
    root: u32,
    ty: u32,
    io: &Io,
    path: Vec<u32>,
    offset: usize,
    out: &mut Vec<Leaf>,
) -> Result<(), String> {
    if let Type::Struct(members) = a.ty(ty)? {
        if io.location.is_some() || io.builtin.is_some() {
            return Err("decorated IO struct unsupported".into());
        }
        let mut end = offset;
        for (i, member) in members.iter().enumerate() {
            let member_io = a.io.get(&(ty, Some(i))).ok_or("IO member lacks builtin")?;
            let start = offset + a.offsets.get(&(ty, i)).copied().unwrap_or(end - offset);
            let mut p = path.clone();
            p.push(i as u32);
            leaves(a, root, *member, member_io, p, start, out)?;
            end = start + a.size(*member)?;
        }
        return Ok(());
    }
    let mut key = match (io.location, io.builtin) {
        (Some(location), None) if location < 32 => {
            format!("location:{location}:component:{}", io.component)
        }
        (None, Some(builtin)) => format!("builtin:{builtin}"),
        _ => return Err("IO must have a supported location or builtin".into()),
    };
    if io.qualifiers.contains(&15) {
        key.insert_str(0, "patch:");
    }
    match a.ty(ty)? {
        Type::Scalar => {
            let scalar = a.scalars.get(&ty).ok_or("output scalar type unsupported")?;
            out.push(Leaf {
                root,
                ty,
                path,
                offset,
                key: format!("{key}:{scalar:?}"),
            });
        }
        Type::Vector(element, count) => {
            if io.component + *count as u32 > 4 {
                return Err("output crosses location".into());
            }
            for i in 0..*count {
                let mut part = io.clone();
                part.component += i as u32;
                let mut p = path.clone();
                p.push(i as u32);
                // Builtin vector components also need their own identity.
                let start = out.len();
                leaves(a, root, *element, &part, p, offset + i * 4, out)?;
                if io.builtin.is_some() {
                    out[start].key.push_str(&format!(":element:{i}"));
                }
            }
        }
        Type::Array(element, Some(count)) if *count <= 8 => {
            for i in 0..*count {
                let mut p = path.clone();
                p.push(i as u32);
                let start = out.len();
                leaves(
                    a,
                    root,
                    *element,
                    io,
                    p,
                    offset + i * a.strides.get(&ty).copied().unwrap_or(a.size(*element)?),
                    out,
                )?;
                for leaf in &mut out[start..] {
                    leaf.key.push_str(&format!(":index:{i}"));
                }
            }
        }
        _ => return Err("output type unsupported".into()),
    }
    Ok(())
}

fn input_words(seed: u64, location: u32, scalar: Scalar, count: usize, vertex: usize) -> Vec<u32> {
    pattern(seed, 0x41545452 ^ vertex as u64, 0, location, count * 4)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| {
            let bits = u32::from_le_bytes(*b);
            // Finite binary fractions avoid NaN and interpolation rounding.
            if scalar == Scalar::Float {
                ((bits % 2048) as f32 / 1024.0 - 1.0).to_bits()
            } else {
                bits
            }
        })
        .collect()
}

fn format(scalar: Scalar, count: usize) -> vk::Format {
    // VkFormat 32-bit component formats: vulkan_core.h:1957-1968.
    (match scalar {
        Scalar::Float => [
            vk::Format::R32_SFLOAT,
            vk::Format::R32G32_SFLOAT,
            vk::Format::R32G32B32_SFLOAT,
            vk::Format::R32G32B32A32_SFLOAT,
        ],
        Scalar::Sint => [
            vk::Format::R32_SINT,
            vk::Format::R32G32_SINT,
            vk::Format::R32G32B32_SINT,
            vk::Format::R32G32B32A32_SINT,
        ],
        Scalar::Uint => [
            vk::Format::R32_UINT,
            vk::Format::R32G32_UINT,
            vk::Format::R32G32B32_UINT,
            vk::Format::R32G32B32A32_UINT,
        ],
    })[count - 1]
}

pub(super) fn prepare(
    words: &[u32],
    stage: u32,
    _entry: u32,
    seed: u64,
    a: &Analysis,
    buffers: &mut Vec<Buffer>,
) -> Result<Graphics, String> {
    if stage == 3 {
        return geometry::prepare(words, a, buffers, seed);
    }
    if matches!(stage, 1 | 2) {
        tessellation::check(stage, a)?;
    }
    let vertex = stage == 0;
    // Three vertex invocations could race on a source storage write. The
    // analyzer currently has no per-invocation address proof for those writes.
    if a.stores != 0 {
        return Err("graphics storage writes require an invocation address proof".into());
    }
    let mut outputs = Vec::new();
    let mut inputs = Vec::new();
    let mut vertex_index = None;
    let mut attributes = Vec::new();
    for (&root, &storage) in &a.interfaces {
        let ty = a.pointers[&root].ty;
        let io = a.io.get(&(root, None)).cloned().unwrap_or_default();
        if storage == 3 {
            let mut builtins = Vec::new();
            if let Some(builtin) = io.builtin {
                builtins.push(builtin);
            }
            if let Type::Struct(members) = a.ty(ty)? {
                for i in 0..members.len() {
                    builtins.push(
                        a.io.get(&(ty, Some(i)))
                            .and_then(|io| io.builtin)
                            .ok_or("output block member lacks builtin")?,
                    );
                }
            }
            if builtins.iter().any(|b| {
                if stage == 1 {
                    !matches!(b, 0 | 1 | 11 | 12)
                } else if vertex || stage == 2 {
                    !matches!(b, 0 | 1)
                } else {
                    !matches!(b, 20 | 22)
                }
            }) {
                return Err("output builtin unsupported".into());
            }
            leaves(a, root, ty, &io, Vec::new(), 0, &mut outputs)?;
            if let Some(location) = io
                .location
                .filter(|_| vertex && a.memory[&root].data.iter().all(|b| *b))
            {
                let (_, scalar, count) = shape(a, ty)?;
                if io.component != 0 {
                    return Err("passthrough output component unsupported".into());
                }
                inputs.push((location, scalar, count, io.qualifiers.clone()));
            }
        } else if let Some(builtin) = io.builtin {
            if vertex && builtin == 42 {
                vertex_index = Some(root);
            }
            if io.location.is_some()
                || if stage == 1 {
                    true
                } else if stage == 2 {
                    !matches!(builtin, 11..=13)
                } else if vertex {
                    !matches!(builtin, 42 | 43)
                } else {
                    !matches!(builtin, 15 | 17 | 20 | 23)
                }
            {
                return Err("input builtin unsupported".into());
            }
            // spirv-val proves builtin type and execution-model constraints.
        } else {
            if matches!(stage, 1 | 2) {
                return Err("tessellation user inputs require a partner IO proof".into());
            }
            let location = io.location.ok_or("input lacks location")?;
            if location >= 16 || io.component != 0 {
                return Err("input location/component unsupported".into());
            }
            let (_, scalar, count) = shape(a, ty)?;
            if vertex {
                let initial = (0..3)
                    .flat_map(|i| input_words(seed, location, scalar, count, i))
                    .flat_map(u32::to_le_bytes)
                    .collect();
                attributes.push(Attribute {
                    location,
                    format: format(scalar, count),
                    buffer: buffers.len(),
                    stride: (count * 4) as u32,
                });
                buffers.push(Buffer {
                    set: 0,
                    binding: location,
                    kind: BufferKind::Vertex,
                    initial,
                    defined: Vec::new(),
                    observation: None,
                    expected: None,
                });
            } else {
                inputs.push((location, scalar, count, io.qualifiers.clone()));
            }
        }
    }
    outputs.sort_by(|x, y| x.key.cmp(&y.key));
    if outputs.is_empty() || outputs.len() > 16 || outputs.windows(2).any(|p| p[0].key == p[1].key)
    {
        return Err("missing, duplicate or too many captured outputs".into());
    }
    if !outputs.iter().any(|o| {
        a.memory[&o.root].data[o.offset..o.offset + 4]
            .iter()
            .all(|b| *b)
    }) {
        return Err("no defined captured stage outputs".into());
    }
    let mut declarations = Binary::new(words[3]);
    let uint = a
        .scalars
        .iter()
        .find_map(|(ty, s)| (*s == Scalar::Uint).then_some(*ty))
        .unwrap_or_else(|| declarations.ty(Op::TypeInt, &[32, 0]));
    let sint = a
        .scalars
        .iter()
        .find_map(|(ty, s)| (*s == Scalar::Sint).then_some(*ty))
        .unwrap_or_else(|| declarations.ty(Op::TypeInt, &[32, 1]));
    let pointer_type = |b: &mut Binary, storage, ty| {
        a.types
            .iter()
            .find_map(|(id, t)| {
                matches!(t, Type::Pointer(class, element) if *class == storage && *element == ty)
                    .then_some(*id)
            })
            .unwrap_or_else(|| b.ty(Op::TypePointer, &[storage, ty]))
    };
    let p_input = pointer_type(&mut declarations, 1, sint);
    let injected_index = vertex && vertex_index.is_none();
    let index_var = if injected_index {
        Some(declarations.result(Op::Variable, &[p_input, 1]))
    } else {
        vertex_index
    };
    let mut decoration = Binary::new(0);
    if injected_index {
        decoration.op(Op::Decorate, &[index_var.unwrap(), 11, 42]);
    }
    let eval = if stage == 2 {
        Some(tessellation::setup(a, &mut declarations, &mut decoration)?)
    } else {
        None
    };
    let mut constants = BTreeMap::new();
    for i in 0..=12 {
        let id = declarations.result(Op::Constant, &[uint, i]);
        constants.insert(i, id);
    }
    let invocations = if vertex || stage == 2 { 3 } else { 1 };
    let array = declarations.ty(Op::TypeArray, &[uint, constants[&invocations]]);
    decoration.op(Op::Decorate, &[array, 6, 4]);
    let block = declarations.ty(Op::TypeStruct, &[array]);
    decoration.op(Op::Decorate, &[block, 2]);
    decoration.op(Op::MemberDecorate, &[block, 0, 35, 0]);
    let p_block = declarations.ty(Op::TypePointer, &[12, block]);
    let p_storage = pointer_type(&mut declarations, 12, uint);
    let mut vars = Vec::new();
    for i in 0..=outputs.len() {
        let var = declarations.result(Op::Variable, &[p_block, 12]);
        decoration.op(Op::Decorate, &[var, 34, 3]);
        decoration.op(Op::Decorate, &[var, 33, 256 + i as u32]);
        vars.push(var);
        let defined = if let Some(output) = outputs.get(i) {
            a.memory[&output.root].data[output.offset..output.offset + 4].to_vec()
        } else {
            vec![true; 4]
        };
        let name = outputs.get(i).map_or_else(
            || "__completion".into(),
            |o| {
                format!(
                    "{}:{}",
                    match stage {
                        0 => "vertex",
                        1 => "tessellation-control",
                        2 => "tessellation-evaluation",
                        _ => "fragment",
                    },
                    o.key
                )
            },
        );
        buffers.push(Buffer {
            set: 3,
            binding: 256 + i as u32,
            kind: BufferKind::Storage,
            initial: vec![0; invocations as usize * 4],
            defined: defined.repeat(invocations as usize),
            observation: Some(name),
            expected: None,
        });
    }
    let mut output_pointers = BTreeMap::new();
    for output in &outputs {
        output_pointers
            .entry(output.ty)
            .or_insert_with(|| pointer_type(&mut declarations, 3, output.ty));
    }
    let mut tail = Binary::new(declarations.next);
    let slot = if let Some(eval) = &eval {
        eval.slot(&mut tail, uint, &constants)
    } else if vertex {
        // AccessChain uses the native signed VertexIndex directly. This saves
        // a cast without changing the slots. IoMap.cs:47 gives its signed type;
        // check-value-diff.py proves all three slots on the validated module.
        tail.result(Op::Load, &[sint, index_var.unwrap()])
    } else {
        constants[&0]
    };
    let complete = tail.result(
        Op::AccessChain,
        &[p_storage, *vars.last().unwrap(), constants[&0], slot],
    );
    let merge = eval
        .as_ref()
        .map(|e| e.claim(&mut tail, uint, complete, &constants));
    for (i, output) in outputs.iter().enumerate() {
        let mask = &a.memory[&output.root].data[output.offset..output.offset + 4];
        if mask.iter().all(|b| !*b) {
            continue;
        }
        if !mask.iter().all(|b| *b) {
            return Err("partially undefined output scalar unsupported".into());
        }
        let p_output = output_pointers[&output.ty];
        let source = if output.path.is_empty() {
            output.root
        } else {
            let mut operands = vec![p_output, output.root];
            operands.extend(output.path.iter().map(|i| constants[i]));
            tail.result(Op::AccessChain, &operands)
        };
        let loaded = tail.result(Op::Load, &[output.ty, source]);
        // An unsigned output already has the observation type. Store its bits
        // directly instead of emitting an identity cast.
        let bits = if output.ty == uint {
            loaded
        } else {
            tail.result(Op::Bitcast, &[uint, loaded])
        };
        let target = tail.result(Op::AccessChain, &[p_storage, vars[i], constants[&0], slot]);
        tail.op(Op::Store, &[target, bits]);
    }
    if let Some(merge) = merge {
        tail.op(Op::Branch, &[merge]);
        tail.op(Op::Label, &[merge]);
    } else {
        tail.op(Op::Store, &[complete, constants[&1]]);
    }
    let mut instrumented = words[..5].to_vec();
    instrumented[3] = tail.next;
    let mut at = 5;
    let mut decorated = false;
    let mut declared = false;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        let opcode = Op::of(words[at]);
        if !decorated
            && matches!(
                opcode,
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
            instrumented.extend_from_slice(&decoration.words);
            decorated = true;
        }
        if opcode == Op::Function && !declared {
            instrumented.extend_from_slice(&declarations.words);
            declared = true;
        }
        if opcode == Op::Return {
            instrumented.extend_from_slice(&tail.words);
        }
        if opcode == Op::EntryPoint {
            let mut operands = words[at + 1..at + count].to_vec();
            if let Some(eval) = &eval {
                if eval.injected {
                    operands.push(eval.coordinate);
                }
            }
            if injected_index {
                operands.push(index_var.unwrap());
            }
            if words[1] >= 0x00010400 {
                operands.extend_from_slice(&vars);
            }
            instrumented.push(((operands.len() as u32 + 1) << 16) | 15);
            instrumented.extend_from_slice(&operands);
        } else {
            instrumented.extend_from_slice(&words[at..at + count]);
        }
        at += count;
    }
    Ok(Graphics {
        words: instrumented,
        partner: partner(vertex, &inputs, seed),
        tess_partner: if matches!(stage, 1 | 2) {
            Some(tessellation::partner(stage == 2))
        } else {
            None
        },
        vertex,
        stage,
        topology: if matches!(stage, 1 | 2) {
            vk::PrimitiveTopology::PATCH_LIST
        } else {
            vk::PrimitiveTopology::TRIANGLE_LIST
        },
        vertices: if matches!(stage, 1 | 2) { 1 } else { 3 },
        streams: false,
        attributes,
    })
}

fn partner(fragment: bool, inputs: &[(u32, Scalar, usize, Vec<u32>)], seed: u64) -> Vec<u32> {
    let mut b = Binary::new(1);
    let void = b.ty(Op::TypeVoid, &[]);
    let function = b.ty(Op::TypeFunction, &[void]);
    let float = b.ty(Op::TypeFloat, &[32]);
    let uint = b.ty(Op::TypeInt, &[32, 0]);
    let sint = b.ty(Op::TypeInt, &[32, 1]);
    let main = b.id();
    let mut decorations = Binary::new(0);
    let mut interfaces = Vec::new();
    let mut stores = Vec::new();
    let mut passthrough = Vec::new();
    let mut position = None;
    if !fragment {
        let vec4 = b.ty(Op::TypeVector, &[float, 4]);
        let p_output = b.ty(Op::TypePointer, &[3, vec4]);
        let output = b.result(Op::Variable, &[p_output, 3]);
        decorations.op(Op::Decorate, &[output, 11, 0]);
        interfaces.push(output);
        let p_index = b.ty(Op::TypePointer, &[1, sint]);
        let v_index = b.result(Op::Variable, &[p_index, 1]);
        decorations.op(Op::Decorate, &[v_index, 11, 42]);
        interfaces.push(v_index);
        // Project experiment: fullscreen triangle over a single pixel. Every
        // partner varying is constant at all vertices, so interpolation cannot
        // introduce input variation. No user shader position is overwritten.
        let three = b.result(Op::Constant, &[uint, 3]);
        let array = b.ty(Op::TypeArray, &[vec4, three]);
        let mut coords = Vec::new();
        for (x, y) in [(-1.0f32, -1.0f32), (3.0, -1.0), (-1.0, 3.0)] {
            let components =
                [x, y, 0.5, 1.0].map(|v| b.result(Op::Constant, &[float, v.to_bits()]));
            let mut args = vec![vec4];
            args.extend(components);
            coords.push(b.result(Op::ConstantComposite, &args));
        }
        let mut args = vec![array];
        args.extend(coords);
        let initial = b.result(Op::ConstantComposite, &args);
        let p_array = b.ty(Op::TypePointer, &[6, array]);
        let positions = b.result(Op::Variable, &[p_array, 6, initial]);
        interfaces.push(positions); // SPIR-V 1.5 lists all used globals.
        let p_vec4 = b.ty(Op::TypePointer, &[6, vec4]);
        position = Some((output, v_index, positions, p_vec4, vec4));
    }
    for (location, scalar, count, qualifiers) in inputs {
        let element = match scalar {
            Scalar::Float => float,
            Scalar::Uint => uint,
            Scalar::Sint => sint,
        };
        let ty = if *count == 1 {
            element
        } else {
            b.ty(Op::TypeVector, &[element, *count as u32])
        };
        let p = b.ty(Op::TypePointer, &[3, ty]);
        let var = b.result(Op::Variable, &[p, 3]);
        decorations.op(Op::Decorate, &[var, 30, *location]);
        interfaces.push(var);
        if fragment {
            let p_input = b.ty(Op::TypePointer, &[1, ty]);
            let input = b.result(Op::Variable, &[p_input, 1]);
            decorations.op(Op::Decorate, &[input, 30, *location]);
            for q in qualifiers
                .iter()
                .filter(|q| matches!(**q, 13 | 14 | 16 | 17))
            {
                decorations.op(Op::Decorate, &[input, *q]);
            }
            if *scalar != Scalar::Float && !qualifiers.contains(&14) {
                decorations.op(Op::Decorate, &[input, 14]);
            }
            interfaces.push(input);
            passthrough.push((var, input, ty));
        } else {
            for q in qualifiers
                .iter()
                .filter(|q| matches!(**q, 13 | 14 | 16 | 17))
            {
                decorations.op(Op::Decorate, &[var, *q]);
            }
            let values = input_words(seed, *location, *scalar, *count, 0)
                .into_iter()
                .map(|v| b.result(Op::Constant, &[element, v]))
                .collect::<Vec<_>>();
            let value = if *count == 1 {
                values[0]
            } else {
                let mut args = vec![ty];
                args.extend(values);
                b.result(Op::ConstantComposite, &args)
            };
            stores.push((var, value));
        }
    }
    let mut prefix = Binary::new(0);
    prefix.op(Op::Capability, &[1]);
    prefix.op(Op::MemoryModel, &[0, 1]);
    let mut entry = vec![if fragment { 4 } else { 0 }, main, 0x6e69616d, 0];
    entry.extend(interfaces);
    prefix.op(Op::EntryPoint, &entry);
    if fragment {
        prefix.op(Op::ExecutionMode, &[main, 7]);
    }
    prefix.words.extend(decorations.words);
    b.op(Op::Function, &[void, main, 0, function]);
    b.ty(Op::Label, &[]);
    if let Some((output, index, positions, pointer, vec4)) = position {
        let loaded = b.result(Op::Load, &[sint, index]);
        let ptr = b.result(Op::AccessChain, &[pointer, positions, loaded]);
        let value = b.result(Op::Load, &[vec4, ptr]);
        stores.push((output, value));
    }
    for (output, input, ty) in passthrough {
        let value = b.result(Op::Load, &[ty, input]);
        stores.push((output, value));
    }
    for (var, value) in stores {
        b.op(Op::Store, &[var, value]);
    }
    b.op(Op::Return, &[]);
    b.op(Op::FunctionEnd, &[]);
    let mut words = vec![0x07230203, 0x00010500, 0, b.next, 0];
    words.extend(prefix.words);
    words.extend(b.words);
    words
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stage(words: &[u32]) -> u32 {
        let mut at = 5;
        while at < words.len() {
            if Op::of(words[at]) == Op::EntryPoint {
                return words[at + 1];
            }
            at += (words[at] >> 16) as usize;
        }
        panic!("partner has no entry point")
    }

    #[test]
    fn partners_use_the_pipeline_slot_execution_model() {
        assert_eq!(stage(&partner(true, &[], 0)), 4);
        assert_eq!(stage(&partner(false, &[], 0)), 0);
    }
}
