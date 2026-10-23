// SPIR-V numeric fields: /usr/include/spirv/unified1/spirv.h:47,79,101,111,
// 135,229-241,1085 (stage, memory model, local size, storage, capability),
// 511-545 (Block, BufferBlock, ArrayStride, Binding, DescriptorSet, Offset),
// 1783-1861,1876-1903,1926-1966,2001-2016,2065,2078 (opcodes).
// Additional stage modes, Patch/Stream, geometry and subgroup capabilities and
// opcodes: spirv.h:79-85,119-147,525,539,1086-1087,1136,1143-1148,
// 1976-1979,2082-2096. Binary word counts follow project emitter.rs:50-53.
// Buffer declarations follow Ryujinx CodeGen/Spirv/Declarations.cs:82-144.
// Uninitialized local memory has no initializer there (:54-69). SSA missing
// definitions are undefined in Translation/Ssa.cs:250-282. Analyze byte
// dependencies instead of comparing arbitrary GPU values for those bytes.
mod compute;
pub(super) mod graphics;
pub(super) mod images;
mod physical;
mod storage;

use shadowbox::spirv::Op;
use std::collections::BTreeMap;

pub(super) const BUFFER_SIZE: usize = 65536;
const MAX_BUFFERS: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum BufferKind {
    Uniform,
    Storage,
    Vertex,
    Physical,
}

pub(super) struct Buffer {
    pub set: u32,
    pub binding: u32,
    pub kind: BufferKind,
    pub initial: Vec<u8>,
    pub defined: Vec<bool>,
    pub observation: Option<String>,
    pub expected: Option<Vec<u8>>,
}

pub(super) struct Plan {
    pub stage: u32,
    pub local_size: [u32; 3],
    pub shared_size: usize,
    pub subgroup: bool,
    pub maximal_reconvergence: bool,
    pub global_capture: Option<storage::GlobalCapture>,
    pub compute_words: Option<Vec<u32>>,
    pub buffers: Vec<Buffer>,
    pub push: Vec<u8>,
    pub images: Vec<images::Resource>,
    pub graphics: Option<graphics::Graphics>,
}

#[derive(Clone)]
enum Type {
    Void,
    Scalar,
    Uint64,
    Vector(u32, usize),
    Array(u32, Option<usize>),
    Struct(Vec<u32>),
    Pointer(u32, u32),
    Function,
    Image(images::ImageType),
    Sampler,
    SampledImage(u32),
}

#[derive(Clone, Copy, Debug)]
struct Pointer {
    root: u32,
    ty: u32,
    offset: usize,
}

#[derive(Clone)]
struct Memory {
    data: Vec<bool>,
    writable: bool,
    written: Vec<bool>,
}

#[derive(Clone, Default)]
struct Analysis {
    types: BTreeMap<u32, Type>,
    constants: BTreeMap<u32, u32>,
    literals: BTreeMap<u32, Vec<u32>>,
    memory_literals: BTreeMap<(u32, usize), u32>,
    resources: BTreeMap<(u32, u32), (u32, images::ResourceKind)>,
    handles: BTreeMap<u32, images::Handle>,
    values: BTreeMap<u32, Vec<bool>>,
    value_types: BTreeMap<u32, u32>,
    pointers: BTreeMap<u32, Pointer>,
    memory: BTreeMap<u32, Memory>,
    storage_classes: BTreeMap<u32, u32>,
    strides: BTreeMap<u32, usize>,
    offsets: BTreeMap<(u32, usize), usize>,
    bindings: BTreeMap<u32, (Option<u32>, Option<u32>)>,
    blocks: BTreeMap<u32, bool>,
    roots: BTreeMap<(u32, u32), (u32, BufferKind)>,
    push: usize,
    cb1_len: Option<usize>,
    host_initial: BTreeMap<u32, Vec<u8>>,
    stores: usize,
    concrete: bool,
    subgroup: bool,
    subgroup_masks: Vec<Vec<u32>>,
    subgroup_traces: Vec<Vec<Vec<u32>>>,
    geometry_events: Vec<graphics::geometry::Event>,
    graphics_modes: BTreeMap<u32, u32>,
    geometry_streams: bool,
    physical: bool,
    maximal_reconvergence: bool,
    wide: BTreeMap<u32, physical::Wide>,
    io: BTreeMap<(u32, Option<usize>), graphics::Io>,
    interfaces: BTreeMap<u32, u32>,
    scalars: BTreeMap<u32, graphics::Scalar>,
}

/// Counter-based mixing. Input depends on seed and interface coordinates,
// not variable IDs, file names, build order, jobs, or descriptor enumeration.
// This project-owned experiment specifies the input pattern; no shader
// instruction encoding or mapping is inferred from the PRNG.
fn pattern(seed: u64, domain: u64, set: u32, binding: u32, len: usize) -> Vec<u8> {
    (0..len)
        .map(|offset| {
            let mut x =
                seed ^ domain ^ (u64::from(set) << 48) ^ (u64::from(binding) << 32) ^ offset as u64;
            x = x.wrapping_add(0x9e3779b97f4a7c15);
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            (x ^ (x >> 31)) as u8
        })
        .collect()
}

impl Analysis {
    fn ty(&self, id: u32) -> Result<&Type, String> {
        self.types
            .get(&id)
            .ok_or_else(|| format!("unknown type %{id}"))
    }

    fn size(&self, id: u32) -> Result<usize, String> {
        self.size_inner(id, 0)
    }

    fn size_inner(&self, id: u32, depth: usize) -> Result<usize, String> {
        if depth > 16 {
            return Err("type nesting exceeds 16".into());
        }
        let size = match self.ty(id)? {
            Type::Scalar => Some(4),
            Type::Uint64 => Some(8),
            Type::Vector(element, count) => {
                self.size_inner(*element, depth + 1)?.checked_mul(*count)
            }
            Type::Array(element, count) => {
                let stride = self
                    .strides
                    .get(&id)
                    .copied()
                    .unwrap_or(self.size_inner(*element, depth + 1)?);
                count.map_or(Some(BUFFER_SIZE), |count| stride.checked_mul(count))
            }
            Type::Struct(members) => {
                let mut end = 0usize;
                for (index, member) in members.iter().enumerate() {
                    let offset = self.offsets.get(&(id, index)).copied().unwrap_or(end);
                    if matches!(self.ty(*member)?, Type::Array(_, None)) {
                        return Ok(BUFFER_SIZE);
                    }
                    end = offset
                        .checked_add(self.size_inner(*member, depth + 1)?)
                        .ok_or("type size overflow")?;
                }
                Some(end)
            }
            _ => return Err("unsupported value type".into()),
        };
        let size = size.ok_or("type size overflow")?;
        if size > BUFFER_SIZE {
            return Err("type exceeds 64 KiB".into());
        }
        Ok(size)
    }

    fn value(&self, id: u32) -> Result<&[bool], String> {
        self.values
            .get(&id)
            .map(Vec::as_slice)
            .ok_or_else(|| format!("unknown value %{id}"))
    }

    fn result(&mut self, ty: u32, id: u32, bytes: Vec<bool>) -> Result<(), String> {
        if !matches!(
            self.ty(ty)?,
            Type::Scalar | Type::Uint64 | Type::Vector(_, _)
        ) {
            return Err("aggregate value unsupported".into());
        }
        if bytes.len() != self.size(ty)? {
            return Err("value extent mismatch".into());
        }
        self.value_types.insert(id, ty);
        self.values.insert(id, bytes);
        Ok(())
    }

    fn pointer(&self, id: u32) -> Result<Pointer, String> {
        self.pointers
            .get(&id)
            .copied()
            .ok_or_else(|| format!("unsupported pointer %{id}"))
    }

    fn range(&self, p: Pointer) -> Result<std::ops::Range<usize>, String> {
        let len = self.size(p.ty)?;
        let end = p.offset.checked_add(len).ok_or("pointer overflow")?;
        if end
            > self
                .memory
                .get(&p.root)
                .ok_or("missing allocation")?
                .data
                .len()
        {
            return Err("out-of-bounds memory access".into());
        }
        Ok(p.offset..end)
    }

    fn chain(&self, base: u32, indices: &[u32]) -> Result<Pointer, String> {
        let mut p = self.pointer(base)?;
        for &index in indices {
            let index = *self
                .constants
                .get(&index)
                .ok_or("dynamic access-chain index is unsupported")?
                as usize;
            let (ty, offset) = match self.ty(p.ty)? {
                Type::Struct(members) => {
                    let ty = *members.get(index).ok_or("struct member out of bounds")?;
                    let mut offset = 0;
                    for (i, &member) in members.iter().enumerate().take(index + 1) {
                        offset = self.offsets.get(&(p.ty, i)).copied().unwrap_or(offset);
                        if i != index {
                            offset += self.size(member)?;
                        }
                    }
                    (ty, offset)
                }
                Type::Array(ty, count) => {
                    if count.is_some_and(|count| index >= count) {
                        return Err("array index out of bounds".into());
                    }
                    let stride = self.strides.get(&p.ty).copied().unwrap_or(self.size(*ty)?);
                    (
                        *ty,
                        index.checked_mul(stride).ok_or("array offset overflow")?,
                    )
                }
                Type::Vector(ty, count) => {
                    if index >= *count {
                        return Err("vector index out of bounds".into());
                    }
                    (*ty, index * self.size(*ty)?)
                }
                _ => return Err("unsupported access-chain type".into()),
            };
            p.offset = p
                .offset
                .checked_add(offset)
                .ok_or("access-chain overflow")?;
            p.ty = ty;
        }
        self.range(p)?;
        Ok(p)
    }

    fn variable(
        &mut self,
        ty: u32,
        id: u32,
        storage: u32,
        init: Option<u32>,
    ) -> Result<(), String> {
        let Type::Pointer(class, value_ty) = *self.ty(ty)? else {
            return Err("variable requires a pointer type".into());
        };
        if class != storage {
            return Err("variable storage mismatch".into());
        }
        if storage == 0 {
            return images::variable(self, id, value_ty, init);
        }
        let size = self.size(value_ty)?;
        let (len, writable, initialized) = match storage {
            2 | 12 => {
                if !matches!(self.ty(value_ty)?, Type::Struct(_)) {
                    return Err("buffer must point to a block struct".into());
                }
                let buffer_block = *self
                    .blocks
                    .get(&value_ty)
                    .ok_or("buffer lacks block decoration")?;
                let kind = if storage == 12 || buffer_block {
                    BufferKind::Storage
                } else {
                    BufferKind::Uniform
                };
                let (set, binding) = self
                    .bindings
                    .get(&id)
                    .ok_or("descriptor lacks set and binding")?;
                let key = (
                    set.ok_or("descriptor lacks set")?,
                    binding.ok_or("descriptor lacks binding")?,
                );
                if key.0 > 3 || key.1 > 255 || self.roots.len() == MAX_BUFFERS {
                    return Err("descriptor exceeds harness limits".into());
                }
                if self.resources.contains_key(&key) || self.roots.insert(key, (id, kind)).is_some()
                {
                    return Err("duplicate descriptor binding".into());
                }
                // Fixed buffers with initialized padding avoid uninitialized
                // host memory. Addresses still must pass range().
                (BUFFER_SIZE, kind == BufferKind::Storage, true)
            }
            9 => {
                if size > 128 || self.push != 0 {
                    return Err("unsupported push constant extent".into());
                }
                self.push = size;
                (size, false, true)
            }
            1 | 3 => {
                self.interfaces.insert(id, storage);
                (size, storage == 3, storage == 1)
            }
            4 | 6 | 7 => (size, true, false),
            _ => return Err(format!("unsupported interface storage class {storage}")),
        };
        if self.memory.values().map(|m| m.data.len()).sum::<usize>() + len > 2 * 1024 * 1024 {
            return Err("memory analysis exceeds 2 MiB".into());
        }
        let mut data = vec![initialized; len];
        // Shadowbox's cbuf interface is set0/binding=slot (emitter.rs:1838-1840).
        // Only saved bytes are defined: Ryujinx DiskCacheGpuAccessor.cs:62-69.
        if self.roots.get(&(0, 1)) == Some(&(id, BufferKind::Uniform)) {
            if let Some(size) = self.cb1_len {
                data[size..].fill(false);
            }
        }
        if let Some(init) = init {
            let value = self.value(init)?;
            if value.len() != size {
                return Err("variable initializer extent mismatch".into());
            }
            data[..size].copy_from_slice(value);
        }
        self.storage_classes.insert(id, storage);
        if let Some(bits) = init.and_then(|id| self.literals.get(&id)) {
            for (i, &bits) in bits.iter().enumerate() {
                self.memory_literals.insert((id, i * 4), bits);
            }
        }
        self.memory.insert(
            id,
            Memory {
                written: vec![false; data.len()],
                data,
                writable,
            },
        );
        self.pointers.insert(
            id,
            Pointer {
                root: id,
                ty: value_ty,
                offset: 0,
            },
        );
        Ok(())
    }

    fn set_literal(&mut self, id: u32, bits: Vec<u32>) {
        if bits.len() == 1 {
            self.constants.insert(id, bits[0]);
        }
        self.literals.insert(id, bits);
    }

    fn instruction(&mut self, op: Op, a: &[u32]) -> Result<(), String> {
        // A loop re-executes result IDs. Erase stale literals before producing
        // a new result so an unknown load cannot reuse a previous iteration.
        if matches!(
            op,
            Op::Undef
                | Op::ConstantTrue
                | Op::ConstantFalse
                | Op::Constant
                | Op::ConstantComposite
                | Op::ConstantSampler
                | Op::ConstantNull
                | Op::Load
                | Op::VectorExtractDynamic
                | Op::CompositeConstruct
                | Op::CompositeExtract
                | Op::CopyObject
                | Op::SampledImage
                | Op::ImageSampleImplicitLod
                | Op::ImageSampleExplicitLod
                | Op::ImageSampleDrefImplicitLod
                | Op::ImageSampleDrefExplicitLod
                | Op::ConvertSToF
                | Op::ConvertUToF
                | Op::Bitcast
                | Op::SNegate
                | Op::FNegate
                | Op::IAdd
                | Op::FAdd
                | Op::ISub
                | Op::FSub
                | Op::IMul
                | Op::FMul
                | Op::LogicalEqual
                | Op::LogicalNotEqual
                | Op::LogicalOr
                | Op::LogicalAnd
                | Op::LogicalNot
                | Op::Select
                | Op::IEqual
                | Op::INotEqual
                | Op::UGreaterThan
                | Op::SGreaterThan
                | Op::UGreaterThanEqual
                | Op::SGreaterThanEqual
                | Op::ULessThan
                | Op::SLessThan
                | Op::ULessThanEqual
                | Op::SLessThanEqual
                | Op::FOrdEqual
                | Op::FUnordEqual
                | Op::FOrdNotEqual
                | Op::FUnordNotEqual
                | Op::FOrdLessThan
                | Op::FUnordLessThan
                | Op::FOrdGreaterThan
                | Op::FUnordGreaterThan
                | Op::FOrdLessThanEqual
                | Op::FUnordLessThanEqual
                | Op::FOrdGreaterThanEqual
                | Op::FUnordGreaterThanEqual
                | Op::ShiftRightLogical
                | Op::ShiftRightArithmetic
                | Op::ShiftLeftLogical
                | Op::BitwiseOr
                | Op::BitwiseXor
                | Op::BitwiseAnd
                | Op::Not
                | Op::BitReverse
                | Op::BitCount
        ) {
            self.constants.remove(&a[1]);
            self.literals.remove(&a[1]);
        }
        match op {
            Op::EmitVertex | Op::EndPrimitive | Op::EmitStreamVertex | Op::EndStreamPrimitive => {
                graphics::geometry::record(self, op, a)?
            }
            Op::TypeInt if a[1..] == [64, 0] && self.physical => {
                self.types.insert(a[0], Type::Uint64);
            }
            Op::Constant | Op::ConstantNull if matches!(self.ty(a[0])?, Type::Uint64) => {
                physical::constant(self, op, a)?
            }
            Op::Load if matches!(self.ty(a[0])?, Type::Uint64) => physical::load_delta(self, a)?,
            Op::UConvert | Op::ConvertUToPtr => physical::instruction(self, op, a)?,
            Op::Bitcast | Op::IAdd | Op::CopyObject if matches!(self.ty(a[0])?, Type::Uint64) => {
                physical::instruction(self, op, a)?
            }
            Op::TypeVoid => {
                self.types.insert(a[0], Type::Void);
            }
            Op::TypeBool => {
                self.types.insert(a[0], Type::Scalar);
            }
            Op::TypeInt if a[1] == 32 => {
                self.types.insert(a[0], Type::Scalar);
                self.scalars.insert(
                    a[0],
                    if a[2] == 0 {
                        graphics::Scalar::Uint
                    } else {
                        graphics::Scalar::Sint
                    },
                );
            }
            Op::TypeFloat if a[1] == 32 => {
                self.types.insert(a[0], Type::Scalar);
                self.scalars.insert(a[0], graphics::Scalar::Float);
            }
            Op::TypeVector => {
                self.types.insert(a[0], Type::Vector(a[1], a[2] as usize));
            }
            Op::TypeImage | Op::TypeSampler | Op::TypeSampledImage => images::declare(self, op, a)?,
            Op::SampledImage
            | Op::ImageSampleImplicitLod
            | Op::ImageSampleExplicitLod
            | Op::ImageSampleDrefImplicitLod
            | Op::ImageSampleDrefExplicitLod => images::instruction(self, op, a)?,
            Op::TypeArray => {
                let count = *self
                    .constants
                    .get(&a[2])
                    .ok_or("array length must be constant")? as usize;
                self.types.insert(a[0], Type::Array(a[1], Some(count)));
            }
            Op::TypeRuntimeArray => {
                self.types.insert(a[0], Type::Array(a[1], None));
            }
            Op::TypeStruct => {
                self.types.insert(a[0], Type::Struct(a[1..].to_vec()));
            }
            Op::TypePointer => {
                self.types.insert(a[0], Type::Pointer(a[1], a[2]));
            }
            Op::TypeFunction => {
                self.types.insert(a[0], Type::Function);
            }
            Op::ConstantTrue | Op::ConstantFalse | Op::Constant | Op::ConstantNull => {
                let len = self.size(a[0])?;
                if len > 16 {
                    return Err("aggregate constants unsupported".into());
                }
                self.result(a[0], a[1], vec![true; len])?;
                self.literals.insert(
                    a[1],
                    if op == Op::Constant {
                        vec![a[2]]
                    } else {
                        vec![u32::from(op == Op::ConstantTrue); len / 4]
                    },
                );
                if len == 4 {
                    self.constants.insert(
                        a[1],
                        match op {
                            Op::ConstantTrue => 1,
                            Op::Constant => a[2],
                            _ => 0,
                        },
                    );
                }
            }
            Op::Undef => {
                if self.size(a[0])? > 16 {
                    return Err("aggregate undef unsupported".into());
                }
                self.result(a[0], a[1], vec![false; self.size(a[0])?])?;
            }
            Op::ConstantComposite | Op::CompositeConstruct => {
                let Type::Vector(element, _) = self.ty(a[0])? else {
                    return Err("only vector construction supported".into());
                };
                if a[2..]
                    .iter()
                    .any(|source| self.value_types.get(source) != Some(element))
                {
                    return Err("vector element type mismatch".into());
                }
                let mut bytes = Vec::new();
                for &source in &a[2..] {
                    bytes.extend_from_slice(self.value(source)?);
                }
                self.result(a[0], a[1], bytes)?;
                if let Some(parts) = a[2..]
                    .iter()
                    .map(|source| self.literals.get(source).cloned())
                    .collect::<Option<Vec<_>>>()
                {
                    self.literals.insert(a[1], parts.concat());
                }
            }
            Op::Variable => self.variable(a[0], a[1], a[2], a.get(3).copied())?,
            Op::Load if self.handles.contains_key(&a[2]) => images::load(self, a)?,
            Op::Load => {
                physical::alignment(self, a[2], &a[3..])?;
                let p = self.pointer(a[2])?;
                if p.ty != a[0] {
                    return Err("load type mismatch".into());
                }
                if !matches!(self.ty(p.ty)?, Type::Scalar | Type::Vector(_, _)) {
                    return Err("aggregate loads unsupported".into());
                }
                let range = self.range(p)?;
                if let Some(bits) = range
                    .clone()
                    .step_by(4)
                    .map(|offset| self.memory_literals.get(&(p.root, offset)).copied())
                    .collect::<Option<Vec<_>>>()
                {
                    if bits.len() == 1 {
                        self.constants.insert(a[1], bits[0]);
                    }
                    self.literals.insert(a[1], bits);
                }
                self.result(a[0], a[1], self.memory[&p.root].data[range].to_vec())?;
            }
            Op::Store => {
                physical::alignment(self, a[0], &a[2..])?;
                let p = self.pointer(a[0])?;
                if !matches!(self.ty(p.ty)?, Type::Scalar | Type::Vector(_, _)) {
                    return Err("aggregate stores unsupported".into());
                }
                let range = self.range(p)?;
                let value = self.value(a[1])?.to_vec();
                if range.len() != value.len() {
                    return Err("store extent mismatch".into());
                }
                for (i, offset) in range.clone().step_by(4).enumerate() {
                    self.memory_literals.remove(&(p.root, offset));
                    if let Some(&bits) = self.literals.get(&a[1]).and_then(|bits| bits.get(i)) {
                        self.memory_literals.insert((p.root, offset), bits);
                    }
                }
                let memory = self.memory.get_mut(&p.root).unwrap();
                if !memory.writable {
                    return Err("write to read-only input".into());
                }
                memory.written[range.clone()].fill(true);
                memory.data[range].copy_from_slice(&value);
                if self.roots.values().any(|(root, kind)| {
                    *root == p.root && matches!(kind, BufferKind::Storage | BufferKind::Physical)
                }) {
                    self.stores += 1;
                }
            }
            Op::AccessChain | Op::InBoundsAccessChain => {
                let p = self.chain(a[2], &a[3..])?;
                let Type::Pointer(_, pointee) = self.ty(a[0])? else {
                    return Err("chain result must be pointer".into());
                };
                if *pointee != p.ty {
                    return Err("chain pointee mismatch".into());
                }
                self.pointers.insert(a[1], p);
            }
            Op::VectorExtractDynamic if self.concrete => {
                let index = *self.constants.get(&a[3]).ok_or("unknown vector index")?;
                self.instruction(Op::CompositeExtract, &[a[0], a[1], a[2], index])?;
            }
            Op::CompositeExtract => {
                // CompositeExtract literals are indices, not ID operands.
                // Value types are tracked by result extent for flat vectors.
                // Nested composites are refused rather than guessed.
                let Type::Vector(element, _) =
                    self.ty(*self.value_types.get(&a[2]).ok_or("missing value type")?)?
                else {
                    return Err("only vector extraction supported".into());
                };
                if *element != a[0] {
                    return Err("extract element type mismatch".into());
                }
                let value = self.value(a[2])?;
                if a.len() != 4 || value.len() > 16 || self.size(a[0])? != 4 {
                    return Err("only scalar extraction from flat vectors is supported".into());
                }
                let offset = a[3] as usize * 4;
                let bytes = value
                    .get(offset..offset + 4)
                    .ok_or("extract out of bounds")?
                    .to_vec();
                self.result(a[0], a[1], bytes)?;
                if let Some(&bits) = self
                    .literals
                    .get(&a[2])
                    .and_then(|bits| bits.get(a[3] as usize))
                {
                    self.set_literal(a[1], vec![bits]);
                }
            }
            Op::CopyObject if self.handles.contains_key(&a[2]) => images::load(self, a)?,
            Op::CopyObject | Op::Bitcast => {
                if self.wide.contains_key(&a[2]) {
                    return Err("physical address used as ordinary data".into());
                }
                if let Some(bits) = self.literals.get(&a[2]).cloned() {
                    self.literals.insert(a[1], bits);
                }
                let bytes = self.value(a[2])?.to_vec();
                self.result(a[0], a[1], bytes)?;
                if let Some(&value) = self.constants.get(&a[2]) {
                    self.constants.insert(a[1], value);
                }
            }
            // Total scalar/componentwise operations. Integer division,
            // float-to-int, shifts with unknown counts, extended math and
            // cross-lane operations stay uncheckable.
            Op::ConvertSToF
            | Op::ConvertUToF
            | Op::SNegate
            | Op::FNegate
            | Op::IAdd
            | Op::FAdd
            | Op::ISub
            | Op::FSub
            | Op::IMul
            | Op::FMul
            | Op::LogicalEqual
            | Op::LogicalNotEqual
            | Op::LogicalOr
            | Op::LogicalAnd
            | Op::LogicalNot
            | Op::IEqual
            | Op::INotEqual
            | Op::UGreaterThan
            | Op::SGreaterThan
            | Op::UGreaterThanEqual
            | Op::SGreaterThanEqual
            | Op::ULessThan
            | Op::SLessThan
            | Op::ULessThanEqual
            | Op::SLessThanEqual
            | Op::FOrdEqual
            | Op::FUnordEqual
            | Op::FOrdNotEqual
            | Op::FUnordNotEqual
            | Op::FOrdLessThan
            | Op::FUnordLessThan
            | Op::FOrdGreaterThan
            | Op::FUnordGreaterThan
            | Op::FOrdLessThanEqual
            | Op::FUnordLessThanEqual
            | Op::FOrdGreaterThanEqual
            | Op::FUnordGreaterThanEqual
            | Op::ShiftRightLogical
            | Op::ShiftRightArithmetic
            | Op::ShiftLeftLogical
            | Op::BitwiseOr
            | Op::BitwiseXor
            | Op::BitwiseAnd
            | Op::Not
            | Op::BitReverse
            | Op::BitCount => {
                let len = self.size(a[0])?;
                if len > 16 {
                    return Err("arithmetic aggregate unsupported".into());
                }
                let inputs = a[2..]
                    .iter()
                    .map(|&source| self.value(source))
                    .collect::<Result<Vec<_>, _>>()?;
                if inputs.iter().any(|input| input.len() != len) {
                    return Err("component count mismatch".into());
                }
                let mut bytes = vec![true; len];
                for (component, out) in bytes.chunks_mut(4).enumerate() {
                    let mut undefined = 0;
                    for input in &inputs {
                        let part = &input[component * 4..component * 4 + 4];
                        if part.iter().all(|b| !*b) {
                            undefined += 1;
                        } else if !part.iter().all(|b| *b) {
                            return Err(
                                "partially undefined arithmetic component unsupported".into()
                            );
                        }
                    }
                    if undefined != 0 {
                        // Project-owned dependency proof: negation, NOT, and
                        // adding, subtracting, or XORing a defined operand
                        // are bijections of the undefined input's domain.
                        // Other operations can collapse undefined inputs
                        // to a defined result, such as x * 0 or x & 0.
                        // Multiple tainted operands can cancel, as x - x.
                        if undefined != 1
                            || !matches!(
                                op,
                                Op::SNegate
                                    | Op::FNegate
                                    | Op::IAdd
                                    | Op::ISub
                                    | Op::LogicalNot
                                    | Op::BitwiseXor
                                    | Op::Not
                            )
                        {
                            return Err(format!(
                                "undefined dependency for opcode {} is not proved",
                                op.number()
                            ));
                        }
                        out.fill(false);
                    }
                }
                if matches!(
                    op,
                    Op::ShiftRightLogical | Op::ShiftRightArithmetic | Op::ShiftLeftLogical
                ) && (!self.concrete
                    || self
                        .literals
                        .get(&a[3])
                        .is_none_or(|bits| bits.iter().any(|&v| v >= 32)))
                {
                    return Err("undefined or unknown shift count".into());
                }
                let folded = (0..len / 4)
                    .map(|i| {
                        let x = *self.literals.get(&a[2])?.get(i)?;
                        let y = if a.len() > 3 {
                            *self.literals.get(&a[3])?.get(i)?
                        } else {
                            0
                        };
                        compute::fold(op, x, y)
                    })
                    .collect::<Option<Vec<_>>>();
                self.result(a[0], a[1], bytes)?;
                if let Some(bits) = folded {
                    self.set_literal(a[1], bits);
                }
            }
            Op::Select => {
                let condition = *self
                    .constants
                    .get(&a[2])
                    .ok_or("nonconstant select unsupported")?;
                let bytes = self
                    .value(if condition == 0 { a[4] } else { a[3] })?
                    .to_vec();
                let selected = if condition == 0 { a[4] } else { a[3] };
                let bits = self.literals.get(&selected).cloned();
                self.result(a[0], a[1], bytes)?;
                if let Some(bits) = bits {
                    self.set_literal(a[1], bits);
                }
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
            | Op::AtomicXor
                if self.concrete =>
            {
                compute::atomic(self, op, a)?
            }
            Op::Function
            | Op::FunctionEnd
            | Op::LoopMerge
            | Op::SelectionMerge
            | Op::Label
            | Op::Branch
            | Op::Switch
            | Op::Return => {}
            // Debug and source metadata, checked by spirv-val before parse.
            Op::Nop
            | Op::SourceContinued
            | Op::Source
            | Op::SourceExtension
            | Op::Name
            | Op::MemberName
            | Op::String
            | Op::Line
            | Op::NoLine
            | Op::ModuleProcessed => {}
            _ => return Err(format!("unsupported SPIR-V opcode {}", op.number())),
        }
        Ok(())
    }
}

// Also used before spirv-val to classify a missing source execution contract.
fn instructions(words: &[u32]) -> Result<Vec<(Op, &[u32])>, String> {
    if words.len() < 5 || words[0] != 0x07230203 || words[3] > 1_000_000 {
        return Err("invalid or oversized SPIR-V header".into());
    }
    let mut instructions = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        if count == 0 || at + count > words.len() {
            return Err("malformed instruction length".into());
        }
        instructions.push((Op::of(words[at]), &words[at + 1..at + count]));
        at += count;
    }
    Ok(instructions)
}

// Token consumers wait at their push target: Ryujinx Instructions/
// InstEmitFlowControl.cs:239-275. The checker requires the production region
// contract even if this device happens to produce equal bytes without it.
// Mode, entry and extension opcodes: Mesa src/compiler/spirv/spirv.h:
// 210,1746,1750-1751; extension association: spirv.core.grammar.json:13565-13570.
pub(super) fn require_maximal_reconvergence(words: &[u32]) -> Result<(), super::Rejected> {
    let instructions = instructions(words)?;
    let entries = instructions
        .iter()
        .filter(|(op, a)| *op == Op::EntryPoint && a.len() >= 4 && a[2..4] == [0x6e69616d, 0])
        .map(|(_, a)| a[1])
        .collect::<Vec<_>>();
    let modes = instructions
        .iter()
        .filter(|(op, a)| *op == Op::ExecutionMode && a.len() == 2 && a[1] == 6023)
        .map(|(_, a)| a[0])
        .collect::<Vec<_>>();
    let extension = instructions.iter().any(|(op, a)| {
        *op == Op::Extension && physical::extension(a, b"SPV_KHR_maximal_reconvergence")
    });
    if entries.len() != 1 || modes != entries || !extension {
        return Err(super::Rejected::TokenRegionUncheckable);
    }
    Ok(())
}

impl Plan {
    /// Only called after spirv-val. The structural checks below additionally
    /// keep direct worker invocations bounded and reject unsupported modules.
    pub fn parse(words: &[u32], seed: u64, inputs: &super::inputs::Inputs) -> Result<Self, String> {
        let cb1 = inputs.cb1.as_deref();
        inputs.validate()?;
        if cb1.is_some_and(|bytes| bytes.len() > BUFFER_SIZE || !bytes.len().is_multiple_of(4)) {
            return Err("invalid CB1 snapshot extent".into());
        }
        let instructions = instructions(words)?;
        let mut analysis = Analysis {
            cb1_len: cb1.map(<[u8]>::len),
            ..Default::default()
        };
        let mut entry = None;
        let mut local = None;
        let mut local_size = [1; 3];
        let mut stage = 5;
        let mut modes = Vec::new();
        let mut maximal_extension = false;
        let mut maximal_entry = None;
        let mut functions = 0;
        let mut function_id = None;
        let mut labels = 0;
        let mut returned = false;
        let mut in_function = false;
        for &(op, a) in &instructions {
            if stage != 5
                && returned
                && op != Op::FunctionEnd
                && !matches!(
                    op,
                    Op::Nop
                        | Op::Undef
                        | Op::SourceContinued
                        | Op::Source
                        | Op::SourceExtension
                        | Op::Name
                        | Op::MemberName
                        | Op::String
                        | Op::Line
                        | Op::NoLine
                        | Op::ModuleProcessed
                )
            {
                return Err("instruction after return is unsupported".into());
            }
            // Prevent direct worker invocations from indexing short operands.
            let minimum = match op {
                Op::TypeVoid
                | Op::TypeBool
                | Op::Label
                | Op::Branch
                | Op::Capability
                | Op::TypeStruct
                | Op::TypeFunction => 1,
                Op::Undef
                | Op::ExtInstImport
                | Op::MemoryModel
                | Op::TypeFloat
                | Op::TypeRuntimeArray
                | Op::ConstantTrue
                | Op::ConstantFalse
                | Op::ConstantComposite
                | Op::ConstantNull
                | Op::Store
                | Op::CompositeConstruct
                | Op::Decorate => 2,
                Op::TypeInt
                | Op::TypeVector
                | Op::TypeArray
                | Op::TypePointer
                | Op::Variable
                | Op::Load
                | Op::AccessChain
                | Op::InBoundsAccessChain
                | Op::Constant
                | Op::CopyObject
                | Op::ConvertSToF
                | Op::ConvertUToF
                | Op::Bitcast
                | Op::SNegate
                | Op::FNegate
                | Op::LogicalNot
                | Op::Not
                | Op::BitReverse
                | Op::BitCount
                | Op::UConvert
                | Op::ConvertUToPtr => 3,
                Op::EntryPoint
                | Op::Function
                | Op::CompositeExtract
                | Op::IAdd
                | Op::FAdd
                | Op::ISub
                | Op::FSub
                | Op::IMul
                | Op::FMul
                | Op::LogicalEqual
                | Op::LogicalNotEqual
                | Op::LogicalOr
                | Op::LogicalAnd
                | Op::IEqual
                | Op::INotEqual
                | Op::UGreaterThan
                | Op::SGreaterThan
                | Op::UGreaterThanEqual
                | Op::SGreaterThanEqual
                | Op::ULessThan
                | Op::SLessThan
                | Op::ULessThanEqual
                | Op::SLessThanEqual
                | Op::FOrdEqual
                | Op::FUnordEqual
                | Op::FOrdNotEqual
                | Op::FUnordNotEqual
                | Op::FOrdLessThan
                | Op::FUnordLessThan
                | Op::FOrdGreaterThan
                | Op::FUnordGreaterThan
                | Op::FOrdLessThanEqual
                | Op::FUnordLessThanEqual
                | Op::FOrdGreaterThanEqual
                | Op::FUnordGreaterThanEqual
                | Op::BitwiseOr
                | Op::BitwiseXor
                | Op::BitwiseAnd
                | Op::MemberDecorate => 4,
                Op::Phi => 4,
                Op::LoopMerge | Op::BranchConditional => 3,
                Op::SelectionMerge | Op::Switch => 2,
                Op::VectorExtractDynamic
                | Op::ShiftRightLogical
                | Op::ShiftRightArithmetic
                | Op::ShiftLeftLogical => 4,
                Op::TypeImage => 8,
                Op::TypeSampler => 1,
                Op::TypeSampledImage => 2,
                Op::SampledImage | Op::ImageSampleImplicitLod | Op::ImageSampleExplicitLod => 4,
                Op::ImageSampleDrefImplicitLod | Op::ImageSampleDrefExplicitLod => 5,
                Op::Select
                | Op::GroupNonUniformShuffle
                | Op::GroupNonUniformShuffleXor
                | Op::GroupNonUniformShuffleUp
                | Op::GroupNonUniformShuffleDown
                | Op::GroupNonUniformUMin => 5,
                Op::GroupNonUniformAll
                | Op::GroupNonUniformAny
                | Op::GroupNonUniformAllEqual
                | Op::GroupNonUniformBallot => 4,
                Op::AtomicExchange
                | Op::AtomicIAdd
                | Op::AtomicISub
                | Op::AtomicSMin
                | Op::AtomicUMin
                | Op::AtomicSMax
                | Op::AtomicUMax
                | Op::AtomicAnd
                | Op::AtomicOr
                | Op::AtomicXor => 6,
                Op::AtomicCompareExchange => 8,
                Op::AtomicIIncrement | Op::AtomicIDecrement => 5,
                Op::ExecutionMode => 2,
                Op::EmitStreamVertex | Op::EndStreamPrimitive => 1,
                _ => 0,
            };
            if a.len() < minimum {
                return Err("short SPIR-V operands".into());
            }
            match op {
                // GLSL import declaration may survive an unoptimized build.
                // ExtInst remains unsupported. Import words: shadowbox/src/emitter.rs:1353-1357;
                // import opcode: /usr/include/spirv/unified1/spirv.h:1785.
                Op::ExtInstImport if a[1..] == [0x4c534c47, 0x6474732e, 0x3035342e, 0] => {}
                Op::Extension if physical::extension(a, b"SPV_KHR_physical_storage_buffer") => {}
                // Mesa src/compiler/spirv/spirv.h:210; project emitter.rs:1399-1415.
                Op::Extension if physical::extension(a, b"SPV_KHR_maximal_reconvergence") => {
                    maximal_extension = true;
                }
                Op::Extension => return Err("unsupported SPIR-V extension".into()),
                Op::MemoryModel if a == [5348, 1] => {
                    analysis.physical = true;
                }
                Op::MemoryModel if a == [0, 1] => {}
                Op::MemoryModel => return Err("physical global memory is not supported yet".into()),
                Op::EntryPoint if a.len() >= 4 && matches!(a[0], 0..=5) && entry.is_none() => {
                    // The runner uses the exact entry point name "main".
                    if a[2..].get(..2) != Some(&[0x6e69616d, 0]) {
                        return Err("entry point must be main".into());
                    }
                    entry = Some(a[1]);
                    stage = a[0];
                }
                Op::EntryPoint => {
                    return Err("only a single shader entry point is supported".into())
                }
                Op::ExecutionMode if a.len() == 2 && a[1] == 6023 => {
                    if maximal_entry.replace(a[0]).is_some() {
                        return Err("duplicate maximal reconvergence execution mode".into());
                    }
                }
                Op::ExecutionMode
                    if (stage == 1 && a.len() == 3 && a[1..] == [26, 1])
                        || (stage == 2 && a.len() == 2 && matches!(a[1], 1 | 4 | 5 | 10 | 22)) =>
                {
                    if Some(a[0]) != entry
                        || analysis
                            .graphics_modes
                            .insert(a[1], a.get(2).copied().unwrap_or(1))
                            .is_some()
                    {
                        return Err("invalid/duplicate tessellation execution mode".into());
                    }
                }
                Op::ExecutionMode
                    if stage == 3 && a.len() == 2 && matches!(a[1], 19 | 22 | 27..=29)
                        || stage == 3 && a.len() == 3 && matches!(a[1], 0 | 26) =>
                {
                    if Some(a[0]) != entry
                        || analysis
                            .graphics_modes
                            .insert(a[1], a.get(2).copied().unwrap_or(1))
                            .is_some()
                    {
                        return Err("invalid/duplicate geometry execution mode".into());
                    }
                }
                Op::ExecutionMode if a.len() == 5 && a[1] == 17 && local.is_none() => {
                    local = Some(a[0]);
                    local_size = compute::local_size(&a[2..])?;
                }
                Op::ExecutionMode if a.len() == 2 && matches!(a[1], 7 | 12) => {
                    modes.push((a[0], a[1]))
                }
                Op::ExecutionMode => return Err("unsupported execution mode".into()),
                Op::Capability if a == [54] => {
                    analysis.geometry_streams = true;
                }
                Op::Capability
                    if a.len() == 1
                        && matches!(a[0], 1 | 11 | 43 | 45 | 61..=66 | 2 | 3 | 54 | 5347) => {}
                Op::Capability => return Err(format!("unsupported capability {a:?}")),
                Op::Decorate if a.len() >= 2 => match a[1] {
                    2 | 3 => {
                        analysis.blocks.insert(a[0], a[1] == 3);
                    }
                    6 if a.len() == 3 => {
                        analysis.strides.insert(a[0], a[2] as usize);
                    }
                    33 | 34 if a.len() == 3 => {
                        let pair = analysis.bindings.entry(a[0]).or_default();
                        if a[1] == 33 {
                            pair.1 = Some(a[2]);
                        } else {
                            pair.0 = Some(a[2]);
                        }
                    }
                    29 if stage != 3 => {
                        return Err("Stream decoration requires geometry capture".into())
                    }
                    11 | 29 | 30 | 31 if a.len() == 3 => {
                        analysis
                            .io
                            .entry((a[0], None))
                            .or_default()
                            .decorate(a[1], a[2])?;
                    }
                    13..=18 if a.len() == 2 => {
                        analysis.io.entry((a[0], None)).or_default().qualify(a[1]);
                    }
                    // NonWritable/NonReadable affect validation; they do not
                    // introduce host resources or byte layout.
                    24 | 25 | 0 => {}
                    _ => return Err(format!("unsupported decoration {a:?}")),
                },
                Op::MemberDecorate if a.len() == 4 && a[2] == 35 => {
                    analysis
                        .offsets
                        .insert((a[0], a[1] as usize), a[3] as usize);
                }
                Op::MemberDecorate if a.len() == 4 && a[2] == 11 => {
                    analysis
                        .io
                        .entry((a[0], Some(a[1] as usize)))
                        .or_default()
                        .decorate(11, a[3])?;
                }
                Op::Function => {
                    functions += 1;
                    function_id = Some(a[1]);
                    in_function = true;
                }
                Op::Label => labels += 1,
                Op::SelectionMerge if a.len() == 2 && a[1] == 0 => {}
                Op::Branch if a.len() == 1 => {}
                Op::Switch
                    if a.len() >= 2
                        && a.len().is_multiple_of(2)
                        && (stage == 5 || a.len() == 2) => {}
                Op::BranchConditional if stage == 5 && matches!(a.len(), 3 | 5) => {}
                Op::LoopMerge if stage == 5 && a.len() == 3 && a[2] == 0 => {}
                Op::Phi if stage == 5 && a.len() >= 4 && a.len().is_multiple_of(2) => {}
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
                | Op::AtomicXor
                    if stage == 5 => {}
                Op::EmitVertex | Op::EndPrimitive if stage == 3 && a.is_empty() => {}
                Op::EmitStreamVertex | Op::EndStreamPrimitive if stage == 3 && a.len() == 1 => {}
                Op::GroupNonUniformAll
                | Op::GroupNonUniformAny
                | Op::GroupNonUniformAllEqual
                | Op::GroupNonUniformBallot
                | Op::GroupNonUniformShuffle
                | Op::GroupNonUniformShuffleXor
                | Op::GroupNonUniformShuffleUp
                | Op::GroupNonUniformShuffleDown
                | Op::GroupNonUniformUMin
                    if stage == 5 =>
                {
                    analysis.subgroup = true;
                }
                Op::SelectionMerge | Op::Branch | Op::Switch => {
                    return Err("only a forward single-path function is supported".into())
                }
                Op::Return if in_function => returned = true,
                Op::FunctionEnd
                | Op::Nop
                | Op::Undef
                | Op::SourceContinued
                | Op::Source
                | Op::SourceExtension
                | Op::Name
                | Op::MemberName
                | Op::String
                | Op::Line
                | Op::TypeVoid
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
                | Op::ConstantTrue
                | Op::ConstantFalse
                | Op::Constant
                | Op::ConstantComposite
                | Op::ConstantSampler
                | Op::ConstantNull
                | Op::Variable
                | Op::ImageTexelPointer
                | Op::Load
                | Op::Store
                | Op::AccessChain
                | Op::InBoundsAccessChain
                | Op::VectorExtractDynamic
                | Op::CompositeConstruct
                | Op::CompositeExtract
                | Op::CopyObject
                | Op::SampledImage
                | Op::ImageSampleImplicitLod
                | Op::ImageSampleExplicitLod
                | Op::ImageSampleDrefImplicitLod
                | Op::ImageSampleDrefExplicitLod
                | Op::ConvertSToF
                | Op::ConvertUToF
                | Op::UConvert
                | Op::ConvertUToPtr
                | Op::Bitcast
                | Op::SNegate
                | Op::FNegate
                | Op::IAdd
                | Op::FAdd
                | Op::ISub
                | Op::FSub
                | Op::IMul
                | Op::FMul
                | Op::LogicalEqual
                | Op::LogicalNotEqual
                | Op::LogicalOr
                | Op::LogicalAnd
                | Op::LogicalNot
                | Op::Select
                | Op::IEqual
                | Op::INotEqual
                | Op::UGreaterThan
                | Op::SGreaterThan
                | Op::UGreaterThanEqual
                | Op::SGreaterThanEqual
                | Op::ULessThan
                | Op::SLessThan
                | Op::ULessThanEqual
                | Op::SLessThanEqual
                | Op::FOrdEqual
                | Op::FUnordEqual
                | Op::FOrdNotEqual
                | Op::FUnordNotEqual
                | Op::FOrdLessThan
                | Op::FUnordLessThan
                | Op::FOrdGreaterThan
                | Op::FUnordGreaterThan
                | Op::FOrdLessThanEqual
                | Op::FUnordLessThanEqual
                | Op::FOrdGreaterThanEqual
                | Op::FUnordGreaterThanEqual
                | Op::ShiftRightLogical
                | Op::ShiftRightArithmetic
                | Op::ShiftLeftLogical
                | Op::BitwiseOr
                | Op::BitwiseXor
                | Op::BitwiseAnd
                | Op::Not
                | Op::BitReverse
                | Op::BitCount
                | Op::NoLine
                | Op::ModuleProcessed => {}
                _ => return Err(format!("unsupported SPIR-V opcode {}", op.number())),
            }
        }
        if entry.is_none() || functions != 1 || entry != function_id || labels == 0 || !returned {
            return Err("only one main function is supported".into());
        }
        // Forward branches may only target the immediately following label.
        // A switch with no cases always selects its default. This accepts the
        // production emitter's structured EXIT wrapper without skipping any
        // instruction or guessing a conditional path. Numeric branch, switch
        // and selection-merge fields: spirv.h:1999-2004.
        for (i, &(op, operands)) in instructions.iter().enumerate() {
            if stage != 5 && matches!(op, Op::Branch | Op::Switch) {
                let target = if op == Op::Branch {
                    operands[0]
                } else {
                    operands[1]
                };
                let next = instructions[i + 1..].iter().find(|(op, _)| {
                    !matches!(
                        op,
                        Op::Nop
                            | Op::Undef
                            | Op::SourceContinued
                            | Op::Source
                            | Op::SourceExtension
                            | Op::Name
                            | Op::MemberName
                            | Op::String
                            | Op::Line
                            | Op::NoLine
                            | Op::ModuleProcessed
                    )
                });
                if !matches!(next, Some((Op::Label, label)) if *label == [target]) {
                    return Err("branch would skip instructions or form a cycle".into());
                }
            }
        }
        if stage == 5 && (entry != local || !modes.is_empty())
            || stage != 5 && local.is_some()
            || stage == 0 && !modes.is_empty()
            || stage == 4
                && (!modes.contains(&(entry.unwrap(), 7))
                    || modes.iter().any(|(id, _)| Some(*id) != entry))
        {
            return Err("execution modes do not match stage".into());
        }
        if maximal_entry.is_some() && (!maximal_extension || maximal_entry != entry) {
            return Err("maximal reconvergence mode lacks its extension or main entry".into());
        }
        if !inputs.storage.is_empty() && stage != 5 {
            return Err("descriptor global capture requires compute".into());
        }
        if analysis.physical && stage != 5 {
            return Err("physical graphics memory unsupported".into());
        }
        analysis.maximal_reconvergence = maximal_entry.is_some();
        if stage == 5 {
            analysis.concrete = true;
            if analysis.physical {
                physical::initialize(&mut analysis);
            }
            for &(op, a) in instructions
                .iter()
                .take_while(|(op, _)| *op != Op::Function)
            {
                if !matches!(
                    op,
                    Op::Extension
                        | Op::ExtInstImport
                        | Op::MemoryModel
                        | Op::EntryPoint
                        | Op::ExecutionMode
                        | Op::Capability
                        | Op::Decorate
                        | Op::MemberDecorate
                ) {
                    analysis.instruction(op, a)?;
                }
            }
            storage::initialize(&mut analysis, inputs, seed)?;
            compute::analyze(&mut analysis, &instructions, seed, local_size)?;
        } else {
            for &(op, a) in &instructions {
                if !matches!(
                    op,
                    Op::Extension
                        | Op::ExtInstImport
                        | Op::MemoryModel
                        | Op::EntryPoint
                        | Op::ExecutionMode
                        | Op::Capability
                        | Op::Decorate
                        | Op::MemberDecorate
                ) {
                    analysis.instruction(op, a)?;
                }
            }
        }
        if stage != 5 {
            storage::initialize(&mut analysis, inputs, seed)?;
        }
        if stage == 5 && analysis.stores == 0 {
            return Err("no captured storage writes".into());
        }
        if stage == 5
            && !analysis.roots.values().any(|(root, kind)| {
                matches!(kind, BufferKind::Storage | BufferKind::Physical)
                    && analysis.memory[root]
                        .written
                        .iter()
                        .zip(&analysis.memory[root].data)
                        .any(|(written, defined)| *written && *defined)
            })
        {
            return Err("no defined captured storage writes".into());
        }
        let mut buffers = Vec::new();
        for ((set, binding), (root, kind)) in &analysis.roots {
            buffers.push(Buffer {
                set: *set,
                binding: *binding,
                kind: *kind,
                initial: analysis.host_initial[root].clone(),
                defined: analysis.memory[root].data.clone(),
                observation: None,
                expected: None,
            });
        }
        if analysis.physical && analysis.push < 8 {
            return Err("physical memory requires the delta push constant".into());
        }
        if let Some(cb1) = cb1 {
            if let Some(buffer) = buffers.iter_mut().find(|buffer| {
                buffer.kind == BufferKind::Uniform && (buffer.set, buffer.binding) == (0, 1)
            }) {
                // Vulkan uniform blocks retain raw saved bits, following Ryujinx
                // CodeGen/Spirv/Declarations.cs:82-144. No seeded CB1 substitute.
                buffer.initial.fill(0);
                buffer.initial[..cb1.len()].copy_from_slice(cb1);
            }
        }
        let compute_words = if analysis.subgroup {
            Some(compute::capture_subgroup(words, &analysis, &mut buffers)?)
        } else {
            None
        };
        let graphics = if stage != 5 {
            Some(graphics::prepare(
                words,
                stage,
                entry.unwrap(),
                seed,
                &analysis,
                &mut buffers,
            )?)
        } else {
            None
        };
        Ok(Self {
            stage,
            local_size,
            subgroup: analysis.subgroup,
            maximal_reconvergence: maximal_entry.is_some(),
            global_capture: storage::capture(&analysis, inputs, seed)?,
            compute_words,
            shared_size: analysis
                .storage_classes
                .iter()
                .filter(|(_, class)| **class == 4)
                .map(|(root, _)| analysis.memory[root].data.len())
                .sum(),
            graphics,
            buffers,
            push: pattern(seed, 0x50555348, 0, 0, analysis.push),
            images: analysis
                .resources
                .into_iter()
                .map(|((set, binding), (_, kind))| images::Resource { set, binding, kind })
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overwriting_constant_memory_invalidates_sampling_literal_proof() {
        let mut a = Analysis::default();
        a.types.insert(1, Type::Scalar);
        a.types.insert(2, Type::Pointer(7, 1));
        a.variable(2, 10, 7, None).unwrap();
        a.instruction(Op::Constant, &[1, 20, 0x3e80_0000]).unwrap();
        a.instruction(Op::Undef, &[1, 21]).unwrap();
        a.instruction(Op::Store, &[10, 20]).unwrap();
        a.instruction(Op::Load, &[1, 30, 10]).unwrap();
        assert_eq!(a.literals[&30], [0x3e80_0000]);
        a.instruction(Op::Store, &[10, 21]).unwrap();
        a.instruction(Op::Load, &[1, 31, 10]).unwrap();
        assert!(!a.literals.contains_key(&31));
        assert_eq!(a.value(31).unwrap(), [false; 4]);
        a.instruction(Op::Load, &[1, 30, 10]).unwrap();
        assert!(!a.literals.contains_key(&30));
        assert!(!a.constants.contains_key(&30));
    }

    #[test]
    fn patterns_depend_on_interface_identity_and_seed() {
        assert_eq!(pattern(42, 1, 0, 3, 64), pattern(42, 1, 0, 3, 64));
        assert_ne!(pattern(42, 1, 0, 3, 64), pattern(43, 1, 0, 3, 64));
        assert_ne!(pattern(42, 1, 0, 3, 64), pattern(42, 1, 0, 4, 64));
    }

    #[test]
    fn uninitialized_load_taints_only_its_written_bytes_and_overwrite_recovers() {
        let mut a = Analysis::default();
        a.types.insert(1, Type::Scalar);
        a.types.insert(2, Type::Pointer(7, 1));
        a.types.insert(3, Type::Pointer(12, 1));
        a.variable(2, 10, 7, None).unwrap();
        a.memory.insert(
            20,
            Memory {
                data: vec![true; 16],
                writable: true,
                written: vec![false; 16],
            },
        );
        a.pointers.insert(
            20,
            Pointer {
                root: 20,
                ty: 1,
                offset: 4,
            },
        );
        a.instruction(Op::Load, &[1, 11, 10]).unwrap();
        a.instruction(Op::Store, &[20, 11]).unwrap();
        assert_eq!(
            a.memory[&20].data,
            [
                true, true, true, true, false, false, false, false, true, true, true, true, true,
                true, true, true
            ]
        );
        a.instruction(Op::Constant, &[1, 12, 123]).unwrap();
        a.instruction(Op::Store, &[20, 12]).unwrap();
        assert!(a.memory[&20].data.iter().all(|b| *b));
    }

    #[test]
    fn ambiguous_undefined_arithmetic_is_uncheckable() {
        let mut a = Analysis::default();
        a.types.insert(1, Type::Scalar);
        a.instruction(Op::Undef, &[1, 10]).unwrap();
        a.instruction(Op::Constant, &[1, 11, 0]).unwrap();
        assert!(a
            .instruction(Op::IMul, &[1, 12, 10, 11])
            .unwrap_err()
            .contains("not proved"));
        assert!(a
            .instruction(Op::BitwiseAnd, &[1, 12, 10, 11])
            .unwrap_err()
            .contains("not proved"));
        assert!(a
            .instruction(Op::ISub, &[1, 12, 10, 10])
            .unwrap_err()
            .contains("not proved"));
        a.instruction(Op::IAdd, &[1, 12, 10, 11]).unwrap();
        assert_eq!(a.value(12).unwrap(), &[false; 4]);
    }

    #[test]
    fn rejects_dynamic_and_out_of_bounds_addresses() {
        let mut a = Analysis::default();
        a.types.insert(1, Type::Scalar);
        a.types.insert(2, Type::Array(1, Some(2)));
        a.types.insert(3, Type::Pointer(7, 2));
        a.variable(3, 10, 7, None).unwrap();
        assert!(a.chain(10, &[999]).unwrap_err().contains("dynamic"));
        a.constants.insert(999, 2);
        assert!(a.chain(10, &[999]).unwrap_err().contains("bounds"));
    }
}

#[cfg(test)]
mod reconvergence_requirement_tests {
    use super::require_maximal_reconvergence;

    // Project-owned declaration-only inputs for the pre-validation check.
    // Header/word count: emitter.rs:50-53; mode/extension fields cited above.
    fn declarations(mode: Option<u32>, extension: bool) -> Vec<u32> {
        let mut words = vec![0x07230203, 0x00010500, 0, 10, 0];
        if extension {
            words.push((9 << 16) | 10);
            for bytes in b"SPV_KHR_maximal_reconvergence\0\0\0".as_chunks::<4>().0 {
                words.push(u32::from_le_bytes(*bytes));
            }
        }
        words.extend([(5 << 16) | 15, 5, 1, 0x6e69616d, 0]);
        if let Some(entry) = mode {
            words.extend([(3 << 16) | 16, entry, 6023]);
        }
        words
    }

    #[test]
    fn token_contract_accepts_mode_on_main_with_extension() {
        require_maximal_reconvergence(&declarations(Some(1), true)).unwrap();
    }

    #[test]
    fn token_contract_rejects_each_removed_declaration() {
        for (mode, extension) in [(None, true), (Some(1), false), (None, false)] {
            assert!(matches!(
                require_maximal_reconvergence(&declarations(mode, extension)),
                Err(super::super::Rejected::TokenRegionUncheckable)
            ));
        }
    }

    #[test]
    fn token_contract_rejects_mode_on_another_entry() {
        assert!(require_maximal_reconvergence(&declarations(Some(2), true)).is_err());
    }
}
