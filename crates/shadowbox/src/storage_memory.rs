//! Descriptor global memory after an equal-source cbuf proof.
//! Independent implementation of Ryujinx GlobalToStorage.cs:305-357,409-445.
//! Mesa nak/sm50.rs:2386-2431,2603-2614,2659-2669 defines the LDG/STG
//! fields; the decoder retains them and rejects unnamed fields.
//! SPIR-V numeric values: Mesa src/compiler/spirv/spirv.h:235,507,511,537-539,
//! 1763-1764,1766,1776,1789,1791-1792,1795,
//! 1893,1962-1965 (StorageBuffer, decorations, types, memory and flow).
use crate::ir::ops;
use crate::{
    emitter::{inst, HandleLeaf, HandleSource},
    fast_hash::HashMap,
    interface::{DescriptorMemory, Stopped, StorageBufferBinding},
    ir::{GlobalAccess, Instruction, InstructionKind, LsSize, LsSize2, Register, Source},
    reaching::ReachingDefinitions,
    spirv::Op,
};
use std::collections::BTreeSet;

pub(crate) struct Access {
    binding: u32,
    word: u32,
    /// The data register's SPIR-V variable index (RZ included).
    data: usize,
    /// What this access does with its element pointer, fixed at plan.
    transfer: Transfer,
}

/// One descriptor word transfer: loads, stores, and their RZ forms.
type Transfer = fn(&StorageMemory, &mut Vec<u32>, &[u32], u32, u32, u32, usize);

/// LDG into a register.
fn load(
    _: &StorageMemory,
    w: &mut Vec<u32>,
    regs: &[u32],
    uint: u32,
    pointer: u32,
    value: u32,
    data: usize,
) {
    inst(w, Op::Load, &[uint, value, pointer]);
    inst(w, Op::Store, &[regs[data], value]);
}

/// LDG into RZ: the load still happens, the result is discarded.
fn load_discard(
    _: &StorageMemory,
    w: &mut Vec<u32>,
    _: &[u32],
    uint: u32,
    pointer: u32,
    value: u32,
    _: usize,
) {
    inst(w, Op::Load, &[uint, value, pointer]);
}

/// STG of a register.
fn store(
    _: &StorageMemory,
    w: &mut Vec<u32>,
    regs: &[u32],
    uint: u32,
    pointer: u32,
    value: u32,
    data: usize,
) {
    inst(w, Op::Load, &[uint, value, regs[data]]);
    inst(w, Op::Store, &[pointer, value]);
}

/// STG of RZ: the stored word is the zero constant.
fn store_zero(
    memory: &StorageMemory,
    w: &mut Vec<u32>,
    _: &[u32],
    _: u32,
    pointer: u32,
    _: u32,
    _: usize,
) {
    inst(w, Op::Store, &[pointer, memory.zero]);
}

/// Load transfers indexed by whether the data register is RZ.
const LOADS: [Transfer; 2] = [load, load_discard];
/// Store transfers indexed by whether the data register is RZ.
const STORES: [Transfer; 2] = [store, store_zero];

pub(crate) fn plan(
    xs: &[Instruction],
    reaching: &ReachingDefinitions,
    bindings: &[StorageBufferBinding],
) -> Result<HashMap<usize, Access>, Stopped> {
    let mut accesses = HashMap::default();
    if bindings.is_empty() {
        return Ok(accesses);
    }
    let invalid = Stopped::DescriptorMemory;
    let mut sources = BTreeSet::new();
    let mut slots = BTreeSet::new();
    // Mesa sm50.rs:364-368: aligned 14-bit word offsets and 5-bit banks.
    // The low word must leave room for the high word within 64 KiB.
    for b in bindings {
        if b.bank >= 32 || b.offset > 65528 || b.offset % 4 != 0 {
            return Err(invalid(DescriptorMemory::MisalignedBase));
        }
        if !sources.insert((b.bank, b.offset)) || !slots.insert(b.binding) {
            return Err(invalid(DescriptorMemory::DuplicateBinding));
        }
    }
    for (index, x) in xs.iter().enumerate() {
        // Set 1 also holds scalar textures. Reject collisions rather than
        // reinterpret an image descriptor as a buffer (emitter.rs's TEXS path).
        if matches!(
            x.kind,
            InstructionKind::Texture(ops::Texture { scalar: true, .. })
        ) {
            return Err(invalid(DescriptorMemory::TextureSetOverlap));
        }
        if !crate::global_memory::is_global(x) {
            continue;
        }
        // The one decision: which transfer this instruction is.
        let (address, offset, data, transfers) = match x.kind {
            InstructionKind::GlobalMemory(ops::GlobalMemory {
                address,
                offset,
                data,
                extended: true,
                access: GlobalAccess::Load(LsSize::B32),
                ..
            }) => (address, offset, data, LOADS),
            InstructionKind::GlobalMemory(ops::GlobalMemory {
                address,
                offset,
                data,
                extended: true,
                access: GlobalAccess::Store(LsSize2::B32),
                ..
            }) => (address, offset, data, STORES),
            _ => return Err(invalid(DescriptorMemory::NotWord)),
        };
        if address.0 >= 254 || offset < 0 || offset % 4 != 0 {
            return Err(invalid(DescriptorMemory::UnalignedAddress));
        }
        let cb = |r| match reaching.source(index, Source::Register(Register(r))) {
            Some(HandleSource::Leaf(HandleLeaf::ConstantBuffer { bank, offset })) => {
                Some((bank, offset))
            }
            _ => None,
        };
        let Some((bank, offset_low)) = cb(address.0) else {
            return Err(invalid(DescriptorMemory::NoLowSource));
        };
        if cb(address.0 + 1) != offset_low.checked_add(4).map(|o| (bank, o)) {
            return Err(invalid(DescriptorMemory::NoHighSource));
        }
        let Some(binding) = bindings
            .iter()
            .find(|b| b.bank == bank && b.offset == offset_low)
        else {
            return Err(invalid(DescriptorMemory::NoBinding));
        };
        // A proven base plus immediate offset needs no runtime subtraction or
        // 64-bit address assembly. This is Ryujinx's inline wordOffset path
        // (GlobalToStorage.cs:409-445), folded at translation time.
        accesses.insert(
            index,
            Access {
                binding: binding.binding,
                word: offset as u32 / 4,
                data: usize::from(data.0),
                transfer: transfers[usize::from(data.0 == 255)],
            },
        );
    }
    Ok(accesses)
}

pub(crate) struct StorageMemory {
    element_pointer: u32,
    buffers: HashMap<u32, u32>,
    words: HashMap<u32, u32>,
    zero: u32,
}
impl StorageMemory {
    pub(crate) fn declare(
        w: &mut Vec<u32>,
        annotations: &mut Vec<u32>,
        n: &mut impl FnMut() -> u32,
        accesses: &HashMap<usize, Access>,
        uint: u32,
    ) -> Self {
        let mut result = Self {
            element_pointer: 0,
            buffers: HashMap::default(),
            words: HashMap::default(),
            zero: 0,
        };
        if accesses.is_empty() {
            return result;
        }
        let array = n();
        let block = n();
        let pointer = n();
        result.element_pointer = n();
        result.zero = n();
        inst(w, Op::TypeRuntimeArray, &[array, uint]);
        inst(w, Op::TypeStruct, &[block, array]);
        inst(w, Op::TypePointer, &[pointer, 12, block]);
        inst(w, Op::TypePointer, &[result.element_pointer, 12, uint]);
        inst(w, Op::Constant, &[uint, result.zero, 0]);
        inst(annotations, Op::Decorate, &[array, 6, 4]);
        inst(annotations, Op::Decorate, &[block, 2]);
        inst(annotations, Op::MemberDecorate, &[block, 0, 35, 0]);
        for binding in accesses
            .values()
            .map(|a| a.binding)
            .collect::<BTreeSet<_>>()
        {
            let variable = n();
            inst(w, Op::Variable, &[pointer, variable, 12]);
            inst(annotations, Op::Decorate, &[variable, 34, 1]);
            inst(annotations, Op::Decorate, &[variable, 33, binding]);
            result.buffers.insert(binding, variable);
        }
        for word in accesses.values().map(|a| a.word).collect::<BTreeSet<_>>() {
            let constant = n();
            inst(w, Op::Constant, &[uint, constant, word]);
            result.words.insert(word, constant);
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit(
        &self,
        w: &mut Vec<u32>,
        id: &mut u32,
        access: &Access,
        regs: &[u32],
        uint: u32,
        boolt: u32,
        pred: Option<(u32, bool)>,
    ) {
        // Ryujinx Translator.cs:359-393 guards the operation itself. Never
        // access a descriptor for a lane whose instruction predicate is false.
        let merge = crate::global_memory::open_guard(w, id, boolt, pred);
        let mut next = || {
            let value = *id;
            *id += 1;
            value
        };
        let pointer = next();
        inst(
            w,
            Op::AccessChain,
            &[
                self.element_pointer,
                pointer,
                self.buffers[&access.binding],
                self.zero,
                self.words[&access.word],
            ],
        );
        // Ryujinx CodeGen/Spirv/Instructions.cs:1658-1669 uses logical
        // descriptor pointers for storage loads/stores.
        let value = next();
        (access.transfer)(self, w, regs, uint, pointer, value, access.data);
        crate::global_memory::close_guard(w, merge);
    }
}
