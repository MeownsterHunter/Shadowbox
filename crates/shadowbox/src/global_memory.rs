//! Flat physical global memory. Encoding provenance is on the decoder arm.
//! Numeric SPIR-V facts below were checked against
//! /usr/include/spirv/unified1/spirv.h:104 (addressing),257 (storage),
//! 1095,1105,1121,1177,1189,1282 (capabilities),513,545 (decorations),
//! 1028 (Aligned),1784,1790,1793,1802,1804,1814,1827,1829-1833,
//! 1839-1840,1847-1848,1878,1885,1889,1892,1957,1960,1963-1964,
//! 1931,2000-2003,2006-2007 (opcodes). PushConstant is :238.
//! Vulkan's SPIR-V environment rule
//! VUID-StandaloneSpirv-PhysicalStorageBuffer64-04708 requires Aligned on
//! physical loads and stores. SPV_KHR_physical_storage_buffer sections 3.4
//! and 3.7 keep logical register pointers separate from physical pointers.

use crate::ir::ops;
use crate::{
    emitter::inst,
    interface::Stopped,
    ir::{AtomicWidth, GlobalAccess, Instruction, InstructionKind, LsSize2, Register},
    spirv::Op,
    TranslationOptions, PUSH_GLOBAL_DELTA_OFFSET,
};
use std::collections::{BTreeSet, HashMap};

/// Usage bit: the instruction touches physical global memory.
const GLOBAL: usize = 1;
/// Usage bit: an STG stores 8-bit subwords.
const STORE8: usize = 2;
/// Usage bit: an STG stores 16-bit subwords.
const STORE16: usize = 4;
/// Narrow-store usage bits of each STG size, indexed by LsSize2.
const STORE_WIDTH: [usize; 7] = [STORE8, STORE8, STORE16, STORE16, 0, 0, 0];
/// Usage bit: a global ATOM/RED accesses a 64-bit word.
const WIDE_ATOMIC: usize = 8;
/// Atomic usage bits of each width, indexed by `AtomicWidth as usize - 1`.
const ATOMIC_WIDTH: [usize; 2] = [0, WIDE_ATOMIC];

/// What one instruction asks of global memory, as usage bits, and its
/// immediate address offset (0 adds nothing).
fn usage(x: &Instruction) -> (usize, i32) {
    match x.kind.legacy_form() {
        InstructionKind::GlobalMemory(ops::GlobalMemory {
            access: GlobalAccess::Store(size),
            offset,
            ..
        }) => (GLOBAL | STORE_WIDTH[size as usize], offset),
        InstructionKind::GlobalMemory(ops::GlobalMemory { offset, .. }) => (GLOBAL, offset),
        InstructionKind::GlobalAtomic(ops::GlobalAtomic { offset, width, .. }) => {
            (GLOBAL | ATOMIC_WIDTH[usize::from(width as u8 - 1)], offset)
        }
        _ => (0, 0),
    }
}

/// Union of every instruction's usage bits.
fn usages(xs: &[Instruction]) -> usize {
    xs.iter().fold(0, |bits, x| bits | usage(x).0)
}

pub(crate) fn is_global(x: &Instruction) -> bool {
    usage(x).0 & GLOBAL != 0
}

/// Cache hints have no Vulkan cache-policy operand. Preserve the decoded hint
/// in module debug output, with no extra executable instructions. OpString is
/// spirv.h:1782. Ryujinx InstEmitMemory.cs:106-111,142-147 also omits cache policy.
pub(crate) fn debug_hints(w: &mut Vec<u32>, n: &mut impl FnMut() -> u32, xs: &[Instruction]) {
    for x in xs {
        // Once for a global instruction, never for any other.
        for _ in 0..usage(x).0 & GLOBAL {
            string_inst(w, Op::String, &[n()], &x.to_string());
        }
    }
}

fn fresh(id: &mut u32) -> u32 {
    let value = *id;
    *id += 1;
    value
}

/// `prefix` is the result id of an OpString, empty for OpExtension.
fn string_inst(w: &mut Vec<u32>, op: Op, prefix: &[u32], text: &str) {
    let mut bytes = text.as_bytes().to_vec();
    bytes.push(0);
    bytes.resize(bytes.len().next_multiple_of(4), 0);
    let mut words = prefix.to_vec();
    words.extend(
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b)),
    );
    inst(w, op, &words);
}

/// Capabilities each usage set needs, indexed by usage bits. Narrow and
/// wide bits never appear without GLOBAL. Int64Atomics is spirv.h
/// Capability 12.
const CAPABILITIES: [&[u32]; 16] = [
    &[],
    &[5347, 11],
    &[],
    &[5347, 11, 39, 4448],
    &[],
    &[5347, 11, 22, 4433],
    &[],
    &[5347, 11, 39, 4448, 22, 4433],
    &[],
    &[5347, 11, 12],
    &[],
    &[5347, 11, 12, 39, 4448],
    &[],
    &[5347, 11, 12, 22, 4433],
    &[],
    &[5347, 11, 12, 39, 4448, 22, 4433],
];

/// Usage bits that select extensions and narrow types.
const STORAGE_BITS: usize = WIDE_ATOMIC - 1;

/// Extensions each usage set needs, indexed by usage bits.
const EXTENSIONS: [&[&str]; 8] = [
    &[],
    &["SPV_KHR_physical_storage_buffer"],
    &[],
    &["SPV_KHR_physical_storage_buffer", "SPV_KHR_8bit_storage"],
    &[],
    &["SPV_KHR_physical_storage_buffer", "SPV_KHR_16bit_storage"],
    &[],
    &[
        "SPV_KHR_physical_storage_buffer",
        "SPV_KHR_8bit_storage",
        "SPV_KHR_16bit_storage",
    ],
];

/// Narrow integer types each usage set declares, as (first LsSize2 index,
/// bit width), indexed by usage bits.
const NARROW_TYPES: [&[(usize, u32)]; 8] = [
    &[],
    &[],
    &[],
    &[(0, 8)],
    &[],
    &[(2, 16)],
    &[],
    &[(0, 8), (2, 16)],
];

pub(crate) fn capabilities(w: &mut Vec<u32>, xs: &[Instruction]) {
    for &capability in CAPABILITIES[usages(xs)] {
        inst(w, Op::Capability, &[capability]);
    }
}

pub(crate) fn extensions(w: &mut Vec<u32>, xs: &[Instruction]) {
    for name in EXTENSIONS[usages(xs) & STORAGE_BITS] {
        string_inst(w, Op::Extension, &[], name);
    }
}

/// The host's 64-bit global address delta: either a push constant added to
/// every address, or the identity when the host promises zero.
#[derive(Clone, Copy)]
struct Delta {
    variable: u32,
    member_pointer: u32,
    /// The loaded delta value, once loaded.
    value: u32,
    load: fn(&GlobalMemoryEmitter, &mut Vec<u32>, &mut u32, u32) -> u32,
    add: fn(&GlobalMemoryEmitter, &mut Vec<u32>, &mut u32, u32) -> u32,
}

/// Declares the host delta push constant.
fn push_delta<N: FnMut() -> u32>(
    w: &mut Vec<u32>,
    annotations: &mut Vec<u32>,
    n: &mut N,
    uint64: u32,
) -> Delta {
    let block = n();
    let block_pointer = n();
    let member_pointer = n();
    let variable = n();
    inst(w, Op::TypeStruct, &[block, uint64]);
    inst(w, Op::TypePointer, &[block_pointer, 9, block]);
    inst(w, Op::TypePointer, &[member_pointer, 9, uint64]);
    inst(annotations, Op::Decorate, &[block, 2]);
    inst(
        annotations,
        Op::MemberDecorate,
        &[block, 0, 35, PUSH_GLOBAL_DELTA_OFFSET],
    );
    inst(w, Op::Variable, &[block_pointer, variable, 9]);
    Delta {
        variable,
        member_pointer,
        value: 0,
        load: |emitter, w, id, zero| {
            let delta = emitter.delta;
            let pointer = fresh(id);
            inst(
                w,
                Op::AccessChain,
                &[delta.member_pointer, pointer, delta.variable, zero],
            );
            let value = fresh(id);
            inst(w, Op::Load, &[emitter.uint64, value, pointer]);
            value
        },
        add: |emitter, w, id, addr| {
            let sum = fresh(id);
            inst(
                w,
                Op::IAdd,
                &[emitter.uint64, sum, addr, emitter.delta.value],
            );
            sum
        },
    }
}

/// The host promises a zero delta: nothing is declared, loaded or added.
fn zero_delta<N: FnMut() -> u32>(_: &mut Vec<u32>, _: &mut Vec<u32>, _: &mut N, _: u32) -> Delta {
    Delta {
        variable: 0,
        member_pointer: 0,
        value: 0,
        load: |_, _, _, _| 0,
        add: |_, _, _, addr| addr,
    }
}

pub(crate) struct GlobalMemoryEmitter {
    uint64: u32,
    types: [u32; 7],
    pointers: [u32; 7],
    /// Atomic pointers indexed by `AtomicWidth as usize - 1`. The 64-bit
    /// PhysicalStorageBuffer pointer to uint64 is declared only for 64-bit
    /// atomics.
    atomic_pointers: [u32; 2],
    offsets: HashMap<i32, u32>,
    word_mask: u32,
    delta: Delta,
}

impl GlobalMemoryEmitter {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn declare<N: FnMut() -> u32>(
        w: &mut Vec<u32>,
        annotations: &mut Vec<u32>,
        n: &mut N,
        xs: &[Instruction],
        options: TranslationOptions,
        uint: u32,
        uvec2: u32,
        uvec4: u32,
    ) -> Self {
        let uint64 = n();
        inst(w, Op::TypeInt, &[uint64, 64, 0]);
        let mut types = [uint, uint, uint, uint, uint, uvec2, uvec4];
        let mut pointers = [0; 7];
        for &(index, bits) in NARROW_TYPES[usages(xs) & STORAGE_BITS] {
            let ty = n();
            inst(w, Op::TypeInt, &[ty, bits, 0]);
            let pointer = n();
            inst(w, Op::TypePointer, &[pointer, 5349, ty]);
            types[index..index + 2].fill(ty);
            pointers[index..index + 2].fill(pointer);
        }
        for index in 4..7 {
            let pointer = n();
            inst(w, Op::TypePointer, &[pointer, 5349, types[index]]);
            pointers[index] = pointer;
        }
        // 64-bit atomic pointer declarations indexed by the wide usage bit.
        type Pointer64<N> = fn(&mut Vec<u32>, &mut N, u32) -> u32;
        let pointer64s: [Pointer64<N>; 2] = [
            |_, _, _| 0,
            |w, n, uint64| {
                let pointer = n();
                inst(w, Op::TypePointer, &[pointer, 5349, uint64]);
                pointer
            },
        ];
        let pointer64 = pointer64s[usages(xs) / WIDE_ATOMIC](w, n, uint64);
        let delta = match options.global_delta {
            crate::GlobalDelta::Pushed => push_delta::<N>(w, annotations, n, uint64),
            crate::GlobalDelta::Zero => zero_delta::<N>(w, annotations, n, uint64),
        };
        let mut offsets = HashMap::new();
        let mut distinct = xs.iter().map(|x| usage(x).1).collect::<BTreeSet<_>>();
        // Offset 0 adds nothing and has no constant.
        distinct.remove(&0);
        for offset in distinct {
            // Sign extension before 64-bit modular addition, including borrow.
            let bits = i64::from(offset) as u64;
            let constant = n();
            inst(
                w,
                Op::Constant,
                &[uint64, constant, bits as u32, (bits >> 32) as u32],
            );
            offsets.insert(offset, constant);
        }
        let word_mask = n();
        inst(w, Op::Constant, &[uint64, word_mask, !3_u32, u32::MAX]);
        Self {
            uint64,
            types,
            pointers,
            atomic_pointers: [pointers[LsSize2::B32 as usize], pointer64],
            offsets,
            word_mask,
            delta,
        }
    }

    pub(crate) fn integer_type(&self) -> u32 {
        // I2F reuses the uint64 type declared above.
        self.uint64
    }

    /// Loads the host delta once, at function entry. Without a delta this
    /// emits nothing.
    pub(crate) fn load_delta(mut self, w: &mut Vec<u32>, id: &mut u32, zero: u32) -> Self {
        self.delta.value = (self.delta.load)(&self, w, id, zero);
        self
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit(
        &self,
        w: &mut Vec<u32>,
        globals: &mut Vec<u32>,
        id: &mut u32,
        x: &Instruction,
        regs: &[u32],
        uint: u32,
        uvec2: u32,
        signed_int: u32,
        boolt: u32,
        zero: u32,
        pred: Option<(u32, bool)>,
    ) -> Result<(), Stopped> {
        let frame = Frame {
            regs,
            uint,
            uvec2,
            signed_int,
            zero,
        };
        let unsupported = || Stopped::Unbuilt(Box::new(*x));
        // The one decision: what this instruction is. Each arm lowers its
        // own body; nothing downstream asks again.
        match x.kind.legacy_form() {
            InstructionKind::GlobalMemory(ops::GlobalMemory {
                access: GlobalAccess::Load(size),
                data,
                address,
                offset,
                extended,
                ..
            }) => self.lower(
                w,
                globals,
                id,
                &frame,
                boolt,
                pred,
                Address::new(address, offset, extended),
                Load {
                    form: LOADS[size.access() as usize],
                    data,
                },
                unsupported,
            ),
            InstructionKind::GlobalMemory(ops::GlobalMemory {
                access: GlobalAccess::Store(size),
                data,
                address,
                offset,
                extended,
                ..
            }) => self.lower(
                w,
                globals,
                id,
                &frame,
                boolt,
                pred,
                Address::new(address, offset, extended),
                Store {
                    size,
                    pack: STORES[size as usize],
                    data,
                },
                unsupported,
            ),
            InstructionKind::GlobalAtomic(ops::GlobalAtomic {
                dst: Some(dst),
                data,
                address,
                offset,
                extended,
                signed,
                op,
                width,
            }) => self.atomic(
                w,
                globals,
                id,
                &frame,
                boolt,
                pred,
                Address::new(address, offset, extended),
                ATOMICS[usize::from(op)][usize::from(signed)],
                WIDTHS[usize::from(width as u8 - 1)],
                dst,
                data,
                unsupported,
            ),
            // RED has no destination; RZ discards alike.
            InstructionKind::GlobalAtomic(ops::GlobalAtomic {
                dst: None,
                data,
                address,
                offset,
                extended,
                signed,
                op,
                width,
            }) => self.atomic(
                w,
                globals,
                id,
                &frame,
                boolt,
                pred,
                Address::new(address, offset, extended),
                ATOMICS[usize::from(op)][usize::from(signed)],
                WIDTHS[usize::from(width as u8 - 1)],
                Register(255),
                data,
                unsupported,
            ),
            _ => unreachable!(),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn atomic(
        &self,
        w: &mut Vec<u32>,
        globals: &mut Vec<u32>,
        id: &mut u32,
        frame: &Frame,
        boolt: u32,
        pred: Option<(u32, bool)>,
        address: Address,
        form: Result<AtomicForm, Unsupported>,
        width: AtomicWidthForm,
        dst: Register,
        data: Register,
        unsupported: impl Fn() -> Stopped,
    ) -> Result<(), Stopped> {
        form.map_err(|Unsupported| unsupported()).and_then(|form| {
            self.lower(
                w,
                globals,
                id,
                frame,
                boolt,
                pred,
                address,
                Atomic {
                    form,
                    width,
                    dst,
                    data,
                },
                &unsupported,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn lower(
        &self,
        w: &mut Vec<u32>,
        globals: &mut Vec<u32>,
        id: &mut u32,
        frame: &Frame,
        boolt: u32,
        pred: Option<(u32, bool)>,
        address: Address,
        body: impl Body,
        unsupported: impl Fn() -> Stopped,
    ) -> Result<(), Stopped> {
        // Reject overflowing register vectors instead of wrapping to R0. RZ
        // alone is valid: stores write zeroes, loads still perform the access.
        let data = body.moved().0;
        if data != 255 && usize::from(data) + usize::from(body.size().registers()) > 256 {
            return Err(unsupported());
        }
        // Ryujinx InstEmitMemory.cs:359-386,457-486 snapshots the address
        // before any destination write, also when it overlaps the loaded vector.
        // Guard the access itself, not just register writes. False predicates
        // must not dereference an invalid address or perform a store.
        let merge = open_guard(w, id, boolt, pred);
        let low = frame.register(w, id, usize::from(address.register.0));
        let addr = fresh(id);
        (address.widen)(self, frame, w, id, address.register, low, addr);
        let addr = OFFSETS[address.offset](self, w, id, addr, address.value);
        let addr = (self.delta.add)(self, w, id, addr);
        let view = body.view();
        let pointer = fresh(id);
        let pointer_addr = body.align(self, w, id, addr);
        inst(
            w,
            Op::ConvertUToPtr,
            &[body.pointer(self), pointer, pointer_addr],
        );
        body.transfer(
            self,
            frame,
            w,
            globals,
            id,
            Site {
                pointer,
                addr,
                ty: self.types[view as usize],
                alignment: view.alignment(),
            },
        );
        close_guard(w, merge);
        Ok(())
    }
}

/// Opens the selection that guards an access with its instruction
/// predicate, returning the merge label to close; nothing without one.
/// Ryujinx Translator.cs:359-393 guards the operation itself.
pub(crate) fn open_guard(
    w: &mut Vec<u32>,
    id: &mut u32,
    boolt: u32,
    pred: Option<(u32, bool)>,
) -> Option<u32> {
    pred.map(|(p, inverted)| {
        let condition = fresh(id);
        inst(w, Op::Load, &[boolt, condition, p]);
        let condition = NEGATE[usize::from(inverted)](w, id, boolt, condition);
        let body = fresh(id);
        let merge = fresh(id);
        inst(w, Op::SelectionMerge, &[merge, 0]);
        inst(w, Op::BranchConditional, &[condition, body, merge]);
        inst(w, Op::Label, &[body]);
        merge
    })
}

/// Closes the selection `open_guard` opened.
pub(crate) fn close_guard(w: &mut Vec<u32>, merge: Option<u32>) {
    merge.into_iter().for_each(|merge| {
        inst(w, Op::Branch, &[merge]);
        inst(w, Op::Label, &[merge]);
    });
}

/// Writes the predicate condition, inverted or as loaded.
type Negate = fn(&mut Vec<u32>, &mut u32, u32, u32) -> u32;
/// Condition writers indexed by the inversion flag.
const NEGATE: [Negate; 2] = [
    |_, _, _, condition| condition,
    |w, id, boolt, condition| {
        let negated = fresh(id);
        inst(w, Op::LogicalNot, &[boolt, negated, condition]);
        negated
    },
];

/// Adds the immediate offset constant, or nothing for offset 0.
type Offset = fn(&GlobalMemoryEmitter, &mut Vec<u32>, &mut u32, u32, i32) -> u32;
/// Offset adders indexed by whether the offset is nonzero.
const OFFSETS: [Offset; 2] = [
    |_, _, _, addr, _| addr,
    |emitter, w, id, addr, offset| {
        let sum = fresh(id);
        inst(
            w,
            Op::IAdd,
            &[emitter.uint64, sum, addr, emitter.offsets[&offset]],
        );
        sum
    },
];

/// Builds the 64-bit address from the low word.
type Widen = fn(&GlobalMemoryEmitter, &Frame, &mut Vec<u32>, &mut u32, Register, u32, u32);
/// Address builders indexed by the E (extended) flag.
const WIDEN: [Widen; 2] = [
    |emitter, _, w, _, _, low, addr| {
        inst(w, Op::UConvert, &[emitter.uint64, addr, low]);
    },
    |emitter, frame, w, id, register, low, addr| {
        // E is a register pair; RZ has no high register (Ryujinx
        // InstEmitMemory.cs:512-532) and reads past it as zero. Bitcast packs
        // low then high words.
        let high = frame.register(w, id, usize::from(register.0) + 1);
        let pair = fresh(id);
        inst(w, Op::CompositeConstruct, &[frame.uvec2, pair, low, high]);
        inst(w, Op::Bitcast, &[emitter.uint64, addr, pair]);
    },
];

/// The address operands of one access.
struct Address {
    register: Register,
    value: i32,
    /// Index into OFFSETS.
    offset: usize,
    widen: Widen,
}

impl Address {
    fn new(register: Register, offset: i32, extended: bool) -> Self {
        Self {
            register,
            value: offset,
            offset: usize::from(offset != 0),
            widen: WIDEN[usize::from(extended)],
        }
    }
}

/// The function-wide ids an access reads and writes registers through.
struct Frame<'a> {
    regs: &'a [u32],
    uint: u32,
    uvec2: u32,
    signed_int: u32,
    zero: u32,
}

/// Reads one register value; RZ and indices past it read zero.
type Read = fn(&Frame, &mut Vec<u32>, &mut u32, usize) -> u32;
/// Register readers indexed by `index / 255`: R0-R254, then RZ and beyond.
/// Indices never exceed 258 (RZ plus a B128 vector).
const READS: [Read; 2] = [
    |frame, w, id, index| {
        let value = fresh(id);
        inst(w, Op::Load, &[frame.uint, value, frame.regs[index]]);
        value
    },
    |frame, _, _, _| frame.zero,
];

impl Frame<'_> {
    fn register(&self, w: &mut Vec<u32>, id: &mut u32, index: usize) -> u32 {
        READS[index / 255](self, w, id, index)
    }

    /// Registers `data..data + count` that exist: RZ and beyond are dropped.
    fn written(data: Register, count: usize) -> std::ops::Range<usize> {
        let data = usize::from(data.0);
        data..(data + count).min(255)
    }
}

/// The pointer an access goes through.
struct Site {
    pointer: u32,
    /// The full 64-bit address, before any word alignment.
    addr: u32,
    ty: u32,
    alignment: u32,
}

/// One kind of global access: what its pointer views, and what it moves.
trait Body {
    /// The first register of the vector the access moves: a load's or an
    /// atomic's destination, a store's source.
    fn moved(&self) -> Register;
    /// The size whose registers the access moves.
    fn size(&self) -> LsSize2;
    /// The size the pointer views.
    fn view(&self) -> LsSize2;
    /// The physical pointer type the access goes through.
    fn pointer(&self, emitter: &GlobalMemoryEmitter) -> u32 {
        emitter.pointers[self.view() as usize]
    }
    /// The address the pointer is built from.
    fn align(
        &self,
        emitter: &GlobalMemoryEmitter,
        w: &mut Vec<u32>,
        id: &mut u32,
        addr: u32,
    ) -> u32;
    fn transfer(
        &self,
        emitter: &GlobalMemoryEmitter,
        frame: &Frame,
        w: &mut Vec<u32>,
        globals: &mut Vec<u32>,
        id: &mut u32,
        site: Site,
    );
}

/// How an LDG of one size reads its containing memory.
#[derive(Clone, Copy)]
struct LoadForm {
    size: LsSize2,
    /// The pointer's view: B32 for sub-word loads.
    view: LsSize2,
    align: fn(&GlobalMemoryEmitter, &mut Vec<u32>, &mut u32, u32) -> u32,
    /// Sub-word width extracted, unused for whole-word forms.
    width: u32,
    extract: Extract,
    write: Write,
}

/// Turns the loaded value into the register value.
type Extract = fn(&Frame, &mut Vec<u32>, &mut Vec<u32>, &mut u32, u32, u32, u32) -> u32;
/// Writes the loaded value into the destination registers.
type Write = fn(&Frame, &mut Vec<u32>, &mut u32, Register, usize, u32);

/// The address as it is.
fn unaligned(_: &GlobalMemoryEmitter, _: &mut Vec<u32>, _: &mut u32, addr: u32) -> u32 {
    addr
}

/// The address of the containing word.
fn word_aligned(emitter: &GlobalMemoryEmitter, w: &mut Vec<u32>, id: &mut u32, addr: u32) -> u32 {
    let aligned = fresh(id);
    inst(
        w,
        Op::BitwiseAnd,
        &[emitter.uint64, aligned, addr, emitter.word_mask],
    );
    aligned
}

/// A whole-word or vector load is its own value.
fn whole(
    _: &Frame,
    _: &mut Vec<u32>,
    _: &mut Vec<u32>,
    _: &mut u32,
    value: u32,
    _: u32,
    _: u32,
) -> u32 {
    value
}

/// Ryujinx GlobalToStorage.cs:928-950 extracts from a containing word. One
/// bit-field extract replaces its shift plus extension. Returns the bit
/// offset and the width constant.
fn subword_position(
    frame: &Frame,
    w: &mut Vec<u32>,
    globals: &mut Vec<u32>,
    id: &mut u32,
    addr: u32,
    width: u32,
) -> (u32, u32) {
    let low_addr = fresh(id);
    inst(w, Op::UConvert, &[frame.uint, low_addr, addr]);
    let three = constant(globals, id, frame.uint, 3);
    let byte = fresh(id);
    inst(w, Op::BitwiseAnd, &[frame.uint, byte, low_addr, three]);
    let bit = fresh(id);
    inst(w, Op::ShiftLeftLogical, &[frame.uint, bit, byte, three]);
    let width = constant(globals, id, frame.uint, width);
    (bit, width)
}

/// Zero-extended sub-word.
fn unsigned_subword(
    frame: &Frame,
    w: &mut Vec<u32>,
    globals: &mut Vec<u32>,
    id: &mut u32,
    value: u32,
    addr: u32,
    width: u32,
) -> u32 {
    let (bit, width) = subword_position(frame, w, globals, id, addr, width);
    let extracted = fresh(id);
    inst(
        w,
        Op::BitFieldUExtract,
        &[frame.uint, extracted, value, bit, width],
    );
    extracted
}

/// Sign-extended sub-word.
fn signed_subword(
    frame: &Frame,
    w: &mut Vec<u32>,
    globals: &mut Vec<u32>,
    id: &mut u32,
    value: u32,
    addr: u32,
    width: u32,
) -> u32 {
    let (bit, width) = subword_position(frame, w, globals, id, addr, width);
    let extracted = fresh(id);
    let signed_word = fresh(id);
    inst(w, Op::Bitcast, &[frame.signed_int, signed_word, value]);
    inst(
        w,
        Op::BitFieldSExtract,
        &[frame.signed_int, extracted, signed_word, bit, width],
    );
    let bits = fresh(id);
    inst(w, Op::Bitcast, &[frame.uint, bits, extracted]);
    bits
}

/// One loaded word into its register.
fn write_scalar(
    frame: &Frame,
    w: &mut Vec<u32>,
    _: &mut u32,
    data: Register,
    _: usize,
    value: u32,
) {
    for reg in Frame::written(data, 1) {
        inst(w, Op::Store, &[frame.regs[reg], value]);
    }
}

/// Each component of a loaded vector into consecutive registers.
fn write_vector(
    frame: &Frame,
    w: &mut Vec<u32>,
    id: &mut u32,
    data: Register,
    count: usize,
    value: u32,
) {
    let first = usize::from(data.0);
    for reg in Frame::written(data, count) {
        let part = fresh(id);
        inst(
            w,
            Op::CompositeExtract,
            &[frame.uint, part, value, (reg - first) as u32],
        );
        inst(w, Op::Store, &[frame.regs[reg], part]);
    }
}

/// Loads read sub-word sizes from their containing word.
const fn subword_load(size: LsSize2, width: u32, extract: Extract) -> LoadForm {
    LoadForm {
        size,
        view: LsSize2::B32,
        align: word_aligned,
        width,
        extract,
        write: write_scalar,
    }
}

/// One vector load reads consecutive words for B64/B128. Splitting it into
/// repeated loads from the same pointer would be incorrect.
const fn word_load(size: LsSize2, write: Write) -> LoadForm {
    LoadForm {
        size,
        view: size,
        align: unaligned,
        width: 32,
        extract: whole,
        write,
    }
}

/// LDG forms indexed by the access size.
const LOADS: [LoadForm; 7] = [
    subword_load(LsSize2::U8, 8, unsigned_subword),
    subword_load(LsSize2::S8, 8, signed_subword),
    subword_load(LsSize2::U16, 16, unsigned_subword),
    subword_load(LsSize2::S16, 16, signed_subword),
    word_load(LsSize2::B32, write_scalar),
    word_load(LsSize2::B64, write_vector),
    word_load(LsSize2::B128, write_vector),
];

struct Load {
    form: LoadForm,
    data: Register,
}

impl Body for Load {
    fn moved(&self) -> Register {
        self.data
    }

    fn size(&self) -> LsSize2 {
        self.form.size
    }

    fn view(&self) -> LsSize2 {
        self.form.view
    }

    fn align(
        &self,
        emitter: &GlobalMemoryEmitter,
        w: &mut Vec<u32>,
        id: &mut u32,
        addr: u32,
    ) -> u32 {
        (self.form.align)(emitter, w, id, addr)
    }

    fn transfer(
        &self,
        _: &GlobalMemoryEmitter,
        frame: &Frame,
        w: &mut Vec<u32>,
        globals: &mut Vec<u32>,
        id: &mut u32,
        site: Site,
    ) {
        let value = fresh(id);
        inst(
            w,
            Op::Load,
            &[site.ty, value, site.pointer, 2, site.alignment],
        );
        let value = (self.form.extract)(frame, w, globals, id, value, site.addr, self.form.width);
        let count = usize::from(self.form.size.registers());
        (self.form.write)(frame, w, id, self.data, count, value);
    }
}

/// Builds the stored value from the data registers.
type Pack = fn(&Frame, &mut Vec<u32>, &mut u32, Register, usize, u32) -> u32;

/// One word from its register.
fn pack_word(
    frame: &Frame,
    w: &mut Vec<u32>,
    id: &mut u32,
    data: Register,
    _: usize,
    _: u32,
) -> u32 {
    frame.register(w, id, usize::from(data.0))
}

/// One register truncated to the narrow stored type.
fn pack_subword(
    frame: &Frame,
    w: &mut Vec<u32>,
    id: &mut u32,
    data: Register,
    _: usize,
    ty: u32,
) -> u32 {
    let value = frame.register(w, id, usize::from(data.0));
    let truncated = fresh(id);
    inst(w, Op::UConvert, &[ty, truncated, value]);
    truncated
}

/// Load every source register first. OpCompositeConstruct takes values,
/// never the ids of Function-storage variables.
fn pack_vector(
    frame: &Frame,
    w: &mut Vec<u32>,
    id: &mut u32,
    data: Register,
    count: usize,
    ty: u32,
) -> u32 {
    let mut args = vec![ty, fresh(id)];
    for component in 0..count {
        args.push(frame.register(w, id, usize::from(data.0) + component));
    }
    inst(w, Op::CompositeConstruct, &args);
    args[1]
}

/// STG packers indexed by the access size.
const STORES: [Pack; 7] = [
    pack_subword,
    pack_subword,
    pack_subword,
    pack_subword,
    pack_word,
    pack_vector,
    pack_vector,
];

struct Store {
    size: LsSize2,
    pack: Pack,
    data: Register,
}

impl Body for Store {
    fn moved(&self) -> Register {
        self.data
    }

    fn size(&self) -> LsSize2 {
        self.size
    }

    fn view(&self) -> LsSize2 {
        self.size
    }

    fn align(
        &self,
        emitter: &GlobalMemoryEmitter,
        w: &mut Vec<u32>,
        id: &mut u32,
        addr: u32,
    ) -> u32 {
        unaligned(emitter, w, id, addr)
    }

    fn transfer(
        &self,
        _: &GlobalMemoryEmitter,
        frame: &Frame,
        w: &mut Vec<u32>,
        _: &mut Vec<u32>,
        id: &mut u32,
        site: Site,
    ) {
        let count = usize::from(self.size.registers());
        let value = (self.pack)(frame, w, id, self.data, count, site.ty);
        inst(w, Op::Store, &[site.pointer, value, 2, site.alignment]);
    }
}

/// An atomic operation field value with no SPIR-V lowering.
#[derive(Clone, Copy)]
struct Unsupported;

/// The ids one atomic instruction is built from.
#[derive(Clone, Copy)]
struct AtomicIds {
    result_type: u32,
    result: u32,
    pointer: u32,
    scope: u32,
    value: u32,
}

/// Writes one atomic instruction into `ids.result`, typed `ids.result_type`.
type AtomicForm = fn(&Frame, &mut Vec<u32>, &mut u32, &Atomic, AtomicIds);

/// Read-modify-write atomics differ only in their opcode.
macro_rules! rmw {
    ($op:ident) => {
        Ok(
            |frame: &Frame,
             w: &mut Vec<u32>,
             _: &mut u32,
             _: &Atomic,
             AtomicIds {
                 result_type,
                 result,
                 pointer,
                 scope,
                 value,
             }| {
                // SMin/SMax interpret integer bits as signed even though the
                // register and physical pointer carry uint bits. This avoids two
                // bitcasts and a duplicate signed pointer type. SPIR-V section
                // 2.8.1 and OpAtomicSMin/SMax define this interpretation.
                inst(
                    w,
                    Op::$op,
                    &[result_type, result, pointer, scope, frame.zero, value],
                );
            },
        )
    };
}

/// Mesa ir.rs:2691, sm50.rs:2697-2708 packs compare first, replacement
/// second. SPIR-V orders replacement then compare.
fn compare_exchange(
    frame: &Frame,
    w: &mut Vec<u32>,
    id: &mut u32,
    atomic: &Atomic,
    AtomicIds {
        result_type: _,
        result,
        pointer,
        scope,
        value,
    }: AtomicIds,
) {
    let replacement = frame.register(w, id, usize::from(atomic.data.0) + 1);
    inst(
        w,
        Op::AtomicCompareExchange,
        &[
            frame.uint,
            result,
            pointer,
            scope,
            frame.zero,
            frame.zero,
            replacement,
            value,
        ],
    );
}

/// ATOM/RED lowering of each operation field value (0 add, 1 min, 2 max,
/// 5 and, 6 or, 7 xor, 8 exchange, 15 CAS), then signedness.
const ATOMICS: [[Result<AtomicForm, Unsupported>; 2]; 256] = {
    let mut table: [[Result<AtomicForm, Unsupported>; 2]; 256] = [[Err(Unsupported); 2]; 256];
    table[0] = [rmw!(AtomicIAdd), rmw!(AtomicIAdd)];
    table[1] = [rmw!(AtomicUMin), rmw!(AtomicSMin)];
    table[2] = [rmw!(AtomicUMax), rmw!(AtomicSMax)];
    table[5] = [rmw!(AtomicAnd), rmw!(AtomicAnd)];
    table[6] = [rmw!(AtomicOr), rmw!(AtomicOr)];
    table[7] = [rmw!(AtomicXor), rmw!(AtomicXor)];
    table[8] = [rmw!(AtomicExchange), rmw!(AtomicExchange)];
    table[15] = [Ok(compare_exchange), Ok(compare_exchange)];
    table
};

/// Builds an atomic's data value from register `data` holding `value`;
/// returns it with the result type.
type Join = fn(&GlobalMemoryEmitter, &Frame, &mut Vec<u32>, &mut u32, Register, u32) -> (u32, u32);

/// What one atomic width changes: the pointer, the data value, the
/// result type and the destination write.
#[derive(Clone, Copy)]
struct AtomicWidthForm {
    width: AtomicWidth,
    /// Builds the operation's data value; returns it with the result type.
    join: Join,
    /// Writes the old value into the destination registers.
    write: fn(&Frame, &mut Vec<u32>, &mut u32, Register, u32),
}

/// Atomic widths indexed by `AtomicWidth as usize - 1`.
const WIDTHS: [AtomicWidthForm; 2] = [
    AtomicWidthForm {
        width: AtomicWidth::B32,
        join: |_, frame, _, _, _, value| (value, frame.uint),
        write: |frame, w, id, dst, result| write_scalar(frame, w, id, dst, 1, result),
    },
    AtomicWidthForm {
        width: AtomicWidth::B64,
        // 64-bit data is the even pair data:data+1, low word first, as for
        // the E address pair. Mesa from_nir.rs:2129-2173.
        join: |emitter, frame, w, id, data, value| {
            let high = frame.register(w, id, usize::from(data.0) + 1);
            let pair = fresh(id);
            inst(w, Op::CompositeConstruct, &[frame.uvec2, pair, value, high]);
            let joined = fresh(id);
            inst(w, Op::Bitcast, &[emitter.uint64, joined, pair]);
            (joined, emitter.uint64)
        },
        write: |frame, w, id, dst, result| {
            // Once for a real destination, never for RZ.
            for first in Frame::written(dst, 1) {
                let pair = fresh(id);
                inst(w, Op::Bitcast, &[frame.uvec2, pair, result]);
                for half in 0..2 {
                    let word = fresh(id);
                    inst(w, Op::CompositeExtract, &[frame.uint, word, pair, half]);
                    inst(w, Op::Store, &[frame.regs[first + half as usize], word]);
                }
            }
        },
    },
];

struct Atomic {
    form: AtomicForm,
    width: AtomicWidthForm,
    dst: Register,
    data: Register,
}

impl Body for Atomic {
    fn moved(&self) -> Register {
        self.dst
    }

    fn size(&self) -> LsSize2 {
        self.width.width.size()
    }

    fn view(&self) -> LsSize2 {
        LsSize2::B32
    }

    fn pointer(&self, emitter: &GlobalMemoryEmitter) -> u32 {
        emitter.atomic_pointers[usize::from(self.width.width as u8 - 1)]
    }

    fn align(
        &self,
        emitter: &GlobalMemoryEmitter,
        w: &mut Vec<u32>,
        id: &mut u32,
        addr: u32,
    ) -> u32 {
        unaligned(emitter, w, id, addr)
    }

    fn transfer(
        &self,
        emitter: &GlobalMemoryEmitter,
        frame: &Frame,
        w: &mut Vec<u32>,
        globals: &mut Vec<u32>,
        id: &mut u32,
        site: Site,
    ) {
        // Ryujinx InstEmitMemory.cs:16-24 reads all inputs before writing
        // the old value to dst. The address and data can alias that dst.
        let value = frame.register(w, id, usize::from(self.data.0));
        let (value, result_type) = (self.width.join)(emitter, frame, w, id, self.data, value);
        let result = fresh(id);
        // Ryujinx CodeGen/Spirv/Instructions.cs:1630-1655 uses Device=1,
        // Relaxed=0 for global atomic operations. Numeric opcodes and scope:
        // /usr/include/spirv/unified1/spirv.h:1042,1984-1997.
        let scope = constant(globals, id, frame.uint, 1);
        (self.form)(
            frame,
            w,
            id,
            self,
            AtomicIds {
                result_type,
                result,
                pointer: site.pointer,
                scope,
                value,
            },
        );
        (self.width.write)(frame, w, id, self.dst, result);
    }
}

fn constant(w: &mut Vec<u32>, id: &mut u32, ty: u32, bits: u32) -> u32 {
    let value = fresh(id);
    inst(w, Op::Constant, &[ty, value, bits]);
    value
}
