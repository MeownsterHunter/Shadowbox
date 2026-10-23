mod edge;
mod families;
pub mod formats;
use crate::flow::Program;
use crate::ir::ops;
use crate::ir::{
    AtomicOp, AtomicWidth, AttributeBank, AttributeLanes, BranchTarget, BrxBase, CSetPCondition,
    Conversion, Denormals, DirectedRounding, DoubleAddendSign, DoubleMinMaxOperation,
    DoubleProductSign, DoubleSource, DoubleSourceModifier, F2fDestination, F2fSource,
    FAddModifiers, FFmaModifiers, FMulModifiers, FMulRounding, FMulScale, FloatCompare, FloatFlags,
    Flush, FunnelType, GatherOffsets, GlobalAccess, HalfFlush, HalfOp, HalfSelect, Instruction,
    InstructionKind, IntCompare, IntegerCompareMode, IntegerWidth, InterpFreq, InterpLoc,
    LdcAddressMode, LdcWidth, LocalStoreCache, LogicOp, LsSize, LsSize2, MufuOp, OutputVertex,
    PartialSumShift, Predicate, PredicateOp, PredicateOutput, PrimitiveVertex, Register,
    RegisterPair, Relation, RoundingMode, Saturation, SetOp, SetResult, ShuffleMode, Signedness,
    Source, SourceAbs, SourceNegation, TargetStack, TexOp, VideoElement, VideoSelection, VoteOp,
    VoteVtgMode, XmadCop, U28,
};
use crate::run::Run;
const fn b(r: u64, s: u32, w: u32) -> u32 {
    ((r >> s) & ((1u64 << w) - 1)) as u32
}

/// Binary64 source modifier, indexed by the abs bit then the negate bit.
const DOUBLE_SOURCE_MODIFIER: [[DoubleSourceModifier; 2]; 2] = [
    [DoubleSourceModifier::Plain, DoubleSourceModifier::Negated],
    [
        DoubleSourceModifier::Absolute,
        DoubleSourceModifier::NegatedAbsolute,
    ],
];
/// The binary64 source modifier formed by the abs bit and the negate bit.
fn double_source_modifier(r: u64, abs: u32, negate: u32) -> DoubleSourceModifier {
    DOUBLE_SOURCE_MODIFIER[b(r, abs, 1) as usize][b(r, negate, 1) as usize]
}

/// The `width`-bit two's-complement field at `start`, sign-extended.
/// The FTZ/FMZ pair at `bit`: neither, FTZ, or FMZ (DNZ). Both set is
/// reserved (envytools gm107.c:1330-1335,1516-1520); only the source-locked
/// FMZ words read this pair, and none sets both.
fn flush(r: u64, bit: u32) -> Flush {
    [Flush::Keep, Flush::Ftz, Flush::Dnz][b(r, bit, 2) as usize]
}
fn signed(r: u64, start: u32, width: u32) -> i32 {
    ((b(r, start, width) << (32 - width)) as i32) >> (32 - width)
}
fn src(r: u64, long: bool, reg: bool) -> Source {
    if long {
        Source::Immediate(b(r, 20, 32))
    } else if reg {
        Source::Register(Register(b(r, 20, 8) as u8))
    } else {
        Source::Immediate(b(r, 20, 19))
    }
}
/// Register B, 20..27.
fn register_b(r: u64) -> Source {
    Source::Register(Register(b(r, 20, 8) as u8))
}
/// Register C, 39..46.
fn register_c(r: u64) -> Source {
    Source::Register(Register(b(r, 39, 8) as u8))
}
/// Constant-buffer B: word offset 20..33, bank 34..38 (Mesa
/// nak/sm50.rs:361-372).
fn constant_b(r: u64) -> Source {
    Source::ConstantBuffer {
        binding: b(r, 34, 5) as u8,
        offset: (b(r, 20, 14) * 4) as u16,
    }
}
/// Short half literal: nine magnitude bits plus signs 29/56, and six
/// implicit zero bits (envytools :1977-1983,2155-2157,240-241).
fn short_b(r: u64) -> Source {
    Source::Immediate((b(r, 20, 10) << 6) | (b(r, 30, 9) << 22) | (b(r, 56, 1) << 31))
}
/// The 32-bit half-pair literal of the long forms.
fn long_b(r: u64) -> Source {
    Source::Immediate(b(r, 20, 32))
}
/// The 20-bit integer immediate B with its sign at bit 56
/// (set_src_imm_i20, sm50.rs:335-358).
fn signed_immediate20(r: u64) -> Source {
    let value = b(r, 20, 19) | (b(r, 56, 1) << 19);
    Source::Immediate(((value << 12) as i32 >> 12) as u32)
}
fn imm32(r: u64) -> Source {
    Source::Immediate((b(r, 20, 19) << 12) | (b(r, 56, 1) << 31))
}
/// Integer and video compare field, 3 bits wide in every encoding that
/// carries it (envytools gm107.c:1445-1453,1588-1596), so each value names
/// a compare.
const CMP: [IntCompare; 8] = [
    IntCompare::False,
    IntCompare::Relation(Relation::Lt),
    IntCompare::Relation(Relation::Eq),
    IntCompare::Relation(Relation::Le),
    IntCompare::Relation(Relation::Gt),
    IntCompare::Relation(Relation::Ne),
    IntCompare::Relation(Relation::Ge),
    IntCompare::True,
];
fn cmp(v: u32) -> IntCompare {
    CMP[v as usize]
}
/// Mesa's 4-bit float compare field (set_float_cmp_op, sm50.rs:775-796):
/// 0 false, 1..6 ordered Lt Eq Le Gt Ne Ge, 7 IsNum, 8 IsNan,
/// 9..14 unordered Lt Eq Le Gt Ne Ge, 15 true.
const FCMP: [FloatCompare; 16] = [
    FloatCompare::False,
    FloatCompare::Ordered(Relation::Lt),
    FloatCompare::Ordered(Relation::Eq),
    FloatCompare::Ordered(Relation::Le),
    FloatCompare::Ordered(Relation::Gt),
    FloatCompare::Ordered(Relation::Ne),
    FloatCompare::Ordered(Relation::Ge),
    FloatCompare::Num,
    FloatCompare::Nan,
    FloatCompare::Unordered(Relation::Lt),
    FloatCompare::Unordered(Relation::Eq),
    FloatCompare::Unordered(Relation::Le),
    FloatCompare::Unordered(Relation::Gt),
    FloatCompare::Unordered(Relation::Ne),
    FloatCompare::Unordered(Relation::Ge),
    FloatCompare::True,
];
fn fcmp(v: u32) -> FloatCompare {
    FCMP[v as usize]
}
/// ISETP/ISET/ICMP bit 48 (envytools gm107.c:1365-1367).
fn signedness(r: u64) -> Signedness {
    [Signedness::Unsigned, Signedness::Signed][b(r, 48, 1) as usize]
}
/// LDG size field (Ryujinx Decoders/InstDecoders.cs:426-436); all eight
/// values are assigned.
const LS_SIZE: [LsSize; 8] = [
    LsSize::U8,
    LsSize::S8,
    LsSize::U16,
    LsSize::S16,
    LsSize::B32,
    LsSize::B64,
    LsSize::B128,
    LsSize::UB128,
];

fn ls_size(v: u32) -> LsSize {
    LS_SIZE[v as usize]
}
fn logic_op(v: u32) -> LogicOp {
    [LogicOp::And, LogicOp::Or, LogicOp::Xor, LogicOp::PassB][(v & 3) as usize]
}
/// LOP/LOP3 predicate output: mode field and destination; PT (7) writes
/// nothing (Ryujinx Decoders/InstDecoders.cs:463-469).
fn predicate_output(mode: u32, dst: u32) -> Option<PredicateOutput> {
    (dst != 7).then_some(PredicateOutput {
        op: [
            PredicateOp::F,
            PredicateOp::T,
            PredicateOp::Z,
            PredicateOp::Nz,
        ][(mode & 3) as usize],
        dst: dst as u8,
    })
}
/// Second source by encoding form, indexed by the opcode's top nibble
/// (bits 60..63): 0x4 constant buffer (set_src_cb(20..39): offset 20..34 in
/// words, bank 34..39), 0x3 a 20-bit immediate with its sign at bit 56,
/// every other nibble register B (set_src_imm_i20 / set_src_imm_f20,
/// sm50.rs:335-358; set_src_cb, sm50.rs:361-372).
const INT_FORMS: [fn(u64) -> Source; 16] = {
    let mut t = [register_b as fn(u64) -> Source; 16];
    t[3] = signed_immediate20;
    t[4] = constant_b;
    t
};
/// As [`INT_FORMS`], with the float f20 immediate.
const FLOAT_FORMS: [fn(u64) -> Source; 16] = {
    let mut t = [register_b as fn(u64) -> Source; 16];
    t[3] = imm32;
    t[4] = constant_b;
    t
};
/// Integer B by the word's encoding form.
fn int_form_src(r: u64) -> Source {
    INT_FORMS[b(r, 60, 4) as usize](r)
}
/// Float B by the word's encoding form.
fn float_form_src(r: u64) -> Source {
    FLOAT_FORMS[b(r, 60, 4) as usize](r)
}
/// The 2-bit rounding field: RN, RM, RP, RZ (Mesa sm50.rs:695-700).
fn rounding(v: u32) -> RoundingMode {
    [
        RoundingMode::NearestEven,
        RoundingMode::NegativeInfinity,
        RoundingMode::PositiveInfinity,
        RoundingMode::Zero,
    ][v as usize]
}
/// The FMUL rounding field, Round39..40: RN, RM, RP, RZ (Mesa
/// sm50.rs:695-700).
fn fmul_rounding(r: u64) -> FMulRounding {
    [
        FMulRounding::NearestEven,
        FMulRounding::Directed(DirectedRounding::NegativeInfinity),
        FMulRounding::Directed(DirectedRounding::PositiveInfinity),
        FMulRounding::Directed(DirectedRounding::Zero),
    ][b(r, 39, 2) as usize]
}
/// The float abs bit at `bit`.
fn source_abs(r: u64, bit: u32) -> SourceAbs {
    [SourceAbs::Plain, SourceAbs::Absolute][b(r, bit, 1) as usize]
}
/// The float negation bit at `bit`.
fn source_negation(r: u64, bit: u32) -> SourceNegation {
    [SourceNegation::Plain, SourceNegation::Negated][b(r, bit, 1) as usize]
}
/// The SAT bit at `bit`.
fn saturation(r: u64, bit: u32) -> Saturation {
    [Saturation::Unclamped, Saturation::Saturated][b(r, bit, 1) as usize]
}
/// The FTZ bit at `bit`.
fn denormals(r: u64, bit: u32) -> Denormals {
    [Denormals::Preserved, Denormals::FlushedToZero][b(r, bit, 1) as usize]
}
/// The concrete state of the float .CC bit at `bit`.
fn float_flags(r: u64, bit: u32) -> FloatFlags {
    [FloatFlags::Kept, FloatFlags::Written][b(r, bit, 1) as usize]
}
/// FADD register, cbuf and f20 forms: abs srcs[0] 46 and srcs[1] 49, neg
/// 48 (srcs[0]) and 45 (srcs[1]), sat 50, FTZ 44, and round 39..40
/// (set_reg_fmod_src(range, abs, neg), Mesa sm50.rs:268-278, FAdd encoder).
/// .CC47 writes Z/N from the final result and keeps C/V (envytools
/// gm107.c:2006,2080,2126; Ryujinx InstEmitAluHelper.cs:124-138).
fn fadd(r: u64, rhs: Source, modifiers: FAddModifiers) -> InstructionKind {
    InstructionKind::FAdd(ops::FAdd {
        dst: Register(b(r, 0, 8) as u8),
        lhs: Register(b(r, 8, 8) as u8),
        rhs,
        modifiers,
        round: rounding(b(r, 39, 2)),
        flags: float_flags(r, 47),
    })
}
/// FADD register and cbuf modifiers.
fn fadd_operand(r: u64) -> FAddModifiers {
    FAddModifiers::Operand {
        lhs_abs: source_abs(r, 46),
        rhs_abs: source_abs(r, 49),
        lhs_negation: source_negation(r, 48),
        rhs_negation: source_negation(r, 45),
        saturation: saturation(r, 50),
        denormals: denormals(r, 44),
    }
}
/// FADD f20 modifiers: the arm matches srcs[1] abs (bit 49) as zero.
fn fadd_immediate20(r: u64) -> FAddModifiers {
    FAddModifiers::Immediate20 {
        lhs_abs: source_abs(r, 46),
        lhs_negation: source_negation(r, 48),
        rhs_negation: source_negation(r, 45),
        saturation: saturation(r, 50),
        denormals: denormals(r, 44),
    }
}
/// FFMA, all forms: bit 48 negates the product and bit 49 the addend, sat
/// 50, rounding 51..52, FTZ 53 and DNZ 54 (Mesa sm50.rs FFma encoder). Only
/// round-to-nearest is modeled, .CC47 is concrete Z/N state, and FTZ+DNZ is
/// reserved (envytools
/// gm107.c:1516-1520; Mesa from_nir.rs:1177-1180 emits only one mode bit),
/// so the family rejects those words.
/// FTZ 53 and DNZ 54 as one mode; 3 (FTZ+DNZ) is reserved and its words
/// stay Unknown, so that entry is never read.
const FFMA_FLUSH: [Flush; 4] = [Flush::Keep, Flush::Ftz, Flush::Dnz, Flush::Keep];
fn ffma(r: u64, rhs: Source, addend: Source) -> InstructionKind {
    let flush = FFMA_FLUSH[b(r, 53, 2) as usize];
    InstructionKind::FFma(ops::FFma {
        dst: Register(b(r, 0, 8) as u8),
        lhs: Register(b(r, 8, 8) as u8),
        rhs,
        addend,
        flags: float_flags(r, 47),
        modifiers: FFmaModifiers {
            product_negate: b(r, 48, 1) != 0,
            addend_negate: b(r, 49, 1) != 0,
            saturate: b(r, 50, 1) != 0,
            flush,
        },
    })
}
pub fn decode_word(w: [u32; 2], address: u32) -> Instruction {
    let x = edge::Word::new(w, address);
    families::FAMILY[families::OPCODE[b(x.r, 48, 16) as usize] as usize](x)
}
/// Decode only code discovered from entry. Ryujinx Decoders/Decoder.cs:
/// 73-86,133-162,188-195 follows fallthrough, branch, push, call and resolved
/// indirect targets. Unvisited words are data regardless of their encoding;
/// a visited unsupported instruction remains Unknown and rejects translation.
pub fn decode_stream<R: Run>(run: R, w: &[u32]) -> Vec<Instruction> {
    decode_reachable(run, w, None)
}

// Reuse reachable discovery storage per translator, following Ryujinx
// CodeGen/Spirv/SpirvGenerator.cs:19-28,38-42 without its shared pool lock.
#[derive(Default)]
pub(crate) struct StackScratch {
    /// Each discovered slot and where its instruction stands.
    slots: Slots,
    /// The instructions as decoded, before any table proof rewrites one.
    decoded: Vec<Instruction>,
    /// The things they arrived at the flow station as.
    found: crate::flow::Found,
    /// The program's control flow as the proofs leave it.
    program: crate::flow::Program,
    pending: Vec<u32>,
}

impl StackScratch {
    /// The slots of the program the last discovery decoded.
    pub(crate) fn slots(&self) -> &Slots {
        &self.slots
    }

    /// The control flow of the program the last discovery decoded.
    pub(crate) fn program(&self) -> &crate::flow::Program {
        &self.program
    }
}

/// Where each slot of one program runs: its address and its position in
/// program order. Discovery establishes it with the instructions; every
/// later pass resolves a target slot through it.
#[derive(Default)]
pub(crate) struct Slots(crate::fast_hash::HashMap<u32, usize>);

impl Slots {
    /// The slots of `xs`, a program not built by discovery.
    pub(crate) fn of(xs: &[Instruction]) -> Self {
        let mut slots = Self::default();
        slots.place(xs);
        slots
    }

    /// Every instruction of `xs` stands at its position.
    fn place<'x>(&mut self, xs: impl IntoIterator<Item = &'x Instruction>) {
        self.0.clear();
        self.0
            .extend(xs.into_iter().enumerate().map(|(i, x)| (x.address, i)));
    }

    /// The position of the instruction running `slot`.
    pub(crate) fn get(&self, slot: u32) -> Option<usize> {
        self.0.get(&slot).copied()
    }

    /// Discovery has decoded `slot`.
    fn holds(&self, slot: u32) -> bool {
        self.0.contains_key(&slot)
    }
}

/// Decode the reachable program into `xs` and build its block graph once in
/// `graph`; every later pass borrows that graph.
pub(crate) fn decode_stream_into<'g, R: Run>(
    run: R,
    w: &[u32],
    cb1: Option<&[u32]>,
    xs: &mut Vec<Instruction>,
    scratch: &mut StackScratch,
    graph: &'g mut crate::structure::GraphStore,
) -> Result<&'g crate::structure::ControlFlowGraph, crate::interface::Stopped> {
    decode_reachable_into(run, w, cb1, xs, scratch, graph)
}

/// Discover one program on the run its host chose.
fn decode_reachable<R: Run>(run: R, w: &[u32], cb1: Option<&[u32]>) -> Vec<Instruction> {
    let mut xs = Vec::new();
    let _ = decode_reachable_into(
        run,
        w,
        cb1,
        &mut xs,
        &mut StackScratch::default(),
        &mut crate::structure::GraphStore::default(),
    );
    xs
}

fn decode_reachable_into<'g, R: Run>(
    run: R,
    w: &[u32],
    cb1: Option<&[u32]>,
    xs: &mut Vec<Instruction>,
    scratch: &mut StackScratch,
    graph: &'g mut crate::structure::GraphStore,
) -> Result<&'g crate::structure::ControlFlowGraph, crate::interface::Stopped> {
    let StackScratch {
        slots,
        decoded,
        found,
        program,
        pending,
    } = scratch;
    slots.0.clear();
    decoded.clear();
    found.clear();
    pending.clear();
    pending.push(ENTRY_SLOT);
    loop {
        while let Some(address) = pending.pop() {
            if slots.holds(address) {
                continue;
            }
            let index = address as usize / 4;
            let Some(raw) = w.get(index..index + 2) else {
                // Missing/partial destinations retain CFG and byte-length errors.
                continue;
            };
            // Branch targets already name the slot they run. The decoded
            // instruction arrives at the flow station here, once.
            let x = decode_word([raw[0], raw[1]], address);
            found.arrive(&x, |successor| pending.push(successor.slot()));
            slots.0.insert(address, decoded.len());
            decoded.push(x);
        }
        decoded.sort_unstable_by_key(|x| x.address);
        slots.place(decoded.iter());
        xs.clear();
        xs.extend_from_slice(decoded);
        program.order(found, xs);
        // Ryujinx Decoder.cs:188-195: new indirect targets require another
        // decoding round. Do not scan unreachable bytes for table producers.
        // Table proofs are recomputed from the original register BRX on each
        // discovery round; new entries can invalidate them (Ryujinx
        // Decoder.cs:577-616). Only a BRX the proof rewrote has new targets:
        // the proof made a new instruction, which arrives at the station.
        let known = crate::brx::Graph::of(program, xs, slots);
        let track = crate::brx::resolve_in(
            run,
            xs,
            slots,
            &known,
            cb1.unwrap_or_default(),
            |at, x, table| {
                program.prove(at, x, table, |successor| {
                    let target = successor.slot();
                    if !slots.holds(target)
                        && w.get(target as usize / 4..target as usize / 4 + 2)
                            .is_some()
                    {
                        pending.push(target);
                    }
                });
            },
        );
        if pending.is_empty() && !track.unlearned().is_empty() {
            // Ryujinx InstEmitFlowControl.cs:38-40. A register BRX whose
            // value is not learned may jump to any target, so the discovery
            // track retains every remaining real slot.
            for slot in (8..w.len() * 4).step_by(8) {
                if !slot.is_multiple_of(32) && slot + 8 <= w.len() * 4 && !slots.holds(slot as u32)
                {
                    pending.push(slot as u32);
                }
            }
        }
        if pending.is_empty() {
            resolve_stack_jumps_with_linear_fallback(run, w, cb1, xs, program, slots);
            // The proofs are done: each instruction leaves as the thing it is.
            program.write(xs);
            // Every pop and table is resolved: the block graph is final.
            // BFI and OUT proofs rewrite no control flow, so they borrow it.
            let blocks = graph.build_in(xs, program, slots);
            if let Ok(cfg) = &blocks {
                resolve_bfi_ranges(xs, cfg);
                resolve_out_streams(xs, program, cb1, cfg);
            }
            return blocks;
        }
    }
}

fn resolve_out_streams(
    xs: &mut [Instruction],
    program: &mut crate::flow::Program,
    cb1: Option<&[u32]>,
    cfg: &crate::structure::ControlFlowGraph,
) {
    if !xs.iter().any(|x| {
        matches!(
            x.kind,
            InstructionKind::GeometryOutDynamic(ops::GeometryOutDynamic { .. })
        )
    }) {
        return;
    }
    // Ryujinx InstEmitHelper.cs:56-76 and ConstantFolding.cs:12-15 retain
    // constant operands; Decoder.cs:501-517 separates branch entries. Resolve
    // the constant directly, avoiding SSA and runtime stream dispatch. Never
    // move a fact across a join, predicated/partial write or unknown operation.
    let read = |src: Source, values: &[Option<u32>; 256]| match src {
        Source::Immediate(value) => Some(value),
        Source::Register(Register(255)) => Some(0),
        Source::Register(Register(r)) => values[usize::from(r)],
        Source::ConstantBuffer { binding: 1, offset } => cb1
            .and_then(|words| words.get(usize::from(offset) / 4))
            .copied(),
        _ => None,
    };
    for block in &cfg.blocks {
        let mut values = [None; 256];
        for index in block.instructions.clone() {
            let x = &mut xs[index];
            if let InstructionKind::GeometryOutDynamic(out) = x.kind {
                // OUT register/constant-buffer streams need a constant
                // SPIR-V stream operand (Mesa
                // src/compiler/spirv/spirv_to_nir.c:5383-5388).
                if let Some(stream) = read(out.stream, &values).filter(|&s| s <= 3) {
                    x.kind = InstructionKind::GeometryOut(out.on(stream as u8));
                    // The proof made a new instruction: it arrives at the
                    // station.
                    program.replace(index, x);
                    continue;
                }
                values.fill(None);
            } else if let InstructionKind::Mov(ops::Mov {
                dst,
                src,
                quad_lanes,
            }) = x.kind
            {
                // MOV lane enables: Mesa sm50.rs:1936,1941,1946. An
                // unconditional full-lane write alone establishes a value.
                values[usize::from(dst.0)] = if x.predicate.is_none() && quad_lanes == 15 {
                    read(src, &values)
                } else {
                    None
                };
            } else if x.kind.written_registers().count != 0 {
                // Includes vector and split destinations. Forget all facts
                // rather than risk retaining an overlapping register value.
                values.fill(None);
            }
        }
    }
}

/// Execution enters at word 0, a control word, so it runs slot 0 (Ryujinx
/// Decoder.cs:265-270).
const ENTRY_SLOT: u32 = crate::ir::run_slot(0);

/// Where control goes next: the 8-byte `word` it reads and the `slot` that
/// runs there. A decoded target already names the slot it runs; the physical
/// word that follows in order may be a bundle control word (Mesa
/// sm50.rs:3418-3428), whose following slot runs (Ryujinx Decoder.cs:265-270).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Successor {
    word: u32,
    slot: u32,
}

impl Successor {
    /// A decoded target runs exactly its slot.
    fn target(slot: u32) -> Self {
        Self { word: slot, slot }
    }

    /// The physical word that follows in order.
    fn word(word: u32) -> Self {
        Self {
            word,
            slot: crate::ir::run_slot(word),
        }
    }

    /// The slot that runs.
    fn slot(self) -> u32 {
        self.slot
    }

    /// Whether control reaching here touches byte `word`: a target runs only
    /// its slot, while a physical word is read even when it is a control
    /// word and the slot after it runs.
    fn touches(self, word: u64) -> bool {
        u64::from(self.word) == word || u64::from(self.slot) == word
    }
}

// Discovery walks each thing it decodes (Ryujinx Decoder.cs:133-162,
// 188-195,501-515 includes push targets and conditional fallthrough).
// Computing static destinations here adds no SPIR-V work.

/// What a guard does to a transfer's fallthrough during discovery: any
/// guard field, PT included, may leave the transfer untaken.
pub(crate) trait FallGuard {
    /// The next word follows where this guard can leave the transfer untaken.
    fn falls(self, next: impl FnOnce());
    /// The next word follows where this guard or the CC test `cc` can
    /// leave the transfer untaken.
    fn falls_tested(self, cc: impl CcFall, next: impl FnOnce());
}

impl FallGuard for crate::guard::Unguarded {
    fn falls(self, _: impl FnOnce()) {}
    fn falls_tested(self, cc: impl CcFall, next: impl FnOnce()) {
        cc.falls(next);
    }
}

macro_rules! guard_field_falls {
    ($($guard:ident),*) => {$(
        impl FallGuard for crate::guard::$guard {
            fn falls(self, next: impl FnOnce()) {
                next();
            }
            fn falls_tested(self, _: impl CcFall, next: impl FnOnce()) {
                next();
            }
        }
    )*};
}
guard_field_falls!(Pt, NotPt, When);

/// What a CC test does to an unguarded transfer's fallthrough.
pub(crate) trait CcFall {
    fn falls(self, next: impl FnOnce());
}

/// CC.T and no predicate guarantees a taken transfer
/// (envytools envydis/gm107.c:1944-1945).
impl CcFall for crate::flow::CcT {
    fn falls(self, _: impl FnOnce()) {}
}

impl CcFall for crate::flow::CcTest {
    fn falls(self, next: impl FnOnce()) {
        next();
    }
}

/// The word after `address`, where execution passes on.
fn next_word(address: u32, visit: &mut dyn FnMut(Successor)) {
    if let Some(next) = address.checked_add(8) {
        visit(Successor::word(next));
    }
}

/// Where a transfer at `address` sends discovery.
pub(crate) trait Walk: Sized {
    fn walk(self, address: u32, guard: impl FallGuard, visit: &mut dyn FnMut(Successor));
    /// [`Self::walk`] while a partial final word must stay unreachable.
    fn reach(
        self,
        address: u32,
        guard: impl FallGuard,
        visit: &mut dyn FnMut(Successor),
    ) -> Result<(), crate::interface::Stopped> {
        self.walk(address, guard, visit);
        Ok(())
    }
}

impl<C: crate::flow::Cc> Walk for crate::flow::Jump<C> {
    fn walk(self, address: u32, guard: impl FallGuard, visit: &mut dyn FnMut(Successor)) {
        visit(Successor::target(self.slot()));
        guard.falls_tested(self.cc(), || next_word(address, visit))
    }
}

impl<C: crate::flow::Cc> Walk for crate::flow::Reconverge<C> {
    fn walk(self, address: u32, guard: impl FallGuard, visit: &mut dyn FnMut(Successor)) {
        visit(Successor::target(self.slot()));
        guard.falls_tested(self.cc(), || next_word(address, visit))
    }
}

/// An outside target has no instruction to visit.
impl<C: crate::flow::Cc, Y: crate::flow::Gap> Walk for crate::flow::NoSlotJump<C, Y> {
    fn walk(self, address: u32, guard: impl FallGuard, visit: &mut dyn FnMut(Successor)) {
        guard.falls_tested(self.cc(), || next_word(address, visit))
    }
}

impl Walk for crate::flow::TableJump {
    fn walk(self, address: u32, guard: impl FallGuard, visit: &mut dyn FnMut(Successor)) {
        for target in self.targets() {
            visit(Successor::target(target.slot()));
        }
        guard.falls(|| next_word(address, visit))
    }
}

/// Only destinations inside the code address space are visited.
impl Walk for crate::flow::OutsideTable {
    fn walk(self, address: u32, guard: impl FallGuard, visit: &mut dyn FnMut(Successor)) {
        for target in self.table().slots() {
            visit(Successor::target(target.slot()));
        }
        guard.falls(|| next_word(address, visit))
    }
}

/// Unconditional BRX has no fallthrough, even before target recovery:
/// Ryujinx Decoder.cs:157-162,501-515; Block.cs:157-160. Otherwise an
/// unreachable data word after BRX could poison the table proof.
/// Failed recovery still enumerates all real slots in the fallback.
impl<C: crate::flow::Cc> Walk for crate::flow::AnyJump<C> {
    fn walk(self, address: u32, guard: impl FallGuard, visit: &mut dyn FnMut(Successor)) {
        guard.falls_tested(self.cc(), || next_word(address, visit))
    }
    /// Ryujinx InstEmitFlowControl.cs:38-40: arbitrary register targets
    /// may select the partial slot.
    fn reach(
        self,
        _: u32,
        _: impl FallGuard,
        _: &mut dyn FnMut(Successor),
    ) -> Result<(), crate::interface::Stopped> {
        Err(crate::interface::Stopped::PartialWord)
    }
}

impl<C: crate::flow::Cc> Walk for crate::flow::Exit<C> {
    fn walk(self, address: u32, guard: impl FallGuard, visit: &mut dyn FnMut(Successor)) {
        guard.falls_tested(self.cc(), || next_word(address, visit))
    }
}

impl<C: crate::flow::Cc> Walk for crate::flow::Pop<C> {
    fn walk(self, address: u32, guard: impl FallGuard, visit: &mut dyn FnMut(Successor)) {
        guard.falls_tested(self.cc(), || next_word(address, visit))
    }
}

impl<P: crate::flow::PushOp> Walk for crate::flow::Push<P> {
    fn walk(self, address: u32, _: impl FallGuard, visit: &mut dyn FnMut(Successor)) {
        visit(Successor::target(self.slot()));
        next_word(address, visit);
    }
}

/// A call returns to the next word; Ryujinx Decoder.cs:146-148 also
/// discovers the function it calls.
impl Walk for crate::flow::Call {
    fn walk(self, address: u32, _: impl FallGuard, visit: &mut dyn FnMut(Successor)) {
        visit(Successor::target(self.slot()));
        next_word(address, visit);
    }
}

impl Walk for crate::flow::NoSlotCall {
    fn walk(self, address: u32, _: impl FallGuard, visit: &mut dyn FnMut(Successor)) {
        next_word(address, visit);
    }
}

/// Unresolved BRX/JMX and other unknown transfers have no proven
/// successors. JMX encoding: envytools gm107.c:1951-1952; no resolved
/// JMX IR exists here.
impl Walk for crate::flow::Unknown {
    fn walk(self, _: u32, _: impl FallGuard, _: &mut dyn FnMut(Successor)) {}
}

/// Every other instruction passes on to the next word.
macro_rules! steps {
    ($($t:ty),*) => {$(
        impl Walk for $t {
            fn walk(self, address: u32, _: impl FallGuard, visit: &mut dyn FnMut(Successor)) {
                next_word(address, visit);
            }
        }
    )*};
}
impl<P: crate::flow::PushOp> Walk for crate::flow::ResolvedPush<P> {
    fn walk(self, address: u32, _: impl FallGuard, visit: &mut dyn FnMut(Successor)) {
        next_word(address, visit);
    }
}

steps!(
    crate::flow::NoSlotPush,
    crate::flow::Step,
    crate::flow::PredicateWrite
);

/// Check a partial final word only when reachable control flow needs it.
/// Ryujinx Decoders/Decoder.cs:73-86,133-162,188-195 discovers branch, push
/// and resolved BRX destinations separately; :332-338 stops each block at
/// its terminator and :501-515 retains conditional fallthrough. Complete
/// reachable slots stay in the IR, including Unknown.
#[cfg(test)]
fn validate_reachable_words(
    xs: &[Instruction],
    code_len: usize,
) -> Result<(), crate::interface::Stopped> {
    if code_len.is_multiple_of(8) {
        return Ok(());
    }
    partial_word_reached(
        &crate::flow::Program::of(xs),
        &Slots::of(xs),
        (code_len - code_len % 8) as u32,
    )
}

/// Whether control, over the program and slots discovery established,
/// reaches the partial word at `partial_address`. Mesa
/// src/nouveau/compiler/nak/sm50.rs:3418-3428: a control word and three
/// instruction slots are each 8 bytes; the CFG validation still checks
/// missing targets.
pub(crate) fn partial_word_reached(
    program: &crate::flow::Program,
    slots: &Slots,
    partial_address: u32,
) -> Result<(), crate::interface::Stopped> {
    let line = program.line();
    let mut visited = vec![false; program.len()];
    // Execution enters at word 0.
    let mut pending = vec![Successor::word(0)];
    while let Some(successor) = pending.pop() {
        // Check both an incoming word and the slot it runs, so a reachable
        // partial control word cannot hide an incomplete stream.
        if successor.touches(partial_address as u64) {
            return Err(crate::interface::Stopped::PartialWord);
        }
        let address = successor.slot();
        let Some(index) = slots.get(address) else {
            // Absent and misaligned targets retain the CFG's existing error.
            continue;
        };
        if std::mem::replace(&mut visited[index], true) {
            continue;
        }
        line.at(index)
            .reach(address, &mut |successor| pending.push(successor))?;
    }
    Ok(())
}

// BFI range proof: Ryujinx InstEmitBitfield.cs:154-162 and
// CodeGen/Spirv/Instructions.cs:2155-2171. Fold proven constants, reject
// proven invalid ranges, and preserve runtime controls for the emitter.
fn resolve_bfi_ranges(xs: &mut [Instruction], cfg: &crate::structure::ControlFlowGraph) {
    if !xs.iter().any(|x| {
        matches!(
            x.kind,
            InstructionKind::BitFieldInsert(ops::BitFieldInsert {
                range: Source::Register(_),
                ..
            })
        )
    }) {
        return;
    }
    let reaching = crate::reaching::ReachingDefinitions::analyze(xs, cfg);
    for (index, x) in xs.iter_mut().enumerate() {
        if let InstructionKind::BitFieldInsert(
            insert @ ops::BitFieldInsert {
                range: Source::Register(register),
                ..
            },
        ) = x.kind
        {
            match reaching.constant(index, register.0) {
                Some(control) if (control & 255) + ((control >> 8) & 255) <= 32 => {
                    x.kind = InstructionKind::BitFieldInsert(ops::BitFieldInsert {
                        range: Source::Immediate(control),
                        ..insert
                    });
                }
                Some(control) => {
                    x.kind =
                        InstructionKind::OverlongInsert(ops::OverlongInsert { insert, control });
                }
                None => {}
            }
        }
    }
}

/// Main 4d4dbc2 decoded every complete instruction slot before its ordered
/// stack pass (decoder.rs:1749-1787). Keep that input independently of reachable
/// discovery: omitting a slot can change the older pass's fallthrough paths.
/// Bundle layout: Mesa sm50.rs:3418-3428; control-word targets: Ryujinx
/// Decoders/Decoder.cs:265-270. This temporary IR is never emitted.
#[cfg(test)]
fn decode_linear_stack_reference(w: &[u32], cb1: Option<&[u32]>) -> Vec<Instruction> {
    linear_stack_reference(crate::run::QuietRun, w, cb1).xs
}

/// The linear reference: its instructions, the things they arrived as, and
/// the slots that place them.
struct Linear {
    xs: Vec<Instruction>,
    program: Program,
}

/// [`decode_linear_stack_reference`] with the program and slots of its own.
fn linear_stack_reference<R: Run>(run: R, w: &[u32], cb1: Option<&[u32]>) -> Linear {
    let mut xs = Vec::new();
    let mut i = 0;
    while i + 2 <= w.len() {
        i += 2;
        for _ in 0..3 {
            if i + 2 > w.len() {
                break;
            }
            xs.push(decode_word([w[i], w[i + 1]], i as u32 * 4));
            i += 2;
        }
    }
    // The reference's own decode edge: each instruction arrives once.
    let mut program = Program::of(&xs);
    let slots = Slots::of(&xs);
    resolve_stack_jumps_linear_in(run, &xs, &mut program, &slots);
    // The table proof reads the instructions as the stack proof left them.
    program.write(&mut xs);
    for cb1 in cb1.into_iter() {
        // Match main's two rounds: ordinary pops first, then resolved BRX
        // edges (main decoder.rs:1956-2022). Table semantics: Ryujinx
        // Decoders/Decoder.cs:527-552,561-613. A rewritten BRX is a new
        // instruction: it arrives at the station.
        let known = crate::brx::Graph::of(&program, &xs, &slots);
        crate::brx::resolve_in(run, &mut xs, &slots, &known, cb1, |at, x, table| {
            program.prove(at, x, table, |_| {});
        });
        resolve_stack_jumps_linear_in(run, &xs, &mut program, &slots);
        program.write(&mut xs);
    }
    Linear { xs, program }
}

/// Preserve main's proven pop destinations when reachable discovery or frame
/// merging loses a proof. Unique targets become branches, following Ryujinx
/// Instructions/InstEmitFlowControl.cs:254-275; no unknown instruction or real
/// operation is substituted, and the independent input is never emitted.
fn resolve_stack_jumps_with_linear_fallback<R: Run>(
    run: R,
    w: &[u32],
    cb1: Option<&[u32]>,
    xs: &[Instruction],
    program: &mut Program,
    slots: &Slots,
) {
    let pops = program.pops();
    resolve_stack_jumps_in(run, xs, program, slots);
    if pops == 0 {
        return;
    }
    let linear = linear_stack_reference(run, w, cb1);
    merge_linear_stack_results_in(run, xs, program, &linear);
}

#[cfg(test)]
fn merge_linear_stack_results(xs: &mut [Instruction], linear: &[Instruction]) {
    let mut program = Program::of(xs);
    let linear = Linear {
        xs: linear.to_vec(),
        program: Program::of(linear),
    };
    merge_linear_stack_results_in(crate::run::QuietRun, xs, &mut program, &linear);
    program.write(xs);
}

/// Carry the linear reference's proofs at each address both programs hold.
fn merge_linear_stack_results_in<R: Run>(
    run: R,
    xs: &[Instruction],
    program: &mut Program,
    linear: &Linear,
) {
    // A recovered pop can consume pushes that the reachable pass did not
    // prove. Carry main's resolved flags, retaining each push and its target.
    // Runtime tag writes only serve consumers (Ryujinx InstEmitFlowControl.cs:
    // 239-251); recovered consumers branch to the original push destinations.
    for (at, fallback) in program.carry(xs, &linear.program, &linear.xs) {
        match fallback {
            crate::flow::Fallback::Resolves(kind) => run.fallback_resolves(&xs[at], *kind),
            crate::flow::Fallback::Leaves => run.fallback_leaves(&xs[at]),
        }
    }
}

/// Compatibility reference from main a6d79923 decoder.rs:2583-2786.
/// Retain both the block proof and the ordered-stack proof with main's limits.
/// Propagate the actual ordered push stack along execution paths. Ryujinx
/// Decoders/Decoder.cs:800-810,815-869 pushes SSY/PBK/PCNT and pops through
/// intervening kinds until the matching kind is found. Only a unique target
/// becomes BRA (Instructions/InstEmitFlowControl.cs:254-275). Computing that
/// target here avoids runtime tags and extra SPIR-V comparisons.
#[cfg(test)]
fn resolve_stack_jumps_linear(xs: &mut [Instruction]) {
    let (mut program, slots) = (Program::of(xs), Slots::of(xs));
    resolve_stack_jumps_linear_in(crate::run::QuietRun, xs, &mut program, &slots);
    program.write(xs);
}

/// [`resolve_stack_jumps_linear`] over the program's things and slots.
fn resolve_stack_jumps_linear_in<R: Run>(
    run: R,
    xs: &[Instruction],
    program: &mut Program,
    slots: &Slots,
) {
    if program.pops() == 0 {
        return;
    }
    // Both proofs see the original pushes and pops. Mutating first would
    // erase the pop transition before the second analysis can propagate it.
    let block_resolution = crate::stack::resolve_in(run, program, xs, slots);
    // Refine only when the shared block proof cannot resolve all pops.
    if block_resolution
        .as_ref()
        .is_none_or(|resolution| !resolution.covers_pops())
    {
        resolve_stack_jumps_linear_ordered_traced_in(xs, program, slots, run.stack());
    }
    for resolution in block_resolution.into_iter() {
        resolution.apply(program, xs);
    }
}

// PT is immutable true: Ryujinx IntermediateRepresentation/OperandHelper.cs:49-51.
fn initial_stack_predicates() -> [Option<bool>; 8] {
    let mut predicates = [None; 8];
    predicates[7] = Some(true);
    predicates
}

// These are edge proofs, not instruction removals. No predicate fact survives
// a possible writer (InstructionKind::may_write_predicate). Predication and
// inversion: Ryujinx Translation/Translator.cs:339-383; Mesa sm50.rs:225-235.
pub(crate) fn trace_predicate_pruning(
    output: &mut Option<&mut dyn std::io::Write>,
    x: &Instruction,
    value: bool,
) {
    if let (Some(output), Some(p)) = (output, x.predicate) {
        write_pruning(output, x.address, p, value);
    }
}

/// The guard `p` at byte `address` is proven `value`: one of its two
/// histories (skipped when `value` disagrees with the guard) is excluded.
fn write_pruning(mut output: impl std::io::Write, address: u32, p: Predicate, value: bool) {
    let edge = ["executed", "skipped"][usize::from(value != p.inverted)];
    // Diagnostics must not change translation, even if the output is closed.
    let _ = writeln!(output, "SHADOWBOX_TRACE_STACK: prune {edge} history at byte 0x{address:x}: predicate P{} proven {value}, inversion {}; no intervening predicate write", p.register, p.inverted);
}

/// Where stack-proof diagnostics go. `Quiet` formats nothing; `Traced`
/// records each exploration event and reports every unresolved pop.
pub(crate) trait StackTrace {
    /// Exploration reached `event` at instruction `at`.
    fn event(&mut self, at: usize, event: impl FnOnce() -> String);
    /// The guard `p` of `x` is proven `value`, excluding one history.
    fn prune(&mut self, x: &Instruction, p: Predicate, value: bool);
    /// Exploration ended with these targets, underflows and abort reason.
    fn report(
        self,
        program: &Program,
        xs: &[Instruction],
        slots: &Slots,
        targets: &[std::collections::HashSet<usize>],
        invalid: &[bool],
        aborted: Option<&str>,
    );
}

/// No diagnostics were requested.
pub(crate) struct Quiet;

impl StackTrace for Quiet {
    fn event(&mut self, _: usize, _: impl FnOnce() -> String) {}
    fn prune(&mut self, _: &Instruction, _: Predicate, _: bool) {}
    fn report(
        self,
        _: &Program,
        _: &[Instruction],
        _: &Slots,
        _: &[std::collections::HashSet<usize>],
        _: &[bool],
        _: Option<&str>,
    ) {
    }
}

/// Opt-in diagnostics over the existing IR, with no additional decoding.
pub(crate) struct Traced<W> {
    output: W,
    events: Vec<(usize, String)>,
}

impl<W: std::io::Write> Traced<W> {
    pub(crate) fn new(output: W) -> Self {
        Self {
            output,
            events: Vec::new(),
        }
    }
}

impl<W: std::io::Write> StackTrace for Traced<W> {
    fn event(&mut self, at: usize, event: impl FnOnce() -> String) {
        self.events.push((at, event()));
    }
    fn prune(&mut self, x: &Instruction, p: Predicate, value: bool) {
        write_pruning(&mut self.output, x.address, p, value);
    }
    fn report(
        mut self,
        program: &Program,
        xs: &[Instruction],
        slots: &Slots,
        targets: &[std::collections::HashSet<usize>],
        invalid: &[bool],
        aborted: Option<&str>,
    ) {
        // Diagnostics must not change translation, even if stderr is closed.
        let _ = write_stack_report(
            &self.events,
            program,
            xs,
            slots,
            targets,
            invalid,
            aborted,
            &mut self.output,
        );
    }
}

/// Why an ordered stack exploration stopped before its fixed point: a
/// target outside the program, an instruction that may hide control flow,
/// or exhausted work. It publishes no proof.
pub(crate) struct Abort;

/// One explored path, standing at one instruction: what each thing does
/// to it. The thing calls these; the explorer holds the path's stack and
/// its known guards.
pub(crate) trait Explore {
    /// The guard `p` forks the path: its skipped history continues to the
    /// next instruction, and `then` explores its executed history.
    fn guard(
        &mut self,
        p: Predicate,
        then: &mut dyn FnMut(&mut dyn Explore) -> Result<(), Abort>,
    ) -> Result<(), Abort>;
    /// The path continues to the next instruction.
    fn next(&mut self);
    /// The instruction may write a predicate: every known guard is lost.
    fn forget(&mut self);
    /// The path pushes `target` on `stack`.
    fn push(&mut self, stack: TargetStack, target: BranchTarget) -> Result<(), Abort>;
    /// A push whose target lies outside the code address space.
    fn outside_push(&mut self, stack: TargetStack) -> Abort;
    /// A branch whose target lies outside the code address space.
    fn outside_branch(&mut self) -> Abort;
    /// A table with an entry outside the code address space.
    fn outside_table(&mut self, table: &crate::flow::OutsideTable) -> Result<(), Abort>;
    /// The path pops `stack`.
    fn pop(&mut self, stack: TargetStack) -> Result<(), Abort>;
    /// The path branches to `slot`.
    fn jump(&mut self, slot: u32) -> Result<(), Abort>;
    /// The path branches to each entry of `table`.
    fn table(&mut self, table: &crate::flow::TableJump) -> Result<(), Abort>;
    /// The path may branch to any instruction.
    fn any(&mut self);
    /// An undecoded instruction could hide an edge or stack operation.
    fn unknown(&mut self) -> Abort;
}

/// What a guard does to an explored path. A skipped pop preserves its
/// stack. Predication semantics: Ryujinx Translation/Translator.cs:359-393;
/// predicate/inversion encoding: Mesa sm50.rs:225-235. Only control flow
/// and stack operations fork on their guard: a guarded ALU or memory
/// instruction never changes the stack or the next instruction.
pub(crate) trait ExploreGuard: Copy {
    fn fork(
        self,
        explore: &mut dyn Explore,
        then: &mut dyn FnMut(&mut dyn Explore) -> Result<(), Abort>,
    ) -> Result<(), Abort>;
}

impl ExploreGuard for crate::guard::Unguarded {
    fn fork(
        self,
        explore: &mut dyn Explore,
        then: &mut dyn FnMut(&mut dyn Explore) -> Result<(), Abort>,
    ) -> Result<(), Abort> {
        then(explore)
    }
}

impl<G: crate::guard::Field> ExploreGuard for G {
    fn fork(
        self,
        explore: &mut dyn Explore,
        then: &mut dyn FnMut(&mut dyn Explore) -> Result<(), Abort>,
    ) -> Result<(), Abort> {
        explore.guard(self.predicate(), then)
    }
}

/// What a CC test does to an executed transfer's path. Ryujinx
/// InstEmitFlowControl.cs:295-312: CC and predicate are conjoined; a failed
/// CC continues to the next instruction (envytools gm107.c:1944-1945).
pub(crate) trait CcExplore {
    fn tested(self, explore: &mut dyn Explore);
}

impl CcExplore for crate::flow::CcT {
    fn tested(self, _: &mut dyn Explore) {}
}

impl CcExplore for crate::flow::CcTest {
    fn tested(self, explore: &mut dyn Explore) {
        explore.next();
    }
}

/// What a transfer does to an explored path.
pub(crate) trait Explores: Copy {
    fn explore<G: crate::guard::Guard>(
        self,
        guard: G,
        explore: &mut dyn Explore,
    ) -> Result<(), Abort>;
}

impl<C: crate::flow::Cc> Explores for crate::flow::Jump<C> {
    fn explore<G: crate::guard::Guard>(self, guard: G, e: &mut dyn Explore) -> Result<(), Abort> {
        guard.fork(e, &mut |e| {
            e.jump(self.slot())?;
            self.cc().tested(e);
            Ok(())
        })
    }
}

impl<C: crate::flow::Cc> Explores for crate::flow::Reconverge<C> {
    fn explore<G: crate::guard::Guard>(self, guard: G, e: &mut dyn Explore) -> Result<(), Abort> {
        guard.fork(e, &mut |e| {
            self.cc().tested(e);
            e.pop(self.stack())
        })
    }
}

impl<C: crate::flow::Cc> Explores for crate::flow::Pop<C> {
    fn explore<G: crate::guard::Guard>(self, guard: G, e: &mut dyn Explore) -> Result<(), Abort> {
        guard.fork(e, &mut |e| {
            self.cc().tested(e);
            e.pop(self.stack())
        })
    }
}

impl<C: crate::flow::Cc> Explores for crate::flow::Exit<C> {
    fn explore<G: crate::guard::Guard>(self, guard: G, e: &mut dyn Explore) -> Result<(), Abort> {
        guard.fork(e, &mut |e| {
            self.cc().tested(e);
            Ok(())
        })
    }
}

impl<C: crate::flow::Cc, Y: crate::flow::Gap> Explores for crate::flow::NoSlotJump<C, Y> {
    fn explore<G: crate::guard::Guard>(self, guard: G, e: &mut dyn Explore) -> Result<(), Abort> {
        guard.fork(e, &mut |e| Err(e.outside_branch()))
    }
}

impl Explores for crate::flow::TableJump {
    fn explore<G: crate::guard::Guard>(self, guard: G, e: &mut dyn Explore) -> Result<(), Abort> {
        guard.fork(e, &mut |e| e.table(&self))
    }
}

impl Explores for crate::flow::OutsideTable {
    fn explore<G: crate::guard::Guard>(self, guard: G, e: &mut dyn Explore) -> Result<(), Abort> {
        guard.fork(e, &mut |e| e.outside_table(&self))
    }
}

impl<C: crate::flow::Cc> Explores for crate::flow::AnyJump<C> {
    fn explore<G: crate::guard::Guard>(self, guard: G, e: &mut dyn Explore) -> Result<(), Abort> {
        guard.fork(e, &mut |e| {
            self.cc().tested(e);
            e.any();
            Ok(())
        })
    }
}

impl<P: crate::flow::PushOp> Explores for crate::flow::Push<P> {
    fn explore<G: crate::guard::Guard>(self, guard: G, e: &mut dyn Explore) -> Result<(), Abort> {
        guard.fork(e, &mut |e| {
            e.push(self.stack(), self.target())?;
            e.next();
            Ok(())
        })
    }
}

impl<P: crate::flow::PushOp> Explores for crate::flow::ResolvedPush<P> {
    fn explore<G: crate::guard::Guard>(self, guard: G, e: &mut dyn Explore) -> Result<(), Abort> {
        guard.fork(e, &mut |e| {
            e.push(self.stack(), self.target())?;
            e.next();
            Ok(())
        })
    }
}

impl Explores for crate::flow::NoSlotPush {
    fn explore<G: crate::guard::Guard>(self, guard: G, e: &mut dyn Explore) -> Result<(), Abort> {
        guard.fork(e, &mut |e| Err(e.outside_push(self.stack())))
    }
}

/// Unknown is not control flow the explorer knows: its guard does not fork.
impl Explores for crate::flow::Unknown {
    fn explore<G: crate::guard::Guard>(self, _: G, e: &mut dyn Explore) -> Result<(), Abort> {
        Err(e.unknown())
    }
}

/// A call is not control flow the explorer models.
impl Explores for crate::flow::Call {
    fn explore<G: crate::guard::Guard>(self, _: G, e: &mut dyn Explore) -> Result<(), Abort> {
        Err(e.unknown())
    }
}

impl Explores for crate::flow::NoSlotCall {
    fn explore<G: crate::guard::Guard>(self, _: G, e: &mut dyn Explore) -> Result<(), Abort> {
        Err(e.unknown())
    }
}

impl Explores for crate::flow::Step {
    fn explore<G: crate::guard::Guard>(self, _: G, e: &mut dyn Explore) -> Result<(), Abort> {
        e.next();
        Ok(())
    }
}

/// Predicate writes conservatively invalidate the known guards.
impl Explores for crate::flow::PredicateWrite {
    fn explore<G: crate::guard::Guard>(self, _: G, e: &mut dyn Explore) -> Result<(), Abort> {
        e.forget();
        e.next();
        Ok(())
    }
}

/// A push on an explored stack: where it is, the stack it uses and the
/// target it pushes. Its position names it; the rest follows from it.
#[derive(Clone, Copy)]
struct Pushed {
    at: usize,
    stack: TargetStack,
    target: BranchTarget,
}

impl PartialEq for Pushed {
    fn eq(&self, other: &Self) -> bool {
        self.at == other.at
    }
}

impl Eq for Pushed {}

impl std::hash::Hash for Pushed {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.at.hash(state);
    }
}

#[cfg(test)]
fn resolve_stack_jumps_linear_ordered(xs: &mut [Instruction]) {
    let (mut program, slots) = (Program::of(xs), Slots::of(xs));
    resolve_stack_jumps_linear_ordered_traced_in(xs, &mut program, &slots, Quiet);
    program.write(xs);
}

#[cfg(test)]
fn resolve_stack_jumps_linear_ordered_traced(xs: &mut [Instruction], trace: impl StackTrace) {
    let (mut program, slots) = (Program::of(xs), Slots::of(xs));
    resolve_stack_jumps_linear_ordered_traced_in(xs, &mut program, &slots, trace);
    program.write(xs);
}

/// The ordered exploration: each path carries its complete push stack.
struct Ordered<'a, T> {
    xs: &'a [Instruction],
    slots: &'a Slots,
    trace: T,
    pending: Vec<(usize, Vec<Pushed>, [Option<bool>; 8])>,
    at: usize,
    stack: Vec<Pushed>,
    predicates: [Option<bool>; 8],
    targets: Vec<std::collections::HashSet<usize>>,
    consumed: std::collections::HashSet<usize>,
    invalid: Vec<bool>,
}

impl<T: StackTrace> Explore for Ordered<'_, T> {
    fn guard(
        &mut self,
        p: Predicate,
        then: &mut dyn FnMut(&mut dyn Explore) -> Result<(), Abort>,
    ) -> Result<(), Abort> {
        let register = usize::from(p.register);
        let execute = !p.inverted;
        for value in self.predicates[register].into_iter() {
            self.trace.prune(&self.xs[self.at], p, value);
        }
        match self.predicates[register] {
            Some(value) if value != execute => {
                self.next();
                Ok(())
            }
            None => {
                let mut skipped = self.predicates;
                skipped[register] = Some(!execute);
                self.pending
                    .push((self.at + 1, self.stack.clone(), skipped));
                self.predicates[register] = Some(execute);
                then(self)
            }
            _ => then(self),
        }
    }
    fn next(&mut self) {
        self.pending
            .push((self.at + 1, self.stack.clone(), self.predicates));
    }
    fn forget(&mut self) {
        self.predicates = initial_stack_predicates();
    }
    fn push(&mut self, stack: TargetStack, target: BranchTarget) -> Result<(), Abort> {
        self.slots.get(target.slot()).ok_or(Abort)?;
        self.stack.push(Pushed {
            at: self.at,
            stack,
            target,
        });
        Ok(())
    }
    fn outside_push(&mut self, _: TargetStack) -> Abort {
        Abort
    }
    fn outside_branch(&mut self) -> Abort {
        Abort
    }
    fn outside_table(&mut self, _: &crate::flow::OutsideTable) -> Result<(), Abort> {
        Err(Abort)
    }
    fn pop(&mut self, kind: TargetStack) -> Result<(), Abort> {
        let Some(position) = self.stack.iter().rposition(|push| push.stack == kind) else {
            self.invalid[self.at] = true;
            return Ok(());
        };
        let target = self
            .slots
            .get(self.stack[position].target.slot())
            .ok_or(Abort)?;
        self.targets[self.at].insert(target);
        self.consumed
            .extend(self.stack.drain(position..).map(|push| push.at));
        self.pending
            .push((target, std::mem::take(&mut self.stack), self.predicates));
        Ok(())
    }
    fn jump(&mut self, slot: u32) -> Result<(), Abort> {
        let target = self.slots.get(slot).ok_or(Abort)?;
        self.pending
            .push((target, self.stack.clone(), self.predicates));
        Ok(())
    }
    fn table(&mut self, table: &crate::flow::TableJump) -> Result<(), Abort> {
        for target in table.targets() {
            let target = self.slots.get(target.slot()).ok_or(Abort)?;
            self.pending
                .push((target, self.stack.clone(), self.predicates));
        }
        Ok(())
    }
    fn any(&mut self) {
        // A branch preserves the stack, unlike Unknown. Runtime targets:
        // Ryujinx InstEmitFlowControl.cs:38-40. Include every supplied
        // executable slot, matching the address dispatcher.
        for target in 0..self.xs.len() {
            self.pending
                .push((target, self.stack.clone(), self.predicates));
        }
    }
    fn unknown(&mut self) -> Abort {
        // Re-run after cbuf BRX resolution, when all edges are available.
        Abort
    }
}

impl<T: StackTrace> Ordered<'_, T> {
    /// Explores every path from the entry, or aborts.
    fn run(mut self, program: &Program) -> Result<Self, Abort> {
        use std::collections::HashSet;
        let line = program.line();
        let mut visited = HashSet::new();
        let mut remaining = self.xs.len().saturating_mul(256);
        while let Some((at, stack, predicates)) = self.pending.pop() {
            // Account for copying/hashing the complete stack and pop searches,
            // not just the number of visited states. This keeps the predicate
            // refinement bounded even for loops whose stack grows each visit.
            remaining = remaining
                .checked_sub(stack.len().saturating_mul(8) + 32)
                .ok_or(Abort)?;
            if at == self.xs.len() || !visited.insert((at, stack.clone(), predicates)) {
                continue;
            }
            self.at = at;
            self.stack = stack;
            self.predicates = predicates;
            line.at(at).explore(&mut self)?;
        }
        Ok(self)
    }
}

fn resolve_stack_jumps_linear_ordered_traced_in(
    xs: &[Instruction],
    program: &mut Program,
    slots: &Slots,
    trace: impl StackTrace,
) {
    use std::collections::HashSet;
    if program.pops() == 0 {
        return;
    }
    let explored = Ordered {
        xs,
        slots,
        trace,
        pending: vec![(0, Vec::new(), initial_stack_predicates())],
        at: 0,
        stack: Vec::new(),
        predicates: initial_stack_predicates(),
        targets: vec![HashSet::new(); xs.len()],
        consumed: HashSet::new(),
        invalid: vec![false; xs.len()],
    }
    .run(program);
    let Ok(Ordered {
        targets,
        consumed,
        invalid,
        ..
    }) = explored
    else {
        return;
    };
    program.lower(&reached(&targets, &invalid), xs);
    program.settle(&crate::flow::Consumed(consumed.into_iter().collect()));
}

/// The pops an exploration reached exactly one destination with, on no
/// path that underflows.
fn reached(targets: &[std::collections::HashSet<usize>], invalid: &[bool]) -> crate::flow::Reached {
    crate::flow::Reached(
        targets
            .iter()
            .zip(invalid)
            .enumerate()
            .filter(|(_, (destinations, &invalid))| destinations.len() == 1 && !invalid)
            .flat_map(|(at, (destinations, _))| destinations.iter().map(move |&to| (at, to)))
            .collect(),
    )
}

/// Report every unresolved pop with the exploration `events` that reached it.
/// Block boundaries follow Ryujinx Decoders/Decoder.cs:133-162,332-338:
/// branch/push destinations and the instruction after a terminator.
#[allow(clippy::too_many_arguments)]
fn write_stack_report(
    events: &[(usize, String)],
    program: &Program,
    xs: &[Instruction],
    slots: &Slots,
    targets: &[std::collections::HashSet<usize>],
    invalid: &[bool],
    aborted: Option<&str>,
    mut output: impl std::io::Write,
) -> std::io::Result<()> {
    use std::collections::BTreeSet;
    let line = program.line();
    let mut starts = BTreeSet::from([0]);
    for (i, x) in xs.iter().enumerate() {
        line.at(i).walk(x.address, &mut |successor| {
            if let Some(target) = slots.get(successor.slot()) {
                // Ordinary fallthrough does not start a new block.
                if target != i + 1 {
                    starts.insert(target);
                }
            }
        });
        // Include adjacent explicit targets, and the instruction after a
        // terminator, as boundaries as well.
        line.at(i).starts(slots, &mut |target| {
            starts.insert(target);
        });
        if i + 1 < xs.len() && xs[i + 1].address > x.address.saturating_add(16) {
            starts.insert(i + 1);
        }
    }
    starts.insert(xs.len());
    let starts = starts.into_iter().collect::<Vec<_>>();
    let mut blocks = vec![(0, 0); xs.len()];
    for pair in starts.windows(2) {
        let range = (xs[pair[0]].address, u64::from(xs[pair[1] - 1].address) + 8);
        blocks[pair[0]..pair[1]].fill(range);
    }
    for (at, stack) in program.waiting() {
        let x = xs[at];
        if aborted.is_none() && targets[at].len() == 1 && !invalid[at] {
            continue;
        }
        // Names/ordered matching: Ryujinx Decoder.cs:824-850;
        // opcodes: Mesa sm50.rs:3050,3061,3084 (SYNC/BRK/CONT).
        let name = match stack {
            TargetStack::Sync => "SYNC",
            TargetStack::Break => "BRK",
            TargetStack::Continue => "CONT",
        };
        writeln!(
            output,
            "SHADOWBOX_TRACE_STACK: unresolved {name} ({stack:?}) at byte 0x{:x}, predicate {:?}",
            x.address, x.predicate
        )?;
        if let Some(reason) = aborted {
            writeln!(
                output,
                "  resolution failed: exploration aborted: {reason}; all proofs discarded"
            )?;
        } else {
            if invalid[at] {
                writeln!(
                    output,
                    "  resolution failed: possible stack underflow / no matching push"
                )?;
            }
            if targets[at].is_empty() {
                writeln!(output, "  resolution failed: no matching target found (pop may not be executed on any examined path)")?;
            } else if targets[at].len() > 1 {
                writeln!(output, "  resolution failed: multiple matching targets")?;
            }
        }
        let destinations = targets[at]
            .iter()
            .map(|&target| xs[target].address)
            .collect::<BTreeSet<_>>();
        writeln!(output, "  candidate target byte offsets: {destinations:x?}")?;
        writeln!(output, "  paths examined (shared exploration, block ends exclusive; frames link possible older pushes):")?;
        for (at, event) in events {
            let (start, end) = blocks[*at];
            writeln!(
                output,
                "    block 0x{start:x}..0x{end:x}, instruction 0x{:x}: {event}",
                xs[*at].address
            )?;
        }
    }
    Ok(())
}

/// Resolve consumers using a finite graph of possible stack predecessors.
/// Ryujinx Decoders/Decoder.cs:753-810 bounds push discovery by visited blocks;
/// :824-869 searches ordered pushes for the matching pop kind. Its emitter
/// writes push tags only for consumers (InstEmitFlowControl.cs:239-251), and
/// :254-275 folds a unique pop target to a branch. Retaining static push sites
/// and their predecessor links avoids enumerating unbounded loop histories.
fn resolve_stack_jumps_finite<R: Run>(
    run: R,
    xs: &[Instruction],
    program: &mut Program,
    slots: &Slots,
) {
    resolve_stack_jumps_traced_in(xs, program, slots, run.stack());
}

#[cfg(test)]
fn resolve_stack_jumps(xs: &mut [Instruction]) {
    let (mut program, slots) = (Program::of(xs), Slots::of(xs));
    resolve_stack_jumps_in(crate::run::QuietRun, xs, &mut program, &slots);
    program.write(xs);
}

/// Resolve every pop of `xs`, as the things of `program`, placed by `slots`.
fn resolve_stack_jumps_in<R: Run>(
    run: R,
    xs: &[Instruction],
    program: &mut Program,
    slots: &Slots,
) {
    // Both proofs see original pops. The finite pass also proves unused pushes
    // have no consumers (Ryujinx InstEmitFlowControl.cs:239-251).
    let block_resolution = crate::stack::resolve_in(run, program, xs, slots);
    if block_resolution.as_ref().is_none_or(|r| !r.covers_pops()) || program.pushes() != 0 {
        resolve_stack_jumps_finite(run, xs, program, slots);
    }
    for resolution in block_resolution.into_iter() {
        resolution.apply(program, xs);
    }
}

#[cfg(test)]
fn resolve_stack_jumps_traced(xs: &mut [Instruction], trace: impl StackTrace) {
    let (mut program, slots) = (Program::of(xs), Slots::of(xs));
    resolve_stack_jumps_traced_in(xs, &mut program, &slots, trace);
    program.write(xs);
}

/// A path state of the finite exploration: the instruction, the frame
/// heading its stack, and its known guards.
type FiniteState = (usize, usize, [Option<bool>; 8]);

/// The finite exploration. Frame zero is the empty stack. A frame
/// identifies a static push and the known guards at that push, with edges
/// to every possible older frame. Cycles represent repeated pushes without
/// allocating additional depth. Merging predecessor links can only add
/// possible histories. Ambiguous targets and all possible underflows stay
/// unresolved. An empty search (Ryujinx Decoder.cs:832-835) supplies no
/// hardware reachability proof.
struct Finite<'a, T> {
    xs: &'a [Instruction],
    slots: &'a Slots,
    trace: T,
    aborted: Option<String>,
    frames: Vec<Option<Pushed>>,
    parents: Vec<std::collections::HashSet<usize>>,
    frame_ids: std::collections::HashMap<(usize, [Option<bool>; 8]), usize>,
    dependents: Vec<std::collections::HashSet<FiniteState>>,
    pending: Vec<FiniteState>,
    visited: std::collections::HashSet<FiniteState>,
    targets: Vec<std::collections::HashSet<usize>>,
    invalid: Vec<bool>,
    at: usize,
    frame: usize,
    predicates: [Option<bool>; 8],
}

impl<T: StackTrace> Finite<'_, T> {
    fn enqueue(&mut self, state: FiniteState) {
        let xs = self.xs;
        self.trace.event(self.at, || {
            format!(
                "queue instruction byte {:?}, frame {}, predicates {:?}",
                xs.get(state.0).map(|x| format!("0x{:x}", x.address)),
                state.1,
                state.2
            )
        });
        self.pending.push(state);
    }

    /// Exploration ends without a proof, for `reason`.
    fn abort(&mut self, reason: String) -> Abort {
        self.aborted = Some(reason);
        Abort
    }

    fn address(&self) -> u32 {
        self.xs[self.at].address
    }

    /// Explores every path from the entry, until a path aborts.
    fn run(&mut self, program: &Program) {
        let line = program.line();
        while let Some((at, frame, predicates)) = self.pending.pop() {
            if at == self.xs.len() || !self.visited.insert((at, frame, predicates)) {
                continue;
            }
            let xs = self.xs;
            self.trace.event(at, || {
                format!(
                    "visit {:?}, frame {frame}, predicates {predicates:?}",
                    xs[at].kind
                )
            });
            self.at = at;
            self.frame = frame;
            self.predicates = predicates;
            if let Err(Abort) = line.at(at).explore(self) {
                return;
            }
        }
    }
}

impl<T: StackTrace> Explore for Finite<'_, T> {
    fn guard(
        &mut self,
        p: Predicate,
        then: &mut dyn FnMut(&mut dyn Explore) -> Result<(), Abort>,
    ) -> Result<(), Abort> {
        let register = usize::from(p.register);
        let execute = !p.inverted;
        for value in self.predicates[register].into_iter() {
            self.trace.prune(&self.xs[self.at], p, value);
        }
        match self.predicates[register] {
            Some(value) if value != execute => {
                self.next();
                Ok(())
            }
            None => {
                let mut skipped = self.predicates;
                skipped[register] = Some(!execute);
                self.enqueue((self.at + 1, self.frame, skipped));
                self.predicates[register] = Some(execute);
                then(self)
            }
            _ => then(self),
        }
    }
    fn next(&mut self) {
        self.enqueue((self.at + 1, self.frame, self.predicates));
    }
    fn forget(&mut self) {
        self.predicates = initial_stack_predicates();
    }
    fn push(&mut self, stack: TargetStack, target: BranchTarget) -> Result<(), Abort> {
        let (at, frame, predicates) = (self.at, self.frame, self.predicates);
        self.trace.event(at, || {
            format!(
                "push {stack:?}, target byte Some({}), older frame {frame}",
                target.slot()
            )
        });
        if self.slots.get(target.slot()).is_none() {
            let reason = format!(
                "push at byte 0x{:x} has missing/out-of-range target Some({})",
                self.address(),
                target.slot()
            );
            return Err(self.abort(reason));
        }
        let Self {
            frames,
            parents,
            dependents,
            ..
        } = self;
        let next = *self.frame_ids.entry((at, predicates)).or_insert_with(|| {
            let id = frames.len();
            frames.push(Some(Pushed { at, stack, target }));
            parents.push(std::collections::HashSet::new());
            dependents.push(std::collections::HashSet::new());
            id
        });
        if self.parents[next].insert(frame) {
            let dependents = &self.dependents;
            self.trace.event(at, || {
                format!(
                    "frame {next} adds older frame {frame}; revisit {} dependent pop states",
                    dependents[next].len()
                )
            });
            // A new predecessor may change a previously visited pop,
            // even when its top frame did not change. Revisit every
            // consumer that inspected this frame's predecessor links.
            let states = self.dependents[next].iter().copied().collect::<Vec<_>>();
            for state in states {
                self.visited.remove(&state);
                self.enqueue(state);
            }
        }
        self.frame = next;
        Ok(())
    }
    fn outside_push(&mut self, kind: TargetStack) -> Abort {
        let frame = self.frame;
        self.trace.event(self.at, || {
            format!("push {kind:?}, target byte None, older frame {frame}")
        });
        let reason = format!(
            "push at byte 0x{:x} has missing/out-of-range target None",
            self.address()
        );
        self.abort(reason)
    }
    fn outside_branch(&mut self) -> Abort {
        let reason = format!(
            "BRA at byte 0x{:x} has missing/out-of-range target None",
            self.address()
        );
        self.abort(reason)
    }
    fn outside_table(&mut self, table: &crate::flow::OutsideTable) -> Result<(), Abort> {
        // Entries before the first missing one are still explored;
        // the first outside entry names no slot and ends exploration.
        for target in table.table().leading() {
            let Some(target) = self.slots.get(target.slot()) else {
                let reason = format!(
                    "BRX at byte 0x{:x} has missing/out-of-range target Some({})",
                    self.address(),
                    target.slot()
                );
                return Err(self.abort(reason));
            };
            self.enqueue((target, self.frame, self.predicates));
        }
        let reason = format!(
            "BRX at byte 0x{:x} has missing/out-of-range target None",
            self.address()
        );
        Err(self.abort(reason))
    }
    fn pop(&mut self, kind: TargetStack) -> Result<(), Abort> {
        let at = self.at;
        let state = (at, self.frame, self.predicates);
        let mut search = vec![self.frame];
        let mut seen = std::collections::HashSet::new();
        while let Some(current) = search.pop() {
            if !seen.insert(current) {
                continue;
            }
            let Some(producer) = self.frames[current] else {
                self.invalid[at] = true;
                self.trace.event(at, || {
                    "pop search reached empty frame 0: stack underflow".into()
                });
                continue;
            };
            self.dependents[current].insert(state);
            let (parents, xs) = (&self.parents, self.xs);
            self.trace.event(at, || {
                let mut older = parents[current].iter().copied().collect::<Vec<_>>();
                older.sort_unstable();
                let relation = ["discard intervening kind", "matching kind"][usize::from(producer.stack == kind)];
                format!("pop searches frame {current}: push {:?} at byte 0x{:x}, older frames {older:?}, {relation}", producer.stack, xs[producer.at].address)
            });
            // Ryujinx Decoder.cs:830-850 discards intervening kinds,
            // then consumes the nearest matching push, preserving the
            // older stack for execution at that push's target.
            if producer.stack != kind {
                search.extend(self.parents[current].iter().copied());
                continue;
            }
            let Some(target) = self.slots.get(producer.target.slot()) else {
                let reason = format!(
                    "pop at byte 0x{:x} consumes push at byte 0x{:x} with missing target",
                    self.address(),
                    self.xs[producer.at].address
                );
                return Err(self.abort(reason));
            };
            self.trace.event(at, || {
                format!(
                    "pop target byte 0x{:x} from push byte 0x{:x}",
                    xs[target].address, xs[producer.at].address
                )
            });
            self.targets[at].insert(target);
            let older = self.parents[current].iter().copied().collect::<Vec<_>>();
            for older in older {
                self.enqueue((target, older, self.predicates));
            }
        }
        Ok(())
    }
    fn jump(&mut self, slot: u32) -> Result<(), Abort> {
        let Some(target) = self.slots.get(slot) else {
            let reason = format!(
                "BRA at byte 0x{:x} has missing/out-of-range target Some({})",
                self.address(),
                slot
            );
            return Err(self.abort(reason));
        };
        self.enqueue((target, self.frame, self.predicates));
        Ok(())
    }
    fn table(&mut self, table: &crate::flow::TableJump) -> Result<(), Abort> {
        for target in table.targets() {
            let Some(target) = self.slots.get(target.slot()) else {
                let reason = format!(
                    "BRX at byte 0x{:x} has missing/out-of-range target Some({})",
                    self.address(),
                    target.slot()
                );
                return Err(self.abort(reason));
            };
            self.enqueue((target, self.frame, self.predicates));
        }
        Ok(())
    }
    fn any(&mut self) {
        // Runtime BRX only changes the PC (Ryujinx InstEmitFlowControl.cs:
        // 38-40). Propagate unchanged stack/guards to every executable
        // slot, including backward and middle-of-block destinations.
        for target in 0..self.xs.len() {
            self.enqueue((target, self.frame, self.predicates));
        }
    }
    fn unknown(&mut self) -> Abort {
        let reason = format!(
            "Unknown instruction at byte 0x{:x} may hide control/stack effects",
            self.address()
        );
        self.abort(reason)
    }
}

fn resolve_stack_jumps_traced_in(
    xs: &[Instruction],
    program: &mut Program,
    slots: &Slots,
    trace: impl StackTrace,
) {
    use std::collections::{HashMap, HashSet};
    if program.pops() == 0 {
        // An empty consumer set needs no stack exploration or runtime tag
        // writes (Ryujinx InstEmitFlowControl.cs:239-251). Check every push
        // target before resolving, preserving missing-destination errors.
        // Discovery still visits these targets (Decoder.cs:133-138).
        program.settle_held(slots);
        return;
    }
    let mut finite = Finite {
        xs,
        slots,
        trace,
        aborted: None,
        frames: vec![None],
        parents: vec![HashSet::new()],
        frame_ids: HashMap::new(),
        dependents: vec![HashSet::new()],
        pending: vec![(0, 0, initial_stack_predicates())],
        visited: HashSet::new(),
        targets: vec![HashSet::new(); xs.len()],
        invalid: vec![false; xs.len()],
        at: 0,
        frame: 0,
        predicates: initial_stack_predicates(),
    };
    finite.run(program);
    let Finite {
        trace,
        aborted,
        targets,
        invalid,
        ..
    } = finite;
    trace.report(program, xs, slots, &targets, &invalid, aborted.as_deref());
    if aborted.is_some() {
        return;
    }
    // Unique destinations only justify lowering when every feasible
    // history has a matching push (Ryujinx InstEmitFlowControl.cs:254-262).
    // Dispatcher presence says nothing about an empty-stack history.
    program.lower(&reached(&targets, &invalid), xs);
    // Every possible pop was explored, including empty-stack paths. A push
    // with no consumer has no tag writes in Ryujinx InstEmitFlowControl.cs:
    // 239-251. Used targets affect the branches above; unresolved consumers
    // still reject translation. Keep every push and its target in the IR.
    program.settle_pushes();
}

/// Resolve the compiler-generated BRX table described by Ryujinx's decoder
/// (Ryujinx.Graphics.Shader/Decoders/Decoder.cs:527-552,561-613). The table
/// is read from constant-buffer 1, and a missing LDC/SHL/unsigned-IMNMX
/// chain deliberately leaves register BRX using general dispatch. The same
/// immutable CB1 snapshot specializes OUT streams, directly or through
/// full-lane MOVs. Re-decode whenever any consumed CB1 value changes.
pub fn decode_stream_with_constant_buffer<R: Run>(
    run: R,
    w: &[u32],
    cb1: &[u32],
) -> Vec<Instruction> {
    decode_reachable(run, w, Some(cb1))
}

// Surface atomic fields: Mesa sm50.rs:2558-2591 and envytools
// gm107.c:769-791,1833-1871,1908,1911-1914. SURED size/BA reuse
// raw surface positions (gm107.c:794-822); its op is 24..27 (769-777).
// envytools omits the SURED size labels; Ryujinx InstDecoders.cs:4740-4742,
// 4756-4758 corroborates the size/BA positions and atomic size meaning.

/// Ryujinx Decoder.cs:561-616's common IMNMX/SHL/LDC chain. Predicated
/// definitions cannot bound all executions. Matching the unsigned MIN is
/// essential: MAX supplies a lower bound and cannot constrain table indices.
/// This independent implementation supports adjacent writes only.
#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs, path::Path};

    fn tail_kind(raw: u64) -> InstructionKind {
        decode_word([raw as u32, (raw >> 32) as u32], 0).kind
    }

    /// An unguarded instruction at `address`.
    fn at(address: u32, kind: InstructionKind) -> Instruction {
        Instruction {
            address,
            predicate: edge::GUARD[7],
            kind,
        }
    }

    /// An instruction at `address` under guard P`register`, inverted or not.
    fn guarded(address: u32, register: u8, inverted: bool, kind: InstructionKind) -> Instruction {
        Instruction {
            address,
            predicate: edge::GUARD[usize::from(register) | usize::from(inverted) << 3],
            kind,
        }
    }

    /// The target the instruction at `from` reaches with encoded byte
    /// displacement `offset` inside the code address space.
    fn target(from: u32, offset: i32) -> BranchTarget {
        BranchTarget::reaching(
            from,
            crate::ir::run_slot(from.wrapping_add_signed(8 + offset)),
        )
    }

    /// How many times the trace `output` says `needle`.
    fn says(output: impl AsRef<[u8]>, needle: &str) -> usize {
        String::from_utf8_lossy(output.as_ref())
            .matches(needle)
            .count()
    }

    /// What emitting `xs` gives: `Ok(())` or the error it reports.
    fn emitted(xs: &[Instruction]) -> String {
        format!("{:?}", crate::emitter::emit(xs).map(drop))
    }

    /// The emission stop for an unresolved stack jump at `byte`.
    fn unresolved(byte: u32) -> String {
        format!("Err(UnprovenPop({byte}))")
    }

    /// The kind an encoding no modeled instruction has: its raw word.
    fn unknown(raw: u64) -> InstructionKind {
        InstructionKind::Unknown(ops::Unknown {
            raw: [raw as u32, (raw >> 32) as u32],
        })
    }

    #[test]
    fn unresolved_brx_preserves_predicate_guarded_stack_proofs() {
        let mut xs = vec![
            at(
                8,
                InstructionKind::Schedule(ops::Schedule {
                    target: target(8, 24),
                }),
            ),
            guarded(
                16,
                0,
                false,
                InstructionKind::PopJump(ops::PopJump {
                    cc_test: 0xf,
                    stack: TargetStack::Sync,
                }),
            ),
            guarded(
                24,
                0,
                true,
                InstructionKind::BrxRegister(ops::BrxRegister {
                    displacement: Register(0),
                    base: BrxBase::decode(24, 0),
                    cc_test: 0xf,
                }),
            ),
            at(40, InstructionKind::Exit(ops::Exit { cc_test: 15 })),
        ];
        let mut output = Vec::new();
        resolve_stack_jumps_traced(&mut xs, Traced::new(&mut output));
        assert_eq!(
            xs[1].kind,
            InstructionKind::Reconverge(ops::Reconverge {
                target: target(xs[1].address, 16),
                cc_test: 15,
                stack: TargetStack::Sync
            })
        );
        assert_eq!(
            xs[2].kind,
            InstructionKind::BrxRegister(ops::BrxRegister {
                displacement: Register(0),
                base: BrxBase::decode(24, 0),
                cc_test: 0xf,
            })
        );
        assert_ne!(says(&output, "predicate P0 proven"), 0);
        // Unknown may hide a push, predicate write, or memory operation.
        // Only the BRX family gets the unchanged-stack transfer.
        xs[0].kind = InstructionKind::Schedule(ops::Schedule {
            target: target(8, 24),
        });
        xs[1].kind = InstructionKind::PopJump(ops::PopJump {
            cc_test: 0xf,
            stack: TargetStack::Sync,
        });
        xs[2].kind = InstructionKind::Unknown(ops::Unknown { raw: [0, 0] });
        resolve_stack_jumps_traced(&mut xs, Traced::new(&mut output));
        assert_eq!(
            xs[1].kind,
            InstructionKind::PopJump(ops::PopJump {
                cc_test: 0xf,
                stack: TargetStack::Sync,
            })
        );
        assert_ne!(says(&output, "Unknown instruction at byte 0x18"), 0);
    }

    #[test]
    fn resolved_brx_discovers_all_targets_and_only_conditional_fallthrough() {
        // envytools gm107.c:1942-1943: base plus byte displacement.
        // Ryujinx Decoder.cs:188-195,265-270,501-515: indirect target
        // discovery, control-word skip and predicated fallthrough.
        let mut targets = [0; 32];
        targets[..4].copy_from_slice(&[16, 24, 48, 0x1000]);
        let mut x = at(
            8,
            InstructionKind::brx(
                Register(2),
                BrxBase::decode(8, 0),
                targets[0],
                &targets[1..3],
            ),
        );
        let mut edges = Vec::new();
        crate::flow::walk(&x, |successor| edges.push(successor.slot()));
        assert_eq!(edges, [40, 40, 72]);
        x.predicate = edge::GUARD[0];
        edges.clear();
        crate::flow::walk(&x, |successor| edges.push(successor.slot()));
        assert_eq!(edges, [40, 40, 72, 16]);
    }

    #[test]
    fn ipa_operands_and_unimplemented_fields_are_not_discarded() {
        // Mesa nak/sm50.rs:2905-2934; envytools gm107.c:859-871,1953.
        let pass = (0xe000u64 << 48)
            | (255 << 8)
            | (7 << 16)
            | (255 << 20)
            | (0x80 << 28)
            | (255 << 39)
            | (7 << 47);
        let mul = (pass & !(255 << 20)) | (7 << 20) | (1 << 54);
        assert_eq!(
            tail_kind(mul),
            InstructionKind::IpaPassMulW(ops::IpaPassMulW {
                dst: Register(0),
                addr: 0x80,
                inv_w: Source::Register(Register(7)),
                offset: Source::Register(Register(255)),
                loc: InterpLoc::Default,
                freq: InterpFreq::PassMulW,
                saturate: false,
            })
        );
        for valid in [pass, pass | (2 << 54), mul] {
            assert_ne!(tail_kind(valid), unknown(valid));
            for bit in [8, 28, 38, 39, 47, 48, 49, 50, 52, 53] {
                let word = valid ^ (1 << bit);
                assert_eq!(
                    tail_kind(word),
                    unknown(word),
                    "IPA {valid:016x} bit{bit} was discarded"
                );
            }
        }
        assert_eq!(
            tail_kind(mul | (1 << 51)),
            InstructionKind::IpaSaturate(ops::IpaSaturate {
                dst: Register(0),
                addr: 0x80,
                inv_w: Source::Register(Register(7)),
            })
        );
        // SAT on PASS and CONSTANT saturates the result (Ryujinx
        // InstEmitAttribute.cs:146-218, MIT).
        assert_eq!(
            tail_kind(pass | (1 << 51)),
            InstructionKind::IpaPass(ops::IpaPass {
                dst: Register(0),
                addr: 0x80,
                inv_w: Source::Register(Register(255)),
                offset: Source::Register(Register(255)),
                loc: InterpLoc::Default,
                freq: InterpFreq::Pass,
                saturate: true,
            })
        );
        assert_eq!(
            tail_kind(pass | (2 << 54) | (1 << 51)),
            InstructionKind::IpaPass(ops::IpaPass {
                dst: Register(0),
                addr: 0x80,
                inv_w: Source::Register(Register(255)),
                offset: Source::Register(Register(255)),
                loc: InterpLoc::Default,
                freq: InterpFreq::Constant,
                saturate: true,
            })
        );
        for word in [
            pass ^ (1 << 20),
            (pass | (2 << 54)) ^ (1 << 20),
            pass | (3 << 54),
        ] {
            assert_eq!(tail_kind(word), unknown(word));
        }
    }

    #[test]
    fn exit_requires_supported_condition_and_reference_count_mode() {
        // Mesa sm50.rs:3106-3110; envytools gm107.c:1925,426-442.
        for condition in 0..32 {
            let raw = (0xe300_u64 << 48) | (7 << 16) | condition;
            assert_eq!(
                tail_kind(raw),
                InstructionKind::Exit(ops::Exit {
                    cc_test: condition as u8
                })
            );
            let keep_reference_count = raw | 1 << 5;
            assert_eq!(
                tail_kind(keep_reference_count),
                unknown(keep_reference_count)
            );
        }
    }

    #[test]
    fn partial_word_validation_checks_all_resolved_brx_targets() {
        // Byte displacements and bundle-control targets: envytools
        // envydis/gm107.c:1942-1943; Ryujinx Decoder.cs:188-195,265-270.
        for displacement in [16, 24] {
            let mut targets = [0; 32];
            targets[1] = displacement;
            let mut xs = [
                at(
                    8,
                    InstructionKind::brx(
                        Register(2),
                        BrxBase::decode(8, 0),
                        targets[0],
                        &targets[1..2],
                    ),
                ),
                at(16, InstructionKind::Exit(ops::Exit { cc_test: 0xf })),
            ];
            assert_eq!(
                format!("{:?}", validate_reachable_words(&xs, 44)),
                "Err(PartialWord)"
            );
            xs[0].kind = InstructionKind::brx(
                Register(2),
                BrxBase::decode(8, 0),
                targets[0],
                &targets[1..1],
            );
            assert_eq!(format!("{:?}", validate_reachable_words(&xs, 44)), "Ok(())");
        }
    }

    #[test]
    fn flo_sources_and_unsupported_modifiers() {
        // Encoding references are at the FLO decoder arm.
        for (opcode, binding, src) in [
            (0x5c30, 0, Source::Register(Register(3))),
            (
                0x4c30,
                2 << 34,
                Source::ConstantBuffer {
                    binding: 2,
                    offset: 12,
                },
            ),
            (0x3930, 0, Source::Immediate(0xfff8_0003)),
        ] {
            let raw = opcode << 48 | binding | 3 << 20 | 7 << 16 | 1;
            let flo = |signed, invert, shift| {
                InstructionKind::Flo(ops::Flo {
                    dst: Register(1),
                    src,
                    signed,
                    invert,
                    shift,
                })
            };
            assert_eq!(tail_kind(raw | 1 << 40), flo(false, true, false));
            assert_eq!(tail_kind(raw | 1 << 48), flo(true, false, false));
            assert_eq!(tail_kind(raw | 1 << 41), flo(false, false, true));
            for rejected in [1 << 47, 1 << 48 | 1 << 41] {
                assert_eq!(tail_kind(raw | rejected), unknown(raw | rejected));
            }
        }
    }

    #[test]
    fn local_load_family_preserves_sizes_addresses_and_rejects_cache() {
        // envytools gm107.c:583-591,602-607,1895 and Mesa sm50.rs:2603-2614.
        use crate::ir::LsSize2;
        for (size, mem_type) in [
            LsSize2::U8,
            LsSize2::S8,
            LsSize2::U16,
            LsSize2::S16,
            LsSize2::B32,
            LsSize2::B64,
            LsSize2::B128,
        ]
        .into_iter()
        .enumerate()
        {
            let raw = (0xef40 + size as u64) << 48 | 0xff_fff0 << 20 | 2 << 8 | 1;
            let load = InstructionKind::LocalLoad(ops::LocalLoad {
                dst: Register(1),
                address: Register(2),
                offset: -16,
                mem_type,
            });
            assert_eq!(tail_kind(raw), load);
            // Bits 44..45 are the LU/CI/CV cache hints (gm107.c:602-607):
            // results are unchanged. Bits 46..47 are unused: reject them.
            for at in 44..46 {
                assert_eq!(tail_kind(raw | 1 << at), load);
            }
            for at in 46..48 {
                assert_eq!(tail_kind(raw | 1 << at), unknown(raw | 1 << at));
            }
        }
        assert_eq!(tail_kind(0xef47 << 48), unknown(0xef47 << 48));
    }

    #[test]
    fn i2f_double_writes_a_pair_and_keeps_unsupported_forms_unknown() {
        // Ryujinx InstEmitConversion.cs:255,268-283,410-415.
        let i2f = |signed, src_size, dst, src| {
            InstructionKind::Convert(ops::Convert {
                kind: Conversion::I2F {
                    signed,
                    dst_size: 3,
                    src_size,
                    round: RoundingMode::NearestEven,
                    byte: 0,
                    abs: false,
                    negate: false,
                },
                dst,
                src,
            })
        };
        for (opcode, src) in [
            (0x5cb8, Source::Register(Register(0))),
            (
                0x4cb8,
                Source::ConstantBuffer {
                    binding: 0,
                    offset: 0,
                },
            ),
            (0x38b8, Source::Immediate(0)),
        ] {
            for size in 0..=2 {
                let raw = opcode << 48 | 3 << 8 | size << 10 | 1 << 13 | 1;
                let kind = tail_kind(raw);
                assert_eq!(kind, i2f(true, size as u8, Register(1), src));
                assert_eq!(
                    kind.written_registers(),
                    crate::ir::Written::new(Register(1), 2)
                );
                assert_eq!(tail_kind(raw | (1 << 47)), unknown(raw | (1 << 47)));
            }
        }
        assert_eq!(
            tail_kind(0x5cb8 << 48 | 3 << 8 | 3 << 10),
            i2f(false, 3, Register(0), Source::Register(Register(0)))
        );
        for opcode in [0x4cb8, 0x38b8] {
            let raw = opcode << 48 | 3 << 8 | 3 << 10;
            assert_eq!(tail_kind(raw), unknown(raw));
        }
    }

    #[test]
    fn only_source_proven_nops_and_scheduling_can_decode_as_nop() {
        // Mesa nak/sm50.rs:3178-3184; envytools gm107.c:1877-1878. The exact
        // The exact 0x50b1..0x50b7 packets are covered by nop_packet and
        // nop_tail_packet.
        for opcode in (0..0x50b0)
            .chain(0x50b8..0xf0f0)
            .chain(0xf100..=u64::from(u16::MAX))
        {
            let raw = (opcode << 48) | (7 << 16);
            assert_ne!(
                tail_kind(raw),
                InstructionKind::Nop(ops::Nop),
                "{opcode:04x}"
            );
        }
    }

    #[test]
    fn fall_through_branch_keeps_its_instruction_kind() {
        // Mesa nak/sm50.rs:3006-3028: BRA opcode and relative displacement.
        // A zero displacement jumps from address 8 to the next slot at 16.
        let bra = (0xe240_u64 << 48) | (7 << 16) | 0xf;
        let exit = (0xe300_u64 << 48) | (7 << 16) | 0xf;
        let nop = (0x50b0_u64 << 48) | (7 << 16);
        let stream = [
            0,
            0,
            bra as u32,
            (bra >> 32) as u32,
            exit as u32,
            (exit >> 32) as u32,
            nop as u32,
            (nop >> 32) as u32,
        ];
        assert_eq!(
            decode_stream(crate::QuietRun, &stream)[0].kind,
            InstructionKind::Branch(ops::Branch {
                target: target(8, 0),
                cc_test: 0xf
            })
        );
    }

    #[test]
    fn ald_decodes_component_sizes_and_addressing_flags() {
        use crate::ir::{AttributeBank, AttributeLanes};
        let load = |address, components| {
            InstructionKind::AttributeLoad(ops::AttributeLoad {
                dst: Register(0),
                vertex: PrimitiveVertex::Current,
                address,
                components,
                lanes: AttributeLanes::vertex(address, components, AttributeBank::decode(false)),
            })
        };
        for components in 1..=4u64 {
            let word = (0xefd8u64 << 48)
                | (0xff << 39)
                | ((components - 1) << 47)
                | (0xff << 8)
                | (0x84 << 20);
            let decoded = decode_word([word as u32, (word >> 32) as u32], 0);
            assert_eq!(decoded.kind, load(0x84, components as u8));
        }
        // Without P, Ra not RZ with a nonzero immediate has no ALD
        // meaning: nvdisasm SM52 rejects this exact word.
        let indexed = (0xefd8u64 << 48) | (0xff << 39) | (3 << 47) | (0x80 << 20) | (2 << 8);
        assert_eq!(tail_kind(indexed), unknown(indexed));
        // A vector physical read stays unproven.
        let physical = (0xefd8u64 << 48) | (0xff << 39) | (1 << 47) | (2 << 8);
        assert_eq!(tail_kind(physical), unknown(physical));
        let flags = (0xefd8u64 << 48) | (0xff << 39) | (0xff << 8) | (1 << 31) | (1 << 32);
        assert_eq!(
            tail_kind(flags),
            InstructionKind::PatchAttributeLoad(ops::PatchAttributeLoad {
                dst: Register(0),
                address: 0,
                components: 1,
                lanes: AttributeLanes::patch(0, 1, AttributeBank::Output),
            })
        );
    }

    #[test]
    fn ald_lanes_follow_the_ryujinx_attribute_map() {
        use crate::ir::{Attribute, AttributeBank, AttributeLane, AttributeLanes, OutputVertex};
        let word = |address: u64, components: u64, offset: u64, flags: u64| {
            (0xefd8u64 << 48)
                | (0x01 << 39)
                | ((components - 1) << 47)
                | (offset << 8)
                | (address << 20)
                | flags
        };
        let ald = |address, components, offset, flags| {
            tail_kind(word(address, components, offset, flags))
        };
        let output = 1 << 32;
        let patch = 1 << 31;
        let input_bank = AttributeBank::decode(false);
        let output_bank = AttributeBank::decode(true);
        // AttributeMap.cs:63-81 maps nothing below 0x60, at 0x2ec or
        // between 0x3a4 and 0x3f8; unaligned offsets miss every key.
        for (address, components, flags) in [
            (0, 1, output),
            (0, 4, output),
            (0x5c, 1, output),
            (0x50, 4, output),
            (0x2, 1, output),
            (0, 1, 0),
            (0x50, 4, 0),
            (0x2ec, 1, 0),
            (0x3a4, 4, 0),
            (0x81, 4, 0),
            (0x200, 1, patch),
        ] {
            assert_eq!(
                ald(address, components, 0xff, flags),
                InstructionKind::UnmappedAttributeLoad(ops::UnmappedAttributeLoad {
                    dst: Register(0),
                    address: address as u16,
                    components: components as u8,
                }),
                "{address:x} {components} {flags:x}"
            );
        }
        // The address is ten bits (Mesa sm50.rs:2873, envytools gm107.c:
        // 314): bit 30 has no ALD meaning, whatever Ra, P and O say.
        for (address, offset, flags) in [
            (0x400, 0xff, 0),
            (0x7fc, 0xff, output),
            (0x400, 0xff, patch),
            (0x400, 2, 0),
            (0x400, 2, patch),
        ] {
            let raw = word(address, 1, offset, flags);
            assert_eq!(
                tail_kind(raw),
                unknown(raw),
                "{address:x} {offset:x} {flags:x}"
            );
        }
        let lanes = |lanes: AttributeLanes, address| {
            lanes
                .iter(address)
                .map(|(_, lane)| lane)
                .collect::<Vec<_>>()
        };
        let lane = |attribute, output| AttributeLane {
            attribute,
            bank: AttributeBank::decode(output),
        };
        let output_load = |address, components| {
            InstructionKind::OutputAttributeLoad(ops::OutputAttributeLoad {
                dst: Register(0),
                vertex: OutputVertex::Lane(Register(1)),
                address,
                components,
                lanes: AttributeLanes::vertex(address, components, output_bank),
            })
        };
        let patch_load = |address, components| {
            InstructionKind::PatchAttributeLoad(ops::PatchAttributeLoad {
                dst: Register(0),
                address,
                components,
                lanes: AttributeLanes::patch(address, components, input_bank),
            })
        };
        let load = |address, components, vertex| {
            InstructionKind::AttributeLoad(ops::AttributeLoad {
                dst: Register(0),
                vertex,
                address,
                components,
                lanes: AttributeLanes::vertex(address, components, input_bank),
            })
        };
        assert_eq!(ald(0x5c, 2, 0xff, output), output_load(0x5c, 2));
        assert_eq!(
            lanes(AttributeLanes::vertex(0x5c, 2, output_bank), 0x5c),
            [
                lane(Attribute::Unmapped, true),
                lane(Attribute::PrimitiveId, true)
            ]
        );
        assert_eq!(
            ald(0x60, 4, 0xff, 0),
            load(0x60, 4, PrimitiveVertex::Current)
        );
        assert_eq!(
            lanes(AttributeLanes::vertex(0x60, 4, input_bank), 0x60),
            [
                lane(Attribute::PrimitiveId, false),
                lane(Attribute::Layer, false),
                lane(Attribute::ViewportIndex, false),
                lane(Attribute::PointSize, false),
            ]
        );
        assert_eq!(ald(0x2e8, 4, 0xff, output), output_load(0x2e8, 4));
        assert_eq!(
            lanes(AttributeLanes::vertex(0x2e8, 4, output_bank), 0x2e8),
            [
                lane(Attribute::FixedFunction, true),
                lane(Attribute::Unmapped, true),
                lane(Attribute::TessellationCoord, false),
                lane(Attribute::TessellationCoord, false),
            ]
        );
        assert_eq!(ald(0x0c, 3, 0xff, patch), patch_load(0x0c, 3));
        assert_eq!(
            lanes(AttributeLanes::patch(0x0c, 3, input_bank), 0x0c),
            [
                lane(Attribute::TessellationLevelOuter, false),
                lane(Attribute::TessellationLevelInner, false),
                lane(Attribute::TessellationLevelInner, false),
            ]
        );
        // Rb is a primitive vertex only without P and O, and never for
        // PrimitiveId or TessCoord (InstEmitAttribute.cs:25-29,297-302).
        assert_eq!(ald(0x80, 1, 0xff, output), output_load(0x80, 1));
        assert_eq!(ald(0x18, 1, 0xff, patch), patch_load(0x18, 1));
        let primitive = PrimitiveVertex::Primitive(Register(1));
        for (address, vertex) in [
            (0x80, primitive),
            (0x60, PrimitiveVertex::Current),
            (0x2f0, PrimitiveVertex::Current),
            (0x2f4, PrimitiveVertex::Current),
            (0x64, primitive),
        ] {
            assert_eq!(
                ald(address, 1, 0xff, 0),
                load(address as u16, 1, vertex),
                "{address:x}"
            );
        }
        // Physical: zero immediate, no P, Ra not RZ (InstDecoders.cs:887).
        assert_eq!(
            ald(0, 1, 2, output),
            InstructionKind::PhysicalOutputAttributeLoad(ops::PhysicalOutputAttributeLoad {
                dst: Register(0),
                vertex: OutputVertex::Lane(Register(1)),
                offset: Register(2),
            })
        );
        // P with Ra not RZ indexes the patch bank (Mesa
        // nak_nir_lower_vtg_io.c:203,208-211).
        assert_eq!(
            ald(0, 1, 2, patch),
            InstructionKind::IndexedPatchLoad(ops::IndexedPatchLoad {
                dst: Register(0),
                base: 0,
                index: Register(2),
                components: 1,
                bank: input_bank,
                lanes: AttributeLanes::patch(0, 1, input_bank),
            })
        );
    }

    #[test]
    fn imnmx_extended_modes_remain_unknown() {
        // envytools gm107.c:1383-1388 distinguishes XLO/XMED/XHI at43..44.
        // Ryujinx InstEmitIntegerMinMax.cs:68 does not define their X flags.
        for op in [0x5c20_u64, 0x4c20, 0x3820] {
            for mode in 1..=3 {
                let raw = op << 48 | mode << 43 | 7 << 39 | 7 << 16 | 2 << 8 | 1;
                assert_eq!(tail_kind(raw), unknown(raw));
            }
        }
    }

    #[test]
    fn reviewed_encodings_keep_their_instruction_kinds() {
        assert_eq!(
            tail_kind((0x5080u64 << 48) | (4 << 20)),
            InstructionKind::MuFu(ops::MuFu {
                dst: Register(0),
                src: Register(0),
                op: MufuOp::Rcp,
                abs: false,
                negate: false,
                saturate: false,
            })
        );
        assert_eq!(
            tail_kind((0x3821u64 << 48) | (1 << 48) | (7 << 39)),
            InstructionKind::IMinMax(ops::IMinMax {
                dst: Register(0),
                lhs: Register(0),
                rhs: Source::Immediate(0),
                min: true,
                signed: true,
                cc: false,
            })
        );
        // dst1 = PT (0..3) and accumulator = PT (39..42), as Mesa encodes.
        // 0x5b65: bit 48 signed, compare 49..52 = 2 (Eq).
        assert_eq!(
            tail_kind((0x5b65u64 << 48) | 7 | (7 << 39)),
            InstructionKind::ISetP(ops::ISetP {
                dst: [0, 7],
                lhs: Register(0),
                rhs: Source::Register(Register(0)),
                compare: IntCompare::Relation(Relation::Eq),
                signedness: Signedness::Signed,
                mode: IntegerCompareMode::Ordinary,
                set_op: SetOp::And,
                accum: 7,
                accum_negate: false,
                abs: [false; 2],
                negate: [false; 2],
            })
        );
        // XMAD decodes, CSFU c-mode (3) included; the undefined c-modes
        // 5..7 and .X stay Unknown.
        let word = (0x5b00u64 << 48) | 7 | (7 << 39);
        assert_eq!(
            tail_kind(word),
            InstructionKind::Xmad(ops::Xmad {
                dst: Register(7),
                a: Register(0),
                b: Source::Register(Register(0)),
                c: Source::Register(Register(7)),
                signed: [false; 2],
                high: [false; 2],
                psl: false,
                mrg: false,
                cmode: XmadCop::Cfull,
                cc: false,
            })
        );
        let csfu = word | (3 << 50);
        assert!(matches!(
            tail_kind(csfu),
            InstructionKind::Xmad(ops::Xmad {
                cmode: XmadCop::Csfu,
                ..
            })
        ));
        let undefined = word | (5 << 50);
        assert_eq!(tail_kind(undefined), unknown(undefined));
        let carry = word | (1 << 38);
        assert_eq!(tail_kind(carry), unknown(carry));
        let imad = (0x5a00u64 << 48) | 7 | (7 << 39);
        assert_eq!(tail_kind(imad).to_string(), "IMUL R7, R0, R0, R7");
    }

    #[test]
    fn iset_and_r2p_follow_envytools_fields() {
        let word = |opcode: u64, bf: u64, signed: u64| {
            (opcode << 48) | 3 | (9 << 8) | (7 << 39) | bf << 44 | signed << 48 | (2 << 49)
        };
        let iset = |signedness, bool_float| {
            InstructionKind::ISet(ops::ISet {
                dst: Register(3),
                lhs: Register(9),
                rhs: Source::Register(Register(0)),
                compare: IntCompare::Relation(Relation::Eq),
                signedness,
                bool_float,
                cc: false,
                x: false,
                accum: 7,
                accum_negate: false,
                set_op: SetOp::And,
            })
        };
        assert_eq!(
            tail_kind(word(0x5b50, 0, 0)),
            iset(Signedness::Unsigned, false)
        );
        assert_eq!(
            tail_kind(word(0x5b50, 1, 1)),
            iset(Signedness::Signed, true)
        );
        // Mask is the immediate at 20.., byte select 41..43.
        let r2p = (0x38f0u64 << 48) | (5 << 8) | (0b0100101 << 20) | (2 << 41);
        assert_eq!(
            tail_kind(r2p),
            InstructionKind::R2P(ops::R2P {
                src: Register(5),
                mask: Source::Immediate(0b0100101),
                byte: 2,
                bank: crate::ir::FlagBank::Predicates,
            })
        );
    }

    /// ISET R0, R0, `rhs` with only its X or CC modifier set.
    fn iset_modifier(rhs: Source, x: bool, cc: bool) -> InstructionKind {
        InstructionKind::ISet(ops::ISet {
            dst: Register(0),
            lhs: Register(0),
            rhs,
            compare: IntCompare::False,
            signedness: Signedness::Unsigned,
            bool_float: false,
            cc,
            x,
            accum: 0,
            accum_negate: false,
            set_op: SetOp::And,
        })
    }

    #[test]
    fn iset_modifiers_are_decoded_and_reserved_combine_stays_unknown() {
        let immediate = Source::Immediate(0);
        assert_eq!(
            tail_kind((0x3650u64 << 48) | (1 << 43)),
            iset_modifier(immediate, true, false)
        );
        assert_eq!(
            tail_kind((0x3650u64 << 48) | (1 << 47)),
            iset_modifier(immediate, false, true)
        );
        let reserved = (0x3650u64 << 48) | (3 << 45);
        assert_eq!(tail_kind(reserved), unknown(reserved));
    }

    #[test]
    fn shift_bits_forms_and_unimplemented_flags_follow_references() {
        // Fields: envytools gm107.c:1990,1994,2000,2017,2018,2021,
        // 2068,2091,2110,2114,2137,2138,2163; subfields 1179-1190,1228-1253.
        let immediate = Source::Immediate(0);
        let register = Source::Register(Register(0));
        let constant = Source::ConstantBuffer {
            binding: 0,
            offset: 0,
        };
        for (opcode, shift, right) in [
            (0x36f8u64, Source::Immediate(17), false),
            (0x38f8, Source::Immediate(17), true),
            (0x5bf8, Source::Register(Register(17)), false),
            (0x5cf8, Source::Register(Register(17)), true),
        ] {
            let raw = (opcode << 48) | (5 << 8) | (17 << 20) | (9 << 39);
            assert_eq!(
                tail_kind(raw),
                InstructionKind::FunnelShift(ops::FunnelShift {
                    dst: Register(0),
                    low: Register(5),
                    high: Register(9),
                    shift,
                    right,
                    dst_high: !right,
                    ty: FunnelType::Bits32,
                    wrap: false,
                    cc: false,
                })
            );
            assert_eq!(tail_kind(raw | (1 << 49)), unknown(raw | (1 << 49)));
            assert_eq!(tail_kind(raw | (1 << 37)), unknown(raw | (1 << 37)));
        }
        assert_eq!(tail_kind(0x4cf8u64 << 48), unknown(0x4cf8u64 << 48));
        for (opcode, xbit, addend) in [(0x5bdfu64, 38, register), (0x1807, 57, constant)] {
            let raw = (opcode << 48) | (5 << 8) | (9 << 39);
            assert_eq!(
                tail_kind(raw),
                InstructionKind::LeaHi(ops::LeaHi {
                    dst: Register(0),
                    low: Register(5),
                    addend,
                    high: Register(9),
                    shift: 0,
                    negate: false,
                })
            );
            for bit in [xbit, 47] {
                assert_eq!(tail_kind(raw | (1 << bit)), unknown(raw | (1 << bit)));
            }
            assert_eq!(tail_kind(raw & !(7 << 48)), unknown(raw & !(7 << 48)));
        }
        let iadd3 = |rhs, negate| {
            InstructionKind::IAdd3(ops::IAdd3 {
                dst: Register(0),
                lhs: Register(0),
                rhs,
                addend: Register(0),
                halves: [HalfSelect::Full; 3],
                shift: PartialSumShift::Unshifted,
                negate,
            })
        };
        for (opcode, rhs) in [
            (0x5cc0u64, register),
            (0x4cc0, constant),
            (0x38c0, immediate),
        ] {
            let raw = (opcode << 48) | (1 << 51) | (1 << 50) | (1 << 49);
            assert_eq!(tail_kind(raw), iadd3(rhs, [true; 3]));
            for bit in [47, 48] {
                assert_eq!(tail_kind(raw | (1 << bit)), unknown(raw | (1 << bit)));
            }
        }
        for at in [31, 33, 35, 37] {
            let raw = (0x5cc0u64 << 48) | (3 << at);
            assert_eq!(tail_kind(raw), unknown(raw));
        }
        for (opcode, range) in [
            (0x5c01u64, register),
            (0x4c01, constant),
            (0x3801, immediate),
        ] {
            let raw = (opcode << 48) | (1 << 40);
            assert_eq!(
                tail_kind(raw),
                InstructionKind::BitFieldExtract(ops::BitFieldExtract {
                    dst: Register(0),
                    base: Register(0),
                    range,
                    signed: true,
                    reverse: true,
                })
            );
            assert_eq!(tail_kind(raw | (1 << 47)), unknown(raw | (1 << 47)));
        }
        assert_eq!(
            tail_kind((0x5c92u64 << 48) | (1 << 45)),
            InstructionKind::Rro(ops::Rro {
                dst: Register(0),
                src: register,
                abs: true,
                negate: true,
                ex2: false,
            })
        );
        // Mesa sm50.rs:335-346: integer immediate bit 56 is the sign.
        assert_eq!(
            tail_kind((0x39c0u64 << 48) | (0x7ffff << 20)),
            iadd3(Source::Immediate(u32::MAX), [false; 3])
        );
    }

    #[test]
    fn unsupported_iset_modifiers_remain_unknown() {
        // Retain semreview-A's register-form coverage alongside main's
        // immediate-form test. Main now implements X43 and CC47:
        // envytools gm107.c:2029,2147; Ryujinx
        // InstEmitIntegerComparison.cs:133-166. Combine45..46=3 is reserved.
        let register = Source::Register(Register(0));
        assert_eq!(
            tail_kind((0x5b50u64 << 48) | (1 << 43)),
            iset_modifier(register, true, false)
        );
        assert_eq!(
            tail_kind((0x5b50u64 << 48) | (1 << 47)),
            iset_modifier(register, false, true)
        );
        let word = (0x5b50u64 << 48) | (3 << 45);
        assert_eq!(tail_kind(word), unknown(word));
    }

    #[test]
    fn float_compare_codes_follow_mesa() {
        for (code, compare) in [
            (1, FloatCompare::Ordered(Relation::Lt)),
            (9, FloatCompare::Unordered(Relation::Lt)),
            (7, FloatCompare::Num),
            (8, FloatCompare::Nan),
            (13, FloatCompare::Unordered(Relation::Ne)),
        ] {
            assert_eq!(
                tail_kind((0x5bb0u64 << 48) | (code << 48) | 7 | (7 << 39)),
                InstructionKind::FSetP(ops::FSetP {
                    dst: [0, 7],
                    lhs: Register(0),
                    rhs: Source::Register(Register(0)),
                    compare,
                    set_op: SetOp::And,
                    accum: 7,
                    accum_negate: false,
                    abs: [false; 2],
                    negate: [false; 2],
                })
            );
        }
    }

    #[test]
    fn sm50_modifier_bits_are_preserved() {
        let word = |opcode: u64, bits: &[(u32, u64)]| {
            bits.iter()
                .fold(opcode << 48, |value, &(bit, set)| value | set << bit)
        };
        let register = Source::Register(Register(0));
        let constant = Source::ConstantBuffer {
            binding: 0,
            offset: 0,
        };
        let ffma = |dst, rhs, addend, modifiers| {
            InstructionKind::FFma(ops::FFma {
                dst,
                lhs: Register(0),
                rhs,
                addend,
                flags: FloatFlags::Kept,
                modifiers,
            })
        };
        // Bit 51 set selects a non-default rounding mode, which is not modeled.
        for (opcode, rhs, addend) in [
            (0x3280, Source::Immediate(0), register),
            (0x4980, constant, register),
            (0x5180, register, constant),
            (0x5980, register, register),
        ] {
            assert_eq!(
                tail_kind(word(opcode, &[(48, 1), (49, 1), (50, 1), (54, 1)])),
                ffma(
                    Register(0),
                    rhs,
                    addend,
                    FFmaModifiers {
                        product_negate: true,
                        addend_negate: true,
                        saturate: true,
                        flush: Flush::Dnz,
                    }
                )
            );
            // envytools gm107.c:1516-1520 reserves mode3.
            for raw in [word(opcode, &[(53, 1), (54, 1)]), word(opcode, &[(51, 1)])] {
                assert_eq!(tail_kind(raw), unknown(raw));
            }
        }
        // An odd destination register must not read as a modifier.
        assert_eq!(
            tail_kind(word(0x5980, &[(0, 1)])),
            ffma(
                Register(1),
                register,
                register,
                FFmaModifiers {
                    product_negate: false,
                    addend_negate: false,
                    saturate: false,
                    flush: Flush::Keep,
                }
            )
        );
        let fadd = |lhs_abs, rhs_abs, negation, saturation, denormals| {
            InstructionKind::FAdd(ops::FAdd {
                dst: Register(0),
                lhs: Register(0),
                rhs: register,
                modifiers: FAddModifiers::Operand {
                    lhs_abs,
                    rhs_abs,
                    lhs_negation: negation,
                    rhs_negation: negation,
                    saturation,
                    denormals,
                },
                round: RoundingMode::NearestEven,
                flags: FloatFlags::Kept,
            })
        };
        assert_eq!(
            tail_kind(word(0x5c58, &[(48, 1), (45, 1), (50, 1), (44, 1)])),
            fadd(
                SourceAbs::Plain,
                SourceAbs::Plain,
                SourceNegation::Negated,
                Saturation::Saturated,
                Denormals::FlushedToZero
            )
        );
        // Absolute value: srcs[0] at 46, srcs[1] at 49.
        assert_eq!(
            tail_kind(word(0x5c58, &[(46, 1)])),
            fadd(
                SourceAbs::Absolute,
                SourceAbs::Plain,
                SourceNegation::Plain,
                Saturation::Unclamped,
                Denormals::Preserved
            )
        );
        assert_eq!(
            tail_kind(word(0x5c58, &[(49, 1)])),
            fadd(
                SourceAbs::Plain,
                SourceAbs::Absolute,
                SourceNegation::Plain,
                Saturation::Unclamped,
                Denormals::Preserved
            )
        );
        let raw = word(0x5c68, &[(48, 1), (50, 1), (44, 1), (45, 1)]);
        assert_eq!(tail_kind(raw), unknown(raw));
        assert_eq!(
            tail_kind(word(0x5c10, &[])),
            InstructionKind::IAdd(ops::IAdd {
                dst: Register(0),
                lhs: Register(0),
                rhs: register,
                cc: false,
                x: false,
                negate: [false; 2],
            })
        );
        assert_eq!(
            tail_kind(word(0x5cc0, &[(20, 7), (39, 9)])),
            InstructionKind::IAdd3(ops::IAdd3 {
                dst: Register(0),
                lhs: Register(0),
                rhs: Source::Register(Register(7)),
                addend: Register(9),
                halves: [HalfSelect::Full; 3],
                shift: PartialSumShift::Unshifted,
                negate: [false; 3],
            })
        );
        let raw = word(0x5cc0, &[(47, 1)]);
        assert_eq!(tail_kind(raw), unknown(raw));
    }

    #[test]
    fn never_guard_and_invalid_immediate_ranges_are_preserved() {
        // gm107.c:408-411: !PT means never, not unguarded execution.
        let raw = (0x38c0u64 << 48) | (15 << 16);
        assert_eq!(
            decode_word([raw as u32, (raw >> 32) as u32], 0).predicate,
            Some(Predicate {
                register: 7,
                inverted: true
            })
        );
        // InstEmitBitfield.cs:151 does not define oversized ranges.
        for range in [0x0120u64, 0x0021, 0xff00, 0x021f] {
            let raw = (0x3800u64 << 48) | (range << 20);
            assert_eq!(tail_kind(raw), unknown(raw));
        }
    }

    #[derive(Debug, PartialEq, Eq)]
    enum Parsed {
        Supported(String),
        Ipa,
        Unknown,
    }

    fn register(token: &str) -> Option<u8> {
        if token == "rZ" {
            Some(255)
        } else {
            token.strip_prefix('r')?.parse().ok()
        }
    }

    fn canonical(line: &str) -> Parsed {
        let line = line.trim().split("//").next().unwrap_or("").trim();
        if line.contains("ipa.") {
            return Parsed::Ipa;
        }
        let Some((lhs, rhs)) = line.split_once('=') else {
            return if line.contains(" exit") || line == "exit" || line == "nop" {
                Parsed::Supported(if line == "nop" {
                    "NOP".into()
                } else {
                    "EXIT".into()
                })
            } else {
                Parsed::Unknown
            };
        };
        let dst = lhs.trim();
        let operands: Vec<_> = rhs.split_whitespace().collect();
        let Some(op) = operands.first().copied() else {
            return Parsed::Unknown;
        };
        let source = |value: &str| {
            register(value).map_or_else(
                || {
                    // NAK prints constant buffers as c[0xB][0xO]; Shadowbox
                    // prints the bank in decimal.
                    if let Some(rest) = value.strip_prefix("c[0x") {
                        if let Some((bank, tail)) = rest.split_once(']') {
                            if let Ok(bank) = u32::from_str_radix(bank, 16) {
                                return format!("c[{bank}]{tail}");
                            }
                        }
                    }
                    value
                        .strip_prefix("0x")
                        .and_then(|hex| u32::from_str_radix(hex, 16).ok())
                        .map_or_else(|| value.to_string(), |number| format!("0x{number:x}"))
                },
                |r| format!("R{r}"),
            )
        };
        match op.split('.').next().unwrap_or(op) {
            "mov" if operands.len() == 2 => {
                let Some(d) = register(dst) else {
                    return Parsed::Unknown;
                };
                Parsed::Supported(format!("MOV R{d}, {}", source(operands[1])))
            }
            "fadd" if operands.len() == 3 => {
                let (Some(d), Some(l)) = (register(dst), register(operands[1])) else {
                    return Parsed::Unknown;
                };
                Parsed::Supported(format!("FADD R{d}, R{l}, {}", source(operands[2])))
            }
            "fsetp" if operands.len() >= 3 => {
                let Some(p) = dst.strip_prefix('p').and_then(|v| v.parse::<u8>().ok()) else {
                    return Parsed::Unknown;
                };
                let Some(l) = register(operands[1]) else {
                    return Parsed::Unknown;
                };
                Parsed::Supported(format!("FSETP P{p}, R{l}, {}", source(operands[2])))
            }
            "sel" if operands.len() == 4 => {
                let Some(d) = register(dst) else {
                    return Parsed::Unknown;
                };
                let Some(l) = register(operands[2]) else {
                    return Parsed::Unknown;
                };
                let Some(p) = operands[1]
                    .trim_start_matches('!')
                    .strip_prefix('p')
                    .and_then(|v| v.parse::<u8>().ok())
                else {
                    return Parsed::Unknown;
                };
                let negate = if operands[1].starts_with('!') {
                    "!"
                } else {
                    ""
                };
                Parsed::Supported(format!(
                    "SEL R{d}, R{l}, {}, {negate}P{p}",
                    source(operands[3])
                ))
            }
            _ => Parsed::Unknown,
        }
    }

    fn normalize_hex(value: &str) -> String {
        value
            .split_whitespace()
            .map(|token| {
                token
                    .strip_prefix("0x")
                    .and_then(|hex| u32::from_str_radix(hex, 16).ok())
                    .map_or_else(|| token.to_string(), |number| format!("0x{number:x}"))
            })
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn decoded_words(path: &Path) -> Vec<Instruction> {
        let bytes = fs::read(path).unwrap();
        let (words, remainder) = bytes.as_chunks::<4>();
        assert_eq!(remainder.len(), 0);
        let words = words
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect::<Vec<_>>();
        decode_stream(crate::QuietRun, &words)
    }

    #[test]
    fn corpus_matches_nak_disassembly() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .unwrap();
        let mut totals = [0usize; 3];
        let mut mismatches = Vec::new();
        for entry in fs::read_dir(root.join("corpus"))
            .unwrap()
            .filter_map(Result::ok)
        {
            let path = entry.path();
            if path.extension().and_then(|x| x.to_str()) != Some("asm") {
                continue;
            }
            let name = path.file_stem().unwrap().to_str().unwrap();
            let expected = fs::read_to_string(&path)
                .unwrap()
                .lines()
                .filter(|line| {
                    line.contains('=')
                        || line.trim().starts_with("exit")
                        || line.trim().starts_with("nop")
                })
                .map(canonical)
                .collect::<Vec<_>>();
            let actual = decoded_words(&path.with_extension("bin"))
                .into_iter()
                .collect::<Vec<_>>();
            // Compare every discovered instruction at its physical slot.
            // Unreachable padding is absent from the stream (Mesa bundle
            // layout sm50.rs:3418-3428; Ryujinx Decoder.cs:133-162).
            for decoded in &actual {
                let index = (decoded.address / 32 * 3 + decoded.address % 32 / 8 - 1) as usize;
                if let Some(item) = expected.get(index) {
                    match item {
                        Parsed::Unknown => totals[2] += 1,
                        Parsed::Supported(expected)
                            if normalize_hex(&decoded.to_string()) == *expected =>
                        {
                            totals[0] += 1
                        }
                        Parsed::Ipa
                            if matches!(
                                decoded.kind,
                                InstructionKind::IpaPass(ops::IpaPass { .. })
                                    | InstructionKind::IpaPassMulW(ops::IpaPassMulW { .. })
                            ) =>
                        {
                            totals[0] += 1
                        }
                        Parsed::Ipa => {
                            totals[1] += 1;
                            mismatches.push(format!("{name}:{index}: IPA decode mismatch"));
                        }
                        Parsed::Supported(expected) => {
                            totals[1] += 1;
                            mismatches.push(format!(
                                "{name}:{index}: expected {expected}, decoded {decoded}"
                            ));
                        }
                    }
                } else {
                    totals[1] += 1;
                    mismatches.push(format!("{name}:{index}: missing disassembled instruction"));
                }
            }
        }
        eprintln!(
            "coverage: supported={}, mismatched={}, unknown={}",
            totals[0], totals[1], totals[2]
        );
        assert_eq!(mismatches.len(), 0, "{}", mismatches.join("\n"));
    }
    fn w(o: u32) -> [u32; 2] {
        [2 | 3 << 8 | 4 << 20, o << 16]
    }
    #[test]
    fn families() {
        for o in [
            0x5c98, 0x0100, 0x5c10, 0x3810, 0x1c00, 0x5c58, 0x3858, 0x0800, 0x5080, 0xe240, 0xe300,
            0x4c98, 0x4c10, 0x4c58, 0x4c68, 0x5b60, 0x3660, 0x4b60, 0x5bb0, 0x36b0, 0x4bb0, 0x5800,
            0x3000, 0x4800, 0x5a00, 0x3400, 0x4a00, 0x5ca0, 0x38a0, 0x4ca0, 0x5cb8, 0x38b8, 0x4cb8,
            0x5cb0, 0x38b0, 0x4cb0, 0x5ca8, 0x38a8, 0x4ca8, 0x5ce0, 0x38e0, 0x4ce0, 0x5c40, 0x3840,
            0x4c40, 0x5c48, 0x3848, 0x4c48, 0x5c28, 0x3828, 0x4c28, 0xeed0, 0xef40, 0xeed8, 0xef50,
            0x0380, 0xdeb8,
        ] {
            let _ = decode_word(w(o), 0);
        }
    }
    #[test]
    fn random_never_panics() {
        for x in 0..100000u64 {
            let _ = decode_word([x as u32, (x >> 32) as u32], 0);
        }
    }
    #[test]
    fn unknown_raw() {
        let r = [0xdead_beef, 0x1234_5678];
        assert_eq!(
            decode_word(r, 0).kind,
            InstructionKind::Unknown(ops::Unknown { raw: r })
        );
    }

    #[test]
    fn fswzadd_preserves_both_sources_and_rejects_unimplemented_fields() {
        // Mesa nak/sm50.rs:913-943; envytools gm107.c:2051.
        for mask in 0..=255_u64 {
            let raw = 0x50f8_u64 << 48 | mask << 28 | 113 << 20 | 7 << 16 | 17 << 8 | 2;
            assert_eq!(
                tail_kind(raw),
                InstructionKind::SwizzleAdd(ops::SwizzleAdd {
                    dst: Register(2),
                    src: [Register(17), Register(113)],
                    mask: mask as u8,
                    ftz: false,
                })
            );
            for bit in (36..48).filter(|bit| *bit != 44) {
                let modified = raw | 1 << bit;
                assert_eq!(tail_kind(modified), unknown(modified), "bit {bit}");
            }
        }
    }

    #[test]
    fn bra_decodes_forward_and_backward_relative_targets() {
        // Condition-code test T (0xF) in bits 0..5, as real encodings carry.
        let forward = decode_word([0xf, 0x0000_e240 << 16], 0x100);
        assert_eq!(
            forward.kind,
            InstructionKind::Branch(ops::Branch {
                target: target(0x100, 0),
                cc_test: 0xf
            })
        );

        let backward = decode_word(
            [(0x0000_0ff8 << 20) | 0xf, 0x0000_0fff | (0x0000_e240 << 16)],
            0x108,
        );
        assert_eq!(
            backward.kind,
            InstructionKind::Branch(ops::Branch {
                target: target(0x108, -8),
                cc_test: 0xf
            })
        );
    }

    #[test]
    fn disjoint_and_nested_sync_regions_lower_to_unique_targets() {
        // Ryujinx Decoder.cs:800-810,821-861 propagates producer targets;
        // InstEmitFlowControl.cs:239-262 folds unique pop targets to branches.
        // Encoding: Mesa sm50.rs:3032-3041; envytools gm107.c:1877-1878,1930-1933.
        let ssy = (0xe290u64 << 48) | (0x18u64 << 20) | 0xf;
        let sync = (0xf0f8u64 << 48) | 0xf;
        let nop = 0x50b0u64 << 48;
        let decode = |slots: [u64; 12]| {
            decode_stream(
                crate::QuietRun,
                &slots
                    .into_iter()
                    .flat_map(|r| [r as u32, (r >> 32) as u32])
                    .collect::<Vec<_>>(),
            )
        };
        let schedule = |at, to| {
            resolved(InstructionKind::Schedule(ops::Schedule {
                target: BranchTarget::reaching(at, to),
            }))
        };
        let sync_to = |at, to| {
            guarded(
                at,
                0,
                false,
                InstructionKind::Reconverge(ops::Reconverge {
                    target: BranchTarget::reaching(at, to),
                    cc_test: 0xf,
                    stack: TargetStack::Sync,
                }),
            )
        };
        let nop_at = |at| guarded(at, 0, false, InstructionKind::Nop(ops::Nop));
        let tail = [nop_at(56), nop_at(72), nop_at(80), nop_at(88)];
        let disjoint = decode([0, ssy, sync, nop, 0, ssy, sync, nop, 0, nop, nop, nop]);
        assert_eq!(
            disjoint,
            [
                [
                    at(8, schedule(8, 40)),
                    sync_to(16, 40),
                    nop_at(24),
                    at(40, schedule(40, 72)),
                    sync_to(48, 72),
                ]
                .as_slice(),
                &tail,
            ]
            .concat()
        );

        let outer = (0xe290u64 << 48) | (0x40u64 << 20) | 0xf;
        let inner = (0xe290u64 << 48) | (0x18u64 << 20) | 0xf;
        let nested = decode([0, outer, inner, sync, 0, nop, sync, nop, 0, nop, nop, nop]);
        assert_eq!(
            nested,
            [
                [
                    at(8, schedule(8, 80)),
                    at(16, schedule(16, 48)),
                    sync_to(24, 48),
                    nop_at(40),
                    sync_to(48, 80),
                ]
                .as_slice(),
                &tail,
            ]
            .concat()
        );
        assert_eq!(emitted(&nested), "Ok(())");
    }

    fn stack_program(kinds: Vec<InstructionKind>) -> Vec<Instruction> {
        kinds
            .into_iter()
            .enumerate()
            .map(|(i, kind)| at(i as u32 * 8, kind))
            .collect()
    }

    /// `kind`, a push, once its consumers are proven.
    fn resolved(kind: InstructionKind) -> InstructionKind {
        let mut xs = [at(0, kind)];
        let mut program = Program::of(&xs);
        program.settle_pushes();
        program.write(&mut xs);
        xs[0].kind
    }

    /// The push each stack has: SSY schedules the SYNC stack, PBK and PCNT
    /// push a target (Mesa sm50.rs:3038-3040,3072-3074,3095-3097).
    const PUSH: [fn(BranchTarget, TargetStack) -> InstructionKind; 3] = [
        |target, _| InstructionKind::Schedule(ops::Schedule { target }),
        |target, stack| InstructionKind::PushTarget(ops::PushTarget { target, stack }),
        |target, stack| InstructionKind::PushTarget(ops::PushTarget { target, stack }),
    ];

    fn push(stack: TargetStack, at: usize, to: usize) -> InstructionKind {
        PUSH[stack as usize](BranchTarget::reaching(at as u32 * 8, to as u32 * 8), stack)
    }

    fn pop(stack: TargetStack) -> InstructionKind {
        InstructionKind::PopJump(ops::PopJump {
            stack,
            cc_test: 0xf,
        })
    }

    fn resolved_pop(stack: TargetStack, at: usize, to: usize) -> InstructionKind {
        InstructionKind::Reconverge(ops::Reconverge {
            target: BranchTarget::reaching(at as u32 * 8, to as u32 * 8),
            cc_test: 0xf,
            stack,
        })
    }

    fn jump(at: usize, to: usize) -> InstructionKind {
        InstructionKind::Branch(ops::Branch {
            target: BranchTarget::reaching(at as u32 * 8, to as u32 * 8),
            cc_test: 0xf,
        })
    }

    #[test]
    fn linear_fallback_recovers_lost_proofs_by_physical_address() {
        // Mesa sm50.rs:3038-3040,3072-3074,3095-3097: push target bits
        // 20..44; :3050-3051,3061-3062,3084-3085: CC.T pop encodings.
        // The push targets a control word, and a dead real BRA remains in
        // main's full decode. Only the reachable consumer may be recovered.
        for (push_opcode, pop_opcode) in
            [(0xe290_u64, 0xf0f8_u64), (0xe2a0, 0xe340), (0xe2b0, 0xe350)]
        {
            let raw = [
                0,
                (push_opcode << 48) | (16 << 20) | 0xf,
                (pop_opcode << 48) | (7 << 16) | 0xf,
                (0xe240_u64 << 48) | (7 << 16) | 0xf,
                0,
                (0xe300_u64 << 48) | (7 << 16) | 0xf,
            ];
            let w = raw
                .into_iter()
                .flat_map(|r| [r as u32, (r >> 32) as u32])
                .collect::<Vec<_>>();
            let linear = decode_linear_stack_reference(&w, None);
            assert_eq!(
                linear.iter().map(|x| x.address).collect::<Vec<_>>(),
                [8, 16, 24, 40]
            );
            let expected = decode_stream(crate::QuietRun, &w);
            assert_eq!(
                expected.iter().map(|x| x.address).collect::<Vec<_>>(),
                [8, 16, 40]
            );
            // Fault injection: retain the original instructions while losing
            // their resolution flags, as happens when a conservative proof
            // fails. The independent main pass must recover this same IR.
            let mut unresolved = expected.clone();
            unresolved[1] = decode_word([w[4], w[5]], 16);
            unresolved[0] = decode_word([w[2], w[3]], 8);
            merge_linear_stack_results(&mut unresolved, &linear);
            assert_eq!(unresolved, expected);
            assert_eq!(emitted(&unresolved), "Ok(())");
        }
        let mut unresolved = stack_program(vec![
            pop(TargetStack::Sync),
            InstructionKind::Unknown(ops::Unknown { raw: [1, 2] }),
        ]);
        let mut linear = unresolved.clone();
        resolve_stack_jumps_linear(&mut linear);
        let original = unresolved.clone();
        merge_linear_stack_results(&mut unresolved, &linear);
        assert_eq!(
            unresolved, original,
            "an absent main proof must not invent control flow"
        );
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "stack resolvers disagree at byte 0x8")]
    fn conflicting_stack_proofs_trigger_debug_assertion() {
        let mut reachable = stack_program(vec![
            InstructionKind::Nop(ops::Nop),
            resolved_pop(TargetStack::Sync, 1, 2),
            InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
            InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
        ]);
        let mut linear = reachable.clone();
        linear[1].kind = resolved_pop(TargetStack::Sync, 1, 3);
        merge_linear_stack_results(&mut reachable, &linear);
    }

    /// A producerless SYNC: Mesa sm50.rs:3050-3051; it remains a real pop.
    const PRODUCERLESS_SYNC: [u32; 4] = {
        let pop = (0xf0f8_u64 << 48) | (7 << 16) | 0xf;
        [0, 0, pop as u32, (pop >> 32) as u32]
    };

    /// The child the opt-in test runs under each environment.
    #[test]
    #[ignore = "run by stack_trace_environment_is_opt_in"]
    fn stack_trace_environment_child() {
        // The child is the host: its entry takes its environment apart once.
        let ir = match std::env::var("SHADOWBOX_TRACE_STACK").as_deref() {
            Ok("1") => decode_stream(crate::TracedRun, &PRODUCERLESS_SYNC),
            _ => decode_stream(crate::QuietRun, &PRODUCERLESS_SYNC),
        };
        println!("{ir:?}");
    }

    #[test]
    fn stack_trace_environment_is_opt_in() {
        let ir = format!("{:?}", decode_stream(crate::QuietRun, &PRODUCERLESS_SYNC));
        let needles = [
            "unresolved SYNC (Sync) at byte 0x8",
            "block 0x8..0x10",
            "main linear fallback also leaves pop",
        ];
        // Exit status, how often stdout holds the decoded IR (the harness
        // adds its elapsed time), whether stderr says each needle, and its
        // length.
        let run = |vars: &[(&str, &str)]| {
            std::env::current_exe()
                .and_then(|exe| {
                    std::process::Command::new(exe)
                        .args([
                            "--exact",
                            "decoder::tests::stack_trace_environment_child",
                            "--ignored",
                            "--nocapture",
                        ])
                        .env_remove("SHADOWBOX_TRACE_STACK")
                        .envs(vars.iter().copied())
                        .output()
                })
                .map(|output| {
                    (
                        output.status.to_string(),
                        says(&output.stdout, &ir),
                        needles.map(|needle| says(&output.stderr, needle).min(1)),
                        output.stderr.len().min(1),
                    )
                })
                .map_err(|error| error.kind())
        };
        let status = "exit status: 0".to_string();
        for vars in [
            &[][..],
            &[("SHADOWBOX_TRACE_STACK", "0")],
            &[("SHADOWBOX_TRACE_STACK", "true")],
        ] {
            assert_eq!(run(vars), Ok((status.clone(), 1, [0; 3], 0)));
        }
        assert_eq!(
            run(&[("SHADOWBOX_TRACE_STACK", "1")]),
            Ok((status, 1, [1; 3], 1))
        );
    }

    #[test]
    fn stack_trace_reports_paths_pushes_underflow_and_ambiguous_targets() {
        // Synthetic IR with both PBK alternatives, an inner SSY/SYNC and a
        // later producerless CONT. Ordered pop matching: Ryujinx
        // Decoder.cs:824-850; unique targets: InstEmitFlowControl.cs:254-275.
        let program = stack_program(vec![
            InstructionKind::Branch(ops::Branch {
                target: BranchTarget::reaching(0, 32),
                cc_test: 0,
            }),
            push(TargetStack::Break, 1, 8),
            jump(2, 5),
            InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
            push(TargetStack::Break, 4, 9),
            push(TargetStack::Sync, 5, 7),
            pop(TargetStack::Sync),
            pop(TargetStack::Break),
            pop(TargetStack::Continue),
            pop(TargetStack::Continue),
        ]);
        let mut quiet = program.clone();
        resolve_stack_jumps_traced(&mut quiet, Quiet);
        let mut traced = program;
        let mut output = Vec::new();
        resolve_stack_jumps_traced(&mut traced, Traced::new(&mut output));
        assert_eq!(traced, quiet);
        let text = String::from_utf8_lossy(&output);
        for expected in [
            "unresolved BRK (Break) at byte 0x38",
            "unresolved CONT (Continue) at byte 0x40",
            "unresolved CONT (Continue) at byte 0x48",
            "multiple matching targets",
            "possible stack underflow",
            "block 0x0..0x8",
            "block 0x8..0x18",
            "block 0x20..0x28",
            "block 0x28..0x38",
            "push Break, target byte Some(64)",
            "push Break, target byte Some(72)",
            "push Sync, target byte Some(56)",
            "pop target byte 0x38 from push byte 0x28",
            "queue instruction byte",
            "older frames",
        ] {
            assert_ne!(says(&*text, expected), 0, "missing {expected:?}: {text}");
        }
    }

    #[test]
    fn stack_trace_reports_aborts_and_ignores_writer_errors() {
        let mut program = stack_program(vec![
            push(TargetStack::Sync, 0, 2),
            InstructionKind::Unknown(ops::Unknown { raw: [0, 0] }),
            pop(TargetStack::Sync),
        ]);
        let original = program.clone();
        let mut output = Vec::new();
        resolve_stack_jumps_traced(&mut program, Traced::new(&mut output));
        assert_eq!(program, original);
        let text = String::from_utf8_lossy(&output);
        assert_ne!(says(&*text, "unresolved SYNC (Sync) at byte 0x10"), 0);
        assert_ne!(says(&*text, "Unknown instruction at byte 0x8"), 0);
        assert_ne!(says(&*text, "all proofs discarded"), 0);
        assert_ne!(says(&*text, "block 0x0..0x10"), 0);
        // A full fixed-size writer returns WriteZero. Logging is best effort.
        resolve_stack_jumps_traced(&mut program, Traced::new([].as_mut_slice()));
        assert_eq!(program, original);
        program[1].kind = pop(TargetStack::Sync);
        program[2].kind = InstructionKind::Exit(ops::Exit { cc_test: 0xf });
        output.clear();
        resolve_stack_jumps_traced(&mut program, Traced::new(&mut output));
        assert_eq!(output.len(), 0, "resolved programs must stay quiet");
    }

    #[test]
    fn mixed_stack_kinds_follow_ordered_pushes_and_matching_pops() {
        // Ryujinx Decoder.cs:824-850 pops intervening kinds, not independent
        // per-kind stacks. The discarded push still participates in the
        // proven stack transition. No original instruction decodes as Nop.
        for outer in [TargetStack::Sync, TargetStack::Break, TargetStack::Continue] {
            for inner in [TargetStack::Sync, TargetStack::Break, TargetStack::Continue] {
                let mut xs = stack_program(vec![
                    push(outer, 0, 5),
                    push(inner, 1, 3),
                    pop(inner),
                    pop(outer),
                    InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
                    InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
                ]);
                resolve_stack_jumps(&mut xs);
                assert_eq!(xs[2].kind, resolved_pop(inner, 2, 3));
                assert_eq!(xs[3].kind, resolved_pop(outer, 3, 5));
                assert_eq!(emitted(&xs), "Ok(())");
            }
        }
        use TargetStack::{Break, Continue, Sync};
        for (outer, inner) in [
            (Sync, Break),
            (Sync, Continue),
            (Break, Sync),
            (Break, Continue),
            (Continue, Sync),
            (Continue, Break),
        ] {
            let mut xs = stack_program(vec![
                push(outer, 0, 4),
                push(inner, 1, 3),
                pop(outer),
                InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
                InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
            ]);
            resolve_stack_jumps(&mut xs);
            assert_eq!(xs[2].kind, resolved_pop(outer, 2, 4));
            assert_eq!(emitted(&xs), "Ok(())");
        }
    }

    #[test]
    fn disjoint_mixed_regions_and_control_word_push_targets_resolve() {
        // SSY relative target: Mesa sm50.rs:3038-3040; PBK/PCNT:
        // envytools gm107.c:1930-1935; pop pred/CC: :1877,1920-1921;
        // bundle control word and slot 0: Mesa sm50.rs:3418-3428.
        let nop = (0x50b0_u64 << 48) | (7 << 16);
        for (push, pop, stack) in [
            (0xe290_u64, 0xf0f8_u64, TargetStack::Sync),
            (0xe2a0, 0xe340, TargetStack::Break),
            (0xe2b0, 0xe350, TargetStack::Continue),
        ] {
            let words = [
                0,
                (push << 48) | (0x10 << 20) | 0xf,
                (pop << 48) | (7 << 16) | 0xf,
                nop,
                0,
                (0xe300_u64 << 48) | (7 << 16) | 0xf,
                nop,
                nop,
            ]
            .into_iter()
            .flat_map(|r| [r as u32, (r >> 32) as u32])
            .collect::<Vec<_>>();
            let xs = decode_stream(crate::QuietRun, &words);
            assert_eq!(
                xs[1].kind,
                InstructionKind::Reconverge(ops::Reconverge {
                    target: target(xs[1].address, 16),
                    cc_test: 0xf,
                    stack
                })
            );
            assert_eq!(emitted(&xs), "Ok(())");
        }
        let mut xs = stack_program(vec![
            push(TargetStack::Sync, 0, 2),
            pop(TargetStack::Sync),
            push(TargetStack::Break, 2, 4),
            pop(TargetStack::Break),
            InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
        ]);
        resolve_stack_jumps(&mut xs);
        assert_eq!(xs[1].kind, resolved_pop(TargetStack::Sync, 1, 2));
        assert_eq!(xs[3].kind, resolved_pop(TargetStack::Break, 3, 4));
        assert_eq!(emitted(&xs), "Ok(())");
    }

    #[test]
    fn cb1_excludes_underflow_edges_and_traces_the_local_proof() {
        // MOV cbuf1: Mesa sm50.rs:1931-1951,361-368. BRX register+base:
        // envytools gm107.c:1943; Ryujinx InstEmitFlowControl.cs:38-40.
        // Saved cb1 supplies exact words: DiskCacheGpuAccessor.cs:62-69.
        let program = stack_program(vec![
            InstructionKind::Mov(ops::Mov {
                dst: Register(2),
                src: Source::ConstantBuffer {
                    binding: 1,
                    offset: 0,
                },
                quad_lanes: 0xf,
            }),
            InstructionKind::BrxRegister(ops::BrxRegister {
                displacement: Register(2),
                base: BrxBase::decode(8, 0),
                cc_test: 0xf,
            }),
            push(TargetStack::Sync, 2, 4),
            pop(TargetStack::Sync),
            InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
        ]);
        // From byte 8, displacement 0 enters the push; 8 bypasses it.
        let traced = |cb1: &[u32]| {
            let mut xs = program.clone();
            let mut trace = Vec::new();
            crate::brx::resolve_traced(&mut xs, cb1, Some(&mut trace), |_, _, _| {});
            let mut quiet = program.clone();
            crate::brx::resolve(&mut quiet, cb1, |_, _, _| {});
            assert_eq!(xs, quiet, "tracing cannot alter a target proof");
            (xs, String::from_utf8_lossy(&trace).into_owned())
        };
        for (cb1, destination, pop_kind, emission) in [
            (
                0,
                "[Some(10)]",
                resolved_pop(TargetStack::Sync, 3, 4),
                "Ok(())".to_string(),
            ),
            (8, "[Some(18)]", pop(TargetStack::Sync), unresolved(0x18)),
        ] {
            let (mut xs, text) = traced(&[cb1]);
            assert_ne!(says(&*text, "prune BRX histories at byte 0x8"), 0, "{text}");
            assert_ne!(
                says(&*text, "finite reaching-value proof under recovered graph"),
                0
            );
            assert_ne!(says(&*text, "immutable cb1 snapshot 1 words"), 0);
            assert_ne!(says(&*text, destination), 0, "{text}");
            resolve_stack_jumps(&mut xs);
            assert_eq!(xs[3].kind, pop_kind);
            assert_eq!(emitted(&xs), emission);
        }
        let (mut xs, text) = traced(&[]);
        assert_eq!(text.len(), 0, "failed proofs must exclude no edges");
        assert_eq!(xs[1].kind, program[1].kind);
        resolve_stack_jumps(&mut xs);
        assert_eq!(xs[3].kind, pop(TargetStack::Sync));
        assert_eq!(emitted(&xs), unresolved(0x18));
    }

    #[test]
    fn predicate_pruning_is_local_traced_and_invalidated_by_writes() {
        // BRA !P0 bypasses SSY, but SYNC P0 cannot then execute unless P0
        // changes. Predication: Ryujinx Translator.cs:374-383. The proof is
        // calculated here, removing runtime stack tags (InstEmitFlowControl.cs:258-262).
        let mut program = stack_program(vec![
            jump(0, 3),
            push(TargetStack::Sync, 1, 4),
            InstructionKind::Nop(ops::Nop),
            pop(TargetStack::Sync),
            InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
        ]);
        program[0].predicate = edge::GUARD[8];
        program[3].predicate = edge::GUARD[0];
        type Resolve = fn(&mut [Instruction], &mut Vec<u8>);
        let traced: [Resolve; 2] = [
            |xs, trace| resolve_stack_jumps_traced(xs, Traced::new(trace)),
            |xs, trace| resolve_stack_jumps_linear_ordered_traced(xs, Traced::new(trace)),
        ];
        for resolve in traced {
            let mut xs = program.clone();
            let mut trace = Vec::new();
            resolve(&mut xs, &mut trace);
            assert_eq!(xs[3].kind, resolved_pop(TargetStack::Sync, 3, 4));
            assert_ne!(
                says(
                    &trace,
                    "prune executed history at byte 0x18: predicate P0 proven false"
                ),
                0
            );
        }
        // Put an actual predicate writer on the bypass path. Its outcome
        // is unconstrained by the earlier BRA. ISETP: Mesa sm50.rs:1505-1541;
        // Ryujinx InstEmitIntegerComparison.cs:183-191. No proof can discard underflow.
        program[0].kind = jump(0, 2);
        program[2].kind = decode_word([0x0007_0007, 0x5b64_0380], 16).kind;
        assert_eq!(
            program[2].kind,
            InstructionKind::ISetP(ops::ISetP {
                dst: [0, 7],
                lhs: Register(0),
                rhs: Source::Register(Register(0)),
                compare: IntCompare::Relation(Relation::Eq),
                signedness: Signedness::Unsigned,
                mode: IntegerCompareMode::Ordinary,
                set_op: SetOp::And,
                accum: 7,
                accum_negate: false,
                abs: [false; 2],
                negate: [false; 2],
            })
        );
        assert!(program[2].kind.may_write_predicate());
        for resolve in [
            resolve_stack_jumps as fn(&mut [Instruction]),
            resolve_stack_jumps_linear_ordered,
        ] {
            let mut xs = program.clone();
            resolve(&mut xs);
            assert_eq!(xs[3].kind, pop(TargetStack::Sync));
            assert_eq!(emitted(&xs), unresolved(0x18));
        }
    }

    #[test]
    fn pt_excludes_only_the_impossible_predicate_edge() {
        // PT is true: Ryujinx OperandHelper.cs:49-51. Keep the guarded
        // instruction in IR, but exclude its impossible execution history.
        let mut program = stack_program(vec![
            push(TargetStack::Sync, 0, 3),
            pop(TargetStack::Sync),
            pop(TargetStack::Sync),
            InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
        ]);
        program[1].predicate = edge::GUARD[15];
        let mut trace = Vec::new();
        resolve_stack_jumps_traced(&mut program, Traced::new(&mut trace));
        assert_eq!(program[1].kind, pop(TargetStack::Sync));
        assert_eq!(program[2].kind, resolved_pop(TargetStack::Sync, 2, 3));
        assert_ne!(
            says(
                &trace,
                "prune executed history at byte 0x8: predicate P7 proven true"
            ),
            0
        );
    }

    #[test]
    fn unrelated_dispatcher_cannot_justify_a_push_bypass() {
        let program = stack_program(vec![
            jump(0, 2),
            push(TargetStack::Sync, 1, 3),
            pop(TargetStack::Sync),
            InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
            InstructionKind::BrxRegister(ops::BrxRegister {
                displacement: Register(2),
                base: BrxBase::decode(32, 0),
                cc_test: 0xf,
            }),
        ]);
        for resolve in [
            resolve_stack_jumps as fn(&mut [Instruction]),
            resolve_stack_jumps_linear_ordered,
        ] {
            let mut xs = program.clone();
            resolve(&mut xs);
            assert_eq!(xs[2].kind, pop(TargetStack::Sync));
            assert_eq!(emitted(&xs), unresolved(0x10));
        }
        assert!(crate::stack::resolve(&program).is_none());
    }

    #[test]
    fn dispatcher_does_not_justify_empty_stack_histories() {
        // BRX contributes every slot. A matching push on one path cannot
        // prove an empty-stack path impossible (InstEmitFlowControl.cs:38-40).
        let dispatcher = InstructionKind::BrxRegister(ops::BrxRegister {
            displacement: Register(2),
            base: BrxBase::decode(0, 0),
            cc_test: 0xf,
        });
        let program = stack_program(vec![
            dispatcher,
            push(TargetStack::Sync, 1, 4),
            pop(TargetStack::Sync),
            InstructionKind::Nop(ops::Nop),
            InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
        ]);
        let mut linear = program.clone();
        resolve_stack_jumps_linear_ordered(&mut linear);
        assert_eq!(linear[2].kind, pop(TargetStack::Sync));
        let mut finite = program.clone();
        let mut trace = Vec::new();
        resolve_stack_jumps_traced(&mut finite, Traced::new(&mut trace));
        assert_eq!(finite[2].kind, pop(TargetStack::Sync));
        let text = String::from_utf8_lossy(&trace);
        assert_ne!(says(&*text, "unresolved SYNC"), 0);
        assert_ne!(says(&*text, "possible stack underflow"), 0);
        assert_eq!(says(&*text, "prune"), 0);
        assert!(crate::stack::resolve(&program).is_none());
        let mut combined = program;
        resolve_stack_jumps(&mut combined);
        assert_eq!(combined[2].kind, pop(TargetStack::Sync));
        assert_eq!(emitted(&combined), unresolved(0x10));
        assert_eq!(
            crate::emitter::emit(&combined)
                .map(drop)
                .map_err(|error| says(error.to_string(), "missing cb1 data or control-flow proof")),
            Err(1)
        );

        for (kinds, at) in [
            (
                vec![
                    dispatcher,
                    pop(TargetStack::Sync),
                    InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
                ],
                1,
            ),
            (
                vec![
                    dispatcher,
                    push(TargetStack::Break, 1, 4),
                    pop(TargetStack::Sync),
                    InstructionKind::Nop(ops::Nop),
                    InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
                ],
                2,
            ),
            (
                vec![
                    dispatcher,
                    push(TargetStack::Sync, 1, 6),
                    jump(2, 4),
                    push(TargetStack::Sync, 3, 7),
                    pop(TargetStack::Sync),
                    InstructionKind::Nop(ops::Nop),
                    InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
                    InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
                ],
                4,
            ),
        ] {
            let mut xs = stack_program(kinds);
            resolve_stack_jumps(&mut xs);
            assert_eq!(xs[at].kind, pop(TargetStack::Sync), "{xs:?}");
            assert_eq!(emitted(&xs), unresolved(at as u32 * 8));
        }
    }

    #[test]
    fn push_bypass_and_repeated_pop_without_push_remain_unresolved() {
        let mut xs = stack_program(vec![
            jump(0, 2),
            push(TargetStack::Sync, 1, 3),
            pop(TargetStack::Sync),
            InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
        ]);
        xs[0].predicate = edge::GUARD[0];
        resolve_stack_jumps(&mut xs);
        assert_eq!(xs[2].kind, pop(TargetStack::Sync));
        assert_eq!(emitted(&xs), unresolved(0x10));

        let mut xs = stack_program(vec![
            push(TargetStack::Sync, 0, 2),
            pop(TargetStack::Sync),
            jump(2, 1),
        ]);
        resolve_stack_jumps(&mut xs);
        assert_eq!(xs[1].kind, pop(TargetStack::Sync));
        assert_eq!(emitted(&xs), unresolved(0x8));
    }

    #[test]
    fn repushing_on_loop_backedge_and_same_guard_push_are_proven() {
        let mut xs = stack_program(vec![
            push(TargetStack::Sync, 0, 2),
            pop(TargetStack::Sync),
            jump(2, 0),
            InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
        ]);
        xs[2].predicate = edge::GUARD[0];
        resolve_stack_jumps(&mut xs);
        assert_eq!(xs[1].kind, resolved_pop(TargetStack::Sync, 1, 2));
        assert_eq!(emitted(&xs), "Ok(())");

        let mut xs = stack_program(vec![
            jump(0, 2),
            push(TargetStack::Sync, 1, 3),
            pop(TargetStack::Sync),
            InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
        ]);
        xs.push(at(32, InstructionKind::Exit(ops::Exit { cc_test: 0xf })));
        xs[0].predicate = edge::GUARD[8];
        xs[2].predicate = edge::GUARD[0];
        resolve_stack_jumps(&mut xs);
        assert_eq!(xs[2].kind, resolved_pop(TargetStack::Sync, 2, 3));
        assert_eq!(emitted(&xs), "Ok(())");
    }

    #[test]
    fn multiple_reaching_stack_targets_remain_unresolved() {
        let mut xs = stack_program(vec![
            jump(0, 3),
            push(TargetStack::Sync, 1, 5),
            jump(2, 4),
            push(TargetStack::Sync, 3, 6),
            pop(TargetStack::Sync),
            InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
            InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
        ]);
        xs[0].predicate = edge::GUARD[0];
        resolve_stack_jumps(&mut xs);
        assert_eq!(xs[4].kind, pop(TargetStack::Sync));
        assert_eq!(emitted(&xs), unresolved(0x20));
    }

    #[test]
    fn new_frame_predecessors_revisit_pop_consumers() {
        // A CC-tested branch keeps both paths without refining predicates
        // (envytools gm107.c:1944-1945). Both paths share the same inner push
        // frame. Its older frame arrives late, after the pop was first visited.
        for older in [
            InstructionKind::Nop(ops::Nop),
            push(TargetStack::Break, 4, 9),
        ] {
            let mut xs = stack_program(vec![
                InstructionKind::Branch(ops::Branch {
                    target: BranchTarget::reaching(0, 32),
                    cc_test: 0,
                }),
                push(TargetStack::Break, 1, 8),
                jump(2, 5),
                InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
                older,
                push(TargetStack::Sync, 5, 7),
                pop(TargetStack::Sync),
                pop(TargetStack::Break),
                InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
                InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
            ]);
            resolve_stack_jumps(&mut xs);
            assert_eq!(xs[6].kind, resolved_pop(TargetStack::Sync, 6, 7));
            // One alternative underflows, the other has a different target.
            // Neither allows the earlier single-target proof for the BRK.
            assert_eq!(xs[7].kind, pop(TargetStack::Break));
        }
    }

    #[test]
    fn repeated_push_frames_preserve_possible_underflow() {
        // One or more iterations push SSY, but two later SYNCs underflow on
        // the one-iteration path. A cyclic predecessor graph must preserve
        // that empty-stack history, rather than assuming unlimited pushes.
        let mut xs = stack_program(vec![
            push(TargetStack::Sync, 0, 4),
            jump(1, 0),
            pop(TargetStack::Sync),
            InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
            pop(TargetStack::Sync),
        ]);
        xs[1].kind = InstructionKind::Branch(ops::Branch {
            target: BranchTarget::reaching(8, 0),
            cc_test: 0,
        });
        resolve_stack_jumps(&mut xs);
        assert_eq!(xs[2].kind, resolved_pop(TargetStack::Sync, 2, 4));
        assert_eq!(xs[4].kind, pop(TargetStack::Sync));
        assert_eq!(emitted(&xs), unresolved(0x20));
    }

    #[test]
    fn resolved_brx_edges_participate_in_stack_proof() {
        let mut targets = [0; 32];
        targets[1] = 8;
        let mut xs = stack_program(vec![
            push(TargetStack::Sync, 0, 4),
            InstructionKind::brx(
                Register(2),
                BrxBase::decode(8, 0),
                targets[0],
                &targets[1..2],
            ),
            pop(TargetStack::Sync),
            pop(TargetStack::Sync),
            InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
        ]);
        resolve_stack_jumps(&mut xs);
        assert_eq!(xs[2].kind, resolved_pop(TargetStack::Sync, 2, 4));
        assert_eq!(xs[3].kind, resolved_pop(TargetStack::Sync, 3, 4));
        assert_eq!(emitted(&xs), "Ok(())");
        let mut bypass = xs;
        bypass[2].kind = pop(TargetStack::Sync);
        bypass[3].kind = pop(TargetStack::Sync);
        bypass[0].kind = InstructionKind::brx(
            Register(2),
            BrxBase::decode(0, 0),
            targets[0],
            &targets[1..2],
        );
        resolve_stack_jumps(&mut bypass);
        assert_eq!(bypass[2].kind, pop(TargetStack::Sync));
    }

    #[test]
    fn resolved_brx_bundle_targets_participate_in_stack_proof() {
        // Physical slots: Mesa sm50.rs:3418-3428. BRX relative base/register:
        // envytools gm107.c:1942-1943; control-word skip: Ryujinx
        // Decoder.cs:265-270. SSY/SYNC target semantics: Ryujinx
        // InstEmitFlowControl.cs:254-275. Both paths consume the same push.
        let mut targets = [0; 32];
        targets[..2].copy_from_slice(&[8, 32]);
        let mut xs = stack_program(vec![
            InstructionKind::Schedule(ops::Schedule {
                target: target(8, 56),
            }),
            InstructionKind::brx(
                Register(2),
                BrxBase::decode(16, 0),
                targets[0],
                &targets[1..2],
            ),
            InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
            pop(TargetStack::Sync),
            InstructionKind::Nop(ops::Nop),
            pop(TargetStack::Sync),
            InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
        ]);
        for (x, address) in xs.iter_mut().zip([8, 16, 24, 40, 48, 56, 72]) {
            x.address = address;
        }
        resolve_stack_jumps(&mut xs);
        assert_eq!(
            xs[0].kind,
            resolved(InstructionKind::Schedule(ops::Schedule {
                target: target(8, 56),
            }))
        );
        assert_eq!(
            xs[3].kind,
            InstructionKind::Reconverge(ops::Reconverge {
                target: target(xs[3].address, 24),
                cc_test: 0xf,
                stack: TargetStack::Sync
            })
        );
        assert_eq!(
            xs[5].kind,
            InstructionKind::Reconverge(ops::Reconverge {
                target: target(xs[5].address, 8),
                cc_test: 0xf,
                stack: TargetStack::Sync
            })
        );
        assert_eq!(emitted(&xs), "Ok(())");
    }

    #[test]
    fn sync_jumps_to_its_ssy_target() {
        // Bundle at 0x00: control word, then SSY at 0x08 with offset 0x18
        // (target 0x28), SYNC at 0x10, NOP at 0x18; bundle at 0x20: control word, NOP 0x28, SYNC 0x30.
        let ssy = (0xe290u64 << 48) | (0x18u64 << 20) | 0xf;
        let sync = (0xf0f8u64 << 48) | 0xf;
        let nop = 0x50b0u64 << 48;
        let words: Vec<u32> = [0, ssy, sync, nop, 0, nop, sync, nop]
            .iter()
            .flat_map(|w| [*w as u32, (*w >> 32) as u32])
            .collect();
        let xs = decode_stream(crate::QuietRun, &words);
        assert_eq!(xs[1].address, 0x10);
        assert_eq!(
            xs[1].kind,
            InstructionKind::Reconverge(ops::Reconverge {
                target: BranchTarget::reaching(0x10, 0x28),
                cc_test: 0xf,
                stack: TargetStack::Sync
            })
        );
        // The SYNC at 0x30 lies outside [0x08, 0x28) and stays unresolved.
        assert_eq!(xs[4].address, 0x30);
        assert_eq!(xs[4].kind, pop(TargetStack::Sync));
        // A prior unresolved pop must prevent proof of later consumers,
        // without panicking or pretending the first pop has no effect.
        let words: Vec<u32> = [0, sync, ssy, sync, 0, nop, nop, nop]
            .iter()
            .flat_map(|w| [*w as u32, (*w >> 32) as u32])
            .collect();
        let xs = decode_stream(crate::QuietRun, &words);
        assert_eq!(xs[0].kind, pop(TargetStack::Sync));
        assert_eq!(xs[2].kind, pop(TargetStack::Sync));
    }

    #[test]
    fn decodes_vote_layout_from_mesa_encoding() {
        let raw = (0x50dau64 << 48) | (3 << 45) | (5 << 39) | (1 << 42) | 7;
        assert_eq!(
            tail_kind(raw),
            InstructionKind::Vote(ops::Vote {
                op: VoteOp::Eq,
                ballot: Register(7),
                vote: 3,
                pred: Predicate {
                    register: 5,
                    inverted: true
                },
            })
        );
    }

    #[test]
    fn conversions_follow_mesa_fields() {
        let word = |opcode: u64, bits: u64| {
            let value = (opcode << 48) | bits;
            tail_kind(value)
        };
        let convert = |kind| {
            InstructionKind::Convert(ops::Convert {
                kind,
                dst: Register(0),
                src: crate::ir::Source::Register(Register(0)),
            })
        };
        let i2f = |signed, src_size, round, byte, modifier| {
            convert(Conversion::I2F {
                signed,
                dst_size: 2,
                src_size,
                round,
                byte,
                abs: modifier,
                negate: modifier,
            })
        };
        use crate::ir::RoundingMode::{NearestEven, NegativeInfinity, Zero};
        // 32-bit sizes: dst 8..9 = 2, src 10..11 = 2.
        let sizes = 0xa << 8;
        assert_eq!(
            word(0x5ca8, sizes | (1 << 42) | (3 << 39)),
            convert(Conversion::F2F {
                source: F2fSource::Float32,
                destination: F2fDestination::Float32,
                integer_round: Some(Zero),
                abs: false,
                negate: false,
                saturate: false,
                ftz: false,
            })
        );
        assert_eq!(
            word(0x5cb0, sizes | (1 << 12) | (1 << 39)),
            convert(Conversion::F2I {
                signed: true,
                dst_size: 2,
                src_size: 2,
                round: NegativeInfinity,
                high: false,
                ftz: false,
                abs: false,
                negate: false,
            })
        );
        assert_eq!(
            word(0x5cb8, sizes | (1 << 13)),
            i2f(true, 2, NearestEven, 0, false)
        );
        // Unsupported I2F field combinations remain Unknown.
        assert_eq!(word(0x5ca8, 0x9 << 8), unknown(0x5ca8 << 48 | 0x9 << 8));
        assert_eq!(
            word(
                0x5cb8,
                (2 << 8) | (1 << 10) | (1 << 13) | (2 << 41) | (1 << 45) | (1 << 49)
            ),
            i2f(true, 1, NearestEven, 2, true)
        );
        // nvdisasm SM52 prints byte 1 of a 16-bit source as INVALIDSIZE1.
        let odd = (0x5cb8 << 48) | (2 << 8) | (1 << 10) | (1 << 41);
        assert_eq!(tail_kind(odd), unknown(odd));
        assert_eq!(
            word(0x5cb8, sizes | (1 << 39)),
            i2f(false, 2, NegativeInfinity, 0, false)
        );
        let raw = (0x5cb8 << 48) | (2 << 8) | (2 << 10) | (1 << 41);
        assert_eq!(tail_kind(raw), unknown(raw));
    }

    #[test]
    fn unsupported_conversion_fields_remain_unknown() {
        // Field citations are at the conversion arms above. Never silently
        // discard CC, unsupported rounding or a selection beyond 32 bits.
        let unknown = |opcode: u64, fields: u64| {
            let raw = opcode << 48 | 7 << 16 | fields;
            assert_eq!(tail_kind(raw), unknown(raw), "{raw:016x}");
        };
        for opcode in [0x5ce0, 0x4ce0, 0x38e0] {
            unknown(opcode, 3 << 8 | 2 << 10);
            unknown(opcode, 2 << 8 | 3 << 10);
            unknown(opcode, 1 << 10 | 3 << 41);
            unknown(opcode, 2 << 10 | 1 << 41);
        }
        for opcode in [0x5cb8, 0x4cb8, 0x38b8] {
            unknown(opcode, 0xa00 | 1 << 47);
            unknown(opcode, 0x800); // Destination float size0 is reserved.
            unknown(opcode, 0xd00); // F16 cannot span 32 bits (Mesa:1868-1872).
            unknown(opcode, 0x600 | 3 << 41);
        }
        for opcode in [0x4cb8, 0x38b8] {
            unknown(opcode, 0xe00);
        }
        for opcode in [0x5cb0, 0x4cb0, 0x38b0] {
            unknown(opcode, 0xa00 | 1 << 47);
            unknown(opcode, 0xa00 | 1 << 41);
            unknown(opcode, 0x600 | 1 << 44);
            unknown(opcode, 0xb00);
            unknown(opcode, 0x800);
        }
    }

    #[test]
    fn lop_predicate_output_modes_decode_for_all_short_forms() {
        use crate::ir::{PredicateOp, Source};
        for (opcode, rhs) in [
            (0x5c40_u64, Source::Register(Register(0))),
            (
                0x4c40,
                Source::ConstantBuffer {
                    binding: 0,
                    offset: 0,
                },
            ),
            (0x3840, Source::Immediate(0)),
        ] {
            for (mode, op) in [
                (1_u64, PredicateOp::T),
                (2, PredicateOp::Z),
                (3, PredicateOp::Nz),
            ] {
                let value = (opcode << 48) | (2 << 48) | (mode << 44);
                assert_eq!(
                    tail_kind(value),
                    InstructionKind::Lop(ops::Lop {
                        dst: Register(0),
                        lhs: Register(0),
                        rhs,
                        op: crate::ir::LogicOp::And,
                        invert: [false, false],
                        predicate_output: Some(PredicateOutput { op, dst: 2 }),
                        extended: false,
                        cc: false,
                    })
                );
            }
        }
    }

    #[test]
    fn iscadd_and_negated_iadd_follow_envytools() {
        use crate::ir::Source;
        // ISCADD R1, R2 << 5, -R3 (srcs[1] negate at 48, shift 39..44).
        let iscadd = (0x5c18u64 << 48) | 1 | (2 << 8) | (3 << 20) | (5 << 39) | (1 << 48);
        assert_eq!(
            tail_kind(iscadd),
            InstructionKind::ShiftAdd(ops::ShiftAdd {
                dst: Register(1),
                lhs: Register(2),
                rhs: Source::Register(Register(3)),
                shift: 5,
                negate: [false, true],
                cc: false,
            })
        );
        let iadd = |dst, lhs, rhs, x, negate| {
            InstructionKind::IAdd(ops::IAdd {
                dst: Register(dst),
                lhs: Register(lhs),
                rhs: Source::Register(Register(rhs)),
                cc: false,
                x,
                negate,
            })
        };
        // IADD source A negation at bit 49 (envytools gm107.c:2015).
        let raw = (0x5c12u64 << 48) | 1 | (2 << 8) | (3 << 20);
        assert_eq!(tail_kind(raw), iadd(1, 2, 3, false, [true, false]));
        // IADD.X consumes C from bit 43 (envytools gm107.c:2015; Mesa
        // sm50.rs:1291-1343).
        assert_eq!(
            tail_kind((0x5c10u64 << 48) | (1 << 43)),
            iadd(0, 0, 0, true, [false, false])
        );
    }

    #[test]
    fn texture_encodings_preserve_mode_fields() {
        let texture = |op, dim, mask| ops::Texture {
            op,
            dst: Register(0),
            src0: Register(0),
            src1: Register(0),
            dim,
            mask,
            offset: false,
            scalar: false,
            packed_f16: false,
            lod_clamp: false,
            multisample: false,
            depth_compare: false,
            nodep: false,
            scalar_dst: None,
            bound_handle: None,
        };
        assert_eq!(
            tail_kind((0xdeb8u64 << 48) | (1 << 31) | (1 << 50)),
            InstructionKind::Texture(ops::Texture {
                depth_compare: true,
                ..texture(TexOp::Sample { lod: 0 }, 0, 1)
            })
        );
        assert_eq!(
            tail_kind((0xdd38u64 << 48) | (1 << 31) | (2 << 28) | (1 << 50) | (1 << 55)),
            InstructionKind::Texture(ops::Texture {
                multisample: true,
                ..texture(TexOp::Fetch { lod: true }, 2, 1)
            })
        );
        assert_eq!(
            tail_kind((0xdd38u64 << 48) | (1 << 54)),
            unknown((0xdd38u64 << 48) | (1 << 54))
        );
        assert_eq!(
            tail_kind((0xdef8u64 << 48) | (2 << 38) | (1 << 50)),
            InstructionKind::Texture(ops::Texture {
                depth_compare: true,
                ..texture(
                    TexOp::Gather {
                        component: 2,
                        offsets: GatherOffsets::Zero,
                    },
                    0,
                    0
                )
            })
        );
        assert_eq!(
            tail_kind((0xdf50u64 << 48) | (1 << 22) | (1 << 31)),
            InstructionKind::Texture(ops::Texture {
                src1: Register(255),
                ..texture(TexOp::Query { query: 1 }, 2, 1)
            })
        );
        assert_eq!(
            tail_kind(0xd800u64 << 48),
            InstructionKind::Texture(ops::Texture {
                scalar: true,
                scalar_dst: Some(Register(0)),
                bound_handle: Some(0),
                ..texture(TexOp::Sample { lod: 1 }, 0, 7)
            })
        );
    }

    #[test]
    fn fmul_ignores_bits_46_and_49() {
        // nvdisasm prints 0x5c68400000070000 and 0x5c6a000000070000
        // as plain FMUL: neither bit is a modifier, so both decode as
        // the word without them.
        let plain = tail_kind(0x5c68u64 << 48);
        assert_eq!(tail_kind((0x5c68u64 << 48) | (1 << 46)), plain);
        assert_eq!(tail_kind((0x5c68u64 << 48) | (1 << 49)), plain);
    }

    #[test]
    fn register_brx_keeps_a_reachable_partial_tail() {
        // Envytools gm107.c:1943; Ryujinx InstEmitFlowControl.cs:38-40.
        let raw = 0xe250_u64 << 48 | 7 << 16 | 2 << 8 | 15;
        let xs = [decode_word([raw as u32, (raw >> 32) as u32], 8)];
        assert_eq!(
            format!("{:?}", validate_reachable_words(&xs, 20)),
            "Err(PartialWord)"
        );
        assert_eq!(format!("{:?}", validate_reachable_words(&xs, 24)), "Ok(())");

        // A non-T condition adds fallthrough without weakening the arbitrary
        // target: both paths can reach the partial tail.
        let conditional = [at(
            8,
            InstructionKind::BrxRegister(ops::BrxRegister {
                displacement: Register(2),
                base: BrxBase::decode(8, 0),
                cc_test: 0,
            }),
        )];
        assert_eq!(
            format!("{:?}", validate_reachable_words(&conditional, 20)),
            "Err(PartialWord)"
        );
        assert_eq!(
            format!("{:?}", validate_reachable_words(&conditional, 24)),
            "Ok(())"
        );
    }

    #[test]
    fn conditional_exit_keeps_a_reachable_partial_tail() {
        // EXIT condition0..4: envytools gm107.c:1925,426-454.
        // Conditional fallthrough: Ryujinx InstEmitFlowControl.cs:153-186.
        for (cc_test, result) in [
            (0, "Err(PartialWord)"),
            (1, "Err(PartialWord)"),
            (15, "Ok(())"),
            (30, "Err(PartialWord)"),
            (31, "Err(PartialWord)"),
        ] {
            let xs = [at(8, InstructionKind::Exit(ops::Exit { cc_test }))];
            assert_eq!(format!("{:?}", validate_reachable_words(&xs, 20)), result);
        }
    }

    #[test]
    fn conditional_exit_retains_a_following_stack_pop() {
        // Ryujinx InstEmitFlowControl.cs:170-185 keeps the fallthrough.
        let mut xs = stack_program(vec![
            push(TargetStack::Sync, 0, 3),
            InstructionKind::Exit(ops::Exit { cc_test: 0 }),
            pop(TargetStack::Sync),
            InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
        ]);
        resolve_stack_jumps(&mut xs);
        assert_eq!(xs[2].kind, resolved_pop(TargetStack::Sync, 2, 3));
        assert_eq!(xs[1].kind, InstructionKind::Exit(ops::Exit { cc_test: 0 }));
    }
}
