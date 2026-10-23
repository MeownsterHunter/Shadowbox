//! A conservative pass for modules produced by our emitter.
//!
//! Ryujinx Translation/Optimizations/Optimizer.cs:56-131 combines constant
//! folding, copy propagation and removal of unused values to a fixed point.
//! We do that directly on SPIR-V, promoting unescaped Function scalars to SSA.
//! No Maxwell decoding or shader operation mapping takes place here.
//!
//! Every opcode reaches its behavior through tables indexed by its number,
//! built once at compile time. Per-instruction facts are computed as values
//! and written through masks; a state with work of its own reaches that work
//! through a two-entry table indexed by the state.
mod cross_block;
mod layouts;

use crate::fast_hash::HashMap;
use crate::spirv::Op;
use std::iter::{once, repeat_with};

/// `yes` where `c` holds and `no` elsewhere: a value chosen by arithmetic.
fn pick<T: Copy>(c: bool, no: T, yes: T) -> T {
    [no, yes][usize::from(c)]
}

/// The table slot an operand word names. A word that is a literal for this
/// opcode still lands inside the table; a masked write keeps that slot as it
/// was, and a masked read is discarded.
fn slot(word: u32, len: usize) -> usize {
    (word as usize).min(len - 1)
}

/// Opcode tables cover the numbers below 401; the last entry stands for every
/// higher number, which no supported grammar uses.
const OPCODES: usize = 402;

fn code(op: Op) -> usize {
    usize::from(op).min(OPCODES - 1)
}

const fn at(op: Op) -> usize {
    op.number() as usize
}

/// An opcode's operand grammar. `results` and `ids` mark operand positions;
/// every `step`-th position from `tail` on is an ID too. The grammar admits
/// `min..=max` operand words.
#[derive(Clone, Copy)]
struct Layout {
    results: u64,
    ids: u64,
    tail: usize,
    step: usize,
    min: usize,
    max: usize,
    pure: bool,
}

/// An instruction without a result operand.
const NO_RESULT: u64 = 0;
/// The result is the first operand: a type, label or other untyped definition.
const RESULT_AT_0: u64 = 1;
/// The result follows its result type.
const RESULT_AT_1: u64 = 2;
/// A tail past every word count the 16-bit field holds: no ID tail.
const NO_TAIL: usize = 1 << 16;
/// No upper bound on the operand count.
const UNBOUNDED: usize = usize::MAX;

impl Layout {
    const fn new(results: u64, ids: u64, tail: usize, min: usize, max: usize, pure: bool) -> Self {
        Self {
            results,
            ids,
            tail,
            step: 1,
            min,
            max,
            pure,
        }
    }

    /// The grammar no operand count satisfies: an unsupported form.
    const REJECT: Self = Self::new(NO_RESULT, 0, NO_TAIL, 1, 0, false);

    /// This grammar where the operand words satisfy it, `REJECT` elsewhere.
    fn admit(self, holds: bool) -> Self {
        pick(holds, Self::REJECT, self)
    }

    fn admits(self, count: usize) -> bool {
        (count >= self.min) & (count <= self.max)
    }
}

const HEAD: usize = 6;

/// Operand words. Most emitted instructions have at most six; those live in
/// `head` with zero padding. A longer list lives whole in `heap`, and `head`
/// mirrors its first six words, so each of the first six reads directly.
/// Operand classification still comes solely from the cited grammar layouts.
#[derive(Clone, PartialEq, Eq, Hash)]
struct Operands {
    head: [u32; HEAD],
    len: usize,
    heap: Vec<u32>,
}

impl Operands {
    fn new(args: &[u32]) -> Self {
        let len = args.len();
        let mut head = [0; HEAD];
        for (word, &arg) in head.iter_mut().zip(args) {
            *word = arg;
        }
        Self {
            head,
            len,
            heap: args[..len * usize::from(len > HEAD)].to_vec(),
        }
    }

    fn words_mut(&mut self) -> &mut [u32] {
        let len = self.len;
        let mut views: [&mut [u32]; 2] = [&mut self.heap[..], &mut self.head[..len.min(HEAD)]];
        std::mem::take(&mut views[usize::from(len <= HEAD)])
    }

    /// Restores `head` as the mirror of the first six words.
    fn mirror(&mut self) {
        let mut head = [0; HEAD];
        for (word, &arg) in head.iter_mut().zip(self.iter()) {
            *word = arg;
        }
        self.head = head;
    }
}

impl std::ops::Deref for Operands {
    type Target = [u32];
    fn deref(&self) -> &[u32] {
        let views: [&[u32]; 2] = [&self.heap, &self.head[..self.len.min(HEAD)]];
        views[usize::from(self.len <= HEAD)]
    }
}

/// Where an instruction's IDs sit. Validation-only min/max lengths need not
/// travel with every instruction. All positions fit: Mesa
/// src/compiler/spirv/spirv.h:48-49 defines the 16-bit word-count field.
#[derive(Clone, Copy)]
struct OperandLayout {
    /// ID positions among the first 64 words.
    ids: u64,
    /// The first tail ID past those 64 words; at or past the end for none.
    wide: u32,
    step: u8,
    /// Position of the result, zero for an instruction without one.
    result: u8,
    /// All ones for an instruction with a result, zero for one without.
    defines: u32,
    /// The result follows a result type.
    typed: bool,
    pure: bool,
}

impl OperandLayout {
    fn new(layout: Layout, count: usize) -> Self {
        let start = layout.tail;
        let limit = ((1u128 << count.min(64)) - 1) as u64;
        let tail = (u64::MAX << start.min(63)) * u64::from(start < 64);
        // A switch alternates literal and label words from its tail on.
        let stride = [u64::MAX, 0x5555_5555_5555_5555u64 << (start % 2)][layout.step - 1];
        let after = start.max(64);
        Self {
            ids: (layout.ids | (tail & stride)) & limit,
            wide: (after + (layout.step - 1) * ((after - start) % 2)) as u32,
            step: layout.step as u8,
            result: (layout.results.trailing_zeros() % 64) as u8,
            defines: u32::MAX * u32::from(layout.results != NO_RESULT),
            typed: layout.results == RESULT_AT_1,
            pure: layout.pure,
        }
    }
}

#[derive(Clone)]
struct Instruction {
    op: Op,
    args: Operands,
    layout: OperandLayout,
    removed: bool,
}

impl Instruction {
    fn with_layout(op: Op, args: &[u32], layout: Layout) -> Self {
        Self {
            op,
            args: Operands::new(args),
            layout: OperandLayout::new(layout, args.len()),
            removed: false,
        }
    }

    /// An instruction this optimizer builds, in a grammar it emits.
    fn new(op: Op, args: &[u32]) -> Self {
        Self::with_layout(op, args, layout(op, args))
    }

    /// The result ID. SPIR-V reserves ID 0, which names no value; an
    /// instruction without a result reports it.
    fn result(&self) -> u32 {
        self.args.head[usize::from(self.layout.result)] & self.layout.defines
    }

    /// One of the first six operand words, zero past the end.
    fn word(&self, at: usize) -> u32 {
        self.args.head[at]
    }

    /// Visits each ID operand position. Set bits of the fixed mask come first;
    /// long interfaces and switches continue with the grammar tail, including
    /// alternating switch literal/label pairs.
    fn inputs(&self, mut visit: impl FnMut(usize)) {
        let mut bits = self.layout.ids;
        for _ in 0..bits.count_ones() {
            visit(bits.trailing_zeros() as usize);
            bits &= bits - 1;
        }
        let tail = self.layout.wide as usize..self.args.len;
        for at in tail.step_by(usize::from(self.layout.step)) {
            visit(at);
        }
    }

    fn rewrite(&mut self, replacements: &[u32]) {
        let mut bits = self.layout.ids;
        let wide = self.layout.wide as usize;
        let step = usize::from(self.layout.step);
        let words = self.args.words_mut();
        for _ in 0..bits.count_ones() {
            let at = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            words[at] = resolve(replacements, words[at]);
        }
        for at in (wide..words.len()).step_by(step) {
            words[at] = resolve(replacements, words[at]);
        }
        self.args.mirror();
    }

    fn annotation(&self) -> bool {
        (self.op == Op::Decorate) | (self.op == Op::MemberDecorate)
    }
}

/// Follows replacement links to an ID mapped to itself. A link can name an
/// ID that a later rewrite replaced in turn, so the walk length is the data's.
fn resolve(replacements: &[u32], mut id: u32) -> u32 {
    while replacements[id as usize] != id {
        id = replacements[id as usize];
    }
    id
}

/// How the parse edge reads an opcode's operands, from the operand words and
/// a zero-padded copy of the first six.
type Form = fn(Op, &[u32], &[u32; HEAD]) -> Layout;

// The generated layouts distinguish ID operands from literals. This is load
// bearing: a constant's bits, a component index or an OpSwitch case can equal
// an ID. Exceptional grammars in
// /usr/include/spirv/unified1/spirv.core.grammar.json:
// :227-235 OpExtInstImport, :260-269 OpEntryPoint, :272-280 OpExecutionMode,
// :704-713 OpLoad, :716-724 OpStore, :838-845 OpDecorate,
// :848-856 OpMemberDecorate, :997-1180 OpImage*, :11926-12026 image operands,
// :2799-2808 OpLoopMerge, :2850-2859 OpSwitch.
// OpSwitch selectors emitted here are uint32, so each case literal is one word.
fn layout(op: Op, a: &[u32]) -> Layout {
    let mut head = [0; HEAD];
    for (word, &arg) in head.iter_mut().zip(a) {
        *word = arg;
    }
    (KINDS[code(op)].form)(op, a, &head)
}

fn generated(op: Op, _: &[u32], _: &[u32; HEAD]) -> Layout {
    layouts::layout(op)
}

/// GLSL.std.450 is the one extended instruction set the emitter imports.
fn ext_inst_import(_: Op, _: &[u32], h: &[u32; HEAD]) -> Layout {
    Layout::new(RESULT_AT_0, 0, NO_TAIL, 5, 5, true)
        .admit(h[1..5] == [0x4c534c47, 0x6474732e, 0x3035342e, 0])
}

/// The name is a nul-terminated string from word 2; interface IDs follow the
/// word holding its first zero byte.
fn entry_point(_: Op, a: &[u32], _: &[u32; HEAD]) -> Layout {
    let mut end = 0;
    let mut found = false;
    for (at, &word) in a.iter().enumerate().skip(2) {
        // Nonzero exactly for a word with a zero byte: the borrow of each
        // byte subtraction is the meaning here.
        let zero = (word.wrapping_sub(0x0101_0101) & !word & 0x8080_8080) != 0;
        end = pick(zero & !found, end, at);
        found |= zero;
    }
    Layout::new(NO_RESULT, 2, end + 1, end + 1, UNBOUNDED, false).admit(found)
}

fn execution_mode(_: Op, a: &[u32], h: &[u32; HEAD]) -> Layout {
    let n = a.len();
    let holds = ((n == 2) & ((h[1] == 7) | (h[1] == 12))) | ((n == 5) & (h[1] == 17));
    Layout::new(NO_RESULT, 1, NO_TAIL, n, n, false).admit(holds)
}

// Optional MemoryAccess operands are deliberately unsupported.
fn load(_: Op, _: &[u32], _: &[u32; HEAD]) -> Layout {
    Layout::new(RESULT_AT_1, 5, NO_TAIL, 3, 3, false)
}

fn store(_: Op, _: &[u32], _: &[u32; HEAD]) -> Layout {
    Layout::new(NO_RESULT, 3, NO_TAIL, 2, 2, false)
}

/// Decorations without an operand, then those with one literal operand.
const BARE_DECORATIONS: u64 = 1 | 1 << 2 | 1 << 13 | 1 << 14 | 1 << 42;
const LITERAL_DECORATIONS: u64 = 1 << 1 | 1 << 6 | 1 << 11 | 1 << 30 | 1 << 33 | 1 << 34 | 1 << 39;

fn decorate(_: Op, a: &[u32], h: &[u32; HEAD]) -> Layout {
    let n = a.len();
    let named = |set: u64| (h[1] < 64) & ((set >> (h[1] % 64)) & 1 != 0);
    let holds = ((n == 2) & named(BARE_DECORATIONS)) | ((n == 3) & named(LITERAL_DECORATIONS));
    Layout::new(NO_RESULT, 1, NO_TAIL, n, n, false).admit(holds)
}

fn member_decorate(_: Op, _: &[u32], h: &[u32; HEAD]) -> Layout {
    Layout::new(NO_RESULT, 1, NO_TAIL, 4, 4, false).admit(h[2] == 35)
}

// OpPhi value/parent pairs are both IDs (grammar :2788-2796).
fn phi(_: Op, a: &[u32], _: &[u32; HEAD]) -> Layout {
    Layout::new(RESULT_AT_1, 1, 2, 4, UNBOUNDED, true).admit(a.len().is_multiple_of(2))
}

fn loop_merge(_: Op, _: &[u32], h: &[u32; HEAD]) -> Layout {
    Layout::new(NO_RESULT, 3, NO_TAIL, 3, 3, false).admit(h[2] == 0)
}

fn switch(_: Op, a: &[u32], _: &[u32; HEAD]) -> Layout {
    Layout {
        step: 2,
        ..Layout::new(NO_RESULT, 3, 3, 2, UNBOUNDED, false)
    }
    .admit(a.len().is_multiple_of(2))
}

/// OpImageSample* through OpImageWrite (grammar :997-1180): the fixed words,
/// then an image operands mask and the operands it selects, both optional.
/// The mask follows the coordinate, and the Dref forms carry one more operand
/// first.
#[derive(Clone, Copy)]
struct ImageForm {
    results: u64,
    /// ID operands among the fixed words.
    ids: u64,
    /// Number of fixed words: the index of the mask.
    fixed: usize,
    /// First word after the mask.
    tail: usize,
    /// Mask bits that select no operand word.
    hints: u32,
    /// The explicit-lod forms need the mask for their Lod or Grad operand.
    mask_required: bool,
}

impl ImageForm {
    const SAMPLE: Self = Self {
        results: RESULT_AT_1,
        ids: 0b1101,
        fixed: 4,
        tail: 5,
        hints: 0,
        mask_required: false,
    };
    const SAMPLE_EXPLICIT: Self = Self {
        mask_required: true,
        ..Self::SAMPLE
    };
    const DREF: Self = Self {
        results: RESULT_AT_1,
        ids: 0b1_1101,
        fixed: 5,
        tail: 6,
        hints: 0,
        mask_required: false,
    };
    const DREF_EXPLICIT: Self = Self {
        mask_required: true,
        ..Self::DREF
    };
    // Nontemporal is a literal image-write hint with no IdRef operand (Mesa
    // spirv.core.grammar.json:11990-11994). Parse it so CS stores can use the
    // normal optimizer instead of forcing the whole module through the
    // unknown-op fallback.
    const WRITE: Self = Self {
        results: NO_RESULT,
        ids: 0b111,
        fixed: 3,
        tail: 4,
        hints: 0x4000,
        mask_required: false,
    };

    fn layout(self, a: &[u32], h: &[u32; HEAD]) -> Layout {
        let n = a.len();
        let present = n > self.fixed;
        let mask = h[self.fixed] * u32::from(present);
        let known = mask & !(0xff | self.hints) == 0;
        // Grad (bit 2) selects two words; every other operand bit selects one.
        let words = (mask & !self.hints).count_ones() as usize + usize::from(mask & 4 != 0);
        let selected = n.max(self.fixed + 1) - (self.fixed + 1);
        let holds = pick(present, !self.mask_required, known & (selected == words));
        Layout::new(
            self.results,
            self.ids,
            self.tail,
            self.fixed,
            UNBOUNDED,
            false,
        )
        .admit(holds)
    }
}

fn sample(_: Op, a: &[u32], h: &[u32; HEAD]) -> Layout {
    ImageForm::SAMPLE.layout(a, h)
}

fn sample_explicit(_: Op, a: &[u32], h: &[u32; HEAD]) -> Layout {
    ImageForm::SAMPLE_EXPLICIT.layout(a, h)
}

fn dref(_: Op, a: &[u32], h: &[u32; HEAD]) -> Layout {
    ImageForm::DREF.layout(a, h)
}

fn dref_explicit(_: Op, a: &[u32], h: &[u32; HEAD]) -> Layout {
    ImageForm::DREF_EXPLICIT.layout(a, h)
}

fn write(_: Op, a: &[u32], h: &[u32; HEAD]) -> Layout {
    ImageForm::WRITE.layout(a, h)
}

/// The 32-bit scalar a declaration names. `Other` is every other type and
/// every value, which this pass neither folds nor forwards.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Scalar {
    Other,
    Bool,
    Int,
    Float,
}

/// The scalar type an instruction declares; `Other` for the rest.
fn declared_scalar(i: &Instruction) -> Scalar {
    let wide = i.word(1) == 32;
    let which = usize::from(i.op == Op::TypeBool)
        | (usize::from((i.op == Op::TypeInt) & wide) * 2)
        | (usize::from((i.op == Op::TypeFloat) & wide) * 3);
    [Scalar::Other, Scalar::Bool, Scalar::Int, Scalar::Float][which]
}

#[derive(Clone, Copy)]
struct Constant {
    ty: u32,
    bits: u32,
    known: bool,
}

const UNKNOWN: Constant = Constant {
    ty: 0,
    bits: 0,
    known: false,
};

/// An exact 32-bit integer or boolean operation, or a bit-preserving cast.
#[derive(Clone, Copy)]
struct Rule {
    /// The folded bits, and whether the operation defines them.
    eval: fn(u32, u32, u32) -> (u32, bool),
    /// Whether the result type and the first operand's type admit folding.
    admits: fn(Scalar, Scalar) -> bool,
    /// The leading operands (from word 2) that must be known constants.
    operands: usize,
}

fn never(_: Scalar, _: Scalar) -> bool {
    false
}

fn int_operand(_: Scalar, x: Scalar) -> bool {
    x == Scalar::Int
}

fn bool_operand(_: Scalar, x: Scalar) -> bool {
    x == Scalar::Bool
}

fn cast(ty: Scalar, x: Scalar) -> bool {
    (ty != Scalar::Bool) & (x != Scalar::Bool)
}

impl Rule {
    const NONE: Self = Self {
        eval: |_, _, _| (0, false),
        admits: never,
        operands: 1,
    };

    const fn int(operands: usize, eval: fn(u32, u32, u32) -> (u32, bool)) -> Self {
        Self {
            eval,
            admits: int_operand,
            operands,
        }
    }

    const fn boolean(operands: usize, eval: fn(u32, u32, u32) -> (u32, bool)) -> Self {
        Self {
            eval,
            admits: bool_operand,
            operands,
        }
    }
}

/// Bits `y..y + z` of `x`, sign extended where `signed`. Widths past the word
/// are undefined in SPIR-V and stay unfolded.
fn bit_field(x: u32, y: u32, z: u32, signed: bool) -> (u32, bool) {
    let defined = (y <= 32) & (z <= 32 - y.min(32));
    let shift = (32 - z.min(32)) & 31;
    let top = (x >> (y & 31)) << shift;
    let value = [top >> shift, ((top as i32) >> shift) as u32][usize::from(signed)];
    (pick(z == 0, value, 0), defined)
}

// Only exact 32-bit integer/boolean operations and bit-preserving casts fold.
// Mesa nak/ir.rs:3802-3821 truncates integer sums to u32; :4390-4400 folds
// AND/OR/XOR bitwise. Ryujinx ConstantFolding.cs:19-49,59-97 does the same.
// SPIR-V opcodes and operand order are from the cited generated grammar.
// Oversized shifts and bitfields are undefined in SPIR-V and stay unfolded.
fn fold(i: &Instruction, constants: &[Constant], types: &[Scalar]) -> (u32, bool) {
    let rule = KINDS[code(i.op)].rule;
    let ty = types[slot(i.word(0), types.len())];
    let [x, y, z] = [2, 3, 4].map(|at| constants[slot(i.word(at), constants.len())]);
    let known = [x.known, y.known, z.known]
        .iter()
        .take(rule.operands)
        .fold(true, |all, &k| all & k);
    let (bits, defined) = (rule.eval)(x.bits, y.bits, z.bits);
    let admitted = (rule.admits)(ty, types[x.ty as usize]);
    (bits, (ty != Scalar::Other) & known & admitted & defined)
}

/// What each opcode is to this pass.
#[derive(Clone, Copy)]
struct Kind {
    form: Form,
    /// Capabilities the opcode needs whatever its operands (grammar
    /// :1224-1276 queries, :3872-4047 subgroups).
    capabilities: &'static [u32],
    /// Position of the image operands mask.
    mask_at: usize,
    /// Image operand bits needing ImageGatherExtended: Offset, ConstOffsets.
    offsets: u32,
    rule: Rule,
}

impl Kind {
    const PLAIN: Self = Self {
        form: generated,
        capabilities: &[],
        mask_at: 0,
        offsets: 0,
        rule: Rule::NONE,
    };

    const fn image(form: Form, mask_at: usize) -> Self {
        Self {
            form,
            mask_at,
            offsets: 0x30,
            ..Self::PLAIN
        }
    }
}

static KINDS: [Kind; OPCODES] = {
    let mut k = [Kind::PLAIN; OPCODES];
    k[at(Op::ExtInstImport)].form = ext_inst_import;
    k[at(Op::EntryPoint)].form = entry_point;
    k[at(Op::ExecutionMode)].form = execution_mode;
    k[at(Op::Load)].form = load;
    k[at(Op::Store)].form = store;
    k[at(Op::Decorate)].form = decorate;
    k[at(Op::MemberDecorate)].form = member_decorate;
    k[at(Op::Phi)].form = phi;
    k[at(Op::LoopMerge)].form = loop_merge;
    k[at(Op::Switch)].form = switch;
    k[at(Op::ImageSampleImplicitLod)] = Kind::image(sample, 4);
    k[at(Op::ImageSampleProjImplicitLod)] = Kind::image(sample, 4);
    k[at(Op::ImageFetch)] = Kind::image(sample, 4);
    k[at(Op::ImageRead)] = Kind {
        capabilities: &[55],
        ..Kind::image(sample, 4)
    };
    k[at(Op::ImageSampleExplicitLod)] = Kind::image(sample_explicit, 4);
    k[at(Op::ImageSampleProjExplicitLod)] = Kind::image(sample_explicit, 4);
    k[at(Op::ImageSampleDrefImplicitLod)] = Kind::image(dref, 5);
    k[at(Op::ImageSampleProjDrefImplicitLod)] = Kind::image(dref, 5);
    k[at(Op::ImageGather)] = Kind::image(dref, 5);
    k[at(Op::ImageDrefGather)] = Kind::image(dref, 5);
    k[at(Op::ImageSampleDrefExplicitLod)] = Kind::image(dref_explicit, 5);
    k[at(Op::ImageSampleProjDrefExplicitLod)] = Kind::image(dref_explicit, 5);
    k[at(Op::ImageWrite)] = Kind {
        capabilities: &[56],
        ..Kind::image(write, 3)
    };
    k[at(Op::ImageQueryFormat)].capabilities = &[50];
    k[at(Op::ImageQueryOrder)].capabilities = &[50];
    k[at(Op::ImageQuerySizeLod)].capabilities = &[50];
    k[at(Op::ImageQuerySize)].capabilities = &[50];
    k[at(Op::ImageQueryLod)].capabilities = &[50];
    k[at(Op::ImageQueryLevels)].capabilities = &[50];
    k[at(Op::ImageQuerySamples)].capabilities = &[50];
    k[at(Op::GroupNonUniformAll)].capabilities = &[61, 62];
    k[at(Op::GroupNonUniformAny)].capabilities = &[61, 62];
    k[at(Op::GroupNonUniformAllEqual)].capabilities = &[61, 62];
    k[at(Op::GroupNonUniformBroadcast)].capabilities = &[61, 64];
    k[at(Op::GroupNonUniformBroadcastFirst)].capabilities = &[61, 64];
    k[at(Op::GroupNonUniformBallot)].capabilities = &[61, 64];
    k[at(Op::GroupNonUniformInverseBallot)].capabilities = &[61, 64];
    k[at(Op::GroupNonUniformBallotBitExtract)].capabilities = &[61, 64];
    k[at(Op::GroupNonUniformBallotBitCount)].capabilities = &[61, 64];
    k[at(Op::GroupNonUniformBallotFindLSB)].capabilities = &[61, 64];
    k[at(Op::GroupNonUniformBallotFindMSB)].capabilities = &[61, 64];
    k[at(Op::GroupNonUniformShuffle)].capabilities = &[61, 65];
    k[at(Op::GroupNonUniformShuffleXor)].capabilities = &[61, 65];
    k[at(Op::GroupNonUniformElect)].capabilities = &[61];
    k[at(Op::GroupNonUniformShuffleUp)].capabilities = &[61];
    k[at(Op::GroupNonUniformShuffleDown)].capabilities = &[61];
    k[at(Op::GroupNonUniformIAdd)].capabilities = &[61];
    k[at(Op::GroupNonUniformFAdd)].capabilities = &[61];
    k[at(Op::GroupNonUniformIMul)].capabilities = &[61];
    k[at(Op::GroupNonUniformFMul)].capabilities = &[61];
    k[at(Op::GroupNonUniformSMin)].capabilities = &[61];
    k[at(Op::GroupNonUniformUMin)].capabilities = &[61];
    k[at(Op::GroupNonUniformFMin)].capabilities = &[61];
    k[at(Op::GroupNonUniformSMax)].capabilities = &[61];
    k[at(Op::GroupNonUniformUMax)].capabilities = &[61];
    k[at(Op::GroupNonUniformFMax)].capabilities = &[61];
    k[at(Op::GroupNonUniformBitwiseAnd)].capabilities = &[61];
    k[at(Op::GroupNonUniformBitwiseOr)].capabilities = &[61];
    k[at(Op::GroupNonUniformBitwiseXor)].capabilities = &[61];
    k[at(Op::GroupNonUniformLogicalAnd)].capabilities = &[61];
    k[at(Op::GroupNonUniformLogicalOr)].capabilities = &[61];
    k[at(Op::GroupNonUniformLogicalXor)].capabilities = &[61];
    k[at(Op::GroupNonUniformQuadBroadcast)].capabilities = &[61];
    k[at(Op::GroupNonUniformQuadSwap)].capabilities = &[61];
    k[at(Op::Bitcast)].rule = Rule {
        eval: |x, _, _| (x, true),
        admits: cast,
        operands: 1,
    };
    k[at(Op::SNegate)].rule = Rule::int(1, |x, _, _| (x.wrapping_neg(), true));
    k[at(Op::LogicalNot)].rule = Rule::boolean(1, |x, _, _| (u32::from(x == 0), true));
    k[at(Op::Not)].rule = Rule::int(1, |x, _, _| (!x, true));
    k[at(Op::BitReverse)].rule = Rule::int(1, |x, _, _| (x.reverse_bits(), true));
    k[at(Op::BitCount)].rule = Rule::int(1, |x, _, _| (x.count_ones(), true));
    k[at(Op::Select)].rule = Rule::boolean(3, |x, y, z| (pick(x != 0, z, y), x <= 1));
    k[at(Op::LogicalEqual)].rule = Rule::boolean(2, |x, y, _| (u32::from(x == y), true));
    k[at(Op::LogicalNotEqual)].rule = Rule::boolean(2, |x, y, _| (u32::from(x != y), true));
    k[at(Op::LogicalOr)].rule = Rule::boolean(2, |x, y, _| (x | y, true));
    k[at(Op::LogicalAnd)].rule = Rule::boolean(2, |x, y, _| (x & y, true));
    k[at(Op::IAdd)].rule = Rule::int(2, |x, y, _| (x.wrapping_add(y), true));
    k[at(Op::ISub)].rule = Rule::int(2, |x, y, _| (x.wrapping_sub(y), true));
    k[at(Op::IMul)].rule = Rule::int(2, |x, y, _| (x.wrapping_mul(y), true));
    k[at(Op::IEqual)].rule = Rule::int(2, |x, y, _| (u32::from(x == y), true));
    k[at(Op::INotEqual)].rule = Rule::int(2, |x, y, _| (u32::from(x != y), true));
    k[at(Op::UGreaterThan)].rule = Rule::int(2, |x, y, _| (u32::from(x > y), true));
    k[at(Op::SGreaterThan)].rule =
        Rule::int(2, |x, y, _| (u32::from((x as i32) > (y as i32)), true));
    k[at(Op::UGreaterThanEqual)].rule = Rule::int(2, |x, y, _| (u32::from(x >= y), true));
    k[at(Op::SGreaterThanEqual)].rule =
        Rule::int(2, |x, y, _| (u32::from((x as i32) >= (y as i32)), true));
    k[at(Op::ULessThan)].rule = Rule::int(2, |x, y, _| (u32::from(x < y), true));
    k[at(Op::SLessThan)].rule = Rule::int(2, |x, y, _| (u32::from((x as i32) < (y as i32)), true));
    k[at(Op::ULessThanEqual)].rule = Rule::int(2, |x, y, _| (u32::from(x <= y), true));
    k[at(Op::SLessThanEqual)].rule =
        Rule::int(2, |x, y, _| (u32::from((x as i32) <= (y as i32)), true));
    k[at(Op::ShiftRightLogical)].rule = Rule::int(2, |x, y, _| (x >> (y & 31), y < 32));
    k[at(Op::ShiftRightArithmetic)].rule =
        Rule::int(2, |x, y, _| (((x as i32) >> (y & 31)) as u32, y < 32));
    k[at(Op::ShiftLeftLogical)].rule = Rule::int(2, |x, y, _| (x << (y & 31), y < 32));
    k[at(Op::BitwiseOr)].rule = Rule::int(2, |x, y, _| (x | y, true));
    k[at(Op::BitwiseXor)].rule = Rule::int(2, |x, y, _| (x ^ y, true));
    k[at(Op::BitwiseAnd)].rule = Rule::int(2, |x, y, _| (x & y, true));
    k[at(Op::BitFieldSExtract)].rule = Rule::int(3, |x, y, z| bit_field(x, y, z, true));
    k[at(Op::BitFieldUExtract)].rule = Rule::int(3, |x, y, z| bit_field(x, y, z, false));
    k
};

#[cfg(test)]
fn parse(words: &[u32]) -> Vec<Instruction> {
    let mut instructions = Vec::new();
    parse_into(words, &mut instructions, &mut Vec::new());
    instructions
}

/// Reads `words` into `instructions`. True where the words form whole
/// instructions, each in a grammar this optimizer knows.
fn parse_into(words: &[u32], instructions: &mut Vec<Instruction>, starts: &mut Vec<usize>) -> bool {
    instructions.clear();
    starts.clear();
    let mut next = 5;
    let mut whole = true;
    for (at, &word) in words.iter().enumerate().skip(5) {
        let header = at == next;
        let count = (word >> 16) as usize;
        whole &= !(header & (count == 0));
        next += usize::from(header) * count.max(1);
        starts.extend(once(at).take(usize::from(header)));
    }
    whole &= next == words.len();
    DECODE[usize::from(whole)](words, instructions, starts)
}

/// Decodes the words at `starts` into instructions; false when they do not decode.
type DecodeWords = fn(&[u32], &mut Vec<Instruction>, &[usize]) -> bool;
const DECODE: [DecodeWords; 2] = [broken, decode];

fn broken(_: &[u32], _: &mut Vec<Instruction>, _: &[usize]) -> bool {
    false
}

fn decode(words: &[u32], instructions: &mut Vec<Instruction>, starts: &[usize]) -> bool {
    instructions.reserve(starts.len());
    let mut known = true;
    for &at in starts {
        let op = Op::of(words[at]);
        let args = &words[at + 1..at + (words[at] >> 16) as usize];
        let layout = layout(op, args);
        known &= layout.admits(args.len());
        instructions.push(Instruction::with_layout(op, args, layout));
    }
    known
}

// OpTypePointer / OpVariable fields: Khronos grammar :433-441, :679-689.
// Function storage class is 7 (:13779-13781).
fn scalar_variables(instructions: &[Instruction], types: &[Scalar], bound: usize) -> Vec<bool> {
    let mut pointers = Vec::new();
    let mut local = Vec::new();
    scalar_variables_into(instructions, types, bound, &mut local, &mut pointers);
    local
}

fn scalar_variables_into(
    instructions: &[Instruction],
    types: &[Scalar],
    bound: usize,
    local: &mut Vec<bool>,
    pointers: &mut Vec<bool>,
) {
    reset(pointers, bound, false);
    reset(local, bound, false);
    for i in instructions {
        let scalar = types[slot(i.word(2), bound)] != Scalar::Other;
        pointers[slot(i.word(0), bound)] |= (i.op == Op::TypePointer) & (i.word(1) == 7) & scalar;
        let variable = (i.op == Op::Variable) & (i.word(2) == 7);
        local[slot(i.word(1), bound)] |= variable & pointers[slot(i.word(0), bound)];
    }
    // A pointer passed to access chains, calls, copies, or any other operation
    // escapes. Only direct unqualified loads and stores can be forwarded.
    for i in instructions {
        i.inputs(|at| {
            let direct = ((i.op == Op::Load) & (at == 2)) | ((i.op == Op::Store) & (at == 0));
            local[i.args[at] as usize] &= direct;
        });
    }
}

#[derive(Default)]
struct UnusedScratch {
    uses: Vec<usize>,
    definitions: Vec<usize>,
    defined: Vec<bool>,
    dead: Vec<bool>,
    pending: Vec<usize>,
}

fn eliminate_unused(instructions: &mut [Instruction], local: &[bool], scratch: &mut UnusedScratch) {
    let UnusedScratch {
        uses,
        definitions,
        defined,
        dead,
        pending,
    } = scratch;
    uses.fill(0);
    defined.fill(false);
    dead.fill(false);
    pending.clear();
    for (index, i) in instructions.iter().enumerate() {
        let live = !i.removed;
        let result = i.result() as usize;
        let defines = live & (i.layout.defines != 0);
        definitions[result] = pick(defines, definitions[result], index);
        defined[result] |= defines;
        let counted = usize::from(live & !i.annotation());
        i.inputs(|at| uses[i.args[at] as usize] += counted);
    }
    for (id, (&n, &d)) in uses.iter().zip(defined.iter()).enumerate() {
        pending.extend(once(id).take(usize::from((n == 0) & d)));
    }
    let mut cursor = 0;
    while cursor < pending.len() {
        let id = pending[cursor];
        cursor += 1;
        let index = definitions[id];
        let i = &mut instructions[index];
        let pure = i.layout.pure
            | ((i.op == Op::Load) & local[slot(i.word(2), local.len())])
            | ((i.op == Op::Variable) & local[id]);
        let drop = !i.removed & pure;
        i.removed |= drop;
        let i = &instructions[index];
        let gone = usize::from(drop);
        i.inputs(|at| {
            let source = i.args[at] as usize;
            uses[source] -= gone;
            let empty = drop & (uses[source] == 0) & defined[source];
            pending.extend(once(source).take(usize::from(empty)));
        });
    }
    for i in instructions.iter() {
        dead[i.result() as usize] |= i.removed & (i.layout.defines != 0);
    }
    for i in instructions.iter_mut() {
        i.removed |= i.annotation() & dead[slot(i.word(0), dead.len())];
    }
}

// Capability predicates are conservative. Unknown capabilities stay declared.
// /usr/include/spirv/unified1/spirv.core.grammar.json:17165-17168,
// :17266-17292, :17306-17310, :17336-17347, :17372-17400 defines these
// capability values; :354-369 gives OpTypeImage fields, :1224-1276 queries,
// :3872-4047 subgroup requirements, :11926-12026 image operand requirements.
// Keep image dimension capabilities for every declared image type of that
// dimension, read or not. Unknown capabilities stay declared.
const CAPABILITIES: usize = 66;

const PRUNABLE: [bool; CAPABILITIES] = {
    let mut prunable = [false; CAPABILITIES];
    prunable[25] = true;
    prunable[43] = true;
    prunable[44] = true;
    prunable[45] = true;
    prunable[47] = true;
    prunable[50] = true;
    prunable[55] = true;
    prunable[56] = true;
    prunable[61] = true;
    prunable[62] = true;
    prunable[64] = true;
    prunable[65] = true;
    prunable
};

/// Marks in `needed` the prunable capabilities `i` needs.
fn note_capabilities(i: &Instruction, needed: &mut [bool]) {
    let kind = &KINDS[code(i.op)];
    for &capability in kind.capabilities {
        needed[capability as usize] = true;
    }
    let mask = i.word(kind.mask_at) * u32::from(i.args.len > kind.mask_at);
    needed[25] |= mask & kind.offsets != 0;
    let image = i.op == Op::TypeImage;
    let [_, _, dim, _, arrayed, _] = i.args.head;
    needed[43] |= image & (dim == 0);
    needed[44] |= image & (dim == 0);
    needed[45] |= image & (dim == 3) & (arrayed == 1);
    needed[47] |= image & (dim == 5);
    let builtin = (i.op == Op::Decorate) & (i.word(1) == 11);
    let value = i.word(2);
    needed[61] |= builtin & (36..=41).contains(&value);
    // Mask input decorations need Ballot even without a ballot opcode:
    // Mesa src/compiler/spirv/spirv.core.grammar.json:16221-16258.
    needed[64] |= builtin & (4416..=4420).contains(&value);
}

/// The value a constant instruction declares, and whether `i` is one.
fn constant_value(i: &Instruction) -> (u32, bool) {
    let bits = pick(
        i.op == Op::ConstantTrue,
        pick(i.op == Op::Constant, 0, i.word(2)),
        1,
    );
    let constant = (i.op == Op::ConstantTrue)
        | (i.op == Op::ConstantFalse)
        | ((i.op == Op::Constant) & (i.args.len == 3));
    (bits, constant)
}

/// Optimise only the known emitter grammar. A future unrecognised opcode or
/// operand form leaves the whole module unchanged, so adding emitter features
/// cannot silently turn their semantics into a guessed operation. False where
/// a switch selector is not a 32-bit integer; the module then stays as it is.
fn optimize_local(
    instructions: &mut Vec<Instruction>,
    bound: usize,
    scratch: &mut LocalScratch,
) -> bool {
    let integral = local_facts(instructions, bound, scratch);
    LOCAL[usize::from(integral)](instructions, bound, scratch);
    integral
}

const LOCAL: [fn(&mut Vec<Instruction>, usize, &mut LocalScratch); 2] = [settled, forward];

fn settled(_: &mut Vec<Instruction>, _: usize, _: &mut LocalScratch) {}

/// Types, decorations and constants; true where every switch selector is a
/// 32-bit integer.
fn local_facts(instructions: &[Instruction], bound: usize, scratch: &mut LocalScratch) -> bool {
    let LocalScratch {
        types,
        constants,
        canonical,
        decorated,
        result_types,
        constant_at,
        ..
    } = scratch;
    reset(types, bound, Scalar::Other);
    reset(constants, bound, UNKNOWN);
    canonical.clear();
    reset(decorated, bound, false);
    for i in instructions {
        let scalar = declared_scalar(i);
        let k = slot(i.word(0), bound);
        types[k] = pick(scalar != Scalar::Other, types[k], scalar);
        decorated[k] |= i.op == Op::Decorate;
    }
    constant_at.clear();
    for (index, i) in instructions.iter().enumerate() {
        let (_, constant) = constant_value(i);
        let typed = (i.op != Op::Constant) | (types[slot(i.word(0), bound)] != Scalar::Other);
        constant_at.extend(once(index).take(usize::from(constant & typed)));
    }
    for &index in constant_at.iter() {
        let i = &instructions[index];
        let (bits, _) = constant_value(i);
        constants[i.word(1) as usize] = Constant {
            ty: i.word(0),
            bits,
            known: true,
        };
        canonical.entry((i.word(0), bits)).or_insert(i.word(1));
    }
    reset(result_types, bound, Scalar::Other);
    for i in instructions {
        let k = slot(i.word(1), bound);
        result_types[k] = pick(
            i.layout.typed,
            result_types[k],
            types[slot(i.word(0), bound)],
        );
    }
    instructions.iter().fold(true, |integral, i| {
        let selector = result_types[slot(i.word(0), bound)];
        integral & !((i.op == Op::Switch) & (selector != Scalar::Int))
    })
}

/// The tables a fold writes: a folded result becomes a canonical constant.
struct Folding<'a> {
    canonical: &'a mut HashMap<(u32, u32), u32>,
    replacements: &'a mut [u32],
    constants: &'a mut [Constant],
    types: &'a [Scalar],
    new_constants: &'a mut Vec<Instruction>,
}

const FOLD: [fn(&mut Instruction, u32, &mut Folding); 2] = [unfolded, fold_constant];

fn unfolded(_: &mut Instruction, _: u32, _: &mut Folding) {}

fn fold_constant(i: &mut Instruction, bits: u32, f: &mut Folding) {
    let (ty, result) = (i.word(0), i.word(1));
    let existing = *f.canonical.entry((ty, bits)).or_insert(result);
    let fresh = existing == result;
    f.replacements[result as usize] = existing;
    let k = result as usize;
    f.constants[k] = pick(
        fresh,
        f.constants[k],
        Constant {
            ty,
            bits,
            known: true,
        },
    );
    let boolean = f.types[ty as usize] == Scalar::Bool;
    let op = pick(
        boolean,
        Op::Constant,
        pick(bits != 0, Op::ConstantFalse, Op::ConstantTrue),
    );
    let words = [ty, result, bits];
    let args = &words[..3 - usize::from(boolean)];
    f.new_constants
        .extend(repeat_with(|| Instruction::new(op, args)).take(usize::from(fresh)));
    i.removed = true;
}

fn forward(instructions: &mut Vec<Instruction>, bound: usize, scratch: &mut LocalScratch) {
    let LocalScratch {
        types,
        constants,
        canonical,
        decorated,
        local_index,
        local,
        pointers,
        values,
        stores,
        stored,
        touched,
        new_constants,
        dead_stores,
        read,
        alive,
        unused,
        replacements,
        needed,
        ..
    } = scratch;
    scalar_variables_into(instructions, types, bound, local, pointers);
    // Local variables number from zero; every other ID names the spare slot
    // past them, which only masked writes reach.
    let count = local.iter().map(|&l| usize::from(l)).sum::<usize>();
    reset(local_index, bound, count);
    let mut next = 0;
    for (id, &is_local) in local.iter().enumerate() {
        local_index[id] = pick(is_local, count, next);
        next += usize::from(is_local);
    }
    reset(values, count + 1, 0);
    reset(stores, count + 1, 0);
    reset(stored, count + 1, false);
    touched.clear();
    new_constants.clear();
    dead_stores.clear();
    reset(read, bound, false);
    reset(alive, bound, false);
    reset(&mut unused.uses, bound, 0);
    reset(&mut unused.definitions, bound, 0);
    reset(&mut unused.defined, bound, false);
    reset(&mut unused.dead, bound, false);
    replacements.clear();
    replacements.extend(0..bound as u32);
    let mut again = true;
    while again {
        let before = instructions.len();
        values.fill(0);
        stored.fill(false);
        touched.clear();
        new_constants.clear();
        dead_stores.clear();
        for (index, i) in instructions.iter_mut().enumerate() {
            i.rewrite(replacements);
            // A label starts a fresh set of facts. No value is assumed on
            // entry to a successor, merge, loop or dispatch case. Only locals
            // touched since the previous label need reset. This preserves the
            // exact block-local forwarding rules.
            let boundary = (i.op == Op::Label) | (i.op == Op::FunctionEnd);
            for pointer in touched.drain(..touched.len() * usize::from(boundary)) {
                values[pointer] = 0;
                stored[pointer] = false;
            }
            let target = slot(i.word(0), bound);
            let store = (i.op == Op::Store) & local[target];
            let p = local_index[target];
            touched.extend(once(p).take(usize::from(store & (values[p] == 0))));
            // Forwarded reads use an immutable SSA value, so even an
            // intervening forwarded read cannot observe an overwritten store.
            // The final store stays for reads in successor blocks.
            dead_stores.extend(once(stores[p]).take(usize::from(store & stored[p])));
            values[p] = pick(store, values[p], i.word(1));
            stores[p] = pick(store, stores[p], index);
            stored[p] |= store;
            let source = slot(i.word(2), bound);
            let result = slot(i.word(1), bound);
            let load = (i.op == Op::Load) & local[source] & !decorated[result];
            let q = local_index[source];
            let value = values[q];
            let forwarded = load & (value != 0);
            replacements[result] = pick(forwarded, replacements[result], value);
            i.removed |= forwarded;
            let first = load & (value == 0);
            touched.extend(once(q).take(usize::from(first)));
            values[q] = pick(first, values[q], i.word(1));
            let (bits, folds) = fold(i, constants, types);
            let folds = folds & !i.removed & i.layout.typed & !decorated[result];
            let mut folding = Folding {
                canonical,
                replacements,
                constants,
                types,
                new_constants,
            };
            FOLD[usize::from(folds)](i, bits, &mut folding);
        }
        let function = instructions
            .iter()
            .enumerate()
            .rev()
            .fold(instructions.len(), |at, (index, i)| {
                pick(i.op == Op::Function, at, index)
            });
        for &index in dead_stores.iter() {
            instructions[index].removed = true;
        }
        instructions.splice(function..function, new_constants.drain(..));
        for i in instructions.iter_mut() {
            i.rewrite(replacements);
        }
        read.fill(false);
        for i in instructions.iter() {
            read[slot(i.word(2), bound)] |= !i.removed & (i.op == Op::Load);
        }
        for i in instructions.iter_mut() {
            let k = slot(i.word(0), bound);
            i.removed |= (i.op == Op::Store) & local[k] & !read[k];
        }
        eliminate_unused(instructions, local, unused);
        instructions.retain(|i| !i.removed);
        alive.fill(false);
        for i in instructions.iter() {
            alive[i.result() as usize] |= i.layout.defines != 0;
        }
        canonical.retain(|_, id| alive[*id as usize]);
        again = instructions.len() != before;
    }
    reset(needed, CAPABILITIES, false);
    for i in instructions.iter() {
        note_capabilities(i, needed);
    }
    instructions.retain(|i| {
        let capability = i.word(0) as usize;
        let k = capability.min(CAPABILITIES - 1);
        let prunable = (capability < CAPABILITIES) & PRUNABLE[k];
        !((i.op == Op::Capability) & prunable & !needed[k])
    });
}

// Keep mutable inline operands for the newer cross-block SSA pass. Reuse
// analysis and module storage per translator, as in Ryujinx
// CodeGen/Spirv/SpirvGenerator.cs:19-28,38-42, without a shared pool lock.
#[derive(Default)]
pub(crate) struct Scratch {
    arena: Vec<u32>,
    instructions: Vec<Instruction>,
    starts: Vec<usize>,
    local: LocalScratch,
}

#[derive(Default)]
struct LocalScratch {
    local: Vec<bool>,
    pointers: Vec<bool>,
    types: Vec<Scalar>,
    constants: Vec<Constant>,
    canonical: HashMap<(u32, u32), u32>,
    decorated: Vec<bool>,
    result_types: Vec<Scalar>,
    constant_at: Vec<usize>,
    local_index: Vec<usize>,
    values: Vec<u32>,
    stores: Vec<usize>,
    stored: Vec<bool>,
    touched: Vec<usize>,
    new_constants: Vec<Instruction>,
    dead_stores: Vec<usize>,
    read: Vec<bool>,
    alive: Vec<bool>,
    unused: UnusedScratch,
    replacements: Vec<u32>,
    needed: Vec<bool>,
}

fn reset<T: Clone>(buffer: &mut Vec<T>, len: usize, value: T) {
    buffer.clear();
    buffer.resize(len, value);
}

impl Scratch {
    pub(crate) fn take_arena(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.arena)
    }
}

/// Optimise only the known emitter grammar. Unknown forms preserve the module.
pub(crate) fn optimize(words: &mut Vec<u32>) {
    optimize_with_scratch(words, &mut Scratch::default());
}

pub(crate) fn optimize_with_scratch(words: &mut Vec<u32>, scratch: &mut Scratch) {
    let known = parse_into(words, &mut scratch.instructions, &mut scratch.starts);
    OPTIMIZE[usize::from(known)](words, scratch);
}

const OPTIMIZE: [fn(&mut Vec<u32>, &mut Scratch); 2] = [preserve, optimize_known];

fn preserve(_: &mut Vec<u32>, _: &mut Scratch) {}

fn optimize_known(words: &mut Vec<u32>, scratch: &mut Scratch) {
    let bound = words[3] as usize;
    let integral = optimize_local(&mut scratch.instructions, bound, &mut scratch.local);
    [preserve, optimize_blocks][usize::from(integral)](words, scratch);
}

const RERUN: [fn(&mut Vec<Instruction>, usize, &mut LocalScratch); 2] = [settled, rerun];

fn rerun(instructions: &mut Vec<Instruction>, bound: usize, scratch: &mut LocalScratch) {
    optimize_local(instructions, bound, scratch);
}

fn optimize_blocks(words: &mut Vec<u32>, scratch: &mut Scratch) {
    let Scratch {
        instructions,
        local,
        arena,
        ..
    } = scratch;
    let mut bound = words[3] as usize;
    let mut promoted = cross_block::promote(instructions, &mut bound);
    let mut again = true;
    while again {
        let changed = cross_block::simplify(instructions, bound);
        // Local optimization already reached a fixed point. Re-running it on
        // unchanged instructions only rebuilds the same facts and rewrites IDs.
        let stale = promoted | changed;
        let before = instructions.len();
        RERUN[usize::from(stale)](instructions, bound, local);
        again = stale & (changed | (instructions.len() != before));
        promoted = false;
    }
    std::mem::swap(words, arena);
    words.clear();
    words.extend_from_slice(&arena[..5]);
    words[3] = bound as u32;
    for i in instructions.iter() {
        words.push(((i.args.len() as u32 + 1) << 16) | u32::from(i.op));
        words.extend_from_slice(&i.args);
    }
}

#[cfg(test)]
mod tests;
