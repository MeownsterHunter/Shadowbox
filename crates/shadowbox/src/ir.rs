use std::fmt;

/// The slot that runs when control reaches byte address `named`. A bundle
/// is one control word and three 8-byte instruction slots (Mesa
/// src/nouveau/compiler/nak/sm50.rs:3418-3428), so a control-word address
/// runs slot 0, as Ryujinx Decoders/Decoder.cs:265-270 does. The largest
/// control-word address is 0xffff_ffe0, so the step never wraps.
pub const fn run_slot(named: u32) -> u32 {
    // 1 when `named` has bits below the bundle size, 0 at a control word.
    let within = ((named & 31) + 31) >> 5;
    named + 8 * (1 - within)
}

/// The exact byte address an immediate control transfer names: the signed
/// byte displacement counts from the next 8-byte slot (Mesa
/// src/nouveau/compiler/nak/sm50.rs:3006-3028; Ryujinx
/// Decoders/InstOp.cs:22-25).
fn named_address(address: u32, offset: i32) -> i64 {
    i64::from(address) + 8 + i64::from(offset)
}

/// A relative control-transfer target inside the code address space,
/// decoded once from the instruction that encodes it. The slot already skips
/// a bundle control word, and the displacement is the one that reaches that
/// slot (the encoded displacement, plus 8 when it named a control word).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BranchTarget {
    offset: i32,
    slot: u32,
}

/// What a relative control transfer names, read once at the word edge.
pub enum Named {
    /// An instruction slot.
    Slot(BranchTarget),
    /// An address between instruction slots.
    Misaligned(MisalignedTarget),
    /// An address outside the code address space.
    Outside(OutsideTarget),
}

/// A relative control-transfer target inside the code address space that
/// is not a multiple of the 8-byte instruction size. It names no slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MisalignedTarget {
    offset: i32,
    address: u32,
}

impl MisalignedTarget {
    /// The encoded byte displacement.
    pub fn offset(self) -> i32 {
        self.offset
    }

    /// The exact byte address, between two slots.
    pub fn address(self) -> u32 {
        self.address
    }
}

/// A relative control-transfer target whose exact byte address lies below
/// zero or past the 32-bit code address space. It names no slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutsideTarget {
    offset: i32,
    address: i64,
}

impl BranchTarget {
    /// What the instruction at `address` with encoded byte displacement
    /// `offset` names: a slot, an address between slots, or an address
    /// outside the code address space. As in Ryujinx Decoders/InstOp.cs:
    /// 22-25 (GetAbsoluteAddress), target sums do not wrap.
    pub fn named(address: u32, offset: i32) -> Named {
        let named = named_address(address, offset);
        match u32::try_from(named) {
            Err(_) => Named::Outside(OutsideTarget {
                offset,
                address: named,
            }),
            Ok(exact) if !exact.is_multiple_of(8) => Named::Misaligned(MisalignedTarget {
                offset,
                address: exact,
            }),
            Ok(exact) => {
                let slot = run_slot(exact);
                Named::Slot(Self {
                    offset: offset + (slot - exact) as i32,
                    slot,
                })
            }
        }
    }

    /// The target that the instruction at `from` encodes to reach slot `to`
    /// (Mesa sm50.rs:3006-3016: target_ip - ip - 8).
    pub fn reaching(from: u32, to: u32) -> Self {
        Self {
            offset: (i64::from(to) - i64::from(from) - 8) as i32,
            slot: to,
        }
    }

    /// The byte displacement that reaches the slot.
    pub fn offset(self) -> i32 {
        self.offset
    }

    /// The code slot this target runs.
    pub fn slot(self) -> u32 {
        self.slot
    }
}

impl OutsideTarget {
    /// The encoded byte displacement.
    pub fn offset(self) -> i32 {
        self.offset
    }

    /// The exact byte address, outside the code address space.
    pub fn address(self) -> i64 {
        self.address
    }
}

/// The 32-bit sum of `a` and `b`, as the GPU adds addresses. When the
/// exact sum reaches 2^32, `b` exceeds `u32::MAX - a`, so
/// `b - (u32::MAX - a) - 1` is that sum minus 2^32; otherwise the exact sum
/// fits.
const fn add_mod_2_32(a: u32, b: u32) -> u32 {
    let room = u32::MAX - a;
    if b > room {
        b - room - 1
    } else {
        a + b
    }
}

/// The immediate-relative base of a BRX: an exact byte address that the
/// runtime register displacement or a proven table entry is added to
/// (envytools envydis/gm107.c:1942-1943; Ryujinx
/// Instructions/InstEmitFlowControl.cs:38-40). The base itself is never run,
/// so it keeps any address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BrxBase {
    word: u32,
    address: i64,
    runtime: u32,
}

impl BrxBase {
    /// Incoming edge for the BRX at `address` whose 24-bit displacement
    /// field (bits 20..44, envytools envydis/gm107.c:1942-1943) is the low
    /// 24 bits of `field`. Every value below is exact u32 or i64 arithmetic:
    /// the field is below 2^24, so its sign extension to 32 bits stays
    /// below 2^32.
    pub fn decode(address: u32, field: u32) -> Self {
        let field = field & 0xff_ffff;
        let sign = field >> 23;
        let word = field + sign * 0xff00_0000;
        Self {
            word,
            address: i64::from(address) + 8 + i64::from(field) - i64::from(sign) * 0x100_0000,
            runtime: add_mod_2_32(add_mod_2_32(address, 8), word),
        }
    }

    /// The encoded byte displacement as its 32-bit two's-complement word,
    /// the value the runtime adds.
    pub fn offset(self) -> u32 {
        self.word
    }

    /// The runtime BRX base address: the 32-bit sum of the address, 8 and
    /// the encoded displacement, as Ryujinx
    /// Instructions/InstEmitFlowControl.cs:38 computes it.
    pub fn runtime(self) -> u32 {
        self.runtime
    }

    /// The exact byte address a loaded table displacement names. The base
    /// is a u32 address plus 8 plus an i32, and the displacement is an i32,
    /// so the sum stays far inside i64.
    fn entry_address(self, loaded: Displacement) -> i64 {
        self.address + loaded.bytes()
    }
}

/// A BRX table word as loaded from the constant buffer: a two's-complement
/// byte displacement that BRX adds to its base (Ryujinx
/// Instructions/InstEmitFlowControl.cs:38-40). The word itself is the
/// value the emitted dispatch compares against.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Displacement(u32);

impl Displacement {
    /// The word as loaded.
    pub fn loaded(self) -> u32 {
        self.0
    }

    /// The signed byte count the word encodes. A u32 is below 2^32, so
    /// subtracting 2^32 once for a set top bit stays far inside i64.
    fn bytes(self) -> i64 {
        i64::from(self.0) - i64::from(self.0 >> 31) * 0x1_0000_0000
    }
}

/// A proven BRX table entry whose destination lies in the code address
/// space: the loaded displacement and the slot it runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BrxEntry {
    loaded: Displacement,
    slot: u32,
}

/// A proven BRX table entry whose exact destination lies below zero or past
/// the 32-bit code address space. It names no slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutsideBrxEntry {
    loaded: Displacement,
}

/// A proven BRX table entry whose exact destination lies in the code
/// address space but off the 8-byte word grid. It names no slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MisalignedBrxEntry {
    loaded: Displacement,
    address: u32,
}

/// One entry of a proven BRX table with at least one destination that
/// names no slot, decoded once when the table is read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BrxTableEntry {
    Slot(BrxEntry),
    Outside(OutsideBrxEntry),
    Misaligned(MisalignedBrxEntry),
}

/// The first table entry that names no slot: the station where the table
/// stops.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoSlotEntry {
    Outside(OutsideBrxEntry),
    Misaligned(MisalignedBrxEntry),
}

/// Fills table entries past `count`, which are never read.
const TABLE_FILL: BrxEntry = BrxEntry {
    loaded: Displacement(0),
    slot: 0,
};

impl BrxEntry {
    /// The loaded table word.
    pub fn loaded(self) -> u32 {
        self.loaded.loaded()
    }

    /// The code slot this entry runs.
    pub fn slot(self) -> u32 {
        self.slot
    }
}

/// A proven BRX table whose every destination lies in the code address
/// space: one to 32 entries, as the proof made it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SlotTable {
    targets: [BrxEntry; 32],
    count: u8,
}

impl SlotTable {
    /// The entries, in table order.
    pub fn targets(&self) -> &[BrxEntry] {
        &self.targets[..usize::from(self.count)]
    }

    /// How many entries the table holds.
    pub fn count(&self) -> u8 {
        self.count
    }
}

impl OutsideBrxEntry {
    /// The loaded table word.
    pub fn loaded(self) -> u32 {
        self.loaded.loaded()
    }

    /// The exact byte address this entry names from `base`, the base of the
    /// BRX it was decoded with; it lies outside the code address space.
    pub fn address(self, base: BrxBase) -> i64 {
        base.entry_address(self.loaded)
    }
}

impl MisalignedBrxEntry {
    /// The loaded table word.
    pub fn loaded(self) -> u32 {
        self.loaded.loaded()
    }

    /// The exact byte address this entry names.
    pub fn address(self) -> u32 {
        self.address
    }
}

impl BrxTableEntry {
    /// Incoming edge for one loaded table word: it becomes the entry it
    /// names from `base`. Only a control-word destination moves to slot 0,
    /// as Ryujinx Decoders/Decoder.cs:265-270 does (layout: Mesa
    /// sm50.rs:3418-3428). As in Ryujinx Decoders/InstOp.cs:22-25, the sum
    /// does not wrap.
    fn decode(base: BrxBase, word: u32) -> Self {
        let loaded = Displacement(word);
        match u32::try_from(base.entry_address(loaded)) {
            Err(_) => Self::Outside(OutsideBrxEntry { loaded }),
            Ok(address) if !address.is_multiple_of(8) => {
                Self::Misaligned(MisalignedBrxEntry { loaded, address })
            }
            Ok(named) => Self::Slot(BrxEntry {
                loaded,
                slot: run_slot(named),
            }),
        }
    }
}

/// What a decoded BRX table entry does for its consumers.
pub trait TableEntry: Copy {
    /// The loaded table word.
    fn loaded(self) -> u32;
    /// The slot entry this is, or none for an outside entry.
    fn slot_entries(&self) -> &[BrxEntry];
    /// Writes the slot this entry runs as `Some(slot)`, or `None` for an
    /// outside entry, in the formatter's integer style.
    fn fmt_slot(self, f: &mut fmt::Formatter<'_>) -> fmt::Result;
}

impl TableEntry for BrxTableEntry {
    fn loaded(self) -> u32 {
        match self {
            Self::Slot(entry) => entry.loaded(),
            Self::Outside(entry) => entry.loaded(),
            Self::Misaligned(entry) => entry.loaded(),
        }
    }

    fn slot_entries(&self) -> &[BrxEntry] {
        match self {
            Self::Slot(entry) => std::slice::from_ref(entry),
            Self::Outside(_) | Self::Misaligned(_) => &[],
        }
    }

    fn fmt_slot(self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Slot(entry) => f.debug_tuple("Some").field(&entry.slot()).finish(),
            Self::Outside(_) | Self::Misaligned(_) => f.write_str("None"),
        }
    }
}

/// How a decoded entry shapes the table it is gathered into.
trait Gather {
    /// Copies a slot entry into `into`; an outside entry is recorded in the
    /// shape.
    fn gather(self, into: &mut BrxEntry, shape: TableShape) -> TableShape;
}

impl Gather for BrxTableEntry {
    fn gather(self, into: &mut BrxEntry, shape: TableShape) -> TableShape {
        match self {
            Self::Slot(entry) => {
                *into = entry;
                shape.slot()
            }
            Self::Outside(entry) => shape.outside(NoSlotEntry::Outside(entry)),
            Self::Misaligned(entry) => shape.outside(NoSlotEntry::Misaligned(entry)),
        }
    }
}

/// A BRX table entry shown by the slot it runs: `Some(slot)`, or `None`
/// when it lies outside the code address space.
#[derive(Clone, Copy)]
pub struct ShownSlot(pub BrxTableEntry);

impl fmt::Debug for ShownSlot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt_slot(f)
    }
}

/// A decoded table seen so far: only slot entries (`leading` of them), or
/// `leading` slot entries and then a first outside entry.
#[derive(Clone, Copy)]
enum TableShape {
    AllSlots { leading: u8 },
    WithOutside { leading: u8, first: NoSlotEntry },
}

/// How a table's shape grows and what it builds.
trait Shape {
    /// The shape after one more slot entry. A table has at most 32 entries,
    /// so `leading` stays at most 32.
    fn slot(self) -> Self;
    /// The shape after an outside entry.
    fn outside(self, entry: NoSlotEntry) -> Self;
    /// What `meets` makes of this shape, from the gathered slot entries or
    /// the decoded table.
    fn meets<M: TableMeets>(self, entries: [BrxEntry; 32], table: BrxTable, meets: M) -> M::Out;
}

/// What meets a proven table as the shape it is.
pub(crate) trait TableMeets {
    type Out;
    /// Every destination lies in the code address space: the first
    /// `count` gathered entries.
    fn slots(self, table: SlotTable) -> Self::Out;
    /// At least one destination lies outside the code address space.
    fn outside(self, table: OutsideBrxTable) -> Self::Out;
}

/// The BRX kind a table makes, reading `index` from `base`.
struct Kinding {
    index: Register,
    base: BrxBase,
}

impl TableMeets for Kinding {
    type Out = InstructionKind;
    fn slots(self, table: SlotTable) -> InstructionKind {
        InstructionKind::Brx(ops::Brx {
            index: self.index,
            base: self.base,
            table,
        })
    }
    fn outside(self, table: OutsideBrxTable) -> InstructionKind {
        InstructionKind::OutsideBrx(ops::OutsideBrx {
            index: self.index,
            base: self.base,
            table,
        })
    }
}

impl Shape for TableShape {
    fn slot(self) -> Self {
        match self {
            Self::AllSlots { leading } => Self::AllSlots {
                leading: leading + 1,
            },
            Self::WithOutside { .. } => self,
        }
    }

    fn outside(self, entry: NoSlotEntry) -> Self {
        match self {
            Self::AllSlots { leading } => Self::WithOutside {
                leading,
                first: entry,
            },
            Self::WithOutside { .. } => self,
        }
    }

    fn meets<M: TableMeets>(self, entries: [BrxEntry; 32], table: BrxTable, meets: M) -> M::Out {
        match self {
            Self::AllSlots { leading } => meets.slots(SlotTable {
                targets: entries,
                count: leading,
            }),
            Self::WithOutside { leading, first } => meets.outside(OutsideBrxTable {
                table,
                leading,
                first,
            }),
        }
    }
}

/// A proven BRX table: every loaded word decoded once from the BRX base.
/// Entries past `count` are unused.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BrxTable {
    entries: [BrxTableEntry; 32],
    count: u8,
}

impl BrxTable {
    /// Incoming edge for a proven table: `first` and each following loaded
    /// word, up to 32, becomes the entry it names from `base`. `count`
    /// grows once per entry, so it is one to 32.
    pub fn decode(base: BrxBase, first: u32, rest: impl IntoIterator<Item = u32>) -> Self {
        let mut entries = [BrxTableEntry::Slot(TABLE_FILL); 32];
        let mut count = 0;
        let loaded = std::iter::once(first).chain(rest);
        for (entry, word) in entries.iter_mut().zip(loaded) {
            *entry = BrxTableEntry::decode(base, word);
            count += 1;
        }
        Self { entries, count }
    }

    /// The decoded entries, in table order.
    pub fn entries(&self) -> impl Iterator<Item = &BrxTableEntry> {
        self.entries.iter().take(usize::from(self.count))
    }

    /// The BRX kind this table makes: `Brx` when every destination lies in
    /// the code address space, else `OutsideBrx`.
    pub fn kind(self, index: Register, base: BrxBase) -> InstructionKind {
        self.meets(Kinding { index, base })
    }

    /// `meets` meets this table as the shape it is.
    pub(crate) fn meets<M: TableMeets>(self, meets: M) -> M::Out {
        let mut entries = [TABLE_FILL; 32];
        let shape = entries.iter_mut().zip(self.entries()).fold(
            TableShape::AllSlots { leading: 0 },
            |shape, (into, entry)| entry.gather(into, shape),
        );
        shape.meets(entries, self, meets)
    }
}

/// A proven BRX table with at least one entry that names no slot (outside
/// the code address space or off the word grid): `leading` slot entries
/// come before the first such one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OutsideBrxTable {
    table: BrxTable,
    leading: u8,
    first: NoSlotEntry,
}

impl OutsideBrxTable {
    /// The decoded entries, in table order.
    pub fn entries(&self) -> impl Iterator<Item = &BrxTableEntry> {
        self.table.entries()
    }

    /// The number of proven entries.
    pub fn count(&self) -> u8 {
        self.table.count
    }

    /// Every slot entry, in table order, including those after an outside
    /// entry.
    pub fn slots(&self) -> impl Iterator<Item = &BrxEntry> {
        self.entries().flat_map(|entry| entry.slot_entries())
    }

    /// The slot entries before the first outside entry.
    pub fn leading(&self) -> impl Iterator<Item = &BrxEntry> {
        self.entries()
            .take(usize::from(self.leading))
            .flat_map(|entry| entry.slot_entries())
    }

    /// The first entry that names no slot.
    pub fn first_outside(&self) -> NoSlotEntry {
        self.first
    }
}

impl InstructionKind {
    /// A BRX with proven table words `first` and `rest`, decoded once as
    /// a [`BrxTable`].
    pub fn brx(index: Register, base: BrxBase, first: u32, rest: &[u32]) -> Self {
        BrxTable::decode(base, first, rest.iter().copied()).kind(index, base)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Register(pub u8);
/// The GPR pair that holds one binary64 value, named by its low word's
/// register. Ryujinx InstEmitHelper.cs:37-40,68-78 (GetDest2, GetSrcReg
/// isFP64): the high word is the register with bit 0 set, so an odd
/// register is both words and RZ pairs with itself.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegisterPair(pub Register);
impl RegisterPair {
    /// The register holding the low word.
    pub fn low(self) -> Register {
        self.0
    }
    /// The register holding the high word.
    pub fn high(self) -> Register {
        Register(self.0 .0 | 1)
    }
    /// The register holding the low word alone; an odd pair has none.
    pub fn low_alone(self) -> Option<Register> {
        (self.0 .0 & 1 == 0).then_some(self.0)
    }
    /// The registers a binary64 write covers: the low through the high.
    pub fn written(self) -> Written {
        Written::new(self.0, self.high().0 - self.0 .0 + 1)
    }
}
impl fmt::Display for RegisterPair {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "R{}", self.0 .0)
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Predicate {
    pub register: u8,
    pub inverted: bool,
}
/// IADD3 component and partial-sum controls (envytools gm107.c:1228-1253).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HalfSelect {
    Full,
    Low,
    High,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartialSumShift {
    /// The partial sum enters unshifted.
    Unshifted,
    Right,
    Left,
}
/// SHF type field (Mesa sm50.rs:1667-1675).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FunnelType {
    Bits32,
    Unsigned64,
    Signed64,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Register(Register),
    Immediate(u32),
    ConstantBuffer { binding: u8, offset: u16 },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpLoc {
    Default,
    Centroid,
    Offset,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpFreq {
    Pass,
    PassMulW,
    Constant,
    State,
}
/// The ordering relation a nonconstant compare tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Relation {
    Lt,
    Eq,
    Le,
    Gt,
    Ne,
    Ge,
}
/// The 3-bit integer and video compare field (envytools gm107.c:1445-1453,
/// 1588-1596): 0 false, 1..6 Lt Eq Le Gt Ne Ge, 7 true.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntCompare {
    False,
    True,
    Relation(Relation),
}
/// The 4-bit float compare field (Mesa set_float_cmp_op, sm50.rs:775-796):
/// 0 false, 1..6 ordered Lt Eq Le Gt Ne Ge, 7 IsNum, 8 IsNan, 9..14
/// unordered Lt Eq Le Gt Ne Ge, 15 true.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatCompare {
    False,
    True,
    /// Both operands are numbers (Mesa FloatCmpOp::IsNum).
    Num,
    /// Either operand is NaN (Mesa FloatCmpOp::IsNan).
    Nan,
    /// The relation holds and neither operand is NaN.
    Ordered(Relation),
    /// The relation holds or either operand is NaN.
    Unordered(Relation),
}
/// An SSY/PBK/PCNT push whose consumers stack resolution checked: each pop
/// is lowered or remains an error. Unused pushes have no tag writes (Ryujinx
/// InstEmitFlowControl.cs:239-251). Only stack resolution makes one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ResolvedPush {
    target: BranchTarget,
}
impl ResolvedPush {
    pub(crate) fn checked(target: BranchTarget) -> Self {
        Self { target }
    }

    pub fn target(self) -> BranchTarget {
        self.target
    }
}
/// Which reconvergence stack an SSY/PBK/PCNT push and SYNC/BRK/CONT use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetStack {
    Sync,
    Break,
    Continue,
}
/// Texture operation and its mode fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TexOp {
    /// TEX: lod mode 0 implicit, 1 zero, 2 bias, 3 explicit LOD.
    Sample { lod: u8 },
    /// TLD: texel fetch at LOD zero or an explicit LOD.
    Fetch { lod: bool },
    /// TLD4: gather of one component with its encoded offset state.
    Gather {
        component: u8,
        offsets: GatherOffsets,
    },
    /// TXQ: query 1 is dimensions, query 2 supplies samples in Z.
    Query { query: u8 },
    /// TMML.LOD: signed 8.8 fixed-point LOD, with Maxwell's reversed XY.
    /// Ryujinx Instructions/InstEmitTexture.cs:897-923.
    Lod,
    /// TXD: sample with explicit derivatives from B (dPdx, then dPdy). A
    /// holds the coordinates, then one packed word: layer in bits 0..15
    /// and signed four-bit offsets from bit 16. Ryujinx
    /// Instructions/InstEmitTexture.cs:929-1008.
    Grad,
}
/// TLD4 offset state. The hardware distinguishes one packed offset from four
/// packed offsets; neither is a boolean property of a gather.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GatherOffsets {
    /// No packed offset: the gather reads at its coordinate.
    Zero,
    One,
    Four,
}
/// Disassembly names: the relation or the constant.
impl fmt::Display for IntCompare {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IntCompare::False => f.write_str("False"),
            IntCompare::True => f.write_str("True"),
            IntCompare::Relation(relation) => write!(f, "{relation:?}"),
        }
    }
}
/// Disassembly names: the relation without its ordered/unordered half,
/// which the instruction text prints separately.
impl fmt::Display for FloatCompare {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FloatCompare::False => f.write_str("False"),
            FloatCompare::True => f.write_str("True"),
            FloatCompare::Num => f.write_str("Num"),
            FloatCompare::Nan => f.write_str("Nan"),
            FloatCompare::Ordered(relation) | FloatCompare::Unordered(relation) => {
                write!(f, "{relation:?}")
            }
        }
    }
}
impl FloatCompare {
    /// The 4-bit encoding this compare was decoded from.
    pub fn field(self) -> u32 {
        let relation = |relation| match relation {
            Relation::Lt => 1,
            Relation::Eq => 2,
            Relation::Le => 3,
            Relation::Gt => 4,
            Relation::Ne => 5,
            Relation::Ge => 6,
        };
        match self {
            FloatCompare::False => 0,
            FloatCompare::Ordered(r) => relation(r),
            FloatCompare::Num => 7,
            FloatCompare::Nan => 8,
            FloatCompare::Unordered(r) => 8 + relation(r),
            FloatCompare::True => 15,
        }
    }
}
/// How an integer compare reads its operands: ISETP bit 48 selects U32 or
/// I32 (sm50.rs OpISetP encode).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Signedness {
    Unsigned,
    Signed,
}
/// The byte, half-word, or whole-word selected from a VMAD source.
/// Selector 7 has no variant because the architecture reserves it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VideoElement {
    Byte0,
    Byte1,
    Byte2,
    Byte3,
    Half0,
    Half1,
    Whole,
}
/// A validated VMAD source selection. The decoder owns the raw selector;
/// emitters consume this causal state without interpreting instruction bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VideoSelection {
    pub element: VideoElement,
    pub signedness: Signedness,
}
impl VideoSelection {
    /// The legacy selector byte used only by stable diagnostic rendering.
    pub fn field(self) -> u8 {
        let element = match self.element {
            VideoElement::Byte0 => 0,
            VideoElement::Byte1 => 1,
            VideoElement::Byte2 => 2,
            VideoElement::Byte3 => 3,
            VideoElement::Half0 => 4,
            VideoElement::Half1 => 5,
            VideoElement::Whole => 6,
        };
        element
            | match self.signedness {
                Signedness::Unsigned => 0,
                Signedness::Signed => 8,
            }
    }

    pub fn is_signed_whole(self) -> bool {
        matches!(
            self,
            Self {
                element: VideoElement::Whole,
                signedness: Signedness::Signed,
            }
        )
    }
}
/// Integer destination width established by the conversion encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntegerWidth {
    Bits16,
    Bits32,
}
/// Whether an integer relation is evaluated directly or extended through the
/// previous Z/C condition-code state (ISETP/ISET bit 43).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntegerCompareMode {
    Ordinary,
    Extended,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SetOp {
    And,
    Or,
    Xor,
}

/// The condition a CSETP or CSET tests: its five-bit selector, each value
/// its own condition. Selectors 24-29 are the CSM conditions (CSM_TA,
/// CSM_TR, CSM_MX, FCSM_TA, FCSM_TR, FCSM_MX in that order).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CSetPCondition(u8);

impl CSetPCondition {
    /// The condition the encoding's five selector bits name.
    pub fn decode(selector: u8) -> Self {
        Self(selector & 0x1f)
    }

    /// The five-bit selector used by Maxwell's instruction encoding.
    pub fn selector(self) -> u8 {
        self.0
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Conversions as Mesa's SM 5.x encoders define them (sm50.rs OpF2F, OpF2I,
/// OpI2F, OpI2I). I2F carries the hardware's destination/source sizes and
/// byte select in addition to signedness and rounding.
pub enum Conversion {
    /// Signed or unsigned integer to float with the encoded rounding mode.
    I2F {
        signed: bool,
        dst_size: u8,
        src_size: u8,
        round: RoundingMode,
        byte: u8,
        abs: bool,
        negate: bool,
    },
    /// Float to integer: round with the mode, then saturate (NaN gives 0).
    F2I {
        signed: bool,
        dst_size: u8,
        src_size: u8,
        round: RoundingMode,
        high: bool,
        /// Mesa sm50.rs:1837: flush float32 subnormals before rounding.
        ftz: bool,
        abs: bool,
        negate: bool,
    },
    /// Float to float of the same size: identity, or round to an integral
    /// value with the mode when integer rounding is set.
    F2F {
        source: F2fSource,
        destination: F2fDestination,
        integer_round: Option<RoundingMode>,
        abs: bool,
        negate: bool,
        saturate: bool,
        /// Mesa sm50.rs:1793: flush float32 subnormals before rounding.
        ftz: bool,
    },
    /// Integer extraction, abs/neg, optional destination clamp and Z/N flags.
    /// Ryujinx Instructions/InstEmitConversion.cs:286-342.
    I2I {
        src_signed: bool,
        dst_signed: bool,
        src_size: u8,
        dst_size: u8,
        byte: u8,
        abs: bool,
        negate: bool,
        saturate: bool,
        cc: bool,
    },
}
/// The source format an F2F reads before its float32 result is formed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F2fSource {
    /// The full 32-bit operand is a binary32 value.
    Float32,
    /// The low 16 bits of the operand are a binary16 value, widened exactly.
    Float16Low,
    /// The high 16 bits of the operand are a binary16 value, widened
    /// exactly (Sh41: Ryujinx InstEmitConversion.cs:344-352 selects it).
    Float16High,
}
/// The register shape an F2F's float32 result takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum F2fDestination {
    /// The whole register is the binary32 result.
    Float32,
    /// The low 16 bits are the result as binary16 and the high 16 bits are
    /// zero: Ryujinx InstEmitConversion.cs:406-409 packs it with 0.
    Float16Low,
    /// The register and the next hold the exactly widened binary64 result,
    /// low word first: Ryujinx InstEmitConversion.cs:149-152 converts with
    /// FP32ConvertToFP64 and :410-416 writes both words.
    Float64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoundingMode {
    NearestEven,
    NegativeInfinity,
    PositiveInfinity,
    Zero,
}

/// The source operation encoded by a qualified binary64 arithmetic operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DoubleSourceModifier {
    Plain,
    Absolute,
    Negated,
    NegatedAbsolute,
}

/// Where a binary64 operand reads its two words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DoubleSource {
    /// A GPR pair, low word first.
    RegisterPair(RegisterPair),
    /// Two consecutive constant-buffer words at `offset` and `offset + 4`
    /// bytes, low word first.
    ConstantBuffer { binding: u8, offset: u16 },
    /// A D20 immediate: this is the binary64 high word, the low word is zero.
    Immediate(u32),
}

impl DoubleSource {
    /// The low word, the high word, and the first word the operand reads.
    /// Ryubing InstEmitHelper.cs:42-66: a cbuf operand packs word offset
    /// (low) with offset + 1 (high); a D20 immediate is the high word over a
    /// zero low word, which reads nothing.
    fn parts(self) -> ([Source; 2], usize) {
        match self {
            Self::RegisterPair(pair) => (
                [Source::Register(pair.low()), Source::Register(pair.high())],
                0,
            ),
            Self::ConstantBuffer { binding, offset } => (
                [
                    Source::ConstantBuffer { binding, offset },
                    Source::ConstantBuffer {
                        binding,
                        offset: offset + 4,
                    },
                ],
                0,
            ),
            Self::Immediate(high) => ([Source::Immediate(0), Source::Immediate(high)], 1),
        }
    }
    /// The low word, then the high word.
    pub fn words(self) -> [Source; 2] {
        self.parts().0
    }
    /// The words the operand reads.
    pub fn slots(self) -> Sources {
        let ([low, high], first) = self.parts();
        let words = [low, high, Source::Immediate(0)];
        Sources::new(
            [words[first], words[first + 1], Source::Immediate(0)],
            2 - first as u8,
        )
    }
}

impl fmt::Display for DoubleSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            DoubleSource::RegisterPair(pair) => write!(f, "{pair}"),
            DoubleSource::ConstantBuffer { binding, offset } => {
                write!(f, "c[{binding}][0x{offset:x}]")
            }
            DoubleSource::Immediate(high) => write!(f, "{}", f64::from_bits(u64::from(high) << 32)),
        }
    }
}

/// The operation DMNMX's predicate source selects: PT the minimum, !PT the
/// maximum, any other predicate the minimum where it holds and the maximum
/// elsewhere (Ryujinx InstEmitFloatMinMax.cs:78-104).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DoubleMinMaxOperation {
    Minimum,
    Maximum,
    Selected(Predicate),
}

/// The concrete 32-bit value written when a floating comparison is true.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SetResult {
    Mask,
    FloatOne,
}

/// DSET's concrete Boolean representation and its coupled condition-code
/// behavior. The two written states differ: a true mask is a negative integer,
/// while a true float-one is positive binary32.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DoubleSetResult {
    Mask(FloatFlags),
    FloatOne(FloatFlags),
}

/// Maxwell DMUL has one shared source-negation bit, so the product itself has
/// one of these two concrete sign states.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DoubleProductSign {
    Positive,
    Negated,
}

/// Whether DFMA adds or subtracts its third binary64 operand.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DoubleAddendSign {
    Added,
    Subtracted,
}
/// Whether a float source is read as its absolute value (applied before
/// its negation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceAbs {
    Plain,
    Absolute,
}
/// Whether a float source is read negated.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceNegation {
    Plain,
    Negated,
}
/// Whether a float result is clamped to [0, 1] (SAT).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Saturation {
    Unclamped,
    Saturated,
}
/// Whether a float result keeps the flags or writes Z/N while preserving C/V.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatFlags {
    Kept,
    Written,
}
impl fmt::Display for FloatFlags {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(["", ".CC"][*self as usize])
    }
}
/// Whether subnormal float inputs and results are kept or flushed to a
/// signed zero (FTZ).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Denormals {
    Preserved,
    FlushedToZero,
}
/// FADD modifiers, one variant per encoding form, each holding only the
/// fields that form encodes (envytools gm107.c:2006,2080,2126,2167;
/// Ryujinx InstDecoders.cs:1704-1777). The register, cbuf and f20 forms
/// encode their rounding separately at bits 39..40 (Mesa sm50.rs:525;
/// envytools gm107.c:1275-1281).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FAddModifiers {
    /// Register and cbuf forms: abs 46/49, neg 48/45, SAT 50, FTZ 44.
    Operand {
        lhs_abs: SourceAbs,
        rhs_abs: SourceAbs,
        lhs_negation: SourceNegation,
        rhs_negation: SourceNegation,
        saturation: Saturation,
        denormals: Denormals,
    },
    /// f20 immediate form: as the register form, but the decoder matches
    /// B abs (bit 49) as zero in the opcode, so B has no abs.
    Immediate20 {
        lhs_abs: SourceAbs,
        lhs_negation: SourceNegation,
        rhs_negation: SourceNegation,
        saturation: Saturation,
        denormals: Denormals,
    },
    /// FADD32I: abs 54/57, neg 56/53, FTZ 55; no SAT and no rounding.
    Immediate32 {
        lhs_abs: SourceAbs,
        rhs_abs: SourceAbs,
        lhs_negation: SourceNegation,
        rhs_negation: SourceNegation,
        denormals: Denormals,
    },
}
/// FMUL rounding modes other than nearest-even (Round39..40, Mesa
/// sm50.rs:695-700), which the emitter implements with a binary64 product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectedRounding {
    NegativeInfinity,
    PositiveInfinity,
    Zero,
}
/// The FMUL rounding field: RN, or one of the directed modes. F2F.F32.F64
/// narrows in the same four modes (bits 39..40).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FMulRounding {
    NearestEven,
    Directed(DirectedRounding),
}
/// FMUL's power-of-two scale of A, bits 41..43: 0 none, 1..3 divide by
/// 2, 4, 8, and 4..6 multiply by 8, 4, 2; 7 is reserved (envytools
/// gm107.c:1331-1345). Ryujinx InstEmitFloatArithmetic.cs:450-476 scales A
/// before the multiply.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FMulScale {
    Unscaled,
    Divide2,
    Divide4,
    Divide8,
    Multiply8,
    Multiply4,
    Multiply2,
}
/// FMUL modifiers, one variant per encoding form, each holding only the
/// fields that form encodes (envytools gm107.c:2004,2078,2124,2160;
/// Ryujinx InstDecoders.cs:2077-2141). The denormal mode is FTZ44/FMZ45
/// (FMUL32I: 53/54), envytools gm107.c:1330-1335,1516-1520.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FMulModifiers {
    /// Register, cbuf and f20 forms: one negation (of B) 48, SAT 50, FTZ
    /// 44/FMZ 45, rounding 39..40 and the scale of A, 41..43 (Mesa
    /// sm50.rs:679-700).
    Operand {
        rhs_negation: SourceNegation,
        saturation: Saturation,
        flush: Flush,
        rounding: FMulRounding,
        scale: FMulScale,
    },
    /// FMUL32I: FTZ 53/FMZ 54 and SAT 55; no negation, rounding or scale.
    Immediate32 {
        saturation: Saturation,
        flush: Flush,
    },
}
impl FMulModifiers {
    /// The register, cbuf and f20 forms carry their rounding; FMUL32I
    /// rounds to nearest-even.
    pub fn rounding(self) -> FMulRounding {
        match self {
            FMulModifiers::Operand { rounding, .. } => rounding,
            FMulModifiers::Immediate32 { .. } => FMulRounding::NearestEven,
        }
    }
}
/// FFMA's and FMUL's denormal mode: FTZ and DNZ (FMZ), the combination
/// being reserved (envytools gm107.c:1330-1335,1516-1520).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flush {
    /// Subnormals kept.
    Keep,
    /// Flush subnormal inputs and result to zero.
    Ftz,
    /// FTZ, and a zero multiplicand makes the product +0 (Mesa
    /// sm50.rs:601-602,669,698; nir_opcodes.py:783-793).
    Dnz,
}
/// FFMA modifiers: negation of the product and of the addend, saturation
/// and the denormal mode. The non-immediate forms encode rounding at bits
/// 51..52; only round-to-nearest is modeled, so the decoder rejects the
/// directed modes and the IR carries no rounding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FFmaModifiers {
    pub product_negate: bool,
    pub addend_negate: bool,
    pub saturate: bool,
    pub flush: Flush,
}
/// The flags R2P writes: P0..P6, or the condition-code flags Z, N, C and V
/// in Ryujinx's flag order (Instructions/InstEmitMove.cs:254-270,
/// InstEmitHelper.cs:12-29, MIT; envytools gm107.c:1194-1207).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FlagBank {
    Predicates,
    ConditionCodes,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoteOp {
    All,
    Any,
    Eq,
}
/// An unsigned 28-bit field, carried in a u32 that holds only its value
/// range: VOTE.VTG immediate20..47 (Ryujinx Decoders/InstDecoders.cs:5231,
/// Imm28).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct U28(u32);
impl U28 {
    /// The low 28 bits of `bits`.
    pub const fn low_bits(bits: u64) -> Self {
        Self((bits & 0x0fff_ffff) as u32)
    }

    /// `value` if it fits in 28 bits.
    pub const fn new(value: u32) -> Option<Self> {
        if value <= 0x0fff_ffff {
            Some(Self(value))
        } else {
            None
        }
    }

    pub const fn get(self) -> u32 {
        self.0
    }
}
impl fmt::LowerHex for U28 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::LowerHex::fmt(&self.0, f)
    }
}
/// Global ATOM/RED data width. ATOM type 49..52 and RED type 20..23 name
/// U32=0, S32=1, U64=2 and S64=5: envytools gm107.c:688-694,731-737;
/// Mesa nak/sm50.rs:2718-2731,2763-2777.
/// Each width's discriminant is its register count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum AtomicWidth {
    B32 = 1,
    B64 = 2,
}
impl AtomicWidth {
    /// Registers moved: one word, or an even pair.
    pub fn registers(self) -> u8 {
        self as u8
    }
    /// The memory size one access moves.
    pub fn size(self) -> LsSize2 {
        [LsSize2::B32, LsSize2::B64][usize::from(self as u8 - 1)]
    }
}
/// VOTE.VTG modes, envytools envydis/gm107.c:1633-1637.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VoteVtgMode {
    R,
    A,
    Ra,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtomicOp {
    Add,
    Min,
    Max,
    And,
    Or,
    Xor,
    Exchange,
}
/// XMAD c operand mode, Ryujinx Decoders/InstDecoders.cs:814-821 (XmadCop).
/// CSFU (3) decodes only for the six words of provenance 0087.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum XmadCop {
    /// c as is.
    Cfull = 0,
    /// The low half of c.
    Clo = 1,
    /// The high half of c.
    Chi = 2,
    /// c minus (sign of each widened half << 16), Ryujinx
    /// InstEmitIntegerArithmetic.cs:629-634.
    Csfu = 3,
    /// c + (b << 16).
    Cbcc = 4,
}

/// IMAD AvgMode 51..52 (Ryujinx Decoders/InstDecoders.cs:354-359,
/// InstEmitIntegerArithmetic.cs:420-421): NegA negates the multiplier B,
/// NegB the addend C. PlusOne has no Ryujinx semantics and never decodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MulNegation {
    None = 0,
    Multiplier = 1,
    Addend = 2,
}
/// IMADSP operand extraction, numbered as the A select field 48..50
/// (Ryujinx Decoders/InstDecoders.cs:831-841, ImadspASelect). Mesa
/// nak/ir.rs IMadSpSrcType::cast gives the meaning: the whole word, its
/// low 24 bits, or one 16-bit half, zero- or sign-extended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum IMadSpType {
    U32 = 0,
    S32 = 1,
    U24 = 2,
    S24 = 3,
    U16Lo = 4,
    S16Lo = 5,
    U16Hi = 6,
    S16Hi = 7,
}
/// MUFU function, Ryujinx Decoders/InstDecoders.cs:494-505 (MufuOp).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum MufuOp {
    Cos = 0,
    Sin = 1,
    Ex2 = 2,
    Lg2 = 3,
    Rcp = 4,
    Rsq = 5,
    Rcp64h = 6,
    Rsq64h = 7,
    Sqrt = 8,
}
/// LDG access size, Ryujinx Decoders/InstDecoders.cs:426-436 (LsSize).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum LsSize {
    U8 = 0,
    S8 = 1,
    U16 = 2,
    S16 = 3,
    B32 = 4,
    B64 = 5,
    B128 = 6,
    /// Four ordinary words, as B128 (Ryujinx InstEmitMemory.cs:489-509).
    UB128 = 7,
}
impl LsSize {
    /// The access this size performs: UB128 reads the same four words as
    /// B128 (Ryujinx InstEmitMemory.cs:489-509, GetStorageKind and
    /// GetVectorCount).
    pub fn access(self) -> LsSize2 {
        LS_ACCESS[self as usize]
    }
}
/// The LsSize2 each LsSize performs, indexed by LsSize.
const LS_ACCESS: [LsSize2; 8] = [
    LsSize2::U8,
    LsSize2::S8,
    LsSize2::U16,
    LsSize2::S16,
    LsSize2::B32,
    LsSize2::B64,
    LsSize2::B128,
    LsSize2::B128,
];
/// Registers each LsSize2 moves, indexed by LsSize2.
const LS_REGISTERS: [u8; 7] = [1, 1, 1, 1, 1, 2, 4];
/// Natural byte alignment of each LsSize2, indexed by LsSize2.
const LS_ALIGNMENT: [u32; 7] = [1, 1, 2, 2, 4, 8, 16];
/// LDL/LDS/STL/STS/STG access size, Ryujinx Decoders/InstDecoders.cs:438-447
/// (LsSize2). Value 7 is unassigned and is not decoded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum LsSize2 {
    U8 = 0,
    S8 = 1,
    U16 = 2,
    S16 = 3,
    B32 = 4,
    B64 = 5,
    B128 = 6,
}

/// LDC address mode established at decode. Ryujinx `AddressMode.Il` leaves
/// the encoded constant-buffer slot and register byte address unchanged.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LdcAddressMode {
    #[default]
    Default,
    Il,
    /// The address register is a packed constant-buffer selector and byte
    /// address: high 16 bits extend the encoded binding, low 16 bits extend
    /// the encoded byte offset.
    IndexedSegmented,
    /// Maxwell's distinct ISL encoding has the same packed bank/address
    /// execution road as IS, but remains a separate established state.
    IndexedSegmentedLocal,
}

/// LDC access width established once at decode. Maxwell LDC has exactly six
/// assigned width states; unlike the related local-memory encoding it has no
/// B128 state (Ryujinx Decoders/InstDecoders.cs, `InstLdc`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LdcWidth {
    U8,
    S8,
    U16,
    S16,
    B32,
    B64,
}

impl LdcWidth {
    pub const fn encoded_name(self) -> &'static str {
        ["0", "1", "2", "3", "4", "5"][self as usize]
    }

    pub const fn registers(self) -> u8 {
        [1, 1, 1, 1, 1, 2][self as usize]
    }
}

impl LdcAddressMode {
    pub const fn suffix(self) -> &'static str {
        ["", ".IL", ".IS", ".ISL"][self as usize]
    }
}

/// STL cache policy, envytools `gm107.c`:594-599. These policies control the
/// local-memory cache path; they do not change the value written.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LocalStoreCache {
    #[default]
    Default,
    Cg,
    Cs,
    Wt,
}

impl LocalStoreCache {
    pub const fn suffix(self) -> &'static str {
        ["", ".CG", ".CS", ".WT"][self as usize]
    }
}
impl LsSize2 {
    /// Registers the access moves: B64 2, B128 4, otherwise 1 (Ryujinx
    /// InstEmitMemory.cs:325-332).
    pub fn registers(self) -> u8 {
        LS_REGISTERS[self as usize]
    }

    /// Natural byte alignment of the access.
    pub fn alignment(self) -> u32 {
        LS_ALIGNMENT[self as usize]
    }

    /// Bit width and signedness of a sub-word access; None for B32, B64
    /// and B128 (Ryujinx InstEmitMemory.cs:552-574, ExtractSmallInt).
    pub fn subword(self) -> Option<(u32, bool)> {
        match self {
            Self::U8 => Some((8, false)),
            Self::S8 => Some((8, true)),
            Self::U16 => Some((16, false)),
            Self::S16 => Some((16, true)),
            Self::B32 | Self::B64 | Self::B128 => None,
        }
    }
}
/// An LDG or STG and its size. LDG takes LsSize and STG LsSize2 (Ryujinx
/// Decoders/InstDecoders.cs:3541-3549 InstLdg, 4530-4538 InstStg).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GlobalAccess {
    Load(LsSize),
    Store(LsSize2),
}
/// LOP operation, Ryujinx Decoders/InstDecoders.cs:471-477 (LogicOp).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum LogicOp {
    And = 0,
    Or = 1,
    Xor = 2,
    /// The second source after its inversion.
    PassB = 3,
}
/// LOP/LOP3 predicate output mode, Ryujinx Decoders/InstDecoders.cs:463-469
/// (PredicateOp); InstEmitIntegerLogical.cs:145-160 writes false, true,
/// result zero or result nonzero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum PredicateOp {
    F = 0,
    T = 1,
    Z = 2,
    Nz = 3,
}
/// A LOP/LOP3 predicate write: the mode and a destination other than PT.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PredicateOutput {
    pub op: PredicateOp,
    pub dst: u8,
}
/// Packed arithmetic uses float32 lanes before the selected output packing.
/// Ryujinx Instructions/InstEmitFloatArithmetic.cs:483-528.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalfOp {
    Add,
    Mul,
    Fma,
}

/// envytools envydis/gm107.c:1157-1168,1083-1087,1782-1786:
/// preserve=0, FTZ=1, FMZ=2; reserved=3 has no IR representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HalfFlush {
    Preserve,
    Ftz,
    Fmz,
}

/// The attribute one static ALD lane names, decoded once from its byte
/// offset as Ryujinx Instructions/AttributeMap.cs:63-92 maps it. Ryujinx is
/// MIT licensed, Copyright (c) Ryujinx Team and Contributors; the offsets
/// and stage masks below are ported from that file under its notice in
/// emitter.rs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Attribute {
    /// No map entry: AttributeMap.cs:129-133 logs the offset and loads 0.
    Unmapped,
    PrimitiveId,
    Layer,
    ViewportIndex,
    PointSize,
    Position,
    /// Generic vec4 locations 0x80..0x280 (IoVariable.UserDefined).
    Generic,
    ClipDistance,
    PointCoord,
    TessellationCoord,
    InstanceId,
    VertexId,
    ViewportMask,
    FrontFacing,
    /// Colors 0x280..0x2c0, FogCoord 0x2e8 and TexCoords 0x300..0x3a0.
    /// InstEmitAttribute.cs:344-366 moves each to a free user attribute
    /// chosen from the linked stages' usage before the map lookup.
    FixedFunction,
    TessellationLevelOuter,
    TessellationLevelInner,
    PatchGeneric,
    /// A fixed-function offset read with P: the same move lands in the
    /// per-patch generic range (AttributeMap.cs:92) or misses the map.
    PatchFixedFunction,
}

/// Ryujinx ShaderStage values as a bit set (AttributeMap.cs:11-25).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Stages(u8);

impl Stages {
    const NONE: Self = Self(0);
    const VERTEX: u8 = 1 << 1;
    const TESS_CONTROL: u8 = 1 << 2;
    const TESS_EVALUATION: u8 = 1 << 3;
    const GEOMETRY: u8 = 1 << 4;
    const FRAGMENT: u8 = 1 << 5;
    const V: Self = Self(Self::VERTEX);
    const TCS: Self = Self(Self::TESS_CONTROL);
    const TES: Self = Self(Self::TESS_EVALUATION);
    const G: Self = Self(Self::GEOMETRY);
    const F: Self = Self(Self::FRAGMENT);
    const VTG: Self =
        Self(Self::VERTEX | Self::TESS_CONTROL | Self::TESS_EVALUATION | Self::GEOMETRY);
    const TGF: Self =
        Self(Self::TESS_CONTROL | Self::TESS_EVALUATION | Self::GEOMETRY | Self::FRAGMENT);
    const ALL_GRAPHICS: Self = Self(Self::VTG.0 | Self::FRAGMENT);

    fn contains(self, stage: crate::headers::Stage) -> bool {
        use crate::headers::Stage;
        let bit = match stage {
            Stage::Compute => 1,
            Stage::Vertex => Self::VERTEX,
            Stage::TessellationControl => Self::TESS_CONTROL,
            Stage::TessellationEvaluation => Self::TESS_EVALUATION,
            Stage::Geometry => Self::GEOMETRY,
            Stage::Fragment => Self::FRAGMENT,
        };
        self.0 & bit != 0
    }
}

/// The attribute bank an ALD reads, chosen by its O bit (Mesa sm50.rs:
/// 2873; Ryujinx InstEmitAttribute.cs:49 picks StorageKind.Output or
/// Input from it).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AttributeBank {
    Input,
    Output,
}

impl AttributeBank {
    /// Decode the O bit.
    pub fn decode(o: bool) -> Self {
        if o {
            Self::Output
        } else {
            Self::Input
        }
    }
}

/// The vertex an input-bank ALD reads, from Rb (bits 39..46, Mesa
/// sm50.rs:2871), decided once at decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrimitiveVertex {
    /// Rb is RZ, or the attribute has no primitive vertex (PrimitiveId and
    /// TessCoord, Ryujinx InstEmitAttribute.cs:25-32,297-302): the
    /// invocation's own vertex.
    Current,
    /// Rb holds the primitive vertex, the handle ISBERD returns (Mesa
    /// nak_nir_lower_vtg_io.c:80-82; Ryujinx AttributeMap.cs:323-333 uses
    /// it as the input array index).
    Primitive(Register),
}

impl PrimitiveVertex {
    /// Ryujinx InstEmitAttribute.cs:32 snapshots the vertex before any
    /// destination is written: a primitive vertex is read with `read`; the
    /// current vertex reads nothing.
    pub fn snapshot(self, read: impl FnOnce(Register) -> u32) -> Option<u32> {
        match self {
            Self::Current => None,
            Self::Primitive(register) => Some(read(register)),
        }
    }
}

/// The vertex an output-bank ALD reads, from Rb (bits 39..46), decided
/// once at decode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutputVertex {
    /// Rb is RZ: the invocation's own vertex.
    Current,
    /// Rb is the lane whose per-vertex outputs are read. Mesa computes it
    /// in tessellation control as lane + vertex - invocation
    /// (nak_nir_lower_vtg_io.c:10-49,86-90); Ryujinx AttributeMap.cs:
    /// 323-328 reads no vertex for O.
    Lane(Register),
}

impl OutputVertex {
    /// The vertex index this read uses: none for the current vertex, and
    /// for a lane whatever `control_point` makes of Rb, which is none
    /// where the stage has no per-vertex outputs to index by lane.
    pub fn snapshot(self, control_point: impl FnOnce(Register) -> Option<u32>) -> Option<u32> {
        match self {
            Self::Current => None,
            Self::Lane(register) => control_point(register),
        }
    }
}

/// One static ALD lane: its attribute and the bank it reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AttributeLane {
    pub attribute: Attribute,
    pub bank: AttributeBank,
}

impl AttributeLane {
    /// Decode the lane at byte `offset` of a static ALD. Offsets follow
    /// AttributeMap.cs:63-81 (regular) and :90-92 (per patch), with the
    /// fixed-function ranges of InstEmitAttribute.cs:344-366. TessCoord
    /// always reads the input bank (InstEmitAttribute.cs:56,98,304-307).
    fn decode(offset: u16, patch: bool, bank: AttributeBank) -> Self {
        let attribute = if !offset.is_multiple_of(4) {
            // Every map key is a multiple of 4, and the fixed-function move
            // keeps the low bits (InstEmitAttribute.cs:376).
            Attribute::Unmapped
        } else if patch {
            match offset {
                0x00..=0x0c => Attribute::TessellationLevelOuter,
                0x10..=0x14 => Attribute::TessellationLevelInner,
                0x18..=0x1fc => Attribute::PatchGeneric,
                0x280..=0x2bc | 0x2e8 | 0x300..=0x39c => Attribute::PatchFixedFunction,
                _ => Attribute::Unmapped,
            }
        } else {
            match offset {
                0x60 => Attribute::PrimitiveId,
                0x64 => Attribute::Layer,
                0x68 => Attribute::ViewportIndex,
                0x6c => Attribute::PointSize,
                0x70..=0x7c => Attribute::Position,
                0x80..=0x27c => Attribute::Generic,
                0x280..=0x2bc | 0x2e8 | 0x300..=0x39c => Attribute::FixedFunction,
                0x2c0..=0x2dc => Attribute::ClipDistance,
                0x2e0..=0x2e4 => Attribute::PointCoord,
                0x2f0..=0x2f4 => Attribute::TessellationCoord,
                0x2f8 => Attribute::InstanceId,
                0x2fc => Attribute::VertexId,
                0x3a0 => Attribute::ViewportMask,
                0x3fc => Attribute::FrontFacing,
                _ => Attribute::Unmapped,
            }
        };
        Self {
            attribute,
            bank: if attribute == Attribute::TessellationCoord {
                AttributeBank::Input
            } else {
                bank
            },
        }
    }

    /// Ryujinx's input or output stage mask for this lane, AttributeMap.cs:
    /// 63-92. A fixed-function lane takes the union of its own entry and
    /// the user attribute it may move to, since only linkage decides.
    fn stages(self) -> Stages {
        let (input, output) = match self.attribute {
            Attribute::Unmapped => (Stages::NONE, Stages::NONE),
            Attribute::PrimitiveId => (Stages::TGF, Stages::G),
            Attribute::Layer | Attribute::ViewportIndex | Attribute::ViewportMask => {
                (Stages::F, Stages::VTG)
            }
            Attribute::PointSize => (Stages::NONE, Stages::VTG),
            Attribute::Position | Attribute::ClipDistance => (Stages::TGF, Stages::VTG),
            Attribute::Generic | Attribute::FixedFunction => (Stages::ALL_GRAPHICS, Stages::VTG),
            Attribute::PointCoord | Attribute::FrontFacing => (Stages::F, Stages::NONE),
            Attribute::TessellationCoord => (Stages::TES, Stages::NONE),
            Attribute::InstanceId | Attribute::VertexId => (Stages::V, Stages::NONE),
            Attribute::TessellationLevelOuter
            | Attribute::TessellationLevelInner
            | Attribute::PatchGeneric
            | Attribute::PatchFixedFunction => (Stages::TES, Stages::TCS),
        };
        match self.bank {
            AttributeBank::Input => input,
            AttributeBank::Output => output,
        }
    }

    /// Whether this lane reads an attribute in `stage`. Otherwise Ryujinx
    /// AttributeMap.cs:129-141 logs the offset and the lane loads 0.
    pub fn reads_in(self, stage: crate::headers::Stage) -> bool {
        self.stages().contains(stage)
    }

    /// Ryujinx AttributeMap.cs:131,139 log text for a lane that loads 0.
    pub fn zero_reason(self, offset: u16, stage: crate::headers::Stage) -> String {
        match self.attribute {
            Attribute::Unmapped => format!("Attribute offset 0x{offset:X} is not valid."),
            // Ryujinx names the generic banks IoVariable.UserDefined.
            Attribute::Generic | Attribute::PatchGeneric => format!(
                "Attribute offset 0x{offset:X} (UserDefined) is not valid for stage {stage:?}."
            ),
            attribute => format!(
                "Attribute offset 0x{offset:X} ({attribute:?}) is not valid for stage {stage:?}."
            ),
        }
    }
}

/// Whether `stage` has the per-patch `bank` an ALD with P reads: outputs
/// in tessellation control, inputs in tessellation evaluation (Ryujinx
/// AttributeMap.cs:90-92).
pub fn patch_bank_in(stage: crate::headers::Stage, bank: AttributeBank) -> bool {
    AttributeLane {
        attribute: Attribute::PatchGeneric,
        bank,
    }
    .reads_in(stage)
}

/// The lanes of one static ALD, in component order; `components` of them
/// are meaningful (InstEmitAttribute.cs:34-55 advances 4 bytes per lane).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AttributeLanes {
    lanes: [AttributeLane; 4],
    count: u8,
}

impl AttributeLanes {
    fn decode(address: u16, components: u8, patch: bool, bank: AttributeBank) -> Self {
        let mut lanes = [AttributeLane::decode(address, patch, bank); 4];
        for (index, lane) in lanes.iter_mut().enumerate().skip(1) {
            *lane = AttributeLane::decode(address + index as u16 * 4, patch, bank);
        }
        Self {
            lanes,
            count: components.min(4),
        }
    }

    /// The lanes of a per-vertex ALD (no P), AttributeMap.cs:63-81.
    pub fn vertex(address: u16, components: u8, bank: AttributeBank) -> Self {
        Self::decode(address, components, false, bank)
    }

    /// The lanes of a per-patch ALD (P), AttributeMap.cs:90-92.
    pub fn patch(address: u16, components: u8, bank: AttributeBank) -> Self {
        Self::decode(address, components, true, bank)
    }

    /// Each lane with its byte offset.
    pub fn iter(self, address: u16) -> impl Iterator<Item = (u16, AttributeLane)> {
        (0..self.count).map(move |index| {
            (
                address + u16::from(index) * 4,
                self.lanes[usize::from(index)],
            )
        })
    }

    /// The load these lanes form: `mapped` when one lane names an
    /// attribute, `unmapped` when none does (AttributeMap.cs:129-133).
    pub fn load(self, mapped: InstructionKind, unmapped: InstructionKind) -> InstructionKind {
        let named = self.lanes[..usize::from(self.count)]
            .iter()
            .map(|lane| ATTRIBUTE_NAMED[lane.attribute as usize])
            .sum::<usize>();
        [unmapped, mapped][named.min(1)]
    }

    pub fn all_unmapped(self) -> bool {
        self.lanes[..usize::from(self.count)]
            .iter()
            .all(|lane| lane.attribute == Attribute::Unmapped)
    }
}

/// Every instruction type, each its own thing. The one place an
/// `InstructionKind` meets its behavior: it hands its instruction to a
/// [`Visit`], which runs that instruction type's own trait methods.
macro_rules! instruction_kinds {
    ($($kind:ident),* $(,)?) => {
        #[derive(Clone, Copy, PartialEq, Eq)]
        pub enum InstructionKind {
            $($kind(ops::$kind),)*
        }

        impl InstructionKind {
            /// Hands this instruction to `visitor` as its own type.
            pub fn visit<V: Visit>(&self, visitor: V) -> V::Out {
                match self {
                    $(Self::$kind(kind) => visitor.visit(kind),)*
                }
            }

            /// Hands this instruction to `visitor` as its own type, for change.
            pub fn visit_mut<V: VisitMut>(&mut self, visitor: V) -> V::Out {
                match self {
                    $(Self::$kind(kind) => visitor.visit(kind),)*
                }
            }
        }

        $(impl From<ops::$kind> for InstructionKind {
            fn from(kind: ops::$kind) -> Self {
                Self::$kind(kind)
            }
        })*
    };
}

instruction_kinds!(
    Nop,
    VoteVtg,
    Vote,
    Shuffle,
    SwizzleAdd,
    Kill,
    S2R,
    IpaPass,
    IpaPassMulW,
    IpaSaturate,
    Mov,
    Ldc,
    IAdd,
    IAdd3,
    Lea,
    LeaHi,
    FunnelShift,
    BitFieldInsert,
    BitFieldExtract,
    IntegerMul,
    IntegerMulAdd,
    FAdd,
    FMul,
    FCmp,
    ICmp,
    HalfArithmetic,
    HalfSet,
    VideoMultiplyAdd,
    VideoMinMax,
    FFma,
    FMnMx,
    FMnMxSelect,
    MuFu,
    Rro,
    IMinMax,
    IMinMaxSelect,
    Flo,
    Popc,
    ISetP,
    ISet,
    Xmad,
    ShiftAdd,
    R2P,
    P2R,
    VideoSetP,
    FSetP,
    DoubleSetP,
    DoubleSet,
    DoubleAdd,
    DoubleMinMax,
    DoubleMultiply,
    DoubleFma,
    CSetP,
    CSet,
    PSet,
    PSetP,
    FSet,
    Sel,
    Convert,
    DoubleToInt,
    DoubleToFloat,
    DoubleRound,
    Lop,
    LopWithPredicate,
    Lop3,
    Lop3WithPredicate,
    Shl,
    Shr,
    Prmt,
    IMadSp,
    GlobalMemory,
    GlobalAtomic,
    GlobalReduce,
    LocalLoad,
    SharedLoad,
    LocalStore,
    SharedStore,
    Atomic,
    SharedCompareExchange,
    GroupMemoryBarrier,
    MemoryBarrier,
    ControlBarrier,
    GeometryOut,
    GeometryOutDynamic,
    AttributeAddress,
    IsbeRead,
    AttributeLoad,
    OutputAttributeLoad,
    PatchAttributeLoad,
    PhysicalAttributeLoad,
    PhysicalOutputAttributeLoad,
    UnmappedAttributeLoad,
    IndexedPatchLoad,
    AttributeStore,
    PhysicalAttributeStore,
    Texture,
    SurfaceAtomic,
    SurfaceReduce,
    Surface,
    Branch,
    Reconverge,
    BrxRegister,
    Brx,
    OutsideBrx,
    Schedule,
    PushTarget,
    CbufPushTarget,
    ResolvedSchedule,
    ResolvedPushTarget,
    OutsideBranch,
    OutsideSchedule,
    OutsidePushTarget,
    MisalignedBranch,
    MisalignedSchedule,
    MisalignedPushTarget,
    PopJump,
    Exit,
    Call,
    OutsideCall,
    MisalignedCall,
    OverlongInsert,
    Padding,
    Unknown,
);

/// The predicate registers an instruction writes, one bit per register
/// P0..PT. A write to PT is discarded (Ryujinx OperandHelper.cs), so its
/// bit names no register anyone reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PredicateSet(u8);

impl PredicateSet {
    pub const NONE: Self = Self(0);
    pub const ALL: Self = Self(0xff);

    /// The two destinations of a compare-and-set (ISETP, FSETP, PSETP,
    /// VSETP, DSETP, CSETP): Ryujinx InstDecoders.cs Dest and DestInv.
    pub fn pair(dst: [u8; 2]) -> Self {
        Self(
            dst.iter()
                .fold(0, |set, register| set | 1 << (register & 7)),
        )
    }

    /// One destination: VOTE's vote result, SHFL's in-bounds flag.
    pub fn one(register: u8) -> Self {
        Self(1 << (register & 7))
    }

    /// Whether `register` is written.
    pub fn contains(self, register: u8) -> bool {
        self.0 >> (register & 7) & 1 != 0
    }
}

/// What every instruction type does, whatever pass asks.
/// The general registers an instruction writes: `count` registers from
/// `base`. Writing nothing is a count of zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Written {
    pub base: Register,
    pub count: u8,
}
impl Written {
    pub const NOTHING: Self = Self {
        base: Register(255),
        count: 0,
    };
    pub fn new(base: Register, count: u8) -> Self {
        Self { base, count }
    }
    pub fn covers(self, register: u8) -> bool {
        (u16::from(self.base.0)..u16::from(self.base.0) + u16::from(self.count))
            .contains(&u16::from(register))
    }
}
/// Up to four source words; the first `len` are present. Three operands
/// at most, and DFMA's two binary64 operands are four words.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Sources {
    items: [Source; 4],
    len: u8,
}
impl Sources {
    pub const NOTHING: Self = Self {
        items: [Source::Immediate(0); 4],
        len: 0,
    };
    pub fn new([first, second, third]: [Source; 3], len: u8) -> Self {
        Self {
            items: [first, second, third, Source::Immediate(0)],
            len,
        }
    }
}
impl Sources {
    /// These sources, then `next` after them; four at most.
    pub fn then(self, next: Sources) -> Self {
        let mut items = self.items;
        let len = (self.len + next.len).min(4);
        (self.len..len).zip(next).for_each(|(slot, source)| {
            items[usize::from(slot)] = source;
        });
        Self { items, len }
    }
}
impl IntoIterator for Sources {
    type Item = Source;
    type IntoIter = std::iter::Take<std::array::IntoIter<Source, 4>>;
    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter().take(usize::from(self.len))
    }
}

/// What the module around a program is made with. Each instruction tells
/// it the things it requires through [`Kind::require`]; an instruction
/// requiring nothing tells it nothing.
pub trait Requirements {
    /// Helper demotion (KIL).
    fn demote(&mut self);
    /// A collective that observes the active set of lanes.
    fn subgroup(&mut self);
    /// Preserved binary32 denormals, signed zero and rounding.
    fn preserve_float32(&mut self);
    /// Binary64 arithmetic.
    fn float64(&mut self);
    /// 64-bit integers.
    fn int64(&mut self);
    /// A storage image written without a declared format.
    fn surface_store(&mut self);
    /// A storage image read without a declared format.
    fn surface_load(&mut self);
    /// Extended storage image formats.
    fn surface_extended(&mut self);
    /// A signed storage image view.
    fn signed_storage(&mut self);
    /// A nontemporal image store.
    fn nontemporal(&mut self);
    /// A 1D sampled image.
    fn sampled_1d(&mut self);
    /// A cube array sampled image.
    fn cube_array(&mut self);
    /// Non-constant gather offsets.
    fn texture_offsets(&mut self);
    /// A 1D storage image.
    fn image_1d(&mut self);
    /// A buffer image.
    fn image_buffer(&mut self);
    /// An image size or level query.
    fn image_query(&mut self);
    /// Any image.
    fn images(&mut self);
    /// A subgroup shuffle.
    fn shuffle(&mut self);
    /// A subgroup vote.
    fn vote(&mut self);
    /// The lane id.
    fn lane_id(&mut self);
    /// A read of another lane's output control point.
    fn output_lanes(&mut self);
    /// The fragment coordinate.
    fn frag_coord(&mut self);
    /// A texture result helpers may leave undefined (NODEP).
    fn nodep(&mut self);
    /// A generic input addressed at run time.
    fn dynamic_input(&mut self);
    /// Half-precision vector arithmetic.
    fn half(&mut self);
    /// The uvec2 the half paths pack through.
    fn half_uvec2(&mut self);
    /// A constant-buffer bank chosen at run time.
    fn ldc_indexed(&mut self);
    /// Shared memory.
    fn shared(&mut self);
    /// Local memory.
    fn local(&mut self);
    /// A BRX through a register.
    fn register_brx(&mut self);
    /// SR_Y_DIRECTION.
    fn y_direction(&mut self);
    /// SR_HELPER (the HelperInvocation input).
    fn helper(&mut self);
    /// SR_INVOCATION_INFO.
    fn invocation_info(&mut self);
    /// SR_TID and its components (LocalInvocationId).
    fn local_invocation_id(&mut self);
    /// SR_CTAID components (WorkgroupId).
    fn workgroup_id(&mut self);
    /// A lane-mask system register, by its index (0x38..=0x3c,
    /// envytools gm107.c:173-177).
    fn lane_mask(&mut self, index: u8);
}

/// What a pass does at an instruction, by whether the emitter can
/// translate it.
pub trait Translation {
    type Out;
    /// A word the decoder could not name.
    fn unknown(self) -> Self::Out;
    /// A zero word: padding, unless control reaches it.
    fn padding(self) -> Self::Out;
    /// A form the decoder names whose lowering is not built.
    fn unbuilt(self) -> Self::Out;
    /// A form a proof found to be one Shadowbox does not take.
    fn unwanted(self) -> Self::Out;
    /// A stack instruction whose consumers were not proven.
    fn unresolved_stack(self) -> Self::Out;
    /// An OUT whose stream no proof named.
    fn unproven_stream(self) -> Self::Out;
    /// Every instruction the emitter translates.
    fn known(self) -> Self::Out;
}

/// A requirement table entry that requires nothing.
fn nothing<M>(_: &mut M) {}

/// What S2R requires by system register index (envytools
/// gm107.c:121-177).
fn system_register<M: Requirements>(idx: u8, module: &mut M) {
    match idx {
        0x00 => module.lane_id(),
        0x12 => module.y_direction(),
        0x13 => module.helper(),
        0x1d => module.invocation_info(),
        0x20..=0x23 => module.local_invocation_id(),
        0x25..=0x27 => module.workgroup_id(),
        0x38..=0x3c => module.lane_mask(idx),
        _ => (),
    }
}
/// By MufuOp: the 64-bit reciprocal forms use binary64.
const MUFU_FLOAT64: [usize; 9] = [0, 0, 0, 0, 0, 0, 1, 1, 0];
/// By texture dimension (Mesa set_tex_dim): Sampled1D for 0 and 1,
/// SampledCubeArray for 7.
fn texture_dim<M: Requirements>(dim: u8, module: &mut M) {
    [
        M::sampled_1d,
        M::sampled_1d,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        M::cube_array,
    ][usize::from(dim)](module)
}
/// By surface target: Image1D for t1d/a1d, ImageBuffer for b1d.
fn surface_target<M: Requirements>(target: u8, module: &mut M) {
    [
        M::image_1d,
        M::image_buffer,
        M::image_1d,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
    ][usize::from(target)](module)
}
/// A raw (format 1) surface by SuSize: extended formats for 0..=3 and 5,
/// a signed view for 1 and 3.
fn raw_surface<M: Requirements>(size: u8, module: &mut M) {
    [
        M::surface_extended,
        |m: &mut M| {
            m.surface_extended();
            m.signed_storage();
        },
        M::surface_extended,
        |m: &mut M| {
            m.surface_extended();
            m.signed_storage();
        },
        nothing,
        M::surface_extended,
        nothing,
        nothing,
    ][usize::from(size)](module)
}
/// MOV quad lanes 1..=14 read the lane id (Mesa sm50.rs:1936-1946).
const QUAD_LANE_ID: [usize; 16] = [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0];

impl Conversion {
    fn require<M: Requirements>(self, module: &mut M) {
        match self {
            // Integer size 3 (64-bit) needs the 64-bit types; a 32-bit to
            // 32-bit conversion in a directed mode rounds through a 64-bit
            // integer.
            Conversion::I2F {
                dst_size,
                src_size,
                round,
                ..
            } => {
                [nothing, nothing, nothing, M::float64][usize::from(dst_size)](module);
                [nothing, nothing, nothing, M::int64][usize::from(src_size)](module);
                [
                    nothing,
                    [nothing, M::int64, M::int64, M::int64][round as usize],
                ][usize::from(src_size == 2 && dst_size == 2)](module);
            }
            Conversion::F2F { destination, .. } => {
                module.preserve_float32();
                [nothing, nothing, M::float64][destination as usize](module);
            }
            Conversion::F2I { .. } | Conversion::I2I { .. } => (),
        }
    }
}

impl Attribute {
    /// A fragment Position lane reads FragCoord.
    fn require<M: Requirements>(self, module: &mut M) {
        if let Attribute::Position = self {
            module.frag_coord()
        }
    }
}

/// By Attribute: 1 for a lane naming an attribute, 0 for Unmapped.
const ATTRIBUTE_NAMED: [usize; 19] = {
    let mut named = [1; 19];
    named[Attribute::Unmapped as usize] = 0;
    named
};

impl TexOp {
    /// What the op requires.
    fn require<M: Requirements>(self, module: &mut M) {
        match self {
            TexOp::Gather {
                offsets: GatherOffsets::One | GatherOffsets::Four,
                ..
            } => module.texture_offsets(),
            TexOp::Query { .. } | TexOp::Lod => module.image_query(),
            TexOp::Sample { .. } | TexOp::Fetch { .. } | TexOp::Gather { .. } | TexOp::Grad => (),
        }
    }
}

impl OutputVertex {
    fn require<M: Requirements>(self, module: &mut M) {
        match self {
            OutputVertex::Current => (),
            OutputVertex::Lane(_) => module.output_lanes(),
        }
    }
}

pub trait Kind: Copy + fmt::Debug + fmt::Display + Into<InstructionKind> {
    /// Whether this instruction may write a predicate register (or ends a
    /// straight-line run). Required, so a new kind states it.
    fn may_write_predicate(&self) -> bool;
    /// General registers written (first register and count).
    fn written_registers(&self) -> Written {
        Written::NOTHING
    }
    /// Every destination range. TEXS has two disjoint pairs (Ryujinx
    /// InstEmitTexture.cs:613-655; envytools gm107.c:1967-1968).
    fn destinations(&self) -> [Written; 2] {
        [self.written_registers(), Written::NOTHING]
    }
    fn writes_register(&self, register: u8) -> bool {
        let [first, second] = self.destinations();
        first.covers(register) || second.covers(register)
    }
    /// Source operands, in order.
    fn source_slots(&self) -> Sources {
        Sources::NOTHING
    }
    /// What the module around this instruction is made with because of
    /// it. A kind requiring nothing tells `module` nothing.
    fn require<M: Requirements>(&self, _module: &mut M) {}
    /// What `pass` does at this instruction, by whether the emitter can
    /// translate it. Every kind it can translate is known.
    fn translation<T: Translation>(&self, pass: T) -> T::Out {
        pass.known()
    }
    /// The rounding this instruction multiplies an FMUL product in. Only
    /// FMUL multiplies in one; every other kind adds nearest-even.
    fn product_rounding(&self) -> FMulRounding {
        FMulRounding::NearestEven
    }
    /// Ryujinx Translator.cs:339-393: whether the guard predicate wraps the
    /// whole instruction, rather than selecting each written value.
    fn guards_whole(&self) -> bool {
        false
    }
    /// Marks the dispatch blocks this instruction starts: its jump targets
    /// and, for a block end, the instruction after it in `next`.
    fn mark_block_starts<E: Extend<u32>>(&self, _starts: &mut E, _next: &[Instruction]) {}
    /// This kind once stack resolution proved its consumers.
    fn resolved(self) -> InstructionKind {
        self.into()
    }
    /// The combined form the emitter's IMUL, ATOM and SUATOM paths still
    /// read, until those forms carry a destination and no addend of their
    /// own. Every other kind is its own form.
    fn legacy_form(self) -> InstructionKind {
        self.into()
    }
    /// Whether this instruction reads a traced handle: texture and surface
    /// operations, BFI with a register range, and global memory when the
    /// caller traces global addresses.
    fn handle_use(&self, _include_global: bool) -> bool {
        false
    }
    /// Whether this instruction really writes a predicate register. Control
    /// flow ends a straight-line run but changes no predicate register
    /// (Mesa sm50.rs:3020-3052; Ryujinx InstEmitFlowControl.cs:239-275).
    fn writes_predicate_register(&self) -> bool {
        self.may_write_predicate()
    }
    /// The predicate registers this instruction writes. A writer with no
    /// exact list names every register.
    fn written_predicates(&self) -> PredicateSet {
        [PredicateSet::NONE, PredicateSet::ALL][usize::from(self.writes_predicate_register())]
    }
    /// Whether every destination register is written whole. A MOV with a
    /// partial quad-lane mask may keep the previous value (Mesa
    /// sm50.rs:1936,1941,1946).
    fn whole_write(&self) -> bool {
        true
    }
    /// How this instruction bounds code detached from the entry run.
    fn bound_detached<B: DetachedBounds>(&self, _bounds: &mut B) {}
    /// What an unguarded instruction proves about generic attribute
    /// addresses in registers: AL2P's sum (Ryujinx
    /// InstEmitAttribute.cs:12-16).
    fn prove_address(&self, _proven: &mut ProvenAddresses) {}
    /// This instruction once its physical attribute offset is a proven
    /// generic address (InstEmitAttribute.cs:43-51,118-127). A kind with
    /// no physical offset stays itself.
    fn at_proven_address(&self, _proven: &ProvenAddresses) -> InstructionKind {
        (*self).into()
    }
    /// The attribute interface variables this instruction touches,
    /// declared through `io`. A kind touching none declares nothing.
    fn declare_attributes<A: AttributeIo>(&self, _io: &mut A) -> Result<(), A::Error> {
        Ok(())
    }
    /// The generic banks this instruction addresses at run time, input
    /// then output (Ryujinx InstEmitAttribute.cs:43-51,118-127).
    fn dynamic_banks(&self) -> [bool; 2] {
        [false; 2]
    }
    /// Whether this instruction reads SR_INVOCATION_ID (envytools
    /// gm107.c:135; Ryujinx InstEmitMove.cs:82-84).
    fn reads_invocation_id(&self) -> bool {
        false
    }
    /// Whether this instruction reads SR_INVOCATION_INFO, which needs the
    /// driver's input topology (Ryujinx InstEmitMove.cs:94-114).
    fn reads_invocation_info(&self) -> bool {
        false
    }
    /// The geometry streams this instruction cuts or emits on.
    fn stream_marks(&self) -> StreamMarks {
        StreamMarks::NONE
    }
    /// The generic output locations this instruction reads or writes,
    /// every one where its address is chosen at run time (Ryujinx
    /// AttributeUsage.cs:91-97; Decoder.cs:349-365,374-385,410-417).
    fn used_outputs(&self) -> u32 {
        0
    }
}

/// What OUT instructions do to geometry streams.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StreamMarks {
    /// An OUT is present.
    pub outs: bool,
    /// An OUT names a stream past 3 or neither cuts nor emits.
    pub unencodable: bool,
    /// Each stream an OUT cuts or emits on.
    pub streams: u8,
    /// Each stream an OUT emits on.
    pub emitted: u8,
}

impl StreamMarks {
    pub const NONE: Self = Self {
        outs: false,
        unencodable: false,
        streams: 0,
        emitted: 0,
    };

    /// Both marks together.
    pub fn with(self, other: Self) -> Self {
        Self {
            outs: self.outs | other.outs,
            unencodable: self.unencodable | other.unencodable,
            streams: self.streams | other.streams,
            emitted: self.emitted | other.emitted,
        }
    }
}

/// The generic output locations `components` words from `address` touch.
fn generic_outputs(address: u16, components: u8) -> u32 {
    (0..components).fold(0, |used, c| {
        let address = address + u16::from(c) * 4;
        let generic = (0x80..0x280).contains(&address);
        used | u32::from(generic) << ((address.wrapping_sub(0x80) / 16) & 31)
    })
}

/// The generic attribute address AL2P left in each register, proven within
/// one block; 0, never a generic address, where nothing is proven.
pub struct ProvenAddresses([u16; 256]);

impl ProvenAddresses {
    pub const EMPTY: Self = Self([0; 256]);

    /// Forget every register `kind` writes (RZ is never proven).
    pub fn forget(&mut self, kind: InstructionKind) {
        for Written { base, count } in kind.destinations() {
            let start = usize::from(base.0);
            let end = (start + usize::from(count)).min(255);
            self.0[start..end].fill(0);
        }
    }

    /// The address proven in `register`, or 0.
    pub fn at(&self, register: Register) -> u16 {
        self.0[usize::from(register.0)]
    }

    /// `register` now holds `address`; a write to RZ proves nothing.
    fn prove(&mut self, register: Register, address: u16) {
        self.0[usize::from(register.0)] = address;
        self.0[255] = 0;
    }
}

/// What an instruction tells a pass about code detached from the entry run.
pub trait DetachedBounds {
    /// The entry run ends here.
    fn end_entry(&mut self);
    /// The block at `slot` starts detached code.
    fn start(&mut self, slot: u32);
    /// The instruction after this one starts detached code.
    fn fall_through(&mut self);
    /// The entry run also ends at `target`.
    fn end_at(&mut self, target: BranchTarget);
}

/// One way an instruction declares its attributes into `A`.
type Declare<A> = fn(&mut A) -> Result<(), <A as AttributeIo>::Error>;
/// One way instruction `K` declares its attributes into `A`.
type DeclareFrom<K, A> = fn(&K, &mut A) -> Result<(), <A as AttributeIo>::Error>;

/// Where a geometry or tessellation stage declares the attribute
/// interface an instruction touches.
pub trait AttributeIo {
    type Error;
    /// The instruction has no meaning in this stage's interface.
    fn reject(&mut self) -> Result<(), Self::Error>;
    /// Static `lanes` read at `address`, in the patch bank or not.
    fn lanes(
        &mut self,
        dst: Register,
        address: u16,
        lanes: AttributeLanes,
        patch: bool,
    ) -> Result<(), Self::Error>;
    /// A scalar physical generic access to the output bank or the input.
    fn physical(&mut self, output: bool) -> Result<(), Self::Error>;
    /// An indexed load of the whole patch `bank`.
    fn patch_bank(&mut self, bank: AttributeBank) -> Result<(), Self::Error>;
    /// A static output word at `address`, in the patch bank or not.
    fn store(&mut self, address: u16, patch: bool) -> Result<(), Self::Error>;
}

macro_rules! kind_calls {
    ($($visitor:ident => $method:ident -> $out:ty;)*) => {$(
        struct $visitor;
        impl Visit for $visitor {
            type Out = $out;
            fn visit<K: Kind>(self, kind: &K) -> $out {
                kind.$method()
            }
        }
    )*};
}
kind_calls! {
    MayWritePredicate => may_write_predicate -> bool;
    WrittenRegisters => written_registers -> Written;
    Destinations => destinations -> [Written; 2];
    SourceSlots => source_slots -> Sources;
    ProductRounding => product_rounding -> FMulRounding;
    GuardsWhole => guards_whole -> bool;
    WritesPredicateRegister => writes_predicate_register -> bool;
    WrittenPredicates => written_predicates -> PredicateSet;
    WholeWrite => whole_write -> bool;
    DynamicBanks => dynamic_banks -> [bool; 2];
    ReadsInvocationId => reads_invocation_id -> bool;
    ReadsInvocationInfo => reads_invocation_info -> bool;
    StreamMarksOf => stream_marks -> StreamMarks;
    UsedOutputs => used_outputs -> u32;
}
struct DeclareAttributes<'a, A>(&'a mut A);
impl<A: AttributeIo> Visit for DeclareAttributes<'_, A> {
    type Out = Result<(), A::Error>;
    fn visit<K: Kind>(self, kind: &K) -> Result<(), A::Error> {
        kind.declare_attributes(self.0)
    }
}
struct Require<'a, M>(&'a mut M);
impl<M: Requirements> Visit for Require<'_, M> {
    type Out = ();
    fn visit<K: Kind>(self, kind: &K) {
        kind.require(self.0)
    }
}
struct TranslationOf<T>(T);
impl<T: Translation> Visit for TranslationOf<T> {
    type Out = T::Out;
    fn visit<K: Kind>(self, kind: &K) -> T::Out {
        kind.translation(self.0)
    }
}
struct HandleUse(bool);
impl Visit for HandleUse {
    type Out = bool;
    fn visit<K: Kind>(self, kind: &K) -> bool {
        kind.handle_use(self.0)
    }
}
struct ProveAddress<'a>(&'a mut ProvenAddresses);
impl Visit for ProveAddress<'_> {
    type Out = ();
    fn visit<K: Kind>(self, kind: &K) {
        kind.prove_address(self.0)
    }
}
struct AtProvenAddress<'a>(&'a ProvenAddresses);
impl Visit for AtProvenAddress<'_> {
    type Out = InstructionKind;
    fn visit<K: Kind>(self, kind: &K) -> InstructionKind {
        kind.at_proven_address(self.0)
    }
}
struct BoundDetached<'a, B>(&'a mut B);
impl<B: DetachedBounds> Visit for BoundDetached<'_, B> {
    type Out = ();
    fn visit<K: Kind>(self, kind: &K) {
        kind.bound_detached(self.0)
    }
}
struct MarkBlockStarts<'a, E>(&'a mut E, &'a [Instruction]);
impl<E: Extend<u32>> Visit for MarkBlockStarts<'_, E> {
    type Out = ();
    fn visit<K: Kind>(self, kind: &K) {
        kind.mark_block_starts(self.0, self.1)
    }
}
struct WritesRegister(u8);
impl Visit for WritesRegister {
    type Out = bool;
    fn visit<K: Kind>(self, kind: &K) -> bool {
        kind.writes_register(self.0)
    }
}
struct LegacyForm;
impl Visit for LegacyForm {
    type Out = InstructionKind;
    fn visit<K: Kind>(self, kind: &K) -> InstructionKind {
        kind.legacy_form()
    }
}

/// A pass's behavior for any instruction type.
pub trait Visit {
    type Out;
    fn visit<K: Kind>(self, kind: &K) -> Self::Out;
}

/// A pass's changing behavior for any instruction type.
pub trait VisitMut {
    type Out;
    fn visit<K: Kind>(self, kind: &mut K) -> Self::Out;
}

/// An instruction kind prints as its own instruction type.
struct DebugKind<'a, 'b>(&'a mut fmt::Formatter<'b>);

impl Visit for DebugKind<'_, '_> {
    type Out = fmt::Result;
    fn visit<K: Kind>(self, kind: &K) -> fmt::Result {
        fmt::Debug::fmt(kind, self.0)
    }
}

impl fmt::Debug for InstructionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.visit(DebugKind(f))
    }
}
impl Kind for ops::Nop {
    fn may_write_predicate(&self) -> bool {
        false
    }
}

impl Kind for ops::VoteVtg {
    fn may_write_predicate(&self) -> bool {
        // No predicate output: Ryujinx Decoders/InstTable.cs:320.
        let ops::VoteVtg {
            mode: _,
            immediate: _,
        } = *self;
        false
    }
}

impl Kind for ops::Vote {
    fn guards_whole(&self) -> bool {
        true
    }

    fn require<M: Requirements>(&self, module: &mut M) {
        module.subgroup();
        module.vote();
        // The ballot word of this lane's 32-lane warp (Ryujinx
        // Instructions/InstEmitWarp.cs:110-143).
        module.lane_id();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::Vote { .. } = *self;
        true
    }

    fn written_predicates(&self) -> PredicateSet {
        let ops::Vote { vote, .. } = *self;
        PredicateSet::one(vote)
    }

    fn written_registers(&self) -> Written {
        let ops::Vote { ballot, .. } = *self;
        Written::new(ballot, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::Vote { .. } = *self;
        Sources::new(
            [
                Source::Immediate(0),
                Source::Immediate(0),
                Source::Immediate(0),
            ],
            0,
        )
    }
}

impl Kind for ops::Shuffle {
    fn guards_whole(&self) -> bool {
        true
    }

    fn require<M: Requirements>(&self, module: &mut M) {
        module.subgroup();
        module.shuffle();
        module.lane_id();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::Shuffle { .. } = *self;
        true
    }

    fn written_predicates(&self) -> PredicateSet {
        let ops::Shuffle { in_bounds, .. } = *self;
        PredicateSet::one(in_bounds)
    }

    fn written_registers(&self) -> Written {
        let ops::Shuffle { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::Shuffle { lane, control, .. } = *self;
        Sources::new([lane, control, Source::Immediate(0)], 2)
    }
}

impl Kind for ops::SwizzleAdd {
    fn require<M: Requirements>(&self, module: &mut M) {
        module.subgroup();
        module.lane_id();
        module.preserve_float32();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::SwizzleAdd {
            dst: _,
            src: _,
            mask: _,
            ftz: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::SwizzleAdd { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::SwizzleAdd { src, mask, .. } = *self;
        Sources::new(
            [
                Source::Register(src[0]),
                Source::Register(src[1]),
                Source::Immediate(u32::from(mask)),
            ],
            3,
        )
    }
}

impl Kind for ops::Kill {
    fn require<M: Requirements>(&self, module: &mut M) {
        module.demote();
    }

    fn may_write_predicate(&self) -> bool {
        false
    }
}

/// The S2R indices naming compute builtins (Ryujinx IoMap.cs:20,44):
/// workgroup and thread IDs, with no geometry or tessellation equivalent.
const COMPUTE_BUILTIN: [bool; 256] = {
    let mut table = [false; 256];
    table[0x20] = true;
    table[0x21] = true;
    table[0x22] = true;
    table[0x23] = true;
    table[0x25] = true;
    table[0x26] = true;
    table[0x27] = true;
    table
};

impl Kind for ops::S2R {
    fn declare_attributes<A: AttributeIo>(&self, io: &mut A) -> Result<(), A::Error> {
        let paths: [Declare<A>; 2] = [|_| Ok(()), A::reject];
        paths[usize::from(COMPUTE_BUILTIN[usize::from(self.idx)])](io)
    }

    fn reads_invocation_id(&self) -> bool {
        self.idx == 0x11
    }

    fn reads_invocation_info(&self) -> bool {
        self.idx == 0x1d
    }

    fn require<M: Requirements>(&self, module: &mut M) {
        let ops::S2R { idx, .. } = *self;
        system_register(idx, module);
    }

    fn may_write_predicate(&self) -> bool {
        let ops::S2R { dst: _, idx: _ } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::S2R { dst, .. } = *self;
        Written::new(dst, 1)
    }
}

impl Kind for ops::IpaPass {
    fn require<M: Requirements>(&self, module: &mut M) {
        module.frag_coord();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::IpaPass {
            dst: _,
            addr: _,
            inv_w: _,
            offset: _,
            loc: _,
            freq: _,
            saturate: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::IpaPass { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::IpaPass { inv_w, offset, .. } = *self;
        Sources::new([inv_w, offset, Source::Immediate(0)], 2)
    }
}

impl Kind for ops::IpaPassMulW {
    fn require<M: Requirements>(&self, module: &mut M) {
        module.frag_coord();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::IpaPassMulW {
            dst: _,
            addr: _,
            inv_w: _,
            offset: _,
            loc: _,
            freq: _,
            saturate: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::IpaPassMulW { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::IpaPassMulW { inv_w, offset, .. } = *self;
        Sources::new([inv_w, offset, Source::Immediate(0)], 2)
    }
}

impl Kind for ops::IpaSaturate {
    fn require<M: Requirements>(&self, module: &mut M) {
        module.frag_coord();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::IpaSaturate {
            dst: _,
            addr: _,
            inv_w: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::IpaSaturate { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::IpaSaturate { inv_w, .. } = *self;
        Sources::new([inv_w, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::Mov {
    fn whole_write(&self) -> bool {
        self.quad_lanes == 0xf
    }

    fn require<M: Requirements>(&self, module: &mut M) {
        let ops::Mov { quad_lanes, .. } = *self;
        [nothing, M::lane_id][QUAD_LANE_ID[usize::from(quad_lanes)]](module);
    }

    fn may_write_predicate(&self) -> bool {
        let ops::Mov {
            dst: _,
            src: _,
            quad_lanes: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::Mov { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::Mov { src, .. } = *self;
        Sources::new([src, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::Ldc {
    fn require<M: Requirements>(&self, module: &mut M) {
        // By LdcAddressMode: IS and ISL can select any of the 32 banks.
        let ops::Ldc { address_mode, .. } = *self;
        [nothing, nothing, M::ldc_indexed, M::ldc_indexed][address_mode as usize](module);
    }

    fn may_write_predicate(&self) -> bool {
        let ops::Ldc {
            dst: _,
            binding: _,
            offset: _,
            immediate: _,
            mem_type: _,
            address_mode: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        // Mesa sm50.rs:2635-2648 and :2863-2876 encode destination plus
        // memory width or attribute component count.
        let ops::Ldc { dst, mem_type, .. } = *self;
        Written::new(dst, mem_type.registers())
    }
}

impl Kind for ops::IAdd {
    fn may_write_predicate(&self) -> bool {
        let ops::IAdd {
            dst: _,
            lhs: _,
            rhs: _,
            x: _,
            cc: _,
            negate: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::IAdd { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::IAdd { rhs, .. } = *self;
        Sources::new([rhs, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::IAdd3 {
    fn may_write_predicate(&self) -> bool {
        let ops::IAdd3 {
            dst: _,
            lhs: _,
            rhs: _,
            addend: _,
            halves: _,
            shift: _,
            negate: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::IAdd3 { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::IAdd3 { rhs, .. } = *self;
        Sources::new([rhs, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::Lea {
    fn may_write_predicate(&self) -> bool {
        let ops::Lea { .. } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::Lea { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::Lea { rhs, .. } = *self;
        Sources::new([rhs, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::LeaHi {
    fn may_write_predicate(&self) -> bool {
        let ops::LeaHi {
            dst: _,
            low: _,
            addend: _,
            high: _,
            shift: _,
            negate: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::LeaHi { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::LeaHi { addend, .. } = *self;
        Sources::new([addend, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::FunnelShift {
    fn may_write_predicate(&self) -> bool {
        let ops::FunnelShift {
            dst: _,
            low: _,
            high: _,
            shift: _,
            right: _,
            dst_high: _,
            ty: _,
            wrap: _,
            cc: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::FunnelShift { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::FunnelShift { shift, .. } = *self;
        Sources::new([shift, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::BitFieldInsert {
    fn handle_use(&self, _include_global: bool) -> bool {
        self.range.reads_register()
    }

    fn may_write_predicate(&self) -> bool {
        let ops::BitFieldInsert { .. } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::BitFieldInsert { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::BitFieldInsert { base, range, .. } = *self;
        Sources::new([base, range, Source::Immediate(0)], 2)
    }
}

impl Kind for ops::BitFieldExtract {
    fn may_write_predicate(&self) -> bool {
        let ops::BitFieldExtract {
            dst: _,
            base: _,
            range: _,
            signed: _,
            reverse: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::BitFieldExtract { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::BitFieldExtract { range, .. } = *self;
        Sources::new([range, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::IntegerMul {
    fn may_write_predicate(&self) -> bool {
        let ops::IntegerMul {
            dst: _,
            lhs: _,
            rhs: _,
            addend: _,
            high: _,
            signed: _,
            negate: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::IntegerMul { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::IntegerMul { rhs, addend, .. } = *self;
        Sources::new([rhs, addend, Source::Immediate(0)], 2)
    }
}

impl Kind for ops::IntegerMulAdd {
    fn may_write_predicate(&self) -> bool {
        let ops::IntegerMulAdd {
            dst: _,
            lhs: _,
            rhs: _,
            addend: _,
            high: _,
            signed: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::IntegerMulAdd { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn legacy_form(self) -> InstructionKind {
        let ops::IntegerMulAdd {
            dst,
            lhs,
            rhs,
            addend,
            high,
            signed,
        } = self;
        InstructionKind::IntegerMul(ops::IntegerMul {
            dst,
            lhs,
            rhs,
            addend: Source::Register(addend),
            high,
            signed,
            negate: MulNegation::None,
        })
    }

    fn source_slots(&self) -> Sources {
        let ops::IntegerMulAdd { rhs, .. } = *self;
        Sources::new([rhs, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::FAdd {
    fn require<M: Requirements>(&self, module: &mut M) {
        module.preserve_float32();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::FAdd {
            dst: _,
            lhs: _,
            rhs: _,
            modifiers: _,
            round: _,
            flags: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::FAdd { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::FAdd { rhs, .. } = *self;
        Sources::new([rhs, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::FMul {
    fn require<M: Requirements>(&self, module: &mut M) {
        module.preserve_float32();
    }

    fn product_rounding(&self) -> FMulRounding {
        self.modifiers.rounding()
    }

    fn may_write_predicate(&self) -> bool {
        let ops::FMul {
            dst: _,
            lhs: _,
            rhs: _,
            modifiers: _,
            flags: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::FMul { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::FMul { rhs, .. } = *self;
        Sources::new([rhs, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::FCmp {
    fn require<M: Requirements>(&self, module: &mut M) {
        module.preserve_float32();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::FCmp {
            dst: _,
            when_true: _,
            when_false: _,
            test: _,
            compare: _,
            ftz: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::FCmp { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::FCmp {
            when_false, test, ..
        } = *self;
        Sources::new([when_false, test, Source::Immediate(0)], 2)
    }
}

impl Kind for ops::ICmp {
    fn may_write_predicate(&self) -> bool {
        let ops::ICmp { .. } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::ICmp { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::ICmp {
            when_false, test, ..
        } = *self;
        Sources::new([when_false, test, Source::Immediate(0)], 2)
    }
}

impl Kind for ops::HalfArithmetic {
    fn require<M: Requirements>(&self, module: &mut M) {
        module.preserve_float32();
        module.half();
        module.half_uvec2();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::HalfArithmetic { .. } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::HalfArithmetic { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::HalfArithmetic { op, src, .. } = *self;
        Sources::new(src, 2 + u8::from(op == HalfOp::Fma))
    }
}

impl Kind for ops::HalfSet {
    fn guards_whole(&self) -> bool {
        self.may_write_predicate()
    }

    fn require<M: Requirements>(&self, module: &mut M) {
        let ops::HalfSet { ftz, .. } = *self;
        module.preserve_float32();
        module.half();
        [nothing, M::half_uvec2][usize::from(ftz)](module);
    }

    fn may_write_predicate(&self) -> bool {
        let ops::HalfSet { dst, .. } = *self;
        matches!(dst, HalfSetDestination::Predicates { .. })
    }

    fn written_registers(&self) -> Written {
        match *self {
            ops::HalfSet {
                dst: HalfSetDestination::Register { dst, .. },
                ..
            } => Written::new(dst, 1),
            _ => Written::NOTHING,
        }
    }

    fn source_slots(&self) -> Sources {
        let ops::HalfSet { src, .. } = *self;
        Sources::new([src[0], src[1], Source::Immediate(0)], 2)
    }
}

impl Kind for ops::VideoMultiplyAdd {
    fn may_write_predicate(&self) -> bool {
        let ops::VideoMultiplyAdd { .. } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::VideoMultiplyAdd { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::VideoMultiplyAdd { rhs, .. } = *self;
        Sources::new([rhs, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::VideoMinMax {
    fn may_write_predicate(&self) -> bool {
        let ops::VideoMinMax {
            dst: _,
            lhs: _,
            rhs: _,
            addend: _,
            select: _,
            max: _,
            signed_output: _,
            saturate: _,
            op: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::VideoMinMax { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::VideoMinMax { rhs, .. } = *self;
        Sources::new([rhs, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::FFma {
    fn require<M: Requirements>(&self, module: &mut M) {
        module.preserve_float32();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::FFma {
            dst: _,
            lhs: _,
            rhs: _,
            addend: _,
            flags: _,
            modifiers: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::FFma { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::FFma { rhs, addend, .. } = *self;
        Sources::new([rhs, addend, Source::Immediate(0)], 2)
    }
}

impl Kind for ops::FMnMx {
    fn may_write_predicate(&self) -> bool {
        let ops::FMnMx {
            dst: _,
            lhs: _,
            rhs: _,
            min: _,
            flags: _,
            abs: _,
            negate: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::FMnMx { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::FMnMx { rhs, .. } = *self;
        Sources::new([rhs, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::FMnMxSelect {
    fn may_write_predicate(&self) -> bool {
        let ops::FMnMxSelect {
            dst: _,
            lhs: _,
            rhs: _,
            min_when: _,
            flags: _,
            abs: _,
            negate: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::FMnMxSelect { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::FMnMxSelect { rhs, .. } = *self;
        Sources::new([rhs, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::MuFu {
    fn require<M: Requirements>(&self, module: &mut M) {
        let ops::MuFu { op, .. } = *self;
        [nothing, M::float64][MUFU_FLOAT64[op as usize]](module);
    }

    fn may_write_predicate(&self) -> bool {
        let ops::MuFu {
            dst: _,
            src: _,
            op: _,
            abs: _,
            negate: _,
            saturate: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::MuFu { dst, .. } = *self;
        Written::new(dst, 1)
    }
}

impl Kind for ops::Rro {
    fn may_write_predicate(&self) -> bool {
        let ops::Rro {
            dst: _,
            src: _,
            abs: _,
            negate: _,
            ex2: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::Rro { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::Rro { src, .. } = *self;
        Sources::new([src, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::IMinMax {
    fn may_write_predicate(&self) -> bool {
        let ops::IMinMax {
            dst: _,
            lhs: _,
            rhs: _,
            min: _,
            signed: _,
            cc: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::IMinMax { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::IMinMax { rhs, .. } = *self;
        Sources::new([rhs, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::IMinMaxSelect {
    fn may_write_predicate(&self) -> bool {
        let ops::IMinMaxSelect {
            dst: _,
            lhs: _,
            rhs: _,
            min_when: _,
            signed: _,
            cc: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::IMinMaxSelect { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::IMinMaxSelect { rhs, .. } = *self;
        Sources::new([rhs, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::Flo {
    fn may_write_predicate(&self) -> bool {
        let ops::Flo { .. } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::Flo { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::Flo { src, .. } = *self;
        Sources::new([src, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::Popc {
    fn may_write_predicate(&self) -> bool {
        let ops::Popc { .. } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::Popc { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::Popc { src, .. } = *self;
        Sources::new([src, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::ISetP {
    fn may_write_predicate(&self) -> bool {
        let ops::ISetP { .. } = *self;
        true
    }

    fn written_predicates(&self) -> PredicateSet {
        let ops::ISetP { dst, .. } = *self;
        PredicateSet::pair(dst)
    }

    fn source_slots(&self) -> Sources {
        let ops::ISetP { rhs, .. } = *self;
        Sources::new([rhs, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::ISet {
    fn may_write_predicate(&self) -> bool {
        let ops::ISet {
            dst: _,
            lhs: _,
            rhs: _,
            compare: _,
            signedness: _,
            bool_float: _,
            cc: _,
            x: _,
            accum: _,
            accum_negate: _,
            set_op: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::ISet { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::ISet { rhs, .. } = *self;
        Sources::new([rhs, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::Xmad {
    fn may_write_predicate(&self) -> bool {
        let ops::Xmad {
            dst: _,
            a: _,
            b: _,
            c: _,
            signed: _,
            high: _,
            psl: _,
            mrg: _,
            cmode: _,
            cc: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::Xmad { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::Xmad { b, c, .. } = *self;
        Sources::new([b, c, Source::Immediate(0)], 2)
    }
}

impl Kind for ops::ShiftAdd {
    fn may_write_predicate(&self) -> bool {
        let ops::ShiftAdd {
            dst: _,
            lhs: _,
            rhs: _,
            shift: _,
            negate: _,
            cc: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::ShiftAdd { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::ShiftAdd { rhs, .. } = *self;
        Sources::new([rhs, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::R2P {
    fn may_write_predicate(&self) -> bool {
        let ops::R2P { bank, .. } = *self;
        bank == FlagBank::Predicates
    }

    fn source_slots(&self) -> Sources {
        let ops::R2P { mask, .. } = *self;
        Sources::new([mask, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::P2R {
    fn may_write_predicate(&self) -> bool {
        let ops::P2R { .. } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::P2R { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::P2R { src, mask, .. } = *self;
        Sources::new([Source::Register(src), mask, Source::Immediate(0)], 2)
    }
}

impl Kind for ops::VideoSetP {
    fn guards_whole(&self) -> bool {
        true
    }

    fn may_write_predicate(&self) -> bool {
        let ops::VideoSetP { .. } = *self;
        true
    }

    fn written_predicates(&self) -> PredicateSet {
        let ops::VideoSetP { dst, .. } = *self;
        PredicateSet::pair(dst)
    }

    fn source_slots(&self) -> Sources {
        let ops::VideoSetP { lhs, rhs, .. } = *self;
        Sources::new([Source::Register(lhs), rhs, Source::Immediate(0)], 2)
    }
}

impl Kind for ops::FSetP {
    fn may_write_predicate(&self) -> bool {
        let ops::FSetP { .. } = *self;
        true
    }

    fn written_predicates(&self) -> PredicateSet {
        let ops::FSetP { dst, .. } = *self;
        PredicateSet::pair(dst)
    }

    fn source_slots(&self) -> Sources {
        let ops::FSetP { rhs, .. } = *self;
        Sources::new([rhs, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::DoubleSetP {
    fn require<M: Requirements>(&self, module: &mut M) {
        module.float64();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::DoubleSetP { .. } = *self;
        true
    }

    fn written_predicates(&self) -> PredicateSet {
        let ops::DoubleSetP { dst, .. } = *self;
        PredicateSet::pair(dst)
    }

    fn source_slots(&self) -> Sources {
        let ops::DoubleSetP { rhs, .. } = *self;
        rhs.slots()
    }
}

impl Kind for ops::DoubleSet {
    fn require<M: Requirements>(&self, module: &mut M) {
        module.float64();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::DoubleSet { .. } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::DoubleSet { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::DoubleSet { rhs, .. } = *self;
        rhs.slots()
    }
}

impl Kind for ops::DoubleAdd {
    fn require<M: Requirements>(&self, module: &mut M) {
        module.float64();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::DoubleAdd { .. } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::DoubleAdd { dst, .. } = *self;
        dst.written()
    }

    fn source_slots(&self) -> Sources {
        let ops::DoubleAdd { rhs, .. } = *self;
        rhs.slots()
    }
}

impl Kind for ops::DoubleMinMax {
    fn require<M: Requirements>(&self, module: &mut M) {
        module.float64();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::DoubleMinMax { .. } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::DoubleMinMax { dst, .. } = *self;
        dst.written()
    }

    fn source_slots(&self) -> Sources {
        let ops::DoubleMinMax { rhs, .. } = *self;
        rhs.slots()
    }
}

impl Kind for ops::DoubleMultiply {
    fn require<M: Requirements>(&self, module: &mut M) {
        module.float64();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::DoubleMultiply { .. } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::DoubleMultiply { dst, .. } = *self;
        dst.written()
    }

    fn source_slots(&self) -> Sources {
        let ops::DoubleMultiply { rhs, .. } = *self;
        rhs.slots()
    }
}

impl Kind for ops::DoubleFma {
    fn require<M: Requirements>(&self, module: &mut M) {
        module.float64();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::DoubleFma { .. } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::DoubleFma { dst, .. } = *self;
        dst.written()
    }

    fn source_slots(&self) -> Sources {
        let ops::DoubleFma {
            multiplier, addend, ..
        } = *self;
        multiplier.slots().then(addend.slots())
    }
}

impl Kind for ops::CSetP {
    fn may_write_predicate(&self) -> bool {
        let ops::CSetP { .. } = *self;
        true
    }

    fn written_predicates(&self) -> PredicateSet {
        let ops::CSetP { dst, .. } = *self;
        PredicateSet::pair(dst)
    }
}

impl Kind for ops::CSet {
    fn may_write_predicate(&self) -> bool {
        let ops::CSet {
            dst: _,
            condition: _,
            set_op: _,
            accum: _,
            result: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::CSet { dst, .. } = *self;
        Written::new(dst, 1)
    }
}

impl Kind for ops::PSet {
    fn may_write_predicate(&self) -> bool {
        let ops::PSet {
            dst: _,
            src: _,
            set_op: _,
            negate: _,
            bool_float: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::PSet { dst, .. } = *self;
        Written::new(dst, 1)
    }
}

impl Kind for ops::PSetP {
    fn guards_whole(&self) -> bool {
        true
    }

    fn may_write_predicate(&self) -> bool {
        let ops::PSetP { .. } = *self;
        true
    }

    fn written_predicates(&self) -> PredicateSet {
        let ops::PSetP { dst, .. } = *self;
        PredicateSet::pair(dst)
    }
}

impl Kind for ops::FSet {
    fn may_write_predicate(&self) -> bool {
        let ops::FSet {
            dst: _,
            lhs: _,
            rhs: _,
            compare: _,
            bool_float: _,
            abs: _,
            negate: _,
            cc: _,
            accum: _,
            set_op: _,
            ftz: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::FSet { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::FSet { rhs, .. } = *self;
        Sources::new([rhs, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::Sel {
    fn may_write_predicate(&self) -> bool {
        let ops::Sel {
            dst: _,
            when_true: _,
            when_false: _,
            condition: _,
            condition_negate: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::Sel { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::Sel { when_false, .. } = *self;
        Sources::new([when_false, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::Convert {
    fn require<M: Requirements>(&self, module: &mut M) {
        let ops::Convert { kind, .. } = *self;
        kind.require(module);
    }

    fn may_write_predicate(&self) -> bool {
        let ops::Convert {
            kind: _,
            dst: _,
            src: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        match *self {
            // Ryujinx InstEmitConversion.cs:410-415 writes the double pair.
            ops::Convert {
                dst,
                kind:
                    Conversion::I2F { dst_size: 3, .. }
                    | Conversion::F2F {
                        destination: F2fDestination::Float64,
                        ..
                    },
                ..
            } => Written::new(dst, 2),
            ops::Convert { dst, .. } => Written::new(dst, 1),
        }
    }

    fn source_slots(&self) -> Sources {
        match *self {
            // Mesa nak/ir.rs:4834-4838,394-397: 64-bit I2F consumes
            // consecutive GPR components. Include the high word in scans.
            ops::Convert {
                kind: Conversion::I2F { src_size: 3, .. },
                src: Source::Register(r),
                ..
            } if r.0 < 255 => Sources::new(
                [
                    Source::Register(r),
                    Source::Register(Register(r.0 + 1)),
                    Source::Immediate(0),
                ],
                2,
            ),
            ops::Convert { src, .. } => {
                Sources::new([src, Source::Immediate(0), Source::Immediate(0)], 1)
            }
        }
    }
}

impl Kind for ops::DoubleToInt {
    fn require<M: Requirements>(&self, module: &mut M) {
        module.float64();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::DoubleToInt { .. } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::DoubleToInt { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::DoubleToInt { src, .. } = *self;
        src.slots()
    }
}

impl Kind for ops::DoubleToFloat {
    fn require<M: Requirements>(&self, module: &mut M) {
        module.float64();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::DoubleToFloat { .. } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::DoubleToFloat { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::DoubleToFloat { src, .. } = *self;
        src.slots()
    }
}

impl Kind for ops::DoubleRound {
    fn require<M: Requirements>(&self, module: &mut M) {
        module.float64();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::DoubleRound { .. } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::DoubleRound { dst, .. } = *self;
        dst.written()
    }

    fn source_slots(&self) -> Sources {
        let ops::DoubleRound { src, .. } = *self;
        src.slots()
    }
}

impl Kind for ops::Lop {
    fn may_write_predicate(&self) -> bool {
        let ops::Lop {
            dst: _,
            lhs: _,
            rhs: _,
            op: _,
            invert: _,
            predicate_output,
            extended: _,
            cc: _,
        } = *self;
        predicate_output.is_some()
    }

    fn written_registers(&self) -> Written {
        let ops::Lop { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::Lop { rhs, .. } = *self;
        Sources::new([rhs, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::Lop3 {
    fn may_write_predicate(&self) -> bool {
        let ops::Lop3 {
            cc,
            predicate_output,
            ..
        } = *self;
        cc || predicate_output.is_some()
    }

    fn written_registers(&self) -> Written {
        let ops::Lop3 { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::Lop3 { rhs, .. } = *self;
        Sources::new([rhs, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::LopWithPredicate {
    fn may_write_predicate(&self) -> bool {
        let ops::LopWithPredicate { .. } = *self;
        true
    }

    fn written_registers(&self) -> Written {
        let ops::LopWithPredicate { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::LopWithPredicate { rhs, .. } = *self;
        Sources::new([rhs, Source::Immediate(0), Source::Immediate(0)], 1)
    }

    fn legacy_form(self) -> InstructionKind {
        let ops::LopWithPredicate {
            dst,
            lhs,
            rhs,
            op,
            invert,
            predicate_output,
            extended,
            cc,
        } = self;
        InstructionKind::Lop(ops::Lop {
            dst,
            lhs,
            rhs,
            op,
            invert,
            predicate_output: Some(predicate_output),
            extended,
            cc,
        })
    }
}

impl Kind for ops::Lop3WithPredicate {
    fn may_write_predicate(&self) -> bool {
        let ops::Lop3WithPredicate { .. } = *self;
        true
    }

    fn written_registers(&self) -> Written {
        let ops::Lop3WithPredicate { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::Lop3WithPredicate { rhs, .. } = *self;
        Sources::new([rhs, Source::Immediate(0), Source::Immediate(0)], 1)
    }

    fn legacy_form(self) -> InstructionKind {
        let ops::Lop3WithPredicate {
            dst,
            lhs,
            rhs,
            third,
            lut,
            predicate_output,
            extended,
            cc,
        } = self;
        InstructionKind::Lop3(ops::Lop3 {
            dst,
            lhs,
            rhs,
            third,
            lut,
            predicate_output: Some(predicate_output),
            extended,
            cc,
        })
    }
}

impl Kind for ops::Shl {
    fn may_write_predicate(&self) -> bool {
        let ops::Shl {
            dst: _,
            lhs: _,
            rhs: _,
            wrap: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::Shl { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::Shl { rhs, .. } = *self;
        Sources::new([rhs, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::Shr {
    fn may_write_predicate(&self) -> bool {
        let ops::Shr {
            dst: _,
            lhs: _,
            rhs: _,
            wrap: _,
            signed: _,
            reverse: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::Shr { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::Shr { rhs, .. } = *self;
        Sources::new([rhs, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::Prmt {
    fn may_write_predicate(&self) -> bool {
        let ops::Prmt {
            dst: _,
            lhs: _,
            rhs: _,
            selector: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::Prmt { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::Prmt { selector, .. } = *self;
        Sources::new([selector, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::IMadSp {
    fn may_write_predicate(&self) -> bool {
        let ops::IMadSp {
            dst: _,
            a: _,
            b: _,
            c: _,
            types: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::IMadSp { dst, .. } = *self;
        Written::new(dst, 1)
    }

    fn source_slots(&self) -> Sources {
        let ops::IMadSp { b, c, .. } = *self;
        Sources::new([b, c, Source::Immediate(0)], 2)
    }
}

impl Kind for ops::GlobalMemory {
    fn handle_use(&self, include_global: bool) -> bool {
        include_global
    }

    fn may_write_predicate(&self) -> bool {
        let ops::GlobalMemory { .. } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        match *self {
            ops::GlobalMemory {
                access: GlobalAccess::Load(size),
                data,
                ..
            } => Written::new(data, size.access().registers()),
            _ => Written::NOTHING,
        }
    }
}

impl Kind for ops::GlobalAtomic {
    fn handle_use(&self, include_global: bool) -> bool {
        include_global
    }

    fn may_write_predicate(&self) -> bool {
        let ops::GlobalAtomic { .. } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        match *self {
            ops::GlobalAtomic {
                dst: Some(dst),
                width,
                ..
            } => Written::new(dst, width.registers()),
            _ => Written::NOTHING,
        }
    }
}

impl Kind for ops::GlobalReduce {
    fn handle_use(&self, include_global: bool) -> bool {
        include_global
    }

    fn may_write_predicate(&self) -> bool {
        let ops::GlobalReduce { .. } = *self;
        false
    }

    fn legacy_form(self) -> InstructionKind {
        let ops::GlobalReduce {
            data,
            address,
            offset,
            extended,
            signed,
            op,
            width,
        } = self;
        InstructionKind::GlobalAtomic(ops::GlobalAtomic {
            dst: None,
            data,
            address,
            offset,
            extended,
            signed,
            op,
            width,
        })
    }
}

impl Kind for ops::LocalLoad {
    fn guards_whole(&self) -> bool {
        true
    }

    fn require<M: Requirements>(&self, module: &mut M) {
        module.local();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::LocalLoad {
            dst: _,
            address: _,
            offset: _,
            mem_type: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        // The access size names the registers written: B64 two, B128
        // four, otherwise one (Ryujinx InstEmitMemory.cs:325-332).
        let ops::LocalLoad { dst, mem_type, .. } = *self;
        Written::new(dst, mem_type.registers())
    }
}

impl Kind for ops::SharedLoad {
    fn guards_whole(&self) -> bool {
        true
    }

    fn require<M: Requirements>(&self, module: &mut M) {
        module.shared();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::SharedLoad {
            dst: _,
            address: _,
            offset: _,
            mem_type: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        // The access size names the registers written: B64 two, B128
        // four, otherwise one (Ryujinx InstEmitMemory.cs:325-332).
        let ops::SharedLoad { dst, mem_type, .. } = *self;
        Written::new(dst, mem_type.registers())
    }
}

impl Kind for ops::LocalStore {
    fn guards_whole(&self) -> bool {
        true
    }

    fn require<M: Requirements>(&self, module: &mut M) {
        module.local();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::LocalStore {
            data: _,
            address: _,
            offset: _,
            mem_type: _,
            cache: _,
        } = *self;
        false
    }
}

impl Kind for ops::SharedStore {
    fn guards_whole(&self) -> bool {
        true
    }

    fn require<M: Requirements>(&self, module: &mut M) {
        module.shared();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::SharedStore {
            data: _,
            address: _,
            offset: _,
            mem_type: _,
        } = *self;
        false
    }
}

impl Kind for ops::Atomic {
    fn guards_whole(&self) -> bool {
        true
    }

    fn require<M: Requirements>(&self, module: &mut M) {
        module.shared();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::Atomic { .. } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::Atomic { dst, .. } = *self;
        Written::new(dst, 1)
    }
}

impl Kind for ops::SharedCompareExchange {
    fn guards_whole(&self) -> bool {
        true
    }

    fn require<M: Requirements>(&self, module: &mut M) {
        module.shared();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::SharedCompareExchange { .. } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::SharedCompareExchange { dst, .. } = *self;
        Written::new(dst, 1)
    }
}

impl Kind for ops::GroupMemoryBarrier {
    fn may_write_predicate(&self) -> bool {
        false
    }
}

impl Kind for ops::MemoryBarrier {
    fn may_write_predicate(&self) -> bool {
        false
    }
}

impl Kind for ops::ControlBarrier {
    fn may_write_predicate(&self) -> bool {
        false
    }
}

impl Kind for ops::GeometryOut {
    fn stream_marks(&self) -> StreamMarks {
        let ops::GeometryOut { stream, emit, cut } = *self;
        let stream = stream & 7;
        StreamMarks {
            outs: true,
            unencodable: (self.stream > 3) | !(emit | cut),
            streams: 1 << stream,
            emitted: u8::from(emit) << stream,
        }
    }

    fn may_write_predicate(&self) -> bool {
        let ops::GeometryOut { .. } = *self;
        false
    }
}

impl Kind for ops::AttributeAddress {
    fn prove_address(&self, proven: &mut ProvenAddresses) {
        // Only an AL2P without a register offset gives a known address, and
        // only a word-aligned generic one names a generic slot.
        let ops::AttributeAddress {
            dst,
            offset,
            address,
        } = *self;
        let generic =
            (offset.0 == 255) & (0x80..0x280).contains(&address) & address.is_multiple_of(4);
        proven.prove(dst, [0, address][usize::from(generic)]);
    }

    fn may_write_predicate(&self) -> bool {
        let ops::AttributeAddress { .. } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::AttributeAddress { dst, .. } = *self;
        Written::new(dst, 1)
    }
}

impl Kind for ops::IsbeRead {
    fn may_write_predicate(&self) -> bool {
        let ops::IsbeRead { .. } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::IsbeRead { dst, .. } = *self;
        Written::new(dst, 1)
    }
}

impl Kind for ops::AttributeLoad {
    fn declare_attributes<A: AttributeIo>(&self, io: &mut A) -> Result<(), A::Error> {
        // A primitive vertex for the fixed attributes from 0x2f0 up has no
        // proven per-vertex meaning in this interface.
        let paths: [DeclareFrom<Self, A>; 2] = [
            |x, io| io.lanes(x.dst, x.address, x.lanes, false),
            |_, io| io.reject(),
        ];
        let primitive = self.vertex.snapshot(|_| 0).iter().count();
        paths[primitive & usize::from(self.address >= 0x2f0)](self, io)
    }

    fn guards_whole(&self) -> bool {
        true
    }

    fn require<M: Requirements>(&self, module: &mut M) {
        let ops::AttributeLoad { address, lanes, .. } = *self;
        lanes
            .iter(address)
            .for_each(|(_, lane)| lane.attribute.require(module));
    }

    fn may_write_predicate(&self) -> bool {
        let ops::AttributeLoad {
            vertex: _,
            dst: _,
            address: _,
            components: _,
            lanes: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::AttributeLoad {
            dst, components, ..
        } = *self;
        Written::new(dst, components)
    }
}

impl Kind for ops::OutputAttributeLoad {
    fn used_outputs(&self) -> u32 {
        generic_outputs(self.address, self.components)
    }

    fn declare_attributes<A: AttributeIo>(&self, io: &mut A) -> Result<(), A::Error> {
        io.lanes(self.dst, self.address, self.lanes, false)
    }

    fn guards_whole(&self) -> bool {
        true
    }

    fn require<M: Requirements>(&self, module: &mut M) {
        let ops::OutputAttributeLoad { vertex, .. } = *self;
        vertex.require(module);
    }

    fn may_write_predicate(&self) -> bool {
        let ops::OutputAttributeLoad {
            dst: _,
            vertex: _,
            address: _,
            components: _,
            lanes: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::OutputAttributeLoad {
            dst, components, ..
        } = *self;
        Written::new(dst, components)
    }
}

impl Kind for ops::PatchAttributeLoad {
    fn declare_attributes<A: AttributeIo>(&self, io: &mut A) -> Result<(), A::Error> {
        io.lanes(self.dst, self.address, self.lanes, true)
    }

    fn guards_whole(&self) -> bool {
        true
    }

    fn may_write_predicate(&self) -> bool {
        let ops::PatchAttributeLoad {
            dst: _,
            address: _,
            components: _,
            lanes: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::PatchAttributeLoad {
            dst, components, ..
        } = *self;
        Written::new(dst, components)
    }
}

impl Kind for ops::PhysicalAttributeLoad {
    // Only scalar physical generic addressing has a proven component
    // result (InstEmitAttribute.cs:43-51).
    fn declare_attributes<A: AttributeIo>(&self, io: &mut A) -> Result<(), A::Error> {
        io.physical(false)
    }

    fn dynamic_banks(&self) -> [bool; 2] {
        [true, false]
    }

    fn at_proven_address(&self, proven: &ProvenAddresses) -> InstructionKind {
        const AT: [fn(ops::PhysicalAttributeLoad, u16) -> InstructionKind; 2] = [
            |x, _| x.into(),
            |ops::PhysicalAttributeLoad { dst, vertex, .. }, address| {
                ops::AttributeLoad {
                    dst,
                    vertex,
                    address,
                    components: 1,
                    lanes: AttributeLanes::vertex(address, 1, AttributeBank::Input),
                }
                .into()
            },
        ];
        let address = proven.at(self.offset);
        AT[usize::from(address != 0)](*self, address)
    }

    fn guards_whole(&self) -> bool {
        true
    }

    fn require<M: Requirements>(&self, module: &mut M) {
        module.dynamic_input();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::PhysicalAttributeLoad {
            dst: _,
            vertex: _,
            offset: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::PhysicalAttributeLoad { dst, .. } = *self;
        Written::new(dst, 1)
    }
}

impl Kind for ops::PhysicalOutputAttributeLoad {
    fn declare_attributes<A: AttributeIo>(&self, io: &mut A) -> Result<(), A::Error> {
        io.physical(true)
    }

    fn dynamic_banks(&self) -> [bool; 2] {
        [false, true]
    }

    fn used_outputs(&self) -> u32 {
        u32::MAX
    }

    fn at_proven_address(&self, proven: &ProvenAddresses) -> InstructionKind {
        const AT: [fn(ops::PhysicalOutputAttributeLoad, u16) -> InstructionKind; 2] = [
            |x, _| x.into(),
            |ops::PhysicalOutputAttributeLoad { dst, vertex, .. }, address| {
                ops::OutputAttributeLoad {
                    dst,
                    vertex,
                    address,
                    components: 1,
                    lanes: AttributeLanes::vertex(address, 1, AttributeBank::Output),
                }
                .into()
            },
        ];
        let address = proven.at(self.offset);
        AT[usize::from(address != 0)](*self, address)
    }

    fn guards_whole(&self) -> bool {
        true
    }

    fn require<M: Requirements>(&self, module: &mut M) {
        let ops::PhysicalOutputAttributeLoad { vertex, .. } = *self;
        vertex.require(module);
    }

    fn may_write_predicate(&self) -> bool {
        let ops::PhysicalOutputAttributeLoad {
            dst: _,
            vertex: _,
            offset: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::PhysicalOutputAttributeLoad { dst, .. } = *self;
        Written::new(dst, 1)
    }
}

impl Kind for ops::UnmappedAttributeLoad {
    fn may_write_predicate(&self) -> bool {
        let ops::UnmappedAttributeLoad {
            dst: _,
            address: _,
            components: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::UnmappedAttributeLoad {
            dst, components, ..
        } = *self;
        Written::new(dst, components)
    }
}

impl Kind for ops::IndexedPatchLoad {
    fn declare_attributes<A: AttributeIo>(&self, io: &mut A) -> Result<(), A::Error> {
        io.patch_bank(self.bank)
    }

    fn guards_whole(&self) -> bool {
        true
    }

    fn may_write_predicate(&self) -> bool {
        let ops::IndexedPatchLoad {
            dst: _,
            base: _,
            index: _,
            components: _,
            bank: _,
            lanes: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::IndexedPatchLoad {
            dst, components, ..
        } = *self;
        Written::new(dst, components)
    }
}

impl Kind for ops::AttributeStore {
    fn declare_attributes<A: AttributeIo>(&self, io: &mut A) -> Result<(), A::Error> {
        // By offset: static words, or physical generic addressing, where
        // only a scalar generic store has a proven component result.
        // Static indirect addressing is TODO in Ryujinx
        // InstEmitAttribute.cs:131 and stays unsupported.
        let shapes: [DeclareFrom<Self, A>; 2] = [
            |x, io| {
                (0..x.components).try_for_each(|c| io.store(x.address + u16::from(c) * 4, x.patch))
            },
            |x, io| {
                let paths: [Declare<A>; 2] = [|io| io.physical(true), A::reject];
                paths[usize::from(x.patch | (x.address != 0) | (x.components != 1))](io)
            },
        ];
        let paths: [DeclareFrom<Self, A>; 2] =
            [shapes[self.offset.iter().count()], |_, io| io.reject()];
        // Unsupported component ranges have no interface.
        let unsupported = (self.components.wrapping_sub(1) > 3) | !self.address.is_multiple_of(4);
        paths[usize::from(unsupported)](self, io)
    }

    fn dynamic_banks(&self) -> [bool; 2] {
        [false, self.offset.iter().count() == 1]
    }

    fn used_outputs(&self) -> u32 {
        // By patch, then by offset: a patch store touches no generic output.
        let generic =
            [generic_outputs(self.address, self.components), u32::MAX][self.offset.iter().count()];
        [generic, 0][usize::from(self.patch)]
    }

    fn at_proven_address(&self, proven: &ProvenAddresses) -> InstructionKind {
        // Only a scalar physical store to a generic slot has a proven
        // static form (InstEmitAttribute.cs:118-127).
        const AT: [fn(ops::AttributeStore, u16) -> InstructionKind; 2] = [
            |x, _| x.into(),
            |ops::AttributeStore { data, .. }, address| {
                ops::AttributeStore {
                    data,
                    address,
                    offset: None,
                    components: 1,
                    patch: false,
                }
                .into()
            },
        ];
        match *self {
            ops::AttributeStore {
                address: 0,
                offset: Some(offset),
                components: 1,
                patch: false,
                ..
            } => {
                let address = proven.at(offset);
                AT[usize::from(address != 0)](*self, address)
            }
            x => x.into(),
        }
    }

    fn guards_whole(&self) -> bool {
        true
    }

    fn may_write_predicate(&self) -> bool {
        let ops::AttributeStore {
            data: _,
            address: _,
            offset: _,
            components: _,
            patch: _,
        } = *self;
        false
    }
}

impl Kind for ops::PhysicalAttributeStore {
    fn declare_attributes<A: AttributeIo>(&self, io: &mut A) -> Result<(), A::Error> {
        // Only a scalar generic store has a proven component result.
        let shape: [Declare<A>; 2] = [|io| io.physical(true), A::reject];
        // Unsupported component ranges have no interface.
        let unsupported = (self.components.wrapping_sub(1) > 3)
            | !self.address.is_multiple_of(4)
            | self.patch
            | (self.address != 0)
            | (self.components != 1);
        shape[usize::from(unsupported)](io)
    }

    fn dynamic_banks(&self) -> [bool; 2] {
        [false, true]
    }

    fn used_outputs(&self) -> u32 {
        // A patch store touches no generic output.
        [u32::MAX, 0][usize::from(self.patch)]
    }

    fn at_proven_address(&self, proven: &ProvenAddresses) -> InstructionKind {
        // Only a scalar physical store to a generic slot has a proven
        // static form (InstEmitAttribute.cs:118-127).
        const AT: [fn(ops::PhysicalAttributeStore, u16) -> InstructionKind; 2] = [
            |x, _| x.into(),
            |ops::PhysicalAttributeStore { data, .. }, address| {
                ops::AttributeStore {
                    data,
                    address,
                    offset: None,
                    components: 1,
                    patch: false,
                }
                .into()
            },
        ];
        let scalar = (self.address == 0) & (self.components == 1) & !self.patch;
        let address = proven.at(self.offset);
        AT[usize::from(scalar & (address != 0))](*self, address)
    }

    fn legacy_form(self) -> InstructionKind {
        let ops::PhysicalAttributeStore {
            data,
            address,
            offset,
            components,
            patch,
        } = self;
        InstructionKind::AttributeStore(ops::AttributeStore {
            data,
            address,
            offset: Some(offset),
            components,
            patch,
        })
    }

    fn guards_whole(&self) -> bool {
        true
    }

    fn may_write_predicate(&self) -> bool {
        let ops::PhysicalAttributeStore {
            data: _,
            address: _,
            offset: _,
            components: _,
            patch: _,
        } = *self;
        false
    }
}

impl Kind for ops::Texture {
    fn handle_use(&self, _include_global: bool) -> bool {
        true
    }

    fn require<M: Requirements>(&self, module: &mut M) {
        let ops::Texture {
            op,
            dim,
            offset,
            nodep,
            ..
        } = *self;
        module.images();
        texture_dim(dim, module);
        [nothing, M::texture_offsets][usize::from(offset)](module);
        [nothing, M::nodep][usize::from(nodep)](module);
        op.require(module);
    }

    fn may_write_predicate(&self) -> bool {
        // SM50 texture ops never write a fault predicate
        // (mesa/src/nouveau/compiler/nak/sm50.rs:2189,2222,2270,2334).
        let ops::Texture {
            op: _,
            dst: _,
            src0: _,
            src1: _,
            dim: _,
            mask: _,
            offset: _,
            scalar: _,
            packed_f16: _,
            lod_clamp: _,
            multisample: _,
            depth_compare: _,
            nodep: _,
            scalar_dst: _,
            bound_handle: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        let ops::Texture {
            dst,
            mask,
            scalar,
            packed_f16,
            ..
        } = *self;
        Written::new(
            dst,
            [
                [mask.count_ones() as u8, (mask.count_ones() as u8).min(2)][usize::from(scalar)],
                1,
            ][usize::from(packed_f16)],
        )
    }

    fn destinations(&self) -> [Written; 2] {
        // F16 packs each pair into one word; its adjacent register stays live.
        let ops::Texture {
            mask,
            scalar,
            scalar_dst,
            packed_f16,
            ..
        } = *self;
        let rest = [(mask.count_ones() as u8).saturating_sub(2), 1][usize::from(packed_f16)];
        let second = scalar_dst
            .into_iter()
            .fold(Written::NOTHING, |_, dst| Written::new(dst, rest));
        [
            self.written_registers(),
            [Written::NOTHING, second][usize::from(scalar)],
        ]
    }
}

impl Kind for ops::SurfaceAtomic {
    fn handle_use(&self, _include_global: bool) -> bool {
        true
    }

    fn guards_whole(&self) -> bool {
        true
    }

    fn require<M: Requirements>(&self, module: &mut M) {
        let ops::SurfaceAtomic { target, signed, .. } = *self;
        module.images();
        surface_target(target, module);
        [nothing, M::signed_storage][usize::from(signed)](module);
    }

    fn may_write_predicate(&self) -> bool {
        let ops::SurfaceAtomic { .. } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        match *self {
            ops::SurfaceAtomic { dst: Some(dst), .. } => Written::new(dst, 1),
            _ => Written::NOTHING,
        }
    }

    fn source_slots(&self) -> Sources {
        let ops::SurfaceAtomic { handle, .. } = *self;
        Sources::new([handle, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::SurfaceReduce {
    fn handle_use(&self, _include_global: bool) -> bool {
        true
    }

    fn guards_whole(&self) -> bool {
        true
    }

    fn require<M: Requirements>(&self, module: &mut M) {
        let ops::SurfaceReduce { target, signed, .. } = *self;
        module.images();
        surface_target(target, module);
        [nothing, M::signed_storage][usize::from(signed)](module);
    }

    fn may_write_predicate(&self) -> bool {
        let ops::SurfaceReduce { .. } = *self;
        false
    }

    fn legacy_form(self) -> InstructionKind {
        let ops::SurfaceReduce {
            op,
            signed,
            target,
            handle,
            coord,
            data,
            byte_address,
        } = self;
        InstructionKind::SurfaceAtomic(ops::SurfaceAtomic {
            op,
            signed,
            target,
            handle,
            coord,
            data,
            dst: None,
            byte_address,
        })
    }

    fn source_slots(&self) -> Sources {
        let ops::SurfaceReduce { handle, .. } = *self;
        Sources::new([handle, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::Surface {
    fn handle_use(&self, _include_global: bool) -> bool {
        true
    }

    fn guards_whole(&self) -> bool {
        true
    }

    fn require<M: Requirements>(&self, module: &mut M) {
        let ops::Surface {
            store,
            target,
            format,
            size,
            cache,
            ..
        } = *self;
        [M::surface_load, M::surface_store][usize::from(store)](module);
        module.images();
        surface_target(target, module);
        [|_: u8, _: &mut M| (), raw_surface::<M>][usize::from(format == 1)](size, module);
        [nothing, M::nontemporal][usize::from(store && cache == 2)](module);
    }

    fn may_write_predicate(&self) -> bool {
        let ops::Surface {
            store: _,
            atomic: _,
            target: _,
            format: _,
            mask: _,
            handle: _,
            coord: _,
            data: _,
            clamp: _,
            size: _,
            byte_address: _,
            cache: _,
        } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        match *self {
            ops::Surface {
                store: false,
                atomic: false,
                data,
                mask,
                ..
            } => Written::new(data, mask.count_ones() as u8),
            _ => Written::NOTHING,
        }
    }

    fn source_slots(&self) -> Sources {
        let ops::Surface { handle, .. } = *self;
        Sources::new([handle, Source::Immediate(0), Source::Immediate(0)], 1)
    }
}

impl Kind for ops::Branch {
    fn bound_detached<B: DetachedBounds>(&self, bounds: &mut B) {
        bounds.end_entry();
        bounds.start(self.target.slot());
        bounds.fall_through();
        bounds.end_at(self.target);
    }

    fn writes_predicate_register(&self) -> bool {
        false
    }

    fn mark_block_starts<E: Extend<u32>>(&self, starts: &mut E, next: &[Instruction]) {
        starts.extend([self.target.slot()]);
        starts.extend(next.first().map(|n| n.address));
    }

    fn may_write_predicate(&self) -> bool {
        let ops::Branch { .. } = *self;
        true
    }
}

impl Kind for ops::Reconverge {
    fn bound_detached<B: DetachedBounds>(&self, bounds: &mut B) {
        bounds.end_entry();
        bounds.start(self.target.slot());
        bounds.fall_through();
        bounds.end_at(self.target);
    }

    fn writes_predicate_register(&self) -> bool {
        false
    }

    fn mark_block_starts<E: Extend<u32>>(&self, starts: &mut E, next: &[Instruction]) {
        starts.extend([self.target.slot()]);
        starts.extend(next.first().map(|n| n.address));
    }

    fn may_write_predicate(&self) -> bool {
        let ops::Reconverge { .. } = *self;
        true
    }
}

impl Kind for ops::BrxRegister {
    fn bound_detached<B: DetachedBounds>(&self, bounds: &mut B) {
        bounds.end_entry();
    }

    fn writes_predicate_register(&self) -> bool {
        false
    }

    fn require<M: Requirements>(&self, module: &mut M) {
        module.register_brx();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::BrxRegister { .. } = *self;
        true
    }

    fn source_slots(&self) -> Sources {
        let ops::BrxRegister { base, .. } = *self;
        Sources::new(
            [
                Source::Immediate(base.offset()),
                Source::Immediate(0),
                Source::Immediate(0),
            ],
            1,
        )
    }
}

impl Kind for ops::Brx {
    fn bound_detached<B: DetachedBounds>(&self, bounds: &mut B) {
        bounds.end_entry();
    }

    fn writes_predicate_register(&self) -> bool {
        false
    }

    fn mark_block_starts<E: Extend<u32>>(&self, starts: &mut E, next: &[Instruction]) {
        starts.extend(self.table.targets().iter().map(|t| t.slot()));
        starts.extend(next.first().map(|n| n.address));
    }

    fn may_write_predicate(&self) -> bool {
        let ops::Brx { .. } = *self;
        true
    }
}

impl Kind for ops::OutsideBrx {
    fn bound_detached<B: DetachedBounds>(&self, bounds: &mut B) {
        bounds.end_entry();
    }

    fn writes_predicate_register(&self) -> bool {
        false
    }

    fn may_write_predicate(&self) -> bool {
        let ops::OutsideBrx { .. } = *self;
        true
    }
}

impl Kind for ops::Schedule {
    fn translation<T: Translation>(&self, pass: T) -> T::Out {
        pass.unresolved_stack()
    }

    fn writes_predicate_register(&self) -> bool {
        false
    }

    fn resolved(self) -> InstructionKind {
        ops::ResolvedSchedule {
            push: ResolvedPush::checked(self.target),
        }
        .into()
    }

    fn may_write_predicate(&self) -> bool {
        let ops::Schedule { .. } = *self;
        false
    }
}

impl Kind for ops::PushTarget {
    fn translation<T: Translation>(&self, pass: T) -> T::Out {
        pass.unresolved_stack()
    }

    fn bound_detached<B: DetachedBounds>(&self, bounds: &mut B) {
        bounds.end_entry();
        bounds.end_at(self.target);
    }

    fn writes_predicate_register(&self) -> bool {
        false
    }

    fn resolved(self) -> InstructionKind {
        ops::ResolvedPushTarget {
            push: ResolvedPush::checked(self.target),
            stack: self.stack,
        }
        .into()
    }

    fn may_write_predicate(&self) -> bool {
        let ops::PushTarget { .. } = *self;
        true
    }
}

impl Kind for ops::CbufPushTarget {
    fn writes_predicate_register(&self) -> bool {
        false
    }

    fn may_write_predicate(&self) -> bool {
        let ops::CbufPushTarget { .. } = *self;
        false
    }
}
impl Kind for ops::ResolvedSchedule {
    fn writes_predicate_register(&self) -> bool {
        false
    }

    fn may_write_predicate(&self) -> bool {
        let ops::ResolvedSchedule { .. } = *self;
        false
    }
}

impl Kind for ops::ResolvedPushTarget {
    fn bound_detached<B: DetachedBounds>(&self, bounds: &mut B) {
        bounds.end_entry();
        bounds.end_at(self.push.target());
    }

    fn writes_predicate_register(&self) -> bool {
        false
    }

    fn may_write_predicate(&self) -> bool {
        let ops::ResolvedPushTarget { .. } = *self;
        true
    }
}

impl Kind for ops::OutsideBranch {
    fn bound_detached<B: DetachedBounds>(&self, bounds: &mut B) {
        bounds.end_entry();
        bounds.fall_through();
    }

    fn writes_predicate_register(&self) -> bool {
        false
    }

    fn may_write_predicate(&self) -> bool {
        let ops::OutsideBranch { .. } = *self;
        true
    }
}

impl Kind for ops::OutsideSchedule {
    fn translation<T: Translation>(&self, pass: T) -> T::Out {
        pass.unresolved_stack()
    }

    fn writes_predicate_register(&self) -> bool {
        false
    }

    fn may_write_predicate(&self) -> bool {
        let ops::OutsideSchedule { .. } = *self;
        false
    }
}

impl Kind for ops::OutsidePushTarget {
    fn translation<T: Translation>(&self, pass: T) -> T::Out {
        pass.unresolved_stack()
    }

    fn bound_detached<B: DetachedBounds>(&self, bounds: &mut B) {
        bounds.end_entry();
    }

    fn writes_predicate_register(&self) -> bool {
        false
    }

    fn may_write_predicate(&self) -> bool {
        let ops::OutsidePushTarget { .. } = *self;
        true
    }
}

impl Kind for ops::PopJump {
    fn translation<T: Translation>(&self, pass: T) -> T::Out {
        pass.unresolved_stack()
    }

    fn bound_detached<B: DetachedBounds>(&self, bounds: &mut B) {
        bounds.end_entry();
    }

    fn writes_predicate_register(&self) -> bool {
        false
    }

    fn may_write_predicate(&self) -> bool {
        let ops::PopJump { .. } = *self;
        true
    }
}

impl Kind for ops::Exit {
    fn bound_detached<B: DetachedBounds>(&self, bounds: &mut B) {
        bounds.end_entry();
    }

    fn writes_predicate_register(&self) -> bool {
        false
    }

    fn mark_block_starts<E: Extend<u32>>(&self, starts: &mut E, next: &[Instruction]) {
        starts.extend(next.first().map(|n| n.address));
    }

    fn may_write_predicate(&self) -> bool {
        let ops::Exit { .. } = *self;
        true
    }
}

impl Kind for ops::Unknown {
    fn translation<T: Translation>(&self, pass: T) -> T::Out {
        pass.unknown()
    }

    fn bound_detached<B: DetachedBounds>(&self, bounds: &mut B) {
        bounds.end_entry();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::Unknown { .. } = *self;
        true
    }
}

impl Kind for ops::Padding {
    fn translation<T: Translation>(&self, pass: T) -> T::Out {
        pass.padding()
    }

    fn bound_detached<B: DetachedBounds>(&self, bounds: &mut B) {
        bounds.end_entry();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::Padding = *self;
        true
    }
}

impl Kind for ops::Call {
    fn translation<T: Translation>(&self, pass: T) -> T::Out {
        pass.unbuilt()
    }

    fn bound_detached<B: DetachedBounds>(&self, bounds: &mut B) {
        bounds.end_entry();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::Call { .. } = *self;
        true
    }
}

impl Kind for ops::OutsideCall {
    fn translation<T: Translation>(&self, pass: T) -> T::Out {
        pass.unbuilt()
    }

    fn bound_detached<B: DetachedBounds>(&self, bounds: &mut B) {
        bounds.end_entry();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::OutsideCall { .. } = *self;
        true
    }
}

impl Kind for ops::OverlongInsert {
    fn translation<T: Translation>(&self, pass: T) -> T::Out {
        pass.unwanted()
    }

    fn may_write_predicate(&self) -> bool {
        let ops::OverlongInsert { .. } = *self;
        false
    }

    fn written_registers(&self) -> Written {
        self.insert.written_registers()
    }

    fn source_slots(&self) -> Sources {
        self.insert.source_slots()
    }
}

impl Kind for ops::MisalignedBranch {
    fn bound_detached<B: DetachedBounds>(&self, bounds: &mut B) {
        bounds.end_entry();
        bounds.fall_through();
    }

    fn writes_predicate_register(&self) -> bool {
        false
    }

    fn may_write_predicate(&self) -> bool {
        let ops::MisalignedBranch { .. } = *self;
        true
    }
}

impl Kind for ops::MisalignedSchedule {
    fn translation<T: Translation>(&self, pass: T) -> T::Out {
        pass.unresolved_stack()
    }

    fn writes_predicate_register(&self) -> bool {
        false
    }

    fn may_write_predicate(&self) -> bool {
        let ops::MisalignedSchedule { .. } = *self;
        false
    }
}

impl Kind for ops::MisalignedPushTarget {
    fn translation<T: Translation>(&self, pass: T) -> T::Out {
        pass.unresolved_stack()
    }

    fn bound_detached<B: DetachedBounds>(&self, bounds: &mut B) {
        bounds.end_entry();
    }

    fn writes_predicate_register(&self) -> bool {
        false
    }

    fn may_write_predicate(&self) -> bool {
        let ops::MisalignedPushTarget { .. } = *self;
        true
    }
}

impl Kind for ops::MisalignedCall {
    fn translation<T: Translation>(&self, pass: T) -> T::Out {
        pass.unbuilt()
    }

    fn bound_detached<B: DetachedBounds>(&self, bounds: &mut B) {
        bounds.end_entry();
    }

    fn may_write_predicate(&self) -> bool {
        let ops::MisalignedCall { .. } = *self;
        true
    }
}

impl Kind for ops::GeometryOutDynamic {
    fn translation<T: Translation>(&self, pass: T) -> T::Out {
        pass.unproven_stream()
    }

    fn may_write_predicate(&self) -> bool {
        let ops::GeometryOutDynamic { .. } = *self;
        false
    }
}

impl ops::GeometryOutDynamic {
    /// This OUT on the stream a proof found its source to hold.
    pub(crate) fn on(self, stream: u8) -> ops::GeometryOut {
        let ops::GeometryOutDynamic { emit, cut, .. } = self;
        ops::GeometryOut { emit, cut, stream }
    }
}

/// Each instruction is its own type.
pub mod ops {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Nop;

    /// Operator-authorized reference matching: Ryujinx InstEmit.cs:330-336
    /// emits nothing. Hardware semantics are unproven. Encoding fields:
    /// envytools envydis/gm107.c:2053,1633-1637; the 28-bit immediate is
    /// Ryujinx Decoders/InstDecoders.cs:5231.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct VoteVtg {
        pub mode: VoteVtgMode,
        pub immediate: U28,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Vote {
        pub op: VoteOp,
        pub ballot: Register,
        pub vote: u8,
        pub pred: Predicate,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Shuffle {
        pub dst: Register,
        pub src: Register,
        pub lane: Source,
        pub mode: ShuffleMode,
        pub control: Source,
        pub in_bounds: u8,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct SwizzleAdd {
        pub dst: Register,
        pub src: [Register; 2],
        pub mask: u8,
        pub ftz: bool,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Kill;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct S2R {
        pub dst: Register,
        pub idx: u8,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct IpaPass {
        pub dst: Register,
        pub addr: u16,
        pub inv_w: Source,
        pub offset: Source,
        pub loc: InterpLoc,
        pub freq: InterpFreq,
        /// SAT51: envytools gm107.c:1953; Ryujinx InstEmitAttribute.cs:215
        /// clamps the final value to [0, 1].
        pub saturate: bool,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct IpaPassMulW {
        pub dst: Register,
        pub addr: u16,
        pub inv_w: Source,
        pub offset: Source,
        pub loc: InterpLoc,
        pub freq: InterpFreq,
        /// SAT51: envytools gm107.c:1953; Ryujinx InstEmitAttribute.cs:215
        /// clamps the final value to [0, 1].
        pub saturate: bool,
    }

    /// Bounded immediate IPA multiply with saturation. Encoding: Mesa
    /// nak/sm50.rs:2905-2934; semantics: Ryujinx InstEmitAttribute.cs:167-175,208-217.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct IpaSaturate {
        pub dst: Register,
        pub addr: u16,
        pub inv_w: Source,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Mov {
        /// Quad position enables, Mesa sm50.rs:1936,1941,1946.
        pub quad_lanes: u8,
        pub dst: Register,
        pub src: Source,
    }

    /// LDC (sm50.rs OpLdc): load from constant buffer `binding` at byte
    /// address register + offset. The decoder establishes the concrete width.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Ldc {
        pub dst: Register,
        pub binding: u8,
        pub offset: Register,
        pub immediate: i32,
        pub mem_type: LdcWidth,
        pub address_mode: LdcAddressMode,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct IAdd {
        pub dst: Register,
        pub lhs: Register,
        pub rhs: Source,
        pub cc: bool,
        pub x: bool,
        pub negate: [bool; 2],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct IAdd3 {
        pub dst: Register,
        pub lhs: Register,
        pub rhs: Source,
        pub addend: Register,
        pub halves: [HalfSelect; 3],
        pub shift: PartialSumShift,
        pub negate: [bool; 3],
    }

    /// LEA low (envytools gm107.c:2022,2093,2140), without carry/CC outputs.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Lea {
        pub dst: Register,
        pub lhs: Register,
        pub rhs: Source,
        pub shift: u8,
        pub negate: bool,
    }

    /// LEA.HI (envytools gm107.c:2021,2163), documented arithmetic only.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct LeaHi {
        pub dst: Register,
        pub low: Register,
        pub addend: Source,
        pub high: Register,
        pub shift: u8,
        pub negate: bool,
    }

    /// Funnel shift (Mesa sm50.rs:1653-1691).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct FunnelShift {
        pub dst: Register,
        pub low: Register,
        pub high: Register,
        pub shift: Source,
        pub right: bool,
        pub dst_high: bool,
        pub ty: FunnelType,
        pub wrap: bool,
        pub cc: bool,
    }

    /// BFI, Ryujinx InstEmitBitfield.cs:75-83,154-162.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct BitFieldInsert {
        pub dst: Register,
        pub insert: Register,
        pub base: Source,
        pub range: Source,
    }

    /// BFE (Mesa sm50.rs:1166-1195).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct BitFieldExtract {
        pub dst: Register,
        pub base: Register,
        pub range: Source,
        pub signed: bool,
        pub reverse: bool,
    }

    /// IMUL/IMAD (Ryujinx InstEmitIntegerArithmetic.cs:84-178,409-457):
    /// the low or high word of lhs * rhs plus the addend. IMUL's addend is
    /// the constant zero, as Ryujinx passes it; IMAD.RC's is a cbuf word.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct IntegerMul {
        pub dst: Register,
        pub lhs: Register,
        pub rhs: Source,
        pub addend: Source,
        pub high: bool,
        pub signed: [bool; 2],
        pub negate: MulNegation,
    }

    /// IMAD: the 32-bit product plus a register addend (Mesa
    /// sm50.rs:1347-1403; envytools gm107.c:2033,2084,2130).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct IntegerMulAdd {
        pub dst: Register,
        pub lhs: Register,
        pub rhs: Source,
        pub addend: Register,
        pub high: bool,
        pub signed: [bool; 2],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct FAdd {
        pub dst: Register,
        pub lhs: Register,
        pub rhs: Source,
        pub modifiers: FAddModifiers,
        pub round: RoundingMode,
        pub flags: FloatFlags,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct FMul {
        pub dst: Register,
        pub lhs: Register,
        pub rhs: Source,
        pub modifiers: FMulModifiers,
        pub flags: FloatFlags,
    }

    /// FCMP selects raw A/B bits by comparing C against +0.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct FCmp {
        pub dst: Register,
        pub when_true: Register,
        pub when_false: Source,
        pub test: Source,
        pub compare: FloatCompare,
        /// FTZ47: Mesa nv50_ir_emit_gm107.cpp:1569,474-477.
        pub ftz: bool,
    }

    /// ICMP selects raw A/B bits by comparing C against integer zero.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct ICmp {
        pub dst: Register,
        pub when_true: Register,
        pub when_false: Source,
        pub test: Source,
        pub compare: IntCompare,
        pub signedness: Signedness,
    }

    /// Half arithmetic, with explicit sources for register/cbuf/immediate forms.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct HalfArithmetic {
        pub op: HalfOp,
        pub flush: HalfFlush,
        pub dst: Register,
        pub src: [Source; 3],
        pub swizzle: [u8; 3],
        pub output: u8,
        pub abs: [bool; 3],
        pub negate: [bool; 3],
        pub saturate: bool,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct HalfSet {
        pub ftz: bool,
        pub dst: HalfSetDestination,
        pub src: [Source; 2],
        pub swizzle: [u8; 2],
        pub abs: [bool; 2],
        pub negate: [bool; 2],
        pub compare: FloatCompare,
        pub accum: u8,
        pub accum_negate: bool,
        pub set_op: SetOp,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct VideoMultiplyAdd {
        pub dst: Register,
        pub lhs: Register,
        pub rhs: Source,
        pub addend: Register,
        pub select: [VideoSelection; 2],
        pub mode: u8,
        pub shift: u8,
        pub saturate: bool,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct VideoMinMax {
        pub dst: Register,
        pub lhs: Register,
        pub rhs: Source,
        pub addend: Register,
        pub select: [u8; 2],
        pub max: bool,
        pub signed_output: bool,
        pub saturate: bool,
        pub op: u8,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct FFma {
        pub dst: Register,
        pub lhs: Register,
        pub rhs: Source,
        pub addend: Source,
        pub flags: FloatFlags,
        pub modifiers: FFmaModifiers,
    }

    /// FMNMX: IEEE minNum/maxNum (a NaN operand yields the other one),
    /// with per-source absolute value then negation.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct FMnMx {
        pub dst: Register,
        pub lhs: Register,
        pub rhs: Source,
        pub min: bool,
        pub flags: FloatFlags,
        pub abs: [bool; 2],
        pub negate: [bool; 2],
    }

    /// FMNMX whose selector39..42 names a runtime predicate: minNum when it
    /// holds, otherwise maxNum (Ryujinx Instructions/InstEmitFloatMinMax.cs:
    /// 78-104; Mesa nak/sm50.rs:605-635 encodes the predicate source).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct FMnMxSelect {
        pub dst: Register,
        pub lhs: Register,
        pub rhs: Source,
        pub min_when: Predicate,
        pub flags: FloatFlags,
        pub abs: [bool; 2],
        pub negate: [bool; 2],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct MuFu {
        pub dst: Register,
        pub src: Register,
        pub op: MufuOp,
        pub abs: bool,
        pub negate: bool,
        pub saturate: bool,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Rro {
        pub dst: Register,
        pub src: Source,
        pub abs: bool,
        pub negate: bool,
        /// Mode bit 39: EX2 when set, SINCOS when clear (envytools
        /// envydis/gm107.c:1320-1323). Kept for audit; both modes lower
        /// the same way (Ryujinx Instructions/InstEmitMultifunction.cs:86-94).
        pub ex2: bool,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct IMinMax {
        pub dst: Register,
        pub lhs: Register,
        pub rhs: Source,
        pub min: bool,
        pub signed: bool,
        /// .CC 47: zero and sign flags of the result (Ryujinx
        /// InstEmitIntegerMinMax.cs:66).
        pub cc: bool,
    }

    /// IMNMX whose selector39..42 names a runtime predicate: the minimum
    /// when it holds, otherwise the maximum (Ryujinx
    /// Instructions/InstEmitIntegerMinMax.cs:45-73; Mesa nak/ir.rs:4093-4105).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct IMinMaxSelect {
        pub dst: Register,
        pub lhs: Register,
        pub rhs: Source,
        pub min_when: Predicate,
        pub signed: bool,
        /// As [`IMinMax::cc`].
        pub cc: bool,
    }

    /// Find the most significant differing bit, or the unsigned shift count.
    /// Encoding: Mesa nak/sm50.rs:1197-1225; semantics: Ryujinx
    /// Instructions/InstEmitBitfield.cs:165-182.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Flo {
        pub dst: Register,
        pub src: Source,
        pub signed: bool,
        pub invert: bool,
        pub shift: bool,
    }

    /// Bit population count of B, optionally complemented before counting.
    /// Ryujinx Instructions/InstEmitBitfield.cs:185-192.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Popc {
        pub dst: Register,
        pub src: Source,
        pub invert: bool,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct ISetP {
        /// The combined comparison, then its combined complement; 7 is PT.
        pub dst: [u8; 2],
        pub lhs: Register,
        pub rhs: Source,
        pub compare: IntCompare,
        pub signedness: Signedness,
        pub mode: IntegerCompareMode,
        pub set_op: SetOp,
        /// Accumulator predicate combined by set_op; 7 is PT.
        pub accum: u8,
        pub accum_negate: bool,
        /// Float sources only: absolute value, then negation, per source.
        pub abs: [bool; 2],
        pub negate: [bool; 2],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct ISet {
        pub dst: Register,
        pub lhs: Register,
        pub rhs: Source,
        pub compare: IntCompare,
        pub signedness: Signedness,
        pub bool_float: bool,
        pub cc: bool,
        pub x: bool,
        pub accum: u8,
        pub accum_negate: bool,
        pub set_op: SetOp,
    }

    /// XMAD (nouveau nv50_ir.h:284-305): 16x16 multiply of the selected
    /// halves of a and b (high half when `high`, sign-extended when
    /// `signed`), plus c adjusted by cmode (0 c, 1 low half of c, 2 high
    /// half of c, 3 c - ((a' >> 31) << 16) - ((b' >> 31) << 16) on the
    /// widened halves a' and b', 4 c + (b << 16)); PSL shifts the product
    /// left by 16 first, MRG replaces the result's high half with b's low half.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Xmad {
        pub dst: Register,
        pub a: Register,
        pub b: Source,
        pub c: Source,
        pub signed: [bool; 2],
        pub high: [bool; 2],
        pub psl: bool,
        pub mrg: bool,
        pub cmode: XmadCop,
        /// .CC 47: carry, overflow, zero and sign of the sum before MRG
        /// (Ryujinx InstEmitIntegerArithmetic.cs:261-274,665-699).
        pub cc: bool,
    }

    /// ISCADD / IADD with negation: (lhs << shift) + rhs, each source
    /// optionally negated (two's complement), modulo 2^32.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct ShiftAdd {
        pub dst: Register,
        pub lhs: Register,
        pub rhs: Source,
        pub shift: u8,
        pub negate: [bool; 2],
        pub cc: bool,
    }

    /// R2P: bit i of the selected byte of src goes to Pi where bit i of
    /// the mask is set (envytools gm107.c:1991,2065,2111; P0..P6).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct R2P {
        pub src: Register,
        pub mask: Source,
        pub byte: u8,
        pub bank: FlagBank,
    }

    /// P2R register merge: envytools gm107.c:1992,2066,2112,1194-1204;
    /// Ryujinx InstEmitPredicate.cs:86-114. Bit 40 selects Z/S/C/V or P0..P6.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct P2R {
        pub dst: Register,
        pub src: Register,
        pub mask: Source,
        pub byte: u8,
        pub cc: bool,
    }

    /// VSETP: envytools gm107.c:2052,1588-1628,1538-1542,1105-1109;
    /// Ryujinx InstEmitVideoMinMax.cs:122-180. Selectors use the video
    /// extraction convention in InstEmitAluHelper.cs:75-91.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct VideoSetP {
        pub dst: [u8; 2],
        pub lhs: Register,
        pub rhs: Source,
        pub select: [u8; 2],
        pub compare: IntCompare,
        pub signedness: Signedness,
        pub set_op: SetOp,
        pub accum: Predicate,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct FSetP {
        /// The combined comparison, then its combined complement; 7 is PT.
        pub dst: [u8; 2],
        pub lhs: Register,
        pub rhs: Source,
        pub compare: FloatCompare,
        pub set_op: SetOp,
        /// Accumulator predicate combined by set_op; 7 is PT.
        pub accum: u8,
        pub accum_negate: bool,
        /// Float sources only: absolute value, then negation, per source.
        pub abs: [bool; 2],
        pub negate: [bool; 2],
    }

    /// DSETP compares two binary64 register/cbuf/immediate pairs and writes
    /// the comparison and its complement to separate predicate destinations.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct DoubleSetP {
        pub dst: [u8; 2],
        pub lhs: RegisterPair,
        pub rhs: DoubleSource,
        pub compare: FloatCompare,
        pub set_op: SetOp,
        pub accum: Predicate,
        pub abs: [bool; 2],
        pub negate: [bool; 2],
    }

    /// DSET compares two binary64 pairs and writes a concrete 32-bit Boolean
    /// representation to one GPR.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct DoubleSet {
        pub dst: Register,
        pub lhs: RegisterPair,
        pub rhs: DoubleSource,
        pub compare: FloatCompare,
        pub set_op: SetOp,
        pub accum: Predicate,
        pub abs: [bool; 2],
        pub negate: [bool; 2],
        pub result: DoubleSetResult,
    }

    /// DADD consumes a binary64 register pair and one binary64 source and
    /// writes one binary64 pair.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct DoubleAdd {
        pub dst: RegisterPair,
        pub lhs: RegisterPair,
        pub rhs: DoubleSource,
        pub modifiers: [DoubleSourceModifier; 2],
        pub round: RoundingMode,
        pub flags: FloatFlags,
    }

    /// DMNMX consumes a binary64 register pair and one binary64 source and
    /// writes one binary64 pair after applying each source's decoded modifier.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct DoubleMinMax {
        pub dst: RegisterPair,
        pub lhs: RegisterPair,
        pub rhs: DoubleSource,
        pub operation: DoubleMinMaxOperation,
        pub modifiers: [DoubleSourceModifier; 2],
        pub flags: FloatFlags,
    }

    /// DMUL consumes a binary64 register pair and one binary64 source and
    /// writes one binary64 pair.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct DoubleMultiply {
        pub dst: RegisterPair,
        pub lhs: RegisterPair,
        pub rhs: DoubleSource,
        pub sign: DoubleProductSign,
        pub flags: FloatFlags,
    }

    /// Fused binary64 multiply-add. Product and addend signs are established
    /// independently at decode and the operation rounds only once.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct DoubleFma {
        pub dst: RegisterPair,
        pub lhs: RegisterPair,
        pub multiplier: DoubleSource,
        pub addend: DoubleSource,
        pub product_sign: DoubleProductSign,
        pub addend_sign: DoubleAddendSign,
        pub flags: FloatFlags,
    }

    /// CSETP: Ryujinx InstEmitConditionCode.cs:35-47.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct CSetP {
        pub dst: [u8; 2],
        pub condition: CSetPCondition,
        pub set_op: SetOp,
        pub accum: Predicate,
    }

    /// CSET: Ryujinx InstEmitConditionCode.cs:12-33. The condition and
    /// accumulator combine once; true writes the mask or 1.0f.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct CSet {
        pub dst: Register,
        pub condition: CSetPCondition,
        pub set_op: SetOp,
        pub accum: Predicate,
        pub result: SetResult,
    }

    /// Ryujinx InstEmitPredicate.cs:12-32: two Boolean ops, then GPR bits.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct PSet {
        pub dst: Register,
        pub src: [u8; 3],
        pub set_op: [SetOp; 2],
        pub negate: [bool; 3],
        pub bool_float: bool,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct PSetP {
        pub dst: [u8; 2],
        pub src: [u8; 3],
        pub set_op: [SetOp; 2],
        pub negate: [bool; 3],
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct FSet {
        pub dst: Register,
        pub lhs: Register,
        pub rhs: Source,
        pub compare: FloatCompare,
        /// Bit 52: write 1.0f for true instead of 0xffffffff.
        pub bool_float: bool,
        pub cc: bool,
        pub abs: [bool; 2],
        pub negate: [bool; 2],
        /// FSET predicate accumulator39..42 and Boolean45..46:
        /// envytools envydis/gm107.c:2036,2106,2154,1430-1434.
        pub accum: Predicate,
        pub set_op: SetOp,
        /// Flush comparison inputs: Mesa sm50.rs:861, ir.rs:2910,8949.
        pub ftz: bool,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Sel {
        pub dst: Register,
        pub when_true: Register,
        pub when_false: Source,
        pub condition: u8,
        /// Bit 42: select on the negated predicate.
        pub condition_negate: bool,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Convert {
        pub kind: Conversion,
        pub dst: Register,
        pub src: Source,
    }

    /// F2I.F64 consumes one binary64 source and produces one signed or
    /// unsigned integer with a concrete width.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct DoubleToInt {
        pub dst: Register,
        pub src: DoubleSource,
        /// abs49/neg45, applied to the binary64 source before rounding.
        pub modifier: DoubleSourceModifier,
        pub signedness: Signedness,
        pub width: IntegerWidth,
        pub round: RoundingMode,
    }

    /// F2F.F32.F64: one binary64 source, abs/neg applied, narrowed to
    /// binary32 in the encoded rounding, then saturated. Ryujinx
    /// InstEmitConversion.cs:121-162 (FPAbsNeg, FP64ConvertToFP32,
    /// FPSaturate in the destination type).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct DoubleToFloat {
        pub dst: Register,
        pub src: DoubleSource,
        pub modifier: DoubleSourceModifier,
        pub round: FMulRounding,
        pub saturate: Saturation,
    }

    /// F2F.F64.F64: one binary64 source, abs/neg applied, then the
    /// identity or an integral rounding, then saturated, written to a
    /// pair. Ryujinx InstEmitConversion.cs:121-162.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct DoubleRound {
        pub dst: RegisterPair,
        pub src: DoubleSource,
        pub modifier: DoubleSourceModifier,
        pub integer_round: Option<RoundingMode>,
        pub saturate: Saturation,
    }

    /// LOP/LOP32I: invert applies a bitwise not to each source first.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Lop {
        pub dst: Register,
        pub lhs: Register,
        pub rhs: Source,
        pub op: LogicOp,
        pub invert: [bool; 2],
        /// envytools gm107.c:1357-1362,2009,2083,2129.
        pub predicate_output: Option<PredicateOutput>,
        /// .X (bit 43; bit 57 for LOP32I): with CC, Z also requires the
        /// entry Z. envytools gm107.c:2009,2083,2129,2168; Ryujinx
        /// InstEmitIntegerLogical.cs:87-115, InstEmitAluHelper.cs:95-121.
        pub extended: bool,
        /// .CC: write the zero and sign condition flags (bit 47; bit 52
        /// for LOP32I).
        pub cc: bool,
    }

    /// LOP/LOP32I that also writes a predicate (envytools
    /// gm107.c:1357-1362,2009,2083,2129; Ryujinx
    /// InstEmitIntegerLogical.cs:145-160).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct LopWithPredicate {
        pub dst: Register,
        pub lhs: Register,
        pub rhs: Source,
        pub op: LogicOp,
        pub invert: [bool; 2],
        pub predicate_output: PredicateOutput,
        /// As [`Lop::extended`].
        pub extended: bool,
        /// As [`Lop::cc`].
        pub cc: bool,
    }

    /// LOP3 that also writes a predicate (gm107.c:1391-1396,2020;
    /// Ryujinx InstEmitIntegerLogical.cs:141-163).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Lop3WithPredicate {
        pub dst: Register,
        pub lhs: Register,
        pub rhs: Source,
        pub third: Register,
        pub lut: u8,
        pub predicate_output: PredicateOutput,
        pub extended: bool,
        pub cc: bool,
    }

    /// LOP3 A/register, B/signed immediate, C/register, LUT48..55.
    /// envytools gm107.c:253,2108; Ryujinx InstDecoders.cs:3756-3764.
    /// LOP3: A register, B by form, C register, and three-input truth table.
    /// gm107.c:2108,2169; Ryujinx InstEmitIntegerLogical.cs:65-84.
    /// LUT index is (A << 2) | (B << 1) | C (Lop3Expression.cs:13-22).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Lop3 {
        pub dst: Register,
        pub lhs: Register,
        pub rhs: Source,
        pub third: Register,
        pub lut: u8,
        /// gm107.c:1391-1396,2020: mode36..38 and destination48..51.
        pub predicate_output: Option<PredicateOutput>,
        /// gm107.c:2020 X38; Ryujinx InstEmitAluHelper.cs:95-121.
        pub extended: bool,
        pub cc: bool,
    }

    /// SHL: wrap (bit 39) masks the shift to 0..31, otherwise shifts of
    /// 32 or more give 0 (Mesa OpShl fold, ir.rs:4603).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Shl {
        pub dst: Register,
        pub lhs: Register,
        pub rhs: Source,
        pub wrap: bool,
    }

    /// SHR: wrap as SHL; signed (bit 48) shifts in the sign bit and gives
    /// x >> 31 for shifts of 32 or more (Mesa OpShr fold, ir.rs:4656).
    /// reverse (.BREV, bit 40) reverses the bits of lhs before the shift
    /// (Ryujinx InstEmitShift.cs:211-214).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Shr {
        pub dst: Register,
        pub lhs: Register,
        pub rhs: Source,
        pub wrap: bool,
        pub signed: bool,
        pub reverse: bool,
    }

    /// PRMT in index mode (Mesa OpPrmt fold, ir.rs:5133-5150).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Prmt {
        pub dst: Register,
        pub lhs: Register,
        pub rhs: Register,
        pub selector: Source,
    }

    /// IMADSP: low word of ext(a) * ext(b) + ext(c), each operand
    /// extracted by its type (Mesa nak/ir.rs OpIMadSp fold).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct IMadSp {
        pub dst: Register,
        pub a: Register,
        pub b: Source,
        pub c: Source,
        pub types: [IMadSpType; 3],
    }

    /// Global LDG/STG: Mesa sm50.rs:2595-2614,2652-2669. Sizes are
    /// sm50.rs:2386-2397; E is sm50.rs:2402-2410. UB128=7 is
    /// envytools gm107.c:650, Ryujinx InstEmitMemory.cs:502-509.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct GlobalMemory {
        pub access: GlobalAccess,
        pub data: Register,
        pub address: Register,
        pub offset: i32,
        pub extended: bool,
        /// Cache 46..48: Mesa sm50.rs:2413-2431. Preserved in module debug text.
        pub cache: u8,
    }

    /// Integer global atomics: Mesa sm50.rs:2713-2789; envytools
    /// gm107.c:667-694,718-737,1897,1903,1905. Packed CAS compares
    /// data and replaces with data+1 (Mesa ir.rs:2691, sm50.rs:2697-2708).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct GlobalAtomic {
        pub dst: Option<Register>,
        pub data: Register,
        pub address: Register,
        pub offset: i32,
        pub extended: bool,
        pub signed: bool,
        /// 0 add, 1 min, 2 max, 5 and, 6 or, 7 xor, 8 exchange, 15 CAS.
        pub op: u8,
        /// Memory and register width. B64 uses the even register pairs
        /// dst:dst+1 and data:data+1, low word first.
        pub width: AtomicWidth,
    }

    /// RED: a global atomic whose old value no register receives
    /// (envytools gm107.c:718-737,1905; Mesa sm50.rs:2713-2789).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct GlobalReduce {
        pub data: Register,
        pub address: Register,
        pub offset: i32,
        pub extended: bool,
        pub signed: bool,
        /// As [`GlobalAtomic::op`].
        pub op: u8,
        pub width: AtomicWidth,
    }

    /// LDL: per-thread local memory, Ryujinx
    /// Instructions/InstEmitMemory.cs:110-118 (StorageKind.LocalMemory).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct LocalLoad {
        pub dst: Register,
        pub address: Register,
        pub offset: i32,
        pub mem_type: LsSize2,
    }

    /// LDS: per-workgroup shared memory, Ryujinx
    /// Instructions/InstEmitMemory.cs:123-131 (StorageKind.SharedMemory).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct SharedLoad {
        pub dst: Register,
        pub address: Register,
        pub offset: i32,
        pub mem_type: LsSize2,
    }

    /// STL: Ryujinx Instructions/InstEmitMemory.cs:146-154.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct LocalStore {
        pub data: Register,
        pub address: Register,
        pub offset: i32,
        pub mem_type: LsSize2,
        pub cache: LocalStoreCache,
    }

    /// STS: Ryujinx Instructions/InstEmitMemory.cs:159-167.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct SharedStore {
        pub data: Register,
        pub address: Register,
        pub offset: i32,
        pub mem_type: LsSize2,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Atomic {
        pub dst: Register,
        pub address: Register,
        pub offset: u32,
        pub data: Register,
        pub op: AtomicOp,
        pub signed: bool,
    }

    /// Packed ATOMS.CAS.B32. Maxwell names the comparator register and uses
    /// the adjacent register as the replacement value.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct SharedCompareExchange {
        pub dst: Register,
        pub address: Register,
        pub offset: i32,
        pub compare: Register,
        pub replacement: Register,
    }

    /// MEMBAR.CTA (Mesa OpMemBar, sm50.rs:2986-3000, scope bits 8..9 = 0).
    /// Ryujinx Instructions/InstEmitBarrier.cs:32-44 emits
    /// Instruction.GroupMemoryBarrier for it.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct GroupMemoryBarrier;

    /// MEMBAR.GL, .SYS or .VC (scope 1..3, Ryujinx
    /// Decoders/InstDecoders.cs:479-485). Ryujinx InstEmitBarrier.cs:32-44
    /// emits Instruction.MemoryBarrier for each. The Ivall field does not
    /// change the barrier.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct MemoryBarrier;

    /// Mesa OpBar (sm50.rs:3114-3124), the sync form used by Maxwell.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct ControlBarrier;

    /// OUT with an immediate stream and unused handle/destination;
    /// Mesa sm50.rs:3257-3284. Other register/cbuf streams require a
    /// block-local constant proof; unproven values stay Unknown.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct GeometryOut {
        pub emit: bool,
        pub cut: bool,
        pub stream: u8,
    }

    /// OUT whose stream a register or cbuf word holds (Mesa
    /// sm50.rs:3257-3284,361-372; envytools gm107.c:1875,1907). A
    /// block-local constant proof names the stream; until then it is
    /// not translated.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct GeometryOutDynamic {
        pub emit: bool,
        pub cut: bool,
        pub stream: Source,
    }

    /// Scalar input AL2P, Mesa sm50.rs:2842-2851. Other forms stay Unknown.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct AttributeAddress {
        pub dst: Register,
        pub offset: Register,
        pub address: u16,
    }

    /// Default input ISBERD, Mesa nv50_ir_emit_gm107.cpp:2549-2553.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct IsbeRead {
        pub dst: Register,
        pub offset: Register,
    }

    /// Static per-vertex input ALD (P and O clear): Ra is not read
    /// (InstEmitAttribute.cs:53-101 uses only Imm11).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct AttributeLoad {
        pub dst: Register,
        pub vertex: PrimitiveVertex,
        pub address: u16,
        pub components: u8,
        /// Each lane's attribute, all in the input bank. Some lanes may be
        /// Unmapped.
        pub lanes: AttributeLanes,
    }

    /// Static per-vertex ALD.O: the output bank, except TessCoord lanes,
    /// which read the input bank (InstEmitAttribute.cs:56,98,304-307).
    /// AttributeMap.cs:323-328 gives O no primitive vertex.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct OutputAttributeLoad {
        pub dst: Register,
        /// What Rb selects.
        pub vertex: OutputVertex,
        pub address: u16,
        pub components: u8,
        pub lanes: AttributeLanes,
    }

    /// Static per-patch ALD.P (AttributeMap.cs:90-92); Rb is not read.
    /// Each lane carries the bank O selects.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct PatchAttributeLoad {
        pub dst: Register,
        pub address: u16,
        pub components: u8,
        pub lanes: AttributeLanes,
    }

    /// Physical input ALD: Imm11 zero, P and O clear and Ra not RZ
    /// (InstDecoders.cs:883-887); Ra holds the byte address in the generic
    /// bank (InstEmitAttribute.cs:43-51). Scalar only.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct PhysicalAttributeLoad {
        pub dst: Register,
        pub vertex: PrimitiveVertex,
        pub offset: Register,
    }

    /// Physical ALD.O, reading the generic output bank at Ra.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct PhysicalOutputAttributeLoad {
        pub dst: Register,
        pub vertex: OutputVertex,
        pub offset: Register,
    }

    /// Static ALD whose every lane names no attribute (Ryujinx
    /// AttributeMap.cs:63-92, see Attribute::Unmapped): any unaligned
    /// offset, regular 0x00..0x60, 0x2ec, 0x3a4..0x3fc and 0x400 up, and
    /// per-patch 0x200 up outside the fixed-function ranges. In every stage
    /// AttributeMap.cs:129-133 logs each lane and loads 0; no interface
    /// variable is read and Rb is irrelevant. Components write 0 to dst,
    /// dst+1, ... until RZ (InstEmitAttribute.cs:34-41).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct UnmappedAttributeLoad {
        pub dst: Register,
        pub address: u16,
        pub components: u8,
    }

    /// ALD with P and Ra not RZ: the per-patch words at `base` plus the
    /// byte offset in `index`, chosen at run time. Mesa emits it for an
    /// indirect patch access, Ra holding the vec4 index times 16 with
    /// physical addressing clear (nak_nir_lower_vtg_io.c:203,208-211;
    /// nak/sm50.rs:2864-2868 lets P take an offset register), and splits
    /// vectors only by alignment (:224-231). Ryujinx InstDecoders.cs:887
    /// and InstEmitAttribute.cs:53-101 ignore Ra here; Mesa settles the
    /// encoding. Rb is not read for P (nak_nir_lower_vtg_io.c:138,142).
    /// `lanes` are the static lanes Ryujinx reads at `base`, which a stage
    /// without the patch bank reads instead.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct IndexedPatchLoad {
        pub dst: Register,
        pub base: u16,
        pub index: Register,
        pub components: u8,
        pub bank: AttributeBank,
        pub lanes: AttributeLanes,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct AttributeStore {
        pub data: Register,
        pub address: u16,
        pub offset: Option<Register>,
        pub components: u8,
        pub patch: bool,
    }

    /// AST with a register offset: the address is chosen at run time
    /// (Ryujinx InstEmitAttribute.cs:105-141; envytools gm107.c:1883).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct PhysicalAttributeStore {
        pub data: Register,
        pub address: u16,
        pub offset: Register,
        pub components: u8,
        pub patch: bool,
    }

    /// Vector or scalar texture instruction. Vector handles start src1
    /// (src0 for TXQ), packed as Mesa nak_nir_lower_tex.c:350-378 describes.
    /// Bound vector/scalar forms use bound_handle; scalar pair layout follows Ryujinx
    /// InstEmitTexture.cs:394-641 and envytools gm107.c:1960,1967-1968.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Texture {
        pub op: TexOp,
        pub dst: Register,
        pub src0: Register,
        pub src1: Register,
        /// Mesa set_tex_dim: 0 1D, 1 1D array, 2 2D, 3 2D array, 4 3D,
        /// 6 cube, 7 cube array.
        pub dim: u8,
        /// Selected channels, rgba bits; packed_f16 stores two per register.
        pub mask: u8,
        /// AOFFI packed offset register follows the LOD in src1.
        pub offset: bool,
        /// Scalar register-pair encoding (envytools gm107.c:1960,1967-1968).
        pub scalar: bool,
        /// F16 scalar results pack selected pairs, padding with +0.
        /// envytools gm107.c:1960,1967-1968; Ryujinx InstEmitTexture.cs:609-655.
        pub packed_f16: bool,
        pub lod_clamp: bool,
        pub multisample: bool,
        pub depth_compare: bool,
        pub nodep: bool,
        pub scalar_dst: Option<Register>,
        /// Bound vector/scalar index: Mesa sm50.rs:2210,2254; envytools
        /// gm107.c:1960,1966-1969, U13_36. Distinct from bindless handle bits.
        pub bound_handle: Option<u16>,
    }

    /// Integer image atomic. Encoding: Mesa sm50.rs:2558-2591 and
    /// envytools gm107.c:769-791,1833-1871,1908,1911-1914.
    /// Semantics: Ryujinx InstEmitSurface.cs:180-291,463-563.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct SurfaceAtomic {
        /// 0..8 add/min/max/inc/dec/and/or/xor/exchange, 9 CAS.
        pub op: u8,
        pub signed: bool,
        pub target: u8,
        pub handle: Source,
        pub coord: Register,
        pub data: Register,
        pub dst: Option<Register>,
        pub byte_address: bool,
    }

    /// SURED: an image atomic whose old value no register receives
    /// (envytools gm107.c:1911-1914; Ryujinx InstEmitSurface.cs:463-563).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct SurfaceReduce {
        /// As [`SurfaceAtomic::op`].
        pub op: u8,
        pub signed: bool,
        pub target: u8,
        pub handle: Source,
        pub coord: Register,
        pub data: Register,
        pub byte_address: bool,
    }

    /// Surface memory operation. Fields preserve every encoded SUST/SULD
    /// field from envytools gm107.c:753-838 and gm107.c:1909-1910.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Surface {
        pub store: bool,
        pub atomic: bool,
        pub target: u8,
        pub format: u8,
        pub mask: u8,
        pub handle: Source,
        pub coord: Register,
        pub data: Register,
        pub clamp: u8,
        /// Raw SuSize 20..23, Mesa sm50.rs:2473-2474,2508-2509. The
        /// decoder rejects 7 (Ryujinx UB128): envytools gm107.c:822 OOPS,
        /// nvdisasm SM52 INVALIDSIZE7.
        pub size: u8,
        /// Raw byte-address X, envytools gm107.c:794-796; Ryujinx
        /// InstEmitSurface.cs:404-408,654-658 shifts by the texel width.
        pub byte_address: bool,
        /// Cache policy 24..26, envytools gm107.c:825-838. CG is coherent
        /// (Ryujinx InstEmitSurface.cs:677-679); store CS is Nontemporal
        /// (Mesa ir.rs:2545-2559; spirv.core.grammar.json:11990-11994).
        pub cache: u8,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Branch {
        pub target: BranchTarget,
        /// Maxwell BRA condition-code test, encoded in bits 0..4
        /// (envytools/envydis/gm107.c:1944-1945). `0xf` is unconditional.
        pub cc_test: u8,
    }

    /// A resolved SYNC/BRK/CONT. Retain the pop kind so ordinary BRA cannot
    /// acquire a token merge. Targets and stack consumption follow Ryujinx
    /// InstEmitFlowControl.cs:254-275 and Decoders/Decoder.cs:824-869.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Reconverge {
        pub target: BranchTarget,
        pub cc_test: u8,
        pub stack: TargetStack,
    }

    /// Register-relative BRX. Byte address = next slot + base + register,
    /// wrapping at 32 bits (Mesa nv50_ir_emit_gm107.cpp:533-555;
    /// Ryujinx InstEmitFlowControl.cs:38-40). Cbuf and LMT forms are rejected.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct BrxRegister {
        pub displacement: Register,
        pub base: BrxBase,
        /// Condition-code test applied after the execution predicate.
        pub cc_test: u8,
    }

    /// A BRX whose register values were proven bounded by backward recovery
    /// (Ryujinx Decoders/Decoder.cs:518-688). The targets are byte
    /// offsets from the BRX absolute base (envytools/envydis/gm107.c:1942-1943).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Brx {
        /// Register containing the loaded byte displacement, not a table index
        /// (Ryujinx Instructions/InstEmitFlowControl.cs:38-40).
        pub index: Register,
        pub base: BrxBase,
        pub table: SlotTable,
    }

    /// A proven BRX with at least one table destination outside the code
    /// address space. It runs no such slot: control-flow analysis rejects it
    /// (Ryujinx Decoders/InstOp.cs:22-25 does not wrap the sum either).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct OutsideBrx {
        pub index: Register,
        pub base: BrxBase,
        /// Each loaded byte displacement decoded once into the slot it runs
        /// or an outside entry.
        pub table: OutsideBrxTable,
    }

    /// SSY (sync), PBK (break) or PCNT (continue) as decoded: pushes a
    /// reconvergence target that a later SYNC, BRK or CONT jumps to. Its
    /// consumers are unproven. Stack resolution turns each push it proves
    /// into `ResolvedSchedule`/`ResolvedPushTarget`; one it cannot prove
    /// stays this kind, which has no lowering.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Schedule {
        pub target: BranchTarget,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct PushTarget {
        pub target: BranchTarget,
        pub stack: TargetStack,
    }

    /// A PCNT/PBK target loaded from a constant-buffer word. The source is
    /// runtime state, not a direct [`BranchTarget`]. An unused push has no
    /// observable effect; a reachable matching pop remains unproven until
    /// the translator can dispatch the runtime target.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct CbufPushTarget {
        pub binding: u8,
        pub offset: i16,
        pub stack: TargetStack,
    }
    /// SSY after stack resolution checked its consumers.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct ResolvedSchedule {
        pub push: ResolvedPush,
    }

    /// PBK/PCNT after stack resolution checked its consumers.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct ResolvedPushTarget {
        pub push: ResolvedPush,
        pub stack: TargetStack,
    }

    /// A BRA whose target lies outside the code address space. It runs no
    /// slot: control-flow analysis rejects it (Ryujinx
    /// Decoders/InstOp.cs:22-25 does not wrap the sum either).
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct OutsideBranch {
        pub target: OutsideTarget,
        /// BRA condition-code test, bits 0..4 (envytools gm107.c:1944-1945).
        pub cc_test: u8,
    }

    /// An SSY whose target lies outside the code address space. No pop can
    /// reach it, so it is never resolved.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct OutsideSchedule {
        pub target: OutsideTarget,
    }

    /// A PBK or PCNT whose target lies outside the code address space. No
    /// pop can reach it, so it is never resolved.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct OutsidePushTarget {
        pub target: OutsideTarget,
        pub stack: TargetStack,
    }

    /// SYNC, BRK or CONT before it is resolved to Reconverge by decode_stream.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct PopJump {
        pub stack: TargetStack,
        /// BRK/CONT condition0..4, envytools envydis/gm107.c:1920-1921,426-458.
        pub cc_test: u8,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Exit {
        /// Condition selector 0..4, envytools envydis/gm107.c:1925,426-454.
        pub cc_test: u8,
    }

    /// Relative CAL: a call into the function at `target`, returning to
    /// the next word. envytools gm107.c:1941,297-299: opcode mask
    /// 0xfff0000000000020, signed 24-bit target at bit 20, relative to
    /// address + 8. Ryujinx Decoder.cs:146-148 discovers the function and
    /// :157-162 its continuation. Its lowering is not built: never replace
    /// a call with a branch or a no-op.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Call {
        pub target: BranchTarget,
    }

    /// A relative CAL whose target lies outside the code address space.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct OutsideCall {
        pub target: OutsideTarget,
    }

    /// A BFI whose range register a proof found to hold `control`, a
    /// range running past bit 31 (offset + count > 32). Ryujinx
    /// InstEmitBitfield.cs:154-162 and CodeGen/Spirv/Instructions.cs:
    /// 2155-2171; SPIR-V leaves such an OpBitFieldInsert undefined.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct OverlongInsert {
        pub insert: BitFieldInsert,
        pub control: u32,
    }

    /// A BRA whose target lies between instruction slots. It runs no
    /// slot: control-flow analysis stops on it.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct MisalignedBranch {
        pub target: MisalignedTarget,
        /// BRA condition-code test, bits 0..4 (envytools gm107.c:1944-1945).
        pub cc_test: u8,
    }

    /// An SSY whose target lies between instruction slots. No pop can
    /// reach it, so it is never resolved.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct MisalignedSchedule {
        pub target: MisalignedTarget,
    }

    /// A PBK or PCNT whose target lies between instruction slots. No pop
    /// can reach it, so it is never resolved.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct MisalignedPushTarget {
        pub target: MisalignedTarget,
        pub stack: TargetStack,
    }

    /// A relative CAL whose target lies between instruction slots.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct MisalignedCall {
        pub target: MisalignedTarget,
    }

    /// The zero word: padding between and after programs.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Padding;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct Unknown {
        pub raw: [u32; 2],
    }
}
/// HSET2 writes packed booleans; HSETP2 writes two predicates.
/// Ryujinx InstEmitFloatComparison.cs:456-523.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HalfSetDestination {
    Register { dst: Register, bool_float: bool },
    Predicates { dst: [u8; 2], horizontal: bool },
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShuffleMode {
    Idx,
    Up,
    Down,
    Bfly,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Instruction {
    pub address: u32,
    pub predicate: Option<Predicate>,
    pub kind: InstructionKind,
}
impl Source {
    /// Whether this operand reads a general register.
    pub fn reads_register(self) -> bool {
        match self {
            Self::Register(_) => true,
            Self::Immediate(_) | Self::ConstantBuffer { .. } => false,
        }
    }
}
impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Register(r) => write!(f, "R{}", r.0),
            Self::Immediate(v) => write!(f, "0x{v:08x}"),
            Self::ConstantBuffer { binding, offset } => write!(f, "c[{binding}][0x{offset:x}]"),
        }
    }
}
impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(p) = self.predicate {
            write!(f, "@{}{} ", ["", "!"][usize::from(p.inverted)], p.register)?;
        }
        self.kind.visit(DisplayKind(f))
    }
}

/// An instruction kind prints as its own instruction type.
struct DisplayKind<'a, 'b>(&'a mut fmt::Formatter<'b>);

impl Visit for DisplayKind<'_, '_> {
    type Out = fmt::Result;
    fn visit<K: Kind>(self, kind: &K) -> fmt::Result {
        fmt::Display::fmt(kind, self.0)
    }
}

impl fmt::Display for InstructionKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.visit(DisplayKind(f))
    }
}

impl fmt::Display for ops::Nop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("NOP")
    }
}
impl fmt::Display for ops::VoteVtg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::VoteVtg { mode, immediate } = *self;
        write!(f, "VOTE.VTG.{mode:?} 0x{immediate:07x}")
    }
}
impl fmt::Display for ops::Vote {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::Vote {
            op,
            ballot,
            vote,
            pred,
        } = *self;
        write!(f, "VOTE.{op:?} R{}, P{}, {}", ballot.0, vote, pred.register)
    }
}
impl fmt::Display for ops::Shuffle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::Shuffle {
            dst,
            src,
            lane,
            mode,
            control,
            in_bounds,
        } = *self;
        write!(
            f,
            "SHFL.{mode:?} P{in_bounds}, R{}, R{}, {lane}, {control}",
            dst.0, src.0
        )
    }
}
impl fmt::Display for ops::SwizzleAdd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::SwizzleAdd {
            dst,
            src,
            mask,
            ftz,
        } = *self;
        write!(
            f,
            "FSWZADD{} R{}, R{}, R{}, {mask:#04x}",
            ["", ".FTZ"][usize::from(ftz)],
            dst.0,
            src[0].0,
            src[1].0
        )
    }
}
impl fmt::Display for ops::Kill {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("KILL")
    }
}
impl fmt::Display for ops::S2R {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::S2R { dst, idx } = *self;
        write!(f, "S2R R{}, 0x{idx:02x}", dst.0)
    }
}
impl fmt::Display for ops::IpaSaturate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::IpaSaturate { dst, .. } = *self;
        write!(f, "IPA.SAT R{}", dst.0)
    }
}
impl fmt::Display for ops::IpaPass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::IpaPass { dst, saturate, .. } = *self;
        write!(
            f,
            "IPA_PASS{} R{}",
            ["", ".SAT"][usize::from(saturate)],
            dst.0
        )
    }
}
impl fmt::Display for ops::IpaPassMulW {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::IpaPassMulW {
            dst,
            inv_w,
            saturate,
            ..
        } = *self;
        write!(
            f,
            "IPA_PASS_MUL_W{} R{}, {inv_w}",
            ["", ".SAT"][usize::from(saturate)],
            dst.0
        )
    }
}
impl fmt::Display for ops::Mov {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::Mov {
            dst,
            src,
            quad_lanes,
        } = *self;
        write!(f, "MOV{} R{}, {src}", QuadLanes(quad_lanes), dst.0)
    }
}
impl fmt::Display for ops::Ldc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::Ldc {
            dst,
            binding,
            offset,
            immediate,
            mem_type,
            address_mode,
        } = *self;
        write!(
            f,
            "LDC{}.{} R{}, c[{binding}][R{} + {immediate:#x}]",
            address_mode.suffix(),
            mem_type.encoded_name(),
            dst.0,
            offset.0
        )
    }
}
impl fmt::Display for ops::IAdd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::IAdd { dst, lhs, rhs, .. } = *self;
        write!(f, "IADD R{}, R{}, {rhs}", dst.0, lhs.0)
    }
}
impl fmt::Display for ops::IAdd3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::IAdd3 {
            dst,
            lhs,
            rhs,
            addend,
            halves,
            shift,
            negate,
        } = *self;
        write!(
            f,
            "IADD3.{} R{}, R{}, {rhs}, R{} {halves:?} {negate:?}",
            PARTIAL_SUM_SHIFT_NAMES[shift as usize], dst.0, lhs.0, addend.0
        )
    }
}
/// IADD3 disassembly names by [`PartialSumShift`]; the unshifted sum keeps
/// its historical spelling.
const PARTIAL_SUM_SHIFT_NAMES: [&str; 3] = ["None", "Right", "Left"];
impl fmt::Display for ops::Lea {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::Lea {
            dst,
            lhs,
            rhs,
            shift,
            negate,
        } = *self;
        write!(f, "LEA R{}, R{}, {rhs}, {shift} neg={negate}", dst.0, lhs.0)
    }
}
impl fmt::Display for ops::LeaHi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::LeaHi {
            dst,
            low,
            addend,
            high,
            shift,
            negate,
        } = *self;
        write!(
            f,
            "LEA.HI R{}, R{}, {addend}, R{}, {shift} neg={negate}",
            dst.0, low.0, high.0
        )
    }
}
impl fmt::Display for ops::FunnelShift {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::FunnelShift {
            dst,
            low,
            high,
            shift,
            right,
            dst_high,
            ty,
            wrap,
            cc,
        } = *self;
        write!(
            f,
            "SHF right={right} hi={dst_high} {ty:?} wrap={wrap} cc={cc} R{}, R{}, {shift}, R{}",
            dst.0, low.0, high.0
        )
    }
}
impl fmt::Display for ops::BitFieldInsert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::BitFieldInsert {
            dst,
            insert,
            base,
            range,
        } = *self;
        write!(f, "BFI R{}, R{}, {range}, {base}", dst.0, insert.0)
    }
}
impl fmt::Display for ops::BitFieldExtract {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::BitFieldExtract {
            dst,
            base,
            range,
            signed,
            reverse,
        } = *self;
        write!(
            f,
            "BFE signed={signed} brev={reverse} R{}, R{}, {range}",
            dst.0, base.0
        )
    }
}
impl fmt::Display for ops::IntegerMul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::IntegerMul {
            dst,
            lhs,
            rhs,
            addend,
            high,
            ..
        } = *self;
        write!(
            f,
            "{} R{}, R{}, {rhs}, {addend}",
            ["IMUL", "IMUL.HI"][usize::from(high)],
            dst.0,
            lhs.0,
        )
    }
}
impl fmt::Display for ops::FAdd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::FAdd {
            dst,
            lhs,
            rhs,
            flags,
            ..
        } = *self;
        write!(f, "FADD R{}{flags}, R{}, {rhs}", dst.0, lhs.0)
    }
}
impl fmt::Display for ops::FMul {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::FMul {
            dst,
            lhs,
            rhs,
            flags,
            ..
        } = *self;
        write!(f, "FMUL R{}{flags}, R{}, {rhs}", dst.0, lhs.0)
    }
}
impl fmt::Display for ops::FCmp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::FCmp {
            dst,
            when_true,
            when_false,
            test,
            compare,
            ftz,
        } = *self;
        write!(
            f,
            "FCMP.{compare}.Float {{ unordered: {} }}{} R{}, R{}, {when_false}, {test}",
            matches!(compare, FloatCompare::Unordered(_)),
            ["", ".FTZ"][usize::from(ftz)],
            dst.0,
            when_true.0
        )
    }
}
impl fmt::Display for ops::ICmp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::ICmp {
            dst,
            when_true,
            when_false,
            test,
            compare,
            signedness,
        } = *self;
        write!(
            f,
            "ICMP.{compare}.{signedness:?} R{}, R{}, {when_false}, {test}",
            dst.0, when_true.0
        )
    }
}
impl fmt::Display for ops::HalfArithmetic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::HalfArithmetic {
            op,
            flush,
            dst,
            src,
            swizzle,
            output,
            abs,
            negate,
            saturate,
        } = *self;
        write!(f, "H{op:?}2 R{}, {src:?} flush={flush:?} swizzle={swizzle:?} output={output} abs={abs:?} neg={negate:?} sat={saturate}", dst.0)
    }
}
impl fmt::Display for ops::HalfSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::HalfSet {
            ftz,
            dst,
            src,
            swizzle,
            abs,
            negate,
            compare,
            accum,
            accum_negate,
            set_op,
        } = *self;
        write!(f, "HSET2 {dst:?}, {src:?} ftz={ftz} swizzle={swizzle:?} abs={abs:?} neg={negate:?} cmp={compare} unordered={} accum=P{accum} inverted={accum_negate} op={set_op:?}", compare.field() & 8 != 0)
    }
}
impl fmt::Display for ops::VideoMultiplyAdd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::VideoMultiplyAdd {
            dst,
            lhs,
            rhs,
            addend,
            select,
            mode,
            shift,
            saturate,
        } = *self;
        let select = [select[0].field(), select[1].field()];
        let signed_result = self
            .select
            .iter()
            .any(|selection| selection.is_signed_whole())
            || matches!(mode, 1 | 2);
        write!(
            f,
            "VMAD R{}, R{}, {rhs}, R{} select={select:?} mode={mode} shift={shift} sat={saturate} signed={signed_result}",
            dst.0, lhs.0, addend.0
        )
    }
}
impl fmt::Display for ops::VideoMinMax {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::VideoMinMax {
            dst,
            lhs,
            rhs,
            addend,
            select,
            max,
            signed_output,
            saturate,
            op,
        } = *self;
        write!(f, "VMNMX R{}, R{}, {rhs}, R{} select={select:?} max={max} signed={signed_output} sat={saturate} op={op}", dst.0, lhs.0, addend.0)
    }
}
impl fmt::Display for ops::FFma {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::FFma {
            dst,
            lhs,
            rhs,
            addend,
            flags,
            modifiers: _,
        } = *self;
        write!(f, "FFMA R{}{flags}, R{}, {rhs}, {addend}", dst.0, lhs.0)
    }
}
impl fmt::Display for ops::FMnMx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::FMnMx {
            dst,
            lhs,
            rhs,
            min,
            flags,
            ..
        } = *self;
        write!(
            f,
            "{} R{}{flags}, R{}, {rhs}",
            ["FMAX", "FMIN"][usize::from(min)],
            dst.0,
            lhs.0
        )
    }
}
impl fmt::Display for ops::FMnMxSelect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::FMnMxSelect {
            dst,
            lhs,
            rhs,
            min_when,
            flags,
            ..
        } = *self;
        write!(
            f,
            "FMNMX R{}{flags}, R{}, {rhs}, {}P{}",
            dst.0,
            lhs.0,
            ["", "!"][usize::from(min_when.inverted)],
            min_when.register
        )
    }
}
impl fmt::Display for ops::MuFu {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::MuFu {
            dst,
            src,
            op,
            abs,
            negate,
            saturate,
        } = *self;
        write!(
            f,
            "MUFU.{}{} R{}, {}{}R{}{}",
            op as u8,
            ["", ".SAT"][usize::from(saturate)],
            dst.0,
            ["", "-"][usize::from(negate)],
            ["", "|"][usize::from(abs)],
            src.0,
            ["", "|"][usize::from(abs)]
        )
    }
}
impl fmt::Display for ops::Rro {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::Rro {
            dst,
            src,
            abs,
            negate,
            ex2,
        } = *self;
        write!(
            f,
            "RRO.{} R{}, {} abs={abs} neg={negate}",
            ["SINCOS", "EX2"][usize::from(ex2)],
            dst.0,
            src
        )
    }
}
impl fmt::Display for ops::IMinMax {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::IMinMax {
            dst,
            lhs,
            rhs,
            min,
            signed,
            cc,
        } = *self;
        write!(
            f,
            "I{}{}{} R{}, R{}, {rhs}",
            ["MAX", "MIN"][usize::from(min)],
            [".U", ".S"][usize::from(signed)],
            ["", ".CC"][usize::from(cc)],
            dst.0,
            lhs.0
        )
    }
}
impl fmt::Display for ops::IMinMaxSelect {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::IMinMaxSelect {
            dst,
            lhs,
            rhs,
            min_when,
            signed,
            cc,
        } = *self;
        write!(
            f,
            "IMNMX{}{} R{}, R{}, {rhs}, {}P{}",
            [".U", ".S"][usize::from(signed)],
            ["", ".CC"][usize::from(cc)],
            dst.0,
            lhs.0,
            ["", "!"][usize::from(min_when.inverted)],
            min_when.register
        )
    }
}
impl fmt::Display for ops::Flo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::Flo {
            dst,
            src,
            signed,
            invert,
            shift,
        } = *self;
        write!(
            f,
            "FLO{}{} R{}, {}{src}",
            [".U32", ".S32"][usize::from(signed)],
            ["", ".SH"][usize::from(shift)],
            dst.0,
            ["", "~"][usize::from(invert)],
        )
    }
}
impl fmt::Display for ops::Popc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::Popc { dst, src, invert } = *self;
        write!(
            f,
            "POPC{} R{}, {src}",
            ["", ".INV"][usize::from(invert)],
            dst.0
        )
    }
}
impl fmt::Display for ops::ISetP {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::ISetP {
            dst: [first, second],
            lhs,
            rhs,
            ..
        } = *self;
        write!(f, "ISETP P{first}")?;
        // A PT second destination is not shown.
        (second != 7)
            .then_some(second)
            .into_iter()
            .try_for_each(|second| write!(f, ", P{second}"))?;
        write!(f, ", R{}, {rhs}", lhs.0)
    }
}
impl fmt::Display for ops::ISet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::ISet { dst, lhs, rhs, .. } = *self;
        write!(f, "ISET R{}, R{}, {rhs}", dst.0, lhs.0)
    }
}
impl fmt::Display for ops::Xmad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::Xmad { dst, a, b, c, .. } = *self;
        write!(f, "XMAD R{}, R{}, {b}, {c}", dst.0, a.0)
    }
}
impl fmt::Display for ops::ShiftAdd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::ShiftAdd {
            dst,
            lhs,
            rhs,
            shift,
            negate,
            ..
        } = *self;
        write!(
            f,
            "ISCADD R{}, {}R{} << {shift}, {}{rhs}",
            dst.0,
            ["", "-"][usize::from(negate[0])],
            lhs.0,
            ["", "-"][usize::from(negate[1])]
        )
    }
}
impl fmt::Display for ops::P2R {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::P2R {
            dst,
            src,
            mask,
            byte,
            cc,
        } = *self;
        write!(
            f,
            "P2R R{}, R{}, mask={mask}, b{byte}, cc={cc}",
            dst.0, src.0
        )
    }
}
impl fmt::Display for ops::VideoSetP {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::VideoSetP {
            dst,
            lhs,
            rhs,
            select,
            compare,
            ..
        } = *self;
        write!(
            f,
            "VSETP P{}, P{}, R{}, {rhs}, {select:?}, {compare}",
            dst[0], dst[1], lhs.0
        )
    }
}
impl fmt::Display for ops::R2P {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::R2P {
            src,
            mask,
            byte,
            bank,
        } = *self;
        let suffix = match bank {
            FlagBank::Predicates => "",
            FlagBank::ConditionCodes => ".CC",
        };
        write!(f, "R2P{suffix} mask={mask}, b{byte}, R{}", src.0)
    }
}
impl fmt::Display for ops::DoubleSetP {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::DoubleSetP { dst, lhs, rhs, .. } = *self;
        write!(f, "DSETP P{}, P{}, {lhs}, {rhs}", dst[0], dst[1])
    }
}
impl fmt::Display for ops::DoubleSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::DoubleSet {
            dst,
            lhs,
            rhs,
            result,
            ..
        } = *self;
        write!(f, "DSET.{result:?} R{}, {lhs}, {rhs}", dst.0)
    }
}
impl fmt::Display for ops::DoubleAdd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::DoubleAdd {
            dst,
            lhs,
            rhs,
            modifiers,
            round,
            flags,
        } = *self;
        write!(f, "DADD.{round:?}.{modifiers:?}{flags} {dst}, {lhs}, {rhs}")
    }
}
impl fmt::Display for ops::DoubleMinMax {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::DoubleMinMax {
            dst,
            lhs,
            rhs,
            operation,
            modifiers,
            flags,
        } = *self;
        write!(
            f,
            "DMNMX.{operation:?}.{modifiers:?}.{flags} {dst}, {lhs}, {rhs}"
        )
    }
}
impl fmt::Display for ops::DoubleMultiply {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::DoubleMultiply {
            dst,
            lhs,
            rhs,
            sign,
            flags,
        } = *self;
        write!(f, "DMUL.{sign:?}{flags} {dst}, {lhs}, {rhs}")
    }
}
impl fmt::Display for ops::DoubleFma {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::DoubleFma {
            dst,
            lhs,
            multiplier,
            addend,
            product_sign,
            addend_sign,
            flags,
        } = *self;
        write!(
            f,
            "DFMA.{product_sign:?}.{addend_sign:?}{flags} {dst}, {lhs}, {multiplier}, {addend}"
        )
    }
}
impl fmt::Display for ops::CSetP {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::CSetP { dst, condition, .. } = *self;
        write!(
            f,
            "CSETP.{:#x} P{}, P{}",
            condition.selector(),
            dst[0],
            dst[1]
        )
    }
}
impl fmt::Display for ops::CSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::CSet { dst, condition, .. } = *self;
        write!(f, "CSET.{:#x} R{}", condition.selector(), dst.0)
    }
}
impl fmt::Display for ops::PSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::PSet { dst, .. } = *self;
        write!(f, "PSET R{}", dst.0)
    }
}
impl fmt::Display for ops::PSetP {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::PSetP { dst, .. } = *self;
        write!(f, "PSETP P{}, P{}", dst[0], dst[1])
    }
}
impl fmt::Display for ops::FSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::FSet { dst, lhs, rhs, .. } = *self;
        write!(f, "FSET R{}, R{}, {rhs}", dst.0, lhs.0)
    }
}
impl fmt::Display for ops::Sel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::Sel {
            dst,
            when_true,
            when_false,
            condition,
            condition_negate,
        } = *self;
        write!(
            f,
            "SEL R{}, R{}, {when_false}, {}P{}",
            dst.0,
            when_true.0,
            ["", "!"][usize::from(condition_negate)],
            condition
        )
    }
}
impl fmt::Display for ops::Convert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::Convert { kind, dst, src } = *self;
        write!(f, "{kind:?} R{}, {src}", dst.0)
    }
}
impl fmt::Display for ops::DoubleToInt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::DoubleToInt {
            dst,
            src,
            modifier,
            signedness,
            width,
            round,
        } = *self;
        write!(
            f,
            "F2I.F64.{round:?}.{}{} R{}, {modifier:?} {src}",
            ["U", "S"][signedness as usize],
            [16, 32][width as usize],
            dst.0,
        )
    }
}
impl fmt::Display for ops::DoubleToFloat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::DoubleToFloat {
            dst,
            src,
            modifier,
            round,
            saturate,
        } = *self;
        write!(
            f,
            "F2F.F32.F64.{round:?}.{saturate:?} R{}, {modifier:?} {src}",
            dst.0
        )
    }
}
impl fmt::Display for ops::DoubleRound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::DoubleRound {
            dst,
            src,
            modifier,
            integer_round,
            saturate,
        } = *self;
        write!(
            f,
            "F2F.F64.F64.{integer_round:?}.{saturate:?} {dst}, {modifier:?} {src}"
        )
    }
}
impl fmt::Display for ops::Lop {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::Lop {
            dst, lhs, rhs, op, ..
        } = *self;
        write!(f, "LOP{} R{}, R{}, {rhs}", op as u8, dst.0, lhs.0)
    }
}
impl fmt::Display for ops::Lop3 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::Lop3 {
            dst,
            lhs,
            rhs,
            third,
            lut,
            ..
        } = *self;
        write!(
            f,
            "LOP3 R{}, R{}, {rhs}, R{}, 0x{lut:02x}",
            dst.0, lhs.0, third.0
        )
    }
}
impl fmt::Display for ops::LopWithPredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::LopWithPredicate {
            dst, lhs, rhs, op, ..
        } = *self;
        write!(f, "LOP{} R{}, R{}, {rhs}", op as u8, dst.0, lhs.0)
    }
}
impl fmt::Display for ops::Lop3WithPredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::Lop3WithPredicate {
            dst,
            lhs,
            rhs,
            third,
            lut,
            ..
        } = *self;
        write!(
            f,
            "LOP3 R{}, R{}, {rhs}, R{}, 0x{lut:02x}",
            dst.0, lhs.0, third.0
        )
    }
}
impl fmt::Display for ops::Shl {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::Shl { dst, lhs, rhs, .. } = *self;
        write!(f, "SHL R{}, R{}, {rhs}", dst.0, lhs.0)
    }
}
impl fmt::Display for ops::Shr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::Shr { dst, lhs, rhs, .. } = *self;
        write!(f, "SHR R{}, R{}, {rhs}", dst.0, lhs.0)
    }
}
impl fmt::Display for ops::Prmt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::Prmt {
            dst,
            lhs,
            rhs,
            selector,
        } = *self;
        write!(f, "PRMT R{}, R{}, {}, {}", dst.0, lhs.0, selector, rhs.0)
    }
}
impl fmt::Display for ops::IMadSp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::IMadSp {
            dst,
            a,
            b,
            c,
            types,
        } = *self;
        let [ta, tb, tc] = types.map(|t| {
            [
                "U32", "S32", "U24", "S24", "U16H0", "S16H0", "U16H1", "S16H1",
            ][t as usize]
        });
        write!(f, "IMADSP.{ta}.{tb}.{tc} R{}, R{}, {b}, {c}", dst.0, a.0)
    }
}
impl fmt::Display for ops::GlobalMemory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::GlobalMemory {
            access,
            data,
            address,
            offset,
            extended,
            cache,
        } = *self;
        write!(
            f,
            "{}{} cache={} size={} R{}, [R{}{:+}]",
            match access {
                GlobalAccess::Load(_) => "LDG",
                GlobalAccess::Store(_) => "STG",
            },
            ["", ".E"][usize::from(extended)],
            cache,
            match access {
                GlobalAccess::Load(size) => size as u8,
                GlobalAccess::Store(size) => size as u8,
            },
            data.0,
            address.0,
            offset
        )
    }
}
impl fmt::Display for ops::IntegerMulAdd {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::IntegerMulAdd {
            dst,
            lhs,
            rhs,
            addend,
            high,
            ..
        } = *self;
        write!(
            f,
            "{} R{}, R{}, {rhs}, R{}",
            ["IMUL", "IMUL.HI"][usize::from(high)],
            dst.0,
            lhs.0,
            addend.0,
        )
    }
}
impl fmt::Display for ops::GlobalReduce {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::GlobalReduce {
            data,
            address,
            offset,
            extended,
            signed,
            op,
            width,
        } = *self;
        write!(
            f,
            "RED{} op={} {}{} dst=None, [R{}{:+}], R{}",
            ["", ".E"][usize::from(extended)],
            op,
            ["U", "S"][usize::from(signed)],
            u32::from(width.registers()) * 32,
            address.0,
            offset,
            data.0
        )
    }
}
impl fmt::Display for ops::GlobalAtomic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::GlobalAtomic {
            dst,
            data,
            address,
            offset,
            extended,
            signed,
            op,
            width,
        } = *self;
        write!(
            f,
            "{}{} op={} {}{} dst={:?}, [R{}{:+}], R{}",
            ["RED", "ATOM"][usize::from(dst.is_some())],
            ["", ".E"][usize::from(extended)],
            op,
            ["U", "S"][usize::from(signed)],
            u32::from(width.registers()) * 32,
            dst.map(|r| r.0),
            address.0,
            offset,
            data.0
        )
    }
}
impl fmt::Display for ops::LocalLoad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::LocalLoad {
            dst,
            address,
            offset,
            ..
        } = *self;
        write!(f, "LDL R{}, [R{}+0x{:x}]", dst.0, address.0, offset)
    }
}
impl fmt::Display for ops::SharedLoad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::SharedLoad {
            dst,
            address,
            offset,
            ..
        } = *self;
        write!(f, "LDS R{}, [R{}+0x{:x}]", dst.0, address.0, offset)
    }
}
impl fmt::Display for ops::LocalStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::LocalStore {
            data,
            address,
            offset,
            cache,
            ..
        } = *self;
        write!(
            f,
            "STL{} [R{}+0x{:x}], R{}",
            cache.suffix(),
            address.0,
            offset,
            data.0
        )
    }
}
impl fmt::Display for ops::SharedStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::SharedStore {
            data,
            address,
            offset,
            ..
        } = *self;
        write!(f, "STS [R{}+0x{:x}], R{}", address.0, offset, data.0)
    }
}
impl fmt::Display for ops::Atomic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::Atomic {
            dst,
            address,
            offset,
            data,
            op,
            ..
        } = *self;
        write!(
            f,
            "ATOMS.{op:?} R{}, [R{}+{}], R{}",
            dst.0, address.0, offset, data.0
        )
    }
}
impl fmt::Display for ops::SharedCompareExchange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::SharedCompareExchange {
            dst,
            address,
            offset,
            compare,
            replacement,
        } = *self;
        write!(
            f,
            "ATOMS.CAS.B32 R{}, [R{}+{}], R{}:R{}",
            dst.0, address.0, offset, compare.0, replacement.0
        )
    }
}
impl fmt::Display for ops::GroupMemoryBarrier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("MEMBAR.GROUP")
    }
}
impl fmt::Display for ops::MemoryBarrier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("MEMBAR")
    }
}
impl fmt::Display for ops::ControlBarrier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("BAR.SYNC")
    }
}
impl fmt::Display for ops::GeometryOut {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::GeometryOut { emit, cut, stream } = *self;
        write!(f, "OUT emit={emit} cut={cut} stream={stream}")
    }
}
impl fmt::Display for ops::AttributeAddress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::AttributeAddress {
            dst,
            offset,
            address,
        } = *self;
        write!(f, "AL2P R{}, R{}, {address:#x}", dst.0, offset.0)
    }
}
impl fmt::Display for ops::IsbeRead {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::IsbeRead { dst, offset } = *self;
        write!(f, "ISBERD R{}, R{}", dst.0, offset.0)
    }
}
impl fmt::Display for ops::Texture {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::Texture {
            op,
            dst,
            src0,
            src1,
            dim,
            mask,
            offset,
            packed_f16,
            ..
        } = *self;
        f.write_str("TEX.")?;
        match op {
            TexOp::Gather {
                component,
                offsets: GatherOffsets::Zero | GatherOffsets::One,
            } => write!(f, "Gather {{ component: {component} }}")?,
            _ => fmt::Debug::fmt(&op, f)?,
        }
        write!(
            f,
            "{} R{}, R{}, R{}, dim{dim}, mask{mask:#x}{}",
            ["", ".F16"][usize::from(packed_f16)],
            dst.0,
            src0.0,
            src1.0,
            if offset
                || matches!(
                    op,
                    TexOp::Gather {
                        offsets: GatherOffsets::One,
                        ..
                    }
                )
            {
                ", aoffi"
            } else {
                ""
            }
        )
    }
}
impl fmt::Display for ops::SurfaceReduce {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::SurfaceReduce {
            op,
            signed,
            target,
            handle,
            coord,
            data,
            byte_address,
        } = *self;
        write!(f, "SUATOM op{op} signed{signed} target{target} {handle:?}, R{}, R{}, None, ba{byte_address}", coord.0, data.0)
    }
}
impl fmt::Display for ops::SurfaceAtomic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::SurfaceAtomic {
            op,
            signed,
            target,
            handle,
            coord,
            data,
            dst,
            byte_address,
        } = *self;
        write!(f, "SUATOM op{op} signed{signed} target{target} {handle:?}, R{}, R{}, {dst:?}, ba{byte_address}", coord.0, data.0)
    }
}
impl fmt::Display for ops::Surface {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::Surface {
            store,
            atomic,
            target,
            format,
            mask,
            handle,
            coord,
            data,
            clamp,
            size,
            byte_address,
            cache,
        } = *self;
        write!(
                f,
                "SU{}{} R{}, R{}, target{}, format{}, mask{}, handle {:?}, clamp{}, size{}, ba{}, cache{}",
                ["LD", "ST"][usize::from(store)],
                ["", "ATOM"][usize::from(atomic)],
                data.0,
                coord.0,
                target,
                format,
                mask,
                handle,
                clamp, size, byte_address, cache
            )
    }
}
impl fmt::Display for ops::AttributeLoad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::AttributeLoad { dst, address, .. } = *self;
        write!(f, "ATTR_LOAD R{}, a{}", dst.0, address)
    }
}
impl fmt::Display for ops::OutputAttributeLoad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::OutputAttributeLoad { dst, address, .. } = *self;
        write!(f, "ATTR_LOAD R{}, a{}", dst.0, address)
    }
}
impl fmt::Display for ops::PatchAttributeLoad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::PatchAttributeLoad { dst, address, .. } = *self;
        write!(f, "ATTR_LOAD R{}, a{}", dst.0, address)
    }
}
impl fmt::Display for ops::PhysicalAttributeLoad {
    // A physical ALD's immediate is zero by decode.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::PhysicalAttributeLoad { dst, .. } = *self;
        write!(f, "ATTR_LOAD R{}, a0", dst.0)
    }
}
impl fmt::Display for ops::PhysicalOutputAttributeLoad {
    // A physical ALD's immediate is zero by decode.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::PhysicalOutputAttributeLoad { dst, .. } = *self;
        write!(f, "ATTR_LOAD R{}, a0", dst.0)
    }
}
impl fmt::Display for ops::UnmappedAttributeLoad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::UnmappedAttributeLoad { dst, address, .. } = *self;
        write!(f, "ATTR_LOAD R{}, a{} (unmapped)", dst.0, address)
    }
}
impl fmt::Display for ops::IndexedPatchLoad {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::IndexedPatchLoad {
            dst, base, index, ..
        } = *self;
        write!(f, "ATTR_LOAD R{}, a[R{}+{}] (patch)", dst.0, index.0, base)
    }
}
impl fmt::Display for ops::AttributeStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::AttributeStore { data, address, .. } = *self;
        write!(f, "ATTR_STORE R{}, a{}", data.0, address)
    }
}
impl fmt::Display for ops::PhysicalAttributeStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::PhysicalAttributeStore { data, address, .. } = *self;
        write!(f, "ATTR_STORE R{}, a{}", data.0, address)
    }
}
impl fmt::Display for ops::Branch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::Branch { target, cc_test } = *self;
        write!(f, "BRA.{cc_test:#x} {:+#x}", target.offset())
    }
}
impl fmt::Display for ops::Reconverge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::Reconverge {
            target, cc_test, ..
        } = *self;
        write!(f, "BRA.{cc_test:#x} {:+#x}", target.offset())
    }
}
impl fmt::Display for ops::BrxRegister {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::BrxRegister {
            displacement,
            base,
            cc_test,
        } = *self;
        write!(
            f,
            "BRX{} R{} {:+#x}",
            CcSuffix(cc_test),
            displacement.0,
            base.offset()
        )
    }
}
impl fmt::Display for ops::Brx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::Brx { index, base, table } = *self;
        write!(
            f,
            "BRX R{} {:+#x} ({} targets)",
            index.0,
            base.offset(),
            table.count()
        )
    }
}
impl fmt::Display for ops::OutsideBrx {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::OutsideBrx { index, base, table } = *self;
        write!(
            f,
            "BRX R{} {:+#x} ({} targets)",
            index.0,
            base.offset(),
            table.count()
        )
    }
}
impl fmt::Display for ops::OutsideBranch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::OutsideBranch { target, cc_test } = *self;
        write!(f, "BRA.{cc_test:#x} {:+#x}", target.offset())
    }
}
impl fmt::Display for ops::Schedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::Schedule { target } = *self;
        write!(f, "SSY {:+#x}", target.offset())
    }
}
impl fmt::Display for ops::ResolvedSchedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::ResolvedSchedule { push } = *self;
        write!(f, "SSY {:+#x}", push.target().offset())
    }
}
impl fmt::Display for ops::OutsideSchedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::OutsideSchedule { target } = *self;
        write!(f, "SSY {:+#x}", target.offset())
    }
}
impl fmt::Display for ops::PushTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::PushTarget { target, stack } = *self;
        write!(f, "PUSH.{stack:?} {:+#x}", target.offset())
    }
}
impl fmt::Display for ops::CbufPushTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::CbufPushTarget {
            binding,
            offset,
            stack,
        } = *self;
        write!(f, "PUSH.{stack:?} c[{binding}][{offset:+#x}]")
    }
}
impl fmt::Display for ops::ResolvedPushTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::ResolvedPushTarget { push, stack } = *self;
        write!(f, "PUSH.{stack:?} {:+#x}", push.target().offset())
    }
}
impl fmt::Display for ops::OutsidePushTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::OutsidePushTarget { target, stack } = *self;
        write!(f, "PUSH.{stack:?} {:+#x}", target.offset())
    }
}
impl fmt::Display for ops::PopJump {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::PopJump { stack, cc_test } = *self;
        write!(f, "POPJUMP.{stack:?}{}", CcSuffix(cc_test))
    }
}
impl fmt::Display for ops::Exit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::Exit { cc_test } = *self;
        write!(f, "EXIT{}", CcSuffix(cc_test))
    }
}
impl fmt::Display for ops::FSetP {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::FSetP {
            dst: [first, second],
            lhs,
            rhs,
            ..
        } = *self;
        write!(
            f,
            "FSETP P{first}{}, R{}, {rhs}",
            SecondPredicate(second),
            lhs.0
        )
    }
}

/// A condition-code test prints as a suffix; CC.T (0xf) is the bare form.
struct CcSuffix(u8);

impl fmt::Display for CcSuffix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const SUFFIX: [fn(u8, &mut fmt::Formatter<'_>) -> fmt::Result; 2] =
            [|cc, f| write!(f, ".{cc:#x}"), |_, _| Ok(())];
        SUFFIX[usize::from(self.0 == 0xf)](self.0, f)
    }
}

/// MOV quad lane enables print where some lane is off; 0xf is the bare form.
struct QuadLanes(u8);

impl fmt::Display for QuadLanes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const LANES: [fn(u8, &mut fmt::Formatter<'_>) -> fmt::Result; 2] =
            [|lanes, f| write!(f, "[{lanes:#x}]"), |_, _| Ok(())];
        LANES[usize::from(self.0 == 0xf)](self.0, f)
    }
}

/// A second destination predicate prints after the first; PT (7) is none.
struct SecondPredicate(u8);

impl fmt::Display for SecondPredicate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const SECOND: [fn(u8, &mut fmt::Formatter<'_>) -> fmt::Result; 2] =
            [|p, f| write!(f, ", P{p}"), |_, _| Ok(())];
        SECOND[usize::from(self.0 == 7)](self.0, f)
    }
}

impl fmt::Display for ops::GeometryOutDynamic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::GeometryOutDynamic { emit, cut, stream } = *self;
        write!(f, "OUT emit={emit} cut={cut} stream={stream}")
    }
}
impl fmt::Display for ops::Call {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::Call { target } = *self;
        write!(f, "CAL {:+#x}", target.offset())
    }
}
impl fmt::Display for ops::OutsideCall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::OutsideCall { target } = *self;
        write!(f, "CAL {:+#x}", target.offset())
    }
}
impl fmt::Display for ops::OverlongInsert {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::OverlongInsert { insert, control } = *self;
        write!(f, "{insert} (range {control:#x})")
    }
}
impl fmt::Display for ops::MisalignedBranch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::MisalignedBranch { target, cc_test } = *self;
        write!(f, "BRA.{cc_test:#x} {:+#x}", target.offset())
    }
}
impl fmt::Display for ops::MisalignedSchedule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::MisalignedSchedule { target } = *self;
        write!(f, "SSY {:+#x}", target.offset())
    }
}
impl fmt::Display for ops::MisalignedPushTarget {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::MisalignedPushTarget { target, stack } = *self;
        write!(f, "PUSH.{stack:?} {:+#x}", target.offset())
    }
}
impl fmt::Display for ops::MisalignedCall {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::MisalignedCall { target } = *self;
        write!(f, "CAL {:+#x}", target.offset())
    }
}
impl fmt::Display for ops::Padding {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::Padding = *self;
        write!(f, "PADDING")
    }
}
impl ops::Unknown {
    /// The disassembler's text of this word when a format learned from it
    /// (`decoder::formats`) prints the word: recognised, still untranslatable.
    pub fn text(&self) -> Option<String> {
        let ops::Unknown { raw } = *self;
        let word = u64::from(raw[0]) | (u64::from(raw[1]) << 32);
        let table = &crate::decoder::formats::TABLE;
        table
            .recognise(word)
            .and_then(|variant| table.text(variant, word))
    }
}
impl fmt::Display for ops::Unknown {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ops::Unknown { raw } = *self;
        write!(f, "UNKNOWN 0x{:08x}{:08x}", raw[1], raw[0])
    }
}

pub struct Program {
    pub instruction_count: usize,
}

impl InstructionKind {
    /// Whether this instruction may write a predicate register (or ends a
    /// straight-line run). Used to prove a predicated definition is still
    /// guarded by the same predicate value at a later use. The match has no
    /// wildcard and lists every field of each `false` arm, so adding a kind
    /// or a field (for example a predicate output) fails to compile until
    /// this is updated.
    pub fn may_write_predicate(&self) -> bool {
        self.visit(MayWritePredicate)
    }

    /// General registers this instruction writes (first register and count),
    /// or None when it writes no general register. Used to trace values
    /// such as texture handles back to their definition.
    /// Destinations: Mesa sm50.rs:1951 (MOV), :1598 (LOP),
    /// envytools gm107.c:2108 (LOP3), :2029 (ISET);
    /// Mesa sm50.rs:2736 (atomics), :2635-2648 (LDC), :2855-2876 (ALD).
    /// Widths: Mesa sm50.rs:2386-2397 (memory types), :2476-2478 (surface
    /// mask); envytools gm107.c:799-814 defines the formatted channel mask.
    pub fn written_registers(&self) -> Written {
        self.visit(WrittenRegisters)
    }

    /// Every destination range, including TEXS's second pair.
    pub fn destinations(&self) -> [Written; 2] {
        self.visit(Destinations)
    }

    /// What the module around this instruction is made with because of
    /// it, told to `module`.
    pub fn require<M: Requirements>(&self, module: &mut M) {
        self.visit(Require(module))
    }

    /// What `pass` does at this instruction, by whether the emitter can
    /// translate it.
    pub fn translation<T: Translation>(&self, pass: T) -> T::Out {
        self.visit(TranslationOf(pass))
    }

    /// The combined IMUL, ATOM or SUATOM form of IMAD, RED and SURED;
    /// every other kind itself.
    pub fn legacy_form(&self) -> InstructionKind {
        self.visit(LegacyForm)
    }

    /// The rounding this instruction multiplies an FMUL product in.
    pub fn product_rounding(&self) -> FMulRounding {
        self.visit(ProductRounding)
    }

    /// Marks the dispatch blocks this instruction starts.
    pub fn mark_block_starts<E: Extend<u32>>(&self, starts: &mut E, next: &[Instruction]) {
        self.visit(MarkBlockStarts(starts, next))
    }

    /// Whether the guard predicate wraps the whole instruction.
    pub fn guards_whole(&self) -> bool {
        self.visit(GuardsWhole)
    }

    /// Whether this instruction reads a traced handle.
    pub fn handle_use(&self, include_global: bool) -> bool {
        self.visit(HandleUse(include_global))
    }

    /// Whether this instruction really writes a predicate register.
    pub fn writes_predicate_register(&self) -> bool {
        self.visit(WritesPredicateRegister)
    }

    /// The predicate registers this instruction writes.
    pub fn written_predicates(&self) -> PredicateSet {
        self.visit(WrittenPredicates)
    }

    /// Whether every destination register is written whole.
    pub fn whole_write(&self) -> bool {
        self.visit(WholeWrite)
    }

    /// How this instruction bounds code detached from the entry run.
    pub fn bound_detached<B: DetachedBounds>(&self, bounds: &mut B) {
        self.visit(BoundDetached(bounds))
    }

    /// What this unguarded instruction proves about attribute addresses.
    pub fn prove_address(&self, proven: &mut ProvenAddresses) {
        self.visit(ProveAddress(proven))
    }

    /// This instruction once its physical offset is a proven address.
    pub fn at_proven_address(&self, proven: &ProvenAddresses) -> InstructionKind {
        self.visit(AtProvenAddress(proven))
    }

    /// Declare the attribute interface this instruction touches.
    pub fn declare_attributes<A: AttributeIo>(&self, io: &mut A) -> Result<(), A::Error> {
        self.visit(DeclareAttributes(io))
    }

    /// The generic banks this instruction addresses at run time.
    pub fn dynamic_banks(&self) -> [bool; 2] {
        self.visit(DynamicBanks)
    }

    /// Whether this instruction reads SR_INVOCATION_ID.
    pub fn reads_invocation_id(&self) -> bool {
        self.visit(ReadsInvocationId)
    }

    /// Whether this instruction reads SR_INVOCATION_INFO.
    pub fn reads_invocation_info(&self) -> bool {
        self.visit(ReadsInvocationInfo)
    }

    /// The geometry streams this instruction cuts or emits on.
    pub fn stream_marks(&self) -> StreamMarks {
        self.visit(StreamMarksOf)
    }

    /// The generic output locations this instruction touches.
    pub fn used_outputs(&self) -> u32 {
        self.visit(UsedOutputs)
    }
}

impl InstructionKind {
    /// Includes both disjoint TEXS destination pairs. Ryujinx
    /// InstEmitTexture.cs:621-641 maps the first two results to dst and
    /// the remaining results to scalar_dst. F16 packs each pair into one
    /// register (InstEmitTexture.cs:652-655).
    pub fn writes_register(&self, register: u8) -> bool {
        [self.visit(WritesRegister(register)), false][usize::from(register == 255)]
    }

    /// Every operand of type Source (register, immediate or constant buffer).
    pub fn sources(&self) -> Vec<Source> {
        self.source_operands().collect()
    }

    /// The same source list without allocating for emitter scans.
    pub(crate) fn source_operands(&self) -> impl Iterator<Item = Source> {
        self.visit(SourceSlots).into_iter()
    }
}
