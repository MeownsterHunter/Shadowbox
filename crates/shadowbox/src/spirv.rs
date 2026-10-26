//! SPIR-V module words the emitter writes by name: the header magic and
//! version words, and every instruction opcode it emits or reads back.
//!
//! Names and numbers: Khronos SPIR-V Unified Specification, sections 2.3
//! (physical layout) and 3.52 (instructions), as published in the
//! machine-readable grammar `spirv.core.grammar.json` (unified1, SPIR-V 1.6
//! revision 7, Copyright 2014-2024 The Khronos Group Inc., MIT license).
//! https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html
//! Sources: docs/reference-sources.md

/// First word of every SPIR-V module (specification section 2.3).
pub const MAGIC: u32 = 0x0723_0203;
/// Version word for SPIR-V 1.3: major in bits 16-23, minor in bits 8-15.
pub const VERSION_1_3: u32 = 0x0001_0300;
/// Version word for SPIR-V 1.6, required by nontemporal image operands.
pub(crate) const VERSION_1_6: u32 = 0x0001_0600;
/// StorageClass Function (specification section 3.7): the only class whose
/// variables are not module globals.
pub(crate) const STORAGE_CLASS_FUNCTION: u32 = 7;

/// Declares `Op` with one variant per named opcode, its incoming edge
/// `Op::of` and its number. Each name is the specification's instruction name
/// without the `Op` prefix.
macro_rules! opcodes {
    ($($name:ident = $number:literal,)*) => {
        /// An instruction opcode: the low 16 bits of an instruction's first
        /// word (specification section 2.3). Every opcode this crate emits or
        /// reads is its own variant; any other number is `Unnamed`.
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub enum Op {
            $($name,)*
            /// An opcode number this crate gives no name.
            Unnamed(UnnamedOpcode),
        }

        impl Op {
            /// The incoming edge for opcodes: the opcode of the instruction
            /// whose first word is `word`, read from the word's low half.
            pub const fn of(word: u32) -> Op {
                let [low, high, _, _] = word.to_le_bytes();
                match u16::from_le_bytes([low, high]) {
                    $($number => Op::$name,)*
                    number => Op::Unnamed(UnnamedOpcode(number)),
                }
            }

            /// The opcode's number, as an instruction's first word carries it.
            pub const fn number(self) -> u16 {
                match self {
                    $(Op::$name => $number,)*
                    Op::Unnamed(UnnamedOpcode(number)) => number,
                }
            }
        }
    };
}

/// An opcode number with no `Op` name. Only `Op::of` builds one, so it never
/// holds a named number.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct UnnamedOpcode(u16);

impl From<Op> for u32 {
    fn from(op: Op) -> u32 {
        u32::from(op.number())
    }
}

impl From<Op> for usize {
    fn from(op: Op) -> usize {
        usize::from(op.number())
    }
}

opcodes! {
    Nop = 0,
    Undef = 1,
    SourceContinued = 2,
    Source = 3,
    SourceExtension = 4,
    Name = 5,
    MemberName = 6,
    String = 7,
    Line = 8,
    Extension = 10,
    ExtInstImport = 11,
    ExtInst = 12,
    MemoryModel = 14,
    EntryPoint = 15,
    ExecutionMode = 16,
    Capability = 17,
    TypeVoid = 19,
    TypeBool = 20,
    TypeInt = 21,
    TypeFloat = 22,
    TypeVector = 23,
    TypeMatrix = 24,
    TypeImage = 25,
    TypeSampler = 26,
    TypeSampledImage = 27,
    TypeArray = 28,
    TypeRuntimeArray = 29,
    TypeStruct = 30,
    TypeOpaque = 31,
    TypePointer = 32,
    TypeFunction = 33,
    TypeEvent = 34,
    TypeDeviceEvent = 35,
    TypeReserveId = 36,
    TypeQueue = 37,
    TypePipe = 38,
    TypeForwardPointer = 39,
    ConstantTrue = 41,
    ConstantFalse = 42,
    Constant = 43,
    ConstantComposite = 44,
    ConstantSampler = 45,
    ConstantNull = 46,
    SpecConstantTrue = 48,
    SpecConstantFalse = 49,
    SpecConstant = 50,
    SpecConstantComposite = 51,
    SpecConstantOp = 52,
    Function = 54,
    FunctionEnd = 56,
    Variable = 59,
    ImageTexelPointer = 60,
    Load = 61,
    Store = 62,
    AccessChain = 65,
    InBoundsAccessChain = 66,
    Decorate = 71,
    MemberDecorate = 72,
    VectorExtractDynamic = 77,
    VectorShuffle = 79,
    CompositeConstruct = 80,
    CompositeExtract = 81,
    CompositeInsert = 82,
    CopyObject = 83,
    SampledImage = 86,
    ImageSampleImplicitLod = 87,
    ImageSampleExplicitLod = 88,
    ImageSampleDrefImplicitLod = 89,
    ImageSampleDrefExplicitLod = 90,
    ImageSampleProjImplicitLod = 91,
    ImageSampleProjExplicitLod = 92,
    ImageSampleProjDrefImplicitLod = 93,
    ImageSampleProjDrefExplicitLod = 94,
    ImageFetch = 95,
    ImageGather = 96,
    ImageDrefGather = 97,
    ImageRead = 98,
    ImageWrite = 99,
    ImageQueryFormat = 101,
    ImageQueryOrder = 102,
    ImageQuerySizeLod = 103,
    ImageQuerySize = 104,
    ImageQueryLod = 105,
    ImageQueryLevels = 106,
    ImageQuerySamples = 107,
    ConvertFToU = 109,
    ConvertFToS = 110,
    ConvertSToF = 111,
    ConvertUToF = 112,
    UConvert = 113,
    SConvert = 114,
    FConvert = 115,
    ConvertUToPtr = 120,
    Bitcast = 124,
    SNegate = 126,
    FNegate = 127,
    IAdd = 128,
    FAdd = 129,
    ISub = 130,
    FSub = 131,
    IMul = 132,
    FMul = 133,
    FDiv = 136,
    IAddCarry = 149,
    UMulExtended = 151,
    SMulExtended = 152,
    IsNan = 156,
    LogicalEqual = 164,
    LogicalNotEqual = 165,
    LogicalOr = 166,
    LogicalAnd = 167,
    LogicalNot = 168,
    Select = 169,
    IEqual = 170,
    INotEqual = 171,
    UGreaterThan = 172,
    SGreaterThan = 173,
    UGreaterThanEqual = 174,
    SGreaterThanEqual = 175,
    ULessThan = 176,
    SLessThan = 177,
    ULessThanEqual = 178,
    SLessThanEqual = 179,
    FOrdEqual = 180,
    FUnordEqual = 181,
    FOrdNotEqual = 182,
    FUnordNotEqual = 183,
    FOrdLessThan = 184,
    FUnordLessThan = 185,
    FOrdGreaterThan = 186,
    FUnordGreaterThan = 187,
    FOrdLessThanEqual = 188,
    FUnordLessThanEqual = 189,
    FOrdGreaterThanEqual = 190,
    FUnordGreaterThanEqual = 191,
    ShiftRightLogical = 194,
    ShiftRightArithmetic = 195,
    ShiftLeftLogical = 196,
    BitwiseOr = 197,
    BitwiseXor = 198,
    BitwiseAnd = 199,
    Not = 200,
    BitFieldInsert = 201,
    BitFieldSExtract = 202,
    BitFieldUExtract = 203,
    BitReverse = 204,
    BitCount = 205,
    EmitVertex = 218,
    EndPrimitive = 219,
    EmitStreamVertex = 220,
    EndStreamPrimitive = 221,
    ControlBarrier = 224,
    MemoryBarrier = 225,
    AtomicLoad = 227,
    AtomicExchange = 229,
    AtomicCompareExchange = 230,
    AtomicCompareExchangeWeak = 231,
    AtomicIIncrement = 232,
    AtomicIDecrement = 233,
    AtomicIAdd = 234,
    AtomicISub = 235,
    AtomicSMin = 236,
    AtomicUMin = 237,
    AtomicSMax = 238,
    AtomicUMax = 239,
    AtomicAnd = 240,
    AtomicOr = 241,
    AtomicXor = 242,
    Phi = 245,
    LoopMerge = 246,
    SelectionMerge = 247,
    Label = 248,
    Branch = 249,
    BranchConditional = 250,
    Switch = 251,
    Kill = 252,
    Return = 253,
    ReturnValue = 254,
    Unreachable = 255,
    NoLine = 317,
    ModuleProcessed = 330,
    GroupNonUniformElect = 333,
    GroupNonUniformAll = 334,
    GroupNonUniformAny = 335,
    GroupNonUniformAllEqual = 336,
    GroupNonUniformBroadcast = 337,
    GroupNonUniformBroadcastFirst = 338,
    GroupNonUniformBallot = 339,
    GroupNonUniformInverseBallot = 340,
    GroupNonUniformBallotBitExtract = 341,
    GroupNonUniformBallotBitCount = 342,
    GroupNonUniformBallotFindLSB = 343,
    GroupNonUniformBallotFindMSB = 344,
    GroupNonUniformShuffle = 345,
    GroupNonUniformShuffleXor = 346,
    GroupNonUniformShuffleUp = 347,
    GroupNonUniformShuffleDown = 348,
    GroupNonUniformIAdd = 349,
    GroupNonUniformFAdd = 350,
    GroupNonUniformIMul = 351,
    GroupNonUniformFMul = 352,
    GroupNonUniformSMin = 353,
    GroupNonUniformUMin = 354,
    GroupNonUniformFMin = 355,
    GroupNonUniformSMax = 356,
    GroupNonUniformUMax = 357,
    GroupNonUniformFMax = 358,
    GroupNonUniformBitwiseAnd = 359,
    GroupNonUniformBitwiseOr = 360,
    GroupNonUniformBitwiseXor = 361,
    GroupNonUniformLogicalAnd = 362,
    GroupNonUniformLogicalOr = 363,
    GroupNonUniformLogicalXor = 364,
    GroupNonUniformQuadBroadcast = 365,
    GroupNonUniformQuadSwap = 366,
    DemoteToHelperInvocation = 5380,
    IsHelperInvocationEXT = 5381,
}

#[cfg(test)]
mod tests {
    use super::Op;

    #[test]
    fn every_wire_opcode_round_trips_through_its_concrete_identity() {
        for number in u16::MIN..=u16::MAX {
            assert_eq!(Op::of(u32::from(number)).number(), number);
        }
    }
}
