//! The incoming edge: one raw 64-bit word, read once, becomes the
//! instruction its bits form. The opcode field selects the family through
//! a table built before runtime; each family's remaining fields select its
//! constructor through a form table, also built before runtime.

use crate::ir::{ops, Instruction, InstructionKind, Predicate};

/// One raw instruction word and the address it was read from.
#[derive(Clone, Copy)]
pub(super) struct Word {
    pub(super) r: u64,
    pub(super) w: [u32; 2],
    pub(super) address: u32,
}

/// A constructor: the raw word becomes one instruction.
pub(super) type Decode = fn(Word) -> Instruction;

impl Word {
    pub(super) fn new(w: [u32; 2], address: u32) -> Self {
        Self {
            r: w[0] as u64 | ((w[1] as u64) << 32),
            w,
            address,
        }
    }
    /// The instruction under the word's guard predicate. The guard is bits
    /// 16..19 for every guarded instruction, KILL included (KILL's bits
    /// 0..5 are its condition-code test, sm50.rs:3167-3175). PT is true, so
    /// inverted PT is a false guard. Preserve inversion.
    /// Mesa sm50.rs:225-235; envytools gm107.c:363-364,371,408-411;
    /// Ryujinx InstDecoders.cs:2685-2686 and Instructions/InstEmitHelper.cs:203-212.
    pub(super) fn guarded(self, kind: InstructionKind) -> Instruction {
        Instruction {
            address: self.address,
            predicate: GUARD[super::b(self.r, 16, 4) as usize],
            kind,
        }
    }
    /// The instruction of an encoding without a guard predicate field.
    pub(super) fn unguarded(self, kind: InstructionKind) -> Instruction {
        Instruction {
            address: self.address,
            predicate: GUARD[7],
            kind,
        }
    }
}

/// The guard formed by predicate register bits 16..18 and inversion bit 19.
pub(super) const GUARD: [Option<Predicate>; 16] = {
    let mut t = [Option::<Predicate>::None; 16];
    let mut i = 0;
    while i < 16 {
        t[i] = Option::Some(Predicate {
            register: (i & 7) as u8,
            inverted: i > 7,
        });
        i += 1;
    }
    t[7] = Option::None;
    t
};

/// An encoding no modeled instruction has: it stays the raw word.
pub(super) fn unknown(x: Word) -> Instruction {
    x.guarded(InstructionKind::Unknown(ops::Unknown { raw: x.w }))
}

/// One raw field as a form table reads it: every value (`bits`), only
/// whether it is zero (`zero`), or only whether it is all ones (`ones`).
#[derive(Clone, Copy)]
pub(super) struct Field {
    start: u32,
    mask: u64,
    flip: u64,
    cap: u64,
    width: u32,
    spread: u64,
}

impl Field {
    /// Every value of the `width`-bit field at `start`.
    pub(super) const fn bits(start: u32, width: u32) -> Self {
        let mask = (1u64 << width) - 1;
        Self {
            start,
            mask,
            flip: 0,
            cap: mask,
            width,
            spread: 1,
        }
    }
    /// Whether the `width`-bit field at `start` is zero (index 0) or not.
    pub(super) const fn zero(start: u32, width: u32) -> Self {
        Self {
            start,
            mask: (1u64 << width) - 1,
            flip: 0,
            cap: 1,
            width: 1,
            spread: 1,
        }
    }
    /// Whether the `width`-bit field at `start` is all ones (index 0) or not.
    pub(super) const fn ones(start: u32, width: u32) -> Self {
        let mask = (1u64 << width) - 1;
        Self {
            start,
            mask,
            flip: mask,
            cap: 1,
            width: 1,
            spread: mask,
        }
    }
    /// The `width`-bit field at `start` read up to `cap`: every value
    /// below `cap`, and `cap` itself for every value from `cap` up.
    pub(super) const fn capped(start: u32, width: u32, cap: u64) -> Self {
        Self {
            start,
            mask: (1u64 << width) - 1,
            flip: 0,
            cap,
            width: u64::BITS - cap.leading_zeros(),
            spread: 1,
        }
    }
    fn read(self, r: u64) -> usize {
        ((((r >> self.start) & self.mask) ^ self.flip).min(self.cap)) as usize
    }
    /// A word whose field reads as `v`.
    const fn deposit(self, v: u64) -> u64 {
        ((v * self.spread) ^ self.flip) << self.start
    }
}

/// The fields a form table is indexed by, lowest index bits first.
pub(super) struct Fields<const N: usize>(pub(super) [Field; N]);

impl<const N: usize> Fields<N> {
    /// The number of index values: one table entry each.
    pub(super) const fn len(&self) -> usize {
        let mut width = 0;
        let mut k = 0;
        while k < N {
            width += self.0[k].width;
            k += 1;
        }
        1 << width
    }
    /// A word whose fields read as index `i`.
    pub(super) const fn deposit(&self, i: usize) -> u64 {
        let mut r = 0;
        let mut shift = 0;
        let mut k = 0;
        while k < N {
            let f = self.0[k];
            r |= f.deposit((i as u64 >> shift) & ((1 << f.width) - 1));
            shift += f.width;
            k += 1;
        }
        r
    }
    /// The table index the word's fields form.
    pub(super) fn index(&self, r: u64) -> usize {
        self.0
            .iter()
            .fold((0, 0), |(i, shift), f| {
                (i | f.read(r) << shift, shift + f.width)
            })
            .0
    }
}

/// A form table over `$fields`: entry `i` is `$rule` of a word whose fields
/// read as `i`, evaluated before runtime.
macro_rules! forms {
    ($fields:path, $rule:path) => {{
        let mut t = [0u8; $fields.len()];
        let mut i = 0;
        while i < t.len() {
            t[i] = $rule($fields.deposit(i));
            i += 1;
        }
        t
    }};
}
pub(super) use forms;
