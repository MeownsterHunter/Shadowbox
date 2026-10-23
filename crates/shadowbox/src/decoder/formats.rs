//! Instruction formats learned from the disassembler, as a decoder of last
//! resort. A word the hand-written families leave unknown may still be one
//! whose format is known: which bits are its opcode, which are register,
//! predicate, immediate, modifier and flag fields, and how each prints.
//!
//! The formats come from a table of bit-flip inference results
//! (`tests/data/decoder-formats.tsv`); `tools/generate-decoder-formats`
//! turns the rows that pass the proof gate into [`table`]. A variant is one
//! known word of an operand form, its template: the variant's mask holds its
//! fixed bits, every other field is free, and a word with the same fixed bits
//! prints as the template with the free fields' values put in. A recognised
//! word has a format and its exact disassembly, not semantics: translation
//! still stops at it.

#[rustfmt::skip]
mod table;

pub use table::TABLE;

/// One contiguous run of a field's bits, `lo..=hi`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Part {
    pub lo: u8,
    pub hi: u8,
}

/// A field's bits: its parts in value order, least significant part first.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bits<'a>(pub &'a [Part]);

impl Bits<'_> {
    pub fn value(self, word: u64) -> u64 {
        let mut value = 0;
        let mut shift = 0;
        for part in self.0 {
            let width = u32::from(part.hi - part.lo) + 1;
            value |= ((word >> part.lo) & ((1u64 << width) - 1)) << shift;
            shift += width;
        }
        value
    }

    pub fn width(self) -> u32 {
        self.0.iter().map(|p| u32::from(p.hi - p.lo) + 1).sum()
    }
}

/// Which atom of which operand a field writes. Operand 0 is the guard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slot {
    pub operand: u8,
    pub atom: u8,
}

/// The register file a register field indexes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RegisterFile {
    /// `R0`..`R254`, `RZ` = 255.
    General,
    /// `P0`..`P6`, `PT` = 7.
    Predicate,
}

impl RegisterFile {
    /// The number of the file's zero register (`RZ`, `PT`).
    pub fn zero(self) -> u8 {
        match self {
            RegisterFile::General => 255,
            RegisterFile::Predicate => 7,
        }
    }

    /// `atom` holding `value` instead, when it is an atom of this file.
    fn write<'a>(self, atom: Atom<'a>, value: u8) -> Option<Atom<'a>> {
        match (self, atom) {
            (RegisterFile::General, Atom::Register(_)) => Some(Atom::Register(value)),
            (RegisterFile::Predicate, Atom::Predicate(_)) => Some(Atom::Predicate(value)),
            _ => None,
        }
    }
}

/// How an immediate field's raw value becomes the printed number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sign {
    Plain,
    TwosComplement,
}

/// One component of a rule's key: a field's value, or only whether that
/// value is zero (an immediate a refusal asks to be zero).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selector<'a> {
    Value(Bits<'a>),
    Nonzero(Bits<'a>),
}

impl Selector<'_> {
    /// The component's value in `word`: the field's value, or 1 when it is
    /// nonzero and 0 when it is zero.
    pub fn read(self, word: u64) -> u64 {
        match self {
            Selector::Value(bits) => bits.value(word),
            Selector::Nonzero(bits) => u64::from(bits.value(word) != 0),
        }
    }
}

/// A result decided by the joint value of selector bits. Key values never
/// seen in the inference have no entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Keyed<'a, T> {
    pub key: &'a [Selector<'a>],
    pub table: &'a [(&'a [u64], T)],
}

impl<'a, T: Copy> Keyed<'a, T> {
    pub fn get(&self, word: u64) -> Option<T> {
        self.table
            .iter()
            .find(|(values, _)| {
                values.len() == self.key.len()
                    && self
                        .key
                        .iter()
                        .zip(*values)
                        .all(|(s, v)| s.read(word) == *v)
            })
            .map(|&(_, result)| result)
    }

    /// Whether `word` and `base` hold the same key value.
    pub fn same(&self, word: u64, base: u64) -> bool {
        self.key.iter().all(|s| s.read(word) == s.read(base))
    }
}

/// A keyed rule with the result it gives for key values never seen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ruled<'a, T> {
    pub rule: Keyed<'a, T>,
    pub unseen: T,
}

impl<T: Copy> Ruled<'_, T> {
    pub fn get(&self, word: u64) -> T {
        self.rule.get(word).unwrap_or(self.unseen)
    }
}

/// An immediate's signedness: fixed, or decided by selector values with a
/// fallback for values the inference never saw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImmediateSign<'a> {
    Fixed(Sign),
    Keyed(Ruled<'a, Sign>),
}

/// Whether an immediate prints its value or a zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Printed {
    Value,
    Zero,
}

/// An immediate the disassembler prints as `0x0` under some selector values
/// (a bound texture's index, printed 0x0 when bindless).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Zeroing<'a> {
    Never,
    Keyed(Ruled<'a, Printed>),
}

/// Whether an immediate prints as minus its value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Negation<'a> {
    Never,
    Keyed(Ruled<'a, bool>),
}

/// What each of an immediate's bits weighs, in value order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scale<'a> {
    /// Bit `k` weighs `2^(k + shift)`.
    Shift(u8),
    /// Bit `k` weighs `2^exponents[k]`.
    Weights(&'a [u8]),
}

/// The IEEE widths the disassembler prints floats in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatWidth {
    Half,
    Single,
    Double,
}

impl FloatWidth {
    pub fn bits(self) -> u32 {
        match self {
            FloatWidth::Half => 16,
            FloatWidth::Single => 32,
            FloatWidth::Double => 64,
        }
    }
}

/// How a float field's value makes an IEEE number: the field is its bits
/// from `shift` up (a negative shift: the field's own bits from `-shift`
/// up), and `Flipped` prints it negated (the sign bit turned).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FloatEncoding {
    pub width: FloatWidth,
    pub shift: i8,
    pub sign: FloatSign,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FloatSign {
    AsEncoded,
    Flipped,
}

impl FloatEncoding {
    /// The IEEE number field value `raw` makes.
    pub fn real(self, raw: u64) -> Real {
        let width = self.width.bits();
        let shifted = match u32::try_from(self.shift) {
            Ok(up) => u128::from(raw) << up,
            Err(_) => u128::from(raw) >> self.shift.unsigned_abs(),
        };
        let bits = (shifted & ((1u128 << width) - 1)) as u64;
        Real {
            width: self.width,
            bits: match self.sign {
                FloatSign::AsEncoded => bits,
                FloatSign::Flipped => bits ^ (1 << (width - 1)),
            },
        }
    }
}

/// Whether the disassembler accepts a word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Validity {
    Accepted,
    Refused,
}

/// The words of a format the disassembler rejects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal<'a> {
    Never,
    Keyed(Ruled<'a, Validity>),
    BackedOff(Backoff<'a>),
}

impl Refusal<'_> {
    pub fn get(&self, word: u64) -> Validity {
        match self {
            Refusal::Never => Validity::Accepted,
            Refusal::Keyed(rule) => rule.get(word),
            Refusal::BackedOff(backoff) => backoff.get(word),
        }
    }
}

/// A refusal whose key values never seen back off to its components: a
/// component value that only ever showed one result decides alone (a
/// refusal that follows either of two selectors). With none deciding, the
/// words where no component decided give `rest`; with two disagreeing,
/// `unseen`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Backoff<'a> {
    pub rule: Keyed<'a, Validity>,
    /// Per key component, the values that decide alone.
    pub alone: &'a [&'a [(u64, Validity)]],
    pub rest: Validity,
    pub unseen: Validity,
}

impl Backoff<'_> {
    pub fn get(&self, word: u64) -> Validity {
        if let Some(seen) = self.rule.get(word) {
            return seen;
        }
        let mut decided = self
            .rule
            .key
            .iter()
            .zip(self.alone)
            .filter_map(|(selector, values)| {
                let value = selector.read(word);
                values.iter().find(|(v, _)| *v == value).map(|&(_, r)| r)
            });
        match decided.next() {
            None => self.rest,
            Some(first) if decided.all(|r| r == first) => first,
            Some(_) => self.unseen,
        }
    }
}

/// How a register atom follows another register field: that field's value
/// plus an offset (the second register of a pair), or the zero register.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Derivation {
    Offset(i16),
    Zero,
}

/// A register atom printed from a register field, as selector values decide.
/// Key values never seen take `unseen`, or leave the atom when it is `None`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Derived<'a> {
    pub slot: Slot,
    pub rule: Keyed<'a, Derivation>,
    pub unseen: Option<Derivation>,
}

/// A modifier no field owns, printed where its selector key says so; key
/// values never seen leave the template's modifiers as they are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Token<'a> {
    pub name: &'a str,
    pub rule: Keyed<'a, bool>,
}

/// One free field of a format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Field<'a> {
    /// A register field, printed xor `xor` (a predicate written as PT for
    /// 0) at its first slot and again at the others (a destination read
    /// back as a source); the `derived` atoms print from its value too.
    Register {
        bits: Bits<'a>,
        file: RegisterFile,
        slot: Slot,
        again: &'a [Slot],
        xor: u8,
        derived: &'a [Derived<'a>],
    },
    /// An integer immediate: the sum of its set bits' weights (the top bit
    /// negative when two's complement), negated where its negation says,
    /// plus `bias`; or zero where its zeroing says.
    Immediate {
        bits: Bits<'a>,
        slot: Slot,
        scale: Scale<'a>,
        bias: i64,
        sign: ImmediateSign<'a>,
        negation: Negation<'a>,
        zero: Zeroing<'a>,
    },
    /// A float immediate, printed from its bits in the encoding its
    /// selectors name.
    Float {
        bits: Bits<'a>,
        slot: Slot,
        encoding: Ruled<'a, FloatEncoding>,
    },
    /// A modifier field: each valid value prints these suffix names.
    Modifier {
        bits: Bits<'a>,
        /// Every name this field prints for some value.
        names: &'a [&'a str],
        values: &'a [(u64, &'a [&'a str])],
    },
    /// Bits that turn an operand's shape into another (`-`, `|..|`, `!`, `~`).
    Flag { bits: Bits<'a>, operand: u8 },
    /// Bits whose value does not change the text.
    Ignored { bits: Bits<'a> },
}

/// A manifest row: what every form of it shares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Row<'a> {
    /// The manifest line (`tests/data/decoder-rows.tsv`).
    pub line: u16,
    /// How `[RZ+0x0]` prints in this row's memory operands.
    pub zero: &'a str,
    /// How often modifier `.0` was printed before `.1`.
    pub order: &'a [(&'a str, &'a str, u32)],
    /// The same counts over the rows printing the same mnemonic, for pairs
    /// this row never printed together.
    pub order_other: &'a [(&'a str, &'a str, u32)],
    /// Flipping `bit` turns operand `operand`'s shape `from` into `to`.
    pub flags: &'a [FlagTurn<'a>],
    /// Operands the disassembler leaves out at one value, in the order they
    /// are tried.
    pub omit: &'a [Omission<'a>],
}

/// An operand left out when it holds its default: in a text of `count`
/// operands, operand `operand` of shape `shape` whose single atom is `value`
/// (RZ 255, PT 7, a number) is not printed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Omission<'a> {
    pub operand: u8,
    pub shape: &'a str,
    pub count: u8,
    pub value: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlagTurn<'a> {
    pub operand: u8,
    pub from: &'a str,
    pub bit: u8,
    pub to: &'a str,
}

/// The operand-form bits a format was learned under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OperandForm {
    /// The row has a single operand form.
    Only,
    /// The row's form bits hold this value.
    Selected(u64),
}

/// One operand form of a row: its free fields and the operand shapes its
/// selector values decide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Format<'a> {
    /// Index into [`Table::rows`].
    pub row: u16,
    pub form: OperandForm,
    pub fields: &'a [Field<'a>],
    pub shapes: &'a [(u8, Keyed<'a, &'a str>)],
    pub tokens: &'a [Token<'a>],
    pub refusal: Refusal<'a>,
}

/// How a number atom prints.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Radix {
    Hex,
    Decimal,
}

/// One value inside an operand. Shapes mark each with a placeholder:
/// `\u{1}` register, `\u{2}` predicate, `\u{3}` number.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Atom<'a> {
    Register(u8),
    Predicate(u8),
    Number {
        value: i64,
        radix: Radix,
    },
    /// A float as printed in the template.
    Float(&'a str),
    /// A float a field wrote.
    Real(Real),
}

/// An IEEE number of one of the printed widths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Real {
    pub width: FloatWidth,
    pub bits: u64,
}

impl Real {
    fn value(self) -> f64 {
        match self.width {
            FloatWidth::Half => half(self.bits as u16),
            FloatWidth::Single => f64::from(f32::from_bits(self.bits as u32)),
            FloatWidth::Double => f64::from_bits(self.bits),
        }
    }

    /// The disassembler's text: +INF/-INF, +QNAN/-QNAN (halves and singles
    /// print widened to double, which makes every NaN quiet), -0.0, plain
    /// decimal of 20 significant digits from 1e-4 up to 1e9, else 20
    /// significant digits in exponent form below one (trailing zeros cut)
    /// and 21 above.
    pub fn text(self) -> String {
        let width = self.width.bits();
        let x = self.value();
        let sign = match (self.bits >> (width - 1)) & 1 {
            1 => "-",
            _ => "+",
        };
        if x.is_infinite() {
            return format!("{sign}INF");
        }
        if x.is_nan() {
            let quiet = match self.width {
                FloatWidth::Double => (self.bits >> 51) & 1 == 1,
                FloatWidth::Half | FloatWidth::Single => true,
            };
            return format!("{sign}{}", if quiet { "QNAN" } else { "SNAN" });
        }
        if x == 0.0 {
            return if sign == "-" { "-0.0" } else { "0" }.to_owned();
        }
        let e = x.abs().log10().floor() as i32;
        if (-4..9).contains(&e) {
            let (digits, exp, negative) = significant(x, 19);
            let mut t = String::from(if negative { "-" } else { "" });
            if exp >= 0 {
                let point = (exp + 1) as usize;
                let padded = format!("{digits:0<point$}");
                t.push_str(&padded[..point]);
                if padded.len() > point {
                    t.push('.');
                    t.push_str(&padded[point..]);
                }
            } else {
                t.push_str("0.");
                t.push_str(&"0".repeat((-exp - 1) as usize));
                t.push_str(&digits);
            }
            if t.contains('.') {
                t = t.trim_end_matches('0').trim_end_matches('.').to_owned();
            }
            return t;
        }
        let places = if e < 0 { 19 } else { 20 };
        let (digits, exp, negative) = significant(x, places);
        let mut mantissa = format!(
            "{}{}.{}",
            if negative { "-" } else { "" },
            &digits[..1],
            &digits[1..]
        );
        if e < 0 {
            mantissa = mantissa
                .trim_end_matches('0')
                .trim_end_matches('.')
                .to_owned();
        }
        let exp_sign = if exp < 0 { '-' } else { '+' };
        format!("{mantissa}e{exp_sign}{:02}", exp.unsigned_abs())
    }
}

/// `x` rounded to `places + 1` significant digits: the digits, the decimal
/// exponent of the first, and whether it is negative.
fn significant(x: f64, places: usize) -> (String, i32, bool) {
    let text = format!("{:.places$e}", x.abs());
    let (mantissa, exp) = text.split_once('e').expect("exponent form");
    (
        mantissa.replace('.', ""),
        exp.parse().expect("an exponent"),
        x < 0.0,
    )
}

/// An IEEE half as the double it widens to.
fn half(bits: u16) -> f64 {
    let sign = if bits >> 15 == 1 { -1.0 } else { 1.0 };
    let exponent = i32::from((bits >> 10) & 0x1f);
    let fraction = f64::from(bits & 0x3ff);
    sign * match exponent {
        0 => fraction * 2f64.powi(-24),
        0x1f if fraction == 0.0 => f64::INFINITY,
        0x1f => f64::NAN,
        _ => (1.0 + fraction / 1024.0) * 2f64.powi(exponent - 15),
    }
}

/// Floats the disassembler pads with a space before what follows.
fn padded(text: &str) -> bool {
    matches!(
        text,
        "+INF" | "-INF" | "+QNAN" | "-QNAN" | "+SNAN" | "-SNAN" | "-0.0"
    )
}

const REGISTER: char = '\u{1}';
const PREDICATE: char = '\u{2}';
const NUMBER: char = '\u{3}';

impl Atom<'_> {
    /// The value an omission compares: a register's or predicate's number,
    /// or an integer.
    fn integer(self) -> Option<i64> {
        match self {
            Atom::Register(r) | Atom::Predicate(r) => Some(i64::from(r)),
            Atom::Number { value, .. } => Some(value),
            Atom::Float(_) | Atom::Real(_) => None,
        }
    }

    fn placeholder(self) -> char {
        match self {
            Atom::Register(_) => REGISTER,
            Atom::Predicate(_) => PREDICATE,
            Atom::Number { .. } | Atom::Float(_) | Atom::Real(_) => NUMBER,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Operand<'a> {
    pub shape: &'a str,
    pub atoms: &'a [Atom<'a>],
}

/// Whether the first operand follows the mnemonic unspaced (`CCTL.QRY1[R0]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spacing {
    Spaced,
    Glued,
}

/// A variant: a format's template word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Variant<'a> {
    pub mask: u64,
    pub value: u64,
    /// Index into [`Table::formats`].
    pub format: u16,
    /// The template word whose text the variant prints from.
    pub base: u64,
    pub spacing: Spacing,
    pub mnemonic: &'a str,
    pub modifiers: &'a [&'a str],
    /// Operand 0 is the guard.
    pub operands: &'a [Operand<'a>],
}

/// Every admitted variant, and per top-16-bit prefix the variants whose
/// fixed bits allow it, in table order.
#[derive(Clone, Copy, Debug)]
pub struct Table<'a> {
    pub rows: &'a [Row<'a>],
    pub formats: &'a [Format<'a>],
    pub variants: &'a [Variant<'a>],
    pub prefixes: &'a [(u16, &'a [u32])],
}

impl<'a> Table<'a> {
    /// The first variant whose fixed bits match `word` and which prints it.
    pub fn recognise(&self, word: u64) -> Option<u32> {
        let prefix = (word >> 48) as u16;
        let at = self
            .prefixes
            .binary_search_by_key(&prefix, |&(p, _)| p)
            .ok()?;
        self.prefixes[at]
            .1
            .iter()
            .copied()
            .find(|&id| self.text(id, word).is_some())
    }

    /// The disassembly of `word` as variant `id` prints it, when its fixed
    /// bits match, the disassembler accepts it and every changed field value
    /// is one the variant knows.
    pub fn text(&self, id: u32, word: u64) -> Option<String> {
        let variant = &self.variants[id as usize];
        if word & variant.mask != variant.value {
            return None;
        }
        let format = &self.formats[usize::from(variant.format)];
        if format.refusal.get(word) == Validity::Refused {
            return None;
        }
        let row = &self.rows[usize::from(format.row)];
        let base = variant.base;
        let mut shapes: Vec<&str> = variant.operands.iter().map(|o| o.shape).collect();
        let mut atoms: Vec<Vec<Atom>> = variant.operands.iter().map(|o| o.atoms.to_vec()).collect();
        let owned = |name: &&str| {
            format.fields.iter().any(|f| match f {
                Field::Modifier { names, .. } => names.contains(name),
                _ => false,
            })
        };
        let mut chains: Vec<Vec<&str>> = vec![variant
            .modifiers
            .iter()
            .copied()
            .filter(|m| !owned(m))
            .collect()];
        for field in format.fields {
            match *field {
                Field::Modifier {
                    bits,
                    names,
                    values,
                } => {
                    let new = bits.value(word);
                    if new == bits.value(base) {
                        chains.push(
                            variant
                                .modifiers
                                .iter()
                                .copied()
                                .filter(|m| names.contains(m))
                                .collect(),
                        );
                    } else {
                        let (_, printed) = values.iter().find(|(v, _)| *v == new)?;
                        chains.push(printed.to_vec());
                    }
                }
                Field::Register {
                    bits,
                    file,
                    slot,
                    again,
                    xor,
                    ..
                } => {
                    let new = bits.value(word);
                    if new == bits.value(base) {
                        continue;
                    }
                    let new = u8::try_from(new ^ u64::from(xor)).ok()?;
                    let atom = slot_atom(&mut atoms, slot)?;
                    *atom = file.write(*atom, new)?;
                    for &slot in again {
                        if let Some(atom) = slot_atom(&mut atoms, slot) {
                            if let Some(new) = file.write(*atom, new) {
                                *atom = new;
                            }
                        }
                    }
                }
                Field::Immediate {
                    bits,
                    slot,
                    scale,
                    bias,
                    sign,
                    negation,
                    zero,
                } => {
                    if let Zeroing::Keyed(rule) = zero {
                        if rule.get(word) == Printed::Zero {
                            let atom = slot_atom(&mut atoms, slot)?;
                            let Atom::Number { radix, .. } = *atom else {
                                return None;
                            };
                            *atom = Atom::Number { value: 0, radix };
                            continue;
                        }
                    }
                    // A zeroing immediate prints its value again on every word.
                    let unchanged = zero == Zeroing::Never
                        && bits.value(word) == bits.value(base)
                        && match sign {
                            ImmediateSign::Fixed(_) => true,
                            ImmediateSign::Keyed(r) => r.rule.same(word, base),
                        }
                        && match negation {
                            Negation::Never => true,
                            Negation::Keyed(r) => r.rule.same(word, base),
                        };
                    if unchanged {
                        continue;
                    }
                    let atom = slot_atom(&mut atoms, slot)?;
                    let Atom::Number { radix, .. } = *atom else {
                        return None;
                    };
                    let sign = match sign {
                        ImmediateSign::Fixed(sign) => sign,
                        ImmediateSign::Keyed(rule) => rule.get(word),
                    };
                    let negated = match negation {
                        Negation::Never => false,
                        Negation::Keyed(rule) => rule.get(word),
                    };
                    let mut value = immediate(bits, scale, sign, word);
                    if negated {
                        value = -value;
                    }
                    *atom = Atom::Number {
                        value: value + bias,
                        radix,
                    };
                }
                Field::Float {
                    bits,
                    slot,
                    encoding,
                } => {
                    let atom = slot_atom(&mut atoms, slot)?;
                    let (Atom::Number { .. } | Atom::Float(_) | Atom::Real(_)) = *atom else {
                        return None;
                    };
                    *atom = Atom::Real(encoding.get(word).real(bits.value(word)));
                }
                Field::Flag { bits, operand } => {
                    let Part { lo, hi } = bits.0[0];
                    let changed = (word ^ base) >> lo;
                    for bit in lo..=hi {
                        if (changed >> (bit - lo)) & 1 == 0 {
                            continue;
                        }
                        let shape = shapes.get_mut(usize::from(operand))?;
                        let turn = row
                            .flags
                            .iter()
                            .find(|t| t.operand == operand && t.bit == bit && t.from == *shape)?;
                        *shape = turn.to;
                    }
                }
                Field::Ignored { .. } => {}
            }
        }
        for field in format.fields {
            let Field::Register {
                bits,
                file,
                xor,
                derived,
                ..
            } = *field
            else {
                continue;
            };
            let zero = file.zero();
            let own = bits.value(word) ^ u64::from(xor);
            for d in derived {
                let Some(derivation) = d.rule.get(word).or(d.unseen) else {
                    continue;
                };
                let Some(atom) = slot_atom(&mut atoms, d.slot) else {
                    continue;
                };
                let value = match derivation {
                    Derivation::Zero => u64::from(zero),
                    _ if own == u64::from(zero) => own,
                    Derivation::Offset(offset) => own.checked_add_signed(i64::from(offset))?,
                };
                if let Some(new) = file.write(*atom, u8::try_from(value).ok()?) {
                    *atom = new;
                }
            }
        }
        for &(operand, rule) in format.shapes {
            let operand = usize::from(operand);
            let (Some(shape), Some(slot)) = (rule.get(word), atoms.get(operand)) else {
                continue;
            };
            if shape
                .chars()
                .filter(|c| [REGISTER, PREDICATE, NUMBER].contains(c))
                .eq(slot.iter().map(|a| a.placeholder()))
            {
                shapes[operand] = shape;
            }
        }
        for token in format.tokens {
            let Some(printed) = token.rule.get(word) else {
                continue;
            };
            chains[0].retain(|m| *m != token.name);
            if printed {
                chains.push(vec![token.name]);
            }
        }
        let modifiers = order(chains, row.order, row.order_other);
        for omission in row.omit {
            let operand = usize::from(omission.operand);
            if shapes.len() == usize::from(omission.count)
                && shapes.get(operand) == Some(&omission.shape)
                && matches!(atoms[operand].as_slice(), [atom] if atom.integer() == Some(omission.value))
            {
                shapes.remove(operand);
                atoms.remove(operand);
            }
        }
        Some(print(variant, row.zero, &shapes, &atoms, &modifiers))
    }
}

fn slot_atom<'v, 'a>(atoms: &'v mut [Vec<Atom<'a>>], slot: Slot) -> Option<&'v mut Atom<'a>> {
    atoms
        .get_mut(usize::from(slot.operand))?
        .get_mut(usize::from(slot.atom))
}

/// An immediate's value: its set bits' weights summed, the top bit's
/// negative under two's complement.
fn immediate(bits: Bits, scale: Scale, sign: Sign, word: u64) -> i64 {
    let raw = bits.value(word);
    let width = bits.width();
    (0..width)
        .filter(|&k| (raw >> k) & 1 == 1)
        .map(|k| {
            let exponent = match scale {
                Scale::Shift(shift) => k + u32::from(shift),
                Scale::Weights(exponents) => u32::from(exponents[k as usize]),
            };
            let weight = 1i128 << exponent;
            match (sign, k + 1 == width) {
                (Sign::TwosComplement, true) => -weight,
                _ => weight,
            }
        })
        .sum::<i128>() as i64
}

/// Merges suffix chains: each chain keeps its own order, and the row's
/// pairwise counts order the chains' heads against each other; a pair the
/// row never printed together takes the counts of the rows printing the same
/// mnemonic (`other`).
fn order<'a>(
    chains: Vec<Vec<&'a str>>,
    before: &[(&str, &str, u32)],
    other: &[(&str, &str, u32)],
) -> Vec<&'a str> {
    let in_table = |table: &[(&str, &str, u32)], a: &str, b: &str| {
        table
            .iter()
            .find(|(x, y, _)| *x == a && *y == b)
            .map_or(0, |&(_, _, n)| n)
    };
    let count = |a: &str, b: &str| match in_table(before, a, b) + in_table(before, b, a) {
        0 => in_table(other, a, b),
        _ => in_table(before, a, b),
    };
    let mut chains: Vec<std::collections::VecDeque<&str>> = chains
        .into_iter()
        .filter(|c| !c.is_empty())
        .map(Into::into)
        .collect();
    let mut out = Vec::new();
    while !chains.is_empty() {
        let heads: Vec<&str> = chains.iter().map(|c| c[0]).collect();
        let first = heads
            .iter()
            .position(|&x| {
                heads
                    .iter()
                    .filter(|&&y| y != x)
                    .all(|&y| count(y, x) <= count(x, y))
            })
            .unwrap_or(0);
        out.extend(chains[first].pop_front());
        chains.retain(|c| !c.is_empty());
    }
    out
}

fn print(
    variant: &Variant,
    zero: &str,
    shapes: &[&str],
    atoms: &[Vec<Atom>],
    modifiers: &[&str],
) -> String {
    let operands: Vec<String> = shapes
        .iter()
        .zip(atoms)
        .enumerate()
        .map(|(j, (shape, atoms))| {
            let mut atoms = atoms.iter();
            let mut text = String::new();
            for ch in shape.chars() {
                if ![REGISTER, PREDICATE, NUMBER].contains(&ch) {
                    text.push(ch);
                    continue;
                }
                match atoms.next() {
                    Some(Atom::Register(255)) => text.push_str("RZ"),
                    Some(Atom::Register(r)) => text.push_str(&format!("R{r}")),
                    Some(Atom::Predicate(7)) => text.push_str("PT"),
                    Some(Atom::Predicate(p)) => text.push_str(&format!("P{p}")),
                    Some(Atom::Number { value, radix }) => text.push_str(&number(*value, *radix)),
                    Some(Atom::Float(raw)) => {
                        text.push_str(raw);
                        if padded(raw) {
                            text.push(' ');
                        }
                    }
                    Some(Atom::Real(real)) => {
                        let raw = real.text();
                        text.push_str(&raw);
                        if padded(&raw) {
                            text.push(' ');
                        }
                    }
                    None => text.push(ch),
                }
            }
            match j {
                0 => text,
                _ => brackets(&text, zero),
            }
        })
        .collect();
    let mut head = variant.mnemonic.to_owned();
    for m in modifiers {
        head.push('.');
        head.push_str(m);
    }
    let guard = match operands[0].as_str() {
        "@PT" => String::new(),
        g => format!("{g} "),
    };
    let line = match (variant.spacing, &operands[1..]) {
        (_, []) => format!("{guard}{head}"),
        (Spacing::Glued, [first]) => format!("{guard}{head}{first}"),
        (Spacing::Glued, [first, rest @ ..]) => format!("{guard}{head}{first} {}", rest.join(", ")),
        (Spacing::Spaced, all) => format!("{guard}{head} {}", all.join(", ")),
    };
    line.trim_end().to_owned()
}

fn number(value: i64, radix: Radix) -> String {
    match radix {
        Radix::Hex if value < 0 => format!("-0x{:x}", value.unsigned_abs()),
        Radix::Hex => format!("0x{value:x}"),
        Radix::Decimal => value.to_string(),
    }
}

/// Memory brackets print without a zero offset or a zero register: the
/// canonical `[Rn+off]` of the templates back to the disassembler's text.
fn brackets(text: &str, zero: &str) -> String {
    // A bare [RZ+0x0] prints as the row prints it; after `c` or `]` it is a
    // constant-buffer offset and prints as a number.
    let mut first = String::new();
    let mut rest = text;
    let mut prev = None;
    while let Some(at) = rest.find("[RZ+0x0]") {
        let before = rest[..at].chars().next_back().or(prev);
        first.push_str(&rest[..at]);
        match before {
            Some('c' | ']') => first.push_str("[RZ+0x0]"),
            _ => {
                first.push('[');
                first.push_str(zero);
                first.push(']');
            }
        }
        prev = Some(']');
        rest = &rest[at + "[RZ+0x0]".len()..];
    }
    first.push_str(rest);
    let second = rewrite(&first, |s| {
        let inner = s.strip_prefix("[RZ+")?;
        let digits = inner
            .strip_prefix('-')
            .unwrap_or(inner)
            .strip_prefix("0x")?;
        let n = digits
            .bytes()
            .take_while(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b))
            .count();
        let end = inner.len() - digits.len() + n;
        (n > 0 && inner[end..].starts_with(']'))
            .then(|| (format!("[{}]", &inner[..end]), 4 + end + 1))
    });
    rewrite(&second, |s| {
        let inner = s.strip_prefix("[R")?;
        let n = inner.bytes().take_while(u8::is_ascii_digit).count();
        (n > 0 && inner[n..].starts_with("+0x0]"))
            .then(|| (format!("[R{}]", &inner[..n]), 2 + n + 5))
    })
}

/// Rewrites, left to right and without overlap, every place `at` matches:
/// `at` returns the replacement and how many bytes it consumed.
fn rewrite(text: &str, at: impl Fn(&str) -> Option<(String, usize)>) -> String {
    let mut out = String::new();
    let mut i = 0;
    while i < text.len() {
        match at(&text[i..]) {
            Some((replacement, used)) => {
                out.push_str(&replacement);
                i += used;
            }
            None => {
                let ch = text[i..].chars().next().expect("inside the text");
                out.push(ch);
                i += ch.len_utf8();
            }
        }
    }
    out
}
