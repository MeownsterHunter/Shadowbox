//! Reads the inferred format table (`decoder-formats/3`), keeps the variants
//! that print every held sample word exactly as the disassembler does, and
//! writes them as Rust statics for `shadowbox::decoder::formats`.
//!
//! The gate is first-match, like the decoder: a word belongs to the first
//! variant whose fixed bits match and which prints it. A variant that misprints
//! any of its words leaves, its words fall to later variants, and the gate runs
//! again until no admitted variant misprints a word.

use shadowbox::decoder::formats::{
    Atom, Backoff, Bits, Derivation, Derived, Field, FlagTurn, FloatEncoding, FloatSign,
    FloatWidth, Format, ImmediateSign, Keyed, Negation, Omission, Operand, OperandForm, Part,
    Printed, Radix, Refusal, RegisterFile, Row, Ruled, Scale, Selector, Sign, Slot, Spacing, Table,
    Token, Validity, Variant, Zeroing,
};
use std::collections::{BTreeMap, HashMap};
use std::fmt::Write;

pub const VERSION: &str = "decoder-formats/3";

fn leak<T>(v: Vec<T>) -> &'static [T] {
    Box::leak(v.into_boxed_slice())
}

fn text(s: &str) -> &'static str {
    Box::leak(s.to_owned().into_boxed_str())
}

/// Where a variant came from in the table.
#[derive(Clone, Debug)]
pub struct Label {
    pub id: u32,
    pub row: u16,
    pub form: String,
    /// The inference's own count of held words it rendered, and how many exactly.
    pub inferred: (u64, u64),
}

pub struct Parsed {
    pub rows: Vec<Row<'static>>,
    pub formats: Vec<Format<'static>>,
    pub variants: Vec<Variant<'static>>,
    pub labels: Vec<Label>,
}

fn int<T: std::str::FromStr>(s: &str, what: &str) -> Result<T, String> {
    s.parse().map_err(|_| format!("bad {what}: {s:?}"))
}

fn hex(s: &str) -> Result<u64, String> {
    u64::from_str_radix(s, 16).map_err(|_| format!("bad hex: {s:?}"))
}

fn bits(s: &str) -> Result<Bits<'static>, String> {
    let parts = s
        .split(',')
        .map(|p| {
            let (hi, lo) = p.split_once(':').ok_or(format!("bad bits: {s:?}"))?;
            Ok(Part {
                lo: int(lo, "bit")?,
                hi: int(hi, "bit")?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(Bits(leak(parts)))
}

fn slot(s: &str) -> Result<Slot, String> {
    if s == "g" {
        return Ok(Slot {
            operand: 0,
            atom: 0,
        });
    }
    let (o, a) = s.split_once('.').ok_or(format!("bad slot: {s:?}"))?;
    Ok(Slot {
        operand: int(o, "operand")?,
        atom: int(a, "atom")?,
    })
}

fn shape(s: &str) -> &'static str {
    text(
        &s.replace("\\R", "\u{1}")
            .replace("\\P", "\u{2}")
            .replace("\\N", "\u{3}"),
    )
}

fn names(s: &str) -> &'static [&'static str] {
    match s {
        "-" => &[],
        s => leak(s.split('.').map(text).collect()),
    }
}

/// A key component: `bits`, or `nz:bits` for whether their value is zero.
fn selector(s: &str) -> Result<Selector<'static>, String> {
    match s.strip_prefix("nz:") {
        Some(b) => Ok(Selector::Nonzero(bits(b)?)),
        None => Ok(Selector::Value(bits(s)?)),
    }
}

/// A keyed rule: its key (`-` for none), its `value=result` entries, and the
/// `*=result` fallback for key values never seen, when it has one.
fn keyed<T: Copy>(
    key: &str,
    entries: &[&str],
    value: impl Fn(&str) -> Result<T, String>,
) -> Result<(Keyed<'static, T>, Option<T>), String> {
    let key = match key {
        "-" => &[][..],
        key => leak(
            key.split('|')
                .map(selector)
                .collect::<Result<Vec<_>, _>>()?,
        ),
    };
    let mut unseen = None;
    let mut table = Vec::new();
    for e in entries {
        let (k, v) = e.split_once('=').ok_or(format!("bad entry: {e:?}"))?;
        if k == "*" {
            unseen = Some(value(v)?);
            continue;
        }
        let k = k
            .split(',')
            .map(|x| int(x, "key value"))
            .collect::<Result<Vec<u64>, _>>()?;
        table.push((leak(k), value(v)?));
    }
    Ok((
        Keyed {
            key,
            table: leak(table),
        },
        unseen,
    ))
}

fn sign(s: &str) -> Result<Sign, String> {
    match s {
        "plain" => Ok(Sign::Plain),
        "twos" => Ok(Sign::TwosComplement),
        _ => Err(format!("bad sign: {s:?}")),
    }
}

fn bit(s: &str) -> Result<bool, String> {
    match s {
        "0" => Ok(false),
        "1" => Ok(true),
        _ => Err(format!("bad 0/1: {s:?}")),
    }
}

fn printed(s: &str) -> Result<Printed, String> {
    match s {
        "value" => Ok(Printed::Value),
        "zero" => Ok(Printed::Zero),
        _ => Err(format!("bad printed: {s:?}")),
    }
}

/// `Z` (the zero register) or an offset from the source field's value.
fn derivation(s: &str) -> Result<Derivation, String> {
    match s {
        "Z" => Ok(Derivation::Zero),
        n => Ok(Derivation::Offset(int(n, "offset")?)),
    }
}

/// `value=result` entries.
fn values<T>(
    entries: &[&str],
    result: impl Fn(&str) -> Result<T, String>,
) -> Result<Vec<(u64, T)>, String> {
    entries
        .iter()
        .map(|e| {
            let (v, r) = e.split_once('=').ok_or(format!("bad value: {e:?}"))?;
            Ok((int(v, "value")?, result(r)?))
        })
        .collect()
}

fn validity(s: &str) -> Result<Validity, String> {
    match s {
        "ok" => Ok(Validity::Accepted),
        "bad" => Ok(Validity::Refused),
        _ => Err(format!("bad validity: {s:?}")),
    }
}

/// `<<s` or `w:e,e,...`.
fn scale(s: &str) -> Result<Scale<'static>, String> {
    if let Some(shift) = s.strip_prefix("<<") {
        return Ok(Scale::Shift(int(shift, "shift")?));
    }
    let weights = s.strip_prefix("w:").ok_or(format!("bad scale: {s:?}"))?;
    Ok(Scale::Weights(leak(
        weights
            .split(',')
            .map(|e| int(e, "weight"))
            .collect::<Result<Vec<u8>, _>>()?,
    )))
}

/// `[-]W<<s`.
fn encoding(s: &str) -> Result<FloatEncoding, String> {
    let (sign, rest) = match s.strip_prefix('-') {
        Some(rest) => (FloatSign::Flipped, rest),
        None => (FloatSign::AsEncoded, s),
    };
    let (width, shift) = rest
        .split_once("<<")
        .ok_or(format!("bad encoding: {s:?}"))?;
    Ok(FloatEncoding {
        width: match width {
            "16" => FloatWidth::Half,
            "32" => FloatWidth::Single,
            "64" => FloatWidth::Double,
            _ => return Err(format!("bad float width: {s:?}")),
        },
        shift: int(shift, "float shift")?,
        sign,
    })
}

fn file(s: &str) -> Result<RegisterFile, String> {
    match s {
        "R" => Ok(RegisterFile::General),
        "P" => Ok(RegisterFile::Predicate),
        f => Err(format!("bad register file {f:?}")),
    }
}

/// A template operand: shape text with each atom inline (`\R5;`).
fn operand(s: &str) -> Result<Operand<'static>, String> {
    let mut shape = String::new();
    let mut atoms = Vec::new();
    let mut rest = s;
    while let Some(at) = rest.find('\\') {
        shape.push_str(&rest[..at]);
        let kind = rest[at + 1..]
            .chars()
            .next()
            .ok_or(format!("bad operand: {s:?}"))?;
        let body = &rest[at + 2..];
        let end = body.find(';').ok_or(format!("bad operand: {s:?}"))?;
        let value = &body[..end];
        let (placeholder, atom) = match kind {
            'R' => ('\u{1}', Atom::Register(int(value, "register")?)),
            'P' => ('\u{2}', Atom::Predicate(int(value, "predicate")?)),
            'N' => {
                let radix = match value.chars().next() {
                    Some('x') => Radix::Hex,
                    Some('d') => Radix::Decimal,
                    _ => return Err(format!("bad number: {s:?}")),
                };
                (
                    '\u{3}',
                    Atom::Number {
                        value: int(&value[1..], "number")?,
                        radix,
                    },
                )
            }
            'F' => ('\u{3}', Atom::Float(text(value))),
            _ => return Err(format!("bad atom: {s:?}")),
        };
        shape.push(placeholder);
        atoms.push(atom);
        rest = &body[end + 1..];
    }
    shape.push_str(rest);
    Ok(Operand {
        shape: text(&shape),
        atoms: leak(atoms),
    })
}

#[derive(Default)]
struct RowParts {
    zero: String,
    order: Vec<(&'static str, &'static str, u32)>,
    order_other: Vec<(&'static str, &'static str, u32)>,
    flags: Vec<FlagTurn<'static>>,
    omit: Vec<Omission<'static>>,
}

#[derive(Default)]
struct FormatParts {
    /// Free fields by their index in the form.
    fields: Vec<(u32, Field<'static>)>,
    /// Derived register atoms by the index of the field they print from.
    derived: Vec<(u32, Derived<'static>)>,
    shapes: Vec<(u8, Keyed<'static, &'static str>)>,
    tokens: Vec<Token<'static>>,
    refusal: Option<Ruled<'static, Validity>>,
    /// A refusal's backoff: per key component, the values deciding alone.
    alone: Vec<(usize, &'static [(u64, Validity)])>,
    rest: Option<Validity>,
}

impl FormatParts {
    /// A `rule` record: the sign, negation or encoding rule of field `index`.
    fn rule(&mut self, index: u32, kind: &str, key: &str, entries: &[&str]) -> Result<(), String> {
        let (_, field) = self
            .fields
            .iter_mut()
            .find(|(k, _)| *k == index)
            .ok_or(format!("rule for missing field {index}"))?;
        match (kind, field) {
            (
                "sign",
                Field::Immediate {
                    sign: sign @ ImmediateSign::Fixed(_),
                    ..
                },
            ) => {
                let ImmediateSign::Fixed(own) = *sign else {
                    unreachable!("matched as fixed")
                };
                let (rule, unseen) = keyed(key, entries, self::sign)?;
                *sign = ImmediateSign::Keyed(Ruled {
                    rule,
                    unseen: unseen.unwrap_or(own),
                });
            }
            (
                "neg",
                Field::Immediate {
                    negation: negation @ Negation::Never,
                    ..
                },
            ) => {
                let (rule, unseen) = keyed(key, entries, bit)?;
                *negation = Negation::Keyed(Ruled {
                    rule,
                    unseen: unseen.unwrap_or(false),
                });
            }
            (
                "zero",
                Field::Immediate {
                    zero: zero @ Zeroing::Never,
                    ..
                },
            ) => {
                let (rule, unseen) = keyed(key, entries, printed)?;
                *zero = Zeroing::Keyed(Ruled {
                    rule,
                    unseen: unseen.unwrap_or(Printed::Value),
                });
            }
            ("enc", Field::Float { encoding, .. }) if encoding.rule.key.is_empty() => {
                let (rule, unseen) = keyed(key, entries, self::encoding)?;
                *encoding = Ruled {
                    rule,
                    unseen: unseen.unwrap_or(encoding.unseen),
                };
            }
            (kind, field) => return Err(format!("rule {kind} does not fit field {field:?}")),
        }
        Ok(())
    }

    /// The free fields with their derived atoms, and the refusal with its
    /// backoff.
    fn finish(mut self) -> Result<(Vec<Field<'static>>, Refusal<'static>), String> {
        for (index, _) in &self.derived {
            if !self
                .fields
                .iter()
                .any(|(k, f)| k == index && matches!(f, Field::Register { .. }))
            {
                return Err(format!("derive for missing register field {index}"));
            }
        }
        let fields = self
            .fields
            .into_iter()
            .map(|(index, field)| match field {
                Field::Register {
                    bits,
                    file,
                    slot,
                    again,
                    xor,
                    ..
                } => Field::Register {
                    bits,
                    file,
                    slot,
                    again,
                    xor,
                    derived: leak(
                        self.derived
                            .iter()
                            .filter(|(k, _)| *k == index)
                            .map(|&(_, d)| d)
                            .collect(),
                    ),
                },
                field => field,
            })
            .collect();
        let refusal = match (self.refusal, self.rest) {
            (None, None) if self.alone.is_empty() => Refusal::Never,
            (Some(r), None) if self.alone.is_empty() => Refusal::Keyed(r),
            (Some(r), Some(rest)) => {
                let mut alone = vec![&[][..]; r.rule.key.len()];
                for (k, values) in std::mem::take(&mut self.alone) {
                    *alone
                        .get_mut(k)
                        .ok_or(format!("backoff component {k} past the key"))? = values;
                }
                Refusal::BackedOff(Backoff {
                    rule: r.rule,
                    alone: leak(alone),
                    rest,
                    unseen: r.unseen,
                })
            }
            _ => return Err("a refusal backoff without its refusal or rest".to_owned()),
        };
        Ok((fields, refusal))
    }
}

/// The format a `field` or `render` record belongs to, listed on first sight.
fn form<'m>(
    forms: &mut Vec<(u16, String)>,
    parts: &'m mut HashMap<(u16, String), FormatParts>,
    c: &[&str],
) -> Result<&'m mut FormatParts, String> {
    let key = (int::<u16>(c[1], "row")?, c[2].to_owned());
    if !parts.contains_key(&key) {
        forms.push(key.clone());
    }
    Ok(parts.entry(key).or_default())
}

pub fn parse(table: &str) -> Result<Parsed, String> {
    let mut lines = table.lines();
    if lines.next() != Some(&format!("# {VERSION}")) {
        return Err(format!("not a {VERSION} table"));
    }
    let mut rows: BTreeMap<u16, RowParts> = BTreeMap::new();
    let mut forms: Vec<(u16, String)> = Vec::new();
    let mut form_parts: HashMap<(u16, String), FormatParts> = HashMap::new();
    let mut raw_variants = Vec::new();
    for line in lines.filter(|l| !l.starts_with('#')) {
        let c: Vec<&str> = line.split('\t').collect();
        let row = || int::<u16>(c[1], "row");
        match c[0] {
            "zero" => rows.entry(row()?).or_default().zero = c[2].to_owned(),
            "order" => rows.entry(row()?).or_default().order.push((
                text(c[2]),
                text(c[3]),
                int(c[4], "count")?,
            )),
            "order-other" => rows.entry(row()?).or_default().order_other.push((
                text(c[2]),
                text(c[3]),
                int(c[4], "count")?,
            )),
            "omit" => rows.entry(row()?).or_default().omit.push(Omission {
                operand: int(c[2], "operand")?,
                shape: shape(c[3]),
                count: int(c[4], "operand count")?,
                value: int(c[5], "omitted value")?,
            }),
            "flag" => rows.entry(row()?).or_default().flags.push(FlagTurn {
                operand: int(c[2], "operand")?,
                from: shape(c[3]),
                bit: int(c[4], "bit")?,
                to: shape(c[5]),
            }),
            "field" => {
                let field = match c[4] {
                    "reg" => Field::Register {
                        bits: bits(c[5])?,
                        file: file(c[6])?,
                        slot: slot(c[7])?,
                        xor: int(c[8], "xor")?,
                        again: leak(c[9..].iter().map(|s| slot(s)).collect::<Result<_, _>>()?),
                        derived: &[],
                    },
                    "imm" => Field::Immediate {
                        bits: bits(c[5])?,
                        slot: slot(c[6])?,
                        scale: scale(c[7])?,
                        bias: int(c[8], "bias")?,
                        sign: ImmediateSign::Fixed(sign(c[9])?),
                        negation: Negation::Never,
                        zero: Zeroing::Never,
                    },
                    "float" => Field::Float {
                        bits: bits(c[5])?,
                        slot: slot(c[6])?,
                        encoding: Ruled {
                            rule: Keyed {
                                key: &[],
                                table: &[],
                            },
                            unseen: encoding(c[7])?,
                        },
                    },
                    "mod" => Field::Modifier {
                        bits: bits(c[5])?,
                        names: names(c[6]),
                        values: leak(
                            c[7..]
                                .iter()
                                .map(|e| {
                                    let (v, n) =
                                        e.split_once('=').ok_or(format!("bad value: {e:?}"))?;
                                    Ok((int(v, "value")?, names(n)))
                                })
                                .collect::<Result<Vec<_>, String>>()?,
                        ),
                    },
                    "flag" => Field::Flag {
                        bits: bits(c[5])?,
                        operand: int(c[6], "operand")?,
                    },
                    "none" => Field::Ignored { bits: bits(c[5])? },
                    // Fixed fields are part of each variant's mask.
                    role if role.starts_with("fixed-") => continue,
                    role => return Err(format!("unknown field role {role:?}")),
                };
                let index = int(c[3], "field index")?;
                form(&mut forms, &mut form_parts, &c)?
                    .fields
                    .push((index, field));
            }
            "rule" => {
                let index = int(c[3], "field index")?;
                form(&mut forms, &mut form_parts, &c)?.rule(index, c[4], c[5], &c[6..])?;
            }
            "derive" => {
                let index = int(c[3], "field index")?;
                let (rule, unseen) = keyed(c[5], &c[6..], derivation)?;
                let derived = Derived {
                    slot: slot(c[4])?,
                    rule,
                    unseen,
                };
                form(&mut forms, &mut form_parts, &c)?
                    .derived
                    .push((index, derived));
            }
            "render" => {
                let (rule, _) = keyed(c[4], &c[5..], |s| Ok(shape(s)))?;
                let operand = int(c[3], "operand")?;
                form(&mut forms, &mut form_parts, &c)?
                    .shapes
                    .push((operand, rule));
            }
            "token" => {
                let (rule, unseen) = keyed(c[4], &c[5..], bit)?;
                if let Some(u) = unseen {
                    return Err(format!("token {} has a fallback {u}", c[3]));
                }
                form(&mut forms, &mut form_parts, &c)?.tokens.push(Token {
                    name: text(c[3]),
                    rule,
                });
            }
            "refuse" => {
                let (rule, unseen) = keyed(c[3], &c[4..], validity)?;
                let parts = form(&mut forms, &mut form_parts, &c)?;
                if parts.refusal.is_some() {
                    return Err(format!("two refusals for row {} form {}", c[1], c[2]));
                }
                parts.refusal = Some(Ruled {
                    rule,
                    unseen: unseen.unwrap_or(Validity::Accepted),
                });
            }
            "refuse-alone" => {
                let component = int(c[3], "key component")?;
                let alone = leak(values(&c[4..], validity)?);
                form(&mut forms, &mut form_parts, &c)?
                    .alone
                    .push((component, alone));
            }
            "refuse-rest" => {
                form(&mut forms, &mut form_parts, &c)?.rest = Some(validity(c[3])?);
            }
            "variant" => raw_variants.push(c),
            kind => return Err(format!("unknown record {kind:?}")),
        }
    }
    let row_index: HashMap<u16, u16> = rows
        .keys()
        .enumerate()
        .map(|(i, &l)| (l, i as u16))
        .collect();
    let rows_out = rows
        .into_iter()
        .map(|(line, mut p)| {
            // Tried from the last operand back, as the inference tries them.
            p.omit.sort_by(|a, b| {
                (b.operand, b.shape, b.count, b.value).cmp(&(a.operand, a.shape, a.count, a.value))
            });
            Row {
                line,
                zero: text(&p.zero),
                order: leak(p.order),
                order_other: leak(p.order_other),
                flags: leak(p.flags),
                omit: leak(p.omit),
            }
        })
        .collect();
    let mut format_index = HashMap::new();
    let mut formats = Vec::new();
    for key in &forms {
        let mut parts = form_parts.remove(key).expect("listed form");
        let shapes = leak(std::mem::take(&mut parts.shapes));
        let tokens = leak(std::mem::take(&mut parts.tokens));
        let (fields, refusal) = parts
            .finish()
            .map_err(|e| format!("row {} form {}: {e}", key.0, key.1))?;
        format_index.insert(key.clone(), formats.len() as u16);
        formats.push(Format {
            row: *row_index
                .get(&key.0)
                .ok_or(format!("row {} has no zero record", key.0))?,
            form: match key.1.as_str() {
                "-" => OperandForm::Only,
                f => OperandForm::Selected(hex(f)?),
            },
            fields: leak(fields),
            shapes,
            tokens,
            refusal,
        });
    }
    let mut variants = Vec::new();
    let mut labels = Vec::new();
    for c in raw_variants {
        let row = int::<u16>(c[2], "row")?;
        let key = (row, c[3].to_owned());
        let format = match format_index.get(&key) {
            Some(&f) => f,
            // A form whose every field is fixed: one word per variant.
            None => {
                format_index.insert(key.clone(), formats.len() as u16);
                formats.push(Format {
                    row: row_index[&row],
                    form: match c[3] {
                        "-" => OperandForm::Only,
                        f => OperandForm::Selected(hex(f)?),
                    },
                    fields: &[],
                    shapes: &[],
                    tokens: &[],
                    refusal: Refusal::Never,
                });
                formats.len() as u16 - 1
            }
        };
        variants.push(Variant {
            mask: hex(c[4])?,
            value: hex(c[5])?,
            format,
            base: hex(c[6])?,
            spacing: match c[9] {
                "glued" => Spacing::Glued,
                "spaced" => Spacing::Spaced,
                s => return Err(format!("bad spacing {s:?}")),
            },
            mnemonic: text(c[10]),
            modifiers: names(c[11]),
            operands: leak(
                c[12..]
                    .iter()
                    .map(|o| operand(o))
                    .collect::<Result<_, _>>()?,
            ),
        });
        labels.push(Label {
            id: int(c[1], "variant id")?,
            row,
            form: c[3].to_owned(),
            inferred: (int(c[7], "samples")?, int(c[8], "exact")?),
        });
    }
    Ok(Parsed {
        rows: rows_out,
        formats,
        variants,
        labels,
    })
}

/// Per top-16-bit prefix, the variants (by position) whose fixed bits allow it.
pub fn prefixes(variants: &[Variant]) -> Vec<(u16, Vec<u32>)> {
    let mut map: BTreeMap<u16, Vec<u32>> = BTreeMap::new();
    for (i, v) in variants.iter().enumerate() {
        let (mask, value) = ((v.mask >> 48) as u16, (v.value >> 48) as u16);
        let free: Vec<u32> = (0..16).filter(|k| (mask >> k) & 1 == 0).collect();
        for n in 0u32..1 << free.len() {
            let mut prefix = value;
            for (j, k) in free.iter().enumerate() {
                prefix |= (((n >> j) & 1) as u16) << k;
            }
            map.entry(prefix).or_default().push(i as u32);
        }
    }
    map.into_iter().collect()
}

pub fn rejected_text(t: &str) -> bool {
    t.starts_with('!') || t.contains("INVALID")
}

/// Why a printed text differs from the disassembler's.
pub fn reason(printed: &str, oracle: &str) -> &'static str {
    let head = |t: &str| {
        let t = t.trim_start();
        let t = match t.starts_with('@') {
            true => t.split_once(' ').map_or("", |(_, r)| r),
            false => t,
        };
        t.split([' ', '[']).next().unwrap_or("").to_owned()
    };
    let guard = |t: &str| {
        t.starts_with('@')
            .then(|| t.split(' ').next().unwrap_or("").to_owned())
    };
    let squash = |t: &str| t.chars().filter(|c| !c.is_whitespace()).collect::<String>();
    if rejected_text(oracle) {
        "prints a word the disassembler rejects"
    } else if squash(printed) == squash(oracle) {
        "spacing"
    } else if guard(printed) != guard(oracle) {
        "guard predicate"
    } else if head(printed).split('.').next() != head(oracle).split('.').next() {
        "mnemonic"
    } else if head(printed) != head(oracle) {
        "modifiers"
    } else {
        "operands"
    }
}

pub struct Rejection {
    pub variant: usize,
    pub claimed: u64,
    pub exact: u64,
    pub reason: &'static str,
    pub word: u64,
    pub oracle: String,
    pub printed: String,
}

pub struct Gate {
    /// Admitted variants by position in `Parsed::variants`, in table order.
    pub admitted: Vec<usize>,
    /// Per admitted variant, the pool positions of the words it prints.
    pub claimed: Vec<Vec<usize>>,
    pub rejected: Vec<Rejection>,
    pub rounds: usize,
}

/// The proof gate: every pool word a variant claims must print exactly.
pub fn gate(parsed: &Parsed, pool: &[(u64, String)]) -> Gate {
    let mut active: Vec<usize> = (0..parsed.variants.len()).collect();
    // A word held with two texts was printed at two addresses (a PC-relative
    // target): no variant prints it exactly.
    let mut texts: HashMap<u64, &str> = HashMap::new();
    let ambiguous: std::collections::HashSet<u64> = pool
        .iter()
        .filter(|(w, t)| *texts.entry(*w).or_insert(t) != t.as_str())
        .map(|(w, _)| *w)
        .collect();
    let mut rejected = Vec::new();
    let mut rounds = 0;
    loop {
        rounds += 1;
        let variants: Vec<Variant> = active.iter().map(|&i| parsed.variants[i]).collect();
        let index = prefixes(&variants);
        let index_ref: Vec<(u16, &[u32])> = index.iter().map(|(p, v)| (*p, v.as_slice())).collect();
        let table = Table {
            rows: &parsed.rows,
            formats: &parsed.formats,
            variants: &variants,
            prefixes: &index_ref,
        };
        let mut claimed = vec![Vec::new(); active.len()];
        let mut first_bad: Vec<Option<(usize, String)>> = vec![None; active.len()];
        let mut bad = vec![0u64; active.len()];
        for (n, (word, oracle)) in pool.iter().enumerate() {
            let Some(id) = table.recognise(*word) else {
                continue;
            };
            let id = id as usize;
            claimed[id].push(n);
            let printed = table.text(id as u32, *word).expect("recognised");
            if printed != *oracle {
                bad[id] += 1;
                let replace = match &first_bad[id] {
                    None => true,
                    Some((m, _)) => ambiguous.contains(word) && !ambiguous.contains(&pool[*m].0),
                };
                if replace {
                    first_bad[id] = Some((n, printed));
                }
            }
        }
        let failed: Vec<usize> = (0..active.len()).filter(|&i| bad[i] > 0).collect();
        let printed: usize = claimed.iter().map(Vec::len).sum();
        eprintln!(
            "round {rounds}: {} variants print {printed} words, {} exactly; {} variants misprint",
            active.len(),
            printed as u64 - bad.iter().sum::<u64>(),
            failed.len()
        );
        if failed.is_empty() {
            let (admitted, claimed): (Vec<usize>, Vec<Vec<usize>>) = active
                .iter()
                .zip(claimed)
                .filter_map(|(&v, c)| {
                    if c.is_empty() {
                        rejected.push(Rejection {
                            variant: v,
                            claimed: 0,
                            exact: 0,
                            reason: "no word of its own: earlier variants print them all",
                            word: parsed.variants[v].base,
                            oracle: String::new(),
                            printed: String::new(),
                        });
                        None
                    } else {
                        Some((v, c))
                    }
                })
                .unzip();
            rejected.sort_by_key(|r| r.variant);
            return Gate {
                admitted,
                claimed,
                rejected,
                rounds,
            };
        }
        for &i in &failed {
            let (n, printed) = first_bad[i].clone().expect("a misprint");
            let (word, oracle) = &pool[n];
            rejected.push(Rejection {
                variant: active[i],
                claimed: claimed[i].len() as u64,
                exact: claimed[i].len() as u64 - bad[i],
                reason: match ambiguous.contains(word) {
                    true => "text depends on the word's address",
                    false => reason(&printed, oracle),
                },
                word: *word,
                oracle: oracle.clone(),
                printed,
            });
        }
        let failed: std::collections::HashSet<usize> = failed.into_iter().collect();
        active = active
            .into_iter()
            .enumerate()
            .filter(|(i, _)| !failed.contains(i))
            .map(|(_, v)| v)
            .collect();
    }
}

fn bits_rs(b: Bits) -> String {
    let parts: Vec<String> =
        b.0.iter()
            .map(|p| format!("Part {{ lo: {}, hi: {} }}", p.lo, p.hi))
            .collect();
    format!("Bits(&[{}])", parts.join(", "))
}

fn selector_rs(s: Selector) -> String {
    match s {
        Selector::Value(b) => format!("Selector::Value({})", bits_rs(b)),
        Selector::Nonzero(b) => format!("Selector::Nonzero({})", bits_rs(b)),
    }
}

fn validity_rs(v: Validity) -> String {
    format!("Validity::{v:?}")
}

fn derivation_rs(d: Derivation) -> String {
    format!("Derivation::{d:?}")
}

fn keyed_rs<T: Copy>(k: Keyed<T>, value: impl Fn(T) -> String) -> String {
    let key: Vec<String> = k.key.iter().map(|&s| selector_rs(s)).collect();
    let table: Vec<String> = k
        .table
        .iter()
        .map(|(vs, t)| format!("(&{vs:?}, {})", value(*t)))
        .collect();
    format!(
        "Keyed {{ key: &[{}], table: &[{}] }}",
        key.join(", "),
        table.join(", ")
    )
}

fn sign_rs(s: Sign) -> String {
    format!("Sign::{s:?}")
}

fn slot_rs(s: Slot) -> String {
    format!("Slot {{ operand: {}, atom: {} }}", s.operand, s.atom)
}

fn ruled_rs<T: Copy>(r: Ruled<T>, value: impl Fn(T) -> String) -> String {
    format!(
        "Ruled {{ rule: {}, unseen: {} }}",
        keyed_rs(r.rule, &value),
        value(r.unseen)
    )
}

fn encoding_rs(e: FloatEncoding) -> String {
    format!(
        "FloatEncoding {{ width: FloatWidth::{:?}, shift: {}, sign: FloatSign::{:?} }}",
        e.width, e.shift, e.sign
    )
}

fn field_rs(f: &Field) -> String {
    match *f {
        Field::Register {
            bits,
            file,
            slot,
            again,
            xor,
            derived,
        } => format!(
            "Field::Register {{ bits: {}, file: RegisterFile::{file:?}, slot: {}, again: &[{}], xor: {xor}, derived: &[{}] }}",
            bits_rs(bits),
            slot_rs(slot),
            again.iter().map(|&s| slot_rs(s)).collect::<Vec<_>>().join(", "),
            derived
                .iter()
                .map(|d| format!(
                    "Derived {{ slot: {}, rule: {}, unseen: {} }}",
                    slot_rs(d.slot),
                    keyed_rs(d.rule, derivation_rs),
                    match d.unseen {
                        Some(u) => format!("Some({})", derivation_rs(u)),
                        None => "None".to_owned(),
                    }
                ))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Field::Immediate {
            bits,
            slot,
            scale,
            bias,
            sign,
            negation,
            zero,
        } => format!(
            "Field::Immediate {{ bits: {}, slot: {}, scale: {}, bias: {bias}, sign: {}, negation: {}, zero: {} }}",
            bits_rs(bits),
            slot_rs(slot),
            match scale {
                Scale::Shift(s) => format!("Scale::Shift({s})"),
                Scale::Weights(w) => format!("Scale::Weights(&{w:?})"),
            },
            match sign {
                ImmediateSign::Fixed(s) => format!("ImmediateSign::Fixed({})", sign_rs(s)),
                ImmediateSign::Keyed(r) => format!("ImmediateSign::Keyed({})", ruled_rs(r, sign_rs)),
            },
            match negation {
                Negation::Never => "Negation::Never".to_owned(),
                Negation::Keyed(r) => format!("Negation::Keyed({})", ruled_rs(r, |b| b.to_string())),
            },
            match zero {
                Zeroing::Never => "Zeroing::Never".to_owned(),
                Zeroing::Keyed(r) => format!("Zeroing::Keyed({})", ruled_rs(r, |p| format!("Printed::{p:?}"))),
            }
        ),
        Field::Float {
            bits,
            slot,
            encoding,
        } => format!(
            "Field::Float {{ bits: {}, slot: {}, encoding: {} }}",
            bits_rs(bits),
            slot_rs(slot),
            ruled_rs(encoding, encoding_rs)
        ),
        Field::Modifier {
            bits,
            names,
            values,
        } => {
            let values: Vec<String> = values
                .iter()
                .map(|(v, n)| format!("({v}, &{n:?})"))
                .collect();
            format!(
                "Field::Modifier {{ bits: {}, names: &{names:?}, values: &[{}] }}",
                bits_rs(bits),
                values.join(", ")
            )
        }
        Field::Flag { bits, operand } => {
            format!(
                "Field::Flag {{ bits: {}, operand: {operand} }}",
                bits_rs(bits)
            )
        }
        Field::Ignored { bits } => format!("Field::Ignored {{ bits: {} }}", bits_rs(bits)),
    }
}

fn atom_rs(a: &Atom) -> String {
    match *a {
        Atom::Register(r) => format!("Atom::Register({r})"),
        Atom::Predicate(p) => format!("Atom::Predicate({p})"),
        Atom::Number { value, radix } => {
            format!("Atom::Number {{ value: {value}, radix: Radix::{radix:?} }}")
        }
        Atom::Float(raw) => format!("Atom::Float({raw:?})"),
        Atom::Real(r) => format!(
            "Atom::Real(Real {{ width: FloatWidth::{:?}, bits: {} }})",
            r.width, r.bits
        ),
    }
}

/// The admitted variants as Rust: only the formats and rows they use.
pub fn emit(parsed: &Parsed, admitted: &[usize], digest: &str) -> String {
    let mut format_new: BTreeMap<u16, u16> = BTreeMap::new();
    for &v in admitted {
        let f = parsed.variants[v].format;
        let next = format_new.len() as u16;
        format_new.entry(f).or_insert(next);
    }
    let mut formats: Vec<(u16, u16)> = format_new.iter().map(|(&old, &new)| (new, old)).collect();
    formats.sort();
    let mut row_new: BTreeMap<u16, u16> = BTreeMap::new();
    for &(_, old) in &formats {
        row_new
            .entry(parsed.formats[usize::from(old)].row)
            .or_insert(0);
    }
    for (i, v) in row_new.values_mut().enumerate() {
        *v = i as u16;
    }
    let mut out = String::new();
    writeln!(
        out,
        "// @generated by tools/generate-decoder-formats from\n\
         // crates/shadowbox/tests/data/decoder-formats.tsv ({VERSION}, sha256 {digest}).\n\
         // Do not edit: regenerate. {} variants of {} formats over {} rows.\n\
         use super::*;\n",
        admitted.len(),
        formats.len(),
        row_new.len()
    )
    .unwrap();
    writeln!(
        out,
        "pub static TABLE: Table<'static> = Table {{\n    rows: &["
    )
    .unwrap();
    for &old in row_new.keys() {
        let r = &parsed.rows[usize::from(old)];
        let flags: Vec<String> = r
            .flags
            .iter()
            .map(|t| {
                format!(
                    "FlagTurn {{ operand: {}, from: {:?}, bit: {}, to: {:?} }}",
                    t.operand, t.from, t.bit, t.to
                )
            })
            .collect();
        let omit: Vec<String> = r
            .omit
            .iter()
            .map(|o| {
                format!(
                    "Omission {{ operand: {}, shape: {:?}, count: {}, value: {} }}",
                    o.operand, o.shape, o.count, o.value
                )
            })
            .collect();
        writeln!(
            out,
            "        Row {{ line: {}, zero: {:?}, order: &{:?}, order_other: &{:?}, flags: &[{}], omit: &[{}] }},",
            r.line,
            r.zero,
            r.order,
            r.order_other,
            flags.join(", "),
            omit.join(", ")
        )
        .unwrap();
    }
    writeln!(out, "    ],\n    formats: &[").unwrap();
    for &(_, old) in &formats {
        let f = &parsed.formats[usize::from(old)];
        let fields: Vec<String> = f.fields.iter().map(field_rs).collect();
        let shapes: Vec<String> = f
            .shapes
            .iter()
            .map(|&(o, k)| format!("({o}, {})", keyed_rs(k, |s| format!("{s:?}"))))
            .collect();
        let form = match f.form {
            OperandForm::Only => "OperandForm::Only".to_owned(),
            OperandForm::Selected(v) => format!("OperandForm::Selected(0x{v:x})"),
        };
        let tokens: Vec<String> = f
            .tokens
            .iter()
            .map(|t| {
                format!(
                    "Token {{ name: {:?}, rule: {} }}",
                    t.name,
                    keyed_rs(t.rule, |b| b.to_string())
                )
            })
            .collect();
        let refusal = match f.refusal {
            Refusal::Never => "Refusal::Never".to_owned(),
            Refusal::Keyed(r) => format!("Refusal::Keyed({})", ruled_rs(r, validity_rs)),
            Refusal::BackedOff(b) => format!(
                "Refusal::BackedOff(Backoff {{ rule: {}, alone: &[{}], rest: {}, unseen: {} }})",
                keyed_rs(b.rule, validity_rs),
                b.alone
                    .iter()
                    .map(|vs| format!(
                        "&[{}]",
                        vs.iter()
                            .map(|&(v, r)| format!("({v}, {})", validity_rs(r)))
                            .collect::<Vec<_>>()
                            .join(", ")
                    ))
                    .collect::<Vec<_>>()
                    .join(", "),
                validity_rs(b.rest),
                validity_rs(b.unseen)
            ),
        };
        writeln!(
            out,
            "        Format {{ row: {}, form: {form}, fields: &[{}], shapes: &[{}], tokens: &[{}], refusal: {refusal} }},",
            row_new[&f.row],
            fields.join(", "),
            shapes.join(", "),
            tokens.join(", ")
        )
        .unwrap();
    }
    writeln!(out, "    ],\n    variants: &[").unwrap();
    let mut kept = Vec::new();
    for &v in admitted {
        let x = &parsed.variants[v];
        let operands: Vec<String> = x
            .operands
            .iter()
            .map(|o| {
                let atoms: Vec<String> = o.atoms.iter().map(atom_rs).collect();
                format!(
                    "Operand {{ shape: {:?}, atoms: &[{}] }}",
                    o.shape,
                    atoms.join(", ")
                )
            })
            .collect();
        writeln!(
            out,
            "        Variant {{ mask: 0x{:016x}, value: 0x{:016x}, format: {}, base: 0x{:016x}, \
             spacing: Spacing::{:?}, mnemonic: {:?}, modifiers: &{:?}, operands: &[{}] }},",
            x.mask,
            x.value,
            format_new[&x.format],
            x.base,
            x.spacing,
            x.mnemonic,
            x.modifiers,
            operands.join(", ")
        )
        .unwrap();
        kept.push(*x);
    }
    writeln!(out, "    ],\n    prefixes: &[").unwrap();
    for (p, ids) in prefixes(&kept) {
        writeln!(out, "        (0x{p:04x}, &{ids:?}),").unwrap();
    }
    writeln!(out, "    ],\n}};").unwrap();
    out
}
