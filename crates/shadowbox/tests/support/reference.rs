// Independent CPU oracle. Dispatch and operands come from encoding bits,
// never from the decoded InstructionKind or emitted SPIR-V.
// Paths below: sm50.rs/ir.rs are Mesa src/nouveau/compiler/nak;
// gm107.c is envytools envydis; Inst*.cs are Ryujinx.Graphics.Shader/Instructions.
// No upstream implementation copied. See docs/gpu-exact.md.
use super::compute::{bfe_cbuf, cbuf, WORDS};

pub fn field(raw: u64, start: u32, width: u32) -> u32 {
    ((raw >> start) & ((1_u64 << width) - 1)) as u32
}
fn flag(raw: u64, at: u32) -> bool {
    field(raw, at, 1) != 0
}
fn sext(v: u32, width: u32) -> u32 {
    ((v << (32 - width)) as i32 >> (32 - width)) as u32
}
fn extract(v: u32, at: u32, width: u32, signed: bool) -> u32 {
    if width == 0 {
        return 0;
    }
    let v = v >> at;
    if width == 32 {
        v
    } else if signed {
        sext(v, width)
    } else {
        v & ((1 << width) - 1)
    }
}
fn neg(v: u32, yes: bool) -> u32 {
    if yes {
        v.wrapping_neg()
    } else {
        v
    }
}
fn absneg(v: u32, abs: bool, negative: bool) -> u32 {
    neg(
        if abs {
            (v as i32).wrapping_abs() as u32
        } else {
            v
        },
        negative,
    )
}
fn fmod(v: u32, abs: bool, negative: bool) -> u32 {
    (if abs { v & 0x7fff_ffff } else { v }) ^ if negative { 0x8000_0000 } else { 0 }
}
pub fn flush(v: u32) -> u32 {
    if v & 0x7fff_ffff < 0x0080_0000 {
        v & 0x8000_0000
    } else {
        v
    }
}

// sm50.rs:335-372: F20/S20 immediate and constant-buffer word offset.
// gm107.c:203-218: common register operands. Bank34..39 and word20..34: sm50.rs:361-372.
fn operand(raw: u64, state: &[u32; WORDS], float: bool) -> u32 {
    match raw >> 60 {
        4 => cbuf(field(raw, 34, 5), field(raw, 20, 14)),
        3 => {
            let bits = field(raw, 20, 19) | field(raw, 56, 1) << 19;
            if float {
                bits << 12
            } else {
                sext(bits, 20)
            }
        }
        _ => state[field(raw, 20, 8) as usize],
    }
}

fn int_cmp(code: u32, a: u32, b: u32, signed: bool) -> bool {
    let lt = if signed { (a as i32) < b as i32 } else { a < b };
    let eq = a == b;
    [false, lt, eq, lt || eq, !lt && !eq, !eq, !lt, true][code as usize]
}
fn float_cmp(code: u32, a: u32, b: u32) -> bool {
    let a = f32::from_bits(a);
    let b = f32::from_bits(b);
    let nan = a.is_nan() || b.is_nan();
    match code {
        0 => false,
        7 => !nan,
        8 => nan,
        15 => true,
        _ => {
            let basic = match code & 7 {
                1 => a < b,
                2 => a == b,
                3 => a <= b,
                4 => a > b,
                5 => a != b,
                6 => a >= b,
                _ => unreachable!(),
            };
            if code >= 9 {
                nan || basic
            } else {
                !nan && basic
            }
        }
    }
}
fn double_cmp(code: u32, a: u64, b: u64) -> bool {
    let a = f64::from_bits(a);
    let b = f64::from_bits(b);
    let nan = a.is_nan() || b.is_nan();
    match code {
        0 => false,
        7 => !nan,
        8 => nan,
        15 => true,
        _ => {
            let basic = match code & 7 {
                1 => a < b,
                2 => a == b,
                3 => a <= b,
                4 => a > b,
                5 => a != b,
                6 => a >= b,
                _ => unreachable!(),
            };
            if code >= 9 {
                nan || basic
            } else {
                !nan && basic
            }
        }
    }
}
fn logical(code: u32, a: bool, b: bool) -> Result<bool, &'static str> {
    match code {
        0 => Ok(a && b),
        1 => Ok(a || b),
        2 => Ok(a ^ b),
        _ => Err("reserved predicate operation"),
    }
}
fn pred(raw: u64, state: &[u32; WORDS], at: u32) -> bool {
    (state[256 + field(raw, at, 3) as usize] != 0) ^ flag(raw, at + 3)
}

pub struct Expected {
    pub words: [u32; WORDS],
    // 0 exact; 1 arithmetic f32 NaN; 2 two arithmetic half NaNs.
    pub nan: [u8; WORDS],
}
impl Expected {
    pub fn new(state: &[u32; WORDS]) -> Self {
        Self {
            words: *state,
            nan: [0; WORDS],
        }
    }
    fn zn(&mut self, v: u32) {
        self.words[264] = u32::from(v == 0);
        self.words[265] = v >> 31;
    }
    fn fpzn(&mut self, v: u32) {
        let f = f32::from_bits(v);
        self.words[264] = u32::from(f == 0.0);
        self.words[265] = u32::from(f < 0.0);
    }
    fn fp64zn(&mut self, v: u64) {
        let f = f64::from_bits(v);
        self.words[264] = u32::from(f == 0.0);
        self.words[265] = u32::from(f < 0.0);
    }
}

// Independent binary16 conversion with nearest-even packing.
// Arithmetic NaN sign/payload are unspecified by GLSL, see normalise below.
// Rounding rule: Mesa src/util/half_float.c:38-44,96-131.
pub fn unpack(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exponent = (h >> 10) & 31;
    let mantissa = h & 1023;
    let bits = match exponent {
        0 => (f32::from(mantissa) * 2.0_f32.powi(-24)).to_bits(),
        31 => 0x7f80_0000 | u32::from(mantissa) << 13,
        _ => u32::from(exponent + 112) << 23 | u32::from(mantissa) << 13,
    };
    f32::from_bits(bits | sign)
}
pub fn pack(f: f32) -> u16 {
    let sign = ((f.to_bits() >> 16) & 0x8000) as u16;
    let value = f.abs();
    if value.is_nan() {
        return sign | 0x7e00;
    }
    if value >= 65520.0 {
        return sign | 0x7c00;
    }
    if value < 2.0_f32.powi(-14) {
        return sign | (value * 2.0_f32.powi(24)).round_ties_even() as u16;
    }
    let exponent = ((value.to_bits() >> 23) & 255) as i32 - 127;
    let significand = (value * 2.0_f32.powi(10 - exponent)).round_ties_even() as u16;
    sign | ((((exponent + 15) as u16) << 10) + significand - 1024)
}
fn rounded(f: f32, mode: u32) -> f32 {
    match mode {
        0 => f.round_ties_even(),
        1 => f.floor(),
        2 => f.ceil(),
        3 => f.trunc(),
        _ => unreachable!(),
    }
}

/// Independent binary64 nearest-even addition using the host's IEEE
/// binary64 operation. NaN payload selection is outside the exact domain.
fn dadd_rn_bits(lhs: u64, rhs: u64) -> Result<u64, &'static str> {
    let a = f64::from_bits(lhs);
    let b = f64::from_bits(rhs);
    if a.is_nan() || b.is_nan() {
        return Err("undefined DADD NaN payload");
    }
    let sum = a + b;
    if sum.is_nan() {
        return Err("undefined DADD NaN payload");
    }
    Ok(sum.to_bits())
}

fn next_down_f64(bits: u64) -> u64 {
    if bits & 0x7fff_ffff_ffff_ffff == 0 {
        0x8000_0000_0000_0001
    } else if bits >> 63 != 0 {
        bits + 1
    } else {
        bits - 1
    }
}

/// Independent binary64 add rounded toward negative infinity. TwoSum
/// recovers the exact residual from the host's nearest-even addition.
fn dadd_rm_bits(lhs: u64, rhs: u64) -> Result<u64, &'static str> {
    let a = f64::from_bits(lhs);
    let b = f64::from_bits(rhs);
    if a.is_nan() || b.is_nan() {
        return Err("undefined DADD NaN payload");
    }
    let sum = a + b;
    if sum.is_nan() {
        return Err("undefined DADD NaN payload");
    }
    if sum == f64::INFINITY && a.is_finite() && b.is_finite() {
        return Ok(f64::MAX.to_bits());
    }
    if !sum.is_finite() {
        return Ok(sum.to_bits());
    }
    if sum == 0.0 {
        return Ok(
            if a == 0.0 && b == 0.0 && a.is_sign_positive() && b.is_sign_positive() {
                0
            } else {
                (-0.0_f64).to_bits()
            },
        );
    }
    let virtual_b = sum - a;
    let virtual_a = sum - virtual_b;
    let error = (a - virtual_a) + (b - virtual_b);
    Ok(if error < 0.0 {
        next_down_f64(sum.to_bits())
    } else {
        sum.to_bits()
    })
}

fn next_up_f64(bits: u64) -> u64 {
    if bits & 0x7fff_ffff_ffff_ffff == 0 {
        1
    } else if bits >> 63 != 0 {
        bits - 1
    } else {
        bits + 1
    }
}

/// Independent binary64 add rounded toward positive infinity. TwoSum
/// recovers the exact residual from the host's nearest-even addition.
fn dadd_rp_bits(lhs: u64, rhs: u64) -> Result<u64, &'static str> {
    let a = f64::from_bits(lhs);
    let b = f64::from_bits(rhs);
    if a.is_nan() || b.is_nan() {
        return Err("undefined DADD NaN payload");
    }
    let sum = a + b;
    if sum.is_nan() {
        return Err("undefined DADD NaN payload");
    }
    if sum == f64::NEG_INFINITY && a.is_finite() && b.is_finite() {
        return Ok((-f64::MAX).to_bits());
    }
    if !sum.is_finite() {
        return Ok(sum.to_bits());
    }
    if sum == 0.0 {
        return Ok(
            if a == 0.0 && b == 0.0 && a.is_sign_negative() && b.is_sign_negative() {
                (-0.0_f64).to_bits()
            } else {
                0
            },
        );
    }
    let virtual_b = sum - a;
    let virtual_a = sum - virtual_b;
    let error = (a - virtual_a) + (b - virtual_b);
    Ok(if error > 0.0 {
        next_up_f64(sum.to_bits())
    } else {
        sum.to_bits()
    })
}

/// Independent binary64 add rounded toward zero. TwoSum recovers the exact
/// residual from the host's nearest-even addition.
fn dadd_rz_bits(lhs: u64, rhs: u64) -> Result<u64, &'static str> {
    let a = f64::from_bits(lhs);
    let b = f64::from_bits(rhs);
    if a.is_nan() || b.is_nan() {
        return Err("undefined DADD NaN payload");
    }
    let sum = a + b;
    if sum.is_nan() {
        return Err("undefined DADD NaN payload");
    }
    if sum.is_infinite() && a.is_finite() && b.is_finite() {
        return Ok(if sum.is_sign_negative() {
            (-f64::MAX).to_bits()
        } else {
            f64::MAX.to_bits()
        });
    }
    if !sum.is_finite() {
        return Ok(sum.to_bits());
    }
    if sum == 0.0 {
        return Ok(
            if a == 0.0 && b == 0.0 && a.is_sign_negative() && b.is_sign_negative() {
                (-0.0_f64).to_bits()
            } else {
                0
            },
        );
    }
    let virtual_b = sum - a;
    let virtual_a = sum - virtual_b;
    let error = (a - virtual_a) + (b - virtual_b);
    Ok(if sum.is_sign_positive() && error < 0.0 {
        next_down_f64(sum.to_bits())
    } else if sum.is_sign_negative() && error > 0.0 {
        next_up_f64(sum.to_bits())
    } else {
        sum.to_bits()
    })
}

// Raw half encodings: envytools gm107.c:1973-1989,2155-2157,
// 240-241,1133-1175,1774-1785. Swizzles/modifiers/packing:
// Ryujinx InstEmitHelper.cs:139-184; arithmetic:
// InstEmitFloatArithmetic.cs:251-383,483-528; comparisons:
// InstEmitFloatComparison.cs:309-366,456-523. No decoder IR is consumed.
fn half_pair(bits: u32, swizzle: u32, absolute: bool, negative: bool) -> [f32; 2] {
    let low = unpack(bits as u16);
    let high = unpack((bits >> 16) as u16);
    let pair = match swizzle {
        0 => [low, high],
        1 => [f32::from_bits(bits); 2],
        2 => [low; 2],
        3 => [high; 2],
        _ => unreachable!(),
    };
    pair.map(|value| f32::from_bits(fmod(value.to_bits(), absolute, negative)))
}

// Source precision follows Ryujinx InstEmitHelper.cs:139-158. The
// binary16 normal boundary and signed-zero flush are Mesa
// src/nouveau/compiler/nak/from_nir.rs:1313-1320. Ryujinx's half emitters
// omit FTZ/FMZ; the project contract is docs/reference-sources.md
// Numeric classification is independent of the emitter's integer bit tests.
fn half_flush_value(value: f32, f32_format: bool) -> f32 {
    let normal = if f32_format {
        f32::MIN_POSITIVE
    } else {
        2.0_f32.powi(-14)
    };
    if value.abs() < normal {
        0.0_f32.copysign(value)
    } else {
        value
    }
}

pub fn is_half(raw: u64) -> bool {
    let o = (raw >> 48) as u16;
    matches!(o & 0xfff8, 0x5d00 | 0x5d08 | 0x5d10 | 0x5d18 | 0x5d20)
        || matches!(
            o & 0xfe00,
            0x2800 | 0x2a00 | 0x2c00 | 0x7800 | 0x7a00 | 0x7c00 | 0x7e00
        )
        || matches!(o & 0xf880, 0x7000 | 0x7080 | 0x6080)
}

fn half_reference(raw: u64, state: &[u32; WORDS]) -> Result<Expected, &'static str> {
    let o = (raw >> 48) as u16;
    let f = |a, b| field(raw, a, b);
    let bit = |at| flag(raw, at);
    let register = o & 0xff00 == 0x5d00;
    let long = matches!(o & 0xfe00, 0x2800 | 0x2a00 | 0x2c00);
    let cbuf_form = !register && !long && bit(55);
    let comparison = matches!(o & 0xfff8, 0x5d18 | 0x5d20) || matches!(o & 0xfe00, 0x7c00 | 0x7e00);
    let fma = o & 0xfff8 == 0x5d00
        || o & 0xfe00 == 0x2800
        || matches!(o & 0xf880, 0x7000 | 0x7080 | 0x6080);
    let add = o & 0xfff8 == 0x5d10 || o & 0xfe00 == 0x2c00 || o & 0xfe00 == 0x7a00;
    let dst = f(0, 8) as usize;
    let predicates = o & 0xfff8 == 0x5d20 || o & 0xfe00 == 0x7e00;
    let rhs = if register {
        state[f(20, 8) as usize]
    } else if long {
        f(20, 32)
    } else if cbuf_form {
        cbuf(f(34, 5), f(20, 14))
    } else {
        // Both signs are inside the literal. U09_20+NEG29, U09_30+NEG56.
        (f(20, 10) << 6) | (f(30, 9) << 22) | (f(56, 1) << 31)
    };
    let mut source = [state[f(8, 8) as usize], rhs, 0];
    let mut swizzle = [
        f(if long { 53 } else { 47 }, 2),
        if register {
            f(28, 2)
        } else {
            u32::from(cbuf_form)
        },
        0,
    ];
    let mut absolute = [false; 3];
    let mut negative = [false; 3];
    if fma {
        source[2] = state[if long { dst } else { f(39, 8) as usize }];
        swizzle[2] = if long {
            0
        } else {
            f(if register { 35 } else { 53 }, 2)
        };
        negative[1] = if register {
            bit(31)
        } else {
            cbuf_form && bit(56)
        };
        negative[2] = bit(if long {
            52
        } else if register {
            30
        } else {
            51
        });
        if o & 0xf880 == 0x6080 {
            source.swap(1, 2);
            swizzle.swap(1, 2);
            negative.swap(1, 2);
        }
    } else {
        absolute[0] = !long && bit(44);
        absolute[1] = register && bit(30) || cbuf_form && bit(54) && (!comparison || predicates);
        negative[0] = if long {
            add && bit(56)
        } else {
            (!register || add || comparison) && bit(43)
        };
        negative[1] = register && bit(31) || cbuf_form && bit(56);
        if cbuf_form && !add && !comparison {
            negative[0] = false;
            negative[1] = bit(43);
        }
    }
    let mut pair: [[f32; 2]; 3] =
        std::array::from_fn(|i| half_pair(source[i], swizzle[i], absolute[i], negative[i]));
    // envytools gm107.c:1973-1989,2155-2157; mode tables
    // :1157-1168,1083-1087,1782-1786. Mesa's mode encoding is
    // nv50_ir_emit_gm107.cpp:474-476, DNZ << 1 | FTZ.
    let flush = if comparison {
        f(
            if predicates {
                6
            } else if register {
                50
            } else {
                54
            },
            1,
        )
    } else if register {
        f(if fma { 37 } else { 39 }, if add { 1 } else { 2 })
    } else if long {
        f(55, if add { 1 } else { 2 })
    } else {
        f(if fma { 57 } else { 39 }, if add { 1 } else { 2 })
    };
    if flush == 3 {
        return Err("reserved half flush mode");
    }
    if flush != 0 {
        for i in 0..if fma { 3 } else { 2 } {
            pair[i] = pair[i].map(|value| half_flush_value(value, swizzle[i] == 1));
        }
    }
    let mut out = Expected::new(state);
    if comparison {
        let mut q = std::array::from_fn::<_, 2, _>(|i| {
            float_cmp(
                f(if register { 35 } else { 49 }, 4),
                pair[0][i].to_bits(),
                pair[1][i].to_bits(),
            )
        });
        if predicates && bit(if register { 49 } else { 53 }) {
            let both = q[0] && q[1];
            q = [both, !both];
        }
        let p = pred(raw, state, 39);
        let q = [logical(f(45, 2), q[0], p)?, logical(f(45, 2), q[1], p)?];
        if predicates {
            out.words[256 + f(3, 3) as usize] = u32::from(q[0]);
            out.words[256 + f(0, 3) as usize] = u32::from(q[1]);
        } else if dst != 255 {
            let mask = if bit(if register { 49 } else { 53 }) {
                0x3c00
            } else {
                0xffff
            };
            out.words[dst] = if q[0] { mask } else { 0 } | if q[1] { mask << 16 } else { 0 };
        }
    } else if dst != 255 {
        let sat = !(long && fma) && bit(if register { 32 } else { 52 });
        let mut result = [0_f32; 2];
        for (i, value) in result.iter_mut().enumerate() {
            *value = if flush == 2 && (pair[0][i] == 0.0 || pair[1][i] == 0.0) {
                // FMZ zero-product contract: docs/reference-sources.md
                // Compute the zero product directly, without reproducing the
                // emitter's operand substitution. FMA still adds C once.
                let negative = pair[0][i].is_sign_negative() ^ pair[1][i].is_sign_negative();
                let product = 0.0_f32.copysign(if negative { -1.0 } else { 1.0 });
                if fma {
                    product + pair[2][i]
                } else {
                    product
                }
            } else if fma {
                pair[0][i].mul_add(pair[1][i], pair[2][i])
            } else if add {
                pair[0][i] + pair[1][i]
            } else {
                pair[0][i] * pair[1][i]
            };
            if sat {
                // Mesa nir_opcodes.py:265, u_math.h:343-371, fsat NaN -> +0.
                *value = f32::from_bits(finish_float(value.to_bits(), true, false)?);
            }
        }
        let output = if long { 0 } else { f(49, 2) };
        if output == 1 {
            out.words[dst] = if flush != 0 {
                half_flush_value(result[0], true)
            } else {
                result[0]
            }
            .to_bits();
            out.nan[dst] = 1;
        } else {
            // Round before output flushing so a result rounding up to the
            // smallest half normal survives: Mesa half_float.c:96-104,122-128.
            // Restore preserved lanes after flushing, InstEmitHelper:173-184.
            if flush != 0 {
                result = result.map(|value| half_flush_value(unpack(pack(value)), false));
            }
            // Preserved halves also undergo unpack/pack, InstEmitHelper:173-184.
            if output == 2 {
                result[1] = unpack((state[dst] >> 16) as u16);
            }
            if output == 3 {
                result[0] = unpack(state[dst] as u16);
            }
            out.words[dst] = u32::from(pack(result[0])) | u32::from(pack(result[1])) << 16;
            out.nan[dst] = 2;
        }
    }
    out.words[255] = 0;
    out.words[263] = 1;
    Ok(if pred(raw, state, 16) {
        out
    } else {
        Expected::new(state)
    })
}
fn bounds(size: u32, signed: bool) -> (i64, i64) {
    let bits = 8 << size;
    if signed {
        (-(1_i64 << (bits - 1)), (1_i64 << (bits - 1)) - 1)
    } else {
        (0, (1_i64 << bits) - 1)
    }
}
fn clamp_int(v: u32, size: u32, signed: bool) -> u32 {
    let (lo, hi) = bounds(size, signed);
    (if signed {
        i64::from(v as i32)
    } else {
        i64::from(v)
    })
    .clamp(lo, hi) as u32
}
fn minmax(a: u32, b: u32, signed: bool, max: bool) -> u32 {
    let a_less = if signed { (a as i32) < b as i32 } else { a < b };
    if a_less ^ max {
        a
    } else {
        b
    }
}

// TwoSum retains the error of a binary64 sum. Binary32 products are exact in
// binary64. This also handles directed FADD with operands far apart in exponent.
fn sum(a: f64, b: f64) -> (f64, f64) {
    let value = a + b;
    let delta = value - a;
    (value, (a - (value - delta)) + (b - delta))
}
fn float_round(value: f64, error: f64, mode: u32) -> u32 {
    let mut rounded = value as f32;
    if value.is_finite() {
        let actual = f64::from(rounded);
        let less = actual < value || actual == value && error > 0.0;
        let greater = actual > value || actual == value && error < 0.0;
        match mode {
            1 if greater => rounded = rounded.next_down(),
            2 if less => rounded = rounded.next_up(),
            3 if value >= 0.0 && greater => rounded = rounded.next_down(),
            3 if value < 0.0 && less => rounded = rounded.next_up(),
            _ => {}
        }
        if mode == 0 && error != 0.0 {
            // A binary64 sum can land exactly on a binary32 midpoint while
            // TwoSum's residual lies on one side of it.
            if error > 0.0 && (actual + f64::from(rounded.next_up())) * 0.5 == value {
                rounded = rounded.next_up();
            } else if error < 0.0 && (actual + f64::from(rounded.next_down())) * 0.5 == value {
                rounded = rounded.next_down();
            }
        }
    }
    rounded.to_bits()
}
// IEEE 754-2019 minimumNumber/maximumNumber: a NaN operand yields the other
// one, and -0 orders below +0.
fn min_max_number(x: u32, y: u32, max: bool) -> u32 {
    if f32::from_bits(x).is_nan() {
        return y;
    }
    if f32::from_bits(y).is_nan() {
        return x;
    }
    let key = |v: u32| if v >> 31 == 1 { !v } else { v | 0x8000_0000 };
    if (key(x) < key(y)) ^ max {
        x
    } else {
        y
    }
}
fn finish_float(bits: u32, sat: bool, ftz: bool) -> Result<u32, &'static str> {
    let bits = if ftz { flush(bits) } else { bits };
    // Mesa nir_opcodes.py:265 and util/u_math.h:343-371 define fsat
    // through min/max-number. NaN and -0 saturate to +0. Unlike bare
    // Ryujinx FClamp (Spirv/Instructions.cs:334-336), this is deterministic.
    if sat && f32::from_bits(bits).is_nan() {
        return Ok(0);
    }
    if sat && bits == 0x8000_0000 {
        return Ok(0);
    }
    Ok(if sat {
        f32::from_bits(bits).clamp(0.0, 1.0).to_bits()
    } else {
        bits
    })
}

/// Evaluate one raw instruction. Unsupported semantics are a coverage gap,
/// never an assumed identity. Guarded-off forms still require an oracle.
pub fn reference(raw: u64, state: &[u32; WORDS]) -> Result<Expected, &'static str> {
    if is_half(raw) {
        return half_reference(raw, state);
    }
    let o = (raw >> 48) as u16;
    let f = |at, width| field(raw, at, width);
    let bit = |at| flag(raw, at);
    let d = f(0, 8) as usize;
    let a = state[f(8, 8) as usize];
    let b = |float| operand(raw, state, float);
    let c = state[f(39, 8) as usize];
    let mut out = Expected::new(state);
    let mut nan = 0;
    let value = match o {
        // CSET: envytools gm107.c:2059,373,406,1430-1434,1647-1680 places
        // condition8..12, accumulator39..41/not42, Boolean45..46 and BF44;
        // Ryujinx InstEmitConditionCode.cs:12-33 combines the condition
        // once with the accumulator and writes all ones or 1.0f.
        _ if matches!(
            raw,
            0x5098_0380_0007_0101
                | 0x5098_1380_0007_0401
                | 0x5098_2400_0007_0201
                | 0x5098_4080_0007_1401
                | 0x5098_3580_0007_0d01
                | 0x5098_0380_0007_1f01
        ) =>
        {
            let cc = super::control_reference::condition(f(8, 5), state)?;
            let combined = logical(f(45, 2), cc, pred(raw, state, 39))?;
            Some(if combined {
                if bit(44) {
                    1.0_f32.to_bits()
                } else {
                    u32::MAX
                }
            } else {
                0
            })
        }
        // CSET selectors LE/NE/GE/LTU/LO/RLE: envytools gm107.c:2059,
        // 1647-1680 names them; Ryujinx InstDecoders.cs:104-111 numbers
        // them and InstEmitConditionCode.cs:12-33,52-84 evaluates the flag
        // formula, combines it once with the accumulator, then writes all
        // ones or 1.0f.
        _ if matches!(
            raw,
            0x5098_0100_0007_0301
                | 0x5098_3380_0007_0501
                | 0x5098_4480_0007_0601
                | 0x5098_1380_0007_0901
                | 0x5098_2400_0007_1101
                | 0x5098_0280_0007_1e01
        ) =>
        {
            let cc = super::control_reference::condition(f(8, 5), state)?;
            let combined = logical(f(45, 2), cc, pred(raw, state, 39))?;
            Some(if combined {
                if bit(44) {
                    1.0_f32.to_bits()
                } else {
                    u32::MAX
                }
            } else {
                0
            })
        }
        // DSET register/cbuf/immediate: envytools gm107.c:2035,2105,2153
        // defines the fields; Ryujinx InstEmitFloatComparison.cs:13-86,
        // 378-419 establishes binary64 comparison and the 32-bit result.
        _ if matches!(
            raw,
            0x5902_0380_0047_0201
                | 0x4901_0380_0047_0201
                | 0x3203_03bf_f007_0201
                | 0x5907_0380_0047_0201
                | 0x5908_0380_0047_0201
                | 0x590d_0380_0047_0201
                | 0x5914_0380_0047_0201
                | 0x5906_2400_0047_0201
                | 0x5909_4080_0047_0201
                | 0x5901_1b80_0047_0201
                | 0x596a_0380_0047_0201
                | 0x491e_0b80_0047_0201
                | 0x5940_bf2f_80e1_2c1a
                | 0x5961_cbc8_e1e6_78bd
                | 0x3248_b199_07bb_bb0d
                | 0x334a_d6b1_3595_21a1
                | 0x593f_94c6_cf78_22b3
                | 0x3255_cb90_b738_8bba
                | 0x3313_b2ae_d9bc_64e7
        ) =>
        {
            // Source modifiers act on the binary64 sign bit: absolute
            // clears it, then negation flips it (absA54/negA43,
            // absB44/negB53).
            let modify = |value: u64, abs: bool, negate: bool| {
                let value = if abs { value & !(1 << 63) } else { value };
                if negate {
                    value ^ 1 << 63
                } else {
                    value
                }
            };
            let pair =
                |register: usize| u64::from(state[register]) | u64::from(state[register | 1]) << 32;
            let lhs = modify(pair(f(8, 8) as usize), bit(54), bit(43));
            let rhs = match raw >> 60 {
                4 => {
                    let bank = f(34, 5);
                    let word = f(20, 14);
                    u64::from(cbuf(bank, word)) | u64::from(cbuf(bank, word + 1)) << 32
                }
                3 => u64::from(operand(raw, state, true)) << 32,
                _ => pair(f(20, 8) as usize),
            };
            let rhs = modify(rhs, bit(44), bit(53));
            let compared = double_cmp(f(48, 4), lhs, rhs);
            let combined = logical(f(45, 2), compared, pred(raw, state, 39))?;
            let value = if combined {
                if bit(52) {
                    1.0_f32.to_bits()
                } else {
                    u32::MAX
                }
            } else {
                0
            };
            if bit(47) {
                if bit(52) {
                    out.fpzn(value);
                } else {
                    out.zn(value);
                }
            }
            Some(value)
        }
        // DMNMX register: Mesa sm50.rs:1046-1075; Ryubing
        // InstEmitFloatMinMax.cs:12-21. Both operands are little-endian
        // binary64 register pairs. The predicate at39 selects minimum when
        // true and maximum when false after source modifiers are applied.
        // C/I forms: envytools gm107.c:2081,2127 put the right operand at
        // slot34..38/word offset20..33 or D20_20 with sign56; Ryubing
        // InstEmitHelper.cs:42-66 packs words offset (low) and offset + 1
        // (high), and the immediate as the high word over a zero low word.
        _ if matches!(o & 0xfff8, 0x5c50 | 0x4c50) || o & 0xfef8 == 0x3850 => {
            let pair =
                |register: usize| u64::from(state[register]) | u64::from(state[register | 1]) << 32;
            let modified = |bits: u64, absolute: bool, negate: bool| {
                let bits = if absolute {
                    bits & !(1_u64 << 63)
                } else {
                    bits
                };
                if negate {
                    bits ^ (1_u64 << 63)
                } else {
                    bits
                }
            };
            let lhs = modified(pair(f(8, 8) as usize), bit(46), bit(48));
            let rhs = match raw >> 60 {
                4 => {
                    let (bank, word) = (f(34, 5), f(20, 14));
                    u64::from(cbuf(bank, word)) | u64::from(cbuf(bank, word + 1)) << 32
                }
                3 => u64::from(f(20, 19) << 12 | f(56, 1) << 31) << 32,
                _ => pair(f(20, 8) as usize),
            };
            let rhs = modified(rhs, bit(49), bit(45));
            // Ryubing lowers FPMinimum/FPMaximum to GLSL.std.450 FMin/FMax.
            // Model the resulting numeric-minimum behavior without invoking
            // Rust's min/max helpers: one NaN yields the numeric operand and
            // equal zeros select -0 for minimum or +0 for maximum. The
            // deterministic both-NaN fixtures make the modified operands
            // bit-identical because FMin/FMax do not promise payload choice.
            let x = f64::from_bits(lhs);
            let y = f64::from_bits(rhs);
            let minimum = pred(raw, state, 39);
            let result = if x.is_nan() {
                if y.is_nan() {
                    lhs
                } else {
                    rhs
                }
            } else if y.is_nan() {
                lhs
            } else if x == 0.0 && y == 0.0 {
                if minimum {
                    lhs | rhs
                } else {
                    lhs & rhs
                }
            } else if minimum {
                if y < x {
                    rhs
                } else {
                    lhs
                }
            } else {
                if x < y {
                    rhs
                } else {
                    lhs
                }
            };
            if bit(47) {
                out.fp64zn(result);
            }
            out.words[d | 1] = (result >> 32) as u32;
            if d & 1 == 0 {
                out.words[d] = result as u32;
            }
            None
        }
        // DFMA register/cbuf/register-cbuf/immediate: Mesa sm50.rs:980-1036;
        // envytools gm107.c:2027,2098,2045,2145; Ryujinx
        // InstEmitFloatArithmetic.cs:42-84. The product and addend signs are
        // independent and `mul_add` supplies the required single rounding.
        // A binary64 cbuf operand is two consecutive words; a D20 immediate
        // is the high word with a zero low word.
        _ if matches!(o & 0xfff0, 0x5b70 | 0x4b70 | 0x5370) || o & 0xfef0 == 0x3670 => {
            let pair =
                |register: usize| u64::from(state[register]) | u64::from(state[register | 1]) << 32;
            let constant = || {
                let bank = f(34, 5);
                let word = f(20, 14);
                u64::from(cbuf(bank, word)) | u64::from(cbuf(bank, word + 1)) << 32
            };
            let lhs = pair(f(8, 8) as usize);
            let (multiplier, addend) = match (o & 0xfff0, o & 0xfef0) {
                (0x4b70, _) => (constant(), pair(f(39, 8) as usize)),
                (0x5370, _) => (pair(f(39, 8) as usize), constant()),
                (_, 0x3670) => (
                    u64::from(operand(raw, state, true)) << 32,
                    pair(f(39, 8) as usize),
                ),
                _ => (pair(f(20, 8) as usize), pair(f(39, 8) as usize)),
            };
            let lhs = if bit(48) { lhs ^ (1_u64 << 63) } else { lhs };
            let addend = if bit(49) {
                addend ^ (1_u64 << 63)
            } else {
                addend
            };
            let result = f64::from_bits(lhs)
                .mul_add(f64::from_bits(multiplier), f64::from_bits(addend))
                .to_bits();
            if bit(47) {
                out.fp64zn(result);
            }
            out.words[d + 1] = (result >> 32) as u32;
            Some(result as u32)
        }
        // DMUL register/cbuf/immediate: Mesa sm50.rs:1079-1117; Ryujinx
        // InstEmitFloatArithmetic.cs:86-114. Binary64 sources are pairs and
        // bit48 negates the product.
        _ if matches!(o & 0xfff8, 0x5c80 | 0x4c80) || o & 0xfef8 == 0x3880 => {
            let pair =
                |register: usize| u64::from(state[register]) | u64::from(state[register | 1]) << 32;
            let lhs = pair(f(8, 8) as usize);
            let rhs = match raw >> 60 {
                4 => {
                    let bank = f(34, 5);
                    let word = f(20, 14);
                    u64::from(cbuf(bank, word)) | u64::from(cbuf(bank, word + 1)) << 32
                }
                3 => u64::from(operand(raw, state, true)) << 32,
                _ => pair(f(20, 8) as usize),
            };
            let lhs = if bit(48) { lhs ^ (1_u64 << 63) } else { lhs };
            let result = (f64::from_bits(lhs) * f64::from_bits(rhs)).to_bits();
            if bit(47) {
                out.fp64zn(result);
            }
            out.words[d + 1] = (result >> 32) as u32;
            Some(result as u32)
        }
        // DADD register: Mesa sm50.rs:947-980; Ryujinx
        // InstEmitFloatArithmetic.cs:12-20. Register pairs are little-endian
        // binary64 and each source applies absolute value before negation.
        _ if matches!(o & 0xfff8, 0x5c70 | 0x4c70) || o & 0xfef8 == 0x3870 => {
            let pair =
                |register: usize| u64::from(state[register]) | u64::from(state[register | 1]) << 32;
            let modified = |bits: u64, absolute: bool, negate: bool| {
                let bits = if absolute {
                    bits & !(1_u64 << 63)
                } else {
                    bits
                };
                if negate {
                    bits ^ (1_u64 << 63)
                } else {
                    bits
                }
            };
            let lhs = modified(pair(f(8, 8) as usize), bit(46), bit(48));
            let rhs = match raw >> 60 {
                4 => {
                    let bank = f(34, 5);
                    let word = f(20, 14);
                    u64::from(cbuf(bank, word)) | u64::from(cbuf(bank, word + 1)) << 32
                }
                3 => u64::from(operand(raw, state, true)) << 32,
                _ => pair(f(20, 8) as usize),
            };
            let rhs = modified(rhs, bit(49), bit(45));
            let result = match f(39, 2) {
                0 => dadd_rn_bits(lhs, rhs)?,
                1 => dadd_rm_bits(lhs, rhs)?,
                2 => dadd_rp_bits(lhs, rhs)?,
                3 => dadd_rz_bits(lhs, rhs)?,
                _ => unreachable!(),
            };
            if bit(47) {
                out.fp64zn(result);
            }
            if d != 255 {
                out.words[d | 1] = (result >> 32) as u32;
                if d & 1 == 0 {
                    out.words[d] = result as u32;
                }
            }
            None
        }
        // MOV: sm50.rs:1931-1951; InstEmitMove.cs:14-44.
        0x0100..=0x01ff => {
            // QuadMask12..15, sm50.rs:1941, is outside this scalar oracle.
            if f(12, 4) != 15 {
                return Err("MOV quad mask needs lane oracle");
            }
            Some(f(20, 32))
        }
        0x5c98 | 0x4c98 => {
            // QuadMask39..42, sm50.rs:1936,1946.
            if f(39, 4) != 15 {
                return Err("MOV quad mask needs lane oracle");
            }
            Some(b(false))
        }
        // IADD and ISCADD: gm107.c:2014-2015; sm50.rs:1249-1286;
        // InstEmitIntegerArithmetic.cs:317-340,460-478.
        _ if matches!(o & 0xfffc, 0x5c10 | 0x4c10) || o & 0xfefc == 0x3810 => {
            let a = neg(a, bit(49));
            let b = neg(b(false), bit(48));
            let carry = if bit(43) { state[266] } else { 0 };
            let total = u64::from(a) + u64::from(b) + u64::from(carry);
            let v = total as u32;
            if bit(47) {
                out.zn(v);
                if bit(43) {
                    out.words[264] &= state[264];
                }
                out.words[266] = (total >> 32) as u32;
                let signed = i64::from(a as i32) + i64::from(b as i32) + i64::from(carry);
                out.words[267] =
                    u32::from(signed < i64::from(i32::MIN) || signed > i64::from(i32::MAX));
            }
            Some(v)
        }
        _ if matches!(o & 0xfffc, 0x5c18 | 0x4c18) || o & 0xfefc == 0x3818 => {
            // ISCADD shifts then negates, and .CC writes Z/N/C/V:
            // Ryujinx InstEmitIntegerArithmetic.cs:469-478,665-698;
            // envytools gm107.c:2014,2088,2134 places .CC at bit47.
            let a = neg(a << f(39, 5), bit(49));
            let b = neg(b(false), bit(48));
            let total = u64::from(a) + u64::from(b);
            let value = total as u32;
            if bit(47) {
                out.zn(value);
                out.words[266] = (total >> 32) as u32;
                let signed = i64::from(a as i32) + i64::from(b as i32);
                out.words[267] =
                    u32::from(signed < i64::from(i32::MIN) || signed > i64::from(i32::MAX));
            }
            Some(value)
        }
        // BFI R/C/I: envytools gm107.c:2019,2092,2139,253,203-207; Mesa
        // nv50_ir_emit_gm107.cpp:2304-2312. Ryujinx InstEmitBitfield.cs:
        // 53-61,154-162 uses C as base, A as insertion, immediate as position/size.
        _ if o & 0xfef8 == 0x36f0 || matches!(o & 0xfff8, 0x53f0 | 0x5bf0 | 0x4bf0) => {
            // BFI.RC swaps range/base: envytools gm107.c:2041,258,302-309;
            // Ryujinx InstEmitBitfield.cs:75-83,154-162. RZ gives count zero.
            let control = if o & 0xfff8 == 0x53f0 { c } else { b(false) };
            let base = if o & 0xfff8 == 0x53f0 {
                cbuf(f(34, 5), f(20, 14))
            } else {
                c
            };
            let offset = control & 255;
            let width = (control >> 8) & 255;
            if offset > 32 || width > 32 - offset {
                return Err("BFI oracle missing out-of-range insert semantics");
            }
            let mask = if width == 32 {
                u32::MAX
            } else {
                (1u32 << width) - 1
            };
            Some(if width == 0 {
                base
            } else {
                (base & !(mask << offset)) | ((a & mask) << offset)
            })
        }
        // CSETP: envytools gm107.c:2058,368-369,373,406,1430-1434,1647-1680.
        // Ryujinx InstEmitConditionCode.cs:35-47 combines condition and its
        // complement independently with the original accumulator.
        0x50a0..=0x50a7 => {
            let cc = match f(8, 5) {
                24..=29 => true,
                condition => super::control_reference::condition(condition, state)?,
            };
            let acc = pred(raw, state, 39);
            out.words[256 + f(3, 3) as usize] = u32::from(logical(f(45, 2), cc, acc)?);
            out.words[256 + f(0, 3) as usize] = u32::from(logical(f(45, 2), !cc, acc)?);
            None
        }
        // IADD32I fields: gm107.c:2162; InstEmitIntegerArithmetic.cs:
        // 43-50,317-341,665-699. NegA/NegB act before carry-in; .X chains
        // the prior Z and consumes prior C exactly like register IADD.
        _ if o & 0xfe40 == 0x1c00 => {
            let a = neg(a, bit(56));
            let b = neg(f(20, 32), bit(55));
            let carry = if bit(53) { state[266] & 1 } else { 0 };
            let total = u64::from(a) + u64::from(b) + u64::from(carry);
            let value = total as u32;
            if bit(52) {
                out.zn(value);
                if bit(53) {
                    out.words[264] &= state[264];
                }
                out.words[266] = (total >> 32) as u32;
                let signed = i64::from(a as i32) + i64::from(b as i32) + i64::from(carry);
                out.words[267] =
                    u32::from(signed < i64::from(i32::MIN) || signed > i64::from(i32::MAX));
            }
            Some(value)
        }
        // ISCADD32I: gm107.c:2164 places CC52, imm32 20..51 and
        // shift53..57; InstEmitIntegerArithmetic.cs:211-219,460-479 adds
        // A << shift to the immediate and .CC writes Z/N/C/V.
        _ if o & 0xfc00 == 0x1400 => {
            let a = a << f(53, 5);
            let b = f(20, 32);
            let total = u64::from(a) + u64::from(b);
            let value = total as u32;
            if bit(52) {
                out.zn(value);
                out.words[266] = (total >> 32) as u32;
                let signed = i64::from(a as i32) + i64::from(b as i32);
                out.words[267] =
                    u32::from(signed < i64::from(i32::MIN) || signed > i64::from(i32::MAX));
            }
            Some(value)
        }
        // IADD3: gm107.c:1228-1253,1994,2068,2114;
        // InstEmitIntegerArithmetic.cs:343-406.
        _ if matches!(o & 0xfff0, 0x5cc0 | 0x4cc0) || o & 0xfef0 == 0x38c0 => {
            let mut v = [a, b(false), c];
            for (i, at) in [35, 33, 31].into_iter().enumerate() {
                if o & 0xfff0 == 0x5cc0 {
                    v[i] = match f(at, 2) {
                        0 => v[i],
                        1 => v[i] & 65535,
                        2 => v[i] >> 16,
                        _ => return Err("reserved IADD3 half"),
                    };
                }
                v[i] = neg(v[i], bit(51 - i as u32));
            }
            let partial = v[0].wrapping_add(v[1]);
            let partial = if o & 0xfff0 == 0x5cc0 {
                match f(37, 2) {
                    0 => partial,
                    1 => partial >> 16,
                    2 => partial << 16,
                    _ => return Err("reserved IADD3 shift"),
                }
            } else {
                partial
            };
            Some(partial.wrapping_add(v[2]))
        }
        // IMUL: sm50.rs:1405-1451; InstEmitIntegerArithmetic.cs:409-457.
        0x5c38 | 0x4c38 | 0x3838 => {
            let as_int = |v: u32, signed| -> i128 {
                if signed {
                    i128::from(v as i32)
                } else {
                    i128::from(v)
                }
            };
            let product = (as_int(a, bit(40)) * as_int(b(false), bit(41))) as u64;
            Some(if bit(39) {
                (product >> 32) as u32
            } else {
                product as u32
            })
        }
        // IMUL32I: InstDecoders.cs InstImul32i and
        // InstEmitIntegerArithmetic.cs:171-178,409-457.
        _ if o & 0xff00 == 0x1f00 => {
            let signed = [bit(54), bit(55)];
            let extend = |value: u32, signed: bool| -> i128 {
                if signed {
                    i128::from(value as i32)
                } else {
                    i128::from(value)
                }
            };
            let product = (extend(a, signed[0]) * extend(f(20, 32), signed[1])) as u64;
            Some(if bit(53) {
                (product >> 32) as u32
            } else {
                product as u32
            })
        }
        // IMAD register/cbuf/immediate/RC: Mesa sm50.rs:1347-1403;
        // Ryujinx InstEmitIntegerArithmetic.cs:84-138,409-457.
        _ if matches!(o & 0xff80, 0x5a00 | 0x4a00 | 0x5200) || o & 0xfe80 == 0x3400 => {
            let (mut multiplier, mut addend) = if o & 0xff80 == 0x5200 {
                (state[f(39, 8) as usize], cbuf(f(34, 5), f(20, 14)))
            } else {
                (b(false), c)
            };
            match f(51, 2) {
                0 => {}
                1 => addend = addend.wrapping_neg(),
                2 => multiplier = multiplier.wrapping_neg(),
                _ => return Err("reserved IMAD PlusOne mode"),
            }
            let extend = |value: u32, signed: bool| -> i128 {
                if signed {
                    i128::from(value as i32)
                } else {
                    i128::from(value)
                }
            };
            let product = (extend(a, bit(48)) * extend(multiplier, bit(53))) as u64;
            let word = if bit(54) {
                (product >> 32) as u32
            } else {
                product as u32
            };
            Some(word.wrapping_add(addend))
        }
        // IMAD32I: InstDecoders.cs InstImad32i and
        // InstEmitIntegerArithmetic.cs:129-138,409-457. C is the old
        // destination, so retain the state snapshot before writing it.
        _ if o & 0xfc00 == 0x1000 => {
            let mut multiplier = f(20, 32);
            let mut addend = state[d];
            match f(55, 2) {
                0 => {}
                1 => addend = addend.wrapping_neg(),
                2 => multiplier = multiplier.wrapping_neg(),
                _ => return Err("reserved IMAD32I PlusOne mode"),
            }
            let extend = |value: u32, signed: bool| -> i128 {
                if signed {
                    i128::from(value as i32)
                } else {
                    i128::from(value)
                }
            };
            let product = (extend(a, bit(54)) * extend(multiplier, bit(57))) as u64;
            let word = if bit(53) {
                (product >> 32) as u32
            } else {
                product as u32
            };
            Some(word.wrapping_add(addend))
        }
        // IMADSP: Mesa nak/ir.rs OpIMadSp fold. Ryujinx supplies the field
        // names but deliberately has no implementation to use as an oracle.
        _ if matches!(o & 0xff80, 0x5a80 | 0x4a80 | 0x5280) || o & 0xfe80 == 0x3480 => {
            let signed_c = bit(48) || bit(53);
            let types = [f(48, 3), f(53, 2) + 2, f(51, 2) * 2 + u32::from(signed_c)];
            let extract = |value: u32, ty: u32| match ty {
                0 | 1 => value,
                2 => value & 0x00ff_ffff,
                3 => ((value << 8) as i32 >> 8) as u32,
                4 => value & 0xffff,
                5 => (value as i16 as i32) as u32,
                6 => value >> 16,
                7 => ((value as i32) >> 16) as u32,
                _ => unreachable!(),
            };
            let (b, c) = if o & 0xff80 == 0x5280 {
                (state[f(39, 8) as usize], cbuf(f(34, 5), f(20, 14)))
            } else {
                (b(false), c)
            };
            Some(
                extract(a, types[0])
                    .wrapping_mul(extract(b, types[1]))
                    .wrapping_add(extract(c, types[2])),
            )
        }
        // XMAD: gm107.c:2031,2063,2107,2131,1937-1993 in Mesa's
        // nv50_ir_emit_gm107.cpp; InstEmitIntegerArithmetic.cs:565-656.
        _ if o & 0xffc0 == 0x5b00
            || o & 0xfec0 == 0x3600
            || o & 0xfe00 == 0x4e00
            || o & 0xff80 == 0x5100 =>
        {
            let cb_b = o & 0xfe00 == 0x4e00;
            let cb_c = o & 0xff80 == 0x5100;
            let immediate = o & 0xfec0 == 0x3600;
            let full_b = if cb_b {
                cbuf(f(34, 5), f(20, 14))
            } else if cb_c {
                c
            } else if immediate {
                f(20, 16)
            } else {
                b(false)
            };
            let mut c = if cb_c { cbuf(f(34, 5), f(20, 14)) } else { c };
            let b_high = if cb_b || cb_c {
                bit(52)
            } else {
                !immediate && bit(35)
            };
            let a = extract(a, if bit(53) { 16 } else { 0 }, 16, bit(48));
            let b = extract(full_b, if b_high { 16 } else { 0 }, 16, bit(49));
            let mut product = a.wrapping_mul(b);
            let psl = if cb_b {
                bit(55)
            } else if cb_c {
                false
            } else {
                bit(36)
            };
            let merge = if cb_b {
                bit(56)
            } else if cb_c {
                false
            } else {
                bit(37)
            };
            if psl {
                product <<= 16;
            }
            c = match f(50, if cb_b || cb_c { 2 } else { 3 }) {
                0 => c,
                1 => c & 65535,
                2 => c >> 16,
                // CSFU: InstEmitIntegerArithmetic.cs:629-634 subtracts the
                // sign bit of each widened half, shifted to bit 16.
                3 => c.wrapping_sub((a >> 31 << 16).wrapping_add(b >> 31 << 16)),
                4 => c.wrapping_add(full_b << 16),
                _ => return Err("unimplemented XMAD mode"),
            };
            let value = product.wrapping_add(c);
            if bit(47) {
                // Ryujinx InstEmitIntegerArithmetic.cs:261-274,665-699:
                // XMAD.CC observes product+C before the optional MRG.
                out.zn(value);
                out.words[266] = u32::from(u64::from(product) + u64::from(c) > u64::from(u32::MAX));
                let signed = i64::from(product as i32) + i64::from(c as i32);
                out.words[267] =
                    u32::from(signed < i64::from(i32::MIN) || signed > i64::from(i32::MAX));
            }
            Some(if merge {
                value & 65535 | full_b << 16
            } else {
                value
            })
        }
        // SHL/SHR: gm107.c:2008,2012,2128,2132; ir.rs:4603-4670;
        // InstEmitShift.cs:180-246.
        _ if matches!(o, 0x5c48 | 0x4c48)
            || o & 0xfeff == 0x3848
            || matches!(
                raw,
                0x5c4a_0733_d3f7_4209
                    | 0x5c4a_07b3_d3f7_4209
                    | 0x4c4a_037a_0367_2672
                    | 0x4c4a_03fa_0367_2672
                    | 0x3849_4406_e117_07a6
                    | 0x3849_4486_e117_07a6
            ) =>
        {
            let shift = if bit(39) { b(false) & 31 } else { b(false) };
            Some(a.checked_shl(shift).unwrap_or(0))
        }
        _ if matches!(o & 0xfffe, 0x5c28 | 0x4c28) || o & 0xfefe == 0x3828 => {
            // .BREV 40 (gm107.c:2012; InstEmitShift.cs:211-214): reverse A
            // before both the shift and the clamped sign fill.
            let a = if bit(40) { a.reverse_bits() } else { a };
            let shift = if bit(39) { b(false) & 31 } else { b(false) };
            Some(if bit(48) {
                (a as i32).checked_shr(shift).unwrap_or((a as i32) >> 31) as u32
            } else {
                a.checked_shr(shift).unwrap_or(0)
            })
        }
        // FLO source/modifiers: sm50.rs:1203-1225, gm107.c:2011,2085,2131.
        // Ryujinx InstEmitBitfield.cs:164-182 uses FindMSB(S/U) or
        // FindLSB(reverseBits) for .SH. Count leading bits directly to
        // avoid reversing then scanning. Ryujinx returns -1 for .SH zero;
        // Mesa ir.rs:3736-3751 returns 32 there. Preserve the explicit
        // Ryujinx result and report that boundary disagreement.
        _ if matches!(o & 0xfff8, 0x5c30 | 0x4c30) || o & 0xfef8 == 0x3830 => {
            let v = if bit(40) { !b(false) } else { b(false) };
            let v = if bit(48) && v & 0x8000_0000 != 0 {
                !v
            } else {
                v
            };
            Some(if v == 0 {
                u32::MAX
            } else if bit(41) {
                v.leading_zeros()
            } else {
                31 - v.leading_zeros()
            })
        }
        // POPC source is B, inversion40: sm50.rs:1616-1640;
        // gm107.c:2016; InstEmitBitfield.cs:185-191.
        0x5c08 | 0x4c08 | 0x3808 => Some((if bit(40) { !b(false) } else { b(false) }).count_ones()),
        // LOP: gm107.c:2009,2083,2129,2168;
        // InstEmitIntegerLogical.cs:87-113,139-162.
        _ if matches!(o & 0xfff8, 0x5c40 | 0x4c40 | 0x3840) || raw >> 58 == 1 => {
            let long = raw >> 58 == 1;
            let a = if bit(if long { 55 } else { 39 }) {
                !a
            } else {
                a
            };
            let b = if long { f(20, 32) } else { b(false) };
            let b = if bit(if long { 56 } else { 40 }) {
                !b
            } else {
                b
            };
            let v = match f(if long { 53 } else { 41 }, 2) {
                0 => a & b,
                1 => a | b,
                2 => a ^ b,
                3 => b,
                _ => unreachable!(),
            };
            if bit(if long { 52 } else { 47 }) {
                out.zn(v);
                // X57 (LOP32I) / X43 (LOP): envytools gm107.c:2009,2083,
                // 2129,2168; Ryujinx InstEmitAluHelper.cs:102-114 keeps Z
                // only when the entry Z was set.
                if bit(if long { 57 } else { 43 }) {
                    out.words[264] &= state[264];
                }
            }
            if !long && f(48, 3) != 7 {
                let q = match f(44, 2) {
                    0 => false,
                    1 => true,
                    2 => v == 0,
                    3 => v != 0,
                    _ => unreachable!(),
                };
                out.words[256 + f(48, 3) as usize] = u32::from(q);
            }
            Some(v)
        }
        // LOP3 immediate: gm107.c:2108 lists S20_20 and LUT48, but omits
        // the third source. Ryujinx Decoders/InstDecoders.cs:3752-3764 and
        // InstEmitIntegerLogical.cs:65-75 supply B=S20 and C=register39.
        // S20 sign56: Mesa sm50.rs:335-346. A occupies truth-table bit2,
        // B bit1, C bit0: Ryujinx Lop3Expression.cs:13-29,34-42.
        _ if o & 0xfc00 == 0x3c00 || o & 0xfff8 == 0x5be0 => {
            let b = operand(raw, state, false);
            let c = state[f(39, 8) as usize];
            // Register LUT28..35 and predicate48..50/mode36..37:
            // envytools gm107.c:2020,1391-1396; Ryujinx
            // InstEmitIntegerLogical.cs:118-162. Evaluate each bit's truth
            // table independently of the translator's boolean expression.
            let register = o & 0xfff8 == 0x5be0;
            let lut = f(if register { 28 } else { 48 }, 8);
            let mut v = 0;
            for n in 0..32 {
                let index = ((a >> n) & 1) << 2 | ((b >> n) & 1) << 1 | ((c >> n) & 1);
                v |= ((lut >> index) & 1) << n;
            }
            if bit(47) {
                out.zn(v);
                // Register X38 propagates the entry zero flag.
                // envytools gm107.c:2020; Ryujinx
                // InstEmitIntegerLogical.cs:118-136, InstEmitAluHelper.cs:102-116.
                if register && bit(38) {
                    out.words[264] &= state[264];
                }
            }
            if register && f(48, 3) != 7 {
                out.words[256 + f(48, 3) as usize] = match f(36, 2) {
                    0 => 0,
                    1 => 1,
                    2 => u32::from(v == 0),
                    3 => u32::from(v != 0),
                    _ => unreachable!(),
                };
            }
            Some(v)
        }
        // PRMT: sm50.rs:1955-2000, ir.rs:5013-5030,5133-5150.
        0x5bc0 | 0x4bc0 | 0x36c0 | 0x37c0 => {
            let selector = b(false);
            let sources = [a, c];
            let mut v = 0;
            for byte in 0..4 {
                let nibble = (selector >> (byte * 4)) & 15;
                let mut selected =
                    (sources[((nibble >> 2) & 1) as usize] >> ((nibble & 3) * 8)) & 255;
                if nibble & 8 != 0 {
                    selected = if selected & 128 != 0 { 255 } else { 0 };
                }
                v |= selected << (byte * 8);
            }
            Some(v)
        }
        // BFE: sm50.rs:1166-1195; gm107.c:2017,2091,2137;
        // InstEmitBitfield.cs:128-151. Undefined dynamic ranges are excluded.
        _ if matches!(o & 0xfff8, 0x5c00 | 0x4c00) || o & 0xfef8 == 0x3800 => {
            let range = if o & 0xfff8 == 0x4c00 {
                bfe_cbuf(f(34, 5), f(20, 14))
            } else {
                b(false)
            };
            let at = range & 255;
            let width = (range >> 8) & 255;
            if at > 32 || width > 32 - at {
                return Err("undefined BFE range");
            }
            let a = if bit(40) { a.reverse_bits() } else { a };
            Some(if width == 0 {
                0
            } else {
                extract(a, at, width, bit(48))
            })
        }
        // LEA.HI: gm107.c:2021,2163; InstEmitIntegerArithmetic.cs:493-523.
        _ if o & 0xfff8 == 0x5bd8 || o & 0xfc00 == 0x1800 => {
            let cb = o & 0xfc00 == 0x1800;
            let shift = f(if cb { 51 } else { 28 }, 5);
            let pair = (u64::from(c) << 32 | u64::from(a)).wrapping_shl(shift);
            let pair = if bit(if cb { 56 } else { 37 }) {
                pair.wrapping_neg()
            } else {
                pair
            };
            let b = if cb {
                cbuf(f(34, 5), f(20, 14))
            } else {
                b(false)
            };
            Some(((pair >> 32) as u32).wrapping_add(b))
        }
        // SHF: sm50.rs:1644-1693; ir.rs:4501-4544.
        // Maxwell high-destination unsigned shift differs from the simplified
        // Ryujinx InstEmitShift.cs:107-174 at boundaries. Use Mesa's fold.
        _ if matches!(o & 0xfff8, 0x5bf8 | 0x5cf8) || matches!(o & 0xfef8, 0x36f8 | 0x38f8) => {
            let right = matches!(o & 0xfef8, 0x5cf8 | 0x38f8);
            let high = !right || bit(48);
            let ty = f(37, 2);
            let limit = if ty == 0 { 32 } else { 64 };
            let shift = if o >> 12 == 3 { f(20, 6) } else { b(false) };
            let shift = if bit(50) {
                shift & (limit - 1)
            } else {
                shift.min(limit)
            };
            let pair = u64::from(a) | u64::from(c) << 32;
            let shifted = if right {
                if ty == 3 {
                    (pair as i64)
                        .checked_shr(shift)
                        .unwrap_or((pair as i64) >> 63) as u64
                } else {
                    pair.checked_shr(shift).unwrap_or(0)
                }
            } else {
                pair.checked_shl(shift).unwrap_or(0)
            };
            let v = if high {
                (shifted >> 32) as u32
            } else {
                shifted as u32
            };
            if bit(47) {
                out.zn(v);
            }
            Some(v)
        }
        // I2I/I2F/F2I encodings: sm50.rs:1806-1922;
        // gm107.c:1200-1225,1256-1307,1993-1996,2067-2070,2113-2116.
        // Semantics: InstEmitConversion.cs:165-342,344-408.
        _ if matches!(o & 0xfff8, 0x5ce0 | 0x4ce0) || o & 0xfef8 == 0x38e0 => {
            let v = extract(b(false), f(41, 2) * 8, 8 << f(10, 2), bit(13));
            let v = absneg(v, bit(49), bit(45));
            let v = if bit(50) {
                clamp_int(v, f(8, 2), bit(12))
            } else {
                v
            };
            if bit(47) {
                out.zn(v);
            }
            Some(v)
        }
        _ if matches!(o & 0xfff8, 0x5cb8 | 0x4cb8) || o & 0xfef8 == 0x38b8 => {
            let v = absneg(b(false), bit(49), bit(45));
            let v = extract(v, f(41, 2) * 8, 8 << f(10, 2), bit(13));
            // I2F.F64 converts the integer directly, with two raw register
            // words. No intermediate f32 rounding. Ryujinx InstEmitConversion.cs:
            // 234-284,398-416; Mesa sm50.rs:1873-1875 selects destination size.
            if field(raw, 8, 2) == 3 {
                let value = if bit(13) {
                    f64::from(v as i32)
                } else {
                    f64::from(v)
                };
                let bits = value.to_bits();
                if d < 254 {
                    out.words[d + 1] = (bits >> 32) as u32;
                }
                Some(bits as u32)
            } else {
                let f = if bit(13) { (v as i32) as f32 } else { v as f32 };
                Some(if field(raw, 8, 2) == 1 {
                    u32::from(pack(f))
                } else {
                    f.to_bits()
                })
            }
        }
        // F2I.F64 register source: gm107.c:1996,1283-1307; source pair at
        // 20..27, abs49, neg45, round39..40, size8..9, sign12. Ryujinx
        // InstEmitConversion.cs:185-221: abs then negate, round, clamp.
        // Project contract: NaN becomes 0 and overflow clamps.
        // F2I.F64 constant-buffer source: gm107.c:2070 slot34..38 and word
        // offset20..33; Ryujinx InstEmitHelper.cs:42-49 packs word offset
        // (low) with offset + 1 (high). Fields otherwise as the register row.
        _ if (o & 0xfffd == 0x5cb0 || o & 0xfffd == 0x4cb0) && f(10, 2) == 3 => {
            let (low, high) = if o >> 12 == 0x4 {
                let (bank, word) = (f(34, 5), f(20, 14));
                (cbuf(bank, word), cbuf(bank, word + 1))
            } else {
                let r = f(20, 8) as usize;
                (state[r], state[r + 1])
            };
            let x = f64::from_bits(u64::from(low) | u64::from(high) << 32);
            let x = if bit(49) { x.abs() } else { x };
            let x = if bit(45) { -x } else { x };
            let x = match f(39, 2) {
                0 => x.round_ties_even(),
                1 => x.floor(),
                2 => x.ceil(),
                _ => x.trunc(),
            };
            let (lo, hi) = bounds(f(8, 2), bit(12));
            Some(if x.is_nan() {
                0
            } else {
                x.clamp(lo as f64, hi as f64) as i64 as u32
            })
        }
        _ if matches!(o & 0xfff8, 0x5cb0 | 0x4cb0) || o & 0xfef8 == 0x38b0 => {
            let source = b(true);
            let source = if f(10, 2) == 1 {
                unpack((source >> (f(41, 1) * 16)) as u16).to_bits()
            } else if bit(44) {
                flush(source)
            } else {
                source
            };
            let value = rounded(f32::from_bits(fmod(source, bit(49), bit(45))), f(39, 2));
            let (lo, hi) = bounds(f(8, 2), bit(12));
            // Existing project contract: F2I overflow clamps, NaN becomes 0.
            // Bare Ryujinx OpConvertFToS/U is undefined outside integer range;
            // docs/reference-sources.md states this distinction.
            Some(if value.is_nan() {
                0
            } else {
                (value as f64).clamp(lo as f64, hi as f64) as i64 as u32
            })
        }
        // F2F: gm107.c:1997,2071,2117,1295-1317; InstEmitConversion.cs:
        // 13-37,105-163. The immediate form reads float20 at 20..38/sign56.
        _ if matches!(o & 0xfff8, 0x5ca8 | 0x4ca8) || o & 0xfef8 == 0x38a8 => {
            let source = b(true);
            // F2F.F32.F16: source size10..11 = 1 widens the low half exactly
            // (InstEmitConversion.cs:344-378, Sh41 clear). Any NaN is accepted.
            let half = f(10, 2) == 1;
            let source = if half {
                let widened = unpack(source as u16);
                if widened.is_nan() {
                    nan = 1;
                }
                widened.to_bits()
            } else {
                source
            };
            let source = if bit(44) { flush(source) } else { source };
            let source = fmod(source, bit(49), bit(45));
            if bit(42) || bit(50) {
                nan = 1;
            }
            let v = if bit(42) {
                rounded(f32::from_bits(source), f(39, 2)).to_bits()
            } else {
                source
            };
            if bit(47) {
                out.fpzn(v);
            }
            Some(finish_float(v, bit(50), false)?)
        }
        // FCMP: gm107.c:2025,2044,2096,2143; InstEmitFloatComparison.cs:369-375.
        _ if matches!(o & 0xfff0, 0x5ba0 | 0x4ba0 | 0x53a0) || o & 0xfef0 == 0x36a0 => {
            let swapped = o & 0xfff0 == 0x53a0;
            let (b, c) = if swapped {
                (c, cbuf(f(34, 5), f(20, 14)))
            } else {
                (b(true), c)
            };
            // Mesa nv50_ir_emit_gm107.cpp:1568-1569,474-477 encodes FTZ47.
            // Ryujinx InstEmitFloatComparison.cs:163-204 omits that modifier;
            // :369-375 selects raw A/B. Flush only comparison C, after the
            // swapped cbuf form has selected it, preserving selected payloads.
            let c = if bit(47) { flush(c) } else { c };
            Some(if float_cmp(f(48, 4), c, 0) { a } else { b })
        }
        // FADD/FMUL: gm107.c:2004,2006,2160,2167; sm50.rs:507-530,679-700.
        // InstEmitFloatArithmetic.cs:386-408,435-480.
        _ if matches!(o & 0xfff8, 0x5c58 | 0x4c58)
            || o & 0xfef8 == 0x3858
            || matches!(o & 0xfff8, 0x5c68 | 0x4c68)
            || o & 0xfef8 == 0x3868
            || raw >> 58 == 2
            || raw >> 56 == 0x1e =>
        {
            let long_add = raw >> 58 == 2;
            let long_mul = raw >> 56 == 0x1e;
            let multiply = long_mul
                || !long_add && o & 0xfef8 & 0xfff0 == 0x3860
                || matches!(o & 0xfff8, 0x5c68 | 0x4c68);
            let ftz_bit = bit(if long_add {
                55
            } else if long_mul {
                53
            } else {
                44
            });
            let dnz = multiply && bit(if long_mul { 54 } else { 45 });
            // DNZ implies FTZ: Mesa from_nir.rs:1271-1285.
            let ftz = ftz_bit || dnz;
            let a = fmod(
                a,
                bit(if long_add { 54 } else { 46 }) && !multiply,
                bit(if long_add { 56 } else { 48 }) && !multiply,
            );
            let b = if long_add || long_mul {
                f(20, 32)
            } else {
                b(true)
            };
            let b = fmod(
                b,
                if long_add {
                    bit(57)
                } else {
                    !multiply && bit(49)
                },
                if long_add {
                    bit(53)
                } else {
                    !long_mul && bit(if multiply { 48 } else { 45 })
                },
            );
            let mut a = if ftz { flush(a) } else { a };
            let b = if ftz { flush(b) } else { b };
            if multiply && !long_mul {
                let exponent = [0, -1, -2, -3, 3, 2, 1]
                    .get(f(41, 3) as usize)
                    .ok_or("reserved FMUL scale")?;
                a = (f32::from_bits(a) * 2.0_f32.powi(*exponent)).to_bits();
            }
            let (x, y) = (f64::from(f32::from_bits(a)), f64::from(f32::from_bits(b)));
            let mode = if long_add || long_mul { 0 } else { f(39, 2) };
            let (mut value, error) = if multiply { (x * y, 0.0) } else { sum(x, y) };
            if !multiply && mode == 1 && value == 0.0 && error == 0.0 && (a ^ b) & 0x8000_0000 != 0
            {
                value = -0.0;
            }
            // Mesa nir_opcodes.py:783-793: DNZ has a positive zero product.
            let v = if dnz && (a & 0x7fff_ffff == 0 || b & 0x7fff_ffff == 0) {
                0
            } else {
                float_round(value, error, mode)
            };
            nan = 1;
            let value = finish_float(v, bit(if long_mul { 55 } else { 50 }) && !long_add, ftz)?;
            if !multiply && bit(if long_add { 52 } else { 47 }) {
                out.fpzn(value);
            }
            Some(value)
        }
        // FFMA32I: gm107.c:2166,1516-1520; InstDecoders.cs:1960-1974 and
        // InstEmitFloatArithmetic.cs:200-209,411-433. B is imm32 20..51 and
        // the addend is the destination register; negA56 negates the
        // product, negC57 the addend, sat55, FTZ53, FMZ54 (DNZ: Mesa
        // from_nir.rs:1167-1180, nir_opcodes.py:1213-1225).
        _ if matches!(
            raw,
            0x0c03_fc00_0007_0201
                | 0x0d0b_f400_0007_0201
                | 0x0e04_0000_0007_0201
                | 0x0c83_f000_0007_0201
                | 0x0c24_0400_0007_0201
                | 0x0c40_0000_0007_0201
        ) =>
        {
            let a = fmod(a, false, bit(56));
            let b = f(20, 32);
            let c = fmod(state[d], false, bit(57));
            let ftz = bit(53) || bit(54);
            let (a, b, c) = if ftz {
                (flush(a), flush(b), flush(c))
            } else {
                (a, b, c)
            };
            let product = if bit(54) && (a & 0x7fff_ffff == 0 || b & 0x7fff_ffff == 0) {
                0.0
            } else {
                f64::from(f32::from_bits(a)) * f64::from(f32::from_bits(b))
            };
            let (value, error) = sum(product, f64::from(f32::from_bits(c)));
            nan = 1;
            Some(finish_float(float_round(value, error, 0), bit(55), ftz)?)
        }
        // FFMA32I combinations: gm107.c:2166; InstDecoders.cs:1960-1974;
        // InstEmitFloatArithmetic.cs:200-209,411-433. imm32 20..51, addend
        // is the destination register, negA56, negC57, sat55. FTZ53/FMZ54
        // and CC52 are not pinned here and are refused.
        _ if matches!(
            raw,
            0x0f03_fc00_0007_0201
                | 0x0d84_0000_0007_0201
                | 0x0e8b_fc00_0007_0201
                | 0x0f83_e800_0007_0201
                | 0x0c04_0800_0007_0305
                | 0x0c08_0000_0007_0607
        ) =>
        {
            if bit(52) || bit(53) || bit(54) {
                return Err("FFMA32I CC/FTZ/FMZ not pinned");
            }
            let a = fmod(a, false, bit(56));
            let b = f(20, 32);
            let c = fmod(state[d], false, bit(57));
            let product = f64::from(f32::from_bits(a)) * f64::from(f32::from_bits(b));
            let (value, error) = sum(product, f64::from(f32::from_bits(c)));
            nan = 1;
            Some(finish_float(float_round(value, error, 0), bit(55), false)?)
        }
        // FFMA: sm50.rs:558-599; InstEmitFloatArithmetic.cs:411-432.
        _ if matches!(o & !0x007f, 0x5980 | 0x4980 | 0x5180) || o & !0x017f == 0x3280 => {
            let swapped = o & !0x007f == 0x5180;
            let (b, c) = if swapped {
                (c, cbuf(f(34, 5), f(20, 14)))
            } else {
                (b(true), c)
            };
            let a = fmod(a, false, false);
            let b = fmod(b, false, bit(48));
            let c = fmod(c, false, bit(49));
            // Mesa sm50.rs:601; from_nir.rs:1167-1180 and
            // nir_opcodes.py:1213-1225: DNZ implies FTZ and a +0 product.
            let ftz = bit(53) || bit(54);
            let (a, b, c) = if ftz {
                (flush(a), flush(b), flush(c))
            } else {
                (a, b, c)
            };
            let product = if bit(54) && (a & 0x7fff_ffff == 0 || b & 0x7fff_ffff == 0) {
                0.0
            } else {
                f64::from(f32::from_bits(a)) * f64::from(f32::from_bits(b))
            };
            let (mut value, error) = sum(product, f64::from(f32::from_bits(c)));
            if f(51, 2) == 1
                && value == 0.0
                && error == 0.0
                && product.is_sign_negative() != (c >> 31 != 0)
            {
                value = -0.0;
            }
            let v = float_round(value, error, f(51, 2));
            nan = 1;
            Some(finish_float(v, bit(50), ftz)?)
        }
        // FMNMX: gm107.c:2005,2079,2125 (abs A46/B49, neg A48/B45, FTZ44,
        // CC47, selector39..42); InstEmitFloatMinMax.cs:78-104 takes minNum
        // when the selector holds, else maxNum.
        _ if matches!(o & 0xfffc, 0x5c60 | 0x4c60) || o & 0xfefc == 0x3860 => {
            if bit(44) || bit(47) {
                return Err("FMNMX FTZ/CC needs an oracle");
            }
            let x = fmod(a, bit(46), bit(48));
            let y = fmod(b(true), bit(49), bit(45));
            let v = min_max_number(x, y, !pred(raw, state, 39));
            if f32::from_bits(v).is_nan() {
                nan = 1;
            }
            Some(v)
        }
        // IMNMX: gm107.c:1383-1388,2013; sm50.rs:1453-1492;
        // InstEmitIntegerMinMax.cs:52-81. XLO/XMED/XHI need a CC oracle.
        _ if matches!(o & 0xfffe, 0x5c20 | 0x4c20) || o & 0xfefe == 0x3820 => {
            if f(43, 2) != 0 {
                return Err("IMNMX extended mode needs a CC oracle");
            }
            let value = minmax(a, b(false), bit(48), !pred(raw, state, 39));
            if bit(47) {
                // Ryujinx InstEmitIntegerMinMax.cs:66: IMNMX.CC writes only
                // zero and sign from the selected result.
                out.zn(value);
            }
            Some(value)
        }
        // RRO: gm107.c:2000; InstEmitMultifunction.cs:86-94 maps either
        // range-reduction mode39 to raw abs/neg, paired with host MUFU.
        _ if o & 0xfff8 == 0x5c90 || o & !2 == 0x4c90 => Some(fmod(b(true), bit(49), bit(45))),
        // RRO immediate: gm107.c:2120,256 (value20..38, sign56, << 12);
        // InstEmitMultifunction.cs:18-23,86-94 (Imm20ToFloat, abs, neg).
        _ if matches!(
            raw,
            0x3890_003f_8007_0003
                | 0x3990_00c0_2007_0003
                | 0x3992_0040_4007_0003
                | 0x3890_20bf_0007_0003
                | 0x3892_207f_8007_0003
                | 0x3992_20ff_c007_0003
        ) =>
        {
            Some(fmod(b(true), bit(49), bit(45)))
        }
        // SEL/R2P: gm107.c:1991,1998,2065,2111;
        // InstEmitMove.cs:252-275. R2P bit40 selects the concrete output
        // bank: P0..P6, or Z/N/C/V in that order.
        0x5ca0 | 0x4ca0 | 0x38a0 => Some(if pred(raw, state, 39) { a } else { b(false) }),
        _ if matches!(o & 0xfff8, 0x5cf0 | 0x4cf0) || o & 0xfef8 == 0x38f0 => {
            let mask = b(false);
            let value = a >> (f(41, 2) * 8);
            let (base, count) = if bit(40) { (264, 4) } else { (256, 7) };
            for p in 0..count {
                if mask & (1 << p) != 0 {
                    out.words[base + p] = (value >> p) & 1;
                }
            }
            None
        }
        // ICMP A/B selection based on C compared with zero. Encoding:
        // envytools gm107.c:2030,2046,2101,2148,1365-1368,1445-1453;
        // A8..15, B20..27 or signed i20/cbuf, C39..46, sign48, compare49..51.
        // Ryujinx InstEmitIntegerComparison.cs:13-54,117-131 swaps B/C in RC
        // and selects the original bits. No arithmetic NaN normalization.
        _ if matches!(o & 0xfff0, 0x5b40 | 0x4b40 | 0x5340) || o & 0xfef0 == 0x3640 => {
            let (test, otherwise) = if o & 0xfff0 == 0x5340 {
                (cbuf(f(34, 5), f(20, 14)), c)
            } else {
                (c, b(false))
            };
            Some(if int_cmp(f(49, 3), test, 0, bit(48)) {
                a
            } else {
                otherwise
            })
        }
        // ISET/ISETP: gm107.c:2028-2029,2099-2100,2146-2147;
        // sm50.rs:1494-1532; InstEmitIntegerComparison.cs:133-285.
        _ if matches!(o & 0xfff0, 0x5b50 | 0x4b50 | 0x5b60 | 0x4b60)
            || matches!(o & 0xfef0, 0x3650 | 0x3660) =>
        {
            let b = b(false);
            let code = f(49, 3);
            let mut q = int_cmp(code, a, b, bit(48));
            if bit(43) && code != 0 && code != 7 {
                let equal = a == b;
                let lt = int_cmp(1, a, b, bit(48));
                let gt = int_cmp(4, a, b, bit(48));
                let z = state[264] != 0;
                let carry = state[266] != 0;
                q = match code {
                    1 => lt || equal && !carry,
                    2 => equal && z,
                    3 => lt || equal && (z || !carry),
                    4 => gt || equal && carry && !z,
                    5 => !equal || !z,
                    6 => gt || equal && carry,
                    _ => unreachable!(),
                };
            }
            let compared = q;
            let q = logical(f(45, 2), compared, pred(raw, state, 39))?;
            if o & 0x10 == 0 {
                out.words[256 + f(3, 3) as usize] = u32::from(q);
                // ISETP second destination 0..2 (gm107.c PRED00; Ryujinx
                // DestPredInv): the complement under the same Boolean.
                if f(0, 3) != 7 {
                    out.words[256 + f(0, 3) as usize] =
                        u32::from(logical(f(45, 2), !compared, pred(raw, state, 39))?);
                }
                None
            } else {
                let value = if q {
                    if bit(44) {
                        0x3f80_0000
                    } else {
                        u32::MAX
                    }
                } else {
                    0
                };
                if bit(47) {
                    if bit(44) {
                        out.fpzn(value);
                    } else {
                        out.zn(value);
                        if bit(43) {
                            out.words[264] &= state[264];
                        }
                    }
                }
                Some(value)
            }
        }
        // FSET/FSETP: gm107.c:2024,2036,2095,2103,2142,2150;
        // InstEmitFloatComparison.cs:378-464. FTZ affects input bits.
        _ if matches!(o & 0xfff0, 0x5bb0 | 0x4bb0)
            || o & 0xfef0 == 0x36b0
            || matches!(o & 0xff00, 0x5800 | 0x4800)
            || o & 0xfe20 == 0x3000 =>
        {
            let setp = matches!(o & 0xfff0, 0x5bb0 | 0x4bb0) || o & 0xfef0 == 0x36b0;
            let a = fmod(a, bit(if setp { 7 } else { 54 }), bit(43));
            let b = fmod(
                b(true),
                bit(44),
                bit(if setp || o & 0xff00 == 0x4800 { 6 } else { 53 }),
            );
            let ftz = bit(if setp { 47 } else { 55 });
            let (a, b) = if ftz { (flush(a), flush(b)) } else { (a, b) };
            let compared = float_cmp(f(48, 4), a, b);
            let q = logical(f(45, 2), compared, pred(raw, state, 39))?;
            if setp {
                out.words[256 + f(3, 3) as usize] = u32::from(q);
                // Second destination 0..2 (gm107.c PRED00; Ryujinx
                // DestPredInv): the complement under the same Boolean.
                if f(0, 3) != 7 {
                    out.words[256 + f(0, 3) as usize] =
                        u32::from(logical(f(45, 2), !compared, pred(raw, state, 39))?);
                }
                None
            } else {
                let value = if q {
                    if bit(52) {
                        0x3f80_0000
                    } else {
                        u32::MAX
                    }
                } else {
                    0
                };
                if bit(47) {
                    if bit(52) {
                        out.fpzn(value);
                    } else {
                        out.zn(value);
                    }
                }
                Some(value)
            }
        }
        // PSET fields: gm107.c:2061,370,372-373,404-406,1683-1687;
        // Ryujinx InstEmitPredicate.cs:12-32. BF44 chooses 1.0 or all ones.
        _ if o & 0xfff8 == 0x5088 => {
            let t = logical(f(24, 2), pred(raw, state, 12), pred(raw, state, 29))?;
            let q = logical(f(45, 2), t, pred(raw, state, 39))?;
            Some(if q {
                if bit(44) {
                    0x3f80_0000
                } else {
                    u32::MAX
                }
            } else {
                0
            })
        }
        // PSETP: sm50.rs:2084-2104; InstEmitPredicate.cs:35-50.
        0x5090 => {
            // Mesa ir.rs:5318-5334 specifies opC(C, opAB(!A, B)) for
            // the second output. Ryujinx InstEmitPredicate.cs:42-47 instead
            // complements opAB(A, B), which differs for AND/OR. The Maxwell
            // fold is the authority for this correction, not GPU output.
            let a = pred(raw, state, 12);
            let b = pred(raw, state, 29);
            let t = logical(f(24, 2), a, b)?;
            let inverse_a = logical(f(24, 2), !a, b)?;
            let p = pred(raw, state, 39);
            out.words[256 + f(3, 3) as usize] = u32::from(logical(f(45, 2), t, p)?);
            out.words[256 + f(0, 3) as usize] = u32::from(logical(f(45, 2), inverse_a, p)?);
            None
        }
        // VMAD fields: gm107.c:1984,1090-1110,1600-1628; mode53..54:
        // Ryujinx InstDecoders.cs:5182-5187. Semantics:
        // InstEmitVideoArithmetic.cs:15-112. Compute the complete 64-bit
        // value on the CPU, independently of the emitter's carry operations.
        _ if o & 0xff00 == 0x5f00 => {
            let extend = |v, sel, signed| match sel {
                0..=3 => extract(v, sel * 8, 8, signed),
                4..=5 => extract(v, (sel - 4) * 16, 16, signed),
                6 => v,
                _ => unreachable!(),
            };
            let x = extend(a, f(36, 3), bit(48));
            let y = if bit(50) {
                extend(b(false), f(28, 3), bit(49))
            } else if bit(49) {
                sext(f(20, 16), 16)
            } else {
                f(20, 16)
            };
            let widen = |v: u32, signed| {
                if signed {
                    v as i32 as i64 as u64
                } else {
                    u64::from(v)
                }
            };
            let mut total = widen(x, bit(48)).wrapping_mul(widen(y, bit(49)));
            let mode = f(53, 2);
            if mode == 1 {
                total = total.wrapping_neg();
            }
            // C is a zero-extended word even in NegB mode, Ryujinx:59-60.
            total = total.wrapping_add(u64::from(neg(c, mode == 2)));
            if mode == 3 {
                total = total.wrapping_add(1);
            }
            let signed =
                f(36, 3) == 6 && bit(48) || f(28, 3) == 6 && bit(49) || matches!(mode, 1 | 2);
            let shift = [0, 7, 15][f(51, 2) as usize];
            total = if signed {
                ((total as i64) >> shift) as u64
            } else {
                total >> shift
            };
            Some(if !bit(55) {
                total as u32
            } else if signed {
                (total as i64).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as u32
            } else if total >> 32 == 0 {
                total as u32
            } else if total >> 63 != 0 {
                0
            } else {
                u32::MAX
            })
        }
        // VMNMX: gm107.c:1532-1560,1600-1628,2109;
        // InstEmitVideoMinMax.cs:12-119, InstEmitAluHelper.cs:75-91.
        _ if o & 0xfe00 == 0x3a00 => {
            let extend = |v, sel: u32, signed| match sel {
                0..=3 => extract(v, sel * 8, 8, signed),
                4..=5 => extract(v, (sel - 4) * 16, 16, signed),
                6 => v,
                _ => 0,
            };
            if f(36, 3) == 7 || bit(50) && f(28, 3) == 7 {
                return Err("reserved video selector");
            }
            let a = extend(a, f(36, 3), bit(48));
            let b = if bit(50) {
                extend(state[f(20, 8) as usize], f(28, 3), bit(49))
            } else if bit(49) {
                sext(f(20, 16), 16)
            } else {
                f(20, 16)
            };
            let signed = if bit(48) == bit(49) {
                bit(48)
            } else {
                !bit(56)
            };
            let mut value = minmax(a, b, signed, bit(56));
            if bit(48) != bit(49) {
                let unsigned = if bit(48) { b } else { a };
                if unsigned & 0x8000_0000 != 0 {
                    value = unsigned;
                }
            }
            if bit(55) {
                if bit(54) && !signed {
                    value = value.min(i32::MAX as u32);
                }
                if !bit(54) && signed {
                    value = (value as i32).max(0) as u32;
                }
            }
            Some(match f(51, 3) {
                0..=3 => {
                    let (offset, width) = [(16, 16), (0, 16), (0, 8), (16, 8)][f(51, 3) as usize];
                    let mask = ((1 << width) - 1) << offset;
                    (c & !mask) | ((value << offset) & mask)
                }
                4 => value.wrapping_add(c),
                5 | 6 => minmax(value, c, bit(54), f(51, 3) == 6),
                7 => value,
                _ => unreachable!(),
            })
        }
        // A cbuf PCNT only pushes a target for a later CONT. The bounded
        // execution program has no CONT, so the register/predicate/CC state
        // is unchanged. Envytools gm107.c:1767 defines the exact cbuf form.
        _ if o & 0xfff0 == 0xe2b0 && raw & 0x0000_ffff_ffff_ffff == 0x20 => None,
        // The direct PCNT opcode aliases push the next instruction address.
        // The exact hardware program has no CONT, so the push cannot affect
        // registers, predicates, or condition codes. Envytools gm107.c:1931.
        _ if o & 0xfff0 == 0xe2b0 && o != 0xe2b0 && raw & 0x0000_ffff_ffff_ffff == 0 => None,
        // A PBK push has no lane-visible result. For the exact c[0][0]
        // packet program, EXIT follows immediately and no BRK observes the
        // pushed target, so every input state remains unchanged. Envytools
        // gm107.c:1932,244-310 supplies the exact form and fields.
        _ if raw & 0xfff0_ffff_ffff_ffff == 0xe2a0_0000_0000_0020 => None,
        _ if nop_packet_contains(raw) || nop_tail_packet_contains(raw) => None,
        // Only Mesa OpNop/scheduling OpSync may otherwise be observational identities.
        // sm50.rs:3178-3184; gm107.c:1878.
        0x50b0 | 0xf0f0..=0xf0f7 => None,
        0xe300 => return Err("EXIT needs a control-flow observation program"),
        _ => return Err("CPU oracle missing"),
    };
    if let Some(value) = value {
        if d != 255 {
            out.words[d] = value;
            out.nan[d] = nan;
        }
    }
    out.words[255] = 0;
    out.words[263] = 1;
    // Common guard16..19: gm107.c:363-364,408-411.
    if !pred(raw, state, 16) {
        return Ok(Expected::new(state));
    }
    Ok(out)
}

fn nop_packet_contains(raw: u64) -> bool {
    const REGIONS: [(u64, u64); 7] = [
        (0xfff9_ffff_fff7_e0ff, 0x50b1_0000_0007_0000),
        (0xfff9_ffff_fff7_e0ff, 0x50b1_28de_18a7_00f2),
        (0xfff9_ffff_fff7_e0ff, 0x50b1_6f7a_8a07_00b2),
        (0xfff9_ffff_fff7_e0ff, 0x50b1_953c_3f97_8072),
        (0xfff9_ffff_fff7_e0ff, 0x50b1_9607_1787_0070),
        (0xfff9_ffff_fff7_e2ff, 0x50b1_87f3_87f7_8061),
        (0xfff9_ffff_fff7_e8ff, 0x50b1_f38f_1097_40df),
    ];
    REGIONS.iter().any(|&(mask, value)| raw & mask == value)
}

fn nop_tail_packet_contains(raw: u64) -> bool {
    const REGIONS: [(u64, u64); 14] = [
        (0xfffb_ffff_fff7_e0ff, 0x50b2_0000_0007_0000),
        (0xfffb_ffff_fff7_e0ff, 0x50b2_28de_18a7_00f2),
        (0xfffb_ffff_fff7_e0ff, 0x50b2_6f7a_8a07_00b2),
        (0xfffb_ffff_fff7_e0ff, 0x50b2_953c_3f97_8072),
        (0xfffb_ffff_fff7_e0ff, 0x50b2_9607_1787_0070),
        (0xfffb_ffff_fff7_e2ff, 0x50b2_87f3_87f7_8061),
        (0xfffb_ffff_fff7_e8ff, 0x50b2_f38f_1097_40df),
        (0xffff_ffff_fff7_e0ff, 0x50b4_0000_0007_0000),
        (0xffff_ffff_fff7_e0ff, 0x50b4_28de_18a7_00f2),
        (0xffff_ffff_fff7_e0ff, 0x50b4_6f7a_8a07_00b2),
        (0xffff_ffff_fff7_e0ff, 0x50b4_953c_3f97_8072),
        (0xffff_ffff_fff7_e0ff, 0x50b4_9607_1787_0070),
        (0xffff_ffff_fff7_e2ff, 0x50b4_87f3_87f7_8061),
        (0xffff_ffff_fff7_e8ff, 0x50b4_f38f_1097_40df),
    ];
    REGIONS.iter().any(|&(mask, value)| raw & mask == value)
}

#[test]
fn nop_packet_has_a_cpu_oracle_for_every_word() {
    const REGIONS: &[(u64, &[u32])] = &[
        (0x50b1_0000_0007_0000, &[8, 9, 10, 11, 12, 19, 49, 50]),
        (0x50b1_28de_18a7_00f2, &[8, 9, 10, 11, 12, 19, 49, 50]),
        (0x50b1_6f7a_8a07_00b2, &[8, 9, 10, 11, 12, 19, 49, 50]),
        (0x50b1_953c_3f97_8072, &[8, 9, 10, 11, 12, 19, 49, 50]),
        (0x50b1_9607_1787_0070, &[8, 9, 10, 11, 12, 19, 49, 50]),
        (0x50b1_87f3_87f7_8061, &[8, 10, 11, 12, 19, 49, 50]),
        (0x50b1_f38f_1097_40df, &[8, 9, 10, 12, 19, 49, 50]),
    ];
    let mut covered = 0;
    for &(base, bits) in REGIONS {
        for selector in 0..(1_u16 << bits.len()) {
            let raw = bits.iter().enumerate().fold(base, |raw, (i, bit)| {
                raw | u64::from((selector >> i) & 1) << bit
            });
            for predicate in [0_u32, 1] {
                let mut state = std::array::from_fn(|i| (i as u32).wrapping_mul(0x0101_0101));
                state[255] = 0;
                state[263] = 1;
                state[256..263].fill(predicate);
                state[264..268].copy_from_slice(&[1, 0, 1, 1]);
                assert_eq!(reference(raw, &state).unwrap().words, state, "{raw:016x}");
                covered += 1;
            }
        }
    }
    assert_eq!(covered, 3_072);
}

#[test]
fn nop_tail_packet_has_a_cpu_oracle_for_every_word() {
    const REGIONS: &[(u64, &[u32])] = &[
        (0x50b2_0000_0007_0000, &[8, 9, 10, 11, 12, 19, 50]),
        (0x50b2_28de_18a7_00f2, &[8, 9, 10, 11, 12, 19, 50]),
        (0x50b2_6f7a_8a07_00b2, &[8, 9, 10, 11, 12, 19, 50]),
        (0x50b2_953c_3f97_8072, &[8, 9, 10, 11, 12, 19, 50]),
        (0x50b2_9607_1787_0070, &[8, 9, 10, 11, 12, 19, 50]),
        (0x50b2_87f3_87f7_8061, &[8, 10, 11, 12, 19, 50]),
        (0x50b2_f38f_1097_40df, &[8, 9, 10, 12, 19, 50]),
        (0x50b4_0000_0007_0000, &[8, 9, 10, 11, 12, 19]),
        (0x50b4_28de_18a7_00f2, &[8, 9, 10, 11, 12, 19]),
        (0x50b4_6f7a_8a07_00b2, &[8, 9, 10, 11, 12, 19]),
        (0x50b4_953c_3f97_8072, &[8, 9, 10, 11, 12, 19]),
        (0x50b4_9607_1787_0070, &[8, 9, 10, 11, 12, 19]),
        (0x50b4_87f3_87f7_8061, &[8, 10, 11, 12, 19]),
        (0x50b4_f38f_1097_40df, &[8, 9, 10, 12, 19]),
    ];
    let mut covered = 0;
    for &(base, bits) in REGIONS {
        for selector in 0..(1_u16 << bits.len()) {
            let raw = bits.iter().enumerate().fold(base, |word, (index, bit)| {
                word | u64::from((selector >> index) & 1) << bit
            });
            for predicate in [0_u32, 1] {
                let mut state =
                    std::array::from_fn(|index| (index as u32).wrapping_mul(0x0101_0101));
                state[255] = 0;
                state[263] = 1;
                state[256..263].fill(predicate);
                state[264..268].copy_from_slice(&[1, 0, 1, 1]);
                assert_eq!(reference(raw, &state).unwrap().words, state, "{raw:016x}");
                covered += 1;
            }
        }
    }
    assert_eq!(covered, 2_304);
}

pub fn normalise(bits: u32, policy: u8) -> u32 {
    fn nan32(bits: u32) -> u32 {
        if bits & 0x7fff_ffff > 0x7f80_0000 {
            0x7fc0_0000
        } else {
            bits
        }
    }
    fn nan16(bits: u16) -> u16 {
        if bits & 0x7fff > 0x7c00 {
            0x7e00
        } else {
            bits
        }
    }
    match policy {
        1 => nan32(bits),
        2 => u32::from(nan16(bits as u16)) | u32::from(nan16((bits >> 16) as u16)) << 16,
        _ => bits,
    }
}

const FADD_FTZ_RM_PACKET_BASES: [u64; 11] = [
    0x5c58_1080_0000_0000,
    0x5c58_1080_0007_0000,
    0x5c58_10c1_e777_1d6e,
    0x5c58_1286_3767_da24,
    0x5c58_14bd_d8e7_972e,
    0x5c58_14f1_e8b7_f5ce,
    0x5c58_18d7_7b67_7d3d,
    0x5c58_1ad8_99f7_42aa,
    0x5c58_1ccf_af87_30a7,
    0x5c58_1e8d_1517_72c3,
    0x5c58_1e90_6e97_1225,
];
const FADD_FTZ_RM_PACKET_VARIABLE_BITS: [u32; 6] = [19, 45, 46, 47, 48, 49];

fn fadd_ftz_rm_packet_words() -> impl Iterator<Item = u64> {
    FADD_FTZ_RM_PACKET_BASES.into_iter().flat_map(|base| {
        (0_u8..64).map(move |selector| {
            FADD_FTZ_RM_PACKET_VARIABLE_BITS
                .iter()
                .enumerate()
                .fold(base, |raw, (index, bit)| {
                    raw | u64::from(selector >> index & 1) << bit
                })
        })
    })
}

#[test]
fn fadd_ftz_rm_cpu_edges() {
    let rm = |a: f32, b: f32| {
        let a = flush(a.to_bits());
        let b = flush(b.to_bits());
        let (value, error) = sum(f64::from(f32::from_bits(a)), f64::from(f32::from_bits(b)));
        let mut bits = float_round(value, error, 1);
        if value == 0.0 && error == 0.0 && !(a == 0 && b == 0) {
            bits = 0x8000_0000;
        }
        flush(bits)
    };
    let quarter_ulp = 2.0_f32.powi(-25);
    assert_eq!(rm(1.0, -quarter_ulp), 1.0_f32.to_bits() - 1);
    assert_eq!(rm(-1.0, -quarter_ulp), (-1.0_f32).to_bits() + 1);
    assert_eq!(rm(f32::MAX, f32::MAX), f32::MAX.to_bits());
    assert_eq!(rm(-f32::MAX, -f32::MAX), f32::NEG_INFINITY.to_bits());
    assert_eq!(rm(1.0, -1.0), (-0.0_f32).to_bits());
    assert_eq!(rm(0.0, 0.0), 0.0_f32.to_bits());
    assert_eq!(rm(-0.0, 0.0), (-0.0_f32).to_bits());
    assert_eq!(rm(f32::from_bits(1), f32::from_bits(1)), 0.0_f32.to_bits());
}

#[test]
fn fadd_ftz_rm_packet_has_a_cpu_oracle_for_every_word() {
    let values = [
        (0.0_f32, -0.0_f32),
        (1.0, 2.0),
        (-1.0, f32::from_bits(1)),
        (1.0, f32::EPSILON / 2.0),
        (f32::MIN_POSITIVE, f32::from_bits(1)),
        (f32::from_bits(1), f32::from_bits(1)),
        (f32::MAX / 2.0, f32::MAX / 4.0),
        (f32::MAX, f32::MAX),
        (-f32::MAX, -f32::MAX),
        (2.0_f32.powi(60), 2.0_f32.powi(35)),
        (-2.0_f32.powi(60), 2.0_f32.powi(59)),
        (1.0, -1.0),
    ];
    let mut covered = 0;
    for raw in fadd_ftz_rm_packet_words() {
        let lhs = field(raw, 8, 8) as usize;
        let rhs = field(raw, 20, 8) as usize;
        for predicate_value in [0_u32, 1] {
            for (a, b) in values {
                let mut state = [0_u32; WORDS];
                state[lhs] = a.to_bits();
                state[rhs] = b.to_bits();
                state[255] = 0;
                state[256..263].fill(predicate_value);
                state[263] = 1;
                state[264..268].copy_from_slice(&[1, 0, 1, 1]);
                reference(raw, &state).unwrap_or_else(|reason| panic!("{raw:016x}: {reason}"));
                covered += 1;
            }
        }
    }
    assert_eq!(covered, 16_896);
}

#[test]
fn dadd_rn_cpu_edges() {
    let rn = |a: f64, b: f64| f64::from_bits(dadd_rn_bits(a.to_bits(), b.to_bits()).unwrap());
    assert_eq!(rn(1.0, f64::EPSILON / 2.0).to_bits(), 1.0_f64.to_bits());
    assert_eq!(rn(1.0, f64::EPSILON).to_bits(), 1.0_f64.to_bits() + 1);
    assert_eq!(
        rn(-1.0, -f64::EPSILON / 2.0).to_bits(),
        (-1.0_f64).to_bits()
    );
    assert_eq!(rn(f64::from_bits(1), f64::from_bits(1)).to_bits(), 2);
    assert_eq!(rn(f64::MAX, f64::MAX), f64::INFINITY);
    assert_eq!(rn(-f64::MAX, -f64::MAX), f64::NEG_INFINITY);
    assert_eq!(rn(1.0, -1.0).to_bits(), 0.0_f64.to_bits());
    assert_eq!(rn(-0.0, -0.0).to_bits(), (-0.0_f64).to_bits());
    assert_eq!(rn(-0.0, 0.0).to_bits(), 0.0_f64.to_bits());
    assert_eq!(rn(f64::INFINITY, 1.0), f64::INFINITY);
    assert!(dadd_rn_bits(f64::NAN.to_bits(), 1.0_f64.to_bits()).is_err());
    assert!(dadd_rn_bits(f64::INFINITY.to_bits(), f64::NEG_INFINITY.to_bits()).is_err());
}

#[test]
fn dadd_rn_cc_immediate_packet_has_a_cpu_oracle_for_every_word() {
    const BASES: [u64; 8] = [
        0x3870_8000_0007_0000,
        0x3870_824d_7ec7_7914,
        0x3870_8c27_1987_c17e,
        0x3870_8e43_ba37_5b2b,
        0x3870_9866_f617_7807,
        0x3870_9e00_0ff7_ffff,
        0x3870_9e04_6767_b14c,
        0x3870_9e7a_5ea7_cf2a,
    ];
    const VARIABLE_BITS: [u32; 7] = [19, 45, 46, 48, 49, 50, 56];

    let mut covered = 0;
    for base in BASES {
        for selector in 0_u8..128 {
            let raw = VARIABLE_BITS
                .iter()
                .enumerate()
                .fold(base, |raw, (index, bit)| {
                    raw | u64::from(selector >> index & 1) << bit
                });
            for predicate_value in [0_u32, 1] {
                let mut state = [0_u32; WORDS];
                state[255] = 0;
                state[263] = 1;
                state[256..263].fill(predicate_value);
                state[264..268].copy_from_slice(&[1, 0, 1, 1]);
                reference(raw, &state).unwrap_or_else(|reason| panic!("{raw:016x}: {reason}"));
                covered += 1;
            }
        }
    }
    assert_eq!(covered, 2_048);
}

#[test]
fn dadd_rm_cpu_edges() {
    let rm = |a: f64, b: f64| f64::from_bits(dadd_rm_bits(a.to_bits(), b.to_bits()).unwrap());
    assert_eq!(
        rm(1.0, -f64::from_bits(1)).to_bits(),
        (1.0_f64.to_bits() - 1)
    );
    assert_eq!(
        rm(-1.0, -f64::from_bits(1)).to_bits(),
        (-1.0_f64).to_bits() + 1
    );
    assert_eq!(rm(1.0, 1.0), 2.0);
    assert_eq!(rm(f64::MAX, f64::MAX), f64::MAX);
    assert_eq!(rm(-f64::MAX, -f64::MAX), f64::NEG_INFINITY);
    assert_eq!(rm(1.0, -1.0).to_bits(), (-0.0_f64).to_bits());
    assert_eq!(rm(0.0, 0.0).to_bits(), 0.0_f64.to_bits());
    assert_eq!(rm(-0.0, 0.0).to_bits(), (-0.0_f64).to_bits());
    assert_eq!(rm(f64::from_bits(1), f64::from_bits(1)).to_bits(), 2);
}

#[test]
fn dadd_rm_immediate_packet_has_a_cpu_oracle_for_every_word() {
    const BASES: [u64; 8] = [
        0x3870_0080_0007_0000,
        0x3870_02cd_7ec7_7914,
        0x3870_0ca7_1987_c17e,
        0x3870_0ec3_ba37_5b2b,
        0x3870_18e6_f617_7807,
        0x3870_1e80_0ff7_ffff,
        0x3870_1e84_6767_b14c,
        0x3870_1efa_5ea7_cf2a,
    ];
    const VARIABLE_BITS: [u32; 8] = [19, 45, 46, 47, 48, 49, 50, 56];

    let mut covered = 0;
    for base in BASES {
        for selector in 0_u16..256 {
            let raw = VARIABLE_BITS
                .iter()
                .enumerate()
                .fold(base, |raw, (index, bit)| {
                    raw | u64::from(selector >> index & 1) << bit
                });
            for predicate_value in [0_u32, 1] {
                let mut state = [0_u32; WORDS];
                state[255] = 0;
                state[263] = 1;
                state[256..263].fill(predicate_value);
                state[264..268].copy_from_slice(&[1, 0, 1, 1]);
                reference(raw, &state).unwrap_or_else(|reason| panic!("{raw:016x}: {reason}"));
                covered += 1;
            }
        }
    }
    assert_eq!(covered, 4_096);
}

#[test]
fn dadd_rm_register_packet_has_a_cpu_oracle_for_every_word() {
    const BASES: [u64; 8] = [
        0x5c70_0080_0007_0000,
        0x5c70_0080_a427_440b,
        0x5c70_02e1_8077_a0d1,
        0x5c70_06dc_cc47_3d36,
        0x5c70_0cb2_f917_817b,
        0x5c70_0cb4_1db7_aa3a,
        0x5c70_1ca4_8b47_e19b,
        0x5c70_1caa_6b67_2c35,
    ];
    const VARIABLE_BITS: [u32; 7] = [19, 45, 46, 47, 48, 49, 50];
    let values = [
        (0.0_f64.to_bits(), (-0.0_f64).to_bits()),
        (1.0_f64.to_bits(), 2.0_f64.to_bits()),
        ((-1.0_f64).to_bits(), f64::from_bits(1).to_bits()),
        (1.0_f64.to_bits(), (f64::EPSILON / 2.0).to_bits()),
        (f64::MIN_POSITIVE.to_bits(), f64::from_bits(1).to_bits()),
        (f64::from_bits(1).to_bits(), f64::from_bits(1).to_bits()),
        ((f64::MAX / 2.0).to_bits(), (f64::MAX / 4.0).to_bits()),
        (f64::MAX.to_bits(), f64::MAX.to_bits()),
        ((-f64::MAX).to_bits(), (-f64::MAX).to_bits()),
        (2.0_f64.powi(500).to_bits(), 2.0_f64.powi(448).to_bits()),
        ((-2.0_f64.powi(500)).to_bits(), 2.0_f64.powi(499).to_bits()),
        (1.0_f64.to_bits(), (-1.0_f64).to_bits()),
    ];

    let set_pair = |state: &mut [u32; WORDS], register: usize, value: u64| {
        state[register] = value as u32;
        state[register | 1] = (value >> 32) as u32;
        if register & 1 == 1 {
            state[register] = (value >> 32) as u32;
        }
    };

    let mut covered = 0;
    for base in BASES {
        for selector in 0_u8..128 {
            let raw = VARIABLE_BITS
                .iter()
                .enumerate()
                .fold(base, |raw, (index, bit)| {
                    raw | u64::from(selector >> index & 1) << bit
                });
            let lhs = field(raw, 8, 8) as usize;
            let rhs = field(raw, 20, 8) as usize;
            let operands = values[covered / 2 % values.len()];
            for predicate_value in [0_u32, 1] {
                let mut state = [0_u32; WORDS];
                set_pair(&mut state, lhs, operands.0);
                set_pair(&mut state, rhs, operands.1);
                state[255] = 0;
                state[263] = 1;
                state[256..263].fill(predicate_value);
                state[264..268].copy_from_slice(&[1, 0, 1, 1]);
                reference(raw, &state).unwrap_or_else(|reason| panic!("{raw:016x}: {reason}"));
                covered += 1;
            }
        }
    }
    assert_eq!(covered, 2_048);
}

#[test]
fn dadd_rm_cbuf_packet_has_a_cpu_oracle_for_every_word() {
    const BASES: [u64; 9] = [
        0x4c70_0080_0000_0000,
        0x4c70_0080_0007_0000,
        0x4c70_0ade_fac7_4812,
        0x4c70_0e83_90c7_921f,
        0x4c70_16d1_9b07_39ac,
        0x4c70_16ec_8347_824a,
        0x4c70_18fd_a827_4194,
        0x4c70_1ce6_b537_0757,
        0x4c70_1e80_0ff7_ffff,
    ];
    const VARIABLE_BITS: [u32; 7] = [19, 45, 46, 47, 48, 49, 50];
    let values = [
        0.0_f64.to_bits(),
        (-0.0_f64).to_bits(),
        1.0_f64.to_bits(),
        (-1.0_f64).to_bits(),
        (1.0_f64.to_bits() - 1),
        (1.0_f64.to_bits() + 1),
        f64::from_bits(1).to_bits(),
        f64::MIN_POSITIVE.to_bits(),
        (f64::MAX / 2.0).to_bits(),
        (-(f64::MAX / 2.0)).to_bits(),
        f64::MAX.to_bits(),
        (-f64::MAX).to_bits(),
    ];
    let set_pair = |state: &mut [u32; WORDS], register: usize, value: u64| {
        state[register] = value as u32;
        state[register | 1] = (value >> 32) as u32;
        if register & 1 == 1 {
            state[register] = (value >> 32) as u32;
        }
    };

    let mut covered = 0;
    for base in BASES {
        for selector in 0_u8..128 {
            let raw = VARIABLE_BITS
                .iter()
                .enumerate()
                .fold(base, |raw, (index, bit)| {
                    raw | u64::from(selector >> index & 1) << bit
                });
            let lhs = field(raw, 8, 8) as usize;
            let value = values[covered / 2 % values.len()];
            for predicate_value in [0_u32, 1] {
                let mut state = [0_u32; WORDS];
                set_pair(&mut state, lhs, value);
                state[255] = 0;
                state[263] = 1;
                state[256..263].fill(predicate_value);
                state[264..268].copy_from_slice(&[1, 0, 1, 1]);
                reference(raw, &state).unwrap_or_else(|reason| panic!("{raw:016x}: {reason}"));
                covered += 1;
            }
        }
    }
    assert_eq!(covered, 2_304);
}

#[test]
fn dadd_rp_cbuf_packet_has_a_cpu_oracle_for_every_word() {
    const BASES: [u64; 9] = [
        0x4c70_0100_0000_0000,
        0x4c70_0100_0007_0000,
        0x4c70_0b5e_fac7_4812,
        0x4c70_0f03_90c7_921f,
        0x4c70_1751_9b07_39ac,
        0x4c70_176c_8347_824a,
        0x4c70_197d_a827_4194,
        0x4c70_1d66_b537_0757,
        0x4c70_1f00_0ff7_ffff,
    ];
    const VARIABLE_BITS: [u32; 7] = [19, 45, 46, 47, 48, 49, 50];
    let values = [
        0.0_f64.to_bits(),
        (-0.0_f64).to_bits(),
        1.0_f64.to_bits(),
        (-1.0_f64).to_bits(),
        (1.0_f64.to_bits() - 1),
        (1.0_f64.to_bits() + 1),
        f64::from_bits(1).to_bits(),
        f64::MIN_POSITIVE.to_bits(),
        (f64::MAX / 2.0).to_bits(),
        (-(f64::MAX / 2.0)).to_bits(),
        f64::MAX.to_bits(),
        (-f64::MAX).to_bits(),
    ];
    let set_pair = |state: &mut [u32; WORDS], register: usize, value: u64| {
        state[register] = value as u32;
        state[register | 1] = (value >> 32) as u32;
        if register & 1 == 1 {
            state[register] = (value >> 32) as u32;
        }
    };

    let mut covered = 0;
    for base in BASES {
        for selector in 0_u8..128 {
            let raw = VARIABLE_BITS
                .iter()
                .enumerate()
                .fold(base, |raw, (index, bit)| {
                    raw | u64::from(selector >> index & 1) << bit
                });
            let lhs = field(raw, 8, 8) as usize;
            let value = values[covered / 2 % values.len()];
            for predicate_value in [0_u32, 1] {
                let mut state = [0_u32; WORDS];
                set_pair(&mut state, lhs, value);
                state[255] = 0;
                state[263] = 1;
                state[256..263].fill(predicate_value);
                state[264..268].copy_from_slice(&[1, 0, 1, 1]);
                reference(raw, &state).unwrap_or_else(|reason| panic!("{raw:016x}: {reason}"));
                covered += 1;
            }
        }
    }
    assert_eq!(covered, 2_304);
}

#[test]
fn dadd_rz_cbuf_packet_has_a_cpu_oracle_for_every_word() {
    const BASES: [u64; 9] = [
        0x4c70_0180_0000_0000,
        0x4c70_0180_0007_0000,
        0x4c70_0bde_fac7_4812,
        0x4c70_0f83_90c7_921f,
        0x4c70_17d1_9b07_39ac,
        0x4c70_17ec_8347_824a,
        0x4c70_19fd_a827_4194,
        0x4c70_1de6_b537_0757,
        0x4c70_1f80_0ff7_ffff,
    ];
    const VARIABLE_BITS: [u32; 7] = [19, 45, 46, 47, 48, 49, 50];
    let values = [
        0.0_f64.to_bits(),
        (-0.0_f64).to_bits(),
        1.0_f64.to_bits(),
        (-1.0_f64).to_bits(),
        1.0_f64.to_bits() - 1,
        1.0_f64.to_bits() + 1,
        f64::from_bits(1).to_bits(),
        f64::MIN_POSITIVE.to_bits(),
        (f64::MAX / 2.0).to_bits(),
        (-(f64::MAX / 2.0)).to_bits(),
        f64::MAX.to_bits(),
        (-f64::MAX).to_bits(),
    ];
    let set_pair = |state: &mut [u32; WORDS], register: usize, value: u64| {
        state[register] = value as u32;
        state[register | 1] = (value >> 32) as u32;
        if register & 1 == 1 {
            state[register] = (value >> 32) as u32;
        }
    };
    let mut covered = 0;
    for base in BASES {
        for selector in 0_u8..128 {
            let raw = VARIABLE_BITS
                .iter()
                .enumerate()
                .fold(base, |raw, (index, bit)| {
                    raw | u64::from(selector >> index & 1) << bit
                });
            let lhs = field(raw, 8, 8) as usize;
            let value = values[covered / 2 % values.len()];
            for predicate_value in [0_u32, 1] {
                let mut state = [0_u32; WORDS];
                set_pair(&mut state, lhs, value);
                state[255] = 0;
                state[263] = 1;
                state[256..263].fill(predicate_value);
                state[264..268].copy_from_slice(&[1, 0, 1, 1]);
                reference(raw, &state).unwrap_or_else(|reason| panic!("{raw:016x}: {reason}"));
                covered += 1;
            }
        }
    }
    assert_eq!(covered, 2_304);
}

#[test]
fn dadd_rp_cpu_edges() {
    let rp = |a: f64, b: f64| f64::from_bits(dadd_rp_bits(a.to_bits(), b.to_bits()).unwrap());
    assert_eq!(
        rp(1.0, f64::from_bits(1)).to_bits(),
        (1.0_f64.to_bits() + 1)
    );
    assert_eq!(
        rp(-1.0, f64::from_bits(1)).to_bits(),
        (-1.0_f64).to_bits() - 1
    );
    assert_eq!(rp(1.0, 1.0), 2.0);
    assert_eq!(rp(f64::MAX, f64::MAX), f64::INFINITY);
    assert_eq!(rp(-f64::MAX, -f64::MAX), -f64::MAX);
    assert_eq!(rp(1.0, -1.0).to_bits(), 0.0_f64.to_bits());
    assert_eq!(rp(0.0, 0.0).to_bits(), 0.0_f64.to_bits());
    assert_eq!(rp(-0.0, -0.0).to_bits(), (-0.0_f64).to_bits());
    assert_eq!(rp(f64::from_bits(1), f64::from_bits(1)).to_bits(), 2);
}

#[test]
fn dadd_rp_immediate_packet_has_a_cpu_oracle_for_every_word() {
    const BASES: [u64; 8] = [
        0x3870_0100_0007_0000,
        0x3870_034d_7ec7_7914,
        0x3870_0d27_1987_c17e,
        0x3870_0f43_ba37_5b2b,
        0x3870_1966_f617_7807,
        0x3870_1f00_0ff7_ffff,
        0x3870_1f04_6767_b14c,
        0x3870_1f7a_5ea7_cf2a,
    ];
    const VARIABLE_BITS: [u32; 8] = [19, 45, 46, 47, 48, 49, 50, 56];

    let mut covered = 0;
    for base in BASES {
        for selector in 0_u16..256 {
            let raw = VARIABLE_BITS
                .iter()
                .enumerate()
                .fold(base, |raw, (index, bit)| {
                    raw | u64::from(selector >> index & 1) << bit
                });
            for predicate_value in [0_u32, 1] {
                let mut state = [0_u32; WORDS];
                state[255] = 0;
                state[263] = 1;
                state[256..263].fill(predicate_value);
                state[264..268].copy_from_slice(&[1, 0, 1, 1]);
                reference(raw, &state).unwrap_or_else(|reason| panic!("{raw:016x}: {reason}"));
                covered += 1;
            }
        }
    }
    assert_eq!(covered, 4_096);
}

#[test]
fn dadd_rp_register_packet_has_a_cpu_oracle_for_every_word() {
    const BASES: [u64; 8] = [
        0x5c70_0100_0007_0000,
        0x5c70_0100_a427_440b,
        0x5c70_0361_8077_a0d1,
        0x5c70_075c_cc47_3d36,
        0x5c70_0d32_f917_817b,
        0x5c70_0d34_1db7_aa3a,
        0x5c70_1d24_8b47_e19b,
        0x5c70_1d2a_6b67_2c35,
    ];
    const VARIABLE_BITS: [u32; 7] = [19, 45, 46, 47, 48, 49, 50];
    let values = [
        (0.0_f64.to_bits(), (-0.0_f64).to_bits()),
        (1.0_f64.to_bits(), 2.0_f64.to_bits()),
        ((-1.0_f64).to_bits(), f64::from_bits(1).to_bits()),
        (1.0_f64.to_bits(), (f64::EPSILON / 2.0).to_bits()),
        (f64::MIN_POSITIVE.to_bits(), f64::from_bits(1).to_bits()),
        (f64::from_bits(1).to_bits(), f64::from_bits(1).to_bits()),
        ((f64::MAX / 2.0).to_bits(), (f64::MAX / 4.0).to_bits()),
        (f64::MAX.to_bits(), f64::MAX.to_bits()),
        ((-f64::MAX).to_bits(), (-f64::MAX).to_bits()),
        (2.0_f64.powi(500).to_bits(), 2.0_f64.powi(448).to_bits()),
        ((-2.0_f64.powi(500)).to_bits(), 2.0_f64.powi(499).to_bits()),
        (1.0_f64.to_bits(), (-1.0_f64).to_bits()),
    ];

    let set_pair = |state: &mut [u32; WORDS], register: usize, value: u64| {
        state[register] = value as u32;
        state[register | 1] = (value >> 32) as u32;
        if register & 1 == 1 {
            state[register] = (value >> 32) as u32;
        }
    };

    let mut covered = 0;
    for base in BASES {
        for selector in 0_u8..128 {
            let raw = VARIABLE_BITS
                .iter()
                .enumerate()
                .fold(base, |raw, (index, bit)| {
                    raw | u64::from(selector >> index & 1) << bit
                });
            let lhs = field(raw, 8, 8) as usize;
            let rhs = field(raw, 20, 8) as usize;
            let operands = values[covered / 2 % values.len()];
            for predicate_value in [0_u32, 1] {
                let mut state = [0_u32; WORDS];
                set_pair(&mut state, lhs, operands.0);
                set_pair(&mut state, rhs, operands.1);
                state[255] = 0;
                state[263] = 1;
                state[256..263].fill(predicate_value);
                state[264..268].copy_from_slice(&[1, 0, 1, 1]);
                reference(raw, &state).unwrap_or_else(|reason| panic!("{raw:016x}: {reason}"));
                covered += 1;
            }
        }
    }
    assert_eq!(covered, 2_048);
}

#[test]
fn dadd_rz_cpu_edges() {
    let rz = |a: f64, b: f64| f64::from_bits(dadd_rz_bits(a.to_bits(), b.to_bits()).unwrap());
    assert_eq!(rz(1.0, -f64::from_bits(1)).to_bits(), 1.0_f64.to_bits() - 1);
    assert_eq!(
        rz(-1.0, f64::from_bits(1)).to_bits(),
        (-1.0_f64).to_bits() - 1
    );
    assert_eq!(rz(1.0, f64::EPSILON / 2.0).to_bits(), 1.0_f64.to_bits());
    assert_eq!(
        rz(-1.0, -f64::EPSILON / 2.0).to_bits(),
        (-1.0_f64).to_bits()
    );
    assert_eq!(rz(f64::MAX, f64::MAX), f64::MAX);
    assert_eq!(rz(-f64::MAX, -f64::MAX), -f64::MAX);
    assert_eq!(rz(1.0, -1.0).to_bits(), 0.0_f64.to_bits());
    assert_eq!(rz(-0.0, -0.0).to_bits(), (-0.0_f64).to_bits());
    assert_eq!(rz(-0.0, 0.0).to_bits(), 0.0_f64.to_bits());
    assert_eq!(rz(f64::from_bits(1), f64::from_bits(1)).to_bits(), 2);
    assert!(dadd_rz_bits(f64::NAN.to_bits(), 1.0_f64.to_bits()).is_err());
    assert!(dadd_rz_bits(f64::INFINITY.to_bits(), f64::NEG_INFINITY.to_bits()).is_err());
}

#[test]
fn dadd_rz_immediate_packet_has_a_cpu_oracle_for_every_word() {
    const BASES: [u64; 8] = [
        0x3870_0180_0007_0000,
        0x3870_03cd_7ec7_7914,
        0x3870_0da7_1987_c17e,
        0x3870_0fc3_ba37_5b2b,
        0x3870_19e6_f617_7807,
        0x3870_1f80_0ff7_ffff,
        0x3870_1f84_6767_b14c,
        0x3870_1ffa_5ea7_cf2a,
    ];
    const VARIABLE_BITS: [u32; 8] = [19, 45, 46, 47, 48, 49, 50, 56];

    let mut covered = 0;
    for base in BASES {
        for selector in 0_u16..256 {
            let raw = VARIABLE_BITS
                .iter()
                .enumerate()
                .fold(base, |raw, (index, bit)| {
                    raw | u64::from(selector >> index & 1) << bit
                });
            for predicate_value in [0_u32, 1] {
                let mut state = [0_u32; WORDS];
                state[255] = 0;
                state[263] = 1;
                state[256..263].fill(predicate_value);
                state[264..268].copy_from_slice(&[1, 0, 1, 1]);
                reference(raw, &state).unwrap_or_else(|reason| panic!("{raw:016x}: {reason}"));
                covered += 1;
            }
        }
    }
    assert_eq!(covered, 4_096);
}

#[test]
fn dadd_rz_register_packet_has_a_cpu_oracle_for_every_word() {
    const BASES: [u64; 8] = [
        0x5c70_0180_0007_0000,
        0x5c70_0180_a427_440b,
        0x5c70_03e1_8077_a0d1,
        0x5c70_07dc_cc47_3d36,
        0x5c70_0db2_f917_817b,
        0x5c70_0db4_1db7_aa3a,
        0x5c70_1da4_8b47_e19b,
        0x5c70_1daa_6b67_2c35,
    ];
    const VARIABLE_BITS: [u32; 7] = [19, 45, 46, 47, 48, 49, 50];
    let values = [
        (0.0_f64.to_bits(), (-0.0_f64).to_bits()),
        (1.0_f64.to_bits(), 2.0_f64.to_bits()),
        ((-1.0_f64).to_bits(), f64::from_bits(1).to_bits()),
        (1.0_f64.to_bits(), (f64::EPSILON / 2.0).to_bits()),
        (f64::MIN_POSITIVE.to_bits(), f64::from_bits(1).to_bits()),
        (f64::from_bits(1).to_bits(), f64::from_bits(1).to_bits()),
        ((f64::MAX / 2.0).to_bits(), (f64::MAX / 4.0).to_bits()),
        (f64::MAX.to_bits(), f64::MAX.to_bits()),
        ((-f64::MAX).to_bits(), (-f64::MAX).to_bits()),
        (2.0_f64.powi(500).to_bits(), 2.0_f64.powi(448).to_bits()),
        ((-2.0_f64.powi(500)).to_bits(), 2.0_f64.powi(499).to_bits()),
        (1.0_f64.to_bits(), (-1.0_f64).to_bits()),
    ];

    let set_pair = |state: &mut [u32; WORDS], register: usize, value: u64| {
        state[register] = value as u32;
        state[register | 1] = (value >> 32) as u32;
        if register & 1 == 1 {
            state[register] = (value >> 32) as u32;
        }
    };

    let mut covered = 0;
    for base in BASES {
        for selector in 0_u8..128 {
            let raw = VARIABLE_BITS
                .iter()
                .enumerate()
                .fold(base, |raw, (index, bit)| {
                    raw | u64::from(selector >> index & 1) << bit
                });
            let lhs = field(raw, 8, 8) as usize;
            let rhs = field(raw, 20, 8) as usize;
            let operands = values[covered / 2 % values.len()];
            for predicate_value in [0_u32, 1] {
                let mut state = [0_u32; WORDS];
                set_pair(&mut state, lhs, operands.0);
                set_pair(&mut state, rhs, operands.1);
                state[255] = 0;
                state[263] = 1;
                state[256..263].fill(predicate_value);
                state[264..268].copy_from_slice(&[1, 0, 1, 1]);
                reference(raw, &state).unwrap_or_else(|reason| panic!("{raw:016x}: {reason}"));
                covered += 1;
            }
        }
    }
    assert_eq!(covered, 2_048);
}

#[test]
fn new_alu_oracle_anchors() {
    // Author-owned values anchored to the cited operations above. These
    // check endpoint masks, condition numbering, FTZ's selection boundary,
    // two packed lanes, and 64-bit VMAD carry before any GPU execution.
    let raw = |opcode: u64| opcode << 48 | 7 << 16 | 2 << 8 | 1;
    let mut state = [0; WORDS];
    state[263] = 1;
    state[2] = 0x1234_5678;
    state[3] = 0x8765_4321;
    state[4] = 1;
    let fcmp = raw(0x5ba5) | 3 << 20 | 4 << 39;
    assert_eq!(reference(fcmp, &state).unwrap().words[1], state[2]);
    assert_eq!(
        reference(fcmp | 1 << 47, &state).unwrap().words[1],
        state[3]
    );
    let bfi = raw(0x36f0) | 32 << 28 | 3 << 39;
    assert_eq!(reference(bfi, &state).unwrap().words[1], state[2]);
    let bfi_empty = raw(0x36f0) | 32 << 20 | 3 << 39;
    assert_eq!(reference(bfi_empty, &state).unwrap().words[1], state[3]);
    state[264] = 1;
    let cset = (0x50a0_u64 << 48) | 30 << 8 | 7 << 39 | 7 << 16 | 1;
    let result = reference(cset, &state).unwrap();
    assert_eq!([result.words[256], result.words[257]], [1, 0]);
    state[2] = 0x3e00_3e00;
    state[3] = 0x4000_4000;
    state[4] = 0x3800_3800;
    let hfma = raw(0x5d00) | 3 << 20 | 4 << 39;
    assert_eq!(reference(hfma, &state).unwrap().words[1], 0x4300_4300);
    state[2] = 0x8000_8000;
    assert_eq!(reference(raw(0x7a00), &state).unwrap().words[1], 0);
    state[2] = 255;
    state[4] = u32::MAX;
    let vmad = raw(0x5f00) | 2 << 20 | 4 << 39;
    assert_eq!(reference(vmad, &state).unwrap().words[1], 509);
    assert_eq!(
        reference(vmad | 1 << 55, &state).unwrap().words[1],
        u32::MAX
    );
    assert_eq!(
        reference(vmad | 1 << 51, &state).unwrap().words[1],
        33_554_435
    );
}

#[test]
fn oracle_anchors() {
    // Project-owned, independently calculated anchors for the upstream facts
    // most likely to be misread. These never use an emitter result.
    let mut state = [0; WORDS];
    state[263] = 1;
    state[2] = 0x5555_5555;
    state[3] = 0xffff_0000;
    // POPC source20, gm107.c:2016.
    let popc = 0x5c08_u64 << 48 | 3 << 20 | 2 << 8 | 7 << 16 | 1;
    assert_eq!(reference(popc, &state).unwrap().words[1], 16);
    assert_eq!(reference(popc | 1 << 40, &state).unwrap().words[1], 16);
    // LOP3 LUT0xf8 is A | (B & C), Lop3Expression.cs:13-29.
    let lop3 = 0x3cf8_u64 << 48 | 2 << 8 | 7 << 16 | 1;
    assert_eq!(reference(lop3, &state).unwrap().words[1], 0x5555_5555);
    // Signed .HI, Mesa ir.rs:4524-4527. Sign-fill survives bit32.
    state[4] = u32::MAX;
    let shf = 0x5cfd_u64 << 48 | 3 << 37 | 4 << 39 | 3 << 20 | 2 << 8 | 7 << 16 | 1;
    state[3] = 33;
    assert_eq!(reference(shf, &state).unwrap().words[1], u32::MAX);
    // Saturation is GLSL FClamp, not Rust's signed-zero clamp.
    assert_eq!(finish_float(0x8000_0000, true, false).unwrap(), 0);
    assert_eq!(finish_float(0x7fc1_2345, true, false).unwrap(), 0);
    // Ryujinx InstEmitBitfield.cs:164-182: .SH of zero is FindLSB(0)=-1.
    state[3] = 0;
    let flo = 0x5c30_u64 << 48 | 1 << 41 | 3 << 20 | 7 << 16 | 1;
    assert_eq!(reference(flo, &state).unwrap().words[1], u32::MAX);
    // I2F.F64 must retain bit0 above f32's exact integer range.
    // InstEmitConversion.cs:268-284,398-416.
    state[3] = 16_777_217;
    let double = 0x5cb8_u64 << 48 | 3 << 20 | 7 << 16 | 2 << 10 | 3 << 8 | 1;
    let result = reference(double, &state).unwrap();
    assert_eq!(result.words[1], 0x1000_0000);
    assert_eq!(result.words[2], 0x4170_0000);
    // Rounding anchors: binary16 has ten fraction bits and RTE.
    assert_eq!(pack(2049.0), 0x6800);
    assert_eq!(pack(2051.0), 0x6802);
    assert_eq!(pack(65520.0), 0x7c00);
    assert_eq!(unpack(1).to_bits(), 2.0_f32.powi(-24).to_bits());
    // Binary64 midpoint plus a retained residual must round above the tie.
    let midpoint = 1.0 + 2.0_f64.powi(-24);
    assert_eq!(float_round(midpoint, 2.0_f64.powi(-80), 0), 0x3f80_0001);
    assert_eq!(float_round(midpoint, -2.0_f64.powi(-80), 0), 0x3f80_0000);
}

#[test]
fn fma_oracle_agrees_with_host_fused_arithmetic() {
    // Independent native fused arithmetic checks the CPU oracle itself.
    // Include exponent gaps, cancellation, normal/subnormal results and ties.
    let raw = 0x5980_u64 << 48 | 4 << 39 | 3 << 20 | 2 << 8 | 7 << 16 | 1;
    let mut rng = 0x5eed_5eed_5eed_5eed_u64;
    for _ in 0..10000 {
        let mut state = [0; WORDS];
        state[263] = 1;
        for slot in [2, 3, 4] {
            rng ^= rng >> 12;
            rng ^= rng << 25;
            rng ^= rng >> 27;
            // Avoid input NaN/infinity; finite overflow is included.
            state[slot] = (rng.wrapping_mul(0x2545_f491_4f6c_dd1d) as u32) & 0xfeff_ffff;
        }
        let expected = f32::from_bits(state[2])
            .mul_add(f32::from_bits(state[3]), f32::from_bits(state[4]))
            .to_bits();
        let actual = reference(raw, &state).unwrap().words[1];
        assert_eq!(actual, expected, "inputs={:08x?}", &state[2..5]);
    }
}

#[test]
fn extended_lop3_zero_uses_the_entry_flag() {
    // Register LOP3 X38/CC47: envytools gm107.c:2020.
    // Extended Z = oldZ && result==0: Ryujinx InstEmitAluHelper.cs:102-116.
    // LUT28..35, predicate48..50 and CC47: envytools gm107.c:2020.
    // GPR0/8/20:203-218; guard16..19:408-411.
    // LUT zero deliberately makes the result independent of every operand.
    let raw = 0x5be0_u64 << 48 | 7 << 48 | 1 << 47 | 3 << 20 | 2 << 8 | 7 << 16 | 1;
    let mut state = [0; WORDS];
    state[263] = 1;
    for old_zero in [0, 1] {
        state[264] = old_zero;
        assert_eq!(reference(raw, &state).unwrap().words[264], 1);
        assert_eq!(
            reference(raw | 1 << 38, &state).unwrap().words[264],
            old_zero
        );
    }
}

#[test]
fn half_flush_oracle_fixed_answers() {
    // Mode fields: envytools gm107.c:1157-1168,1977-1989;
    // HSETP2 FTZ6:1973-1976. Source and merge precision:
    // Ryujinx InstEmitHelper.cs:139-184. Signed denormal/zero-product
    // semantics: docs/reference-sources.md
    // GPR0/8/20: envytools gm107.c:203-218; guard16..19:408-411.
    // Output49..50 and A/B swizzles47..48/28..29:1133-1153.
    // FMA addend39..46:1989. Comparison accumulator39..42:1985,363-365.
    let raw = |opcode: u64| opcode << 48 | 3 << 20 | 2 << 8 | 7 << 16 | 1;
    let mut state = [0; WORDS];
    state[263] = 1;
    state[2] = 0x8001_0001;
    let add = raw(0x5d10);
    assert_eq!(reference(add, &state).unwrap().words[1], 0x8001_0001);
    assert_eq!(reference(add | 1 << 39, &state).unwrap().words[1], 0);

    state[2] = 0x8000_0000;
    state[3] = 0x7c00_7c00;
    let mul = raw(0x5d08);
    assert_eq!(
        normalise(reference(mul, &state).unwrap().words[1], 2),
        0x7e00_7e00
    );
    assert_eq!(
        reference(mul | 2 << 39, &state).unwrap().words[1],
        0x8000_0000
    );
    state[4] = 0xbc00_3c00;
    assert_eq!(
        reference(raw(0x5d00) | 2 << 37 | 4 << 39, &state)
            .unwrap()
            .words[1],
        state[4]
    );

    state[1] = 0x8001_0001;
    state[2] = 0x8001_0001;
    state[3] = 0;
    assert_eq!(
        reference(add | 1 << 39 | 2 << 49, &state).unwrap().words[1],
        0x8001_0000
    );

    // Equal comparison changes from false to true when tiny inputs flush.
    // H_AND49 and cmp35..38: envytools gm107.c:1985,1113-1129.
    let setp = raw(0x5d20) | 2 << 35 | 7 << 39;
    assert_eq!(reference(setp, &state).unwrap().words[256], 0);
    assert_eq!(reference(setp | 1 << 6, &state).unwrap().words[256], 1);

    // Output flush follows half rounding. Half 0x03ff times f32 just above
    // one rounds up to 0x0400; flushing before rounding would lose it.
    // Mesa half_float.c:96-104,122-128.
    state[2] = 0x3f80_2000;
    state[3] = 0x03ff_03ff;
    assert_eq!(
        reference(mul | 1 << 47 | 1 << 39, &state).unwrap().words[1],
        0
    );
    state[3] = 0x387f_c000; // F32 value of half 0x03ff, so input remains normal32.
    assert_eq!(
        reference(mul | 1 << 47 | 1 << 28 | 1 << 39, &state)
            .unwrap()
            .words[1],
        0x0400_0400
    );
}

#[test]
fn dmnmx_cc_cpu_exceptional_matrix() {
    // Exact register form with dst/lhs/rhs all R0:R1. That alias makes the
    // selected min/max irrelevant and isolates final-result flag semantics.
    let raw = 0x5c50_8000_0007_0000_u64;
    for (value, expected_zn) in [
        (0.0_f64, [1, 0]),
        (-0.0_f64, [1, 0]),
        (-2.0_f64, [0, 1]),
        (3.0_f64, [0, 0]),
        (f64::INFINITY, [0, 0]),
        (f64::NEG_INFINITY, [0, 1]),
        (f64::from_bits(0x7ff8_1234_5678_9abc), [0, 0]),
    ] {
        let mut state = [0; WORDS];
        state[263] = 1;
        state[266] = 1;
        state[267] = 1;
        state[0] = value.to_bits() as u32;
        state[1] = (value.to_bits() >> 32) as u32;
        let result = reference(raw, &state).unwrap();
        assert_eq!(&result.words[264..266], &expected_zn);
        assert_eq!(&result.words[266..268], &[1, 1], "C/V changed");
    }

    // One of the exact inventory forms is guarded by a non-PT predicate.
    // With that guard false, destination and every condition bit are kept.
    let guarded = 0x5c51_e26d_1df0_3060_u64;
    let mut state = [0; WORDS];
    state[263] = 1;
    state[0] = 0x0123_4567;
    state[1] = 0x89ab_cdef;
    state[264..268].copy_from_slice(&[1, 0, 1, 1]);
    let guard = field(guarded, 16, 3) as usize;
    state[256 + guard] = u32::from(flag(guarded, 19));
    let result = reference(guarded, &state).unwrap();
    assert_eq!(&result.words[0..2], &state[0..2]);
    assert_eq!(&result.words[264..268], &state[264..268]);
}

#[test]
fn dmnmx_immediate_cc_packet_has_a_cpu_oracle_for_every_word() {
    const BASES: [u64; 7] = [
        0x3850_8000_0007_0000,
        0x3850_805d_3db7_5ce2,
        0x3850_822e_ade7_8146,
        0x3850_82e3_d857_c6ee,
        0x3850_83b0_d857_31fe,
        0x3850_885c_2af7_8254,
        0x3850_8a5a_3f57_c9ab,
    ];
    const VARIABLE_BITS: [u32; 8] = [19, 42, 45, 46, 48, 49, 50, 56];

    let mut covered = 0;
    for base in BASES {
        for selector in 0_u16..256 {
            let raw = VARIABLE_BITS
                .iter()
                .enumerate()
                .fold(base, |raw, (index, bit)| {
                    raw | u64::from(selector >> index & 1) << bit
                });
            for predicate_value in [0_u32, 1] {
                let mut state = [0_u32; WORDS];
                for (register, word) in state[..255].iter_mut().enumerate() {
                    *word = (register as u32).wrapping_mul(0x0101_0101) ^ 0x8000_0042;
                }
                state[255] = 0;
                state[263] = 1;
                state[256..263].fill(predicate_value);
                state[264..268].copy_from_slice(&[1, 0, 1, 1]);

                let result =
                    reference(raw, &state).unwrap_or_else(|reason| panic!("{raw:016x}: {reason}"));
                assert_eq!(&result.words[266..268], &[1, 1], "{raw:016x}");
                if flag(raw, 19) {
                    assert_eq!(result.words, state, "inactive @!PT word {raw:016x}");
                }
                covered += 1;
            }
        }
    }
    assert_eq!(covered, 3_584);
}
