//! Bounded raw Maxwell interpreter for project-owned helper-lane fixtures.
//! No decoded IR or translated module enters this oracle.
#[derive(Clone, Copy)]
struct Lane {
    registers: [u32; 256],
    predicates: [bool; 8],
    helper: bool,
}
fn field(raw: u64, start: u32, width: u32) -> u32 {
    ((raw >> start) & ((1u64 << width) - 1)) as u32
}
fn flush(bits: u32) -> u32 {
    if bits & 0x7f80_0000 == 0 {
        bits & 0x8000_0000
    } else {
        bits
    }
}

/// Execute one covered 2x2 quad in instruction order, including KIL helpers.
/// Only the cited encodings used by this fixture are admitted.
pub fn quad(words: &[u64], x: usize, y: usize) -> [Option<[u32; 4]>; 4] {
    let mut lanes = [Lane {
        registers: [0; 256],
        predicates: [false, false, false, false, false, false, false, true],
        helper: false,
    }; 4];
    for &raw in words {
        let before = lanes;
        for (index, lane) in lanes.iter_mut().enumerate() {
            // Mesa sm50.rs:225-235: guard16..18, inversion19.
            if before[index].predicates[field(raw, 16, 3) as usize] == (field(raw, 19, 1) != 0) {
                continue;
            }
            let dst = field(raw, 0, 8) as usize;
            let a = field(raw, 8, 8) as usize;
            let b = field(raw, 20, 8) as usize;
            match raw >> 48 {
                // Mesa sm50.rs:1938-1951: MOV32I immediate20..51, quad12..15.
                _ if raw >> 60 == 0 && raw & (15 << 56) == 1 << 56 => {
                    if field(raw, 12, 4) & (1 << index) != 0 {
                        lane.registers[dst] = field(raw, 20, 32);
                    }
                }
                // Mesa sm50.rs:2861-2876: ALD Position xy at0x70/74.
                // Attribute map: nak_private.h:45-57.
                0xefd8 => {
                    assert_eq!(a, 255);
                    assert_eq!(field(raw, 39, 8), 255);
                    assert_eq!(field(raw, 31, 2), 0);
                    let address = field(raw, 20, 10);
                    let count = field(raw, 47, 2) + 1;
                    for component in 0..count {
                        let value = match address + component * 4 {
                            0x70 => (x + index % 2) as f32 + 0.5,
                            0x74 => (y + index / 2) as f32 + 0.5,
                            _ => panic!("unsupported ALD attribute"),
                        };
                        lane.registers[dst + component as usize] = value.to_bits();
                    }
                }
                // Mesa sm50.rs:871-901,798-805: FSETP.LT.AND Pn,PT,A,B,PT.
                0x5bb1 => {
                    assert_eq!(field(raw, 39, 4), 7);
                    assert_eq!(field(raw, 45, 2), 0);
                    let p = field(raw, 3, 3) as usize;
                    let q = field(raw, 0, 3) as usize;
                    let result = f32::from_bits(before[index].registers[a])
                        < f32::from_bits(before[index].registers[b]);
                    if p != 7 {
                        lane.predicates[p] = result;
                    }
                    if q != 7 {
                        lane.predicates[q] = !result;
                    }
                }
                // Mesa from_nir.rs:3035-3050; sm50.rs:3172-3174:
                // KIL is demotion, so register execution and guards continue.
                0xe330 => {
                    assert_eq!(raw & 0x0000_ffff_fff0_ffff, 15);
                    lane.helper = true;
                }
                // Mesa sm50.rs:2043-2080: SHFL.BFLY immediate1/2, control1c03,
                // PT predicate destination. All lanes execute in these fixtures.
                // Ryujinx InstEmitWarp.cs:24-45; HelperFunctionManager.cs:319-339.
                0xef17 => {
                    assert_eq!(field(raw, 28, 2), 3);
                    assert_eq!(field(raw, 30, 2), 3);
                    assert_eq!(field(raw, 34, 13), 0x1c03);
                    assert_eq!(field(raw, 48, 3), 7);
                    let distance = field(raw, 20, 5) as usize;
                    assert!(matches!(distance, 1 | 2));
                    lane.registers[dst] = before[index ^ distance].registers[a];
                }
                // Mesa sm50.rs:913-943 / ir.rs:8952: lane ops28..35 and FTZ44.
                // Mesa from_nir.rs:2463-2492,2524-2553: subtraction signs.
                // Ryujinx Optimizer.cs:240-286 recognizes the derivative pair.
                0x50f8 => {
                    assert_eq!(field(raw, 38, 3), 0);
                    assert_eq!(field(raw, 47, 1), 0);
                    let ftz = field(raw, 44, 1) != 0;
                    let bits = |r: usize| {
                        let v = before[index].registers[r];
                        if ftz {
                            flush(v)
                        } else {
                            v
                        }
                    };
                    let left = f32::from_bits(bits(a));
                    let right = f32::from_bits(bits(b));
                    let value = match field(raw, 28 + index as u32 * 2, 2) {
                        0 => left + right,
                        1 => left - right,
                        2 => right - left,
                        3 => left * 0.0 + right,
                        _ => unreachable!(),
                    }
                    .to_bits();
                    lane.registers[dst] = if ftz { flush(value) } else { value };
                }
                // Mesa sm50.rs:1931-1936,1951: register MOV, lanes39..42.
                0x5c98 => {
                    if field(raw, 39, 4) & (1 << index) != 0 {
                        lane.registers[dst] = before[index].registers[b];
                    }
                }
                // Mesa sm50.rs:681-704: register FMUL, RN, no modifiers.
                // Ryujinx InstEmitFloatArithmetic.cs:435-480: binary32 product.
                0x5c68 => {
                    assert_eq!(field(raw, 39, 9), 0);
                    lane.registers[dst] = (f32::from_bits(before[index].registers[a])
                        * f32::from_bits(before[index].registers[b]))
                    .to_bits();
                }
                // Bound TEX AUTO: Mesa sm50.rs:2171-2197; full opcode
                // nv50_ir_emit_gm107.cpp:2909-2924. srcA coordinates xy,
                // srcB RZ, handle0, 2D, RGBA, no NODEP/DC/NDV/offset/LOD.
                // Ryujinx InstEmitTexture.cs:28-32; Spirv/Instructions.cs:1443-1445.
                0xc038 => {
                    assert_eq!(b, 255);
                    assert_eq!(field(raw, 28, 3), 2);
                    assert_eq!(field(raw, 31, 4), 15);
                    assert_eq!(field(raw, 35, 13), 0);
                    assert_eq!(field(raw, 49, 2), 0);
                    assert_eq!(field(raw, 54, 3), 0);
                    // The fixture has exactly affine power-of-two gradients.
                    // Vulkan-Docs v1.4.351 textures.adoc:1554-1560,1618-1628:
                    // Euclidean footprint, scaled by base image extent. The
                    // permitted approximations still select the same clamped
                    // mip for these two axis-aligned power-of-two gradients.
                    let uv = |i: usize| {
                        [
                            f32::from_bits(before[i].registers[a]),
                            f32::from_bits(before[i].registers[a + 1]),
                        ]
                    };
                    let lo_x = uv(index & !1);
                    let hi_x = uv(index | 1);
                    let lo_y = uv(index & !2);
                    let hi_y = uv(index | 2);
                    let squared = |lo: [f32; 2], hi: [f32; 2]| {
                        let dx = (hi[0] - lo[0]) * 4.0;
                        let dy = (hi[1] - lo[1]) * 4.0;
                        dx * dx + dy * dy
                    };
                    let rho2 = squared(lo_x, hi_x).max(squared(lo_y, hi_y));
                    assert!(rho2 == 0.25 || rho2 == 4.0);
                    // textures.adoc:1677-1687,1730-1761: nearest mip clamped
                    // to0/1. Avoid host log2 and boundary tolerances. At scale
                    // 1/8, even the largest allowed footprint has lambda<0;
                    // at scale1/2, even the smallest footprint has lambda>=1.
                    let level = usize::from(rho2 == 4.0);
                    let size = 4 >> level;
                    let coord = uv(index);
                    // textures.adoc:1909-1922: nearest texel floor, shift0.
                    let tx = ((coord[0] * size as f32).floor() as usize).min(size - 1);
                    let ty = ((coord[1] * size as f32).floor() as usize).min(size - 1);
                    for channel in 0..4 {
                        // Independently mirror immutable project-owned texels,
                        // not the SPIR-V sample operation or its result.
                        let numerator = 1 + level * 128 + ty * 32 + tx * 4 + channel;
                        lane.registers[dst + channel] = (numerator as f32 / 256.0).to_bits();
                    }
                }
                _ => panic!("quad oracle has no implementation for {raw:016x}"),
            }
            lane.registers[255] = 0;
        }
    }
    lanes.map(|lane| (!lane.helper).then(|| lane.registers[..4].try_into().unwrap()))
}

/// Full framebuffer words, including unchanged clear values for demoted lanes.
pub fn framebuffer(words: &[u64]) -> Vec<u32> {
    // Project-owned graphics transport clears untouched pixels to these
    // exact RGBA32F words (graphics/mod.rs, clear_values).
    let clear = [-16f32, -32., -64., -128.].map(f32::to_bits);
    let mut result = clear.repeat(8 * 8);
    for y in (0..8).step_by(2) {
        for x in (0..8).step_by(2) {
            for (lane, color) in quad(words, x, y).into_iter().enumerate() {
                if let Some(color) = color {
                    let pixel = (y + lane / 2) * 8 + x + lane % 2;
                    result[pixel * 4..pixel * 4 + 4].copy_from_slice(&color);
                }
            }
        }
    }
    result
}
