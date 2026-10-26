// Project-owned words and independent texel equations. Encodings: Mesa
// sm50.rs:2171-2197; envytools gm107.c:916-923,1026-1033,1962,1970.
// Operand order and signed offsets: Ryujinx InstEmitTexture.cs:203-299.
// Dynamic offsets: Translation/Transforms/TexturePass.cs:501-525,591-650.
// No upstream implementation copied.
use super::{compute, validate, work, Compute, WORDS};
use shadowbox::spirv::Op;
use shadowbox::{decoder::decode_stream, emitter::emit_unoptimized_with_header, headers};

fn code(raw: u64, bound: bool) -> Vec<u32> {
    // Bound TEX uses table index9; bindless uses the proven handle1 in B20.
    // MOV immediate: Mesa sm50.rs:1938-1941. NOP: sm50.rs:3178-3184.
    let first = if bound {
        0x50b0u64 << 48 | 7 << 16
    } else {
        0x0100u64 << 48 | 1 << 20 | 15 << 12 | 7 << 16 | 20
    };
    [0, first, raw, 0xe300u64 << 48 | 7 << 16 | 15]
        .into_iter()
        .flat_map(|word| [word as u32, (word >> 32) as u32])
        .collect()
}

#[test]
fn sample_runtime_zero_lod_offsets_exact_on_radv() {
    let gpu = Compute::new();
    let header = headers::parse(&[0; 32]).unwrap();
    let mut checked = 0;
    let mut forms = 0;
    for dim in 0..5u64 {
        let array = matches!(dim, 1 | 3);
        let axes = match dim {
            0 | 1 => 1,
            2 | 3 => 2,
            _ => 3,
        };
        for depth in [false, true] {
            if dim == 4 && depth {
                continue;
            }
            for bound in [false, true] {
                for lod in 0..2u64 {
                    for mask in 1..16u64 {
                        // Rotate guards and aliases through every channel mask.
                        let guard = [0u64, 7, 8][mask as usize % 3];
                        let dst = [8u64, 20, 32, 254, 255][mask as usize % 5];
                        let raw = (if bound { 0xc038u64 } else { 0xdeb8 }) << 48
                            | lod << if bound { 55 } else { 37 }
                            | 1 << if bound { 54 } else { 36 }
                            | if bound { 9 << 36 } else { 0 }
                            | u64::from(depth) << 50
                            | (mask & 1) << 49
                            | mask << 31
                            | dim << 28
                            | 20 << 20
                            | guard << 16
                            | 8 << 8
                            | dst;
                        let mut module = emit_unoptimized_with_header(
                            &decode_stream(shadowbox::QuietRun, &code(raw, bound)),
                            Some(&header),
                        )
                        .unwrap();
                        assert_eq!(
                            compute::ops(&module.words)
                                .filter(|(op, _)| *op == Op::ImageQuerySizeLod)
                                .count(),
                            1
                        );
                        assert!(compute::ops(&module.words).any(|(op, _)| {
                            op == if depth {
                                Op::ImageSampleDrefExplicitLod
                            } else {
                                Op::ImageSampleExplicitLod
                            }
                        }));
                        compute::observe_group(&mut module, 1);
                        let original = module.words.clone();
                        module.optimize();
                        let cases = (0..64u32)
                            .map(|case| {
                                let mut state = std::array::from_fn(|word| {
                                    0x1357_9bdfu32.wrapping_mul(word as u32 + 1) ^ case
                                });
                                state[255..268].fill(0);
                                for (bit, value) in state[256..268].iter_mut().enumerate() {
                                    *value = (case >> (bit % 6)) & 1;
                                }
                                state[256] = (case / 16) & 1;
                                state[263] = 1;
                                let layer = if array { case % 3 } else { 0 };
                                if array {
                                    state[8] = layer;
                                }
                                let mut packed = 0;
                                let mut texel = [0; 3];
                                for (axis, texel) in texel.iter_mut().enumerate().take(axes) {
                                    let x = (case / (axis as u32 + 1)) % 4;
                                    state[8 + usize::from(array) + axis] =
                                        ((x as f32 + 0.5) / 4.0).to_bits();
                                    let nibble = (case + axis as u32 * 7) & 15;
                                    packed |= nibble << (axis * 4);
                                    let offset = if nibble < 8 {
                                        nibble as i32
                                    } else {
                                        nibble as i32 - 16
                                    };
                                    *texel = (x as i32 + offset).clamp(0, 3) as u32;
                                }
                                // Unused packed data bits vary, including bit31.
                                state[if bound { 20 } else { 21 }] =
                                    packed | (case.wrapping_mul(0x9e37) << 12);
                                if depth {
                                    let value = 1
                                        + layer * 512
                                        + texel[2] * 128
                                        + texel[1] * 32
                                        + texel[0] * 4;
                                    state[if bound { 21 } else { 22 }] =
                                        ((value as f32 + (case % 3) as f32 - 1.0) / 8192.0)
                                            .to_bits();
                                }
                                state
                            })
                            .collect::<Vec<[u32; WORDS]>>();
                        for words in [&original, &module.words] {
                            validate(words, &work().join("sample-runtime-offset-validation.spv"));
                            let actual = gpu.run_images(words, &cases, false).0;
                            for (case, (input, output)) in cases.iter().zip(actual).enumerate() {
                                let mut expected = *input;
                                if !bound {
                                    expected[20] = 1;
                                }
                                let active = match guard {
                                    0 => input[256] != 0,
                                    7 => true,
                                    8 => input[256] == 0,
                                    _ => false,
                                };
                                if active {
                                    // Image and nearest/clamp sampler are allocated
                                    // independently in support/texture.rs:48-80,231-240.
                                    let layer = if array { input[8] } else { 0 };
                                    let packed = input[if bound { 20 } else { 21 }];
                                    let mut texel = [0; 3];
                                    for (axis, texel) in texel.iter_mut().enumerate().take(axes) {
                                        let coord =
                                            f32::from_bits(input[8 + usize::from(array) + axis]);
                                        let nibble = (packed >> (axis * 4)) & 15;
                                        let offset = if nibble < 8 {
                                            nibble as i32
                                        } else {
                                            nibble as i32 - 16
                                        };
                                        *texel = ((coord * 4.0).floor() as i32 + offset).clamp(0, 3)
                                            as u32;
                                    }
                                    let base = 1
                                        + layer * 512
                                        + texel[2] * 128
                                        + texel[1] * 32
                                        + texel[0] * 4;
                                    let mut written = 0;
                                    for channel in 0..4u32 {
                                        if mask & (1 << channel) != 0 {
                                            let value = if depth {
                                                let compare = f32::from_bits(
                                                    input[if bound { 21 } else { 22 }],
                                                );
                                                f32::from(compare <= base as f32 / 8192.0).to_bits()
                                            } else {
                                                ((base + channel) as f32 / 256.0).to_bits()
                                            };
                                            let register = dst as usize + written;
                                            if register < 255 {
                                                expected[register] = value;
                                            }
                                            written += 1;
                                        }
                                    }
                                }
                                assert_eq!(output, expected, "{raw:016x}, case={case}");
                                checked += WORDS;
                            }
                        }
                        forms += 1;
                    }
                }
            }
        }
    }
    eprintln!("TEX dynamic zero-LOD offsets: {forms} forms, {checked} exact state words, zero differences, raw and optimized, on {}", gpu.description);
}

#[test]
fn sample_runtime_3d_offset_combinations_exact_on_radv() {
    let gpu = Compute::new();
    let header = headers::parse(&[0; 32]).unwrap();
    let mut checked = 0;
    for bound in [false, true] {
        // Held-out joint domain: every combination of three signed nibbles.
        // Fields: Mesa sm50.rs:2171-2197; envytools gm107.c:916-923,1970.
        let raw = (if bound { 0xc038u64 } else { 0xdeb8 }) << 48
            | 1 << if bound { 55 } else { 37 }
            | 1 << if bound { 54 } else { 36 }
            | if bound { 9 << 36 } else { 0 }
            | 15 << 31
            | 4 << 28
            | 20 << 20
            | 7 << 16
            | 8 << 8
            | 32;
        let mut module = emit_unoptimized_with_header(
            &decode_stream(shadowbox::QuietRun, &code(raw, bound)),
            Some(&header),
        )
        .unwrap();
        compute::observe_group(&mut module, 1);
        let original = module.words.clone();
        module.optimize();
        for words in [&original, &module.words] {
            validate(words, &work().join("sample-runtime-3d-validation.spv"));
            for batch in 0..64u32 {
                let cases = (batch * 64..(batch + 1) * 64)
                    .map(|packed| {
                        let mut state = [0; WORDS];
                        state[263] = 1;
                        for axis in 0..3 {
                            let x = (packed >> (axis * 4)) & 3;
                            state[8 + axis] = ((x as f32 + 0.5) / 4.0).to_bits();
                        }
                        state[if bound { 20 } else { 21 }] = packed | 0xfedc_0000;
                        state[32..36].fill(0xdead_beef);
                        state
                    })
                    .collect::<Vec<[u32; WORDS]>>();
                for (input, output) in cases.iter().zip(gpu.run_images(words, &cases, false).0) {
                    let mut expected = *input;
                    if !bound {
                        expected[20] = 1;
                    }
                    let packed = input[if bound { 20 } else { 21 }];
                    let mut texel = [0; 3];
                    for (axis, texel) in texel.iter_mut().enumerate() {
                        let nibble = (packed >> (axis * 4)) & 15;
                        let offset = if nibble < 8 {
                            nibble as i32
                        } else {
                            nibble as i32 - 16
                        };
                        let x = (f32::from_bits(input[8 + axis]) * 4.0).floor() as i32;
                        *texel = (x + offset).clamp(0, 3) as u32;
                    }
                    // Independent allocation equation, support/texture.rs:70-80.
                    for channel in 0..4 {
                        expected[32 + channel] =
                            ((1 + texel[2] * 128 + texel[1] * 32 + texel[0] * 4 + channel as u32)
                                as f32
                                / 256.0)
                                .to_bits();
                    }
                    assert_eq!(output, expected, "{raw:016x}, offset={packed:08x}");
                    checked += WORDS;
                }
            }
        }
    }
    eprintln!("TEX dynamic 3D offsets: 4096 combinations per handle form, {checked} exact state words, zero differences, raw and optimized, on {}", gpu.description);
}
