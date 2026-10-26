#![cfg(feature = "gpu-check")]
//! Independent hardware image-store observations. Encoding and semantics:
//! gm107.c:1909,753-838; sm50.rs:2506-2529; Ryujinx
//! InstEmitSurface.cs:566-679. Only formatted 2D is covered here.
#[path = "support/compute.rs"]
mod compute;
#[path = "support/spirv_validation.rs"]
mod spirv_validation;
use compute::{Compute, WORDS};
use shadowbox::ir::ops;
use shadowbox::spirv::Op;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{Instruction, InstructionKind as K, Register as R, Source as S},
};
use std::{fs, process::Command};

fn glsl(body: &str, path: &std::path::Path) -> Vec<u32> {
    let source = format!("#version 450\nlayout(local_size_x=1) in;\nlayout(set=0,binding=2048,rgba32f) uniform image2D target;\nlayout(set=3,binding=0,std430) readonly buffer In {{uint a[];}};\nlayout(set=3,binding=1,std430) writeonly buffer Out {{uint b[];}};\nvoid main(){{uint n=gl_GlobalInvocationID.x; ivec2 p=ivec2(n%4,n/4); {body} }}");
    compile_glsl(&source, path)
}

fn compile_glsl(source: &str, path: &std::path::Path) -> Vec<u32> {
    let input = path.with_extension("comp");
    let output = path.with_extension("spv");
    fs::write(&input, source).unwrap();
    let check = Command::new("glslangValidator")
        .args(["-V", "--target-env", "vulkan1.2", "-o"])
        .arg(&output)
        .arg(&input)
        .output()
        .unwrap();
    assert!(
        check.status.success(),
        "{}",
        String::from_utf8_lossy(&check.stdout)
    );
    let bytes = fs::read(&output).unwrap();
    let words = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|v| u32::from_le_bytes(*v))
        .collect::<Vec<_>>();
    fs::remove_file(input).unwrap();
    fs::remove_file(output).unwrap();
    words
}

#[test]
fn sust_target3_all_formatted_masks_bound_and_bindless_on_gpu() {
    let gpu = Compute::new();
    let scratch = std::env::temp_dir().join(format!("field-probes-gpu-{}", std::process::id()));
    fs::create_dir_all(&scratch).unwrap();
    // A GLSL reader observes every texel independently of Shadowbox's
    // surface coordinate, mask, format and handle lowering.
    let reset = glsl(
        "imageStore(target,p,vec4(19,23,29,31));",
        &scratch.join("reset"),
    );
    let reader = glsl(&format!("uvec4 v=floatBitsToUint(imageLoad(target,p)); for(uint c=0;c<4;c++) b[n*{WORDS}+c]=v[c];"),&scratch.join("reader"));
    let cases = (0..16)
        .map(|n| {
            let mut state = [0u32; WORDS];
            state[2] = n % 4;
            state[3] = n / 4;
            for c in 0..4 {
                state[8 + c] = ((n * 4 + c as u32 + 1) as f32 / 16.0).to_bits();
            }
            state
        })
        .collect::<Vec<_>>();
    let header = headers::parse(&[0; 32]).unwrap();
    let mut comparisons = 0;
    for bound in [false, true] {
        for cache in 0..2u64 {
            for mask in 1..16u64 {
                let raw = 0xeb20u64 << 48
                    | 7 << 16
                    | 3 << 33
                    | mask << 20
                    | cache << 24
                    | 2 << 8
                    | 8
                    | if bound { 1 << 51 | 9 << 36 } else { 20 << 39 };
                let xs = [
                    Instruction {
                        address: 0,
                        predicate: None,
                        kind: K::Mov(ops::Mov {
                            quad_lanes: 15,
                            dst: R(20),
                            src: S::ConstantBuffer {
                                binding: 0,
                                offset: 0,
                            },
                        }),
                    },
                    decode_word([raw as u32, (raw >> 32) as u32], 8),
                    Instruction {
                        address: 16,
                        predicate: None,
                        kind: K::Exit(ops::Exit { cc_test: 15 }),
                    },
                ];
                let mut module = emit_unoptimized_with_header(&xs, Some(&header)).unwrap();
                assert_eq!(module.surface_descriptors.len(), 1);
                let descriptor = &module.surface_descriptors[0];
                assert_eq!(descriptor.target, 3);
                // Observer uses set3 for its SSBOs. Remap only the verified
                // bound storage descriptor to the test image in set0.
                if bound {
                    assert_eq!(descriptor.set, 2);
                    assert_eq!(descriptor.binding, 9);
                    let var = compute::ops(&module.words)
                        .find(|(op, a)| *op == Op::Decorate && a[1..] == [34, 2])
                        .unwrap()
                        .1[0];
                    let mut at = 5;
                    while at < module.words.len() {
                        let size = (module.words[at] >> 16) as usize;
                        if Op::of(module.words[at]) == Op::Decorate && module.words[at + 1] == var {
                            match module.words[at + 2] {
                                34 => module.words[at + 3] = 0,
                                33 => module.words[at + 3] = 2048,
                                _ => {}
                            }
                        }
                        at += size;
                    }
                }
                compute::observe_group(&mut module, 1);
                let raw_words = module.words.clone();
                module.optimize();
                for words in [&raw_words, &module.words] {
                    let check = spirv_validation::validate(words);
                    assert!(
                        check.status.success(),
                        "{}",
                        String::from_utf8_lossy(&check.stderr)
                    );
                    gpu.run(&reset, &cases);
                    gpu.run(words, &cases);
                    let actual = gpu.run(&reader, &cases);
                    for (n, result) in actual.iter().enumerate() {
                        // Ryujinx InstEmitSurface.cs:637-645 emits packed
                        // sources; CodeGen/Spirv/Instructions.cs:808-831
                        // fills consecutive channels then pads with zero.
                        for (c, &bits) in result[..4].iter().enumerate() {
                            let expected = if c < mask.count_ones() as usize {
                                cases[n][8 + c]
                            } else {
                                0
                            };
                            assert_eq!(bits, expected, "{raw:016x}, texel {n}, channel {c}");
                            comparisons += 1;
                        }
                    }
                }
            }
        }
    }
    fs::remove_dir(scratch).unwrap();
    eprintln!(
        "{}; SUST target3 formatted: {comparisons} exact words, 0 differences",
        format_args!(
            "{} Float64={}",
            gpu.description, gpu.supports_shader_float64
        )
    );
}

#[test]
fn bound_txq_dimension_and_level_masks_use_driver_shape_on_gpu() {
    // Encoding: sm50.rs:2351-2381; Ryujinx InstDecoders.cs:5082-5093.
    // Driver type: InstEmitTexture.cs:1116-1119. SPIR-V size/levels:
    // CodeGen/Spirv/Instructions.cs:1483-1521.
    use shadowbox::{
        emitter::emit_unoptimized_with_graphics_state,
        interface::{BoundTexture, GraphicsState, TextureDimension},
    };
    let gpu = Compute::new();
    let header = headers::parse(&[0; 32]).unwrap();
    let bindings = [BoundTexture {
        binding: 9,
        dimension: TextureDimension::D2,
        depth_compare: false,
        multisample: false,
    }];
    let state = GraphicsState {
        texture_bindings: &bindings,
        ..Default::default()
    };
    let cases = (0..2)
        .map(|lod| {
            let mut s = [0; WORDS];
            s[2] = lod;
            s
        })
        .collect::<Vec<_>>();
    let mut comparisons = 0;
    for mask in [1u64, 2, 3, 8, 9, 10, 11] {
        let raw = 0xdf48u64 << 48
            | 9 << 36
            | mask << 31
            | 1 << 22
            | 7 << 16
            | (if mask == 8 { 255 } else { 2 }) << 8
            | 8;
        let xs = [
            decode_word([raw as u32, (raw >> 32) as u32], 0),
            Instruction {
                address: 8,
                predicate: None,
                kind: K::Exit(ops::Exit { cc_test: 15 }),
            },
        ];
        assert!(emit_unoptimized_with_header(&xs, Some(&header)).is_err());
        let mut module = emit_unoptimized_with_graphics_state(&xs, Some(&header), &state).unwrap();
        compute::observe_group(&mut module, 1);
        let original = module.words.clone();
        module.optimize();
        for words in [&original, &module.words] {
            let check = spirv_validation::validate(words);
            assert!(
                check.status.success(),
                "{}",
                String::from_utf8_lossy(&check.stderr)
            );
            let actual = gpu.run(words, &cases);
            for (lod, registers) in actual.iter().enumerate() {
                let channels = [4 >> lod, 4 >> lod, 1, 2];
                let mut packed = 0;
                for (channel, value) in channels.into_iter().enumerate() {
                    if mask & (1 << channel) != 0 {
                        assert_eq!(registers[8 + packed], value);
                        packed += 1;
                        comparisons += 1;
                    }
                }
            }
        }
    }
    eprintln!(
        "{}; TXQ: {comparisons} exact words, 0 differences",
        gpu.description
    );
}

#[path = "support/brk_cases.rs"]
mod brk_cases;

#[test]
fn brk_all_cc_flags_and_predicate_polarities_on_gpu() {
    let gpu = Compute::new();
    let header = headers::parse(&[0; 32]).unwrap();
    let cases = (0..32u8)
        .map(|i| {
            let mut state = [0; WORDS];
            state[256] = u32::from(i & 16 != 0);
            for flag in 0..4 {
                state[264 + flag] = u32::from(i & (1 << flag) != 0);
            }
            state
        })
        .collect::<Vec<_>>();
    let mut comparisons = 0;
    for continuation in [false, true] {
        for cc in brk_cases::CONDITIONS {
            for guard in [0u8, 8, 7, 15] {
                let xs = shadowbox::decoder::decode_stream(
                    shadowbox::QuietRun,
                    &brk_cases::program_for_stack(cc, guard, continuation),
                );
                // !PT has no executed pop history or proved destination.
                // Ryujinx OperandHelper.cs:49-51; Decoder.cs:824-869.
                // The strict stack proof retains the pop and rejects the
                // module. This is an unsupported case, not a GPU match.
                if guard == 15 {
                    assert!(matches!(
                        emit_unoptimized_with_header(&xs, Some(&header)),
                        Err(shadowbox::interface::Stopped::UnprovenPop(_))
                    ));
                    continue;
                }
                let mut module = emit_unoptimized_with_header(&xs, Some(&header)).unwrap();
                compute::observe_group(&mut module, 1);
                let original = module.words.clone();
                module.optimize();
                for words in [&original, &module.words] {
                    let validation = spirv_validation::validate(words);
                    assert!(
                        validation.status.success(),
                        "{}",
                        String::from_utf8_lossy(&validation.stderr)
                    );
                    let results = gpu.run(words, &cases);
                    for (i, state) in results.iter().enumerate() {
                        let pred = match guard {
                            7 => true,
                            15 => false,
                            _ => (i & 16 != 0) ^ (guard == 8),
                        };
                        let expected = if pred && brk_cases::taken(cc, i as u8 & 15) {
                            2
                        } else {
                            1
                        };
                        assert_eq!(
                            state[8], expected,
                            "CONT={continuation}, CC{cc}, guard{guard}, flags/predicate{i}"
                        );
                        comparisons += 1;
                    }
                }
            }
        }
    }
    eprintln!(
        "{}; BRK/CONT: {comparisons} exact words, 0 differences; 52 !PT programs rejected without a pop destination proof",
        gpu.description
    );
}

#[test]
fn failed_brk_cc_keeps_outer_target_on_gpu() {
    let gpu = Compute::new();
    let header = headers::parse(&[0; 32]).unwrap();
    let slots = [
        brk_cases::push(0, 6),
        brk_cases::brk(2, 7),
        brk_cases::mov(9, 11),
        brk_cases::push(3, 8),
        brk_cases::brk(15, 7),
        brk_cases::exit(),
        brk_cases::mov(8, 23),
        brk_cases::exit(),
        brk_cases::brk(15, 7),
    ];
    let xs = shadowbox::decoder::decode_stream(shadowbox::QuietRun, &brk_cases::words(&slots));
    let cases = (0..16u8)
        .map(|i| {
            let mut state = [0; WORDS];
            state[9] = 29;
            for flag in 0..4 {
                state[264 + flag] = u32::from(i & (1 << flag) != 0);
            }
            state
        })
        .collect::<Vec<_>>();
    let mut module = emit_unoptimized_with_header(&xs, Some(&header)).unwrap();
    compute::observe_group(&mut module, 1);
    let original = module.words.clone();
    module.optimize();
    for words in [&original, &module.words] {
        let check = spirv_validation::validate(words);
        assert!(
            check.status.success(),
            "{}",
            String::from_utf8_lossy(&check.stderr)
        );
        let actual = gpu.run(words, &cases);
        for (i, state) in actual.iter().enumerate() {
            assert_eq!(state[8], 23);
            assert_eq!(state[9], if brk_cases::taken(2, i as u8) { 29 } else { 11 });
        }
    }
    eprintln!(
        "{}; nested BRK: 64 exact words, 0 differences",
        gpu.description
    );
}

#[test]
fn s2r_compute_builtins_constants_and_masks_on_gpu() {
    // Explicit Ryujinx cases: InstEmitMove.cs:78-163. InvocationId11 is
    // geometry/TCS, checked by tests/s2r.rs with exact external inputs.
    let gpu = Compute::new();
    let header = headers::parse(&[0; 32]).unwrap();
    let indices = [
        0, 0x12, 0x13, 0x1d, 0x20, 0x21, 0x22, 0x23, 0x25, 0x26, 0x27, 0x38, 0x39, 0x3a, 0x3b, 0x3c,
    ];
    let mut xs = indices
        .iter()
        .enumerate()
        .map(|(dst, &idx)| {
            let raw = 0xf0c8u64 << 48 | 7 << 16 | u64::from(idx as u8) << 20 | dst as u64;
            decode_word([raw as u32, (raw >> 32) as u32], dst as u32 * 8)
        })
        .collect::<Vec<_>>();
    xs.push(Instruction {
        address: xs.len() as u32 * 8,
        predicate: None,
        kind: K::Exit(ops::Exit { cc_test: 15 }),
    });
    let mut module = emit_unoptimized_with_header(&xs, Some(&header)).unwrap();
    compute::observe_group(&mut module, 32);
    let original = module.words.clone();
    module.optimize();
    let cases = vec![[0; WORDS]; 64];
    let mut comparisons = 0;
    for words in [&original, &module.words] {
        let validation = spirv_validation::validate(words);
        assert!(
            validation.status.success(),
            "{}",
            String::from_utf8_lossy(&validation.stderr)
        );
        let results = gpu.run(words, &cases);
        for (i, registers) in results.iter().enumerate() {
            let lane = registers[0];
            assert!(lane < 32);
            // A complete workgroup has one Maxwell warp. Subgroup word
            // selection: InstEmitMove.cs:183-215; thread packing:118-126.
            let eq = 1u32 << lane;
            let lt = eq - 1;
            let expected = [
                lane,
                1.0f32.to_bits(),
                0,
                0,
                i as u32 % 32,
                i as u32 % 32,
                0,
                0,
                i as u32 / 32,
                0,
                0,
                eq,
                lt,
                lt | eq,
                !(lt | eq),
                !lt,
            ];
            assert_eq!(&registers[..indices.len()], &expected, "invocation{i}");
            comparisons += indices.len();
        }
    }
    eprintln!(
        "{}; S2R compute: {comparisons} exact words, 0 differences",
        gpu.description
    );
}

#[test]
fn tex_lba_lla_and_existing_lod_modes_match_explicit_glsl_on_gpu() {
    let gpu = Compute::new();
    let scratch = std::env::temp_dir().join(format!("field-probes-lod-{}", std::process::id()));
    fs::create_dir_all(&scratch).unwrap();
    let oracle = compile_glsl(
        &format!(
            "#version 450
layout(local_size_x=1) in;
layout(set=1,binding=18) uniform texture2D tex;
layout(set=1,binding=19) uniform sampler samp;
layout(set=3,binding=0,std430) readonly buffer In {{uint a[];}};
layout(set=3,binding=1,std430) writeonly buffer Out {{uint b[];}};
void main(){{uint base=gl_GlobalInvocationID.x*{WORDS}u;
vec2 p=vec2(uintBitsToFloat(a[base+2]),uintBitsToFloat(a[base+3]));
vec4 v=textureLod(sampler2D(tex,samp),p,uintBitsToFloat(a[base+21]));
for(uint c=0;c<4;c++) b[base+8+c]=floatBitsToUint(v[c]);}}"
        ),
        &scratch.join("oracle"),
    );
    let cases = (0..32)
        .map(|i| {
            let mut state = [0; WORDS];
            state[2] = ((i % 4) as f32 / 4.0 + 0.09375).to_bits();
            state[3] = ((i / 4 % 4) as f32 / 4.0 + 0.15625).to_bits();
            state[21] = ((i % 2) as f32).to_bits();
            state
        })
        .collect::<Vec<_>>();
    let expected = gpu.run(&oracle, &cases);
    let header = headers::parse(&[0; 32]).unwrap();
    let mut comparisons = 0;
    for bound in [false, true] {
        for lod in [2u64, 3, 6, 7] {
            let raw = if bound {
                0xc038u64 << 48 | lod << 55 | 9 << 36
            } else {
                0xdeb8u64 << 48 | lod << 37
            };
            let raw = raw
                | 7 << 16
                | 2 << 28
                | 15 << 31
                | (if bound { 21 } else { 20 }) << 20
                | 2 << 8
                | 8;
            let xs = [
                Instruction {
                    address: 0,
                    predicate: None,
                    kind: K::Mov(ops::Mov {
                        quad_lanes: 15,
                        dst: R(20),
                        src: S::ConstantBuffer {
                            binding: 8,
                            offset: 0,
                        },
                    }),
                },
                decode_word([raw as u32, (raw >> 32) as u32], 8),
                Instruction {
                    address: 16,
                    predicate: None,
                    kind: K::Exit(ops::Exit { cc_test: 15 }),
                },
            ];
            let mut module = emit_unoptimized_with_header(&xs, Some(&header)).unwrap();
            compute::observe_group(&mut module, 1);
            let original = module.words.clone();
            module.optimize();
            for words in [&original, &module.words] {
                let check = spirv_validation::validate(words);
                assert!(
                    check.status.success(),
                    "{}",
                    String::from_utf8_lossy(&check.stderr)
                );
                let actual = gpu.run(words, &cases);
                for (i, registers) in actual.iter().enumerate() {
                    assert_eq!(
                        &registers[8..12],
                        &expected[i][8..12],
                        "raw{raw:016x}, case{i}"
                    );
                    comparisons += 4;
                }
            }
        }
    }
    fs::remove_dir(scratch).unwrap();
    eprintln!(
        "{}; TEX LOD2/3/6/7: {comparisons} exact words, 0 differences",
        gpu.description
    );
}
