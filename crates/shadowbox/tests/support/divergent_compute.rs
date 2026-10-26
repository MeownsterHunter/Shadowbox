//! Independent CPU equations for multi-instruction GPU execution.
//! This module uses the existing compute observer, preserving all shader ops.
use super::*;
use shadowbox::ir::ops;
use shadowbox::ir::InstructionKind;

const PT: u64 = 7 << 16;
fn flow(op: u64, predicate: u8, inverted: bool) -> u64 {
    // Mesa nak/sm50.rs:225-235,3025-3110; envytools gm107.c:1877,1920-1935.
    op << 48 | u64::from(predicate) << 16 | u64::from(inverted) << 19 | 15
}
fn encode(mut xs: Vec<u64>, targets: &[(usize, usize)]) -> Vec<u32> {
    // Mesa nak/sm50.rs:3006-3016,3418-3468,3178-3187.
    let pc = |i: usize| ((i / 3 * 4 + i % 3 + 1) * 8) as i64;
    for &(from, to) in targets {
        xs[from] |= ((pc(to) - pc(from) - 8) as u64 & 0xff_ffff) << 20;
    }
    xs.push(flow(0xe300, 7, false));
    while !xs.len().is_multiple_of(3) {
        xs.push(0x50b0 << 48 | PT | 15 << 8);
    }
    xs.chunks(3)
        .flat_map(|s| [0, s[0], s[1], s[2]])
        .flat_map(|x| [x as u32, (x >> 32) as u32])
        .collect()
}
fn inputs(pattern: usize) -> Vec<[u32; WORDS]> {
    (0..32)
        .map(|lane| {
            let mut s = [0; WORDS];
            for (r, v) in s[..255].iter_mut().enumerate() {
                *v = 0x9123_0000 | (lane as u32) << 8 | r as u32;
            }
            // Lane-dependent, runtime guards. PT and RZ are architectural constants.
            // Mesa nak/sm50.rs:204-209; Ryujinx RegisterConsts.cs:5-11.
            s[256] = u32::from(match pattern {
                0 => lane < 16,
                1 => lane % 2 == 0,
                2 => lane / 4 % 2 == 0,
                _ => lane % 8 != 0,
            });
            s[257] = u32::from(lane % 3 == 0);
            s[263] = 1;
            for (i, v) in s[268..].iter_mut().enumerate() {
                *v = 0x7000_0000 | (lane as u32) << 8 | i as u32;
            }
            s
        })
        .collect()
}
struct Exact {
    gpu: Compute,
    dir: PathBuf,
    rows: Vec<String>,
    differences: usize,
    unsupported: usize,
}
impl Exact {
    fn new(family: &str) -> Self {
        let dir = PathBuf::from(std::env::var_os("CARGO_TARGET_DIR").unwrap())
            .join("gpu-stage-exact")
            .join(family);
        fs::create_dir_all(&dir).unwrap();
        let gpu = Compute::new();
        fs::write(dir.join("device.txt"), &gpu.description).unwrap();
        Self {
            gpu,
            dir,
            rows: Vec::new(),
            differences: 0,
            unsupported: 0,
        }
    }
    fn check(
        &mut self,
        name: &str,
        code: &[u32],
        input: &[[u32; WORDS]],
        expected: &[[u32; WORDS]],
        flat: bool,
    ) {
        self.check_storage(name, code, input, expected, flat, &[]);
    }
    fn check_storage(
        &mut self,
        name: &str,
        code: &[u32],
        input: &[[u32; WORDS]],
        expected: &[[u32; WORDS]],
        flat: bool,
        storage_buffers: &[shadowbox::interface::StorageBufferBinding],
    ) {
        fs::write(self.dir.join("current-case.txt"), name).unwrap();
        let decoded = decode_stream(shadowbox::QuietRun, code);
        assert!(
            decoded
                .iter()
                .all(|x| !matches!(x.kind, InstructionKind::Unknown(ops::Unknown { .. }))),
            "{name}: Unknown instruction"
        );
        let h = headers::parse(&[0; 32]).unwrap();
        let mut module = match shadowbox::emitter::emit_unoptimized_with_graphics_state(
            &decoded,
            Some(&h),
            &shadowbox::interface::GraphicsState {
                storage_buffers,
                ..Default::default()
            },
        ) {
            Ok(module) => module,
            Err(error) => {
                eprintln!("{name}: {error}");
                self.unsupported += 1;
                fs::write(
                    self.dir.join(format!("{name}-unsupported.txt")),
                    error.to_string(),
                )
                .unwrap();
                for variant in ["raw", "optimized"] {
                    self.rows.push(format!(
                        "{name}\t{variant}\tCompute\t0\t0\t0\tUNSUPPORTED\n"
                    ));
                }
                self.save();
                return;
            }
        };
        if !storage_buffers.is_empty() {
            assert!(module.requires_maximal_reconvergence);
            assert!(
                !compute::ops(&module.words).any(|(op, a)| op == Op::MemoryModel && a[0] == 5348),
                "descriptor fixture accidentally used physical addressing"
            );
            assert!(
                compute::ops(&module.words).any(|(op, a)| op == Op::Variable && a[2] == 12),
                "descriptor fixture has no translated SSBO"
            );
        }
        compute::observe_group(&mut module, 32);
        let raw = module.words.clone();
        module.optimize();
        for (variant, words) in [("raw", &raw), ("optimized", &module.words)] {
            validate(
                words,
                &self.dir.join(format!("{name}-{variant}.invalid.spv")),
            );
            let actual = self.gpu.run_images(words, input, flat).0;
            assert_eq!(actual.len(), expected.len());
            let mut defects = Vec::new();
            for (lane, (a, e)) in actual.iter().zip(expected).enumerate() {
                for (slot, (a, e)) in a.iter().zip(e).enumerate() {
                    if a != e {
                        defects.push(format!("{lane}\t{slot}\t{a:08x}\t{e:08x}\n"));
                    }
                }
            }
            let status = if defects.is_empty() {
                "MATCH"
            } else {
                "MISMATCH"
            };
            if !defects.is_empty() {
                fs::write(
                    self.dir.join(format!("{name}-{variant}-defects.tsv")),
                    format!("lane\tslot\tactual\texpected\n{}", defects.concat()),
                )
                .unwrap();
                eprintln!(
                    "{name} {variant}: {} differing words, first {}",
                    defects.len(),
                    defects[0].trim()
                );
            }
            self.differences += defects.len();
            self.rows.push(format!(
                "{name}\t{variant}\tCompute\t9600\t0\t0\t{status}\n"
            ));
        }
        self.save();
    }
    fn save(&self) {
        fs::write(
            self.dir.join("coverage.tsv"),
            format!(
                "case\tvariant\tstage\tstorage_words\tcolor_words\tdepth_words\tstatus\n{}",
                self.rows.concat()
            ),
        )
        .unwrap();
    }
    fn finish(self, count: usize) {
        assert_eq!(self.rows.len(), count * 2);
        fs::write(
            self.dir.join("complete.txt"),
            format!(
                "{count}\t{}\t{}\t{}\n",
                self.rows.len() - self.unsupported * 2,
                self.differences,
                self.unsupported
            ),
        )
        .unwrap();
        assert_eq!(
            self.unsupported, 0,
            "family has unsupported translated programs"
        );
        assert_eq!(
            self.differences,
            0,
            "exact family contains mismatches: {}",
            self.dir.display()
        );
    }
}
fn vote(mode: u64, ballot: u8, predicate: u8) -> u64 {
    // Mesa nak/sm50.rs:3233-3247: op48..50, vote dst45..48,
    // source39..43, ballot0..8. Ryujinx InstEmitWarp.cs:48-101.
    (0x50d8 | mode) << 48 | u64::from(predicate) << 45 | 1 << 39 | PT | u64::from(ballot)
}
fn reference_vote(
    s: &mut [u32; WORDS],
    ballot_slot: usize,
    predicate_slot: usize,
    mask: u32,
    true_mask: u32,
    mode: u64,
) {
    // Ryujinx InstEmitWarp.cs:88-100: independent active-mask equations.
    s[ballot_slot] = true_mask;
    s[256 + predicate_slot] = u32::from(match mode {
        0 => true_mask == mask,
        1 => true_mask != 0,
        _ => true_mask == 0 || true_mask == mask,
    });
}

#[test]
#[ignore = "RADV divergent subgroup exact execution"]
fn gpu_divergent_vote_exact() {
    let mut suite = Exact::new("divergent-vote");
    for pattern in 0..4 {
        for source in 0..3 {
            for mode in 0..3 {
                let mut input = inputs(pattern);
                for s in &mut input {
                    if source < 2 {
                        s[257] = source;
                    }
                }
                let mask = input
                    .iter()
                    .enumerate()
                    .fold(0, |m, (i, s)| m | ((1 - s[256]) << i));
                let true_mask = input
                    .iter()
                    .enumerate()
                    .fold(0, |m, (i, s)| m | (((1 - s[256]) * s[257]) << i));
                let all_true = input
                    .iter()
                    .enumerate()
                    .fold(0, |m, (i, s)| m | (s[257] << i));
                let mut expected = input.clone();
                for s in &mut expected {
                    if s[256] == 0 {
                        reference_vote(s, 50, 2, mask, true_mask, mode);
                    }
                    reference_vote(s, 51, 3, u32::MAX, all_true, mode);
                }
                let code = encode(
                    vec![
                        flow(0xe290, 7, false),
                        flow(0xf0f8, 0, false),
                        vote(mode, 50, 2),
                        flow(0xf0f8, 7, false),
                        vote(mode, 51, 3),
                    ],
                    &[(0, 4)],
                );
                suite.check(
                    &format!("vote-mode{mode}-source{source}-mask{pattern}"),
                    &code,
                    &input,
                    &expected,
                    false,
                );
            }
        }
    }
    suite.finish(36);
}

#[test]
#[ignore = "RADV partial-warp shuffle"]
fn gpu_divergent_shuffle_exact() {
    let mut suite = Exact::new("divergent-shuffle");
    for pattern in 0..4 {
        for inverted in [false, true] {
            for mode in 0..4 {
                for alias in [false, true] {
                    for branch in [false, true] {
                        let mut input = inputs(pattern);
                        // Every requested source is active, including nonzero
                        // UP/DOWN/XOR distances. No inactive-source results
                        // enter this defined-domain CPU comparison.
                        let active = |s: &[u32; WORDS]| (s[256] != 0) ^ inverted;
                        let source = if mode == 2 {
                            input.iter().rposition(active).unwrap()
                        } else {
                            input.iter().position(active).unwrap()
                        } as u32;
                        for (lane, s) in input.iter_mut().enumerate() {
                            s[20] = match mode {
                                0 => source,
                                1 => (lane as u32).wrapping_sub(source),
                                2 => source.wrapping_sub(lane as u32),
                                _ => lane as u32 ^ source,
                            };
                            s[21] = 31;
                        }
                        let predicate = if alias { 0usize } else { 2 };
                        let mut expected = input.clone();
                        for (lane, s) in expected.iter_mut().enumerate() {
                            if active(&input[lane]) {
                                s[50] = input[source as usize][10];
                                s[256 + predicate] = 1;
                            }
                        }
                        // Mesa sm50.rs:2043-2080: modes, B/C register
                        // operands20..28,39..47, predicate dst48..51.
                        // Ryujinx InstEmitWarp.cs:24-45;
                        // HelperFunctionManager.cs:245-361; whole-instruction
                        // guards: Translator.cs:359-393.
                        let guard = if branch {
                            PT
                        } else {
                            u64::from(inverted) << 19
                        };
                        let raw = 0xef10_u64 << 48
                            | (predicate as u64) << 48
                            | 21 << 39
                            | mode << 30
                            | 20 << 20
                            | guard
                            | 10 << 8
                            | 50;
                        let code = if branch {
                            encode(
                                vec![
                                    flow(0xe290, 7, false),
                                    flow(0xf0f8, 0, !inverted),
                                    raw,
                                    flow(0xf0f8, 7, false),
                                ],
                                &[(0, 4)],
                            )
                        } else {
                            encode(vec![raw], &[])
                        };
                        suite.check(&format!("shuffle-mode{mode}-invert{inverted}-mask{pattern}-alias{alias}-branch{branch}"),
                            &code,&input,&expected,false);
                    }
                }
            }
        }
    }
    suite.finish(128);
}

#[test]
#[ignore = "RADV FSWZADD arithmetic"]
fn gpu_fswzadd_exact() {
    let mut suite = Exact::new("divergent-swizzle");
    for mask in 0..=255u64 {
        let mut input = inputs(1);
        for (lane, s) in input.iter_mut().enumerate() {
            s[10] = (lane as f32 * 0.25 - 3.).to_bits();
            s[11] = (lane as f32 * -0.5 + 4.).to_bits();
        }
        let mut expected = input.clone();
        for (lane, s) in expected.iter_mut().enumerate() {
            let a = f32::from_bits(s[10]);
            let b = f32::from_bits(s[11]);
            // Mesa from_nir.rs:2463-2492,2524-2553 establishes sign order
            // for the two subtraction modes. Move uses Ryujinx's arithmetic
            // fallback, Spirv/Instructions.cs:1150-1166, including A*0.
            s[50] = match (mask >> ((lane % 4) * 2)) & 3 {
                0 => a + b,
                1 => a - b,
                2 => b - a,
                _ => a * 0. + b,
            }
            .to_bits();
        }
        let raw = 0x50f8_u64 << 48 | mask << 28 | 11 << 20 | PT | 10 << 8 | 50;
        suite.check(
            &format!("swizzle-{mask:02x}"),
            &encode(vec![raw], &[]),
            &input,
            &expected,
            false,
        );
    }
    // FTZ on both inputs and result. Integer flushing is independent of
    // host CPU denormal flags. Mesa from_nir.rs:1313-1320 and
    // nak/ir.rs:8952; encoder sm50.rs:942.
    let flush = |bits: u32| {
        if bits & 0x7fff_ffff < 0x0080_0000 {
            bits & 0x8000_0000
        } else {
            bits
        }
    };
    let edges = [
        1,
        0x8000_0001,
        0x0080_0001,
        0x0080_0000,
        0x007f_ffff,
        0x807f_ffff,
        0,
        0x8000_0000,
    ];
    for mask in [0x00u64, 0x55, 0xaa, 0xff] {
        for ftz in [false, true] {
            let mut input = inputs(0);
            for (lane, s) in input.iter_mut().enumerate() {
                s[10] = edges[lane % 8];
                s[11] = edges[(lane / 4 + 3) % 8];
            }
            let mut expected = input.clone();
            for (lane, s) in expected.iter_mut().enumerate() {
                let a = if ftz { flush(s[10]) } else { s[10] };
                let b = if ftz { flush(s[11]) } else { s[11] };
                let a = f32::from_bits(a);
                let b = f32::from_bits(b);
                let sum = match (mask >> ((lane % 4) * 2)) & 3 {
                    0 => a + b,
                    1 => a - b,
                    2 => b - a,
                    _ => a * 0. + b,
                }
                .to_bits();
                s[50] = if ftz { flush(sum) } else { sum };
            }
            let raw =
                0x50f8_u64 << 48 | u64::from(ftz) << 44 | mask << 28 | 11 << 20 | PT | 10 << 8 | 50;
            suite.check(
                &format!("swizzle-edge-{mask:02x}-ftz{ftz}"),
                &encode(vec![raw], &[]),
                &input,
                &expected,
                false,
            );
        }
    }
    suite.finish(264);
}

#[test]
#[ignore = "RADV resources in divergent regions and loops"]
fn gpu_divergent_resources_exact() {
    let mut suite = Exact::new("divergent-resources");
    for resource in ["cbuf", "flat", "texture"] {
        for pattern in 0..4 {
            for path in ["region", "nested", "loop"] {
                let looping = path == "loop";
                let mut input = inputs(pattern);
                for (lane, s) in input.iter_mut().enumerate() {
                    s[20] = compute::GUEST_BASE as u32 + (lane * 128) as u32;
                    s[21] = 0;
                    s[30] = (lane % 4) as u32;
                    s[31] = (lane / 4 % 4) as u32;
                    s[40] = (lane % 4 + 1) as u32;
                }
                // Mesa sm50.rs:1943-1951,361-368: MOV c[2][word19].
                // Ryujinx InstEmitMove.cs:12-37, InstEmitMemory.cs:82-102.
                let cb = 0x4c98_u64 << 48 | 15 << 39 | 2 << 34 | 19 << 20 | PT | 50;
                // Mesa sm50.rs:2603-2614,2659-2669,2386-2431: LDG/STG.E.B32.
                // Ryujinx InstEmitMemory.cs:106-110,512-541; flat relocation
                // follows the flat global memory contract.
                let load = 0xeed4_u64 << 48 | 1 << 45 | PT | 20 << 8 | 50;
                let store = 0xeedc_u64 << 48 | 1 << 45 | 4 << 20 | PT | 20 << 8 | 50;
                // Mesa sm50.rs:2209-2236,2112-2124: bound TLD 2D, R mask,
                // LZ, table0, coordinates R30/R31. Ryujinx InstEmitTexture.cs:56-62.
                let tex = 0xdc38_u64 << 48 | 1 << 31 | 2 << 28 | 255 << 20 | PT | 30 << 8 | 50;
                let mut ops = match resource {
                    "cbuf" => vec![cb],
                    "flat" => vec![load, store],
                    _ => vec![tex],
                };
                let mut expected = input.clone();
                for s in &mut expected {
                    if s[256] == 0 && (path != "nested" || s[257] == 0) {
                        s[50] = match resource {
                            "cbuf" => compute::cbuf(2, 19),
                            "flat" => s[268],
                            // Independent fixture image formula. Same immutable
                            // upload as support/texture.rs:61-80, red channel.
                            _ => ((1 + s[31] * 32 + s[30] * 4) as f32 / 256.).to_bits(),
                        };
                        if resource == "flat" {
                            s[269] = s[50];
                        }
                    }
                    if looping {
                        s[40] = 0;
                        s[258] = 1;
                    }
                }
                let (xs, targets) = if looping {
                    let mut xs = vec![
                        flow(0xe2a0, 7, false),
                        // ISETP.EQ.U32.AND P2,PT,R40,RZ,PT.
                        // Mesa sm50.rs:1508-1541,810-825.
                        0x5b64_u64 << 48 | 7 << 39 | 255 << 20 | PT | 40 << 8 | 2 << 3 | 7,
                        flow(0xe340, 2, false),
                        flow(0xe2b0, 7, false),
                        // IADD32I R40,R40,-1, Mesa sm50.rs:1255-1263.
                        0x1c00_u64 << 48 | u64::from(u32::MAX) << 20 | PT | 40 << 8 | 40,
                        flow(0xe350, 0, false),
                    ];
                    xs.append(&mut ops);
                    xs.push(flow(0xe350, 7, false));
                    let latch = xs.len();
                    xs.push(flow(0xe240, 7, false));
                    let end = xs.len();
                    (xs, vec![(0, end), (3, latch), (latch, 1)])
                } else if path == "nested" {
                    let mut xs = vec![
                        flow(0xe290, 7, false),
                        flow(0xf0f8, 0, false),
                        flow(0xe290, 7, false),
                        flow(0xf0f8, 1, false),
                    ];
                    xs.append(&mut ops);
                    xs.push(flow(0xf0f8, 7, false));
                    let outer_pop = xs.len();
                    xs.push(flow(0xf0f8, 7, false));
                    let end = xs.len();
                    (xs, vec![(0, end), (2, outer_pop)])
                } else {
                    let mut xs = vec![flow(0xe290, 7, false), flow(0xf0f8, 0, false)];
                    xs.append(&mut ops);
                    xs.push(flow(0xf0f8, 7, false));
                    let end = xs.len();
                    (xs, vec![(0, end)])
                };
                suite.check(
                    &format!("{resource}-{path}-mask{pattern}"),
                    &encode(xs, &targets),
                    &input,
                    &expected,
                    resource == "flat",
                );
            }
        }
    }
    suite.finish(36);
}

// Region cases cover all token kinds, nested stack kinds, and loop iterations
// with different active sets. Mesa nak/sm50.rs:3032-3105 encodes these tokens;
// Ryujinx InstEmitFlowControl.cs:239-275, Decoder.cs:824-869 consumes them.
const TOKENS: [(u64, u64); 3] = [(0xe290, 0xf0f8), (0xe2a0, 0xe340), (0xe2b0, 0xe350)];
fn masks(input: &[[u32; WORDS]], active: impl Fn(&[u32; WORDS]) -> bool) -> (u32, u32) {
    input
        .iter()
        .enumerate()
        .fold((0, 0), |(mask, truth), (lane, s)| {
            let bit = u32::from(active(s)) << lane;
            (mask | bit, truth | (bit * s[257]))
        })
}
#[test]
#[ignore = "RADV token reconvergence"]
fn gpu_token_regions_exact() {
    let mut suite = Exact::new("divergent-tokens");
    for (outer, &(push, pop)) in TOKENS.iter().enumerate() {
        for (inner, &(inner_push, inner_pop)) in TOKENS.iter().enumerate() {
            for pattern in 0..4 {
                for mode in 0..3 {
                    let input = inputs(pattern);
                    let (outer_mask, outer_true) = masks(&input, |s| s[256] == 0);
                    let (inner_mask, inner_true) = masks(&input, |s| s[256] == 0 && s[257] == 0);
                    let (_, all_true) = masks(&input, |_| true);
                    let mut expected = input.clone();
                    for (lane, s) in expected.iter_mut().enumerate() {
                        if s[256] == 0 {
                            if s[257] == 0 {
                                reference_vote(s, 50, 2, inner_mask, inner_true, mode);
                            }
                            reference_vote(s, 51, 3, outer_mask, outer_true, mode);
                        }
                        reference_vote(s, 52, 4, u32::MAX, all_true, mode);
                        s[54] = input[lane ^ 1][10];
                        s[261] = 1;
                    }
                    // SHFL.BFLY R54,P5,R10,1,0x1c03 after outer reconvergence.
                    // Mesa sm50.rs:2043-2080; Ryujinx InstEmitWarp.cs:24-45.
                    let shuffle = 0xef10_u64 << 48
                        | 5 << 48
                        | 0x1c03 << 34
                        | 1 << 28
                        | 1 << 29
                        | 3 << 30
                        | 1 << 20
                        | PT
                        | 10 << 8
                        | 54;
                    let code = encode(
                        vec![
                            flow(push, 7, false),
                            flow(pop, 0, false),
                            flow(inner_push, 7, false),
                            flow(inner_pop, 1, false),
                            vote(mode, 50, 2),
                            flow(inner_pop, 7, false),
                            vote(mode, 51, 3),
                            flow(pop, 7, false),
                            vote(mode, 52, 4),
                            shuffle,
                        ],
                        &[(0, 8), (2, 6)],
                    );
                    suite.check(
                        &format!("nested-o{outer}-i{inner}-mode{mode}-mask{pattern}"),
                        &code,
                        &input,
                        &expected,
                        false,
                    );
                }
            }
        }
    }
    suite.finish(108);
}

#[test]
#[ignore = "RADV lane-dependent token loops"]
fn gpu_token_loops_exact() {
    let mut suite = Exact::new("divergent-token-loops");
    for pattern in 0..4 {
        for mode in 0..3 {
            let mut input = inputs(pattern);
            for (lane, s) in input.iter_mut().enumerate() {
                s[40] = (lane % 4 + 1) as u32;
            }
            let (_, all_true) = masks(&input, |_| true);
            let mut expected = input.clone();
            for s in &mut expected {
                let last = s[40] - 1;
                if s[256] == 0 {
                    let (mask, truth) = masks(&input, |s| s[40] > last && s[256] == 0);
                    reference_vote(s, 50, 3, mask, truth, mode);
                }
                let (mask, truth) = masks(&input, |s| s[40] > last);
                reference_vote(s, 51, 4, mask, truth, mode);
                reference_vote(s, 52, 5, u32::MAX, all_true, mode);
                s[40] = 0;
                s[258] = 1;
            }
            // ISETP.EQ.U32 and IADD32I fields: Mesa sm50.rs:1508-1541,
            // 810-825,1255-1263. The loop is PBK/BRK; each iteration has a
            // PCNT/CONT merge. Compare before CONT, after CONT and after BRK.
            let code = encode(
                vec![
                    flow(0xe2a0, 7, false),
                    0x5b64_u64 << 48 | 7 << 39 | 255 << 20 | PT | 40 << 8 | 2 << 3 | 7,
                    flow(0xe340, 2, false),
                    flow(0xe2b0, 7, false),
                    0x1c00_u64 << 48 | u64::from(u32::MAX) << 20 | PT | 40 << 8 | 40,
                    flow(0xe350, 0, false),
                    vote(mode, 50, 3),
                    flow(0xe350, 7, false),
                    vote(mode, 51, 4),
                    flow(0xe240, 7, false),
                    vote(mode, 52, 5),
                ],
                &[(0, 10), (3, 8), (9, 1)],
            );
            suite.check(
                &format!("loop-mode{mode}-mask{pattern}"),
                &code,
                &input,
                &expected,
                false,
            );
        }
    }
    suite.finish(12);
}

#[test]
#[ignore = "RADV descriptor memory and maximal reconvergence"]
fn gpu_divergent_storage_exact() {
    let mut suite = Exact::new("divergent-storage-buffer");
    let bindings = [shadowbox::interface::StorageBufferBinding {
        bank: 2,
        offset: 76,
        binding: 9,
    }];
    for pattern in 0..4 {
        for path in ["region", "nested", "loop"] {
            let mut input = inputs(pattern);
            let active = |s: &[u32; WORDS]| s[256] == 0 && (path != "nested" || s[257] == 0);
            let writer = input.iter().position(active).unwrap();
            for (lane, s) in input.iter_mut().enumerate() {
                s[258] = u32::from(lane == writer);
                s[40] = (lane % 4 + 1) as u32;
            }
            // Mesa nak/sm50.rs:1943-1951,361-368: adjacent c[2] MOV words.
            // :2386-2431,2603-2614,2659-2669: LDG/STG.E.B32, offset4.
            // Ryujinx GlobalToStorage.cs:305-357,409-445 selects the descriptor.
            let prefix = vec![
                0x4c98_u64 << 48 | 15 << 39 | 2 << 34 | 19 << 20 | PT | 20,
                0x4c98_u64 << 48 | 15 << 39 | 2 << 34 | 20 << 20 | PT | 21,
            ];
            let mut body = vec![
                0xeed4_u64 << 48 | 1 << 45 | PT | 20 << 8 | 50,
                0xeedc_u64 << 48 | 1 << 45 | 4 << 20 | 2 << 16 | 20 << 8 | 50,
                // RZ stores zero; an RZ destination still performs the load.
                // Ryujinx InstEmitMemory.cs:457-486; RegisterConsts.cs:5-11.
                0xeedc_u64 << 48 | 1 << 45 | 8 << 20 | 2 << 16 | 20 << 8 | 255,
                0xeed4_u64 << 48 | 1 << 45 | PT | 20 << 8 | 255,
                // False runtime P5 guards must suppress invalid accesses.
                // Ryujinx Translator.cs:359-393. Offset is signed24 bytes
                // (Mesa sm50.rs:2611,2665); largest aligned positive value.
                0xeedc_u64 << 48 | 1 << 45 | 0x7f_fffc << 20 | 5 << 16 | 20 << 8 | 50,
                0xeed4_u64 << 48 | 1 << 45 | 0x7f_fffc << 20 | 5 << 16 | 20 << 8 | 52,
            ];
            let mut expected = input.clone();
            let mut backing = input
                .iter()
                .flat_map(|s| s[268..].iter().copied())
                .collect::<Vec<_>>();
            let base =
                u64::from(compute::cbuf(2, 19)) | (u64::from(compute::cbuf(2, 20) & 255) << 32);
            for (lane, s) in expected.iter_mut().enumerate() {
                for &raw in &prefix {
                    *s = reference::reference(raw, s).unwrap().words;
                }
                if active(&input[lane]) {
                    let repeats = if path == "loop" { input[lane][40] } else { 1 };
                    for _ in 0..repeats {
                        for &raw in &body {
                            *s = memory_reference::descriptor_reference(
                                raw,
                                s,
                                &[base],
                                &mut backing,
                            )
                            .unwrap()
                            .words;
                        }
                    }
                }
                if path == "loop" {
                    s[40] = 0;
                    s[260] = 1;
                }
                reference_vote(
                    s,
                    51,
                    3,
                    u32::MAX,
                    input
                        .iter()
                        .enumerate()
                        .fold(0, |m, (i, s)| m | (s[257] << i)),
                    0,
                );
            }
            for (s, memory) in expected.iter_mut().zip(backing.as_chunks::<32>().0) {
                s[268..].copy_from_slice(memory);
            }
            let (xs, targets) = if path == "loop" {
                let mut xs = prefix;
                xs.extend([
                    flow(0xe2a0, 7, false),
                    // Mesa sm50.rs:1508-1541,810-825: EQ.U32.AND P4,PT,R40,RZ,PT.
                    0x5b64_u64 << 48 | 7 << 39 | 255 << 20 | PT | 40 << 8 | 4 << 3 | 7,
                    flow(0xe340, 4, false),
                    flow(0xe2b0, 7, false),
                    // Mesa sm50.rs:1255-1263: IADD32I R40,R40,-1.
                    0x1c00_u64 << 48 | u64::from(u32::MAX) << 20 | PT | 40 << 8 | 40,
                    flow(0xe350, 0, false),
                ]);
                xs.append(&mut body);
                xs.push(flow(0xe350, 7, false));
                let latch = xs.len();
                xs.push(flow(0xe240, 7, false));
                let end = xs.len();
                xs.push(vote(0, 51, 3));
                (xs, vec![(2, end), (5, latch), (latch, 3)])
            } else if path == "nested" {
                let mut xs = prefix;
                xs.extend([
                    flow(0xe290, 7, false),
                    flow(0xf0f8, 0, false),
                    flow(0xe290, 7, false),
                    flow(0xf0f8, 1, false),
                ]);
                xs.append(&mut body);
                xs.push(flow(0xf0f8, 7, false));
                let outer_pop = xs.len();
                xs.push(flow(0xf0f8, 7, false));
                let end = xs.len();
                xs.push(vote(0, 51, 3));
                (xs, vec![(2, end), (4, outer_pop)])
            } else {
                let mut xs = prefix;
                xs.extend([flow(0xe290, 7, false), flow(0xf0f8, 0, false)]);
                xs.append(&mut body);
                xs.push(flow(0xf0f8, 7, false));
                let end = xs.len();
                xs.push(vote(0, 51, 3));
                (xs, vec![(2, end)])
            };
            suite.check_storage(
                &format!("descriptor-{path}-mask{pattern}"),
                &encode(xs, &targets),
                &input,
                &expected,
                true,
                &bindings,
            );
        }
    }
    suite.finish(12);
}

#[test]
#[ignore = "RADV compiler control-flow shapes"]
fn gpu_compiler_regions_exact() {
    let mut suite = Exact::new("divergent-compiler-regions");
    for pattern in 0..4 {
        for mode in 0..3 {
            for path in [
                "forward",
                "cc-forward",
                "shared-token-target",
                "shared-tail",
                "exit-path",
                "pred-exit",
                "nested-cont",
                "nested-break",
                "plain-loop",
                "conditional-latch",
            ] {
                let mut input = inputs(pattern);
                for (lane, s) in input.iter_mut().enumerate() {
                    s[40] = (lane % 4 + 1) as u32;
                    s[264] = u32::from(lane % 5 == 0);
                    s[265] = u32::from(lane % 7 == 0);
                    s[266] = u32::from(lane % 6 == 0);
                    s[267] = u32::from(lane % 9 == 0);
                }
                let mut expected = input.clone();
                let (_, all_true) = masks(&input, |_| true);
                let (left, left_true) = masks(&input, |s| s[256] == 0);
                let (right, right_true) = masks(&input, |s| s[256] != 0);
                let (inner, inner_true) = masks(&input, |s| s[256] == 0 && s[257] == 0);
                let (exit_body, exit_true) = masks(&input, |s| s[256] != 0 && s[257] == 0);
                // ISETP.EQ.U32 P2,PT,R40,RZ,PT and IADD32I R40,R40,-1.
                // Mesa sm50.rs:1508-1541,810-825,1255-1263.
                let compare = 0x5b64_u64 << 48 | 7 << 39 | 255 << 20 | PT | 40 << 8 | 2 << 3 | 7;
                let decrement = 0x1c00_u64 << 48 | u64::from(u32::MAX) << 20 | PT | 40 << 8 | 40;
                let (xs, targets) = match path {
                    "forward" => {
                        for s in &mut expected {
                            if s[256] == 0 {
                                if s[257] == 0 {
                                    reference_vote(s, 50, 2, inner, inner_true, mode);
                                }
                                reference_vote(s, 51, 3, left, left_true, mode);
                            } else {
                                reference_vote(s, 52, 4, right, right_true, mode);
                            }
                            reference_vote(s, 53, 5, u32::MAX, all_true, mode);
                        }
                        (
                            vec![
                                flow(0xe240, 0, false),
                                flow(0xe240, 1, false),
                                vote(mode, 50, 2),
                                vote(mode, 51, 3),
                                flow(0xe240, 7, false),
                                vote(mode, 52, 4),
                                vote(mode, 53, 5),
                            ],
                            vec![(0, 5), (1, 3), (4, 6)],
                        )
                    }
                    "cc-forward" => {
                        // CC.EQU is Z: envytools gm107.c:437 and Ryujinx
                        // InstEmitConditionCode.cs:66.
                        // The instruction guard ANDs with the CC test,
                        // Ryujinx InstEmitFlowControl.cs:293-312.
                        let (body, body_true) = masks(&input, |s| s[256] == 0 || s[264] == 0);
                        for s in &mut expected {
                            if s[256] == 0 || s[264] == 0 {
                                reference_vote(s, 50, 2, body, body_true, mode);
                            }
                            reference_vote(s, 51, 3, u32::MAX, all_true, mode);
                        }
                        (
                            vec![
                                flow(0xe240, 0, false) & !0x1f | 0xa,
                                vote(mode, 50, 2),
                                vote(mode, 51, 3),
                            ],
                            vec![(0, 2)],
                        )
                    }
                    "shared-token-target" => {
                        for s in &mut expected {
                            if s[256] == 0 && s[257] == 0 {
                                reference_vote(s, 50, 2, inner, inner_true, mode);
                            }
                            reference_vote(s, 51, 3, u32::MAX, all_true, mode);
                        }
                        // Nested SSY entries can share a forward target.
                        // Keep both pushes and resolve each SYNC by stack kind.
                        // Ryujinx Decoder.cs:800-869; InstEmitFlowControl.cs:239-275.
                        (
                            vec![
                                flow(0xe290, 7, false),
                                flow(0xf0f8, 0, false),
                                flow(0xe290, 7, false),
                                flow(0xf0f8, 1, false),
                                vote(mode, 50, 2),
                                flow(0xf0f8, 7, false),
                                vote(mode, 51, 3),
                            ],
                            vec![(0, 6), (2, 6)],
                        )
                    }
                    "shared-tail" => {
                        for s in &mut expected {
                            if s[256] == 0 {
                                if s[257] == 0 {
                                    reference_vote(s, 50, 2, inner, inner_true, mode);
                                }
                                reference_vote(s, 51, 3, left, left_true, mode);
                                s[41] = s[41].wrapping_add(1);
                            }
                            reference_vote(s, 52, 4, u32::MAX, all_true, mode);
                        }
                        // BRA escapes SSY without consuming it. Its destination is
                        // beyond the SYNC target and rejoins the slower target path.
                        (
                            vec![
                                flow(0xe290, 7, false),
                                flow(0xe240, 0, false),
                                flow(0xe240, 1, false),
                                vote(mode, 50, 2),
                                flow(0xf0f8, 7, false),
                                vote(mode, 51, 3),
                                0x1c00_u64 << 48 | 1 << 20 | PT | 41 << 8 | 41,
                                flow(0xe240, 7, false),
                                vote(mode, 52, 4),
                            ],
                            vec![(0, 5), (1, 8), (2, 4), (7, 8)],
                        )
                    }
                    "exit-path" => {
                        for s in &mut expected {
                            if s[256] != 0 {
                                if s[257] == 0 {
                                    reference_vote(s, 50, 2, exit_body, exit_true, mode);
                                }
                                reference_vote(s, 51, 3, right, right_true, mode);
                            }
                        }
                        (
                            vec![
                                flow(0xe290, 7, false),
                                flow(0xe240, 0, false),
                                flow(0xe300, 7, false),
                                flow(0xf0f8, 1, false),
                                vote(mode, 50, 2),
                                flow(0xf0f8, 7, false),
                                vote(mode, 51, 3),
                            ],
                            vec![(0, 6), (1, 3)],
                        )
                    }
                    "pred-exit" => {
                        for s in &mut expected {
                            if s[256] == 0 {
                                if s[257] == 0 {
                                    reference_vote(s, 50, 2, inner, inner_true, mode);
                                }
                                reference_vote(s, 51, 3, left, left_true, mode);
                            }
                        }
                        (
                            vec![
                                flow(0xe290, 7, false),
                                flow(0xe300, 0, false),
                                flow(0xe240, 1, false),
                                vote(mode, 50, 2),
                                flow(0xf0f8, 7, false),
                                vote(mode, 51, 3),
                            ],
                            vec![(0, 5), (2, 4)],
                        )
                    }
                    "nested-cont" => {
                        for s in &mut expected {
                            let last = s[40] - 1;
                            let (active, truth) = masks(&input, |s| s[40] > last);
                            let (body, body_true) = masks(&input, |s| s[40] > last && s[256] == 0);
                            let (part, part_true) =
                                masks(&input, |s| s[40] > last && s[256] == 0 && s[257] == 0);
                            if s[256] == 0 {
                                if s[257] == 0 {
                                    reference_vote(s, 50, 3, part, part_true, mode);
                                    reference_vote(s, 52, 5, part, part_true, mode);
                                }
                                reference_vote(s, 51, 4, body, body_true, mode);
                                reference_vote(s, 53, 6, body, body_true, mode);
                            }
                            reference_vote(s, 54, 3, active, truth, mode);
                            reference_vote(s, 55, 4, u32::MAX, all_true, mode);
                            s[40] = 0;
                            s[258] = 1;
                        }
                        // CONT consumes PCNT through both intervening SSY entries.
                        // Ryujinx Decoder.cs:824-869; InstEmitFlowControl.cs:146-150.
                        (
                            vec![
                                flow(0xe2a0, 7, false),
                                compare,
                                flow(0xe340, 2, false),
                                flow(0xe2b0, 7, false),
                                decrement,
                                flow(0xe290, 7, false),
                                flow(0xe290, 7, false),
                                flow(0xe350, 0, false),
                                flow(0xe240, 1, false),
                                vote(mode, 50, 3),
                                flow(0xf0f8, 7, false),
                                vote(mode, 51, 4),
                                flow(0xf0f8, 1, false),
                                vote(mode, 52, 5),
                                flow(0xf0f8, 7, false),
                                vote(mode, 53, 6),
                                flow(0xe350, 7, false),
                                vote(mode, 54, 3),
                                flow(0xe240, 7, false),
                                vote(mode, 55, 4),
                            ],
                            vec![(0, 19), (3, 17), (5, 15), (6, 11), (8, 10), (18, 1)],
                        )
                    }
                    "nested-break" => {
                        for s in &mut expected {
                            if s[40] > 1 {
                                let last = s[40] - 2;
                                let (active, truth) = masks(&input, |s| s[40] > last + 1);
                                let (part, part_true) =
                                    masks(&input, |s| s[40] > last + 1 && s[256] == 0);
                                if s[256] == 0 {
                                    reference_vote(s, 50, 3, part, part_true, mode);
                                }
                                reference_vote(s, 51, 4, active, truth, mode);
                                reference_vote(s, 52, 5, active, truth, mode);
                            }
                            reference_vote(s, 53, 6, u32::MAX, all_true, mode);
                            s[40] = 0;
                            s[258] = 1;
                        }
                        // BRK consumes PBK through nested SSY, including lanes
                        // which terminate the loop before either inner ballot.
                        // Ryujinx Decoder.cs:824-869; InstEmitFlowControl.cs:19-24.
                        (
                            vec![
                                flow(0xe2a0, 7, false),
                                flow(0xe290, 7, false),
                                flow(0xe290, 7, false),
                                decrement,
                                compare,
                                flow(0xe340, 2, false),
                                flow(0xe240, 0, false),
                                vote(mode, 50, 3),
                                flow(0xf0f8, 7, false),
                                vote(mode, 51, 4),
                                flow(0xf0f8, 7, false),
                                vote(mode, 52, 5),
                                flow(0xe240, 7, false),
                                vote(mode, 53, 6),
                            ],
                            vec![(0, 13), (1, 11), (2, 9), (6, 8), (12, 1)],
                        )
                    }
                    _ => {
                        for s in &mut expected {
                            let last = s[40] - 1;
                            let (active, truth) = masks(&input, |s| s[40] > last);
                            let (part, part_true) = masks(&input, |s| s[40] > last && s[256] == 0);
                            if s[256] == 0 {
                                reference_vote(s, 50, 3, part, part_true, mode);
                            }
                            reference_vote(s, 51, 4, active, truth, mode);
                            reference_vote(s, 52, 5, u32::MAX, all_true, mode);
                            s[40] = 0;
                            s[258] = 1;
                        }
                        if path == "plain-loop" {
                            (
                                vec![
                                    compare,
                                    flow(0xe240, 2, false),
                                    decrement,
                                    flow(0xe240, 0, false),
                                    vote(mode, 50, 3),
                                    vote(mode, 51, 4),
                                    flow(0xe240, 7, false),
                                    vote(mode, 52, 5),
                                ],
                                vec![(1, 7), (3, 5), (6, 0)],
                            )
                        } else {
                            (
                                vec![
                                    decrement,
                                    flow(0xe240, 0, false),
                                    vote(mode, 50, 3),
                                    vote(mode, 51, 4),
                                    compare,
                                    flow(0xe240, 2, true),
                                    vote(mode, 52, 5),
                                ],
                                vec![(1, 3), (5, 0)],
                            )
                        }
                    }
                };
                suite.check(
                    &format!("{path}-mode{mode}-mask{pattern}"),
                    &encode(xs, &targets),
                    &input,
                    &expected,
                    false,
                );
            }
        }
    }
    suite.finish(120);
}
