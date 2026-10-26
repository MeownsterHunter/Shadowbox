//! Exact divergent fragment execution using the existing unmodified draw transport.
use super::*;
#[path = "quad_reference.rs"]
mod quad_reference;

const FS_PREFIX: &str =
    "#version 450\nlayout(location=0) out vec4 color;void main(){color=vec4(0.25,0.5,0.75,1);";

// All fixture words are project-authored. Field citations are next to builders.
pub(super) fn flow(op: u64, pred: u8, invert: bool) -> u64 {
    // Mesa nak/sm50.rs:225-235,3025-3097,3106-3110,3173-3174.
    op << 48 | u64::from(pred) << 16 | u64::from(invert) << 19 | 15
}
fn fix(mut xs: Vec<u64>, targets: &[(usize, usize)]) -> Vec<u64> {
    // Mesa nak/sm50.rs:3006-3016,3418-3468: rel24 includes schedule slots.
    let pc = |i: usize| ((i / 3 * 4 + i % 3 + 1) * 8) as i64;
    for &(from, to) in targets {
        xs[from] |= ((pc(to) - pc(from) - 8) as u64 & 0xff_ffff) << 20;
    }
    xs
}
fn less(pred: u8, a: u8, b: u8) -> u64 {
    // Mesa nak/sm50.rs:871-901,798-805: FSETP.LT.AND, PT accumulator.
    0x5bb0 << 48
        | 1 << 48
        | 7 << 39
        | u64::from(b) << 20
        | PT
        | u64::from(a) << 8
        | u64::from(pred) << 3
        | 7
}
fn add(dst: u8, src: u8, immediate: u32) -> u64 {
    // Mesa nak/sm50.rs:1255-1263; gm107.c:2162: IADD32I, no CC/X/SAT/NEG.
    0x1c00 << 48 | u64::from(immediate) << 20 | PT | u64::from(src) << 8 | u64::from(dst)
}
fn swz(dst: u8, a: u8, b: u8, mask: u8) -> u64 {
    // Mesa nak/sm50.rs:913-943: both sources, four lane ops, RN, no FTZ/CC/NDV.
    0x50f8 << 48
        | u64::from(mask) << 28
        | u64::from(b) << 20
        | PT
        | u64::from(a) << 8
        | u64::from(dst)
}
fn shfl(dst: u8, src: u8, xor: u8) -> u64 {
    // Mesa nak/sm50.rs:2043-2080: BFLY, immediate B/C, mask0x1c03, PT output.
    0xef10 << 48
        | 7 << 48
        | 0x1c03 << 34
        | 3 << 30
        | 3 << 28
        | u64::from(xor) << 20
        | PT
        | u64::from(src) << 8
        | u64::from(dst)
}
pub(super) fn setup(threshold: f32) -> Vec<u64> {
    vec![
        ald(10, 0x70, 2, 255, false, false),
        mov(12, threshold.to_bits()),
        less(0, 10, 12),
        less(1, 11, 12),
        mov(0, 0.25f32.to_bits()),
        mov(1, 0.5f32.to_bits()),
        mov(2, 0.75f32.to_bits()),
        mov(3, 1f32.to_bits()),
    ]
}
fn draw(suite: &mut Suite, producer: &[u32], name: String, xs: Vec<u64>, body: &str) {
    let input = (0..6)
        .map(|i| {
            let p = [[-1., -1.], [3., -1.], [-1., 3.]][i % 3];
            [p[0], p[1], 0.25, 1., 0., 0., 0., 0.]
        })
        .collect::<Vec<_>>();
    suite.check_vertices(
        Case {
            name,
            stage: Stage::Fragment,
            xs,
            h: header(Stage::Fragment),
            state: GraphicsState::default(),
            reference: format!("{FS_PREFIX}{body}}}"),
            shaders: vec![(Stage::Vertex, producer.to_vec())],
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            discard: false,
            storage_active: false,
        },
        &input,
    );
}

#[test]
#[ignore = "RADV exact execution"]
fn gpu_divergent_control_exact() {
    let mut suite = Suite::new("divergent-control");
    let producer = suite.compile("producer", Stage::Vertex, PRODUCER);
    for threshold in [2f32, 4., 6.] {
        let base = setup(threshold);
        let n = base.len();
        let mut xs = base.clone();
        // Ryujinx InstEmitFlowControl.cs:239-275: a taken SYNC consumes
        // the SSY target; non-taken lanes fall through to the second SYNC.
        xs.extend([
            flow(0xe290, 7, false),
            flow(0xf0f8, 0, false),
            mov(0, 0.875f32.to_bits()),
            flow(0xf0f8, 7, false),
            mov(1, 0.625f32.to_bits()),
        ]);
        draw(
            &mut suite,
            &producer,
            format!("sync-partial-{threshold}"),
            fix(xs, &[(n, n + 4)]),
            &format!("if(gl_FragCoord.x>={threshold}.)color.r=0.875;color.g=0.625;"),
        );

        let mut xs = base.clone();
        xs.extend([
            flow(0xe290, 7, false),
            flow(0xe290, 7, false),
            flow(0xf0f8, 0, false),
            mov(0, 0.875f32.to_bits()),
            flow(0xf0f8, 7, false),
            flow(0xf0f8, 1, false),
            mov(1, 0.625f32.to_bits()),
            flow(0xf0f8, 7, false),
            mov(2, 0.375f32.to_bits()),
        ]);
        draw(&mut suite, &producer, format!("sync-nested-{threshold}"),
            fix(xs, &[(n,n+8),(n+1,n+5)]),
            &format!("if(gl_FragCoord.x>={threshold}.)color.r=0.875;if(gl_FragCoord.y>={threshold}.)color.g=0.625;color.b=0.375;"));

        for (name, terminal, body) in [("exit", 0xe300, "return;"), ("kill", 0xe330, "discard;")] {
            let mut xs = base.clone();
            xs.extend([
                flow(0xe290, 7, false),
                flow(terminal, 0, false),
                flow(0xf0f8, 1, false),
                mov(0, 0.875f32.to_bits()),
                flow(0xf0f8, 7, false),
                mov(1, 0.625f32.to_bits()),
            ]);
            // Ryujinx InstEmitFlowControl.cs:153-195: EXIT exports current
            // outputs, KIL discards the fragment. Both can be predicated.
            draw(&mut suite, &producer, format!("{name}-region-{threshold}"),
                fix(xs, &[(n,n+5)]),
                &format!("if(gl_FragCoord.x<{threshold}.){{{body}}}if(gl_FragCoord.y>={threshold}.)color.r=0.875;color.g=0.625;"));
        }

        let mut xs = base;
        xs.extend([
            mov(20, 1),
            p0(mov(20, 3)),
            mov(21, 0),
            flow(0xe2a0, 7, false),
            predicate_zero(20),
            flow(0xe340, 0, false),
            flow(0xe2b0, 7, false),
            add(20, 20, u32::MAX),
            predicate_zero(20),
            flow(0xe350, 0, false),
            add(21, 21, 1),
            flow(0xe350, 7, false),
            flow(0xe240, 7, false),
            mov(0, 0.875f32.to_bits()),
            predicate_zero(21),
            p0(mov(0, 0.25f32.to_bits())),
        ]);
        // PCNT is pushed once per iteration and consumed on both paths.
        // PBK remains beneath it until BRK. Ryujinx Decoder.cs:800-869,
        // InstEmitFlowControl.cs:19-24,146-150,197-209,239-275.
        draw(&mut suite, &producer, format!("break-continue-{threshold}"),
            fix(xs, &[(n+3,n+13),(n+6,n+12),(n+12,n+4)]),
            &format!("int count=gl_FragCoord.x<{threshold}.?3:1;int sum=0;while(count!=0){{--count;if(count==0)continue;sum++;}}color.r=sum==0?0.25:0.875;"));
    }
    suite.finish(15);
}

#[test]
#[ignore = "RADV fragment quads"]
fn gpu_divergent_derivatives_exact() {
    let mut suite = Suite::new("divergent-derivatives");
    let producer = suite.compile("producer", Stage::Vertex, PRODUCER);
    for threshold in [2f32, 4., 6.] {
        for y in [false, true] {
            for divergent in [false, true] {
                for ftz in [false, true] {
                    let mut xs = setup(threshold);
                    // A runtime depth condition is uniform over this draw.
                    // GLSL cannot fold gl_FragCoord.z at compile time.
                    let uniform_threshold = if threshold == 2. { 0.125f32 } else { 0.5f32 };
                    if !divergent {
                        xs.extend([
                            ald(14, 0x78, 1, 255, false, false),
                            mov(12, uniform_threshold.to_bits()),
                            less(0, 14, 12),
                        ]);
                    }
                    let region = xs.len();
                    xs.extend([flow(0xe290, 7, false), flow(0xf0f8, 0, false)]);
                    xs.extend([
                        shfl(13, if y { 11 } else { 10 }, if y { 2 } else { 1 }),
                        swz(0, 13, if y { 11 } else { 10 }, if y { 0xa5 } else { 0x99 })
                            | u64::from(ftz) << 44,
                    ]);
                    xs.push(flow(0xf0f8, 7, false));
                    xs = fix(xs, &[(region, region + 5)]);
                    // dFdxFine/dFdyFine is independent GLSL for the SHFL+FSWZADD
                    // pair. Quad-uniform branch edges keep derivative inputs
                    // defined. Ryujinx Optimizer.cs:216-288 proves the pair.
                    let deriv = if y {
                        "dFdyFine(gl_FragCoord.y)"
                    } else {
                        "dFdxFine(gl_FragCoord.x)"
                    };
                    let body = if divergent {
                        format!("if(gl_FragCoord.x>={threshold}.)color.r={deriv};")
                    } else {
                        format!("if(gl_FragCoord.z>={uniform_threshold})color.r={deriv};")
                    };
                    draw(
                        &mut suite,
                        &producer,
                        format!("quad-{threshold}-y{y}-branch{divergent}-ftz{ftz}"),
                        xs,
                        &body,
                    );
                }
            }
        }
    }
    // Recheck the actual Mesa-generated derivative corpus, including its
    // FTZ operands, against its original project-owned GLSL.
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let binary = fs::read(root.join("corpus/gamelike_derivatives.bin")).unwrap();
    let xs = binary
        .as_chunks::<8>()
        .0
        .chunks(4)
        .flat_map(|bundle| bundle[1..].iter().map(|word| u64::from_le_bytes(*word)))
        .collect();
    let input = (0..6)
        .map(|i| {
            let p = [[-1., -1.], [3., -1.], [-1., 3.]][i % 3];
            [
                p[0],
                p[1],
                0.25,
                1.,
                (p[0] + 1.) * 0.5,
                (p[1] + 1.) * 0.5,
                0.,
                0.,
            ]
        })
        .collect::<Vec<_>>();
    suite.check_vertices(
        Case {
            name: "mesa-derivative-corpus".into(),
            stage: Stage::Fragment,
            xs,
            h: fs::read(root.join("corpus/gamelike_derivatives.hdr")).unwrap(),
            state: GraphicsState::default(),
            reference: fs::read_to_string(root.join("corpus/gamelike_derivatives.frag")).unwrap(),
            shaders: vec![(Stage::Vertex, producer)],
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            discard: false,
            storage_active: false,
        },
        &input,
    );
    suite.finish(25);
}

#[test]
#[ignore = "RADV EXIT/KIL inside SSY with subgroup votes"]
fn gpu_compiler_terminal_regions_exact() {
    let mut suite = Suite::new("divergent-terminal-regions");
    let producer = suite.compile("producer", Stage::Vertex, PRODUCER);
    let input = (0..6)
        .map(|i| {
            let p = [[-1., -1.], [3., -1.], [-1., 3.]][i % 3];
            [p[0], p[1], 0.25, 1., 0., 0., 0., 0.]
        })
        .collect::<Vec<_>>();
    for threshold in [2f32, 4., 6.] {
        for (name, terminal, statement) in
            [("exit", 0xe300, "return;"), ("kill", 0xe330, "discard;")]
        {
            for (mode, glsl) in ["subgroupAll", "subgroupAny", "subgroupAllEqual"]
                .iter()
                .enumerate()
            {
                for branch_path in [false, true] {
                    let mut xs = setup(threshold);
                    let n = xs.len();
                    xs.push(flow(0xe290, 7, false));
                    let targets = if branch_path {
                        xs.extend([flow(0xe240, 0, true), flow(terminal, 7, false)]);
                        vec![(n, n + 6), (n + 1, n + 3)]
                    } else {
                        xs.push(flow(terminal, 0, false));
                        vec![(n, n + 5)]
                    };
                    xs.extend([
                        flow(0xf0f8, 1, false),
                        mov(0, 0.875f32.to_bits()),
                        flow(0xf0f8, 7, false),
                        // Mesa sm50.rs:3233-3247: ALL/ANY/EQ, P2,
                        // P1 source and R50 ballot. Ryujinx InstEmitWarp.cs:48-101.
                        (0x50d8 | mode as u64) << 48 | 2 << 45 | 1 << 39 | PT | 50,
                        // Instruction guard P2, Mesa sm50.rs:225-235.
                        mov(1, 0.625f32.to_bits()) & !PT | 2 << 16,
                    ]);
                    suite.check_vertices(
                        Case {
                            name: format!("{name}-branch{branch_path}-mode{mode}-{threshold}"),
                            stage: Stage::Fragment,
                            xs: fix(xs, &targets),
                            h: header(Stage::Fragment),
                            state: GraphicsState::default(),
                            reference: format!(
                                "#version 450\n#extension GL_KHR_shader_subgroup_vote: require\nlayout(location=0) out vec4 color;void main(){{color=vec4(0.25,0.5,0.75,1);if(gl_FragCoord.x<{threshold}.){{{statement}}}if(gl_FragCoord.y>={threshold}.)color.r=0.875;if({glsl}(gl_FragCoord.y<{threshold}.))color.g=0.625;}}"
                            ),
                            shaders: vec![(Stage::Vertex, producer.clone())],
                            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                            discard: false,
                            storage_active: false,
                        },
                        &input,
                    );
                }
            }
        }
    }
    suite.finish(36);
}

// Full quads and split quads after demotion have a defined independent oracle.
// A split branch is deliberately excluded: Vulkan leaves its sources undefined.
#[test]
#[ignore = "RADV helper quads"]
fn gpu_kil_quad_derivatives_exact() {
    let mut suite = Suite::new("divergent-kil-derivatives");
    let producer = suite.compile("producer", Stage::Vertex, PRODUCER);
    for threshold in [1f32, 3., 5., 7.] {
        for killed_axes in 0..4 {
            for y in [false, true] {
                for ftz in [false, true] {
                    let mut xs = setup(threshold);
                    if killed_axes & 1 != 0 {
                        xs.push(flow(0xe330, 0, false));
                    }
                    if killed_axes & 2 != 0 {
                        xs.push(flow(0xe330, 1, false));
                    }
                    let source = if y { 11 } else { 10 };
                    xs.extend([
                        shfl(13, source, if y { 2 } else { 1 }),
                        swz(0, 13, source, if y { 0xa5 } else { 0x99 }) | u64::from(ftz) << 44,
                        // Observe the neighbor independently of FSWZADD so
                        // a derivative peephole cannot conceal a bad SHFL.
                        // Mesa sm50.rs:1931-1936,1951: MOV R1,R13, all lanes.
                        0x5c98 << 48 | 15 << 39 | 13 << 20 | PT | 1,
                    ]);
                    let expected = quad_reference::framebuffer(&xs);
                    let kill = format!(
                        "{}{}",
                        if killed_axes & 1 != 0 {
                            format!("if(gl_FragCoord.x<{threshold}.)demote;")
                        } else {
                            String::new()
                        },
                        if killed_axes & 2 != 0 {
                            format!("if(gl_FragCoord.y<{threshold}.)demote;")
                        } else {
                            String::new()
                        }
                    );
                    let coord = if y { "y" } else { "x" };
                    // Independent GLSL checks the CPU interpreter as well as
                    // both installed Maxwell variants. Derivatives are uniform.
                    let reference = format!(
                        "#version 450\n#extension GL_EXT_demote_to_helper_invocation: require\nlayout(location=0) out vec4 color;void main(){{{kill}color=vec4(dFd{coord}Fine(gl_FragCoord.{coord}),float(int(gl_FragCoord.{coord})^1)+0.5,0.75,1);}}");
                    let input = (0..6)
                        .map(|i| {
                            let p = [[-1., -1.], [3., -1.], [-1., 3.]][i % 3];
                            [p[0], p[1], 0.25, 1., 0., 0., 0., 0.]
                        })
                        .collect::<Vec<_>>();
                    let actual = suite.check_vertices(
                        Case {
                            name: format!("kill-axes{killed_axes}-{coord}-ftz{ftz}-{threshold}"),
                            stage: Stage::Fragment,
                            xs,
                            h: header(Stage::Fragment),
                            state: GraphicsState::default(),
                            reference,
                            shaders: vec![(Stage::Vertex, producer.clone())],
                            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                            discard: false,
                            storage_active: false,
                        },
                        &input,
                    );
                    assert_eq!(actual.color, expected, "independent raw quad interpreter");
                    let expected_depth = (0..8)
                        .flat_map(|py| {
                            (0..8).map(move |px| {
                                let killed = (killed_axes & 1 != 0
                                    && (px as f32 + 0.5) < threshold)
                                    || (killed_axes & 2 != 0 && (py as f32 + 0.5) < threshold);
                                if killed { 1f32 } else { 0.25f32 }.to_bits()
                            })
                        })
                        .collect::<Vec<_>>();
                    assert_eq!(actual.depth, expected_depth);
                }
            }
        }
    }
    suite.finish(64);
}

#[test]
#[ignore = "RADV implicit derivatives after KIL"]
fn gpu_kil_implicit_lod_exact() {
    let mut suite = Suite::new("divergent-kil-implicit-lod");
    let producer = suite.compile("producer", Stage::Vertex, PRODUCER);
    let input = (0..6)
        .map(|i| {
            let p = [[-1., -1.], [3., -1.], [-1., 3.]][i % 3];
            [p[0], p[1], 0.25, 1., 0., 0., 0., 0.]
        })
        .collect::<Vec<_>>();
    for threshold in [1f32, 3., 5., 7.] {
        for killed_axes in 0..4 {
            for scale in [0.125f32, 0.5] {
                let mut xs = setup(threshold);
                if killed_axes & 1 != 0 {
                    xs.push(flow(0xe330, 0, false));
                }
                if killed_axes & 2 != 0 {
                    xs.push(flow(0xe330, 1, false));
                }
                xs.push(mov(12, scale.to_bits()));
                // FMUL: Mesa sm50.rs:681-704, RN, no modifiers.
                for reg in [10u64, 11] {
                    xs.push(0x5c68 << 48 | 12 << 20 | PT | reg << 8 | reg);
                }
                // TEX bound0, AUTO, 2D, RGBA, R10.xy coordinates, RZ srcB.
                // Mesa sm50.rs:2171-2197; older emitter full c038 opcode:
                // nv50_ir_emit_gm107.cpp:2909-2924. No fields are replaced.
                xs.push(0xc038 << 48 | 255 << 20 | 2 << 28 | 15 << 31 | PT | 10 << 8);
                let expected = quad_reference::framebuffer(&xs);
                let kill_x = if killed_axes & 1 != 0 {
                    format!("if(gl_FragCoord.x<{threshold}.)demote;")
                } else {
                    String::new()
                };
                let kill_y = if killed_axes & 2 != 0 {
                    format!("if(gl_FragCoord.y<{threshold}.)demote;")
                } else {
                    String::new()
                };
                let actual = suite.check_vertices_with_texture(Case {
                    name: format!("tex-kill{killed_axes}-scale{scale}-{threshold}"),
                    stage: Stage::Fragment,
                    xs,
                    h: header(Stage::Fragment),
                    state: GraphicsState::default(),
                    reference: format!("#version 450\n#extension GL_EXT_demote_to_helper_invocation: require\nlayout(set=1,binding=0) uniform texture2D image;layout(set=1,binding=1) uniform sampler filterState;layout(location=0) out vec4 color;void main(){{{kill_x}{kill_y}color=texture(sampler2D(image,filterState),gl_FragCoord.xy*{scale});}}"),
                    shaders: vec![(Stage::Vertex, producer.clone())],
                    topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                    discard: false,
                    storage_active: false,
                }, &input, true);
                assert_eq!(
                    actual.color, expected,
                    "raw quad TEX implicit-LOD interpreter"
                );
            }
        }
    }
    suite.finish(32);
}
