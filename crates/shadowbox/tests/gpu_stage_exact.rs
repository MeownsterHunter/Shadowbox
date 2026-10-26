#![cfg(feature = "gpu-check")]
//! Project-owned draws. References are independently authored GLSL, never IR
//! interpretations.
#[path = "support/demote_cost.rs"]
mod demote_cost;
#[path = "support/divergent_graphics.rs"]
mod divergent_graphics;
#[path = "../src/decoder/geometry_current_ast_words.rs"]
mod geometry_current_ast_words;
#[path = "support/graphics/mod.rs"]
mod graphics;
#[path = "support/spirv_validation.rs"]
mod spirv_validation;
#[path = "support/stage_reference.rs"]
mod stage_reference;

use ash::vk;
use graphics::{Draw, Graphics};
use shadowbox::ir::ops;
use shadowbox::{
    decoder::{decode_stream, decode_word},
    emitter::emit_unoptimized_with_graphics_state,
    headers::{self, Stage},
    interface::{
        GeometryInput, GraphicsState, PatchIo, PatchLayout, ShaderInput, TessellationDomain,
        TessellationSpacing, TessellationState,
    },
    ir::InstructionKind,
    translate_with_graphics_state,
};
use std::{fs, io::Write, path::PathBuf, process::Command};

// Mesa nak/sm50.rs:225-235: common predicate16..19, PT=7, no inversion.
const PT: u64 = 7 << 16;

fn mov(dst: u8, bits: u32) -> u64 {
    // Mesa nak/sm50.rs:1938-1941,1951: literal20..52, lanes12..16, dst0..8.
    (0x0100u64 << 48) | (u64::from(bits) << 20) | (15 << 12) | PT | u64::from(dst)
}
fn mov_reg(dst: u8, src: u8) -> u64 {
    // Mesa nak/sm50.rs:1936-1947: register MOV, full 32-bit lane.
    (0x5c98u64 << 48) | (15 << 39) | (u64::from(src) << 20) | PT | u64::from(dst)
}
fn ald(dst: u8, address: u16, count: u8, vertex: u8, patch: bool, output: bool) -> u64 {
    // Mesa nak/sm50.rs:2861-2876: dst, offset RZ, vertex, byte address,
    // patch31, output32, count-1 in47..49. RZ: sm50.rs:204-205.
    (0xefd8u64 << 48)
        | u64::from(dst)
        | (255 << 8)
        | PT
        | (u64::from(address) << 20)
        | (u64::from(patch) << 31)
        | (u64::from(output) << 32)
        | (u64::from(vertex) << 39)
        | (u64::from(count - 1) << 47)
}
fn ast(data: u8, address: u16, count: u8, patch: bool) -> u64 {
    // Mesa nak/sm50.rs:2886-2896: data0..8, offset8..16, vtx39..47,
    // byte address20..30, patch31, output32, count47..49.
    (0xeff0u64 << 48)
        | u64::from(data)
        | (255 << 8)
        | PT
        | (u64::from(address) << 20)
        | (u64::from(patch) << 31)
        | (1 << 32)
        | (255 << 39)
        | (u64::from(count - 1) << 47)
}
fn s2r(dst: u8, index: u8) -> u64 {
    // Mesa nak/sm50.rs:3215-3224: dst0..8, system register20..28.
    // Ryujinx Decoders/InstDecoders.cs:162: InvocationId=17;
    // Instructions/InstEmitMove.cs:82-85 maps it to the stage invocation.
    (0xf0c8u64 << 48) | u64::from(dst) | PT | (u64::from(index) << 20)
}
fn al2p(dst: u8, offset: u8, address: u16) -> u64 {
    // Mesa nak/sm50.rs:2842-2851: scalar input address, PT predicate result.
    (0xefa0u64 << 48)
        | u64::from(dst)
        | (u64::from(offset) << 8)
        | PT
        | (u64::from(address) << 20)
        | (7 << 44)
}
fn isberd(dst: u8, offset: u8) -> u64 {
    // Mesa nv50_ir_emit_gm107.cpp:2549-2553: default input offset8..16.
    // Ryujinx InstEmitAttribute.cs:224-229 propagates the local vertex index.
    (0xefd0u64 << 48) | u64::from(dst) | (u64::from(offset) << 8) | PT
}
fn physical(raw: u64, offset: u8) -> u64 {
    // Mesa nak/sm50.rs:2864-2870; Ryujinx InstDecoders.cs:887,902:
    // immediate zero, no P, non-RZ offset selects physical addressing.
    (raw & !(255 << 8)) | (u64::from(offset) << 8)
}
fn predicate_zero(reg: u8) -> u64 {
    // Mesa nak/sm50.rs:1508-1541,810-825: ISETP.EQ.U32.AND P0,PT,R, RZ,PT.
    (0x5b60u64 << 48) | 7 | (u64::from(reg) << 8) | PT | (255 << 20) | (7 << 39) | (2 << 49)
}
fn p0(raw: u64) -> u64 {
    // Mesa nak/sm50.rs:225-235: instruction predicate16..19, inversion19.
    raw & !PT
}
fn out(mode: u8, immediate: bool) -> u64 {
    // Mesa nak/sm50.rs:3260-3284: register RZ or immediate0 stream,
    // handle8..16 and dst0..8 RZ, EMIT/CUT flags39..41.
    ((if immediate { 0xf6e0u64 } else { 0xfbe0u64 }) << 48)
        | 255
        | (255 << 8)
        | PT
        | (if immediate { 0 } else { 255 << 20 })
        | (u64::from(mode) << 39)
}
fn ipa(dst: u8, address: u16, frequency: u8) -> u64 {
    // Mesa nak/sm50.rs:2905-2934: dst0..8, unused address8..16 RZ,
    // inv_w20..28 RZ, address28..38, no IDX38, offset39..47 RZ,
    // predicate result47..50 PT, no SAT51, default loc52..54, freq54..56.
    (0xe000u64 << 48)
        | u64::from(dst)
        | (255 << 8)
        | PT
        | (255 << 20)
        | (u64::from(address) << 28)
        | (255 << 39)
        | (7 << 47)
        | (u64::from(frequency) << 54)
}
fn ipa_mul(dst: u8, address: u16, inv_w: u8) -> u64 {
    // Mesa nak/sm50.rs:2905-2934: MUL frequency1, inv_w20..28.
    (ipa(dst, address, 1) & !(255 << 20)) | (u64::from(inv_w) << 20)
}
fn code(xs: &[u64]) -> Vec<u8> {
    // Mesa nak/sm50.rs:3099-3111 EXIT;3178-3184 NOP;3418-3468 bundles.
    let mut words = xs.to_vec();
    words.push((0xe300u64 << 48) | PT | 15);
    while !words.len().is_multiple_of(3) {
        words.push((0x50b0u64 << 48) | PT);
    }
    words
        .chunks(3)
        .flat_map(|s| [0, s[0], s[1], s[2]])
        .flat_map(u64::to_le_bytes)
        .collect()
}
fn header(stage: Stage) -> Vec<u8> {
    // Mesa nak/sph.rs:204-215,305-312,326-350; cla097sph.h:37-42,66-73.
    // Generic component maps: sph.rs:130-135,166-170. FS target map185-188.
    let kind = match stage {
        Stage::Vertex => 1,
        Stage::TessellationControl => 2,
        Stage::TessellationEvaluation => 3,
        Stage::Geometry => 4,
        Stage::Fragment => 5,
        Stage::Compute => unreachable!(),
    };
    let mut h = [0u32; 32];
    h[0] = kind << 10;
    h[2] = 3 << 24;
    h[3] = 7 << 24;
    h[4] = 6;
    h[6] = 255;
    h[13] = 255 << 16;
    if stage == Stage::Fragment {
        h[6] = 0xffff;
        h[18] = 15;
    }
    h.into_iter().flat_map(u32::to_le_bytes).collect()
}
fn patch_state(layout: PatchLayout) -> GraphicsState<'static> {
    let mut locations = [None; 31];
    locations[0] = Some(2);
    locations[1] = Some(3);
    GraphicsState {
        patch_io: Some(PatchIo { layout, locations }),
        tessellation: Some(TessellationState {
            domain: TessellationDomain::Triangles,
            spacing: TessellationSpacing::Equal,
            clockwise: false,
        }),
        ..Default::default()
    }
}

// Known clip positions plus finite binary32 payloads, including signed zero.
// Every triangle is distinct. Consumer SSBO floatBitsToUint preserves payloads.
fn vertices() -> Vec<[f32; 8]> {
    (0..6)
        .map(|i| {
            let p = [[-1., -1.], [1., -1.], [-1., 1.]][i % 3];
            [
                p[0],
                p[1],
                0.25,
                1.,
                i as f32 + 0.125,
                -(i as f32) - 0.25,
                if i % 2 == 0 { -0.0 } else { 0.0 },
                i as f32 + 1.,
            ]
        })
        .collect()
}
const PRODUCER: &str = "#version 450\nlayout(location=0) in vec4 p;layout(location=1) in vec4 a;layout(location=0) out vec4 v;layout(location=1) out vec4 ids;void main(){gl_Position=p;v=a;ids=vec4(uintBitsToFloat(uint(gl_VertexIndex)),uintBitsToFloat(uint(gl_InstanceIndex)),0,0);}";
const VS_REFERENCE: &str = "#version 450\nlayout(location=0) in vec4 p;layout(location=1) in vec4 a;layout(location=0) out vec4 v;layout(location=1) out vec4 ids;void main(){gl_Position=p;v=a;ids=vec4(uintBitsToFloat(uint(gl_VertexIndex)),uintBitsToFloat(uint(gl_InstanceIndex)),0,0);}";
// Capture each assembled primitive and each vertex with disjoint slots. The
// geometry input is an ordinary stage interface, no tested SPIR-V is rewritten.
// Vulkan-Docs chapters/shaders.adoc:2856-2857,2876-2881 permits duplicate
// TES/GS execution. Atomic exchanges make identical duplicate records safe.
const TRIANGLE_CAPTURE: &str = "#version 450\nlayout(triangles) in;layout(points,max_vertices=1) out;layout(location=0) in vec4 v[];layout(location=1) in vec4 ids[];layout(set=3,binding=0,std430) buffer Capture {uint words[];};void main(){uint base=uint(gl_PrimitiveIDIn)*64;atomicExchange(words[base],0x12345678u);for(int i=0;i<3;i++){uint b=base+1+uint(i)*12;for(int c=0;c<4;c++){atomicExchange(words[b+c],floatBitsToUint(gl_in[i].gl_Position[c]));atomicExchange(words[b+4+c],floatBitsToUint(v[i][c]));atomicExchange(words[b+8+c],floatBitsToUint(ids[i][c]));}}}";
const FLAT_FRAGMENT_CAPTURE: &str = "#version 450\nlayout(location=0) flat in vec4 v;layout(location=1) flat in vec4 ids;layout(location=0) out vec4 color;layout(set=3,binding=0,std430) buffer Capture {uint words[];};void main(){uint group=uint(gl_PrimitiveID)*3+floatBitsToUint(ids.x);uint b=group*512+(uint(gl_FragCoord.y)*8+uint(gl_FragCoord.x))*8;for(int c=0;c<4;c++){atomicExchange(words[b+c],floatBitsToUint(v[c]));atomicExchange(words[b+4+c],floatBitsToUint(ids[c]));}atomicExchange(words[4090+group],0x12345678u);color=v;}";

struct Suite {
    gpu: Graphics,
    scratch: PathBuf,
    report: PathBuf,
    rows: Vec<String>,
}
struct Case<'a> {
    name: String,
    stage: Stage,
    xs: Vec<u64>,
    h: Vec<u8>,
    state: GraphicsState<'a>,
    reference: String,
    shaders: Vec<(Stage, Vec<u32>)>,
    topology: vk::PrimitiveTopology,
    discard: bool,
    storage_active: bool,
}
impl Suite {
    fn new(family: &str) -> Self {
        let scratch = PathBuf::from(std::env::var_os("TMPDIR").expect("TMPDIR required"))
            .join(format!("gpu-stage-{family}-{}", std::process::id()));
        let report =
            PathBuf::from(std::env::var_os("CARGO_TARGET_DIR").expect("CARGO_TARGET_DIR required"))
                .join("gpu-stage-exact")
                .join(family);
        fs::create_dir_all(&scratch).unwrap();
        fs::create_dir_all(&report).unwrap();
        for file in ["complete.txt", "coverage.tsv", "current-case.txt"] {
            let _ = fs::remove_file(report.join(file));
        }
        let gpu = Graphics::new();
        fs::write(report.join("device.txt"), &gpu.description).unwrap();
        Self {
            gpu,
            scratch,
            report,
            rows: Vec::new(),
        }
    }
    fn compile(&self, name: &str, stage: Stage, source: &str) -> Vec<u32> {
        let suffix = match stage {
            Stage::Vertex => "vert",
            Stage::Fragment => "frag",
            Stage::Geometry => "geom",
            Stage::TessellationControl => "tesc",
            Stage::TessellationEvaluation => "tese",
            Stage::Compute => unreachable!(),
        };
        let input = self.scratch.join(format!("{name}.{suffix}"));
        let output = input.with_extension(format!("{suffix}.spv"));
        fs::write(&input, source).unwrap();
        let result = Command::new("glslc")
            .arg("--target-env=vulkan1.2")
            .arg(&input)
            .arg("-o")
            .arg(&output)
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        let bytes = fs::read(output).unwrap();
        assert!(bytes.len().is_multiple_of(4));
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect()
    }
    fn check(&mut self, case: Case<'_>) -> graphics::Observation {
        self.check_vertices(case, &vertices())
    }
    fn check_vertices(&mut self, case: Case<'_>, input: &[[f32; 8]]) -> graphics::Observation {
        self.check_vertices_with_texture(case, input, false)
    }
    fn check_vertices_with_texture(
        &mut self,
        mut case: Case<'_>,
        input: &[[f32; 8]],
        sampled: bool,
    ) -> graphics::Observation {
        fs::write(self.report.join("current-case.txt"), &case.name).unwrap();
        for &word in &case.xs {
            assert!(
                !matches!(
                    decode_word([word as u32, (word >> 32) as u32], 0).kind,
                    InstructionKind::Unknown(ops::Unknown { .. })
                ),
                "{}: required word {word:016x} is Unknown",
                case.name
            );
        }
        let bytes = code(&case.xs);
        let translated = Result::from(translate_with_graphics_state(
            shadowbox::QuietRun,
            &ShaderInput {
                code: &bytes,
                program_record: &case.h,
            },
            &case.state,
        ))
        .unwrap_or_else(|error| panic!("{}: {error:?}", case.name));
        // The graphics transport binds size 32 for actual Vote/Ballot
        // modules, including references.
        let parsed = headers::parse(
            &case
                .h
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let decoded = decode_stream(
            shadowbox::QuietRun,
            &bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect::<Vec<_>>(),
        );
        let raw = emit_unoptimized_with_graphics_state(&decoded, Some(&parsed), &case.state)
            .unwrap()
            .words;
        let reference = self.compile(&case.name, case.stage, &case.reference);
        case.shaders.push((case.stage, reference));
        case.shaders.sort_by_key(|(stage, _)| match stage {
            Stage::Vertex => 0,
            Stage::TessellationControl => 1,
            Stage::TessellationEvaluation => 2,
            Stage::Geometry => 3,
            Stage::Fragment => 4,
            Stage::Compute => unreachable!(),
        });
        let index = case
            .shaders
            .iter()
            .position(|(stage, _)| *stage == case.stage)
            .unwrap();
        for (_, module) in &case.shaders {
            let result = spirv_validation::validate(module);
            assert!(
                result.status.success(),
                "{} reference pipeline: {}",
                case.name,
                String::from_utf8_lossy(&result.stderr)
            );
        }
        let topology = case.topology;
        let discard = case.discard;
        let expected = self.gpu.run_with_texture(
            &Draw {
                shaders: &case.shaders,
                vertices: input,
                vertex_count: 6,
                topology,
                discard,
                samples: vk::SampleCountFlags::TYPE_1,
            },
            sampled,
        );
        if case.storage_active {
            assert!(
                expected.storage.contains(&0x1234_5678),
                "{}: vacuous reference capture",
                case.name
            );
        }
        assert_eq!(expected.storage.len(), 4096);
        for (variant, module) in [("raw", raw), ("optimized", translated.spirv)] {
            let result = spirv_validation::validate(&module);
            assert!(
                result.status.success(),
                "{} {variant}: {}",
                case.name,
                String::from_utf8_lossy(&result.stderr)
            );
            case.shaders[index].1 = module;
            let actual = self.gpu.run_with_texture(
                &Draw {
                    shaders: &case.shaders,
                    vertices: input,
                    vertex_count: 6,
                    topology,
                    discard,
                    samples: vk::SampleCountFlags::TYPE_1,
                },
                sampled,
            );
            for (plane, actual, expected) in [
                ("storage", &actual.storage, &expected.storage),
                ("color", &actual.color, &expected.color),
                ("depth", &actual.depth, &expected.depth),
            ] {
                assert_eq!(actual.len(), expected.len());
                let differences: Vec<_> = actual
                    .iter()
                    .zip(expected)
                    .enumerate()
                    .filter(|(_, (a, b))| a != b)
                    .map(|(i, (a, b))| format!("{i}\t{a:08x}\t{b:08x}\n"))
                    .collect();
                if !differences.is_empty() {
                    fs::write(
                        self.report
                            .join(format!("{}-{variant}-{plane}-defects.tsv", case.name)),
                        differences.concat(),
                    )
                    .unwrap();
                }
                assert!(
                    differences.is_empty(),
                    "{} {variant} {plane}: {} differing words; {}",
                    case.name,
                    differences.len(),
                    differences.first().unwrap_or(&String::new())
                );
            }
            self.rows.push(format!(
                "{}\t{variant}\t{:?}\t{}\t{}\t{}\tMATCH\n",
                case.name,
                case.stage,
                expected.storage.len(),
                expected.color.len(),
                expected.depth.len()
            ));
        }
        let mut data = fs::File::create(self.report.join("coverage.tsv")).unwrap();
        data.write_all(b"case\tvariant\tstage\tstorage_words\tcolor_words\tdepth_words\tstatus\n")
            .unwrap();
        data.write_all(self.rows.concat().as_bytes()).unwrap();
        expected
    }
    fn finish(self, cases: usize) {
        assert_eq!(self.rows.len(), cases * 2);
        fs::write(
            self.report.join("complete.txt"),
            format!("{cases}\t{}\t0\n", self.rows.len()),
        )
        .unwrap();
        fs::remove_dir_all(self.scratch).unwrap();
    }
}

#[test]
#[ignore = "requires RADV hardware, glslc and spirv-val"]
fn gpu_stage_vertex_exact() {
    let mut suite = Suite::new("vertex");
    let consumer = suite.compile("capture", Stage::Geometry, TRIANGLE_CAPTURE);
    // Ryujinx InstEmitAttribute.cs:19-100,105-143; AttributeConsts.cs:23-24:
    // generic input, output-bank reads, VertexId/InstanceId bits are observable.
    for count in 1..=4 {
        let mut xs = vec![ald(0, 0x80, 4, 255, false, false), ast(0, 0x70, 4, false)];
        for c in 0..4 {
            xs.push(mov(4 + c, 0));
        }
        xs.extend([
            ast(4, 0x80, 4, false),
            ald(4, 0x90, count, 255, false, false),
            ast(4, 0x80, count, false),
            ald(8, 0x80, 4, 255, false, true),
            ast(8, 0x80, 4, false),
            ald(12, 0x2fc, 1, 255, false, false),
            ald(13, 0x2f8, 1, 255, false, false),
            mov(14, 0),
            mov(15, 0),
            ast(12, 0x90, 4, false),
        ]);
        let components = ["x", "xy", "xyz", "xyzw"][usize::from(count - 1)];
        let reference =
            VS_REFERENCE.replace("v=a;", &format!("v=vec4(0);v.{components}=a.{components};"));
        suite.check(Case {
            name: format!("vertex-ald-ast-{count}"),
            stage: Stage::Vertex,
            xs,
            h: header(Stage::Vertex),
            state: GraphicsState::default(),
            reference,
            shaders: vec![(Stage::Geometry, consumer.clone())],
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            discard: true,
            storage_active: true,
        });
    }
    for component in 0..4 {
        let mut xs = vec![ald(0, 0x80, 4, 255, false, false), ast(0, 0x70, 4, false)];
        for r in 4..8 {
            xs.push(mov(r, 0));
        }
        xs.extend([
            ast(4, 0x80, 4, false),
            mov(20, u32::from(component) * 4),
            al2p(20, 20, 0x90),
            physical(ald(21, 0, 1, 255, false, false), 20),
            ast(21, 0x80 + component * 4, 1, false),
            ald(12, 0x2fc, 1, 255, false, false),
            ald(13, 0x2f8, 1, 255, false, false),
            mov(14, 0),
            mov(15, 0),
            ast(12, 0x90, 4, false),
        ]);
        let reference =
            VS_REFERENCE.replace("v=a;", &format!("v=vec4(0);v[{component}]=a[{component}];"));
        suite.check(Case {
            name: format!("vertex-physical-byte-component-{component}"),
            stage: Stage::Vertex,
            xs,
            h: header(Stage::Vertex),
            state: GraphicsState::default(),
            reference,
            shaders: vec![(Stage::Geometry, consumer.clone())],
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            discard: true,
            storage_active: true,
        });
    }
    // The same block-local AL2P proof used by GS/TCS/TES also permits vertex
    // physical output stores and output-bank reads. AL2P remains executable.
    // Ryujinx InstEmitAttribute.cs:12-16,43-51,118-127.
    for component in 0..4 {
        let mut xs = vec![ald(0, 0x80, 4, 255, false, false), ast(0, 0x70, 4, false)];
        for r in 4..8 {
            xs.push(mov(r, 0));
        }
        xs.extend([
            ast(4, 0x80, 4, false),
            ald(21, 0x90 + component * 4, 1, 255, false, false),
            al2p(20, 255, 0x80 + component * 4),
            physical(ast(21, 0, 1, false), 20),
            physical(ald(22, 0, 1, 255, false, true), 20),
            ast(22, 0x80 + component * 4, 1, false),
            ald(12, 0x2fc, 1, 255, false, false),
            ald(13, 0x2f8, 1, 255, false, false),
            mov(14, 0),
            mov(15, 0),
            ast(12, 0x90, 4, false),
        ]);
        suite.check(Case {
            name: format!("vertex-proven-physical-output-component-{component}"),
            stage: Stage::Vertex,
            xs,
            h: header(Stage::Vertex),
            state: GraphicsState::default(),
            reference: VS_REFERENCE
                .replace("v=a;", &format!("v=vec4(0);v[{component}]=a[{component}];")),
            shaders: vec![(Stage::Geometry, consumer.clone())],
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            discard: true,
            storage_active: true,
        });
    }
    generic_slot_matrix(&mut suite, Stage::Vertex);
    suite.finish(76);
}

// Mesa sm50.rs:2861-2876,2886-2896 defines widths 1..4 and byte offsets.
// Ryujinx InstEmitAttribute.cs:34-55,109-140 advances each lane by four
// bytes, including across generic locations. AttributeMap.cs:68 defines
// all 32 locations. Eight locations plus one crossing slot need at most
// 40 components with Position, below RADV's geometry input limit of 64.
fn generic_slot_matrix(suite: &mut Suite, stage: Stage) {
    for first in (0..32u16).step_by(8) {
        for count in 1..=4u8 {
            for component in 0..4u8 {
                let crossing = component + count > 4;
                let end = (first + 8 + u16::from(crossing)).min(32);
                let control = stage == Stage::TessellationControl;
                let geometry = stage == Stage::Geometry;
                let vertex_stage = stage == Stage::Vertex;
                let mut input_declarations = String::new();
                let mut output_declarations = String::new();
                let mut input_body = String::new();
                let mut body = String::new();
                let mut capture = String::new();
                for location in first..end {
                    let array = if vertex_stage { "" } else { "[]" };
                    input_declarations.push_str(&format!(
                        "layout(location={location}) in vec4 a{location}{array};"
                    ));
                    output_declarations.push_str(&format!(
                        "layout(location={location}) {}out vec4 v{location}{};",
                        if geometry { "flat " } else { "" },
                        if control { "[]" } else { "" }
                    ));
                    input_body.push_str(&format!("a{location}=vec4(float(gl_VertexIndex)+{location}.125,-float(gl_VertexIndex)-{location}.25,-0.0,{location}.0+1.0);"));
                    body.push_str(&format!(
                        "v{location}{}=vec4(-1);",
                        if control { "[gl_InvocationID]" } else { "" }
                    ));
                    capture.push_str(&format!("for(int c=0;c<4;c++)atomicExchange(words[b+1+{}+c],floatBitsToUint(v{location}{}[c]));", (location-first)*4, if geometry { "" } else { "[i]" }));
                }
                let mut xs = Vec::new();
                let repeats = if geometry { 3 } else { 1 };
                for selected in 0..repeats {
                    let vertex = if vertex_stage { 255 } else { 21 };
                    if control {
                        xs.push(s2r(20, 0x11));
                    } else if !vertex_stage {
                        xs.push(mov(20, selected));
                    }
                    if !vertex_stage {
                        xs.push(isberd(21, 20));
                    }
                    xs.extend([
                        ald(
                            0,
                            if vertex_stage { 0x80 } else { 0x70 },
                            4,
                            vertex,
                            false,
                            false,
                        ),
                        ast(0, 0x70, 4, false),
                    ]);
                    for r in 4..8 {
                        xs.push(mov(r, (-1f32).to_bits()));
                    }
                    for location in first..end {
                        xs.push(ast(4, 0x80 + location * 16, 4, false));
                    }
                    for location in first..first + 8 {
                        if location == 31 && crossing {
                            continue;
                        }
                        let address = 0x80 + location * 16 + u16::from(component) * 4;
                        xs.extend([
                            ald(8, address, count, vertex, false, false),
                            ast(8, address, count, false),
                            ald(12, address, count, 255, false, true),
                            ast(12, address, count, false),
                        ]);
                    }
                    if geometry {
                        xs.extend([
                            ald(24, 0x60, 1, 255, false, false),
                            ast(24, 0x60, 1, false),
                            out(1, true),
                        ]);
                    }
                }
                for location in first..first + 8 {
                    if location == 31 && crossing {
                        continue;
                    }
                    for lane in 0..count {
                        let index = u16::from(component + lane);
                        let slot = location + index / 4;
                        let lane = index % 4;
                        body.push_str(&format!(
                            "v{slot}{}[{lane}]=a{slot}{}[{lane}];",
                            if control { "[gl_InvocationID]" } else { "" },
                            if vertex_stage {
                                ""
                            } else if control {
                                "[gl_InvocationID]"
                            } else if geometry {
                                "[selected]"
                            } else {
                                "[0]"
                            }
                        ));
                    }
                }
                let producer_source = format!("#version 450\nlayout(location=0) in vec4 p;{}void main(){{gl_Position=p;{input_body}}}", input_declarations.replace("in vec4", "out vec4").replace("[]", ""));
                let mut shaders = Vec::new();
                if !vertex_stage {
                    shaders.push((
                        Stage::Vertex,
                        suite.compile("generic-producer", Stage::Vertex, &producer_source),
                    ));
                }
                let source_index = if control { "gl_InvocationID" } else { "0" };
                let position = if vertex_stage {
                    "p".to_string()
                } else {
                    format!("gl_in[{source_index}].gl_Position")
                };
                let reference = if geometry {
                    format!("#version 450\nlayout(triangles) in;layout(triangle_strip,max_vertices=3) out;{input_declarations}{output_declarations}void main(){{for(int selected=0;selected<3;selected++){{gl_Position=gl_in[selected].gl_Position;{body}gl_PrimitiveID=gl_PrimitiveIDIn;EmitVertex();}}EndPrimitive();}}")
                } else {
                    let prefix = if control {
                        "layout(vertices=3) out;"
                    } else if vertex_stage {
                        ""
                    } else {
                        "layout(triangles,equal_spacing,ccw) in;"
                    };
                    let position_output = if control {
                        "gl_out[gl_InvocationID].gl_Position"
                    } else {
                        "gl_Position"
                    };
                    let vertex_position = if vertex_stage && first == 0 {
                        ""
                    } else if vertex_stage {
                        "layout(location=0) in vec4 p;"
                    } else {
                        ""
                    };
                    let position = if vertex_stage && first == 0 {
                        "a0"
                    } else {
                        &position
                    };
                    let levels = if control {
                        "if(gl_InvocationID==0){for(int j=0;j<4;j++)gl_TessLevelOuter[j]=1;for(int j=0;j<2;j++)gl_TessLevelInner[j]=1;}"
                    } else {
                        ""
                    };
                    format!("#version 450\n{prefix}{vertex_position}{input_declarations}{output_declarations}void main(){{{position_output}={position};{body}{levels}}}")
                };
                if control {
                    xs.extend([predicate_zero(20), mov(24, 1f32.to_bits())]);
                    for address in [0, 4, 8, 12, 16, 20] {
                        xs.push(p0(ast(24, address, 1, true)));
                    }
                } else if !vertex_stage && !geometry {
                    let outputs = input_declarations.replace("in vec4", "out vec4");
                    let copies = (first..end)
                        .map(|location| {
                            format!(
                                "a{location}[gl_InvocationID]=inputA{location}[gl_InvocationID];"
                            )
                        })
                        .collect::<String>();
                    let source = format!("#version 450\nlayout(vertices=3) out;{}{outputs}void main(){{gl_out[gl_InvocationID].gl_Position=gl_in[gl_InvocationID].gl_Position;{copies}if(gl_InvocationID==0){{for(int j=0;j<4;j++)gl_TessLevelOuter[j]=1;for(int j=0;j<2;j++)gl_TessLevelInner[j]=1;}}}}", input_declarations.replace("vec4 a", "vec4 inputA"));
                    shaders.push((
                        Stage::TessellationControl,
                        suite.compile(
                            "generic-control-producer",
                            Stage::TessellationControl,
                            &source,
                        ),
                    ));
                }
                let declarations = output_declarations.replace("out vec4", "in vec4");
                let observer_stage;
                let observer;
                if geometry {
                    observer_stage = Stage::Fragment;
                    observer = format!("#version 450\n{declarations}layout(location=0) out vec4 color;layout(set=3,binding=0,std430) buffer Capture{{uint words[];}};void main(){{uint b=uint(gl_PrimitiveID)*64;atomicExchange(words[b],0x12345678u);{capture}color=vec4(0.125,0.25,0.5,1);}}");
                } else if control {
                    observer_stage = Stage::TessellationEvaluation;
                    observer = format!("#version 450\nlayout(triangles,equal_spacing,ccw) in;{declarations}layout(set=3,binding=0,std430) buffer Capture{{uint words[];}};void main(){{if(gl_TessCoord.x==1)for(int i=0;i<3;i++){{uint b=uint(gl_PrimitiveID)*128+uint(i)*40;atomicExchange(words[b],0x12345678u);{capture}}}gl_Position=vec4(gl_TessCoord.xy,0,1);}}");
                } else {
                    observer_stage = Stage::Geometry;
                    let declarations = declarations.replace(";", "[];");
                    observer = format!("#version 450\nlayout(triangles) in;layout(points,max_vertices=1) out;{declarations}layout(set=3,binding=0,std430) buffer Capture{{uint words[];}};void main(){{for(int i=0;i<3;i++){{uint b=uint(gl_PrimitiveIDIn)*128+uint(i)*40;atomicExchange(words[b],0x12345678u);{capture}}}}}");
                }
                shaders.push((
                    observer_stage,
                    suite.compile("generic-observer", observer_stage, &observer),
                ));
                let mut h = header(stage);
                // Mesa sph.rs:166-170: one output bit per generic component
                // at bit432. Declare only the observed batch for vertex output.
                h[54..70].fill(0);
                for location in first..end {
                    for component in 0..4usize {
                        let bit = 432 + usize::from(location) * 4 + component;
                        h[bit / 8] |= 1 << (bit % 8);
                    }
                }
                if geometry {
                    // Mesa sph.rs:305-312,326-350: one invocation, triangle strip.
                    h[11] = 1;
                    h[4] = 3;
                }
                suite.check(Case {
                    name: format!("{stage:?}-generic-{first}-width-{count}-component-{component}"),
                    stage,
                    xs,
                    h,
                    state: GraphicsState {
                        geometry_input: geometry.then_some(GeometryInput::Triangles),
                        ..patch_state(PatchLayout::Ryujinx)
                    },
                    reference,
                    shaders,
                    topology: if control || stage == Stage::TessellationEvaluation {
                        vk::PrimitiveTopology::PATCH_LIST
                    } else {
                        vk::PrimitiveTopology::TRIANGLE_LIST
                    },
                    discard: !geometry,
                    storage_active: true,
                });
            }
        }
    }
}

fn patch_slot_matrix(suite: &mut Suite, stage: Stage) {
    let producer = suite.compile("patch-producer", Stage::Vertex, PRODUCER);
    // Ryujinx AttributeMap.cs:90-92,109-121: 31 vec4 patch slots at0x18,
    // bounded below0x200. Mesa nak_private.h:29-39 uses generic base0x20.
    // Batch eight slots per draw, keeping the downstream geometry interface
    // below RADV's 64-component limit. Every byte slot is still observed.
    for layout in [PatchLayout::Ryujinx, PatchLayout::Mesa] {
        let base = if matches!(layout, PatchLayout::Ryujinx) {
            0x18
        } else {
            0x20
        };
        let slots = if matches!(layout, PatchLayout::Ryujinx) {
            31u16
        } else {
            30
        };
        for first in (0..slots).step_by(8) {
            let operation_end = (first + 8).min(slots);
            for count in 1..=4u8 {
                for component in 0..4u8 {
                    let end = (operation_end + u16::from(component + count > 4)).min(slots);
                    let per_vertex_width = count.min(4 - component);
                    let mut locations = [None; 31];
                    let mut declarations = String::new();
                    let mut patch_body = String::new();
                    let mut capture_body = String::new();
                    let mut xs = Vec::new();
                    let control = stage == Stage::TessellationControl;
                    let vertex = if control { 20 } else { 255 };
                    if control {
                        xs.push(s2r(20, 0x11));
                    }
                    xs.extend([
                        isberd(21, vertex),
                        ald(0, 0x70, 4, 21, false, false),
                        ast(0, 0x70, 4, false),
                    ]);
                    for r in 4..8 {
                        xs.push(mov(r, (-1f32).to_bits()));
                    }
                    xs.extend([
                        ast(4, 0x80, 4, false),
                        ald(
                            4,
                            0x80 + u16::from(component) * 4,
                            per_vertex_width,
                            21,
                            false,
                            false,
                        ),
                        ast(4, 0x80 + u16::from(component) * 4, per_vertex_width, false),
                    ]);
                    if control {
                        xs.push(predicate_zero(20));
                        xs.push(mov(24, 1f32.to_bits()));
                        for address in [0, 4, 8, 12, 16, 20] {
                            xs.push(p0(ast(24, address, 1, true)));
                        }
                    }
                    // Initialize the whole batch before cross-slot writes.
                    // Otherwise the next slot's initializer could erase the
                    // very crossing components this experiment observes.
                    xs.push(mov(32, (-1f32).to_bits()));
                    for slot in first..end {
                        patch_body.push_str(&format!("p{slot}=vec4(-1);"));
                        let valid = ((0x200 - (base + slot * 16)) / 4).min(4);
                        for c in 0..valid {
                            xs.push(if control {
                                p0(ast(32, base + slot * 16 + c * 4, 1, true))
                            } else {
                                ast(32, 0xa0 + (slot - first) * 16 + c * 4, 1, false)
                            });
                        }
                    }
                    for slot in first..end {
                        let local = slot - first;
                        locations[usize::from(slot)] = Some(2 + u32::from(local));
                        declarations.push_str(&format!(
                            "layout(location={}) patch {} vec4 p{slot};",
                            2 + local,
                            if control { "out" } else { "in" }
                        ));
                        let valid = ((0x200 - (base + slot * 16)) / 4).min(4);
                        let remaining = (0x200 - (base + slot * 16)) / 4;
                        let width = if slot < operation_end {
                            u16::from(count).min(remaining.saturating_sub(u16::from(component)))
                        } else {
                            0
                        };
                        let address = base + slot * 16;
                        for c in 0..valid {
                            let b = local * 4 + c;
                            capture_body.push_str(&format!(
                                "atomicExchange(words[b+9+{b}],floatBitsToUint(p{slot}[{c}]));"
                            ));
                        }
                        for c in 0..width {
                            let lane = u16::from(component) + c;
                            let target_slot = slot + lane / 4;
                            let target_component = lane % 4;
                            let value = match target_component {
                                0 => f32::from(target_slot + 1),
                                1 => -f32::from(target_slot + 1),
                                2 => f32::from(target_slot * 4 + target_component) + 0.125,
                                _ => -0.,
                            };
                            xs.push(mov(32 + c as u8, value.to_bits()));
                            patch_body.push_str(&format!(
                                "p{target_slot}[{target_component}]={value:?};"
                            ));
                        }
                        if width != 0 {
                            let address = address + u16::from(component) * 4;
                            if control {
                                xs.extend([
                                    p0(ast(32, address, width as u8, true)),
                                    p0(ald(36, address, width as u8, 255, true, true)),
                                    p0(ast(36, address, width as u8, true)),
                                ]);
                            } else {
                                xs.extend([
                                    ald(36, address, width as u8, 255, true, false),
                                    ast(
                                        36,
                                        0xa0 + local * 16 + u16::from(component) * 4,
                                        width as u8,
                                        false,
                                    ),
                                ]);
                            }
                        }
                    }
                    let lanes =
                        &"xyzw"[usize::from(component)..usize::from(component + per_vertex_width)];
                    // The reference selects source components directly; it never
                    // interprets the constructed words or translated IR.
                    let per_vertex = if control {
                        format!("gl_out[gl_InvocationID].gl_Position=gl_in[gl_InvocationID].gl_Position;v[gl_InvocationID]=vec4(-1);v[gl_InvocationID].{lanes}=inputV[gl_InvocationID].{lanes};")
                    } else {
                        format!("gl_Position=gl_in[0].gl_Position;v=vec4(-1);v.{lanes}=inputV[0].{lanes};")
                    };
                    let observer;
                    let reference;
                    let mut shaders = vec![(Stage::Vertex, producer.clone())];
                    if control {
                        reference = format!("#version 450\nlayout(vertices=3) out;layout(location=0) in vec4 inputV[];layout(location=0) out vec4 v[];{declarations}void main(){{{per_vertex}if(gl_InvocationID==0){{for(int i=0;i<4;i++)gl_TessLevelOuter[i]=1;for(int i=0;i<2;i++)gl_TessLevelInner[i]=1;{patch_body}}}}}");
                        let declarations = declarations.replace("patch out", "patch in");
                        let source = format!("#version 450\nlayout(triangles,equal_spacing,ccw) in;layout(location=0) in vec4 v[];{declarations}layout(set=3,binding=0,std430) buffer Capture{{uint words[];}};void main(){{if(gl_TessCoord.x==1){{for(int i=0;i<3;i++){{uint b=uint(gl_PrimitiveID)*192+uint(i)*64;atomicExchange(words[b],0x12345678u);for(int c=0;c<4;c++){{atomicExchange(words[b+1+c],floatBitsToUint(gl_in[i].gl_Position[c]));atomicExchange(words[b+5+c],floatBitsToUint(v[i][c]));}}{capture_body}}}}}gl_Position=vec4(gl_TessCoord.xy,0,1);}}");
                        observer =
                            suite.compile("patch-observer", Stage::TessellationEvaluation, &source);
                        shaders.push((Stage::TessellationEvaluation, observer));
                    } else {
                        let producer_declarations = declarations.replace("patch in", "patch out");
                        let source = format!("#version 450\nlayout(vertices=3) out;layout(location=0) in vec4 inputV[];layout(location=0) out vec4 v[];{producer_declarations}void main(){{gl_out[gl_InvocationID].gl_Position=gl_in[gl_InvocationID].gl_Position;v[gl_InvocationID]=inputV[gl_InvocationID];if(gl_InvocationID==0){{for(int i=0;i<4;i++)gl_TessLevelOuter[i]=1;for(int i=0;i<2;i++)gl_TessLevelInner[i]=1;{patch_body}}}}}");
                        shaders.push((
                            Stage::TessellationControl,
                            suite.compile("patch-input", Stage::TessellationControl, &source),
                        ));
                        let outputs = (first..end)
                            .map(|slot| {
                                format!("layout(location={}) out vec4 p{slot};", 2 + slot - first)
                            })
                            .collect::<String>();
                        let copies = (first..end)
                            .map(|slot| format!("p{slot}=inputP{slot};"))
                            .collect::<String>();
                        reference = format!("#version 450\nlayout(triangles,equal_spacing,ccw) in;layout(location=0) in vec4 inputV[];layout(location=0) out vec4 v;{}{outputs}void main(){{{per_vertex}{copies}}}", declarations.replace("vec4 p", "vec4 inputP"));
                        let inputs = outputs.replace("out vec4", "in vec4").replace(";", "[];");
                        let capture = (first..end).fold(capture_body.clone(), |body, slot| {
                            body.replace(&format!("p{slot}["), &format!("p{slot}[i]["))
                        });
                        let source = format!("#version 450\nlayout(triangles) in;layout(points,max_vertices=1) out;layout(location=0) in vec4 v[];{inputs}layout(set=3,binding=0,std430) buffer Capture{{uint words[];}};void main(){{for(int i=0;i<3;i++){{uint b=uint(gl_PrimitiveIDIn)*192+uint(i)*64;atomicExchange(words[b],0x12345678u);for(int c=0;c<4;c++){{atomicExchange(words[b+1+c],floatBitsToUint(gl_in[i].gl_Position[c]));atomicExchange(words[b+5+c],floatBitsToUint(v[i][c]));}}{capture}}}}}");
                        shaders.push((
                            Stage::Geometry,
                            suite.compile("patch-observer", Stage::Geometry, &source),
                        ));
                    }
                    let mut state = patch_state(layout);
                    state.patch_io = Some(PatchIo { layout, locations });
                    suite.check(Case {
                        name: format!("{stage:?}-{layout:?}-patch-{first}-width-{count}-component-{component}"),
                        stage,
                        xs,
                        h: header(stage),
                        state,
                        reference,
                        shaders,
                        topology: vk::PrimitiveTopology::PATCH_LIST,
                        discard: true,
                        storage_active: true,
                    });
                }
            }
        }
    }
}

#[test]
#[ignore = "requires RADV hardware, glslc and spirv-val"]
fn gpu_stage_tess_control_exact() {
    let mut suite = Suite::new("tess-control");
    let producer = suite.compile("producer", Stage::Vertex, PRODUCER);
    let capture = "#version 450\nlayout(triangles,equal_spacing,ccw) in;layout(location=0) in vec4 v[];layout(location=1) in vec4 ids[];layout(location=2) patch in vec4 patch0;layout(set=3,binding=0,std430) buffer Capture {uint words[];};void main(){if(gl_TessCoord.x==1.0){uint b=uint(gl_PrimitiveID)*128;atomicExchange(words[b],0x12345678u);for(int i=0;i<3;i++)for(int c=0;c<4;c++){atomicExchange(words[b+1+i*12+c],floatBitsToUint(gl_in[i].gl_Position[c]));atomicExchange(words[b+5+i*12+c],floatBitsToUint(v[i][c]));atomicExchange(words[b+9+i*12+c],floatBitsToUint(ids[i][c]));}for(int i=0;i<4;i++){atomicExchange(words[b+40+i],floatBitsToUint(gl_TessLevelOuter[i]));atomicExchange(words[b+48+i],floatBitsToUint(patch0[i]));}for(int i=0;i<2;i++)atomicExchange(words[b+44+i],floatBitsToUint(gl_TessLevelInner[i]));}gl_Position=vec4(gl_TessCoord.xy,0,1);}";
    let observer = suite.compile("capture", Stage::TessellationEvaluation, capture);
    for layout in [PatchLayout::Ryujinx, PatchLayout::Mesa] {
        let base = if matches!(layout, PatchLayout::Ryujinx) {
            0x18
        } else {
            0x20
        };
        let mut xs = vec![
            s2r(20, 0x11),
            isberd(21, 20),
            ald(0, 0x70, 4, 21, false, false),
            ast(0, 0x70, 4, false),
            ald(4, 0x80, 4, 21, false, false),
            ast(4, 0x80, 4, false),
            mov(8, 0),
            mov(9, 0),
            mov(10, 0),
            mov(11, 0),
            ast(8, 0x90, 4, false),
            predicate_zero(20),
        ];
        for r in 24..30 {
            let level = if r < 28 {
                f32::from(r - 23)
            } else {
                f32::from(r - 26)
            };
            xs.push(mov(r, level.to_bits()));
        }
        xs.extend([p0(ast(24, 0, 4, true)), p0(ast(28, 0x10, 2, true))]);
        for (r, v) in [(32, 0.125f32), (33, -2.), (34, 8.), (35, -0.0)] {
            xs.push(mov(r, v.to_bits()));
        }
        xs.push(p0(ast(32, base, 4, true)));
        // BAR.SYNC: Mesa nak/sm50.rs:3114-3124. Patch/output reads occur
        // after all invocations have completed the producer writes.
        xs.push((0xf0a8u64 << 48) | PT | (7 << 39) | (1 << 43) | (1 << 44));
        xs.extend([
            ald(8, base, 1, 255, true, true),
            ast(8, 0x90, 1, false),
            ast(20, 0x94, 1, false),
            ald(10, 0x84, 1, 255, false, true),
            ast(10, 0x98, 1, false),
            ald(11, 0x60, 1, 255, false, false),
            ast(11, 0x9c, 1, false),
        ]);
        let reference = "#version 450\nlayout(vertices=3) out;layout(location=0) in vec4 inputV[];layout(location=0) out vec4 v[];layout(location=1) out vec4 ids[];layout(location=2) patch out vec4 patch0;void main(){gl_out[gl_InvocationID].gl_Position=gl_in[gl_InvocationID].gl_Position;v[gl_InvocationID]=inputV[gl_InvocationID];if(gl_InvocationID==0){for(int i=0;i<4;i++)gl_TessLevelOuter[i]=float(i+1);for(int i=0;i<2;i++)gl_TessLevelInner[i]=float(i+2);patch0=vec4(0.125,-2,8,-0.0);}barrier();ids[gl_InvocationID]=vec4(patch0.x,uintBitsToFloat(uint(gl_InvocationID)),v[gl_InvocationID].y,uintBitsToFloat(uint(gl_PrimitiveID)));}";
        suite.check(Case {
            name: format!("tcs-{layout:?}-patch-levels-ids-output-read"),
            stage: Stage::TessellationControl,
            xs,
            h: header(Stage::TessellationControl),
            state: patch_state(layout),
            reference: reference.into(),
            shaders: vec![
                (Stage::Vertex, producer.clone()),
                (Stage::TessellationEvaluation, observer.clone()),
            ],
            topology: vk::PrimitiveTopology::PATCH_LIST,
            discard: true,
            storage_active: true,
        });
    }
    patch_slot_matrix(&mut suite, Stage::TessellationControl);
    generic_slot_matrix(&mut suite, Stage::TessellationControl);
    suite.finish(194);
}

const TCS_PRODUCER: &str = "#version 450\nlayout(vertices=3) out;layout(location=0) in vec4 inputV[];layout(location=0) out vec4 v[];layout(location=2) patch out vec4 patch0;void main(){gl_out[gl_InvocationID].gl_Position=gl_in[gl_InvocationID].gl_Position;v[gl_InvocationID]=inputV[gl_InvocationID];if(gl_InvocationID==0){for(int i=0;i<4;i++)gl_TessLevelOuter[i]=1;for(int i=0;i<2;i++)gl_TessLevelInner[i]=1;patch0=vec4(0.125,-2,8,-0.0);}}";

#[test]
#[ignore = "requires RADV hardware, glslc and spirv-val"]
fn gpu_stage_tess_evaluation_exact() {
    let mut suite = Suite::new("tess-evaluation");
    let producer = suite.compile("producer", Stage::Vertex, PRODUCER);
    let tcs = suite.compile("tcs", Stage::TessellationControl, TCS_PRODUCER);
    let observer = suite.compile("capture", Stage::Geometry, TRIANGLE_CAPTURE);
    for layout in [PatchLayout::Ryujinx, PatchLayout::Mesa] {
        let base = if matches!(layout, PatchLayout::Ryujinx) {
            0x18
        } else {
            0x20
        };
        for vertex in 0..3 {
            for component in 0..4 {
                let xs = vec![
                    mov(20, vertex),
                    isberd(21, 20),
                    ald(0, 0x70, 4, 21, false, false),
                    ast(0, 0x70, 4, false),
                    ald(4, 0x80, 2, 21, false, false),
                    ast(4, 0x80, 2, false),
                    ald(6, 0x2f0, 2, 255, false, false),
                    ast(6, 0x88, 2, false),
                    ald(8, base + component * 4, 1, 255, true, false),
                    ald(9, component * 4, 1, 255, true, false),
                    ald(10, 0x10 + (component % 2) * 4, 1, 255, true, false),
                    ald(11, 0x60, 1, 255, false, false),
                    ast(8, 0x90, 4, false),
                ];
                let inner = component % 2;
                let reference = format!("#version 450\nlayout(triangles,equal_spacing,ccw) in;layout(location=0) in vec4 inputV[];layout(location=2) patch in vec4 patch0;layout(location=0) out vec4 v;layout(location=1) out vec4 ids;void main(){{gl_Position=gl_in[{vertex}].gl_Position;v=vec4(inputV[{vertex}].xy,gl_TessCoord.xy);ids=vec4(patch0[{component}],gl_TessLevelOuter[{component}],gl_TessLevelInner[{inner}],uintBitsToFloat(uint(gl_PrimitiveID)));}}");
                suite.check(Case {
                    name: format!(
                        "tes-{layout:?}-vertex-{vertex}-patch-component-{component}-coordinates"
                    ),
                    stage: Stage::TessellationEvaluation,
                    xs,
                    h: header(Stage::TessellationEvaluation),
                    state: patch_state(layout),
                    reference,
                    shaders: vec![
                        (Stage::Vertex, producer.clone()),
                        (Stage::TessellationControl, tcs.clone()),
                        (Stage::Geometry, observer.clone()),
                    ],
                    topology: vk::PrimitiveTopology::PATCH_LIST,
                    discard: true,
                    storage_active: true,
                });
            }
        }
    }
    patch_slot_matrix(&mut suite, Stage::TessellationEvaluation);
    // Ryujinx SpirvGenerator.cs:194-237 maps all domains, spacing modes and
    // driver-adjusted winding. Noninteger levels exercise fractional placement.
    let tcs = suite.compile(
        "fractional-producer",
        Stage::TessellationControl,
        &TCS_PRODUCER
            .replace("Outer[i]=1", "Outer[i]=2.25")
            .replace("Inner[i]=1", "Inner[i]=2.25"),
    );
    for (domain, domain_name) in [
        (TessellationDomain::Triangles, "triangles"),
        (TessellationDomain::Quads, "quads"),
        (TessellationDomain::Isolines, "isolines"),
    ] {
        let consumer = if matches!(domain, TessellationDomain::Isolines) {
            suite.compile(
                "line-capture",
                Stage::Geometry,
                &TRIANGLE_CAPTURE
                    .replace("layout(triangles)", "layout(lines)")
                    .replace("i<3", "i<2"),
            )
        } else {
            observer.clone()
        };
        for (spacing, spacing_name) in [
            (TessellationSpacing::Equal, "equal_spacing"),
            (
                TessellationSpacing::FractionalEven,
                "fractional_even_spacing",
            ),
            (TessellationSpacing::FractionalOdd, "fractional_odd_spacing"),
        ] {
            for clockwise in [false, true] {
                let order = if clockwise { "cw" } else { "ccw" };
                let xs = vec![
                    ald(0, 0x2f0, 2, 255, false, false),
                    mov(2, 0.25f32.to_bits()),
                    mov(3, 1f32.to_bits()),
                    ast(0, 0x70, 4, false),
                    ald(6, 0, 2, 255, true, false),
                    ast(0, 0x80, 2, false),
                    ast(6, 0x88, 2, false),
                    ald(8, 0x60, 1, 255, false, false),
                    mov(9, 0),
                    mov(10, 0),
                    mov(11, 0),
                    ast(8, 0x90, 4, false),
                ];
                let reference = format!("#version 450\nlayout({domain_name},{spacing_name},{order}) in;layout(location=0) out vec4 v;layout(location=1) out vec4 ids;void main(){{gl_Position=vec4(gl_TessCoord.xy,0.25,1);v=vec4(gl_TessCoord.xy,gl_TessLevelOuter[0],gl_TessLevelOuter[1]);ids=vec4(uintBitsToFloat(uint(gl_PrimitiveID)),0,0,0);}}");
                suite.check(Case {
                    name: format!("tes-domain-{domain_name}-spacing-{spacing_name}-{order}"),
                    stage: Stage::TessellationEvaluation,
                    xs,
                    h: header(Stage::TessellationEvaluation),
                    state: GraphicsState {
                        tessellation: Some(TessellationState {
                            domain,
                            spacing,
                            clockwise,
                        }),
                        ..Default::default()
                    },
                    reference,
                    shaders: vec![
                        (Stage::Vertex, producer.clone()),
                        (Stage::TessellationControl, tcs.clone()),
                        (Stage::Geometry, consumer.clone()),
                    ],
                    topology: vk::PrimitiveTopology::PATCH_LIST,
                    discard: true,
                    storage_active: true,
                });
            }
        }
    }
    generic_slot_matrix(&mut suite, Stage::TessellationEvaluation);
    suite.finish(234);
}

#[test]
#[ignore = "requires RADV hardware, glslc and spirv-val"]
fn gpu_stage_geometry_exact() {
    let mut suite = Suite::new("geometry");
    let producer = suite.compile("producer", Stage::Vertex, PRODUCER);
    let observer = suite.compile("capture", Stage::Fragment, FLAT_FRAGMENT_CAPTURE);
    for immediate in [false, true] {
        for physical_address in [false, true] {
            for rotate in 0..3 {
                for combined_cut in [false, true] {
                    let mut xs = Vec::new();
                    let mut body = String::new();
                    // Two independent strips. CUT must prevent the inter-strip bridge.
                    // Ryujinx InstEmitAttribute.cs:253-294 maps EMIT then CUT.
                    for strip in 0..2 {
                        for vertex in 0..3 {
                            let selected = (vertex + rotate) % 3;
                            let origin = if strip == 0 { -0.75f32 } else { 0.25 };
                            let x = origin + if vertex == 1 { 0.5 } else { 0.0 };
                            let y = origin + if vertex == 2 { 0.5 } else { 0.0 };
                            xs.extend([
                                mov(20, selected),
                                isberd(21, 20),
                                ald(0, 0x70, 4, 21, false, false),
                                mov(0, x.to_bits()),
                                mov(1, y.to_bits()),
                                ast(0, 0x70, 4, false),
                            ]);
                            if physical_address {
                                for c in 0..4 {
                                    xs.extend([
                                        al2p(22, 255, 0x80 + c * 4),
                                        physical(ald(4 + c as u8, 0, 1, 21, false, false), 22),
                                    ]);
                                }
                            } else {
                                xs.push(ald(4, 0x80, 4, 21, false, false));
                            }
                            if physical_address {
                                for c in 0..4 {
                                    xs.extend([
                                        al2p(22, 255, 0x80 + c * 4),
                                        physical(ast(4 + c as u8, 0, 1, false), 22),
                                    ]);
                                }
                            } else {
                                xs.push(ast(4, 0x80, 4, false));
                            }
                            xs.extend([
                                s2r(8, 0x11),
                                ald(9, 0x60, 1, 255, false, false),
                                mov(10, 0),
                                mov(11, 0),
                                ast(8, 0x90, 4, false),
                                ast(9, 0x60, 1, false),
                            ]);
                            // Output-bank load and store must preserve the same value.
                            xs.extend([
                                ald(12, 0x84, 1, 255, false, true),
                                ast(12, 0x84, 1, false),
                                out(if vertex == 2 && combined_cut { 3 } else { 1 }, immediate),
                            ]);
                            body.push_str(&format!("gl_Position=vec4({x:?},{y:?},gl_in[{selected}].gl_Position.zw);v=inputV[{selected}];ids=vec4(uintBitsToFloat(uint(gl_InvocationID)),uintBitsToFloat(uint(gl_PrimitiveIDIn)),0,0);gl_PrimitiveID=gl_PrimitiveIDIn;EmitVertex();"));
                        }
                        if !combined_cut {
                            xs.push(out(2, immediate));
                        }
                        body.push_str("EndPrimitive();");
                    }
                    let reference=format!("#version 450\nlayout(triangles,invocations=3) in;layout(triangle_strip,max_vertices=6) out;layout(location=0) in vec4 inputV[];layout(location=0) flat out vec4 v;layout(location=1) flat out vec4 ids;void main(){{{body}}}");
                    suite.check(Case {
                name: format!("gs-stream0-immediate-{immediate}-physical-{physical_address}-rotate-{rotate}-combined-{combined_cut}"),
                stage: Stage::Geometry,
                xs,
                h: header(Stage::Geometry),
                state: GraphicsState {
                    geometry_input: Some(GeometryInput::Triangles),
                    ..Default::default()
                },
                reference,
                shaders: vec![
                    (Stage::Vertex, producer.clone()),
                    (Stage::Fragment, observer.clone()),
                ],
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                discard: false,
                storage_active: true,
            });
                }
            }
        }
    }
    // Ryujinx EmitterContext.cs:386-426 forwards primary vertices at return:
    // lines adjacency1,2; triangles adjacency0,2,4. AttributeUsage.cs:91-97
    // excludes whole output-used locations from automatic forwarding.
    let passthrough_capture = FLAT_FRAGMENT_CAPTURE.replace(
        "uint group=uint(gl_PrimitiveID)*3+floatBitsToUint(ids.x);",
        "uint group=uint(gl_PrimitiveID);",
    );
    let capture = suite.compile("passthrough-capture", Stage::Fragment, &passthrough_capture);
    for (input, topology, glsl_input, glsl_output, primary) in [
        (
            GeometryInput::Lines,
            vk::PrimitiveTopology::LINE_LIST,
            "lines",
            "line_strip",
            vec![0, 1],
        ),
        (
            GeometryInput::LinesAdjacency,
            vk::PrimitiveTopology::LINE_LIST_WITH_ADJACENCY,
            "lines_adjacency",
            "line_strip",
            vec![1, 2],
        ),
        (
            GeometryInput::Triangles,
            vk::PrimitiveTopology::TRIANGLE_LIST,
            "triangles",
            "triangle_strip",
            vec![0, 1, 2],
        ),
        (
            GeometryInput::TrianglesAdjacency,
            vk::PrimitiveTopology::TRIANGLE_LIST_WITH_ADJACENCY,
            "triangles_adjacency",
            "triangle_strip",
            vec![0, 2, 4],
        ),
    ] {
        for overwrite in [false, true] {
            let mut h = header(Stage::Geometry);
            // Mesa nak/sph.rs:240-243,305-312,326-350: passthrough24,
            // invocation count88..96, output topology120..124 (lines6/triangles7).
            h[3] |= 1;
            h[11] = 1;
            h[15] = if primary.len() == 2 { 6 } else { 7 };
            let mut xs = vec![ald(12, 0x60, 1, 255, false, false), ast(12, 0x60, 1, false)];
            if overwrite {
                for (r, value) in [0.125f32, 0.25, 0.5, 1.].into_iter().enumerate() {
                    xs.push(mov(r as u8, value.to_bits()));
                }
                xs.push(ast(0, 0x80, 4, false));
            }
            let mut body = String::new();
            for vertex in &primary {
                let payload = if overwrite {
                    "vec4(0.125,0.25,0.5,1)".into()
                } else {
                    format!("inputV[{vertex}]")
                };
                body.push_str(&format!("gl_Position=gl_in[{vertex}].gl_Position;v={payload};ids=inputIds[{vertex}];gl_PrimitiveID=gl_PrimitiveIDIn;EmitVertex();"));
            }
            let reference = format!("#version 450\nlayout({glsl_input}) in;layout({glsl_output},max_vertices={}) out;layout(location=0) in vec4 inputV[];layout(location=1) in vec4 inputIds[];layout(location=0) flat out vec4 v;layout(location=1) flat out vec4 ids;void main(){{{body}EndPrimitive();}}", primary.len());
            suite.check(Case {
                name: format!("gs-passthrough-{input:?}-overwrite-{overwrite}"),
                stage: Stage::Geometry,
                xs,
                h,
                state: GraphicsState {
                    geometry_input: Some(input),
                    geometry_next_inputs: Some(3),
                    ..Default::default()
                },
                reference,
                shaders: vec![
                    (Stage::Vertex, producer.clone()),
                    (Stage::Fragment, capture.clone()),
                ],
                topology,
                discard: false,
                storage_active: true,
            });
        }
    }
    // An assembled triangle followed by one/two vertices has no second
    // primitive, with or without CUT. OUT remains a real SPIR-V emission.
    // Ryujinx InstEmitAttribute.cs:253-294; SPIR-V EmitVertex/EndPrimitive
    // shapes follow CodeGen/Spirv/Instructions.cs:517-528.
    for incomplete in 1..=2 {
        for cut in [false, true] {
            let mut xs = Vec::new();
            let mut body = String::new();
            for vertex in 0..3 + incomplete {
                let selected = vertex % 3;
                xs.extend([
                    mov(20, selected),
                    isberd(21, 20),
                    ald(0, 0x70, 4, 21, false, false),
                    ast(0, 0x70, 4, false),
                    ald(4, 0x80, 4, 21, false, false),
                    ast(4, 0x80, 4, false),
                    s2r(8, 0x11),
                    ald(9, 0x60, 1, 255, false, false),
                    mov(10, 0),
                    mov(11, 0),
                    ast(8, 0x90, 4, false),
                    ast(9, 0x60, 1, false),
                    out(1, true),
                ]);
                body.push_str(&format!("gl_Position=gl_in[{selected}].gl_Position;v=inputV[{selected}];ids=vec4(uintBitsToFloat(uint(gl_InvocationID)),uintBitsToFloat(uint(gl_PrimitiveIDIn)),0,0);gl_PrimitiveID=gl_PrimitiveIDIn;EmitVertex();"));
                if vertex == 2 || (vertex == 2 + incomplete && cut) {
                    xs.push(out(2, true));
                    body.push_str("EndPrimitive();");
                }
            }
            let reference = format!("#version 450\nlayout(triangles,invocations=3) in;layout(triangle_strip,max_vertices=6) out;layout(location=0) in vec4 inputV[];layout(location=0) flat out vec4 v;layout(location=1) flat out vec4 ids;void main(){{{body}}}");
            suite.check(Case {
                name: format!("gs-incomplete-{incomplete}-cut-{cut}"),
                stage: Stage::Geometry,
                xs,
                h: header(Stage::Geometry),
                state: GraphicsState {
                    geometry_input: Some(GeometryInput::Triangles),
                    ..Default::default()
                },
                reference,
                shaders: vec![
                    (Stage::Vertex, producer.clone()),
                    (Stage::Fragment, observer.clone()),
                ],
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                discard: false,
                storage_active: true,
            });
        }
    }
    generic_slot_matrix(&mut suite, Stage::Geometry);
    suite.finish(100);
}

#[test]
#[ignore = "requires RADV hardware, glslc and spirv-val"]
fn gpu_stage_geometry_current_handle_ast_exact() {
    const OUT_WORD: u64 = 0xfbe0_0080_0ff7_0000;
    const WORDS: &[u64] = geometry_current_ast_words::WORDS;
    const VALUE: u32 = 0.625_f32.to_bits();

    fn fields(raw: u64) -> (u8, u16) {
        (raw as u8, ((raw >> 20) & 0x3ff) as u16)
    }

    fn stored_value(data: u8, address: u16) -> u32 {
        if data == 255 {
            return 0;
        }
        match address {
            0x64 => 0,
            0x70 | 0x74 => 0.25_f32.to_bits(),
            0x78 => 0.5_f32.to_bits(),
            0x7c => 1.0_f32.to_bits(),
            _ => VALUE,
        }
    }

    let mut suite = Suite::new("geometry-current-handle-ast");
    let producer = suite.compile("geometry-current-producer", Stage::Vertex, PRODUCER);
    for &word in WORDS {
        let (data, address) = fields(word);
        let value = stored_value(data, address);
        let mut position = [
            (-0.25_f32).to_bits(),
            (-0.25_f32).to_bits(),
            0.25_f32.to_bits(),
            1.0_f32.to_bits(),
        ];
        if (0x70..=0x7c).contains(&address) {
            position[usize::from((address - 0x70) / 4)] = value;
        }

        let generic = address >= 0x80;
        let base = if generic { address & !0xf } else { 0x80 };
        let location = usize::from((base - 0x80) / 16);
        let component = if generic {
            usize::from((address - base) / 4)
        } else {
            0
        };
        let target_value = if generic { value } else { VALUE };
        let xs = vec![
            mov(20, (-0.25_f32).to_bits()),
            mov(21, (-0.25_f32).to_bits()),
            mov(22, 0.25_f32.to_bits()),
            mov(23, 1.0_f32.to_bits()),
            ast(20, 0x70, 4, false),
            mov(24, if generic { 0 } else { VALUE }),
            mov(25, 0),
            mov(26, 0),
            mov(27, 0),
            ast(24, base, 4, false),
            mov(data, value),
            word,
            OUT_WORD,
        ];
        let mut h = header(Stage::Geometry);
        h[15] = 1;
        let reference = format!(
            "#version 450\nlayout(triangles) in;layout(points,max_vertices=1) out;layout(location={location}) flat out vec4 target;void main(){{gl_Position=vec4(uintBitsToFloat({}u),uintBitsToFloat({}u),uintBitsToFloat({}u),uintBitsToFloat({}u));gl_Layer=0;target=vec4(0);target[{component}]=uintBitsToFloat({target_value}u);EmitVertex();}}",
            position[0], position[1], position[2], position[3]
        );
        let observer = suite.compile(
            &format!("geometry-current-observer-{word:016x}"),
            Stage::Fragment,
            &format!(
                "#version 450\nlayout(location={location}) flat in vec4 target;layout(location=0) out vec4 color;void main(){{color=vec4(target[{component}],0,0,1);}}"
            ),
        );
        suite.check(Case {
            name: format!("geometry-current-{word:016x}"),
            stage: Stage::Geometry,
            xs,
            h,
            state: GraphicsState {
                geometry_input: Some(GeometryInput::Triangles),
                ..Default::default()
            },
            reference,
            shaders: vec![
                (Stage::Vertex, producer.clone()),
                (Stage::Fragment, observer),
            ],
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            discard: false,
            storage_active: false,
        });
    }
    suite.finish(WORDS.len());
}

// Ryujinx AttributeMap.cs:68, InstEmitAttribute.cs:34-55,167-175,208-217:
// generic ALD advances the immediate per lane; IPA reconstructs the plane
// only for a perspective-linked input, then MUL applies the B register.
fn fragment_generic_matrix(suite: &mut Suite) {
    for location in 0..32u16 {
        let next = (location + 1).min(31);
        let declarations = if next == location {
            format!("layout(location={location}) flat out vec4 a;")
        } else {
            format!("layout(location={location}) flat out vec4 a;layout(location={next}) flat out vec4 b;")
        };
        let assignment = if next == location {
            String::new()
        } else {
            format!("b=vec4({next}.125,-{next}.25,-0.0,{}.0);", next + 1)
        };
        let producer = suite.compile("fragment-generic-producer",Stage::Vertex,&format!("#version 450\n{declarations}void main(){{vec2 p[3]=vec2[](vec2(-1,-1),vec2(3,-1),vec2(-1,3));gl_Position=vec4(p[gl_VertexIndex%3],0.25,1);a=vec4({location}.125,-{location}.25,-0.0,{}.0);{assignment}}}",location+1));
        for count in 1..=4u8 {
            let component = (location % 4) as u8;
            if location == 31 && component + count > 4 {
                continue;
            }
            let mut h = header(Stage::Fragment);
            // Mesa sph.rs:138-143: two-bit PS component map at bit192.
            // Ryujinx ShaderHeader.cs:7-12 defines Constant as 1.
            h[24..56].fill(0);
            for slot in location..=next {
                h[24 + usize::from(slot)] = 0x55;
            }
            let mut xs = (0..4)
                .map(|r| mov(r, (-1f32).to_bits()))
                .collect::<Vec<_>>();
            xs.push(ald(
                0,
                0x80 + location * 16 + u16::from(component) * 4,
                count,
                255,
                false,
                false,
            ));
            let lanes = (0..count)
                .map(|lane| {
                    let index = component + lane;
                    format!("{}[{}]", if index < 4 { "a" } else { "b" }, index % 4)
                })
                .collect::<Vec<_>>();
            let mut body = String::from("color=vec4(-1);");
            for (lane, expression) in lanes.iter().enumerate() {
                body.push_str(&format!("color[{lane}]={expression};"));
            }
            let reference = format!(
                "#version 450\n{}layout(location=0) out vec4 color;void main(){{{body}}}",
                declarations.replace("out vec4", "in vec4")
            );
            let expected = suite.check(Case {
                name: format!(
                    "fs-generic-ald-location-{location}-width-{count}-component-{component}"
                ),
                stage: Stage::Fragment,
                xs,
                h,
                state: GraphicsState::default(),
                reference,
                shaders: vec![(Stage::Vertex, producer.clone())],
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                discard: false,
                storage_active: false,
            });
            let mut values = [-1f32; 4];
            for lane in 0..count {
                let index = component + lane;
                let slot = location + u16::from(index / 4);
                values[usize::from(lane)] = [
                    f32::from(slot) + 0.125,
                    -f32::from(slot) - 0.25,
                    -0.0,
                    f32::from(slot) + 1.,
                ][usize::from(index % 4)];
            }
            assert_eq!(
                expected.color,
                values.map(f32::to_bits).repeat(64),
                "analytic generic ALD reference"
            );
        }
        for (mode, qualifier) in [(1u32, "flat"), (2, ""), (3, "noperspective")] {
            let producer = suite.compile("fragment-generic-ipa-producer",Stage::Vertex,&format!("#version 450\nlayout(location={location}) {qualifier} out vec4 a;void main(){{vec2 p[3]=vec2[](vec2(-1,-1),vec2(3,-1),vec2(-1,3));gl_Position=vec4(p[gl_VertexIndex%3]*2,0.5,2);a=vec4({location}.125,-{location}.25,-0.0,{}.0);}}",location+1));
            let mut h = header(Stage::Fragment);
            h[24..56].fill(0);
            h[24 + usize::from(location)] = (mode * 0x55) as u8;
            let mut xs = vec![mov(20, 2f32.to_bits())];
            for component in 0..4u8 {
                let address = 0x80 + location * 16 + u16::from(component) * 4;
                xs.push(if mode == 1 {
                    ipa(component, address, 2)
                } else {
                    ipa_mul(component, address, 20)
                });
            }
            let expression = match mode {
                1 => "a",
                2 => "(a*gl_FragCoord.w)*2",
                _ => "a*2",
            };
            let reference=format!("#version 450\nlayout(location={location}) {qualifier} in vec4 a;layout(location=0) out vec4 color;void main(){{color={expression};}}");
            let expected = suite.check(Case {
                name: format!("fs-generic-ipa-location-{location}-mode-{mode}"),
                stage: Stage::Fragment,
                xs,
                h,
                state: GraphicsState::default(),
                reference,
                shaders: vec![(Stage::Vertex, producer)],
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                discard: false,
                storage_active: false,
            });
            let factor = if mode == 3 { 2. } else { 1. };
            let values = [
                f32::from(location) + 0.125,
                -f32::from(location) - 0.25,
                -0.0,
                f32::from(location) + 1.,
            ];
            assert_eq!(
                expected.color,
                values
                    .map(|v| {
                        // This RADV's constant perspective plane adds a
                        // positive-zero gradient, changing -0 to +0.
                        // Flat and screen-linear inputs preserve -0 here.
                        // Mesa aco_isel_helpers.cpp:405-430 emits two FP32
                        // interpolation steps. fs-generic-ipa-location-0-mode-2
                        // independently observed +0 in GLSL and both translations.
                        // Model the plane addition in the analytic oracle only;
                        // every actual color bit still compares unchanged.
                        let linked = if mode == 2 { v + 0.0 } else { v };
                        (linked * factor).to_bits()
                    })
                    .repeat(64),
                "analytic generic IPA reference"
            );
        }
    }
}

#[test]
#[ignore = "requires RADV hardware, glslc and spirv-val"]
fn gpu_stage_texcoord0_pass_lanes_exact() {
    const WORDS: [(u64, usize, u8); 2] =
        [(0xe003_ffb0_cff7_ff04, 3, 4), (0xe003_ffb0_4ff7_ff05, 1, 5)];
    const SENTINEL: f32 = -7.0;
    const INPUTS: [f32; 4] = [0.25, -0.5, 0.75, 1.0];

    let mut suite = Suite::new("texcoord0-pass-lanes");
    let producer = suite.compile(
        "texcoord0-pass-producer",
        Stage::Vertex,
        "#version 450\nlayout(location=5) out vec4 v;void main(){vec2 p[3]=vec2[](vec2(-1,-1),vec2(3,-1),vec2(-1,3));gl_Position=vec4(p[gl_VertexIndex%3]*2,0.5,2);v=vec4(0.25,-0.5,0.75,1.0);}",
    );
    for (word, component, dst) in WORDS {
        let mut xs = vec![mov(10, SENTINEL.to_bits()), word, mov_reg(10, dst)];
        xs.extend([
            mov(0, 0),
            mov(1, 0),
            mov(2, 0),
            mov(3, 1.0f32.to_bits()),
            mov_reg(0, 10),
        ]);
        let mut fragment_header = header(Stage::Fragment);
        fragment_header[24..56].fill(0);
        let reference = format!(
            "#version 450\nlayout(location=5) in vec4 v;layout(location=0) out vec4 color;void main(){{color=vec4(v[{component}]*gl_FragCoord.w,0,0,1);}}"
        );
        let expected = suite.check(Case {
            name: format!("texcoord0-pass-{word:016x}"),
            stage: Stage::Fragment,
            xs,
            h: fragment_header,
            state: GraphicsState::default(),
            reference,
            shaders: vec![(Stage::Vertex, producer.clone())],
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            discard: false,
            storage_active: false,
        });
        assert_eq!(expected.color[0], (INPUTS[component] * 0.5).to_bits());
    }
    suite.finish(2);
}

#[test]
#[ignore = "requires RADV hardware, glslc and spirv-val"]
fn gpu_stage_texcoord0_replay_lanes_exact() {
    const WORDS: [(u64, usize, u8); 4] = [
        (0xe003_ffb0_0ff7_ff04, 0, 4),
        (0xe003_ffb0_8ff7_ff02, 2, 2),
        (0xe003_ffb0_0ff7_ff00, 0, 0),
        (0xe003_ffb0_4ff7_ff01, 1, 1),
    ];
    const SENTINEL: f32 = -7.0;
    const INPUTS: [f32; 4] = [0.25, -0.5, 0.75, 1.0];

    let mut suite = Suite::new("texcoord0-replay-lanes");
    let producer = suite.compile(
        "texcoord0-replay-producer",
        Stage::Vertex,
        "#version 450\nlayout(location=5) out vec4 v;void main(){vec2 p[3]=vec2[](vec2(-1,-1),vec2(3,-1),vec2(-1,3));gl_Position=vec4(p[gl_VertexIndex%3]*2,0.5,2);v=vec4(0.25,-0.5,0.75,1.0);}",
    );
    for (word, component, dst) in WORDS {
        let mut xs = vec![mov(10, SENTINEL.to_bits()), word, mov_reg(10, dst)];
        xs.extend([
            mov(0, 0),
            mov(1, 0),
            mov(2, 0),
            mov(3, 1.0f32.to_bits()),
            mov_reg(0, 10),
        ]);
        let mut fragment_header = header(Stage::Fragment);
        fragment_header[24..56].fill(0);
        let reference = format!(
            "#version 450\nlayout(location=5) in vec4 v;layout(location=0) out vec4 color;void main(){{color=vec4(v[{component}]*gl_FragCoord.w,0,0,1);}}"
        );
        let expected = suite.check(Case {
            name: format!("texcoord0-replay-{word:016x}"),
            stage: Stage::Fragment,
            xs,
            h: fragment_header,
            state: GraphicsState::default(),
            reference,
            shaders: vec![(Stage::Vertex, producer.clone())],
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            discard: false,
            storage_active: false,
        });
        assert_eq!(expected.color[0], (INPUTS[component] * 0.5).to_bits());
    }
    suite.finish(4);
}

#[test]
#[ignore = "requires RADV hardware, glslc and spirv-val"]
fn gpu_stage_front_color_sc_lane_exact() {
    const SENTINEL: f32 = -7.0;

    let mut suite = Suite::new("front-color-sc-lane");
    let smooth = suite.compile(
        "front-color-sc-smooth-producer",
        Stage::Vertex,
        "#version 450\nlayout(location=1) out vec4 v;void main(){vec2 p[3]=vec2[](vec2(-1,-1),vec2(3,-1),vec2(-1,3));gl_Position=vec4(p[gl_VertexIndex%3]*2,0.5,2);v=vec4(0.25,-0.5,0.75,1.0);}",
    );
    let flat = suite.compile(
        "front-color-sc-flat-producer",
        Stage::Vertex,
        "#version 450\nlayout(location=1) flat out vec4 v;void main(){vec2 p[3]=vec2[](vec2(-1,-1),vec2(3,-1),vec2(-1,3));gl_Position=vec4(p[gl_VertexIndex%3]*2,0.5,2);v=vec4(0.25,-0.5,0.75,1.0);}",
    );
    for (word, dst, lane, input) in [
        (0xe0c3_ffa8_0047_ff00, 0, "x", 0.25_f32),
        (0xe0c3_ffa8_4047_ff01, 1, "y", -0.5_f32),
        (0xe0c3_ffa8_8047_ff02, 2, "z", 0.75_f32),
        (0xe0c3_ffa8_c047_ff03, 3, "w", 1.0_f32),
    ] {
        for shading_flat in [false, true] {
            let mut xs = vec![
                mov(4, 2.0_f32.to_bits()),
                mov(10, SENTINEL.to_bits()),
                word,
                mov_reg(10, dst),
            ];
            xs.extend([
                mov(0, 0),
                mov(1, 0),
                mov(2, 0),
                mov(3, 1.0f32.to_bits()),
                mov_reg(0, 10),
            ]);
            let mut fragment_header = header(Stage::Fragment);
            fragment_header[24..56].fill(0);
            let qualifier = if shading_flat { "flat" } else { "" };
            let expression = if shading_flat {
                format!("v.{lane}")
            } else {
                format!("(v.{lane}*gl_FragCoord.w)*2.0")
            };
            let reference = format!(
                "#version 450\nlayout(location=1) {qualifier} in vec4 v;layout(location=0) out vec4 color;void main(){{color=vec4({expression},0,0,1);}}"
            );
            let expected = suite.check(Case {
                name: format!("front-color-sc-{lane}-flat-{shading_flat}"),
                stage: Stage::Fragment,
                xs,
                h: fragment_header,
                state: GraphicsState {
                    legacy_color_flat_shading: Some(shading_flat),
                    ..Default::default()
                },
                reference,
                shaders: vec![(
                    Stage::Vertex,
                    if shading_flat {
                        flat.clone()
                    } else {
                        smooth.clone()
                    },
                )],
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                discard: false,
                storage_active: false,
            });
            assert_eq!(expected.color[0], input.to_bits());
        }
    }
    suite.finish(8);
}

#[test]
#[ignore = "requires RADV hardware, glslc and spirv-val"]
fn gpu_stage_legacy_varying_ast_exact() {
    const VALUE: f32 = -0.625;
    const WORDS: [u64; 49] = [
        0xeff0_7f80_2807_ff2a,
        0xeff0_7f80_2847_ff2b,
        0xeff0_7f80_2887_ff2c,
        0xeff0_7f80_28c7_ff29,
        0xeff0_7f80_2907_ff26,
        0xeff0_7f80_2947_ff25,
        0xeff0_7f80_2987_ff24,
        0xeff0_7f80_29c7_ff23,
        0xeff0_7f80_2a07_ff2a,
        0xeff0_7f80_2a47_ff2b,
        0xeff0_7f80_2a87_ff2c,
        0xeff0_7f80_2ac7_ff29,
        0xeff0_7f80_2b07_ff26,
        0xeff0_7f80_2b47_ff25,
        0xeff0_7f80_2b87_ff24,
        0xeff0_7f80_2bc7_ff23,
        0xeff0_7f80_2e87_ff06,
        0xeff0_7f80_3007_ff22,
        0xeff0_7f80_3047_ff21,
        0xeff0_7f80_3087_ff20,
        0xeff0_7f80_30c7_ff1f,
        0xeff0_7f80_3107_ff1e,
        0xeff0_7f80_3147_ff1d,
        0xeff0_7f80_3187_ff19,
        0xeff0_7f80_31c7_ff1c,
        0xeff0_7f80_3207_ff1b,
        0xeff0_7f80_3247_ff1a,
        0xeff0_7f80_3287_ff15,
        0xeff0_7f80_32c7_ff18,
        0xeff0_7f80_3307_ff17,
        0xeff0_7f80_3347_ff16,
        0xeff0_7f80_3387_ff11,
        0xeff0_7f80_33c7_ff14,
        0xeff0_7f80_3407_ff13,
        0xeff0_7f80_3447_ff12,
        0xeff0_7f80_3487_ff0d,
        0xeff0_7f80_34c7_ff10,
        0xeff0_7f80_3507_ff0f,
        0xeff0_7f80_3547_ff0e,
        0xeff0_7f80_3587_ff11,
        0xeff0_7f80_35c7_ff0c,
        0xeff0_7f80_3607_ff0b,
        0xeff0_7f80_3647_ff0a,
        0xeff0_7f80_3687_ff08,
        0xeff0_7f80_36c7_ff19,
        0xeff0_7f80_3707_ff03,
        0xeff0_7f80_3747_ff02,
        0xeff0_7f80_3787_ff01,
        0xeff0_7f80_37c7_ff00,
    ];

    let mut suite = Suite::new("legacy-color-ast-lanes");
    for word in WORDS {
        let instruction = decode_word([word as u32, (word >> 32) as u32], 0);
        let InstructionKind::LegacyVaryingStore(ops::LegacyVaryingStore { data, varying }) =
            instruction.kind
        else {
            panic!("{word:016x} did not enter the typed legacy store path")
        };
        let location = varying.location(0).unwrap();
        let lane = ["x", "y", "z", "w"][usize::from(varying.component())];
        let observer = suite.compile(
            &format!("legacy-ast-{word:016x}-observer"),
            Stage::Fragment,
            &format!("#version 450\nlayout(location={location}) in vec4 v;layout(location=0) out vec4 color;void main(){{color=vec4(v.{lane},0,0,1);}}"),
        );
        let xs = vec![
            ald(0, 0x80, 4, 255, false, false),
            ast(0, 0x70, 4, false),
            mov(data.0, VALUE.to_bits()),
            word,
        ];
        let mut vertex_header = header(Stage::Vertex);
        vertex_header[54..70].fill(0);
        let expected = suite.check(Case {
            name: format!("legacy-varying-ast-{word:016x}"),
            stage: Stage::Vertex,
            xs,
            h: vertex_header,
            state: GraphicsState::default(),
            reference: format!(
                "#version 450\nlayout(location=0) in vec4 p;layout(location={location}) out vec4 v;void main(){{gl_Position=p;v.{lane}={VALUE:?};}}"
            ),
            shaders: vec![(Stage::Fragment, observer)],
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            discard: false,
            storage_active: false,
        });
        assert_eq!(expected.color[0], VALUE.to_bits());
    }
    suite.finish(49);
}

#[test]
#[ignore = "requires RADV hardware, glslc and spirv-val"]
fn gpu_stage_point_size_ast_exact() {
    const WORD: u64 = 0xeff0_7f80_06c7_ff05;
    const POSITIONS: [[f32; 8]; 6] = [
        [-0.75, -0.75, 0.25, 1.0, 0.0, 0.0, 0.0, 0.0],
        [-0.25, -0.25, 0.25, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.75, 0.75, 0.25, 1.0, 0.0, 0.0, 0.0, 0.0],
        [-0.75, 0.75, 0.25, 1.0, 0.0, 0.0, 0.0, 0.0],
        [0.75, -0.75, 0.25, 1.0, 0.0, 0.0, 0.0, 0.0],
    ];
    let mut suite = Suite::new("point-size-ast");
    let fragment = suite.compile(
        "point-size-observer",
        Stage::Fragment,
        "#version 450\nlayout(location=0) out vec4 color;void main(){color=vec4(1);}",
    );
    for size in [2.0_f32, 4.0, 8.0] {
        let expected = suite.check_vertices(
            Case {
                name: format!("point-size-{size}"),
                stage: Stage::Vertex,
                xs: vec![
                    ald(0, 0x80, 4, 255, false, false),
                    ast(0, 0x70, 4, false),
                    mov(5, size.to_bits()),
                    WORD,
                ],
                h: header(Stage::Vertex),
                state: GraphicsState::default(),
                reference: format!(
                    "#version 450\nlayout(location=0) in vec4 p;void main(){{gl_Position=p;gl_PointSize={size:?};}}"
                ),
                shaders: vec![(Stage::Fragment, fragment.clone())],
                topology: vk::PrimitiveTopology::POINT_LIST,
                discard: false,
                storage_active: false,
            },
            &POSITIONS,
        );
        assert!(
            expected.color.iter().any(|&pixel| pixel != 0),
            "point size {size} produced a vacuous reference image"
        );
    }
    suite.finish(3);
}

#[test]
#[ignore = "requires RADV hardware, glslc and spirv-val"]
fn gpu_stage_fragment_exact() {
    let mut suite = Suite::new("fragment");
    // Ryujinx InstEmitAttribute.cs:167-175,208-217: reconstruct the
    // perspective plane for PASS, then apply the actual B operand for MUL.
    // Constant W=2 and binary fractions keep both GLSL operations exact.
    for (mode, qualifier) in [(0u32, ""), (1, "flat"), (2, ""), (3, "noperspective")] {
        let source = format!("#version 450\nlayout(location=0) {qualifier} out vec4 v;void main(){{vec2 p[3]=vec2[](vec2(-1,-1),vec2(3,-1),vec2(-1,3));gl_Position=vec4(p[gl_VertexIndex%3]*2,0.5,2);v=vec4(0.125,0.25,0.5,1);}}");
        let producer = suite.compile(&format!("w2-{mode}"), Stage::Vertex, &source);
        for (multiplier, saturate) in [
            (None, false),
            (Some(0f32), false),
            (Some(0.5), false),
            (Some(2.), false),
            (Some(-2.), false),
            (Some(0.), true),
            (Some(0.5), true),
            (Some(2.), true),
            (Some(-2.), true),
        ] {
            let mut h = header(Stage::Fragment);
            // Mesa nak/sph.rs:138-143: two-bit interpolation map at192.
            // Ryujinx Translation/ShaderHeader.cs:7-12,30-48: flat1, perspective2,
            // screen-linear3, with the first used component selecting the mode.
            h[24..28].copy_from_slice(&(mode * 0x55).to_le_bytes());
            let mut xs = Vec::new();
            if let Some(value) = multiplier {
                xs.push(mov(20, value.to_bits()));
            }
            for c in 0..4 {
                xs.push(if multiplier.is_some() {
                    // Mesa nak/sm50.rs:2916: saturation bit51.
                    ipa_mul(c, 0x80 + u16::from(c) * 4, 20) | (u64::from(saturate) << 51)
                } else {
                    ipa(c, 0x80 + u16::from(c) * 4, if mode == 1 { 2 } else { 0 })
                });
            }
            let plane = if mode == 2 { "v*gl_FragCoord.w" } else { "v" };
            let expression =
                multiplier.map_or_else(|| plane.into(), |value| format!("({plane})*{value:?}"));
            let expression = if saturate {
                format!("clamp({expression},0.0,1.0)")
            } else {
                expression
            };
            let reference = format!("#version 450\nlayout(location=0) {qualifier} in vec4 v;layout(location=0) out vec4 color;void main(){{color={expression};}}");
            let expected = suite.check(Case {
                name: format!("fs-ipa-mode-{mode}-mul-{multiplier:?}-sat-{saturate}"),
                stage: Stage::Fragment,
                xs,
                h,
                state: GraphicsState::default(),
                reference,
                shaders: vec![(Stage::Vertex, producer.clone())],
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                discard: false,
                storage_active: false,
            });
            let factor = if mode == 2 { 0.5 } else { 1. };
            let color = [0.125f32, 0.25, 0.5, 1.].map(|v| {
                let result = v * factor * multiplier.unwrap_or(1.);
                if saturate {
                    result.clamp(0., 1.).to_bits()
                } else {
                    result.to_bits()
                }
            });
            assert_eq!(expected.color, color.repeat(64), "analytic IPA reference");
        }
    }
    // Full-screen triangle, constant attributes avoid implementation-dependent
    // interpolation rounding. Fragment coordinates exercise nonconstant IPA.
    let producer=suite.compile("producer",Stage::Vertex,"#version 450\nlayout(location=0) out vec4 v;void main(){vec2 p[3]=vec2[](vec2(-1,-1),vec2(3,-1),vec2(-1,3));gl_Position=vec4(p[gl_VertexIndex%3],0.25,1);v=vec4(0.125,0.25,0.5,1);}");
    for mask in [0u32, 1, u32::MAX] {
        for depth in [0.0f32, 0.375, 1.] {
            let mut h = header(Stage::Fragment);
            // Mesa cla097sph.h:628-633: sample-mask bit608, depth bit609.
            h[76] = 3;
            let xs = vec![
                ipa(0, 0x80, 0),
                ipa(1, 0x84, 0),
                ipa(2, 0x88, 0),
                ipa(3, 0x8c, 0),
                mov(4, mask),
                mov(5, depth.to_bits()),
            ];
            let reference=format!("#version 450\nlayout(location=0) in vec4 v;layout(location=0) out vec4 color;void main(){{color=v;gl_SampleMask[0]=int({mask}u);gl_FragDepth={depth:?};}}");
            let expected = suite.check(Case {
                name: format!("fs-ipa-mask-{mask}-depth-{:08x}", depth.to_bits()),
                stage: Stage::Fragment,
                xs,
                h,
                state: GraphicsState::default(),
                reference,
                shaders: vec![(Stage::Vertex, producer.clone())],
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                discard: false,
                storage_active: false,
            });
            let color = if mask & 1 == 0 {
                [-16f32, -32., -64., -128.]
            } else {
                [0.125, 0.25, 0.5, 1.]
            };
            assert_eq!(
                expected.color,
                color.map(f32::to_bits).repeat(64),
                "independent reference coverage"
            );
            assert_eq!(
                expected.depth,
                vec![
                    if mask & 1 == 0 {
                        1f32.to_bits()
                    } else {
                        depth.to_bits()
                    };
                    64
                ],
                "independent reference depth"
            );
        }
    }
    // Ryujinx InstEmitAttribute.cs:176-193,208-217: ordinary Position
    // components use FragCoord directly and MUL still applies its B operand.
    for multiply in [false, true] {
        let mut xs = vec![mov(20, 2f32.to_bits())];
        for c in 0..4 {
            xs.push(if multiply {
                ipa_mul(c, 0x70 + u16::from(c) * 4, 20)
            } else {
                ipa(c, 0x70 + u16::from(c) * 4, 0)
            });
        }
        let expression = if multiply {
            "gl_FragCoord*2"
        } else {
            "gl_FragCoord"
        };
        suite.check(Case {
            name: format!("fs-ipa-fragcoord-multiply-{multiply}"), stage: Stage::Fragment,
            xs, h: header(Stage::Fragment), state: GraphicsState::default(),
            reference: format!("#version 450\nlayout(location=0) out vec4 color;void main(){{color={expression};}}"),
            shaders: vec![(Stage::Vertex, producer.clone())],
            topology: vk::PrimitiveTopology::TRIANGLE_LIST, discard: false, storage_active: false,
        });
    }
    // ALD FragCoord reads and ordinary register-packed color export.
    let expected = suite.check(Case {
        name: "fs-ald-fragcoord".into(),
        stage: Stage::Fragment,
        xs: vec![ald(0, 0x70, 4, 255, false, false)],
        h: header(Stage::Fragment),
        state: GraphicsState::default(),
        reference:
            "#version 450\nlayout(location=0) out vec4 color;void main(){color=gl_FragCoord;}"
                .into(),
        shaders: vec![(Stage::Vertex, producer)],
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        discard: false,
        storage_active: false,
    });
    let color = (0..8)
        .flat_map(|y| {
            (0..8).flat_map(move |x| [x as f32 + 0.5, y as f32 + 0.5, 0.25, 1.].map(f32::to_bits))
        })
        .collect::<Vec<_>>();
    assert_eq!(expected.color, color, "independent FragCoord reference");
    fragment_generic_matrix(&mut suite);
    suite.finish(269);
}

#[test]
#[ignore = "requires RADV hardware, glslc and spirv-val"]
fn gpu_stage_fragment_ipa_saturate_exact() {
    // Six IPA.SAT words: envytools gm107.c:1953; Ryujinx
    // InstEmitAttribute.cs:146-218. The GLSL references and the raw-word CPU
    // oracle in support/stage_reference.rs are both independent of the decoder.
    let mut suite = Suite::new("fragment-ipa-saturate");
    let mut registers = [0u32; 256];
    registers[20] = 0.25f32.to_bits();
    registers[21] = (-2f32).to_bits();
    // W=2 producer: FragCoord = (x+0.5, y+0.5, 0.25, 0.5) on the 8x8 target.
    let frag_coord = |pixel: usize| {
        [
            (pixel % 8) as f32 + 0.5,
            (pixel / 8) as f32 + 0.5,
            0.25,
            0.5,
        ]
    };
    let generic = [3f32, 0.25, 0.5, -4.];
    let fill = [0.25f32, 0.5, 0.75, 0.125];
    for (name, mode, qualifier, word, lane) in [
        (
            "pass-perspective",
            2u32,
            "",
            0xe00b_ff88_0ff7_ff00u64,
            0usize,
        ),
        ("constant-flat", 1, "flat", 0xe08b_ff88_cff7_ff03, 3),
    ] {
        let producer = suite.compile(&format!("ipa-sat-{name}-producer"), Stage::Vertex, &format!("#version 450\nlayout(location=0) {qualifier} out vec4 v;void main(){{vec2 p[3]=vec2[](vec2(-1,-1),vec2(3,-1),vec2(-1,3));gl_Position=vec4(p[gl_VertexIndex%3]*2,0.5,2);v=vec4(3.0,0.25,0.5,-4.0);}}"));
        let mut h = header(Stage::Fragment);
        // Mesa nak/sph.rs:138-143; Ryujinx ShaderHeader.cs:7-12,30-48.
        h[24..28].copy_from_slice(&(mode * 0x55).to_le_bytes());
        let mut xs: Vec<u64> = (0..4u8)
            .filter(|&c| usize::from(c) != lane)
            .map(|c| mov(c, fill[usize::from(c)].to_bits()))
            .collect();
        xs.push(word);
        let plane = if mode == 2 {
            "v.x*gl_FragCoord.w"
        } else {
            "v.w"
        };
        let mut components = fill.map(|v| format!("{v:?}"));
        components[lane] = format!("clamp({plane},0.0,1.0)");
        let expected = suite.check(Case {
            name: format!("fs-ipa-sat-{name}"),
            stage: Stage::Fragment,
            xs,
            h,
            state: GraphicsState::default(),
            reference: format!("#version 450\nlayout(location=0) {qualifier} in vec4 v;layout(location=0) out vec4 color;void main(){{color=vec4({});}}", components.join(",")),
            shaders: vec![(Stage::Vertex, producer)],
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            discard: false,
            storage_active: false,
        });
        let oracle: Vec<u32> = (0..64)
            .flat_map(|pixel| {
                let mut color = fill.map(f32::to_bits);
                color[lane] = stage_reference::ipa_saturate_fragment(
                    word,
                    generic,
                    mode == 2,
                    frag_coord(pixel),
                    &registers,
                );
                color
            })
            .collect();
        assert_eq!(expected.color, oracle, "IPA.SAT CPU oracle {name}");
    }
    // Position: PASS.SAT x and w, MUL.SAT y by R20 and z by R21.
    let words = [
        0xe00b_ff87_0ff7_ff00u64,
        0xe04b_ff87_4147_ff01,
        0xe04b_ff87_8157_ff02,
        0xe00b_ff87_cff7_ff03,
    ];
    let producer = suite.compile("ipa-sat-position-producer", Stage::Vertex, "#version 450\nvoid main(){vec2 p[3]=vec2[](vec2(-1,-1),vec2(3,-1),vec2(-1,3));gl_Position=vec4(p[gl_VertexIndex%3]*2,0.5,2);}");
    let mut xs = vec![mov(20, registers[20]), mov(21, registers[21])];
    xs.extend(words);
    let expected = suite.check(Case {
        name: "fs-ipa-sat-position".into(),
        stage: Stage::Fragment,
        xs,
        h: header(Stage::Fragment),
        state: GraphicsState::default(),
        reference: "#version 450\nlayout(location=0) out vec4 color;void main(){color=clamp(vec4(gl_FragCoord.x,gl_FragCoord.y*0.25,gl_FragCoord.z*-2.0,gl_FragCoord.w),0.0,1.0);}".into(),
        shaders: vec![(Stage::Vertex, producer)],
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        discard: false,
        storage_active: false,
    });
    let oracle: Vec<u32> = (0..64)
        .flat_map(|pixel| {
            words.map(|word| {
                stage_reference::ipa_saturate_fragment(
                    word,
                    generic,
                    false,
                    frag_coord(pixel),
                    &registers,
                )
            })
        })
        .collect();
    assert_eq!(expected.color, oracle, "IPA.SAT CPU oracle position");
    suite.finish(3);
}

#[test]
fn stage_decoder_has_only_the_two_cited_noop_arms() {
    assert_eq!(
        include_str!("../src/decoder/families.rs")
            .matches("x.guarded(InstructionKind::Nop(ops::Nop))")
            .count(),
        2
    );
}
