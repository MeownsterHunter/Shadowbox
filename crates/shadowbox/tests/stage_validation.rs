//! Project-owned synthetic Maxwell programs. Encoding citations live beside
//! the constructors; no vendor binaries or captures are used.
#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::spirv::Op;
use shadowbox::{
    decoder::decode_word,
    headers::{parse, Stage},
    interface::{
        GeometryInput, GeometryStreamLocations, GeometryStreams, GraphicsState, PatchIo,
        PatchLayout, ShaderInput, TessellationDomain, TessellationSpacing, TessellationState,
    },
    ir::InstructionKind,
    translate, translate_with_graphics_state,
};

fn header(stage: u32) -> Vec<u8> {
    // Mesa cla097sph.h:37-42,66-73; sph.rs:134,169 (generic maps).
    let mut words = [0u32; 32];
    words[0] = stage << 10;
    words[2] = 3 << 24;
    words[3] = 7 << 24;
    words[4] = 3;
    words[6] = 15;
    words[13] = 15 << 16;
    words.into_iter().flat_map(u32::to_le_bytes).collect()
}

fn ald(dst: u8, address: u16, count: u8, vertex: u8, patch: bool, output: bool) -> u64 {
    // Mesa sm50.rs:2861-2876: dst, offset RZ, vertex, address, P/O, count.
    (0xefd8u64 << 48)
        | u64::from(dst)
        | (255 << 8)
        | (7 << 16)
        | (u64::from(address) << 20)
        | (u64::from(patch) << 31)
        | (u64::from(output) << 32)
        | (u64::from(vertex) << 39)
        | (u64::from(count - 1) << 47)
}
fn ast(data: u8, address: u16, count: u8, patch: bool) -> u64 {
    // Mesa sm50.rs:2886-2896: data, offset/vertex RZ, address, P/O, count.
    (0xeff0u64 << 48)
        | u64::from(data)
        | (255 << 8)
        | (7 << 16)
        | (u64::from(address) << 20)
        | (u64::from(patch) << 31)
        | (1 << 32)
        | (255 << 39)
        | (u64::from(count - 1) << 47)
}
fn s2r(dst: u8, index: u8) -> u64 {
    // Mesa sm50.rs:3215-3224: dst 0..8, system register 20..28.
    (0xf0c8u64 << 48) | u64::from(dst) | (7 << 16) | (u64::from(index) << 20)
}
fn out(mode: u64) -> u64 {
    // Mesa sm50.rs:3260-3261,3274-3284: stream RZ, unused handle/dst, type.
    (0xfbe0u64 << 48) | 255 | (255 << 8) | (7 << 16) | (255 << 20) | (mode << 39)
}
fn out_stream(mode: u64, stream: u8) -> u64 {
    // Mesa sm50.rs:3263-3265,3274-3284: immediate stream and type.
    (0xf6e0u64 << 48) | 255 | (255 << 8) | (7 << 16) | (u64::from(stream) << 20) | (mode << 39)
}
fn out_register(mode: u64, stream: u8) -> u64 {
    // Mesa sm50.rs:3260-3261,3274-3284: stream GPR, unused handle/dst.
    (out(mode) & !(255 << 20)) | (u64::from(stream) << 20)
}
fn out_cbuf(mode: u64, binding: u8, offset: u16) -> u64 {
    // Mesa sm50.rs:3267-3269,361-372: offset in words, binding34..39.
    (out(mode) & !(0xffff << 48) & !(255 << 20))
        | (0xebe0 << 48)
        | (u64::from(binding) << 34)
        | (u64::from(offset / 4) << 20)
}
fn mov(dst: u8, value: u32) -> u64 {
    // Mesa sm50.rs:1938-1941: MOV32I with full quad lane mask.
    (0x0100u64 << 48) | (u64::from(value) << 20) | (15 << 12) | (7 << 16) | u64::from(dst)
}
fn stream_state(native: u8) -> GraphicsState<'static> {
    GraphicsState {
        geometry_input: Some(GeometryInput::Points),
        geometry_streams: Some(GeometryStreams {
            builtin_stream: native,
            locations: std::array::from_fn(|s| {
                Some(GeometryStreamLocations {
                    // Explicitly disjoint locations, including builtin aliases.
                    // This is test-owned driver linkage, not a Maxwell field.
                    generic_base: s as u32 * 40,
                    position: (s != usize::from(native)).then_some(s as u32 * 40 + 32),
                    primitive_id: (s != usize::from(native)).then_some(s as u32 * 40 + 33),
                })
            }),
        }),
        ..Default::default()
    }
}
fn code(instructions: &[u64]) -> Vec<u8> {
    // Mesa sm50.rs:3418-3428,3438-3468: control word + three instructions.
    // OpExit sm50.rs:3099-3111; OpNop sm50.rs:3178-3184.
    let mut words = instructions.to_vec();
    words.push((0xe300u64 << 48) | (7 << 16) | 0xf);
    while !words.len().is_multiple_of(3) {
        words.push((0x50b0u64 << 48) | (7 << 16));
    }
    words
        .chunks(3)
        .flat_map(|c| [0, c[0], c[1], c[2]])
        .flat_map(u64::to_le_bytes)
        .collect()
}
fn validate(name: &str, instructions: &[u64], h: &[u8], state: GraphicsState) -> Vec<u32> {
    let bytes = code(instructions);
    let input = ShaderInput {
        code: &bytes,
        program_record: h,
    };
    let words = Result::from(translate_with_graphics_state(
        shadowbox::QuietRun,
        &input,
        &state,
    ))
    .unwrap()
    .spirv;
    let result = spirv_validation::validate(&words);
    assert!(
        result.status.success(),
        "{name}: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    words
}

fn al2p(dst: u8, offset: u8, address: u16) -> u64 {
    // Mesa sm50.rs:2842-2851: scalar input AL2P, predicate result PT.
    (0xefa0u64 << 48)
        | u64::from(dst)
        | (u64::from(offset) << 8)
        | (7 << 16)
        | (u64::from(address) << 20)
        | (7 << 44)
}

fn isberd(dst: u8, offset: u8) -> u64 {
    // Mesa nv50_ir_emit_gm107.cpp:2549-2553: default input ISBERD.
    (0xefd0u64 << 48) | u64::from(dst) | (u64::from(offset) << 8) | (7 << 16)
}

fn physical(raw: u64, offset: u8) -> u64 {
    // Mesa sm50.rs:2864-2870; Ryujinx InstDecoders.cs:887,902: zero
    // immediate, no P, and a register source distinguish physical addressing.
    (raw & !(255 << 8)) | (u64::from(offset) << 8)
}

#[test]
fn physical_attributes_use_byte_addresses_and_share_static_banks() {
    for (stage, state) in [
        (
            4,
            GraphicsState {
                geometry_input: Some(GeometryInput::Triangles),
                ..Default::default()
            },
        ),
        (2, GraphicsState::default()),
        (
            3,
            GraphicsState {
                tessellation: Some(TessellationState {
                    domain: TessellationDomain::Triangles,
                    spacing: TessellationSpacing::Equal,
                    clockwise: false,
                }),
                ..Default::default()
            },
        ),
    ] {
        let mut xs = vec![
            s2r(3, 0), // runtime source, preventing constant folding
            al2p(4, 3, 0x80),
            isberd(5, 255),
            physical(ald(0, 0, 1, 5, false, false), 4),
            physical(ast(0, 0, 1, false), 4),
            physical(ald(1, 0, 1, 255, false, true), 4),
            ald(2, 0x94, 1, 5, false, false),
            ast(2, 0x94, 1, false),
        ];
        if stage == 4 {
            xs.push(out(3));
        }
        let words = validate(&format!("physical-{stage}"), &xs, &header(stage), state);
        let ops = instructions(&words);
        // The same generic variable serves static and physical access. The
        // outer vertex dimension adds one index for input and TCS output.
        let location_zero: Vec<_> = ops
            .iter()
            .filter(|(op, a)| *op == Op::Decorate && a[1..] == [30, 0])
            .map(|(_, a)| a[0])
            .collect();
        assert_eq!(location_zero.len(), 2);
        let input_bank = location_zero
            .iter()
            .copied()
            .find(|var| {
                ops.iter()
                    .any(|(op, a)| *op == Op::Variable && a[1] == *var && a[2] == 1)
            })
            .unwrap();
        let chains: Vec<_> = ops
            .iter()
            .filter(|(op, a)| *op == Op::AccessChain && a[2] == input_bank)
            .collect();
        assert_eq!(chains.len(), 2);
        assert!(chains.iter().all(|(_, a)| a.len() == 6));
        let definition = |id: u32| {
            ops.iter()
                .find(|(op, a)| {
                    matches!(
                        *op,
                        Op::Constant | Op::IAdd | Op::ISub | Op::ShiftRightLogical | Op::BitwiseAnd
                    ) && a[1] == id
                })
                .unwrap()
        };
        let constant = |id: u32, value: u32| {
            let (op, a) = definition(id);
            assert_eq!(*op, Op::Constant);
            assert_eq!(a[2], value);
        };
        let dynamic = chains
            .iter()
            .find(|(_, a)| definition(a[4]).0 == Op::ShiftRightLogical)
            .unwrap()
            .1;
        let (op, location) = definition(dynamic[4]);
        assert_eq!(*op, Op::ShiftRightLogical);
        constant(location[3], 4);
        let (op, mask) = definition(dynamic[5]);
        assert_eq!(*op, Op::BitwiseAnd);
        constant(mask[3], 3);
        let (op, component) = definition(mask[2]);
        assert_eq!(*op, Op::ShiftRightLogical);
        constant(component[3], 2);
        assert_eq!(location[2], component[2]);
        let (op, relative) = definition(location[2]);
        assert_eq!(*op, Op::ISub);
        constant(relative[3], 0x80);
        let static_access = chains
            .iter()
            .find(|(_, a)| definition(a[4]).0 == Op::Constant)
            .unwrap()
            .1;
        constant(static_access[4], 1); // 0x94 is location 1, component 1.
        constant(static_access[5], 1);
        // Exact byte decomposition, not register-as-location indexing:
        // (R - 0x80) >> 4 and ((R - 0x80) >> 2) & 3.
        assert!(has(&words, Op::ISub, &[]));
        assert!(has(&words, Op::ShiftRightLogical, &[]));
        assert!(has(&words, Op::BitwiseAnd, &[]));
        assert!(!has(&words, Op::CompositeExtract, &[])); // direct scalar chains, no vec4 extraction
    }
}

#[test]
fn vector_attribute_load_snapshots_aliased_vertex_source() {
    for (stage, state) in [
        (
            4,
            GraphicsState {
                geometry_input: Some(GeometryInput::Lines),
                ..Default::default()
            },
        ),
        (2, GraphicsState::default()),
        (
            3,
            GraphicsState {
                tessellation: Some(TessellationState {
                    domain: TessellationDomain::Triangles,
                    spacing: TessellationSpacing::Equal,
                    clockwise: false,
                }),
                ..Default::default()
            },
        ),
    ] {
        let words = validate(
            &format!("alias-{stage}"),
            &[
                al2p(0, 255, 1),
                ald(0, 0x70, 4, 0, false, false),
                ast(0, 0x70, 4, false),
            ],
            &header(stage),
            state,
        );
        let ops = instructions(&words);
        let input_position = ops
            .iter()
            .filter(|(op, a)| *op == Op::Decorate && a[1..] == [11, 0])
            .map(|(_, a)| a[0])
            .find(|v| {
                ops.iter()
                    .any(|(op, a)| *op == Op::Variable && a[1] == *v && a[2] == 1)
            })
            .unwrap();
        let chains: Vec<_> = ops
            .iter()
            .filter(|(op, a)| *op == Op::AccessChain && a[2] == input_position)
            .collect();
        assert_eq!(chains.len(), 4);
        let saved_vertex = chains[0].1[3];
        assert!(chains.iter().all(|(_, a)| a[3] == saved_vertex));
    }
}

#[test]
fn unproven_address_and_isbe_modifiers_stay_unknown() {
    let input = al2p(0, 1, 0x80);
    let read = isberd(0, 1);
    let load = physical(ald(0, 0, 1, 255, false, false), 1);
    let store = physical(ast(0, 0, 1, false), 1);
    for raw in [
        input | (1 << 30),
        input | (1 << 32),
        input & !(7 << 44),
        input | (1 << 47),
        read | (1 << 31),
        read | (1 << 32),
        read | (1 << 33),
        read | (1 << 34),
        read | (1 << 47),
        read | (1 << 48),
        read | (1 << 49),
        store | (0x80 << 20),
        store | (1 << 31),
        load | (1 << 47),
        store | (1 << 47),
        store | (1 << 30),
    ] {
        assert!(
            matches!(
                decode_word([raw as u32, (raw >> 32) as u32], 0).kind,
                InstructionKind::Unknown(ops::Unknown { .. })
            ),
            "{raw:016x}"
        );
    }
    for raw in [input, read, load, store] {
        assert!(!matches!(
            decode_word([raw as u32, (raw >> 32) as u32], 0).kind,
            InstructionKind::Unknown(ops::Unknown { .. }) | InstructionKind::Nop(ops::Nop)
        ));
    }
    // Without P, Ra not RZ with a nonzero immediate has no ALD meaning:
    // nvdisasm SM52 rejects this word. With P, Mesa adds Ra to the
    // immediate (nak_nir_lower_vtg_io.c:203,208-211). The address is ten
    // bits (Mesa sm50.rs:2873, envytools gm107.c:314), so bit 30 has no
    // ALD meaning.
    assert!(matches!(
        decode_word(
            [
                (load | (0x80 << 20)) as u32,
                ((load | (0x80 << 20)) >> 32) as u32
            ],
            0
        )
        .kind,
        InstructionKind::Unknown(ops::Unknown { .. })
    ));
    assert!(matches!(
        decode_word(
            [(load | (1 << 31)) as u32, ((load | (1 << 31)) >> 32) as u32],
            0
        )
        .kind,
        InstructionKind::IndexedPatchLoad(ops::IndexedPatchLoad {
            base: 0,
            index: shadowbox::ir::Register(1),
            ..
        })
    ));
    assert!(matches!(
        decode_word(
            [(load | (1 << 30)) as u32, ((load | (1 << 30)) >> 32) as u32],
            0
        )
        .kind,
        InstructionKind::Unknown(ops::Unknown { .. })
    ));
}

#[test]
fn scalar_address_and_isbe_forms_preserve_sources_and_predicates() {
    for source in [0, 1, 254, 255] {
        for address in [0, 0x80, 0x27c, 0x3ff] {
            let raw = al2p(2, source, address);
            assert!(
                matches!(decode_word([raw as u32, (raw >> 32) as u32], 0).kind,
                InstructionKind::AttributeAddress(ops::AttributeAddress { dst: shadowbox::ir::Register(2), offset, address: a })
                    if offset.0 == source && a == address)
            );
        }
        let raw = isberd(2, source);
        assert!(
            matches!(decode_word([raw as u32, (raw >> 32) as u32], 0).kind,
            InstructionKind::IsbeRead(ops::IsbeRead { dst: shadowbox::ir::Register(2), offset }) if offset.0 == source)
        );
    }
    for predicate in [0, 7, 8, 15] {
        let guard = |raw: u64| (raw & !(15 << 16)) | (predicate << 16);
        let words = validate(
            &format!("address-predicate-{predicate}"),
            &[
                guard(al2p(2, 255, 0x80)),
                guard(isberd(3, 255)),
                guard(physical(ald(4, 0, 1, 3, false, false), 2)),
                guard(physical(ast(4, 0, 1, false), 2)),
                out(1),
            ],
            &header(4),
            GraphicsState {
                geometry_input: Some(GeometryInput::Points),
                ..Default::default()
            },
        );
        // A dynamic predicate must guard register updates and attribute writes.
        if predicate == 0 || predicate == 8 {
            assert!(has(&words, Op::Select, &[]));
        }
    }
}

#[test]
fn inactive_physical_attributes_branch_before_constructing_pointers() {
    for (stage, state) in [
        (
            4,
            GraphicsState {
                geometry_input: Some(GeometryInput::Points),
                ..Default::default()
            },
        ),
        (2, GraphicsState::default()),
        (
            3,
            GraphicsState {
                tessellation: Some(TessellationState {
                    domain: TessellationDomain::Triangles,
                    spacing: TessellationSpacing::Equal,
                    clockwise: false,
                }),
                ..Default::default()
            },
        ),
    ] {
        // R10 starts at zero, outside the generic byte address range. The
        // default-false P0 must suppress the access before pointer creation.
        let predicate = |raw: u64| raw & !(7 << 16);
        let words = validate(
            &format!("inactive-physical-{stage}"),
            &[
                predicate(physical(ald(0, 0, 1, 255, false, false), 10)),
                predicate(physical(ast(0, 0, 1, false), 10)),
            ],
            &header(stage),
            state,
        );
        let ops = instructions(&words);
        let mut pointers = 0;
        for (index, (op, _)) in ops.iter().enumerate() {
            if *op != Op::AccessChain {
                continue;
            }
            pointers += 1;
            let block = ops[..index]
                .iter()
                .rev()
                .find(|(op, _)| *op == Op::Label)
                .unwrap()
                .1[0];
            assert!(ops[..index]
                .iter()
                .any(|(op, a)| *op == Op::BranchConditional && a[1] == block));
            // Attribute access is inside the true block, with no result select.
            // Conditional EXIT elsewhere may legitimately use OpSelect.
            let end = ops[index + 1..]
                .iter()
                .position(|(op, _)| *op == Op::Label)
                .map_or(ops.len(), |next| index + 1 + next);
            assert!(!ops[index..end].iter().any(|(op, _)| *op == Op::Select));
        }
        assert_eq!(pointers, 2);
    }
}

#[test]
fn primitive_ids_are_scalar_inputs_and_persistent_geometry_outputs() {
    for (stage, state) in [
        (
            4,
            GraphicsState {
                geometry_input: Some(GeometryInput::Points),
                ..Default::default()
            },
        ),
        (2, GraphicsState::default()),
        (
            3,
            GraphicsState {
                tessellation: Some(TessellationState {
                    domain: TessellationDomain::Triangles,
                    spacing: TessellationSpacing::Equal,
                    clockwise: false,
                }),
                ..Default::default()
            },
        ),
    ] {
        let mut xs = vec![ald(0, 0x60, 1, 255, false, false), ast(0, 0x80, 1, false)];
        if stage == 4 {
            xs.extend([
                ast(0, 0x60, 1, false),
                ald(1, 0x60, 1, 255, false, true),
                ast(1, 0x84, 1, false),
                out(1),
                out(3),
            ]);
        }
        let words = validate(&format!("primitive-id-{stage}"), &xs, &header(stage), state);
        let ops = instructions(&words);
        let primitives: Vec<_> = ops
            .iter()
            .filter(|(op, a)| *op == Op::Decorate && a[1..] == [11, 7])
            .map(|(_, a)| a[0])
            .collect();
        assert_eq!(primitives.len(), if stage == 4 { 2 } else { 1 });
        for primitive in primitives {
            let variable = ops
                .iter()
                .find(|(op, a)| *op == Op::Variable && a[1] == primitive)
                .unwrap()
                .1;
            let pointer = ops
                .iter()
                .find(|(op, a)| *op == Op::TypePointer && a[0] == variable[0])
                .unwrap()
                .1;
            assert!(ops
                .iter()
                .any(|(op, a)| *op == Op::TypeInt && a[0] == pointer[2] && a[1] == 32));
            if variable[2] == 3 {
                assert_eq!(
                    ops.iter()
                        .filter(|(op, a)| *op == Op::Store && a[0] == primitive)
                        .count(),
                    2
                );
            }
        }
        // A primitive ID has no primitive-vertex index: Ryujinx
        // InstEmitAttribute.cs:27-29,297-302 ignores Rb for it.
        let raw = ald(0, 0x60, 1, 1, false, false);
        assert!(matches!(
            decode_word([raw as u32, (raw >> 32) as u32], 0).kind,
            InstructionKind::AttributeLoad(ops::AttributeLoad {
                vertex: shadowbox::ir::PrimitiveVertex::Current,
                ..
            })
        ));
        if stage != 4 {
            assert!(Result::from(translate_with_graphics_state(
                shadowbox::QuietRun,
                &ShaderInput {
                    code: &code(&[ast(0, 0x60, 1, false)]),
                    program_record: &header(stage),
                },
                &state
            ))
            .is_err());
        }
    }
}
fn instructions(words: &[u32]) -> Vec<(Op, &[u32])> {
    let mut result = Vec::new();
    let mut index = 5;
    while index < words.len() {
        let len = (words[index] >> 16) as usize;
        result.push((Op::of(words[index]), &words[index + 1..index + len]));
        index += len;
    }
    result
}
fn has(words: &[u32], opcode: Op, suffix: &[u32]) -> bool {
    instructions(words)
        .iter()
        .any(|(op, args)| *op == opcode && args.ends_with(suffix))
}

#[test]
fn geometry_emits_cuts_and_retains_output_bank() {
    // Register vertex selection is exercised on input Position and generic I/O.
    let xs = [
        ald(0, 0x70, 4, 255, false, false),
        ast(0, 0x70, 4, false),
        ald(4, 0x84, 1, 2, false, false),
        ast(4, 0x84, 1, false),
        s2r(10, 0x11),
        ast(10, 0x90, 1, false),
        s2r(11, 0),
        ast(11, 0x94, 1, false),
        ald(12, 0x84, 1, 255, false, true),
        ast(12, 0x98, 1, false),
        out(1),
        out(3),
    ];
    for (name, topology, mode, count) in [
        ("points", GeometryInput::Points, 19, 1),
        ("lines", GeometryInput::Lines, 20, 2),
        ("lines-adj", GeometryInput::LinesAdjacency, 21, 4),
        ("triangles", GeometryInput::Triangles, 22, 3),
        ("triangles-adj", GeometryInput::TrianglesAdjacency, 23, 6),
    ] {
        let words = validate(
            name,
            &xs,
            &header(4),
            GraphicsState {
                geometry_input: Some(topology),
                ..Default::default()
            },
        );
        assert!(instructions(&words)
            .iter()
            .any(|(op, a)| *op == Op::EntryPoint && a[0] == 3));
        assert!(has(&words, Op::ExecutionMode, &[mode]));
        assert!(has(&words, Op::ExecutionMode, &[26, 3]));
        assert!(has(&words, Op::ExecutionMode, &[0, 3]));
        assert!(has(&words, Op::Constant, &[count]));
        assert_eq!(
            instructions(&words)
                .iter()
                .filter(|(op, _)| *op == Op::EmitVertex)
                .count(),
            2
        );
        assert!(has(&words, Op::EndPrimitive, &[]));
        // Both emissions must copy the persistent Private bank to Output.
        assert!(has(&words, Op::Variable, &[6]));
    }
    for (topology, mode) in [(1, 27), (6, 28), (7, 29)] {
        let mut h = header(4);
        h[15] = topology;
        let words = validate(
            &format!("output-{mode}"),
            &xs,
            &h,
            GraphicsState {
                geometry_input: Some(GeometryInput::Triangles),
                ..Default::default()
            },
        );
        assert!(has(&words, Op::ExecutionMode, &[mode]));
    }
    let mut predicated = out(3);
    predicated &= !(7 << 16); // P0
    let words = validate(
        "predicated-out",
        &[predicated],
        &header(4),
        GraphicsState {
            geometry_input: Some(GeometryInput::Points),
            ..Default::default()
        },
    );
    assert!(has(&words, Op::BranchConditional, &[]));
    assert!(has(&words, Op::EmitVertex, &[]));
    assert!(has(&words, Op::EndPrimitive, &[]));
}

#[test]
fn immediate_geometry_streams_preserve_selected_output_interface() {
    let mut h = header(4);
    h[15] = 1; // Mesa cla097sph.h:68-71: PointList.
    for stream in 0..=3 {
        let mut predicated = out_stream(3, stream);
        predicated &= !(7 << 16); // P0, Mesa sm50.rs:225-235.
        let words = validate(
            &format!("stream-{stream}"),
            &[
                ald(0, 0x70, 4, 255, false, false),
                ast(0, 0x70, 4, false),
                ast(0, 0x80, 4, false),
                ald(8, 0x60, 1, 255, false, false),
                ast(8, 0x60, 1, false),
                out_stream(1, stream),
                predicated,
                out_stream(2, stream),
            ],
            &h,
            GraphicsState {
                geometry_input: Some(GeometryInput::Points),
                ..Default::default()
            },
        );
        if stream == 0 {
            assert!(!has(&words, Op::Capability, &[54]));
            assert!(has(&words, Op::EmitVertex, &[]));
            assert!(has(&words, Op::EndPrimitive, &[]));
        } else {
            assert!(has(&words, Op::Capability, &[54]));
            assert!(has(&words, Op::Decorate, &[29, u32::from(stream)]));
            let ops = instructions(&words);
            let stream_constants: Vec<_> = ops
                .iter()
                .filter(|(op, a)| *op == Op::Constant && a[2] == u32::from(stream))
                .map(|(_, a)| a[1])
                .collect();
            assert!(ops
                .iter()
                .any(|(op, a)| *op == Op::EmitStreamVertex && stream_constants.contains(&a[0])));
            assert!(ops
                .iter()
                .any(|(op, a)| *op == Op::EndStreamPrimitive && stream_constants.contains(&a[0])));
            // Every emission publishes persistent attributes to its own stream.
            for (_, decoration) in ops
                .iter()
                .filter(|(op, a)| *op == Op::Decorate && a[1..] == [29, u32::from(stream)])
            {
                assert!(ops
                    .iter()
                    .any(|(op, a)| *op == Op::Store && a[0] == decoration[0]));
            }
        }
    }
    for raw in [
        out_stream(1, 4),
        out_stream(1, 255),
        out_stream(1, 1) | (1 << 56),
    ] {
        assert!(matches!(
            decode_word([raw as u32, (raw >> 32) as u32], 0).kind,
            InstructionKind::Unknown(ops::Unknown { .. })
        ));
    }
    for streams in [[0, 1], [1, 0], [1, 2]] {
        assert!(Result::from(translate_with_graphics_state(
            shadowbox::QuietRun,
            &ShaderInput {
                code: &code(&[out_stream(1, streams[0]), out_stream(1, streams[1])]),
                program_record: &h,
            },
            &GraphicsState {
                geometry_input: Some(GeometryInput::Points),
                ..Default::default()
            }
        ))
        .is_err());
    }
    assert!(Result::from(translate_with_graphics_state(
        shadowbox::QuietRun,
        &ShaderInput {
            code: &code(&[out_stream(1, 1)]),
            program_record: &header(4),
        },
        &GraphicsState {
            geometry_input: Some(GeometryInput::Triangles),
            ..Default::default()
        }
    ))
    .is_err());
}
#[test]
fn tessellation_control_writes_patch_and_invocation_outputs_and_barriers() {
    // BAR.SYNC: Mesa sm50.rs:3114-3124 and decoder's supported all-thread form.
    let bar = (0xf0a8u64 << 48) | (7 << 16) | (7 << 39) | (1 << 43) | (1 << 44);
    let xs = [
        s2r(2, 0x11),
        ald(0, 0x70, 4, 2, false, false),
        ast(0, 0x70, 4, false),
        ast(0, 0x80, 4, false),
        ast(0, 0, 4, true),
        ast(0, 0x10, 2, true),
        bar,
        ald(8, 0x14, 1, 255, true, true),
        ast(8, 0x90, 1, false),
        ald(4, 0x84, 1, 255, false, true),
    ];
    let words = validate("tcs", &xs, &header(2), GraphicsState::default());
    assert!(has(&words, Op::Capability, &[3]));
    assert!(has(&words, Op::ExecutionMode, &[26, 3]));
    assert!(has(&words, Op::Decorate, &[11, 8]));
    assert!(has(&words, Op::Decorate, &[11, 11]));
    assert!(has(&words, Op::Decorate, &[11, 12]));
    assert!(has(&words, Op::Decorate, &[15]));
    assert!(has(&words, Op::ControlBarrier, &[]));
    assert!(has(&words, Op::Constant, &[4]));
}
#[test]
fn tessellation_evaluation_reads_patch_vertices_and_tesscoord() {
    let xs = [
        ald(0, 0x70, 4, 2, false, false),
        ast(0, 0x70, 4, false),
        ald(4, 0x84, 1, 2, false, false),
        ast(4, 0x84, 1, false),
        ald(8, 0x2f0, 2, 255, false, false),
        ast(8, 0x88, 2, false),
        ald(10, 0, 4, 255, true, false),
        ald(14, 0x10, 2, 255, true, false),
    ];
    for (name, domain, mode) in [
        ("isolines", TessellationDomain::Isolines, 25),
        ("triangles", TessellationDomain::Triangles, 22),
        ("quads", TessellationDomain::Quads, 24),
    ] {
        for (spacing, sm) in [
            (TessellationSpacing::Equal, 1),
            (TessellationSpacing::FractionalEven, 2),
            (TessellationSpacing::FractionalOdd, 3),
        ] {
            for clockwise in [false, true] {
                let words = validate(
                    &format!("tes-{name}-{sm}-{clockwise}"),
                    &xs,
                    &header(3),
                    GraphicsState {
                        tessellation: Some(TessellationState {
                            domain,
                            spacing,
                            clockwise,
                        }),
                        ..Default::default()
                    },
                );
                assert!(instructions(&words)
                    .iter()
                    .any(|(op, a)| *op == Op::EntryPoint && a[0] == 2));
                assert!(has(&words, Op::ExecutionMode, &[mode]));
                assert!(has(&words, Op::ExecutionMode, &[sm]));
                assert!(has(
                    &words,
                    Op::ExecutionMode,
                    &[if clockwise { 4 } else { 5 }]
                ));
                assert!(has(&words, Op::Decorate, &[11, 13]));
            }
        }
    }
}
#[test]
fn unsupported_state_and_out_forms_fail_without_noops() {
    for (raw, expected) in [
        (2, Stage::TessellationControl),
        (3, Stage::TessellationEvaluation),
        (4, Stage::Geometry),
    ] {
        let h = header(raw);
        let words = h
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect::<Vec<_>>();
        assert_eq!(parse(&words).unwrap().stage(), expected);
        if raw != 2 {
            assert!(Result::from(translate(
                shadowbox::QuietRun,
                &ShaderInput {
                    code: &code(&[]),
                    program_record: &h
                }
            ))
            .is_err());
        }
    }
    // A register stream is an OUT whose stream a proof names.
    assert!(matches!(
        decode_word(
            [
                (out(1) & !(255 << 20)) as u32,
                ((out(1) & !(255 << 20)) >> 32) as u32
            ],
            0
        )
        .kind,
        InstructionKind::GeometryOutDynamic(ops::GeometryOutDynamic { .. })
    ));
    for raw in [out(0), out(1) & !255, out(1) & !(255 << 8)] {
        assert!(matches!(
            decode_word([raw as u32, (raw >> 32) as u32], 0).kind,
            InstructionKind::Unknown(ops::Unknown { .. })
        ));
    }
    for raw in [
        ast(0, 0x200, 1, true),
        ast(0, 0x1fc, 2, true),
        ast(0, 0x18, 1, true) & !(255 << 8),
        ast(0, 0x18, 1, true) | (1 << 30),
    ] {
        assert!(matches!(
            decode_word([raw as u32, (raw >> 32) as u32], 0).kind,
            InstructionKind::Unknown(ops::Unknown { .. })
        ));
    }
    // Patch ALD is never physical and ignores Rb (Ryujinx
    // InstDecoders.cs:887; InstEmitAttribute.cs:25), and the patch map ends
    // at 0x200 (AttributeMap.cs:90-92,129-133). Ra not RZ indexes the
    // patch bank (Mesa nak_nir_lower_vtg_io.c:203,208-211). Bit 30 is
    // outside the ten-bit address (Mesa sm50.rs:2873).
    assert!(matches!(
        {
            let raw = ald(0, 0x18, 1, 255, true, false) | (1 << 30);
            decode_word([raw as u32, (raw >> 32) as u32], 0).kind
        },
        InstructionKind::Unknown(ops::Unknown { .. })
    ));
    for (raw, unmapped) in [
        (ald(0, 0x200, 1, 255, true, false), true),
        (ald(0, 0x1fc, 2, 255, true, false), false),
        (ald(0, 0x18, 1, 1, true, false), false),
    ] {
        let kind = decode_word([raw as u32, (raw >> 32) as u32], 0).kind;
        if unmapped {
            assert!(matches!(
                kind,
                InstructionKind::UnmappedAttributeLoad(ops::UnmappedAttributeLoad { .. })
            ));
        } else {
            assert!(matches!(
                kind,
                InstructionKind::PatchAttributeLoad(ops::PatchAttributeLoad { .. })
            ));
        }
    }
    let raw = ald(0, 0x18, 1, 255, true, false) & !(255 << 8);
    assert!(matches!(
        decode_word([raw as u32, (raw >> 32) as u32], 0).kind,
        InstructionKind::IndexedPatchLoad(ops::IndexedPatchLoad {
            base: 0x18,
            index: shadowbox::ir::Register(0),
            ..
        })
    ));
    let nop_arms = include_str!("../src/decoder/families.rs")
        .matches("x.guarded(InstructionKind::Nop(ops::Nop))")
        .count();
    assert_eq!(nop_arms, 2);
}

#[test]
fn stage_specific_system_registers_and_state_are_checked() {
    for (stage, state) in [
        (
            4,
            GraphicsState {
                geometry_input: Some(GeometryInput::Triangles),
                ..Default::default()
            },
        ),
        (2, GraphicsState::default()),
        (
            3,
            GraphicsState {
                tessellation: Some(TessellationState {
                    domain: TessellationDomain::Triangles,
                    spacing: TessellationSpacing::Equal,
                    clockwise: false,
                }),
                ..Default::default()
            },
        ),
    ] {
        for index in [0x21, 0x22, 0x23, 0x25, 0x26, 0x27] {
            assert!(Result::from(translate_with_graphics_state(
                shadowbox::QuietRun,
                &ShaderInput {
                    code: &code(&[s2r(0, index)]),
                    program_record: &header(stage)
                },
                &state
            ))
            .is_err());
        }
        if stage == 3 {
            assert!(Result::from(translate_with_graphics_state(
                shadowbox::QuietRun,
                &ShaderInput {
                    code: &code(&[s2r(0, 0x11)]),
                    program_record: &header(stage)
                },
                &state
            ))
            .is_err());
        }
    }
    let mut passthrough = header(4);
    passthrough[3] |= 1;
    assert!(Result::from(translate_with_graphics_state(
        shadowbox::QuietRun,
        &ShaderInput {
            code: &code(&[]),
            program_record: &passthrough
        },
        &GraphicsState {
            geometry_input: Some(GeometryInput::Triangles),
            ..Default::default()
        }
    ))
    .is_err());
}

#[test]
fn geometry_store_reads_zero_register_as_zero() {
    // Ryujinx OperandHelper.cs:43-47, InstEmitAttribute.cs:109-114.
    let words = validate(
        "rz-store",
        &[s2r(255, 0x11), ast(255, 0x70, 1, false), out(1)],
        &header(4),
        GraphicsState {
            geometry_input: Some(GeometryInput::Points),
            ..Default::default()
        },
    );
    let ops = instructions(&words);
    // Read the stored bit pattern through constants and bitcasts; the
    // optimizer may keep a uint-zero bitcast or fold it into float zero.
    fn bits(ops: &[(Op, &[u32])], id: u32) -> Option<u32> {
        ops.iter().find_map(|(op, a)| {
            if matches!(op, Op::Constant | Op::Bitcast) && a[1] == id {
                if *op == Op::Constant {
                    Some(a[2])
                } else {
                    bits(ops, a[2])
                }
            } else {
                None
            }
        })
    }
    let shadow = ops
        .iter()
        .find(|(op, a)| *op == Op::Variable && a[2] == 6)
        .unwrap()
        .1[1];
    let pointer = ops
        .iter()
        .find(|(op, a)| *op == Op::AccessChain && a[2] == shadow)
        .unwrap()
        .1[1];
    assert!(ops
        .iter()
        .any(|(op, a)| *op == Op::Store && a[0] == pointer && bits(&ops, a[1]) == Some(0)));
}

#[test]
fn geometry_combines_driver_state_global_options_and_double_output() {
    // LDG.E B32: Mesa sm50.rs:2609-2614,2386-2410.
    // PT guard16..19: Mesa sm50.rs:225-235. Register I2F source20..28
    // and destination: sm50.rs:1851-1852,1866; formats:1873-1875.
    // I2F.F64: Mesa sm50.rs:1873-1879; double pair semantics:
    // Ryujinx InstEmitConversion.cs:268-272,410-415.
    let load = (0xeed4_u64 << 48) | (1 << 45) | (7 << 16) | (10 << 8) | 1;
    let convert = (0x5cb8_u64 << 48) | (1 << 20) | (7 << 16) | (2 << 10) | (3 << 8) | 4;
    let bytes = code(&[load, convert, ast(4, 0x70, 2, false), out(1)]);
    let h = header(4);
    let state = GraphicsState {
        geometry_input: Some(GeometryInput::Lines),
        tessellation: None,
        ..Default::default()
    };
    for global_delta in [shadowbox::GlobalDelta::Pushed, shadowbox::GlobalDelta::Zero] {
        let words = Result::from(shadowbox::translate_with_options_and_graphics_state(
            shadowbox::QuietRun,
            &ShaderInput {
                code: &bytes,
                program_record: &h,
            },
            shadowbox::TranslationOptions {
                global_delta,
                ..Default::default()
            },
            &state,
        ))
        .unwrap()
        .spirv;
        let check = spirv_validation::validate(&words);
        assert!(
            check.status.success(),
            "{}",
            String::from_utf8_lossy(&check.stderr)
        );
        assert!(has(&words, Op::MemoryModel, &[5348, 1]));
        assert!(has(&words, Op::Capability, &[2]));
        assert!(has(&words, Op::Capability, &[10]));
        assert!(has(&words, Op::ExecutionMode, &[20]));
        assert!(has(&words, Op::EmitVertex, &[]));
        assert_eq!(
            has(&words, Op::Variable, &[9]),
            global_delta == shadowbox::GlobalDelta::Pushed
        );
    }
}

fn patch_state(layout: PatchLayout) -> GraphicsState<'static> {
    GraphicsState {
        tessellation: Some(TessellationState {
            domain: TessellationDomain::Triangles,
            spacing: TessellationSpacing::Equal,
            clockwise: false,
        }),
        patch_io: Some(PatchIo {
            layout,
            locations: std::array::from_fn(|slot| Some(slot as u32 + 1)),
        }),
        ..Default::default()
    }
}

// Read actual scalar access chains, their storage classes and decorations.
// This checks component selection and linking in addition to module validity.
fn patch_accesses(words: &[u32]) -> Vec<(u32, u32, u32)> {
    let ops = instructions(words);
    let constant = |id| {
        ops.iter()
            .find(|(op, a)| *op == Op::Constant && a[1] == id)
            .unwrap()
            .1[2]
    };
    ops.iter()
        .filter_map(|(op, a)| {
            if *op != Op::AccessChain {
                return None;
            }
            let base = a[2];
            if !ops
                .iter()
                .any(|(op, a)| *op == Op::Decorate && a == &[base, 15])
            {
                return None;
            }
            let location = ops
                .iter()
                .find(|(op, a)| *op == Op::Decorate && a[0] == base && a[1] == 30)?
                .1[2];
            let storage = ops
                .iter()
                .find(|(op, a)| *op == Op::Variable && a[1] == base)
                .unwrap()
                .1[2];
            assert_eq!(a.len(), 4, "patch attributes must not have a vertex index");
            Some((storage, location, constant(a[3])))
        })
        .collect()
}

#[test]
fn generic_patch_slots_keep_each_producers_components_and_linked_locations() {
    // PatchLayout selects the producer's layout. Ryujinx starts generic patch
    // attributes at 0x18. Mesa starts them at 0x20. Neither base is universal.
    for (name, layout, base) in [
        ("ryujinx", PatchLayout::Ryujinx, 0x18),
        ("mesa", PatchLayout::Mesa, 0x20),
    ] {
        let state = patch_state(layout);
        // Crossing vec4 boundaries and the final partial slot exposes base
        // rounding errors. Ryujinx AttributeMap.cs:92,109-121; Mesa
        // nak_private.h:39 and nak_nir.c:422-423.
        let addresses = (base..0x200).step_by(4).collect::<Vec<_>>();
        let stores = addresses
            .iter()
            .map(|&a| ast(0, a, 1, true))
            .collect::<Vec<_>>();
        let reads = addresses
            .iter()
            .map(|&a| ald(0, a, 1, 255, true, true))
            .collect::<Vec<_>>();
        let xs = stores.into_iter().chain(reads).collect::<Vec<_>>();
        let tcs = validate(&format!("{name}-all-patch-tcs"), &xs, &header(2), state);
        let xs = addresses
            .iter()
            .map(|&a| ald(0, a, 1, 255, true, false))
            .collect::<Vec<_>>();
        let all_tes = validate(&format!("{name}-all-patch-tes"), &xs, &header(3), state);
        let expected = addresses
            .iter()
            .map(|&a| {
                (
                    3,
                    u32::from((a - base) / 16) + 1,
                    u32::from((a - base) % 16 / 4),
                )
            })
            .collect::<Vec<_>>();
        // One scalar chain for each AST and each output-bank ALD.
        assert_eq!(
            patch_accesses(&tcs),
            expected
                .iter()
                .chain(&expected)
                .copied()
                .collect::<Vec<_>>()
        );
        assert_eq!(
            patch_accesses(&all_tes),
            expected
                .iter()
                .map(|&(_, l, c)| (1, l, c))
                .collect::<Vec<_>>()
        );
        // Consume TES reads so scalar input accesses remain after optimization.
        let xs = [
            ald(0, base + 12, 4, 255, true, false),
            ast(0, 0x80, 4, false),
        ];
        let tes = validate(
            &format!("{name}-patch-cross-vec-tes"),
            &xs,
            &header(3),
            state,
        );
        assert_eq!(
            patch_accesses(&tes),
            vec![(1, 1, 3), (1, 2, 0), (1, 2, 1), (1, 2, 2)]
        );
        assert!(has(&tcs, Op::Decorate, &[15]));
        assert!(has(&tes, Op::Decorate, &[15]));
    }
}

#[test]
fn indexed_patch_loads_add_ra_to_the_immediate() {
    // Mesa nak_nir_lower_vtg_io.c:203,208-211 and nak/sm50.rs:2864-2868:
    // P with Ra not RZ reads the patch word at the immediate plus Ra.
    let indexed = |dst, address, count, output| {
        (ald(dst, address, count, 255, true, output) & !(255 << 8)) | (1 << 8)
    };
    let state = patch_state(PatchLayout::Ryujinx);
    // Every bank word is a candidate: four outer levels, two inner levels
    // and four words per linked generic slot.
    let words = 4 + 2 + 31 * 4;
    let compares = |module: &[u32]| {
        instructions(module)
            .iter()
            .filter(|(op, _)| *op == Op::IEqual)
            .count()
    };
    for (name, stage, count, output) in [("tes", 3, 4, false), ("tcs", 2, 2, true)] {
        let module = validate(
            &format!("indexed-patch-{name}"),
            &[mov(1, 0x10), indexed(4, 0x18, count, output)],
            &header(stage),
            state,
        );
        assert_eq!(compares(&module), usize::from(count) * words, "{name}");
    }
    // Tessellation control has no patch input bank: each lane reads 0,
    // as the static lane does (Ryujinx AttributeMap.cs:129-141).
    let module = validate(
        "indexed-patch-tcs-input",
        &[mov(1, 0x10), indexed(4, 0x18, 1, false)],
        &header(2),
        state,
    );
    assert_eq!(compares(&module), 0);
    let raw = indexed(4, 0x18, 3, true);
    assert_eq!(
        decode_word([raw as u32, (raw >> 32) as u32], 0).kind,
        InstructionKind::IndexedPatchLoad(ops::IndexedPatchLoad {
            dst: shadowbox::ir::Register(4),
            base: 0x18,
            index: shadowbox::ir::Register(1),
            components: 3,
            bank: shadowbox::ir::AttributeBank::Output,
            lanes: shadowbox::ir::AttributeLanes::patch(
                0x18,
                3,
                shadowbox::ir::AttributeBank::Output
            ),
        })
    );
}

#[test]
fn tessellation_control_output_reads_take_rb_as_a_lane() {
    // Mesa nak_nir_lower_vtg_io.c:10-49,86-90: in tessellation control
    // Rb of ALD.O is lane - invocation + vertex, so the control point is
    // Rb - LANEID + InvocationId. SubgroupLocalInvocationId is BuiltIn 41,
    // InvocationId 8 (Mesa spirv.core.grammar.json).
    let builtin = |module: &[u32], value: u32| {
        instructions(module).iter().any(|(op, args)| {
            *op == Op::Decorate && args.len() == 3 && args[1] == 11 && args[2] == value
        })
    };
    let state = patch_state(PatchLayout::Ryujinx);
    let read = ald(4, 0x80, 1, 1, false, true);
    let tcs = validate("tcs-output-lane", &[mov(1, 2), read], &header(2), state);
    assert!(builtin(&tcs, 41) && builtin(&tcs, 8));
    // RZ keeps the invocation's own control point; no lane is read.
    let own = validate(
        "tcs-output-own",
        &[ald(4, 0x80, 1, 255, false, true)],
        &header(2),
        state,
    );
    assert!(!builtin(&own, 41));
    // Outside tessellation control the output bank is not per vertex and
    // Rb stays unread, as Ryujinx AttributeMap.cs:323-328 reads it.
    let geometry = GraphicsState {
        geometry_input: Some(GeometryInput::Triangles),
        ..Default::default()
    };
    let gs = validate("gs-output-lane", &[mov(1, 2), read], &header(4), geometry);
    assert!(!builtin(&gs, 41));
    assert!(matches!(
        decode_word([read as u32, (read >> 32) as u32], 0).kind,
        InstructionKind::OutputAttributeLoad(ops::OutputAttributeLoad {
            vertex: shadowbox::ir::OutputVertex::Lane(shadowbox::ir::Register(1)),
            ..
        })
    ));
}

#[test]
fn generic_patch_driver_layout_is_required_and_checked() {
    let code = code(&[ast(0, 0x18, 1, true)]);
    let h = header(2);
    let input = ShaderInput {
        code: &code,
        program_record: &h,
    };
    for state in [
        GraphicsState::default(),
        patch_state(PatchLayout::Mesa),
        GraphicsState {
            patch_io: Some(PatchIo {
                layout: PatchLayout::Ryujinx,
                locations: [None; 31],
            }),
            ..Default::default()
        },
    ] {
        assert!(Result::from(translate_with_graphics_state(
            shadowbox::QuietRun,
            &input,
            &state
        ))
        .is_err());
    }
    let mut state = patch_state(PatchLayout::Ryujinx);
    state.patch_io.as_mut().unwrap().locations[0] = Some(0);
    // Header-declared per-vertex output location 0 must not overlap Patch.
    assert!(Result::from(translate_with_graphics_state(
        shadowbox::QuietRun,
        &input,
        &state
    ))
    .is_err());
    state.patch_io.as_mut().unwrap().locations[0] = Some(2);
    let code = code_for_duplicate_patch_slots();
    assert!(Result::from(translate_with_graphics_state(
        shadowbox::QuietRun,
        &ShaderInput {
            code: &code,
            program_record: &h
        },
        &state
    ))
    .is_err());
}

fn code_for_duplicate_patch_slots() -> Vec<u8> {
    code(&[ast(0, 0x18, 1, true), ast(0, 0x28, 1, true)])
}

#[test]
fn out_register_and_immediate_zero_keep_emit_cut_and_predication() {
    for mode in 1..=3 {
        for immediate in [false, true] {
            let raw = if immediate {
                // Mesa sm50.rs:3263-3265: signed i20 stream zero.
                (out(mode) & !(0xffff << 48) & !(255 << 20)) | (0xf6e0 << 48)
            } else {
                out(mode)
            };
            let mut unpredicated_branches = 0;
            for guard in [7, 0, 8] {
                let raw = (raw & !(15 << 16)) | (guard << 16);
                let words = validate(
                    &format!("out-{mode}-{immediate}-{guard}"),
                    &[raw],
                    &header(4),
                    GraphicsState {
                        geometry_input: Some(GeometryInput::Points),
                        ..Default::default()
                    },
                );
                assert_eq!(has(&words, Op::EmitVertex, &[]), mode & 1 != 0);
                assert_eq!(has(&words, Op::EndPrimitive, &[]), mode & 2 != 0);
                let branches = instructions(&words)
                    .iter()
                    .filter(|(op, _)| *op == Op::BranchConditional)
                    .count();
                if guard == 7 {
                    unpredicated_branches = branches;
                } else {
                    assert_eq!(branches, unpredicated_branches + 1);
                }
            }
        }
    }
    // A real stream, handle or result cannot be silently discarded. Mesa
    // sm50.rs:3260-3284; ir.rs:7986-7995. Cbuf stream: sm50.rs:3267-3269.
    // Register and cbuf streams are OUTs whose stream a proof names.
    for raw in [
        (out(1) & !(255 << 20)) | (1 << 20),
        (out(1) & !(0xffff << 48) & !(255 << 20)) | (0xebe0 << 48),
    ] {
        assert!(matches!(
            decode_word([raw as u32, (raw >> 32) as u32], 0).kind,
            InstructionKind::GeometryOutDynamic(ops::GeometryOutDynamic { .. })
        ));
    }
    for raw in [
        out_stream(1, 4),
        (out(1) & !(0xffff << 48) & !(255 << 20)) | (0xf7e0 << 48),
        out(1) & !255,
        out(1) & !(255 << 8),
    ] {
        assert!(matches!(
            decode_word([raw as u32, (raw >> 32) as u32], 0).kind,
            InstructionKind::Unknown(ops::Unknown { .. })
        ));
    }
}

#[test]
fn generic_attribute_io_is_validated_in_each_stage() {
    let compute = validate(
        "compute-without-attributes",
        &[],
        &header(0),
        GraphicsState::default(),
    );
    assert!(has(&compute, Op::ExecutionMode, &[17, 1, 1, 1]));
    for (stage, mut state) in [
        (1, GraphicsState::default()),
        (5, GraphicsState::default()),
        (2, GraphicsState::default()),
        (3, patch_state(PatchLayout::Ryujinx)),
        (
            4,
            GraphicsState {
                geometry_input: Some(GeometryInput::Triangles),
                ..Default::default()
            },
        ),
    ] {
        let mut h = header(stage);
        let xs = if stage == 5 {
            // Fragment export R0: Mesa from_nir.rs:2990-3031. Flat generic
            // location zero: Mesa sph.rs:137-142,531-532; Ryujinx Declarations.cs:628-638.
            h[24] = 0x55;
            h[72] = 15;
            vec![ald(0, 0x80, 4, 255, false, false)]
        } else {
            let mut xs = vec![
                ald(0, 0x80, 4, 255, false, false),
                ast(0, 0x80, 4, false),
                ald(4, 0x80, 4, 255, false, true),
                ast(4, 0x70, 4, false),
            ];
            if stage == 4 {
                xs.push(out(1));
            }
            xs
        };
        state.patch_io = None;
        let words = validate(&format!("generic-stage-{stage}"), &xs, &h, state);
        assert!(has(&words, Op::Decorate, &[30, 0]));
        if stage == 5 {
            assert!(has(&words, Op::Decorate, &[14]));
        }
    }
    // Compute has no graphics attribute interface (Ryujinx AttributeMap.cs:63-81),
    // and fragment Position is no output: AttributeMap.cs:136-141 logs such
    // an ALD lane and loads 0. Fragment outputs are GPR exports, not AST.
    let mut h = header(5);
    h[72] = 1;
    validate(
        "fragment-output-position-ald",
        &[ald(0, 0x70, 1, 255, false, true)],
        &h,
        GraphicsState::default(),
    );
    validate(
        "compute-generic-ald",
        &[ald(0, 0x80, 1, 255, false, false)],
        &header(0),
        GraphicsState::default(),
    );
    for (stage, raw) in [(0, ast(0, 0x70, 1, false)), (5, ast(0, 0x70, 1, false))] {
        assert!(Result::from(translate_with_graphics_state(
            shadowbox::QuietRun,
            &ShaderInput {
                code: &code(&[raw]),
                program_record: &header(stage)
            },
            &GraphicsState::default()
        ))
        .is_err());
    }
}

#[test]
fn fragment_ald_position_is_fragcoord_and_vertex_rz_stores_are_zero() {
    let mut h = header(5);
    h[72] = 15; // Four exported GPR lanes, Mesa from_nir.rs:2990-3031.
    let words = validate(
        "fragment-ald-position",
        &[ald(0, 0x70, 4, 255, false, false)],
        &h,
        GraphicsState::default(),
    );
    assert!(has(&words, Op::Decorate, &[11, 15])); // Ryujinx IoMap.cs:22.
                                                   // Near-RZ ALD must not index beyond the register bank.
    validate(
        "vertex-near-rz",
        &[
            ald(254, 0x80, 4, 255, false, false),
            ast(254, 0x70, 4, false),
        ],
        &header(1),
        GraphicsState::default(),
    );
    let words = validate(
        "vertex-rz-zero",
        &[s2r(255, 0), ast(255, 0x70, 1, false)],
        &header(1),
        GraphicsState::default(),
    );
    let ops = instructions(&words);
    let position = ops
        .iter()
        .find(|(op, a)| *op == Op::Decorate && a.ends_with(&[11, 0]))
        .unwrap()
        .1[0];
    let pointer = ops
        .iter()
        .find(|(op, a)| *op == Op::AccessChain && a[2] == position)
        .unwrap()
        .1[1];
    let value = ops
        .iter()
        .find(|(op, a)| *op == Op::Store && a[0] == pointer)
        .unwrap()
        .1[1];
    // Ast reads literal zero even when an earlier instruction targets RZ.
    fn bits(ops: &[(Op, &[u32])], id: u32) -> Option<u32> {
        ops.iter().find_map(|(op, a)| {
            if matches!(op, Op::Constant | Op::Bitcast) && a[1] == id {
                if *op == Op::Constant {
                    Some(a[2])
                } else {
                    bits(ops, a[2])
                }
            } else {
                None
            }
        })
    }
    assert_eq!(bits(&ops, value), Some(0));
}

#[test]
fn geometry_passthrough_forwards_exact_vertices_and_keeps_written_attributes() {
    // Bit24: Mesa nak/sph.rs:240-243, Ryujinx ShaderHeader.cs:125.
    // Forwarding and adjacency indices: Ryujinx EmitterContext.cs:360-426;
    // written locations excluded: AttributeUsage.cs:91-97.
    for (input, topology, vertices) in [
        (GeometryInput::Points, 1, vec![0]),
        (GeometryInput::Lines, 6, vec![0, 1]),
        (GeometryInput::LinesAdjacency, 6, vec![1, 2]),
        (GeometryInput::Triangles, 7, vec![0, 1, 2]),
        (GeometryInput::TrianglesAdjacency, 7, vec![0, 2, 4]),
    ] {
        let mut h = header(4);
        h[3] |= 1;
        h[11] = 1;
        h[15] = topology;
        // The passthrough maximum comes from input topology, even if SPH is 0.
        h[16..18].fill(0);
        h[24..40].fill(0);
        h[54..70].fill(0);
        let words = validate(
            "passthrough",
            &[s2r(0, 0x11), ast(0, 0x80, 1, false)],
            &h,
            GraphicsState {
                geometry_input: Some(input),
                geometry_next_inputs: Some(1 | (1 << 2) | (1 << 31)),
                ..Default::default()
            },
        );
        let ops = instructions(&words);
        assert!(has(&words, Op::ExecutionMode, &[26, vertices.len() as u32]));
        assert_eq!(
            ops.iter().filter(|(op, _)| *op == Op::EmitVertex).count(),
            vertices.len()
        );
        assert_eq!(
            ops.iter().filter(|(op, _)| *op == Op::EndPrimitive).count(),
            1
        );
        let variable = |decoration: u32, value: u32, storage: u32| {
            ops.iter()
                .find(|(op, a)| {
                    *op == Op::Decorate
                        && a[1..] == [decoration, value]
                        && ops
                            .iter()
                            .any(|(op, v)| *op == Op::Variable && v[1] == a[0] && v[2] == storage)
                })
                .unwrap()
                .1[0]
        };
        let constant = |id| {
            ops.iter()
                .find(|(op, a)| *op == Op::Constant && a[1] == id)
                .unwrap()
                .1[2]
        };
        for (decoration, value) in [(11, 0), (30, 2), (30, 31)] {
            let input = variable(decoration, value, 1);
            let output = variable(decoration, value, 3);
            let stores: Vec<_> = ops
                .iter()
                .filter(|(op, a)| *op == Op::Store && a[0] == output)
                .collect();
            assert_eq!(stores.len(), vertices.len());
            for (store, vertex) in stores.iter().zip(&vertices) {
                let value = store.1[1];
                let load = ops
                    .iter()
                    .find(|(op, a)| *op == Op::Load && a[1] == value)
                    .unwrap();
                let pointer = load.1[2];
                let chain = ops
                    .iter()
                    .find(|(op, a)| *op == Op::AccessChain && a[1] == pointer)
                    .unwrap();
                assert_eq!(chain.1.len(), 4); // Whole vec4, one vertex index.
                assert_eq!(chain.1[2], input);
                assert_eq!(constant(chain.1[3]), *vertex);
            }
        }
        // Location 0 was written. Each emission republishes the same private
        // bank, and no input interface is invented for that location.
        assert!(!ops.iter().any(|(op, a)| *op == Op::Decorate
            && a[1..] == [30, 0]
            && ops
                .iter()
                .any(|(op, v)| *op == Op::Variable && v[1] == a[0] && v[2] == 1)));
        let output = variable(30, 0, 3);
        let stores: Vec<_> = ops
            .iter()
            .filter(|(op, a)| *op == Op::Store && a[0] == output)
            .collect();
        assert_eq!(stores.len(), vertices.len());
        let mut bank = None;
        for store in stores {
            let load = ops
                .iter()
                .find(|(op, a)| *op == Op::Load && a[1] == store.1[1])
                .unwrap();
            assert!(ops
                .iter()
                .any(|(op, a)| *op == Op::Variable && a[1] == load.1[2] && a[2] == 6));
            assert_eq!(*bank.get_or_insert(load.1[2]), load.1[2]);
        }
    }
}

#[test]
fn geometry_passthrough_uses_indexed_input_banks_and_return_epilogue() {
    let mut h = header(4);
    h[3] |= 1;
    // A physical ALD declares an indexed generic array. Forwarding slot 2
    // must add its location index after the per-vertex index in that bank.
    let words = validate(
        "passthrough-indexed-input",
        &[
            s2r(3, 0x11),
            al2p(4, 3, 0x80),
            physical(ald(0, 0, 1, 3, false, false), 4),
            ast(0, 0x80, 1, false),
            // Mesa sm50.rs:3099-3111,225-235: @P0 EXIT followed by PT EXIT.
            (0xe300_u64 << 48) | 0xf,
        ],
        &h,
        GraphicsState {
            geometry_input: Some(GeometryInput::TrianglesAdjacency),
            geometry_next_inputs: Some(1 << 2),
            ..Default::default()
        },
    );
    let ops = instructions(&words);
    let input = ops
        .iter()
        .find(|(op, a)| {
            *op == Op::Decorate
                && a[1..] == [30, 0]
                && ops
                    .iter()
                    .any(|(op, v)| *op == Op::Variable && v[1] == a[0] && v[2] == 1)
        })
        .unwrap()
        .1[0];
    let constant = |id| {
        ops.iter()
            .find(|(op, a)| *op == Op::Constant && a[1] == id)
            .unwrap()
            .1[2]
    };
    let chains: Vec<_> = ops
        .iter()
        .filter(|(op, a)| *op == Op::AccessChain && a[2] == input && a.len() == 5)
        .collect();
    assert_eq!(chains.len(), 3);
    for (chain, vertex) in chains.iter().zip([0, 2, 4]) {
        assert_eq!(constant(chain.1[3]), vertex);
        assert_eq!(constant(chain.1[4]), 2);
    }
    assert_eq!(
        ops.iter().filter(|(op, _)| *op == Op::EmitVertex).count(),
        3
    );
    assert_eq!(
        ops.iter().filter(|(op, _)| *op == Op::EndPrimitive).count(),
        1
    );
}

#[test]
fn geometry_passthrough_requires_linked_state_and_rejects_explicit_out() {
    let mut h = header(4);
    h[3] |= 1;
    let state = GraphicsState {
        geometry_input: Some(GeometryInput::Points),
        geometry_next_inputs: Some(0),
        ..Default::default()
    };
    validate("passthrough-position-only", &[], &h, state);
    for xs in [
        vec![out(1)],
        vec![out(2)],
        vec![out(3)],
        vec![out_stream(1, 1)],
    ] {
        assert!(Result::from(translate_with_graphics_state(
            shadowbox::QuietRun,
            &ShaderInput {
                code: &code(&xs),
                program_record: &h,
            },
            &state
        ))
        .is_err());
    }
    let missing = GraphicsState {
        geometry_next_inputs: None,
        ..state
    };
    assert!(Result::from(translate_with_graphics_state(
        shadowbox::QuietRun,
        &ShaderInput {
            code: &code(&[]),
            program_record: &h,
        },
        &missing
    ))
    .is_err());
    // Unknown remains an error even when passthrough could produce a valid
    // primitive without executing the instruction.
    let unknown = 0xffff_u64 << 48 | (7 << 16);
    assert!(matches!(
        decode_word([unknown as u32, (unknown >> 32) as u32], 0).kind,
        InstructionKind::Unknown(ops::Unknown { .. })
    ));
    assert!(Result::from(translate_with_graphics_state(
        shadowbox::QuietRun,
        &ShaderInput {
            code: &code(&[unknown]),
            program_record: &h,
        },
        &state
    ))
    .is_err());
}

#[test]
fn patch_locations_cannot_overlap_an_indexed_generic_bank() {
    // Ryujinx AttributeUsage.cs:65-87 requires disjoint patch/regular I/O.
    for stage in [2, 3] {
        let state = patch_state(PatchLayout::Ryujinx);
        let patch = if stage == 2 {
            ast(0, 0x18, 1, true)
        } else {
            ald(0, 0x18, 1, 255, true, false)
        };
        let generic = if stage == 2 {
            physical(ast(0, 0, 1, false), 3)
        } else {
            physical(ald(0, 0, 1, 255, false, false), 3)
        };
        // patch_state maps slots above regular location 0. The indexed bank
        // occupies all 32 locations, so equality of base locations is insufficient.
        for xs in [[generic, patch], [patch, generic]] {
            assert!(Result::from(translate_with_graphics_state(
                shadowbox::QuietRun,
                &ShaderInput {
                    code: &code(&xs),
                    program_record: &header(stage),
                },
                &state
            ))
            .is_err());
        }
    }
}

#[test]
fn geometry_passthrough_output_reads_and_vector_spans_exclude_forwarding() {
    // Ryujinx Decoder.cs:349-385,410-417 counts AST and ALD.O as output
    // usage, including every vector component and the entire indexed bank.
    for xs in [
        vec![ald(0, 0x8c, 2, 255, false, true), ast(0, 0xa0, 1, false)],
        vec![ast(0, 0x8c, 2, false)],
        vec![
            physical(ald(0, 0, 1, 255, false, true), 3),
            ast(0, 0x70, 1, false),
        ],
        vec![physical(ast(0, 0, 1, false), 3)],
    ] {
        let mut h = header(4);
        h[3] |= 1;
        h[24..40].fill(0);
        h[54..70].fill(0);
        let words = validate(
            "passthrough-output-usage",
            &xs,
            &h,
            GraphicsState {
                geometry_input: Some(GeometryInput::Points),
                geometry_next_inputs: Some(3),
                ..Default::default()
            },
        );
        let ops = instructions(&words);
        // Only Position needs an input interface. Both linked generic slots
        // are output-used by a cross-slot span or an indexed access.
        assert!(!ops.iter().any(|(op, a)| *op == Op::Decorate
            && a[1] == 30
            && ops
                .iter()
                .any(|(op, v)| *op == Op::Variable && v[1] == a[0] && v[2] == 1)));
        assert_eq!(
            ops.iter().filter(|(op, _)| *op == Op::EmitVertex).count(),
            1
        );
    }
}

#[test]
fn physical_generic_bank_checks_the_entire_patch_location_range() {
    // Mesa sph.rs:130-135,166-170: 32 generic vec4s. Ryujinx
    // AttributeUsage.cs:65-87: linked patch locations must be disjoint.
    let code = code(&[physical(ast(0, 0, 1, false), 4), ast(0, 0x18, 1, true)]);
    let h = header(2);
    let input = ShaderInput {
        code: &code,
        program_record: &h,
    };
    for location in [1, 31, 32] {
        let mut state = patch_state(PatchLayout::Ryujinx);
        state.patch_io.as_mut().unwrap().locations[0] = Some(location);
        let result = Result::from(translate_with_graphics_state(
            shadowbox::QuietRun,
            &input,
            &state,
        ));
        assert_eq!(result.is_ok(), location == 32, "patch location {location}");
    }
}

#[test]
fn mixed_streams_publish_only_the_selected_outputs_and_keep_one_builtin() {
    let mut h = header(4);
    h[15] = 1; // Mesa cla097sph.h:68-71: PointList.
    h[16] = 8; // Mesa cla097sph.h:73: maximum total emitted vertices.
    for native in 0..4 {
        let mut xs = vec![
            ald(0, 0x70, 4, 255, false, false),
            ast(0, 0x70, 4, false),
            ast(0, 0x80, 4, false),
            ald(8, 0x60, 1, 255, false, false),
            ast(8, 0x60, 1, false),
        ];
        // Interleave all streams, including repeated emits without AST.
        for s in [3, 0, 2, 1, 3, 1, 0, 2] {
            xs.push(out_stream(3, s));
        }
        let words = validate("four-stream-bank", &xs, &h, stream_state(native));
        let ops = instructions(&words);
        assert!(has(&words, Op::Capability, &[54]));
        let mut output_streams = std::collections::BTreeMap::new();
        let mut locations = std::collections::BTreeMap::new();
        for (op, a) in &ops {
            if *op == Op::Decorate && a[1] == 29 {
                output_streams.insert(a[0], a[2]);
            }
            if *op == Op::Decorate && a[1] == 30 {
                locations.insert(a[0], a[2]);
            }
        }
        // Position and output PrimitiveId exist once, on the native stream.
        for builtin in [0, 7] {
            let builtins = ops
                .iter()
                .filter(|(op, a)| {
                    *op == Op::Decorate
                        && a[1..] == [11, builtin]
                        && output_streams.contains_key(&a[0])
                })
                .collect::<Vec<_>>();
            assert_eq!(builtins.len(), 1);
            assert_eq!(output_streams[&builtins[0].1[0]], u32::from(native));
        }
        assert_eq!(output_streams.len(), 12); // Position, generic0, PrimitiveId x 4.
        for s in 0..4u32 {
            let outputs = output_streams.iter().filter(|(_, stream)| **stream == s);
            let actual = outputs
                .filter_map(|(v, _)| locations.get(v).copied())
                .collect::<std::collections::BTreeSet<_>>();
            let expected = if s == u32::from(native) {
                vec![s * 40]
            } else {
                vec![s * 40, s * 40 + 32, s * 40 + 33]
            };
            assert_eq!(actual, expected.into_iter().collect());
        }
        let constants = ops
            .iter()
            .filter(|(op, _)| *op == Op::Constant)
            .map(|(_, a)| (a[1], a[2]))
            .collect::<std::collections::BTreeMap<_, _>>();
        let mut published = Vec::new();
        let mut emissions = Vec::new();
        let mut cuts = Vec::new();
        let mut sources = std::collections::BTreeMap::new();
        let mut persistent = std::collections::BTreeMap::new();
        for (op, a) in &ops {
            if *op == Op::Load {
                sources.insert(a[1], a[2]);
            }
            if *op == Op::Store && output_streams.contains_key(&a[0]) {
                published.push(a[0]);
                // Each output variable is republished from the same persistent
                // source across repeated emissions. No cross-stream AST bank.
                let source = sources[&a[1]];
                if let Some(previous) = persistent.insert(a[0], source) {
                    assert_eq!(previous, source);
                }
            }
            if *op == Op::EmitStreamVertex {
                let s = constants[&a[0]];
                assert_eq!(published.len(), 3);
                assert!(published.iter().all(|v| output_streams[v] == s));
                published.clear();
                emissions.push(s);
            }
            if *op == Op::EndStreamPrimitive {
                cuts.push(constants[&a[0]]);
            }
        }
        assert_eq!(emissions, [3, 0, 2, 1, 3, 1, 0, 2]);
        assert_eq!(cuts, emissions);
        assert!(published.is_empty());
        // Three shared banks feed all four streams, rather than independent
        // uninitialized per-stream banks. Output ALD still reads these banks.
        assert_eq!(
            persistent
                .values()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            3
        );
    }
}

#[test]
fn mixed_cut_streams_need_no_output_aliases_and_keep_predication() {
    let mut h = header(4);
    h[15] = 1;
    for emitted in 0..4 {
        for cut in 0..4 {
            let words = validate(
                "emit-and-independent-cut",
                &[
                    ast(255, 0x70, 4, false),
                    out_stream(1, emitted),
                    (out_stream(2, cut) & !(15 << 16)) | (8 << 16), // !P0
                ],
                &h,
                GraphicsState {
                    geometry_input: Some(GeometryInput::Points),
                    ..Default::default()
                },
            );
            let ops = instructions(&words);
            if emitted != 0 || cut != 0 {
                assert!(has(&words, Op::Capability, &[54]));
                let constants = ops
                    .iter()
                    .filter(|(op, _)| *op == Op::Constant)
                    .map(|(_, a)| (a[1], a[2]))
                    .collect::<std::collections::BTreeMap<_, _>>();
                assert!(ops
                    .iter()
                    .any(|(op, a)| *op == Op::EmitStreamVertex
                        && constants[&a[0]] == u32::from(emitted)));
                assert!(ops
                    .iter()
                    .any(|(op, a)| *op == Op::EndStreamPrimitive
                        && constants[&a[0]] == u32::from(cut)));
            }
            assert!(has(&words, Op::LogicalNot, &[])); // inversion retained
            assert!(has(&words, Op::BranchConditional, &[])); // guarded cut
                                                              // CUT on another stream never creates an output interface there.
            assert!(ops
                .iter()
                .filter(|(op, a)| *op == Op::Decorate && a[1] == 29)
                .all(|(_, a)| a[2] == u32::from(emitted)));
        }
    }
}

#[test]
fn stream_routing_rejects_missing_aliases_collisions_and_overflow() {
    let mut h = header(4);
    h[15] = 1;
    let bytes = code(&[ast(0, 0x60, 1, false), out_stream(1, 0), out_stream(1, 1)]);
    let input = ShaderInput {
        code: &bytes,
        program_record: &h,
    };
    for change in 0..9 {
        let mut state = stream_state(0);
        let routing = state.geometry_streams.as_mut().unwrap();
        match change {
            0 => routing.locations[1] = None,
            1 => routing.locations[1].as_mut().unwrap().position = None,
            2 => routing.locations[1].as_mut().unwrap().primitive_id = None,
            3 => routing.locations[1].as_mut().unwrap().generic_base = 0,
            4 => routing.locations[1].as_mut().unwrap().position = Some(40),
            5 => routing.locations[1].as_mut().unwrap().position = Some(73),
            6 => routing.locations[0].as_mut().unwrap().position = Some(0),
            7 => routing.builtin_stream = 4,
            8 => routing.locations[1].as_mut().unwrap().generic_base = u32::MAX,
            _ => unreachable!(),
        }
        // Add generic slot 1 to test actual base addition overflow.
        let overflow = code(&[
            ast(0, 0x90, 1, false),
            ast(0, 0x60, 1, false),
            out_stream(1, 0),
            out_stream(1, 1),
        ]);
        let checked = if change == 8 {
            ShaderInput {
                code: &overflow,
                program_record: &h,
            }
        } else {
            ShaderInput { ..input }
        };
        assert!(
            Result::from(translate_with_graphics_state(
                shadowbox::QuietRun,
                &checked,
                &state
            ))
            .is_err(),
            "routing case {change}"
        );
    }
    // Missing routing cannot make a multiple-emission fixture pass.
    assert!(Result::from(translate_with_graphics_state(
        shadowbox::QuietRun,
        &input,
        &GraphicsState {
            geometry_input: Some(GeometryInput::Points),
            ..Default::default()
        }
    ))
    .is_err());
}

#[test]
fn register_out_streams_require_a_real_unconditional_constant_proof() {
    use shadowbox::decoder::decode_stream;
    let mut h = header(4);
    h[15] = 1;
    for s in 0..4 {
        for mode in 1..=3 {
            let mut unpredicated_branches = 0;
            for guard in [7, 0, 8] {
                let raw = (out_register(mode, 254) & !(15 << 16)) | (guard << 16);
                let words = validate(
                    "constant-register-out",
                    &[mov(254, s), raw],
                    &h,
                    GraphicsState {
                        geometry_input: Some(GeometryInput::Points),
                        ..Default::default()
                    },
                );
                let ops = instructions(&words);
                let emits = ops
                    .iter()
                    .filter(|(op, _)| *op == Op::EmitVertex || *op == Op::EmitStreamVertex)
                    .count();
                let cuts = ops
                    .iter()
                    .filter(|(op, _)| *op == Op::EndPrimitive || *op == Op::EndStreamPrimitive)
                    .count();
                assert_eq!(emits, usize::from(mode & 1 != 0));
                assert_eq!(cuts, usize::from(mode & 2 != 0));
                let branches = ops
                    .iter()
                    .filter(|(op, _)| *op == Op::BranchConditional)
                    .count();
                if guard == 7 {
                    unpredicated_branches = branches;
                } else {
                    assert_eq!(branches, unpredicated_branches + 1);
                }
            }
        }
    }
    for preceding in [
        vec![],
        vec![mov(4, 4)],
        vec![mov(4, u32::MAX)],
        vec![(mov(4, 1) & !(15 << 16))], // predicated definition
        vec![mov(4, 1) & !(15 << 12)],   // partial lanes
        vec![mov(4, 1), ald(3, 0x80, 2, 255, false, false)], // overlaps R4
        vec![mov(4, 1), out_register(1, 5)], // unknown may clobber anything
    ] {
        let mut xs = preceding;
        xs.push(out_register(1, 4));
        let bytes = code(&xs);
        let decoded = decode_stream(
            shadowbox::QuietRun,
            &bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| u32::from_le_bytes(*w))
                .collect::<Vec<_>>(),
        );
        assert!(decoded.iter().any(|x| matches!(
            x.kind,
            InstructionKind::GeometryOutDynamic(ops::GeometryOutDynamic {
                stream: shadowbox::ir::Source::Register(shadowbox::ir::Register(4)),
                ..
            })
        )));
        assert!(Result::from(translate_with_graphics_state(
            shadowbox::QuietRun,
            &ShaderInput {
                code: &bytes,
                program_record: &h
            },
            &GraphicsState {
                geometry_input: Some(GeometryInput::Points),
                ..Default::default()
            }
        ))
        .is_err());
    }
}

#[test]
fn cbuf_out_streams_use_only_the_supplied_immutable_binding_one_snapshot() {
    use shadowbox::{
        decoder::decode_stream_with_constant_buffer, emitter::emit_with_graphics_state,
    };
    let mut h = header(4);
    h[15] = 1;
    let parsed = parse(
        &h.as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    for s in 0..4 {
        for mode in 1..=3 {
            for guard in [7, 0, 8] {
                let raw = (out_cbuf(mode, 1, 12) & !(15 << 16)) | (guard << 16);
                assert!(matches!(
                    decode_word([raw as u32, (raw >> 32) as u32], 0).kind,
                    InstructionKind::GeometryOutDynamic(ops::GeometryOutDynamic { .. })
                ));
                let bytes = code(&[raw]);
                let words = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|w| u32::from_le_bytes(*w))
                    .collect::<Vec<_>>();
                let decoded = decode_stream_with_constant_buffer(
                    shadowbox::QuietRun,
                    &words,
                    &[99, 99, 99, s],
                );
                assert_eq!(
                    decoded[0].kind,
                    InstructionKind::GeometryOut(ops::GeometryOut {
                        emit: mode & 1 != 0,
                        cut: mode & 2 != 0,
                        stream: s as u8
                    })
                );
                let module = emit_with_graphics_state(
                    &decoded,
                    Some(&parsed),
                    &GraphicsState {
                        geometry_input: Some(GeometryInput::Points),
                        ..Default::default()
                    },
                )
                .unwrap();
                let result = spirv_validation::validate(&module.words);
                assert!(
                    result.status.success(),
                    "{}",
                    String::from_utf8_lossy(&result.stderr)
                );
            }
        }
    }
    for (binding, offset, snapshot) in [
        (0, 0, vec![1]),
        (2, 0, vec![1]),
        (1, 4, vec![1]),
        (1, 0, vec![4]),
        (1, 0, vec![u32::MAX]),
    ] {
        let bytes = code(&[out_cbuf(1, binding, offset)]);
        let decoded = decode_stream_with_constant_buffer(
            shadowbox::QuietRun,
            &bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| u32::from_le_bytes(*w))
                .collect::<Vec<_>>(),
            &snapshot,
        );
        assert!(matches!(
            decoded[0].kind,
            InstructionKind::GeometryOutDynamic(ops::GeometryOutDynamic { .. })
        ));
    }
}

#[test]
fn out_specialization_keeps_the_complete_continuation_and_block_entries() {
    use shadowbox::decoder::decode_stream;
    let decode = |xs: &[u64]| {
        let bytes = code(xs);
        decode_stream(
            shadowbox::QuietRun,
            &bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| u32::from_le_bytes(*w))
                .collect::<Vec<_>>(),
        )
    };
    let mut h = header(4);
    h[15] = 1;
    let xs = [
        mov(4, 0),
        out_register(1, 4),
        ast(255, 0x70, 4, false),
        out_register(3, 4),
        out_stream(2, 3),
    ];
    let decoded = decode(&xs);
    assert_eq!(decoded.len(), xs.len() + 1); // complete continuation plus EXIT
    assert!(matches!(
        decoded[3].kind,
        InstructionKind::GeometryOut(ops::GeometryOut {
            emit: true,
            cut: true,
            stream: 0
        })
    ));
    let words = validate(
        "out-complete-continuation",
        &xs,
        &h,
        GraphicsState {
            geometry_input: Some(GeometryInput::Points),
            ..Default::default()
        },
    );
    assert_eq!(
        instructions(&words)
            .iter()
            .filter(|(op, _)| *op == Op::EmitStreamVertex)
            .count(),
        2
    );
    // Unknown following a now-supported instruction must still reject.
    let bytes = code(&[mov(4, 0), out_register(1, 4), out(1) & !255]);
    assert!(Result::from(translate_with_graphics_state(
        shadowbox::QuietRun,
        &ShaderInput {
            code: &bytes,
            program_record: &h
        },
        &GraphicsState {
            geometry_input: Some(GeometryInput::Points),
            ..Default::default()
        }
    ))
    .is_err());
    // Mesa sm50.rs:3020-3028: BRA relative to next slot, CC.T. Splitting
    // at the destination prevents a preceding MOV being assumed at entry.
    let branch = (0xe240u64 << 48) | (7 << 16) | 15;
    let decoded = decode(&[mov(4, 1), branch, out_register(1, 4)]);
    assert!(matches!(
        decoded[2].kind,
        InstructionKind::GeometryOutDynamic(ops::GeometryOutDynamic { .. })
    ));
    // Conditional branch from slot16 skips slot24 and enters slot40.
    let branch = (branch & !(15 << 16)) | (16 << 20);
    let decoded = decode(&[mov(4, 1), branch, mov(4, 2), out_register(1, 4)]);
    assert!(matches!(
        decoded[3].kind,
        InstructionKind::GeometryOutDynamic(ops::GeometryOutDynamic { .. })
    ));
    // Even a provable stream may not discard a handle, destination or modifier.
    for raw in [
        out_register(1, 4) & !255,
        out_register(1, 4) & !(255 << 8),
        out_register(0, 4),
        out_register(1, 4) | (1 << 28),
        out_register(1, 4) | (1 << 38),
        out_register(1, 4) | (1 << 41),
        out_register(1, 4) | (1 << 47),
    ] {
        let decoded = decode(&[mov(4, 1), raw]);
        assert!(matches!(
            decoded[1].kind,
            InstructionKind::Unknown(ops::Unknown { .. })
        ));
    }
    // AST has implicit output storage: Mesa's older emitter:2538-2545;
    // Ryujinx InstDecoders.cs:890-902 and InstEmitAttribute.cs:105-141.
    for patch in [false, true] {
        let raw = ast(0, if patch { 0x18 } else { 0x80 }, 1, patch);
        let with_bit = decode_word([raw as u32, (raw >> 32) as u32], 0).kind;
        let clear = raw & !(1 << 32);
        assert_eq!(
            decode_word([clear as u32, (clear >> 32) as u32], 0).kind,
            with_bit
        );
        assert!(matches!(
            with_bit,
            InstructionKind::AttributeStore(ops::AttributeStore { .. })
        ));
    }
}

#[test]
fn constant_buffer_mov_and_register_copies_prove_the_same_stream() {
    use shadowbox::{
        decoder::decode_stream_with_constant_buffer, emitter::emit_with_graphics_state,
    };
    let mut h = header(4);
    h[15] = 1;
    let parsed = parse(
        &h.as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes(*w))
            .collect::<Vec<_>>(),
    )
    .unwrap();
    // MOV.CBUF: Mesa sm50.rs:1943-1946,361-372. MOV.REG:1933-1936.
    let mov_cbuf = (0x4c98u64 << 48) | (15 << 39) | (1 << 34) | (3 << 20) | (7 << 16) | 4;
    let copy = (0x5c98u64 << 48) | (15 << 39) | (4 << 20) | (7 << 16) | 5;
    for s in 0..4 {
        let bytes = code(&[mov_cbuf, copy, out_register(3, 5), out_cbuf(2, 1, 12)]);
        let decoded = decode_stream_with_constant_buffer(
            shadowbox::QuietRun,
            &bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| u32::from_le_bytes(*w))
                .collect::<Vec<_>>(),
            &[99, 99, 99, s],
        );
        assert_eq!(decoded.len(), 5);
        assert_eq!(
            decoded[2].kind,
            InstructionKind::GeometryOut(ops::GeometryOut {
                emit: true,
                cut: true,
                stream: s as u8
            })
        );
        let module = emit_with_graphics_state(
            &decoded,
            Some(&parsed),
            &GraphicsState {
                geometry_input: Some(GeometryInput::Points),
                ..Default::default()
            },
        )
        .unwrap();
        let result = spirv_validation::validate(&module.words);
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
}

#[test]
fn out_forms_are_geometry_only_and_stream_state_is_stage_specific() {
    for stage in [0, 1, 2, 3, 5] {
        let mut state = stream_state(1);
        state.tessellation = Some(TessellationState {
            domain: TessellationDomain::Triangles,
            spacing: TessellationSpacing::Equal,
            clockwise: false,
        });
        // Geometry linkage must not change another stage's module or cache key.
        // Ryujinx SpirvGenerator.cs:190-261 selects modes by stage.
        let empty = validate("non-geometry-stream-state", &[], &header(stage), state);
        state.geometry_streams = None;
        let without = validate("non-geometry-no-stream-state", &[], &header(stage), state);
        assert_eq!(empty, without);
        for s in 0..4 {
            for raw in [out_stream(1, s), out_register(3, 4)] {
                let bytes = code(&[mov(4, u32::from(s)), raw]);
                assert!(Result::from(translate_with_graphics_state(
                    shadowbox::QuietRun,
                    &ShaderInput {
                        code: &bytes,
                        program_record: &header(stage)
                    },
                    &state
                ))
                .is_err());
            }
        }
    }
}

#[test]
fn indexed_geometry_output_routes_the_whole_bank_without_location_collisions() {
    let mut h = header(4);
    h[15] = 1;
    let mut state = stream_state(0);
    // Runtime address prevents AL2P resolution to a single location. Each
    // stream retains all 32 vec4s (Mesa sph.rs:130-135,166-170).
    let xs = [
        s2r(4, 0),
        physical(ast(0, 0, 1, false), 4),
        out_stream(1, 0),
        out_stream(1, 1),
    ];
    let words = validate("indexed-multiple-stream-bank", &xs, &h, state);
    assert!(has(&words, Op::Decorate, &[30, 0]));
    assert!(has(&words, Op::Decorate, &[30, 40]));
    // The alias overlaps the indexed bank's last location, not its base.
    state.geometry_streams.as_mut().unwrap().locations[1]
        .as_mut()
        .unwrap()
        .position = Some(71);
    assert!(Result::from(translate_with_graphics_state(
        shadowbox::QuietRun,
        &ShaderInput {
            code: &code(&xs),
            program_record: &h
        },
        &state
    ))
    .is_err());
    // A later stream's base overlaps the first bank's interior.
    state.geometry_streams.as_mut().unwrap().locations[1]
        .as_mut()
        .unwrap()
        .position = Some(72);
    state.geometry_streams.as_mut().unwrap().locations[1]
        .as_mut()
        .unwrap()
        .generic_base = 31;
    assert!(Result::from(translate_with_graphics_state(
        shadowbox::QuietRun,
        &ShaderInput {
            code: &code(&xs),
            program_record: &h
        },
        &state
    ))
    .is_err());
}

#[test]
fn legacy_ast_stores_the_same_values_in_vertex_geometry_and_patch_outputs() {
    // AST count and P: Mesa nv50_ir_emit_gm107.cpp:2541-2545;
    // output semantics: Ryujinx InstEmitAttribute.cs:105-141.
    // Compare whole validated modules to cover real stores and output loads,
    // not just decoder acceptance. Constant source bits exercise each lane.
    for stage in [1, 2, 4] {
        for components in 1..=4 {
            for patch in [false, true] {
                if patch && stage != 2 {
                    continue;
                }
                let address = if patch { 0 } else { 0x80 };
                let store = ast(0, address, components, patch);
                let mut legacy = vec![
                    mov(0, 0x3f800000),
                    mov(1, 0x80000000),
                    mov(2, 0x7fc12345),
                    mov(3, 0x01234567),
                    store & !(1 << 32),
                    ald(8, address, components, 255, patch, true),
                    ast(8, 0x70, components, false),
                ];
                if stage == 4 {
                    legacy.push(out(1));
                }
                let mut current = legacy.clone();
                current[4] = store;
                let state = GraphicsState {
                    geometry_input: (stage == 4).then_some(GeometryInput::Points),
                    ..Default::default()
                };
                let h = header(stage);
                let legacy_words = validate("legacy-ast", &legacy, &h, state);
                let current_words = validate("nak-ast", &current, &h, state);
                assert_eq!(
                    legacy_words, current_words,
                    "stage={stage} count={components} patch={patch}"
                );
                assert!(
                    has(&legacy_words, Op::Store, &[]),
                    "AST must produce stores"
                );
            }
        }
    }
}

#[test]
fn unmapped_output_reads_load_zero() {
    // Ryujinx AttributeMap.cs:63,129-133: no attribute below 0x60, so ALD.O
    // there loads 0 in every stage. 0x58 + 2 components ends at 0x60.
    let read = ald(1, 0x58, 2, 255, false, true);
    assert!(matches!(
        decode_word([read as u32, (read >> 32) as u32], 0).kind,
        InstructionKind::UnmappedAttributeLoad(ops::UnmappedAttributeLoad {
            address: 0x58,
            components: 2,
            ..
        })
    ));
    let zero_stores = |words: &[u32]| {
        let ops = instructions(words);
        let zeros: Vec<u32> = ops
            .iter()
            .filter(|(op, a)| *op == Op::Constant && a.len() == 3 && a[2] == 0)
            .map(|(_, a)| a[1])
            .collect();
        ops.iter()
            .filter(|(op, a)| *op == Op::Store && zeros.contains(&a[1]))
            .count()
    };
    let triangles = GraphicsState {
        geometry_input: Some(GeometryInput::Triangles),
        ..Default::default()
    };
    for (stage, state) in [
        (1, GraphicsState::default()),
        (2, GraphicsState::default()),
        (4, triangles),
    ] {
        let mut base = vec![mov(1, 5), mov(2, 5), ast(1, 0x80, 2, false)];
        if stage == 4 {
            base.push(out(3));
        }
        let mut with_read = base.clone();
        with_read.insert(2, read);
        let before = validate(
            &format!("unmapped-base-{stage}"),
            &base,
            &header(stage),
            state,
        );
        let after = validate(
            &format!("unmapped-read-{stage}"),
            &with_read,
            &header(stage),
            state,
        );
        assert_eq!(
            zero_stores(&after),
            zero_stores(&before) + 2,
            "stage {stage}"
        );
    }
}

#[test]
fn attribute_lanes_without_a_stage_attribute_load_zero() {
    // Ryujinx AttributeMap.cs:63-81,136-141: a mapped attribute outside its
    // stage mask logs and loads 0, lane by lane (InstEmitAttribute.cs:34-55).
    // PrimitiveId is an output only in geometry and an input only in
    // TCS/TES/geometry/fragment; Position is no vertex input.
    let zero_stores = |words: &[u32]| {
        let ops = instructions(words);
        let zeros: Vec<u32> = ops
            .iter()
            .filter(|(op, a)| *op == Op::Constant && a.len() == 3 && a[2] == 0)
            .map(|(_, a)| a[1])
            .collect();
        ops.iter()
            .filter(|(op, a)| *op == Op::Store && zeros.contains(&a[1]))
            .count()
    };
    let triangles = GraphicsState {
        geometry_input: Some(GeometryInput::Triangles),
        ..Default::default()
    };
    let tessellation = GraphicsState {
        tessellation: Some(TessellationState {
            domain: TessellationDomain::Triangles,
            spacing: TessellationSpacing::Equal,
            clockwise: false,
        }),
        ..Default::default()
    };
    for (stage, state, read, zeros) in [
        (
            1,
            GraphicsState::default(),
            ald(1, 0x5c, 2, 255, false, true),
            2,
        ),
        (
            1,
            GraphicsState::default(),
            ald(1, 0x60, 1, 255, false, true),
            1,
        ),
        (
            1,
            GraphicsState::default(),
            ald(1, 0x60, 1, 255, false, false),
            1,
        ),
        (
            1,
            GraphicsState::default(),
            ald(1, 0x70, 2, 255, false, false),
            2,
        ),
        (
            2,
            GraphicsState::default(),
            ald(1, 0x5c, 2, 255, false, true),
            2,
        ),
        (3, tessellation, ald(1, 0x60, 1, 255, false, true), 1),
        (4, triangles, ald(1, 0x5c, 2, 255, false, true), 1),
    ] {
        let mut base = vec![mov(1, 5), mov(2, 5), ast(1, 0x80, 2, false)];
        if stage == 4 {
            base.push(out(3));
        }
        let mut with_read = base.clone();
        with_read.insert(2, read);
        let before = validate(&format!("lane-base-{stage}"), &base, &header(stage), state);
        let after = validate(
            &format!("lane-read-{stage}-{read:016x}"),
            &with_read,
            &header(stage),
            state,
        );
        assert_eq!(
            zero_stores(&after),
            zero_stores(&before) + zeros,
            "stage {stage} {read:016x}"
        );
    }
    // Vertex Layer and PointSize outputs exist in Ryujinx but have no
    // declaration here: an error, never a panic.
    for read in [
        ald(1, 0x64, 1, 255, false, true),
        ald(1, 0x6c, 1, 255, false, true),
        ald(1, 0x68, 4, 255, false, true),
    ] {
        assert!(Result::from(translate_with_graphics_state(
            shadowbox::QuietRun,
            &ShaderInput {
                code: &code(&[read]),
                program_record: &header(1),
            },
            &GraphicsState::default()
        ))
        .is_err());
    }
}
