//! Descriptor global memory must be backed by a proven cbuf pair.
use shadowbox::spirv::Op;
use shadowbox::{
    decoder::decode_stream,
    emitter::{emit_unoptimized_with_graphics_state, SpirvModule},
    headers,
    interface::{GraphicsState, ShaderInput, StorageBufferBinding},
    CacheContext,
};
use std::io::Write;
use std::process::{Command, Stdio};

/// A program's identity under `context`, through [`shadowbox::Program`].
fn identity(
    input: &shadowbox::interface::ShaderInput<'_>,
    context: shadowbox::CacheContext,
) -> Result<String, shadowbox::interface::Stopped> {
    Ok(shadowbox::Program::parse(input, context.graphics_state)?
        .identity(context.generation, context.options))
}
const PT: u64 = 7 << 16;
fn cb(dst: u8, bank: u8, word: u16) -> u64 {
    // Mesa nak/sm50.rs:1943-1951,361-368: MOV cbuf, all quad positions.
    0x4c98 << 48 | 15 << 39 | u64::from(bank) << 34 | u64::from(word) << 20 | PT | u64::from(dst)
}
fn load(dst: u8) -> u64 {
    // Mesa sm50.rs:2386-2431,2603-2614: LDG.E.B32 Rdst,[R20].
    0xeed4 << 48 | 1 << 45 | PT | 20 << 8 | u64::from(dst)
}
fn code(mut xs: Vec<u64>) -> Vec<u32> {
    // Mesa sm50.rs:3178-3187,3418-3468: EXIT, scheduling bundles.
    xs.push(0xe300 << 48 | PT | 15);
    while !xs.len().is_multiple_of(3) {
        xs.push(0x50b0 << 48 | PT | 15 << 8);
    }
    xs.chunks(3)
        .flat_map(|s| [0, s[0], s[1], s[2]])
        .flat_map(|w| [w as u32, (w >> 32) as u32])
        .collect()
}
fn bindings() -> [StorageBufferBinding; 1] {
    [StorageBufferBinding {
        bank: 2,
        offset: 76,
        binding: 9,
    }]
}
fn emit(
    xs: &[u64],
    bs: &[StorageBufferBinding],
) -> Result<SpirvModule, shadowbox::interface::Stopped> {
    emit_unoptimized_with_graphics_state(
        &decode_stream(shadowbox::QuietRun, &code(xs.to_vec())),
        Some(&headers::parse(&[0; 32]).unwrap()),
        &GraphicsState {
            storage_buffers: bs,
            ..Default::default()
        },
    )
}
fn validate(module: &SpirvModule) {
    let mut child = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.2"])
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(
            &module
                .words
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
}
#[test]
fn proven_pair_uses_logical_storage_and_preserves_load() {
    let module = emit(&[cb(20, 2, 19), cb(21, 2, 20), load(50)], &bindings()).unwrap();
    validate(&module);
    let mut at = 5;
    let mut storage = false;
    let mut descriptor_load = false;
    while at < module.words.len() {
        let n = (module.words[at] >> 16) as usize;
        let args = &module.words[at + 1..at + n];
        match Op::of(module.words[at]) {
            Op::MemoryModel => assert_eq!(args[0], 0, "must not use physical addresses"),
            Op::Variable => storage |= args[2] == 12,
            Op::AccessChain => descriptor_load = true,
            _ => {}
        }
        at += n;
    }
    assert!(storage && descriptor_load);
}
#[test]
fn both_address_words_must_reach_from_adjacent_cbuf_words() {
    for (bank, word) in [(3, 20), (2, 21), (2, 18)] {
        assert!(emit(&[cb(20, 2, 19), cb(21, bank, word), load(50)], &bindings()).is_err());
    }
    // A conditional write creates a conflicting definition on one path.
    assert!(emit(
        &[cb(20, 2, 19), cb(21, 2, 20), cb(20, 2, 21) & !PT, load(50)],
        &bindings()
    )
    .is_err());
    // An aliasing load invalidates the base for subsequent accesses.
    assert!(emit(
        &[cb(20, 2, 19), cb(21, 2, 20), load(20), load(50)],
        &bindings()
    )
    .is_err());
    validate(&emit(&[cb(20, 2, 19), cb(21, 2, 20), load(20)], &bindings()).unwrap());
}
#[test]
fn unsupported_descriptor_shapes_keep_failing() {
    for instruction in [
        load(50) & !(1 << 45),
        load(50) | (1 << 48),
        load(50) | (1 << 20),
        load(50) | (0x80_0000 << 20),
    ] {
        assert!(emit(&[cb(20, 2, 19), cb(21, 2, 20), instruction], &bindings()).is_err());
    }
    let mut other = bindings();
    other[0].offset += 4;
    assert!(emit(&[cb(20, 2, 19), cb(21, 2, 20), load(50)], &other).is_err());
    let duplicates = [bindings()[0], bindings()[0]];
    assert!(emit(&[cb(20, 2, 19), cb(21, 2, 20), load(50)], &duplicates).is_err());
}
#[test]
fn rz_memory_operations_still_validate() {
    for store in [false, true] {
        // Mesa sm50.rs:2659-2669: STG.E.B32. RZ must remain a real access.
        let x = if store {
            0xeedc << 48 | 1 << 45 | PT | 20 << 8 | 255
        } else {
            load(255)
        };
        validate(&emit(&[cb(20, 2, 19), cb(21, 2, 20), x], &bindings()).unwrap());
    }
}
#[test]
fn descriptor_source_and_binding_are_in_cache_identity() {
    let words = code(vec![cb(20, 2, 19), cb(21, 2, 20), load(50)]);
    let bytes = words
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    let input = ShaderInput {
        code: &bytes,
        program_record: &[0; 128],
    };
    let key = |bs: &[StorageBufferBinding]| {
        identity(
            &input,
            CacheContext {
                graphics_state: GraphicsState {
                    storage_buffers: bs,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap()
    };
    let original = bindings();
    for field in 0..3 {
        let mut other = original;
        match field {
            0 => other[0].bank += 1,
            1 => other[0].offset += 4,
            _ => other[0].binding += 1,
        }
        assert_ne!(key(&original), key(&other));
    }
    assert_ne!(key(&original), key(&[]));
}
