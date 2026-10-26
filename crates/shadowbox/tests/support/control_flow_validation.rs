use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_stream,
    emitter::emit_unoptimized_with_graphics_state,
    headers,
    interface::{GraphicsState, ShaderInput},
    ir::InstructionKind,
    translate_with_graphics_state,
};

pub fn check(name: &str, code: &[u8], header: &[u8], state: &GraphicsState) {
    let words = |bytes: &[u8]| {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect::<Vec<_>>()
    };
    let instructions = decode_stream(shadowbox::QuietRun, &words(code));
    assert!(
        instructions.iter().all(|i| !matches!(
            i.kind,
            InstructionKind::Unknown(ops::Unknown { .. })
                | InstructionKind::PopJump(ops::PopJump { .. })
                | InstructionKind::Schedule(ops::Schedule { .. })
                | InstructionKind::PushTarget(ops::PushTarget { .. })
        )),
        "{name}: every synthetic instruction must be implemented"
    );
    let header_parsed = headers::parse(&words(header)).unwrap();
    let raw = emit_unoptimized_with_graphics_state(&instructions, Some(&header_parsed), state)
        .unwrap_or_else(|e| panic!("{name}: raw emission: {e}"));
    let optimized = Result::from(translate_with_graphics_state(
        shadowbox::QuietRun,
        &ShaderInput {
            code,
            program_record: header,
        },
        state,
    ))
    .unwrap_or_else(|e| panic!("{name}: optimized translation: {e}"));
    for (variant, words) in [("raw", &raw.words), ("optimized", &optimized.spirv)] {
        let result = super::spirv_validation::validate(words);
        assert!(
            result.status.success(),
            "{name} {variant}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
    }
}
