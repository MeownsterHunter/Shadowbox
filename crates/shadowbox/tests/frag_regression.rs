#![cfg(feature = "gpu-check")]
#[path = "support/fragment.rs"]
mod fragment;
#[path = "support/fragment_program.rs"]
mod program;

#[test]
fn novena_fragment_mov32i_exports_requested_color() {
    for green in [0.25_f32, 0.75] {
        let color = [1.0_f32, green, 0.5, 1.0];
        let instructions = color
            .into_iter()
            .enumerate()
            .map(|(register, value)| program::mov(register as u8, value.to_bits(), 15, false))
            .collect::<Vec<_>>();
        // Both Novena's conditional-false tail and a real unconditional EXIT.
        for cc in [0, 15] {
            fragment::assert_pixels(&program::modules(&instructions, cc), color);
        }
    }
}

#[test]
fn disabled_fragment_mov32i_preserves_existing_color() {
    let color = [1.0_f32, 0.25, 0.5, 1.0];
    let mut instructions = color
        .into_iter()
        .enumerate()
        .map(|(register, value)| program::mov(register as u8, value.to_bits(), 15, false))
        .collect::<Vec<_>>();
    for (mask, inverted) in [(0, false), (15, true)] {
        for register in 0..4 {
            instructions.push(program::mov(register, 0.75_f32.to_bits(), mask, inverted));
        }
    }
    fragment::assert_pixels(&program::modules(&instructions, 15), color);
    let disabled = color
        .into_iter()
        .enumerate()
        .map(|(register, value)| program::mov(register as u8, value.to_bits(), 0, false))
        .collect::<Vec<_>>();
    fragment::assert_pixels(&program::modules(&disabled, 15), [0.0; 4]);
}
