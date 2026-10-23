// Independent CPU oracle for graphics-stage words, the stage-test sibling of
// reference.rs (whose compute dependencies a draw test cannot load).
// Dispatch and operands come from encoding bits, never from the decoded
// InstructionKind or emitted SPIR-V. No upstream implementation copied.

fn field(raw: u64, start: u32, width: u32) -> u32 {
    ((raw >> start) & ((1_u64 << width) - 1)) as u32
}

/// IPA with SAT51 for one fragment, read from the raw word alone.
/// envytools gm107.c:1953 (address28..37, B20..27, SAT51, mode54..55 via
/// tabe000_0 :859-864); Ryujinx InstEmitAttribute.cs:146-218: Position
/// 0x70..0x7c reads FragCoord, a perspective generic is multiplied by
/// FragCoord.w (lines 167-175), MUL multiplies by B (208-212), then SAT
/// clamps to [0, 1] (215). `generic` is the host-interpolated input vector
/// of the addressed location; `registers` holds the B register values.
pub fn ipa_saturate_fragment(
    raw: u64,
    generic: [f32; 4],
    perspective: bool,
    frag_coord: [f32; 4],
    registers: &[u32; 256],
) -> u32 {
    let address = field(raw, 28, 10);
    let mode = field(raw, 54, 2);
    let lane = ((address >> 2) & 3) as usize;
    let mut value = if (0x70..0x80).contains(&address) {
        frag_coord[lane]
    } else {
        let value = generic[lane];
        if perspective && mode != 2 {
            value * frag_coord[3]
        } else {
            value
        }
    };
    if mode == 1 {
        value *= f32::from_bits(registers[field(raw, 20, 8) as usize]);
    }
    if field(raw, 51, 1) != 0 {
        value = if value.is_nan() {
            0.
        } else {
            value.clamp(0., 1.)
        };
    }
    value.to_bits()
}
