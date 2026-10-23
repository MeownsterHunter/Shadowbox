//! Field-complete family representatives shared by validation and speed checks.
use super::families::Family;
use shadowbox::interface::{
    GeometryInput, GraphicsState, PatchIo, PatchLayout, TessellationDomain, TessellationSpacing,
    TessellationState,
};
fn word(opcode: u16, dst: u8, src: u8) -> [u32; 2] {
    // gm107.c:1990,1994,408-411: dst 0..7, lhs 8..15, PT 16..19.
    let raw = (opcode as u64) << 48 | 7_u64 << 16 | (src as u64) << 8 | dst as u64;
    // Mesa sm50.rs:3106-3110: every synthetic EXIT terminator uses CC.T.
    let raw = if opcode == 0xe300 { raw | 0xf } else { raw };
    [raw as u32, (raw >> 32) as u32]
}

pub fn program(family: Family) -> Vec<u8> {
    let opcode = family.opcode;
    // decoder.rs:1106-1112: two control words followed by three instruction
    // slots.  The next bundle supplies a terminator in the same layout.
    let mut words = vec![0; 16];
    // CSETP has condition bits at 8..13 rather than an A register
    // (envytools gm107.c:2058,1647-1680). Keep its supplied condition intact.
    let mut first = word(
        opcode,
        1,
        if matches!(opcode, 0xef98 | 0x50a0 | 0xf0c8) {
            0
        } else {
            2
        },
    );
    if matches!(
        opcode & 0xfef8,
        0x5ce0 | 0x4ce0 | 0x38e0 | 0x5cb0 | 0x4cb0 | 0x38b0 | 0x5cb8 | 0x4cb8 | 0x38b8
    ) {
        // Mesa sm50.rs:1831-1833,1873-1875,1913-1916: 32-bit sizes.
        if family.name.starts_with("I2F_F64") {
            // Mesa sm50.rs:1873-1875: explicit destination/source sizes.
            first[0] &= !0x2f00;
        } else {
            first[0] |= 0xa00;
        }
    }
    let mut raw = u64::from(first[0]) | (u64::from(first[1]) << 32);
    if family.name.starts_with("BRA_EXACT_")
        || family.name.starts_with("EXIT_EXACT_")
        || family.name == "SSY_SYNC_EXACT"
    {
        raw &= !0xff;
    }
    let raw = raw | family.fields;
    let first = [raw as u32, (raw >> 32) as u32];
    words[2..4].copy_from_slice(&first);
    // Mesa sm50.rs:1938-1941: the setup literal must write all lanes.
    let mut setup = word(0x0100, 2, 0);
    setup[0] |= 15 << 12;
    words[4..6].copy_from_slice(&setup);
    if opcode == 0xdf48 && family.required {
        // A matching 3D bound TEX supplies all three axes requested by TXQ.
        // Encoding: Mesa sm50.rs:2170-2197; shape: Ryujinx
        // InstEmitTexture.cs:1116-1119. Keep the standalone Unknown case too.
        let typed_tex = 0xc038_u64 << 48
            | 7 << 16
            | 1 << 55
            | 4 << 28
            | 15 << 31
            | ((raw >> 36) & 0x1fff) << 36
            | 255 << 20
            | 16;
        words[4..6].copy_from_slice(&[typed_tex as u32, (typed_tex >> 32) as u32]);
    }
    words[6..8].copy_from_slice(&word(0xe300, 15, 0));
    words[10..12].copy_from_slice(&word(0xe300, 15, 0));
    // Unused slots use Mesa's encoded NOP (sm50.rs:3178-3184),
    // never an arbitrary zero word that would become Unknown.
    words[12..14].copy_from_slice(&word(0x50b0, 0, 0));
    words[14..16].copy_from_slice(&word(0x50b0, 0, 0));
    words.into_iter().flat_map(u32::to_le_bytes).collect()
}

pub fn compute_header() -> Vec<u8> {
    // headers.rs:59-69 requires a 128-byte record; zero bits 10..13 select compute.
    vec![0; 128]
}

pub fn family_context(family: Family) -> (Vec<u8>, GraphicsState<'static>) {
    // Stage masks: Ryujinx Instructions/AttributeMap.cs:63-92.
    let stage = match family.opcode {
        0xfbe0 | 0xf6e0 | 0xebe0 => 4,
        0xeff0 | 0xeff1 => 2,
        0xefd8 | 0xefd9 if family.fields & (1 << 32) != 0 => 2,
        0xefd8 | 0xefd9 => 3,
        // TMML derivatives and IPA inputs are fragment-only. Ryujinx
        // InstEmitAttribute.cs:167-175; Spirv/Instructions.cs:859-898.
        0xdf58 | 0xe04b | 0xe06b => 5,
        _ => 0,
    };
    let mut header = compute_header();
    header[1] = stage << 2;
    // Perspective component map: headers.rs:84-101.
    if stage == 5 {
        header[24] = 2;
    }
    // Mesa cla097sph.h:66-73: output control points, topology, vertices.
    header[11] = 3;
    header[15] = 7;
    header[16] = 3;
    let state = GraphicsState {
        storage_buffers: &[],
        constant_buffer1: None,
        texture_bindings: &[],
        texture_arrays: &[],
        geometry_input: Some(GeometryInput::Triangles),
        tessellation: Some(TessellationState {
            domain: TessellationDomain::Triangles,
            spacing: TessellationSpacing::Equal,
            clockwise: false,
        }),
        geometry_next_inputs: None,
        geometry_streams: None,
        patch_io: Some(PatchIo {
            layout: PatchLayout::Ryujinx,
            locations: std::array::from_fn(|slot| Some(slot as u32)),
        }),
    };
    (header, state)
}
