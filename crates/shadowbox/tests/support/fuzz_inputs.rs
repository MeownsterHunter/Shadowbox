//! Deterministic inputs shared by SPIR-V validation and output identity gates.
pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

// Same inputs and order as fuzz_validation.rs, including default seed and
// eight samples. Mesa nak/sph.rs:66-79,138-142: two bits per fragment
// attribute from bit 192, perspective=2. Stage type field and values:
// Mesa src/nouveau/headers/nvidia/classes/cla097sph.h:37-42.
// All-zero headers use the existing compute convention (headers.rs:65).
pub fn fuzz_headers() -> [Vec<u8>; 6] {
    let graphics = |stage: u8| {
        let mut header = vec![0_u8; 128];
        header[1] = stage << 2;
        header[24..32].fill(0xaa);
        // Mesa cla097sph.h:66-73: invocation/control-point count,
        // geometry output topology and maximum output vertices.
        header[11] = 3;
        header[15] = 7;
        header[16] = 3;
        header
    };
    [
        vec![0; 128],
        graphics(5),
        graphics(1),
        graphics(4),
        graphics(2),
        graphics(3),
    ]
}

pub fn fuzz_program(raw: u64) -> Vec<u8> {
    // Mesa nak/sm50.rs:3101-3110 EXIT; :3183-3187 OpNop;
    // :225-235 PT predicate; :3418-3428,3438-3468 bundle layout.
    // CC.T at0..4 terminates padding after conditional EXIT is implemented
    // (Mesa sm50.rs:3106-3110); retain the random tested word unchanged.
    let exit = 0xe300_u64 << 48 | 7 << 16 | 15;
    let nop = 0x50b0_u64 << 48 | 7 << 16;
    [0, raw, exit, exit, 0, exit, nop, nop]
        .into_iter()
        .flat_map(u64::to_le_bytes)
        .collect()
}

// Ryujinx ShaderDefinitions.cs:18-21,158-160 and
// CodeGen/Spirv/SpirvGenerator.cs:194-261: driver-provided stage state.
pub fn graphics_state() -> shadowbox::interface::GraphicsState<'static> {
    use shadowbox::interface::*;
    GraphicsState {
        // Patch slots: Ryujinx AttributeMap.cs:90-92. Adapted from Ryujinx
        // Translation/AttributeUsage.cs:63-87 (MIT): patch locations take
        // the regular locations the stage leaves free, lowest first, and
        // slots past the last free location stay unlinked.
        patch_io: Some(PatchIo {
            layout: PatchLayout::Ryujinx,
            locations: {
                let mut free = !regular_locations();
                std::array::from_fn(|_| {
                    (free != 0).then(|| {
                        let location = free.trailing_zeros();
                        free &= free - 1;
                        location
                    })
                })
            },
        }),
        geometry_input: Some(GeometryInput::Triangles),
        tessellation: Some(TessellationState {
            domain: TessellationDomain::Triangles,
            spacing: TessellationSpacing::Equal,
            clockwise: false,
        }),
        ..Default::default()
    }
}

// The regular locations the graphics fuzz headers read or write: four
// component bits per location in imap_g_vtg (bits 192..320) and omap_g
// (bits 432..560), Mesa nak/sph.rs; all graphics headers share them.
fn regular_locations() -> u32 {
    let header = &fuzz_headers()[1];
    let map =
        |first: usize| u128::from_le_bytes(header[first / 8..first / 8 + 16].try_into().unwrap());
    let components = map(192) | map(432);
    (0..32)
        .filter(|location| components >> (location * 4) & 0xf != 0)
        .fold(0, |used, location| used | 1 << location)
}
