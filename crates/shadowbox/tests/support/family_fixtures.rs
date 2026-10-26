//! Synthetic encodings only, independently authored from MIT references.
const PT: u64 = 7 << 16; // envytools envydis/gm107.c:371,409,411.

pub fn ast(bit32: bool, components: u64, patch: bool) -> u64 {
    // Mesa nv50_ir_emit_gm107.cpp:2538-2545 emits AST with bit32 clear;
    // nak/sm50.rs:2886-2896 sets it. Both store outputs, as Ryujinx
    // InstDecoders.cs:890-902 and InstEmitAttribute.cs:105-141 establish.
    // Data0..7, offset8..15, address20..29, patch31, vertex39..46,
    // count47..48: those Mesa lines and envytools gm107.c:314,325,475-479,1883.
    (0xeff0 << 48)
        | PT
        | (255 << 8)
        | (0x80 << 20)
        | (u64::from(bit32) << 32)
        | (255 << 39)
        | ((components - 1) << 47)
        | (u64::from(patch) << 31)
}

pub fn cases() -> Vec<(String, Vec<u64>)> {
    let mut cases = Vec::new();
    for bit in [false, true] {
        for count in 1..=4 {
            cases.push((
                format!("AST.bit32={bit}.count={count}"),
                vec![ast(bit, count, false)],
            ));
        }
    }
    // MOV32I full quad mask and immediate: Mesa nak/sm50.rs:1938-1941.
    let mov = |reg: u64, value: u64| (0x0100 << 48) | PT | (15 << 12) | reg | (value << 20);
    for bound in [false, true] {
        for dim in [0, 1, 2, 3, 4, 6, 7_u64] {
            for lod in [0, 1, 2, 3, 6, 7_u64] {
                // TEX opcode, D/A/B, dim, mask, table index and LOD:
                // envytools gm107.c:875-880,916-923,1962,1970;
                // Mesa nak/sm50.rs:2181-2197; old emitter:2908-2924.
                // Bindless B starts with the handle: Ryujinx
                // InstEmitTexture.cs:207-210,252-291.
                let tex = (if bound { 0xc038 } else { 0xdeb8 }) << 48
                    | PT
                    | (2 << 28)
                    | (15 << 31)
                    | (8 << 8)
                    | ((if bound && lod <= 1 { 255 } else { 20 }) << 20)
                    | if bound {
                        (9 << 36) | (lod << 55)
                    } else {
                        lod << 37
                    };
                let tex = (tex & !(7 << 28)) | (dim << 28);
                cases.push((
                    format!("TEX.Sample.bound={bound}.dim={dim}.lod={lod}"),
                    vec![mov(20, 9), tex, ast(true, 4, false)],
                ));
            }
            if dim == 6 || dim == 7 {
                continue;
            }
            for lod in [false, true] {
                for offset in [false, true] {
                    // TLD: Mesa nak/sm50.rs:2206-2236, envytools
                    // gm107.c:1965,1975. No multisampling in this grid.
                    let tld = (if bound { 0xdc38 } else { 0xdd38 }) << 48
                        | PT
                        | (dim << 28)
                        | (15 << 31)
                        | (8 << 8)
                        | ((if bound && !lod && !offset { 255 } else { 20 }) << 20)
                        | (u64::from(lod) << 55)
                        | (u64::from(offset) << 35)
                        | if bound { 9 << 36 } else { 0 };
                    cases.push((
                        format!("TEX.Fetch.bound={bound}.dim={dim}.lod={lod}.offset={offset}"),
                        vec![mov(20, 9), tld, ast(true, 4, false)],
                    ));
                }
            }
        }
        for dim in 0..=5 {
            for raw in [false, true] {
                // SUST: Mesa nak/sm50.rs:2504-2529; envytools
                // gm107.c:788-838,1909. Coords8..15, data0..7,
                // dim33..35, raw52, mask/size20..23, bound51, handle.
                // Ryujinx InstEmitSurface.cs:566-795 stores image data.
                let sust = (0xeb20 << 48)
                    | PT
                    | (2 << 8)
                    | (dim << 33)
                    | (u64::from(raw) << 52)
                    | ((if raw { 4 } else { 15 }) << 20)
                    | if bound {
                        (1 << 51) | (9 << 36)
                    } else {
                        20 << 39
                    };
                cases.push((
                    format!("SUST.bound={bound}.dim={dim}.raw={raw}"),
                    vec![mov(20, 9), sust],
                ));
            }
        }
    }
    cases
}

pub fn program(instructions: &[u64]) -> Vec<u8> {
    // EXIT and bundles: Mesa nak/sm50.rs:3101-3110,3418-3428,3438-3468.
    let mut slots = instructions.to_vec();
    slots.push((0xe300 << 48) | PT | 15);
    slots
        .chunks(3)
        .flat_map(|chunk| {
            let mut bundle = [
                0,
                0x50b0_0000_0007_0000,
                0x50b0_0000_0007_0000,
                0x50b0_0000_0007_0000,
            ];
            // Padding is Mesa OpNop: nak/sm50.rs:3178-3184.
            bundle[1..1 + chunk.len()].copy_from_slice(chunk);
            bundle
        })
        .flat_map(u64::to_le_bytes)
        .collect()
}

pub fn header() -> Vec<u8> {
    // Vertex type and generic output map: Mesa cla097sph.h:37-42,66-73;
    // nak/sph.rs:134,169. A vertex header also permits explicit texture LOD.
    let mut words = [0u32; 32];
    words[0] = 1 << 10;
    words[13] = 15 << 16;
    words.into_iter().flat_map(u32::to_le_bytes).collect()
}
