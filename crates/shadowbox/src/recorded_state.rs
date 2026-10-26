//! Typed driver state recovered from a decompressed shared cache record.

use crate::interface::{BoundTexture, GeometryInput, TextureArray, TextureDimension};

fn u32_at(bytes: &[u8], at: usize) -> Result<u32, String> {
    let value = bytes.get(at..at + 4).ok_or("truncated shared state")?;
    Ok(u32::from_le_bytes(value.try_into().unwrap()))
}

fn sized<'a>(bytes: &'a [u8], at: usize, magic: &[u8; 4]) -> Result<(&'a [u8], usize), String> {
    if bytes.get(at..at + 4) != Some(magic) {
        return Err(format!("expected {} magic", String::from_utf8_lossy(magic)));
    }
    let end = at + 8 + u32_at(bytes, at + 4)? as usize;
    Ok((
        bytes.get(at + 8..end).ok_or("truncated specialization")?,
        end,
    ))
}

fn texture_section(shared: &[u8]) -> Result<(u32, usize, u16), String> {
    let mask = u32_at(shared, 0)?;
    let compute = mask == 0;
    let mut at = 4 + 4 * mask.max(1).count_ones() as usize;
    let queried = u32_at(shared, at)?;
    at = sized(shared, at + 5, if compute { b"COMS" } else { b"GFXS" })?.1;
    let cbufs = *shared.get(at).ok_or("truncated specialization")?;
    at += 1 + 4 * cbufs.count_ones() as usize;
    let pipeline = *shared.get(at).ok_or("truncated specialization")?;
    at += 1;
    if pipeline != 0 {
        at = sized(shared, at, b"PGPS")?.1;
    }
    let count_at = |at: usize| -> Result<u16, String> {
        let bytes = shared.get(at..at + 2).ok_or("truncated specialization")?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    };
    if queried & (1 << 3) != 0 {
        let count = count_at(at)?;
        at += 2;
        for _ in 0..count {
            at = sized(shared, at, b"TFBD")?.1;
        }
    }
    let count = count_at(at)?;
    Ok((queried, at + 2, count))
}

/// Return the geometry input topology recorded by the graphics driver.
///
/// Ryujinx serializes its unmanaged graphics-state struct in the GFXS record.
/// Natural `u32` alignment puts `PrimitiveTopology` at byte 4. Keep the same
/// closed mapping used by the cache importer instead of inferring topology
/// from shader instructions.
pub fn geometry_input(shared: &[u8]) -> Result<Option<GeometryInput>, String> {
    let mask = u32_at(shared, 0)?;
    if mask == 0 {
        return Ok(None);
    }
    let at = 4 + 4 * mask.count_ones() as usize;
    let (graphics, _) = sized(shared, at + 5, b"GFXS")?;
    if graphics.len() < 8 {
        return Ok(None);
    }
    Ok(Some(match u32_at(graphics, 4)? {
        0 | 7..=9 => GeometryInput::Points,
        1..=3 => GeometryInput::Lines,
        4..=6 => GeometryInput::Triangles,
        10 | 11 => GeometryInput::LinesAdjacency,
        12 | 13 => GeometryInput::TrianglesAdjacency,
        topology => return Err(format!("invalid primitive topology {topology}")),
    }))
}

/// Return the bound texture shapes recorded for one shader stage.
///
/// A bound texture query has no encoded dimension. The runtime must supply the
/// matching shape instead of allowing translation to assume one.
pub fn queried_textures(shared: &[u8], stage_index: u32) -> Result<Vec<BoundTexture>, String> {
    let (_, mut at, count) = texture_section(shared)?;
    let mut textures = Vec::new();
    for _ in 0..count {
        let (key, next) = sized(shared, at, b"TEXK")?;
        let (state, next) = sized(shared, next, b"TEXS")?;
        at = next;
        let stage = u32_at(key, 0)?;
        let handle = u32_at(key, 4)?;
        let slot = u32_at(key, 8)? as i32;
        if stage != stage_index || slot != -1 || u32_at(state, 0)? & 2 == 0 {
            continue;
        }
        let dimension = match *state.get(9).ok_or("truncated texture state")? {
            0 => TextureDimension::D1,
            1 | 7 => TextureDimension::D2,
            2 => TextureDimension::D3,
            3 => TextureDimension::Cube,
            4 => TextureDimension::Array1D,
            5 => TextureDimension::Array2D,
            8 => TextureDimension::ArrayCube,
            6 => continue,
            target => return Err(format!("invalid texture target {target}")),
        };
        textures.push(BoundTexture {
            binding: u16::try_from(handle).map_err(|_| "texture handle out of range")?,
            dimension,
            depth_compare: false,
            multisample: false,
        });
    }
    Ok(textures)
}

/// Return constant-buffer-backed descriptor-array lengths for one stage.
///
/// Ryujinx records these after its ordinary texture specialization records
/// when `TextureArrayFromBuffer` is queried. The emitter uses the length to
/// retain the exact fixed descriptor-array boundary.
pub fn texture_arrays(shared: &[u8], stage_index: u32) -> Result<Vec<TextureArray>, String> {
    let (queried, mut at, count) = texture_section(shared)?;
    for _ in 0..count {
        at = sized(shared, at, b"TEXK")?.1;
        at = sized(shared, at, b"TEXS")?.1;
    }
    if queried & (1 << 4) == 0 {
        return Ok(Vec::new());
    }
    let count = shared
        .get(at..at + 2)
        .map(|bytes| u16::from_le_bytes([bytes[0], bytes[1]]))
        .ok_or("truncated texture array count")?;
    at += 2;
    let mut arrays = Vec::new();
    for _ in 0..count {
        let (key, next) = sized(shared, at, b"TEXK")?;
        at = next;
        let stage = u32_at(key, 0)?;
        let handle = u32_at(key, 4)?;
        let slot = u32_at(key, 8)? as i32;
        let length = u32_at(shared, at)?;
        at += 4;
        if stage != stage_index {
            continue;
        }
        let binding = u8::try_from(slot).map_err(|_| "texture array bank out of range")?;
        if binding >= 32 || length < 2 {
            return Err("invalid texture array declaration".into());
        }
        arrays.push(TextureArray {
            binding,
            handle: u16::try_from(handle).map_err(|_| "texture array handle out of range")?,
            length,
        });
    }
    Ok(arrays)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sized_record(magic: &[u8; 4], body: &[u8], output: &mut Vec<u8>) {
        output.extend_from_slice(magic);
        output.extend_from_slice(&(body.len() as u32).to_le_bytes());
        output.extend_from_slice(body);
    }

    fn fixture(target: u8) -> Vec<u8> {
        let mut shared = Vec::new();
        shared.extend_from_slice(&2u32.to_le_bytes());
        shared.extend_from_slice(&0u32.to_le_bytes());
        shared.extend_from_slice(&0u32.to_le_bytes());
        shared.push(0);
        sized_record(b"GFXS", &[], &mut shared);
        shared.push(0);
        shared.push(0);
        shared.extend_from_slice(&1u16.to_le_bytes());
        let mut key = Vec::new();
        key.extend_from_slice(&0u32.to_le_bytes());
        key.extend_from_slice(&7u32.to_le_bytes());
        key.extend_from_slice(&u32::MAX.to_le_bytes());
        sized_record(b"TEXK", &key, &mut shared);
        let mut state = vec![0; 10];
        state[..4].copy_from_slice(&2u32.to_le_bytes());
        state[9] = target;
        sized_record(b"TEXS", &state, &mut shared);
        shared
    }

    fn array_fixture() -> Vec<u8> {
        let mut shared = fixture(1);
        shared[8..12].copy_from_slice(&(1u32 << 4).to_le_bytes());
        shared.extend_from_slice(&1u16.to_le_bytes());
        let mut key = Vec::new();
        key.extend_from_slice(&0u32.to_le_bytes());
        key.extend_from_slice(&0u32.to_le_bytes());
        key.extend_from_slice(&2u32.to_le_bytes());
        sized_record(b"TEXK", &key, &mut shared);
        shared.extend_from_slice(&17u32.to_le_bytes());
        shared
    }

    fn geometry_fixture(topology: u32) -> Vec<u8> {
        let mut shared = Vec::new();
        shared.extend_from_slice(&2u32.to_le_bytes());
        shared.extend_from_slice(&0u32.to_le_bytes());
        shared.extend_from_slice(&0u32.to_le_bytes());
        shared.push(0);
        let mut graphics = vec![0; 8];
        graphics[4..8].copy_from_slice(&topology.to_le_bytes());
        sized_record(b"GFXS", &graphics, &mut shared);
        shared
    }

    #[test]
    fn recorded_query_shape_is_typed_and_stage_scoped() {
        assert_eq!(
            queried_textures(&fixture(1), 0).unwrap(),
            [BoundTexture {
                binding: 7,
                dimension: TextureDimension::D2,
                depth_compare: false,
                multisample: false,
            }]
        );
        assert!(queried_textures(&fixture(1), 1).unwrap().is_empty());
        assert_eq!(
            queried_textures(&fixture(6), 0).unwrap(),
            Vec::<BoundTexture>::new()
        );
        assert_eq!(
            queried_textures(&fixture(9), 0).unwrap_err(),
            "invalid texture target 9"
        );
    }

    #[test]
    fn recorded_texture_array_is_typed_and_stage_scoped() {
        assert_eq!(
            texture_arrays(&array_fixture(), 0).unwrap(),
            [TextureArray {
                binding: 2,
                handle: 0,
                length: 17,
            }]
        );
        assert!(texture_arrays(&array_fixture(), 1).unwrap().is_empty());
        assert!(texture_arrays(&fixture(1), 0).unwrap().is_empty());
    }

    #[test]
    fn recorded_geometry_topology_is_typed_and_fail_closed() {
        assert!(matches!(
            geometry_input(&geometry_fixture(5)).unwrap(),
            Some(GeometryInput::Triangles)
        ));
        assert_eq!(
            geometry_input(&geometry_fixture(15)).unwrap_err(),
            "invalid primitive topology 15"
        );

        let mut truncated = geometry_fixture(5);
        truncated.truncate(20);
        assert!(geometry_input(&truncated).is_err());
    }
}
