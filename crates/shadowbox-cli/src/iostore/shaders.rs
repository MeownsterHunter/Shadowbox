//! Per-shader IoStore revision-1 extents. Pinned MIT retoc references:
//! docs/reference-sources.md No instruction mapping is added.
use super::*;
use crate::ue_shaderlib::{self as ue, Map, Reader, Shader};
use serde_json::{json, Value};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Group {
    pub id: [u8; 12],
    pub compressed: u32,
    pub uncompressed: u32,
    pub shaders: Vec<Shader>,
}

pub(crate) struct Library {
    pub groups: Vec<Group>,
    pub metadata: Value,
}

pub(crate) fn library(data: &[u8]) -> io::Result<Library> {
    let mut r = Reader::new(data)?;
    // retoc/src/shader_library.rs:119-143,240-243: revision 1, seven arrays.
    let version = r.u32()?;
    if version != 1 {
        return Err(invalid(format!(
            "unsupported IoStore shader library version {version} (only 1 supported)"
        )));
    }
    // Hash width: retoc/src/lib.rs:1105; group ID width: lib.rs:765-776.
    let map_hashes = r.array(20, |r| r.bytes::<20>())?;
    let hashes = r.array(20, |r| r.bytes::<20>())?;
    let ids = r.array(12, |r| r.bytes::<12>())?;
    // retoc/src/shader_library.rs:21-27: indices offset and shader count.
    let raw_maps = r.array(8, |r| Ok((r.u32()?, r.u32()?)))?;
    // retoc/src/shader_library.rs:39-70: packed u64, frequency 0..4,
    // group 4..34, offset 34..64. Every bit is consumed and reported.
    let entries = r.array(8, Reader::u64)?;
    // retoc/src/shader_library.rs:90-105: index offset, count, expanded/stored sizes.
    let raw_groups = r.array(16, |r| Ok((r.u32()?, r.u32()?, r.u32()?, r.u32()?)))?;
    // retoc/src/shader_library.rs:133,267-273: u32 shader ordinals.
    let indices = r.array(4, Reader::u32)?;
    if hashes.len() != entries.len()
        || map_hashes.len() != raw_maps.len()
        || ids.len() != raw_groups.len()
        || r.at != data.len()
    {
        return Err(invalid("IoStore shader library counts or extent disagree"));
    }
    let maps: Vec<_> = map_hashes
        .into_iter()
        .zip(raw_maps)
        .map(|(hash, (start, count))| Map {
            hash,
            start,
            count,
            preload: None,
        })
        .collect();
    let mut shaders: Vec<_> = hashes
        .into_iter()
        .zip(&entries)
        .enumerate()
        .map(|(index, (hash, &packed))| Shader {
            index,
            hash,
            offset: packed >> 34,
            size: 0,
            uncompressed: 0,
            frequency: (packed & 15) as u8,
            maps: Vec::new(),
        })
        .collect();
    ue::memberships(&maps, &indices, &mut shaders)?;
    let mut seen_ids = HashSet::new();
    let mut seen_shaders = HashSet::new();
    let mut references = 0usize;
    let mut work = 0u64;
    let mut groups = Vec::new();
    for (group_index, (id, (start, count, uncompressed, compressed))) in
        ids.into_iter().zip(raw_groups.iter().copied()).enumerate()
    {
        references += count as usize;
        work += u64::from(uncompressed);
        if !seen_ids.insert(id)
            || references > 16_000_000
            || work > 1024 * 1024 * 1024
            || compressed == 0
            || uncompressed == 0
            || compressed as usize > ue::MAX_BYTES
            || uncompressed as usize > ue::MAX_BYTES
        {
            return Err(invalid("invalid or oversized IoStore shader group"));
        }
        let end = (start as usize)
            .checked_add(count as usize)
            .ok_or_else(|| invalid("IoStore group index overflow"))?;
        let mut members = Vec::new();
        // retoc/src/shader_library.rs:267-284: resolve indices and verify that
        // every shader's packed group index agrees with its containing group.
        for &index in indices
            .get(start as usize..end)
            .ok_or_else(|| invalid("IoStore group index range out of bounds"))?
        {
            let index = index as usize;
            let shader = shaders
                .get(index)
                .ok_or_else(|| invalid("IoStore shader index out of bounds"))?;
            if !seen_shaders.insert(index)
                || ((entries[index] >> 4) & 0x3fff_ffff) as usize != group_index
                || shader.offset >= u64::from(uncompressed)
            {
                return Err(invalid("conflicting IoStore shader group or offset"));
            }
            members.push(shader.clone());
        }
        if members.is_empty() {
            return Err(invalid("empty IoStore shader group"));
        }
        // retoc/src/shader_library.rs:287-305: sort offsets, next offset ends
        // each shader; the last shader ends at the expanded group size.
        members.sort_by_key(|s| s.offset);
        for i in 0..members.len() {
            let end = members
                .get(i + 1)
                .map_or(u64::from(uncompressed), |s| s.offset);
            let size = end
                .checked_sub(members[i].offset)
                .filter(|&n| n != 0)
                .ok_or_else(|| invalid("duplicate IoStore shader offsets"))?;
            members[i].size = size as u32;
            members[i].uncompressed = size as u32;
        }
        groups.push(Group {
            id,
            compressed,
            uncompressed,
            shaders: members,
        });
    }
    // retoc/src/shader_library.rs:310-320: every shader must belong to a group.
    if seen_shaders.len() != shaders.len() {
        return Err(invalid("IoStore shader is absent from its group"));
    }
    Ok(Library {
        groups,
        metadata: json!({"version": version,
        "shader_maps": maps.iter().map(Map::metadata).collect::<Vec<_>>(),
        "shader_indices": indices, "group_entries": raw_groups}),
    })
}

pub(crate) fn expand(data: &[u8], group: &Group) -> io::Result<(Vec<u8>, &'static str)> {
    ue::expand(data, group.compressed, group.uncompressed)
}
