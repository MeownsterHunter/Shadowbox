//! Original readers for the pinned MIT retoc formats. Every format citation is
//! relative to retoc at 885a8dae740cb1ce1e41ff2e74f67f9f0c118237; see
//! docs/reference-sources.md No instruction semantics are added.
use std::io::{self, Read};

use serde_json::{json, Value};
use shadowbox::headers::Stage;

pub(super) const MAX_BYTES: usize = 512 * 1024 * 1024;
const MAX_ITEMS: usize = 1_000_000;
const MAX_REFERENCES: usize = 16_000_000;

pub(super) fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

pub(super) struct Reader<'a> {
    data: &'a [u8],
    pub at: usize,
}
impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> io::Result<Self> {
        if data.len() > MAX_BYTES {
            return Err(invalid("shader library exceeds resource limit"));
        }
        Ok(Self { data, at: 0 })
    }
    pub fn bytes<const N: usize>(&mut self) -> io::Result<[u8; N]> {
        let end = self
            .at
            .checked_add(N)
            .ok_or_else(|| invalid("table overflow"))?;
        let bytes = self
            .data
            .get(self.at..end)
            .ok_or_else(|| invalid("truncated shader library table"))?;
        self.at = end;
        Ok(bytes.try_into().unwrap())
    }
    // retoc/src/ser.rs:200-227: little-endian primitive integers.
    pub fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.bytes()?))
    }
    pub fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(self.bytes()?))
    }
    // retoc/src/ser.rs:111-114: u32 array count. Check extent before allocation.
    pub fn array<T>(
        &mut self,
        width: usize,
        mut read: impl FnMut(&mut Self) -> io::Result<T>,
    ) -> io::Result<Vec<T>> {
        let count = self.u32()? as usize;
        if count > MAX_ITEMS || count > (self.data.len() - self.at) / width {
            return Err(invalid("invalid or oversized shader library array"));
        }
        (0..count).map(|_| read(self)).collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Map {
    pub hash: [u8; 20],
    pub start: u32,
    pub count: u32,
    pub preload: Option<(u32, u32)>,
}
impl Map {
    pub fn metadata(&self) -> Value {
        json!({"hash": crate::iostore::hex(&self.hash), "shader_indices_offset": self.start,
            "num_shaders": self.count, "preload_range": self.preload})
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Shader {
    pub index: usize,
    pub hash: [u8; 20],
    pub offset: u64,
    pub size: u32,
    pub uncompressed: u32,
    pub frequency: u8,
    pub maps: Vec<usize>,
}
impl Shader {
    pub fn metadata(&self) -> Value {
        json!({"shader_index": self.index, "shader_hash": crate::iostore::hex(&self.hash),
            "offset": self.offset, "size": self.size, "uncompressed_size": self.uncompressed,
            "frequency": self.frequency, "shader_maps": self.maps})
    }
    pub fn stage(&self) -> io::Result<Stage> {
        // Legacy container frequencies use 1 for hull and 2 for domain.
        // GPU stage mapping follows Ryujinx ShaderStage.cs:3-10 and Mesa
        // nak/sph.rs:204-215, as already used by headers.rs:74-81. The record
        // header must independently agree; no GPU facts come from retoc.
        match self.frequency {
            0 => Ok(Stage::Vertex),
            1 => Ok(Stage::TessellationControl),
            2 => Ok(Stage::TessellationEvaluation),
            3 => Ok(Stage::Fragment),
            4 => Ok(Stage::Geometry),
            5 => Ok(Stage::Compute),
            n => Err(invalid(format!(
                "unsupported or ambiguous Unreal shader frequency {n}"
            ))),
        }
    }
}

pub(super) fn memberships(maps: &[Map], indices: &[u32], shaders: &mut [Shader]) -> io::Result<()> {
    let mut references = 0usize;
    for (map_index, map) in maps.iter().enumerate() {
        references = references
            .checked_add(map.count as usize)
            .ok_or_else(|| invalid("shader map work overflow"))?;
        if references > MAX_REFERENCES {
            return Err(invalid("shader map reference limit"));
        }
        let end = (map.start as usize)
            .checked_add(map.count as usize)
            .ok_or_else(|| invalid("shader map range overflow"))?;
        for &index in indices
            .get(map.start as usize..end)
            .ok_or_else(|| invalid("shader map range out of bounds"))?
        {
            shaders
                .get_mut(index as usize)
                .ok_or_else(|| invalid("shader index out of bounds"))?
                .maps
                .push(map_index);
        }
    }
    if indices.iter().any(|&i| i as usize >= shaders.len()) {
        return Err(invalid("shader index out of bounds"));
    }
    Ok(())
}

pub(super) struct Loose {
    pub shaders: Vec<Shader>,
    pub metadata: Value,
    pub code_start: usize,
}

pub(super) fn loose(data: &[u8]) -> io::Result<Loose> {
    let mut r = Reader::new(data)?;
    // retoc/src/shader_library.rs:1077-1083: only loose revision 2, offsets
    // relative to the end of the header. Revision 1 here is not IoStore.
    let version = r.u32()?;
    if version != 2 {
        return Err(invalid(format!(
            "unsupported loose shader library version {version} (only 2 supported)"
        )));
    }
    // retoc/src/shader_library.rs:424-431: six counted arrays, in this order;
    // retoc/src/lib.rs:1105: 20-byte SHA hashes, preserved rather than verified.
    let map_hashes = r.array(20, |r| r.bytes::<20>())?;
    let hashes = r.array(20, |r| r.bytes::<20>())?;
    // retoc/src/shader_library.rs:335-348: four u32 fields per map.
    let raw_maps = r.array(16, |r| Ok((r.u32()?, r.u32()?, r.u32()?, r.u32()?)))?;
    // retoc/src/shader_library.rs:361-375: u64 offset, u32 stored and
    // uncompressed sizes, u8 frequency. No host struct padding on disk.
    let entries = r.array(17, |r| {
        Ok((r.u64()?, r.u32()?, r.u32()?, r.bytes::<1>()?[0]))
    })?;
    // retoc/src/shader_library.rs:388-395: signed i64 code-relative ranges.
    let preloads = r.array(16, |r| {
        Ok((
            i64::from_le_bytes(r.bytes()?),
            i64::from_le_bytes(r.bytes()?),
        ))
    })?;
    // retoc/src/shader_library.rs:412,1114-1116: u32 shader ordinals.
    let indices = r.array(4, Reader::u32)?;
    if map_hashes.len() != raw_maps.len() || hashes.len() != entries.len() {
        return Err(invalid("shader library hash/entry counts disagree"));
    }
    let code_start = r.at;
    let code_size = data.len() - code_start;
    let range = |offset: u64, size: u64| {
        offset
            .checked_add(size)
            .is_some_and(|end| end <= code_size as u64)
    };
    for &(offset, size) in &preloads {
        if offset < 0 || size < 0 || !range(offset as u64, size as u64) {
            return Err(invalid("shader preload range out of bounds"));
        }
    }
    let maps: Vec<_> = map_hashes
        .into_iter()
        .zip(raw_maps)
        .map(|(hash, (start, count, first, num))| {
            if (first as usize)
                .checked_add(num as usize)
                .is_none_or(|end| end > preloads.len())
            {
                return Err(invalid("shader map preload range out of bounds"));
            }
            Ok(Map {
                hash,
                start,
                count,
                preload: Some((first, num)),
            })
        })
        .collect::<io::Result<_>>()?;
    let mut work = 0u64;
    let mut shaders: Vec<_> = hashes
        .into_iter()
        .zip(entries)
        .enumerate()
        .map(|(index, (hash, (offset, size, uncompressed, frequency)))| {
            work += u64::from(uncompressed);
            if size == 0
                || uncompressed == 0
                || size as usize > MAX_BYTES
                || uncompressed as usize > MAX_BYTES
                || work > 1024 * 1024 * 1024
                || !range(offset, u64::from(size))
            {
                return Err(invalid("invalid or oversized shader code range"));
            }
            Ok(Shader {
                index,
                hash,
                offset,
                size,
                uncompressed,
                frequency,
                maps: Vec::new(),
            })
        })
        .collect::<io::Result<_>>()?;
    memberships(&maps, &indices, &mut shaders)?;
    Ok(Loose {
        shaders,
        code_start,
        metadata: json!({"version": version, "code_start": code_start,
        "shader_maps": maps.iter().map(Map::metadata).collect::<Vec<_>>(),
        "preloads": preloads, "shader_indices": indices}),
    })
}

pub(super) fn named(name: &str) -> bool {
    name.to_ascii_lowercase().ends_with(".ushaderbytecode")
}

pub(super) fn read(input: &mut impl Read) -> io::Result<Vec<u8>> {
    let mut data = Vec::new();
    input.take(MAX_BYTES as u64 + 1).read_to_end(&mut data)?;
    if data.len() > MAX_BYTES {
        return Err(invalid("shader library exceeds resource limit"));
    }
    Ok(data)
}

pub(super) fn expand(
    data: &[u8],
    size: u32,
    uncompressed: u32,
) -> io::Result<(Vec<u8>, &'static str)> {
    if data.len() != size as usize {
        return Err(invalid("shader compressed size mismatch"));
    }
    if size == uncompressed {
        return Ok((data.to_vec(), "None"));
    }
    // retoc/src/shader_library.rs:191-214,1127-1129: no stored codec;
    // validated codec attempts and exact expansion sizes, never Oodle.
    for method in ["Zlib", "Zstd", "LZ4"] {
        if let Ok(bytes) = crate::iostore::decompress(method, data, uncompressed as usize) {
            return Ok((bytes, method));
        }
    }
    Err(invalid(
        "unsupported or invalid shader compression; Oodle is unsupported",
    ))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    fn p32(data: &mut Vec<u8>, n: u32) {
        data.extend(n.to_le_bytes());
    }
    // Synthetic writers use exactly the cited field widths above. Hashes are
    // fabricated identifiers, never hashes or bytes obtained from game data.
    pub(crate) fn loose_fixture(shaders: &[(u8, Vec<u8>)], codec: &str) -> Vec<u8> {
        let encoded: Vec<_> = shaders
            .iter()
            .map(|(_, data)| crate::iostore::tests::compressed(codec, data))
            .collect();
        let mut v = Vec::new();
        p32(&mut v, 2);
        p32(&mut v, 1);
        v.extend([0xab; 20]);
        p32(&mut v, shaders.len() as u32);
        for i in 0..shaders.len() {
            v.extend([i as u8 + 1; 20]);
        }
        p32(&mut v, 1);
        for n in [0, shaders.len() as u32, 0, 1] {
            p32(&mut v, n);
        }
        p32(&mut v, shaders.len() as u32);
        let size: usize = encoded.iter().map(Vec::len).sum();
        let mut end = size;
        // Reverse physical ordering proves the reader follows offsets.
        for ((frequency, raw), compressed) in shaders.iter().zip(&encoded) {
            end -= compressed.len();
            v.extend((end as u64).to_le_bytes());
            p32(&mut v, compressed.len() as u32);
            p32(&mut v, raw.len() as u32);
            v.push(*frequency);
        }
        p32(&mut v, 1);
        v.extend(0i64.to_le_bytes());
        v.extend((size as i64).to_le_bytes());
        p32(&mut v, shaders.len() as u32);
        for i in (0..shaders.len()).rev() {
            p32(&mut v, i as u32);
        }
        for data in encoded.iter().rev() {
            v.extend(data);
        }
        v
    }

    pub(crate) fn group_fixture(
        id: [u8; 12],
        shaders: &[(u8, Vec<u8>)],
        compressed: u32,
    ) -> Vec<u8> {
        let mut v = Vec::new();
        p32(&mut v, 1);
        p32(&mut v, 1);
        v.extend([0xab; 20]);
        p32(&mut v, shaders.len() as u32);
        for i in 0..shaders.len() {
            v.extend([i as u8 + 1; 20]);
        }
        p32(&mut v, 1);
        v.extend(id);
        p32(&mut v, 1);
        p32(&mut v, 0);
        p32(&mut v, shaders.len() as u32);
        p32(&mut v, shaders.len() as u32);
        let mut offset = 0u64;
        for (frequency, bytes) in shaders {
            v.extend(((offset << 34) | u64::from(*frequency)).to_le_bytes());
            offset += bytes.len() as u64;
        }
        p32(&mut v, 1);
        for n in [0, shaders.len() as u32, offset as u32, compressed] {
            p32(&mut v, n);
        }
        p32(&mut v, shaders.len() as u32);
        for i in (0..shaders.len()).rev() {
            p32(&mut v, i as u32);
        }
        v
    }

    #[test]
    fn loose_tables_extents_and_compression_are_exact() {
        let entries = [(3, vec![0x27; 320]), (5, vec![0x18; 512])];
        for codec in ["None", "Zlib", "Zstd", "LZ4"] {
            let bytes = loose_fixture(&entries, codec);
            let library = loose(&bytes).unwrap();
            assert_eq!(library.shaders[0].maps, [0]);
            assert_eq!(library.metadata["shader_indices"], json!([1, 0]));
            for (shader, (_, expected)) in library.shaders.iter().zip(&entries) {
                let start = library.code_start + shader.offset as usize;
                let (actual, method) = expand(
                    &bytes[start..start + shader.size as usize],
                    shader.size,
                    shader.uncompressed,
                )
                .unwrap();
                assert_eq!(&actual, expected);
                assert_eq!(method, codec);
            }
            for len in 0..library.code_start {
                assert!(loose(&bytes[..len]).is_err(), "length {len}");
            }
            assert!(loose(&bytes[..bytes.len() - 1]).is_err());
        }
        let original = loose_fixture(&entries, "None");
        // Version, array counts, map indices/preload indices, code offset/sizes,
        // preload signed offset/size and shader index must affect validation.
        for at in [
            0, 4, 28, 72, 76, 80, 84, 88, 92, 100, 104, 108, 130, 134, 142, 150, 154,
        ] {
            let mut bad = original.clone();
            bad[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
            assert!(loose(&bad).is_err(), "field at {at}");
        }
        let mut changed = original.clone();
        changed[8] ^= 1;
        assert_ne!(
            loose(&changed).unwrap().metadata,
            loose(&original).unwrap().metadata
        );
        let mut changed = original.clone();
        changed[32] ^= 1;
        assert_ne!(
            loose(&changed).unwrap().shaders[0].hash,
            loose(&original).unwrap().shaders[0].hash
        );
        let mut changed = original.clone();
        changed[112] = 1;
        assert_eq!(
            loose(&changed).unwrap().shaders[0].stage().unwrap(),
            Stage::TessellationControl
        );
        assert!(expand(&[0xff; 4], 4, 32)
            .unwrap_err()
            .to_string()
            .contains("Oodle"));
    }

    #[test]
    fn iostore_packed_entries_and_memberships_are_checked() {
        let mut id = [0; 12];
        id[11] = 9;
        let entries = [(3, vec![0; 160]), (5, vec![0; 288])];
        let original = group_fixture(id, &entries, 448);
        let library = crate::iostore::shaders::library(&original).unwrap();
        assert_eq!(library.groups[0].shaders[0].size, 160);
        assert_eq!(library.groups[0].shaders[1].offset, 160);
        assert_eq!(
            library.groups[0].shaders[1].stage().unwrap(),
            Stage::Compute
        );
        assert_eq!(library.groups[0].shaders[1].maps, [0]);
        for len in 0..original.len() {
            assert!(crate::iostore::shaders::library(&original[..len]).is_err());
        }
        // Packed first entry at 104: group bits and offset bits.
        for (at, bit) in [(104, 4), (104, 63), (112, 63)] {
            let mut bad = original.clone();
            let mut packed = u64::from_le_bytes(bad[at..at + 8].try_into().unwrap());
            packed ^= 1 << bit;
            bad[at..at + 8].copy_from_slice(&packed.to_le_bytes());
            assert!(
                crate::iostore::shaders::library(&bad).is_err(),
                "at {at} bit {bit}"
            );
        }
    }
}
