//! Ryujinx cache reader. Reference paths and layouts are documented in
//! docs/reference-sources.md No translator code is copied.
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fmt::Write as _, fs, io::Read, path::Path};

// DiskCacheGuestStorage.cs:14-21; DiskCacheHostStorage.cs:16-28.
const GUEST_VERSION: u32 = 0x0001_0001;
pub(super) const SHARED_VERSION: u32 = 0x0001_0002;
// Translator.cs:12,35-49,65; shadowbox/src/lib.rs:87-104.
pub(super) const HEADER_SIZE: usize = shadowbox::TargetGeneration::Sm5.shader_header_size();
// Reader resource limit, not a format field. Reject rather than truncate.
const MAX_ENTRY: u64 = 64 * 1024 * 1024;

pub(super) fn read(path: &Path) -> Result<Vec<u8>, String> {
    fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))
}
fn write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    fs::write(path, bytes).map_err(|e| format!("write {}: {e}", path.display()))
}
pub(super) fn u32_at(bytes: &[u8], offset: usize) -> Result<u32, String> {
    let word = bytes.get(offset..offset + 4).ok_or("truncated u32")?;
    Ok(u32::from_le_bytes(word.try_into().unwrap()))
}
pub(super) fn u64_at(bytes: &[u8], offset: usize) -> Result<u64, String> {
    let word = bytes.get(offset..offset + 8).ok_or("truncated u64")?;
    Ok(u64::from_le_bytes(word.try_into().unwrap()))
}
pub(super) fn toc<'a>(
    bytes: &'a [u8],
    magic: &[u8; 4],
    version: u32,
    stride: usize,
) -> Result<&'a [u8], String> {
    // Guest TocHeader: DiskCacheGuestStorage.cs:28-59; shared TocHeader:
    // DiskCacheHostStorage.cs:37-68. Both occupy 32 bytes on the LE writer.
    if bytes.len() < 32 || bytes.get(..4) != Some(magic) {
        return Err(format!(
            "invalid {} TOC header",
            String::from_utf8_lossy(magic)
        ));
    }
    if u32_at(bytes, 4)? != version {
        return Err(format!(
            "unsupported {} cache version 0x{:08x}",
            String::from_utf8_lossy(magic),
            u32_at(bytes, 4)?
        ));
    }
    if !(bytes.len() - 32).is_multiple_of(stride) {
        return Err("truncated TOC entry".into());
    }
    Ok(&bytes[32..])
}
pub(super) fn decompress(bytes: &[u8], expected: Option<usize>) -> Result<Vec<u8>, String> {
    // CompressionAlgorithm.cs:6-21; BinarySerializer.cs:123-140,191-218.
    // Byte tag precedes raw bytes, raw RFC1951 Deflate, or Brotli.
    let (&algorithm, payload) = bytes.split_first().ok_or("missing compression tag")?;
    if expected.is_some_and(|size| size as u64 > MAX_ENTRY) {
        return Err("entry exceeds 64 MiB reader limit".into());
    }
    let reader: Box<dyn Read + '_> = match algorithm {
        0 => Box::new(payload),
        1 => Box::new(flate2::read::DeflateDecoder::new(payload)),
        2 => Box::new(brotli::Decompressor::new(payload, 4096)),
        value => return Err(format!("unsupported compression tag {value}")),
    };
    let mut result = Vec::new();
    reader
        .take(expected.map_or(MAX_ENTRY, |size| size as u64) + 1)
        .read_to_end(&mut result)
        .map_err(|e| format!("decompression: {e}"))?;
    if result.len() as u64 > MAX_ENTRY || expected.is_some_and(|size| size != result.len()) {
        return Err("decompressed entry size mismatch or reader limit exceeded".into());
    }
    Ok(result)
}
pub(super) fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub(super) struct Guest {
    pub(super) code: Vec<u8>,
    cb1: Vec<u8>,
    source_hash: u32,
}
pub(super) fn guests(toc_bytes: &[u8], data: &[u8]) -> Result<Vec<Guest>, String> {
    let entries = toc(toc_bytes, b"TOCG", GUEST_VERSION, 16)?;
    // ModificationsCount increments once per appended entry:
    // DiskCacheGuestStorage.cs:398-413. Padding/reserved are preserved in output.
    if u32_at(toc_bytes, 12)? as usize != entries.len() / 16 {
        return Err("guest modification count does not match TOC entries".into());
    }
    let offsets = entries
        .as_chunks::<16>()
        .0
        .iter()
        .map(|entry| u32_at(entry, 0).map(|x| x as usize))
        .collect::<Result<Vec<_>, _>>()?;
    let mut boundaries = offsets.clone();
    boundaries.sort_unstable();
    if boundaries.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("duplicate guest data offset".into());
    }
    entries
        .as_chunks::<16>()
        .0
        .iter()
        .enumerate()
        .map(|(index, entry)| {
            // Offset, CodeSize, Cb1DataSize, Hash: DiskCacheGuestStorage.cs:64-85.
            let start = offsets[index];
            let end = boundaries
                .get(boundaries.binary_search(&start).unwrap() + 1)
                .copied()
                .unwrap_or(data.len());
            let bytes = data
                .get(start..end)
                .ok_or("guest data offset out of range")?;
            let cb_size = u32_at(entry, 8)? as usize;
            // CB1 is uncompressed before tagged code: same file:222-224,395-396.
            let cb1 = bytes.get(..cb_size).ok_or("truncated CB1 data")?.to_vec();
            let code = decompress(
                bytes.get(cb_size..).ok_or("truncated guest code")?,
                Some(u32_at(entry, 4)? as usize),
            )?;
            Ok(Guest {
                code,
                cb1,
                source_hash: u32_at(entry, 12)?,
            })
        })
        .collect()
}
const STAGES: [&str; 6] = [
    "vertex-a",
    "vertex",
    "tessellation-control",
    "tessellation-evaluation",
    "geometry",
    "fragment",
];

pub fn extract(input: &Path, output: &Path) -> Result<(), String> {
    // Names: DiskCacheGuestStorage.cs:20-21; DiskCacheHostStorage.cs:27-28.
    let guest_toc = read(&input.join("guest.toc"))?;
    let guest_data = read(&input.join("guest.data"))?;
    let shared_toc = read(&input.join("shared.toc"))?;
    let shared_data = read(&input.join("shared.data"))?;
    let guests = guests(&guest_toc, &guest_data).map_err(|e| format!("guest cache: {e}"))?;
    let entries = toc(&shared_toc, b"TOCS", SHARED_VERSION, 8)?;
    // Shared TOC u64 offsets: DiskCacheHostStorage.cs:331-343,592-593.
    let offsets = entries
        .as_chunks::<8>()
        .0
        .iter()
        .map(|x| usize::try_from(u64_at(x, 0)?).map_err(|_| "shared offset too large".into()))
        .collect::<Result<Vec<_>, String>>()?;
    let mut boundaries = offsets.clone();
    boundaries.sort_unstable();
    if boundaries.windows(2).any(|pair| pair[0] == pair[1]) {
        return Err("duplicate shared data offset".into());
    }
    let mut artifacts = BTreeMap::new();
    // Preserve all TOC fields without assigning translation semantics to metadata.
    artifacts.insert("guest.toc".to_string(), guest_toc.clone());
    artifacts.insert("shared.toc".to_string(), shared_toc.clone());
    let mut manifest = String::from("program\tslot\tstage\tfile\theader_bytes\tcode_bytes\trecord_bytes\tcb1_file\tcb1_bytes\tguest_index\tguest_hash\tshared_file\tshared_bytes\n");
    let mut records = std::collections::BTreeSet::new();
    for (program, &start) in offsets.iter().enumerate() {
        let end = boundaries
            .get(boundaries.binary_search(&start).unwrap() + 1)
            .copied()
            .unwrap_or(shared_data.len());
        let shared = decompress(
            shared_data
                .get(start..end)
                .ok_or("shared offset out of range")?,
            None,
        )
        .map_err(|e| format!("shared program {program}: {e}"))?;
        // Zero means compute; bits 0..5 select guest indices in ascending order:
        // DiskCacheHostStorage.cs:105-110,344-374. Slot 0 is Vertex A; slots
        // 1..5 follow graphics stages: ParallelDiskCacheLoader.cs:604-629.
        let mask = u32_at(&shared, 0)?;
        if mask & !0x3f != 0 {
            return Err(format!("invalid stage mask 0x{mask:x}"));
        }
        let compute = mask == 0;
        let slots = if compute { 1 } else { mask };
        let mut cursor = 4;
        let shared_file = format!("{}.shared", hash(&shared));
        for (slot, stage_name) in STAGES.iter().enumerate() {
            if slots & (1 << slot) == 0 {
                continue;
            }
            // Signed GuestCodeIndex: DiskCacheHostStorage.cs:94-100,366-371.
            let index = u32_at(&shared, cursor)? as i32;
            cursor += 4;
            let guest = usize::try_from(index)
                .ok()
                .and_then(|i| guests.get(i))
                .ok_or_else(|| format!("invalid guest index {index}"))?;
            let mut record = Vec::new();
            if compute {
                // Compute has no guest SPH (Translator.cs:35-49,65). This is a
                // Shadowbox envelope, stage 0 = compute (headers.rs:64-69).
                // Keep every code byte. Specialization stays in the shared blob.
                record.resize(HEADER_SIZE, 0);
            } else {
                if guest.code.len() < HEADER_SIZE {
                    return Err("graphics code shorter than SPH".into());
                }
                // SPH stage bits 10..14: Ryujinx ShaderHeader.cs:109;
                // Mesa sph.rs:204-215, cla097sph.h:37-42,407-412. Check, never rewrite.
                let stage = (u32_at(&guest.code, 0)? >> 10) & 0xf;
                let expected = if slot == 0 { 1 } else { slot as u32 };
                if stage != expected {
                    return Err(format!(
                        "SPH stage {stage} disagrees with shared slot {slot}"
                    ));
                }
            }
            record.extend_from_slice(&guest.code);
            // SM50 instructions and schedule words are 8 bytes:
            // Mesa sm50.rs:3418-3428. Reject partial words before translation.
            if !(record.len() - HEADER_SIZE).is_multiple_of(8) {
                return Err("guest code contains a partial instruction word".into());
            }
            let code_size = record.len() - HEADER_SIZE;
            // Saved cb1 belongs to this code: DiskCacheGuestStorage.cs:222-229;
            // GuestCodeAndCbData.cs:11-26. Embed it so duplicate code with
            // different tables cannot lose its association during translation.
            let record = shadowbox::encode_header_prefixed_with_constant_buffer1(
                &record,
                shadowbox::TargetGeneration::Sm5,
                &guest.cb1,
            )
            .map_err(|e| e.to_string())?;
            let file = format!("{}.bin", hash(&record));
            let cb_file = format!("{}.cb1", hash(&guest.cb1));
            records.insert(file.clone());
            writeln!(manifest, "{program}\t{slot}\t{}\t{file}\t{HEADER_SIZE}\t{}\t{}\t{cb_file}\t{}\t{index}\t{:08x}\t{shared_file}\t{}",
                if compute { "compute" } else { stage_name }, code_size, record.len(), guest.cb1.len(), guest.source_hash, shared.len()).unwrap();
            artifacts.insert(file, record);
            artifacts.insert(cb_file, guest.cb1.clone());
        }
        // Remaining bytes are specialization, not shader code:
        // DiskCacheHostStorage.cs:376,618; ShaderSpecializationState.cs:972-1061.
        // Preserve the whole decompressed shared entry, including its prefix.
        artifacts.insert(shared_file, shared);
    }
    // Validate everything before writing. Empty destination prevents stale records.
    // This constraint is CLI policy, not a format fact.
    if output.exists()
        && fs::read_dir(output)
            .map_err(|e| e.to_string())?
            .next()
            .is_some()
    {
        return Err("output directory must be empty".into());
    }
    fs::create_dir_all(output).map_err(|e| format!("create {}: {e}", output.display()))?;
    for (name, bytes) in artifacts {
        write(&output.join(name), &bytes)?;
    }
    write(&output.join("manifest.tsv"), manifest.as_bytes())?;
    println!(
        "extracted {} unique shader records from {} shared programs and {} guest entries",
        records.len(),
        offsets.len(),
        guests.len()
    );
    Ok(())
}
