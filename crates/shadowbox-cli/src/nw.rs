//! First-party container readers. Pinned sources and field map:
//! docs/reference-sources.md and 0021-nw-versions.md.
//! No instruction semantics live here.
use shadowbox::headers::Stage;
use std::{io::Read, ops::Range};

// Resource policy, not format limits. Also bound cumulative work in the caller.
pub(super) const MAX_BYTES: usize = 256 * 1024 * 1024;
pub(super) const MAX_DEPTH: usize = 16;
const MAX_DICTIONARIES: usize = 128;
const MAX_DICTIONARY_BYTES: usize = 16 * 1024 * 1024;
pub(super) type Result<T> = std::result::Result<T, String>;

fn slice(bytes: &[u8], offset: usize, size: usize) -> Result<&[u8]> {
    let end = offset.checked_add(size).ok_or("range overflow")?;
    bytes
        .get(offset..end)
        .ok_or_else(|| "truncated or out-of-range container field".into())
}
fn u16_at(bytes: &[u8], at: usize, be: bool) -> Result<u16> {
    let v = slice(bytes, at, 2)?.try_into().unwrap();
    Ok(if be {
        u16::from_be_bytes(v)
    } else {
        u16::from_le_bytes(v)
    })
}
fn u32_at(bytes: &[u8], at: usize, be: bool) -> Result<u32> {
    let v = slice(bytes, at, 4)?.try_into().unwrap();
    Ok(if be {
        u32::from_be_bytes(v)
    } else {
        u32::from_le_bytes(v)
    })
}
fn offset_at(bytes: &[u8], at: usize) -> Result<usize> {
    let v = u64::from_le_bytes(slice(bytes, at, 8)?.try_into().unwrap());
    usize::try_from(v).map_err(|_| "offset exceeds address space".into())
}
fn range(bytes: &[u8], at: usize, size: usize) -> Result<Range<usize>> {
    slice(bytes, at, size)?;
    Ok(at..at + size)
}

pub(super) fn kind(bytes: &[u8]) -> Option<&'static str> {
    // Nintendo-File-Formats src/libs/sead/{yaz0.md:7,sarc.md:7};
    // ShaderLibrary/BnshFile.cs:44, BfshaFile.cs:39; BFRES ResFileParser.cs:16.
    // RFC 8878:3.1.1,3.1.2: LE frame and skippable magic.
    match bytes.get(..4)? {
        b"Yaz0" => Some("Yaz0"),
        b"SARC" => Some("SARC"),
        b"BNSH" => Some("BNSH"),
        b"FSHA" => Some("BFSHA"),
        b"FRES" => Some("BFRES"),
        // BntxLibrary/BntxReader.cs:20-21; BntxFile.cs:550-567.
        b"BNTX" => Some("BNTX"),
        [0x28, 0xb5, 0x2f, 0xfd] | [0x50..=0x5f, 0x2a, 0x4d, 0x18] => Some("zstd"),
        _ => None,
    }
}

pub(super) fn yaz0(bytes: &[u8]) -> Result<Vec<u8>> {
    // Nintendo-File-Formats src/libs/sead/yaz0.md:3,7-11: BE header.
    if slice(bytes, 0, 4)? != b"Yaz0" {
        return Err("missing Yaz0 magic".into());
    }
    let size = u32_at(bytes, 4, true)? as usize;
    let alignment = u32_at(bytes, 8, true)?;
    if size > MAX_BYTES {
        return Err("Yaz0 exceeds resource limit".into());
    }
    if alignment != 0 && !alignment.is_power_of_two() {
        return Err("invalid Yaz0 alignment".into());
    }
    if u32_at(bytes, 12, true)? != 0 {
        return Err("unsupported Yaz0 reserved field".into());
    }
    let mut out = Vec::with_capacity(size);
    let mut at = 16;
    while out.len() < size {
        // Public Yaz0 format table, mkwiiki YAZ0_(File_Format),
        // Data Groups rows 1-8 and size table (rendered lines 30-51).
        // MSB first; literal or 12-bit distance +1, nibble length +2,
        // or third byte +0x12. Overlapping references copy incrementally.
        let group = slice(bytes, at, 1)?[0];
        at += 1;
        for bit in (0..8).rev() {
            if out.len() == size {
                break;
            }
            if group & (1 << bit) != 0 {
                out.push(slice(bytes, at, 1)?[0]);
                at += 1;
            } else {
                let pair = slice(bytes, at, 2)?;
                at += 2;
                let distance = (((pair[0] as usize & 15) << 8) | pair[1] as usize) + 1;
                let length = if pair[0] >> 4 == 0 {
                    let n = slice(bytes, at, 1)?[0] as usize + 0x12;
                    at += 1;
                    n
                } else {
                    (pair[0] >> 4) as usize + 2
                };
                if distance > out.len() || length > size - out.len() {
                    return Err("invalid Yaz0 backreference".into());
                }
                for _ in 0..length {
                    out.push(out[out.len() - distance]);
                }
            }
        }
    }
    Ok(out)
}

pub(super) struct SarcEntry {
    pub name: String,
    pub name_hex: Option<String>,
    pub hash: u32,
    pub attributes: u32,
    pub range: Range<usize>,
}
pub(super) struct Sarc {
    pub entries: Vec<SarcEntry>,
    pub big_endian: bool,
    pub multiplier: u32,
}
pub(super) fn sarc(bytes: &[u8]) -> Result<Sarc> {
    // Nintendo-File-Formats src/libs/sead/sarc.md:3,7-14: all header fields.
    let be = match slice(bytes, 6, 2)? {
        [0xfe, 0xff] => true,
        [0xff, 0xfe] => false,
        _ => return Err("invalid SARC BOM".into()),
    };
    if slice(bytes, 0, 4)? != b"SARC" || u16_at(bytes, 4, be)? != 0x14 {
        return Err("invalid SARC header".into());
    }
    let version = u16_at(bytes, 0x10, be)?;
    if version != 0x100 {
        return Err(format!("unsupported SARC version 0x{version:x}"));
    }
    let size = u32_at(bytes, 8, be)? as usize;
    if size != bytes.len() {
        return Err("SARC size mismatch".into());
    }
    let data = u32_at(bytes, 12, be)? as usize;
    // Same file:25-29: SFAT magic, size, node count, multiplier.
    if slice(bytes, 0x14, 4)? != b"SFAT" || u16_at(bytes, 0x18, be)? != 12 {
        return Err("invalid SFAT header".into());
    }
    let count = u16_at(bytes, 0x1a, be)? as usize;
    if count > 0x3fff {
        return Err("SARC node count exceeds format limit".into());
    }
    let multiplier = u32_at(bytes, 0x1c, be)?;
    let names = 0x20 + count * 16;
    // Same file:62,66-69: SFNT header and zero-terminated names.
    if slice(bytes, names, 4)? != b"SFNT" || u16_at(bytes, names + 4, be)? != 8 {
        return Err("invalid SFNT header".into());
    }
    if data < names + 8 || data > size {
        return Err("invalid SARC data offset".into());
    }
    let mut entries = Vec::with_capacity(count);
    for i in 0..count {
        // Same file:36-39,57-59: hash, attributes, relative data bounds;
        // low 24 name-offset bits in units of four, high byte collision ordinal.
        let at = 0x20 + i * 16;
        let hash = u32_at(bytes, at, be)?;
        let attributes = u32_at(bytes, at + 4, be)?;
        let (name, name_hex) = if attributes == 0 {
            (format!("#{hash:08x}"), None)
        } else {
            if attributes >> 24 == 0 {
                return Err("invalid SARC name attributes".into());
            }
            let start = names + 8 + (attributes as usize & 0xff_ffff) * 4;
            let tail = bytes
                .get(start..data)
                .ok_or("SARC name outside name table")?;
            let end = tail
                .iter()
                .position(|&b| b == 0)
                .ok_or("unterminated SARC name")?;
            let name = &tail[..end];
            (String::from_utf8_lossy(name).into_owned(), Some(hex(name)))
        };
        let start = u32_at(bytes, at + 8, be)? as usize;
        let end = u32_at(bytes, at + 12, be)? as usize;
        if start > end || end > size - data {
            return Err("invalid SARC file range".into());
        }
        entries.push(SarcEntry {
            name,
            name_hex,
            hash,
            attributes,
            range: data + start..data + end,
        });
    }
    Ok(Sarc {
        entries,
        big_endian: be,
        multiplier,
    })
}

pub(super) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub(super) struct Dictionary {
    pub name: String,
    pub id: Option<u32>,
    pub bytes: Vec<u8>,
}
#[derive(Default)]
pub(super) struct Dictionaries {
    pub items: Vec<Dictionary>,
    bytes: usize,
}
pub(super) fn dictionary_hint(name: &str, bytes: &[u8]) -> bool {
    // RFC 8878:5: formatted dictionary magic. Raw dictionary names are
    // caller hints, not a claim about a standard filename convention.
    bytes.starts_with(&[0x37, 0xa4, 0x30, 0xec]) || name.to_ascii_lowercase().ends_with(".zsdic")
}
impl Dictionaries {
    pub fn add(&mut self, name: &str, bytes: &[u8]) -> Result<()> {
        if self.items.iter().any(|d| d.bytes == bytes) {
            return Ok(());
        }
        if bytes.is_empty()
            || self.items.len() >= MAX_DICTIONARIES
            || bytes.len() > MAX_DICTIONARY_BYTES - self.bytes
        {
            return Err("dictionary resource limit".into());
        }
        // RFC 8878:5: dictionary ID follows magic. Library reads/validates it.
        let id = zstd::zstd_safe::get_dict_id_from_dict(bytes).map(u32::from);
        if bytes.starts_with(&[0x37, 0xa4, 0x30, 0xec]) && id.is_none() {
            return Err("invalid formatted dictionary ID".into());
        }
        self.bytes += bytes.len();
        self.items.push(Dictionary {
            name: name.into(),
            id,
            bytes: bytes.to_vec(),
        });
        Ok(())
    }
    fn decode_frame(&self, frame: &[u8], remaining: usize) -> Result<(Vec<u8>, Option<String>)> {
        // RFC 8878:3.1.1.1.3,6: a nonzero ID selects its dictionary;
        // an omitted ID requires external knowledge. Never guess among conflicts.
        let id = zstd::zstd_safe::get_dict_id_from_frame(frame).map(u32::from);
        let decode = |dict: &[u8]| -> Result<Vec<u8>> {
            let mut decoder = zstd::stream::read::Decoder::with_dictionary(frame, dict)
                .map_err(|e| e.to_string())?;
            decoder.window_log_max(28).map_err(|e| e.to_string())?;
            let mut out = Vec::new();
            decoder
                .take(remaining as u64 + 1)
                .read_to_end(&mut out)
                .map_err(|e| e.to_string())?;
            if out.len() > remaining {
                return Err("zstd exceeds resource limit".into());
            }
            Ok(out)
        };
        let candidates: Vec<_> = self
            .items
            .iter()
            .filter(|d| id.is_none() || d.id == id)
            .collect();
        if id.is_none() {
            match decode(&[]) {
                Ok(bytes) => return Ok((bytes, None)),
                Err(error) if candidates.is_empty() || error.contains("resource limit") => {
                    return Err(error)
                }
                Err(_) => {}
            }
        }
        let dict = match candidates.as_slice() {
            [dict] => dict,
            [] => return Err(format!("missing zstd dictionary {id:?}")),
            _ => return Err(format!("ambiguous zstd dictionary {id:?}")),
        };
        Ok((decode(&dict.bytes)?, Some(dict.name.clone())))
    }
    pub fn zstd(&self, bytes: &[u8]) -> Result<(Vec<u8>, Vec<String>)> {
        // RFC 8878:3.1: concatenated frames concatenate their output;
        // :3.1.2: skippable frames carry no decompressed content.
        let mut at = 0;
        let mut out = Vec::new();
        let mut used = Vec::new();
        while at < bytes.len() {
            let tail = &bytes[at..];
            if kind(tail) != Some("zstd") {
                return Err("invalid zstd frame magic".into());
            }
            let len = zstd::zstd_safe::find_frame_compressed_size(tail)
                .map_err(|e| zstd::zstd_safe::get_error_name(e).to_owned())?;
            if len == 0 {
                return Err("empty zstd frame".into());
            }
            let frame = slice(bytes, at, len)?;
            if frame[0] == 0x28 {
                let (decoded, dictionary) = self.decode_frame(frame, MAX_BYTES - out.len())?;
                out.extend_from_slice(&decoded);
                if let Some(d) = dictionary {
                    used.push(d);
                }
            }
            at += len;
        }
        Ok((out, used))
    }
}

pub(super) struct BinaryHeader {
    pub version: u32,
    pub size: usize,
}
pub(super) fn binary_header(bytes: &[u8]) -> Result<BinaryHeader> {
    // ShaderLibrary/Structs.cs:14-27: magic/padding, micro/minor/major,
    // BOM +0xc, total size +0x1c. Switch loader is LE (BnshLoader.cs:16).
    // BFRES Switch/ResFileParser.cs:16-27 uses the same fields;
    // Shared/ResFile.cs:368-372 splits Switch versions identically.
    if slice(bytes, 12, 2)? != [0xff, 0xfe] {
        return Err("unsupported binary container byte order".into());
    }
    let version = u32_at(bytes, 8, false)?;
    let size = u32_at(bytes, 0x1c, false)? as usize;
    if !(32..=MAX_BYTES).contains(&size) || size > bytes.len() {
        return Err("invalid binary container size".into());
    }
    Ok(BinaryHeader { version, size })
}

pub(super) struct StageCode {
    pub stage: Stage,
    pub descriptor: usize,
    pub control: Range<usize>,
    pub code: Range<usize>,
}
pub(super) struct Variation {
    pub offset: usize,
    pub binary_offset: usize,
    pub stages: Vec<StageCode>,
}
pub(super) struct Bnsh {
    pub header: BinaryHeader,
    pub variations: Vec<Result<Variation>>,
}
pub(super) fn bnsh(bytes: &[u8]) -> Result<Bnsh> {
    if kind(bytes) != Some("BNSH") {
        return Err("missing BNSH magic".into());
    }
    let header = binary_header(bytes)?;
    // BnshFile.cs:44-50 constructs 2.1.5. EPD Common nn::util (oldid=155),
    // Common nn__util.wiki:18-21 defines micro revisions as nonbreaking.
    // Admit the earlier micros of this schema, not other majors/minors or
    // future revisions. This compatibility inference is recorded in 0021.
    if !(0x0002_0100..=0x0002_0105).contains(&header.version) {
        return Err(format!("unsupported BNSH version 0x{:08x}", header.version));
    }
    let bytes = &bytes[..header.size];
    // BnshLoader.cs:18-26: 32-byte binary header +64 padding, then GRSC.
    // Structs.cs:246-258: GRSC magic, variation count +0x1c, pointer +0x20.
    if slice(bytes, 0x60, 4)? != b"grsc" {
        return Err("missing BNSH grsc section".into());
    }
    let count = u32_at(bytes, 0x7c, false)? as usize;
    let start = offset_at(bytes, 0x80)?;
    if count > 65536 {
        return Err("BNSH variation resource limit".into());
    }
    // Structs.cs:265-275: eight u64s per variation. Binary program at +0x10.
    range(bytes, start, count * 64)?;
    let variations = (0..count)
        .map(|i| {
            let at = start + i * 64;
            let binary = offset_at(bytes, at + 16)?;
            if binary == 0 {
                return Err("variation has no binary program".into());
            }
            range(bytes, binary, 176)?; // Structs.cs:281-313: full program header.
            let mut stages = Vec::new();
            // Structs.cs:287-292; BnshLoader.cs:53-58: six stage pointers,
            // ordered vertex, tess control, tess eval, geometry, fragment, compute.
            for (i, stage) in [
                Stage::Vertex,
                Stage::TessellationControl,
                Stage::TessellationEvaluation,
                Stage::Geometry,
                Stage::Fragment,
                Stage::Compute,
            ]
            .into_iter()
            .enumerate()
            {
                let descriptor = offset_at(bytes, binary + 8 + i * 8)?;
                if descriptor == 0 {
                    continue;
                } // BnshLoader.cs:83: absent stage.
                range(bytes, descriptor, 64)?;
                // BnshLoader.cs:87-102: reserved u64, control/code absolute u64
                // offsets, code/control u32 sizes, then 32 reserved bytes.
                let control = offset_at(bytes, descriptor + 8)?;
                let code = offset_at(bytes, descriptor + 16)?;
                let code_size = u32_at(bytes, descriptor + 24, false)? as usize;
                let control_size = u32_at(bytes, descriptor + 28, false)? as usize;
                stages.push(StageCode {
                    stage,
                    descriptor,
                    control: range(bytes, control, control_size)?,
                    code: range(bytes, code, code_size)?,
                });
            }
            Ok(Variation {
                offset: at,
                binary_offset: binary,
                stages,
            })
        })
        .collect();
    Ok(Bnsh { header, variations })
}

pub(super) fn wrapper(bytes: &[u8], kind: &str) -> Result<BinaryHeader> {
    if self::kind(bytes) != Some(kind) {
        return Err(format!("missing {kind} magic"));
    }
    let header = binary_header(bytes)?;
    // BFSHA reader BfshaLoader.cs:258-301 distinguishes v4,5,7,8 layouts,
    // :176-186,314-378 adds v9. BFRES ResFileParser.cs:31-42 adds v9/10;
    // Shared/ResFile.cs:42-47 explicitly constructs 5.0.3.
    // The v5 branch also covers v6 (BfshaLoader.cs:272-295).
    // BFRES v8: Shared/Models/Skeleton/Skeleton.cs:124-125 and Bone.cs:470-471.
    // BFRES 10.2.0: EPD Bfres.wiki (oldid=141):1; its root header is shared
    // with 10.0.0. Meshlet changes are outside embedded discovery (same
    // document's ResShape section). See 0021 for the exact field map.
    // BNTX 4.0.0/4.1.0: public 3DSkit doc/BNTX.md:6-11,16-25;
    // BntxLibrary/BntxFile.cs:550-578. Only the common header is interpreted.
    // Restrict to these revisions as project policy. Embedded discovery uses
    // self-contained child headers, not guessed model or material offsets.
    let supported = match kind {
        "BFSHA" => matches!(
            header.version,
            0x0004_0000 | 0x0005_0000 | 0x0006_0000 | 0x0007_0000 | 0x0008_0000 | 0x0009_0000
        ),
        "BFRES" => matches!(
            header.version,
            0x0005_0003 | 0x0008_0000 | 0x0009_0000 | 0x000a_0000 | 0x000a_0200
        ),
        "BNTX" => matches!(header.version, 0x0004_0000 | 0x0004_0100),
        _ => false,
    };
    if !supported {
        return Err(format!(
            "unsupported {kind} version 0x{:08x}",
            header.version
        ));
    }
    Ok(header)
}

#[cfg(test)]
pub(super) mod tests;
