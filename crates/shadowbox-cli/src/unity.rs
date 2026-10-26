//! Unity readers. References are pinned in docs/reference-sources.md
//! (AssetStudio) and docs/reference-sources.md (UnityPy).
//! Paths and line numbers below refer to that MIT reference, not game data.
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{self, Cursor, Read, Seek, SeekFrom, Write},
    ops::Range,
};

mod common_strings;
mod fixed_layout;
mod shared_trees;
mod sub_file;
use shared_trees::read_shared_metadata;
pub(super) use sub_file::SubFile;

// Local resource policy, not Unity format limits. Reject rather than truncate.
pub(super) const MAX_BYTES: usize = 512 * 1024 * 1024;
const MAX_ITEMS: usize = 1_000_000;
const MAX_VISITS: usize = 8_000_000;

fn invalid(reason: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, reason.into())
}
fn size(n: u64) -> io::Result<usize> {
    let n = usize::try_from(n).map_err(|_| invalid("Unity size exceeds address space"))?;
    if n > MAX_BYTES {
        return Err(invalid("Unity size exceeds resource limit"));
    }
    Ok(n)
}
fn range(start: usize, len: usize, total: usize) -> io::Result<Range<usize>> {
    let end = start
        .checked_add(len)
        .filter(|&n| n <= total)
        .ok_or_else(|| invalid("Unity range outside input"))?;
    Ok(start..end)
}

struct Reader<'a> {
    data: &'a [u8],
    at: usize,
    le: bool,
    origin: usize,
}
impl<'a> Reader<'a> {
    fn new(data: &'a [u8], le: bool) -> Self {
        Self {
            data,
            at: 0,
            le,
            origin: 0,
        }
    }
    fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
        let r = range(self.at, n, self.data.len())?;
        self.at = r.end;
        Ok(&self.data[r])
    }
    fn u8(&mut self) -> io::Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn u16(&mut self) -> io::Result<u16> {
        let b = self.take(2)?.try_into().unwrap();
        Ok(if self.le {
            u16::from_le_bytes(b)
        } else {
            u16::from_be_bytes(b)
        })
    }
    fn u32(&mut self) -> io::Result<u32> {
        let b = self.take(4)?.try_into().unwrap();
        Ok(if self.le {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        })
    }
    fn u64(&mut self) -> io::Result<u64> {
        let b = self.take(8)?.try_into().unwrap();
        Ok(if self.le {
            u64::from_le_bytes(b)
        } else {
            u64::from_be_bytes(b)
        })
    }
    fn count(&mut self) -> io::Result<usize> {
        let n = self.u32()? as usize;
        if n > MAX_ITEMS {
            return Err(invalid("Unity count exceeds resource limit"));
        }
        Ok(n)
    }
    fn align(&mut self, n: usize) -> io::Result<()> {
        let absolute = self
            .origin
            .checked_add(self.at)
            .ok_or_else(|| invalid("Unity alignment overflow"))?;
        self.take((n - absolute % n) % n)?;
        Ok(())
    }
    fn cstr(&mut self) -> io::Result<String> {
        let len = self.data[self.at..]
            .iter()
            .position(|&b| b == 0)
            .filter(|&n| n <= MAX_ITEMS)
            .ok_or_else(|| invalid("unterminated Unity string"))?;
        let s = std::str::from_utf8(self.take(len)?)
            .map_err(|_| invalid("invalid Unity UTF-8"))?
            .to_owned();
        self.take(1)?;
        Ok(s)
    }
    fn boolean(&mut self) -> io::Result<bool> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(invalid("invalid Unity boolean")),
        }
    }
}

struct LimitedOutput {
    bytes: Vec<u8>,
    limit: usize,
}
impl Write for LimitedOutput {
    fn write(&mut self, b: &[u8]) -> io::Result<usize> {
        range(self.bytes.len(), b.len(), self.limit)?;
        self.bytes.extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

// AssetStudio/BundleFile.cs:25-31,250-281,321-354: codec IDs and raw LZ4
// (LZ4HC uses the same decoder). SevenZipHelper.cs:37-46: Unity LZMA has
// five property bytes and an externally supplied uncompressed length.
fn decompress(data: &[u8], expected: usize, codec: u32) -> io::Result<Vec<u8>> {
    size(expected as u64)?;
    let output = match codec {
        0 => data.to_vec(),
        2 | 3 => lz4_flex::block::decompress(data, expected)
            .map_err(|e| invalid(format!("Unity LZ4: {e}")))?,
        1 => {
            let mut input = Cursor::new(data);
            let mut output = LimitedOutput {
                bytes: Vec::new(),
                limit: expected,
            };
            let options = lzma_rs::decompress::Options {
                unpacked_size: lzma_rs::decompress::UnpackedSize::UseProvided(Some(
                    expected as u64,
                )),
                memlimit: Some(MAX_BYTES),
                ..Default::default()
            };
            lzma_rs::lzma_decompress_with_options(&mut input, &mut output, &options)
                .map_err(|e| invalid(format!("Unity LZMA: {e}")))?;
            output.bytes
        }
        _ => return Err(invalid(format!("unsupported Unity compression {codec}"))),
    };
    if output.len() != expected {
        return Err(invalid("Unity decompressed size mismatch"));
    }
    Ok(output)
}

pub(super) struct BundleFile {
    pub name: String,
    pub flags: u32,
    pub range: Range<usize>,
}
pub(super) struct Bundle {
    pub version: u32,
    pub unity_version: String,
    pub revision: String,
    pub flags: u32,
    pub data_hash: [u8; 16],
    #[cfg(test)]
    pub data: Vec<u8>,
    pub files: Vec<BundleFile>,
}
impl Bundle {
    #[cfg(test)]
    pub fn read(bytes: &[u8]) -> io::Result<Self> {
        size(bytes.len() as u64)?;
        let mut output = LimitedOutput {
            bytes: Vec::new(),
            limit: MAX_BYTES,
        };
        let mut bundle = Self::read_stream(&mut Cursor::new(bytes), &mut output)?;
        bundle.data = output.bytes;
        Ok(bundle)
    }

    pub fn read_stream(
        input: &mut (impl Read + Seek),
        output: &mut impl Write,
    ) -> io::Result<Self> {
        let length = usize::try_from(input.seek(SeekFrom::End(0))?)
            .map_err(|_| invalid("UnityFS file exceeds address space"))?;
        input.seek(SeekFrom::Start(0))?;
        // AssetStudio/BundleFile.cs:72-75,93-99,218-223: fixed signature
        // and version, two cstrings, then size and block-info fields.
        let mut header = vec![0; 12];
        input.read_exact(&mut header)?;
        if !header.starts_with(b"UnityFS\0") {
            return Err(invalid("missing UnityFS signature"));
        }
        for _ in 0..2 {
            for count in 0..=MAX_ITEMS {
                let mut byte = [0];
                input.read_exact(&mut byte)?;
                header.push(byte[0]);
                if byte[0] == 0 {
                    break;
                }
                if count == MAX_ITEMS {
                    return Err(invalid("unterminated Unity string"));
                }
            }
        }
        let at = header.len();
        header.resize(at + 20, 0);
        input.read_exact(&mut header[at..])?;
        // Padding bytes have no interpreted fields; retain enough room for
        // Reader's position calculation, then bound the real stream below.
        header.resize(header.len() + 15, 0);
        // AssetStudio/EndianBinaryReader.cs:13-16: big endian by default.
        // BundleFile.cs:72-75,218-223: signature, version, strings, sizes, flags.
        let mut r = Reader::new(&header, false);
        if r.cstr()? != "UnityFS" {
            return Err(invalid("missing UnityFS signature"));
        }
        let version = r.u32()?;
        if !(6..=8).contains(&version) {
            return Err(invalid(format!("unsupported UnityFS version {version}")));
        }
        let unity_version = r.cstr()?;
        let revision = r.cstr()?;
        if r.u64()? != length as u64 {
            return Err(invalid("UnityFS file size mismatch"));
        }
        let compressed = size(u64::from(r.u32()?))?;
        let uncompressed = size(u64::from(r.u32()?))?;
        let flags = r.u32()?;
        // BundleFile.cs:11-15,233-247,311-314: combined directory, end info,
        // header alignment from v7, data padding. Reject unknown flag meanings.
        if flags & !0x3ff != 0 || flags & 0x40 == 0 {
            return Err(invalid("unsupported UnityFS flags or separate directory"));
        }
        let engine = fixed_layout::Version::parse(&revision).ok();
        // UnityPy/files/BundleFile.py:120-128; enums/BundleFile.py:12-26:
        // 0x200 meant encryption before these branch-specific padding fixes.
        // Never treat encrypted input as padding or uncompressed data.
        let padding_flag = engine.is_some_and(|v| match v.0 {
            2020 => (v.1, v.2) >= (3, 34),
            2021 => (v.1, v.2) >= (3, 2),
            2022 => (v.1, v.2) >= (1, 1),
            major => major > 2022,
        });
        if flags & 0x200 != 0 && !padding_flag {
            return Err(invalid(
                "unsupported UnityFS encryption or ambiguous 0x200 flag",
            ));
        }
        // UnityPy/files/BundleFile.py:133-137 also aligns format 6 for the
        // 2019.4.15+ branch. This is an explicit rule, not a padding probe.
        if version >= 7 || engine.is_some_and(|v| v.0 == 2019 && (v.1, v.2) >= (4, 15)) {
            r.align(16)?;
        }
        let at_end = flags & 0x80 != 0;
        let info_start = if at_end {
            length
                .checked_sub(compressed)
                .filter(|&n| n >= r.at)
                .ok_or_else(|| invalid("UnityFS end info overlaps header"))?
        } else {
            r.at
        };
        range(info_start, compressed, length)?;
        input.seek(SeekFrom::Start(info_start as u64))?;
        let mut packed_info = vec![0; compressed];
        input.read_exact(&mut packed_info)?;
        let info = decompress(&packed_info, uncompressed, flags & 0x3f)?;
        drop(packed_info);
        if !at_end {
            r.at =
                r.at.checked_add(compressed)
                    .ok_or_else(|| invalid("UnityFS size overflow"))?;
        }
        if flags & 0x200 != 0 {
            r.at =
                r.at.checked_add(15)
                    .ok_or_else(|| invalid("UnityFS alignment overflow"))?
                    & !15;
        }
        let data_end = if at_end { info_start } else { length };
        range(r.at, 0, data_end)?;
        // BundleFile.cs:285-307: hash, block count, unpacked/packed lengths,
        // u16 block flags, node count, i64 offset/size, u32 flags, path.
        let mut i = Reader::new(&info, false);
        let data_hash = i.take(16)?.try_into().unwrap();
        let mut blocks = Vec::new();
        let mut total = 0usize;
        for _ in 0..i.count()? {
            let unpacked = i.u32()? as usize;
            let packed = i.u32()? as usize;
            let block_flags = i.u16()?;
            if block_flags & !0x7f != 0 {
                return Err(invalid("unsupported UnityFS block flags"));
            }
            match block_flags & 0x3f {
                0 if packed == unpacked => (),
                0 => return Err(invalid("Unity decompressed size mismatch")),
                1..=3 => {
                    size(packed as u64)?;
                    size(unpacked as u64)?;
                }
                codec => return Err(invalid(format!("unsupported Unity compression {codec}"))),
            }
            total = total
                .checked_add(unpacked)
                .ok_or_else(|| invalid("UnityFS size overflow"))?;
            blocks.push((unpacked, packed, block_flags));
        }
        let mut files = Vec::new();
        for _ in 0..i.count()? {
            let offset = usize::try_from(i.u64()?)
                .map_err(|_| invalid("UnityFS directory offset exceeds address space"))?;
            let len = usize::try_from(i.u64()?)
                .map_err(|_| invalid("UnityFS directory length exceeds address space"))?;
            let flags = i.u32()?;
            let name = i.cstr()?;
            files.push(BundleFile {
                name,
                flags,
                range: range(offset, len, total)?,
            });
        }
        if i.at != info.len() {
            return Err(invalid("UnityFS trailing block info"));
        }
        input.seek(SeekFrom::Start(r.at as u64))?;
        for (unpacked, packed, block_flags) in blocks {
            range(r.at, packed, data_end)?;
            // AssetStudio/BundleFile.cs:317-357: directory ranges address
            // the concatenated decoded blocks. Spool one block at a time;
            // the total archive/directory size has no in-memory buffer cap.
            if block_flags & 0x3f == 0 {
                // BundleFile.cs:321-327: codec 0 is a direct block copy.
                // Stream that copy, including blocks above the buffer limit.
                if io::copy(&mut input.take(packed as u64), output)? != packed as u64 {
                    return Err(io::Error::new(
                        io::ErrorKind::UnexpectedEof,
                        "truncated UnityFS block",
                    ));
                }
            } else {
                let mut bytes = vec![0; packed];
                input.read_exact(&mut bytes)?;
                output.write_all(&decompress(
                    &bytes,
                    unpacked,
                    u32::from(block_flags) & 0x3f,
                )?)?;
            }
            r.at += packed;
        }
        if r.at != data_end {
            return Err(invalid("UnityFS unaccounted block bytes"));
        }
        Ok(Self {
            version,
            unity_version,
            revision,
            flags,
            data_hash,
            #[cfg(test)]
            data: Vec::new(),
            files,
        })
    }
}

#[derive(Clone)]
struct Node {
    ty: String,
    name: String,
    level: u8,
    byte_size: i32,
    align: bool,
    end: usize,
    shared: Option<u64>,
}
struct SerializedType {
    class: i32,
    nodes: Vec<Node>,
    references: Vec<[u8; 16]>,
}
fn tree_string(strings: &[u8], offset: u32) -> io::Result<String> {
    // AssetStudio/SerializedFile.cs:354-367; CommonString.cs:9-117.
    if offset & 0x8000_0000 != 0 {
        return common_strings::get(offset & 0x7fff_ffff)
            .map(str::to_owned)
            .ok_or_else(|| invalid("unknown Unity common string offset"));
    }
    let mut r = Reader::new(strings, false);
    r.take(offset as usize)?;
    r.cstr()
}
struct TreeBudget {
    nodes: usize,
    strings: usize,
}
fn read_type(
    r: &mut Reader<'_>,
    version: u32,
    budget: &mut TreeBudget,
    has_tree: bool,
) -> io::Result<SerializedType> {
    read_type_kind(r, version, budget, has_tree, false)
}

fn read_type_kind(
    r: &mut Reader<'_>,
    version: u32,
    budget: &mut TreeBudget,
    has_tree: bool,
    is_reference: bool,
) -> io::Result<SerializedType> {
    // UnityPy/files/SerializedFile.py:121-138: reference types also carry
    // a script GUID when their script index is nonnegative.
    let class = r.u32()? as i32;
    r.boolean()?;
    let script_index = r.u16()? as i16;
    if class == 114 || (is_reference && script_index >= 0) {
        r.take(16)?;
    }
    r.take(16)?;
    let (nodes, references) = if !has_tree {
        (Vec::new(), Vec::new())
    } else if version >= 23 {
        read_tree_handler(r, budget)?.1
    } else {
        (read_tree_blob(r, version, budget)?, Vec::new())
    };
    // UnityPy/files/SerializedFile.py:140-156: dependencies/names only
    // accompany a supplied tree. Reference types use three cstrings.
    if has_tree && version >= 21 {
        if is_reference {
            r.cstr()?;
            r.cstr()?;
            r.cstr()?;
        } else {
            let n = r.count()?;
            r.take(n * 4)?;
        }
    }
    if version < 26 && nodes.iter().any(|n| n.shared.is_some()) {
        return Err(invalid(
            "unsupported Unity shared subtree before version 26",
        ));
    }
    Ok(SerializedType {
        class,
        nodes,
        references,
    })
}

type TreeInfo = (Vec<Node>, Vec<[u8; 16]>);

fn read_tree_handler(
    r: &mut Reader<'_>,
    budget: &mut TreeBudget,
) -> io::Result<([u8; 16], TreeInfo)> {
    // UnityPy/helpers/TypeTreeNode.py:349-356,376-393: hash, sized mhtt
    // payload, node format version, node blob, reference count/hash table.
    let hash = r.take(16)?.try_into().unwrap();
    let len = size(u64::from(r.u32()?))?;
    if len == 0 {
        return Ok((hash, (Vec::new(), Vec::new())));
    }
    let origin = r.origin + r.at;
    let mut tree = Reader::new(r.take(len)?, r.le);
    tree.origin = origin;
    if tree.take(4)? != b"mhtt" {
        return Err(invalid("invalid Unity type-tree blob magic"));
    }
    let node_version = tree.u32()?;
    if !(17..=26).contains(&node_version) {
        return Err(invalid("unsupported Unity type-tree blob version"));
    }
    let nodes = read_tree_blob(&mut tree, node_version, budget)?;
    let count = tree.count()?;
    let mut references = Vec::new();
    for _ in 0..count {
        references.push(tree.take(16)?.try_into().unwrap());
    }
    if tree.at != tree.data.len() {
        return Err(invalid("Unity type-tree wrapper size mismatch"));
    }
    Ok((hash, (nodes, references)))
}

fn read_tree_blob(
    r: &mut Reader<'_>,
    version: u32,
    budget: &mut TreeBudget,
) -> io::Result<Vec<Node>> {
    // SerializedFile.cs:323-342: node count, string size, node version, level,
    // type flags, type/name offsets, byte size, index, meta flags, v19+ ref hash.
    let count = r.count()?;
    budget.nodes = budget
        .nodes
        .checked_sub(count)
        .ok_or_else(|| invalid("Unity total tree nodes exceed resource limit"))?;
    let string_size = size(u64::from(r.u32()?))?;
    let mut raw = Vec::new();
    for _ in 0..count {
        r.u16()?;
        let level = r.u8()?;
        let type_flags = r.u8()?;
        if type_flags & !0x21 != 0 || (type_flags & 0x20 != 0 && version < 19) {
            return Err(invalid("unsupported Unity type-tree node flags"));
        }
        let ty = r.u32()?;
        let name = r.u32()?;
        let byte_size = r.u32()? as i32;
        r.u32()?;
        let meta = r.u32()?;
        // UnityPy/helpers/TypeTreeNode.py:316-318;
        // files/SerializedFile.py:364-384: flag 0x20 makes this u64 an
        // index into the wrapper's 16-byte reference hash table.
        let reference = if version >= 19 { r.u64()? } else { 0 };
        raw.push((
            level,
            ty,
            name,
            byte_size,
            meta,
            (type_flags & 0x20 != 0).then_some(reference),
        ));
    }
    let strings = r.take(string_size)?;
    let mut nodes: Vec<Node> = Vec::new();
    let mut stack: Vec<usize> = Vec::new();
    for (index, (level, ty, name, byte_size, meta, shared)) in raw.into_iter().enumerate() {
        // TypeTreeHelper.cs:308-323: a subtree ends at the next same/lower level.
        while stack.last().is_some_and(|&j| nodes[j].level >= level) {
            nodes[stack.pop().unwrap()].end = index;
        }
        if (index == 0 && level != 0)
            || (index != 0 && stack.last().is_none_or(|&j| nodes[j].level + 1 != level))
        {
            return Err(invalid("invalid Unity type-tree hierarchy"));
        }
        let ty = tree_string(strings, ty)?;
        let name = tree_string(strings, name)?;
        budget.strings = budget
            .strings
            .checked_sub(ty.len() + name.len())
            .ok_or_else(|| invalid("Unity decoded tree strings exceed resource limit"))?;
        nodes.push(Node {
            ty,
            name,
            level,
            byte_size,
            align: meta & 0x4000 != 0,
            end: count,
            shared,
        });
        stack.push(index);
    }
    Ok(nodes)
}

pub(super) struct ShaderObject<'a> {
    pub path_id: i64,
    data: Option<&'a [u8]>,
    length: usize,
    nodes: &'a [Node],
    le: bool,
    origin: usize,
    unity_version: &'a str,
    target: i32,
    has_tree: bool,
}
pub(super) struct SerializedFile<'a> {
    pub version: u32,
    pub unity_version: String,
    pub target: i32,
    types: Vec<SerializedType>,
    objects: Vec<(i64, Range<usize>, usize)>,
    bytes: Option<&'a [u8]>,
    le: bool,
    has_tree: bool,
}
impl<'a> SerializedFile<'a> {
    #[cfg(test)]
    pub fn read(bytes: &'a [u8]) -> io::Result<Self> {
        let mut file = Self::read_stream(&mut Cursor::new(bytes))?;
        file.bytes = Some(bytes);
        Ok(file)
    }

    pub fn read_stream(input: &mut (impl Read + Seek)) -> io::Result<Self> {
        let length = input.seek(SeekFrom::End(0))?;
        let length =
            usize::try_from(length).map_err(|_| invalid("Unity file exceeds address space"))?;
        input.seek(SeekFrom::Start(0))?;
        let mut header = [0; 48];
        input.read_exact(&mut header[..20])?;
        let header_version = u32::from_be_bytes(header[8..12].try_into().unwrap());
        if header_version >= 22 {
            input.read_exact(&mut header[20..])?;
        }
        // AssetStudio/SerializedFile.cs:43-52,60-71: BE header, endian byte,
        // reserved bytes, v22+ extended sizes and unknown i64.
        // UnityPy/files/SerializedFile.py:278-285; ObjectReader.py:81-89
        // retain the extended header and object starts for all versions >=22.
        // No interpretation of reserved/unknown bytes is needed to locate metadata and objects.
        let mut r = Reader::new(&header, false);
        let old_metadata = r.u32()?;
        let old_file_size = r.u32()?;
        let version = r.u32()?;
        if !(17..=26).contains(&version) {
            return Err(invalid(format!(
                "unsupported Unity serialized version {version}"
            )));
        }
        let old_data_offset = r.u32()?;
        let le = !r.boolean()?;
        r.take(3)?;
        let (metadata, file_size, data_offset) = if version >= 22 {
            let metadata = size(u64::from(r.u32()?))?;
            let file_size = usize::try_from(r.u64()?)
                .map_err(|_| invalid("Unity file exceeds address space"))?;
            let data_offset = usize::try_from(r.u64()?)
                .map_err(|_| invalid("Unity offset exceeds address space"))?;
            r.u64()?;
            (metadata, file_size, data_offset)
        } else {
            (
                size(u64::from(old_metadata))?,
                old_file_size as usize,
                old_data_offset as usize,
            )
        };
        if file_size != length || data_offset > file_size {
            return Err(invalid("Unity serialized file size/offset mismatch"));
        }
        let meta_range = range(r.at, metadata, data_offset)?;
        // Read only bounded metadata. Object ranges are seek positions, not
        // buffer sizes, so the file/data offset can exceed MAX_BYTES.
        let mut metadata_bytes = vec![0; metadata];
        input.read_exact(&mut metadata_bytes)?;
        let mut m = Reader::new(&metadata_bytes, le);
        m.origin = meta_range.start;
        // SerializedFile.cs:73-97: Unity version, target, tree-present, types.
        let unity_version = m.cstr()?;
        let target = m.u32()? as i32;
        let has_tree = m.boolean()?;
        let mut types = Vec::new();
        let mut budget = TreeBudget {
            nodes: MAX_ITEMS,
            strings: MAX_BYTES,
        };
        for _ in 0..m.count()? {
            types.push(read_type(&mut m, version, &mut budget, has_tree)?);
        }
        // SerializedFile.cs:105-144; UnityPy/files/ObjectReader.py:81-89:
        // 4-byte-aligned i64 path ID, v22+ i64
        // start (earlier u32), u32 size, type index giving the class ID.
        let mut objects = Vec::new();
        for _ in 0..m.count()? {
            m.align(4)?;
            let id = m.u64()? as i64;
            let start = if version >= 22 {
                usize::try_from(m.u64()?)
                    .map_err(|_| invalid("Unity object offset exceeds address space"))?
            } else {
                m.u32()? as usize
            };
            let len = m.u32()? as usize;
            let ty = m.u32()? as usize;
            if ty >= types.len() {
                return Err(invalid("Unity object type index out of range"));
            }
            let start = data_offset
                .checked_add(start)
                .ok_or_else(|| invalid("Unity object offset overflow"))?;
            objects.push((id, range(start, len, length)?, ty));
        }
        if version >= 26 {
            read_shared_metadata(&mut m, version, has_tree, &mut types, &mut budget)?;
        }
        // Before v26 the scripts/external/reference tail is not needed for
        // extraction (AssetStudio/SerializedFile.cs:162-215).
        Ok(Self {
            version,
            unity_version,
            target,
            types,
            objects,
            bytes: None,
            le,
            has_tree,
        })
    }
    pub fn shaders(&self) -> impl Iterator<Item = ShaderObject<'_>> {
        // AssetStudio/ClassIDType.cs:37: Shader = 48.
        self.objects
            .iter()
            .filter(|(_, _, ty)| self.types[*ty].class == 48)
            .map(|(id, range, ty)| ShaderObject {
                path_id: *id,
                data: self.bytes.map(|bytes| &bytes[range.clone()]),
                length: range.len(),
                nodes: &self.types[*ty].nodes,
                le: self.le,
                origin: range.start,
                unity_version: &self.unity_version,
                target: self.target,
                has_tree: self.has_tree,
            })
    }
}

#[derive(Debug)]
enum Field {
    UInt(u32),
    Array(Vec<Field>),
    Bytes(Vec<u8>),
    Text(String),
    Skipped,
}

// TypeTreeHelper.cs:185-305: primitives, aligned strings, TypelessData,
// Array size/data, class children, and meta flag 0x4000 alignment. Traverse
// unwanted fields without materializing them. Budget visits to bound arrays
// whose element schema could consume zero bytes.
fn value(
    r: &mut Reader<'_>,
    nodes: &[Node],
    at: usize,
    capture: bool,
    visits: &mut usize,
) -> io::Result<Field> {
    *visits = visits
        .checked_sub(1)
        .ok_or_else(|| invalid("Unity type-tree traversal exceeds resource limit"))?;
    let node = &nodes[at];
    let mut align = node.align;
    let field = if node.ty == "string" {
        let n = size(u64::from(r.u32()?))?;
        let bytes = r.take(n)?;
        r.align(4)?;
        if capture {
            Field::Text(
                std::str::from_utf8(bytes)
                    .map_err(|_| invalid("invalid Shader name UTF-8"))?
                    .to_owned(),
            )
        } else {
            Field::Skipped
        }
    } else if node.ty == "TypelessData" {
        let n = size(u64::from(r.u32()?))?;
        let bytes = r.take(n)?;
        if capture {
            Field::Bytes(bytes.to_vec())
        } else {
            Field::Skipped
        }
    } else if at + 1 < node.end && nodes[at + 1].ty == "Array" {
        let array = &nodes[at + 1];
        align |= array.align;
        if at + 3 >= node.end
            || nodes[at + 2].name != "size"
            || nodes[at + 2].ty != "int"
            || nodes[at + 3].name != "data"
            || nodes[at + 2].end != at + 3
            || nodes[at + 3].end != array.end
            || array.end != node.end
        {
            return Err(invalid("unsupported Unity Array schema"));
        }
        let element = &nodes[at + 3];
        let byte_array = element.ty == "UInt8" && !element.align && element.end == at + 4;
        if byte_array && element.byte_size != 1 {
            return Err(invalid("Unity byte array size disagrees with type tree"));
        }
        let n = if byte_array {
            size(u64::from(r.u32()?))?
        } else {
            r.count()?
        };
        // Shader.cs:996: compressedBlob is a byte array. Avoid one allocation
        // per byte and allow blobs larger than the general element-count cap.
        if byte_array {
            let bytes = r.take(n)?;
            if capture {
                Field::Bytes(bytes.to_vec())
            } else {
                Field::Skipped
            }
        } else {
            let mut elements = Vec::new();
            for _ in 0..n {
                let v = value(r, nodes, at + 3, capture, visits)?;
                if capture {
                    elements.push(v);
                }
            }
            if capture {
                Field::Array(elements)
            } else {
                Field::Skipped
            }
        }
    } else if at + 1 < node.end {
        let mut child = at + 1;
        while child < node.end {
            value(r, nodes, child, false, visits)?;
            child = nodes[child].end;
        }
        if capture {
            return Err(invalid("unsupported captured Shader field schema"));
        }
        Field::Skipped
    } else {
        let width = match node.ty.as_str() {
            "bool" | "SInt8" | "UInt8" => 1,
            "short" | "SInt16" | "UInt16" | "unsigned short" => 2,
            "int" | "SInt32" | "UInt32" | "unsigned int" | "Type*" | "float" => 4,
            "long long" | "SInt64" | "UInt64" | "unsigned long long" | "FileSize" | "double" => 8,
            // Opaque fixed-sized fields can be skipped using the supplied tree.
            _ if !capture && node.byte_size >= 0 => node.byte_size as usize,
            _ => return Err(invalid(format!("unsupported Unity leaf type {}", node.ty))),
        };
        if node.byte_size >= 0 && node.byte_size as usize != width {
            return Err(invalid("Unity primitive size disagrees with type tree"));
        }
        if capture && matches!(node.ty.as_str(), "UInt32" | "unsigned int") {
            Field::UInt(r.u32()?)
        } else {
            r.take(width)?;
            if capture {
                return Err(invalid("Shader field is not UInt32"));
            }
            Field::Skipped
        }
    };
    if align {
        r.align(4)?;
    }
    Ok(field)
}

fn uints(field: &Field) -> io::Result<Vec<u32>> {
    let Field::Array(values) = field else {
        return Err(invalid("Shader field is not a UInt32 array"));
    };
    values
        .iter()
        .map(|v| {
            if let Field::UInt(n) = v {
                Ok(*n)
            } else {
                Err(invalid("Shader array element is not UInt32"))
            }
        })
        .collect()
}
fn rows(field: &Field) -> io::Result<Vec<Vec<u32>>> {
    let Field::Array(values) = field else {
        return Err(invalid("Shader field is not an array"));
    };
    if values.iter().all(|v| matches!(v, Field::UInt(_))) {
        return Ok(uints(field)?.into_iter().map(|n| vec![n]).collect());
    }
    values.iter().map(uints).collect()
}

pub(super) struct ShaderBlob {
    pub platform: u32,
    pub segment: usize,
    pub offset: u32,
    pub compressed_length: u32,
    pub decompressed_length: u32,
    pub bytes: Vec<u8>,
}
pub(super) struct Shader {
    pub name: String,
    pub blobs: Vec<ShaderBlob>,
}
impl ShaderObject<'_> {
    fn tree_fields<'a>(&'a self, r: &mut Reader<'_>) -> io::Result<BTreeMap<&'a str, Field>> {
        let mut visits = MAX_VISITS;
        let mut fields = BTreeMap::new();
        let mut child = 1;
        while child < self.nodes.len() {
            let node = &self.nodes[child];
            let capture = matches!(
                node.name.as_str(),
                "m_Name"
                    | "platforms"
                    | "offsets"
                    | "compressedLengths"
                    | "decompressedLengths"
                    | "compressedBlob"
            );
            let v = value(r, self.nodes, child, capture, &mut visits)?;
            if capture && fields.insert(node.name.as_str(), v).is_some() {
                return Err(invalid("duplicate Shader field"));
            }
            child = node.end;
        }
        if self.nodes[0].align {
            r.align(4)?;
        }
        if r.at != r.data.len() {
            return Err(invalid("Shader object size disagrees with type tree"));
        }
        Ok(fields)
    }
    pub fn extract(&self) -> io::Result<Shader> {
        self.extract_bytes(
            self.data
                .ok_or_else(|| invalid("Shader needs seekable input"))?,
        )
    }
    pub fn extract_stream(&self, input: &mut (impl Read + Seek)) -> io::Result<Shader> {
        if self.data.is_some() {
            return self.extract();
        }
        let length = size(self.length as u64)?;
        let mut bytes = vec![0; length];
        input.seek(SeekFrom::Start(self.origin as u64))?;
        input.read_exact(&mut bytes)?;
        self.extract_bytes(&bytes)
    }
    fn extract_bytes(&self, bytes: &[u8]) -> io::Result<Shader> {
        if self.has_tree && self.nodes.len() < 2 {
            return Err(invalid("Shader object missing type tree"));
        }
        let mut r = Reader::new(bytes, self.le);
        r.origin = self.origin;
        let fields = if self.has_tree {
            self.tree_fields(&mut r)?
        } else {
            fixed_layout::fields(&mut r, self.unity_version, self.target).map_err(|e| {
                invalid(format!(
                    "Shader layout parsing failed at object byte {} (serialized byte {}), Unity {}: {e}",
                    r.at, self.origin + r.at, self.unity_version
                ))
            })?
        };
        let get = |name| {
            fields
                .get(name)
                .ok_or_else(|| invalid(format!("Shader missing {name}")))
        };
        let Field::Text(name) = get("m_Name")? else {
            return Err(invalid("Shader name is not a string"));
        };
        // Shader.cs:983-996: platforms and one-dimensional (pre-2019.3) or
        // nested lengths/offsets, followed by compressedBlob. The type tree
        // or the explicitly selected fixed layout supplies the array shape.
        let platforms = uints(get("platforms")?)?;
        let offsets = rows(get("offsets")?)?;
        let compressed = rows(get("compressedLengths")?)?;
        let decompressed = rows(get("decompressedLengths")?)?;
        let Field::Bytes(blob) = get("compressedBlob")? else {
            return Err(invalid("Shader compressedBlob is not bytes"));
        };
        if [offsets.len(), compressed.len(), decompressed.len()]
            .iter()
            .any(|&n| n != platforms.len())
        {
            return Err(invalid("Shader platform array length mismatch"));
        }
        let mut blobs = Vec::new();
        let mut total = 0usize;
        for (p, &platform) in platforms.iter().enumerate() {
            if offsets[p].len() != compressed[p].len() || offsets[p].len() != decompressed[p].len()
            {
                return Err(invalid("Shader segment array length mismatch"));
            }
            for segment in 0..offsets[p].len() {
                let offset = offsets[p][segment];
                let compressed_length = compressed[p][segment];
                let decompressed_length = decompressed[p][segment];
                let expected = size(u64::from(decompressed_length))?;
                total = size(
                    total
                        .checked_add(expected)
                        .ok_or_else(|| invalid("Shader blob size overflow"))?
                        as u64,
                )?;
                // AssetStudioUtility/ShaderConverter.cs:41-47: LZ4 per platform/segment,
                // with offset, compressed length and decompressed length.
                let bytes = decompress(
                    &blob[range(offset as usize, compressed_length as usize, blob.len())?],
                    expected,
                    2,
                )?;
                blobs.push(ShaderBlob {
                    platform,
                    segment,
                    offset,
                    compressed_length,
                    decompressed_length,
                    bytes,
                });
            }
        }
        Ok(Shader {
            name: name.clone(),
            blobs,
        })
    }
}

pub(super) fn serialized_name(name: &str) -> bool {
    let name = name
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(name)
        .to_ascii_lowercase();
    name.ends_with(".assets") || name == "globalgamemanagers" || name.starts_with("level")
}
// AssetStudio/SerializedFile.cs:43-52,60-65; UnityPy/files/SerializedFile.py:281-285:
// content hint for bundle files
// with arbitrary names. This only selects the strict parser, never a layout.
#[cfg(test)]
pub(super) fn serialized_hint(bytes: &[u8]) -> bool {
    serialized_header_hint(bytes, bytes.len() as u64)
}
pub(super) fn serialized_header_hint(bytes: &[u8], len: u64) -> bool {
    if bytes.len() < 20 || bytes[16] > 1 {
        return false;
    }
    let word = |at| u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap());
    if word(0) != 0 && u64::from(word(4)) == len && word(12) >= 20 && u64::from(word(12)) <= len {
        return true;
    }
    // Extended-header shape is a discovery hint even for future versions.
    // SerializedFile::read accepts only the referenced versions 17..=26.
    bytes.len() >= 48
        && word(8) >= 22
        && word(20) != 0
        && u64::from_be_bytes(bytes[24..32].try_into().unwrap()) == len
        && (48..=len).contains(&u64::from_be_bytes(bytes[32..40].try_into().unwrap()))
}

#[cfg(test)]
pub(super) mod tests;
