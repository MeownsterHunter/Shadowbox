//! Original IoStore reader. Pinned, permitted format references and limits:
//! docs/reference-sources.md `retoc` citations are relative to that repo.
use std::collections::HashSet;
use std::fs::{self, File};
use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use super::pak::AesKey;
pub(super) mod shaders;

// retoc/src/lib.rs:1214-1215.
pub(super) const MAGIC: &[u8; 16] = b"-==--==--==--==-";
// Local resource policy. Oversized extents fail; none are truncated for scanning.
const MAX_TOC: u64 = 256 * 1024 * 1024;
pub(super) const MAX_CHUNK: u64 = 512 * 1024 * 1024;
const MAX_BLOCK: u32 = 16 * 1024 * 1024;
const MAX_ITEMS: u32 = 1_000_000;
const MAX_STRING: u32 = 1024 * 1024;

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}
fn bytes<const N: usize>(r: &mut impl Read) -> io::Result<[u8; N]> {
    let mut b = [0; N];
    r.read_exact(&mut b)?;
    Ok(b)
}
// retoc/src/ser.rs:200-227: primitive fields are little endian.
fn u32r(r: &mut impl Read) -> io::Result<u32> {
    Ok(u32::from_le_bytes(bytes(r)?))
}
fn count(r: &mut impl Read) -> io::Result<usize> {
    bounded(u32r(r)?, MAX_ITEMS)
}
fn bounded(n: u32, max: u32) -> io::Result<usize> {
    if n > max {
        return Err(invalid("IoStore count exceeds resource limit"));
    }
    Ok(n as usize)
}
fn skip(r: &mut Cursor<&[u8]>, n: usize) -> io::Result<()> {
    let end = (r.position() as usize)
        .checked_add(n)
        .filter(|&end| end <= r.get_ref().len())
        .ok_or_else(|| invalid("truncated IoStore table"))?;
    r.set_position(end as u64);
    Ok(())
}
// retoc/src/ser.rs:95-98,274-283: signed FString length, NUL termination,
// negative lengths count UTF-16 units. Require well-formed names.
fn string(r: &mut Cursor<&[u8]>) -> io::Result<String> {
    let n = i32::from_le_bytes(bytes(r)?);
    bounded(n.unsigned_abs(), MAX_STRING)?;
    if n == 0 {
        return Ok(String::new());
    }
    let size = n.unsigned_abs() as usize * if n < 0 { 2 } else { 1 };
    let start = r.position() as usize;
    skip(r, size)?;
    let data = &r.get_ref()[start..start + size];
    if n < 0 {
        let mut words: Vec<_> = data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes(*b))
            .collect();
        if words.pop() != Some(0) || words.contains(&0) {
            return Err(invalid("invalid IoStore UTF-16 terminator"));
        }
        String::from_utf16(&words).map_err(|_| invalid("invalid IoStore UTF-16"))
    } else {
        if data.last() != Some(&0) || data[..size - 1].contains(&0) {
            return Err(invalid("invalid IoStore string terminator"));
        }
        String::from_utf8(data[..size - 1].to_vec()).map_err(|_| invalid("invalid IoStore UTF-8"))
    }
}
fn array<'a, T>(
    r: &mut Cursor<&'a [u8]>,
    mut read: impl FnMut(&mut Cursor<&'a [u8]>) -> io::Result<T>,
) -> io::Result<Vec<T>> {
    // retoc/src/ser.rs:111-114: u32 count followed by elements.
    let n = count(r)?;
    (0..n).map(|_| read(r)).collect()
}
pub(super) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

pub(super) struct Chunk {
    // retoc/src/lib.rs:765-776,860-863: 12 raw bytes; type at byte 11.
    pub id: [u8; 12],
    pub offset: u64,
    pub length: u64,
    pub name: Option<String>,
    pub hash: String,
    pub flags: u8,
}
#[derive(Clone, Copy)]
struct Block {
    offset: u64,
    compressed: u32,
    uncompressed: u32,
    method: u8,
}
pub(super) struct Toc {
    pub version: u8,
    pub container_id: String,
    pub encryption_guid: String,
    pub flags: u8,
    // Retain the complete header, including opaque reserved bytes, for reporting.
    pub header: [u8; 144],
    pub mount: String,
    pub chunks: Vec<Chunk>,
    pub methods: Vec<String>,
    pub partition_count: u32,
    pub partition_size: u64,
    pub block_size: u32,
    blocks: Vec<Block>,
}
impl Toc {
    pub fn open(path: &Path, key: Option<&AesKey>) -> io::Result<Self> {
        let file = File::open(path)?;
        if file.metadata()?.len() > MAX_TOC {
            return Err(invalid("IoStore TOC exceeds resource limit"));
        }
        let mut data = Vec::new();
        file.take(MAX_TOC + 1).read_to_end(&mut data)?;
        if data.len() as u64 > MAX_TOC {
            return Err(invalid("IoStore TOC exceeds resource limit"));
        }
        Self::parse(&data, key)
    }

    fn parse(data: &[u8], key: Option<&AesKey>) -> io::Result<Self> {
        let mut r = Cursor::new(data);
        // retoc/src/lib.rs:206-237,1189-1215: the complete 144-byte header.
        let header = bytes::<144>(&mut r)?;
        if &header[..16] != MAGIC {
            return Err(invalid("unsupported or missing IoStore magic"));
        }
        let version = header[16];
        // retoc/src/lib.rs:1164-1174. Later revisions stay unsupported.
        if !(1..=8).contains(&version) {
            return Err(invalid(format!(
                "unsupported IoStore TOC version {version}"
            )));
        }
        let h32 = |at| u32::from_le_bytes(header[at..at + 4].try_into().unwrap());
        let h64 = |at| u64::from_le_bytes(header[at..at + 8].try_into().unwrap());
        if h32(20) != 144 || h32(32) != 12 {
            return Err(invalid(
                "unsupported IoStore header or compression entry size",
            ));
        }
        let entries = bounded(h32(24), MAX_ITEMS)?;
        let block_count = bounded(h32(28), MAX_ITEMS)?;
        let method_count = bounded(h32(36), 255)?;
        let method_length = bounded(h32(40), MAX_STRING)?;
        if method_count != 0 && method_length == 0 {
            return Err(invalid("empty IoStore compression name width"));
        }
        let block_size = h32(44);
        if block_size == 0 || block_size > MAX_BLOCK {
            return Err(invalid(
                "IoStore compression block size exceeds resource limit or is zero",
            ));
        }
        let directory_size = h32(48) as usize;
        let flags = header[80];
        // retoc/src/lib.rs:1128-1132; CUE4Parse FIoStoreTocHeader.cs:27-34.
        // On-demand metadata and unknown flag families are deliberately unsupported.
        if flags & !15 != 0 {
            return Err(invalid(
                "unsupported IoStore container flags (including OnDemand)",
            ));
        }
        if directory_size != 0 && (version < 2 || flags & 8 == 0) {
            return Err(invalid(
                "IoStore directory index without indexed flag/version",
            ));
        }
        let encrypted = flags & 2 != 0;
        if encrypted && key.is_none() {
            return Err(invalid(
                "IoStore container is encrypted; supply --pak-aes-key-file or --pak-aes-key-env",
            ));
        }
        // CUE4Parse FIoStoreTocResource.cs:70-74: pre-PartitionSize fields
        // are reserved, so use a single unbounded partition for versions 1/2.
        let (partition_count, partition_size) = if version < 3 {
            (1, u64::MAX)
        } else {
            (h32(52), h64(88))
        };
        if partition_count == 0 || partition_count > 1024 || partition_size == 0 {
            return Err(invalid("invalid or oversized IoStore partition layout"));
        }
        // retoc/src/lib.rs:272-278,467-475: IDs, logical ranges, hash
        // accelerator tables, compression blocks, methods, signatures, directory, meta.
        let ids = (0..entries)
            .map(|_| bytes::<12>(&mut r))
            .collect::<io::Result<Vec<_>>>()?;
        let mut chunks = Vec::with_capacity(entries);
        let mut seen = HashSet::new();
        for id in ids {
            if !seen.insert(id) {
                return Err(invalid("duplicate IoStore chunk ID"));
            }
            let raw = bytes::<10>(&mut r)?;
            // retoc/src/lib.rs:934-964: two big-endian 40-bit values.
            let offset = raw[..5].iter().fold(0u64, |n, &b| (n << 8) | u64::from(b));
            let length = raw[5..].iter().fold(0u64, |n, &b| (n << 8) | u64::from(b));
            chunks.push(Chunk {
                id,
                offset,
                length,
                name: None,
                hash: String::new(),
                flags: 0,
            });
        }
        // retoc/src/lib.rs:282-296: ordinal scanning does not need the hash
        // accelerator contents, but versioned counts must locate the next table.
        if version >= 4 {
            skip(&mut r, bounded(h32(84), MAX_ITEMS)? * 4)?;
        }
        if version >= 5 {
            skip(&mut r, bounded(h32(96), MAX_ITEMS)? * 4)?;
        }
        let mut blocks = Vec::with_capacity(block_count);
        for _ in 0..block_count {
            let b = bytes::<12>(&mut r)?;
            // retoc/src/lib.rs:1005-1027: LE40 offset, LE24 stored and
            // expanded sizes, then 1-based method index (0 means uncompressed).
            let offset = b[..5]
                .iter()
                .rev()
                .fold(0u64, |n, &b| (n << 8) | u64::from(b));
            let compressed = u32::from_le_bytes([b[5], b[6], b[7], 0]);
            let uncompressed = u32::from_le_bytes([b[8], b[9], b[10], 0]);
            if compressed == 0
                || uncompressed == 0
                || uncompressed > block_size
                || usize::from(b[11]) > method_count
                || (b[11] == 0 && compressed != uncompressed)
            {
                return Err(invalid("invalid IoStore compression block"));
            }
            blocks.push(Block {
                offset,
                compressed,
                uncompressed,
                method: b[11],
            });
        }
        // retoc/src/lib.rs:305-312: fixed-width NUL-padded method names.
        let mut methods = vec!["None".to_owned()];
        for _ in 0..method_count {
            let start = r.position() as usize;
            skip(&mut r, method_length)?;
            let b = &data[start..start + method_length];
            let end = b.iter().position(|&b| b == 0).unwrap_or(b.len());
            let method = std::str::from_utf8(&b[..end])
                .map_err(|_| invalid("invalid IoStore compression name"))?;
            if method.is_empty() || b[end..].iter().any(|&b| b != 0) {
                return Err(invalid("invalid IoStore compression name padding"));
            }
            methods.push(method.to_owned());
        }
        // retoc/src/lib.rs:324-335: two sized signatures and one SHA-1 per
        // block. These are framed, not cryptographically verified by this reader.
        if flags & 4 != 0 {
            let n = bounded(u32r(&mut r)?, MAX_STRING)?;
            skip(&mut r, n * 2 + block_count * 20)?;
        }
        if directory_size > MAX_TOC as usize
            || directory_size > data.len().saturating_sub(r.position() as usize)
        {
            return Err(invalid("IoStore directory exceeds resource limit"));
        }
        let mut directory = vec![0; directory_size];
        r.read_exact(&mut directory)?;
        // retoc/src/lib.rs:339-348: directory bytes use AES-256 ECB.
        if encrypted && !directory.is_empty() {
            key.unwrap().decrypt(&mut directory)?;
        }
        let mount = directory_index(&directory, encrypted, &mut chunks)?;
        // retoc/src/lib.rs:355-368: v8 uses a 20-byte hash, flags and 3
        // padding bytes; older versions use a 32-byte hash followed by flags.
        for chunk in &mut chunks {
            let hash_len = if version >= 8 { 20 } else { 32 };
            let start = r.position() as usize;
            skip(&mut r, hash_len)?;
            chunk.hash = hex(&data[start..start + hash_len]);
            chunk.flags = bytes::<1>(&mut r)?[0];
            // retoc/src/lib.rs:1135-1137: compressed and memory mapped.
            if chunk.flags & !3 != 0 {
                return Err(invalid("unsupported IoStore chunk metadata flags"));
            }
            if version >= 8 {
                skip(&mut r, 3)?;
            }
        }
        if r.position() != data.len() as u64 {
            return Err(invalid("trailing IoStore TOC bytes"));
        }
        let toc = Self {
            version,
            container_id: hex(&header[56..64]),
            encryption_guid: hex(&header[64..80]),
            flags,
            header,
            mount,
            chunks,
            methods,
            partition_count,
            partition_size,
            block_size,
            blocks,
        };
        let mut references = 0u64;
        for chunk in &toc.chunks {
            if chunk.length != 0 {
                references += (chunk.offset + chunk.length - 1) / u64::from(block_size)
                    - chunk.offset / u64::from(block_size)
                    + 1;
                if references > 16_000_000 {
                    return Err(invalid(
                        "IoStore chunk/block references exceed resource limit",
                    ));
                }
            }
            toc.chunk_blocks(chunk)?;
        }
        Ok(toc)
    }

    pub fn shader_code(&self, chunk: &Chunk) -> bool {
        // retoc/src/lib.rs:860-863,1263-1298: UE4 and UE5 type numbering.
        chunk.id[11] == if self.version > 4 { 9 } else { 12 }
    }
    pub fn shader_library(&self, chunk: &Chunk) -> bool {
        chunk.id[11] == if self.version > 4 { 8 } else { 11 }
    }
    fn chunk_blocks(&self, chunk: &Chunk) -> io::Result<std::ops::Range<usize>> {
        if chunk.length == 0 {
            return Ok(0..0);
        }
        // CUE4Parse IoStoreReader.cs:252-257: logical block indices and
        // offset inside the first block, including ranges not aligned to blocks.
        let first = usize::try_from(chunk.offset / u64::from(self.block_size))
            .map_err(|_| invalid("IoStore block index exceeds address space"))?;
        let last = usize::try_from((chunk.offset + chunk.length - 1) / u64::from(self.block_size))
            .map_err(|_| invalid("IoStore block index exceeds address space"))?;
        if last >= self.blocks.len() {
            return Err(invalid("IoStore chunk outside logical blocks"));
        }
        for index in first..=last {
            let start = if index == first {
                chunk.offset % u64::from(self.block_size)
            } else {
                0
            };
            let end = (chunk.offset + chunk.length - index as u64 * u64::from(self.block_size))
                .min(u64::from(self.block_size));
            if start >= end || end > u64::from(self.blocks[index].uncompressed) {
                return Err(invalid("IoStore chunk references unavailable block bytes"));
            }
        }
        Ok(first..last + 1)
    }
    pub fn compression(&self, chunk: &Chunk) -> Vec<&str> {
        // parse has already validated the logical range.
        let mut seen = HashSet::new();
        self.chunk_blocks(chunk)
            .unwrap()
            .map(|i| self.methods[usize::from(self.blocks[i].method)].as_str())
            .filter(|method| seen.insert(*method))
            .collect()
    }
    pub fn extract(
        &self,
        path: &Path,
        chunk: &Chunk,
        key: Option<&AesKey>,
        out: &mut impl Write,
    ) -> io::Result<()> {
        if chunk.length > MAX_CHUNK {
            return Err(invalid("IoStore chunk exceeds resource limit"));
        }
        let key = if self.flags & 2 != 0 {
            Some(key.ok_or_else(|| invalid("missing IoStore AES key"))?)
        } else {
            None
        };
        // Retain one partition handle while its consecutive blocks are read.
        let mut current_file: Option<(u64, File)> = None;
        for index in self.chunk_blocks(chunk)? {
            let block = self.blocks[index];
            let method = &self.methods[usize::from(block.method)];
            if !supported(method) {
                return Err(invalid(format!("unsupported IoStore compression {method}")));
            }
            // CUE4Parse IoStoreReader.cs:281-293: partition division/modulo;
            // retoc/src/lib.rs:678-702: ECB sizes rounded up to 16 bytes.
            let partition = block.offset / self.partition_size;
            if partition >= u64::from(self.partition_count) {
                return Err(invalid("IoStore block partition out of range"));
            }
            let offset = block.offset % self.partition_size;
            let stored = if key.is_some() {
                (block.compressed as usize + 15) & !15
            } else {
                block.compressed as usize
            };
            if offset
                .checked_add(stored as u64)
                .is_none_or(|end| end > self.partition_size)
            {
                return Err(invalid("IoStore block crosses partition boundary"));
            }
            if current_file
                .as_ref()
                .is_none_or(|(index, _)| *index != partition)
            {
                current_file = Some((partition, companion(&partition_path(path, partition))?));
            }
            let file = &mut current_file.as_mut().unwrap().1;
            let file_len = file.metadata()?.len();
            if offset
                .checked_add(stored as u64)
                .is_none_or(|end| end > file_len)
            {
                return Err(invalid("IoStore block outside data file"));
            }
            file.seek(SeekFrom::Start(offset))?;
            let mut data = vec![0; stored];
            file.read_exact(&mut data)?;
            if let Some(key) = key {
                key.decrypt(&mut data)?;
            }
            data.truncate(block.compressed as usize);
            let data = decompress(method, &data, block.uncompressed as usize)?;
            let start = if index as u64 == chunk.offset / u64::from(self.block_size) {
                (chunk.offset % u64::from(self.block_size)) as usize
            } else {
                0
            };
            let end = (chunk.offset + chunk.length - index as u64 * u64::from(self.block_size))
                .min(u64::from(self.block_size)) as usize;
            out.write_all(&data[start..end])?;
        }
        Ok(())
    }
}
fn companion(path: &Path) -> io::Result<File> {
    // Scanner policy applies to companion files as well as directory traversal.
    if !fs::symlink_metadata(path)?.file_type().is_file() {
        return Err(invalid(
            "IoStore data companion must be a regular file, not a symlink",
        ));
    }
    File::open(path)
}
fn partition_path(path: &Path, index: u64) -> PathBuf {
    // CUE4Parse/UE4/IO/IoStoreReader.cs:82-103: base.ucas, base_s1.ucas, ...
    if index == 0 {
        return path.with_extension("ucas");
    }
    let mut name = path.file_stem().unwrap_or_default().to_os_string();
    name.push(format!("_s{index}.ucas"));
    path.with_file_name(name)
}
fn supported(method: &str) -> bool {
    ["None", "Zlib", "Zstd", "LZ4"]
        .iter()
        .any(|m| method.eq_ignore_ascii_case(m))
}
// retoc/src/compression.rs:40-53: zlib stream, zstd stream, raw LZ4 block.
// Use exact output lengths and bounded streaming rather than read_exact alone,
// which would accept a stream expanding beyond its declared size.
pub(super) fn decompress(method: &str, data: &[u8], size: usize) -> io::Result<Vec<u8>> {
    if size as u64 > MAX_CHUNK {
        return Err(invalid("IoStore expanded data exceeds resource limit"));
    }
    let result = if method.eq_ignore_ascii_case("None") {
        data.to_vec()
    } else if method.eq_ignore_ascii_case("Zlib") {
        let mut result = Vec::new();
        let mut decoder = flate2::read::ZlibDecoder::new(data);
        (&mut decoder)
            .take(size as u64 + 1)
            .read_to_end(&mut result)?;
        if decoder.total_in() != data.len() as u64 {
            return Err(invalid("trailing IoStore zlib bytes"));
        }
        result
    } else if method.eq_ignore_ascii_case("Zstd") {
        let decoder = zstd::stream::read::Decoder::new(data)?;
        let mut result = Vec::new();
        decoder.take(size as u64 + 1).read_to_end(&mut result)?;
        result
    } else if method.eq_ignore_ascii_case("LZ4") {
        lz4_flex::block::decompress(data, size).map_err(|e| invalid(e.to_string()))?
    } else {
        return Err(invalid(format!("unsupported IoStore compression {method}")));
    };
    if result.len() != size {
        return Err(invalid("IoStore decompressed size mismatch"));
    }
    Ok(result)
}

fn directory_index(data: &[u8], encrypted: bool, chunks: &mut [Chunk]) -> io::Result<String> {
    if data.is_empty() {
        return Ok(String::new());
    }
    let mut r = Cursor::new(data);
    // retoc/src/lib.rs:1353-1366,1545-1582; absent links are 0xffffffff
    // (1510-1520). user_data is the TOC ordinal (487-491).
    let mount = string(&mut r)?;
    let dirs = array(&mut r, bytes::<16>)?;
    let files = array(&mut r, bytes::<12>)?;
    let strings = array(&mut r, string)?;
    let padding = data.len() - r.position() as usize;
    if padding != 0 && !(encrypted && padding < 16 && data.len().is_multiple_of(16)) {
        return Err(invalid("trailing IoStore directory bytes"));
    }
    let field = |b: &[u8], at| u32::from_le_bytes(b[at..at + 4].try_into().unwrap());
    let name = |index: u32| {
        strings
            .get(index as usize)
            .ok_or_else(|| invalid("IoStore directory name index out of range"))
    };
    // retoc/src/lib.rs:1381-1414: root ordinal 0, child/sibling/file lists.
    // Iterative traversal and visited sets bound malformed cycles and depth.
    let mut work = if dirs.is_empty() {
        vec![]
    } else {
        vec![(0u32, String::new())]
    };
    let mut seen_dirs = HashSet::new();
    let mut seen_files = HashSet::new();
    let mut path_bytes = 0usize;
    while let Some((index, parent)) = work.pop() {
        if !seen_dirs.insert(index) {
            return Err(invalid("cycle or shared IoStore directory"));
        }
        let dir = dirs
            .get(index as usize)
            .ok_or_else(|| invalid("IoStore directory index out of range"))?;
        let dir_name = field(dir, 0);
        let mut path = parent.clone();
        if dir_name != u32::MAX {
            path.push_str(name(dir_name)?);
            path.push('/');
        }
        path_bytes = path_bytes
            .checked_add(path.len())
            .filter(|&n| n <= MAX_TOC as usize)
            .ok_or_else(|| invalid("IoStore directory paths exceed resource limit"))?;
        if path.len() > MAX_STRING as usize {
            return Err(invalid("IoStore directory path exceeds resource limit"));
        }
        let sibling = field(dir, 8);
        if sibling != u32::MAX {
            work.push((sibling, parent));
        }
        let child = field(dir, 4);
        if child != u32::MAX {
            work.push((child, path.clone()));
        }
        let mut file = field(dir, 12);
        while file != u32::MAX {
            if !seen_files.insert(file) {
                return Err(invalid("cycle or shared IoStore file entry"));
            }
            let entry = files
                .get(file as usize)
                .ok_or_else(|| invalid("IoStore file index out of range"))?;
            let full_name = format!("{path}{}", name(field(entry, 0))?);
            path_bytes = path_bytes
                .checked_add(full_name.len())
                .filter(|&n| n <= MAX_TOC as usize)
                .ok_or_else(|| invalid("IoStore directory paths exceed resource limit"))?;
            let chunk = chunks
                .get_mut(field(entry, 8) as usize)
                .ok_or_else(|| invalid("IoStore directory chunk index out of range"))?;
            if chunk.name.replace(full_name).is_some() {
                return Err(invalid("duplicate IoStore chunk path"));
            }
            file = field(entry, 4);
        }
    }
    if seen_dirs.len() != dirs.len() || seen_files.len() != files.len() {
        return Err(invalid("unreachable IoStore directory entries"));
    }
    Ok(mount)
}

#[cfg(test)]
pub(crate) mod tests;
