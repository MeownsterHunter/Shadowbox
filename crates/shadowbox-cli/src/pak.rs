//! UE pak reader. Format references are repak at commit
//! 355b5f62f51959c7cc6dd5a51708646ef483065d, paths relative to that repository.
//! See docs/reference-sources.md and 0019-ue4-pak-aes.md.
//! No archive paths are used for writes.
use aes::cipher::{block_padding::NoPadding, BlockDecryptMut, KeyInit};
use sha1::{Digest, Sha1};
use std::collections::{BTreeMap, HashSet};
use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};
use zeroize::Zeroizing;

// No Debug implementation: keys must never enter reports or diagnostics.
pub(super) struct AesKey(Zeroizing<[u8; 32]>);
impl AesKey {
    pub fn parse(value: &[u8]) -> io::Result<Self> {
        let value = value.trim_ascii();
        let value = value.strip_prefix(b"0x").unwrap_or(value);
        let failure = || {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "pak AES key must contain exactly 64 hex characters",
            )
        };
        if value.len() != 64 {
            return Err(failure());
        }
        let mut key = Zeroizing::new([0; 32]);
        let hex = |b: u8| match b {
            b'0'..=b'9' => Some(b - b'0'),
            b'a'..=b'f' => Some(b - b'a' + 10),
            b'A'..=b'F' => Some(b - b'A' + 10),
            _ => None,
        };
        for (out, pair) in key.iter_mut().zip(value.as_chunks::<2>().0) {
            *out = (hex(pair[0]).ok_or_else(failure)? << 4) | hex(pair[1]).ok_or_else(failure)?;
        }
        Ok(Self(key))
    }
    // repak/src/lib.rs:128-139; pak.rs:151-161: AES-256, independent
    // 16-byte blocks (ECB). UE padding is size alignment, not PKCS#7.
    pub(super) fn decrypt(&self, data: &mut [u8]) -> io::Result<()> {
        ecb::Decryptor::<aes::Aes256>::new((&*self.0).into())
            .decrypt_padded_mut::<NoPadding>(data)
            .map_err(|_| {
                diagnostic(
                    Diagnostic::TruncatedData,
                    "encrypted pak size is not a multiple of 16",
                )
            })?;
        Ok(())
    }
}
// repak/src/entry.rs:34-37,349-367: round stored sizes up to an AES block.
fn padded(size: u64) -> io::Result<u64> {
    size.checked_add(15)
        .map(|n| n & !15)
        .ok_or_else(|| invalid("pak padding overflow"))
}

// repak/src/lib.rs:11,68-90; footer.rs:25-61.
const MAGIC: u32 = 0x5a6f12e1;
// Local resource policy, not format limits. Reject, never truncate.
const MAX_INDEX: u64 = 256 * 1024 * 1024;
const MAX_ITEMS: u32 = 1_000_000;
const MAX_STRING: u32 = 1024 * 1024;

// Reporting policy: every reported entry has an explicit integrity state.
// Only an actual matching payload checksum establishes Verified.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) enum Integrity {
    Verified,
    #[default]
    Unverified,
}
impl Integrity {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Verified => "verified",
            Self::Unverified => "unverified",
        }
    }
}

// repak/src/data.rs:173,184,195: plaintext entry hashes cover stored bytes.
// Reporting policy: every reported entry has an explicit integrity state. An absent
// hash (including intentionally skipped encrypted checks) proves nothing.
fn check_entry_hash(r: &mut impl Read, size: u64, hash: Option<[u8; 20]>) -> io::Result<Integrity> {
    let Some(hash) = hash else {
        return Ok(Integrity::Unverified);
    };
    let mut hasher = Sha1::new();
    let mut remaining = size;
    let mut buffer = [0; 64 * 1024];
    while remaining != 0 {
        let n = remaining.min(buffer.len() as u64) as usize;
        r.read_exact(&mut buffer[..n])?;
        hasher.update(&buffer[..n]);
        remaining -= n as u64;
    }
    if hash != <[u8; 20]>::from(hasher.finalize()) {
        return Err(diagnostic(Diagnostic::BadHash, "pak entry SHA-1 mismatch"));
    }
    Ok(Integrity::Verified)
}

// Diagnostic categories describe evidence, not a guessed cause. In particular,
// random wrong-key plaintext and damaged ciphertext are indistinguishable here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Diagnostic {
    MalformedIndex,
    UnsupportedIndexVariant,
    TruncatedData,
    BadHash,
}
#[derive(Debug)]
struct PakError {
    category: Diagnostic,
    reason: String,
}
impl std::fmt::Display for PakError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let category = match self.category {
            Diagnostic::MalformedIndex => "malformed index",
            Diagnostic::UnsupportedIndexVariant => "unsupported index variant",
            Diagnostic::TruncatedData => "truncated data",
            Diagnostic::BadHash => "bad hash",
        };
        write!(f, "{category}: {}", self.reason)
    }
}
impl std::error::Error for PakError {}
fn diagnostic(category: Diagnostic, reason: &str) -> io::Error {
    io::Error::new(
        if category == Diagnostic::TruncatedData {
            io::ErrorKind::UnexpectedEof
        } else {
            io::ErrorKind::InvalidData
        },
        PakError {
            category,
            reason: reason.into(),
        },
    )
}
fn invalid(reason: &str) -> io::Error {
    diagnostic(Diagnostic::MalformedIndex, reason)
}
fn classify(error: io::Error) -> io::Error {
    if error.kind() == io::ErrorKind::UnexpectedEof
        && !error.get_ref().is_some_and(|e| e.is::<PakError>())
    {
        diagnostic(
            Diagnostic::TruncatedData,
            "pak index ended before its declared fields",
        )
    } else {
        error
    }
}
fn bytes<const N: usize>(r: &mut impl Read) -> io::Result<[u8; N]> {
    let mut b = [0; N];
    r.read_exact(&mut b)?;
    Ok(b)
}
fn u8r(r: &mut impl Read) -> io::Result<u8> {
    Ok(bytes::<1>(r)?[0])
}
fn u32r(r: &mut impl Read) -> io::Result<u32> {
    Ok(u32::from_le_bytes(bytes(r)?))
}
fn i32r(r: &mut impl Read) -> io::Result<i32> {
    Ok(i32::from_le_bytes(bytes(r)?))
}
fn u64r(r: &mut impl Read) -> io::Result<u64> {
    Ok(u64::from_le_bytes(bytes(r)?))
}
fn boolean(value: u32) -> io::Result<bool> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(invalid("invalid pak boolean")),
    }
}
fn count(r: &mut impl Read) -> io::Result<u32> {
    let n = u32r(r)?;
    if n > MAX_ITEMS {
        return Err(invalid("pak item count exceeds resource limit"));
    }
    Ok(n)
}
// repak/src/ext.rs:69-80: signed FString length includes the terminator;
// negative lengths count UTF-16 code units. Require well-formed names.
fn string(r: &mut impl Read) -> io::Result<String> {
    let n = i32r(r)?;
    if n.unsigned_abs() > MAX_STRING {
        return Err(invalid("pak string exceeds resource limit"));
    }
    if n == 0 {
        return Ok(String::new());
    }
    let mut data = vec![0; n.unsigned_abs() as usize * if n < 0 { 2 } else { 1 }];
    r.read_exact(&mut data)?;
    if n < 0 {
        let mut words: Vec<_> = data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        if words.pop() != Some(0) || words.contains(&0) {
            return Err(invalid("invalid pak UTF-16 terminator"));
        }
        String::from_utf16(&words).map_err(|_| invalid("invalid pak UTF-16"))
    } else {
        if data.pop() != Some(0) || data.contains(&0) {
            return Err(invalid("invalid pak string terminator"));
        }
        String::from_utf8(data).map_err(|_| invalid("invalid pak UTF-8"))
    }
}
fn end(r: &Cursor<Vec<u8>>) -> io::Result<()> {
    if r.position() != r.get_ref().len() as u64 {
        return Err(invalid("trailing pak index bytes"));
    }
    Ok(())
}
// repak/src/pak.rs:385-400,411-418,442-450: encrypted index sizes
// include AES alignment bytes. Consume at most 15 bytes after the grammar.
fn end_index(r: &Cursor<Vec<u8>>, encrypted: bool) -> io::Result<()> {
    if encrypted && padded(r.position())? == r.get_ref().len() as u64 {
        Ok(())
    } else {
        end(r)
    }
}
fn range(offset: u64, size: u64, limit: u64) -> io::Result<()> {
    if offset.checked_add(size).is_none_or(|end| end > limit) {
        return Err(diagnostic(
            Diagnostic::TruncatedData,
            "pak range outside data area",
        ));
    }
    Ok(())
}

// Bounded reader over one encrypted extent. ECB has no inter-block state;
// decrypt 64 KiB at a time and expose only the declared meaningful bytes.
// repak/src/entry.rs:349-367 truncates AES alignment after decryption.
struct DataReader<'a, R> {
    source: &'a mut R,
    key: Option<&'a AesKey>,
    remaining: u64,
    buffer: [u8; 64 * 1024],
    at: usize,
    len: usize,
}
impl<'a, R: Read> DataReader<'a, R> {
    fn new(source: &'a mut R, size: u64, key: Option<&'a AesKey>) -> Self {
        Self {
            source,
            key,
            remaining: size,
            buffer: [0; 64 * 1024],
            at: 0,
            len: 0,
        }
    }
}
impl<R: Read> Read for DataReader<'_, R> {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        if self.at == self.len {
            self.len = self.remaining.min(self.buffer.len() as u64) as usize;
            self.at = 0;
            if self.len == 0 {
                return Ok(0);
            }
            let stored = if self.key.is_some() {
                padded(self.len as u64)? as usize
            } else {
                self.len
            };
            self.source.read_exact(&mut self.buffer[..stored])?;
            if let Some(key) = self.key {
                key.decrypt(&mut self.buffer[..stored])?;
            }
            self.remaining -= self.len as u64;
        }
        let n = out.len().min(self.len - self.at);
        out[..n].copy_from_slice(&self.buffer[self.at..self.at + n]);
        self.at += n;
        Ok(n)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Entry {
    pub offset: u64,
    pub compressed: u64,
    pub uncompressed: u64,
    pub method: u32,
    pub flags: u8,
    pub block_size: u32,
    pub blocks: Vec<(u64, u64)>,
    hash: Option<[u8; 20]>,
}
impl Entry {
    // repak/src/entry.rs:39-43,116-153: v8A uses a byte method index;
    // other versions use u32, followed by SHA-1, blocks, flags, block size.
    fn read(r: &mut impl Read, byte_method: bool) -> io::Result<Self> {
        let offset = u64r(r)?;
        let compressed = u64r(r)?;
        let uncompressed = u64r(r)?;
        let method = if byte_method {
            u8r(r)? as u32
        } else {
            u32r(r)?
        };
        let hash = Some(bytes(r)?);
        let mut blocks = Vec::new();
        if method != 0 {
            for _ in 0..count(r)? {
                blocks.push((u64r(r)?, u64r(r)?));
            }
        }
        let flags = u8r(r)?;
        let block_size = u32r(r)?;
        Ok(Self {
            offset,
            compressed,
            uncompressed,
            method,
            hash,
            blocks,
            flags,
            block_size,
        })
    }
    // repak/src/entry.rs:71-98,194-268: compact flags, variable-width
    // integers, implicit single-block range and AES padding for multiple blocks.
    fn encoded(r: &mut impl Read) -> io::Result<Self> {
        let bits = u32r(r)?;
        let method = (bits >> 23) & 63;
        let flags = ((bits >> 22) & 1) as u8;
        let n = (bits >> 6) & 0xffff;
        let block_size = if bits & 63 == 63 {
            u32r(r)?
        } else {
            (bits & 63) << 11
        };
        let mut var = |bit: u32| {
            if bits & (1u32 << bit) != 0 {
                u32r(r).map(u64::from)
            } else {
                u64r(r)
            }
        };
        let offset = var(31)?;
        let uncompressed = var(30)?;
        let compressed = if method == 0 { uncompressed } else { var(29)? };
        if method == 0 && n != 0 {
            return Err(invalid("uncompressed compact entry has blocks"));
        }
        let mut at = 53
            + if method != 0 {
                4 + 16 * u64::from(n)
            } else {
                0
            };
        let mut blocks = Vec::new();
        for _ in 0..n {
            let len = if n == 1 && flags == 0 {
                compressed
            } else {
                u64::from(u32r(r)?)
            };
            let next = at
                .checked_add(len)
                .ok_or_else(|| invalid("pak block overflow"))?;
            blocks.push((at, next));
            at = if flags == 0 {
                next
            } else {
                at.checked_add(
                    len.checked_add(15)
                        .ok_or_else(|| invalid("pak padding overflow"))?
                        & !15,
                )
                .ok_or_else(|| invalid("pak block overflow"))?
            };
        }
        Ok(Self {
            offset,
            compressed,
            uncompressed,
            method,
            flags,
            block_size,
            blocks,
            hash: None,
        })
    }
}

pub(super) struct Pak {
    pub version: u32,
    pub mount: String,
    pub entries: Vec<(String, Entry)>,
    pub path_hash_seed: Option<u64>,
    pub encryption_guid: Option<String>,
    byte_method: bool,
    methods: Vec<String>,
    data_end: u64,
    key: Option<AesKey>,
    index_structure_checked: bool,
}
struct Footer {
    version: u32,
    byte_method: bool,
    offset: u64,
    size: u64,
    hash: [u8; 20],
    methods: Vec<String>,
    start: u64,
    encrypted: bool,
    frozen: bool,
    guid: Option<String>,
}
impl Footer {
    fn read(r: &mut (impl Read + Seek)) -> io::Result<Option<Self>> {
        let len = r.seek(SeekFrom::End(0))?;
        // repak/src/lib.rs:68-90 and footer.rs:25-61. Probe only exact
        // footer layouts at EOF, never magic elsewhere in archive content.
        for (version, slots) in [
            (11, 5),
            (10, 5),
            (9, 5),
            (8, 5),
            (8, 4),
            (7, 0),
            (6, 0),
            (5, 0),
            (4, 0),
            (3, 0),
        ] {
            let size = 44
                + u64::from(version >= 4)
                + if version >= 7 { 16 } else { 0 }
                + u64::from(version == 9)
                + slots * 32;
            if len < size {
                continue;
            }
            r.seek(SeekFrom::Start(len - size))?;
            let guid = if version >= 7 {
                Some(bytes::<16>(r)?.iter().map(|b| format!("{b:02x}")).collect())
            } else {
                None
            };
            let encrypted = if version >= 4 { u8r(r)? } else { 0 };
            if u32r(r)? != MAGIC || u32r(r)? != version {
                continue;
            }
            let offset = u64r(r)?;
            let size_index = u64r(r)?;
            let hash = bytes(r)?;
            let frozen = if version == 9 {
                boolean(u8r(r)? as u32)?
            } else {
                false
            };
            let mut methods = vec!["None".into()];
            for _ in 0..slots {
                let name = bytes::<32>(r)?;
                let n = name.iter().position(|b| *b == 0).unwrap_or(32);
                if name[n..].iter().any(|b| *b != 0) {
                    return Err(invalid("invalid pak compression name padding"));
                }
                methods.push(
                    String::from_utf8(name[..n].to_vec())
                        .map_err(|_| invalid("invalid pak codec name"))?,
                );
            }
            if version < 8 {
                methods.extend(["Zlib".into(), "Gzip".into(), "Oodle".into()]);
            }
            range(offset, size_index, len - size)?;
            return Ok(Some(Self {
                version,
                byte_method: version == 8 && slots == 4,
                offset,
                size: size_index,
                hash,
                methods,
                start: len - size,
                encrypted: boolean(encrypted as u32)?,
                frozen,
                guid,
            }));
        }
        Ok(None)
    }
}
// Cheap candidate screening is NOT key authentication. A rejection may also
// mean a damaged ciphertext or a malformed FString. Never convert it into a
// wrong-key diagnostic when opening an archive.
// repak/src/pak.rs:385-398 and ext.rs:69-80: decrypt at the absolute offset,
// then the first four bytes are the signed FString length. Inspect one ECB
// block before allocating the primary or visiting any secondary index.
fn key_candidate_prefix(r: &mut (impl Read + Seek), f: &Footer, key: &AesKey) -> io::Result<bool> {
    if !f.encrypted {
        return Err(diagnostic(
            Diagnostic::UnsupportedIndexVariant,
            "key screening needs an encrypted index",
        ));
    }
    if f.size < 16 || !f.size.is_multiple_of(16) {
        return Err(diagnostic(
            Diagnostic::TruncatedData,
            "encrypted index size is not a positive AES block multiple",
        ));
    }
    r.seek(SeekFrom::Start(f.offset))?;
    let mut prefix = bytes::<16>(r)?;
    key.decrypt(&mut prefix)?;
    let n = i32::from_le_bytes(prefix[..4].try_into().unwrap());
    let units = n.unsigned_abs();
    let string_bytes = u64::from(units) * if n < 0 { 2 } else { 1 };
    // At least the FString and entry count must fit. Zero is a valid FString.
    Ok(units <= MAX_STRING && string_bytes + 8 <= f.size)
}

// repak/src/pak.rs:631,636,646,663,674-678: SHA-1 index hashes.
fn index(
    r: &mut (impl Read + Seek),
    offset: u64,
    size: u64,
    hash: [u8; 20],
    limit: u64,
    key: Option<&AesKey>,
) -> io::Result<Cursor<Vec<u8>>> {
    range(offset, size, limit)?;
    if size > MAX_INDEX {
        return Err(invalid("pak index exceeds resource limit"));
    }
    // repak/src/pak.rs:385-398: IndexOffset is absolute from byte zero of
    // the pak. Seek exactly there. AES block alignment applies to size only,
    // never to the file offset (ECB block definition: pak.rs:151-161).
    r.seek(SeekFrom::Start(offset))?;
    let mut data = vec![0; size as usize];
    r.read_exact(&mut data)?;
    if let Some(key) = key {
        key.decrypt(&mut data)?;
    }
    // repak does not verify encrypted index hashes (pak.rs:385-400).
    // Its public reference does not define the encrypted hash domain.
    // Validate their complete grammar below, without guessing that domain.
    if key.is_none() && Sha1::digest(&data)[..] != hash {
        return Err(diagnostic(Diagnostic::BadHash, "pak index SHA-1 mismatch"));
    }
    Ok(Cursor::new(data))
}
// repak/src/pak.rs:405-408,435-438: optional secondary index descriptors.
fn descriptor(r: &mut impl Read) -> io::Result<Option<(u64, u64, [u8; 20])>> {
    if !boolean(u32r(r)?)? {
        return Ok(None);
    }
    Ok(Some((u64r(r)?, u64r(r)?, bytes(r)?)))
}
// repak/src/pak.rs:699-716: seeded FNV-1a over lowercase UTF-16LE paths.
fn path_hash(path: &str, seed: u64) -> u64 {
    path.to_lowercase()
        .encode_utf16()
        .flat_map(u16::to_le_bytes)
        .fold(0xcbf29ce484222325u64.wrapping_add(seed), |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x00000100000001b3)
        })
}
impl Pak {
    #[cfg(test)]
    pub fn open(r: &mut (impl Read + Seek)) -> io::Result<Option<Self>> {
        Self::open_with_key(r, None)
    }
    pub fn open_with_key(
        r: &mut (impl Read + Seek),
        key: Option<&AesKey>,
    ) -> io::Result<Option<Self>> {
        let Some(f) = Footer::read(r).map_err(classify)? else {
            return Ok(None);
        };
        if f.encrypted && key.is_none() {
            return Err(diagnostic(
                Diagnostic::UnsupportedIndexVariant,
                "encrypted pak index: no key supplied: skipped",
            ));
        }
        if f.frozen {
            return Err(diagnostic(
                Diagnostic::UnsupportedIndexVariant,
                "frozen v9 pak index is unsupported: skipped",
            ));
        }
        if f.encrypted && !key_candidate_prefix(r, &f, key.unwrap()).map_err(classify)? {
            return Err(invalid(
                "invalid mount FString length; key candidate rejected or index damaged",
            ));
        }
        // No parse failure is evidence of a wrong key. Preserve the concrete
        // grammar, unsupported-variant, truncation or checksum diagnostic.
        Self::read_index(r, f, key).map_err(classify)
    }

    fn read_index(
        r: &mut (impl Read + Seek),
        f: Footer,
        key: Option<&AesKey>,
    ) -> io::Result<Option<Self>> {
        let index_key = if f.encrypted { key } else { None };
        let mut idx = index(r, f.offset, f.size, f.hash, f.start, index_key)?;
        // repak/src/pak.rs:397-402,510-516: mount point, entry count,
        // legacy named entries, or the v10+ path hash seed.
        let mount = string(&mut idx)?;
        let n = count(&mut idx)?;
        let mut entries = BTreeMap::new();
        let mut path_hash_seed = None;
        let mut data_end = f.offset;
        if f.version < 10 {
            for _ in 0..n {
                let name = string(&mut idx)?;
                if entries
                    .insert(name, Entry::read(&mut idx, f.byte_method)?)
                    .is_some()
                {
                    return Err(invalid("duplicate pak entry name"));
                }
            }
        } else {
            path_hash_seed = Some(u64r(&mut idx)?);
            let phi = descriptor(&mut idx)?;
            let fdi = descriptor(&mut idx)?;
            // repak/src/pak.rs:470-477: compact entry byte array followed
            // by ordinary entries addressed with negative locations.
            let size = u32r(&mut idx)? as u64;
            if size > MAX_INDEX {
                return Err(invalid("compact entry array exceeds resource limit"));
            }
            if size > idx.get_ref().len() as u64 - idx.position() {
                return Err(diagnostic(
                    Diagnostic::TruncatedData,
                    "truncated compact entry array",
                ));
            }
            let mut encoded = vec![0; size as usize];
            idx.read_exact(&mut encoded)?;
            let mut plain = Vec::new();
            for _ in 0..count(&mut idx)? {
                plain.push(Entry::read(&mut idx, false)?);
            }
            // repak/src/pak.rs:564-571: encoded records are concatenated.
            // Parse every record, including unreferenced ones. Locations must
            // name a record boundary, never bytes inside another record.
            let mut encoded = Cursor::new(encoded);
            let mut compact_entries = BTreeMap::new();
            while encoded.position() < encoded.get_ref().len() as u64 {
                if compact_entries.len() >= MAX_ITEMS as usize {
                    return Err(invalid("compact entry count exceeds resource limit"));
                }
                compact_entries.insert(encoded.position(), Entry::encoded(&mut encoded)?);
            }
            let mut resolve = |location: i32| -> io::Result<Option<Entry>> {
                // repak/src/pak.rs:484-494: deletion sentinel, positive byte
                // offset or negative ordinary-entry index minus one.
                if location == i32::MIN {
                    return Ok(None);
                }
                if location < 0 {
                    return plain
                        .get((-i64::from(location) - 1) as usize)
                        .cloned()
                        .map(Some)
                        .ok_or_else(|| invalid("invalid pak plain entry location"));
                }
                compact_entries
                    .get(&(location as u64))
                    .cloned()
                    .map(Some)
                    .ok_or_else(|| invalid("invalid pak compact entry location"))
            };
            let mut hashes = BTreeMap::new();
            if let Some((offset, size, hash)) = phi {
                data_end = data_end.min(offset);
                let mut idx = index(r, offset, size, hash, f.start, index_key)?;
                // repak/src/pak.rs:421-427: hash -> signed entry location.
                let mut seen_hashes = HashSet::new();
                for _ in 0..count(&mut idx)? {
                    let hash = u64r(&mut idx)?;
                    let location = i32r(&mut idx)?;
                    if !seen_hashes.insert(hash) {
                        return Err(invalid("duplicate pak path hash"));
                    }
                    if let Some(entry) = resolve(location)? {
                        hashes.insert(hash, entry);
                    }
                }
                // repak/src/pak.rs:694 writes a zero trailer. Its reader
                // does not document a nonempty trailer's fields, so reject
                // that variant rather than invent its layout.
                if u32r(&mut idx)? != 0 {
                    return Err(diagnostic(
                        Diagnostic::UnsupportedIndexVariant,
                        "nonempty path hash index trailer is unsupported",
                    ));
                }
                end_index(&idx, f.encrypted)?;
            }
            if let Some((offset, size, hash)) = fdi {
                data_end = data_end.min(offset);
                let mut idx = index(r, offset, size, hash, f.start, index_key)?;
                read_directories(&mut idx, &mut resolve, &mut entries)?;
                end_index(&idx, f.encrypted)?;
            }
            // Validate both maps instead of silently discarding the path hash
            // records when a full directory exists. Hashing: repak/src/pak.rs:
            // 699-716; both generators use the same locations: :681-694,732-757.
            if phi.is_some() && fdi.is_some() {
                if hashes.len() != entries.len() {
                    return Err(invalid("pak path hash and directory index counts disagree"));
                }
                for (name, entry) in &entries {
                    if hashes.get(&path_hash(name, path_hash_seed.unwrap())) != Some(entry) {
                        return Err(invalid("pak path hash and directory indexes disagree"));
                    }
                }
            }
            // Hash-only archives still allow content discovery. Names cannot
            // be recovered from hashes; label them explicitly rather than guess.
            let mut offsets: HashSet<_> = entries.values().map(|e| e.offset).collect();
            for (hash, entry) in hashes {
                if offsets.insert(entry.offset) {
                    entries.insert(format!("<path-hash:{hash:016x}>"), entry);
                }
            }
            if entries.len() > n as usize {
                return Err(invalid("pak index entry count mismatch"));
            }
            if entries.len() < n as usize {
                return Err(invalid("pak indexes do not enumerate all entries"));
            }
        }
        end_index(&idx, f.encrypted)?;
        if f.encrypted {
            // Local trust policy: parsed entry metadata must point into the data
            // area. Header lengths follow repak/src/entry.rs:71-98, flags :65-69.
            for entry in entries.values().filter(|e| e.flags & 2 == 0) {
                let header_size = 53 - if f.byte_method { 3 } else { 0 }
                    + if entry.method != 0 {
                        4 + 16 * entry.blocks.len() as u64
                    } else {
                        0
                    };
                range(entry.offset, header_size, data_end)?;
                let size = if entry.flags & 1 != 0 {
                    padded(entry.compressed)?
                } else {
                    entry.compressed
                };
                range(entry.offset + header_size, size, data_end)?;
                if entry.method == 0 && entry.compressed != entry.uncompressed {
                    return Err(invalid("uncompressed pak sizes disagree"));
                }
            }
        }
        Ok(Some(Self {
            version: f.version,
            mount,
            entries: entries.into_iter().collect(),
            path_hash_seed,
            encryption_guid: f.guid,
            byte_method: f.byte_method,
            methods: f.methods,
            data_end,
            key: key.map(|k| AesKey(Zeroizing::new(*k.0))),
            index_structure_checked: f.encrypted,
        }))
    }
    pub fn method(&self, entry: &Entry) -> &str {
        self.methods
            .get(entry.method as usize)
            .map(String::as_str)
            .filter(|s| !s.is_empty())
            .unwrap_or("Unknown")
    }
    pub fn skip_reason(&self, entry: &Entry) -> Option<String> {
        // repak/src/entry.rs:65-69: encryption and deletion flags.
        if entry.flags & 1 != 0 && self.key.is_none() {
            return Some("encrypted pak entry".into());
        }
        if entry.flags & 1 != 0 && !self.index_structure_checked {
            return Some(
                "encrypted pak entry: key cannot be verified without an encrypted index".into(),
            );
        }
        if entry.flags & 2 != 0 {
            return Some("deleted pak entry".into());
        }
        if entry.flags & !3 != 0 {
            return Some("unsupported pak entry flags".into());
        }
        if !matches!(self.method(entry), "None" | "Zlib" | "Zstd") {
            return Some(format!(
                "unsupported compression method {} (slot {})",
                self.method(entry),
                entry.method
            ));
        }
        None
    }
    pub fn extract(
        &self,
        r: &mut (impl Read + Seek),
        entry: &Entry,
        out: &mut impl Write,
        integrity: &mut Integrity,
    ) -> io::Result<()> {
        *integrity = Integrity::Unverified;
        if let Some(reason) = self.skip_reason(entry) {
            return Err(invalid(&reason));
        }
        range(
            entry.offset,
            53 - if self.byte_method { 3 } else { 0 },
            self.data_end,
        )?;
        r.seek(SeekFrom::Start(entry.offset))?;
        let header = Entry::read(r, self.byte_method)?;
        let data_start = r.stream_position()?;
        // repak/src/entry.rs:163-166: the on-disk header offset is zero.
        let mut expected = entry.clone();
        expected.offset = 0;
        expected.hash = header.hash;
        if header != expected {
            return Err(invalid("pak entry header disagrees with index"));
        }
        if entry.hash.is_some() && entry.hash != header.hash {
            // repak/src/entry.rs:121-135,154-176: ordinary index records
            // and data headers both carry the entry SHA-1. A disagreement
            // is a checksum failure, not an index grammar failure.
            return Err(diagnostic(
                Diagnostic::BadHash,
                "pak entry hash disagrees with index",
            ));
        }
        let key = if entry.flags & 1 != 0 {
            self.key.as_ref()
        } else {
            None
        };
        let stored_size = if key.is_some() {
            padded(entry.compressed)?
        } else {
            entry.compressed
        };
        range(data_start, stored_size, self.data_end)?;
        // repak/src/data.rs:173,184,195: for unencrypted entries SHA-1 covers the stored bytes,
        // compressed when applicable. Verify before exposing decompressed data.
        // A bounded extra read avoids keeping an entire entry in memory.
        // repak/src/entry.rs:349-367,388-427 decrypts encrypted extents
        // without checking their SHA-1. Their hash domain is not specified by
        // the permitted reference. A fully parsed encrypted index must have
        // checked the decrypted grammar before extraction is allowed.
        *integrity = check_entry_hash(r, entry.compressed, header.hash.filter(|_| key.is_none()))?;
        r.seek(SeekFrom::Start(data_start))?;
        if entry.method == 0 {
            if entry.compressed != entry.uncompressed {
                return Err(invalid("uncompressed pak sizes disagree"));
            }
            let copied = io::copy(&mut DataReader::new(r, entry.compressed, key), out)?;
            if copied != entry.uncompressed {
                return Err(invalid("truncated pak entry"));
            }
            return Ok(());
        }
        if entry.blocks.is_empty() || entry.block_size == 0 {
            return Err(invalid("compressed pak entry has no blocks or block size"));
        }
        // repak/src/entry.rs:371-375: absolute v3/v4 block offsets;
        // v5+ block offsets are relative to the entry header.
        let base = if self.version >= 5 { entry.offset } else { 0 };
        let mut previous = data_start;
        let mut total = 0;
        let mut stored = 0u64;
        for &(start, end) in &entry.blocks {
            let start = base
                .checked_add(start)
                .ok_or_else(|| invalid("pak block overflow"))?;
            let end = base
                .checked_add(end)
                .ok_or_else(|| invalid("pak block overflow"))?;
            if start != previous || end <= start || end > data_start + entry.compressed {
                return Err(invalid("invalid pak compression block range"));
            }
            r.seek(SeekFrom::Start(start))?;
            let size = end - start;
            stored += size;
            let expected = (entry.uncompressed - total).min(u64::from(entry.block_size));
            let mut input = DataReader::new(r, size, key);
            // repak/src/entry.rs:388-427: independent zlib/zstd frames per
            // block. Bound output by the declared block length plus one byte.
            let copied = match self.method(entry) {
                "Zlib" => {
                    let mut decoder = flate2::read::ZlibDecoder::new(&mut input);
                    let copied = io::copy(&mut decoder.by_ref().take(expected + 1), out)?;
                    if decoder.total_in() != size {
                        return Err(invalid("zlib block has trailing or unread bytes"));
                    }
                    copied
                }
                "Zstd" => {
                    let mut decoder = zstd::stream::read::Decoder::new(&mut input)?;
                    decoder.window_log_max(27)?; // Local 128 MiB window limit.
                    io::copy(&mut decoder.take(expected + 1), out)?
                }
                _ => unreachable!(),
            };
            if copied != expected {
                return Err(invalid("pak decompressed block size mismatch"));
            }
            total += copied;
            // repak/src/entry.rs:244-256: encrypted block starts advance
            // by padded lengths; block end excludes that block's padding.
            previous = if key.is_some() {
                start
                    .checked_add(padded(size)?)
                    .ok_or_else(|| invalid("pak block overflow"))?
            } else {
                end
            };
        }
        if total != entry.uncompressed
            || (if key.is_some() {
                previous - data_start
            } else {
                stored
            }) != stored_size
        {
            return Err(invalid("pak entry block totals disagree"));
        }
        Ok(())
    }
}
// repak/src/pak.rs:454-464,496-500: directory map, filename map and
// signed entry locations for the full directory map.
fn read_directories(
    idx: &mut Cursor<Vec<u8>>,
    resolve: &mut impl FnMut(i32) -> io::Result<Option<Entry>>,
    entries: &mut BTreeMap<String, Entry>,
) -> io::Result<()> {
    // repak/src/pak.rs:454-464,750-757: counted directory/file maps.
    // Duplicate keys are rejected even if the referenced entry is identical.
    let mut directories = HashSet::new();
    for _ in 0..count(idx)? {
        let dir = string(idx)?;
        if !directories.insert(dir.clone()) {
            return Err(invalid("duplicate pak directory"));
        }
        let mut files = HashSet::new();
        for _ in 0..count(idx)? {
            let file = string(idx)?;
            if !files.insert(file.clone()) {
                return Err(invalid("duplicate pak directory filename"));
            }
            let location = i32r(idx)?;
            if let Some(entry) = resolve(location)? {
                let name = format!("{}{file}", dir.strip_prefix('/').unwrap_or(&dir));
                if let Some(old) = entries.get(&name) {
                    if old != &entry {
                        return Err(invalid("conflicting pak directory entries"));
                    }
                } else {
                    if entries.len() >= MAX_ITEMS as usize {
                        return Err(invalid("pak directory exceeds resource limit"));
                    }
                    entries.insert(name, entry);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests;
