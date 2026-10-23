use super::*;
use flate2::{write::ZlibEncoder, Compression};

// Project-owned synthetic writer, using only the field references in pak.rs.
// It does not read or embed upstream fixture archives or game content.
fn p32(v: &mut Vec<u8>, n: u32) {
    v.extend(n.to_le_bytes());
}
fn p64(v: &mut Vec<u8>, n: u64) {
    v.extend(n.to_le_bytes());
}
fn text(v: &mut Vec<u8>, s: &str) {
    p32(v, s.len() as u32 + 1);
    v.extend(s.as_bytes());
    v.push(0);
}
fn digest(data: &[u8]) -> [u8; 20] {
    Sha1::digest(data).into()
}
fn header(e: &Entry, byte_method: bool, data: bool) -> Vec<u8> {
    let mut v = Vec::new();
    p64(&mut v, if data { 0 } else { e.offset });
    p64(&mut v, e.compressed);
    p64(&mut v, e.uncompressed);
    if byte_method {
        v.push(e.method as u8);
    } else {
        p32(&mut v, e.method);
    }
    v.extend(e.hash.unwrap());
    if e.method != 0 {
        p32(&mut v, e.blocks.len() as u32);
        for &(start, end) in &e.blocks {
            p64(&mut v, start);
            p64(&mut v, end);
        }
    }
    v.push(e.flags);
    p32(&mut v, e.block_size);
    v
}
fn compact(e: &Entry, wide: bool) -> Vec<u8> {
    let mut v = Vec::new();
    let size_code = if e.block_size.is_multiple_of(2048) && e.block_size / 2048 < 63 {
        e.block_size / 2048
    } else {
        63
    };
    let bits = size_code
        | ((e.blocks.len() as u32) << 6)
        | (u32::from(e.flags & 1) << 22)
        | (e.method << 23)
        | if wide { 0 } else { 7 << 29 };
    p32(&mut v, bits);
    if size_code == 63 {
        p32(&mut v, e.block_size);
    }
    if wide {
        p64(&mut v, e.offset);
        p64(&mut v, e.uncompressed);
    } else {
        p32(&mut v, e.offset as u32);
        p32(&mut v, e.uncompressed as u32);
    }
    if e.method != 0 {
        if wide {
            p64(&mut v, e.compressed);
        } else {
            p32(&mut v, e.compressed as u32);
        }
        if e.blocks.len() > 1 || e.flags != 0 {
            for &(start, end) in &e.blocks {
                p32(&mut v, (end - start) as u32);
            }
        }
    }
    v
}
#[derive(Clone, Copy)]
pub(crate) enum Codec {
    None,
    Zlib,
    Zstd,
    Oodle,
}
fn compressed(codec: Codec, data: &[u8]) -> Vec<u8> {
    match codec {
        Codec::None | Codec::Oodle => data.to_vec(),
        Codec::Zstd => zstd::stream::encode_all(data, 1).unwrap(),
        Codec::Zlib => {
            let mut encoder = ZlibEncoder::new(Vec::new(), Compression::fast());
            encoder.write_all(data).unwrap();
            encoder.finish().unwrap()
        }
    }
}
#[derive(Default)]
pub(crate) struct Options {
    pub byte_method: bool,
    pub raw_shader_name: bool,
    pub shader_only: bool,
    pub encrypted_index: bool,
    pub encrypted_entry: bool,
    pub key: Option<[u8; 32]>,
    pub hash_only: bool,
    pub ordinary: bool,
    pub wide: bool,
    pub frozen: bool,
    pub utf16: bool,
    pub block_size: u32,
    pub secondary_after_primary: bool,
}
fn name(v: &mut Vec<u8>, s: &str, utf16: bool) {
    if utf16 {
        let units: Vec<_> = s.encode_utf16().chain([0]).collect();
        p32(v, (-(units.len() as i32)) as u32);
        for unit in units {
            v.extend(unit.to_le_bytes());
        }
    } else {
        text(v, s);
    }
}
pub(crate) fn random_key() -> [u8; 32] {
    let mut key = [0; 32];
    getrandom::fill(&mut key).unwrap();
    key
}
fn encrypt(data: &mut Vec<u8>, key: &[u8; 32]) {
    use aes::cipher::BlockEncryptMut;
    // Nonzero padding proves readers honor size alignment rather than relying
    // on zero padding or PKCS#7. No upstream or real archive bytes are used.
    data.resize(padded(data.len() as u64).unwrap() as usize, 0xa7);
    let len = data.len();
    ecb::Encryptor::<aes::Aes256>::new(key.into())
        .encrypt_padded_mut::<NoPadding>(data, len)
        .unwrap();
}
fn encrypt_index(data: &mut Vec<u8>, options: &Options) {
    if options.encrypted_index {
        if let Some(key) = &options.key {
            encrypt(data, key);
        }
    }
}
pub(crate) fn build(version: u32, codec: Codec, content: &[u8], options: Options) -> Vec<u8> {
    let byte_method = version == 8 && options.byte_method;
    let mut pak = vec![0xa5; 17]; // Ensure nonzero archive offset.
    let mut entries = Vec::new();
    for (name, data) in [
        (
            if options.raw_shader_name {
                "nested/synthetic-shaders.dat"
            } else {
                "nested/ShaderArchive-Synthetic.ushaderbytecode"
            },
            content,
        ),
        ("misc/anonymous.dat", content),
    ] {
        if options.shader_only && name == "misc/anonymous.dat" {
            continue;
        }
        let method = match codec {
            Codec::None => 0,
            Codec::Zlib => 1,
            Codec::Zstd => 2,
            Codec::Oodle => {
                if version < 8 {
                    3
                } else {
                    2
                }
            }
        };
        let block_size = if options.block_size == 0 {
            97
        } else {
            options.block_size as usize
        }; // Deliberately split record magic/header/bundles.
        let pieces: Vec<_> = if method == 0 {
            vec![data.to_vec()]
        } else {
            data.chunks(block_size)
                .map(|b| compressed(codec, b))
                .collect()
        };
        let mut stored = Vec::new();
        for piece in &pieces {
            let mut piece = piece.clone();
            if options.encrypted_entry {
                if let Some(key) = &options.key {
                    encrypt(&mut piece, key);
                }
            }
            stored.extend(piece);
        }
        let offset = pak.len() as u64;
        let mut e = Entry {
            offset,
            compressed: if method == 0 {
                data.len() as u64
            } else {
                stored.len() as u64
            },
            uncompressed: data.len() as u64,
            method,
            flags: u8::from(options.encrypted_entry),
            block_size: block_size as u32,
            blocks: Vec::new(),
            hash: Some(digest(&stored)),
        };
        if method != 0 {
            let mut start = 53 - if byte_method { 3 } else { 0 } + 4 + 16 * pieces.len() as u64;
            if version < 5 {
                start += offset;
            }
            for piece in pieces {
                let end = start + piece.len() as u64;
                e.blocks.push((start, end));
                start = if options.encrypted_entry && options.key.is_some() {
                    start + padded(piece.len() as u64).unwrap()
                } else {
                    end
                };
            }
        }
        pak.extend(header(&e, byte_method, true));
        pak.extend(&stored);
        entries.push((name, e));
    }
    let mut idx = Vec::new();
    let mut following_indexes: Vec<u8> = Vec::new();
    name(&mut idx, "../../../Synthetic/", options.utf16);
    p32(&mut idx, entries.len() as u32);
    if version < 10 {
        for (path, e) in &entries {
            name(&mut idx, path, options.utf16);
            idx.extend(header(e, byte_method, false));
        }
    } else {
        p64(&mut idx, 0x1234); // repak/src/pak.rs:402,560-562: path hash seed.
        let mut encoded = Vec::new();
        let mut locations = Vec::new();
        for (_, e) in &entries {
            locations.push(if options.ordinary {
                -(locations.len() as i32) - 1
            } else {
                encoded.len() as i32
            });
            if !options.ordinary {
                encoded.extend(compact(e, options.wide));
            }
        }
        let mut phi = Vec::new();
        p32(&mut phi, entries.len() as u32);
        for ((path, _), location) in entries.iter().zip(&locations) {
            // repak/src/pak.rs:699-716. Independent fixtures below do not use
            // this project writer or its hash routine.
            p64(&mut phi, path_hash(path, 0x1234));
            p32(&mut phi, *location as u32);
        }
        p32(&mut phi, 0); // empty pruned directory map.
        let mut fdi = Vec::new();
        p32(&mut fdi, entries.len() as u32);
        for ((path, _), location) in entries.iter().zip(&locations) {
            let (dir, file) = path.rsplit_once('/').unwrap();
            name(&mut fdi, &format!("/{dir}/"), options.utf16);
            p32(&mut fdi, 1);
            name(&mut fdi, file, options.utf16);
            p32(&mut fdi, *location as u32);
        }
        encrypt_index(&mut phi, &options);
        encrypt_index(&mut fdi, &options);
        // Exercise both secondary-index placements using explicit offsets.
        let phi_offset = pak.len() as u64;
        if !options.secondary_after_primary {
            pak.extend(&phi);
        }
        let fdi_offset = pak.len() as u64;
        if !options.hash_only && !options.secondary_after_primary {
            pak.extend(&fdi);
        }
        p32(&mut idx, 1);
        let phi_pointer = idx.len();
        p64(&mut idx, phi_offset);
        p64(&mut idx, phi.len() as u64);
        idx.extend(digest(&phi));
        p32(&mut idx, u32::from(!options.hash_only));
        let fdi_pointer = idx.len();
        if !options.hash_only {
            p64(&mut idx, fdi_offset);
            p64(&mut idx, fdi.len() as u64);
            idx.extend(digest(&fdi));
        }
        p32(&mut idx, encoded.len() as u32);
        idx.extend(encoded);
        p32(
            &mut idx,
            if options.ordinary {
                entries.len() as u32
            } else {
                0
            },
        );
        if options.ordinary {
            for (_, e) in &entries {
                idx.extend(header(e, false, false));
            }
        }
        if options.secondary_after_primary {
            let primary_size = if options.encrypted_index && options.key.is_some() {
                padded(idx.len() as u64).unwrap()
            } else {
                idx.len() as u64
            };
            let phi_offset = pak.len() as u64 + primary_size;
            idx[phi_pointer..phi_pointer + 8].copy_from_slice(&phi_offset.to_le_bytes());
            following_indexes.extend(&phi);
            if !options.hash_only {
                let fdi_offset = phi_offset + phi.len() as u64;
                idx[fdi_pointer..fdi_pointer + 8].copy_from_slice(&fdi_offset.to_le_bytes());
                following_indexes.extend(&fdi);
            }
        }
    }
    encrypt_index(&mut idx, &options);
    let offset = pak.len() as u64;
    pak.extend(&idx);
    pak.extend(following_indexes);
    if version >= 7 {
        pak.extend([0; 16]);
    }
    if version >= 4 {
        pak.push(u8::from(options.encrypted_index));
    }
    p32(&mut pak, MAGIC);
    p32(&mut pak, version);
    p64(&mut pak, offset);
    p64(&mut pak, idx.len() as u64);
    pak.extend(digest(&idx));
    if version == 9 {
        pak.push(u8::from(options.frozen));
    }
    if version >= 8 {
        let second = if matches!(codec, Codec::Oodle) {
            "Oodle"
        } else {
            "Zstd"
        };
        for method in ["Zlib", second, "", "", ""]
            .iter()
            .take(if byte_method { 4 } else { 5 })
        {
            let mut slot = [0; 32];
            slot[..method.len()].copy_from_slice(method.as_bytes());
            pak.extend(slot);
        }
    }
    pak
}

// Project experiment: append a library to the committed mixed-integrity pak.
// tools/pak-fixtures/integrity.py:36-63 defines this fixture's v7 layout;
// pak/tests.rs:15-40 serializes ordinary headers; pak.rs:431-494 reads footers.
// Original entry bytes and checksum evidence stay intact. The public fixture
// key and synthetic encrypted checksum domain are never authentication evidence.
pub(crate) fn integrity_library_fixture(content: &[u8], encrypted: bool) -> Vec<u8> {
    let original = include_bytes!("fixtures/synthetic-integrity.pak");
    let key_bytes = std::array::from_fn(|i| i as u8);
    let mut input = Cursor::new(original);
    let footer = Footer::read(&mut input).unwrap().unwrap();
    let mut archive = Pak::open_with_key(&mut input, Some(&AesKey(key_bytes.into())))
        .unwrap()
        .unwrap();
    let mut pak = original[..footer.offset as usize].to_vec();
    let mut stored = content.to_vec();
    if encrypted {
        encrypt(&mut stored, &key_bytes);
    }
    let entry = Entry {
        offset: pak.len() as u64,
        compressed: content.len() as u64,
        uncompressed: content.len() as u64,
        method: 0,
        hash: Some(digest(&stored)),
        flags: u8::from(encrypted),
        block_size: 0,
        blocks: Vec::new(),
    };
    pak.extend(header(&entry, false, true));
    pak.extend(stored);
    archive
        .entries
        .push(("libraries/Synthetic.ushaderbytecode".into(), entry));
    let mut index = Vec::new();
    text(&mut index, &archive.mount);
    p32(&mut index, archive.entries.len() as u32);
    for (name, entry) in &archive.entries {
        text(&mut index, name);
        index.extend(header(entry, false, false));
    }
    let index_hash = digest(&index);
    encrypt(&mut index, &key_bytes);
    let offset = pak.len() as u64;
    let size = index.len() as u64;
    pak.extend(index);
    pak.extend([0; 16]);
    pak.push(1);
    p32(&mut pak, MAGIC);
    p32(&mut pak, 7);
    p64(&mut pak, offset);
    p64(&mut pak, size);
    pak.extend(index_hash);
    pak
}

#[test]
fn all_versions_and_codecs_round_trip_multiple_blocks() {
    let payload: Vec<_> = (0..12345).map(|i| (i * 17) as u8).collect();
    for version in 3..=11 {
        for codec in [Codec::None, Codec::Zlib, Codec::Zstd] {
            if version < 8 && matches!(codec, Codec::Zstd) {
                continue;
            }
            for byte_method in [false, true] {
                let mut input = Cursor::new(build(
                    version,
                    codec,
                    &payload,
                    Options {
                        byte_method,
                        secondary_after_primary: byte_method,
                        utf16: true,
                        ..Options::default()
                    },
                ));
                let pak = Pak::open(&mut input).unwrap().unwrap();
                assert_eq!(pak.version, version);
                assert_eq!(pak.entries.len(), 2);
                for (_, entry) in &pak.entries {
                    let mut out = Vec::new();
                    pak.extract(&mut input, entry, &mut out, &mut Integrity::Unverified)
                        .unwrap();
                    assert_eq!(out, payload);
                }
            }
        }
    }
}
#[test]
fn compact_wide_ordinary_and_hash_only_indexes() {
    for options in [
        Options {
            wide: true,
            ..Options::default()
        },
        Options {
            ordinary: true,
            ..Options::default()
        },
        Options {
            block_size: 2048,
            ..Options::default()
        },
        Options {
            hash_only: true,
            ..Options::default()
        },
    ] {
        let mut input = Cursor::new(build(11, Codec::Zstd, b"synthetic content", options));
        let pak = Pak::open(&mut input).unwrap().unwrap();
        assert_eq!(pak.path_hash_seed, Some(0x1234));
        for (_, entry) in &pak.entries {
            let mut out = Vec::new();
            pak.extract(&mut input, entry, &mut out, &mut Integrity::Unverified)
                .unwrap();
            assert_eq!(out, b"synthetic content");
        }
    }
}
#[test]
fn encrypted_and_unsupported_archives_are_explicit() {
    for version in 4..=11 {
        let mut input = Cursor::new(build(
            version,
            Codec::None,
            b"synthetic",
            Options {
                encrypted_index: true,
                ..Options::default()
            },
        ));
        assert!(Pak::open(&mut input)
            .err()
            .unwrap()
            .to_string()
            .contains("encrypted pak index"));
    }
    let mut input = Cursor::new(build(
        9,
        Codec::None,
        b"synthetic",
        Options {
            frozen: true,
            ..Options::default()
        },
    ));
    assert!(Pak::open(&mut input)
        .err()
        .unwrap()
        .to_string()
        .contains("frozen"));
    for (codec, options, reason) in [
        (Codec::Oodle, Options::default(), "Oodle"),
        (
            Codec::Zlib,
            Options {
                encrypted_entry: true,
                ..Options::default()
            },
            "encrypted",
        ),
    ] {
        let mut input = Cursor::new(build(11, codec, b"synthetic", options));
        let pak = Pak::open(&mut input).unwrap().unwrap();
        for (_, entry) in &pak.entries {
            assert!(pak.skip_reason(entry).unwrap().contains(reason));
        }
    }
}
#[test]
fn corrupt_hash_range_header_and_block_fail_without_panics() {
    let original = build(11, Codec::Zlib, b"synthetic content", Options::default());
    let mut bad = original.clone();
    let last = bad.len() - 221;
    bad[last - 1] ^= 1;
    assert!(Pak::open(&mut Cursor::new(bad))
        .err()
        .unwrap()
        .to_string()
        .contains("SHA-1"));
    let mut bad = original.clone();
    bad[last + 25..last + 33].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(Pak::open(&mut Cursor::new(bad)).is_err());
    let mut input = Cursor::new(original.clone());
    let pak = Pak::open(&mut input).unwrap().unwrap();
    let entry = &pak.entries[0].1;
    // Corrupt a per-entry field without changing the authenticated index.
    input.get_mut()[entry.offset as usize + 8] ^= 1;
    assert!(pak
        .extract(
            &mut input,
            entry,
            &mut Vec::new(),
            &mut Integrity::Unverified
        )
        .unwrap_err()
        .to_string()
        .contains("disagrees"));
    let mut input = Cursor::new(original);
    let pak = Pak::open(&mut input).unwrap().unwrap();
    let entry = &pak.entries[0].1;
    input.get_mut()[(entry.offset + entry.blocks[0].0) as usize] ^= 0xff;
    assert!(pak
        .extract(
            &mut input,
            entry,
            &mut Vec::new(),
            &mut Integrity::Unverified
        )
        .is_err());
    for len in 0..230 {
        assert!(Pak::open(&mut Cursor::new(vec![0; len])).unwrap().is_none());
    }
}

#[test]
fn extraction_reads_and_writes_bounded_chunks() {
    struct Meter<R> {
        inner: R,
        largest: usize,
    }
    impl<R: Read> Read for Meter<R> {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            self.largest = self.largest.max(out.len());
            self.inner.read(out)
        }
    }
    impl<R: Seek> Seek for Meter<R> {
        fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
            self.inner.seek(position)
        }
    }
    struct Sink {
        hash: Sha1,
        size: usize,
        largest: usize,
    }
    impl Write for Sink {
        fn write(&mut self, data: &[u8]) -> io::Result<usize> {
            self.hash.update(data);
            self.size += data.len();
            self.largest = self.largest.max(data.len());
            Ok(data.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let payload: Vec<_> = (0..2 * 1024 * 1024).map(|i| (i * 13) as u8).collect();
    for codec in [Codec::None, Codec::Zlib, Codec::Zstd] {
        let mut input = Meter {
            inner: Cursor::new(build(
                11,
                codec,
                &payload,
                Options {
                    block_size: 65536,
                    ..Options::default()
                },
            )),
            largest: 0,
        };
        let pak = Pak::open(&mut input).unwrap().unwrap();
        input.largest = 0;
        let mut sink = Sink {
            hash: Sha1::new(),
            size: 0,
            largest: 0,
        };
        pak.extract(
            &mut input,
            &pak.entries[0].1,
            &mut sink,
            &mut Integrity::Unverified,
        )
        .unwrap();
        assert_eq!(sink.size, payload.len());
        assert_eq!(sink.hash.finalize()[..], digest(&payload));
        assert!(input.largest <= 64 * 1024, "read {}", input.largest);
        assert!(sink.largest <= 64 * 1024, "write {}", sink.largest);
    }
}

#[test]
fn authenticated_bad_frames_sizes_and_resource_limits_are_rejected() {
    let data = build(
        11,
        Codec::Zlib,
        b"synthetic frame payload",
        Options::default(),
    );
    let mut input = Cursor::new(data);
    let pak = Pak::open(&mut input).unwrap().unwrap();
    let e = &pak.entries[0].1;
    let start = (e.offset + e.blocks[0].0) as usize;
    let end = start + e.compressed as usize;
    input.get_mut()[start] ^= 0xff;
    // Compact entries have no index hash, so update the data-header hash to
    // test actual decompression rejection rather than checksum rejection.
    let hash = digest(&input.get_ref()[start..end]);
    input.get_mut()[e.offset as usize + 28..e.offset as usize + 48].copy_from_slice(&hash);
    let mut integrity = Integrity::Unverified;
    assert!(pak
        .extract(&mut input, e, &mut Vec::new(), &mut integrity)
        .is_err());
    // A later decompression failure does not erase a matching stored-byte hash.
    assert_eq!(integrity, Integrity::Verified);
    let mut input = Cursor::new(build(
        11,
        Codec::Zstd,
        b"synthetic frame payload",
        Options::default(),
    ));
    let pak = Pak::open(&mut input).unwrap().unwrap();
    let mut e = pak.entries[0].1.clone();
    e.uncompressed += 1;
    input.get_mut()[e.offset as usize + 16..e.offset as usize + 24]
        .copy_from_slice(&e.uncompressed.to_le_bytes());
    assert!(pak
        .extract(&mut input, &e, &mut Vec::new(), &mut Integrity::Unverified)
        .unwrap_err()
        .to_string()
        .contains("size mismatch"));
    assert!(string(&mut Cursor::new(i32::MIN.to_le_bytes())).is_err());
    assert!(count(&mut Cursor::new((MAX_ITEMS + 1).to_le_bytes())).is_err());
    let mut input = Cursor::new(vec![0; 1]);
    assert!(index(&mut input, 0, MAX_INDEX + 1, [0; 20], MAX_INDEX + 1, None).is_err());
}

#[test]
fn encrypted_indexes_and_entries_round_trip() {
    let key = random_key();
    let key = AesKey(Zeroizing::new(key));
    let content: Vec<_> = (0..131_119).map(|i| (i * 13) as u8).collect();
    for version in 4..=11 {
        for codec in [Codec::None, Codec::Zlib, Codec::Zstd] {
            if version < 8 && matches!(codec, Codec::Zstd) {
                continue;
            }
            for placement in [false, true] {
                for encrypted_entry in [false, true] {
                    let mut input = Cursor::new(build(
                        version,
                        codec,
                        &content,
                        Options {
                            encrypted_index: true,
                            encrypted_entry,
                            key: Some(*key.0),
                            secondary_after_primary: placement,
                            byte_method: placement,
                            ordinary: placement,
                            utf16: placement,
                            block_size: 32771,
                            hash_only: !placement,
                            wide: !placement,
                            ..Options::default()
                        },
                    ));
                    let pak = Pak::open_with_key(&mut input, Some(&key)).unwrap().unwrap();
                    assert_eq!(pak.entries.len(), 2);
                    for (_, entry) in &pak.entries {
                        let mut out = Vec::new();
                        pak.extract(&mut input, entry, &mut out, &mut Integrity::Unverified)
                            .unwrap();
                        assert_eq!(out, content);
                    }
                }
            }
        }
    }
}

#[test]
fn wrong_keys_and_damaged_encrypted_indexes_are_rejected() {
    let key = AesKey(Zeroizing::new(random_key()));
    let wrong = AesKey(Zeroizing::new(random_key()));
    for version in [4, 8, 10, 11] {
        let archive = build(
            version,
            Codec::None,
            b"synthetic",
            Options {
                encrypted_index: true,
                key: Some(*key.0),
                ..Options::default()
            },
        );
        let error = Pak::open_with_key(&mut Cursor::new(&archive), Some(&wrong))
            .err()
            .unwrap();
        assert!(error.to_string().starts_with("malformed index:"));
        let mut input = Cursor::new(archive);
        let f = Footer::read(&mut input).unwrap().unwrap();
        let mut data = input.into_inner();
        let idx = &mut data[f.offset as usize..(f.offset + f.size) as usize];
        key.decrypt(idx).unwrap();
        // Valid encryption but invalid grammar must still reject, irrespective
        // of the hash: a count exceeding policy cannot be trusted.
        let at = 4 + "../../../Synthetic/".len() + 1;
        idx[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
        let mut idx = idx.to_vec();
        encrypt(&mut idx, &key.0);
        data[f.offset as usize..(f.offset + f.size) as usize].copy_from_slice(&idx);
        let error = Pak::open_with_key(&mut Cursor::new(data), Some(&key))
            .err()
            .unwrap();
        assert!(error.to_string().starts_with("malformed index:"));
    }
}

#[test]
fn encrypted_index_alignment_and_entry_boundaries_are_checked() {
    let key = AesKey(Zeroizing::new(random_key()));
    let mut archive = build(
        7,
        Codec::None,
        b"short",
        Options {
            encrypted_index: true,
            encrypted_entry: true,
            key: Some(*key.0),
            ..Options::default()
        },
    );
    let mut input = Cursor::new(&archive);
    let f = Footer::read(&mut input).unwrap().unwrap();
    let pak = Pak::open_with_key(&mut input, Some(&key)).unwrap().unwrap();
    let mut entry = pak.entries[1].1.clone();
    entry.offset = pak.data_end - 53 - entry.compressed;
    assert!(pak
        .extract(
            &mut input,
            &entry,
            &mut Vec::new(),
            &mut Integrity::Unverified
        )
        .is_err());
    // Footer index-size field: guid + flag + magic + version + offset.
    let at = f.start as usize + 16 + 1 + 8 + 8;
    archive[at..at + 8].copy_from_slice(&(f.size - 1).to_le_bytes());
    assert_eq!(
        Pak::open_with_key(&mut Cursor::new(archive), Some(&key))
            .err()
            .unwrap()
            .to_string(),
        "truncated data: encrypted index size is not a positive AES block multiple"
    );
}

#[test]
fn plain_index_never_claims_to_verify_an_encrypted_entry_key() {
    let key = AesKey(Zeroizing::new(random_key()));
    let mut input = Cursor::new(build(
        7,
        Codec::None,
        b"synthetic",
        Options {
            encrypted_entry: true,
            key: Some(*key.0),
            ..Options::default()
        },
    ));
    let pak = Pak::open_with_key(&mut input, Some(&key)).unwrap().unwrap();
    assert!(pak
        .skip_reason(&pak.entries[0].1)
        .unwrap()
        .contains("cannot be verified"));
    assert!(pak
        .extract(
            &mut input,
            &pak.entries[0].1,
            &mut Vec::new(),
            &mut Integrity::Unverified
        )
        .is_err());
}

#[test]
fn key_parser_accepts_hex_and_rejects_other_formats_without_disclosure() {
    let key = random_key();
    for uppercase in [false, true] {
        let hex: String = key
            .iter()
            .map(|b| {
                if uppercase {
                    format!("{b:02X}")
                } else {
                    format!("{b:02x}")
                }
            })
            .collect();
        for value in [hex.clone(), format!(" \n0x{hex}\r\n")] {
            assert!(*AesKey::parse(value.as_bytes()).unwrap().0 == key);
        }
    }
    for value in [
        b"secret-invalid-content".as_slice(),
        &[b'f'; 63],
        &[b'f'; 65],
        &[b'g'; 64],
        &[0xff; 64],
    ] {
        assert_eq!(
            AesKey::parse(value).err().unwrap().to_string(),
            "pak AES key must contain exactly 64 hex characters"
        );
    }
}

#[test]
fn secondary_index_decryption_must_also_pass_structure_checks() {
    let key = AesKey(Zeroizing::new(random_key()));
    let mut input = Cursor::new(build(
        11,
        Codec::Zstd,
        b"synthetic",
        Options {
            encrypted_index: true,
            encrypted_entry: true,
            key: Some(*key.0),
            ..Options::default()
        },
    ));
    let f = Footer::read(&mut input).unwrap().unwrap();
    let mut primary = index(&mut input, f.offset, f.size, f.hash, f.start, Some(&key)).unwrap();
    string(&mut primary).unwrap();
    count(&mut primary).unwrap();
    u64r(&mut primary).unwrap();
    descriptor(&mut primary).unwrap();
    let (offset, size, _) = descriptor(&mut primary).unwrap().unwrap();
    let mut archive = input.into_inner();
    let mut secondary = archive[offset as usize..(offset + size) as usize].to_vec();
    key.decrypt(&mut secondary).unwrap();
    secondary[..4].copy_from_slice(&u32::MAX.to_le_bytes());
    encrypt(&mut secondary, &key.0);
    archive[offset as usize..(offset + size) as usize].copy_from_slice(&secondary);
    assert_eq!(
        Pak::open_with_key(&mut Cursor::new(archive), Some(&key))
            .err()
            .unwrap()
            .to_string(),
        "malformed index: pak item count exceeds resource limit"
    );
}

// Independent serializer: pinned repak, not build()/compact()/header() above.
// Generator and checksum limitations: tools/pak-fixtures/generate.py,
// docs/reference-sources.md All bytes and the key are synthetic.
const INDEPENDENT_KEY: [u8; 32] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31,
];
const INDEPENDENT_ENCRYPTED: &[u8] = include_bytes!("fixtures/repak-ascii-encrypted.pak");

#[test]
fn independent_repak_indexes_and_entries() {
    let key = AesKey(Zeroizing::new(INDEPENDENT_KEY));
    for (data, mount, encrypted) in [
        (
            include_bytes!("fixtures/repak-ascii-plain.pak").as_slice(),
            "../../../SyntheticIndependentFixture/",
            false,
        ),
        (
            INDEPENDENT_ENCRYPTED,
            "../../../SyntheticIndependentFixture/",
            true,
        ),
        (
            include_bytes!("fixtures/repak-utf16-plain.pak").as_slice(),
            "../../../Synthetic独立Fixture/",
            false,
        ),
        (
            include_bytes!("fixtures/repak-utf16-encrypted.pak").as_slice(),
            "../../../Synthetic独立Fixture/",
            true,
        ),
    ] {
        let mut input = Cursor::new(data);
        let f = Footer::read(&mut input).unwrap().unwrap();
        assert_eq!(f.version, 11);
        assert_eq!(f.start, data.len() as u64 - 221);
        assert_eq!(f.encrypted, encrypted);
        assert!(!f.offset.is_multiple_of(16));
        input.set_position(113); // Seek origin is byte zero, not caller position.
        let pak = Pak::open_with_key(&mut input, Some(&key)).unwrap().unwrap();
        assert_eq!(pak.mount, mount);
        assert!(pak.mount.len() > 16);
        assert_eq!(pak.path_hash_seed, Some(0x1234));
        assert_eq!(pak.entries.len(), 2);
        for (name, entry) in &pak.entries {
            let mut out = Vec::new();
            pak.extract(&mut input, entry, &mut out, &mut Integrity::Unverified)
                .unwrap();
            if name == "plain/payload.txt" {
                assert_eq!(entry.method, 0);
                assert_eq!(out, b"Independent synthetic plain payload.");
            } else {
                assert_eq!(name, "compressed/repeated.txt");
                assert_eq!(pak.method(entry), "Zlib");
                assert_eq!(out, vec![b'Z'; 150_000]);
            }
        }
        // repak/src/pak.rs:397-408,435-438: start with FString, then
        // count/seed and both secondary descriptors. Each extent is decrypted.
        let mut primary = index(
            &mut input,
            f.offset,
            f.size,
            f.hash,
            f.start,
            encrypted.then_some(&key),
        )
        .unwrap();
        assert_eq!(string(&mut primary).unwrap(), mount);
        assert_eq!(count(&mut primary).unwrap(), 2);
        assert_eq!(u64r(&mut primary).unwrap(), 0x1234);
        let phi = descriptor(&mut primary).unwrap().unwrap();
        let fdi = descriptor(&mut primary).unwrap().unwrap();
        for (offset, size, hash) in [phi, fdi] {
            let mut secondary = index(
                &mut input,
                offset,
                size,
                hash,
                f.start,
                encrypted.then_some(&key),
            )
            .unwrap();
            assert!(count(&mut secondary).unwrap() > 0);
            if encrypted {
                assert!(size.is_multiple_of(16));
                assert!(!offset.is_multiple_of(16));
            }
        }
    }
}

#[test]
fn independent_entry_hash_failures_are_bad_hash_and_write_no_output() {
    // repak/src/data.rs:173,184,195: SHA-1 covers compressed stored
    // bytes or the uncompressed payload. entry.rs:121-135,154-176:
    // legacy ordinary indexes and data headers both carry that hash.
    for (data, name, reason) in [
        (
            include_bytes!("fixtures/repak-plain-entry-bad-hash.pak").as_slice(),
            "plain/payload.txt",
            "pak entry SHA-1 mismatch",
        ),
        (
            include_bytes!("fixtures/repak-compressed-entry-bad-hash.pak").as_slice(),
            "compressed/repeated.txt",
            "pak entry SHA-1 mismatch",
        ),
        (
            include_bytes!("fixtures/repak-legacy-entry-hash-disagreement.pak").as_slice(),
            "plain/payload.txt",
            "pak entry hash disagrees with index",
        ),
    ] {
        let mut input = Cursor::new(data);
        // The primary and secondary index checksums and grammar still pass.
        let pak = Pak::open(&mut input).unwrap().unwrap();
        let entry = &pak.entries.iter().find(|(path, _)| path == name).unwrap().1;
        let mut output = Vec::new();
        let error = pak
            .extract(&mut input, entry, &mut output, &mut Integrity::Unverified)
            .unwrap_err();
        assert_eq!(category(&error), Diagnostic::BadHash);
        assert_eq!(error.to_string(), format!("bad hash: {reason}"));
        assert!(output.is_empty());
        assert!(!error.to_string().contains("wrong key"));
        // The other entry is intact, including its own SHA-1.
        let other = &pak.entries.iter().find(|(path, _)| path != name).unwrap().1;
        pak.extract(&mut input, other, &mut output, &mut Integrity::Unverified)
            .unwrap();
        assert!(!output.is_empty());
    }
}

#[test]
fn independent_ecb_blocks_and_first_fstring() {
    let key = AesKey(Zeroizing::new(INDEPENDENT_KEY));
    for (data, expected, negative) in [
        (
            INDEPENDENT_ENCRYPTED,
            "../../../SyntheticIndependentFixture/",
            false,
        ),
        (
            include_bytes!("fixtures/repak-utf16-encrypted.pak").as_slice(),
            "../../../Synthetic独立Fixture/",
            true,
        ),
    ] {
        let mut input = Cursor::new(data);
        let f = Footer::read(&mut input).unwrap().unwrap();
        let ciphertext = &data[f.offset as usize..(f.offset + f.size) as usize];
        let mut independent = Vec::new();
        for block in ciphertext.as_chunks::<16>().0.iter() {
            let mut block = block.to_vec();
            key.decrypt(&mut block).unwrap();
            independent.extend(block);
        }
        let mut whole = ciphertext.to_vec();
        key.decrypt(&mut whole).unwrap();
        assert_eq!(whole, independent);
        assert_eq!(
            i32::from_le_bytes(whole[..4].try_into().unwrap()) < 0,
            negative
        );
        assert_eq!(string(&mut Cursor::new(whole.clone())).unwrap(), expected);
        // Reversing ciphertext blocks reverses plaintext blocks. ECB has no
        // chaining state or IV (repak/src/pak.rs:151-161,763-768).
        let mut reversed: Vec<_> = ciphertext
            .as_chunks::<16>()
            .0
            .iter()
            .rev()
            .flatten()
            .copied()
            .collect();
        key.decrypt(&mut reversed).unwrap();
        assert_eq!(
            reversed,
            whole
                .as_chunks::<16>()
                .0
                .iter()
                .rev()
                .flatten()
                .copied()
                .collect::<Vec<_>>()
        );
        assert_eq!(string(&mut Cursor::new(0i32.to_le_bytes())).unwrap(), "");
    }
}

#[test]
fn key_candidate_screen_rejects_all_one_bit_misses_with_one_block_read() {
    struct Meter {
        data: Cursor<&'static [u8]>,
        read: usize,
        seeks: Vec<u64>,
    }
    impl Read for Meter {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            let n = self.data.read(out)?;
            self.read += n;
            Ok(n)
        }
    }
    impl Seek for Meter {
        fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
            let at = self.data.seek(pos)?;
            self.seeks.push(at);
            Ok(at)
        }
    }
    let f = Footer::read(&mut Cursor::new(INDEPENDENT_ENCRYPTED))
        .unwrap()
        .unwrap();
    for bit in 0..=256 {
        let mut raw = INDEPENDENT_KEY;
        if bit < 256 {
            raw[bit / 8] ^= 1 << (bit % 8);
        }
        let key = AesKey(Zeroizing::new(raw));
        let mut input = Meter {
            data: Cursor::new(INDEPENDENT_ENCRYPTED),
            read: 0,
            seeks: vec![],
        };
        assert_eq!(
            key_candidate_prefix(&mut input, &f, &key).unwrap(),
            bit == 256
        );
        // A deterministic work bound is stronger than a flaky timing threshold.
        assert_eq!(input.read, 16);
        assert_eq!(input.seeks, [f.offset]);
    }
    // Passing the screen still requires full primary and secondary parsing.
    let key = AesKey(Zeroizing::new(INDEPENDENT_KEY));
    assert!(
        Pak::open_with_key(&mut Cursor::new(INDEPENDENT_ENCRYPTED), Some(&key))
            .unwrap()
            .is_some()
    );
}

fn category(error: &io::Error) -> Diagnostic {
    error
        .get_ref()
        .unwrap()
        .downcast_ref::<PakError>()
        .unwrap()
        .category
}

#[test]
fn structural_errors_keep_their_categories_with_the_right_key() {
    let key = AesKey(Zeroizing::new(INDEPENDENT_KEY));
    let mut input = Cursor::new(INDEPENDENT_ENCRYPTED);
    let f = Footer::read(&mut input).unwrap().unwrap();
    let mut primary = index(&mut input, f.offset, f.size, f.hash, f.start, Some(&key)).unwrap();
    string(&mut primary).unwrap();
    let count_at = primary.position() as usize;
    count(&mut primary).unwrap();
    u64r(&mut primary).unwrap();
    let phi = descriptor(&mut primary).unwrap().unwrap();
    let fdi = descriptor(&mut primary).unwrap().unwrap();
    for (offset, size, field, value, expected) in [
        (
            f.offset,
            f.size,
            count_at,
            u32::MAX,
            Diagnostic::MalformedIndex,
        ),
        (
            phi.0,
            phi.1,
            4 + 2 * 12,
            1,
            Diagnostic::UnsupportedIndexVariant,
        ),
        (fdi.0, fdi.1, 0, u32::MAX, Diagnostic::MalformedIndex),
    ] {
        let mut data = INDEPENDENT_ENCRYPTED.to_vec();
        let extent = &mut data[offset as usize..(offset + size) as usize];
        key.decrypt(extent).unwrap();
        extent[field..field + 4].copy_from_slice(&value.to_le_bytes());
        let mut encrypted = extent.to_vec();
        encrypt(&mut encrypted, &key.0);
        extent.copy_from_slice(&encrypted);
        let error = Pak::open_with_key(&mut Cursor::new(data), Some(&key))
            .err()
            .unwrap();
        assert_eq!(category(&error), expected, "{error}");
        assert!(!error.to_string().contains("wrong key"));
    }
    let mut data = INDEPENDENT_ENCRYPTED.to_vec();
    data[f.start as usize + 33..f.start as usize + 41].copy_from_slice(&(f.size - 1).to_le_bytes());
    let error = Pak::open_with_key(&mut Cursor::new(data), Some(&key))
        .err()
        .unwrap();
    assert_eq!(category(&error), Diagnostic::TruncatedData);
    let mut data = include_bytes!("fixtures/repak-ascii-plain.pak").to_vec();
    let f = Footer::read(&mut Cursor::new(&data)).unwrap().unwrap();
    data[f.start as usize + 41] ^= 1;
    let error = Pak::open(&mut Cursor::new(data)).err().unwrap();
    assert_eq!(category(&error), Diagnostic::BadHash);
}

fn replace_encrypted_extent(data: &mut [u8], offset: u64, size: u64, edit: impl FnOnce(&mut [u8])) {
    let key = AesKey(Zeroizing::new(INDEPENDENT_KEY));
    let extent = &mut data[offset as usize..(offset + size) as usize];
    key.decrypt(extent).unwrap();
    edit(extent);
    let mut encrypted = extent.to_vec();
    encrypt(&mut encrypted, &key.0);
    extent.copy_from_slice(&encrypted);
}

#[test]
fn independent_secondary_hashes_and_record_boundaries_are_checked() {
    let key = AesKey(Zeroizing::new(INDEPENDENT_KEY));
    for encrypted in [false, true] {
        let original = if encrypted {
            INDEPENDENT_ENCRYPTED
        } else {
            include_bytes!("fixtures/repak-ascii-plain.pak").as_slice()
        };
        let mut input = Cursor::new(original);
        let f = Footer::read(&mut input).unwrap().unwrap();
        let mut primary = index(
            &mut input,
            f.offset,
            f.size,
            f.hash,
            f.start,
            encrypted.then_some(&key),
        )
        .unwrap();
        string(&mut primary).unwrap();
        count(&mut primary).unwrap();
        u64r(&mut primary).unwrap();
        let phi = descriptor(&mut primary).unwrap().unwrap();
        let fdi_size_at = primary.position() as usize + 12;
        let fdi = descriptor(&mut primary).unwrap().unwrap();
        for extent in [phi, fdi] {
            let mut data = original.to_vec();
            if encrypted {
                // Alter a secondary field while preserving valid AES blocks.
                // Secondary truncation errors must not become wrong-key errors.
                replace_encrypted_extent(&mut data, f.offset, f.size, |primary| {
                    primary[fdi_size_at..fdi_size_at + 8].copy_from_slice(&16u64.to_le_bytes());
                });
            } else {
                data[extent.0 as usize + 4] ^= 1;
            }
            let error = Pak::open_with_key(&mut Cursor::new(data), Some(&key))
                .err()
                .unwrap();
            assert_eq!(
                category(&error),
                if encrypted {
                    Diagnostic::TruncatedData
                } else {
                    Diagnostic::BadHash
                }
            );
        }
        if encrypted {
            for edit in [0, 1] {
                let mut data = original.to_vec();
                replace_encrypted_extent(&mut data, phi.0, phi.1, |phi| {
                    if edit == 0 {
                        // First location points into, rather than at, a record.
                        phi[12..16].copy_from_slice(&1i32.to_le_bytes());
                    } else {
                        // Valid record location, but hash no longer matches FDI.
                        phi[4] ^= 1;
                    }
                });
                let error = Pak::open_with_key(&mut Cursor::new(data), Some(&key))
                    .err()
                    .unwrap();
                assert_eq!(category(&error), Diagnostic::MalformedIndex);
                assert!(error.to_string().contains(if edit == 0 {
                    "compact entry location"
                } else {
                    "indexes disagree"
                }));
            }
        }
    }
}

#[test]
fn encrypted_zero_length_mount_is_an_empty_fstring() {
    let key = AesKey(Zeroizing::new(INDEPENDENT_KEY));
    let mut input = Cursor::new(INDEPENDENT_ENCRYPTED);
    let f = Footer::read(&mut input).unwrap().unwrap();
    let plain_f = Footer::read(&mut Cursor::new(
        include_bytes!("fixtures/repak-ascii-plain.pak").as_slice(),
    ))
    .unwrap()
    .unwrap();
    let mut idx = index(&mut input, f.offset, f.size, f.hash, f.start, Some(&key)).unwrap();
    string(&mut idx).unwrap();
    let mut primary = 0i32.to_le_bytes().to_vec();
    primary.extend_from_slice(&idx.get_ref()[idx.position() as usize..plain_f.size as usize]);
    encrypt(&mut primary, &key.0);
    let mut data = INDEPENDENT_ENCRYPTED.to_vec();
    data[f.offset as usize..f.offset as usize + primary.len()].copy_from_slice(&primary);
    data[f.start as usize + 33..f.start as usize + 41]
        .copy_from_slice(&(primary.len() as u64).to_le_bytes());
    let pak = Pak::open_with_key(&mut Cursor::new(data), Some(&key))
        .unwrap()
        .unwrap();
    assert!(pak.mount.is_empty());
    assert_eq!(pak.entries.len(), 2);
}

#[test]
fn entry_integrity_requires_a_matching_hash() {
    let data = b"Synthetic integrity reporting payload.";
    let mut input = Cursor::new(data);
    assert_eq!(
        check_entry_hash(&mut input, data.len() as u64, None).unwrap(),
        Integrity::Unverified
    );
    assert_eq!(input.position(), 0);
    assert_eq!(
        check_entry_hash(&mut input, data.len() as u64, Some(digest(data))).unwrap(),
        Integrity::Verified
    );
    input.set_position(0);
    let error = check_entry_hash(&mut input, data.len() as u64, Some([0; 20])).unwrap_err();
    assert_eq!(category(&error), Diagnostic::BadHash);
}
