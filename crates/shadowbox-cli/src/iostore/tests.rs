use super::*;
use aes::cipher::{block_padding::NoPadding, BlockEncryptMut, KeyInit};
use flate2::{write::ZlibEncoder, Compression};

// All bytes are project-authored synthetic containers using the field citations
// in iostore.rs and shaders.rs. No upstream fixtures or game data are read.
pub(crate) fn p32(v: &mut Vec<u8>, n: u32) {
    v.extend(n.to_le_bytes());
}
fn put32(v: &mut [u8], at: usize, n: u32) {
    v[at..at + 4].copy_from_slice(&n.to_le_bytes());
}
fn put64(v: &mut [u8], at: usize, n: u64) {
    v[at..at + 8].copy_from_slice(&n.to_le_bytes());
}
fn text(v: &mut Vec<u8>, text: &str) {
    if text.is_ascii() {
        p32(v, text.len() as u32 + 1);
        v.extend(text.as_bytes());
        v.push(0);
    } else {
        let words: Vec<_> = text.encode_utf16().collect();
        v.extend((-(words.len() as i32 + 1)).to_le_bytes());
        for w in words {
            v.extend(w.to_le_bytes());
        }
        v.extend([0; 2]);
    }
}
pub(crate) fn encrypt(v: &mut Vec<u8>) {
    v.resize((v.len() + 15) & !15, 0);
    let n = v.len();
    ecb::Encryptor::<aes::Aes256>::new((&[0x37; 32]).into())
        .encrypt_padded_mut::<NoPadding>(v, n)
        .unwrap();
}
pub(crate) fn key() -> AesKey {
    AesKey::parse("37".repeat(32).as_bytes()).unwrap()
}
pub(crate) fn code_id(version: u8, n: u8) -> [u8; 12] {
    let mut id = [0; 12];
    id[0] = n;
    id[11] = if version > 4 { 9 } else { 12 };
    id
}
pub(crate) struct Fixture {
    pub toc: Vec<u8>,
    pub data: Vec<Vec<u8>>,
}
impl Fixture {
    pub fn write(&self, path: &Path) {
        fs::write(path, &self.toc).unwrap();
        for (i, bytes) in self.data.iter().enumerate() {
            fs::write(partition_path(path, i as u64), bytes).unwrap();
        }
    }
}
pub(crate) fn compressed(method: &str, bytes: &[u8]) -> Vec<u8> {
    match method {
        "None" | "Oodle" => bytes.to_vec(),
        "Zlib" => {
            let mut z = ZlibEncoder::new(Vec::new(), Compression::default());
            z.write_all(bytes).unwrap();
            z.finish().unwrap()
        }
        "Zstd" => zstd::stream::encode_all(bytes, 1).unwrap(),
        "LZ4" => lz4_flex::block::compress(bytes),
        _ => panic!("unknown synthetic codec"),
    }
}

pub(crate) fn container(
    version: u8,
    entries: &[([u8; 12], Option<&str>, Vec<u8>)],
    codec: &str,
    encrypted: bool,
    partition_size: u64,
) -> Fixture {
    let mut logical = vec![0x55; 9]; // chunk offsets deliberately start inside a block.
    let mut ranges = Vec::new();
    for (_, _, bytes) in entries {
        ranges.push((logical.len() as u64, bytes.len() as u64));
        logical.extend(bytes);
    }
    let methods = ["Zlib", "Zstd", "LZ4", "Oodle"];
    let mut data = vec![Vec::new()];
    let mut blocks = Vec::new();
    for (i, block) in logical.chunks(64).enumerate() {
        let method = if codec == "mixed" {
            ["None", "Zlib", "Zstd", "LZ4"][i % 4]
        } else {
            codec
        };
        let index = methods
            .iter()
            .position(|m| m == &method)
            .map_or(0, |n| n + 1);
        let mut stored = compressed(method, block);
        let size = stored.len();
        if encrypted {
            encrypt(&mut stored);
        } else {
            stored.resize((stored.len() + 15) & !15, 0);
        }
        if data.last().unwrap().len() as u64 + stored.len() as u64 > partition_size {
            data.push(Vec::new());
        }
        let offset = (data.len() as u64 - 1) * partition_size + data.last().unwrap().len() as u64;
        blocks.extend(&offset.to_le_bytes()[..5]);
        blocks.extend(&(size as u32).to_le_bytes()[..3]);
        blocks.extend(&(block.len() as u32).to_le_bytes()[..3]);
        blocks.push(index as u8);
        data.last_mut().unwrap().extend(stored);
    }
    let mut directory = Vec::new();
    let named: Vec<_> = entries
        .iter()
        .enumerate()
        .filter_map(|(i, (_, name, _))| name.map(|n| (i, n)))
        .collect();
    if !named.is_empty() {
        text(&mut directory, "../../../Synthetic/");
        p32(&mut directory, 2);
        for n in [u32::MAX, 1, u32::MAX, u32::MAX, 0, u32::MAX, u32::MAX, 0] {
            p32(&mut directory, n);
        }
        p32(&mut directory, named.len() as u32);
        for (j, &(i, _)) in named.iter().enumerate() {
            p32(&mut directory, j as u32 + 1);
            p32(
                &mut directory,
                if j + 1 == named.len() {
                    u32::MAX
                } else {
                    j as u32 + 1
                },
            );
            p32(&mut directory, i as u32);
        }
        p32(&mut directory, named.len() as u32 + 1);
        text(&mut directory, "Shaders");
        for &(_, n) in &named {
            text(&mut directory, n);
        }
        if encrypted {
            encrypt(&mut directory);
        }
    }
    let mut toc = vec![0; 144];
    toc[..16].copy_from_slice(MAGIC);
    toc[16] = version;
    put32(&mut toc, 20, 144);
    put32(&mut toc, 24, entries.len() as u32);
    put32(&mut toc, 28, blocks.len() as u32 / 12);
    put32(&mut toc, 32, 12);
    put32(&mut toc, 36, 4);
    put32(&mut toc, 40, 32);
    put32(&mut toc, 44, 64);
    put32(&mut toc, 48, directory.len() as u32);
    put32(&mut toc, 52, data.len() as u32);
    put64(&mut toc, 56, 0x1122334455667788);
    toc[64..80].fill(0x44);
    toc[80] = u8::from(codec != "None")
        | (u8::from(encrypted) << 1)
        | (u8::from(!directory.is_empty()) << 3);
    put64(&mut toc, 88, partition_size);
    for &(id, _, _) in entries {
        toc.extend(id);
    }
    for &(offset, length) in &ranges {
        toc.extend(&offset.to_be_bytes()[3..]);
        toc.extend(&length.to_be_bytes()[3..]);
    }
    if version >= 4 {
        put32(&mut toc, 84, 2);
        toc.extend([0x51; 8]);
    }
    if version >= 5 {
        put32(&mut toc, 96, 1);
        toc.extend(0u32.to_le_bytes());
    }
    toc.extend(blocks);
    for method in methods {
        let mut m = [0; 32];
        m[..method.len()].copy_from_slice(method.as_bytes());
        toc.extend(m);
    }
    toc.extend(directory);
    for _ in entries {
        toc.extend(vec![0x99; if version >= 8 { 20 } else { 32 }]);
        toc.push(u8::from(codec != "None"));
        if version >= 8 {
            toc.extend([0; 3]);
        }
    }
    Fixture { toc, data }
}

#[test]
fn header_versions_directory_names_and_block_methods() {
    for version in 2..=8 {
        let fixture = container(
            version,
            &[
                (
                    code_id(version, 1),
                    Some("Éclairage.ushaderbytecode"),
                    vec![0xab; 157],
                ),
                (code_id(version, 2), None, vec![0x42; 89]),
            ],
            "mixed",
            false,
            u64::MAX,
        );
        let toc = Toc::parse(&fixture.toc, None).unwrap();
        assert_eq!(toc.version, version);
        assert_eq!(toc.chunks[0].offset, 9);
        assert_eq!(toc.chunks[0].length, 157);
        assert!(toc.shader_code(&toc.chunks[0]));
        assert_eq!(
            toc.chunks[0].name.as_deref(),
            Some("Shaders/Éclairage.ushaderbytecode")
        );
        assert_eq!(toc.mount, "../../../Synthetic/");
        assert_eq!(toc.compression(&toc.chunks[0]), ["None", "Zlib", "Zstd"]);
    }
    let fixture = container(
        1,
        &[(code_id(1, 1), None, vec![0x42; 100])],
        "None",
        false,
        u64::MAX,
    );
    assert!(Toc::parse(&fixture.toc, None).is_ok());
}

#[test]
fn unsupported_versions_bad_counts_and_truncated_tables() {
    let fixture = container(
        8,
        &[(code_id(8, 1), Some("a"), vec![0xab; 150])],
        "None",
        false,
        u64::MAX,
    );
    for len in [0, 15, 143, 155, 165, fixture.toc.len() - 1] {
        assert!(Toc::parse(&fixture.toc[..len], None).is_err());
    }
    for (at, value) in [
        (20, 145),
        (24, u32::MAX),
        (28, u32::MAX),
        (32, 13),
        (36, 256),
        (40, u32::MAX),
        (44, 0),
        (48, u32::MAX),
        (52, 0),
        (84, u32::MAX),
        (96, u32::MAX),
    ] {
        let mut data = fixture.toc.clone();
        put32(&mut data, at, value);
        assert!(Toc::parse(&data, None).is_err(), "field {at}");
    }
    for version in [0, 9, 255] {
        let mut data = fixture.toc.clone();
        data[16] = version;
        assert!(Toc::parse(&data, None).is_err());
    }
    for flags in [16, 128] {
        let mut data = fixture.toc.clone();
        data[80] |= flags;
        assert!(Toc::parse(&data, None).is_err());
    }
    let mut extra = fixture.toc.clone();
    extra.push(0);
    assert!(Toc::parse(&extra, None).is_err());
    let mut outside = fixture.toc.clone();
    outside[156..161].fill(0xff);
    assert!(Toc::parse(&outside, None).is_err());
}

#[test]
fn directory_cycles_invalid_links_and_padding() {
    let fixture = container(
        8,
        &[(code_id(8, 1), Some("x"), vec![1; 20])],
        "None",
        false,
        u64::MAX,
    );
    let directory_at = 144 + 12 + 10 + 8 + 4 + 12 + 128;
    let mount_size = 4 + "../../../Synthetic/".len() + 1;
    let root_at = directory_at + mount_size + 4;
    for (at, value) in [
        (root_at + 4, 0),
        (root_at + 4, 999),
        (root_at + 16, 999),
        (root_at + 28, 999),
        (root_at + 36 + 8, 999),
        (root_at + 36 + 4, 0),
    ] {
        let mut data = fixture.toc.clone();
        put32(&mut data, at, value);
        assert!(Toc::parse(&data, None).is_err(), "directory field {at}");
    }
    let mut data = fixture.toc.clone();
    data[directory_at + 4] = 0xff;
    assert!(Toc::parse(&data, None).is_err());
}

#[test]
fn codecs_require_exact_expanded_lengths() {
    let original = vec![0x5a; 240];
    for codec in ["None", "Zlib", "Zstd", "LZ4"] {
        let encoded = compressed(codec, &original);
        assert_eq!(
            decompress(codec, &encoded, original.len()).unwrap(),
            original
        );
        assert!(decompress(codec, &encoded, original.len() - 1).is_err());
        assert!(decompress(codec, &encoded, original.len() + 1).is_err());
        if codec != "None" {
            assert!(decompress(codec, &encoded[..encoded.len() - 2], original.len()).is_err());
        }
    }
    assert!(decompress("Oodle", &[1; 3], 4)
        .unwrap_err()
        .to_string()
        .contains("unsupported"));
}

pub(crate) fn library(group: [u8; 12], compressed: u32, uncompressed: u32) -> Vec<u8> {
    let mut v = Vec::new();
    p32(&mut v, 1);
    p32(&mut v, 0); // map hashes
    p32(&mut v, 1);
    v.extend([0; 20]); // shader hashes
    p32(&mut v, 1);
    v.extend(group); // group IDs
    p32(&mut v, 0); // map entries
    p32(&mut v, 1);
    v.extend(3u64.to_le_bytes()); // retoc shader_library.rs:54-64,443: pixel frequency
    p32(&mut v, 1);
    for n in [0, 1, uncompressed, compressed] {
        p32(&mut v, n);
    }
    p32(&mut v, 1);
    p32(&mut v, 0);
    v
}
#[test]
fn shader_group_header_and_second_compression_layer() {
    let data = vec![0x17; 320];
    for codec in ["None", "Zlib", "Zstd", "LZ4"] {
        let encoded = compressed(codec, &data);
        let header = library(code_id(8, 1), encoded.len() as u32, data.len() as u32);
        let group = shaders::library(&header).unwrap().groups.pop().unwrap();
        let (expanded, method) = shaders::expand(&encoded, &group).unwrap();
        assert_eq!(expanded, data);
        assert_eq!(method, codec);
        assert!(shaders::expand(&encoded[..encoded.len() - 1], &group).is_err());
    }
    let mut header = library(code_id(8, 1), 4, 32);
    header[0] = 2;
    assert!(shaders::library(&header).is_err());
    let header = library(code_id(8, 1), 4, 32);
    assert!(shaders::library(&header[..header.len() - 1]).is_err());
    assert!(
        shaders::expand(&[0xff; 4], &shaders::library(&header).unwrap().groups[0])
            .unwrap_err()
            .to_string()
            .contains("Oodle is unsupported")
    );
}

#[test]
fn block_fields_and_aes_directory_alignment_fail_closed() {
    let fixture = container(
        8,
        &[(code_id(8, 1), Some("a"), vec![0xab; 100])],
        "None",
        false,
        u64::MAX,
    );
    let block = 144 + 22 + 8 + 4;
    for at in [block + 5, block + 8, block + 11] {
        let mut data = fixture.toc.clone();
        data[at] = 0xff;
        assert!(Toc::parse(&data, None).is_err(), "block field {at}");
    }
    let mut fixture = container(
        8,
        &[(code_id(8, 1), Some("a"), vec![0xab; 100])],
        "None",
        true,
        u64::MAX,
    );
    let size = u32::from_le_bytes(fixture.toc[48..52].try_into().unwrap());
    put32(&mut fixture.toc, 48, size - 1);
    assert!(Toc::parse(&fixture.toc, Some(&key())).is_err());
}
