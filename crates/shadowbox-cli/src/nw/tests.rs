use super::*;

fn put16(bytes: &mut [u8], at: usize, value: u16, be: bool) {
    bytes[at..at + 2].copy_from_slice(&if be {
        value.to_be_bytes()
    } else {
        value.to_le_bytes()
    });
}
fn put32(bytes: &mut [u8], at: usize, value: u32, be: bool) {
    bytes[at..at + 4].copy_from_slice(&if be {
        value.to_be_bytes()
    } else {
        value.to_le_bytes()
    });
}
fn put64(bytes: &mut [u8], at: usize, value: usize) {
    bytes[at..at + 8].copy_from_slice(&(value as u64).to_le_bytes());
}

// All builders use the public field layouts cited at their readers. They never
// load files, reference fixtures, SDK output or captured shader binaries.
pub(crate) fn yaz0_literals(data: &[u8]) -> Vec<u8> {
    let mut bytes = vec![0; 16];
    bytes[..4].copy_from_slice(b"Yaz0");
    put32(&mut bytes, 4, data.len() as u32, true);
    for chunk in data.chunks(8) {
        bytes.push(0xff);
        bytes.extend_from_slice(chunk);
    }
    bytes
}

pub(crate) fn sarc_file(entries: &[(&str, Vec<u8>)], be: bool) -> Vec<u8> {
    let mut entries = entries.to_vec();
    let hash = |name: &str| {
        name.bytes().fold(0u32, |h, b| {
            h.wrapping_mul(101).wrapping_add(b as i8 as u32)
        })
    };
    entries.sort_by_key(|(name, _)| hash(name));
    let names = 0x20 + entries.len() * 16;
    let mut bytes = vec![0; names + 8];
    bytes[..4].copy_from_slice(b"SARC");
    put16(&mut bytes, 4, 0x14, be);
    put16(&mut bytes, 6, 0xfeff, be);
    put16(&mut bytes, 16, 0x100, be);
    bytes[0x14..0x18].copy_from_slice(b"SFAT");
    put16(&mut bytes, 0x18, 12, be);
    put16(&mut bytes, 0x1a, entries.len() as u16, be);
    put32(&mut bytes, 0x1c, 101, be);
    bytes[names..names + 4].copy_from_slice(b"SFNT");
    put16(&mut bytes, names + 4, 8, be);
    for (i, (name, _)) in entries.iter().enumerate() {
        let name_offset = bytes.len() - names - 8;
        put32(&mut bytes, 0x20 + i * 16, hash(name), be);
        if !name.is_empty() {
            put32(
                &mut bytes,
                0x24 + i * 16,
                0x100_0000 | (name_offset as u32 / 4),
                be,
            );
            bytes.extend_from_slice(name.as_bytes());
            bytes.push(0);
            while !bytes.len().is_multiple_of(4) {
                bytes.push(0);
            }
        }
    }
    let data = bytes.len();
    put32(&mut bytes, 12, data as u32, be);
    for (i, (_, content)) in entries.iter().enumerate() {
        let start = bytes.len() - data;
        bytes.extend_from_slice(content);
        let end = bytes.len() - data;
        put32(&mut bytes, 0x28 + i * 16, start as u32, be);
        put32(&mut bytes, 0x2c + i * 16, end as u32, be);
    }
    let size = bytes.len();
    put32(&mut bytes, 8, size as u32, be);
    bytes
}

pub(crate) fn binary_file(magic: &[u8; 4], version: u32, payload: &[u8]) -> Vec<u8> {
    let mut bytes = vec![0; 32];
    bytes[..4].copy_from_slice(magic);
    put32(&mut bytes, 8, version, false);
    put16(&mut bytes, 12, 0xfeff, false);
    bytes.extend_from_slice(payload);
    let size = bytes.len();
    put32(&mut bytes, 28, size as u32, false);
    bytes
}

pub(crate) fn bnsh_file(variations: &[Vec<(Stage, Vec<u8>)>]) -> Vec<u8> {
    // Structs.cs:14-27,244-313; BnshLoader.cs:81-102.
    let mut bytes = vec![0; 0xc0 + 64 * variations.len()];
    bytes[..4].copy_from_slice(b"BNSH");
    put32(&mut bytes, 8, 0x0002_0105, false);
    put16(&mut bytes, 12, 0xfeff, false);
    put16(&mut bytes, 0x16, 0x60, false);
    bytes[0x60..0x64].copy_from_slice(b"grsc");
    put32(&mut bytes, 0x7c, variations.len() as u32, false);
    put64(&mut bytes, 0x80, 0xc0);
    for (i, stages) in variations.iter().enumerate() {
        let program = bytes.len();
        bytes.resize(program + 176, 0);
        put64(&mut bytes, 0xc0 + i * 64 + 16, program);
        for (stage, code) in stages {
            let slot = match stage {
                Stage::Vertex => 0,
                Stage::TessellationControl => 1,
                Stage::TessellationEvaluation => 2,
                Stage::Geometry => 3,
                Stage::Fragment => 4,
                Stage::Compute => 5,
            };
            let descriptor = bytes.len();
            bytes.resize(descriptor + 64, 0);
            put64(&mut bytes, program + 8 + slot * 8, descriptor);
            let control = bytes.len();
            bytes.extend_from_slice(b"opaque-control");
            let code_offset = bytes.len();
            bytes.extend_from_slice(code);
            put64(&mut bytes, descriptor + 8, control);
            put64(&mut bytes, descriptor + 16, code_offset);
            put32(&mut bytes, descriptor + 24, code.len() as u32, false);
            put32(&mut bytes, descriptor + 28, 14, false);
        }
    }
    let size = bytes.len();
    put32(&mut bytes, 28, size as u32, false);
    bytes
}

pub(crate) fn trained_dictionary(seed: usize) -> Vec<u8> {
    // Project-owned text only. Enough diverse samples for dictionary training.
    let samples: Vec<_> = (0..200).map(|i| format!("synthetic material {seed} shader variation {} control code test strings {} {} bindings vertex fragment compute {}", i % 27, i * 137, i % 13, "test payload repeated ".repeat(i % 15 + 1)).into_bytes()).collect();
    zstd::dict::from_samples(&samples, 1024).unwrap()
}
pub(crate) fn compress_with_dict(data: &[u8], dictionary: &[u8]) -> Vec<u8> {
    let mut encoder = zstd::stream::Encoder::with_dictionary(Vec::new(), 3, dictionary).unwrap();
    std::io::Write::write_all(&mut encoder, data).unwrap();
    encoder.finish().unwrap()
}

#[test]
fn yaz0_literals_short_long_overlap_and_errors() {
    let original = b"abcdefghijklmno";
    assert_eq!(yaz0(&yaz0_literals(original)).unwrap(), original);
    let mut short = yaz0_literals(b"ababab");
    short.truncate(16);
    short.extend_from_slice(&[0xc0, b'a', b'b', 0x20, 1]);
    assert_eq!(yaz0(&short).unwrap(), b"ababab");
    let mut long = yaz0_literals(&[b'a'; 19]);
    long.truncate(16);
    long.extend_from_slice(&[0x80, b'a', 0, 0, 0]);
    assert_eq!(yaz0(&long).unwrap(), [b'a'; 19]);
    for n in 0..long.len() {
        assert!(yaz0(&long[..n]).is_err(), "prefix {n}");
    }
    let mut bad = short.clone();
    *bad.last_mut().unwrap() = 0xff;
    assert!(yaz0(&bad).is_err());
    put32(&mut bad, 4, 1, true);
    bad[16] = 0;
    assert!(yaz0(&bad).is_err());
    put32(&mut bad, 4, MAX_BYTES as u32 + 1, true);
    assert!(yaz0(&bad).is_err());
    let mut bad = short.clone();
    put32(&mut bad, 8, 3, true);
    assert!(yaz0(&bad).is_err());
    let mut bad = short;
    put32(&mut bad, 12, 1, true);
    assert!(yaz0(&bad).is_err());
}

#[test]
fn sarc_both_endians_names_ranges_and_rejections() {
    for be in [false, true] {
        let bytes = sarc_file(
            &[("shaders/test.bnsh", b"abc".to_vec()), ("", vec![1, 2])],
            be,
        );
        let archive = sarc(&bytes).unwrap();
        assert_eq!(archive.big_endian, be);
        assert_eq!(archive.entries.len(), 2);
        for entry in archive.entries {
            if entry.name == "shaders/test.bnsh" {
                assert_eq!(&bytes[entry.range], b"abc");
            } else {
                assert!(entry.name.starts_with('#'));
                assert_eq!(&bytes[entry.range], &[1, 2]);
            }
        }
        for n in 0..bytes.len() {
            assert!(sarc(&bytes[..n]).is_err());
        }
        let mut bad = bytes.clone();
        put16(&mut bad, 16, 0x101, be);
        assert!(sarc(&bad).err().unwrap().contains("version"));
        let mut bad = bytes.clone();
        put32(&mut bad, 0x2c, u32::MAX, be);
        assert!(sarc(&bad).is_err());
        let mut bad = bytes.clone();
        put32(&mut bad, 12, 0, be);
        assert!(sarc(&bad).is_err());
        let mut bad = bytes.clone();
        put32(&mut bad, 0x24, 0x1ff_ffff, be);
        assert!(sarc(&bad).is_err());
        let mut bad = bytes;
        bad[6..8].fill(0);
        assert!(sarc(&bad).is_err());
    }
}

#[test]
fn bnsh_variations_all_stages_absolute_ranges_and_unknown_revisions() {
    let stages = [
        Stage::Vertex,
        Stage::TessellationControl,
        Stage::TessellationEvaluation,
        Stage::Geometry,
        Stage::Fragment,
        Stage::Compute,
    ];
    let bytes = bnsh_file(&[
        stages
            .iter()
            .enumerate()
            .map(|(i, &stage)| (stage, vec![i as u8; 17 + i]))
            .collect(),
        vec![(Stage::Compute, vec![0xab; 32])],
    ]);
    let parsed = bnsh(&bytes).unwrap();
    assert_eq!(parsed.variations.len(), 2);
    for (i, code) in parsed.variations[0]
        .as_ref()
        .unwrap()
        .stages
        .iter()
        .enumerate()
    {
        assert_eq!(code.stage, stages[i]);
        assert_eq!(&bytes[code.control.clone()], b"opaque-control");
        assert_eq!(&bytes[code.code.clone()], vec![i as u8; 17 + i]);
    }
    assert_eq!(parsed.variations[1].as_ref().unwrap().stages.len(), 1);
    let mut bad = bytes.clone();
    put32(&mut bad, 8, 0x0002_0106, false);
    assert!(bnsh(&bad).err().unwrap().contains("version"));
    let mut bad = bytes.clone();
    put64(&mut bad, 0x80, usize::MAX);
    assert!(bnsh(&bad).is_err());
    let mut bad = bytes.clone();
    put64(&mut bad, 0xc0 + 16, usize::MAX);
    assert!(bnsh(&bad).unwrap().variations[0].is_err());
    let descriptor = parsed.variations[0].as_ref().unwrap().stages[0].descriptor;
    let mut bad = bytes.clone();
    put64(&mut bad, descriptor + 8, usize::MAX);
    assert!(bnsh(&bad).unwrap().variations[0].is_err());
    let mut bad = bytes.clone();
    put32(&mut bad, descriptor + 24, u32::MAX, false);
    assert!(bnsh(&bad).unwrap().variations[0].is_err());
    for n in 0..bytes.len() {
        assert!(bnsh(&bytes[..n]).is_err());
    }
}

#[test]
fn zstd_multiple_frames_skippable_dictionary_missing_conflicting_and_idless() {
    let plain = zstd::stream::encode_all(&b"synthetic frame"[..], 1).unwrap();
    let mut concatenated = vec![0x50, 0x2a, 0x4d, 0x18, 3, 0, 0, 0, 1, 2, 3];
    concatenated.extend_from_slice(&plain);
    concatenated.extend_from_slice(&plain);
    assert_eq!(
        Dictionaries::default().zstd(&concatenated).unwrap().0,
        b"synthetic framesynthetic frame"
    );
    let dictionary = trained_dictionary(1);
    let data = b"synthetic material 1 shader variation 5 control code test strings 137 2 bindings vertex fragment compute test payload repeated ";
    let compressed = compress_with_dict(data, &dictionary);
    let mut dicts = Dictionaries::default();
    assert!(dicts.zstd(&compressed).err().unwrap().contains("missing"));
    dicts.add("test.zsdic", &dictionary).unwrap();
    assert_eq!(
        dicts.zstd(&compressed).unwrap(),
        (data.to_vec(), vec!["test.zsdic".into()])
    );
    dicts.add("duplicate.zsdic", &dictionary).unwrap();
    assert_eq!(dicts.items.len(), 1);
    let mut conflict = trained_dictionary(2);
    conflict[4..8].copy_from_slice(&dictionary[4..8]);
    dicts.add("conflict.zsdic", &conflict).unwrap();
    assert!(dicts.zstd(&compressed).err().unwrap().contains("ambiguous"));
    let raw = b"synthetic material raw dictionary for a test with a long repeated phrase";
    let compressed = compress_with_dict(raw, raw);
    let mut dicts = Dictionaries::default();
    dicts.add("raw.zsdic", raw).unwrap();
    assert_eq!(dicts.zstd(&compressed).unwrap().0, raw);
    for n in 1..plain.len() {
        assert!(dicts.zstd(&plain[..n]).is_err());
    }
    let mut trailing = plain;
    trailing.push(0);
    assert!(dicts.zstd(&trailing).is_err());
}

#[test]
fn earlier_bnsh_micros_preserve_all_stage_ranges() {
    // Common nn__util.wiki:18-21 (EPD oldid=155): nonbreaking micros.
    let mut bytes = bnsh_file(&[vec![
        (Stage::Vertex, vec![0x31; 43]),
        (Stage::Compute, vec![0x72; 57]),
    ]]);
    for version in 0x0002_0100..=0x0002_0105 {
        put32(&mut bytes, 8, version, false);
        let parsed = bnsh(&bytes).unwrap();
        assert_eq!(parsed.header.version, version);
        let variation = parsed.variations[0].as_ref().unwrap();
        assert_eq!(variation.stages.len(), 2);
        for (stage, expected, size) in [(Stage::Vertex, 0x31, 43), (Stage::Compute, 0x72, 57)] {
            let block = variation.stages.iter().find(|s| s.stage == stage).unwrap();
            assert_eq!(&bytes[block.code.clone()], vec![expected; size]);
            assert_eq!(&bytes[block.control.clone()], b"opaque-control");
        }
    }
    // Unproven major/minor layouts and future micros stay unsupported.
    for version in [
        0x0001_0000,
        0x0002_0000,
        0x0002_0106,
        0x0002_0200,
        0x0003_0105,
    ] {
        put32(&mut bytes, 8, version, false);
        assert!(bnsh(&bytes)
            .err()
            .unwrap()
            .contains("unsupported BNSH version"));
    }
    bytes[..4].copy_from_slice(b"FRES");
    assert!(bnsh(&bytes).err().unwrap().contains("magic"));
}

// Admission policy and its source map: docs/reference-sources.md
// Keep an explicit independent list to catch accidental widening of admission.
pub(crate) const WRAPPER_REVISIONS: &[(&str, &[u8; 4], u32)] = &[
    ("BFSHA", b"FSHA", 0x0004_0000),
    ("BFSHA", b"FSHA", 0x0005_0000),
    ("BFSHA", b"FSHA", 0x0006_0000),
    ("BFSHA", b"FSHA", 0x0007_0000),
    ("BFSHA", b"FSHA", 0x0008_0000),
    ("BFSHA", b"FSHA", 0x0009_0000),
    ("BFRES", b"FRES", 0x0005_0003),
    ("BFRES", b"FRES", 0x0008_0000),
    ("BFRES", b"FRES", 0x0009_0000),
    ("BFRES", b"FRES", 0x000a_0000),
    ("BFRES", b"FRES", 0x000a_0200),
    ("BNTX", b"BNTX", 0x0004_0000),
    ("BNTX", b"BNTX", 0x0004_0100),
];

#[test]
fn binary_wrapper_revisions_and_extent_validation() {
    for &(format, magic, version) in WRAPPER_REVISIONS {
        let bytes = binary_file(magic, version, &bnsh_file(&[]));
        assert_eq!(kind(&bytes), Some(format));
        let h = wrapper(&bytes, format).unwrap();
        assert_eq!(h.version, version);
        assert_eq!(h.size, bytes.len());
        for n in 0..bytes.len() {
            assert!(wrapper(&bytes[..n], format).is_err());
        }
        let mut extended = bytes.clone();
        extended.extend_from_slice(b"not part of the declared container");
        assert_eq!(wrapper(&extended, format).unwrap().size, bytes.len());
        let mut bad = bytes.clone();
        put32(&mut bad, 8, version + 0x0100_0000, false);
        assert!(wrapper(&bad, format).err().unwrap().contains("version"));
        let mut bad = bytes.clone();
        bad[12..14].reverse();
        assert!(wrapper(&bad, format).err().unwrap().contains("byte order"));
        let mut bad = bytes;
        bad[..4].copy_from_slice(b"fake");
        assert!(wrapper(&bad, format).err().unwrap().contains("magic"));
    }
}

pub(crate) fn hash_only_sarc(contents: &[Vec<u8>], be: bool) -> Vec<u8> {
    // Nintendo-File-Formats src/libs/sead/sarc.md:36-39,57,66-69:
    // zero attributes, distinct hashes, empty SFNT, preserved file ranges.
    let entries: Vec<_> = contents.iter().map(|data| ("", data.clone())).collect();
    let mut bytes = sarc_file(&entries, be);
    for i in 0..contents.len() {
        put32(&mut bytes, 0x20 + i * 16, 0x1234_0000 + i as u32, be);
    }
    bytes
}

#[test]
fn sarc_hashes_without_any_names_preserve_identity_and_payloads() {
    let contents = [
        b"first hash only member".to_vec(),
        b"second member".to_vec(),
    ];
    for be in [false, true] {
        let bytes = hash_only_sarc(&contents, be);
        let archive = sarc(&bytes).unwrap();
        assert_eq!(archive.entries.len(), contents.len());
        for (i, entry) in archive.entries.iter().enumerate() {
            assert_eq!(entry.name, format!("#{:08x}", 0x1234_0000 + i));
            assert_eq!(entry.name_hex, None);
            assert_eq!(entry.attributes, 0);
            assert_eq!(&bytes[entry.range.clone()], contents[i]);
        }
        // No decoded name or name offset is invented for hash-only members.
        let names = 0x20 + 16 * contents.len();
        assert_eq!(u32_at(&bytes, 12, be).unwrap() as usize, names + 8);
        // A claimed name must still resolve; empty names do not excuse it.
        let mut bad = bytes;
        put32(&mut bad, 0x24, 0x0100_0000, be);
        assert!(sarc(&bad).err().unwrap().contains("unterminated"));
    }
}
