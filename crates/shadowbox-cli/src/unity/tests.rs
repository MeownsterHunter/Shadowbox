use super::*;

// Synthetic writers below encode only layouts cited in unity.rs. They contain
// project-authored bytes, and never read bundles or serialized assets from disk.
struct Writer {
    bytes: Vec<u8>,
    le: bool,
}
impl Writer {
    fn new(le: bool) -> Self {
        Self {
            bytes: Vec::new(),
            le,
        }
    }
    fn u16(&mut self, n: u16) {
        self.bytes.extend(if self.le {
            n.to_le_bytes()
        } else {
            n.to_be_bytes()
        });
    }
    fn u32(&mut self, n: u32) {
        self.bytes.extend(if self.le {
            n.to_le_bytes()
        } else {
            n.to_be_bytes()
        });
    }
    fn u64(&mut self, n: u64) {
        self.bytes.extend(if self.le {
            n.to_le_bytes()
        } else {
            n.to_be_bytes()
        });
    }
    fn cstr(&mut self, s: &str) {
        self.bytes.extend(s.as_bytes());
        self.bytes.push(0);
    }
    fn align(&mut self, n: usize) {
        self.bytes.resize(self.bytes.len().next_multiple_of(n), 0);
    }
    fn string(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.bytes.extend(s.as_bytes());
        self.align(4);
    }
}
struct Spec {
    ty: &'static str,
    name: &'static str,
    level: u8,
    width: i32,
    meta: u32,
}
fn node(
    nodes: &mut Vec<Spec>,
    ty: &'static str,
    name: &'static str,
    level: u8,
    width: i32,
    align: bool,
) {
    nodes.push(Spec {
        ty,
        name,
        level,
        width,
        meta: if align { 0x4000 } else { 0 },
    });
}
fn vector(nodes: &mut Vec<Spec>, name: &'static str, level: u8, nested: bool, byte: bool) {
    node(nodes, "vector", name, level, -1, false);
    node(nodes, "Array", "Array", level + 1, -1, byte);
    node(nodes, "int", "size", level + 2, 4, false);
    if nested {
        vector(nodes, "data", level + 2, false, false);
    } else {
        node(
            nodes,
            if byte { "UInt8" } else { "UInt32" },
            "data",
            level + 2,
            if byte { 1 } else { 4 },
            false,
        );
    }
}
fn schema(nested: bool) -> Vec<Spec> {
    schema_layout(nested, false)
}
fn schema_layout(nested: bool, modern: bool) -> Vec<Spec> {
    let mut nodes = Vec::new();
    node(&mut nodes, "Shader", "Base", 0, -1, false);
    node(&mut nodes, "string", "m_Name", 1, -1, false);
    // A nested parsed form with a map, aligned strings, and an opaque fixed
    // leaf proves that extraction does not assume the compiled fields' offset.
    node(&mut nodes, "SerializedShader", "m_ParsedForm", 1, -1, false);
    node(&mut nodes, "map", "tags", 2, -1, false);
    node(&mut nodes, "Array", "Array", 3, -1, true);
    node(&mut nodes, "int", "size", 4, 4, false);
    node(&mut nodes, "pair", "data", 4, -1, false);
    node(&mut nodes, "string", "first", 5, -1, false);
    node(&mut nodes, "string", "second", 5, -1, false);
    node(&mut nodes, "Hash128", "hash", 2, 16, false);
    if modern {
        // UnityPy/classes/generated.py:10481-10495: optional player tables
        // and parameter blob indices. This synthetic tree declares order and
        // widths explicitly; it makes no claim about a stripped byte layout.
        node(&mut nodes, "SerializedProgram", "progVertex", 2, -1, false);
        node(&mut nodes, "vector", "m_PlayerSubPrograms", 3, -1, false);
        node(&mut nodes, "Array", "Array", 4, -1, false);
        node(&mut nodes, "int", "size", 5, 4, false);
        node(&mut nodes, "vector", "data", 5, -1, false);
        node(&mut nodes, "Array", "Array", 6, -1, false);
        node(&mut nodes, "int", "size", 7, 4, false);
        node(
            &mut nodes,
            "SerializedPlayerSubProgram",
            "data",
            7,
            -1,
            false,
        );
        node(&mut nodes, "UInt32", "m_BlobIndex", 8, 4, false);
        node(&mut nodes, "int", "m_GpuProgramType", 8, 4, false);
        short_vector(&mut nodes, "m_KeywordIndices", 8);
        node(&mut nodes, "SInt64", "m_ShaderRequirements", 8, 8, false);
        vector(&mut nodes, "m_ParameterBlobIndices", 3, true, false);
        short_vector(&mut nodes, "m_SerializedKeywordStateMask", 3);
    }
    vector(&mut nodes, "platforms", 1, false, false);
    for name in ["offsets", "compressedLengths", "decompressedLengths"] {
        vector(&mut nodes, name, 1, nested, false);
    }
    vector(&mut nodes, "compressedBlob", 1, false, true);
    node(&mut nodes, "bool", "m_ShaderIsBaked", 1, 1, true);
    nodes
}
fn short_vector(nodes: &mut Vec<Spec>, name: &'static str, level: u8) {
    // AssetStudio/TypeTreeHelper.cs:193-240,271-285,303-304: counted
    // primitive array with alignment declared by the supplied tree.
    node(nodes, "vector", name, level, -1, false);
    node(nodes, "Array", "Array", level + 1, -1, true);
    node(nodes, "int", "size", level + 2, 4, false);
    node(nodes, "UInt16", "data", level + 2, 2, false);
}
fn write_type(w: &mut Writer, version: u32, class: u32, nodes: &[Spec]) {
    w.u32(class);
    w.bytes.push(0);
    w.u16(u16::MAX);
    if class == 114 {
        w.bytes.extend([0; 16]);
    }
    w.bytes.extend([0; 16]);
    let mut strings = Writer::new(false);
    let mut offsets = Vec::new();
    for s in nodes {
        let ty = strings.bytes.len() as u32;
        strings.cstr(s.ty);
        let name = strings.bytes.len() as u32;
        strings.cstr(s.name);
        // Use both local strings and the shared table in every fixture.
        offsets.push((
            if s.ty == "UInt32" {
                0x8000_0000 | 914
            } else {
                ty
            },
            if s.name == "m_Name" {
                0x8000_0000 | 427
            } else {
                name
            },
        ));
    }
    let blob_start = w.bytes.len();
    w.u32(nodes.len() as u32);
    w.u32(strings.bytes.len() as u32);
    for (index, (s, (ty, name))) in nodes.iter().zip(offsets).enumerate() {
        w.u16(1);
        w.bytes.push(s.level);
        w.bytes.push(u8::from(s.ty == "Array"));
        w.u32(ty);
        w.u32(name);
        w.u32(s.width as u32);
        w.u32(index as u32);
        w.u32(s.meta);
        if version >= 19 {
            w.u64(0);
        }
    }
    w.bytes.extend(strings.bytes);
    if version >= 23 {
        // UnityPy/helpers/TypeTreeNode.py:349-356,376-393: inline mhtt tree.
        let blob = w.bytes.split_off(blob_start);
        let mut wrapper = Writer::new(w.le);
        wrapper.bytes.extend(b"mhtt");
        wrapper.u32(22);
        wrapper.bytes.extend(blob);
        wrapper.u32(0); // reference hashes
        w.bytes.extend([0x5a; 16]);
        w.u32(wrapper.bytes.len() as u32);
        w.bytes.extend(wrapper.bytes);
    }
    if version >= 21 {
        w.u32(2);
        w.u32(0);
        w.u32(1);
    }
}
fn shader_data(le: bool, nested: bool, modern: bool, payload: &[u8], index: usize) -> Vec<u8> {
    let mut w = Writer::new(le);
    w.string(&format!("../synthetic/shader-{index}"));
    w.u32(1);
    w.string("render");
    w.string("synthetic");
    w.bytes.extend([0xa5; 16]);
    if modern {
        // Per-platform tables in the supplied synthetic schema above, with
        // different row lengths, nonzero indices and odd u16 counts.
        w.u32(2);
        for indices in [vec![0, 1], vec![2]] {
            w.u32(indices.len() as u32);
            for index in indices {
                w.u32(index);
                w.u32(26);
                w.u32(3);
                for keyword in [7, 11, 13] {
                    w.u16(keyword);
                }
                w.align(4);
                w.u64(0x1234_5678_0000_0001);
            }
        }
        w.u32(2);
        for indices in [vec![3, 4], vec![5]] {
            w.u32(indices.len() as u32);
            for index in indices {
                w.u32(index);
            }
        }
        w.u32(3);
        for mask in [1, 2, 4] {
            w.u16(mask);
        }
        w.align(4);
    }
    let inputs = if nested {
        vec![
            vec![payload, b"second segment"],
            vec![b"other platform".as_slice()],
        ]
    } else {
        vec![vec![payload], vec![b"other platform".as_slice()]]
    };
    let mut offsets = Vec::new();
    let mut compressed = Vec::new();
    let mut decompressed = Vec::new();
    let mut blob = vec![0x42; 3]; // Nonzero first offset, with alignment padding afterwards.
    for platform in &inputs {
        let mut o = Vec::new();
        let mut c = Vec::new();
        let mut d = Vec::new();
        for bytes in platform {
            let packed = lz4_flex::block::compress(bytes);
            o.push(blob.len() as u32);
            c.push(packed.len() as u32);
            d.push(bytes.len() as u32);
            blob.extend(packed);
        }
        offsets.push(o);
        compressed.push(c);
        decompressed.push(d);
    }
    w.u32(2);
    w.u32(7);
    w.u32(26);
    for array in [offsets, compressed, decompressed] {
        w.u32(array.len() as u32);
        for row in array {
            if nested {
                w.u32(row.len() as u32);
            }
            for n in row {
                w.u32(n);
            }
        }
    }
    w.u32(blob.len() as u32);
    w.bytes.extend(blob);
    w.align(4);
    w.bytes.push(1);
    w.align(4);
    w.bytes
}

pub(crate) fn serialized(version: u32, le: bool, nested: bool, payloads: &[Vec<u8>]) -> Vec<u8> {
    serialized_layout(version, le, nested, None, payloads)
}
pub(crate) fn modern_serialized(
    version: u32,
    le: bool,
    unity: &str,
    payloads: &[Vec<u8>],
) -> Vec<u8> {
    serialized_layout(version, le, true, Some(unity), payloads)
}
fn serialized_layout(
    version: u32,
    le: bool,
    nested: bool,
    unity: Option<&str>,
    payloads: &[Vec<u8>],
) -> Vec<u8> {
    let header_size = if version >= 22 { 48 } else { 20 };
    let mut m = Writer::new(le);
    m.bytes.resize(header_size, 0);
    m.cstr(unity.unwrap_or("2020.1.0f1"));
    m.u32(38);
    m.bytes.push(1);
    m.u32(2);
    write_type(&mut m, version, 48, &schema_layout(nested, unity.is_some()));
    // The optional script GUID is exercised, and non-Shader objects containing
    // payload bytes must not be treated as compiled Shader blobs.
    write_type(&mut m, version, 114, &[]);
    m.u32(payloads.len() as u32 + 1);
    let mut data = Writer::new(le);
    for (index, bytes) in payloads.iter().enumerate() {
        data.align(8);
        let start = data.bytes.len();
        let object = shader_data(le, nested, unity.is_some(), bytes, index);
        m.align(4);
        m.u64(index as u64 + 100);
        if version >= 22 {
            m.u64(start as u64);
        } else {
            m.u32(start as u32);
        }
        m.u32(object.len() as u32);
        m.u32(0);
        data.bytes.extend(object);
    }
    m.align(4);
    m.u64(999);
    if version >= 22 {
        m.u64(data.bytes.len() as u64);
    } else {
        m.u32(data.bytes.len() as u32);
    }
    let other = payloads.first().map(Vec::as_slice).unwrap_or(b"not shader");
    m.u32(other.len() as u32);
    m.u32(1);
    data.bytes.extend(other);
    m.u32(0);
    m.u32(0);
    if version >= 20 {
        m.u32(0);
    }
    if version >= 26 {
        m.u32(0); // UnityPy/files/SerializedFile.py:331-338: shared trees.
    }
    m.cstr("");
    let metadata_size = m.bytes.len() - header_size;
    m.align(16);
    let data_offset = m.bytes.len();
    m.bytes.extend(data.bytes);
    let mut header = Writer::new(false);
    header.u32(if version >= 22 {
        0
    } else {
        metadata_size as u32
    });
    header.u32(if version >= 22 {
        0
    } else {
        m.bytes.len() as u32
    });
    header.u32(version);
    header.u32(if version >= 22 { 0 } else { data_offset as u32 });
    header.bytes.push(u8::from(!le));
    header.bytes.extend([0; 3]);
    if version >= 22 {
        header.u32(metadata_size as u32);
        header.u64(m.bytes.len() as u64);
        header.u64(data_offset as u64);
        header.u64(0);
    }
    m.bytes[..header_size].copy_from_slice(&header.bytes);
    m.bytes
}
fn pack(bytes: &[u8], codec: u32) -> Vec<u8> {
    match codec {
        0 => bytes.to_vec(),
        2 | 3 => lz4_flex::block::compress(bytes),
        1 => {
            let mut result = Vec::new();
            lzma_rs::lzma_compress(&mut Cursor::new(bytes), &mut result).unwrap();
            result.drain(5..13);
            result
        }
        _ => panic!("test codec"),
    }
}
pub(crate) fn bundle(
    version: u32,
    at_end: bool,
    padding: bool,
    info_codec: u32,
    block_codec: u32,
    files: &[(&str, Vec<u8>)],
) -> Vec<u8> {
    bundle_revision(
        version,
        "2022.3.0f1",
        at_end,
        padding,
        info_codec,
        block_codec,
        files,
    )
}
fn bundle_revision(
    version: u32,
    revision: &str,
    at_end: bool,
    padding: bool,
    info_codec: u32,
    block_codec: u32,
    files: &[(&str, Vec<u8>)],
) -> Vec<u8> {
    let mut data = Vec::new();
    let mut nodes = Writer::new(false);
    nodes.u32(files.len() as u32);
    for (name, bytes) in files {
        nodes.u64(data.len() as u64);
        nodes.u64(bytes.len() as u64);
        nodes.u32(4);
        nodes.cstr(name);
        data.extend(bytes);
    }
    // Three blocks, with file ranges crossing boundaries.
    let mut info = Writer::new(false);
    info.bytes.extend([0; 16]);
    info.u32(3);
    let split = data.len() / 3;
    let mut packed = Vec::new();
    for chunk in [&data[..split], &data[split..split * 2], &data[split * 2..]] {
        let b = pack(chunk, block_codec);
        info.u32(chunk.len() as u32);
        info.u32(b.len() as u32);
        info.u16(block_codec as u16 | 0x40);
        packed.extend(b);
    }
    info.bytes.extend(nodes.bytes);
    let encoded_info = pack(&info.bytes, info_codec);
    let mut h = Writer::new(false);
    h.cstr("UnityFS");
    h.u32(version);
    h.cstr("5.x.x");
    h.cstr(revision);
    let size_at = h.bytes.len();
    h.u64(0);
    h.u32(encoded_info.len() as u32);
    h.u32(info.bytes.len() as u32);
    h.u32(0x40 | info_codec | if at_end { 0x80 } else { 0 } | if padding { 0x200 } else { 0 });
    if version >= 7 || revision == "2019.4.15f1" {
        h.align(16);
    }
    if !at_end {
        h.bytes.extend(&encoded_info);
    }
    if padding {
        h.align(16);
    }
    h.bytes.extend(packed);
    if at_end {
        h.bytes.extend(encoded_info);
    }
    let len = h.bytes.len() as u64;
    h.bytes[size_at..size_at + 8].copy_from_slice(&len.to_be_bytes());
    h.bytes
}

#[test]
fn all_serialized_versions_endianness_and_array_shapes() {
    for version in 17..=26 {
        for le in [false, true] {
            for nested in [false, true] {
                let bytes = serialized(
                    version,
                    le,
                    nested,
                    &[b"first blob".to_vec(), b"second blob".to_vec()],
                );
                let file = SerializedFile::read(&bytes).unwrap();
                let objects = file.shaders().collect::<Vec<_>>();
                assert_eq!(objects.len(), 2);
                for (index, object) in objects.iter().enumerate() {
                    assert_eq!(object.path_id, index as i64 + 100);
                    let shader = object.extract().unwrap();
                    assert_eq!(shader.name, format!("../synthetic/shader-{index}"));
                    assert_eq!(shader.blobs.len(), if nested { 3 } else { 2 });
                    assert_eq!(
                        shader.blobs[0].bytes,
                        if index == 0 {
                            b"first blob".as_slice()
                        } else {
                            b"second blob".as_slice()
                        }
                    );
                    assert_eq!(shader.blobs[0].platform, 7);
                    assert_eq!(shader.blobs[0].offset, 3);
                    assert_eq!(shader.blobs.last().unwrap().platform, 26);
                }
            }
        }
    }
}
#[test]
fn unityfs_codecs_end_info_padding_and_cross_block_files() {
    let files = vec![
        ("../opaque", vec![1; 300]),
        (
            "CAB-synthetic",
            serialized(22, true, true, &[b"compiled".to_vec()]),
        ),
    ];
    for version in 6..=8 {
        for at_end in [false, true] {
            for padding in [false, true] {
                for info_codec in 0..=3 {
                    for block_codec in 0..=3 {
                        let bytes =
                            bundle(version, at_end, padding, info_codec, block_codec, &files);
                        let read = Bundle::read(&bytes).unwrap();
                        assert_eq!(read.files.len(), 2);
                        for (file, (name, original)) in read.files.iter().zip(&files) {
                            assert_eq!(&file.name, name);
                            assert_eq!(&read.data[file.range.clone()], original);
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn reject_unknown_versions_missing_trees_and_bad_ranges() {
    let original = serialized(17, true, false, &[b"blob".to_vec()]);
    for version in [16u32, 27, u32::MAX] {
        let mut bytes = original.clone();
        bytes[8..12].copy_from_slice(&version.to_be_bytes());
        assert!(SerializedFile::read(&bytes)
            .err()
            .unwrap()
            .to_string()
            .contains("unsupported Unity serialized version"));
    }
    let mut missing = original.clone();
    missing[20 + "2020.1.0f1".len() + 1 + 4] = 0;
    // Clearing only the flag leaves the old tree bytes in metadata. This
    // remains malformed; a real stripped file omits them entirely.
    assert!(SerializedFile::read(&missing).is_err());
    let mut offset = original.clone();
    offset[12..16].copy_from_slice(&u32::MAX.to_be_bytes());
    assert!(SerializedFile::read(&offset).is_err());
    for cut in 0..original.len() {
        assert!(SerializedFile::read(&original[..cut]).is_err());
    }
    let packed = bundle(7, true, true, 2, 3, &[("level0", original)]);
    for cut in 0..packed.len() {
        assert!(Bundle::read(&packed[..cut]).is_err());
    }
    let mut bad = packed;
    bad[8..12].copy_from_slice(&99u32.to_be_bytes());
    assert!(Bundle::read(&bad).is_err());
}
#[test]
fn reject_shader_schema_lengths_offsets_and_compression() {
    let bytes = serialized(22, true, true, &[b"payload".to_vec()]);
    let file = SerializedFile::read(&bytes).unwrap();
    let object = file.shaders().next().unwrap();
    // Locate fields by the type tree even in the corruption tests.
    let mut r = Reader::new(object.data.unwrap(), true);
    r.origin = object.origin;
    let mut visits = MAX_VISITS;
    let mut starts = BTreeMap::new();
    let mut child = 1;
    while child < object.nodes.len() {
        starts.insert(object.nodes[child].name.as_str(), r.at);
        value(&mut r, object.nodes, child, false, &mut visits).unwrap();
        child = object.nodes[child].end;
    }
    let mut corruptions = Vec::new();
    for (field, delta, word) in [
        ("offsets", 8, u32::MAX),
        ("compressedLengths", 8, u32::MAX),
        ("decompressedLengths", 8, 1),
        ("platforms", 0, 1),
        ("offsets", 4, 1),
    ] {
        let mut data = object.data.unwrap().to_vec();
        let at = starts[field] + delta;
        data[at..at + 4].copy_from_slice(&word.to_le_bytes());
        corruptions.push(data);
    }
    let mut data = object.data.unwrap().to_vec();
    data[starts["compressedBlob"] + 4 + 3] = 0xff;
    corruptions.push(data);
    for data in &corruptions {
        let corrupt = ShaderObject {
            data: Some(data),
            length: data.len(),
            nodes: object.nodes,
            path_id: object.path_id,
            le: true,
            origin: object.origin,
            unity_version: object.unity_version,
            target: object.target,
            has_tree: object.has_tree,
        };
        assert!(corrupt.extract().is_err());
    }
    assert!(decompress(&[0; 5], 1, 4).is_err());
    assert!(decompress(&[0; 5], 1, 2).is_err());
    assert!(decompress(b"short", 99, 0).is_err());
    assert!(decompress(&[0; 5], MAX_BYTES + 1, 1).is_err());
}

#[test]
fn type_tree_schema_and_metadata_budgets_fail_explicitly() {
    let bytes = serialized(22, true, true, &[b"payload".to_vec()]);
    let mut file = SerializedFile::read(&bytes).unwrap();
    let element = file.types[0]
        .nodes
        .iter()
        .position(|n| n.ty == "UInt32")
        .unwrap();
    file.types[0].nodes[element].ty = "float".into();
    assert!(file
        .shaders()
        .next()
        .unwrap()
        .extract()
        .err()
        .unwrap()
        .to_string()
        .contains("not UInt32"));
    let mut bad_bytes = SerializedFile::read(&bytes).unwrap();
    let element = bad_bytes.types[0]
        .nodes
        .iter()
        .position(|n| n.ty == "UInt8")
        .unwrap();
    bad_bytes.types[0].nodes[element].byte_size = 4;
    assert!(bad_bytes
        .shaders()
        .next()
        .unwrap()
        .extract()
        .err()
        .unwrap()
        .to_string()
        .contains("byte array size"));
    let empty = ShaderObject {
        data: Some(&[]),
        length: 0,
        nodes: &[],
        path_id: 1,
        le: true,
        origin: 0,
        unity_version: "2020.1.0f1",
        target: 38,
        has_tree: true,
    };
    assert!(empty
        .extract()
        .err()
        .unwrap()
        .to_string()
        .contains("missing type tree"));
    let mut w = Writer::new(true);
    write_type(&mut w, 22, 48, &schema(true));
    let mut r = Reader::new(&w.bytes, true);
    assert!(read_type(
        &mut r,
        22,
        &mut TreeBudget {
            nodes: 0,
            strings: MAX_BYTES
        },
        true
    )
    .is_err());
    let mut r = Reader::new(&w.bytes, true);
    assert!(read_type(
        &mut r,
        22,
        &mut TreeBudget {
            nodes: MAX_ITEMS,
            strings: 0
        },
        true
    )
    .is_err());
    // First node's type offset is at byte 35: class/stripped/script/hash,
    // counts, then node version/depth/flags. Reject unknown common offsets.
    w.bytes[35..39].copy_from_slice(&u32::MAX.to_le_bytes());
    let mut r = Reader::new(&w.bytes, true);
    assert!(read_type(
        &mut r,
        22,
        &mut TreeBudget {
            nodes: MAX_ITEMS,
            strings: MAX_BYTES
        },
        true
    )
    .is_err());
    let mut r = Reader::new(&[0; 20], false);
    assert!(read_type(
        &mut r,
        17,
        &mut TreeBudget {
            nodes: MAX_ITEMS,
            strings: MAX_BYTES
        },
        true
    )
    .is_err());
}

// Fixed-layout fixtures follow the field order in AssetStudio/Classes/Shader.cs
// (pinned in 0017-unity.md), independently of the parser. Every variable list
// contains at least one item, and short/byte lists have odd lengths so padding
// is exercised. No source assets or instruction fixtures are read from disk.
fn fixed_shader(le: bool, version: (u32, u32, u32), payload: &[u8]) -> Vec<u8> {
    let mut w = Writer::new(le);
    let (year, minor, _) = version;
    let common = (year == 2020 && version >= (2020, 3, 2))
        || (year == 2021 && version >= (2021, 1, 1))
        || year > 2021;
    let partial = (year == 2020 && version >= (2020, 3, 2))
        || (year == 2021 && version >= (2021, 1, 4))
        || year > 2021;
    fn tags(w: &mut Writer) {
        w.u32(1);
        w.string("key");
        w.string("value");
    }
    fn float(w: &mut Writer) {
        w.u32(0x3f80_0000);
        w.string("x");
    }
    fn shorts(w: &mut Writer) {
        w.u32(3);
        for v in [0, 1, 2] {
            w.u16(v);
        }
        w.align(4);
    }
    fn vector(w: &mut Writer) {
        for v in [1, 2, 3] {
            w.u32(v);
        }
        w.bytes.extend([0, 4]);
        w.align(4);
    }
    fn pointer(w: &mut Writer) {
        w.u32(1);
        w.u64(123);
    }
    fn params(w: &mut Writer, year: u32, minor: u32, partial: bool) {
        // Vector and matrix lists (Shader.cs:617-629,320-325,339-344).
        for _ in 0..2 {
            w.u32(1);
            vector(w);
        }
        // Texture list (:631-636,359-367).
        w.u32(1);
        for v in [1, 2, 3] {
            w.u32(v);
        }
        if (year, minor) >= (2017, 3) {
            w.bytes.push(1);
        }
        w.bytes.push(2);
        w.align(4);
        // Buffer binding (:638-643,381-385).
        w.u32(1);
        w.u32(1);
        w.u32(2);
        if year >= 2020 {
            w.u32(3);
        }
        // Constant buffer (:645-650,403-437), including nested struct params.
        w.u32(1);
        w.u32(7);
        for _ in 0..2 {
            w.u32(1);
            vector(w);
        }
        if (year, minor) >= (2017, 3) {
            w.u32(1);
            for v in [1, 2, 3, 4] {
                w.u32(v);
            }
            for _ in 0..2 {
                w.u32(1);
                vector(w);
            }
        }
        w.u32(128);
        if partial {
            w.bytes.push(1);
            w.align(4);
        }
        // Constant buffer binding, UAV, sampler (:652-674,449-451,53-54).
        w.u32(1);
        w.u32(1);
        w.u32(2);
        if year >= 2020 {
            w.u32(3);
        }
        w.u32(1);
        for v in [1, 2, 3] {
            w.u32(v);
        }
        w.u32(1);
        w.u32(4);
        w.u32(5);
    }
    w.string("synthetic stripped shader"); // NamedObject.cs:14.
                                           // Property sheet (Shader.cs:103-109,76-77,119-123).
    w.u32(1);
    w.string("prop");
    w.string("description");
    w.u32(2);
    w.string("a");
    w.string("bbb");
    w.u32(4);
    w.u32(1);
    for _ in 0..4 {
        w.u32(0x3f80_0000);
    }
    w.string("white");
    w.u32(2);
    w.u32(1); // subshader count (:894).
    w.u32(1); // pass count (:839).
    if (year, minor) >= (2020, 2) {
        // Editor hashes, platform bytes, keyword masks (:759-775).
        w.u32(1);
        w.bytes.extend([0x12; 16]);
        w.align(4);
        w.u32(3);
        w.bytes.extend([1, 2, 3]);
        w.align(4);
        if (year, minor) < (2021, 2) {
            shorts(&mut w);
            shorts(&mut w);
        }
    }
    w.u32(1);
    w.string("index");
    w.u32(0); // name indices (:779-783).
    w.u32(0); // pass type (:786).
              // State (:238-275). Use independently counted named groups.
    w.string("state");
    for _ in 0..8 {
        for _ in 0..7 {
            float(&mut w);
        }
    }
    w.bytes.push(1);
    w.align(4);
    if (year, minor) >= (2017, 2) {
        float(&mut w);
    }
    for _ in 0..3 {
        float(&mut w);
    } // zTest/zWrite/culling.
    if year >= 2020 {
        float(&mut w);
    }
    for _ in 0..3 {
        float(&mut w);
    } // offsets/alphaToMask.
    for _ in 0..3 {
        for _ in 0..4 {
            float(&mut w);
        }
    } // stencil ops.
    for _ in 0..6 {
        float(&mut w);
    } // stencil masks/ref and fog scalars.
    for _ in 0..4 {
        float(&mut w);
    } // fog color.
    w.string("fog");
    w.u32(1);
    w.u32(2);
    tags(&mut w);
    w.u32(100);
    w.bytes.push(1);
    w.align(4);
    w.u32(31); // program mask (:788).
    for _ in 0..if (year, minor) >= (2019, 3) { 6 } else { 5 } {
        w.u32(1); // subprogram count (:701).
                  // Blob index, channels, source map (:584-585,298-306).
        w.u32(0);
        w.u32(3);
        w.bytes.extend([0, 1, 1, 2, 2, 3]);
        w.align(4);
        w.u32(7);
        shorts(&mut w);
        if (2019..=2021).contains(&year) && (year, minor) < (2021, 2) {
            shorts(&mut w);
        }
        w.bytes.extend([1, 26]);
        w.align(4); // tier/type (:603-605).
        params(&mut w, year, minor, partial);
        if year >= 2021 {
            w.u64(0x1234_5678_0000_0001); // Shader.cs:679-681.
        } else if (year, minor) >= (2017, 2) {
            w.u32(1);
        }
        // RazTools/Studio AssetStudio/Classes/Shader.cs:789-806: distinct
        // branch gates, nested player records before common parameters.
        // zhangjiequan/AssetStudio AssetStudio/Classes/Shader.cs:731-750:
        // the later supported families retain both nested player tables.
        if (year == 2021 && version >= (2021, 3, 10))
            || (year == 2022 && version >= (2022, 1, 13))
            || year >= 2023
        {
            w.u32(2);
            for indices in [vec![3, 5], vec![7]] {
                w.u32(indices.len() as u32);
                for index in indices {
                    w.u32(index);
                    shorts(&mut w);
                    w.u64(0x1234_5678_0000_0001);
                    w.bytes.push(26);
                    w.align(4);
                }
            }
            w.u32(2);
            w.u32(2);
            w.u32(11);
            w.u32(13);
            w.u32(1);
            w.u32(17);
        }
        if common {
            params(&mut w, year, minor, partial);
        }
        if year >= 2022 {
            shorts(&mut w); // Shader.cs:717-721: program mask.
        }
    }
    // Pass flags, strings/tags; subshader tags/LOD (:798-807,846-847).
    w.bytes.push(1);
    if year >= 2018 {
        w.bytes.push(1);
    }
    w.align(4);
    w.string("use");
    w.string("pass");
    w.string("tex");
    tags(&mut w);
    if year == 2021 && minor >= 2 {
        shorts(&mut w); // Shader.cs:808-811: pass mask.
    }
    tags(&mut w);
    w.u32(100);
    // Shader.cs:901-905: keyword names/flags are nonempty and need padding.
    if (year, minor) >= (2021, 2) {
        w.u32(3);
        for keyword in ["FOO", "BAR", "BAZ"] {
            w.string(keyword);
        }
        w.u32(3);
        w.bytes.extend([1, 2, 4]);
        w.align(4);
    }
    // Parsed name/custom editor/fallback, dependencies, flag (:908-930).
    w.string("parsed");
    w.string("editor");
    w.string("fallback");
    w.u32(1);
    w.string("from");
    w.string("to");
    if year >= 2021 {
        // Shader.cs:919-926,870-871: editor/pipeline string pairs.
        w.u32(1);
        w.string("pipeline editor");
        w.string("pipeline type");
    }
    w.bytes.push(1);
    w.align(4);
    // Compiled arrays (:983-997): two platforms, multiple segments after 2019.3.
    let inputs = if (year, minor) >= (2019, 3) {
        vec![vec![payload, b"second"], vec![b"third".as_slice()]]
    } else {
        vec![vec![payload], vec![b"third".as_slice()]]
    };
    let mut blob = vec![0x5a; 3];
    let mut offsets = Vec::new();
    let mut compressed = Vec::new();
    let mut decoded = Vec::new();
    for platform in inputs {
        let mut o = Vec::new();
        let mut c = Vec::new();
        let mut d = Vec::new();
        for bytes in platform {
            let packed = lz4_flex::block::compress(bytes);
            o.push(blob.len() as u32);
            c.push(packed.len() as u32);
            d.push(bytes.len() as u32);
            blob.extend(packed);
        }
        offsets.push(o);
        compressed.push(c);
        decoded.push(d);
    }
    w.u32(2);
    w.u32(19);
    w.u32(18);
    for array in [offsets, compressed, decoded] {
        w.u32(2);
        for row in array {
            if (year, minor) >= (2019, 3) {
                w.u32(row.len() as u32);
            }
            for v in row {
                w.u32(v);
            }
        }
    }
    w.u32(blob.len() as u32);
    w.bytes.extend(blob);
    w.align(4);
    // RazTools/Studio AssetStudio/Classes/Shader.cs:1120-1125:
    // stageCounts follows compressedBlob, at later patch gates.
    // zhangjiequan/AssetStudio AssetStudio/Classes/Shader.cs:1047-1055.
    if (year == 2021 && version >= (2021, 3, 12))
        || (year == 2022 && version >= (2022, 1, 21))
        || year >= 2023
    {
        w.u32(2);
        w.u32(6);
        w.u32(2);
    }
    w.u32(1);
    pointer(&mut w); // dependencies (:999-1003).
    if year >= 2018 {
        w.u32(1);
        w.string("fixedtex");
        pointer(&mut w);
    }
    w.bytes.push(1);
    w.align(4); // baked (:1015-1016).
    w.bytes
}

fn stripped_objects(unity_version: &str, le: bool, objects: &[Vec<u8>]) -> Vec<u8> {
    // SerializedFile.cs:43-52,73-144,236-286: no tree counts or dependencies
    // occur in a serialized type when m_EnableTypeTree is false, including v21.
    let version = if unity_version.starts_with("2017") {
        17
    } else if unity_version[..4].parse::<u32>().unwrap_or(0) >= 2021 {
        22
    } else {
        21
    };
    let header_size = if version >= 22 { 48 } else { 20 };
    let mut m = Writer::new(le);
    m.bytes.resize(header_size, 0);
    m.cstr(unity_version);
    m.u32(38);
    m.bytes.push(0);
    m.u32(2);
    for class in [48, 49] {
        m.u32(class);
        m.bytes.push(0);
        m.u16(u16::MAX);
        m.bytes.extend([0; 16]);
    }
    m.u32(objects.len() as u32 + 1);
    let mut data = Writer::new(le);
    for (i, object) in objects.iter().enumerate() {
        data.align(8);
        m.align(4);
        m.u64(100 + i as u64);
        if version >= 22 {
            m.u64(data.bytes.len() as u64);
        } else {
            m.u32(data.bytes.len() as u32);
        }
        m.u32(object.len() as u32);
        m.u32(0);
        data.bytes.extend(object);
    }
    // Non-Shader object contains blob-like data but must never be scanned.
    m.align(4);
    m.u64(999);
    if version >= 22 {
        m.u64(data.bytes.len() as u64);
    } else {
        m.u32(data.bytes.len() as u32);
    }
    m.u32(3);
    m.u32(1);
    data.bytes.extend([0xaa, 0xbb, 0xcc]);
    m.u32(0);
    m.u32(0);
    if version >= 20 {
        m.u32(0);
    }
    if version >= 26 {
        m.u32(0); // UnityPy/files/SerializedFile.py:331-338: shared trees.
    }
    m.cstr("");
    let metadata = m.bytes.len() - header_size;
    m.align(16);
    let offset = m.bytes.len();
    m.bytes.extend(data.bytes);
    let mut h = Writer::new(false);
    h.u32(if version >= 22 { 0 } else { metadata as u32 });
    h.u32(if version >= 22 {
        0
    } else {
        m.bytes.len() as u32
    });
    h.u32(version);
    h.u32(if version >= 22 { 0 } else { offset as u32 });
    h.bytes.push(u8::from(!le));
    h.bytes.extend([0; 3]);
    if version >= 22 {
        h.u32(metadata as u32);
        h.u64(m.bytes.len() as u64);
        h.u64(offset as u64);
        h.u64(0);
    }
    m.bytes[..header_size].copy_from_slice(&h.bytes);
    m.bytes
}

pub(crate) fn stripped(unity_version: &str, le: bool, payloads: &[Vec<u8>]) -> Vec<u8> {
    let parts: Vec<_> = unity_version.split('.').collect();
    let patch: String = parts[2].chars().take_while(char::is_ascii_digit).collect();
    let v = (
        parts[0].parse().unwrap(),
        parts[1].parse().unwrap(),
        patch.parse().unwrap(),
    );
    let objects: Vec<_> = payloads.iter().map(|p| fixed_shader(le, v, p)).collect();
    stripped_objects(unity_version, le, &objects)
}

#[test]
fn stripped_shader_layouts_and_version_boundaries() {
    for version in [
        "2017.1.0f1",
        "2017.2.0f1",
        "2017.3.0f1",
        "2017.4.9f1",
        "2018.4.0f1",
        "2019.2.0f1",
        "2019.4.1f1",
        "2020.1.0f1",
        "2020.2.0f1",
        "2020.3.1f1",
        "2020.3.2f1",
        "2021.1.0f1",
        "2021.1.1f1",
        "2021.1.3f1",
        "2021.1.4f1",
        "2021.2.0f1",
        "2021.3.0f1",
        "2021.3.9f1",
        "2021.3.10f1",
        "2021.3.11f1",
        "2021.3.12f1",
        "2022.1.0f1",
        "2022.1.1f1",
        "2022.1.12f1",
        "2022.1.13f1",
        "2022.1.20f1",
        "2022.1.21f1",
        "2022.1.24f1",
        "2022.2.0f1",
        "2022.3.0f1",
        "2023.1.0f1",
        "2023.2.0f1",
    ] {
        for le in [true, false] {
            let bytes = stripped(version, le, &[b"synthetic bytes".to_vec()]);
            let file = SerializedFile::read(&bytes).unwrap();
            assert_eq!(file.unity_version, version);
            let shaders: Vec<_> = file.shaders().collect();
            assert_eq!(shaders.len(), 1);
            let shader = shaders[0].extract().unwrap();
            assert_eq!(shader.name, "synthetic stripped shader");
            assert_eq!(shader.blobs[0].bytes, b"synthetic bytes");
            assert_eq!((shader.blobs[0].platform, shader.blobs[0].offset), (19, 3));
            let nested = version >= "2019.3";
            assert_eq!(shader.blobs.len(), if nested { 3 } else { 2 });
            if nested {
                assert_eq!(shader.blobs[1].bytes, b"second");
                assert_eq!(shader.blobs[1].segment, 1);
            }
            assert_eq!(shader.blobs.last().unwrap().bytes, b"third");
        }
    }
}

#[test]
fn stripped_layout_failures_report_position_and_preserve_other_objects() {
    // Cut at every byte, including string/byte padding and the trailing fields.
    for version in [
        "2017.4.9f1",
        "2022.1.12f1",
        "2022.1.13f1",
        "2022.1.20f1",
        "2022.1.21f1",
        "2022.2.0f1",
        "2022.3.0f1",
        "2023.1.0f1",
        "2023.2.0f1",
    ] {
        let v = fixed_layout::Version::parse(version).unwrap();
        let good = fixed_shader(true, (v.0, v.1, v.2), b"test");
        for cut in 0..good.len() {
            let bytes = stripped_objects(version, true, &[good[..cut].to_vec(), good.clone()]);
            let file = SerializedFile::read(&bytes).unwrap();
            let objects: Vec<_> = file.shaders().collect();
            let error = objects[0].extract().err().unwrap().to_string();
            assert!(error.contains("object byte"), "{version} {cut}: {error}");
            assert!(error.contains("serialized byte"));
            assert_eq!(objects[1].extract().unwrap().blobs[0].bytes, b"test");
        }
    }
    let good = fixed_shader(true, (2017, 4, 9), b"test");
    for version in [
        "2016.4.0f1",
        "2023.3.0f1",
        "6000.0.0f1",
        "6000.1.0f1",
        "6000.2.0f1",
        "6000.3.0f1",
        "2019.5.0f1",
        "2017.4",
        "2017.4.9custom",
    ] {
        let bytes = stripped_objects(version, true, std::slice::from_ref(&good));
        let file = SerializedFile::read(&bytes).unwrap();
        let error = file
            .shaders()
            .next()
            .unwrap()
            .extract()
            .err()
            .unwrap()
            .to_string();
        assert!(
            error.contains("unsupported Unity Shader layout version"),
            "{error}"
        );
        assert!(error.contains("object byte 0"));
    }
}

#[test]
fn stripped_editor_prefix_and_exact_failure_offset() {
    // Object.cs:30-33, EditorExtension.cs:12-16, PPtr.cs:15-16, BuildTarget.cs:10.
    for le in [true, false] {
        let mut prefix = Writer::new(le);
        prefix.u32(7);
        for _ in 0..2 {
            prefix.u32(0);
            prefix.u64(0);
        }
        prefix
            .bytes
            .extend(fixed_shader(le, (2017, 4, 9), b"editor"));
        let version = "2017.4.9f1";
        let mut bytes = stripped_objects(version, le, &[prefix.bytes]);
        let at = 20 + version.len() + 1;
        bytes[at..at + 4].copy_from_slice(&if le {
            (-2i32).to_le_bytes()
        } else {
            (-2i32).to_be_bytes()
        });
        assert_eq!(
            SerializedFile::read(&bytes)
                .unwrap()
                .shaders()
                .next()
                .unwrap()
                .extract()
                .unwrap()
                .blobs[0]
                .bytes,
            b"editor"
        );
    }
    // Aligned m_Name declares three bytes but only one is present. Failure
    // is at its payload, byte four, rather than at the end of the object.
    let bytes = stripped_objects("2017.4.9f1", true, &[vec![3, 0, 0, 0, b'x']]);
    let file = SerializedFile::read(&bytes).unwrap();
    let object = file.shaders().next().unwrap();
    let error = object.extract().err().unwrap().to_string();
    assert!(error.contains("object byte 4"), "{error}");
    assert!(error.contains(&format!("serialized byte {}", object.origin + 4)));
}

#[test]
fn modern_shader_trees_traverse_player_tables_and_parameter_blob_indices() {
    for unity in ["2021.3.2f1", "2022.3.0f1", "2023.2.0f1", "6000.0.0f1"] {
        for version in 22..=26 {
            for le in [false, true] {
                let bytes =
                    modern_serialized(version, le, unity, &[b"modern compiled bytes".to_vec()]);
                let file = SerializedFile::read(&bytes).unwrap();
                assert_eq!(file.unity_version, unity);
                let shader = file.shaders().next().unwrap().extract().unwrap();
                assert_eq!(shader.blobs.len(), 3);
                assert_eq!(shader.blobs[0].bytes, b"modern compiled bytes");
                assert_eq!(shader.blobs[1].bytes, b"second segment");
                assert_eq!(shader.blobs[2].bytes, b"other platform");
                assert_eq!(shader.blobs[0].offset, 3);
            }
        }
    }
}

#[test]
fn extended_header_and_object_offsets_never_truncate_high_bits() {
    for version in 22..=26 {
        for le in [false, true] {
            let mut original = serialized(version, le, true, &[b"offset test".to_vec()]);
            // SerializedFile.py:281-285: the extended values replace the old
            // u32 header values. The endian flag applies only to metadata.
            for at in [0, 4, 12] {
                original[at..at + 4].copy_from_slice(&u32::MAX.to_be_bytes());
            }
            assert!(serialized_hint(&original));
            let file = SerializedFile::read(&original).unwrap();
            assert_eq!(
                file.shaders().next().unwrap().extract().unwrap().blobs[0].bytes,
                b"offset test"
            );
            for at in [24, 32] {
                let mut bytes = original.clone();
                let low = u64::from_be_bytes(bytes[at..at + 8].try_into().unwrap());
                bytes[at..at + 8].copy_from_slice(&(low | 1 << 32).to_be_bytes());
                assert!(SerializedFile::read(&bytes)
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("size/offset mismatch"));
            }
            // Locate the object table by traversing the metadata, then set a
            // path ID above u32 and a start whose low word still looks valid.
            let mut m = Reader::new(&original, le);
            m.at = 48;
            m.cstr().unwrap();
            m.u32().unwrap();
            m.boolean().unwrap();
            let mut budget = TreeBudget {
                nodes: MAX_ITEMS,
                strings: MAX_BYTES,
            };
            for _ in 0..m.count().unwrap() {
                read_type(&mut m, version, &mut budget, true).unwrap();
            }
            m.count().unwrap();
            m.align(4).unwrap();
            let id_at = m.at;
            let start_at = id_at + 8;
            let mut bytes = original.clone();
            let id = 0x1234_5678_0000_0064u64;
            bytes[id_at..id_at + 8].copy_from_slice(&if le {
                id.to_le_bytes()
            } else {
                id.to_be_bytes()
            });
            assert_eq!(
                SerializedFile::read(&bytes)
                    .unwrap()
                    .shaders()
                    .next()
                    .unwrap()
                    .path_id,
                id as i64
            );
            let start = 1u64 << 32;
            bytes[start_at..start_at + 8].copy_from_slice(&if le {
                start.to_le_bytes()
            } else {
                start.to_be_bytes()
            });
            assert!(SerializedFile::read(&bytes)
                .err()
                .unwrap()
                .to_string()
                .contains("range outside input"));
        }
    }
}

#[test]
fn wrapped_type_trees_reject_bad_sizes_magic_versions_and_truncation() {
    for le in [false, true] {
        let mut w = Writer::new(le);
        write_type(&mut w, 23, 48, &schema(true));
        // SerializedType prefix is 23 bytes; hash is 16, size is 4.
        let size_at = 39;
        let payload_at = 43;
        for (at, replacement, reason) in [
            (size_at, 7u32, "range outside input"),
            (payload_at, 0, "magic"),
            (payload_at + 4, 99, "blob version"),
        ] {
            let mut bytes = w.bytes.clone();
            bytes[at..at + 4].copy_from_slice(&if le {
                replacement.to_le_bytes()
            } else {
                replacement.to_be_bytes()
            });
            let error = read_type(
                &mut Reader::new(&bytes, le),
                23,
                &mut TreeBudget {
                    nodes: MAX_ITEMS,
                    strings: MAX_BYTES,
                },
                true,
            )
            .err()
            .unwrap()
            .to_string();
            assert!(error.contains(reason), "{error}");
        }
        for cut in 0..w.bytes.len() {
            assert!(read_type(
                &mut Reader::new(&w.bytes[..cut], le),
                23,
                &mut TreeBudget {
                    nodes: MAX_ITEMS,
                    strings: MAX_BYTES
                },
                true
            )
            .is_err());
        }
    }
}

#[test]
fn unityfs_revision_gates_padding_encryption_and_format6_alignment() {
    let files = [("level0", b"synthetic bundle contents".to_vec())];
    for revision in [
        "2020.3.34f1",
        "2021.3.2f1",
        "2022.1.1f1",
        "2023.2.0f1",
        "6000.0.0f1",
    ] {
        for at_end in [false, true] {
            for info_codec in 0..=3 {
                for block_codec in 0..=3 {
                    let bytes =
                        bundle_revision(8, revision, at_end, true, info_codec, block_codec, &files);
                    let bundle = Bundle::read(&bytes).unwrap();
                    assert_eq!(bundle.revision, revision);
                    assert_eq!(bundle.data, files[0].1);
                }
            }
        }
    }
    for revision in ["2020.3.33f1", "2021.3.1f1", "2022.1.0f1", "custom"] {
        let bytes = bundle_revision(7, revision, true, true, 2, 3, &files);
        assert!(Bundle::read(&bytes)
            .err()
            .unwrap()
            .to_string()
            .contains("encryption or ambiguous"));
    }
    for revision in ["2019.4.14f1", "2019.4.15f1"] {
        for at_end in [false, true] {
            let bytes = bundle_revision(6, revision, at_end, false, 3, 1, &files);
            assert_eq!(Bundle::read(&bytes).unwrap().data, files[0].1);
        }
    }
}

// UnityPy/helpers/TypeTreeNode.py:349-356,376-393: project-authored
// wrapped node records with an explicit hash table. Offsets below come from
// :304-318 (32-byte v22 nodes), not from reference assets.
fn write_stripped_type(w: &mut Writer, class: u32) {
    // UnityPy/files/SerializedFile.py:121-138: type prefix without a tree.
    w.u32(class);
    w.bytes.push(0);
    w.u16(u16::MAX);
    if class == 114 {
        w.bytes.extend([0; 16]);
    }
    w.bytes.extend([0; 16]);
}

fn shared_handler(
    le: bool,
    hash: [u8; 16],
    specs: &[Spec],
    shared: &[(usize, u64)],
    references: &[[u8; 16]],
) -> Vec<u8> {
    let mut w = Writer::new(le);
    write_type(&mut w, 26, 48, specs);
    w.bytes.truncate(w.bytes.len() - 12); // dependency array belongs to the type
    let mut handler = w.bytes.split_off(23);
    handler[..16].copy_from_slice(&hash);
    for &(index, reference) in shared {
        let at = 20 + 16 + index * 32;
        handler[at + 3] |= 0x20;
        handler[at + 24..at + 32].copy_from_slice(&if le {
            reference.to_le_bytes()
        } else {
            reference.to_be_bytes()
        });
    }
    handler.truncate(handler.len() - 4); // replace reference count
    let mut tail = Writer::new(le);
    tail.u32(references.len() as u32);
    for hash in references {
        tail.bytes.extend(hash);
    }
    handler.extend(tail.bytes);
    let len = (handler.len() - 20) as u32;
    handler[16..20].copy_from_slice(&if le {
        len.to_le_bytes()
    } else {
        len.to_be_bytes()
    });
    handler
}

pub(crate) fn shared_serialized(le: bool, payload: &[u8], fault: &str) -> Vec<u8> {
    const UINT: [u8; 16] = [1; 16];
    const ROWS: [u8; 16] = [2; 16];
    const BYTES: [u8; 16] = [3; 16];
    let full = schema(true);
    let mut inline = Vec::new();
    let mut links = Vec::new();
    let mut at = 0;
    while at < full.len() {
        let spec = &full[at];
        let reference = match spec.name {
            "platforms" => Some(0),
            "offsets" | "compressedLengths" | "decompressedLengths" => Some(1),
            "compressedBlob" => Some(2),
            _ => None,
        };
        if let Some(reference) = reference {
            links.push((
                inline.len(),
                if fault == "index" {
                    u64::MAX
                } else {
                    reference
                },
            ));
        }
        inline.push(Spec {
            ty: spec.ty,
            name: spec.name,
            level: spec.level,
            width: spec.width,
            meta: spec.meta,
        });
        at += 1;
        if reference.is_some() {
            while at < full.len() && full[at].level > spec.level {
                at += 1;
            }
        }
    }
    let hashes = [
        if fault == "missing" { [99; 16] } else { UINT },
        ROWS,
        BYTES,
    ];
    let handler = shared_handler(le, [9; 16], &inline, &links, &hashes);
    let mut m = Writer::new(le);
    m.bytes.resize(48, 0);
    m.cstr("6000.3.0f1");
    m.u32(38);
    m.bytes.push(1);
    m.u32(1);
    write_stripped_type(&mut m, 48);
    m.bytes.extend(handler);
    m.u32(0); // type dependencies
    let object = shader_data(le, true, false, payload, 0);
    m.u32(1);
    m.align(4);
    m.u64(100);
    m.u64(0);
    m.u32(object.len() as u32);
    m.u32(0);
    // UnityPy/files/SerializedFile.py:47-53,79-85,318-338: nonempty
    // scripts, externals and reference types before the shared table.
    m.u32(1);
    m.u32(7);
    m.align(4);
    m.u64(0x1234_5678_0000_0001);
    m.u32(1);
    m.cstr("");
    m.bytes.extend([7; 16]);
    m.u32(2);
    m.cstr("synthetic-reference.assets");
    m.u32(1);
    let mut ref_type = Writer::new(le);
    write_stripped_type(&mut ref_type, 114);
    ref_type
        .bytes
        .extend(shared_handler(le, [10; 16], &inline, &links, &hashes));
    ref_type.cstr("Synthetic");
    ref_type.cstr("Namespace");
    ref_type.cstr("Assembly");
    m.bytes.extend(ref_type.bytes);
    m.u32(if fault == "duplicate" { 4 } else { 3 });
    let mut uint = Vec::new();
    vector(&mut uint, "Base", 0, false, false);
    m.bytes.extend(shared_handler(le, UINT, &uint, &[], &[]));
    let mut rows = Vec::new();
    vector(&mut rows, "Base", 0, true, false);
    // Replace the nested vector element with the shared u32 vector's children.
    rows.truncate(4);
    m.bytes.extend(shared_handler(
        le,
        ROWS,
        &rows,
        &[(3, 0)],
        &[if fault == "cycle" { ROWS } else { UINT }],
    ));
    let mut bytes = Vec::new();
    vector(&mut bytes, "Base", 0, false, true);
    m.bytes.extend(shared_handler(le, BYTES, &bytes, &[], &[]));
    if fault == "duplicate" {
        m.bytes.extend(shared_handler(le, UINT, &uint, &[], &[]));
    }
    m.cstr("");
    let metadata = m.bytes.len() - 48;
    m.align(16);
    let data_offset = m.bytes.len();
    m.bytes.extend(object);
    let mut h = Writer::new(false);
    for value in [0, 0, 26, 0] {
        h.u32(value);
    }
    h.bytes.push(u8::from(!le));
    h.bytes.extend([0; 3]);
    h.u32(metadata as u32);
    h.u64(m.bytes.len() as u64);
    h.u64(data_offset as u64);
    h.u64(0);
    m.bytes[..48].copy_from_slice(&h.bytes);
    m.bytes
}

#[test]
fn shared_trees_expand_nested_reused_arrays_and_keep_all_segments() {
    for le in [false, true] {
        let bytes = shared_serialized(le, b"shared compiled bytes", "");
        let file = SerializedFile::read(&bytes).unwrap();
        let object = file.shaders().next().unwrap();
        let shader = object.extract().unwrap();
        assert_eq!(shader.name, "../synthetic/shader-0");
        assert_eq!(shader.blobs.len(), 3);
        assert_eq!(shader.blobs[0].bytes, b"shared compiled bytes");
        assert_eq!(shader.blobs[1].bytes, b"second segment");
        assert_eq!(shader.blobs[2].bytes, b"other platform");
        assert!(object.nodes.iter().all(|n| n.shared.is_none()));
    }
}

#[test]
fn shared_trees_reject_missing_hash_indices_cycles_duplicates_and_expansion_limits() {
    for le in [false, true] {
        for (fault, reason) in [
            ("missing", "missing Unity shared subtree hash"),
            ("index", "Unity shared reference index out of range"),
            ("cycle", "cyclic Unity shared subtree reference"),
            ("duplicate", "duplicate Unity shared subtree hash"),
        ] {
            let bytes = shared_serialized(le, b"synthetic", fault);
            let error = SerializedFile::read(&bytes).err().unwrap().to_string();
            assert!(error.contains(reason), "{fault}: {error}");
        }
        let mut nodes = Vec::new();
        vector(&mut nodes, "Base", 0, false, false);
        let bytes = shared_handler(le, [1; 16], &nodes, &[], &[]);
        let (_, tree) = read_tree_handler(
            &mut Reader::new(&bytes, le),
            &mut TreeBudget {
                nodes: MAX_ITEMS,
                strings: MAX_BYTES,
            },
        )
        .unwrap();
        let mut types = [SerializedType {
            class: 48,
            nodes: tree.0,
            references: tree.1,
        }];
        let mut tail = Writer::new(le);
        for _ in 0..4 {
            tail.u32(0);
        }
        tail.cstr("");
        let error = read_shared_metadata(
            &mut Reader::new(&tail.bytes, le),
            26,
            true,
            &mut types,
            &mut TreeBudget {
                nodes: 0,
                strings: MAX_BYTES,
            },
        )
        .err()
        .unwrap()
        .to_string();
        assert!(error.contains("expanded tree nodes"));
        let error = read_shared_metadata(
            &mut Reader::new(&tail.bytes, le),
            26,
            true,
            &mut types,
            &mut TreeBudget {
                nodes: MAX_ITEMS,
                strings: 0,
            },
        )
        .err()
        .unwrap()
        .to_string();
        assert!(error.contains("expanded tree strings"));
    }
}

// Synthetic sparse storage has no backing bytes in the gap. Any attempt to
// read that gap fails, proving extraction seeks rather than reading the file.
struct SparseInput {
    parts: Vec<(u64, Vec<u8>)>,
    length: u64,
    at: u64,
    read: usize,
}
impl Read for SparseInput {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        if bytes.is_empty() || self.at == self.length {
            return Ok(0);
        }
        let (origin, data) = self
            .parts
            .iter()
            .find(|(origin, data)| (*origin..*origin + data.len() as u64).contains(&self.at))
            .ok_or_else(|| invalid("synthetic sparse gap was read"))?;
        let at = (self.at - origin) as usize;
        let count = bytes.len().min(data.len() - at);
        bytes[..count].copy_from_slice(&data[at..at + count]);
        self.at += count as u64;
        self.read += count;
        Ok(count)
    }
}
impl Seek for SparseInput {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        let at = match from {
            SeekFrom::Start(at) => i128::from(at),
            SeekFrom::End(offset) => i128::from(self.length) + i128::from(offset),
            SeekFrom::Current(offset) => i128::from(self.at) + i128::from(offset),
        };
        if !(0..=i128::from(self.length)).contains(&at) {
            return Err(invalid("synthetic invalid seek"));
        }
        self.at = at as u64;
        Ok(self.at)
    }
}

pub(crate) fn sparse_serialized_parts(bytes: &[u8], object_gap: u64) -> (Vec<u8>, u64, Vec<u8>) {
    let le = bytes[16] == 0;
    let version = u32::from_be_bytes(bytes[8..12].try_into().unwrap());
    let original_offset = if version >= 22 {
        u64::from_be_bytes(bytes[32..40].try_into().unwrap()) as usize
    } else {
        u32::from_be_bytes(bytes[12..16].try_into().unwrap()) as usize
    };
    let data = bytes[original_offset..].to_vec();
    let mut prefix = bytes[..original_offset].to_vec();
    let mut m = Reader::new(&prefix, le);
    m.at = if version >= 22 { 48 } else { 20 };
    m.cstr().unwrap();
    m.u32().unwrap();
    let has_tree = m.boolean().unwrap();
    let mut budget = TreeBudget {
        nodes: MAX_ITEMS,
        strings: MAX_BYTES,
    };
    for _ in 0..m.count().unwrap() {
        read_type(&mut m, version, &mut budget, has_tree).unwrap();
    }
    let mut starts = Vec::new();
    for _ in 0..m.count().unwrap() {
        m.align(4).unwrap();
        m.u64().unwrap();
        let at = m.at;
        let old = if version >= 22 {
            m.u64().unwrap()
        } else {
            u64::from(m.u32().unwrap())
        };
        starts.push((at, old + object_gap));
        m.take(8).unwrap();
    }
    for (at, start) in starts {
        let encoded = if version >= 22 {
            if le {
                start.to_le_bytes()
            } else {
                start.to_be_bytes()
            }
            .to_vec()
        } else {
            let start = u32::try_from(start).unwrap();
            if le {
                start.to_le_bytes()
            } else {
                start.to_be_bytes()
            }
            .to_vec()
        };
        prefix[at..at + encoded.len()].copy_from_slice(&encoded);
    }
    let data_offset = if version >= 22 {
        (1u64 << 32) + 4096
    } else {
        MAX_BYTES as u64 + 4096
    };
    let position = data_offset + object_gap;
    let length = position + data.len() as u64;
    if version >= 22 {
        prefix[24..32].copy_from_slice(&length.to_be_bytes());
        prefix[32..40].copy_from_slice(&data_offset.to_be_bytes());
    } else {
        prefix[4..8].copy_from_slice(&u32::try_from(length).unwrap().to_be_bytes());
        prefix[12..16].copy_from_slice(&u32::try_from(data_offset).unwrap().to_be_bytes());
    }
    (prefix, position, data)
}

#[test]
fn serialized_stream_seeks_above_512_mib_and_u32_without_reading_gaps() {
    for version in 17..=26 {
        for le in [false, true] {
            let large_gap = if version >= 22 {
                (1u64 << 32) + 16
            } else {
                MAX_BYTES as u64 + 16
            };
            for gap in [0, large_gap] {
                let bytes = serialized(version, le, true, &[b"large sparse input".to_vec()]);
                let (prefix, at, data) = sparse_serialized_parts(&bytes, gap);
                let original_size = bytes.len();
                assert!(serialized_header_hint(
                    &prefix[..48],
                    at + data.len() as u64
                ));
                let mut input = SparseInput {
                    length: at + data.len() as u64,
                    at: 0,
                    read: 0,
                    parts: vec![(0, prefix), (at, data)],
                };
                let file = SerializedFile::read_stream(&mut input).unwrap();
                let shader = file
                    .shaders()
                    .next()
                    .unwrap()
                    .extract_stream(&mut input)
                    .unwrap();
                assert_eq!(shader.blobs[0].bytes, b"large sparse input");
                assert!(input.read <= original_size);
            }
        }
    }
}

#[test]
fn unityfs_stream_spools_totals_and_directory_ranges_above_512_mib() {
    // AssetStudio/BundleFile.cs:218-223,285-308,317-357: many small
    // compressed blocks whose concatenated directory exceeds the buffer cap.
    // Generate compressed zero blocks, then count output bytes without storing
    // them. This tests the size boundary without a large memory/disk fixture.
    const BLOCK: usize = 64 * 1024;
    let count = MAX_BYTES / BLOCK + 1;
    let packed = lz4_flex::block::compress(&vec![0; BLOCK]);
    let mut info = Writer::new(false);
    info.bytes.extend([0; 16]);
    info.u32(count as u32);
    for _ in 0..count {
        info.u32(BLOCK as u32);
        info.u32(packed.len() as u32);
        info.u16(2);
    }
    info.u32(1);
    info.u64(MAX_BYTES as u64 + 16);
    info.u64((BLOCK - 16) as u64);
    info.u32(0);
    info.cstr("synthetic.resS");
    let mut w = Writer::new(false);
    w.cstr("UnityFS");
    w.u32(8);
    w.cstr("2022.1.21f1");
    w.cstr("2022.1.21f1");
    let size_at = w.bytes.len();
    w.u64(0);
    w.u32(info.bytes.len() as u32);
    w.u32(info.bytes.len() as u32);
    w.u32(0x40);
    w.align(16);
    w.bytes.extend(info.bytes);
    for _ in 0..count {
        w.bytes.extend(&packed);
    }
    let length = w.bytes.len() as u64;
    w.bytes[size_at..size_at + 8].copy_from_slice(&length.to_be_bytes());
    #[derive(Default)]
    struct CountOutput {
        bytes: usize,
        largest: usize,
    }
    impl Write for CountOutput {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.bytes += bytes.len();
            self.largest = self.largest.max(bytes.len());
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut output = CountOutput::default();
    let index = Bundle::read_stream(&mut Cursor::new(&w.bytes), &mut output).unwrap();
    assert_eq!(output.bytes, MAX_BYTES + BLOCK);
    assert_eq!(output.largest, BLOCK);
    assert_eq!(index.files[0].range, MAX_BYTES + 16..MAX_BYTES + BLOCK);
    struct FailOutput;
    impl Write for FailOutput {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::other("synthetic spool failure"))
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let error = Bundle::read_stream(&mut Cursor::new(&w.bytes), &mut FailOutput)
        .err()
        .unwrap();
    assert_eq!(error.kind(), io::ErrorKind::Other);
    assert_eq!(error.to_string(), "synthetic spool failure");
}

#[test]
fn unityfs_subfile_bounds_relative_seeks_and_large_origins() {
    let origin = (1u64 << 32) + 16;
    let mut input = SparseInput {
        parts: vec![(origin, b"directory bytes".to_vec())],
        length: origin + 15,
        at: 0,
        read: 0,
    };
    let mut file = SubFile::new(&mut input, origin as usize..origin as usize + 15);
    let mut bytes = [0; 32];
    assert_eq!(file.read(&mut bytes).unwrap(), 15);
    assert_eq!(&bytes[..15], b"directory bytes");
    assert_eq!(file.read(&mut bytes).unwrap(), 0);
    assert_eq!(file.seek(SeekFrom::End(-5)).unwrap(), 10);
    assert_eq!(file.read(&mut bytes).unwrap(), 5);
    assert_eq!(&bytes[..5], b"bytes");
    assert_eq!(file.seek(SeekFrom::Current(-15)).unwrap(), 0);
    assert!(file.seek(SeekFrom::Current(-1)).is_err());
    assert!(file.seek(SeekFrom::Start(16)).is_err());
    assert!(file.seek(SeekFrom::End(1)).is_err());
}

#[test]
fn unityfs_raw_blocks_above_buffer_limit_use_bounded_copy() {
    // AssetStudio/BundleFile.cs:285-308,321-327: a single raw block with a
    // directory near its end. Only the generated header/info has backing bytes.
    let block_length = MAX_BYTES + 16;
    let mut info = Writer::new(false);
    info.bytes.extend([0; 16]);
    info.u32(1);
    info.u32(block_length as u32);
    info.u32(block_length as u32);
    info.u16(0);
    info.u32(1);
    info.u64(MAX_BYTES as u64);
    info.u64(16);
    info.u32(0);
    info.cstr("synthetic.resS");
    let mut w = Writer::new(false);
    w.cstr("UnityFS");
    w.u32(8);
    w.cstr("2022.3.0f1");
    w.cstr("2022.3.0f1");
    let size_at = w.bytes.len();
    w.u64(0);
    w.u32(info.bytes.len() as u32);
    w.u32(info.bytes.len() as u32);
    w.u32(0x40);
    w.align(16);
    w.bytes.extend(info.bytes);
    let length = w.bytes.len() as u64 + block_length as u64;
    w.bytes[size_at..size_at + 8].copy_from_slice(&length.to_be_bytes());
    struct ZeroInput {
        prefix: Vec<u8>,
        length: u64,
        at: u64,
        calls: usize,
    }
    impl Read for ZeroInput {
        fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
            assert!(bytes.len() <= 64 * 1024, "unbounded raw block read");
            self.calls += 1;
            let count = bytes.len().min((self.length - self.at) as usize);
            if self.at < self.prefix.len() as u64 {
                let count = count.min(self.prefix.len() - self.at as usize);
                bytes[..count]
                    .copy_from_slice(&self.prefix[self.at as usize..self.at as usize + count]);
                self.at += count as u64;
                return Ok(count);
            }
            bytes[..count].fill(0);
            self.at += count as u64;
            Ok(count)
        }
    }
    impl Seek for ZeroInput {
        fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
            self.at = match from {
                SeekFrom::Start(at) => at,
                SeekFrom::End(0) => self.length,
                _ => return Err(invalid("unexpected synthetic seek")),
            };
            assert!(self.at <= self.length);
            Ok(self.at)
        }
    }
    let mut input = ZeroInput {
        prefix: w.bytes,
        length,
        at: 0,
        calls: 0,
    };
    let bundle = Bundle::read_stream(&mut input, &mut io::sink()).unwrap();
    assert_eq!(bundle.files[0].range, MAX_BYTES..MAX_BYTES + 16);
    assert_eq!(input.at, length);
    assert!(input.calls > MAX_BYTES / (64 * 1024));
}
