//! Fixed Shader layouts from AssetStudio/Classes/Shader.cs at the commit in
//! docs/reference-sources.md, with patch gates from the MIT RazTools/Studio
//! and zhangjiequan/AssetStudio references pinned in
//! docs/reference-sources.md This locates
//! compiled blobs; it does not translate shader instructions or render state.
use super::*;

#[derive(Clone, Copy)]
pub(super) struct Version(pub(super) u32, pub(super) u32, pub(super) u32);
impl Version {
    pub(super) fn parse(text: &str) -> io::Result<Self> {
        let unsupported = || invalid(format!("unsupported Unity Shader layout version {text:?}"));
        let mut parts = text.splitn(3, '.');
        let major = parts
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(unsupported)?;
        let minor = parts
            .next()
            .and_then(|s| s.parse().ok())
            .ok_or_else(unsupported)?;
        let tail = parts.next().ok_or_else(unsupported)?;
        let digits = tail.bytes().take_while(u8::is_ascii_digit).count();
        let patch = tail[..digits].parse().map_err(|_| unsupported())?;
        // Restrict support to released version families covered by this reader.
        // Unknown/custom version strings cannot select a nearby layout.
        let suffix = &tail[digits..];
        if !(matches!(
            (major, minor),
            (2017..=2019, 1..=4) | (2020..=2023, 1..=3) | (6000, 0..=3)
        ) && suffix.len() >= 2
            && matches!(suffix.as_bytes()[0], b'a' | b'b' | b'f' | b'p')
            && suffix[1..].bytes().all(|b| b.is_ascii_digit()))
        {
            return Err(unsupported());
        }
        Ok(Self(major, minor, patch))
    }
    fn since(self, major: u32, minor: u32) -> bool {
        (self.0, self.1) >= (major, minor)
    }
    fn common_parameters(self) -> bool {
        // AssetStudio/Classes/Shader.cs:607-614,708-715: shared parameters
        // start at 2020.3.2 and 2021.1.1 (separate release branches).
        (self.0 == 2020 && (self.1, self.2) >= (3, 2))
            || (self.0 == 2021 && (self.1, self.2) >= (1, 1))
            || self.0 > 2021
    }
    fn player_tables(self) -> bool {
        // RazTools/Studio AssetStudio/Classes/Shader.cs:789-806: 2021.3.10
        // and 2022.1.13 add nested player programs and parameter blob indices.
        // zhangjiequan/AssetStudio AssetStudio/Classes/Shader.cs:731-750
        // also explicitly includes 2022.2+ and 2023+.
        (self.0 == 2021 && (self.1, self.2) >= (3, 10))
            || (self.0 == 2022 && (self.1, self.2) >= (1, 13))
            || self.0 >= 2023
    }
    fn stage_counts(self) -> bool {
        // RazTools/Studio AssetStudio/Classes/Shader.cs:1120-1125: a distinct
        // gate, 2021.3.12 / 2022.1.21, adds the trailing u32 stageCounts array.
        // zhangjiequan/AssetStudio AssetStudio/Classes/Shader.cs:1047-1055
        // also explicitly includes 2022.2+ and 2023+.
        (self.0 == 2021 && (self.1, self.2) >= (3, 12))
            || (self.0 == 2022 && (self.1, self.2) >= (1, 21))
            || self.0 >= 2023
    }
    fn partial_buffer(self) -> bool {
        // AssetStudio/Classes/Shader.cs:429-437: partial CB starts later,
        // at 2021.1.4; do not use the common-parameters gate here.
        (self.0 == 2020 && (self.1, self.2) >= (3, 2))
            || (self.0 == 2021 && (self.1, self.2) >= (1, 4))
            || self.0 > 2021
    }
}

struct Layout<'r, 'data> {
    r: &'r mut Reader<'data>,
    v: Version,
    visits: usize,
}
impl Layout<'_, '_> {
    fn words(&mut self, n: usize) -> io::Result<()> {
        self.r.take(n * 4)?;
        Ok(())
    }
    fn string(&mut self) -> io::Result<&[u8]> {
        // AssetStudio/Extensions/BinaryReaderExtensions.cs:25-35: signed
        // byte length, UTF-8, four-byte alignment. Reject malformed lengths.
        let n = size(u64::from(self.r.u32()?))?;
        let bytes = self.r.take(n)?;
        std::str::from_utf8(bytes).map_err(|_| invalid("invalid Unity UTF-8"))?;
        self.r.align(4)?;
        Ok(bytes)
    }
    fn count(&mut self) -> io::Result<usize> {
        let n = self.r.count()?;
        self.visits = self
            .visits
            .checked_sub(n)
            .ok_or_else(|| invalid("Unity layout visits exceed resource limit"))?;
        Ok(n)
    }
    fn list(&mut self, mut item: impl FnMut(&mut Self) -> io::Result<()>) -> io::Result<()> {
        let n = self.count()?;
        for _ in 0..n {
            item(self)?;
        }
        Ok(())
    }
    fn string_array(&mut self) -> io::Result<()> {
        // BinaryReaderExtensions.cs:145-147: count, aligned strings.
        self.list(|s| {
            s.string()?;
            Ok(())
        })
    }
    fn short_array(&mut self) -> io::Result<()> {
        // BinaryReaderExtensions.cs:105-107; Shader.cs:589-600,772-775:
        // count, u16 values, then align to four.
        let n = self.r.count()?;
        self.r.take(n * 2)?;
        self.r.align(4)
    }
    fn tags(&mut self) -> io::Result<()> {
        // Shader.cs:822-827: count, (aligned key, aligned value) pairs.
        self.list(|s| {
            s.string()?;
            s.string()?;
            Ok(())
        })
    }
    fn pointer(&mut self) -> io::Result<()> {
        // AssetStudio/Classes/PPtr.cs:15-16: i32 file ID, i64 path ID in
        // serialized versions >=14 (our reader supports 17..=26).
        self.words(3)
    }
    fn property(&mut self) -> io::Result<()> {
        // Shader.cs:103-109,76-77: name, description, attributes, i32 type,
        // u32 flags, four f32 defaults, texture default name and i32 dimension.
        self.string()?;
        self.string()?;
        self.string_array()?;
        self.words(6)?;
        self.string()?;
        self.words(1)
    }
    fn float_value(&mut self) -> io::Result<()> {
        // Shader.cs:135-136: f32 value, aligned parameter name.
        self.words(1)?;
        self.string()?;
        Ok(())
    }
    fn state(&mut self) -> io::Result<()> {
        // Shader.cs:238-245,152-158: name, eight RT blends of seven float
        // values (src/dst, src/dst alpha, op/op alpha, color mask), boolean.
        self.string()?;
        for _ in 0..56 {
            self.float_value()?;
        }
        self.r.boolean()?;
        self.r.align(4)?;
        // Shader.cs:246-259: zClip from 2017.2, zTest, zWrite, culling,
        // conservative from 2020.1, offsetFactor, offsetUnits, alphaToMask.
        if self.v.since(2017, 2) {
            self.float_value()?;
        }
        for _ in 0..3 {
            self.float_value()?;
        }
        if self.v.since(2020, 1) {
            self.float_value()?;
        }
        for _ in 0..3 {
            self.float_value()?;
        }
        // Shader.cs:260-269,171-174,188-192: three stencil operations with
        // pass/fail/zFail/comp; read/write masks, ref, fog start/end/density;
        // fogColor's four float values and name.
        for _ in 0..22 {
            self.float_value()?;
        }
        self.string()?;
        // Shader.cs:270-275: fogMode, gpuProgramID, tags, LOD, lighting.
        self.words(2)?;
        self.tags()?;
        self.words(1)?;
        self.r.boolean()?;
        self.r.align(4)
    }
    fn vector_or_matrix(&mut self) -> io::Result<()> {
        // Shader.cs:320-325,339-344: name index, index, array size, sbyte
        // type, sbyte dimension/row count, alignment.
        self.words(3)?;
        self.r.take(2)?;
        self.r.align(4)
    }
    fn texture(&mut self) -> io::Result<()> {
        // Shader.cs:359-367: name/index/sampler index, multisampled boolean
        // from 2017.3, sbyte dimension, alignment.
        self.words(3)?;
        if self.v.since(2017, 3) {
            self.r.boolean()?;
        }
        self.r.u8()?;
        self.r.align(4)
    }
    fn binding(&mut self) -> io::Result<()> {
        // Shader.cs:381-385: name index, index, array size from 2020.1.
        self.words(if self.v.since(2020, 1) { 3 } else { 2 })
    }
    fn constant_buffer(&mut self) -> io::Result<()> {
        // Shader.cs:403-427: name index, matrix list, vector list, struct
        // list from 2017.3, size. Struct fields Shader.cs:25-41: name/index/
        // array size/struct size, vector list then matrix list.
        self.words(1)?;
        self.list(Self::vector_or_matrix)?;
        self.list(Self::vector_or_matrix)?;
        if self.v.since(2017, 3) {
            self.list(|s| {
                s.words(4)?;
                s.list(Self::vector_or_matrix)?;
                s.list(Self::vector_or_matrix)
            })?;
        }
        self.words(1)?;
        // Shader.cs:429-437: isPartialCB, aligned, from 2020.3.2/2021.1.4.
        if self.v.partial_buffer() {
            self.r.boolean()?;
            self.r.align(4)?;
        }
        Ok(())
    }
    fn parameters(&mut self) -> io::Result<()> {
        // Shader.cs:505-559,617-674: vector, matrix, texture, buffer,
        // constant buffer, constant buffer binding, UAV and sampler lists.
        self.list(Self::vector_or_matrix)?;
        self.list(Self::vector_or_matrix)?;
        self.list(Self::texture)?;
        self.list(Self::binding)?;
        self.list(Self::constant_buffer)?;
        self.list(Self::binding)?;
        // Shader.cs:449-451: UAV name/index/original index;
        // Shader.cs:53-54: sampler u32 and i32 bind point.
        self.list(|s| s.words(3))?;
        self.list(|s| s.words(2))
    }
    fn subprogram(&mut self) -> io::Result<()> {
        // Shader.cs:584-585,298-306,286-287: blob index, channel count,
        // sbyte source/target pairs, alignment, u32 source map.
        self.words(1)?;
        let channels = self.r.count()?;
        self.r.take(channels * 2)?;
        self.r.align(4)?;
        self.words(1)?;
        // Shader.cs:587-605: separate global/local indices for 2019..2021.1;
        // earlier and later versions have one keyword array. Tier and program
        // type are sbytes, followed by four-byte alignment.
        self.short_array()?;
        if self.v.since(2019, 1) && !self.v.since(2021, 2) {
            self.short_array()?;
        }
        self.r.take(2)?;
        self.r.align(4)?;
        self.parameters()?;
        // Shader.cs:677-686: requirements are i32 from 2017.2, i64 from 2021.
        if self.v.since(2017, 2) {
            self.words(if self.v.since(2021, 1) { 2 } else { 1 })?;
        }
        Ok(())
    }
    fn player_subprogram(&mut self) -> io::Result<()> {
        // RazTools/Studio AssetStudio/Classes/Shader.cs:759-766: u32 blob
        // index, aligned u16 keywords, i64 requirements, sbyte GPU type,
        // alignment. The generated UnityPy class is not serialized order.
        self.words(1)?;
        self.short_array()?;
        self.words(2)?;
        self.r.u8()?;
        self.r.align(4)
    }
    fn program(&mut self) -> io::Result<()> {
        // Shader.cs:701-721: subprogram list, versioned common parameters,
        // aligned u16 keyword state mask from 2022.1.
        self.list(Self::subprogram)?;
        if self.v.player_tables() {
            // RazTools/Studio AssetStudio/Classes/Shader.cs:794-806:
            // outer platform count, per-platform player count, player records,
            // then nested u32 parameter blob indices. These locate compiled
            // data; neither index is a shader instruction or a blob filter.
            self.list(|s| s.list(Self::player_subprogram))?;
            self.list(|s| {
                let n = s.count()?;
                s.words(n)
            })?;
        }
        if self.v.common_parameters() {
            self.parameters()?;
        }
        if self.v.since(2022, 1) {
            self.short_array()?;
        }
        Ok(())
    }
    fn pass(&mut self) -> io::Result<()> {
        // Shader.cs:759-775,14: 2020.2 adds editor hashes (16 bytes each),
        // aligned platform byte array, local/global u16 keyword masks.
        if self.v.since(2020, 2) {
            let n = self.r.count()?;
            self.r.take(n * 16)?;
            self.r.align(4)?;
            let n = size(u64::from(self.r.u32()?))?;
            self.r.take(n)?;
            self.r.align(4)?;
            // Shader.cs:770-775: separate masks end after 2021.1.
            if !self.v.since(2021, 2) {
                self.short_array()?;
                self.short_array()?;
            }
        }
        // Shader.cs:779-793: name/index pairs, pass type, state, program
        // mask, vertex/fragment/geometry/hull/domain programs.
        self.list(|s| {
            s.string()?;
            s.words(1)
        })?;
        self.words(1)?;
        self.state()?;
        self.words(1)?;
        for _ in 0..5 {
            self.program()?;
        }
        // Shader.cs:794-807: ray tracing from 2019.3; instancing boolean,
        // procedural instancing from 2018, alignment; use/name/texture, tags.
        if self.v.since(2019, 3) {
            self.program()?;
        }
        self.r.boolean()?;
        if self.v.since(2018, 1) {
            self.r.boolean()?;
        }
        self.r.align(4)?;
        self.string()?;
        self.string()?;
        self.string()?;
        self.tags()?;
        // Shader.cs:808-811: the pass mask exists only in 2021.2/2021.3.
        // From 2022.1 it belongs to each program (:717-721).
        if self.v.0 == 2021 && self.v.1 >= 2 {
            self.short_array()?;
        }
        Ok(())
    }
    fn parsed_form(&mut self) -> io::Result<()> {
        // Shader.cs:119-123,892-899,839-847: property sheet, subshader
        // list; each subshader has passes, tags and i32 LOD.
        self.list(Self::property)?;
        self.list(|s| {
            s.list(Self::pass)?;
            s.tags()?;
            s.words(1)
        })?;
        // Shader.cs:901-905: keyword names and byte flags from 2021.2.
        if self.v.since(2021, 2) {
            self.string_array()?;
            let n = size(u64::from(self.r.u32()?))?;
            self.r.take(n)?;
            self.r.align(4)?;
        }
        // Shader.cs:908-916,858-859: name, editor, fallback, dependencies.
        self.string()?;
        self.string()?;
        self.string()?;
        self.tags()?;
        // Shader.cs:919-926,870-871: pipeline editors from 2021, pairs of
        // aligned custom editor name and render pipeline type strings.
        if self.v.since(2021, 1) {
            self.tags()?;
        }
        // Shader.cs:929-930: disableNoSubshadersMessage and alignment.
        self.r.boolean()?;
        self.r.align(4)
    }
    fn uint_array(&mut self) -> io::Result<Field> {
        // BinaryReaderExtensions.cs:120-127: signed count and u32 elements.
        let n = self.count()?;
        // Check the range before allocating. Counts share the traversal budget
        // across all tables, bounding allocations for nested rows as well.
        range(self.r.at, n * 4, self.r.data.len())?;
        let values = (0..n)
            .map(|_| self.r.u32().map(Field::UInt))
            .collect::<io::Result<_>>()?;
        Ok(Field::Array(values))
    }
    fn rows(&mut self) -> io::Result<Field> {
        // Shader.cs:984-995: flat tables before 2019.3, nested after.
        if !self.v.since(2019, 3) {
            return self.uint_array();
        }
        let n = self.count()?;
        let values = (0..n)
            .map(|_| self.uint_array())
            .collect::<io::Result<_>>()?;
        Ok(Field::Array(values))
    }
}

pub(super) fn fields(
    r: &mut Reader<'_>,
    unity_version: &str,
    target: i32,
) -> io::Result<BTreeMap<&'static str, Field>> {
    let v = Version::parse(unity_version)?;
    // UnityPy/classes/generated.py:10476: a dynamic keyword mask also exists.
    // No inspected MIT reader establishes its serialized order/version gate
    // for Unity 6. A >=2023 condition in an older reader is insufficient proof
    // of the 6000.x layout. Require supplied trees for those builds.
    if v.0 >= 6000 || (v.0 == 2023 && v.1 > 2) {
        return Err(invalid(format!(
            "unsupported Unity Shader layout version {unity_version:?}: newer layout requires a type tree"
        )));
    }
    let mut s = Layout {
        r,
        v,
        visits: MAX_VISITS,
    };
    // Object.cs:30-33; EditorExtension.cs:12-16; BuildTarget.cs:10:
    // NoTarget (-2) has hide flags and two prefab pointers before m_Name.
    if target == -2 {
        s.words(1)?;
        s.pointer()?;
        s.pointer()?;
    }
    // NamedObject.cs:12-14; Shader.cs:978-983: aligned m_Name then parsed form.
    let name = std::str::from_utf8(s.string()?).unwrap().to_owned();
    s.parsed_form()?;
    let mut fields = BTreeMap::new();
    fields.insert("m_Name", Field::Text(name));
    // Shader.cs:983-997: platforms, offsets, compressed/decompressed lengths,
    // byte-counted compressed blob and alignment.
    fields.insert("platforms", s.uint_array()?);
    for field in ["offsets", "compressedLengths", "decompressedLengths"] {
        fields.insert(field, s.rows()?);
    }
    let n = size(u64::from(s.r.u32()?))?;
    fields.insert("compressedBlob", Field::Bytes(s.r.take(n)?.to_vec()));
    s.r.align(4)?;
    if v.stage_counts() {
        // RazTools/Studio AssetStudio/Classes/Shader.cs:1120-1125:
        // stageCounts is a counted u32 array after the aligned compressedBlob.
        let n = s.count()?;
        s.words(n)?;
    }
    // Shader.cs:999-1016: dependency pointers; nonmodifiable texture name/
    // pointer pairs from 2018; baked boolean and alignment.
    s.list(|s| s.pointer())?;
    if v.since(2018, 1) {
        s.list(|s| {
            s.string()?;
            s.pointer()
        })?;
    }
    s.r.boolean()?;
    s.r.align(4)?;
    if s.r.at != s.r.data.len() {
        return Err(invalid("Shader object size disagrees with fixed layout"));
    }
    Ok(fields)
}
