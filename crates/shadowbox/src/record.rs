//! Project-owned cb1 record trailer. Layout: docs/reference-sources.md
//! Guest bytes and sizes: Ryujinx DiskCacheGuestStorage.cs:69-79,222-229.
use crate::{
    interface::{Malformed, Stopped},
    TargetGeneration,
};

const MAGIC: &[u8; 16] = b"SBXCB1RECORD0001";
const FOOTER_SIZE: usize = 40;

pub(crate) struct Record<'a> {
    pub header: &'a [u8],
    pub code: &'a [u8],
    pub cb1: &'a [u8],
}

/// What a caller does with each shape a program record's tail can have.
pub(crate) trait Envelope<'a> {
    type Output;
    /// No cb1 trailer: the whole input is a legacy record.
    fn legacy(self, input: &'a [u8]) -> Self::Output;
    /// A well-formed cb1 trailer and the parts it delimits.
    fn extended(self, record: Record<'a>) -> Self::Output;
    /// A tail that names a cb1 trailer but does not form one.
    fn malformed(self, error: Malformed) -> Self::Output;
}

/// Read the record's tail once and take the path its shape names.
pub(crate) fn open<'a, E: Envelope<'a>>(input: &'a [u8], path: E) -> E::Output {
    let magic = &input[input.len().saturating_sub(MAGIC.len())..];
    if !magic.starts_with(b"SBXCB1RECORD") {
        return path.legacy(input);
    }
    if magic != MAGIC || input.len() < FOOTER_SIZE {
        return path.malformed(Malformed::Trailer);
    }
    let footer = input.len() - FOOTER_SIZE;
    let sizes = input[footer..footer + 24].as_chunks::<8>().0;
    let [header_size, code_size, cb1_size] = [0, 1, 2].map(|i| u64::from_le_bytes(sizes[i]));
    // Header generations: Mesa nvk_shader.c:605-606, nvk_shader.h:29-30.
    // Complete machine words: Mesa nak/sm50.rs:3418-3428. The three sizes
    // must tile the bytes before the footer exactly; the sum cannot
    // overflow u128, and equality bounds each size by the input length.
    if !matches!(header_size, 80 | 128)
        || !code_size.is_multiple_of(8)
        || u128::from(header_size) + u128::from(code_size) + u128::from(cb1_size) != footer as u128
    {
        return path.malformed(Malformed::Trailer);
    }
    let code_end = (header_size + code_size) as usize;
    path.extended(Record {
        header: &input[..header_size as usize],
        code: &input[header_size as usize..code_end],
        cb1: &input[code_end..footer],
    })
}

/// Immutable little-endian cb1 words of a snapshot. Saved uint reads follow
/// Ryujinx DiskCache/DiskCacheGpuAccessor.cs:62-69.
pub(crate) fn words(bytes: &[u8]) -> Result<Vec<u32>, Malformed> {
    // Reject partial words rather than inventing unavailable bytes.
    if !bytes.len().is_multiple_of(4) {
        return Err(Malformed::PartialCb1Word);
    }
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| u32::from_le_bytes(*word))
        .collect())
}

/// Appending a trailer: only a legacy record takes one.
struct Append<'a> {
    generation: TargetGeneration,
    cb1: &'a [u8],
}

impl<'a> Envelope<'a> for Append<'_> {
    type Output = Result<Vec<u8>, Stopped>;
    fn legacy(self, input: &'a [u8]) -> Self::Output {
        let shader = crate::legacy_input(input, self.generation)?;
        if !shader.code.len().is_multiple_of(8) {
            return Err(Stopped::PartialWord);
        }
        let mut result = input.to_vec();
        result.extend_from_slice(self.cb1);
        for size in [
            shader.program_record.len(),
            shader.code.len(),
            self.cb1.len(),
        ] {
            result.extend_from_slice(&(size as u64).to_le_bytes());
        }
        result.extend_from_slice(MAGIC);
        Ok(result)
    }
    fn extended(self, _: Record<'a>) -> Self::Output {
        Err(Malformed::TrailerExists.into())
    }
    fn malformed(self, error: Malformed) -> Self::Output {
        Err(error.into())
    }
}

/// Append an immutable cb1 snapshot to a legacy header-prefixed record.
/// All header, instruction and snapshot bytes are preserved. See
/// docs/reference-sources.md for the versioned trailer layout.
pub fn encode_header_prefixed_with_constant_buffer1(
    input: &[u8],
    generation: TargetGeneration,
    cb1: &[u8],
) -> Result<Vec<u8>, Stopped> {
    open(input, Append { generation, cb1 })
}

/// Reading a record's snapshot words.
struct Snapshot;

impl<'a> Envelope<'a> for Snapshot {
    type Output = Result<Option<Vec<u32>>, Malformed>;
    fn legacy(self, _: &'a [u8]) -> Self::Output {
        Ok(None)
    }
    fn extended(self, record: Record<'a>) -> Self::Output {
        words(record.cb1).map(Some)
    }
    fn malformed(self, error: Malformed) -> Self::Output {
        Err(error)
    }
}

/// Read immutable little-endian cb1 words from a program record.
/// Legacy records return None. Saved uint reads follow Ryujinx
/// DiskCache/DiskCacheGpuAccessor.cs:62-69; trailer layout is project-owned,
/// documented in docs/reference-sources.md
pub fn snapshot(input: &[u8]) -> Result<Option<Vec<u32>>, Malformed> {
    open(input, Snapshot)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{header_prefixed_input, CacheContext};

    /// A program's identity under `context`, through [`crate::Program`].
    fn identity(
        input: &crate::interface::ShaderInput<'_>,
        context: CacheContext,
    ) -> Result<String, crate::interface::Stopped> {
        Ok(crate::Program::parse(input, context.graphics_state)?
            .identity(context.generation, context.options))
    }

    #[test]
    fn snapshots_round_trip_and_share_the_explicit_snapshot_cache_identity() {
        for generation in [
            TargetGeneration::Sm5,
            TargetGeneration::Sm6,
            TargetGeneration::Sm7_5Plus,
        ] {
            let mut legacy = vec![0; generation.shader_header_size()];
            // Schedule word followed by EXIT.T/PT: Mesa sm50.rs:3418-3428,
            // 3106-3110,225-235. Legacy code bytes remain unchanged.
            legacy.extend(0_u64.to_le_bytes());
            legacy.extend(0xe300_0000_0007_000f_u64.to_le_bytes());
            let cb1 = [0x1234_5678_u32, 0xdead_beef];
            let cb_bytes: Vec<_> = cb1.iter().flat_map(|word| word.to_le_bytes()).collect();
            let extended =
                encode_header_prefixed_with_constant_buffer1(&legacy, generation, &cb_bytes)
                    .unwrap();
            let input = header_prefixed_input(&extended, generation).unwrap();
            let old_input = header_prefixed_input(&legacy, generation).unwrap();
            assert_eq!(input.code, old_input.code);
            assert_eq!(snapshot(input.program_record).unwrap(), Some(cb1.to_vec()));
            let default_context = CacheContext {
                generation,
                ..Default::default()
            };
            let embedded_key = identity(&input, default_context).unwrap();
            let explicit_context = CacheContext {
                graphics_state: crate::interface::GraphicsState {
                    constant_buffer1: Some(&cb1),
                    ..Default::default()
                },
                ..default_context
            };
            assert_eq!(
                embedded_key,
                identity(&old_input, explicit_context).unwrap()
            );
            assert_eq!(embedded_key, identity(&input, explicit_context).unwrap());
            assert_ne!(embedded_key, identity(&old_input, default_context).unwrap());
            for index in 0..cb1.len() {
                let mut changed = cb1;
                changed[index] ^= 1;
                let bytes: Vec<_> = changed.iter().flat_map(|word| word.to_le_bytes()).collect();
                let record =
                    encode_header_prefixed_with_constant_buffer1(&legacy, generation, &bytes)
                        .unwrap();
                let changed_input = header_prefixed_input(&record, generation).unwrap();
                assert_ne!(
                    embedded_key,
                    identity(&changed_input, default_context).unwrap()
                );
                assert!(identity(&changed_input, explicit_context).is_err());
                assert!(
                    Result::from(crate::translate_header_prefixed_with_graphics_state(
                        crate::QuietRun,
                        &record,
                        generation,
                        &explicit_context.graphics_state
                    ))
                    .is_err()
                );
            }
            assert_eq!(
                Result::from(crate::translate_header_prefixed_for_generation(
                    crate::QuietRun,
                    &extended,
                    generation
                ))
                .unwrap()
                .spirv,
                Result::from(crate::translate_header_prefixed_for_generation(
                    crate::QuietRun,
                    &legacy,
                    generation
                ))
                .unwrap()
                .spirv
            );
            let empty =
                encode_header_prefixed_with_constant_buffer1(&legacy, generation, &[]).unwrap();
            assert_eq!(snapshot(&empty).unwrap(), Some(vec![]));
            assert_eq!(snapshot(&legacy).unwrap(), None);
            assert!(
                encode_header_prefixed_with_constant_buffer1(&extended, generation, &cb_bytes)
                    .is_err()
            );
        }
    }

    #[test]
    fn malformed_trailers_and_conflicting_generations_are_rejected() {
        let mut legacy = vec![0; 80];
        legacy.extend([0; 16]);
        let record = encode_header_prefixed_with_constant_buffer1(
            &legacy,
            TargetGeneration::Sm5,
            &[1, 2, 3, 4],
        )
        .unwrap();
        for (offset, value) in [
            (0, 0),
            (0, 128),
            (8, 15),
            (8, u64::MAX),
            (16, 5),
            (16, u64::MAX),
        ] {
            let mut changed = record.clone();
            let at = changed.len() - FOOTER_SIZE + offset;
            changed[at..at + 8].copy_from_slice(&value.to_le_bytes());
            assert!(header_prefixed_input(&changed, TargetGeneration::Sm5).is_err());
        }
        let mut unknown_version = record.clone();
        *unknown_version.last_mut().unwrap() = b'2';
        assert!(header_prefixed_input(&unknown_version, TargetGeneration::Sm5).is_err());
        assert!(header_prefixed_input(&record, TargetGeneration::Sm7_5Plus).is_err());
        let partial_cb1 = encode_header_prefixed_with_constant_buffer1(
            &legacy,
            TargetGeneration::Sm5,
            &[1, 2, 3],
        )
        .unwrap();
        let shader = header_prefixed_input(&partial_cb1, TargetGeneration::Sm5).unwrap();
        assert_eq!(shader.code, &[0; 16]);
        assert!(snapshot(shader.program_record).is_err());
        assert!(identity(&shader, CacheContext::default()).is_err());
        assert!(Result::from(crate::translate_header_prefixed(
            crate::QuietRun,
            &partial_cb1
        ))
        .is_err());
    }
}
