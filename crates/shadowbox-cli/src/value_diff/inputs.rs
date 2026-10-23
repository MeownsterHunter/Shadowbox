// Checker-owned host state. Cbuf allocation identity and inline word offsets:
// Ryujinx Translation/Optimizations/GlobalToStorage.cs:305-357,409-445.
// Host binding fields: project interface.rs:205-210, storage_memory.rs:36-44.
// No upstream implementation is copied. The bounded virtual layout is an
// experiment specification, not an inferred guest ABI.
use super::{constant_buffer1, module::BUFFER_SIZE, read_limited};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, fs, path::Path};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct StorageBinding {
    pub bank: u8,
    pub offset: u16,
    pub binding: u32,
}

#[derive(Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Inputs {
    pub cb1: Option<Vec<u8>>,
    pub storage: Vec<StorageBinding>,
}

impl Inputs {
    pub fn load(bytes: &[u8], program: &Path) -> Result<Self, String> {
        let path = program.with_extension("storage.json");
        let storage = match fs::symlink_metadata(&path) {
            Ok(m) if m.is_file() => serde_json::from_slice(&read_limited(&path, 65536)?)
                .map_err(|e| format!("storage sidecar: {e}"))?,
            Ok(_) => return Err("storage sidecar must be a regular file".into()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Vec::new(),
            Err(e) => return Err(format!("storage sidecar: {e}")),
        };
        let result = Self {
            cb1: constant_buffer1(bytes, &program.with_extension("cb1"))?,
            storage,
        };
        result.validate()?;
        Ok(result)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.storage.len() > 16 {
            return Err("storage capture exceeds 16 allocations".into());
        }
        let mut pairs = BTreeSet::new();
        let mut bindings = BTreeSet::new();
        for b in &self.storage {
            if b.bank >= 32 || b.offset > 65528 || b.offset % 4 != 0 || b.binding > 255 {
                return Err(
                    "storage capture requires aligned cbuf pairs and bounded bindings".into(),
                );
            }
            if !bindings.insert(b.binding)
                || !pairs.insert((b.bank, b.offset))
                || self.storage.iter().any(|other| {
                    other.bank == b.bank
                        && other.offset != b.offset
                        && other.offset.abs_diff(b.offset) < 8
                })
            {
                return Err(
                    "storage capture has duplicate or overlapping cbuf pairs/bindings".into(),
                );
            }
            if b.bank == 1 && self.cb1.is_some() {
                return Err("storage cbuf pair overlaps immutable CB1 snapshot".into());
            }
        }
        Ok(())
    }

    // Sort by allocation identity, never by descriptor enumeration or JSON
    // order. Power-of-two windows keep all bases aligned for GPU buffers.
    pub fn range(&self, binding: u32) -> Option<std::ops::Range<usize>> {
        let b = self.storage.iter().find(|b| b.binding == binding)?;
        let index = self
            .storage
            .iter()
            .filter(|other| (other.bank, other.offset) < (b.bank, b.offset))
            .count();
        let extent = BUFFER_SIZE / self.storage.len().next_power_of_two();
        Some(index * extent..(index + 1) * extent)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding(bank: u8, offset: u16, binding: u32) -> StorageBinding {
        StorageBinding {
            bank,
            offset,
            binding,
        }
    }

    #[test]
    fn allocation_layout_is_independent_of_descriptor_and_sidecar_order() {
        let mut inputs = Inputs {
            storage: vec![binding(2, 76, 9), binding(3, 12, 7)],
            ..Default::default()
        };
        inputs.validate().unwrap();
        assert_eq!(inputs.range(9), Some(0..32768));
        assert_eq!(inputs.range(7), Some(32768..65536));
        inputs.storage.reverse();
        assert_eq!(inputs.range(9), Some(0..32768));
        inputs.storage[0].binding = 11;
        assert_eq!(inputs.range(11), Some(32768..65536));
    }

    #[test]
    fn overlapping_or_ambiguous_storage_inputs_fail_closed() {
        for other in [binding(2, 76, 10), binding(2, 80, 10), binding(3, 12, 9)] {
            let inputs = Inputs {
                storage: vec![binding(2, 76, 9), other],
                ..Default::default()
            };
            assert!(inputs.validate().is_err());
        }
    }
}
