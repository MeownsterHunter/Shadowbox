// Bounded descriptor views of the same guest memory as physical capture.
// Ryujinx GlobalToStorage.cs:305-357,409-445 resolves cbuf allocation bases;
// CodeGen/Spirv/Declarations.cs:88-143 sets descriptor layout/identity.
// Shadowbox storage_memory.rs:135-157 declares set1 uint runtime arrays.
// No upstream implementation is copied.
use super::{pattern, physical, Analysis, BufferKind, Type, BUFFER_SIZE};
use crate::value_diff::{inputs::Inputs, Capture};
use std::collections::BTreeMap;

pub(in crate::value_diff) struct GlobalCapture {
    initial: Vec<u8>,
    ranges: BTreeMap<u32, std::ops::Range<usize>>,
}

impl GlobalCapture {
    pub fn normalize(&self, captures: Vec<Capture>) -> Vec<Capture> {
        let mut flat = Capture {
            set: 3,
            binding: 256,
            bytes: self.initial.clone(),
            defined: vec![true; BUFFER_SIZE],
            output: None,
        };
        let mut result = Vec::new();
        for c in captures {
            if c.set == 1 && c.output.is_none() {
                if let Some(range) = self.ranges.get(&c.binding) {
                    flat.bytes[range.clone()].copy_from_slice(&c.bytes);
                    flat.defined[range.clone()].copy_from_slice(&c.defined);
                    continue;
                }
            }
            result.push(c);
        }
        result.push(flat);
        result.sort_by_key(|c| (c.set, c.binding));
        result
    }
}

pub(super) fn initialize(a: &mut Analysis, inputs: &Inputs, seed: u64) -> Result<(), String> {
    for (&(set, binding), &(root, kind)) in &a.roots {
        let mut initial = if kind == BufferKind::Physical {
            pattern(seed, physical::DOMAIN, 0, 0, BUFFER_SIZE)
        } else if set == 1 && kind == BufferKind::Storage && !inputs.storage.is_empty() {
            if a.physical {
                // Separate VkBuffers would alias guest memory incorrectly.
                return Err("mixed physical and descriptor global memory unsupported".into());
            }
            let range = inputs
                .range(binding)
                .ok_or("unmapped descriptor storage binding")?;
            let Type::Struct(members) = a.ty(a.pointer(root)?.ty)? else {
                return Err("descriptor storage requires a block".into());
            };
            let valid = members.len() == 1
                && a.offsets.get(&(a.pointer(root)?.ty, 0)) == Some(&0)
                && matches!(a.ty(members[0])?, Type::Array(element, None)
                    if a.scalars.get(element) == Some(&super::graphics::Scalar::Uint)
                        && a.strides.get(&members[0]) == Some(&4));
            if !valid {
                return Err("descriptor storage requires a uint runtime-array view".into());
            }
            let memory = a.memory.get_mut(&root).unwrap();
            memory.data.truncate(range.len());
            memory.written.truncate(range.len());
            pattern(seed, physical::DOMAIN, 0, 0, BUFFER_SIZE)[range].to_vec()
        } else {
            pattern(seed, 0x42554646, set, binding, BUFFER_SIZE)
        };
        if kind == BufferKind::Uniform {
            if (set, binding) == (0, 1) {
                if let Some(bytes) = &inputs.cb1 {
                    initial.fill(0);
                    initial[..bytes.len()].copy_from_slice(bytes);
                }
            }
            for b in &inputs.storage {
                if (set, binding) == (0, u32::from(b.bank)) {
                    let base = inputs.range(b.binding).unwrap().start as u64;
                    let offset = usize::from(b.offset);
                    initial[offset..offset + 8].copy_from_slice(&base.to_le_bytes());
                }
            }
        }
        a.host_initial.insert(root, initial);
    }
    Ok(())
}

pub(super) fn capture(
    a: &Analysis,
    inputs: &Inputs,
    seed: u64,
) -> Result<Option<GlobalCapture>, String> {
    let mut ranges = BTreeMap::new();
    if !inputs.storage.is_empty() && !a.physical {
        for (&(set, binding), &(_, kind)) in &a.roots {
            if set == 1 && kind == BufferKind::Storage {
                ranges.insert(
                    binding,
                    inputs
                        .range(binding)
                        .ok_or("unmapped descriptor storage binding")?,
                );
            }
        }
        if ranges.is_empty() {
            return Err("storage capture has no modeled global memory".into());
        }
    }
    Ok((!ranges.is_empty()).then(|| GlobalCapture {
        initial: pattern(seed, physical::DOMAIN, 0, 0, BUFFER_SIZE),
        ranges,
    }))
}
