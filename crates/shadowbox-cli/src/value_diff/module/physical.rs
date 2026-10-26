// Bounded flat-memory observations. Ryujinx resolves global addresses to a
// unique buffer base plus offset (Translation/Optimizations/GlobalToStorage.cs:
// 299-340). Preserve Shadowbox's physical module and supply its existing delta
// instead of substituting storage instructions. Address formation follows
// Ryujinx InstEmitMemory.cs:359-386,457-486,512-532 and project global_memory.rs:140-149,189-196,291-314.
// Delta byte offset: project lib.rs:32.
// SPIR-V fields: spirv.h:104,257,1028,1095,1282,1793,1804,1814,1817,1829-1830,1850,1878,1885,1889,1892;
// operands: spirv.core.grammar.json:679-729,838-859,1320-1329,1404-1418,1457-1469.
use super::{Analysis, BufferKind, Memory, Pointer, Type, BUFFER_SIZE};
use shadowbox::spirv::Op;

pub(super) const DOMAIN: u64 = 0x474c4f42;

#[derive(Clone, Copy)]
pub(super) enum Wide {
    Offset(u64),
    Address(u64),
}

pub(super) fn extension(args: &[u32], name: &[u8]) -> bool {
    let bytes = args
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    bytes.starts_with(name)
        && bytes.get(name.len()).is_some()
        && bytes[name.len()..].iter().all(|b| *b == 0)
}

pub(super) fn initialize(a: &mut Analysis) {
    // ID zero cannot be a module ID. The capture key is outside the accepted
    // source binding range, so this flat allocation cannot alias a descriptor.
    a.storage_classes.insert(0, 5349);
    a.roots.insert((3, 256), (0, BufferKind::Physical));
    a.memory.insert(
        0,
        Memory {
            data: vec![true; BUFFER_SIZE],
            written: vec![false; BUFFER_SIZE],
            writable: true,
        },
    );
}

fn result(a: &mut Analysis, ty: u32, id: u32, value: Wide) -> Result<(), String> {
    a.result(ty, id, vec![true; 8])?;
    a.wide.insert(id, value);
    Ok(())
}

pub(super) fn constant(a: &mut Analysis, op: Op, args: &[u32]) -> Result<(), String> {
    let value = match (op, args) {
        (Op::Constant, [_, _, low, high]) => u64::from(*low) | u64::from(*high) << 32,
        (Op::ConstantNull, [_, _]) => 0,
        _ => return Err("invalid uint64 constant".into()),
    };
    result(a, args[0], args[1], Wide::Offset(value))
}

pub(super) fn load_delta(a: &mut Analysis, args: &[u32]) -> Result<(), String> {
    let p = a.pointer(args[2])?;
    if args.len() != 3
        || p.ty != args[0]
        || a.storage_classes[&p.root] != 9
        || p.offset != shadowbox::PUSH_GLOBAL_DELTA_OFFSET as usize
    {
        return Err("uint64 load must be the global delta push constant".into());
    }
    a.range(p)?;
    result(a, args[0], args[1], Wide::Address(0))
}

pub(super) fn instruction(a: &mut Analysis, op: Op, args: &[u32]) -> Result<(), String> {
    if !a.physical {
        return Err("physical address instruction in logical module".into());
    }
    let value = match op {
        Op::UConvert if args.len() == 3 && matches!(a.ty(args[0])?, Type::Uint64) => {
            let value = *a
                .constants
                .get(&args[2])
                .ok_or("unknown global address offset")?;
            if !matches!(
                a.scalars.get(&a.value_types[&args[2]]),
                Some(super::graphics::Scalar::Uint)
            ) {
                return Err("global address conversion requires uint32".into());
            }
            Wide::Offset(u64::from(value))
        }
        Op::Bitcast if args.len() == 3 => {
            let Some(bits) = a.literals.get(&args[2]) else {
                return Err("unknown global address pair".into());
            };
            if bits.len() != 2 {
                return Err("global address bitcast requires two words".into());
            }
            Wide::Offset(u64::from(bits[0]) | u64::from(bits[1]) << 32)
        }
        Op::CopyObject if args.len() == 3 => *a.wide.get(&args[2]).ok_or("unknown uint64 copy")?,
        Op::IAdd if args.len() == 4 => match (a.wide.get(&args[2]), a.wide.get(&args[3])) {
            (Some(Wide::Offset(x)), Some(Wide::Offset(y))) => Wide::Offset(x.wrapping_add(*y)),
            (Some(Wide::Address(x)), Some(Wide::Offset(y)))
            | (Some(Wide::Offset(y)), Some(Wide::Address(x))) => Wide::Address(x.wrapping_add(*y)),
            _ => return Err("global address must contain exactly one delta".into()),
        },
        Op::ConvertUToPtr if args.len() == 3 => {
            let Type::Pointer(5349, ty) = *a.ty(args[0])? else {
                return Err("invalid physical pointer type".into());
            };
            let Some(Wide::Address(offset)) = a.wide.get(&args[2]) else {
                return Err("physical pointer lacks proved delta".into());
            };
            let p = Pointer {
                root: 0,
                ty,
                offset: usize::try_from(*offset).map_err(|_| "global offset overflow")?,
            };
            a.range(p)?;
            a.pointers.insert(args[1], p);
            return Ok(());
        }
        _ => return Err("unsupported physical address operation".into()),
    };
    result(a, args[0], args[1], value)
}

pub(super) fn alignment(a: &Analysis, pointer: u32, operands: &[u32]) -> Result<(), String> {
    let p = a.pointer(pointer)?;
    if a.physical && a.storage_classes.get(&p.root) == Some(&9) {
        let range = a.range(p)?;
        let delta = shadowbox::PUSH_GLOBAL_DELTA_OFFSET as usize;
        if range.start < delta + 8 && range.end > delta {
            return Err("physical delta read as ordinary data".into());
        }
    }
    if p.root == 0 {
        // Aligned is mandatory for physical loads/stores. The runner checks
        // a 16-byte-aligned device base; larger assertions are unsupported.
        let [2, alignment] = operands else {
            return Err("physical access requires only Aligned".into());
        };
        if !alignment.is_power_of_two()
            || *alignment > 16
            || !p.offset.is_multiple_of(*alignment as usize)
        {
            return Err("unproved physical access alignment".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analysis() -> Analysis {
        let mut a = Analysis {
            physical: true,
            concrete: true,
            ..Analysis::default()
        };
        initialize(&mut a);
        a.types.insert(1, Type::Scalar);
        a.scalars.insert(1, super::super::graphics::Scalar::Uint);
        a.types.insert(2, Type::Uint64);
        a.types.insert(3, Type::Vector(1, 2));
        a.types.insert(4, Type::Pointer(5349, 1));
        a.storage_classes.insert(9, 9);
        a.memory.insert(
            9,
            Memory {
                data: vec![true; 8],
                written: vec![false; 8],
                writable: false,
            },
        );
        a.pointers.insert(
            9,
            Pointer {
                root: 9,
                ty: 2,
                offset: 0,
            },
        );
        load_delta(&mut a, &[2, 10, 9]).unwrap();
        a
    }

    #[test]
    fn extension_strings_accept_zero_word_padding_and_reject_suffixes() {
        let name = b"SPV_KHR_maximal_reconvergence";
        let encode = |bytes: &[u8]| {
            bytes
                .chunks(4)
                .map(|chunk| {
                    let mut word = [0; 4];
                    word[..chunk.len()].copy_from_slice(chunk);
                    u32::from_le_bytes(word)
                })
                .collect::<Vec<_>>()
        };
        let mut bytes = name.to_vec();
        bytes.push(0);
        assert!(extension(&encode(&bytes), name));
        bytes.push(1);
        assert!(!extension(&encode(&bytes), name));
        assert!(!extension(
            &encode(b"SPV_KHR_maximal_reconvergence_extra\0"),
            name
        ));
    }

    #[test]
    fn bounded_offset_requires_one_delta_and_asserted_alignment() {
        let mut a = analysis();
        constant(&mut a, Op::Constant, &[2, 11, 24, 0]).unwrap();
        instruction(&mut a, Op::IAdd, &[2, 12, 10, 11]).unwrap();
        instruction(&mut a, Op::ConvertUToPtr, &[4, 13, 12]).unwrap();
        assert_eq!(a.pointer(13).unwrap().offset, 24);
        alignment(&a, 13, &[2, 8]).unwrap();
        assert!(alignment(&a, 13, &[2, 16]).is_err());
        assert!(alignment(&a, 13, &[]).is_err());
        assert!(instruction(&mut a, Op::ConvertUToPtr, &[4, 14, 11]).is_err());
        assert!(instruction(&mut a, Op::IAdd, &[2, 14, 10, 10]).is_err());
    }

    #[test]
    fn modular_negative_offsets_stay_out_of_bounds() {
        let mut a = analysis();
        constant(&mut a, Op::Constant, &[2, 11, u32::MAX, u32::MAX]).unwrap();
        instruction(&mut a, Op::IAdd, &[2, 12, 10, 11]).unwrap();
        assert!(instruction(&mut a, Op::ConvertUToPtr, &[4, 13, 12]).is_err());
    }

    #[test]
    fn delta_cannot_escape_into_ordinary_words() {
        let mut a = analysis();
        assert!(a
            .instruction(Op::Bitcast, &[3, 11, 10])
            .unwrap_err()
            .contains("ordinary data"));
        // A differently typed push member can alias the delta bytes. Never
        // give that read seeded literals while the GPU receives an address.
        a.pointers.insert(
            15,
            Pointer {
                root: 9,
                ty: 1,
                offset: 4,
            },
        );
        assert!(a
            .instruction(Op::Load, &[1, 16, 15])
            .unwrap_err()
            .contains("ordinary data"));
        instruction(&mut a, Op::CopyObject, &[2, 11, 10]).unwrap();
        instruction(&mut a, Op::ConvertUToPtr, &[4, 12, 11]).unwrap();
        assert_eq!(a.pointer(12).unwrap().offset, 0);
    }
}
