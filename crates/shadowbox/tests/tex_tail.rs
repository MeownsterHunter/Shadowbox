//! Unknown boundaries are tested independently of the probe report.
#[path = "support/spirv_validation.rs"]
mod spirv_validation;
use shadowbox::ir::ops;
use shadowbox::spirv::Op;
use shadowbox::{
    decoder::decode_word,
    ir::{InstructionKind as K, Register, Source},
};
fn decode(raw: u64) -> K {
    decode_word([raw as u32, (raw >> 32) as u32], 0).kind
}
const PT: u64 = 7 << 16;

#[test]
fn lea_fields_and_unimplemented_outputs() {
    // envytools gm107.c:2022,2093,2140; Ryujinx InstDecoders.cs:3581-3627.
    let raw = 0x5bd7u64 << 48 | 3 << 20 | PT | 2 << 8 | 1;
    for shift in 0..32u64 {
        for neg in 0..2u64 {
            assert_eq!(
                decode(raw | shift << 39 | neg << 45),
                K::Lea(ops::Lea {
                    dst: Register(1),
                    lhs: Register(2),
                    rhs: Source::Register(Register(3)),
                    shift: shift as u8,
                    negate: neg != 0,
                })
            );
        }
    }
    for bit in 28..=38 {
        assert!(matches!(
            decode(raw | 1 << bit),
            K::Unknown(ops::Unknown { .. })
        ));
    }
    for bit in [44, 46, 47] {
        assert!(matches!(
            decode(raw | 1 << bit),
            K::Unknown(ops::Unknown { .. })
        ));
    }
    for predicate in 0..7u64 {
        assert!(matches!(
            decode((raw & !(7 << 48)) | predicate << 48),
            K::Unknown(ops::Unknown { .. })
        ));
    }
    // Immediate sign56 is distinct from opcode. Cbuf offset is word-scaled.
    assert!(matches!(
        decode(0x37d7u64 << 48 | 0x7ffff << 20 | PT),
        K::Lea(ops::Lea {
            rhs: Source::Immediate(u32::MAX),
            ..
        })
    ));
    assert!(matches!(
        decode(0x4bd7u64 << 48 | 2 << 34 | 19 << 20 | PT),
        K::Lea(ops::Lea {
            rhs: Source::ConstantBuffer {
                binding: 2,
                offset: 76
            },
            ..
        })
    ));
}

#[test]
fn unnamed_texture_and_surface_fields_stay_unknown() {
    // Mesa sm50.rs:2181-2197,2215-2236,2491-2494; envytools gm107.c:1962,1965,1909.
    let tex = 0xdeb8u64 << 48 | 15 << 31 | 2 << 28 | 20 << 20 | PT | 2 << 8 | 8;
    for bit in 39..=48 {
        assert!(matches!(
            decode(tex | 1 << bit),
            K::Unknown(ops::Unknown { .. })
        ));
    }
    let fetch = 0xdd38u64 << 48 | 15 << 31 | 2 << 28 | 20 << 20 | PT | 2 << 8 | 8;
    for bit in 36..=48 {
        assert!(matches!(
            decode(fetch | 1 << bit),
            K::Unknown(ops::Unknown { .. })
        ));
    }
    let surface = 0xeb20u64 << 48 | 1 << 52 | 40 << 39 | 3 << 33 | 4 << 20 | PT | 2 << 8 | 8;
    assert!(matches!(decode(surface), K::Surface(ops::Surface { .. })));
    for bit in (26..=32).chain(36..=38).chain(47..=48) {
        assert!(
            matches!(decode(surface | 1 << bit), K::Unknown(ops::Unknown { .. })),
            "bit {bit}"
        );
    }
    for base in [tex, fetch] {
        assert!(matches!(
            decode(base & !(15 << 31)),
            K::Unknown(ops::Unknown { .. })
        ));
        assert!(matches!(
            decode((base & !(7 << 28)) | 5 << 28),
            K::Unknown(ops::Unknown { .. })
        ));
    }
    // No clock or hardware register is replaced with Ryujinx's default zero.
    // InstEmitMove.cs:165-167 is a fallback, not register semantics.
    for idx in [1, 2, 3, 0x28, 0x50, 0x51, 0x52, 0x53] {
        assert!(matches!(
            decode(0xf0c8u64 << 48 | idx << 20 | PT),
            K::Unknown(ops::Unknown { .. })
        ));
    }
}

#[test]
fn surface_type_aliases_preserve_distinct_descriptor_fields() {
    use shadowbox::{emitter::emit_unoptimized_with_header, headers, ir::Instruction};
    // Encoding: envytools gm107.c:753-838,1909. Ryujinx ResourceManager.cs:
    // 32,308-325 requires separate typed views for the same handle.
    let surface =
        0xeb20u64 << 48 | 1 << 52 | 1 << 51 | 9 << 36 | 3 << 33 | 4 << 20 | PT | 2 << 8 | 8;
    let header = headers::parse(&[0; 32]).unwrap();
    for (changed, shape) in [
        ((surface & !(7 << 33)) | 5 << 33, (5, 33)),
        ((surface & !(7 << 20)) | 5 << 20, (3, 35)),
        (surface & !(1 << 52), (3, 0)),
    ] {
        let xs = [surface, changed, 0xe300u64 << 48 | PT | 15]
            .map(|raw| decode_word([raw as u32, (raw >> 32) as u32], 0));
        let xs: Vec<_> = xs
            .into_iter()
            .enumerate()
            .map(|(i, x)| Instruction {
                address: i as u32 * 8,
                ..x
            })
            .collect();
        let module = emit_unoptimized_with_header(&xs, Some(&header)).unwrap();
        let views = &module.surface_descriptors;
        assert_eq!(views.len(), 2);
        assert_eq!((views[0].target, views[0].image_format), (3, 33));
        assert_eq!((views[1].target, views[1].image_format), shape);
        assert_eq!(views[0].source, views[1].source);
        assert_ne!(
            (views[0].set, views[0].binding),
            (views[1].set, views[1].binding)
        );
    }
}

#[test]
fn sample_dynamic_offset_boundaries_preserve_constant_forms() {
    use shadowbox::{
        emitter::emit_unoptimized_with_header,
        headers::{self, Stage},
        ir::Instruction,
    };
    // Fields: Mesa sm50.rs:2171-2197; envytools gm107.c:916-923,1970.
    // Offset operands: Ryujinx InstEmitTexture.cs:203-276. Normalization:
    // Translation/Transforms/TexturePass.cs:501-525,591-650.
    // SPH stage field bits 10..14: compute 0, fragment 5 (Mesa
    // sph.rs:204-215).
    let headers = [(Stage::Compute, 0), (Stage::Fragment, 5)].map(|(stage, field)| {
        let mut words = [0; 32];
        words[0] = field << 10;
        (stage, headers::parse(&words).unwrap())
    });
    for bound in [false, true] {
        for dim in 0..5u64 {
            for lod in [0u64, 1, 2, 3, 6, 7] {
                let raw = (if bound { 0xc038u64 } else { 0xdeb8 }) << 48
                    | lod << if bound { 55 } else { 37 }
                    | 1 << if bound { 54 } else { 36 }
                    | if bound { 9 << 36 } else { 0 }
                    | 15 << 31
                    | dim << 28
                    | 20 << 20
                    | PT
                    | 8 << 8
                    | 32;
                let mut xs = Vec::new();
                if !bound {
                    xs.push(Instruction {
                        address: 0,
                        predicate: None,
                        kind: K::Mov(ops::Mov {
                            dst: Register(20),
                            src: Source::Immediate(1),
                            quad_lanes: 15,
                        }),
                    });
                }
                xs.push(decode_word(
                    [raw as u32, (raw >> 32) as u32],
                    xs.len() as u32 * 8,
                ));
                xs.push(Instruction {
                    address: xs.len() as u32 * 8,
                    predicate: None,
                    kind: K::Exit(ops::Exit { cc_test: 15 }),
                });
                for (stage, header) in &headers {
                    let module = emit_unoptimized_with_header(&xs, Some(header))
                        .unwrap_or_else(|e| panic!("{raw:016x}, {stage:?}: {e}"));
                    let check = spirv_validation::validate(&module.words);
                    assert!(
                        check.status.success(),
                        "{}",
                        String::from_utf8_lossy(&check.stderr)
                    );
                }
                // Every previously supported literal offset retains its path,
                // including bias and explicit LOD modes.
                let offset = 20 + u8::from(!bound) + u8::from(lod >= 2);
                xs.insert(
                    usize::from(!bound),
                    Instruction {
                        address: 0,
                        predicate: None,
                        kind: K::Mov(ops::Mov {
                            dst: Register(offset),
                            src: Source::Immediate(0x1f),
                            quad_lanes: 15,
                        }),
                    },
                );
                for (index, instruction) in xs.iter_mut().enumerate() {
                    instruction.address = index as u32 * 8;
                }
                for (_, header) in &headers {
                    let module = emit_unoptimized_with_header(&xs, Some(header)).unwrap();
                    let mut at = 5;
                    while at < module.words.len() {
                        let word = module.words[at];
                        assert_ne!(
                            Op::of(word),
                            Op::ImageQuerySizeLod,
                            "literal offset must fold"
                        );
                        at += (word >> 16) as usize;
                    }
                }
            }
        }
    }
}
