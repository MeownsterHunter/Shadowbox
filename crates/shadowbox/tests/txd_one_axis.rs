#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{GradOffset, Instruction, InstructionKind, Predicate, Register, TexOp},
};

struct Form {
    name: &'static str,
    raw: u64,
    dim: u8,
    mask: u8,
    offset: bool,
    nodep: bool,
    binding: u16,
    /// Packed A word after the coordinate: layer 0..15, offset 16..19.
    packed: Option<u32>,
}

fn forms() -> [Form; 6] {
    [
        Form {
            name: "TXD 1D 0xf",
            raw: 0xde38_0017_8107_0820,
            dim: 0,
            mask: 0xf,
            offset: false,
            nodep: false,
            binding: 1,
            packed: None,
        },
        Form {
            name: "TXD.NODEP 1D 0x5",
            raw: 0xde3a_0022_8107_0820,
            dim: 0,
            mask: 0x5,
            offset: false,
            nodep: true,
            binding: 2,
            packed: None,
        },
        Form {
            name: "TXD ARRAY_1D 0xf",
            raw: 0xde38_0037_9107_0820,
            dim: 1,
            mask: 0xf,
            offset: false,
            nodep: false,
            binding: 3,
            packed: None,
        },
        Form {
            name: "TXD.AOFFI 1D 0xf",
            raw: 0xde38_001f_8107_0820,
            dim: 0,
            mask: 0xf,
            offset: true,
            nodep: false,
            binding: 1,
            packed: Some(0x0001_0000),
        },
        Form {
            name: "TXD.AOFFI ARRAY_1D 0xb",
            raw: 0xde38_004d_9107_0820,
            dim: 1,
            mask: 0xb,
            offset: true,
            nodep: false,
            binding: 4,
            packed: Some(0x000f_0002),
        },
        Form {
            name: "@!P2 TXD ARRAY_1D 0x2",
            raw: 0xde38_0051_110a_0820,
            dim: 1,
            mask: 0x2,
            offset: false,
            nodep: false,
            binding: 5,
            packed: None,
        },
    ]
}

fn decode(raw: u64, address: u32) -> Instruction {
    decode_word([raw as u32, (raw >> 32) as u32], address)
}

/// MOV32I R9 = packed (Mesa sm50.rs:1938-1941), then the texture, then EXIT.
fn program(form: &Form, guard: Option<Predicate>) -> Vec<Instruction> {
    let mut xs = Vec::new();
    if let Some(packed) = form.packed {
        xs.push(decode(
            0x0100_u64 << 48 | u64::from(packed) << 20 | 15 << 12 | 7 << 16 | 9,
            8,
        ));
    }
    let mut texture = decode(form.raw, 16);
    if guard.is_some() {
        texture.predicate = guard;
    }
    xs.push(texture);
    xs.push(Instruction {
        address: 24,
        predicate: None,
        kind: InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
    });
    xs
}

#[test]
fn six_txd_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for form in forms() {
        let name = form.name;
        let instruction = decode(form.raw, 16);
        assert_eq!(
            instruction.kind,
            InstructionKind::Texture(ops::Texture {
                op: TexOp::Grad {
                    gradients: Register(16),
                    offset: [
                        GradOffset::Absent,
                        GradOffset::Packed {
                            word: Register(9),
                            axes: 1,
                        },
                    ][usize::from(form.offset)],
                },
                dst: Register(32),
                src0: Register(8),
                src1: Register(16),
                dim: form.dim,
                mask: form.mask,
                offset: form.offset,
                scalar: false,
                packed_f16: false,
                lod_clamp: false,
                multisample: false,
                depth_compare: false,
                nodep: form.nodep,
                scalar_dst: None,
                bound_handle: Some(form.binding),
            }),
            "{name}"
        );
        for guard in [
            None,
            Some(Predicate {
                register: 1,
                inverted: true,
            }),
        ] {
            let mut module =
                emit_unoptimized_with_header(&program(&form, guard), Some(&header)).unwrap();
            let raw_check = spirv_validation::validate(&module.words);
            assert!(
                raw_check.status.success(),
                "{name} raw: {}",
                String::from_utf8_lossy(&raw_check.stderr)
            );
            module.optimize();
            let optimized_check = spirv_validation::validate(&module.words);
            assert!(
                optimized_check.status.success(),
                "{name} optimized: {}",
                String::from_utf8_lossy(&optimized_check.stderr)
            );
        }
    }
}
