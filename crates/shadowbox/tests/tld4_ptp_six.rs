#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{GatherOffsets, Instruction, InstructionKind, Predicate, Register, TexOp},
};

struct Form {
    name: &'static str,
    raw: u64,
    component: u8,
    dim: u8,
    mask: u8,
    /// Bound texture index 36..48; bindless reads its handle from B.
    binding: Option<u16>,
}

fn forms() -> [Form; 6] {
    [
        Form {
            name: "TLD4.B.B.PTP 2D 0xf",
            raw: 0xdef8_00a7_a107_0820,
            component: 2,
            dim: 2,
            mask: 0xf,
            binding: None,
        },
        Form {
            name: "TLD4.R.B.PTP ARRAY_2D 0x3",
            raw: 0xdef8_0021_b107_0820,
            component: 0,
            dim: 3,
            mask: 0x3,
            binding: None,
        },
        Form {
            name: "TLD4.R.PTP 0x1 2D 0x5",
            raw: 0xc8b8_0012_a107_0820,
            component: 0,
            dim: 2,
            mask: 0x5,
            binding: Some(1),
        },
        Form {
            name: "@!P2 TLD4.G.PTP 0x3 ARRAY_2D 0xf",
            raw: 0xc9b8_0037_b10a_0820,
            component: 1,
            dim: 3,
            mask: 0xf,
            binding: Some(3),
        },
        Form {
            name: "@P3 TLD4.G.B.PTP 2D 0x8",
            raw: 0xdef8_0064_2103_0820,
            component: 1,
            dim: 2,
            mask: 0x8,
            binding: None,
        },
        Form {
            name: "TLD4.A.PTP 0x2 ARRAY_2D 0xa",
            raw: 0xcbb8_0025_3107_0820,
            component: 3,
            dim: 3,
            mask: 0xa,
            binding: Some(2),
        },
    ]
}

fn decode(raw: u64, address: u32) -> Instruction {
    decode_word([raw as u32, (raw >> 32) as u32], address)
}

/// MOV32I R16..R18 (Mesa sm50.rs:1938-1941): a bindless handle or the first
/// packed offset word, then the offset words. Then TLD4, then EXIT.
fn program(form: &Form, guard: Option<Predicate>) -> Vec<Instruction> {
    let mut xs = [(16, 1), (17, 0x01_3f_02_3e), (18, 0x03_3d_04_3c)]
        .into_iter()
        .enumerate()
        .map(|(i, (dst, value)): (usize, (u64, u64))| {
            decode(
                0x0100_u64 << 48 | value << 20 | 15 << 12 | 7 << 16 | dst,
                i as u32 * 8,
            )
        })
        .collect::<Vec<_>>();
    let mut texture = decode(form.raw, 24);
    if guard.is_some() {
        texture.predicate = guard;
    }
    xs.push(texture);
    xs.push(Instruction {
        address: 32,
        predicate: None,
        kind: InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
    });
    xs
}

#[test]
fn six_tld4_ptp_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for form in forms() {
        let name = form.name;
        let instruction = decode(form.raw, 24);
        assert_eq!(
            instruction.kind,
            InstructionKind::Texture(ops::Texture {
                op: TexOp::Gather {
                    component: form.component,
                    offsets: GatherOffsets::Four,
                },
                dst: Register(32),
                src0: Register(8),
                src1: Register(16),
                dim: form.dim,
                mask: form.mask,
                offset: false,
                scalar: false,
                packed_f16: false,
                lod_clamp: false,
                multisample: false,
                depth_compare: false,
                nodep: false,
                scalar_dst: None,
                bound_handle: form.binding,
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
