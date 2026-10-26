#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{Instruction, InstructionKind, InterpFreq, InterpLoc, Predicate, Register, Source},
};

struct Form {
    name: &'static str,
    raw: u64,
    dst: u8,
    addr: u16,
    inv_w: u8,
    freq: InterpFreq,
}

// envytools gm107.c:1953 and Ryujinx InstDecoders.cs:3265-3282 (InstIpa).
fn forms() -> [Form; 6] {
    [
        Form {
            name: "PASS.SAT R0, a[0x80]",
            raw: 0xe00b_ff88_0ff7_ff00,
            dst: 0,
            addr: 0x80,
            inv_w: 255,
            freq: InterpFreq::Pass,
        },
        Form {
            name: "CONSTANT.SAT R3, a[0x8c]",
            raw: 0xe08b_ff88_cff7_ff03,
            dst: 3,
            addr: 0x8c,
            inv_w: 255,
            freq: InterpFreq::Constant,
        },
        Form {
            name: "PASS.SAT R0, a[0x70]",
            raw: 0xe00b_ff87_0ff7_ff00,
            dst: 0,
            addr: 0x70,
            inv_w: 255,
            freq: InterpFreq::Pass,
        },
        Form {
            name: "PASS.SAT R3, a[0x7c]",
            raw: 0xe00b_ff87_cff7_ff03,
            dst: 3,
            addr: 0x7c,
            inv_w: 255,
            freq: InterpFreq::Pass,
        },
        Form {
            name: "MUL.SAT R1, a[0x74], R20",
            raw: 0xe04b_ff87_4147_ff01,
            dst: 1,
            addr: 0x74,
            inv_w: 20,
            freq: InterpFreq::PassMulW,
        },
        Form {
            name: "MUL.SAT R2, a[0x78], R21",
            raw: 0xe04b_ff87_8157_ff02,
            dst: 2,
            addr: 0x78,
            inv_w: 21,
            freq: InterpFreq::PassMulW,
        },
    ]
}

fn decode(raw: u64) -> Instruction {
    decode_word([raw as u32, (raw >> 32) as u32], 8)
}

// Fragment program header: Mesa nak/sph.rs:204-215 (kind 5 at bit 10),
// generic input map at word 6, interpolation map words 6..7 at bit 192
// (sph.rs:138-143; Ryujinx ShaderHeader.cs:30-48), color target 0 RGBA.
fn fragment_header(interpolation: u32) -> headers::ShaderHeader {
    let mut h = [0u32; 32];
    h[0] = 5 << 10;
    h[2] = 3 << 24;
    h[3] = 7 << 24;
    h[4] = 6;
    h[6] = interpolation;
    h[13] = 255 << 16;
    h[18] = 15;
    headers::parse(&h).unwrap()
}

#[test]
fn six_ipa_saturate_forms_decode_and_validate_raw_and_optimized() {
    for form in forms() {
        let name = form.name;
        let instruction = decode(form.raw);
        let expected = (
            Register(form.dst),
            form.addr,
            Source::Register(Register(form.inv_w)),
            Source::Register(Register(255)),
            InterpLoc::Default,
            form.freq,
            true,
        );
        let actual = match instruction.kind {
            InstructionKind::IpaPass(ops::IpaPass {
                dst,
                addr,
                inv_w,
                offset,
                loc,
                freq,
                saturate,
            })
            | InstructionKind::IpaPassMulW(ops::IpaPassMulW {
                dst,
                addr,
                inv_w,
                offset,
                loc,
                freq,
                saturate,
            }) => (dst, addr, inv_w, offset, loc, freq, saturate),
            other => panic!("{name}: {other:?}"),
        };
        assert_eq!(actual, expected, "{name}");
        assert_eq!(
            matches!(
                instruction.kind,
                InstructionKind::IpaPassMulW(ops::IpaPassMulW { .. })
            ),
            form.freq == InterpFreq::PassMulW,
            "{name}"
        );
        // Flat map for CONSTANT, perspective otherwise.
        let header = fragment_header(if form.freq == InterpFreq::Constant {
            0x55
        } else {
            0xaa
        });
        for predicate in [
            None,
            Some(Predicate {
                register: 1,
                inverted: true,
            }),
        ] {
            let mut module = emit_unoptimized_with_header(
                &[
                    Instruction {
                        predicate,
                        ..instruction
                    },
                    Instruction {
                        address: 16,
                        predicate: None,
                        kind: InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
                    },
                ],
                Some(&header),
            )
            .unwrap_or_else(|error| panic!("{name}: {error:?}"));
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
