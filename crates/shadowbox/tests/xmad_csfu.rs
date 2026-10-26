#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    emitter::emit_unoptimized_with_header,
    headers,
    ir::{Instruction, InstructionKind, Predicate, Register, Source, XmadCop},
};

struct Form {
    name: &'static str,
    raw: u64,
    predicate: Option<Predicate>,
    b: Source,
    c: Source,
    signed: [bool; 2],
    high: [bool; 2],
    psl: bool,
}

const R2: Source = Source::Register(Register(2));
const R4: Source = Source::Register(Register(4));
const P1: Option<Predicate> = Some(Predicate {
    register: 1,
    inverted: false,
});
const NOT_P0: Option<Predicate> = Some(Predicate {
    register: 0,
    inverted: true,
});

fn forms() -> [Form; 6] {
    [
        Form {
            name: "register S16 x S16",
            raw: 0x5b0f_0200_0027_0103,
            predicate: None,
            b: R2,
            c: R4,
            signed: [true, true],
            high: [false, false],
            psl: false,
        },
        Form {
            name: "register S16 a.H1 x U16 b.H1, @P1",
            raw: 0x5b2d_0208_0021_0103,
            predicate: P1,
            b: R2,
            c: R4,
            signed: [true, false],
            high: [true, true],
            psl: false,
        },
        Form {
            name: "register U16 x S16 PSL, @!P0",
            raw: 0x5b0e_0210_0028_0103,
            predicate: NOT_P0,
            b: R2,
            c: R4,
            signed: [false, true],
            high: [false, false],
            psl: true,
        },
        Form {
            name: "immediate S16 x S16 0x8001",
            raw: 0x360f_0208_0017_0103,
            predicate: None,
            b: Source::Immediate(0x8001),
            c: R4,
            signed: [true, true],
            high: [false, false],
            psl: false,
        },
        Form {
            name: "cbuf b c[0][8].H1, S16 x S16",
            raw: 0x4e1f_0200_0027_0103,
            predicate: None,
            b: Source::ConstantBuffer {
                binding: 0,
                offset: 8,
            },
            c: R4,
            signed: [true, true],
            high: [false, true],
            psl: false,
        },
        Form {
            name: "cbuf c c[0][4], S16 a.H1 x S16, @!P0",
            raw: 0x512f_0200_0018_0103,
            predicate: NOT_P0,
            b: R4,
            c: Source::ConstantBuffer {
                binding: 0,
                offset: 4,
            },
            signed: [true, true],
            high: [true, false],
            psl: false,
        },
    ]
}

fn decode(raw: u64) -> Instruction {
    decode_word([raw as u32, (raw >> 32) as u32], 8)
}

#[test]
fn six_xmad_csfu_forms_decode_and_validate_raw_and_optimized() {
    let header = headers::parse(&[0; 32]).unwrap();
    for form in forms() {
        let name = form.name;
        let instruction = decode(form.raw);
        assert_eq!(instruction.predicate, form.predicate, "{name}");
        assert_eq!(
            instruction.kind,
            InstructionKind::Xmad(ops::Xmad {
                dst: Register(3),
                a: Register(1),
                b: form.b,
                c: form.c,
                signed: form.signed,
                high: form.high,
                psl: form.psl,
                mrg: false,
                cmode: XmadCop::Csfu,
                cc: false,
            }),
            "{name}"
        );
        let mut module = emit_unoptimized_with_header(
            &[
                instruction,
                Instruction {
                    address: 16,
                    predicate: None,
                    kind: InstructionKind::Exit(ops::Exit { cc_test: 0xf }),
                },
            ],
            Some(&header),
        )
        .unwrap();
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
