//! I2I admission is full-word exact: bits envytools gm107.c:1993,2067,2113
//! leaves unnamed (and Ryujinx InstI2iR/I/C, InstDecoders.cs:2731-2781,
//! never reads) must be zero, else the word stays Unknown.
use shadowbox::ir::ops;
use shadowbox::{
    decoder::decode_word,
    ir::{Conversion, InstructionKind, Register, Source},
};

fn kind(raw: u64) -> InstructionKind {
    decode_word([raw as u32, (raw >> 32) as u32], 0).kind
}

// R1 = I2I.S32.S16 byte2 (H1) of the source, neg45, sat50, abs49 (via 0x5ce2).
const REGISTER: u64 = 0x5ce6_2400_0027_3601;
const CBUF: u64 = 0x4ce4_2400_0047_3601;
const IMMEDIATE: u64 = 0x38e4_2400_0047_3601;

#[test]
fn canonical_words_decode_unchanged() {
    for (raw, src) in [
        (REGISTER, Some(Source::Register(Register(2)))),
        (CBUF, None),
        (IMMEDIATE, None),
    ] {
        let InstructionKind::Convert(ops::Convert {
            kind:
                Conversion::I2I {
                    src_signed,
                    dst_signed,
                    src_size,
                    dst_size,
                    byte,
                    abs,
                    negate,
                    saturate,
                    cc,
                },
            dst,
            src: decoded,
        }) = kind(raw)
        else {
            panic!("{raw:016x}: not I2I");
        };
        assert_eq!(
            (src_signed, dst_signed, src_size, dst_size, byte),
            (true, true, 1, 2, 2),
            "{raw:016x}"
        );
        assert_eq!(
            (abs, negate, saturate, cc),
            (raw >> 49 & 1 == 1, true, true, false),
            "{raw:016x}"
        );
        assert_eq!(dst, Register(1));
        if let Some(src) = src {
            assert_eq!(decoded, src);
        }
    }
}

#[test]
fn unnamed_bits_stay_unknown() {
    let common = [14, 15, 39, 40, 43, 44, 46, 48];
    for (base, extra) in [(REGISTER, 28..39), (CBUF, 0..0), (IMMEDIATE, 0..0)] {
        assert!(matches!(
            kind(base),
            InstructionKind::Convert(ops::Convert { .. })
        ));
        for bit in common.into_iter().chain(extra) {
            let raw = base | 1 << bit;
            assert!(
                matches!(kind(raw), InstructionKind::Unknown(ops::Unknown { .. })),
                "{raw:016x}: bit {bit} admitted"
            );
        }
    }
}
