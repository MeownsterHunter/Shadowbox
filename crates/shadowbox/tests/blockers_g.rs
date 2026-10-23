//! Synthetic instruction words only. Encoding: Mesa sm50.rs:3107-3110,
//! 3026-3028,3418-3428; envytools gm107.c:2020,1391-1396,1925,2171.
//! Semantics: Ryujinx Decoders/Decoder.cs:501-517 and
//! Instructions/InstEmitIntegerLogical.cs:118-163.
#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use shadowbox::ir::ops;
use shadowbox::{
    decoder::{decode_stream, decode_word},
    interface::{ShaderInput, Stopped},
    ir::{InstructionKind, PredicateOp, PredicateOutput, Register, Source},
    translate,
};

const PT: u64 = 7 << 16;
const EXIT: u64 = 0xe300_u64 << 48 | PT | 0xf;

fn code(slots: [u64; 3]) -> Vec<u8> {
    [0, slots[0], slots[1], slots[2]]
        .into_iter()
        .flat_map(u64::to_le_bytes)
        .collect()
}

fn translate_slots(slots: [u64; 3]) -> Result<(), Stopped> {
    Result::from(translate(
        shadowbox::QuietRun,
        &ShaderInput {
            code: &code(slots),
            program_record: &[],
        },
    ))
    .map(|module| {
        let output = spirv_validation::validate(&module.spirv);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    })
}

#[test]
fn zero_words_are_invalid_when_reachable_and_padding_when_proved_dead() {
    assert!(translate_slots([0, EXIT, 0]).is_err());
    assert!(translate_slots([EXIT, 0, 0]).is_ok());
    // Reachable discovery omits dead data. Direct decoding names padding
    // (envytools gm107.c:2171; Ryujinx Decoder.cs:501-515).
    let words = code([EXIT, 0, 0])
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| u32::from_le_bytes(*w))
        .collect::<Vec<_>>();
    let decoded = decode_stream(shadowbox::QuietRun, &words);
    assert_eq!(decoded.len(), 1);
    assert_eq!(
        decode_word([0, 0], 16).kind,
        InstructionKind::Padding(ops::Padding)
    );
    // P0 can be false. EXIT.CC.F cannot prove that following words are dead.
    assert!(translate_slots([EXIT & !PT, 0, 0]).is_err());
    assert!(translate_slots([EXIT & !0xf, 0, 0]).is_err());
    assert!(translate_slots([EXIT | 1 << 5, 0, 0]).is_err());
    // Dead nonzero data is omitted, executable data still rejects.
    assert!(translate_slots([EXIT, 1, 0]).is_ok());
    assert!(translate_slots([1, EXIT, 0]).is_err());
}

#[test]
fn zero_padding_proof_tracks_branch_targets_and_fallthrough() {
    let bra = 0xe240_u64 << 48 | PT | 0xf;
    assert!(translate_slots([bra | 8 << 20, 0, EXIT]).is_ok());
    assert!(translate_slots([bra, 0, EXIT]).is_err());
    assert!(translate_slots([(bra | 8 << 20) & !PT, 0, EXIT]).is_err());
    // BRA.CC.NEU has a possible fallthrough even with PT.
    assert!(translate_slots([(bra & !0xf) | 0xd | 8 << 20, 0, EXIT]).is_err());
    // The proof permits dead data after a self loop, without decoding it.
    assert!(translate_slots([bra | (0xfffff8_u64 << 20), 0, 0]).is_ok());
}

#[test]
fn register_lop3_fields_have_exact_ir_and_reject_unimplemented_forms() {
    let raw =
        0x5be1_u64 << 48 | PT | 9 | 2 << 8 | 3 << 20 | 0x96 << 28 | 3 << 36 | 4 << 39 | 1 << 47;
    let decoded = decode_word([raw as u32, (raw >> 32) as u32], 8);
    assert_eq!(
        decoded.kind,
        InstructionKind::Lop3(ops::Lop3 {
            dst: Register(9),
            lhs: Register(2),
            rhs: Source::Register(Register(3)),
            third: Register(4),
            lut: 0x96,
            predicate_output: Some(PredicateOutput {
                op: PredicateOp::Nz,
                dst: 1,
            }),
            extended: false,
            cc: true,
        })
    );
    assert!(decoded.kind.may_write_predicate());
    let extended = raw | 1 << 38;
    assert!(matches!(
        decode_word([extended as u32, (extended >> 32) as u32], 8).kind,
        InstructionKind::Lop3(ops::Lop3 { extended: true, .. })
    ));
    for rejected in [
        (extended & !(1 << 47)),
        raw | 1 << 51,
        (raw & !(7 << 48)) | 7 << 48,
    ] {
        assert_eq!(
            decode_word([rejected as u32, (rejected >> 32) as u32], 8).kind,
            InstructionKind::Unknown(ops::Unknown {
                raw: [rejected as u32, (rejected >> 32) as u32]
            })
        );
    }
}
