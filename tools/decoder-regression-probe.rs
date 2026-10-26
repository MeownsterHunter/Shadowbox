//! Run only against permitted project source snapshots, not private shaders.
#[path = "support.rs"]
mod fixtures;
use shadowbox::{decoder::decode_word, interface::ShaderInput, ir::InstructionKind, translate};
use std::collections::BTreeMap;
fn main() {
    if let Some(samples) = std::env::args().nth(1) {
        let mut accepted = BTreeMap::new();
        let mut probe = |raw: u64| {
            let kind = decode_word([raw as u32, (raw >> 32) as u32], 0).kind;
            if !matches!(kind, InstructionKind::Unknown { .. }) {
                let name = format!("{kind:?}")
                    .split([' ', '{'])
                    .next()
                    .unwrap()
                    .to_owned();
                accepted.insert(raw, name);
            }
        };
        // Exhaust every high16 opcode with four canonical operand profiles,
        // plus all cited row/field samples from envytools gm107.c:1874-2182.
        // PT:371,409,411; registers:203-218; AST:1883. MOV lane mask:
        // Mesa nak/sm50.rs:1936,1941; AST bit32:2895. Payloads are
        // probes, not a claim that every arbitrary field combination is legal.
        for opcode in 0..=u16::MAX {
            for payload in [
                7 << 16,
                (7 << 16) | (255 << 8) | (255 << 20) | (255 << 39),
                (7 << 16) | (255 << 8) | (255 << 39),
                (7 << 16) | (15 << 12) | (255 << 8) | (255 << 39) | (1 << 32),
            ] {
                probe((u64::from(opcode) << 48) | payload);
            }
        }
        for line in std::fs::read_to_string(samples)
            .unwrap()
            .lines()
            .filter(|s| !s.starts_with('#'))
        {
            if let Some(raw) = line.split('\t').nth(3) {
                probe(u64::from_str_radix(raw, 16).unwrap());
            }
        }
        for (raw, kind) in accepted {
            println!("{raw:016x}\t{kind}");
        }
    } else {
        let header = fixtures::header();
        for (name, instructions) in fixtures::cases() {
            let code = fixtures::program(&instructions);
            match translate(&ShaderInput {
                code: &code,
                program_record: &header,
            }) {
                Ok(_) => println!("{name}\tPASS"),
                Err(error) => println!("{name}\tFAIL {error}"),
            }
        }
    }
}
