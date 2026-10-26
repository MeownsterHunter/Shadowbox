//! Golden of how every probed instruction encoding is accepted or rejected,
//! and by which stage, so a decoder change shows as a reviewable diff of
//! tests/data/reject-golden.tsv instead of a silent shift.
//!
//! Inputs are generated, not hand-picked: every high16 opcode with the four
//! canonical operand profiles of tools/decoder-regression-probe.rs (PT guard
//! envytools gm107.c:371,409,411; registers :203-218; AST :1883), followed
//! by every row/field sample of tests/data/decoder-samples.tsv (MIT
//! envytools/envydis/gm107.c:1874-2182). Each word is translated in the
//! one-instruction program and the six stage headers that fuzz_validation.rs
//! uses (Mesa nak/sm50.rs:3418-3468 bundle layout; cla097sph.h:37-42 stage
//! types).
//!
//! Per word the golden records two stages:
//! - decode: `reject` when decode_word returns Unknown, otherwise the
//!   InstructionKind name plus any field the decoder left as `Unknown(n)`;
//! - translate: `ok`, or the exact Stopped state (name and text) per
//!   stage header, with the word's own hex replaced by `<word>`. A panic is
//!   recorded by its message; a debug-build overflow panic wraps in
//!   release instead, so the golden is for the default test profile.
//!
//! Consecutive words with the same record are one run, so a change to one
//! encoding splits its run and shows as a few changed lines.
//! Regenerate after an approved change with
//! `SHADOWBOX_UPDATE_REJECTS=1 cargo test --test reject_golden`.

// Rng serves the randomized gates; these inputs are systematic.
#[allow(dead_code)]
#[path = "support/fuzz_inputs.rs"]
mod fuzz_inputs;

use shadowbox::ir::ops;
use std::{
    collections::HashSet,
    fmt::Write,
    fs,
    panic::{self, AssertUnwindSafe},
    path::Path,
};

use shadowbox::{
    decoder::decode_word, interface::ShaderInput, ir::InstructionKind, TranslationOptions,
    Translator,
};

const SAMPLES: &str = include_str!("data/decoder-samples.tsv");
const GOLDEN: &str = include_str!("data/reject-golden.tsv");
/// Labels for fuzz_inputs::fuzz_headers(), in its order. Stage type values:
/// Mesa cla097sph.h:37-42 (1 vertex, 2 TCS, 3 TES, 4 geometry, 5 fragment);
/// the all-zero header is compute (headers.rs:65).
const STAGES: [&str; 6] = ["cs", "fs", "vs", "gs", "tcs", "tes"];
/// The address of the probed word in fuzz_inputs::fuzz_program.
const ADDRESS: u32 = 8;

fn high16_sweep() -> Vec<u64> {
    let mut words = Vec::with_capacity(4 << 16);
    for opcode in 0..=u16::MAX {
        for payload in [
            7_u64 << 16,
            (7 << 16) | (255 << 8) | (255 << 20) | (255 << 39),
            (7 << 16) | (255 << 8) | (255 << 39),
            (7 << 16) | (15 << 12) | (255 << 8) | (255 << 39) | (1 << 32),
        ] {
            words.push((u64::from(opcode) << 48) | payload);
        }
    }
    words
}

fn envytools_samples() -> Vec<u64> {
    SAMPLES
        .lines()
        .filter(|line| !line.starts_with('#') && !line.is_empty())
        .map(|line| u64::from_str_radix(line.split('\t').nth(3).unwrap(), 16).unwrap())
        .collect()
}

/// `reject`, or the kind name followed by each field left as `Unknown(n)`.
fn decode_record(raw: u64) -> String {
    let kind = decode_word([raw as u32, (raw >> 32) as u32], ADDRESS).kind;
    if matches!(kind, InstructionKind::Unknown(ops::Unknown { .. })) {
        return "reject".to_owned();
    }
    let debug = format!("{kind:?}");
    let mut record = debug.split([' ', '{', '(']).next().unwrap().to_owned();
    for (at, _) in debug.match_indices("Unknown(") {
        let before = &debug[..at];
        let field = before[..before.rfind(": ").unwrap()]
            .rsplit([' ', '{', '(', '['])
            .next()
            .unwrap();
        let value = &debug[at..at + debug[at..].find(')').unwrap() + 1];
        write!(record, " {field}={value}").unwrap();
    }
    record
}

/// Outcome per stage header, grouped so identical outcomes print once.
fn translate_record(outcomes: &[String]) -> String {
    if outcomes.iter().all(|o| *o == outcomes[0]) {
        return format!("all: {}", outcomes[0]);
    }
    let mut groups: Vec<(Vec<&str>, &str)> = Vec::new();
    for (stage, outcome) in STAGES.iter().zip(outcomes) {
        match groups.iter_mut().find(|(_, o)| *o == outcome) {
            Some((stages, _)) => stages.push(stage),
            None => groups.push((vec![stage], outcome)),
        }
    }
    groups
        .iter()
        .map(|(stages, outcome)| format!("{}: {outcome}", stages.join(",")))
        .collect::<Vec<_>>()
        .join(" || ")
}

/// One outcome per word and stage header, in word order. A panic is an
/// outcome too (recorded by message, not location); the translator that
/// panicked is replaced so no partial scratch reaches the next word.
fn translate_outcomes(words: &[u64]) -> Vec<Vec<String>> {
    let headers = fuzz_inputs::fuzz_headers();
    let state = fuzz_inputs::graphics_state();
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let mut outcomes = vec![Vec::with_capacity(STAGES.len()); words.len()];
    let per_thread = words.len().div_ceil(threads);
    let hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    std::thread::scope(|scope| {
        for (chunk, slots) in words
            .chunks(per_thread)
            .zip(outcomes.chunks_mut(per_thread))
        {
            let (headers, state) = (&headers, &state);
            scope.spawn(move || {
                let mut translator = Translator::default();
                for (&raw, slot) in chunk.iter().zip(slots) {
                    let code = fuzz_inputs::fuzz_program(raw);
                    for header in headers {
                        let input = ShaderInput {
                            code: &code,
                            program_record: header,
                        };
                        let result = panic::catch_unwind(AssertUnwindSafe(|| {
                            Result::from(translator.translate_with_options_and_graphics_state(
                                shadowbox::QuietRun,
                                &input,
                                TranslationOptions::default(),
                                state,
                            ))
                        }));
                        slot.push(match result {
                            Ok(Ok(_)) => "ok".to_owned(),
                            Ok(Err(stopped)) => {
                                let debug = format!("{stopped:?}");
                                let state =
                                    debug.split_once('(').map_or(&*debug, |(state, _)| state);
                                format!("{state}: {stopped}")
                                    .replace(&format!("{raw:016x}"), "<word>")
                            }
                            Err(payload) => {
                                translator = Translator::default();
                                let message = payload
                                    .downcast_ref::<&str>()
                                    .map(|s| s.to_string())
                                    .or_else(|| payload.downcast_ref::<String>().cloned())
                                    .unwrap_or_default();
                                format!("panic: {message}")
                            }
                        });
                    }
                }
            });
        }
    });
    panic::set_hook(hook);
    outcomes
}

fn write_set(golden: &mut String, name: &str, words: &[u64]) {
    let records = words
        .iter()
        .zip(translate_outcomes(words))
        .map(|(&raw, outcomes)| (decode_record(raw), translate_record(&outcomes)))
        .collect::<Vec<_>>();
    let mut body = String::new();
    let mut runs = 0;
    let mut start = 0;
    for index in 1..=records.len() {
        if index == records.len() || records[index] != records[start] {
            let (decode, translate) = &records[start];
            writeln!(
                body,
                "{start}\t{}\t{:016x}\t{decode}\t{translate}",
                index - start,
                words[start]
            )
            .unwrap();
            runs += 1;
            start = index;
        }
    }
    let rejected = records.iter().filter(|(d, _)| d == "reject").count();
    let field = records
        .iter()
        .filter(|(d, _)| d.contains("=Unknown("))
        .count();
    let failed = records
        .iter()
        .filter(|(d, t)| d != "reject" && t != "all: ok")
        .count();
    writeln!(
        golden,
        "## {name}: {} words, {runs} runs; decode rejects {rejected}, \
         decoded with an Unknown field {field}, decoded but failing in some stage {failed}",
        words.len()
    )
    .unwrap();
    golden.push_str("# index\tcount\tfirst word\tdecode\ttranslate\n");
    golden.push_str(&body);
}

#[test]
fn every_probed_encoding_keeps_its_decode_and_translate_outcome() {
    let mut golden = String::from(
        "# Generated by `SHADOWBOX_UPDATE_REJECTS=1 cargo test --test reject_golden`.\n\
         # Do not edit; see tests/reject_golden.rs for inputs and columns.\n\
         # Stages: cs fs vs gs tcs tes. decode `reject` = decode_word Unknown.\n",
    );
    let high16 = high16_sweep();
    let samples = envytools_samples();
    assert_eq!(high16.len(), 262_144);
    assert_eq!(samples.len(), 36_877);
    write_set(&mut golden, "high16 x 4 operand profiles", &high16);
    write_set(&mut golden, "envytools gm107.c row/field samples", &samples);

    if std::env::var_os("SHADOWBOX_UPDATE_REJECTS").is_some() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/reject-golden.tsv");
        fs::write(path, &golden).unwrap();
        return;
    }
    if golden != GOLDEN {
        let expected = GOLDEN.lines().collect::<HashSet<_>>();
        let actual = golden.lines().collect::<HashSet<_>>();
        let mut diff = String::new();
        for line in GOLDEN.lines().filter(|l| !actual.contains(l)).take(20) {
            writeln!(diff, "- {line}").unwrap();
        }
        for line in golden.lines().filter(|l| !expected.contains(l)).take(20) {
            writeln!(diff, "+ {line}").unwrap();
        }
        panic!("reject outcomes changed; review, then regenerate with SHADOWBOX_UPDATE_REJECTS=1:\n{diff}");
    }
}
