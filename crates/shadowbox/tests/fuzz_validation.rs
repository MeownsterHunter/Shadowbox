//! Modifier fuzzing: every encoding the translator accepts must produce
//! valid SPIR-V, and no encoding may panic.
//!
//! The family test uses one plain form per opcode, so wrong SPIR-V behind a
//! modifier bit (abs, negate, saturate, sizes, predicate outputs) passes it.
//! This test keeps the 16-bit opcode prefix of every instruction slot and
//! fills the other 48 bits pseudo-randomly, in the same bundle layout as
//! family_validation.rs (decoder.rs:1106-1112). Each translation that
//! succeeds is checked with spirv-val; an Err is fine (the form is Unknown).
//!
//! Slow, so ignored by default: `cargo test --test fuzz_validation --
//! --ignored`. SHADOWBOX_FUZZ_SAMPLES sets the samples per prefix
//! (default 4) and SHADOWBOX_FUZZ_SEED the seed.

#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use std::{
    collections::HashSet,
    panic::{self, AssertUnwindSafe},
    process::Command,
};

use shadowbox::{interface::ShaderInput, translate_with_graphics_state};

#[path = "support/fuzz_inputs.rs"]
mod fuzz_inputs;

#[test]
#[ignore]
fn accepted_encodings_validate_and_nothing_panics() {
    if !Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
    {
        eprintln!("fuzz validation skipped: spirv-val is not on PATH");
        return;
    }
    let samples: u32 = std::env::var("SHADOWBOX_FUZZ_SAMPLES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let seed: u64 = std::env::var("SHADOWBOX_FUZZ_SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0x5eed_5eed_5eed_5eed);
    let mut rng = fuzz_inputs::Rng(seed | 1);
    let headers = fuzz_inputs::fuzz_headers();
    let state = fuzz_inputs::graphics_state();

    panic::set_hook(Box::new(|_| {}));
    let mut panics = Vec::new();
    let mut seen = HashSet::new();
    let mut accepted = 0_u32;
    let mut invalid = Vec::new();
    for prefix in 0..=0xffff_u64 {
        for _ in 0..samples {
            let raw = prefix << 48 | (rng.next() & 0xffff_ffff_ffff);
            for (h, header) in headers.iter().enumerate() {
                let code = fuzz_inputs::fuzz_program(raw);
                let result = panic::catch_unwind(AssertUnwindSafe(|| {
                    Result::from(translate_with_graphics_state(
                        shadowbox::QuietRun,
                        &ShaderInput {
                            code: &code,
                            program_record: header,
                        },
                        &state,
                    ))
                }));
                match result {
                    Err(_) => panics.push((raw, h)),
                    Ok(Ok(output)) => {
                        accepted += 1;
                        if seen.insert(output.spirv.clone()) {
                            let check = spirv_validation::validate(&output.spirv);
                            if !check.status.success() {
                                let message = String::from_utf8_lossy(&check.stderr);
                                invalid.push(format!(
                                    "{raw:016x} stage {h}: {}",
                                    message
                                        .lines()
                                        .next()
                                        .unwrap_or("")
                                        .chars()
                                        .take(160)
                                        .collect::<String>()
                                ));
                            }
                        }
                    }
                    Ok(Err(_)) => {}
                }
            }
        }
    }
    let _ = panic::take_hook();
    eprintln!(
        "fuzz: {} accepted, {} distinct validated, {} invalid, {} panics",
        accepted,
        seen.len(),
        invalid.len(),
        panics.len()
    );
    for line in invalid.iter().take(40) {
        eprintln!("INVALID {line}");
    }
    for (raw, h) in panics.iter().take(40) {
        eprintln!("PANIC {raw:016x} stage {h}");
    }
    assert!(invalid.is_empty() && panics.is_empty());
}

#[path = "support/control_flow_inputs.rs"]
mod control_flow_inputs;
#[path = "support/control_flow_validation.rs"]
mod control_flow_validation;

#[test]
#[ignore = "randomized structured control-flow gate"]
fn randomized_control_flow_shapes_validate() {
    let samples = std::env::var("SHADOWBOX_CONTROL_FLOW_SAMPLES")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(256);
    let mut rng = fuzz_inputs::Rng(0x5eed_c0de);
    let headers = fuzz_inputs::fuzz_headers();
    let state = fuzz_inputs::graphics_state();
    for _ in 0..samples {
        let seed = rng.next();
        for (name, code) in control_flow_inputs::programs(seed) {
            for (stage, header) in headers.iter().enumerate() {
                let case = format!("{name} seed {seed:016x} stage {stage}");
                control_flow_validation::check(&case, &code, header, &state);
            }
        }
    }
    eprintln!(
        "control-flow fuzz: {} programs, raw and optimized validated",
        samples * control_flow_inputs::programs(0).len() * headers.len()
    );
}
