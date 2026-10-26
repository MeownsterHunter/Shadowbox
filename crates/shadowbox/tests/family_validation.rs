//! Decoder-family smoke coverage.
//!
//! Each decoder arm cites the permitted encoding references. Common
//! register operands are envytools envydis/gm107.c:1990,1994,2017,2021;
//! predicate fields are gm107.c:363-364,371,408-411. Bundle control words
//! and three instruction slots follow Mesa nak/sm50.rs:3418-3428,3438-3468.
//! Implemented forms must translate; other smoke cases can remain Unknown.

#[path = "support/spirv_validation.rs"]
mod spirv_validation;

use std::{fs, process::Command};

use shadowbox::{interface::ShaderInput, translate, translate_with_options_and_graphics_state};

#[path = "support/families.rs"]
mod families;
use families::{Family, FAMILIES};

#[path = "support/family_inputs.rs"]
mod family_inputs;
use family_inputs::{compute_header, family_context, program};

#[test]
fn every_decoder_family_is_validated_or_named_unknown() {
    let names = FAMILIES
        .iter()
        .map(|family| family.name)
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(names.len(), FAMILIES.len(), "duplicate family labels");
    let spirv_val = match Command::new("spirv-val").arg("--version").output() {
        Ok(output) if output.status.success() => true,
        _ => {
            eprintln!("family validation skipped: spirv-val is not on PATH");
            return;
        }
    };
    assert!(spirv_val);

    let output_dir = std::env::temp_dir().join(format!("shadowbox-family-{}", std::process::id()));
    fs::create_dir_all(&output_dir).unwrap();
    let mut results = Vec::new();

    for family in FAMILIES {
        let code = program(*family);
        let (header, state) = family_context(*family);
        let result = Result::from(translate_with_options_and_graphics_state(
            shadowbox::QuietRun,
            &ShaderInput {
                code: &code,
                program_record: &header,
            },
            shadowbox::TranslationOptions {
                float64: shadowbox::HostFloat64::Supported,
                ..Default::default()
            },
            &state,
        ));
        match result {
            Ok(output) => {
                assert!(
                    ![
                        "PSET_CC_UNKNOWN",
                        "TMML_NDV_UNKNOWN",
                        "IPA_MUL_SAT_OFFSET_UNKNOWN",
                        "MUFU.RESERVED",
                        "IADD.C.SAT",
                        "IADD.C.PLUS_ONE",
                        "ISCADD.C.PLUS_ONE",
                        "STG_RESERVED",
                        "ATOM_INC_UNKNOWN_S32",
                        "RED_INC_UNKNOWN_S32",
                        "ATOM_DEC_UNKNOWN_S32",
                        "RED_DEC_UNKNOWN_S32"
                    ]
                    .contains(&family.name),
                    "{} must remain Unknown",
                    family.name
                );
                let path = output_dir.join(format!("{}.spv", family.name));
                fs::write(
                    &path,
                    output
                        .spirv
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect::<Vec<_>>(),
                )
                .unwrap();
                let check = Command::new("spirv-val")
                    .args(["--target-env", "vulkan1.2"])
                    .arg(&path)
                    .output()
                    .unwrap();
                assert!(
                    check.status.success(),
                    "{}: {}",
                    family.name,
                    String::from_utf8_lossy(&check.stderr)
                );
                results.push((family.name, "translated".to_string()));
            }
            Err(error) => {
                assert!(
                    !family.required
                        && !family.name.starts_with("I2I_")
                        && !family.name.starts_with("F2I_")
                        && !family.name.starts_with("I2F_"),
                    "{} must translate: {error}",
                    family.name
                );
                let reason = error.to_string();
                assert!(!family.required, "{} must translate: {reason}", family.name);
                assert!(
                    reason.starts_with("unsupported instruction:"),
                    "{}: {reason}",
                    family.name
                );
                results.push((family.name, format!("expected-Unknown: {reason}")));
            }
        }
    }

    eprintln!("family | result");
    for (family, result) in results {
        eprintln!("{family} | {result}");
    }
    let _ = fs::remove_dir_all(output_dir);
}

#[test]
fn conversion_sizes_and_modifiers_validate() {
    use std::collections::HashSet;
    if !Command::new("spirv-val")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
    {
        eprintln!("conversion matrix skipped: spirv-val unavailable");
        return;
    }
    let header = compute_header();
    let mut modules = HashSet::new();
    let mut count = 0;
    let mut check = |raw: u64| {
        let mut code = program(Family {
            name: "NOP",
            opcode: 0x50b0,
            fields: 0,
            required: false,
        });
        code[8..16].copy_from_slice(&raw.to_le_bytes());
        let output = Result::from(translate(
            shadowbox::QuietRun,
            &ShaderInput {
                code: &code,
                program_record: &header,
            },
        ))
        .unwrap_or_else(|e| panic!("{raw:016x}: {e}"));
        count += 1;
        if modules.insert(output.spirv.clone()) {
            let result = spirv_validation::validate(&output.spirv);
            assert!(
                result.status.success(),
                "{raw:016x}: {}",
                String::from_utf8_lossy(&result.stderr)
            );
        }
    };
    // Mesa sm50.rs:1889-1922 and envytools gm107.c:1200-1225,
    // 1993,2067,2113: I2I source forms, sizes, signs and modifiers.
    for opcode in [0x5ce0_u64, 0x4ce0, 0x38e0, 0x39e0] {
        for src_size in 0..=2_u64 {
            for dst_size in 0..=2_u64 {
                for signs in 0..4_u64 {
                    for byte in 0..=3_u64 {
                        // nvdisasm SM52: a 16-bit source selects H0 or H1.
                        if byte * 8 + (8 << src_size) > 32 || byte % (1 << src_size) != 0 {
                            continue;
                        }
                        for modifiers in 0..16_u64 {
                            let raw = opcode << 48
                                | 7 << 16
                                | 1
                                | src_size << 10
                                | dst_size << 8
                                | signs << 12
                                | byte << 41
                                | (modifiers & 1) << 45
                                | ((modifiers >> 1) & 1) << 49
                                | ((modifiers >> 2) & 1) << 50
                                | ((modifiers >> 3) & 1) << 47;
                            check(raw);
                        }
                    }
                }
            }
        }
    }
    // Mesa sm50.rs:1848-1879, envytools gm107.c:1256-1272,
    // 1995,2069,2115: I2F float16/32 and unsigned/signed byte forms.
    for opcode in [0x5cb8_u64, 0x4cb8, 0x38b8, 0x39b8] {
        for src_size in 0..=2_u64 {
            for dst_size in 1..=2_u64 {
                for sign in 0..=1_u64 {
                    for byte in 0..=3_u64 {
                        // nvdisasm SM52: a 16-bit source selects H0 or H1.
                        if byte * 8 + (8 << src_size) > 32 || byte % (1 << src_size) != 0 {
                            continue;
                        }
                        for modifiers in 0..4_u64 {
                            check(
                                opcode << 48
                                    | 7 << 16
                                    | 1
                                    | src_size << 10
                                    | dst_size << 8
                                    | sign << 13
                                    | byte << 41
                                    | (modifiers & 1) << 45
                                    | (modifiers >> 1) << 49,
                            );
                        }
                    }
                }
            }
        }
    }
    // Mesa sm50.rs:1806-1838, envytools gm107.c:1283-1307,
    // 1996,2070,2116: F2I rounding, signs, abs/neg, half selection and FTZ.
    for opcode in [0x5cb0_u64, 0x4cb0, 0x38b0, 0x39b0] {
        for src_size in 1..=2_u64 {
            for dst_size in 1..=2_u64 {
                for sign in 0..=1_u64 {
                    for rounding in 0..4_u64 {
                        for modifiers in 0..8_u64 {
                            let extra = (modifiers >> 2) & 1;
                            check(
                                opcode << 48
                                    | 7 << 16
                                    | 1
                                    | src_size << 10
                                    | dst_size << 8
                                    | sign << 12
                                    | rounding << 39
                                    | (modifiers & 1) << 45
                                    | ((modifiers >> 1) & 1) << 49
                                    | extra << if src_size == 1 { 41 } else { 44 },
                            );
                        }
                    }
                }
            }
        }
    }
    eprintln!(
        "conversion matrix: {count} accepted, {} distinct validated",
        modules.len()
    );
}

#[test]
fn double_conversion_and_images_share_the_uint_pair_type() {
    // Prepend an independent bundle to project-owned texture shaders.
    // I2F.F64 fields: Mesa sm50.rs:1873-1879; RZ source: envytools
    // gm107.c:116; NOP: Mesa sm50.rs:3178-3184. The relative branch
    // distances inside each original shader are preserved by the prefix.
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let convert = (0x5cb8_u64 << 48) | 0xb00 | (255 << 20) | (7 << 16) | 254;
    let nop = (0x50b0_u64 << 48) | (7 << 16);
    for name in ["texture_sample", "texture_fetch"] {
        let mut code = [0, convert, nop, nop]
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect::<Vec<_>>();
        code.extend(fs::read(root.join(format!("corpus/{name}.bin"))).unwrap());
        let header = fs::read(root.join(format!("corpus/{name}.hdr"))).unwrap();
        let output = Result::from(translate(
            shadowbox::QuietRun,
            &ShaderInput {
                code: &code,
                program_record: &header,
            },
        ))
        .unwrap();
        let path = std::env::temp_dir().join(format!(
            "shadowbox-double-image-{}-{name}.spv",
            std::process::id()
        ));
        fs::write(
            &path,
            output
                .spirv
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let check = Command::new("spirv-val")
            .args(["--target-env", "vulkan1.2"])
            .arg(&path)
            .output()
            .unwrap();
        fs::remove_file(path).unwrap();
        assert!(
            check.status.success(),
            "{name}: {}",
            String::from_utf8_lossy(&check.stderr)
        );
    }
}
