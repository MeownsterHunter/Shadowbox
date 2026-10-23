use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};

fn begins_with_decimal(text: &str) -> bool {
    text.trim_start()
        .as_bytes()
        .first()
        .is_some_and(u8::is_ascii_digit)
}

fn has_numeric_call(source: &str, call: &str) -> bool {
    source
        .match_indices(call)
        .any(|(at, _)| begins_with_decimal(&source[at + call.len()..]))
}

fn has_numeric_free_op_call(source: &str) -> bool {
    source.lines().any(|line| {
        line.find("op(&mut")
            .and_then(|at| line[at..].split_once(','))
            .is_some_and(|(_, opcode)| begins_with_decimal(opcode))
    })
}

fn has_numeric_op_comparison(source: &str) -> bool {
    source.lines().any(|line| {
        let Some((_, right)) = line.split_once("Op::of(") else {
            return false;
        };
        right
            .find(')')
            .map(|end| &right[end + 1..])
            .and_then(|tail| tail.split_once("==").or_else(|| tail.split_once("!=")))
            .is_some_and(|(_, operand)| begins_with_decimal(operand))
    })
}

fn tracked_rust_sources(root: &Path) -> Vec<PathBuf> {
    let output = Command::new("git")
        .args(["-C", root.to_str().unwrap(), "ls-files", "-z", "--", "*.rs"])
        .output()
        .unwrap();
    assert!(output.status.success());
    output
        .stdout
        .split(|byte| *byte == 0)
        .filter(|path| !path.is_empty())
        .map(|path| root.join(String::from_utf8(path.to_vec()).unwrap()))
        .collect()
}

fn allows_numeric_opcode_fields(path: &Path) -> bool {
    [
        "tools/generate-spirv-layouts/src/lib.rs",
        "crates/shadowbox/tests/global_memory.rs",
        "crates/shadowbox/tests/support/family_inputs.rs",
        "crates/shadowbox/tests/support/families.rs",
        // Novena's test-only SPIR-V fixture builder takes the opcode as data;
        // production consumers still use the generated symbolic identities.
        "vendor/novena/crates/novena/src/gpu/graphics.rs",
        "vendor/novena/crates/novena/src/gpu/pipelines.rs",
        "vendor/novena/crates/novena/src/gpu/uniforms.rs",
    ]
    .iter()
    .any(|suffix| path.ends_with(suffix))
}

#[test]
fn tracked_rust_consumers_do_not_recover_numeric_spirv_opcodes() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let this_test = root.join("crates/shadowbox/tests/spirv_identity.rs");
    let sources = tracked_rust_sources(&root);
    assert!(
        sources.len() > 100,
        "tracked Rust manifest is unexpectedly small"
    );
    for path in sources {
        if path == this_test {
            continue;
        }
        let source = fs::read_to_string(&path).unwrap();
        assert!(!has_numeric_call(&source, ".op("), "{}", path.display());
        assert!(!has_numeric_call(&source, ".result("), "{}", path.display());
        assert!(!has_numeric_call(&source, ".ty("), "{}", path.display());
        assert!(!has_numeric_free_op_call(&source), "{}", path.display());
        assert!(!has_numeric_op_comparison(&source), "{}", path.display());
        if !allows_numeric_opcode_fields(&path) {
            for signature in ["op: u16", "op: u32", "opcode: u16", "opcode: u32"] {
                assert!(
                    !source.contains(signature),
                    "{}: {signature}",
                    path.display()
                );
            }
        }
    }
}
