use std::{
    io::Write,
    process::{Command, Output, Stdio},
};

pub fn validate(words: &[u32]) -> Output {
    // spirv-val --help: "-" reads the same binary format from standard input.
    // Stream bytes to avoid disk writeback throttling in exhaustive sweeps.
    let mut child = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.2", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let bytes = words
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect::<Vec<_>>();
    child.stdin.take().unwrap().write_all(&bytes).unwrap();
    child.wait_with_output().unwrap()
}
