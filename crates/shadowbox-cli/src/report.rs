use std::{collections::BTreeMap, env, fs, path::Path, process::Command};
struct Scratch(std::path::PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn blocker(stderr: &[u8]) -> String {
    String::from_utf8_lossy(stderr)
        .lines()
        .find(|line| !line.trim().is_empty())
        .unwrap_or("command failed without diagnostic")
        .trim()
        .trim_start_matches("shadowbox: ")
        .to_string()
}
pub fn report(directory: &Path) -> Result<(), String> {
    let executable = env::current_exe().map_err(|e| e.to_string())?;
    let validator = Command::new("spirv-val")
        .arg("--version")
        .output()
        .map_err(|e| format!("run spirv-val: {e}"))?;
    if !validator.status.success() {
        return Err("spirv-val --version failed".into());
    }
    let mut files = fs::read_dir(directory)
        .map_err(|e| e.to_string())?
        .map(|entry| entry.map(|e| e.path()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| e.to_string())?;
    files.retain(|file| file.is_file() && file.extension().is_some_and(|ext| ext == "bin"));
    files.sort();
    let scratch_path = directory.join(format!(".cache-report-{}", std::process::id()));
    // Refuse to reuse another run's directory, and only clean up after creation.
    fs::create_dir(&scratch_path).map_err(|e| format!("create report scratch: {e}"))?;
    let scratch = Scratch(scratch_path);
    let output = scratch.0.join("shader.spv");
    let mut translated = 0;
    let mut validated = 0;
    let mut histogram = BTreeMap::<String, usize>::new();
    for file in &files {
        let translation = Command::new(&executable)
            .arg("translate")
            .arg(file)
            .arg("--header-prefixed")
            .arg("-o")
            .arg(&output)
            .output()
            .map_err(|e| format!("run translate: {e}"))?;
        let failure = if translation.status.success() {
            translated += 1;
            let validation = Command::new("spirv-val")
                .args(["--target-env", "vulkan1.2"])
                .arg(&output)
                .output()
                .map_err(|e| format!("run spirv-val: {e}"))?;
            if validation.status.success() {
                validated += 1;
                None
            } else {
                Some(format!("validate: {}", blocker(&validation.stderr)))
            }
        } else {
            Some(format!("translate: {}", blocker(&translation.stderr)))
        };
        if let Some(failure) = failure {
            *histogram.entry(failure).or_default() += 1;
        }
    }
    println!("summary: total={} translated={translated} translate_failed={} validated={validated} validate_failed={}", files.len(), files.len() - translated, translated - validated);
    let mut entries = histogram.into_iter().collect::<Vec<_>>();
    entries.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    for (failure, count) in entries {
        println!("{count}\t{failure}");
    }
    Ok(())
}
