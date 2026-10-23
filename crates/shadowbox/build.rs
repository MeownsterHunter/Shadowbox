//! Original build fingerprint. Schema: docs/reference-sources.md.
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

fn sources(dir: &Path, files: &mut Vec<std::path::PathBuf>) {
    for entry in fs::read_dir(dir).expect("read translator sources") {
        let path = entry.expect("translator source entry").path();
        if path.is_dir() {
            sources(&path, files);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            files.push(path);
        }
    }
}

fn main() {
    // Watching the directory also catches added and removed implementation files.
    println!("cargo:rerun-if-changed=src");
    let mut files = Vec::new();
    sources(Path::new("src"), &mut files);
    files.extend(
        [
            "build.rs",
            "Cargo.toml",
            "../../Cargo.toml",
            "../../Cargo.lock",
        ]
        .map(Into::into),
    );
    files.sort();
    let mut hash = Sha256::new();
    hash.update(b"shadowbox-translator-sources-v1");
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        let name = path.to_str().expect("UTF-8 source path").as_bytes();
        let bytes = fs::read(&path).expect("read translator source");
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name);
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    println!(
        "cargo:rustc-env=SHADOWBOX_TRANSLATOR_VERSION={:x}",
        hash.finalize()
    );
}
