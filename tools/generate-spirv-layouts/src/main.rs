use clap::Parser;
use generate_spirv_layouts::{check, check_opcodes, generate};
use std::{
    fs,
    path::{Path, PathBuf},
    process::ExitCode,
};

#[derive(Parser)]
#[command(about = "Generate conservative operand layouts from the Khronos SPIR-V grammar")]
struct Args {
    #[arg(
        long,
        default_value = "/usr/include/spirv/unified1/spirv.core.grammar.json"
    )]
    grammar: PathBuf,
    #[arg(long)]
    grammar_label: Option<PathBuf>,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long)]
    op_source: Option<PathBuf>,
    #[arg(long)]
    check: bool,
}

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("generate-spirv-layouts: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Args) -> Result<(), String> {
    let output = args.output.unwrap_or_else(default_output);
    let raw = fs::read_to_string(&args.grammar)
        .map_err(|error| format!("read {}: {error}", args.grammar.display()))?;
    let grammar_label = args.grammar_label.as_deref().unwrap_or(&args.grammar);
    let generated = generate(&raw, grammar_label)?;
    if args.check {
        let checked_in = fs::read_to_string(&output)
            .map_err(|error| format!("read {}: {error}", output.display()))?;
        check(&generated, &checked_in).map_err(|error| format!("{}: {error}", output.display()))?;
        let op_source = args.op_source.unwrap_or_else(default_op_source);
        let op_source_text = fs::read_to_string(&op_source)
            .map_err(|error| format!("read {}: {error}", op_source.display()))?;
        check_opcodes(&raw, &op_source_text)
            .map_err(|error| format!("{}: {error}", op_source.display()))
    } else {
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("create {}: {error}", parent.display()))?;
        }
        fs::write(&output, generated)
            .map_err(|error| format!("write {}: {error}", output.display()))
    }
}

fn default_output() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../crates/shadowbox/src/optimizer/layouts.rs")
}

fn default_op_source() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../crates/shadowbox/src/spirv.rs")
}
