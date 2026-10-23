use shadowbox::{decoder::decode_word, ir::{ops, InstructionKind}};
use std::io::{Read, Write};
fn name(kind: &InstructionKind) -> String {
    match kind {
        InstructionKind::Unknown(ops::Unknown { .. }) => "Unknown".to_owned(),
        kind => format!("{kind:?}").split([' ', '{', '(']).next().unwrap().to_owned(),
    }
}
fn main() {
    let mut buf = Vec::new();
    std::io::stdin().read_to_end(&mut buf).unwrap();
    let out = std::io::stdout();
    let mut out = std::io::BufWriter::new(out.lock());
    for c in buf.chunks_exact(8) {
        let raw = u64::from_le_bytes(c.try_into().unwrap());
        let i = decode_word([raw as u32, (raw >> 32) as u32], 0);
        writeln!(out, "{:016x}\t{}", raw, name(&i.kind)).unwrap();
    }
}
