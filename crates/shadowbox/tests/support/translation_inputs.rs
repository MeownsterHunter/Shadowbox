//! Project-owned throughput inputs. No new instruction semantics are introduced.
use std::{fs, path::Path};

pub struct Case {
    pub name: String,
    pub code: Vec<u8>,
    pub header: Vec<u8>,
}

pub fn corpus() -> Vec<Case> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus");
    let mut paths: Vec<_> = fs::read_dir(&root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|ext| ext == "bin"))
        .collect();
    paths.sort();
    paths
        .into_iter()
        .map(|path| Case {
            name: path.file_stem().unwrap().to_str().unwrap().into(),
            code: fs::read(&path).unwrap(),
            header: fs::read(path.with_extension("hdr")).unwrap(),
        })
        .collect()
}

// Mesa nak/sm50.rs:225-235 predicate 16..19 (PT=7), :321-328 dst 0..8;
// :1938-1941 MOV32I immediate 20..52 and quad lanes 12..16;
// :1255-1263 IADD32I src0 8..16, immediate 20..52, no carry or .X;
// envytools envydis/gm107.c:2162 adds saturation at 54, clear here.
fn mov(dst: u8, value: u32) -> u64 {
    0x0100_u64 << 48 | u64::from(value) << 20 | 0xf << 12 | u64::from(dst) | 7 << 16
}
fn address(index: usize) -> usize {
    (index / 3 * 4 + index % 3 + 1) * 8
}

pub fn synthetic() -> Vec<Case> {
    let header =
        fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/constant_color.hdr"))
            .unwrap();
    let mut cases = Vec::new();
    for length in [2048, 8192] {
        for kind in ["arithmetic", "blocks", "handles"] {
            let mut instructions = vec![mov(0, 0x3f000000), mov(20, 1), mov(8, 0), mov(9, 0)];
            for i in 0..length {
                let raw = if kind != "arithmetic" && i % 16 == 15 {
                    let index = instructions.len();
                    let relative = address(index + 1) - address(index) - 8;
                    // Mesa sm50.rs:3025-3028 BRA, rel24 at 20..44, CC.T=15.
                    // Predicate P0 (16..19=0) deliberately creates many blocks.
                    0xe240_u64 << 48 | (relative as u64) << 20 | 0xf
                } else if kind == "handles" && i % 16 == 0 {
                    mov(21, 0x12)
                } else if kind == "handles" && i % 16 == 14 {
                    // Mesa sm50.rs:2201-2236 TLD bindless opcode, dst 0..8,
                    // src0 8..16, src1 20..28, dim 28..31, mask 31..35,
                    // offset 35, nodep 49, multisample 50, LOD 55.
                    // envytools envydis/gm107.c:1965 clamp bit 54 is clear.
                    // :2105-2119 dimension 2 = 2D. Other modifier bits clear.
                    0xdd38_u64 << 48 | 7 << 16 | 8 << 8 | 20 << 20 | 2 << 28 | 0xf << 31 | 1 << 35
                } else {
                    0x1c00_u64 << 48 | 7 << 16 | 1 << 20
                };
                instructions.push(raw);
            }
            instructions.extend([mov(1, 0x3f800000), mov(2, 0x3f800000), mov(3, 0x3f800000)]);
            // Mesa sm50.rs:3101-3110 EXIT opcode and CC.T field.
            instructions.push(0xe300_u64 << 48 | 7 << 16 | 0xf);
            let mut code = Vec::with_capacity(instructions.len().div_ceil(3) * 32);
            for bundle in instructions.chunks(3) {
                // Mesa sm50.rs:3418-3428,3438-3468: schedule + 3 slots.
                code.extend_from_slice(&0_u64.to_le_bytes());
                for slot in 0..3 {
                    // Mesa sm50.rs:3183-3187: OpNop, CC.T=0xf.
                    let raw = bundle
                        .get(slot)
                        .copied()
                        .unwrap_or(0x50b0_u64 << 48 | 7 << 16 | 0xf << 8);
                    code.extend_from_slice(&raw.to_le_bytes());
                }
            }
            cases.push(Case {
                name: format!("synthetic_{kind}_{length}"),
                code,
                header: header.clone(),
            });
        }
    }
    cases
}
