//! Generated, project-owned compute workloads. Encoding references are Mesa
//! src/nouveau/compiler/nak/sm50.rs unless another path is given.
//! sm50.rs:204-209 defines RZ=255 and PT=7; :2105-2119 defines 2D=2.
//! Memory address mode/type: :2402-2410. All address-mode bits are zero.
//! Existing Shadowbox lowering is exercised without introducing new semantics.
use super::inputs::Case;
use std::{fs, path::Path};

fn address(i: usize) -> usize {
    (i / 3 * 4 + i % 3 + 1) * 8
}

// sm50.rs:225-235 PT, :321-328 destination, :1938-1941 MOV32I
// immediate and quad lanes. All lanes enabled, no modifiers.
fn mov(dst: u8, value: u32) -> u64 {
    0x0100_u64 << 48 | u64::from(value) << 20 | 0xf << 12 | u64::from(dst) | 7 << 16
}

pub fn synthetic() -> Vec<Case> {
    let header =
        fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../corpus/compute_empty.hdr"))
            .unwrap();
    let mut cases = Vec::new();
    for length in [200, 500, 1000, 2000] {
        for variant in 0..3 {
            let mut xs = vec![mov(8, 0), mov(9, 0), mov(20, 1), mov(21, 0x12)];
            while xs.len() + 33 < length {
                let start = xs.len();
                // sm50.rs:3038-3040 SSY relative target 20..44, CC.T.
                // envytools envydis/gm107.c:1932-1933 direct target, bit 5 clear.
                let relative = address(start + 32) - address(start) - 8;
                xs.push(0xe290_u64 << 48 | (relative as u64) << 20 | 0xf);
                for slot in 1..31 {
                    let offset = ((start + slot + variant * 17) % 256 * 4) as u64;
                    let raw = match slot % 8 {
                        0 => {
                            // sm50.rs:2215-2236 bindless TLD, dst 0..8,
                            // coordinates 8..16, handle 20..28, dim 28..31,
                            // mask 31..35, offset 35, NODEP 49, MS 50, LOD 55.
                            // envytools gm107.c:1965 clamp bit 54 clear.
                            // 2D, four components, constant packed offset R21.
                            0xdd38_u64 << 48
                                | 7 << 16
                                | 8 << 8
                                | 20 << 20
                                | 2 << 28
                                | 0xf << 31
                                | 1 << 35
                        }
                        1 | 3 | 5 | 7 => {
                            // sm50.rs:2633-2648 LDC dst 0..8, address 8..16,
                            // byte offset 20..36, bank 36..41, mode 44..46,
                            // type 48..51. B32=4 (:2386-2400), indexed=0, RZ.
                            // Ryujinx Instructions/InstEmitMemory.cs:59-103,
                            // :169-177 loads the indexed constant buffer word.
                            0xef94_u64 << 48
                                | 7 << 16
                                | 255 << 8
                                | offset << 20
                                | (variant as u64 + 1) << 36
                                | 24
                        }
                        2 => {
                            // sm50.rs:2659-2669 STS data 0..8, address 8..16,
                            // signed offset 20..44, type 48..51 (:2386-2400),
                            // cache 46..48. B32, RZ, default cache.
                            // Ryujinx InstEmitMemory.cs:156-166 shared storage.
                            0xef5c_u64 << 48 | 7 << 16 | 255 << 8 | offset << 20 | 24
                        }
                        4 => {
                            // sm50.rs:2603-2614 LDS dst 0..8, address 8..16,
                            // offset 20..44, type 48..51 (:2386-2400).
                            // envytools gm107.c:1894 uniform hint 44 clear.
                            // Ryujinx InstEmitMemory.cs:120-130 shared storage.
                            0xef4c_u64 << 48 | 7 << 16 | 255 << 8 | offset << 20 | 25
                        }
                        _ => {
                            // sm50.rs:1255-1263 IADD32I dst 0..8, src0 8..16,
                            // immediate 20..52, no carry/.X; gm107.c:2162
                            // saturation 54 clear. Feed loaded value to R24.
                            0x1c00_u64 << 48 | 7 << 16 | 25 << 8 | 1 << 20 | 24
                        }
                    };
                    xs.push(raw);
                }
                // sm50.rs:3050-3051 SYNC, CC.T. Predicate PT
                // sm50.rs:225-235; stack semantics Ryujinx
                // Instructions/InstEmitFlowControl.cs:225-275. This is a
                // stack pop resolved to its real SSY target, never a NOP.
                xs.push(0xf0f8_u64 << 48 | 7 << 16 | 0xf);
            }
            while xs.len() + 1 < length {
                // Same indexed B32 LDC fields as above, vary byte offset.
                let offset = (xs.len() % 256 * 4) as u64;
                xs.push(0xef94_u64 << 48 | 7 << 16 | 255 << 8 | offset << 20 | 1 << 36 | 24);
            }
            // sm50.rs:3101-3110 EXIT opcode, CC.T and PT (:225-235).
            xs.push(0xe300_u64 << 48 | 7 << 16 | 0xf);
            let mut code = Vec::with_capacity(xs.len().div_ceil(3) * 32);
            for bundle in xs.chunks(3) {
                // sm50.rs:3418-3428,3438-3468 schedule + three slots.
                code.extend_from_slice(&0_u64.to_le_bytes());
                for slot in 0..3 {
                    // sm50.rs:3183-3187 real OpNop, CC.T, PT (:225-235).
                    let raw = bundle
                        .get(slot)
                        .copied()
                        .unwrap_or(0x50b0_u64 << 48 | 7 << 16 | 0xf << 8);
                    code.extend_from_slice(&raw.to_le_bytes());
                }
            }
            cases.push(Case {
                name: format!("mixed_{length}_{variant}"),
                code,
                header: header.clone(),
            });
        }
    }
    cases
}
