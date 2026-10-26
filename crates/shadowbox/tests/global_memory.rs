//! Project-owned flat-address experiments. Encoding fields: Mesa
//! src/nouveau/compiler/nak/sm50.rs:2603-2614,2659-2669,2386-2410;
//! envytools/envydis/gm107.c:316,331,1898-1899. MOV cbuf forms:
//! sm50.rs:359-371,1926-1952. Predication: sm50.rs:225-236.
use shadowbox::ir::ops;
use shadowbox::spirv::Op;
use shadowbox::{
    decoder::decode_word,
    interface::ShaderInput,
    ir::{AtomicWidth, GlobalAccess, InstructionKind},
    translate_with_options, TranslationOptions, PUSH_GLOBAL_DELTA_OFFSET, PUSH_GLOBAL_DELTA_SIZE,
};
use std::{
    cell::Cell,
    fs,
    io::Write,
    path::PathBuf,
    process::{Command, Stdio},
};

fn raw(op: u16, data: u8, addr: u8, offset: i32, extended: bool) -> u64 {
    u64::from(op) << 48
        | u64::from(data)
        | u64::from(addr) << 8
        | 7 << 16
        | u64::from(offset as u32 & 0xffffff) << 20
        | u64::from(extended) << 45
}
fn program(instructions: &[u64]) -> Vec<u8> {
    let mut code = Vec::new();
    for chunk in instructions.chunks(3) {
        code.extend(0_u64.to_le_bytes());
        for index in 0..3 {
            code.extend(
                chunk
                    .get(index)
                    .copied()
                    .unwrap_or(0x50b0_u64 << 48 | 7 << 16)
                    .to_le_bytes(),
            );
        }
    }
    code
}
fn translate(instructions: &[u64], zero: bool) -> Vec<u32> {
    Result::from(translate_with_options(
        shadowbox::QuietRun,
        &ShaderInput {
            code: &program(instructions),
            program_record: &[0; 128],
        },
        TranslationOptions {
            global_delta: if zero {
                shadowbox::GlobalDelta::Zero
            } else {
                shadowbox::GlobalDelta::Pushed
            },
            ..TranslationOptions::default()
        },
    ))
    .unwrap()
    .spirv
}
fn ops(words: &[u32]) -> Vec<(Op, &[u32])> {
    let mut at = 5;
    let mut out = Vec::new();
    while at < words.len() {
        let count = (words[at] >> 16) as usize;
        out.push((Op::of(words[at]), &words[at + 1..at + count]));
        at += count;
    }
    out
}
struct WorkDir(PathBuf, Cell<usize>);
impl WorkDir {
    fn new(label: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("shadowbox-gmem-{label}-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        Self(dir, Cell::new(0))
    }
    fn module(&self, words: &[u32]) -> PathBuf {
        validate_module(words);
        // Keep each execution case in separate files. Rewriting the same
        // inode thousands of times can stall behind filesystem writeback.
        let sequence = self.1.get();
        self.1.set(sequence + 1);
        let path = self.0.join(format!("test-{sequence}.spv"));
        fs::write(
            &path,
            words
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        path
    }
}
fn validate_module(words: &[u32]) {
    let mut child = Command::new("spirv-val")
        .args(["--target-env", "vulkan1.2", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(
            &words
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let check = child.wait_with_output().unwrap();
    assert!(
        check.status.success(),
        "{}",
        String::from_utf8_lossy(&check.stderr)
    );
}

impl Drop for WorkDir {
    fn drop(&mut self) {
        fs::remove_dir_all(&self.0).unwrap();
    }
}

#[test]
fn global_decode_fields_and_unknown_modifiers() {
    for store in [false, true] {
        for size in 0..=if store { 6 } else { 7 } {
            for extended in [false, true] {
                let opcode = if store { 0xeed8 } else { 0xeed0 } | size;
                let word = raw(opcode, 8, 2, -31, extended);
                let decoded = decode_word([word as u32, (word >> 32) as u32], 8);
                let (s, mem_type) = match decoded.kind {
                    InstructionKind::GlobalMemory(ops::GlobalMemory {
                        access: GlobalAccess::Load(size),
                        ..
                    }) => (false, size as u16),
                    InstructionKind::GlobalMemory(ops::GlobalMemory {
                        access: GlobalAccess::Store(size),
                        ..
                    }) => (true, size as u16),
                    _ => panic!("not global memory: {decoded}"),
                };
                assert!(
                    matches!(decoded.kind, InstructionKind::GlobalMemory(ops::GlobalMemory {
                    data, address, offset: -31, extended: e, cache: 0, ..
                }) if s == store && data.0 == 8 && address.0 == 2 && e == extended && mem_type == size)
                );
                for cache in 1..=3 {
                    let modified = word | cache << 46;
                    assert!(matches!(
                        decode_word([modified as u32, (modified >> 32) as u32], 8).kind,
                        InstructionKind::GlobalMemory(ops::GlobalMemory { cache: c, .. }) if u64::from(c) == cache
                    ));
                }
            }
        }
    }
    for word in [
        raw(0xeedf, 8, 2, 0, true),
        raw(0xeed4, 8, 2, 0, true) | 1 << 44,
    ] {
        assert!(matches!(
            decode_word([word as u32, (word >> 32) as u32], 8).kind,
            InstructionKind::Unknown(ops::Unknown { .. })
        ));
    }
    let decoder = fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/decoder/families.rs"),
    )
    .unwrap();
    assert_eq!(
        decoder
            .matches("x.guarded(InstructionKind::Nop(ops::Nop))")
            .count(),
        2
    );
}

#[test]
fn global_declarations_and_accesses_validate() {
    let exit = 0xe300_u64 << 48 | 7 << 16 | 0xf;
    for store in [false, true] {
        for size in 0..=if store { 6 } else { 7 } {
            for extended in [false, true] {
                for zero in [false, true] {
                    // Validate both 32-bit and extended address forms in compute.
                    let instructions = [
                        raw(
                            if store { 0xeed8 } else { 0xeed0 } | size,
                            8,
                            2,
                            -16,
                            extended,
                        ),
                        exit,
                    ];
                    let words = translate(&instructions, zero);
                    for cache in 1..=3 {
                        let modified = translate(&[instructions[0] | cache << 46, exit], zero);
                        assert_ne!(words, modified);
                        // Only the decoded cache text changes. No executable
                        // SPIR-V operation or operand depends on cache policy.
                        assert_eq!(
                            ops(&words)
                                .into_iter()
                                .filter(|(op, _)| *op != Op::String)
                                .collect::<Vec<_>>(),
                            ops(&modified)
                                .into_iter()
                                .filter(|(op, _)| *op != Op::String)
                                .collect::<Vec<_>>()
                        );
                        validate_module(&modified);
                    }
                    if size == 7 {
                        let plain = translate(&[instructions[0] & !(1 << 48), exit], zero);
                        assert_ne!(words, plain);
                        assert_eq!(
                            ops(&words)
                                .into_iter()
                                .filter(|(op, _)| *op != Op::String)
                                .collect::<Vec<_>>(),
                            ops(&plain)
                                .into_iter()
                                .filter(|(op, _)| *op != Op::String)
                                .collect::<Vec<_>>()
                        );
                    }
                    validate_module(&words);
                    let instructions = ops(&words);
                    assert!(instructions
                        .iter()
                        .any(|(op, a)| *op == Op::MemoryModel && *a == [5348, 1]));
                    assert_eq!(
                        instructions
                            .iter()
                            .filter(|(op, a)| *op == Op::Variable && a[2] == 9)
                            .count(),
                        usize::from(!zero)
                    );
                    if !zero {
                        assert!(instructions.iter().any(|(op, a)| *op == Op::MemberDecorate
                            && a[1..] == [0, 35, PUSH_GLOBAL_DELTA_OFFSET]));
                        assert_eq!(PUSH_GLOBAL_DELTA_SIZE, 8);
                    }
                    assert!(instructions.iter().any(|(op, a)| *op == Op::Constant
                        && a.len() == 4
                        && a[2..] == [(-16_i32) as u32, u32::MAX]));
                    assert_eq!(
                        instructions
                            .iter()
                            .filter(|(op, _)| *op == Op::ConvertUToPtr)
                            .count(),
                        1
                    );
                    let narrow_store = store && size < 4;
                    for cap in [39, 4448, 22, 4433] {
                        let required = narrow_store
                            && if size < 2 {
                                cap == 39 || cap == 4448
                            } else {
                                cap == 22 || cap == 4433
                            };
                        assert_eq!(
                            instructions
                                .iter()
                                .any(|(op, a)| *op == Op::Capability && *a == [cap]),
                            required
                        );
                    }
                }
            }
        }
    }
    let words = translate(&[exit], false);
    validate_module(&words);
    let instructions = ops(&words);
    assert!(instructions
        .iter()
        .any(|(op, a)| *op == Op::MemoryModel && *a == [0, 1]));
    assert!(!instructions.iter().any(
        |(op, a)| (*op == Op::Capability && *a == [5347]) || (*op == Op::Variable && a[2] == 9)
    ));
}

fn cbuf(dst: u8, word: u64) -> u64 {
    // Mesa sm50.rs:1943-1946: all-lane constant-buffer setup.
    0x4c98_u64 << 48 | 15 << 39 | word << 20 | 7 << 16 | u64::from(dst)
}

#[test]
#[ignore = "requires Vulkan 1.2, buffer device addresses, and g++"]
fn global_memory_executes_on_vulkan() {
    let work = WorkDir::new("vulkan");
    let runner = vulkan_runner(&work);
    let mut cases = 0;
    for (mode, cache) in [
        ("delta", 0_u64),
        ("delta", 1),
        ("delta", 2),
        ("delta", 3),
        ("zero", 0),
        ("a32", 0),
        ("negative", 0),
        ("borrow", 0),
    ] {
        for store in [false, true] {
            for size in 0..=if store { 6 } else { 7 } {
                for destination in [2, 8] {
                    for guard in ["true", "false", "inverted-pt"] {
                        if destination == 2 && guard != "true" {
                            continue;
                        }
                        let count = match size {
                            5 => 2,
                            6 | 7 => 4,
                            _ => 1,
                        };
                        let width = match size {
                            0 | 1 => 1,
                            2 | 3 => 2,
                            _ => count * 4,
                        };
                        let byte = if size < 2 {
                            1
                        } else if size < 4 {
                            2
                        } else {
                            0
                        };
                        let mut initial = vec![0xa5; 128];
                        initial[..16].copy_from_slice(&[
                            0x10, 0x80, 0xff, 0x80, 0x67, 0x45, 0x23, 0x01, 0xef, 0xbe, 0xad, 0xde,
                            0x78, 0x56, 0x34, 0x12,
                        ]);
                        let mut expected = initial.clone();
                        let mut instructions =
                            vec![cbuf(2, 0), cbuf(3, 1), cbuf(12, 0), cbuf(13, 1)];
                        for (dst, word) in [(4, 2), (5, 3), (6, 4), (7, 5)] {
                            instructions.push(cbuf(dst, word));
                        }
                        // R8..R11 start zero. R2/R3 alias the address for active loads.
                        let guard_word = |word: u64| match guard {
                            "false" => word & !(7 << 16), // P0 starts false.
                            "inverted-pt" => word | 1 << 19,
                            _ => word,
                        };
                        if store {
                            instructions.push(guard_word(raw(
                                0xeed8 | size,
                                4,
                                if guard == "true" { 2 } else { 255 },
                                if guard == "true" { byte as i32 - 16 } else { 0 },
                                mode != "a32",
                            )));
                            if guard == "true" {
                                let values = [0xff80aabb_u32, 0x76543210, 0xdeadbeef, 0x12345678];
                                let bytes = values
                                    .iter()
                                    .flat_map(|v| v.to_le_bytes())
                                    .collect::<Vec<_>>();
                                expected[byte..byte + width].copy_from_slice(&bytes[..width]);
                            }
                        }
                        // A load always follows the store case so we check both the
                        // memory write and its later signed/unsigned/vector read.
                        instructions.push(if store {
                            raw(
                                0xeed0 | size,
                                destination,
                                2,
                                byte as i32 - 16,
                                mode != "a32",
                            )
                        } else {
                            guard_word(raw(
                                0xeed0 | size,
                                destination,
                                if guard == "true" { 2 } else { 255 },
                                if guard == "true" { byte as i32 - 16 } else { 0 },
                                mode != "a32",
                            ))
                        });
                        let mut result = [0_u32; 4];
                        if store || guard == "true" {
                            for (i, value) in result.iter_mut().enumerate().take(count) {
                                *value = match size {
                                    0 => u32::from(expected[byte]),
                                    1 => i32::from(expected[byte] as i8) as u32,
                                    2 => u32::from(u16::from_le_bytes(
                                        expected[byte..byte + 2].try_into().unwrap(),
                                    )),
                                    3 => i32::from(i16::from_le_bytes(
                                        expected[byte..byte + 2].try_into().unwrap(),
                                    )) as u32,
                                    _ => u32::from_le_bytes(
                                        expected[byte + i * 4..byte + i * 4 + 4]
                                            .try_into()
                                            .unwrap(),
                                    ),
                                };
                            }
                        }
                        // Use a saved address pair because the load may overwrite R2/R3.
                        let output_size = match count {
                            2 => 5,
                            4 => 6,
                            _ => 4,
                        };
                        instructions.push(raw(
                            0xeed8 | output_size,
                            destination,
                            12,
                            16,
                            mode != "a32",
                        ));
                        instructions.push(0xe300_u64 << 48 | 7 << 16 | 0xf);
                        for (index, value) in result.iter().enumerate().take(count) {
                            expected[32 + index * 4..36 + index * 4]
                                .copy_from_slice(&value.to_le_bytes());
                        }
                        for word in &mut instructions {
                            if matches!((*word >> 48) & 0xfff8, 0xeed0 | 0xeed8) {
                                *word |= cache << 46;
                            }
                        }
                        let module = work.module(&translate(&instructions, mode == "zero"));
                        fs::write(module.with_extension("initial.bin"), &initial).unwrap();
                        fs::write(module.with_extension("expected.bin"), &expected).unwrap();
                        let check = Command::new(&runner)
                            .env("MESA_SHADER_CACHE_DISABLE", "true")
                            .env("XDG_CACHE_HOME", work.0.join("driver-cache"))
                            .arg(&module)
                            .arg(module.with_extension("initial.bin"))
                            .arg(module.with_extension("expected.bin"))
                            .arg(mode)
                            .output()
                            .unwrap();
                        assert_ne!(
                            check.status.code(),
                            Some(77),
                            "No Vulkan device with required address and narrow-storage features"
                        );
                        assert!(
                            check.status.success(),
                            "{mode} store={store} size={size} guard={guard}: {}",
                            String::from_utf8_lossy(&check.stderr)
                        );
                        cases += 1;
                    }
                }
            }
        }
    }
    eprintln!("flat global memory Vulkan cases: {cases} MATCH");
}

fn vulkan_runner(work: &WorkDir) -> PathBuf {
    let runner = work.0.join("runner");
    let source =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tools/global-memory-vulkan.cpp");
    assert!(Command::new("g++")
        .args([
            "-std=c++17",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Wno-missing-field-initializers"
        ])
        .arg(source)
        .arg("-lvulkan")
        .arg("-o")
        .arg(&runner)
        .status()
        .unwrap()
        .success());
    runner
}

// Global ATOM/RED: Mesa sm50.rs:2717-2789, envytools gm107.c:321,667-737.
#[allow(clippy::too_many_arguments)]
fn atomic(
    red: bool,
    op: u8,
    signed: bool,
    dst: u8,
    data: u8,
    address: u8,
    offset: i32,
    extended: bool,
) -> u64 {
    let family = if red {
        0xebf8_u64 << 48
    } else if op == 15 {
        0xeef0_u64 << 48
    } else {
        0xed00_u64 << 48
    };
    family
        | 7 << 16
        | u64::from(address) << 8
        | u64::from(extended) << 48
        | u64::from(offset as u32 & 0xfffff) << 28
        | if red {
            u64::from(data) | u64::from(signed) << 20 | u64::from(op) << 23
        } else {
            u64::from(dst)
                | u64::from(data) << 20
                | u64::from(signed) << 49
                | if op == 15 { 0 } else { u64::from(op) << 52 }
        }
}

#[test]
fn global_atomics_decode_and_validate() {
    for red in [false, true] {
        for signed in [false, true] {
            for op in 0..=15 {
                let implemented = matches!(op, 0..=2 | 5..=7)
                    || (!signed && matches!(op, 3 | 4))
                    || (!red && (op == 8 || (op == 15 && !signed)));
                // RED has only three operation bits; do not accidentally generate a different opcode.
                if red && op > 7 {
                    continue;
                }
                // For CAS, bit 49 selects U64 (ATOM.CAS.64), a 64-bit form.
                if op == 15 && signed {
                    continue;
                }
                for extended in [false, true] {
                    for offset in [-(1 << 19), -16, 0, 16, (1 << 19) - 1] {
                        let word = atomic(red, op, signed, 8, 4, 2, offset, extended);
                        let decoded = decode_word([word as u32, (word >> 32) as u32], 8);
                        if !implemented {
                            assert!(matches!(
                                decoded.kind,
                                InstructionKind::Unknown(ops::Unknown { .. })
                            ));
                            continue;
                        }
                        assert!(
                            matches!(decoded.kind, InstructionKind::GlobalAtomic(ops::GlobalAtomic {
                            dst, data, address, offset: off, extended: e, signed: s, op: o,
                            width: AtomicWidth::B32,
                        }) if dst.map(|r| r.0) == (!red).then_some(8) && data.0 == 4 && address.0 == 2
                             && off == offset && e == extended && s == signed && o == op)
                        );
                        let words = translate(&[word, 0xe300_u64 << 48 | 7 << 16 | 0xf], false);
                        validate_module(&words);
                        let instructions = ops(&words);
                        let atom = instructions
                            .iter()
                            .find(|(o, _)| {
                                matches!(
                                    o,
                                    Op::AtomicExchange
                                        | Op::AtomicCompareExchange
                                        | Op::AtomicCompareExchangeWeak
                                        | Op::AtomicIIncrement
                                        | Op::AtomicIDecrement
                                        | Op::AtomicIAdd
                                        | Op::AtomicISub
                                        | Op::AtomicSMin
                                        | Op::AtomicUMin
                                        | Op::AtomicSMax
                                        | Op::AtomicUMax
                                        | Op::AtomicAnd
                                        | Op::AtomicOr
                                        | Op::AtomicXor
                                )
                            })
                            .unwrap();
                        let expected = match op {
                            0 => Op::AtomicIAdd,
                            1 => {
                                if signed {
                                    Op::AtomicSMin
                                } else {
                                    Op::AtomicUMin
                                }
                            }
                            2 => {
                                if signed {
                                    Op::AtomicSMax
                                } else {
                                    Op::AtomicUMax
                                }
                            }
                            5 => Op::AtomicAnd,
                            6 => Op::AtomicOr,
                            7 => Op::AtomicXor,
                            8 => Op::AtomicExchange,
                            // INC and DEC are compare-exchange loops.
                            3 | 4 | 15 => Op::AtomicCompareExchange,
                            _ => unreachable!(),
                        };
                        assert_eq!(atom.0, expected);
                        for (operand, bits) in [(atom.1[3], 1), (atom.1[4], 0)] {
                            assert!(instructions
                                .iter()
                                .any(|(o, a)| *o == Op::Constant && a[1..] == [operand, bits]));
                        }
                        assert_eq!(
                            instructions
                                .iter()
                                .filter(|(o, _)| *o == Op::ConvertUToPtr)
                                .count(),
                            1
                        );
                    }
                }
            }
        }
    }
    // F16x2 and invalid types, SAFEADD and CAS layouts are Unknown;
    // ADD.F32.FTZ.RN (type 3) and ATOM.CAS.64 (field 1) decode.
    for red in [false, true] {
        for ty in [4, 6, 7_u64] {
            let word = atomic(red, 0, false, 8, 4, 2, 0, true) | ty << if red { 20 } else { 49 };
            assert!(matches!(
                decode_word([word as u32, (word >> 32) as u32], 8).kind,
                InstructionKind::Unknown(ops::Unknown { .. })
            ));
        }
    }
    for field in 2..=7_u64 {
        let word = atomic(false, 15, false, 8, 4, 2, 0, true) | field << 49;
        assert!(matches!(
            decode_word([word as u32, (word >> 32) as u32], 8).kind,
            InstructionKind::Unknown(ops::Unknown { .. })
        ));
    }
    for data in [254, 255] {
        let word = atomic(false, 15, false, 8, data, 2, 0, true);
        assert!(matches!(
            decode_word([word as u32, (word >> 32) as u32], 8).kind,
            InstructionKind::Unknown(ops::Unknown { .. })
        ));
    }
}

#[test]
fn fallthrough_bra_stays_branch() {
    // BRA target bits 20..44 and CC test 0xf: Mesa sm50.rs:3005-3027;
    // decoder arm documents the family. A next-slot jump still performs BRA.
    let bytes = program(&[
        0xe240_u64 << 48 | 7 << 16 | 0xf,
        0xe300_u64 << 48 | 7 << 16 | 0xf,
    ]);
    let words = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect::<Vec<_>>();
    let decoded = shadowbox::decoder::decode_stream(shadowbox::QuietRun, &words);
    assert!(matches!(
        decoded[0].kind,
        InstructionKind::Branch(ops::Branch { target, .. }) if target.offset() == 0
    ));
}

#[test]
#[ignore = "requires Vulkan 1.2, buffer device addresses, and g++"]
fn global_atomics_execute_on_vulkan() {
    let work = WorkDir::new("atomic-vulkan");
    let runner = vulkan_runner(&work);
    let mut cases = 0;
    let value = 0xff80aabb_u32;
    let replacement = 0x76543210_u32;
    for mode in ["delta", "zero", "a32", "negative", "borrow"] {
        for red in [false, true] {
            for signed in [false, true] {
                for op in [0, 1, 2, 5, 6, 7, 8, 15] {
                    if (red && op >= 8) || (op == 15 && signed) {
                        continue;
                    }
                    for (dst, guard) in [(8, 0), (2, 0), (4, 0), (255, 0), (8, 1), (8, 2)] {
                        if red && dst != 8 {
                            continue;
                        }
                        // CAS success/failure and signed min/max both cross the sign bit.
                        for old in [value, 0x12345678_u32] {
                            let initial =
                                (0..32).flat_map(|_| old.to_le_bytes()).collect::<Vec<_>>();
                            let mut expected = initial.clone();
                            let mut instructions = vec![
                                cbuf(2, 0),
                                cbuf(3, 1),
                                cbuf(12, 0),
                                cbuf(13, 1),
                                cbuf(4, 2),
                                cbuf(5, 3),
                            ];
                            // Exercise positive and negative signed offsets. Cbuf is guest base +16.
                            let offset = if old == value { -16 } else { 16 };
                            let byte = (offset + 16) as usize;
                            let word = atomic(
                                red,
                                op,
                                signed,
                                dst,
                                4,
                                if guard == 0 { 2 } else { 255 },
                                if guard == 0 { offset } else { 0 },
                                mode != "a32",
                            );
                            instructions.push(match guard {
                                1 => word & !(7 << 16),
                                2 => word | 1 << 19,
                                _ => word,
                            });
                            if guard == 0 {
                                let new = match op {
                                    0 => old.wrapping_add(value),
                                    1 => {
                                        if signed {
                                            (old as i32).min(value as i32) as u32
                                        } else {
                                            old.min(value)
                                        }
                                    }
                                    2 => {
                                        if signed {
                                            (old as i32).max(value as i32) as u32
                                        } else {
                                            old.max(value)
                                        }
                                    }
                                    5 => old & value,
                                    6 => old | value,
                                    7 => old ^ value,
                                    8 => value,
                                    15 => {
                                        if old == value {
                                            replacement
                                        } else {
                                            old
                                        }
                                    }
                                    _ => unreachable!(),
                                };
                                expected[byte..byte + 4].copy_from_slice(&new.to_le_bytes());
                            }
                            if !red && dst != 255 {
                                instructions.push(raw(0xeedc, dst, 12, 48, mode != "a32"));
                                expected[64..68].copy_from_slice(
                                    &if guard == 0 { old } else { 0 }.to_le_bytes(),
                                );
                            }
                            instructions.push(0xe300_u64 << 48 | 7 << 16 | 0xf);
                            let module = work.module(&translate(&instructions, mode == "zero"));
                            fs::write(module.with_extension("initial.bin"), initial).unwrap();
                            fs::write(module.with_extension("expected.bin"), expected).unwrap();
                            let check = Command::new(&runner)
                                .env("MESA_SHADER_CACHE_DISABLE", "true")
                                .env("XDG_CACHE_HOME", work.0.join("driver-cache"))
                                .arg(&module)
                                .arg(module.with_extension("initial.bin"))
                                .arg(module.with_extension("expected.bin"))
                                .arg(mode)
                                .output()
                                .unwrap();
                            assert!(check.status.success(), "mode={mode} red={red} signed={signed} op={op} dst={dst} guard={guard} old={old:x}: {}", String::from_utf8_lossy(&check.stderr));
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    // RZ data is zero, and 64 workgroups contend on the same Device-scope atomic.
    for (data, groups) in [(4, 64_u32), (255, 1)] {
        let initial = vec![0; 128];
        let mut expected = initial.clone();
        expected[..4].copy_from_slice(
            &if data == 255 {
                0
            } else {
                value.wrapping_mul(groups)
            }
            .to_le_bytes(),
        );
        let instructions = [
            cbuf(2, 0),
            cbuf(3, 1),
            cbuf(4, 2),
            atomic(false, 0, false, 255, data, 2, -16, true),
            0xe300_u64 << 48 | 7 << 16 | 0xf,
        ];
        let module = work.module(&translate(&instructions, false));
        fs::write(module.with_extension("initial.bin"), initial).unwrap();
        fs::write(module.with_extension("expected.bin"), expected).unwrap();
        let check = Command::new(&runner)
            .env("MESA_SHADER_CACHE_DISABLE", "true")
            .env("XDG_CACHE_HOME", work.0.join("driver-cache"))
            .arg(&module)
            .arg(module.with_extension("initial.bin"))
            .arg(module.with_extension("expected.bin"))
            .arg("delta")
            .arg(groups.to_string())
            .output()
            .unwrap();
        assert!(
            check.status.success(),
            "Device contention/RZ: {}",
            String::from_utf8_lossy(&check.stderr)
        );
        cases += 1;
    }
    eprintln!("flat global atomic Vulkan cases: {cases} MATCH");
}

// MUFU: envytools gm107.c:1690-1700,2062; Mesa sm50.rs:749-768.
// Run project-owned arithmetic inputs through LDG.E -> MUFU -> STG.E.
// Values with exact binary32 results avoid CPU/GPU transcendental rounding
// comparisons. SIN/COS range and every modifier are covered by float_lane.
#[test]
#[ignore = "requires Vulkan 1.2, buffer device addresses, and g++"]
fn mufu_and_ldg_execute_on_vulkan() {
    let work = WorkDir::new("mufu");
    let runner = vulkan_runner(&work);
    let mut cases = 0;
    for input in [
        0.0_f32,
        -0.0,
        0.25,
        -0.25,
        1.0,
        -1.0,
        4.0,
        -4.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        for base_mod in [0, 4] {
            let mut initial = vec![0xa5; 128];
            initial[..4].copy_from_slice(&input.to_bits().to_le_bytes());
            let mut expected = initial.clone();
            let mut instructions = vec![cbuf(2, 0), cbuf(3, 1), raw(0xeed4, 4, 2, -16, true)];
            let mut output = 16;
            for modifiers in base_mod..base_mod + 4 {
                let mut x = if modifiers & 1 != 0 {
                    input.abs()
                } else {
                    input
                };
                if modifiers & 2 != 0 {
                    x = -x;
                }
                for function in [0, 1, 2, 3, 4, 5, 8] {
                    // Library NaN payloads and FClamp(NaN) are unspecified.
                    // Negative LG2/RSQ have an explicit canonical result.
                    if (function <= 1 && x != 0.0) || (function == 8 && x < 0.0) {
                        continue;
                    }
                    let value = match function {
                        0 => x.cos(),
                        1 => x.sin(),
                        2 => x.exp2(),
                        3 => x.log2(),
                        4 => 1.0 / x,
                        5 => 1.0 / x.sqrt(),
                        8 => x.sqrt(),
                        _ => unreachable!(),
                    };
                    // FClamp's sign of zero is not specified without
                    // float controls. Avoid comparing that sign to Rust's clamp.
                    if modifiers & 4 != 0 && (value.is_nan() || value.to_bits() == 0x8000_0000) {
                        continue;
                    }
                    let value = if modifiers & 4 != 0 {
                        value.clamp(0.0, 1.0)
                    } else {
                        value
                    };
                    let bits = if value.is_nan() {
                        0x7fc0_0000
                    } else {
                        value.to_bits()
                    };
                    let word = (0x5080_u64 << 48)
                        | (function << 20)
                        | (7 << 16)
                        | (4 << 8)
                        | 8
                        | ((modifiers & 1) << 46)
                        | (((modifiers >> 1) & 1) << 48)
                        | (((modifiers >> 2) & 1) << 50);
                    instructions.push(word);
                    instructions.push(raw(0xeedc, 8, 2, output as i32 - 16, true));
                    expected[output..output + 4].copy_from_slice(&bits.to_le_bytes());
                    output += 4;
                    cases += 1;
                }
            }
            instructions.push(0xe300_u64 << 48 | 7 << 16 | 0xf);
            let module = work.module(&translate(&instructions, false));
            fs::write(module.with_extension("initial.bin"), initial).unwrap();
            fs::write(module.with_extension("expected.bin"), expected).unwrap();
            let check = Command::new(&runner)
                .env("MESA_SHADER_CACHE_DISABLE", "true")
                .env("XDG_CACHE_HOME", work.0.join("driver-cache"))
                .arg(&module)
                .arg(module.with_extension("initial.bin"))
                .arg(module.with_extension("expected.bin"))
                .arg("delta")
                .output()
                .unwrap();
            assert!(
                check.status.success(),
                "input={input} modifiers={base_mod}: {}",
                String::from_utf8_lossy(&check.stderr)
            );
        }
    }
    eprintln!("MUFU + LDG Vulkan cases: {cases} MATCH");
}
