// Project-owned synthetic programs. Every hardware field is cited below.
pub fn mov(register: u8, bits: u32, mask: u8, inverted: bool) -> u64 {
    // Mesa src/nouveau/compiler/nak/sm50.rs:1938-1941,1951:
    // MOV32I opcode, literal20..51, quad enables12..15.
    // sm50.rs:321-328 places the destination in bits0..7.
    // envytools envydis/gm107.c:2170,408-411: PT guard16..18, invert19.
    0x0100_0000_0000_0000
        | (7 << 16)
        | (u64::from(inverted) << 19)
        | (u64::from(mask) << 12)
        | u64::from(register)
        | (u64::from(bits) << 20)
}

pub fn modules(instructions: &[u64], cc: u8) -> Vec<(&'static str, Vec<u32>)> {
    // Mesa src/nouveau/headers/nvidia/classes/cla097sph.h:403-415:
    // type2, version3, pixel5, MRT enable14. sph.rs:185-190,431-433
    // and cla097sph.h:616-627: RGBA target0 in bits576..579.
    let mut header = [0_u32; 32];
    header[0] = 2 | (3 << 5) | (5 << 10) | (1 << 14);
    header[18] = 15;
    let mut instructions = instructions.to_vec();
    // envytools envydis/gm107.c:1925,426-445: EXIT condition0..4.
    // cc=0 preserves Novena's EXIT.F; cc=15 is Mesa's unconditional
    // EXIT.T (src/nouveau/compiler/nak/sm50.rs:3101-3110).
    instructions.push(0xe300_0000_0000_0000 | (7 << 16) | u64::from(cc));
    // Mesa src/nouveau/compiler/nak/sm50.rs:3178-3184,3418-3428:
    // real NOP padding and one schedule word followed by three operations.
    let mut words = Vec::new();
    for bundle in instructions.chunks(3) {
        words.extend([0, 0]);
        for slot in 0..3 {
            let raw = bundle.get(slot).copied().unwrap_or(0x50b0_0000_0007_0000);
            words.extend([raw as u32, (raw >> 32) as u32]);
        }
    }
    let decoded = shadowbox::decoder::decode_stream(shadowbox::QuietRun, &words);
    let parsed = shadowbox::headers::parse(&header).unwrap();
    let raw = shadowbox::emitter::emit_unoptimized_with_header(&decoded, Some(&parsed))
        .unwrap()
        .words;
    // crates/shadowbox/src/lib.rs:178-192 consumes 20 SPH words before
    // bundled code. Novena removes its program-record prefix before calling it.
    let mut program = Vec::new();
    program.extend(header[..20].iter().flat_map(|word| word.to_le_bytes()));
    program.extend(words.iter().flat_map(|word| word.to_le_bytes()));
    let translated = Result::from(shadowbox::translate_header_prefixed(
        shadowbox::QuietRun,
        &program,
    ))
    .unwrap()
    .spirv;
    vec![("raw", raw), ("header-prefixed optimized", translated)]
}
