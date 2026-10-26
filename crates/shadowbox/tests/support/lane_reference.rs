// Raw encodings and independent cross-lane results. No emitter introspection.
// sm50.rs is Mesa src/nouveau/compiler/nak; gm107.c is envytools envydis.
use super::compute::{cbuf, WORDS};
use super::reference::{field, Expected};

pub fn is_lane(raw: u64) -> bool {
    let o = (raw >> 48) as u16;
    o & 0xfff8 == 0xef10
        || matches!(o, 0x5c98 | 0x4c98 | 0xf0c8)
        || o & 0xfef8 == 0x3898
        || (0x0100..=0x01ff).contains(&o)
        || o & 0xfffc == 0x50d8
}

pub fn is_system(raw: u64) -> bool {
    raw >> 48 == 0xf0c8
}

/// Independent compute builtins, with the same 32-invocation geometry as the
/// translated module. S2R selector20..27: Mesa sm50.rs:3220-3224;
/// envytools gm107.c:121,136-137,147,150-157,173-177. Host mapping:
/// Ryujinx InstEmitMove.cs:70-180; Spirv/IoMap.cs:20,35-40,44.
pub fn system_source(raw: u64) -> Result<String, &'static str> {
    let f = |a, b| field(raw, a, b);
    let value = match f(20, 8) {
        0 => "gl_SubgroupInvocationID & 31u",
        0x12 => "floatBitsToUint(1.0)",
        0x13 | 0x1d => "0u",
        0x20 => "gl_LocalInvocationID.x | (gl_LocalInvocationID.y << 16u) | (gl_LocalInvocationID.z << 26u)",
        0x21 => "gl_LocalInvocationID.x",
        0x22 => "gl_LocalInvocationID.y",
        0x23 => "gl_LocalInvocationID.z",
        0x25 => "gl_WorkGroupID.x",
        0x26 => "gl_WorkGroupID.y",
        0x27 => "gl_WorkGroupID.z",
        0x38 => "gl_SubgroupEqMask.x",
        0x39 => "gl_SubgroupLtMask.x",
        0x3a => "gl_SubgroupLeMask.x",
        0x3b => "gl_SubgroupGtMask.x",
        0x3c => "gl_SubgroupGeMask.x",
        _ => return Err("GLSL oracle missing system register"),
    };
    Ok(format!(
        "#version 450
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_ballot : require
layout(local_size_x=32) in;
layout(set=3,binding=0,std430) readonly buffer In {{uint input_bits[];}};
layout(set=3,binding=1,std430) writeonly buffer Out {{uint output_bits[];}};
void main(){{uint base=gl_GlobalInvocationID.x*{WORDS}u;
for(uint i=0u;i<{WORDS}u;i++) output_bits[base+i]=input_bits[base+i];
if((input_bits[base+{predicate}u]!=0u)!={inverted} && {dst}u!=255u)
output_bits[base+{dst}u]={value};
}}",
        predicate = 256 + f(16, 3),
        inverted = f(19, 1) != 0,
        dst = f(0, 8)
    ))
}
pub fn reference(raw: u64, states: &[[u32; WORDS]]) -> Result<Vec<Expected>, &'static str> {
    assert_eq!(states.len(), 32);
    let f = |a, b| field(raw, a, b);
    let o = (raw >> 48) as u16;
    let active = |s: &[u32; WORDS]| (s[256 + f(16, 3) as usize] != 0) ^ (f(19, 1) != 0);
    states
        .iter()
        .enumerate()
        .map(|(lane, state)| {
            let mut out = Expected::new(state);
            if !active(state) {
                return Ok(out);
            }
            let dst = f(0, 8) as usize;
            let value = if o & 0xfff8 == 0xef10 {
                // Fields: sm50.rs:2043-2080, gm107.c:610-623,1896.
                // Results: Ryujinx Translation/HelperFunctionManager.cs:245-361.
                let index = if f(28, 1) != 0 {
                    f(20, 5)
                } else {
                    state[f(20, 8) as usize]
                };
                let control = if f(29, 1) != 0 {
                    f(34, 13)
                } else {
                    state[f(39, 8) as usize]
                };
                let segment = (control >> 8) & 31;
                let min = lane as u32 & segment;
                let max = ((control & 31) & !segment) | min;
                let target = match f(30, 2) {
                    0 => index & !segment | min,
                    1 => (lane as u32).wrapping_sub(index),
                    2 => (lane as u32).wrapping_add(index),
                    3 => lane as u32 ^ index,
                    _ => unreachable!(),
                };
                let valid = if f(30, 2) == 1 {
                    (target as i32) >= min as i32
                } else {
                    target <= max
                };
                let src = f(8, 8) as usize;
                if valid && (target >= 32 || !active(&states[target as usize])) {
                    return Err("undefined inactive or out-of-warp shuffle source");
                }
                let pred = f(48, 3) as usize;
                if pred != 7 {
                    out.words[256 + pred] = u32::from(valid);
                }
                if valid {
                    states[target as usize][src]
                } else {
                    state[src]
                }
            } else if o & 0xfffc == 0x50d8 {
                // sm50.rs:3227-3247; Ryujinx InstEmitWarp.cs:48-101.
                let mut ballot = 0_u32;
                let mut mask = 0_u32;
                for (i, s) in states.iter().enumerate() {
                    if active(s) {
                        mask |= 1 << i;
                        if (s[256 + f(39, 3) as usize] != 0) ^ (f(42, 1) != 0) {
                            ballot |= 1 << i;
                        }
                    }
                }
                let result = match f(48, 2) {
                    0 => ballot == mask,
                    1 => ballot != 0,
                    2 => ballot == 0 || ballot == mask,
                    _ => return Err("reserved vote mode"),
                };
                if f(45, 3) != 7 {
                    out.words[256 + f(45, 3) as usize] = u32::from(result);
                }
                ballot
            } else if o == 0xf0c8 {
                // S2R register20: sm50.rs:3215-3224; numeric selectors:
                // Ryujinx Decoders/InstDecoders.cs:148,178-184.
                // InstEmitMove.cs:70-79,88-117,117-145,173-180 maps host IDs.
                match f(20, 8) {
                    0 => lane as u32,
                    0x21 => lane as u32,
                    // YDirection selector18 and default float1 are independent
                    // reference facts: Decoders/InstDecoders.cs:163 and
                    // Instructions/InstEmitMove.cs:86-88. InvocationId17
                    // requires a graphics-stage invocation, not compute zero.
                    0x12 => 1.0_f32.to_bits(),
                    0x13 | 0x1d | 0x22 | 0x23 | 0x25..=0x27 => 0,
                    _ => return Err("lane oracle missing system register"),
                }
            } else {
                // MOV quad enable masks: sm50.rs:1926-1953. Each bit controls a
                // quad lane. Ryujinx InstEmitMove.cs:14-44 supplies the raw value.
                let immediate32 = (0x0100..=0x01ff).contains(&o);
                let immediate20 = o & 0xfef8 == 0x3898;
                let mask = f(if immediate32 { 12 } else { 39 }, 4);
                if mask & (1 << (lane & 3)) == 0 {
                    return Ok(out);
                }
                if immediate32 {
                    f(20, 32)
                } else if immediate20 {
                    // Ryujinx InstDecoders.cs:3820-3829 constructs this raw
                    // zero-extended 20-bit payload before InstEmitMove copies it.
                    f(20, 19) | (f(56, 1) << 19)
                } else if o == 0x4c98 {
                    cbuf(f(34, 5), f(20, 14))
                } else {
                    state[f(20, 8) as usize]
                }
            };
            if dst != 255 {
                out.words[dst] = value;
            }
            Ok(out)
        })
        .collect()
}
