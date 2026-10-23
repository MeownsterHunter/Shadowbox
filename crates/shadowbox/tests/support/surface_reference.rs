// Independent GLSL image operations, decoded from raw Maxwell fields.
// Encodings: envytools envydis/gm107.c:753-838,1833-1871,1908-1914;
// Mesa src/nouveau/compiler/nak/sm50.rs:2435-2591.
// Semantics: Ryujinx Instructions/InstEmitSurface.cs:202-291,321-458,
// 483-563,637-680,693-795; CodeGen/Spirv/Instructions.cs:665-680,810-831.
use super::{compute::WORDS, reference::field};

struct Form {
    target: u32,
    raw: bool,
    size: u32,
    mask: u32,
    bound: bool,
    handle: u32,
    coord: u32,
    data: u32,
    store: bool,
    atomic: Option<u32>,
    reduction: bool,
    byte: bool,
}
pub fn is_surface(raw: u64) -> bool {
    matches!(
        (raw >> 48) & 0xffe0,
        0xeb00 | 0xeb20 | 0xeb40 | 0xea60 | 0xeac0
    ) || matches!((raw >> 48) & 0xffc0, 0xea00 | 0xea80 | 0xeb80)
}
fn form(raw: u64) -> Result<Form, &'static str> {
    let f = |a, b| field(raw, a, b);
    let o = raw >> 48;
    let reduction = o & 0xffe0 == 0xeb40;
    let flexible = matches!(o & 0xffe0, 0xea60 | 0xeac0) || o & 0xffc0 == 0xeb80;
    let atomic = reduction || flexible || matches!(o & 0xffc0, 0xea00 | 0xea80);
    let bound = if atomic && !flexible && !reduction {
        true
    } else {
        f(51, 1) != 0
    };
    let size = if atomic && !reduction {
        f(if flexible { 36 } else { 51 }, 3)
    } else if f(52, 1) != 0 {
        f(20, 3)
    } else {
        0
    };
    let cas = o & 0xffc0 == 0xea80 || o & 0xffe0 == 0xeac0;
    let op = if cas {
        9
    } else if reduction {
        f(24, 3)
    } else {
        f(29, 4)
    };
    let mask = if f(52, 1) == 0 {
        f(20, 4)
    } else {
        match size {
            5 => 3,
            6 | 7 => 15,
            _ => 1,
        }
    };
    if f(33, 3) > 5 || size > 7 || atomic && (size > 1 || op > 9) {
        return Err("unsupported surface form");
    }
    Ok(Form {
        target: f(33, 3),
        raw: atomic || f(52, 1) != 0,
        size,
        mask,
        bound,
        handle: f(39, 8),
        coord: f(8, 8),
        data: if reduction || !atomic {
            f(0, 8)
        } else {
            f(20, 8)
        },
        store: o & 0xffe0 == 0xeb20,
        atomic: atomic.then_some(op),
        reduction,
        byte: f(if atomic && !reduction { 28 } else { 23 }, 1) != 0,
    })
}
pub fn setup(raw: u64) -> Result<Vec<(u32, u32)>, &'static str> {
    let s = form(raw)?;
    Ok(if s.bound {
        Vec::new()
    } else {
        vec![(s.handle, 1)]
    })
}
pub fn prepare(raw: u64, case: usize, state: &mut [u32; WORDS]) -> Result<(), &'static str> {
    let s = form(raw)?;
    let shift = if s.atomic.is_some() {
        2
    } else {
        match s.size {
            0 | 1 => 0,
            2 | 3 => 1,
            4 => 2,
            5 => 3,
            _ => 4,
        }
    };
    let count = match s.target {
        0 | 1 => 1,
        2 | 3 => 2,
        _ => 3,
    };
    for i in 0..count {
        let reg = (s.coord + i).min(255);
        if reg < 255 {
            state[reg as usize] = if i == 0 {
                (case as u32) << if s.byte { shift } else { 0 }
            } else if i == count - 1 && matches!(s.target, 2 | 4) {
                (case / 4 % 3) as u32
            } else {
                (case / 4 % 4) as u32
            };
        }
    }
    // CAS must cover both success and failure. Initial raw bytes are shared
    // test data, not a CPU oracle for the operation.
    if s.atomic == Some(9) && case.is_multiple_of(2) && s.data < 255 {
        let x = case as u32;
        let y = if count > 1 {
            state[(s.coord + 1) as usize]
        } else {
            0
        };
        let z = if count > 2 {
            state[(s.coord + 2) as usize]
        } else {
            0
        };
        let at = (x + 64 * y + 256 * z) * 4;
        state[s.data as usize] = u32::from_le_bytes(std::array::from_fn(|i| {
            ((at + i as u32) as u8).wrapping_mul(73).wrapping_add(129)
        }));
    }
    for (reg, value) in setup(raw)? {
        if reg < 255 {
            state[reg as usize] = value;
        }
    }
    state[255] = 0;
    Ok(())
}
pub fn source(raw: u64) -> Result<String, &'static str> {
    let s = form(raw)?;
    let bits = |r: u32| format!("input_bits[base+{}u]", r.min(255));
    let signed = s.raw
        && if s.atomic.is_some() {
            s.size == 1
        } else {
            matches!(s.size, 1 | 3)
        };
    let kind = if !s.raw {
        ""
    } else if signed {
        "i"
    } else {
        "u"
    };
    let name = match s.target {
        0 => "1D",
        1 => "Buffer",
        2 => "1DArray",
        3 => "2D",
        4 => "2DArray",
        5 => "3D",
        _ => unreachable!(),
    };
    let format = if !s.raw {
        "rgba32f"
    } else if s.atomic.is_some() {
        if signed {
            "r32i"
        } else {
            "r32ui"
        }
    } else {
        match s.size {
            0 => "r8ui",
            1 => "r8i",
            2 => "r16ui",
            3 => "r16i",
            4 => "r32ui",
            5 => "rg32ui",
            _ => "rgba32ui",
        }
    };
    let count = match s.target {
        0 | 1 => 1,
        2 | 3 => 2,
        _ => 3,
    };
    let mut coords = (0..count)
        .map(|i| format!("int({})", bits(s.coord + i)))
        .collect::<Vec<_>>();
    if s.byte {
        let shift = if s.atomic.is_some() {
            2
        } else {
            match s.size {
                0 | 1 => 0,
                2 | 3 => 1,
                4 => 2,
                5 => 3,
                _ => 4,
            }
        };
        coords[0] = format!("({}>>{shift})", coords[0]);
    }
    let position = if count == 1 {
        coords[0].clone()
    } else {
        format!("ivec{count}({})", coords.join(","))
    };
    let vector = format!("{kind}vec4");
    let mut action = String::new();
    if let Some(op) = s.atomic {
        let value = |r| format!("{}({})", if signed { "int" } else { "uint" }, bits(r));
        let (function, args) = match op {
            0 => ("imageAtomicAdd", value(s.data)),
            1 => ("imageAtomicMin", value(s.data)),
            2 => ("imageAtomicMax", value(s.data)),
            3 => ("imageAtomicAdd", if signed { "1" } else { "1u" }.into()),
            4 => (
                "imageAtomicAdd",
                if signed { "-1" } else { "0xffffffffu" }.into(),
            ),
            5 => ("imageAtomicAnd", value(s.data)),
            6 => ("imageAtomicOr", value(s.data)),
            7 => ("imageAtomicXor", value(s.data)),
            8 => ("imageAtomicExchange", value(s.data)),
            9 => (
                "imageAtomicCompSwap",
                format!("{},{}", value(s.data), value(s.data + 1)),
            ),
            _ => unreachable!(),
        };
        action.push_str(&format!(
            "{} old={function}(img,{position},{args});",
            if signed { "int" } else { "uint" }
        ));
        if !s.reduction && field(raw, 0, 8) != 255 {
            action.push_str(&format!(
                "output_bits[base+{}u]=uint(old);",
                field(raw, 0, 8)
            ));
        }
    } else if s.store {
        let n = s.mask.count_ones();
        let parts = (0..4)
            .map(|i| {
                if i < n {
                    if !s.raw {
                        format!("uintBitsToFloat({})", bits(s.data + i))
                    } else {
                        format!(
                            "{}({})",
                            if signed { "int" } else { "uint" },
                            bits(s.data + i)
                        )
                    }
                } else {
                    if !s.raw {
                        "0.0"
                    } else if signed {
                        "0"
                    } else {
                        "0u"
                    }
                    .into()
                }
            })
            .collect::<Vec<_>>();
        action = format!("imageStore(img,{position},{vector}({}));", parts.join(","));
    } else {
        action = format!("{vector} result=imageLoad(img,{position});");
        let mut at = 0;
        for c in 0..4 {
            if s.mask & (1 << c) != 0 {
                let reg = (s.data + at).min(255);
                if reg != 255 {
                    action.push_str(&format!(
                        "output_bits[base+{reg}u]={};",
                        if s.raw {
                            format!("uint(result[{c}])")
                        } else {
                            format!("floatBitsToUint(result[{c}])")
                        }
                    ));
                }
                at += 1;
            }
        }
    }
    let binding = if s.bound { field(raw, 36, 13) } else { 2048 };
    let set = if s.bound { 2 } else { 0 };
    Ok(format!(
        "#version 450
layout(local_size_x=1) in;
layout(set={set},binding={binding},{format}) uniform {kind}image{name} img;
layout(set=3,binding=0,std430) readonly buffer In {{uint input_bits[];}};
layout(set=3,binding=1,std430) writeonly buffer Out {{uint output_bits[];}};
void main(){{uint base=gl_GlobalInvocationID.x*{WORDS}u;
for(uint i=0u;i<{WORDS}u;i++) output_bits[base+i]=input_bits[base+i];
if((input_bits[base+{predicate}u]!=0u)!={inverted}) {{{action}}}
}}",
        predicate = 256 + field(raw, 16, 3),
        inverted = field(raw, 19, 1) != 0
    ))
}
