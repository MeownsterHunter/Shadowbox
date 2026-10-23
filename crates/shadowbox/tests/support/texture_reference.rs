// Independent raw-encoding GLSL oracle. Never consume InstructionKind.
// Encodings: Mesa src/nouveau/compiler/nak/sm50.rs:2105-2149,2164-2381;
// envytools envydis/gm107.c:893-1008,1011-1023,1960-1970.
// Register/result semantics: Ryujinx Instructions/InstEmitTexture.cs:
// 159-306,394-601,609-655,708-776,1084-1182.
// Compute bias becomes explicit LOD: Ryujinx CodeGen/Spirv/Instructions.cs:1339-1348.
use super::compute::WORDS;
use super::reference::field;

#[derive(Clone, Copy, PartialEq)]
enum Op {
    Sample,
    Fetch,
    Gather,
    Query,
    Grad,
}
struct Shape {
    op: Op,
    dim: u32,
    lod: u32,
    offset: bool,
    offsets4: bool,
    depth: bool,
    half: bool,
    scalar: bool,
    bound: bool,
    ms: bool,
    component: u32,
    mask: u32,
}
pub fn is_texture(raw: u64) -> bool {
    raw >> 60 == 0xd || matches!((raw >> 48) & 0xfc38, 0xc038 | 0xc838)
}
pub fn is_query(raw: u64) -> bool {
    (raw >> 48) & !3 == 0xdf48
}
fn shape(raw: u64) -> Result<Shape, &'static str> {
    let f = |a, b| field(raw, a, b);
    let o = raw >> 48;
    let mut s = Shape {
        op: Op::Sample,
        dim: f(28, 3),
        lod: 0,
        offset: false,
        offsets4: false,
        depth: f(50, 1) != 0,
        half: false,
        scalar: false,
        bound: false,
        ms: false,
        component: 0,
        mask: f(31, 4),
    };
    if o & 0xf600 == 0xd000 || o & 0xf600 == 0xd200 {
        s.scalar = true;
        s.bound = true;
        s.half = f(59, 1) == 0;
        let selector = f(50, 3) as usize;
        s.mask = if f(28, 8) == 255 {
            [1, 2, 4, 8, 3, 9, 10, 12][selector]
        } else if selector <= 4 {
            [7, 11, 13, 14, 15][selector]
        } else {
            return Err("reserved scalar mask");
        };
        let mode = f(53, 4);
        if f(57, 1) != 0 {
            s.op = Op::Fetch;
            let (dim, lod, offset) = match mode {
                0 => (0, 1, false),
                1 => (0, 3, false),
                2 => (2, 1, false),
                4 => (2, 1, true),
                5 => (2, 3, false),
                6 => {
                    s.ms = true;
                    (2, 1, false)
                }
                7 => (4, 1, false),
                8 => (3, 1, false),
                12 => (2, 3, true),
                _ => return Err("reserved scalar fetch mode"),
            };
            s.dim = dim;
            s.lod = lod;
            s.offset = offset;
            s.depth = false;
        } else {
            let (dim, lod, depth) = match mode {
                0 => (0, 1, false),
                1 => (2, 0, false),
                2 => (2, 1, false),
                3 => (2, 3, false),
                4 => (2, 0, true),
                5 => (2, 3, true),
                6 => (2, 1, true),
                7 => (3, 0, false),
                8 => (3, 1, false),
                9 => (3, 1, true),
                10 => (4, 0, false),
                11 => (4, 1, false),
                12 => (6, 0, false),
                13 => (6, 3, false),
                _ => return Err("reserved scalar sample mode"),
            };
            s.dim = dim;
            s.lod = lod;
            s.depth = depth;
        }
    } else if o & 0xff40 == 0xdf00 {
        s.op = Op::Gather;
        s.dim = 2;
        s.scalar = true;
        s.bound = true;
        s.offset = f(51, 1) != 0;
        s.half = f(55, 1) != 0;
        s.component = f(52, 2);
        s.mask = if f(28, 8) == 255 { 3 } else { 15 };
    } else if matches!(o & !2, 0xdf58 | 0xdf60) {
        return Err("TMML requires fragment derivatives, compute transport cannot run it");
    } else if o & !2 == 0xdf50 || is_query(raw) {
        // Bound TXQ uses fixture-owned driver shape. Mesa sm50.rs:2351-2381;
        // Ryujinx InstEmitTexture.cs:1116-1119. Preserve bindless queries too.
        s.bound = is_query(raw);
        s.op = Op::Query;
        s.lod = f(22, 6);
        s.depth = false;
        if s.lod == 2 {
            s.ms = true;
        } else if s.lod != 1 {
            return Err("unsupported texture query");
        }
        s.dim = if !s.ms && s.mask & 4 != 0 { 4 } else { 2 };
    } else if o & !7 == 0xdeb8 || o & 0xfc38 == 0xc038 {
        s.bound = o & 0xfc38 == 0xc038;
        s.lod = f(if s.bound { 55 } else { 37 }, 3);
        s.offset = f(if s.bound { 54 } else { 36 }, 1) != 0;
        if !matches!(s.lod, 0..=3 | 6..=7) {
            return Err("unsupported vector LOD mode");
        }
    } else if matches!(o & !0xc7, 0xdc38 | 0xdd38) {
        s.op = Op::Fetch;
        s.bound = f(56, 1) == 0;
        s.depth = false;
        s.lod = if f(55, 1) != 0 { 3 } else { 1 };
        s.offset = f(35, 1) != 0;
        s.ms = f(50, 1) != 0;
    } else if o & !3 == 0xde38 {
        // Bound TXD: envytools gm107.c:1964, aoffi35, index36..48. Ryujinx
        // InstEmitTexture.cs:929-1008 reads B as dPdx then dPdy; Mesa
        // nak_nir_lower_tex.c:340-345 interleaves per axis. Only one axis
        // makes both agree.
        s.op = Op::Grad;
        s.bound = true;
        s.depth = false;
        s.offset = f(35, 1) != 0;
        if s.dim > 1 {
            return Err("TXD derivative order is pinned only for one axis");
        }
    } else if o & !6 == 0xdef8 || o & 0xfc38 == 0xc838 {
        s.op = Op::Gather;
        s.bound = o & 0xfc38 == 0xc838;
        s.component = f(if s.bound { 56 } else { 38 }, 2);
        let offset_mode = f(if s.bound { 54 } else { 36 }, 2);
        s.offset = offset_mode != 0;
        s.offsets4 = offset_mode == 2;
    } else {
        return Err("texture oracle missing opcode");
    }
    if !matches!(s.dim, 0 | 1 | 2 | 3 | 4 | 6 | 7) {
        return Err("invalid dimension");
    }
    Ok(s)
}
struct Sources {
    coords: Vec<u32>,
    layer: Option<u32>,
    lod: Option<u32>,
    offset: Option<u32>,
    offset2: Option<u32>,
    depth: Option<u32>,
    handle: Option<u32>,
    sample: Option<u32>,
    gradients: Option<u32>,
}
fn sources(raw: u64, s: &Shape) -> Sources {
    let a = |i: u32| (field(raw, 8, 8) + i).min(255);
    let b = |i: u32| (field(raw, 20, 8) + i).min(255);
    let count = match s.dim {
        0 | 1 => 1,
        2 | 3 => 2,
        _ => 3,
    };
    let mut r = Sources {
        coords: Vec::new(),
        layer: None,
        lod: None,
        offset: None,
        offset2: None,
        depth: None,
        handle: None,
        sample: None,
        gradients: None,
    };
    if s.op == Op::Grad {
        // One coordinate, then the packed layer/offset word; B is dPdx, dPdy.
        r.coords = vec![a(0)];
        if s.dim == 1 {
            r.layer = Some(a(1));
        }
        if s.offset {
            r.offset = Some(a(1));
        }
        r.gradients = Some(b(0));
        return r;
    }
    if s.op == Op::Query {
        r.handle = (!s.bound).then_some(a(0));
        if !s.ms {
            r.lod = Some(a(u32::from(!s.bound)));
        }
        return r;
    }
    if !s.scalar {
        let array = matches!(s.dim, 1 | 3 | 7);
        if array {
            r.layer = Some(a(0));
        }
        r.coords = (0..count).map(|i| a(i + u32::from(array))).collect();
        let mut next = 0;
        if !s.bound {
            r.handle = Some(b(next));
            next += 1;
        }
        if s.op != Op::Gather && s.lod >= 2 {
            r.lod = Some(b(next));
            next += 1;
        }
        if s.offset {
            r.offset = Some(b(next));
            next += 1;
        }
        if s.offsets4 {
            r.offset2 = Some(b(next));
            next += 1;
        }
        if s.depth {
            r.depth = Some(b(next));
            next += 1;
        }
        if s.ms {
            r.sample = Some(b(next));
        }
    } else {
        match s.op {
            Op::Sample if s.dim == 3 => {
                r.layer = Some(a(0));
                r.coords = vec![a(1), b(0)];
                if s.depth {
                    r.depth = Some(b(1));
                }
            }
            Op::Sample if s.dim == 0 => r.coords = vec![a(0)],
            Op::Sample if s.dim == 2 && !s.depth && s.lod != 3 => r.coords = vec![a(0), b(0)],
            Op::Sample => {
                r.coords = vec![a(0), a(1)];
                if count == 3 {
                    r.coords.push(b(0));
                }
                if s.depth {
                    r.depth = Some(b(0));
                }
                if s.lod == 3 {
                    r.lod = Some(b(u32::from(count == 3 || s.depth)));
                }
            }
            Op::Fetch if s.dim == 3 => {
                r.layer = Some(a(0));
                r.coords = vec![b(0), b(1)];
            }
            Op::Fetch if s.dim == 0 => {
                r.coords = vec![a(0)];
                if s.lod == 3 {
                    r.lod = Some(b(0));
                }
            }
            Op::Fetch if s.dim == 2 && s.lod == 1 && !s.offset && !s.ms => {
                r.coords = vec![a(0), b(0)]
            }
            Op::Fetch => {
                r.coords = vec![a(0), a(1)];
                if count == 3 {
                    r.coords.push(b(0));
                }
                if s.lod == 3 {
                    r.lod = Some(b(0));
                }
                if s.offset {
                    r.offset = Some(b(u32::from(s.lod == 3)));
                }
                if s.ms {
                    r.sample = Some(b(0));
                }
            }
            Op::Gather => {
                r.coords = if s.offset || s.depth {
                    vec![a(0), a(1)]
                } else {
                    vec![a(0), b(0)]
                };
                if s.offset {
                    r.offset = Some(b(0));
                }
                if s.offsets4 {
                    r.offset2 = Some(b(1));
                }
                if s.depth {
                    r.depth = Some(b(if s.offsets4 { 2 } else { u32::from(s.offset) }));
                }
            }
            Op::Query | Op::Grad => unreachable!(),
        }
    }
    r
}
pub fn setup(raw: u64) -> Result<Vec<(u32, u32)>, &'static str> {
    let s = shape(raw)?;
    let r = sources(raw, &s);
    let mut pairs = Vec::new();
    if let Some(handle) = r.handle {
        pairs.push((handle, 1));
    }
    if let Some(offset) = r.offset {
        // Include positive and negative components, all inside Vulkan's minimum
        // offset range. TEX/TLD nibbles: InstEmitTexture.cs:272-276,559-561;
        // gather six-bit bytes: 756-760. Runtime fetch/gather also uses these
        // inputs; sample offsets need proven constants for Vulkan ConstOffset.
        pairs.push((
            offset,
            match s.op {
                Op::Gather => 0x3f01,
                // TXD packs offsets from bit16 over the layer: x=+1 for 1D;
                // layer2 and x=-1 for 1D arrays.
                Op::Grad if s.dim == 1 => 0x000f_0002,
                Op::Grad => 0x0001_0000,
                _ => 0x1f1,
            },
        ));
    }
    if let Some(offset) = r.offset2 {
        pairs.push((offset, 0x3d03_3c04));
    }
    if pairs.iter().any(|(p, _)| {
        *p == 255
            || r.coords.contains(p)
            || (r.layer == Some(*p) && s.op != Op::Grad)
            || r.lod == Some(*p)
            || r.depth == Some(*p)
    }) {
        return Err("aliased setup source needs a separate input domain");
    }
    Ok(pairs)
}
pub fn prepare(raw: u64, case: usize, state: &mut [u32; WORDS]) -> Result<(), &'static str> {
    let s = shape(raw)?;
    let r = sources(raw, &s);
    let set = |state: &mut [u32; WORDS], register: u32, value| {
        if register < 255 {
            state[register as usize] = value;
        }
    };
    let level = (case % 2) as u32;
    for (i, &register) in r.coords.iter().enumerate() {
        let texel = (((case / 2) >> (2 * i)) % 4) as u32;
        let value = if s.op == Op::Fetch {
            texel >> if r.lod.is_some() && !s.ms { level } else { 0 }
        } else {
            ((texel as f32 + 0.375) / 4.0).to_bits()
        };
        set(state, register, value);
    }
    if matches!(s.dim, 6 | 7) {
        // Six face-center directions and off-center coordinates. Cube array
        // layer values name cubes, not individual faces.
        let major = case % 3;
        let sign = if (case / 3).is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        for (i, &register) in r.coords.iter().enumerate() {
            set(
                state,
                register,
                if i == major {
                    sign
                } else {
                    ((case % 4) as f32 - 1.5) / 4.0
                }
                .to_bits(),
            );
        }
    }
    if let Some(layer) = r.layer {
        // TXD ignores bits16..31 without AOFFI: InstEmitTexture.cs:976-985.
        let high = if s.op == Op::Grad {
            ((case % 4) as u32) << 16
        } else {
            0
        };
        set(
            state,
            layer,
            high | (case / 6 % if s.dim == 7 { 2 } else { 3 }) as u32,
        );
    }
    if let Some(gradients) = r.gradients {
        // Footprints below, at and above one texel of the four-texel level.
        let values = [0.0f32, 0.125, 0.25, 0.3, 0.5, 1.0];
        set(state, gradients, values[case % 6].to_bits());
        set(
            state,
            (gradients + 1).min(255),
            values[case / 6 % 6].to_bits(),
        );
    }
    if s.op == Op::Fetch && s.offset {
        let packed = setup(raw)?
            .into_iter()
            .find(|(reg, _)| Some(*reg) == r.offset)
            .unwrap()
            .1;
        for (axis, &register) in r.coords.iter().enumerate() {
            let off = ((packed >> (axis * 4)) << 28) as i32 >> 28;
            if register < 255 {
                state[register as usize] = state[register as usize].wrapping_sub(off as u32);
            }
        }
    }
    if let Some(lod) = r.lod {
        set(
            state,
            lod,
            if s.op == Op::Fetch || s.op == Op::Query {
                level
            } else {
                [-1.0f32, 0.0, 0.25, 0.75, 1.0, 2.0][case % 6].to_bits()
            },
        );
    }
    if let Some(sample) = r.sample {
        set(state, sample, (case % 4) as u32);
    }
    if let Some(depth) = r.depth {
        set(
            state,
            depth,
            if case.is_multiple_of(3) {
                0.0f32
            } else if case % 3 == 1 {
                0.01
            } else {
                0.8
            }
            .to_bits(),
        );
    }
    for (register, value) in setup(raw)? {
        set(state, register, value);
    }
    state[255] = 0;
    Ok(())
}
pub fn prepare_layer_edge(
    raw: u64,
    case: usize,
    state: &mut [u32; WORDS],
) -> Result<(), &'static str> {
    prepare(raw, case, state)?;
    let s = shape(raw)?;
    let r = sources(raw, &s);
    let layer = r.layer.ok_or("layer edge needs an array")?;
    if layer < 255 {
        state[layer as usize] = [-4i32, -1, 0, 1, 2, 3, 0x7fffffff][case % 7] as u32;
    }
    Ok(())
}
pub fn source(raw: u64) -> Result<String, &'static str> {
    source_inner(raw, false)
}
pub fn source_dynamic(raw: u64) -> Result<String, &'static str> {
    source_inner(raw, true)
}
pub fn runtime_setup(raw: u64) -> Result<Vec<(u32, u32)>, &'static str> {
    let s = shape(raw)?;
    let r = sources(raw, &s);
    Ok(setup(raw)?
        .into_iter()
        .filter(|(reg, _)| Some(*reg) != r.offset)
        .collect())
}
pub fn prepare_dynamic(
    raw: u64,
    case: usize,
    state: &mut [u32; WORDS],
) -> Result<(), &'static str> {
    prepare(raw, case, state)?;
    let s = shape(raw)?;
    let r = sources(raw, &s);
    let register = r.offset.ok_or("dynamic fixture needs offset")?;
    let width = if s.op == Op::Gather { 6 } else { 4 };
    let stride = if s.op == Op::Gather { 8 } else { 4 };
    let old = state[register as usize];
    let mut packed = 0;
    for axis in 0..r.coords.len() {
        let off = ((case / (axis + 1) % 3) as i32) - 1;
        packed |= (off as u32 & ((1 << width) - 1)) << (axis * stride);
        if s.op == Op::Fetch && r.coords[axis] < 255 {
            let old_off = ((old >> (axis * stride)) << (32 - width)) as i32 >> (32 - width);
            let coordinate = &mut state[r.coords[axis] as usize];
            *coordinate = coordinate
                .wrapping_add(old_off as u32)
                .wrapping_sub(off as u32);
        }
    }
    state[register as usize] = packed;
    Ok(())
}
fn source_inner(raw: u64, dynamic: bool) -> Result<String, &'static str> {
    let s = shape(raw)?;
    let r = sources(raw, &s);
    let bits = |r: u32| format!("input_bits[base+{r}u]");
    let fl = |r: u32| format!("uintBitsToFloat({})", bits(r));
    let name = match s.dim {
        0 => "1D",
        1 => "1DArray",
        2 => "2D",
        3 => "2DArray",
        4 => "3D",
        6 => "Cube",
        7 => "CubeArray",
        _ => unreachable!(),
    };
    let name = if s.ms {
        if s.dim == 3 {
            "2DMSArray"
        } else {
            "2DMS"
        }
    } else {
        name
    };
    let sampler = format!(
        "sampler{name}{}(tex,samp)",
        if s.depth { "Shadow" } else { "" }
    );
    let vector = |values: &[String], int: bool| {
        if values.len() == 1 {
            values[0].clone()
        } else {
            format!(
                "{}vec{}({})",
                if int { "i" } else { "" },
                values.len(),
                values.join(",")
            )
        }
    };
    let mut coords = r
        .coords
        .iter()
        .map(|&reg| {
            if s.op == Op::Fetch {
                format!("int({})", bits(reg))
            } else {
                fl(reg)
            }
        })
        .collect::<Vec<_>>();
    if let Some(layer) = r.layer {
        // Signed layers: Ryujinx CodeGen/Spirv/Instructions.cs:1218-1225.
        coords.push(if s.op == Op::Fetch {
            format!("int({})", bits(layer))
        } else if s.op == Op::Grad {
            format!("float(int({}&0xffffu))", bits(layer))
        } else {
            format!("float(int({}))", bits(layer))
        });
    }
    let coordinate = vector(&coords, s.op == Op::Fetch);
    let lod = r.lod.map_or("0.0".into(), fl);
    let packed = setup(raw)?
        .into_iter()
        .find(|(reg, _)| Some(*reg) == r.offset)
        .map(|(_, v)| v)
        .unwrap_or(0);
    let offset = (0..r.coords.len())
        .map(|i| {
            let (stride, width) = if s.op == Op::Gather { (8, 6) } else { (4, 4) };
            let value = ((packed >> (i * stride)) << (32 - width)) as i32 >> (32 - width);
            value.to_string()
        })
        .collect::<Vec<_>>();
    let off = if dynamic {
        let (stride, width) = if s.op == Op::Gather { (8, 6) } else { (4, 4) };
        vector(
            &(0..r.coords.len())
                .map(|axis| {
                    format!(
                        "bitfieldExtract(int({}),{},{} )",
                        bits(r.offset.unwrap()),
                        axis * stride,
                        width
                    )
                })
                .collect::<Vec<_>>(),
            true,
        )
    } else {
        vector(&offset, true)
    };
    let operation = match s.op {
        Op::Query if s.ms => format!("uvec4(0u,0u,uint(textureSamples({sampler})),0u)"),
        Op::Query => format!(
            "uvec4(textureSize({sampler},int({})),{},uint(textureQueryLevels({sampler})))",
            bits(r.lod.unwrap()),
            if s.dim == 4 { "" } else { "1u," }
        )
        .replace(",)", ")")
        .replace(",,", ","),
        Op::Fetch => {
            let coordinate = if s.offset {
                let offs = if r.layer.is_some() {
                    format!("ivec{}({off},0)", coords.len())
                } else {
                    off.clone()
                };
                format!("({coordinate}+{offs})")
            } else {
                coordinate
            };
            format!(
                "texelFetch({sampler},{coordinate},int({}))",
                if s.ms {
                    bits(r.sample.unwrap())
                } else {
                    r.lod.map_or("0u".into(), bits)
                }
            )
        }
        Op::Gather => {
            let dref = r.depth.map_or_else(|| s.component.to_string(), fl);
            if s.offsets4 {
                let values = [r.offset.unwrap(), r.offset2.unwrap()]
                    .into_iter()
                    .flat_map(|register| {
                        let packed = setup(raw)
                            .unwrap()
                            .into_iter()
                            .find(|(candidate, _)| *candidate == register)
                            .unwrap()
                            .1;
                        [0, 16].map(move |shift| {
                            let x = ((packed << (26 - shift)) as i32) >> 26;
                            let y = ((packed << (18 - shift)) as i32) >> 26;
                            format!("ivec2({x},{y})")
                        })
                    })
                    .collect::<Vec<_>>();
                format!(
                    "vec4(textureGatherOffset({sampler},{coordinate},{},{dref}).w,textureGatherOffset({sampler},{coordinate},{},{dref}).w,textureGatherOffset({sampler},{coordinate},{},{dref}).w,textureGatherOffset({sampler},{coordinate},{},{dref}).w)",
                    values[0], values[1], values[2], values[3]
                )
            } else {
                let function = if s.offset {
                    "textureGatherOffset"
                } else {
                    "textureGather"
                };
                // Shadow gather's comparison precedes offset. Mesa GLSL
                // builtin_functions.cpp:7852-7859,7882-7888 orders comparison then offset.
                let args = if s.depth && s.offset {
                    format!("{dref},{off}")
                } else if s.offset {
                    format!("{off},{dref}")
                } else {
                    dref
                };
                format!("{function}({sampler},{coordinate},{args})")
            }
        }
        Op::Grad => {
            let gradient = r.gradients.unwrap();
            let derivatives = format!("{},{}", fl(gradient), fl((gradient + 1).min(255)));
            if s.offset {
                let x = ((packed >> 16) << 28) as i32 >> 28;
                format!("textureGradOffset({sampler},{coordinate},{derivatives},{x})")
            } else {
                format!("textureGrad({sampler},{coordinate},{derivatives})")
            }
        }
        Op::Sample => {
            if let Some(depth) = r.depth {
                if s.dim == 0 {
                    coords.push("0.0".into());
                }
                if s.dim != 7 {
                    coords.push(fl(depth));
                }
            }
            let coords = vector(&coords, false);
            let function = if s.offset {
                "textureLodOffset"
            } else {
                "textureLod"
            };
            let separate_depth = if s.depth && s.dim == 7 {
                format!(",{}", fl(r.depth.unwrap()))
            } else {
                String::new()
            };
            let result = format!(
                "{function}({sampler},{coords}{separate_depth},{lod}{})",
                if s.offset {
                    format!(",{off}")
                } else {
                    String::new()
                }
            );
            if s.depth {
                format!("vec4({result})")
            } else {
                result
            }
        }
    };
    let dst = field(raw, 0, 8);
    let dst2 = field(raw, 28, 8);
    let components = (0..4)
        .filter(|c| s.mask & (1 << c) != 0)
        .collect::<Vec<_>>();
    let mut stores = String::new();
    if s.half {
        let mut parts = [
            "0.0".to_owned(),
            "0.0".to_owned(),
            "0.0".to_owned(),
            "0.0".to_owned(),
        ];
        for (i, c) in components.iter().enumerate() {
            parts[i] = format!("result[{c}]");
        }
        for (pair, reg) in [dst, dst2].into_iter().enumerate() {
            if reg != 255 {
                stores.push_str(&format!(
                    "output_bits[base+{reg}u]=packHalf2x16(vec2({},{}));\n",
                    parts[pair * 2],
                    parts[pair * 2 + 1]
                ));
            }
        }
    } else {
        for (i, c) in components.iter().enumerate() {
            let reg = if s.scalar {
                if i < 2 {
                    dst
                } else {
                    dst2
                }
            } else {
                dst
            };
            let reg = if reg == 255 {
                255
            } else {
                (reg + if s.scalar { i as u32 % 2 } else { i as u32 }).min(255)
            };
            if reg != 255 {
                stores.push_str(&format!(
                    "output_bits[base+{reg}u]={};\n",
                    if s.op == Op::Query {
                        format!("result[{c}]")
                    } else {
                        format!("floatBitsToUint(result[{c}])")
                    }
                ));
            }
        }
    }
    let image = if s.bound { 2 * field(raw, 36, 13) } else { 0 };
    let set = if s.bound { 1 } else { 0 };
    Ok(format!(
        "#version 450
#extension GL_EXT_texture_shadow_lod : enable
layout(local_size_x=1) in;
layout(set={set},binding={image}) uniform texture{name} tex;
layout(set={set},binding={sampler_binding}) uniform sampler{suf} samp;
layout(set=3,binding=0,std430) readonly buffer In {{uint input_bits[];}};
layout(set=3,binding=1,std430) writeonly buffer Out {{uint output_bits[];}};
void main(){{uint base=gl_GlobalInvocationID.x*{WORDS}u;
for(uint i=0u;i<{WORDS}u;i++) output_bits[base+i]=input_bits[base+i];
{result_type} result={operation};
if((input_bits[base+{predicate}u]!=0u)!={inverted}) {{{stores}}}
}}",
        sampler_binding = image + 1,
        suf = if s.depth { "Shadow" } else { "" },
        result_type = if s.op == Op::Query { "uvec4" } else { "vec4" },
        predicate = 256 + field(raw, 16, 3),
        inverted = field(raw, 19, 1) != 0
    ))
}
