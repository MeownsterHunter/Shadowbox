#[path = "shader_features.rs"]
mod shader_features;
// Project-owned test transport. Instruction operations are unchanged.
// SPIR-V grammar: /usr/include/spirv/unified1/spirv.core.grammar.json:
// 260-269 entry point; 401-445 types; 679-730 variable/load/store;
// 756-770 access chain;
// 838-860 decorations.
#[path = "surface.rs"]
mod surface;
#[path = "texture.rs"]
mod texture;
#[path = "texture_ms.rs"]
mod texture_ms;
use ash::{vk, Entry};
use shadowbox::emitter::SpirvModule;
use shadowbox::spirv::Op;
use std::collections::BTreeMap;

pub const STATE_WORDS: usize = 268; // R0..R255, P0..P7, Z/N/C/V.
pub const MEMORY_WORDS: usize = 32;
pub const WORDS: usize = STATE_WORDS + MEMORY_WORDS;
pub const GUEST_BASE: u64 = 0x2340_0000;
pub const MAX_CASES: usize = 64;

pub fn ops(words: &[u32]) -> impl Iterator<Item = (Op, &[u32])> {
    let mut at = 5;
    std::iter::from_fn(move || {
        if at == words.len() {
            return None;
        }
        let n = (words[at] >> 16) as usize;
        assert!(n > 0 && at + n <= words.len());
        let result = (Op::of(words[at]), &words[at + 1..at + n]);
        at += n;
        Some(result)
    })
}
fn inst(out: &mut Vec<u32>, op: Op, a: &[u32]) {
    out.push(((a.len() as u32 + 1) << 16) | u32::from(op));
    out.extend(a);
}

fn is_type(op: Op) -> bool {
    matches!(
        op,
        Op::TypeVoid
            | Op::TypeBool
            | Op::TypeInt
            | Op::TypeFloat
            | Op::TypeVector
            | Op::TypeMatrix
            | Op::TypeImage
            | Op::TypeSampler
            | Op::TypeSampledImage
            | Op::TypeArray
            | Op::TypeRuntimeArray
            | Op::TypeStruct
            | Op::TypeOpaque
            | Op::TypePointer
            | Op::TypeFunction
            | Op::TypeEvent
            | Op::TypeDeviceEvent
            | Op::TypeReserveId
            | Op::TypeQueue
            | Op::TypePipe
            | Op::TypeForwardPointer
    )
}

/// Add runtime state inputs and observable uint stores before optimisation.
/// Preserve every emitted operation, guard and branch.
/// Group size changes only test dispatch geometry. Lane operations run with
/// a complete Maxwell warp, not 32 independent single-lane workgroups.
pub fn observe_group(module: &mut SpirvModule, group_size: u32) {
    let original = &module.words;
    let instructions = ops(original).collect::<Vec<_>>();
    let uint = instructions
        .iter()
        .find(|(op, a)| *op == Op::TypeInt && a[1..] == [32, 0])
        .unwrap()
        .1[0];
    let boolt = instructions
        .iter()
        .find(|(op, _)| *op == Op::TypeBool)
        .unwrap()
        .1[0];
    let vec3 = instructions
        .iter()
        .find(|(op, a)| *op == Op::TypeVector && a[1..] == [uint, 3])
        .unwrap()
        .1[0];
    let pointer = |ty| {
        instructions
            .iter()
            .find(|(op, a)| *op == Op::TypePointer && a[1..] == [7, ty])
            .unwrap()
            .1[0]
    };
    let vars = |ty| {
        instructions
            .iter()
            .filter(|(op, a)| *op == Op::Variable && a[0] == pointer(ty))
            .map(|(_, a)| a[1])
            .collect::<Vec<_>>()
    };
    let registers = vars(uint);
    let predicates = vars(boolt);
    // emitter.rs creates 256 registers, the PC, eight predicates and Z/N/C/V.
    // Fail loudly if that layout changes.
    assert!(
        matches!(registers.len(), 257 | 513),
        "expected registers plus PC, with an optional parallel texture-array index bank"
    );
    assert_eq!(predicates.len(), 12);
    let state = registers[..256]
        .iter()
        .chain(&predicates)
        .copied()
        .collect::<Vec<_>>();
    // Workgroup and function arrays are the emitter's shared/local memory.
    // Initialise and export a bounded window, leaving memory operations intact.
    let memory = instructions.iter().find_map(|(op, a)| {
        if *op != Op::Variable {
            return None;
        }
        let ty = instructions
            .iter()
            .find(|(op, t)| *op == Op::TypePointer && t[0] == a[0])?
            .1;
        let array = instructions
            .iter()
            .find(|(op, t)| *op == Op::TypeArray && t[0] == ty[2] && t[1] == uint)?;
        let _ = array;
        let element = instructions
            .iter()
            .find(|(op, t)| *op == Op::TypePointer && t[1..] == [ty[1], uint])?
            .1[0];
        Some((a[1], element))
    });
    assert!(
        memory.is_none() || group_size == 1,
        "memory observer requires one lane per workgroup"
    );
    let mut next = original[3];
    let mut fresh = || {
        let id = next;
        next += 1;
        id
    };
    let array = fresh();
    let block = fresh();
    let buffer_pointer = fresh();
    let element_pointer = fresh();
    let input = fresh();
    let output = fresh();
    let id_pointer = fresh();
    let invocation = fresh();
    let invocation_value = fresh();
    let index = fresh();
    let base = fresh();
    let mut globals = Vec::new();
    let mut constants = BTreeMap::new();
    for (op, a) in &instructions {
        if *op == Op::Constant && a[0] == uint {
            constants.insert(a[2], a[1]);
        }
    }
    for v in 0..=WORDS as u32 {
        constants.entry(v).or_insert_with(|| {
            let id = fresh();
            inst(&mut globals, Op::Constant, &[uint, id, v]);
            id
        });
    }
    inst(&mut globals, Op::TypeRuntimeArray, &[array, uint]);
    inst(&mut globals, Op::TypeStruct, &[block, array]);
    inst(&mut globals, Op::TypePointer, &[buffer_pointer, 12, block]);
    inst(&mut globals, Op::TypePointer, &[element_pointer, 12, uint]);
    inst(&mut globals, Op::Variable, &[buffer_pointer, input, 12]);
    inst(&mut globals, Op::Variable, &[buffer_pointer, output, 12]);
    inst(&mut globals, Op::TypePointer, &[id_pointer, 1, vec3]);
    inst(&mut globals, Op::Variable, &[id_pointer, invocation, 1]);
    let mut annotations = Vec::new();
    inst(&mut annotations, Op::Decorate, &[array, 6, 4]);
    inst(&mut annotations, Op::Decorate, &[block, 2]);
    inst(&mut annotations, Op::MemberDecorate, &[block, 0, 35, 0]);
    for (var, binding) in [(input, 0), (output, 1)] {
        inst(&mut annotations, Op::Decorate, &[var, 34, 3]);
        inst(&mut annotations, Op::Decorate, &[var, 33, binding]);
    }
    inst(&mut annotations, Op::Decorate, &[invocation, 11, 28]);
    let mut prefix = Vec::new();
    inst(&mut prefix, Op::Load, &[vec3, invocation_value, invocation]);
    inst(
        &mut prefix,
        Op::CompositeExtract,
        &[uint, index, invocation_value, 0],
    );
    inst(
        &mut prefix,
        Op::IMul,
        &[uint, base, index, constants[&(WORDS as u32)]],
    );
    let mut exports = Vec::new();
    for (slot, &var) in state.iter().enumerate() {
        let at = fresh();
        let ptr = fresh();
        let value = fresh();
        inst(
            &mut exports,
            Op::IAdd,
            &[uint, at, base, constants[&(slot as u32)]],
        );
        inst(
            &mut exports,
            Op::AccessChain,
            &[element_pointer, ptr, output, constants[&0], at],
        );
        inst(
            &mut exports,
            Op::Load,
            &[if slot < 256 { uint } else { boolt }, value, var],
        );
        let bits = if slot < 256 {
            value
        } else {
            let bits = fresh();
            inst(
                &mut exports,
                Op::Select,
                &[uint, bits, value, constants[&1], constants[&0]],
            );
            bits
        };
        inst(&mut exports, Op::Store, &[ptr, bits]);
    }
    for word in 0..MEMORY_WORDS {
        let at = fresh();
        let in_ptr = fresh();
        let value = fresh();
        inst(
            &mut prefix,
            Op::IAdd,
            &[uint, at, base, constants[&((STATE_WORDS + word) as u32)]],
        );
        inst(
            &mut prefix,
            Op::AccessChain,
            &[element_pointer, in_ptr, input, constants[&0], at],
        );
        inst(&mut prefix, Op::Load, &[uint, value, in_ptr]);
        if let Some((variable, element)) = memory {
            let ptr = fresh();
            inst(
                &mut prefix,
                Op::AccessChain,
                &[element, ptr, variable, constants[&(word as u32)]],
            );
            inst(&mut prefix, Op::Store, &[ptr, value]);
        }
        let at = fresh();
        let ptr = fresh();
        inst(
            &mut exports,
            Op::IAdd,
            &[uint, at, base, constants[&((STATE_WORDS + word) as u32)]],
        );
        inst(
            &mut exports,
            Op::AccessChain,
            &[element_pointer, ptr, output, constants[&0], at],
        );
        let value = if let Some((variable, element)) = memory {
            let ptr = fresh();
            let value = fresh();
            inst(
                &mut exports,
                Op::AccessChain,
                &[element, ptr, variable, constants[&(word as u32)]],
            );
            inst(&mut exports, Op::Load, &[uint, value, ptr]);
            value
        } else {
            value
        };
        inst(&mut exports, Op::Store, &[ptr, value]);
    }
    let mut out = original[..5].to_vec();
    let mut annotated = false;
    let mut prefixed = false;
    let mut initialised = [false; STATE_WORDS];
    for (op, a) in instructions {
        if !annotated && is_type(op) {
            out.extend(&annotations);
            annotated = true;
        }
        if op == Op::Function {
            out.extend(&globals);
        }
        if op == Op::ExecutionMode && a[1] == 17 {
            inst(&mut out, op, &[a[0], 17, group_size, 1, 1]);
            continue;
        }
        if op == Op::EntryPoint {
            let mut a = a.to_vec();
            a.push(invocation);
            inst(&mut out, op, &a);
            continue;
        }
        if op == Op::Store {
            if !prefixed {
                out.extend(&prefix);
                prefixed = true;
            }
            if let Some(slot) = state.iter().position(|var| *var == a[0]) {
                if !initialised[slot] {
                    initialised[slot] = true;
                    if slot != 255 && slot != 263 {
                        // RZ and PT.
                        let at = fresh();
                        let ptr = fresh();
                        let bits = fresh();
                        inst(
                            &mut out,
                            Op::IAdd,
                            &[uint, at, base, constants[&(slot as u32)]],
                        );
                        inst(
                            &mut out,
                            Op::AccessChain,
                            &[element_pointer, ptr, input, constants[&0], at],
                        );
                        inst(&mut out, Op::Load, &[uint, bits, ptr]);
                        let value = if slot < 256 {
                            bits
                        } else {
                            let value = fresh();
                            inst(
                                &mut out,
                                Op::INotEqual,
                                &[boolt, value, bits, constants[&0]],
                            );
                            value
                        };
                        inst(&mut out, Op::Store, &[a[0], value]);
                        continue;
                    }
                }
            }
        }
        if op == Op::Return {
            out.extend(&exports);
        }
        inst(&mut out, op, a);
    }
    assert!(initialised.iter().all(|v| *v));
    out[3] = next;
    module.words = out;
}

struct Buffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *mut u32,
}
pub struct Compute {
    _entry: Entry,
    instance: ash::Instance,
    physical: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    pool: vk::CommandPool,
    command: vk::CommandBuffer,
    inputs: Buffer,
    outputs: Buffer,
    uniform: Buffer,
    flat: Buffer,
    delta: u64,
    textures: BTreeMap<(u32, bool, bool, bool), texture::Texture>,
    storage_image: texture::StorageImage,
    shader_features: shader_features::ShaderFeatures,
    pub description: String,
    pub supports_shader_float64: bool,
}
impl Compute {
    pub fn new() -> Self {
        unsafe {
            let entry = Entry::load().unwrap();
            let instance = entry
                .create_instance(
                    &vk::InstanceCreateInfo::default().application_info(
                        &vk::ApplicationInfo::default()
                            .application_name(c"shadowbox-gpu-exact")
                            .api_version(vk::API_VERSION_1_2),
                    ),
                    None,
                )
                .unwrap();
            let physical = instance
                .enumerate_physical_devices()
                .unwrap()
                .into_iter()
                .find(|p| {
                    instance
                        .get_physical_device_properties(*p)
                        .device_name_as_c_str()
                        .unwrap()
                        .to_string_lossy()
                        .contains("RADV")
                })
                .expect("gpu_exact requires RADV hardware; no silent skip");
            let properties = instance.get_physical_device_properties(physical);
            let description = format!(
                "{} driver={} api={}",
                properties.device_name_as_c_str().unwrap().to_string_lossy(),
                properties.driver_version,
                properties.api_version
            );
            let family = instance
                .get_physical_device_queue_family_properties(physical)
                .iter()
                .position(|p| p.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .unwrap() as u32;
            // Flat physical pointers require BDA; sub-word stores require storage
            // features. Vulkan headers vulkan_core.h:6004-6022,6214-6248.
            // 64-bit global ATOM/RED use SPIR-V Int64Atomics on physical
            // buffers; enable it only where the device reports it.
            let mut supported12 = vk::PhysicalDeviceVulkan12Features::default();
            let mut query = vk::PhysicalDeviceFeatures2::default().push_next(&mut supported12);
            instance.get_physical_device_features2(physical, &mut query);
            let int64_atomics = supported12.shader_buffer_int64_atomics != 0;
            let sampled_image_array_non_uniform =
                supported12.shader_sampled_image_array_non_uniform_indexing != 0;
            let mut features12 = vk::PhysicalDeviceVulkan12Features::default()
                .shader_int8(true)
                .buffer_device_address(true)
                .storage_buffer8_bit_access(true)
                .shader_buffer_int64_atomics(int64_atomics)
                .shader_sampled_image_array_non_uniform_indexing(sampled_image_array_non_uniform);
            let mut features11 =
                vk::PhysicalDeviceVulkan11Features::default().storage_buffer16_bit_access(true);
            // Ryujinx SpirvGenerator.cs:58-60 gates Float64 on host support.
            let supports_shader_float64 = instance
                .get_physical_device_features(physical)
                .shader_float64
                != 0;
            let features = vk::PhysicalDeviceFeatures::default()
                .shader_int64(true)
                .shader_image_gather_extended(true)
                .shader_int16(true)
                .shader_float64(supports_shader_float64)
                .image_cube_array(true)
                .shader_storage_image_extended_formats(true)
                .shader_storage_image_multisample(true)
                .shader_storage_image_read_without_format(true)
                .shader_storage_image_write_without_format(true);
            let shader_features = shader_features::ShaderFeatures::query(&instance, physical);
            let description = format!("{description} {}", shader_features.describe());
            let extensions = shader_features.extension_names();
            let mut reconvergence =
                vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR::default()
                    .shader_maximal_reconvergence(shader_features.maximal_reconvergence);
            let mut quad = vk::PhysicalDeviceShaderQuadControlFeaturesKHR::default()
                .shader_quad_control(shader_features.quad_control);
            let mut demote = vk::PhysicalDeviceShaderDemoteToHelperInvocationFeatures::default()
                .shader_demote_to_helper_invocation(shader_features.demote_to_helper_invocation);
            let mut device_info = vk::DeviceCreateInfo::default();
            if shader_features.maximal_reconvergence {
                device_info = device_info.push_next(&mut reconvergence);
            }
            if shader_features.quad_control {
                device_info = device_info.push_next(&mut quad);
            }
            if shader_features.demote_to_helper_invocation {
                device_info = device_info.push_next(&mut demote);
            }
            let device = instance
                .create_device(
                    physical,
                    &device_info
                        .enabled_extension_names(&extensions)
                        .enabled_features(&features)
                        .push_next(&mut features12)
                        .push_next(&mut features11)
                        .queue_create_infos(&[vk::DeviceQueueCreateInfo::default()
                            .queue_family_index(family)
                            .queue_priorities(&[1.0])]),
                    None,
                )
                .unwrap();
            let queue = device.get_device_queue(family, 0);
            let pool = device
                .create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
                .unwrap();
            let command = device
                .allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
                .unwrap()[0];
            let memory = instance.get_physical_device_memory_properties(physical);
            let buffer = |words, usage, address| {
                let buffer = device
                    .create_buffer(
                        &vk::BufferCreateInfo::default().size(words * 4).usage(usage),
                        None,
                    )
                    .unwrap();
                let req = device.get_buffer_memory_requirements(buffer);
                let ty = (0..memory.memory_type_count)
                    .find(|i| {
                        req.memory_type_bits & (1 << i) != 0
                            && memory.memory_types[*i as usize].property_flags.contains(
                                vk::MemoryPropertyFlags::HOST_VISIBLE
                                    | vk::MemoryPropertyFlags::HOST_COHERENT,
                            )
                    })
                    .unwrap();
                let mut flags = vk::MemoryAllocateFlagsInfo::default()
                    .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
                let mut allocate = vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(ty);
                if address {
                    allocate = allocate.push_next(&mut flags);
                }
                let allocation = device.allocate_memory(&allocate, None).unwrap();
                device.bind_buffer_memory(buffer, allocation, 0).unwrap();
                let mapped = device
                    .map_memory(allocation, 0, req.size, vk::MemoryMapFlags::empty())
                    .unwrap()
                    .cast();
                Buffer {
                    buffer,
                    memory: allocation,
                    mapped,
                }
            };
            let inputs = buffer(
                (WORDS * MAX_CASES) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                false,
            );
            let outputs = buffer(
                (WORDS * MAX_CASES) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
                false,
            );
            // 32 banks, 64 KiB each. Bank and word fields: sm50.rs:361-372.
            let uniform = buffer(32 * 16384, vk::BufferUsageFlags::UNIFORM_BUFFER, false);
            for bank in 0..32 {
                for word in 0..16384 {
                    *uniform.mapped.add(bank * 16384 + word) = cbuf(bank as u32, word as u32);
                }
            }
            let flat = buffer(
                (MEMORY_WORDS * MAX_CASES) as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                true,
            );
            let address = device.get_buffer_device_address(
                &vk::BufferDeviceAddressInfo::default().buffer(flat.buffer),
            );
            let delta = address.wrapping_sub(GUEST_BASE);
            let mut textures = BTreeMap::new();
            for dim in 0..4 {
                for arrayed in [false, true] {
                    if dim == 2 && arrayed {
                        continue;
                    }
                    for depth in [false, true] {
                        if dim == 2 && depth {
                            continue;
                        }
                        textures.insert(
                            (dim, arrayed, depth, false),
                            texture::Texture::new(
                                &instance,
                                &device,
                                physical,
                                command,
                                queue,
                                (dim, arrayed, depth),
                            ),
                        );
                    }
                }
            }
            for arrayed in [false, true] {
                textures.insert(
                    (1, arrayed, false, true),
                    texture_ms::create(&instance, &device, physical, command, queue, arrayed),
                );
            }
            let storage_image =
                texture::StorageImage::new(&instance, &device, physical, command, queue);
            Self {
                storage_image,
                _entry: entry,
                instance,
                physical,
                device,
                queue,
                pool,
                command,
                inputs,
                outputs,
                uniform,
                flat,
                delta,
                textures,
                description,
                shader_features,
                supports_shader_float64,
            }
        }
    }
    #[allow(dead_code)]
    pub fn run(&self, words: &[u32], inputs: &[[u32; WORDS]]) -> Vec<[u32; WORDS]> {
        self.run_images_config(words, inputs, false, None, GUEST_BASE, true)
            .0
    }
    #[allow(dead_code)]
    pub fn run_images(
        &self,
        words: &[u32],
        inputs: &[[u32; WORDS]],
        flat: bool,
    ) -> (Vec<[u32; WORDS]>, Vec<u8>) {
        self.run_images_config(words, inputs, flat, None, GUEST_BASE, false)
    }
    pub fn run_images_config(
        &self,
        words: &[u32],
        inputs: &[[u32; WORDS]],
        flat: bool,
        cbuf_override: Option<(u32, u32, u32)>,
        guest_base: u64,
        persistent_storage: bool,
    ) -> (Vec<[u32; WORDS]>, Vec<u8>) {
        assert!(!inputs.is_empty() && inputs.len() <= MAX_CASES);
        let mut sets = BTreeMap::new();
        self.shader_features.require_module(words);
        let mut bindings = BTreeMap::new();
        for (op, a) in ops(words) {
            if op == Op::Decorate && a[1] == 34 {
                sets.insert(a[0], a[2]);
            }
            if op == Op::Decorate && a[1] == 33 {
                bindings.insert(a[0], a[2]);
            }
        }
        // Resolve descriptor types from declarations, not binding numbers.
        // Production CBs and images can both live in set0; scalar images in
        // set1; surfaces in set2; observer buffers in set3. A bound handle keeps its table index.
        let declarations = ops(words).collect::<Vec<_>>();
        let mut descriptors = [BTreeMap::new(), BTreeMap::new(), BTreeMap::new()];
        let mut image_keys = BTreeMap::new();
        for (&var, &set) in &sets {
            if set >= 3 {
                continue;
            }
            let variable = declarations
                .iter()
                .find(|(op, a)| *op == Op::Variable && a[1] == var)
                .unwrap()
                .1;
            let pointer = declarations
                .iter()
                .find(|(op, a)| *op == Op::TypePointer && a[0] == variable[0])
                .unwrap()
                .1;
            let pointed = declarations
                .iter()
                .find(|(op, a)| is_type(*op) && a.first() == Some(&pointer[2]))
                .unwrap();
            let (ty, count) = if pointed.0 == Op::TypeArray {
                let length = declarations
                    .iter()
                    .find(|(op, a)| *op == Op::Constant && a[1] == pointed.1[2])
                    .unwrap()
                    .1[2];
                let element = declarations
                    .iter()
                    .find(|(op, a)| is_type(*op) && a.first() == Some(&pointed.1[1]))
                    .unwrap();
                (element, length)
            } else {
                (pointed, 1)
            };
            let descriptor = match ty.0 {
                Op::TypeImage if ty.1[6] == 2 && ty.1[2] == 5 => {
                    vk::DescriptorType::STORAGE_TEXEL_BUFFER
                }
                Op::TypeImage if ty.1[6] == 2 => vk::DescriptorType::STORAGE_IMAGE,
                Op::TypeImage => {
                    let shape = |a: &[u32]| (a[2], a[4] != 0, a[3] == 1, a[5] != 0);
                    let mut key = shape(ty.1);
                    // glslc declares a separate texture with Depth=0 even when
                    // its OpSampledImage type is a shadow image. Resolve the
                    // used type through that variable's load, not a global
                    // shadow flag, so both modules bind the same depth data.
                    // /usr/include/spirv/unified1/spirv.core.grammar.json:
                    // OpTypeImage354-366, OpTypeSampledImage380-386,
                    // OpLoad704-717 and OpSampledImage985-993.
                    for (_, sampled) in declarations
                        .iter()
                        .filter(|(op, _)| *op == Op::SampledImage)
                    {
                        let from_variable = declarations
                            .iter()
                            .any(|(op, a)| *op == Op::Load && a[1] == sampled[2] && a[2] == var);
                        if from_variable {
                            let sampled_type = declarations
                                .iter()
                                .find(|(op, a)| *op == Op::TypeSampledImage && a[0] == sampled[0])
                                .unwrap()
                                .1;
                            let image_type = declarations
                                .iter()
                                .find(|(op, a)| *op == Op::TypeImage && a[0] == sampled_type[1])
                                .unwrap()
                                .1;
                            key = shape(image_type);
                        }
                    }
                    image_keys.insert((set, bindings[&var]), key);
                    vk::DescriptorType::SAMPLED_IMAGE
                }
                Op::TypeSampler => vk::DescriptorType::SAMPLER,
                Op::TypeStruct if pointer[1] == 12 => vk::DescriptorType::STORAGE_BUFFER,
                Op::TypeStruct => vk::DescriptorType::UNIFORM_BUFFER,
                _ => panic!("unsupported test descriptor type {:?}", ty.0),
            };
            descriptors[set as usize].insert(bindings[&var], (descriptor, count));
        }
        let image_shapes = sets
            .iter()
            .filter_map(|(&var, &set)| {
                if set >= 3 {
                    return None;
                }
                let variable = declarations
                    .iter()
                    .find(|(op, a)| *op == Op::Variable && a[1] == var)?
                    .1;
                let pointer = declarations
                    .iter()
                    .find(|(op, a)| *op == Op::TypePointer && a[0] == variable[0])?
                    .1;
                let image = declarations
                    .iter()
                    .find(|(op, a)| *op == Op::TypeImage && a[0] == pointer[2])?
                    .1;
                if image[6] == 2 {
                    return None;
                }
                Some((
                    (set as usize, bindings[&var]),
                    (image[2], image[4] != 0, image[3] == 1, image[5] != 0),
                ))
            })
            .collect::<BTreeMap<_, _>>();
        let mut storage = BTreeMap::new();
        for (&var, &set) in &sets {
            if set >= 3 {
                continue;
            }
            let variable = declarations
                .iter()
                .find(|(op, a)| *op == Op::Variable && a[1] == var)
                .unwrap()
                .1;
            let pointer = declarations
                .iter()
                .find(|(op, a)| *op == Op::TypePointer && a[0] == variable[0])
                .unwrap()
                .1;
            if let Some((_, image)) = declarations
                .iter()
                .find(|(op, a)| *op == Op::TypeImage && a[0] == pointer[2] && a[6] == 2)
            {
                if persistent_storage {
                    assert_eq!(
                        (image[2], image[4]),
                        (1, 0),
                        "persistent writer/reader image must be 2D"
                    );
                    // Unknown0 permits formatless access to the same rgba32f
                    // fixture; Rgba32f1 is explicit. Mesa src/compiler/spirv/spirv.h:292-293.
                    assert!(
                        matches!(image[7], 0 | 1),
                        "persistent image only supports float rgba32f access"
                    );
                    continue;
                }
                storage.insert((set as usize, bindings[&var]), unsafe {
                    surface::Surface::new(
                        &self.instance,
                        &self.device,
                        self.physical,
                        self.command,
                        self.queue,
                        (image[2], image[4] != 0, image[7]),
                    )
                });
            }
        }
        unsafe {
            let d = &self.device;
            if let Some((bank, word, value)) = cbuf_override {
                *self.uniform.mapped.add((bank * 16384 + word) as usize) = value;
            }
            std::ptr::copy_nonoverlapping(
                inputs.as_ptr().cast::<u32>(),
                self.inputs.mapped,
                inputs.len() * WORDS,
            );
            for (case, state) in inputs.iter().enumerate() {
                std::ptr::copy_nonoverlapping(
                    state[STATE_WORDS..].as_ptr(),
                    self.flat.mapped.add(case * MEMORY_WORDS),
                    MEMORY_WORDS,
                );
            }
            std::ptr::write_bytes(self.outputs.mapped, 0xa5, inputs.len() * WORDS);
            let binding = |b, ty, count| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b)
                    .descriptor_type(ty)
                    .descriptor_count(count)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            };
            let production = descriptors.each_ref().map(|set| {
                set.iter()
                    .map(|(&b, &(ty, count))| binding(b, ty, count))
                    .collect::<Vec<_>>()
            });
            let sb = [
                binding(0, vk::DescriptorType::STORAGE_BUFFER, 1),
                binding(1, vk::DescriptorType::STORAGE_BUFFER, 1),
            ];
            let layouts = [
                production[0].as_slice(),
                production[1].as_slice(),
                production[2].as_slice(),
                sb.as_slice(),
            ]
            .map(|b| {
                d.create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(b),
                    None,
                )
                .unwrap()
            });
            let push = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(8)];
            let layout = d
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&layouts)
                        .push_constant_ranges(&push),
                    None,
                )
                .unwrap();
            let sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: descriptors
                        .iter()
                        .flat_map(|set| set.values())
                        .filter(|(ty, _)| *ty == vk::DescriptorType::STORAGE_IMAGE)
                        .map(|(_, count)| count)
                        .sum::<u32>()
                        .max(1),
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_TEXEL_BUFFER,
                    descriptor_count: descriptors
                        .iter()
                        .flat_map(|set| set.values())
                        .filter(|(ty, _)| *ty == vk::DescriptorType::STORAGE_TEXEL_BUFFER)
                        .map(|(_, count)| count)
                        .sum::<u32>()
                        .max(1),
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::SAMPLED_IMAGE,
                    descriptor_count: descriptors
                        .iter()
                        .flat_map(|set| set.values())
                        .filter(|(ty, _)| *ty == vk::DescriptorType::SAMPLED_IMAGE)
                        .map(|(_, count)| count)
                        .sum::<u32>()
                        .max(1),
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::SAMPLER,
                    descriptor_count: descriptors
                        .iter()
                        .flat_map(|set| set.values())
                        .filter(|(ty, _)| *ty == vk::DescriptorType::SAMPLER)
                        .map(|(_, count)| count)
                        .sum::<u32>()
                        .max(1),
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::UNIFORM_BUFFER,
                    descriptor_count: descriptors
                        .iter()
                        .flat_map(|set| set.values())
                        .filter(|(ty, _)| *ty == vk::DescriptorType::UNIFORM_BUFFER)
                        .map(|(_, count)| count)
                        .sum::<u32>()
                        .max(1),
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 2 + descriptors
                        .iter()
                        .flat_map(|set| set.values())
                        .filter(|(ty, _)| *ty == vk::DescriptorType::STORAGE_BUFFER)
                        .map(|(_, count)| count)
                        .sum::<u32>(),
                },
            ];
            let pool = d
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(4)
                        .pool_sizes(&sizes),
                    None,
                )
                .unwrap();
            let sets = d
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(pool)
                        .set_layouts(&layouts),
                )
                .unwrap();
            for (set, descriptors) in descriptors.iter().enumerate() {
                for (&b, &(ty, count)) in descriptors {
                    if let Some(surface) = storage.get(&(set, b)) {
                        let write = vk::WriteDescriptorSet::default()
                            .dst_set(sets[set])
                            .dst_binding(b)
                            .descriptor_type(ty);
                        let image = [vk::DescriptorImageInfo::default()
                            .image_view(surface.view)
                            .image_layout(vk::ImageLayout::GENERAL)];
                        let view = [surface.buffer_view];
                        d.update_descriptor_sets(
                            &[if ty == vk::DescriptorType::STORAGE_TEXEL_BUFFER {
                                write.texel_buffer_view(&view)
                            } else {
                                write.image_info(&image)
                            }],
                            &[],
                        );
                        continue;
                    }
                    // Descriptor global memory uses the same independently
                    // initialized backing words as flat memory. No operation
                    // in the translated module is changed by this binding.
                    let buffer = [if ty == vk::DescriptorType::STORAGE_BUFFER {
                        vk::DescriptorBufferInfo::default()
                            .buffer(self.flat.buffer)
                            .offset(0)
                            .range((inputs.len() * MEMORY_WORDS * 4) as u64)
                    } else {
                        vk::DescriptorBufferInfo::default()
                            .buffer(self.uniform.buffer)
                            .offset(u64::from(b) * 65536)
                            .range(65536)
                    }];
                    let shape = image_keys
                        .get(&(set as u32, b))
                        .or_else(|| image_keys.get(&(set as u32, b.saturating_sub(1))))
                        .or_else(|| {
                            image_shapes
                                .get(&(set, b))
                                .or_else(|| image_shapes.get(&(set, b.saturating_sub(1))))
                        });
                    let texture =
                        &self.textures[&shape.copied().unwrap_or((1, false, false, false))];
                    let image = vec![
                        vk::DescriptorImageInfo::default()
                            .sampler(texture.sampler)
                            .image_view(if ty == vk::DescriptorType::STORAGE_IMAGE {
                                self.storage_image.view
                            } else {
                                texture.view
                            })
                            .image_layout(if ty == vk::DescriptorType::STORAGE_IMAGE {
                                vk::ImageLayout::GENERAL
                            } else {
                                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
                            });
                        count as usize
                    ];
                    let write = vk::WriteDescriptorSet::default()
                        .dst_set(sets[set])
                        .dst_binding(b)
                        .descriptor_type(ty);
                    let write = if matches!(
                        ty,
                        vk::DescriptorType::UNIFORM_BUFFER | vk::DescriptorType::STORAGE_BUFFER
                    ) {
                        write.buffer_info(&buffer)
                    } else {
                        write.image_info(&image)
                    };
                    d.update_descriptor_sets(&[write], &[]);
                }
            }
            for (b, buffer) in [self.inputs.buffer, self.outputs.buffer]
                .into_iter()
                .enumerate()
            {
                d.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(sets[3])
                        .dst_binding(b as u32)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&[vk::DescriptorBufferInfo::default()
                            .buffer(buffer)
                            .range((WORDS * MAX_CASES * 4) as u64)])],
                    &[],
                );
            }
            let shader = d
                .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None)
                .unwrap();
            let pipeline = d
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::ComputePipelineCreateInfo::default()
                        .layout(layout)
                        .stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::COMPUTE)
                                .module(shader)
                                .name(c"main"),
                        )],
                    None,
                )
                .unwrap()[0];
            d.reset_command_buffer(self.command, vk::CommandBufferResetFlags::empty())
                .unwrap();
            d.begin_command_buffer(self.command, &vk::CommandBufferBeginInfo::default())
                .unwrap();
            // Persist the test image across independent writer/reader dispatches.
            d.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE)],
                &[],
                &[],
            );
            d.cmd_bind_pipeline(self.command, vk::PipelineBindPoint::COMPUTE, pipeline);
            d.cmd_bind_descriptor_sets(
                self.command,
                vk::PipelineBindPoint::COMPUTE,
                layout,
                0,
                &sets,
                &[],
            );
            d.cmd_push_constants(
                self.command,
                layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &self
                    .delta
                    .wrapping_add(GUEST_BASE)
                    .wrapping_sub(guest_base)
                    .to_ne_bytes(),
            );
            let local_size = ops(words)
                .find(|(op, a)| *op == Op::ExecutionMode && a[1] == 17)
                .unwrap()
                .1[2] as usize;
            assert!(inputs.len().is_multiple_of(local_size));
            d.cmd_dispatch(self.command, (inputs.len() / local_size) as u32, 1, 1);
            d.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                    .dst_access_mask(vk::AccessFlags::TRANSFER_READ)],
                &[],
                &[],
            );
            for surface in storage.values() {
                surface.copy_back(d, self.command);
            }
            d.cmd_pipeline_barrier(
                self.command,
                vk::PipelineStageFlags::COMPUTE_SHADER | vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(
                        vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::TRANSFER_WRITE,
                    )
                    .dst_access_mask(vk::AccessFlags::HOST_READ)],
                &[],
                &[],
            );
            d.end_command_buffer(self.command).unwrap();
            submit_and_wait(d, self.queue, self.command);
            let mut result = std::slice::from_raw_parts(
                self.outputs.mapped.cast::<[u32; WORDS]>(),
                inputs.len(),
            )
            .to_vec();
            if let Some((bank, word, _)) = cbuf_override {
                *self.uniform.mapped.add((bank * 16384 + word) as usize) = cbuf(bank, word);
            }
            if flat {
                for (case, state) in result.iter_mut().enumerate() {
                    state[STATE_WORDS..].copy_from_slice(std::slice::from_raw_parts(
                        self.flat.mapped.add(case * MEMORY_WORDS),
                        MEMORY_WORDS,
                    ));
                }
            }
            d.destroy_pipeline(pipeline, None);
            d.destroy_shader_module(shader, None);
            d.destroy_descriptor_pool(pool, None);
            d.destroy_pipeline_layout(layout, None);
            for layout in layouts {
                d.destroy_descriptor_set_layout(layout, None);
            }
            let image_bytes = storage.values().flat_map(|s| s.bytes()).collect::<Vec<_>>();
            for surface in storage.values() {
                surface.destroy(d);
            }
            (result, image_bytes)
        }
    }
}
impl Drop for Compute {
    fn drop(&mut self) {
        unsafe {
            for b in [&self.inputs, &self.outputs, &self.uniform, &self.flat] {
                self.device.unmap_memory(b.memory);
                self.device.destroy_buffer(b.buffer, None);
                self.device.free_memory(b.memory, None);
            }
            for texture in self.textures.values() {
                texture.destroy(&self.device);
            }
            self.storage_image.destroy(&self.device);
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}
pub fn cbuf(bank: u32, word: u32) -> u32 {
    // Project-owned distinct values expose both bank and word addressing.
    // Word zero also gives BFE a valid offset/width pair in every bank.
    // BFE range bytes: Ryujinx InstEmitBitfield.cs:141-143.
    if word == 0 {
        return bank << 16 | 16 << 8 | bank & 15;
    }
    (word.wrapping_mul(0x9e37_79b9) ^ bank.wrapping_mul(0x85eb_ca6b) ^ 0x8001_8081)
        .rotate_left(word & 31)
}

// A timeout must end this isolated test process. Unwinding would destroy
// resources still used by the GPU. The parent gate has a separate deadline
// covering pipeline compilation and records the current fixture and phase.
pub fn submit_and_wait(device: &ash::Device, queue: vk::Queue, command: vk::CommandBuffer) {
    unsafe {
        let fence = device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .unwrap();
        let submitted = device.queue_submit(
            queue,
            &[vk::SubmitInfo::default().command_buffers(&[command])],
            fence,
        );
        let result = submitted.and_then(|()| device.wait_for_fences(&[fence], true, 5_000_000_000));
        if let Err(error) = result {
            eprintln!("GPU submission failed: {error:?}; fence deadline=5 seconds");
            std::process::exit(124);
        }
        device.destroy_fence(fence, None);
    }
}

#[allow(dead_code)]
pub fn bfe_cbuf(bank: u32, word: u32) -> u32 {
    let original = cbuf(bank, word);
    let offset = original & 255;
    let count = original >> 8 & 255;
    if offset <= 32 && count <= 32 - offset {
        return original;
    }
    // Range bytes: Ryujinx InstEmitBitfield.cs:141-147; Mesa
    // sm50.rs:1174-1179. Preserve every previously defined cbuf input,
    // and supply a defined range for the addresses that had no active cases.
    // Upper bits still identify both bank and word and remain observable.
    // Give formerly empty domains an actual extraction, not only width zero.
    let offset = offset % 32;
    let count = 1 + count % (32 - offset);
    original & !65535 | count << 8 | offset
}
