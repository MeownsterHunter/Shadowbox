// Project-owned Vulkan runner, not shader translation code. API field facts:
// /usr/include/vulkan/vulkan_core.h:3783-3788 (allocation),
// 3887-3896 (buffer),4038-4043 (barrier),4104-4118 (compute stage/pipeline),
// 4207-4220 (descriptor layout),4551,4569,4596,4633 (allocation, bounded
// fence wait, barrier, dispatch),6357,6475-6482 (RADV driver identity).
// Descriptor identities follow Ryujinx CodeGen/Spirv/Declarations.cs:136-144.
mod images;

use super::{
    module::{BufferKind, Plan},
    Capture,
};
use ash::{vk, Device, Entry, Instance};

struct Allocation {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    mapped: *const u8,
}

struct Gpu {
    _entry: Entry,
    instance: Instance,
    device: Device,
    allocations: Vec<Allocation>,
    images: Vec<images::ImageAllocation>,
    set_layouts: Vec<vk::DescriptorSetLayout>,
    descriptor_pool: vk::DescriptorPool,
    layout: vk::PipelineLayout,
    shader: vk::ShaderModule,
    pipeline: vk::Pipeline,
    partner: vk::ShaderModule,
    tess_partner: vk::ShaderModule,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    command_pool: vk::CommandPool,
    fence: vk::Fence,
}

impl Gpu {
    unsafe fn allocate(
        &mut self,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
        initial: &[u8],
    ) -> Result<usize, String> {
        let handle = self
            .device
            .create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(initial.len() as u64)
                    .usage(usage)
                    .sharing_mode(vk::SharingMode::EXCLUSIVE),
                None,
            )
            .map_err(error)?;
        self.allocations.push(Allocation {
            buffer: handle,
            memory: vk::DeviceMemory::null(),
            mapped: std::ptr::null(),
        });
        let allocation = self.allocations.last_mut().unwrap();
        let req = self.device.get_buffer_memory_requirements(handle);
        let flags = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let memory_type = (0..memory_properties.memory_type_count)
            .find(|&i| {
                req.memory_type_bits & (1 << i) != 0
                    && memory_properties.memory_types[i as usize]
                        .property_flags
                        .contains(flags)
            })
            .ok_or("no coherent host-visible memory")?;
        // Device-address usage/allocation flags: vulkan_core.h:3067,5567,5711-5716.
        let mut flags = vk::MemoryAllocateFlagsInfo::default().flags(
            if usage.contains(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS) {
                vk::MemoryAllocateFlags::DEVICE_ADDRESS
            } else {
                vk::MemoryAllocateFlags::empty()
            },
        );
        allocation.memory = self
            .device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(req.size)
                    .memory_type_index(memory_type)
                    .push_next(&mut flags),
                None,
            )
            .map_err(error)?;
        self.device
            .bind_buffer_memory(handle, allocation.memory, 0)
            .map_err(error)?;
        let mapped = self
            .device
            .map_memory(
                allocation.memory,
                0,
                initial.len() as u64,
                vk::MemoryMapFlags::empty(),
            )
            .map_err(error)?;
        allocation.mapped = mapped.cast();
        std::ptr::copy_nonoverlapping(initial.as_ptr(), mapped.cast::<u8>(), initial.len());
        Ok(self.allocations.len() - 1)
    }
}

impl Drop for Gpu {
    fn drop(&mut self) {
        // The worker exits immediately on a failed fence wait. This Drop
        // runs only before submission or after the submission completes.
        unsafe {
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_shader_module(self.shader, None);
            self.device.destroy_shader_module(self.partner, None);
            self.device.destroy_shader_module(self.tess_partner, None);
            self.device.destroy_framebuffer(self.framebuffer, None);
            self.device.destroy_render_pass(self.render_pass, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            for &layout in &self.set_layouts {
                self.device.destroy_descriptor_set_layout(layout, None);
            }
            images::destroy(self);
            for allocation in &self.allocations {
                if !allocation.mapped.is_null() {
                    self.device.unmap_memory(allocation.memory);
                }
                self.device.destroy_buffer(allocation.buffer, None);
                self.device.free_memory(allocation.memory, None);
            }
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn error(result: vk::Result) -> String {
    format!("Vulkan: {result:?}")
}

pub(super) fn execute(words: &[u32], plan: &Plan) -> Result<Vec<Capture>, String> {
    // Safety: all Vulkan object handles belong to this worker and remain live
    // until the fence signals. Host pointers cover coherent mapped 64-KiB
    // allocations. Module inspection proves the supported accesses in bounds.
    unsafe { execute_inner(words, plan) }
}

unsafe fn execute_inner(words: &[u32], plan: &Plan) -> Result<Vec<Capture>, String> {
    let entry = Entry::load().map_err(|e| format!("load Vulkan: {e}"))?;
    let instance = entry
        .create_instance(
            &vk::InstanceCreateInfo::default().application_info(
                &vk::ApplicationInfo::default()
                    .application_name(c"shadowbox-value-diff")
                    .api_version(vk::API_VERSION_1_2),
            ),
            None,
        )
        .map_err(error)?;
    let physical = instance
        .enumerate_physical_devices()
        .map_err(error)?
        .into_iter()
        .find(|&physical| {
            let mut driver = vk::PhysicalDeviceDriverProperties::default();
            let mut properties = vk::PhysicalDeviceProperties2::default().push_next(&mut driver);
            instance.get_physical_device_properties2(physical, &mut properties);
            driver.driver_id == vk::DriverId::MESA_RADV
        })
        .ok_or("no RADV device available")?;
    let graphics = plan.graphics.as_ref();
    let stage = match graphics {
        Some(g) if g.stage == 1 => vk::ShaderStageFlags::TESSELLATION_CONTROL,
        Some(g) if g.stage == 2 => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
        Some(g) if g.stage == 3 => vk::ShaderStageFlags::GEOMETRY,
        Some(g) if g.vertex => vk::ShaderStageFlags::VERTEX,
        Some(_) => vk::ShaderStageFlags::FRAGMENT,
        None => vk::ShaderStageFlags::COMPUTE,
    };
    let queue_flags = if graphics.is_some() {
        vk::QueueFlags::GRAPHICS
    } else {
        vk::QueueFlags::COMPUTE
    };
    let family = instance
        .get_physical_device_queue_family_properties(physical)
        .iter()
        .position(|p| p.queue_flags.contains(queue_flags))
        .ok_or("no suitable queue")? as u32;
    let properties = instance.get_physical_device_properties(physical);
    // Explicit wave size and full-subgroup contract: public Vulkan header
    // vulkan_core.h:3269,6009-6016,7771-7791. Do not assume local invocation
    // numbering equals subgroup numbering; the proof uses only subgroup IDs.
    let mut subgroup_features = vk::PhysicalDeviceSubgroupSizeControlFeatures::default();
    let mut subgroup_properties = vk::PhysicalDeviceSubgroupSizeControlProperties::default();
    let mut subgroup_operations = vk::PhysicalDeviceSubgroupProperties::default();
    if plan.subgroup {
        let extensions = instance
            .enumerate_device_extension_properties(physical)
            .map_err(error)?;
        if !extensions.iter().any(|e| {
            std::ffi::CStr::from_ptr(e.extension_name.as_ptr())
                == ash::ext::subgroup_size_control::NAME
        }) {
            return Err("subgroup size control extension unavailable".into());
        }
        let mut f = vk::PhysicalDeviceFeatures2::default().push_next(&mut subgroup_features);
        instance.get_physical_device_features2(physical, &mut f);
        let mut p = vk::PhysicalDeviceProperties2::default()
            .push_next(&mut subgroup_properties)
            .push_next(&mut subgroup_operations);
        instance.get_physical_device_properties2(physical, &mut p);
        let mut required = vk::SubgroupFeatureFlags::BASIC
            | vk::SubgroupFeatureFlags::VOTE
            | vk::SubgroupFeatureFlags::BALLOT
            | vk::SubgroupFeatureFlags::SHUFFLE
            | vk::SubgroupFeatureFlags::SHUFFLE_RELATIVE;
        if plan.maximal_reconvergence {
            required |= vk::SubgroupFeatureFlags::ARITHMETIC;
        }
        if subgroup_features.subgroup_size_control == 0
            || subgroup_features.compute_full_subgroups == 0
            || subgroup_properties.min_subgroup_size > 32
            || subgroup_properties.max_subgroup_size < 32
            || !subgroup_properties
                .required_subgroup_size_stages
                .contains(vk::ShaderStageFlags::COMPUTE)
            || !subgroup_operations
                .supported_stages
                .contains(vk::ShaderStageFlags::COMPUTE)
            || !subgroup_operations.supported_operations.contains(required)
        {
            return Err("device cannot execute the proved 32-lane subgroup contract".into());
        }
    }

    // LocalSize and shared-memory limits: vulkan_core.h:3636-3639.
    if plan.graphics.is_none()
        && (plan.shared_size > properties.limits.max_compute_shared_memory_size as usize
            || plan.local_size.into_iter().product::<u32>()
                > properties.limits.max_compute_work_group_invocations
            || plan
                .local_size
                .iter()
                .zip(properties.limits.max_compute_work_group_size)
                .any(|(&n, limit)| n > limit))
    {
        return Err("compute workgroup exceeds device limits".into());
    }
    if properties.limits.max_uniform_buffer_range < super::module::BUFFER_SIZE as u32
        || properties.limits.max_storage_buffer_range < super::module::BUFFER_SIZE as u32
    {
        return Err("RADV buffer range limit is below 64 KiB".into());
    }
    // Descriptor and vertex-input limits: vulkan_core.h:3598-3615.
    let storage_count = plan
        .buffers
        .iter()
        .filter(|b| b.kind == BufferKind::Storage)
        .count() as u32;
    let uniform_count = plan
        .buffers
        .iter()
        .filter(|b| b.kind == BufferKind::Uniform)
        .count() as u32;
    let image_count = plan
        .images
        .iter()
        .filter(|r| matches!(r.kind, super::module::images::ResourceKind::Image { .. }))
        .count() as u32;
    let sampler_count = plan
        .images
        .iter()
        .filter(|r| {
            matches!(
                r.kind,
                super::module::images::ResourceKind::Sampler { .. }
                    | super::module::images::ResourceKind::Image { combined: true, .. }
            )
        })
        .count() as u32;
    if image_count > properties.limits.max_per_stage_descriptor_sampled_images
        || image_count > properties.limits.max_descriptor_set_sampled_images
        || sampler_count > properties.limits.max_per_stage_descriptor_samplers
        || sampler_count > properties.limits.max_descriptor_set_samplers
        || plan
            .images
            .iter()
            .any(|r| r.set >= properties.limits.max_bound_descriptor_sets)
        || storage_count > properties.limits.max_per_stage_descriptor_storage_buffers
        || storage_count > properties.limits.max_descriptor_set_storage_buffers
        || uniform_count > properties.limits.max_per_stage_descriptor_uniform_buffers
        || plan
            .buffers
            .iter()
            .filter(|b| matches!(b.kind, BufferKind::Uniform | BufferKind::Storage))
            .any(|b| b.set >= properties.limits.max_bound_descriptor_sets)
    {
        return Err("descriptor interface exceeds device limits".into());
    }
    if let Some(graphics) = graphics {
        if graphics.attributes.len() as u32 > properties.limits.max_vertex_input_bindings
            || graphics.attributes.iter().any(|a| {
                a.location >= properties.limits.max_vertex_input_attributes
                    || !instance
                        .get_physical_device_format_properties(physical, a.format)
                        .buffer_features
                        .contains(vk::FormatFeatureFlags::VERTEX_BUFFER)
            })
        {
            return Err("vertex attribute interface exceeds device limits".into());
        }
    }
    let priority = [1.0];
    let queues = [vk::DeviceQueueCreateInfo::default()
        .queue_family_index(family)
        .queue_priorities(&priority)];
    // Feature fields: vulkan_core.h:3552-3553. Observation stores require
    // the stage store feature even though source shaders have no stores.
    let available = instance.get_physical_device_features(physical);
    if graphics.is_some()
        && (available.vertex_pipeline_stores_and_atomics == 0
            || available.fragment_stores_and_atomics == 0)
    {
        return Err("graphics shader stores unavailable".into());
    }
    if plan.stage == 3
        && (available.geometry_shader == 0
            || properties.limits.max_geometry_output_vertices < 32
            || properties.limits.max_geometry_total_output_components < 512)
    {
        return Err("geometry capture device limits unavailable".into());
    }
    // Geometry streams feature/limits: vulkan_core.h:15252-15272.
    let streams = graphics.is_some_and(|g| g.streams);
    let mut stream_features = vk::PhysicalDeviceTransformFeedbackFeaturesEXT::default();
    if streams {
        let extensions = instance
            .enumerate_device_extension_properties(physical)
            .map_err(error)?;
        if !extensions.iter().any(|e| {
            std::ffi::CStr::from_ptr(e.extension_name.as_ptr())
                == ash::ext::transform_feedback::NAME
        }) {
            return Err("geometry stream extension unavailable".into());
        }
        let mut f = vk::PhysicalDeviceFeatures2::default().push_next(&mut stream_features);
        instance.get_physical_device_features2(physical, &mut f);
        let mut p = vk::PhysicalDeviceTransformFeedbackPropertiesEXT::default();
        let mut query = vk::PhysicalDeviceProperties2::default().push_next(&mut p);
        instance.get_physical_device_properties2(physical, &mut query);
        if stream_features.geometry_streams == 0 || p.max_transform_feedback_streams < 4 {
            return Err("four geometry streams unavailable".into());
        }
        stream_features.transform_feedback = 0;
    }
    if matches!(plan.stage, 1 | 2) && available.tessellation_shader == 0 {
        return Err("tessellation shaders unavailable".into());
    }
    let cube_array = plan.images.iter().any(|r| matches!(r.kind, super::module::images::ResourceKind::Image { ty, .. } if ty.dim == 3 && ty.arrayed));
    if cube_array && available.image_cube_array == 0 {
        return Err("cube array images unavailable".into());
    }
    // BDA feature and address query: vulkan_core.h:3567,5803-5807,6695-6707,7060-7062.
    // Extension name and feature structure: vulkan_core.h:13103-13111.
    // Query before device creation, then enable both for execution mode 6023
    // (Mesa src/compiler/spirv/spirv.h:210). Never submit without the feature.
    let mut maximal = vk::PhysicalDeviceShaderMaximalReconvergenceFeaturesKHR::default();
    if plan.maximal_reconvergence {
        let extensions = instance
            .enumerate_device_extension_properties(physical)
            .map_err(error)?;
        let has_extension = extensions.iter().any(|e| {
            std::ffi::CStr::from_ptr(e.extension_name.as_ptr())
                == ash::khr::shader_maximal_reconvergence::NAME
        });
        let mut query = vk::PhysicalDeviceFeatures2::default().push_next(&mut maximal);
        instance.get_physical_device_features2(physical, &mut query);
        require_maximal_reconvergence(has_extension, maximal.shader_maximal_reconvergence != 0)?;
        maximal.shader_maximal_reconvergence = vk::TRUE;
    }
    let physical_memory = plan.buffers.iter().any(|b| b.kind == BufferKind::Physical);
    let mut addressing = vk::PhysicalDeviceBufferDeviceAddressFeatures::default();
    let mut extended = vk::PhysicalDeviceFeatures2::default().push_next(&mut addressing);
    instance.get_physical_device_features2(physical, &mut extended);
    if physical_memory && (addressing.buffer_device_address == 0 || available.shader_int64 == 0) {
        return Err("physical memory device features unavailable".into());
    }
    let mut addressing = vk::PhysicalDeviceBufferDeviceAddressFeatures::default()
        .buffer_device_address(physical_memory);
    let features = vk::PhysicalDeviceFeatures::default()
        .geometry_shader(plan.stage == 3)
        .tessellation_shader(matches!(plan.stage, 1 | 2))
        .shader_int64(physical_memory)
        .image_cube_array(cube_array)
        .vertex_pipeline_stores_and_atomics(graphics.is_some())
        .fragment_stores_and_atomics(graphics.is_some());
    let mut extension_names = if plan.subgroup {
        vec![ash::ext::subgroup_size_control::NAME.as_ptr()]
    } else {
        Vec::new()
    };
    if plan.maximal_reconvergence {
        extension_names.push(ash::khr::shader_maximal_reconvergence::NAME.as_ptr());
    }
    if streams {
        extension_names.push(ash::ext::transform_feedback::NAME.as_ptr());
    }
    let mut device_info = vk::DeviceCreateInfo::default()
        .queue_create_infos(&queues)
        .enabled_features(&features)
        .enabled_extension_names(&extension_names)
        .push_next(&mut addressing);
    if plan.subgroup {
        device_info = device_info.push_next(&mut subgroup_features);
    }
    if streams {
        device_info = device_info.push_next(&mut stream_features);
    }
    if plan.maximal_reconvergence {
        device_info = device_info.push_next(&mut maximal);
    }
    let device = instance
        .create_device(physical, &device_info, None)
        .map_err(error)?;
    let queue = device.get_device_queue(family, 0);
    let mut gpu = Gpu {
        _entry: entry,
        instance,
        device,
        allocations: Vec::new(),
        images: Vec::new(),
        set_layouts: Vec::new(),
        descriptor_pool: vk::DescriptorPool::null(),
        layout: vk::PipelineLayout::null(),
        shader: vk::ShaderModule::null(),
        pipeline: vk::Pipeline::null(),
        partner: vk::ShaderModule::null(),
        tess_partner: vk::ShaderModule::null(),
        render_pass: vk::RenderPass::null(),
        framebuffer: vk::Framebuffer::null(),
        command_pool: vk::CommandPool::null(),
        fence: vk::Fence::null(),
    };
    let memory_properties = gpu.instance.get_physical_device_memory_properties(physical);
    for buffer in &plan.buffers {
        let usage = match buffer.kind {
            BufferKind::Uniform => vk::BufferUsageFlags::UNIFORM_BUFFER,
            BufferKind::Storage => vk::BufferUsageFlags::STORAGE_BUFFER,
            BufferKind::Vertex => vk::BufferUsageFlags::VERTEX_BUFFER,
            BufferKind::Physical => {
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            }
        };
        gpu.allocate(&memory_properties, usage, &buffer.initial)?;
    }
    images::create(&mut gpu, plan, &memory_properties, physical)?;
    let descriptor_type = |kind| match kind {
        BufferKind::Uniform => vk::DescriptorType::UNIFORM_BUFFER,
        BufferKind::Storage => vk::DescriptorType::STORAGE_BUFFER,
        BufferKind::Vertex | BufferKind::Physical => unreachable!(),
    };
    let max_set = plan
        .buffers
        .iter()
        .filter(|b| matches!(b.kind, BufferKind::Uniform | BufferKind::Storage))
        .map(|b| b.set)
        .chain(plan.images.iter().map(|r| r.set))
        .max()
        .unwrap_or(0);
    for set in 0..=max_set {
        let bindings = plan
            .buffers
            .iter()
            .filter(|b| b.set == set && matches!(b.kind, BufferKind::Uniform | BufferKind::Storage))
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b.binding)
                    .descriptor_count(1)
                    .descriptor_type(descriptor_type(b.kind))
                    .stage_flags(stage)
            })
            .chain(plan.images.iter().filter(|r| r.set == set).map(|r| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(r.binding)
                    .descriptor_count(1)
                    .descriptor_type(r.kind.descriptor())
                    .stage_flags(stage)
            }))
            .collect::<Vec<_>>();
        gpu.set_layouts.push(
            gpu.device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
                    None,
                )
                .map_err(error)?,
        );
    }
    let pool_sizes = [BufferKind::Uniform, BufferKind::Storage]
        .into_iter()
        .filter_map(|kind| {
            let count = plan.buffers.iter().filter(|b| b.kind == kind).count() as u32;
            (count != 0).then_some(vk::DescriptorPoolSize {
                ty: descriptor_type(kind),
                descriptor_count: count,
            })
        })
        .chain(
            [
                vk::DescriptorType::SAMPLER,
                vk::DescriptorType::SAMPLED_IMAGE,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            ]
            .into_iter()
            .filter_map(|ty| {
                let count = plan
                    .images
                    .iter()
                    .filter(|r| r.kind.descriptor() == ty)
                    .count() as u32;
                (count != 0).then_some(vk::DescriptorPoolSize {
                    ty,
                    descriptor_count: count,
                })
            }),
        )
        .collect::<Vec<_>>();
    gpu.descriptor_pool = gpu
        .device
        .create_descriptor_pool(
            &vk::DescriptorPoolCreateInfo::default()
                .max_sets(gpu.set_layouts.len() as u32)
                .pool_sizes(&pool_sizes),
            None,
        )
        .map_err(error)?;
    let sets = gpu
        .device
        .allocate_descriptor_sets(
            &vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(gpu.descriptor_pool)
                .set_layouts(&gpu.set_layouts),
        )
        .map_err(error)?;
    for (buffer, allocation) in plan.buffers.iter().zip(&gpu.allocations) {
        if matches!(buffer.kind, BufferKind::Vertex | BufferKind::Physical) {
            continue;
        }
        let info = [vk::DescriptorBufferInfo {
            buffer: allocation.buffer,
            offset: 0,
            range: buffer.initial.len() as u64,
        }];
        let writes = [vk::WriteDescriptorSet::default()
            .dst_set(sets[buffer.set as usize])
            .dst_binding(buffer.binding)
            .descriptor_type(descriptor_type(buffer.kind))
            .buffer_info(&info)];
        gpu.device.update_descriptor_sets(&writes, &[]);
    }
    images::descriptors(&gpu, plan, &sets);
    let pushes = if plan.push.is_empty() {
        vec![]
    } else {
        vec![vk::PushConstantRange {
            stage_flags: stage,
            offset: 0,
            size: plan.push.len() as u32,
        }]
    };
    gpu.layout = gpu
        .device
        .create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default()
                .set_layouts(&gpu.set_layouts)
                .push_constant_ranges(&pushes),
            None,
        )
        .map_err(error)?;
    gpu.shader = gpu
        .device
        .create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(
                graphics.map_or(plan.compute_words.as_deref().unwrap_or(words), |g| &g.words),
            ),
            None,
        )
        .map_err(error)?;
    if let Some(graphics) = graphics {
        create_graphics_pipeline(&mut gpu, graphics)?;
    } else {
        let mut required_size = vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo::default()
            .required_subgroup_size(32);
        let mut shader_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(gpu.shader)
            .name(c"main");
        if plan.subgroup {
            shader_stage = shader_stage
                .flags(vk::PipelineShaderStageCreateFlags::REQUIRE_FULL_SUBGROUPS)
                .push_next(&mut required_size);
        }
        let pipeline_info = [vk::ComputePipelineCreateInfo::default()
            .layout(gpu.layout)
            .stage(shader_stage)];
        gpu.pipeline = gpu
            .device
            .create_compute_pipelines(vk::PipelineCache::null(), &pipeline_info, None)
            .map_err(|(partial, result)| {
                for pipeline in partial {
                    gpu.device.destroy_pipeline(pipeline, None);
                }
                error(result)
            })?[0];
    }
    gpu.command_pool = gpu
        .device
        .create_command_pool(
            &vk::CommandPoolCreateInfo::default().queue_family_index(family),
            None,
        )
        .map_err(error)?;
    let command = gpu
        .device
        .allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(gpu.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1),
        )
        .map_err(error)?[0];
    gpu.device
        .begin_command_buffer(command, &vk::CommandBufferBeginInfo::default())
        .map_err(error)?;
    images::upload(&gpu, plan, command);
    let bind_point = if graphics.is_some() {
        vk::PipelineBindPoint::GRAPHICS
    } else {
        vk::PipelineBindPoint::COMPUTE
    };
    gpu.device
        .cmd_bind_pipeline(command, bind_point, gpu.pipeline);
    gpu.device
        .cmd_bind_descriptor_sets(command, bind_point, gpu.layout, 0, &sets, &[]);
    if !plan.push.is_empty() {
        let mut push = plan.push.clone();
        for (buffer, allocation) in plan.buffers.iter().zip(&gpu.allocations) {
            if buffer.kind == BufferKind::Physical {
                let base = gpu.device.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(allocation.buffer),
                );
                if base == 0 || !base.is_multiple_of(16) {
                    return Err("invalid flat global device address".into());
                }
                let offset = shadowbox::PUSH_GLOBAL_DELTA_OFFSET as usize;
                push[offset..offset + 8].copy_from_slice(&base.to_le_bytes());
            }
        }
        gpu.device
            .cmd_push_constants(command, gpu.layout, stage, 0, &push);
    }
    if let Some(graphics) = graphics {
        let area = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
        };
        gpu.device.cmd_begin_render_pass(
            command,
            &vk::RenderPassBeginInfo::default()
                .render_pass(gpu.render_pass)
                .framebuffer(gpu.framebuffer)
                .render_area(area),
            vk::SubpassContents::INLINE,
        );
        for (binding, attribute) in graphics.attributes.iter().enumerate() {
            gpu.device.cmd_bind_vertex_buffers(
                command,
                binding as u32,
                &[gpu.allocations[attribute.buffer].buffer],
                &[0],
            );
        }
        gpu.device.cmd_draw(command, graphics.vertices, 1, 0, 0);
        gpu.device.cmd_end_render_pass(command);
    } else {
        gpu.device.cmd_dispatch(command, 1, 1, 1);
    }
    let barriers = [vk::MemoryBarrier::default()
        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
        .dst_access_mask(vk::AccessFlags::HOST_READ)];
    gpu.device.cmd_pipeline_barrier(
        command,
        if graphics.is_some() {
            vk::PipelineStageFlags::ALL_GRAPHICS
        } else {
            vk::PipelineStageFlags::COMPUTE_SHADER
        },
        vk::PipelineStageFlags::HOST,
        vk::DependencyFlags::empty(),
        &barriers,
        &[],
        &[],
    );
    gpu.device.end_command_buffer(command).map_err(error)?;
    gpu.fence = gpu
        .device
        .create_fence(&vk::FenceCreateInfo::default(), None)
        .map_err(error)?;
    let commands = [command];
    let submits = [vk::SubmitInfo::default().command_buffers(&commands)];
    gpu.device
        .queue_submit(queue, &submits, gpu.fence)
        .map_err(error)?;
    if let Err(result) = gpu
        .device
        .wait_for_fences(&[gpu.fence], true, 2_000_000_000)
    {
        eprintln!("Vulkan dispatch did not complete within 2000 ms: {result:?}");
        // Do not call vkDeviceWaitIdle or destroy a device with pending work.
        // Process isolation lets the parent regain control even if the driver
        // stalls during process exit; the parent also has its own deadline.
        std::process::exit(124);
    }
    let mut captures = Vec::new();
    for (buffer, allocation) in plan.buffers.iter().zip(&gpu.allocations) {
        if matches!(buffer.kind, BufferKind::Storage | BufferKind::Physical) {
            if let Some(expected) = &buffer.expected {
                let bytes = std::slice::from_raw_parts(allocation.mapped, buffer.initial.len());
                if bytes != expected {
                    return Err("GPU active-lane masks differ from the collective proof".into());
                }
            }
            if buffer.observation.as_deref() == Some("__completion") {
                let bytes = std::slice::from_raw_parts(allocation.mapped, buffer.initial.len());
                if bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .any(|b| *b != 1_u32.to_le_bytes())
                {
                    return Err("draw did not capture every expected invocation".into());
                }
                continue;
            }
            captures.push(Capture {
                set: buffer.set,
                binding: buffer.binding,
                bytes: std::slice::from_raw_parts(allocation.mapped, buffer.initial.len()).to_vec(),
                defined: buffer.defined.clone(),
                output: buffer.observation.clone(),
            });
        }
    }
    Ok(if let Some(global) = &plan.global_capture {
        global.normalize(captures)
    } else {
        captures
    })
}

// Graphics API fields: /usr/include/vulkan/vulkan_core.h:4300-4417,
// 4436-4481,4520-4528,4639,4655-4656,4664,4666. This one-pixel
// attachment-free draw observes shader values before attachment conversion,
// depth testing or sample-mask suppression. Color, depth and mask remain
// shader outputs and are copied by the observation epilogue.
unsafe fn create_graphics_pipeline(
    gpu: &mut Gpu,
    graphics: &super::module::graphics::Graphics,
) -> Result<(), String> {
    gpu.partner = gpu
        .device
        .create_shader_module(
            &vk::ShaderModuleCreateInfo::default().code(&graphics.partner),
            None,
        )
        .map_err(error)?;
    let subpasses =
        [vk::SubpassDescription::default().pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)];
    gpu.render_pass = gpu
        .device
        .create_render_pass(
            &vk::RenderPassCreateInfo::default().subpasses(&subpasses),
            None,
        )
        .map_err(error)?;
    gpu.framebuffer = gpu
        .device
        .create_framebuffer(
            &vk::FramebufferCreateInfo::default()
                .render_pass(gpu.render_pass)
                .width(1)
                .height(1)
                .layers(1),
            None,
        )
        .map_err(error)?;
    if let Some(words) = &graphics.tess_partner {
        gpu.tess_partner = gpu
            .device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(words), None)
            .map_err(error)?;
    }
    let stages = if matches!(graphics.stage, 1 | 2) {
        vec![
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(gpu.partner)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::TESSELLATION_CONTROL)
                .module(if graphics.stage == 1 {
                    gpu.shader
                } else {
                    gpu.tess_partner
                })
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::TESSELLATION_EVALUATION)
                .module(if graphics.stage == 2 {
                    gpu.shader
                } else {
                    gpu.tess_partner
                })
                .name(c"main"),
        ]
    } else if graphics.stage == 3 {
        vec![
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(gpu.partner)
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::GEOMETRY)
                .module(gpu.shader)
                .name(c"main"),
        ]
    } else {
        vec![
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(if graphics.vertex {
                    gpu.shader
                } else {
                    gpu.partner
                })
                .name(c"main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(if graphics.vertex {
                    gpu.partner
                } else {
                    gpu.shader
                })
                .name(c"main"),
        ]
    };
    let bindings = graphics
        .attributes
        .iter()
        .enumerate()
        .map(|(i, a)| vk::VertexInputBindingDescription {
            binding: i as u32,
            stride: a.stride,
            input_rate: vk::VertexInputRate::VERTEX,
        })
        .collect::<Vec<_>>();
    let attributes = graphics
        .attributes
        .iter()
        .enumerate()
        .map(|(i, a)| vk::VertexInputAttributeDescription {
            location: a.location,
            binding: i as u32,
            format: a.format,
            offset: 0,
        })
        .collect::<Vec<_>>();
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attributes);
    let assembly = vk::PipelineInputAssemblyStateCreateInfo::default().topology(graphics.topology);
    let viewports = [vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: 1.0,
        height: 1.0,
        min_depth: 0.0,
        max_depth: 1.0,
    }];
    let scissors = [vk::Rect2D {
        offset: vk::Offset2D { x: 0, y: 0 },
        extent: vk::Extent2D {
            width: 1,
            height: 1,
        },
    }];
    let viewport = vk::PipelineViewportStateCreateInfo::default()
        .viewports(&viewports)
        .scissors(&scissors);
    let raster = vk::PipelineRasterizationStateCreateInfo::default()
        .rasterizer_discard_enable(matches!(graphics.stage, 1..=3))
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1);
    let depth = vk::PipelineDepthStencilStateCreateInfo::default();
    let blend = vk::PipelineColorBlendStateCreateInfo::default();
    // Patch control point fields: public vulkan_core.h:4370-4375.
    let tessellation = vk::PipelineTessellationStateCreateInfo::default().patch_control_points(1);
    let info = [vk::GraphicsPipelineCreateInfo::default()
        .tessellation_state(&tessellation)
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&assembly)
        .viewport_state(&viewport)
        .rasterization_state(&raster)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth)
        .color_blend_state(&blend)
        .layout(gpu.layout)
        .render_pass(gpu.render_pass)];
    gpu.pipeline = gpu
        .device
        .create_graphics_pipelines(vk::PipelineCache::null(), &info, None)
        .map_err(|(partial, result)| {
            for pipeline in partial {
                gpu.device.destroy_pipeline(pipeline, None);
            }
            error(result)
        })?[0];
    Ok(())
}

fn require_maximal_reconvergence(extension: bool, feature: bool) -> Result<(), String> {
    if !extension || !feature {
        return Err("uncheckable: device lacks maximal reconvergence".into());
    }
    Ok(())
}

#[cfg(test)]
mod maximal_tests {
    use super::require_maximal_reconvergence;

    #[test]
    fn extension_and_feature_are_both_required() {
        for (extension, feature) in [(false, false), (true, false), (false, true)] {
            assert_eq!(
                require_maximal_reconvergence(extension, feature).unwrap_err(),
                "uncheckable: device lacks maximal reconvergence"
            );
        }
        require_maximal_reconvergence(true, true).unwrap();
    }
}
